"""#2293 N1: an empty presence check skips reads without changing the audit plan or stored rows."""
from unittest.mock import patch

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_2466_presence_trip_keeps_mains_plan import PAD, SESSION, _a, _multiset, _trip, _u


def _run(tmp_path, monkeypatch, kind, *, force=False, rewrite=False, candidate=False):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"),
                                      large_output_externalization_enabled=False),
                       hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    seen, snapshots = [], []
    try:
        view = [_u("U0 first", 1000), _a("R0 reply"), _u("U1 second", 1001), _a("R1 reply"),
                _u("X steer note"), _a("RX reply"), _u("U2 third", 1002), _a("R2 reply")]
        engine.ingest(view)
        _trip(kind, engine, monkeypatch)
        real, audit = engine._identity_anchor_unaccounted, engine._identity_anchor_audit

        def spy(messages, identity_messages, cursor, start, identity_at, held, capped, consumed, plan, matched,
                excluded, missed):
            with patch.object(engine, "_rotation_predecessor_session_ids",
                              wraps=engine._rotation_predecessor_session_ids) as ancestry, patch.object(
                    engine._store, "get_session_tail", wraps=engine._store.get_session_tail) as tail:
                with monkeypatch.context() as control:
                    if force:  # Only inside presence: a sentinel rewrite forces the unchanged full path.
                        control.setattr(engine, "_identity_anchor_rewritten", lambda *_: [cursor])
                    found = real(messages, identity_messages, cursor, start, identity_at, held, capped, consumed,
                                 plan, matched, excluded, missed)
                seen.append((found, ancestry.call_count, tail.call_count, cursor, list(missed)))
                return found

        def audit_spy(*args, **kwargs):
            before = len(seen)
            out = audit(*args, **kwargs)
            if len(seen) > before:
                snapshots.append((args[7]["cursor"], set(args[7].get("positional", ()))))
            return out

        monkeypatch.setattr(engine, "_identity_anchor_unaccounted", spy)
        monkeypatch.setattr(engine, "_identity_anchor_audit", audit_spy)
        view[1] = _a("R0 a replaced reply")
        view[2] = _u("U1 a replaced second prompt", 1004)
        view[6] = _u("U2 a replaced third prompt", 1003)
        if rewrite:
            view[4]["content"] = "continue with the plan" + PAD
        if candidate:
            view[3:5] = [{"role": "assistant", "content": "", "tool_calls": [
                {"id": "new", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "new", "content": "new result"}]
        engine.ingest(view)
        engine.ingest(view)
        return seen, snapshots, _multiset(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("kind", ["plain", "window-cap", "tool-shape", "read-error"])
def test_empty_check_preserves_plan_and_rows_without_reads(tmp_path, monkeypatch, kind):
    with monkeypatch.context() as control:
        seen, plans, rows = _run(tmp_path / "skip", control, kind)
    with monkeypatch.context() as control:
        full, full_plans, full_rows = _run(tmp_path / "full", control, kind, force=True)
    assert seen and plans
    assert all(found == [] and ancestry == tail == 0 for found, ancestry, tail, *_ in seen)
    assert plans == full_plans and rows == full_rows
    assert full[0][0] == ([] if kind == "plain" else None)
    assert full[0][1] > 0
    assert full[0][2] > 0 if kind == "plain" else full[0][2] == 0


@pytest.mark.parametrize("kind", ["window-cap", "tool-shape", "read-error"])
def test_rewritten_row_still_trips_and_keeps_mains_plan(tmp_path, monkeypatch, kind):
    seen, plans, _ = _run(tmp_path, monkeypatch, kind, rewrite=True)
    found, ancestry, _, cursor, missed = seen[0]
    assert found is None and ancestry > 0
    assert plans[0] == (min(missed), set(range(min(missed), cursor)) - set(missed))
    assert 4 in plans[0][1] and min(missed) < 4


def test_unstored_call_and_result_still_run_check_and_are_stored(tmp_path, monkeypatch):
    seen, _, rows = _run(tmp_path, monkeypatch, "plain", candidate=True)
    assert seen[0][0] == [3, 4]
    assert seen[0][1] > 0 and seen[0][2] > 0
    assert sum(count for key, count in rows.items() if key[3] == "new") == 1
    assert sum(count for key, count in rows.items() if key[2] == "new") == 1
