"""#2466 review, scope cut: every trip of the presence check (the window cap, a non-standard stored tool shape, a failed
read) keeps main's plan exactly. The audit returns None; the caller then makes every row from the first missed row to
the cursor, except the missed rows, a positional replay, with no R6 exclusion, and logs main's INFO line. The shape is
an R6 in-place rewrite (the release manager's pre-check) on a turn whose audit misses rows."""
from __future__ import annotations

import json
import logging
from collections import Counter

import pytest

from hermes_lcm import identity_anchor
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-trip-plan"
PAD = " alpha beta gamma delta" * 40


def _u(text, ts=None):
    row = {"role": "user", "content": text + PAD}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _a(text):
    return {"role": "assistant", "content": text + PAD}


def _multiset(engine):
    rows = Counter()
    for r in engine._store.get_session_messages(SESSION):
        calls = r.get("tool_calls") or []
        calls = json.loads(calls) if isinstance(calls, str) else calls
        ids = ",".join(sorted(str(c.get("id")) for c in calls if isinstance(c, dict))) if isinstance(calls, list) else "?"
        rows[(r.get("role"), str(r.get("content") or ""), r.get("tool_call_id") or "", ids)] += 1
    return rows


def _trip(kind, engine, monkeypatch):
    if kind == "window-cap":
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1)
    elif kind == "tool-shape":
        engine._store.append_batch(SESSION, [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "odd", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}]}])
    elif kind == "read-error":
        def fail(*_a, **_k):
            raise OSError("injected presence read failure")
        monkeypatch.setattr(engine, "_rotation_predecessor_session_ids", fail)
    elif kind == "forced":
        monkeypatch.setattr(engine, "_identity_anchor_unaccounted", lambda *a, **k: None)


def _run(tmp_path, monkeypatch, caplog, kind):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / kind / "lcm.db"),
                                        large_output_externalization_enabled=False),
                       hermes_home=str(tmp_path / kind / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    seen, snapshots = [], []
    try:
        view = [_u("U0 first", 1000), _a("R0 reply"), _u("U1 second", 1001), _a("R1 reply"),
                _u("X steer note"), _a("RX reply"), _u("U2 third", 1002), _a("R2 reply")]
        engine.ingest(view)
        _trip(kind, engine, monkeypatch)
        real = engine._identity_anchor_unaccounted

        def spy(messages, identity_messages, cursor, start, identity_at, held, capped, consumed, plan, matched,
                excluded, missed):
            found = real(messages, identity_messages, cursor, start, identity_at, held, capped, consumed, plan,
                         matched, excluded, missed)
            seen.append((found, cursor, list(missed)))
            return found

        monkeypatch.setattr(engine, "_identity_anchor_unaccounted", spy)
        audit = engine._identity_anchor_audit

        def audit_spy(*args, **kwargs):
            before = len(seen)
            out = audit(*args, **kwargs)
            if len(seen) > before:  # this audit ran the presence check: its plan, before the R6 rewind moves it
                plan = args[7]
                snapshots.append((plan.get("cursor"), set(plan.get("positional", ()))))
            return out

        monkeypatch.setattr(engine, "_identity_anchor_audit", audit_spy)
        view[1] = _a("R0 a replaced reply")  # unstamped and replaced: the audit misses it
        view[4]["content"] = "continue with the plan" + PAD  # R6: the stored host object, rewritten in place
        view[6] = _u("U2 a replaced third prompt", 1003)
        view[2] = _u("U1 a replaced second prompt", 1004)  # missed before X, so X is in main's positional range
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="hermes_lcm.identity_anchor"):
            engine.ingest(view)
        records = [(r.levelno, r.getMessage()) for r in caplog.records]
        after_one = _multiset(engine)
        engine.ingest(view)
        return [(*entry, snap) for entry, snap in zip(seen, snapshots)], records, after_one, _multiset(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("kind", ["window-cap", "tool-shape", "read-error"])
def test_a_trip_keeps_mains_plan_with_an_r6_rewrite(tmp_path, monkeypatch, caplog, kind):
    seen, records, after_one, after_two = _run(tmp_path, monkeypatch, caplog, kind)
    assert seen, "the audit ran"
    found, cursor, missed, (plan_cursor, positional) = seen[0]  # the turn-end ingest
    assert found is None
    assert plan_cursor == min(missed)
    assert positional == {i for i in range(min(missed), cursor) if i not in missed}  # main's formula
    assert 4 in positional and min(missed) < 4  # the R6 row is inside the range, so the exclusion would show
    changed = [m for level, m in records if "host changed its list before the cursor" in m]
    assert changed and all("unstored rows from" in m for m in changed)
    assert not [m for level, m in records if level >= logging.WARNING and "host changed its list" in m]
    monkeypatch.undo()
    _seen, _records, forced_one, forced_two = _run(tmp_path, monkeypatch, caplog, "forced")
    if kind == "tool-shape":  # the extra stored row differs; compare the view's rows only
        odd = [k for k in after_one if k[3] == "odd"]
        for k in odd:
            after_one.pop(k)
            after_two.pop(k)
    assert (after_one, after_two) == (forced_one, forced_two)
