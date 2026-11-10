"""#2599: at default settings, a host saved-output marker whose payload cannot be written is stored as the host sent it.

Default settings externalize the content recovered from a Hermes saved-output marker. When that write fails (the payload
folder cannot be created, or the payload file write fails), ingest keeps the marker inline. v0.30.0-rc5 added LCM-X's
file-generation line to it; the session-end compare then never matched the host's marker, so an off-current end (or a
busy end's retried tail) lost the rows after it, usually the final reply, and a delegated child stored the result again
on each model call. The fix stores the marker bare in that case, as v0.29.2 does; the generation line stays for
externalization off.

Self-contained: helpers come only from test modules present on main and on release/v0.30.x.
"""

import copy
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from tests.invariants.test_p1e_cold_tails import fail_end_read
from tests.test_issue_10_persisted_output_spillover import _write_output
from tests.test_issue_1931_delegate_children import child_rows, host  # noqa: F401
from tests.test_issue_2055_ingest_ledger_writes import engines  # noqa: F401
from tests.test_issue_2482_interrupted_child import _setup as setup_child, call

CALL = "m1"
LCM_LINE = "[LCM persisted-output"
FAULTS = pytest.mark.parametrize("fault", ["write", "folder"])
KINDS = pytest.mark.parametrize("kind", ["spillover", "tempdir"])


@pytest.fixture(autouse=True)
def isolated_tempdir(tmp_path, monkeypatch):
    root = tmp_path / "tmpdir"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))


def _payload_fault(monkeypatch, engine, fault):
    """Fail externalization in the engine's own package copy (a delegated child's host loads a separate copy):
    ``write`` fails the payload file write; ``folder`` puts a regular file where the payload folder goes."""
    externalize = sys.modules[type(engine).__module__.rsplit(".", 1)[0] + ".externalize"]
    if fault == "write":
        def fails(*_a, **_k):
            raise OSError("injected payload write failure")
        monkeypatch.setattr(externalize, "_write_externalized_payload", fails)
        return
    folder = externalize.get_large_output_storage_dir(engine._config, hermes_home=str(engine._hermes_home),
                                                      create=False)
    if folder.is_dir():
        pytest.fail("setup: the payload folder already exists")
    folder.parent.mkdir(parents=True, exist_ok=True)
    folder.write_bytes(b"")


def _marker_path(kind, home, call=CALL):
    if kind == "spillover":
        return Path(home) / "cache" / "spillover" / f"{call}.txt"
    return Path(tempfile.gettempdir()) / "hermes-results" / f"{call}.txt"


def _history(marker):
    rows = [("user", "A"), ("assistant", "B"), ("user", "C")]
    msgs = [{"role": r, "content": t, "timestamp": float(i + 1)} for i, (r, t) in enumerate(rows)]
    msgs.append({"role": "assistant", "content": "", "timestamp": 4.0, "tool_calls": [
        {"id": CALL, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
    msgs.append({"role": "tool", "tool_call_id": CALL, "content": marker, "timestamp": 5.0})
    msgs += [{"role": r, "content": t, "timestamp": float(6 + i)} for i, (r, t) in
             enumerate((("assistant", "D"), ("user", "E"), ("assistant", "F"), ("user", "G"), ("assistant", "H")))]
    return msgs


def _assert_bare(stored, marker):
    assert stored.startswith("<persisted-output>") and LCM_LINE not in stored, "stored form is not the bare marker"
    assert stored == marker


def _offcurrent(make_engines, monkeypatch, kind, fault):
    engine = make_engines(sessions=[("S1", None, None), ("S2", None, None)])
    _payload_fault(monkeypatch, engine, fault)
    marker = _write_output(_marker_path(kind, engine._hermes_home))
    history = _history(marker)
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    engine.ingest(copy.deepcopy(history))
    rows = engine._store.get_session_messages("S1")
    assert len(rows) == len(history)
    _assert_bare([r for r in rows if r["role"] == "tool"][0]["content"], marker)
    engine.on_session_start("S2", platform="cli", context_length=200_000, conversation_id="conv2")
    view = copy.deepcopy(history) + [{"role": "assistant", "content": "FINAL", "timestamp": 11.0}]
    return engine, view, len(history)


def _final_and_rows(engine):
    rows = engine._store.get_session_messages("S1")
    return sum(r["content"] == "FINAL" for r in rows), len(rows)


@FAULTS
@KINDS
def test_an_off_current_end_after_a_payload_fault_stores_the_final_reply(engines, monkeypatch, kind, fault):  # noqa: F811
    engine, view, n = _offcurrent(engines, monkeypatch, kind, fault)
    engine.on_session_end("S1", view)
    assert _final_and_rows(engine) == (1, n + 1)


@FAULTS
@KINDS
def test_a_busy_end_retried_after_a_payload_fault_stores_the_final_reply(engines, monkeypatch, kind, fault):  # noqa: F811
    engine, view, n = _offcurrent(engines, monkeypatch, kind, fault)
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine))
        engine.on_session_end("S1", view)
    assert len(engine._pending_session_tails) == 1, "no cold tail queued"
    engine._retry_pending_session_tails()
    engine._retry_pending_session_tails()
    assert (*_final_and_rows(engine), len(engine._pending_session_tails)) == (1, n + 1, 0)


@FAULTS
@KINDS
def test_a_delegated_child_after_a_payload_fault_stores_its_result_once(host, monkeypatch, kind, fault):  # noqa: F811
    _, _, ctx, parent, child, engine = setup_child(host, "own_engine")
    _payload_fault(monkeypatch, engine, fault)
    marker = _write_output(_marker_path(kind, engine._hermes_home, "c1"))
    handed = [{"role": "user", "content": "Dump the ledger."}, call(1),
              {"role": "tool", "tool_call_id": "c1", "content": marker}]
    copies = []
    for n in range(3):
        child.api_call(ctx, handed, n + 1)
        rows = child_rows(parent)
        if n == 0:
            _assert_bare([r for r in rows if r["role"] == "tool"][0]["content"], marker)
        copies.append(Counter(r.get("tool_call_id") for r in rows if r["role"] == "tool")["c1"])
    child.end(engine, [*handed, {"role": "assistant", "content": "FINAL"}])
    rows = child_rows(parent)
    copies.append(Counter(r.get("tool_call_id") for r in rows if r["role"] == "tool")["c1"])
    assert (copies, len(rows), sum(r["content"] == "FINAL" for r in rows)) == ([1, 1, 1, 1], 4, 1)


@KINDS
def test_externalization_off_still_writes_the_generation_line(engines, kind):  # noqa: F811
    """Control: the change is limited to externalization on; at externalization off the inline marker keeps LCM-X's
    identity and generation lines (#2547's stored form)."""
    engine = engines(sessions=[("S1", None, None)])
    engine._config.large_output_externalization_enabled = False
    marker = _write_output(_marker_path(kind, engine._hermes_home))
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    engine.ingest(copy.deepcopy(_history(marker)))
    stored = [r for r in engine._store.get_session_messages("S1") if r["role"] == "tool"][0]["content"]
    assert "[LCM persisted-output marker identity:" in stored and "[LCM persisted-output file generation:" in stored


@KINDS
def test_without_a_fault_the_marker_is_externalized(engines, kind):  # noqa: F811
    """Control: with no fault the recovered content is externalized as before."""
    engine = engines(sessions=[("S1", None, None)])
    marker = _write_output(_marker_path(kind, engine._hermes_home))
    engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
    engine.ingest(copy.deepcopy(_history(marker)))
    stored = [r for r in engine._store.get_session_messages("S1") if r["role"] == "tool"][0]["content"]
    assert stored.startswith("[Externalized") and LCM_LINE not in stored


# Cold replays of a store written under a payload fault (each ingest a fresh engine, as at a restart), at default
# settings with the fault gone. Counts are (rows, tool rows); a replay that stores nothing again leaves (7, 1).
# The upgrade replay's keep for a bare stored marker covers spillover markers compared byte for byte on every branch,
# and on main also temp-folder markers and redacted stored text (#2584); without that keep the replay of a stampless
# history stores it again, the #2558 shape. v0.29.2 stores as many rows or more in every cell.
SECRET_BODY = "api_key=differentialsecret1234567890abcdef " + "z" * 2600
_REPLAY_PINS = {  # (kind, redacted preview, stamps): (with #2584's keep, without it, v0.29.2)
    ("spillover", False, "all"): ((7, 1), (7, 1), (7, 1)),
    ("spillover", False, "none"): ((7, 1), (7, 1), (19, 3)),
    ("spillover", True, "all"): ((7, 1), (7, 1), (7, 1)),
    ("spillover", True, "none"): ((12, 2), (12, 2), (19, 3)),
    ("tempdir", False, "all"): ((7, 1), (7, 1), (9, 2)),
    ("tempdir", False, "none"): ((7, 1), (12, 2), (12, 2)),
    ("tempdir", True, "all"): ((7, 1), (7, 1), (9, 2)),
    ("tempdir", True, "none"): ((12, 2), (12, 2), (12, 2)),
}


def _cold_ingest(home, redact, history):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    config = LCMConfig(database_path=str(home / "lcm.db"), large_output_externalization_path=str(home / "externalized"))
    config.sensitive_patterns_enabled = redact
    config.sensitive_patterns = ["all"] if redact else []
    engine = LCMEngine(config=config, hermes_home=str(home))
    try:
        engine.on_session_start("S1", platform="cli", context_length=200_000, conversation_id="conv1")
        engine.ingest(copy.deepcopy(history))
        return [(r["role"], r["content"]) for r in engine._store.get_session_messages("S1")], engine
    finally:
        engine.shutdown()


def _replay_history(marker, extra=0):
    rows = [("user", "A"), ("assistant", "B"), ("user", "C")]
    msgs = [{"role": r, "content": t, "timestamp": 1762700000.0 + i} for i, (r, t) in enumerate(rows)]
    msgs.append({"role": "assistant", "content": "", "timestamp": 1762700003.0, "tool_calls": [
        {"id": CALL, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]})
    msgs.append({"role": "tool", "tool_call_id": CALL, "content": marker, "timestamp": 1762700004.0})
    msgs += [{"role": ("user", "assistant")[i % 2], "content": f"later {i}", "timestamp": 1762700005.0 + i}
             for i in range(extra)]
    return msgs


@pytest.mark.parametrize("stamps", ["all", "none"])
@pytest.mark.parametrize("redact", [False, True], ids=["plain-preview", "redacted-preview"])
@KINDS
def test_cold_replays_of_a_store_written_under_a_payload_fault(tmp_path, monkeypatch, kind, redact, stamps):
    home = tmp_path / "hermes"
    marker = _write_output(_marker_path(kind, home), SECRET_BODY)
    with monkeypatch.context() as patch:
        from hermes_lcm import externalize

        def fails(*_a, **_k):
            raise OSError("injected payload write failure")
        patch.setattr(externalize, "_write_externalized_payload", fails)
        first, engine = _cold_ingest(home, redact, _replay_history(marker))
    tool = [content for role, content in first if role == "tool"][0]
    assert tool.startswith("<persisted-output>") and LCM_LINE not in tool, "stored form is not the bare marker"
    assert ("[LCM sensitive redaction:" in tool) is redact
    replay = _replay_history(marker, extra=2)
    if stamps == "none":
        for row in replay:
            row.pop("timestamp", None)
    for _ in range(2):
        rows, _engine = _cold_ingest(home, redact, replay)
    with_keep, without_keep, v0292 = _REPLAY_PINS[(kind, redact, stamps)]
    expected = with_keep if hasattr(type(engine), "_bare_marker_identity") else without_keep
    got = (len(rows), sum(role == "tool" for role, _content in rows))
    assert got == expected, f"got {got}; pinned {expected}; v0.29.2 {v0292}"
