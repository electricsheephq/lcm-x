"""#752: resuming a session after /new must not restore a frontier whose summaries the reset deleted.

Real engine, lifecycle and store code; only the summariser (the model call) is faked.
"""
import re

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _turn(i):
    return (
        {"role": "user", "content": f"[T{i:02d}] user turn {i}: " + "alpha beta gamma delta " * 40},
        {"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}."},
    )


def _summarize(*args, **kwargs):
    turns = sorted(set(re.findall(r"\[(T\d\d)\] user", kwargs.get("text") or "")))
    return "Summary covers " + " ".join(turns) + ".", 1


def _engine(tmp_path, **overrides):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=6,
        leaf_chunk_tokens=400,
        large_output_externalization_path=str(tmp_path / "externalized"),
        **overrides,
    )
    return LCMEngine(config=config, hermes_home=str(tmp_path / "home"))


def _ingest_turns(engine, host, turns):
    for i in turns:
        user, reply = _turn(i)
        host.append(user)
        engine.ingest(host)
        host.append(reply)
        engine.ingest(host)


def _compact_then_new_then_resume(tmp_path, monkeypatch, *, end_first=True, restart=True, **overrides):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _summarize)
    engine = _engine(tmp_path, **overrides)
    engine.on_session_start("S1", platform="cli", context_length=200_000)
    host = []
    _ingest_turns(engine, host, range(1, 13))
    engine.compress(list(host), force=True)  # d0 leaf over T01-T09 (store ids 1-18)
    assert engine._last_compacted_store_id == 18
    if end_first:
        engine.on_session_end("S1", host)
    engine.on_session_reset()
    engine.on_session_start("S2", platform="cli", context_length=200_000)
    if restart:
        engine.shutdown()
        engine = _engine(tmp_path, **overrides)
    engine.on_session_start("S1", platform="cli", context_length=200_000)
    return engine


def _covered_max(engine, session_id="S1"):
    return max((sid for node in engine._dag.get_session_nodes(session_id) for sid in node.source_ids), default=0)


@pytest.mark.parametrize(
    "end_first,restart,retain",
    [(True, True, 2), (False, True, 2), (True, False, 2), (True, True, 0)],
    ids=["end-reset-start-restart", "no-end", "same-engine", "retain-0"],
)
def test_resume_after_new_binds_at_zero_when_the_reset_dropped_the_summaries(
    tmp_path, monkeypatch, end_first, restart, retain
):
    resumed = _compact_then_new_then_resume(
        tmp_path, monkeypatch, end_first=end_first, restart=restart, new_session_retain_depth=retain
    )
    try:
        assert _covered_max(resumed) == 0
        assert resumed._last_compacted_store_id == 0
        state = resumed._lifecycle.get_by_conversation("S1")
        assert state.current_session_id == "S1"
        assert state.last_finalized_session_id == "S1"
        assert state.last_finalized_frontier_store_id == 18  # the recorded frontier is kept as metadata
        assert state.last_finalized_at < state.last_reset_at  # but it is not resumable
    finally:
        resumed.shutdown()


@pytest.mark.parametrize("resumed_view", ["new-turns-only", "full-raw-transcript"])
def test_next_leaf_after_resume_covers_the_pre_new_user_turns(tmp_path, monkeypatch, resumed_view):
    resumed = _compact_then_new_then_resume(tmp_path, monkeypatch)
    try:
        host = [] if resumed_view == "new-turns-only" else [m for i in range(1, 13) for m in _turn(i)]
        _ingest_turns(resumed, host, range(13, 21))
        context = resumed.compress(list(host), force=True)
        assert resumed._last_compression_status == "compacted"
        covered = {
            sid for node in resumed._dag.get_session_nodes("S1") if node.depth == 0 for sid in node.source_ids
        }
        assert 1 in covered
        shown = {str(message.get("content")) for message in context}
        user_rows = resumed._store._conn.execute(
            "SELECT store_id, content FROM messages WHERE session_id = 'S1' AND role = 'user' AND store_id <= ?",
            (max(covered),),
        ).fetchall()
        skipped = [c[1:4] for sid, c in user_rows if sid not in covered and c not in shown]
        assert skipped == []
    finally:
        resumed.shutdown()


def test_retain_all_keeps_the_resumable_frontier(tmp_path, monkeypatch):
    resumed = _compact_then_new_then_resume(tmp_path, monkeypatch, new_session_retain_depth=-1)
    try:
        assert _covered_max(resumed) == 18
        assert resumed._last_compacted_store_id == 18
    finally:
        resumed.shutdown()


def test_frontier_published_after_the_reset_stays_resumable(tmp_path):
    engine = _engine(tmp_path)
    try:
        engine.on_session_start("S1", platform="cli", context_length=200_000)
        first = engine._store.append("S1", {"role": "user", "content": "before reset"}, token_estimate=7, source="cli")
        engine._last_compacted_store_id = first
        engine.on_session_reset()
        second = engine._store.append(
            "S1", {"role": "assistant", "content": "same session continued after reset"}, token_estimate=11, source="cli"
        )
        engine._last_compacted_store_id = second
        engine._persist_frontier_marker()
        engine.on_session_start("S2", platform="cli", context_length=200_000)
        state = engine._lifecycle.get_by_conversation("S1")
        assert state.last_finalized_frontier_store_id == second
        assert state.last_finalized_at >= state.last_reset_at
        engine.on_session_start("S1", platform="cli", context_length=200_000)
        assert engine._last_compacted_store_id == second
    finally:
        engine.shutdown()


def test_in_place_compression_boundary_after_resume_keeps_frontier_zero(tmp_path, monkeypatch):
    resumed = _compact_then_new_then_resume(tmp_path, monkeypatch)
    try:
        resumed.on_session_start(
            "S1", boundary_reason="compression", old_session_id="S1", platform="cli", context_length=200_000
        )
        assert resumed._last_compacted_store_id == 0
    finally:
        resumed.shutdown()


@pytest.mark.parametrize("retain", [2, 0])
def test_a_reset_whose_deletion_fails_keeps_the_frontier_resumable(tmp_path, monkeypatch, retain):
    # The summaries survive a failed deletion, so the frontier must stay resumable as on main;
    # binding at 0 would make every later leaf collide with the surviving one.
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _summarize)
    engine = _engine(tmp_path, new_session_retain_depth=retain)
    engine.on_session_start("S1", platform="cli", context_length=200_000)
    host = []
    _ingest_turns(engine, host, range(1, 13))
    engine.compress(list(host), force=True)
    assert engine._last_compacted_store_id == 18
    engine.on_session_end("S1", host)

    def _locked(*args, **kwargs):
        raise lcm_engine_module.sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine._dag, "delete_below_depth", _locked)
    monkeypatch.setattr(engine._dag, "delete_session_nodes", _locked)
    with pytest.raises(lcm_engine_module.sqlite3.OperationalError):
        engine.on_session_reset()
    monkeypatch.undo()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _summarize)
    engine.on_session_start("S2", platform="cli", context_length=200_000)
    engine.shutdown()
    resumed = _engine(tmp_path, new_session_retain_depth=retain)
    try:
        resumed.on_session_start("S1", platform="cli", context_length=200_000)
        assert _covered_max(resumed) == 18
        assert resumed._last_compacted_store_id == 18
        _ingest_turns(resumed, host, range(13, 21))
        resumed.compress(list(host), force=True)
        assert resumed._last_compression_status == "compacted"
    finally:
        resumed.shutdown()
