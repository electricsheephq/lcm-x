"""#771: surviving summaries must keep their finalized frontier resumable."""
import json

import pytest

import hermes_lcm.lifecycle_state as lifecycle_module
from tests.test_issue_752_reset_resume_frontier import _engine, _ingest_turns, _summarize

import hermes_lcm.engine as engine_module


def _seed(tmp_path, monkeypatch, retain=-1):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _summarize)
    engine = _engine(tmp_path, new_session_retain_depth=retain)
    engine.on_session_start("S1", platform="cli", context_length=200_000)
    host = []
    _ingest_turns(engine, host, range(1, 13))
    engine.compress(list(host), force=True)
    assert engine._last_compacted_store_id == 18
    return engine, host


def _leaves(engine):
    return sorted(
        (min(node.source_ids), max(node.source_ids))
        for node in engine._dag.get_session_nodes("S1")
        if node.depth == 0 and node.source_ids
    )


def _restart(tmp_path, engine, retain):
    engine.shutdown()
    resumed = _engine(tmp_path, new_session_retain_depth=retain)
    resumed.on_session_start("S1", platform="cli", context_length=200_000)
    return resumed


@pytest.mark.parametrize("reset", [True, False], ids=["T1-retain-all-reset", "T2-no-reset-control"])
def test_four_compactions_after_resume(tmp_path, monkeypatch, reset, caplog):
    engine, host = _seed(tmp_path, monkeypatch)
    engine.on_session_end("S1", host)
    if reset:
        engine.on_session_reset()
    resumed = _restart(tmp_path, engine, -1)
    try:
        bound_frontier = resumed._last_compacted_store_id
        statuses = []
        for first in (13, 17, 21, 25):
            _ingest_turns(resumed, host, range(first, first + 4))
            resumed.compress(list(host), force=True)
            statuses.append(resumed._last_compression_status)
        leaves = _leaves(resumed)
        conflicts = [record.getMessage() for record in caplog.records if "publication_invariant_conflict" in record.getMessage()]
        print(json.dumps({"reset": reset, "bound_frontier": bound_frontier, "statuses": statuses,
                          "leaves": leaves, "publication_conflicts": conflicts}))
        # Assert only after measuring every iteration: base evidence must show all four errors.
        assert bound_frontier == 18
        assert statuses == ["compacted"] * 4
        assert leaves == [(1, 18), (19, 26), (27, 34), (35, 42), (43, 50)]
    finally:
        resumed.shutdown()


@pytest.mark.parametrize("retain", [0, 2], ids=["T3-retain-0", "T3-retain-2"])
def test_successful_deletion_still_binds_zero(tmp_path, monkeypatch, retain):
    engine, host = _seed(tmp_path, monkeypatch, retain)
    engine.on_session_end("S1", host)
    engine.on_session_reset()
    resumed = _restart(tmp_path, engine, retain)
    try:
        state = resumed._lifecycle.get_by_conversation("S1")
        print(json.dumps({"retain": retain, "bound_frontier": resumed._last_compacted_store_id,
                          "leaves": _leaves(resumed)}))
        assert resumed._last_compacted_store_id == 0
        assert _leaves(resumed) == []
        assert state.last_finalized_frontier_store_id == 18
        assert state.last_finalized_at < state.last_reset_at
    finally:
        resumed.shutdown()


def _fail_deletion(*args, **kwargs):
    raise engine_module.sqlite3.OperationalError("synthetic deletion failure")


def test_failed_deletion_keeps_resume_and_next_compaction(tmp_path, monkeypatch):
    engine, host = _seed(tmp_path, monkeypatch, 2)
    engine.on_session_end("S1", host)
    monkeypatch.setattr(engine._dag, "delete_below_depth", _fail_deletion)
    with pytest.raises(engine_module.sqlite3.OperationalError, match="synthetic deletion failure"):
        engine.on_session_reset()
    resumed = _restart(tmp_path, engine, 2)
    try:
        frontier = resumed._last_compacted_store_id
        _ingest_turns(resumed, host, range(13, 17))
        resumed.compress(list(host), force=True)
        print(json.dumps({"deletion": "OperationalError", "bound_frontier": frontier,
                          "status": resumed._last_compression_status, "leaves": _leaves(resumed)}))
        assert frontier == 18
        assert resumed._last_compression_status == "compacted"
        assert _leaves(resumed) == [(1, 18), (19, 26)]
    finally:
        resumed.shutdown()


@pytest.mark.parametrize("retain,fail", [(-1, False), (0, False), (2, False), (2, True)],
                         ids=["T5-retain-all", "T5-retain-0", "T5-retain-2", "T5-deletion-failure"])
def test_reset_epoch_is_unchanged(tmp_path, monkeypatch, retain, fail):
    engine, host = _seed(tmp_path, monkeypatch, retain)
    try:
        engine.on_session_end("S1", host)
        before = engine._emission_binding()
        if fail:
            monkeypatch.setattr(engine._dag, "delete_below_depth", _fail_deletion)
        with monkeypatch.context() as clock:
            clock.setattr(lifecycle_module.time, "time", lambda: 2_000_000_000.0)
            if fail:
                with pytest.raises(engine_module.sqlite3.OperationalError):
                    engine.on_session_reset()
            else:
                engine.on_session_reset()
        state = engine._lifecycle.get_by_conversation("S1")
        after = engine._emission_binding()
        print(json.dumps({"retain": retain, "failed_deletion": fail, "previous_reset_epoch": before["reset_epoch"],
                          "last_reset_at": state.last_reset_at, "reset_epoch": after["reset_epoch"],
                          "last_finalized_at": state.last_finalized_at}))
        assert before["reset_epoch"] is None
        assert after["reset_epoch"] == state.last_reset_at == 2_000_000_000.0
        assert {k: v for k, v in before.items() if k != "reset_epoch"} == {
            k: v for k, v in after.items() if k != "reset_epoch"
        }
    finally:
        engine.shutdown()


def test_acp_reset_only_same_session_full_host_view(tmp_path, monkeypatch):
    # ACP calls reset without end or a new session id, then reuses the full host view.
    engine, host = _seed(tmp_path, monkeypatch)
    try:
        engine.on_session_reset()
        state = engine._lifecycle.get_by_conversation("S1")
        zeroed_marker = engine._last_compacted_store_id
        _ingest_turns(engine, host, range(13, 17))
        engine.compress(list(host), force=True)
        print(json.dumps({"ACP_reset_only": True, "zeroed_marker": zeroed_marker,
                          "current_session_id": state.current_session_id,
                          "current_frontier": state.current_frontier_store_id,
                          "status": engine._last_compression_status, "leaves": _leaves(engine)}))
        assert zeroed_marker == 0
        assert state.current_session_id == "S1"
        # Measurement only: the finalized-row fix deliberately leaves active rows unchanged.
        assert engine._last_compression_status == "error"
        assert _leaves(engine) == [(1, 18)]
    finally:
        engine.shutdown()


def test_failed_second_reset_does_not_revive_deleted_frontier(tmp_path, monkeypatch):
    engine, host = _seed(tmp_path, monkeypatch, 2)
    engine.on_session_end("S1", host)
    with monkeypatch.context() as clock:
        clock.setattr(lifecycle_module.time, "time", lambda: 2_000_000_010.0)
        engine.on_session_reset()
    before = engine._lifecycle.get_by_conversation("S1")
    monkeypatch.setattr(engine._dag, "delete_below_depth", _fail_deletion)
    with monkeypatch.context() as clock:
        clock.setattr(lifecycle_module.time, "time", lambda: 2_000_000_020.0)
        with pytest.raises(engine_module.sqlite3.OperationalError, match="synthetic deletion failure"):
            engine.on_session_reset()
    after = engine._lifecycle.get_by_conversation("S1")
    resumed = _restart(tmp_path, engine, 2)
    try:
        print(json.dumps({"repeated_reset": True, "leaves": _leaves(resumed),
                          "bound_frontier": resumed._last_compacted_store_id,
                          "last_finalized_at": after.last_finalized_at, "reset_epoch": after.last_reset_at}))
        assert before.last_finalized_at == after.last_finalized_at
        assert after.last_finalized_at < after.last_reset_at
        assert resumed._last_compacted_store_id == 0
        assert _leaves(resumed) == []
        _ingest_turns(resumed, host, range(13, 17))
        resumed.compress(list(host), force=True)
        assert resumed._last_compression_status == "compacted"
        assert _leaves(resumed) == [(1, 26)]
    finally:
        resumed.shutdown()


def test_partial_committed_deletion_does_not_revive_frontier(tmp_path, monkeypatch):
    engine, host = _seed(tmp_path, monkeypatch, 2)
    engine.on_session_end("S1", host)
    before = engine._lifecycle.get_by_conversation("S1")
    original = engine._dag.delete_node_batch
    calls = []
    deleted_ids = []

    def fail_second_batch(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise engine_module.sqlite3.OperationalError("synthetic next-batch lock after committed deletion")
        ids = original(*args, **kwargs)
        deleted_ids.extend(ids)
        return ids

    assert not engine._config.embeddings_enabled
    monkeypatch.setattr(engine._dag, "delete_node_batch", fail_second_batch)
    with pytest.raises(engine_module.sqlite3.OperationalError, match="synthetic next-batch lock after committed deletion"):
        engine.on_session_reset()
    after = engine._lifecycle.get_by_conversation("S1")
    dropped = engine._pending_reset_drops_summaries
    resumed = _restart(tmp_path, engine, 2)
    try:
        print(json.dumps({"partial_deletion": True, "committed_deleted_count": len(deleted_ids),
                          "pending_drops_flag": dropped, "bound_frontier": resumed._last_compacted_store_id,
                          "leaves": _leaves(resumed), "finalized_unchanged": after.last_finalized_at == before.last_finalized_at}))
        assert len(calls) == 2 and len(deleted_ids) == 1
        assert after.last_finalized_at == before.last_finalized_at
        assert after.last_finalized_at < after.last_reset_at
        assert resumed._last_compacted_store_id == 0
        assert _leaves(resumed) == []
        _ingest_turns(resumed, host, range(13, 17))
        resumed.compress(list(host), force=True)
        assert resumed._last_compression_status == "compacted"
        assert _leaves(resumed) == [(1, 26)]
    finally:
        resumed.shutdown()
