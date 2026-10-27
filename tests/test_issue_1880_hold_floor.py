"""#1880: locally armed holds keep their floor; restart proof is session-scoped."""
import copy
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.dag import SummaryNode
from tests.test_issue_1880_persisted_hold import (
    KEY, _r1_tail_engine, _r1_views, _reorder, arm, isolated as isolated, rebuild,
)


@pytest.fixture
def clock(monkeypatch):
    wall, mono = engine_module.time.time(), engine_module.time.monotonic()
    age = [0]
    monkeypatch.setattr(engine_module.time, "time", lambda: wall + age[0])
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: mono + age[0])
    return age


def local_hold(path):
    engine, view = _reorder(path, False)
    engine._config.fresh_tail_pressure_yield_enabled = False
    engine._config.empty_lifecycle_gc_enabled = False
    view = engine.compress(view, current_tokens=10_000)
    engine._no_progress_hold = None
    view = engine.compress(view, current_tokens=10_000)
    assert engine._last_compress_leaves[1] == 0
    assert engine._no_progress_hold[1] == "no_progress"
    assert engine._store.read_metadata_json(KEY)["reason"] == "no_progress"
    return engine, view


def extended(view):
    return copy.deepcopy(view) + [
        {"role": "user", "content": "new synthetic objective " * 100},
        {"role": "assistant", "content": "new synthetic reply " * 100},
    ]


def test_local_floor_survives_ingest_and_pressure(tmp_path, monkeypatch, clock):
    engine, view = local_hold(tmp_path)
    try:
        deadline = engine._no_progress_hold[0]
        until = engine._store.read_metadata_json(KEY)["expires_at"]
        before = engine._store.max_store_id()
        engine._ingest_messages(extended(view))
        assert engine._store.max_store_id() > before
        engine._no_progress_observed_tokens = engine.last_prompt_tokens = 10_001
        read = Mock(side_effect=AssertionError("local floor read metadata"))
        fingerprint = Mock(side_effect=AssertionError("local floor computed fingerprint"))
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "read_metadata_json", read)
            patch.setattr(engine, "_no_progress_fingerprint", fingerprint)
            for age in (1, 300, 599):
                clock[0] = age
                assert engine._no_progress_hold_active()
                assert engine._no_progress_hold[0] == deadline
                assert engine._no_progress_hold_status() == {"reason": "no_progress", "until": until}
            read.assert_not_called()
            fingerprint.assert_not_called()
        clock[0] = 601
        assert not engine._no_progress_hold_active()
    finally:
        engine.shutdown()


@pytest.mark.parametrize("other", ["session", "summary"])
def test_restart_ignores_other_session_activity(tmp_path, clock, other):
    # v0.29.1 #1880: restores the v0.28.1 local hold floor
    view, _ = arm(tmp_path)
    writer = rebuild(tmp_path)
    try:
        if other == "summary":
            writer._dag.add_node(SummaryNode(session_id="other", depth=0, summary="Synthetic summary",
                                            token_count=2, source_ids=[], source_type="messages"))
        else:
            writer._store.append("other",
                                 {"role": "user", "content": "other synthetic turn"},
                                 conversation_id="conv")
    finally:
        writer.shutdown()
    clock[0] = 100
    engine = rebuild(tmp_path)
    try:
        engine._ingest_messages(view)
        assert not engine.should_compress(10_000)
        assert engine._no_progress_hold_active()
    finally:
        engine.shutdown()


@pytest.mark.parametrize("changed", [False, True])
def test_after_local_deadline_consults_persisted_proof(tmp_path, monkeypatch, clock, changed):
    engine, view = local_hold(tmp_path)
    try:
        record = engine._store.read_metadata_json(KEY)
        # Exercise expiry of a shorter local floor while durable time remains.
        engine._no_progress_hold = (engine._no_progress_hold[0] - 300, "no_progress")
        engine._no_progress_local_hold = engine._no_progress_hold
        if changed:
            engine._ingest_messages(extended(view))
        read = Mock(wraps=engine._store.read_metadata_json)
        monkeypatch.setattr(engine._store, "read_metadata_json", read)
        clock[0] = 301
        assert engine._no_progress_hold_active() is (not changed)
        assert read.call_count == 1
        assert engine._store.read_metadata_json(KEY) == record
        clock[0] = 601
        assert not engine._no_progress_hold_active()
    finally:
        engine.shutdown()


def test_stored_leaf_clears_local_floor_and_record(tmp_path, clock):
    engine = _r1_tail_engine(tmp_path)
    view, longer = _r1_views()
    try:
        engine.compress(view, current_tokens=10_000)
        assert engine._no_progress_hold_active()
        assert engine._store.read_metadata_json(KEY) is not None
        clock[0] = 1
        engine.compress(longer, current_tokens=10_000, force=True)
        assert engine._last_compress_leaves[1] >= 1
        assert engine._no_progress_hold is None
        assert engine._store.read_metadata_json(KEY) is None
        assert not engine._no_progress_hold_active()
    finally:
        engine.shutdown()


def test_compress_held_after_new_turn(tmp_path, monkeypatch, clock):
    engine, view = local_hold(tmp_path)
    try:
        clock[0] = 100
        view = extended(view)
        engine._ingest_messages(view)
        scan = Mock(wraps=engine._prepare_compression_sweep)
        monkeypatch.setattr(engine, "_prepare_compression_sweep", scan)
        assert engine.compress(view, current_tokens=10_001) == view
        assert engine._last_compression_status == "sanitized"
        assert engine._last_compress_leaves[1] == 0
        scan.assert_not_called()
    finally:
        engine.shutdown()


def test_scoped_maxima_use_indexes(tmp_path):
    engine = rebuild(tmp_path)
    try:
        message_plan = engine._store._conn.execute(
            "EXPLAIN QUERY PLAN SELECT COALESCE(MAX(store_id), 0) FROM messages "
            "WHERE conversation_id = ? AND session_id = ?", ("conv", "S")).fetchall()
        node_plan = engine._dag._conn.execute(
            "EXPLAIN QUERY PLAN SELECT COALESCE(MAX(node_id), 0) FROM summary_nodes "
            "WHERE session_id = ?", ("S",)).fetchall()
        assert any("idx_msg_conversation_session" in row[3] for row in message_plan)
        assert any("idx_nodes_session_node" in row[3] for row in node_plan)
    finally:
        engine.shutdown()
