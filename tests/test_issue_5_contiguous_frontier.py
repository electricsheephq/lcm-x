"""Contiguous compaction-frontier regressions for LCM-X issue #5."""

from __future__ import annotations

import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _summary(**_kwargs) -> tuple[str, int]:
    return "Earlier conversation. Expand for details: repeated turns", 1


def _engine(database_path, identity: str) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(database_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    engine.on_session_start(
        identity,
        platform="cli",
        conversation_id=identity,
        context_length=200_000,
    )
    return engine


def test_duplicate_active_identity_cannot_orphan_a_durable_occurrence(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-5-duplicate-frontier.db"
    identity = "issue-5-duplicate-frontier"
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)

    durable = [
        {"role": "assistant", "content": "durable occurrence before duplicate"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "fresh tail"},
    ]
    durable_ids = engine._store.append_batch(
        identity,
        durable,
        conversation_id=identity,
    )
    active = [
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": ""},
        durable[0],
        durable[2],
    ]
    engine._ingest_cursor = len(active)

    try:
        result = engine.compress(active)
        nodes = engine._dag.get_session_nodes(identity)
        state = engine._lifecycle.get_by_conversation(identity)
    finally:
        engine.shutdown()

    assert result == active
    assert nodes == []
    assert state is not None
    assert state.current_frontier_store_id == 0
    assert engine.last_compression_status == "error"
    assert engine.last_compression_noop_reason == (
        "summary publication could not prove contiguous source coverage"
    )
    published_frontier = state.current_frontier_store_id
    covered = {
        source_id
        for node in nodes
        if node.source_type == "messages"
        for source_id in node.source_ids
    }
    expected_covered = {
        store_id
        for store_id in durable_ids
        if store_id <= published_frontier
    }
    assert covered == expected_covered


def test_repeated_durable_content_publishes_a_contiguous_frontier(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-5-repeated-content.db"
    identity = "issue-5-repeated-content"
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    active = [
        {"role": "assistant", "content": "same"},
        {"role": "assistant", "content": "same"},
        {"role": "user", "content": "fresh tail"},
    ]
    durable_ids = engine._store.append_batch(
        identity,
        active,
        conversation_id=identity,
    )
    engine._ingest_cursor = len(active)

    try:
        engine.compress(active)
        nodes = engine._dag.get_session_nodes(identity)
        state = engine._lifecycle.get_by_conversation(identity)
    finally:
        engine.shutdown()

    assert state is not None
    assert state.current_frontier_store_id == durable_ids[1]
    assert len(nodes) == 1
    assert nodes[0].source_ids == durable_ids[:2]
    assert engine.last_compression_status == "compacted"


def test_session_contiguity_allows_interleaved_global_store_ids(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-5-interleaved-session.db"
    identity = "issue-5-interleaved-session"
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    active = [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "fresh tail"},
    ]
    first_id = engine._store.append_batch(
        identity,
        active[:1],
        conversation_id=identity,
    )[0]
    engine._store.append_batch(
        "another-session",
        [{"role": "assistant", "content": "unrelated"}],
        conversation_id="another-conversation",
    )
    remaining_ids = engine._store.append_batch(
        identity,
        active[1:],
        conversation_id=identity,
    )
    engine._ingest_cursor = len(active)

    try:
        engine.compress(active)
        nodes = engine._dag.get_session_nodes(identity)
        state = engine._lifecycle.get_by_conversation(identity)
    finally:
        engine.shutdown()

    assert state is not None
    assert state.current_frontier_store_id == remaining_ids[0]
    assert len(nodes) == 1
    assert nodes[0].source_ids == [first_id, remaining_ids[0]]
    assert engine.last_compression_status == "compacted"


def test_stale_expected_frontier_rolls_back_node_publication(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-5-stale-frontier.db"
    identity = "issue-5-stale-frontier"
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    active = [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "fresh tail"},
    ]
    durable_ids = engine._store.append_batch(
        identity,
        active,
        conversation_id=identity,
    )
    engine._ingest_cursor = len(active)
    original_add_node = engine._dag.add_node

    def advance_then_add(node, *, before_commit=None):
        engine._lifecycle.advance_frontier(identity, identity, durable_ids[0])
        return original_add_node(node, before_commit=before_commit)

    monkeypatch.setattr(engine._dag, "add_node", advance_then_add)

    try:
        result = engine.compress(active)
        nodes = engine._dag.get_session_nodes(identity)
        state = engine._lifecycle.get_by_conversation(identity)
    finally:
        engine.shutdown()

    assert result == active
    assert nodes == []
    assert state is not None
    assert state.current_frontier_store_id == durable_ids[0]
    assert engine.last_compression_status == "error"


def test_foreign_conversation_rows_cannot_advance_the_frontier(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "issue-5-foreign-conversation.db"
    identity = "issue-5-foreign-conversation"
    engine = _engine(database_path, identity)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    active = [
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "fresh tail"},
    ]
    engine._store.append_batch(
        identity,
        active,
        conversation_id="another-conversation",
    )
    engine._ingest_cursor = len(active)

    try:
        result = engine.compress(active)
        nodes = engine._dag.get_session_nodes(identity)
        state = engine._lifecycle.get_by_conversation(identity)
    finally:
        engine.shutdown()

    assert result == active
    assert nodes == []
    assert state is not None
    assert state.current_frontier_store_id == 0
    assert engine.last_compression_status == "error"
    assert engine.last_compression_noop_reason == (
        "summary publication could not prove contiguous source coverage"
    )
