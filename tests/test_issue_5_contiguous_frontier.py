"""Contiguous compaction-frontier regressions for LCM-X issue #5."""

from __future__ import annotations

import hermes_lcm.engine as lcm_engine
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError


def _summary(**_kwargs) -> tuple[str, int]:
    return "Earlier conversation. Expand for details: repeated turns", 1


class _ContainsPattern:
    pattern = "DROP_ME"

    def search(self, text, timeout=None):
        del timeout
        return object() if self.pattern in str(text) else None


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


def test_later_publication_conflict_reassembles_committed_leaves(
    tmp_path,
    monkeypatch,
) -> None:
    identity = "issue-5-later-conflict"
    engine = _engine(tmp_path / "issue-5-later-conflict.db", identity)
    engine._config.threshold_full_sweep_enabled = True
    engine.threshold_tokens = 1
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    active = [
        {"role": "assistant", "content": f"historical-{index}"}
        for index in range(4)
    ] + [{"role": "user", "content": "fresh tail"}]
    engine._store.append_batch(identity, active, conversation_id=identity)
    engine._ingest_cursor = len(active)
    original_stage = engine._lifecycle.stage_compaction_publication
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise LifecyclePublicationConflictError("injected later conflict")
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(
        engine._lifecycle,
        "stage_compaction_publication",
        fail_second,
    )

    try:
        result = engine.compress(active, current_tokens=100)
        nodes = engine._dag.get_session_nodes(identity)
    finally:
        engine.shutdown()

    assert calls == 2
    assert len(nodes) == 1
    assert "Earlier conversation" in "\n".join(
        str(message.get("content") or "") for message in result
    )
    assert engine.last_compression_status == "error"


@pytest.mark.parametrize("rotate_ahead", [False, True])
def test_filter_exclusion_content_is_revalidated_in_publication_snapshot(
    tmp_path,
    monkeypatch,
    rotate_ahead,
) -> None:
    identity = "issue-5-filter-race"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "issue-5-filter-race.db"),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    engine._compiled_ignore_message_patterns = [_ContainsPattern()]
    engine.on_session_start(
        identity,
        platform="cli",
        conversation_id=identity,
        context_length=200_000,
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    active = [
        {"role": "assistant", "content": "DROP_ME"},
        {"role": "assistant", "content": "covered"},
        {"role": "user", "content": "fresh tail"},
    ]
    durable_ids = engine._store.append_batch(
        identity,
        active,
        conversation_id=identity,
    )
    expected_frontier = durable_ids[-1] if rotate_ahead else 0
    if rotate_ahead:
        engine._lifecycle.advance_frontier(
            identity,
            identity,
            expected_frontier,
        )
    engine._ingest_cursor = len(active)
    original_scan = engine._stored_publication_filter_exclusions

    def rewrite_then_scan(*args, **kwargs):
        engine._store.connection.execute(
            "UPDATE messages SET content = ? WHERE store_id = ?",
            ("no longer filtered", durable_ids[0]),
        )
        engine._store.connection.commit()
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(
        engine,
        "_stored_publication_filter_exclusions",
        rewrite_then_scan,
    )

    try:
        engine.compress(active)
        nodes = engine._dag.get_session_nodes(identity)
        state = engine._lifecycle.get_by_conversation(identity)
    finally:
        engine.shutdown()

    assert nodes == []
    assert state is not None
    assert state.current_frontier_store_id == expected_frontier
    assert engine.last_compression_status == "error"


def test_filter_exclusion_scan_pages_to_the_covered_end(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path / "issue-5-filter-pages.db", "issue-5-filter-pages")
    engine._compiled_ignore_message_patterns = [_ContainsPattern()]
    first_page = [
        {"store_id": store_id, "content": "keep"}
        for store_id in range(1, 10_001)
    ]
    last_page = [{"store_id": 10_001, "content": "DROP_ME"}]
    calls = []

    def paged_rows(_session_id, after_store_id=0, limit=10_000):
        calls.append((after_store_id, limit))
        return first_page if after_store_id == 0 else last_page

    monkeypatch.setattr(engine._store, "get_session_messages_after", paged_rows)
    try:
        proofs = engine._stored_publication_filter_exclusions(
            0,
            10_001,
            [],
            {},
        )
    finally:
        engine.shutdown()

    assert calls == [(0, 10_000), (10_000, 10_000)]
    assert proofs == {10_001: "DROP_ME"}


def test_below_frontier_source_lineage_is_claimed_once(tmp_path) -> None:
    identity = "issue-5-below-frontier-claim"
    engine = _engine(tmp_path / "issue-5-below-frontier-claim.db", identity)
    source_id = engine._store.append(
        identity,
        {"role": "assistant", "content": "covered before rotate"},
        conversation_id=identity,
    )
    engine._lifecycle.advance_frontier(identity, identity, source_id)

    def publish() -> None:
        node = SummaryNode(
            session_id=identity,
            summary="claimed once",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
        )

        def stage(conn, node_id) -> None:
            engine._lifecycle.stage_compaction_publication(
                conn,
                identity,
                identity,
                node_id,
                source_id,
                [source_id],
            )

        engine._dag.add_node(node, before_commit=stage)

    try:
        publish()
        with pytest.raises(
            LifecyclePublicationConflictError,
            match="already claimed",
        ):
            publish()
        nodes = engine._dag.get_session_nodes(identity)
    finally:
        engine.shutdown()

    assert len(nodes) == 1


def test_excluded_rows_must_share_conversation_ownership(tmp_path) -> None:
    identity = "issue-5-excluded-ownership"
    engine = _engine(tmp_path / "issue-5-excluded-ownership.db", identity)
    covered_start = engine._store.append(
        identity,
        {"role": "assistant", "content": "covered start"},
        conversation_id=identity,
    )
    excluded = engine._store.append(
        identity,
        {"role": "assistant", "content": "DROP_ME"},
        conversation_id="foreign-conversation",
    )
    covered_end = engine._store.append(
        identity,
        {"role": "assistant", "content": "covered end"},
        conversation_id=identity,
    )
    node = SummaryNode(
        session_id=identity,
        summary="must not cross foreign exclusion",
        token_count=1,
        source_token_count=1,
        source_ids=[covered_start, covered_end],
    )

    def stage(conn, node_id) -> None:
        engine._lifecycle.stage_compaction_publication(
            conn,
            identity,
            identity,
            node_id,
            0,
            [covered_start, covered_end],
            [excluded],
            {excluded: "DROP_ME"},
        )

    try:
        with pytest.raises(
            LifecyclePublicationConflictError,
            match="source ownership changed",
        ):
            engine._dag.add_node(node, before_commit=stage)
        nodes = engine._dag.get_session_nodes(identity)
    finally:
        engine.shutdown()

    assert nodes == []
