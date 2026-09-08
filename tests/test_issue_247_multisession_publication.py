"""Real-engine regression coverage for issue #247 session rollover."""

from __future__ import annotations

import hermes_lcm.engine as lcm_engine
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError


def _summary(**_kwargs) -> tuple[str, int]:
    return "Earlier conversation. Expand for details: turns", 1


def test_publication_credits_only_a_retained_leading_summary_prefix(
    tmp_path,
    monkeypatch,
) -> None:
    conversation_id = "issue-247-leading-prefix"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "leading-prefix.db"),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    engine.on_session_start(
        "current-session",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    prior_ids = engine._store.append_batch(
        "current-session",
        [
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ],
        conversation_id=conversation_id,
    )
    prior_node = SummaryNode(
        session_id="current-session",
        depth=0,
        summary="Retained prior summary.",
        token_count=4,
        source_token_count=4,
        source_ids=prior_ids,
        source_type="messages",
    )
    engine._dag.add_node(prior_node)
    before_node_count = len(engine._dag.get_session_nodes("current-session"))
    context = [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "fresh tail"},
    ]

    try:
        result = engine.compress(context, force=True)
        state = engine._lifecycle.get_by_conversation(conversation_id)
        nodes = engine._dag.get_session_nodes("current-session")
        new_nodes = [node for node in nodes if node.node_id != prior_node.node_id]
        assembled = engine._assemble_context(None, result[-1:])
    finally:
        engine.shutdown()

    assert engine.last_compression_status == "compacted"
    assert state is not None and state.current_frontier_store_id > max(prior_ids)
    assert len(nodes) == before_node_count + 1
    assert len(new_nodes) == 1
    assert set(new_nodes[0].source_ids).isdisjoint(prior_ids)
    assembled_text = "\n".join(str(message.get("content") or "") for message in assembled)
    assert "Retained prior summary." in assembled_text


@pytest.mark.parametrize("prior_state", ["missing_root", "interior_hole"])
def test_publication_rejects_unretained_or_incomplete_prior_coverage(
    tmp_path,
    monkeypatch,
    prior_state,
) -> None:
    conversation_id = f"issue-247-negative-{prior_state}"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / f"{prior_state}.db"),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    engine.on_session_start(
        "current-session",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    prior_ids = engine._store.append_batch(
        "current-session",
        [
            {"role": "user", "content": "prior one"},
            {"role": "assistant", "content": "prior two"},
            {"role": "user", "content": "prior three"},
        ],
        conversation_id=conversation_id,
    )
    prior_node = SummaryNode(
        session_id=(
            "detached-session" if prior_state == "missing_root" else "current-session"
        ),
        depth=0,
        summary="Prior summary with deliberately incomplete reachability.",
        token_count=7,
        source_token_count=7,
        source_ids=(prior_ids if prior_state == "missing_root" else prior_ids[::2]),
        source_type="messages",
    )
    engine._dag.add_node(prior_node)
    before_node_count = len(engine._dag.get_session_nodes("current-session"))
    context = [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "fresh tail"},
    ]

    try:
        result = engine.compress(context, force=True)
        state = engine._lifecycle.get_by_conversation(conversation_id)
        after_node_count = len(engine._dag.get_session_nodes("current-session"))
    finally:
        engine.shutdown()

    assert engine.last_compression_status == "error"
    assert result == context
    assert state is not None and state.current_frontier_store_id == 0
    assert after_node_count == before_node_count


def test_retained_conversation_compacts_after_session_rollover(
    tmp_path,
    monkeypatch,
) -> None:
    conversation_id = "issue-247-conversation"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    engine.on_session_start(
        "session-one",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    context = [
        {"role": "assistant", "content": "first old turn"},
        {"role": "assistant", "content": "second old turn"},
        {"role": "user", "content": "old tail"},
    ]

    try:
        engine.ingest(context)
        context = engine.compress(context, force=True)
        first_frontier = engine._last_compacted_store_id
        assert engine.last_compression_status == "compacted"
        assert first_frontier > 0

        engine.on_session_start(
            "session-two",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="session-one",
        )
        for turn in range(2):
            context = context + [
                {"role": "assistant", "content": f"new turn {turn} a"},
                {"role": "assistant", "content": f"new turn {turn} b"},
                {"role": "user", "content": f"new turn {turn} tail"},
            ]
            engine.ingest(context)
            context = engine.compress(context, force=True)
            assert engine.last_compression_status == "compacted"
    finally:
        final_frontier = engine._last_compacted_store_id
        engine.shutdown()

    assert final_frontier > first_frontier


def test_publication_rejects_sources_claimed_below_a_refreshed_frontier(
    tmp_path,
) -> None:
    conversation_id = "issue-247-concurrent-frontier"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "concurrent.db")),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "current-session",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_ids = engine._store.append_batch(
        "source-producer-session",
        [
            {"role": "user", "content": f"source {index}"}
            for index in range(4)
        ],
        conversation_id=conversation_id,
    )
    prior = SummaryNode(
        session_id="source-producer-session",
        depth=0,
        summary="Prior concurrent publication.",
        token_count=3,
        source_token_count=3,
        source_ids=source_ids[:2],
        source_type="messages",
    )
    engine._dag.add_node(prior)
    engine._lifecycle.advance_frontier(
        conversation_id,
        "current-session",
        source_ids[1],
    )
    candidate = SummaryNode(
        session_id="current-session",
        depth=0,
        summary="Stale candidate publication.",
        token_count=3,
        source_token_count=6,
        source_ids=source_ids,
        source_type="messages",
    )
    before_node_count = len(engine._dag.get_session_nodes("current-session"))

    try:
        def stage_candidate(conn, candidate_id):
            engine._lifecycle.stage_compaction_publication(
                conn,
                conversation_id,
                "current-session",
                candidate_id,
                source_ids[1],
                source_ids,
            )

        with pytest.raises(LifecyclePublicationConflictError):
            engine._dag.add_node(candidate, before_commit=stage_candidate)
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_frontier_store_id == source_ids[1]
        assert len(engine._dag.get_session_nodes("current-session")) == before_node_count
    finally:
        engine.shutdown()


def test_blank_legacy_rows_require_an_unambiguous_conversation_binding(
    tmp_path,
) -> None:
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "ambiguous.db")),
        hermes_home=str(tmp_path / "home"),
    )
    session_id = "shared-legacy-session"
    first_conversation = "issue-247-first-owner"
    second_conversation = "issue-247-second-owner"
    engine._lifecycle.bind_session(session_id, conversation_id=first_conversation)
    engine._lifecycle.finalize_session(first_conversation, session_id)
    engine._lifecycle.bind_session(session_id, conversation_id=second_conversation)
    [source_id] = engine._store.append_batch(
        session_id,
        [{"role": "user", "content": "ambiguous legacy row"}],
    )

    try:
        assert engine._store.get_conversation_messages_after(
            first_conversation,
            legacy_session_ids={session_id},
        ) == []
        assert engine._store.get_conversation_messages_after(
            second_conversation,
            legacy_session_ids={session_id},
        ) == []
        candidate = SummaryNode(
            session_id=session_id,
            depth=0,
            summary="Ambiguous candidate.",
            token_count=2,
            source_token_count=2,
            source_ids=[source_id],
            source_type="messages",
        )
        before_node_count = len(engine._dag.get_session_nodes(session_id))

        def stage_candidate(conn, candidate_id):
            engine._lifecycle.stage_compaction_publication(
                conn,
                second_conversation,
                session_id,
                candidate_id,
                0,
                [source_id],
            )

        with pytest.raises(LifecyclePublicationConflictError):
            engine._dag.add_node(candidate, before_commit=stage_candidate)
        state = engine._lifecycle.get_by_conversation(second_conversation)
        assert state is not None and state.current_frontier_store_id == 0
        assert len(engine._dag.get_session_nodes(session_id)) == before_node_count
    finally:
        engine.shutdown()
