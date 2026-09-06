"""Real-engine regression coverage for lossless multisession rollover."""

from __future__ import annotations

import hermes_lcm.engine as lcm_engine
import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError


def _summary(**_kwargs) -> tuple[str, int]:
    return "Earlier conversation. Expand for details: turns", 1


def test_rollover_preserves_logical_conversation_progression(tmp_path, monkeypatch):
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
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    context = [
        {"role": "assistant", "content": "s1 a"},
        {"role": "assistant", "content": "s1 b"},
        {"role": "user", "content": "s1 tail"},
    ]
    statuses = []
    try:
        engine.ingest(context)
        context = engine.compress(context)
        statuses.append((engine.last_compression_status, engine._lifecycle.get_by_conversation(conversation_id)))
        engine.on_session_start(
            "s2",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="s1",
        )
        for turn in range(1, 5):
            context = context + [
                {"role": "assistant", "content": f"s2 t{turn} a"},
                {"role": "assistant", "content": f"s2 t{turn} b"},
                {"role": "user", "content": f"s2 t{turn} tail"},
            ]
            engine.ingest(context)
            context = engine.compress(context)
            statuses.append((engine.last_compression_status, engine._lifecycle.get_by_conversation(conversation_id)))
    finally:
        engine.shutdown()

    assert statuses[0][0] == "compacted"
    assert [status for status, _state in statuses[1:]] == ["compacted"] * 4
    assert all(state is not None for _status, state in statuses)
    assert statuses[-1][1].current_frontier_store_id > statuses[0][1].current_frontier_store_id


def test_source_mapping_spans_current_and_last_finalized_sessions(tmp_path):
    conversation_id = "issue-247-mapping"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "mapping.db"),
            fresh_tail_count=24,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_ids = engine._store.append_batch(
        "s1",
        [
            {"role": "assistant", "content": "old-a"},
            {"role": "assistant", "content": "old-b"},
        ],
        conversation_id=conversation_id,
    )
    old_rows = engine._store.get_session_messages("s1")
    try:
        engine.rollover_session(
            "s1",
            "s2",
            previous_messages=[],
            platform="cli",
            context_length=200_000,
        )
        mapped = engine._get_store_id_map_for_messages(old_rows)
    finally:
        engine.shutdown()

    assert [mapped[id(row)] for row in old_rows] == source_ids


def test_publication_accepts_cross_session_sources_owned_by_conversation(tmp_path):
    conversation_id = "issue-247-publication"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "publication.db"),
            fresh_tail_count=24,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "source from finalized session"},
        conversation_id=conversation_id,
    )
    try:
        engine.rollover_session(
            "s1",
            "s2",
            previous_messages=[],
            platform="cli",
            context_length=200_000,
        )
        node = SummaryNode(
            session_id="s2",
            summary="summary published after rollover",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
        )

        def stage(conn, node_id):
            engine._lifecycle.stage_compaction_publication(
                conn,
                conversation_id,
                "s2",
                node_id,
                0,
                [source_id],
            )

        engine._dag.add_node(node, before_commit=stage)
        state = engine._lifecycle.get_by_conversation(conversation_id)
    finally:
        engine.shutdown()

    assert state is not None
    assert state.current_frontier_store_id == source_id


def test_blank_legacy_rows_require_proven_session_and_foreign_rows_fail_closed(tmp_path):
    conversation_id = "issue-247-legacy"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "legacy.db"),
            fresh_tail_count=24,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    legacy_source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "legacy source"},
    )
    foreign_source_id = engine._store.append(
        "foreign-session",
        {"role": "assistant", "content": "foreign source"},
        conversation_id="foreign-conversation",
    )
    try:
        engine.rollover_session(
            "s1",
            "s2",
            previous_messages=[],
            platform="cli",
            context_length=200_000,
        )
        legacy_rows = engine._store.get_session_messages("s1")
        mapped = engine._get_store_id_map_for_messages(legacy_rows)
        assert mapped[id(legacy_rows[0])] == legacy_source_id

        node = SummaryNode(
            session_id="s2",
            summary="must reject cross-conversation source",
            token_count=1,
            source_token_count=2,
            source_ids=[legacy_source_id, foreign_source_id],
        )

        def stage(conn, node_id):
            engine._lifecycle.stage_compaction_publication(
                conn,
                conversation_id,
                "s2",
                node_id,
                0,
                [legacy_source_id, foreign_source_id],
            )

        try:
            engine._dag.add_node(node, before_commit=stage)
        except LifecyclePublicationConflictError:
            pass
        else:
            raise AssertionError("foreign conversation source was published")
        assert engine._dag.get_session_nodes("s2") == []
    finally:
        engine.shutdown()


def test_cross_session_lineage_claim_is_idempotent_and_unique(tmp_path):
    conversation_id = "issue-247-lineage"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lineage.db"),
            fresh_tail_count=24,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
            new_session_retain_depth=-1,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "claim once"},
        conversation_id=conversation_id,
    )

    def publish(session_id, expected_frontier=0):
        node = SummaryNode(
            session_id=session_id,
            summary="one logical source claim",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
        )

        def stage(conn, node_id):
            engine._lifecycle.stage_compaction_publication(
                conn,
                conversation_id,
                session_id,
                node_id,
                expected_frontier,
                [source_id],
            )

        engine._dag.add_node(node, before_commit=stage)

    try:
        engine.rollover_session(
            "s1",
            "s2",
            previous_messages=[],
            platform="cli",
            context_length=200_000,
        )
        publish("s2")
        engine.rollover_session(
            "s2",
            "s3",
            previous_messages=[],
            platform="cli",
            context_length=200_000,
        )
        try:
            publish("s3")
        except LifecyclePublicationConflictError:
            pass
        else:
            raise AssertionError("cross-session source was claimed twice")
        assert len(engine._dag.get_session_nodes("s3")) == 1
    finally:
        engine.shutdown()


def test_fresh_tail_merged_assistant_run_preserves_every_source_id(tmp_path, monkeypatch):
    conversation_id = "issue-247-fresh-tail"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "fresh-tail.db"),
            fresh_tail_count=24,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
            new_session_retain_depth=-1,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", _summary)
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    first_turns = [
        {"role": "assistant", "content": f"s1-{index}"}
        for index in range(32)
    ] + [{"role": "user", "content": "s1-tail"}]
    try:
        engine.ingest(first_turns)
        context = engine.compress(first_turns, force=True)
        assert engine.last_compression_status == "compacted"
        folded_lineage = engine._load_folded_tail_lineage(context)
        assert folded_lineage is not None
        _folded_message, retained_rows = folded_lineage
        assert len(retained_rows) > 1

        engine.on_session_start(
            "s2",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="s1",
        )
        second_turns = context + [
            {"role": "assistant", "content": f"s2-{index}"}
            for index in range(32)
        ] + [{"role": "user", "content": "s2-tail"}]
        engine.ingest(second_turns)
        engine.compress(second_turns, force=True)

        assert engine.last_compression_status == "compacted"
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_frontier_store_id > 0
        nodes = engine._dag.get_session_nodes("s2")
        published_source_ids = [
            source_id
            for node in nodes
            if node.source_type == "messages"
            for source_id in node.source_ids
        ]
        assert published_source_ids == sorted(set(published_source_ids))
        assert published_source_ids
        assert all(
            engine._store.get(source_id).get("conversation_id") == conversation_id
            for source_id in published_source_ids
        )
    finally:
        engine.shutdown()


def test_atomic_rollover_rolls_back_lifecycle_and_node_reassignment(tmp_path, monkeypatch):
    conversation_id = "issue-247-atomic"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "atomic.db"),
            new_session_retain_depth=-1,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    engine._dag.add_node(
        SummaryNode(
            session_id="s1",
            summary="retained before rollover",
            token_count=1,
            source_token_count=1,
            source_ids=[],
        )
    )
    try:
        original_reassign = lcm_engine.SummaryDAG.stage_reassign_session_nodes

        def fail_after_lifecycle(*_args, **_kwargs):
            raise RuntimeError("synthetic rollover fault")

        monkeypatch.setattr(
            lcm_engine.SummaryDAG,
            "stage_reassign_session_nodes",
            staticmethod(fail_after_lifecycle),
        )
        try:
            engine._atomic_rollover_lcm_state(
                conversation_id,
                "s1",
                "s2",
            )
        except RuntimeError as exc:
            assert str(exc) == "synthetic rollover fault"
        else:
            raise AssertionError("synthetic rollover fault was not raised")

        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_session_id == "s1"
        assert state.last_finalized_session_id is None
        assert len(engine._dag.get_session_nodes("s1")) == 1
        assert engine._dag.get_session_nodes("s2") == []

        monkeypatch.setattr(
            lcm_engine.SummaryDAG,
            "stage_reassign_session_nodes",
            staticmethod(original_reassign),
        )
        moved = engine._atomic_rollover_lcm_state(
            conversation_id,
            "s1",
            "s2",
        )
        assert moved == 1
        duplicate = engine._atomic_rollover_lcm_state(
            conversation_id,
            "s1",
            "s2",
        )
        assert duplicate == 0
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_session_id == "s2"
        assert state.last_finalized_session_id == "s1"
        assert engine._dag.get_session_nodes("s1") == []
        assert len(engine._dag.get_session_nodes("s2")) == 1
    finally:
        engine.shutdown()


def test_sweep_deadline_rejects_a_late_summary_result(tmp_path, monkeypatch):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "deadline.db")),
        hermes_home=str(tmp_path / "home"),
    )
    calls = {}
    clock = iter((10.0, 111.0))

    def late_summary(**kwargs):
        calls.update(kwargs)
        return "late summary", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", late_summary)
    monkeypatch.setattr(lcm_engine.time, "monotonic", lambda: next(clock))
    try:
        with pytest.raises(TimeoutError, match="time budget exhausted"):
            engine._summarize_leaf_chunk_with_rescue(
                [{"role": "user", "content": "synthetic deadline input"}],
                deadline=110.0,
            )
        assert calls["deadline"] == 110.0
        assert engine._dag.get_session_nodes("") == []
    finally:
        engine.shutdown()
