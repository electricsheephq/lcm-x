"""Real-engine regression coverage for lossless multisession rollover."""

from __future__ import annotations

import sqlite3
import time

import hermes_lcm.escalation as lcm_escalation
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


def test_ordinary_rollover_preserves_committed_frontier_for_fresh_tail(tmp_path, monkeypatch):
    conversation_id = "issue-247-ordinary-frontier"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "ordinary-frontier.db"),
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
    first_context = [
        {"role": "assistant", "content": f"old-{index}"}
        for index in range(4)
    ] + [{"role": "user", "content": "old-tail"}]
    try:
        engine.ingest(first_context)
        first_context = engine.compress(first_context, force=True)
        first_frontier = engine._last_compacted_store_id
        assert first_frontier > 0

        engine.rollover_session(
            "s1",
            "s2",
            previous_messages=first_context,
            platform="cli",
            context_length=200_000,
        )
        assert engine._last_compacted_store_id == first_frontier

        second_context = first_context + [
            {"role": "assistant", "content": "new-one"},
            {"role": "assistant", "content": "new-two"},
            {"role": "user", "content": "new-tail"},
        ]
        engine.ingest(second_context)
        engine.compress(second_context, force=True)
    finally:
        status = engine.last_compression_status
        final_frontier = engine._last_compacted_store_id
        engine.shutdown()

    assert status == "compacted"
    assert final_frontier > first_frontier


def test_replayed_scaffold_and_later_duplicate_keep_distinct_source_ownership(tmp_path):
    conversation_id = "issue-247-scaffold-ownership"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "scaffold-ownership.db"),
            fresh_tail_count=1,
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
    scaffold_content = (
        "[Current user objective preserved from compacted history]\n"
        "replayed objective"
    )
    source_ids = engine._store.append_batch(
        "s1",
        [
            {"role": "user", "content": scaffold_content},
            {"role": "user", "content": scaffold_content},
        ],
        conversation_id=conversation_id,
    )
    scaffold = {"role": "user", "content": scaffold_content}
    later_duplicate = {"role": "user", "content": scaffold_content}
    active = [
        {"role": "system", "content": "system"},
        scaffold,
        {"role": "assistant", "content": "durable boundary"},
        later_duplicate,
    ]

    try:
        pass_map = engine._get_store_id_map_for_messages(active)
        excluded = engine._get_store_ids_for_messages(
            [scaffold],
            mapped_ids_by_message_id=pass_map,
        )
        covered = engine._get_store_ids_for_messages(
            [later_duplicate],
            mapped_ids_by_message_id=pass_map,
        )
    finally:
        engine.shutdown()

    assert pass_map[id(scaffold)] == source_ids[0]
    assert pass_map[id(later_duplicate)] == source_ids[1]
    assert excluded == [source_ids[0]]
    assert covered == [source_ids[1]]
    assert not set(excluded) & set(covered)


def test_scaffold_exclusion_does_not_overlap_compacted_source_lookup(
    tmp_path,
    monkeypatch,
):
    conversation_id = "issue-247-scaffold-publication"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "scaffold-publication.db"),
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
    scaffold_content = (
        "[Current user objective preserved from compacted history]\n"
        "replayed objective"
    )
    duplicate_content = "replayed objective"
    active = [
        {"role": "user", "content": scaffold_content},
        {"role": "user", "content": duplicate_content},
        {"role": "assistant", "content": "fresh tail"},
    ]
    engine.ingest(active)
    source_ids = [
        row["store_id"] for row in engine._store.get_session_messages("s1")
    ]

    try:
        result = engine.compress(active)
        state = engine._lifecycle.get_by_conversation(conversation_id)
        nodes = engine._dag.get_session_nodes("s1")
    finally:
        engine.shutdown()

    assert engine.last_compression_status == "compacted"
    assert state is not None
    assert state.current_frontier_store_id == source_ids[1]
    assert len(nodes) == 1
    assert nodes[0].source_ids == [source_ids[1]]
    assert result[-1] == active[-1]


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

        state_db_path = tmp_path / "home" / "state.db"
        state_db_path.parent.mkdir(parents=True, exist_ok=True)
        state_db = sqlite3.connect(str(state_db_path))
        state_db.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                end_reason TEXT,
                ended_at REAL,
                model_config TEXT,
                source TEXT,
                started_at REAL
            );
            """
        )
        state_db.executemany(
            "INSERT INTO sessions(id, parent_session_id, end_reason, ended_at, model_config, source, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("s1", None, "compression", 1.0, "{}", "cli", 1.0),
                ("s2", "s1", None, None, "{}", "cli", 2.0),
            ],
        )
        state_db.commit()
        state_db.close()

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


def test_tool_cleanup_folded_assistant_run_preserves_exact_source_ids(tmp_path):
    conversation_id = "issue-247-tool-cleanup-lineage"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "tool-cleanup-lineage.db"),
            fresh_tail_count=24,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    try:
        engine._store.append(
            "s1",
            {"role": "assistant", "content": "assistant-one"},
            conversation_id=conversation_id,
        )
        engine._store.append(
            "s1",
            {
                "role": "tool",
                "tool_call_id": "orphan-call",
                "content": "late orphan result",
            },
            conversation_id=conversation_id,
        )
        engine._store.append(
            "s1",
            {"role": "assistant", "content": "assistant-two"},
            conversation_id=conversation_id,
        )
        engine._store.append(
            "s1",
            {"role": "user", "content": "continue"},
            conversation_id=conversation_id,
        )
        stored = engine._store.get_session_messages("s1")
        context = engine._assemble_context(
            {"role": "system", "content": "system"},
            stored,
            assembly_cap_override=200_000,
        )
        folded_lineage = engine._load_folded_tail_lineage(context)
        assert folded_lineage is not None
        folded_message, source_rows = folded_lineage
        assert folded_message["content"] == "assistant-one\nassistant-two"
        assert [row["store_id"] for row in source_rows] == [
            stored[0]["store_id"],
            stored[2]["store_id"],
        ]
        assert stored[1]["store_id"] not in [row["store_id"] for row in source_rows]
    finally:
        engine.shutdown()


def test_generated_fold_after_orphan_tool_keeps_post_cleanup_lineage(tmp_path):
    conversation_id = "issue-247-generated-fold-cleanup"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "generated-fold-cleanup.db"),
            fresh_tail_count=24,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    first_source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "assistant-one"},
        conversation_id=conversation_id,
    )
    engine._store.append(
        "s1",
        {
            "role": "tool",
            "tool_call_id": "orphan-call",
            "content": "late orphan result",
        },
        conversation_id=conversation_id,
    )
    second_source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "assistant-two"},
        conversation_id=conversation_id,
    )
    engine._dag.add_node(
        SummaryNode(
            session_id="s1",
            summary="prior durable summary",
            token_count=1,
            source_token_count=1,
            source_ids=[],
        )
    )
    try:
        context = engine._assemble_context(
            {"role": "system", "content": "system"},
            engine._store.get_session_messages("s1"),
            assembly_cap_override=200_000,
            retained_user_message={"role": "user", "content": "current objective"},
        )
        folded_lineage = engine._load_folded_tail_lineage(context)
        assert folded_lineage is not None
        folded_message, source_rows = folded_lineage
        assert "assistant-one" in folded_message["content"]
        assert "assistant-two" in folded_message["content"]
        assert [row["store_id"] for row in source_rows] == [
            first_source_id,
            second_source_id,
        ]
    finally:
        engine.shutdown()


def test_reassembling_existing_fold_preserves_all_source_ids(tmp_path):
    conversation_id = "issue-247-reassembled-fold"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "reassembled-fold.db"),
            fresh_tail_count=24,
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
    first_source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "assistant-one"},
        conversation_id=conversation_id,
    )
    second_source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "assistant-two"},
        conversation_id=conversation_id,
    )
    try:
        engine._dag.add_node(
            SummaryNode(
                session_id="s1",
                summary="retained summary",
                token_count=1,
                source_token_count=2,
                source_ids=[first_source_id, second_source_id],
            )
        )
        anchor = {"role": "user", "content": "current objective"}
        first_context = engine._assemble_context(
            None,
            [
                {"role": "assistant", "content": "assistant-one"},
                {"role": "assistant", "content": "assistant-two"},
                {"role": "user", "content": "tail"},
            ],
            assembly_cap_override=200_000,
            retained_user_message=anchor,
        )
        first_lineage = engine._load_folded_tail_lineage(first_context)
        assert first_lineage is not None
        assert [row["store_id"] for row in first_lineage[1]] == [
            first_source_id,
            second_source_id,
        ]

        second_context = engine._assemble_context(
            None,
            first_context[1:],
            assembly_cap_override=200_000,
            retained_user_message=anchor,
        )
        second_lineage = engine._load_folded_tail_lineage(second_context)
    finally:
        engine.shutdown()

    assert second_lineage is not None
    assert [row["store_id"] for row in second_lineage[1]] == [
        first_source_id,
        second_source_id,
    ]


def test_generated_fold_lineage_failure_restores_merged_tail(tmp_path, monkeypatch):
    conversation_id = "issue-247-generated-fold-lineage-failure"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "generated-fold-lineage-failure.db"),
            fresh_tail_count=24,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    first_source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "assistant-one"},
        conversation_id=conversation_id,
    )
    engine._store.append(
        "s1",
        {
            "role": "tool",
            "tool_call_id": "orphan-call",
            "content": "late orphan result",
        },
        conversation_id=conversation_id,
    )
    second_source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "assistant-two"},
        conversation_id=conversation_id,
    )
    engine._dag.add_node(
        SummaryNode(
            session_id="s1",
            summary="prior durable summary",
            token_count=1,
            source_token_count=1,
            source_ids=[],
        )
    )
    monkeypatch.setattr(engine, "_write_folded_tail_lineage", lambda *_args: False)
    try:
        context = engine._assemble_context(
            {"role": "system", "content": "system"},
            engine._store.get_session_messages("s1"),
            assembly_cap_override=200_000,
            retained_user_message={"role": "user", "content": "current objective"},
        )
        assert [message.get("content") for message in context[-2:]] == [
            "assistant-one",
            "assistant-two",
        ]
        assert not any("prior durable summary" in str(message.get("content")) for message in context)
        mapped = engine._get_store_id_map_for_messages(context[1:])
    finally:
        engine.shutdown()

    assert mapped[id(context[-2])] == first_source_id
    assert mapped[id(context[-1])] == second_source_id


def test_legacy_conversation_alias_refuses_unproven_current_session(tmp_path):
    conversation_id = "legacy-session-alias"
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    state_db = hermes_home / "state.db"
    host = sqlite3.connect(state_db)
    host.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            end_reason TEXT,
            ended_at REAL,
            model_config TEXT,
            source TEXT,
            started_at REAL
        );
        """
    )
    host.executemany(
        "INSERT INTO sessions(id, parent_session_id, end_reason, ended_at, model_config, source, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("s1", None, "compression", 1.0, "{}", "cli", 1.0),
            ("s3", "s1", None, None, "{}", "cli", 3.0),
        ],
    )
    host.commit()
    host.close()

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "legacy-alias.db")),
        hermes_home=str(hermes_home),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "committed source"},
        conversation_id=conversation_id,
    )
    engine._lifecycle.advance_frontier(conversation_id, "s1", source_id)
    engine.on_session_start(
        "s4",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    before = engine._lifecycle.get_by_conversation(conversation_id)
    assert before is not None
    assert before.current_session_id == "s4"
    engine.on_session_start(
        "s3",
        platform="cli",
        conversation_id=conversation_id,
        boundary_reason="compression",
        old_session_id="s1",
        context_length=200_000,
        hermes_home=str(hermes_home),
    )
    after = engine._lifecycle.get_by_conversation(conversation_id)
    try:
        assert after is not None
        assert after.current_session_id == "s4"
        assert after.current_frontier_store_id == before.current_frontier_store_id
        assert engine._store.get(source_id)["session_id"] == "s1"
        assert engine._session_id == "s4"
        assert engine._store.get_session_count("s3") == 0
    finally:
        engine.shutdown()


def test_finalized_alias_refuses_stale_callback_without_host_proof(tmp_path):
    conversation_id = "finalized-alias-stale"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "finalized-stale.db")),
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
        {"role": "assistant", "content": "finalized source"},
        conversation_id=conversation_id,
    )
    engine._lifecycle.advance_frontier(conversation_id, "s1", source_id)
    engine._atomic_rollover_lcm_state(conversation_id, "s1", "s2")
    engine._lifecycle.finalize_session(
        conversation_id,
        "s2",
        frontier_store_id=source_id,
    )
    engine._session_id = "s2"
    before = engine._lifecycle.get_by_conversation(conversation_id)
    assert before is not None
    assert before.current_session_id is None
    assert before.last_finalized_session_id == "s2"
    try:
        engine.on_session_start(
            "s3",
            platform="cli",
            conversation_id=conversation_id,
            boundary_reason="compression",
            old_session_id="s1",
            context_length=200_000,
        )
        after = engine._lifecycle.get_by_conversation(conversation_id)
        assert after == before
        assert engine._session_id == "s2"
        assert engine._store.get_session_count("s3") == 0
        assert engine._store.get(source_id)["session_id"] == "s1"
    finally:
        engine.shutdown()


def test_bound_finalized_alias_refuses_stale_callback_without_host_proof(tmp_path):
    """A process still bound to finalized old state cannot guess a successor."""
    conversation_id = "finalized-alias-bound-old-stale"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "finalized-bound-stale.db")),
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
        {"role": "assistant", "content": "bound finalized source"},
        conversation_id=conversation_id,
    )
    engine._dag.add_node(
        SummaryNode(
            session_id="s1",
            summary="bound finalized summary",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
        )
    )
    engine._lifecycle.advance_frontier(conversation_id, "s1", source_id)
    engine._lifecycle.finalize_session(
        conversation_id,
        "s1",
        frontier_store_id=source_id,
    )
    before = engine._lifecycle.get_by_conversation(conversation_id)
    before_nodes = engine._dag.get_session_nodes("s1")
    assert before is not None
    assert before.current_session_id is None
    assert before.last_finalized_session_id == "s1"
    assert before.last_finalized_frontier_store_id == source_id
    assert engine._session_id == "s1"
    try:
        engine.on_session_start(
            "s3",
            platform="cli",
            conversation_id=conversation_id,
            boundary_reason="compression",
            old_session_id="s1",
            context_length=200_000,
        )
        after = engine._lifecycle.get_by_conversation(conversation_id)
        assert after == before
        assert engine._session_id == "s1"
        assert engine._store.get_session_count("s3") == 0
        assert engine._store.get(source_id)["session_id"] == "s1"
        assert engine._dag.get_session_nodes("s1") == before_nodes
        assert engine._dag.get_session_nodes("s3") == []
    finally:
        engine.shutdown()


def test_finalized_alias_resumes_only_on_proven_host_chain(tmp_path):
    conversation_id = "finalized-alias-proven"
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    state_db = hermes_home / "state.db"
    host = sqlite3.connect(state_db)
    host.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            end_reason TEXT,
            ended_at REAL,
            model_config TEXT,
            source TEXT,
            started_at REAL
        );
        """
    )
    host.executemany(
        "INSERT INTO sessions(id, parent_session_id, end_reason, ended_at, model_config, source, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("s1", None, "compression", 1.0, "{}", "cli", 1.0),
            ("s2", "s1", "compression", 2.0, "{}", "cli", 2.0),
            ("s3", "s2", None, None, "{}", "cli", 3.0),
        ],
    )
    host.commit()
    host.close()

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "finalized-proven.db")),
        hermes_home=str(hermes_home),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "proven finalized source"},
        conversation_id=conversation_id,
    )
    engine._lifecycle.advance_frontier(conversation_id, "s1", source_id)
    engine._atomic_rollover_lcm_state(conversation_id, "s1", "s2")
    engine._lifecycle.finalize_session(
        conversation_id,
        "s2",
        frontier_store_id=source_id,
    )
    engine._session_id = "s2"
    try:
        engine.on_session_start(
            "s3",
            platform="cli",
            conversation_id=conversation_id,
            boundary_reason="compression",
            old_session_id="s1",
            context_length=200_000,
            hermes_home=str(hermes_home),
        )
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_session_id == "s3"
        assert state.last_finalized_session_id == "s2"
        assert state.last_finalized_frontier_store_id == source_id
        assert state.current_frontier_store_id == source_id
        assert engine._session_id == "s3"
        assert engine._store.get_session_count("s3") == 0
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
                record_reset=True,
            )
        except RuntimeError as exc:
            assert str(exc) == "synthetic rollover fault"
        else:
            raise AssertionError("synthetic rollover fault was not raised")

        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_session_id == "s1"
        assert state.last_finalized_session_id is None
        assert state.last_reset_at is None
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
            record_reset=True,
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
        assert state.last_reset_at is not None
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
    clock = 10.0

    def late_summary(**kwargs):
        calls.update(kwargs)
        return "late summary", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", late_summary)
    def monotonic():
        nonlocal clock
        current = clock
        clock += 101.0
        return current

    monkeypatch.setattr(lcm_engine.time, "monotonic", monotonic)
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


def test_escalation_deadline_is_not_renewed_for_level_two(monkeypatch):
    calls = []
    clock = 10.0

    def fake_summary(*_args, **kwargs):
        calls.append(kwargs.get("timeout"))
        return None

    monkeypatch.setattr(lcm_escalation, "_invoke_summary_llm", fake_summary)
    def monotonic():
        nonlocal clock
        current = clock
        clock += 101.0
        return current

    monkeypatch.setattr(lcm_escalation.time, "monotonic", monotonic)
    summary, level = lcm_escalation.summarize_with_escalation(
        "synthetic deadline source",
        source_tokens=4,
        token_budget=2,
        timeout=100.0,
        deadline=110.0,
        l3_truncate_tokens=2,
    )

    assert level == 3
    assert summary
    assert calls == [100.0]


def test_ownership_audit_rejects_ambiguous_and_orphan_legacy_bindings(tmp_path):
    conversation_id = "issue-247-audit"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "audit.db")),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    engine._store.append("s1", {"role": "assistant", "content": "legacy"})
    engine._store.append("orphan", {"role": "assistant", "content": "legacy orphan"})
    engine._lifecycle.record_rollover(
        "empty-alias",
        old_session_id="s1",
        new_session_id="alias-session",
    )
    alias_audit = engine._lifecycle.audit_conversation_ownership(
        engine._dag.connection,
        conversation_id,
        "s1",
    )
    assert alias_audit["ok"] is True
    assert alias_audit["non_owning_aliases"] == ["empty-alias"]
    engine._store.append(
        "other-session",
        {"role": "assistant", "content": "other source"},
        conversation_id="other-conversation",
    )
    engine._lifecycle.record_rollover(
        "other-conversation",
        old_session_id="s1",
        new_session_id="other-session",
    )
    try:
        audit = engine._lifecycle.audit_conversation_ownership(
            engine._dag.connection,
            conversation_id,
            "s1",
        )
        assert audit["ok"] is False
        assert audit["ambiguous_sessions"] == ["s1"]
        with pytest.raises(LifecyclePublicationConflictError):
            engine._bind_lifecycle_state(
                "ambiguous-session",
                conversation_id=conversation_id,
            )
        orphan_audit = engine._lifecycle.audit_conversation_ownership(
            engine._dag.connection,
            conversation_id,
            "orphan",
        )
        assert orphan_audit["ok"] is False
        assert orphan_audit["orphan_legacy_sessions"] == ["orphan"]
    finally:
        engine.shutdown()


def test_host_successor_replay_is_idempotent_and_stale_binding_preserves_frontier(tmp_path):
    conversation_id = "issue-247-host-replay"
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    state_db = hermes_home / "state.db"
    host = sqlite3.connect(state_db)
    host.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            end_reason TEXT,
            ended_at REAL,
            model_config TEXT,
            source TEXT,
            started_at REAL
        );
        """
    )
    host.executemany(
        "INSERT INTO sessions(id, parent_session_id, end_reason, ended_at, model_config, source, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("s1", None, "compression", 1.0, "{}", "cli", 1.0),
            ("s2", "s1", None, None, "{}", "cli", 2.0),
        ],
    )
    host.commit()
    host.close()

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "host-replay.db")),
        hermes_home=str(hermes_home),
    )
    engine.on_session_start(
        "s1",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_id = engine._store.append(
        "s1",
        {"role": "assistant", "content": "retained source"},
        conversation_id=conversation_id,
    )
    engine._lifecycle.advance_frontier(conversation_id, "s1", source_id)
    engine._dag.add_node(
        SummaryNode(
            session_id="s1",
            summary="retained summary",
            token_count=1,
            source_token_count=1,
            source_ids=[source_id],
        )
    )
    try:
        engine.on_session_start(
            "s2",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="s1",
            hermes_home=str(hermes_home),
        )
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_session_id == "s2"
        assert state.last_finalized_session_id == "s1"
        assert state.last_finalized_frontier_store_id == source_id
        assert len(engine._dag.get_session_nodes("s2")) == 1

        engine.on_session_start(
            "s2",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="s1",
            hermes_home=str(hermes_home),
        )
        assert len(engine._dag.get_session_nodes("s2")) == 1
        assert engine._lifecycle.get_by_conversation(conversation_id).current_frontier_store_id == source_id

        host = sqlite3.connect(state_db)
        host.execute(
            "UPDATE sessions SET end_reason = 'compression', ended_at = 3.0 WHERE id = 's2'"
        )
        host.execute(
            "INSERT INTO sessions(id, parent_session_id, end_reason, ended_at, model_config, source, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("s3", "s2", None, None, "{}", "cli", 3.0),
        )
        host.commit()
        host.close()

        engine.on_session_start(
            "s3",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="s1",
            hermes_home=str(hermes_home),
        )
        state = engine._lifecycle.get_by_conversation(conversation_id)
        assert state is not None
        assert state.current_session_id == "s3"
        assert state.last_finalized_session_id == "s2"
        assert state.current_frontier_store_id == source_id
        assert len(engine._dag.get_session_nodes("s3")) == 1

        before = engine._lifecycle.get_by_conversation(conversation_id)
        engine.on_session_start(
            "stale-child",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="unknown-old",
            hermes_home=str(hermes_home),
        )
        after = engine._lifecycle.get_by_conversation(conversation_id)
        assert before is not None and after is not None
        assert engine._session_id == "s3"
        assert after.current_session_id == "s3"
        assert after.current_frontier_store_id == before.current_frontier_store_id

        engine.on_session_start(
            "s4",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
        )
        engine._lifecycle.advance_frontier(conversation_id, "s4", source_id)
        before = engine._lifecycle.get_by_conversation(conversation_id)
        engine.on_session_start(
            "s3",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="s1",
            hermes_home=str(hermes_home),
        )
        after = engine._lifecycle.get_by_conversation(conversation_id)
        assert before is not None and after is not None
        assert engine._session_id == "s4"
        assert after.current_session_id == "s4"
        assert after.current_frontier_store_id == before.current_frontier_store_id == source_id
    finally:
        engine.shutdown()


def test_publication_validation_scales_with_fixed_owner_scope(tmp_path):
    conversation_id = "issue-247-publication-perf"
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "publication-perf.db")),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "legitimate-session",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )
    source_ids = [
        engine._store.append(
            "legitimate-session",
            {"role": "assistant", "content": f"source-{index}"},
            conversation_id=conversation_id,
        )
        for index in range(4)
    ]
    conn = engine._dag.connection
    assert conn is not None

    def measure(rounds=30):
        timings = []
        conn.execute("BEGIN")
        try:
            for _ in range(rounds):
                conn.execute("SAVEPOINT issue247_perf")
                started = time.perf_counter()
                engine._lifecycle.stage_compaction_publication(
                    conn,
                    conversation_id,
                    "legitimate-session",
                    -1,
                    0,
                    source_ids,
                )
                timings.append(time.perf_counter() - started)
                conn.execute("ROLLBACK TO issue247_perf")
                conn.execute("RELEASE issue247_perf")
        finally:
            conn.rollback()
        return sum(timings) / len(timings)

    try:
        baseline = measure()
        conn.execute("BEGIN")
        conn.executemany(
            """
            INSERT INTO summary_nodes(
                session_id, depth, summary, token_count, source_token_count,
                source_ids, source_type, created_at
            ) VALUES (?, 0, 'foreign', 1, 1, '[]', 'messages', ?)
            """,
            [
                (f"foreign-session-{index % 500}", float(index))
                for index in range(20_000)
            ],
        )
        conn.commit()
        expanded = measure()
        assert expanded <= baseline * 2.0
    finally:
        engine.shutdown()
