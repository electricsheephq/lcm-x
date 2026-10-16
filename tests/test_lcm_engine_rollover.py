"""LCM engine integration tests: rollover."""

import json
import logging
import sqlite3
import time

import pytest

import hermes_lcm.engine as lcm_engine

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

from .engine_helpers import (
    engine as engine,
    TestSessionRollover as _SessionRolloverHelpers,
)


class TestSessionRollover:
    HostAgentFrame = _SessionRolloverHelpers.__dict__["HostAgentFrame"]
    _start_host_child = _SessionRolloverHelpers.__dict__["_start_host_child"]

    def test_on_session_end_fails_open_when_ingest_store_is_locked(self, engine, monkeypatch, caplog):
        engine.on_session_start("test-session", platform="discord")

        def locked_ingest(messages, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine, "_ingest_messages", locked_ingest)

        with caplog.at_level(logging.WARNING):
            engine.on_session_end("test-session", [{"role": "user", "content": "hello"}])

        assert "LCM session-end raw-message ingest skipped due to SQLite lock" in caplog.text

    def test_on_session_end_fails_open_when_ingest_is_interrupted(self, engine, monkeypatch, caplog):
        engine.on_session_start("test-session", platform="discord")

        def interrupted_ingest(messages, **_kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(engine, "_ingest_messages", interrupted_ingest)

        with caplog.at_level(logging.WARNING):
            engine.on_session_end("test-session", [{"role": "user", "content": "hello"}])

        assert "LCM session-end raw-message ingest interrupted" in caplog.text

    def test_on_session_end_fails_open_when_finalize_store_is_locked(self, engine, monkeypatch, caplog):
        engine.on_session_start("test-session", platform="discord")

        def locked_finalize(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(engine._lifecycle, "finalize_session", locked_finalize)

        with caplog.at_level(logging.WARNING):
            engine.on_session_end("test-session", [{"role": "user", "content": "hello"}])

        assert "LCM session-end lifecycle finalization skipped due to SQLite lock" in caplog.text

    def test_on_session_end_fails_open_when_finalize_is_interrupted(self, engine, monkeypatch, caplog):
        engine.on_session_start("test-session", platform="discord")

        def interrupted_finalize(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(engine._lifecycle, "finalize_session", interrupted_finalize)

        with caplog.at_level(logging.WARNING):
            engine.on_session_end("test-session", [{"role": "user", "content": "hello"}])

        assert "LCM session-end lifecycle finalization interrupted" in caplog.text

    def test_on_session_end_returns_quickly_under_real_sqlite_writer_lock(self, engine, caplog):
        engine.on_session_start("test-session", platform="discord")
        engine._store._conn.execute("PRAGMA busy_timeout=750")
        engine._lifecycle._conn.execute("PRAGMA busy_timeout=750")

        locker = sqlite3.connect(str(engine._store.db_path), timeout=1.0, isolation_level=None)
        locker.execute("PRAGMA journal_mode=WAL")
        locker.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            with caplog.at_level(logging.WARNING):
                engine.on_session_end("test-session", [{"role": "user", "content": "hello"}])
            elapsed = time.monotonic() - started
        finally:
            locker.execute("ROLLBACK")
            locker.close()

        assert elapsed < 0.3
        assert engine._store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 750
        assert engine._lifecycle._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 750
        assert "LCM session-end raw-message ingest skipped due to SQLite lock" in caplog.text

    def test_on_session_end_reraises_non_lock_errors(self, engine, monkeypatch):
        engine.on_session_start("test-session", platform="discord")

        def broken_ingest(messages, **_kwargs):
            raise RuntimeError("not a lock")

        monkeypatch.setattr(engine, "_ingest_messages", broken_ingest)

        with pytest.raises(RuntimeError, match="not a lock"):
            engine.on_session_end("test-session", [{"role": "user", "content": "hello"}])

    def test_rollover_session_rebinds_engine_and_carries_retained_nodes(self, engine):
        engine._config.new_session_retain_depth = 2
        from hermes_lcm.dag import SummaryNode
        import time

        engine.on_session_start("old-session", platform="cli", context_length=200000)
        for depth in range(4):
            engine._dag.add_node(SummaryNode(
                session_id="old-session", depth=depth,
                summary=f"old d{depth}", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))

        moved = engine.rollover_session(
            "old-session",
            "new-session",
            previous_messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            platform="cli",
            context_length=200000,
        )

        assert moved == 2
        assert engine._session_id == "new-session"
        assert engine._session_platform == "cli"
        assert engine._store.get_session_count("old-session") == 3
        assert engine._dag.get_session_nodes("old-session") == []
        new_nodes = engine._dag.get_session_nodes("new-session")
        assert len(new_nodes) == 2
        assert all(node.depth >= 2 for node in new_nodes)

    def test_rollover_session_supports_repeated_new_session_boundaries_without_duplicate_nodes(self, engine):
        engine._config.new_session_retain_depth = 2
        from hermes_lcm.dag import SummaryNode
        import time

        engine.on_session_start("s1", platform="cli", context_length=200000)
        for depth in range(4):
            engine._dag.add_node(SummaryNode(
                session_id="s1", depth=depth,
                summary=f"seed d{depth}", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))

        moved1 = engine.rollover_session("s1", "s2", previous_messages=[], platform="cli", context_length=200000)
        assert moved1 == 2

        engine._dag.add_node(SummaryNode(
            session_id="s2", depth=2,
            summary="fresh d2", token_count=100,
            source_token_count=500, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))
        engine._dag.add_node(SummaryNode(
            session_id="s2", depth=0,
            summary="fresh d0", token_count=100,
            source_token_count=500, source_ids=[],
            source_type="messages", created_at=time.time(),
        ))

        moved2 = engine.rollover_session("s2", "s3", previous_messages=[], platform="cli", context_length=200000)

        assert moved2 == 3
        s3_nodes = engine._dag.get_session_nodes("s3")
        assert len(s3_nodes) == 3
        assert sorted(node.summary for node in s3_nodes) == ["fresh d2", "seed d2", "seed d3"]
        assert engine._dag.get_session_nodes("s2") == []
        assert engine._session_id == "s3"

    def test_rollover_session_current_session_retrieval_uses_new_session_after_carry_over(self, engine):
        engine._config.new_session_retain_depth = 2
        engine.on_session_start("old-retrieval", platform="cli", context_length=200000)
        old_store_id = engine._store.append(
            "old-retrieval",
            {"role": "user", "content": "phoenix raw old-only context"},
            token_estimate=9,
            source="cli",
        )
        retained_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-retrieval",
            depth=2,
            summary="phoenix retained rollover summary",
            token_count=7,
            source_token_count=9,
            source_ids=[old_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        pruned_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-retrieval",
            depth=0,
            summary="phoenix pruned rollover summary",
            token_count=7,
            source_token_count=9,
            source_ids=[old_store_id],
            source_type="messages",
            created_at=time.time(),
        ))

        moved = engine.rollover_session(
            "old-retrieval",
            "new-retrieval",
            previous_messages=[],
            platform="cli",
            context_length=200000,
        )

        assert moved == 1
        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "phoenix", "session_scope": "current", "sort": "relevance", "limit": 10},
        ))
        assert result["session_scope"] == "current"
        assert result["total_results"] == 1
        assert result["results"] == [
            {
                "type": "summary",
                "depth": "d2",
                "node_id": retained_node_id,
                "session_id": "new-retrieval",
                "snippet": "phoenix retained rollover summary",
                "token_count": 7,
                "expand_hint": "",
                "earliest_at": None,
                "latest_at": None,
                "from_current_session": True,
            }
        ]
        assert engine._dag.get_node(pruned_node_id) is None
        assert engine._store.get_session_count("old-retrieval") == 1
        assert engine._store.get_session_count("new-retrieval") == 0

    def test_expand_retained_rollover_summary_recovers_original_session_sources(self, engine):
        engine._config.new_session_retain_depth = 2
        engine.on_session_start("old-expand", platform="discord", context_length=200000)
        old_store_id = engine._store.append(
            "old-expand",
            {"role": "user", "content": "discord carried source payload"},
            token_estimate=9,
            source="discord",
        )
        retained_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-expand",
            depth=2,
            summary="carried rollover summary about discord payload",
            token_count=7,
            source_token_count=9,
            source_ids=[old_store_id],
            source_type="messages",
            created_at=time.time(),
        ))

        moved = engine.rollover_session(
            "old-expand",
            "new-expand",
            previous_messages=[],
            platform="discord",
            context_length=200000,
        )
        expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": retained_node_id}))

        assert moved == 1
        assert expanded["pagination"]["total_sources"] == 1
        assert expanded["pagination"]["returned_sources"] == 1
        assert expanded["expanded"][0]["store_id"] == old_store_id
        assert expanded["expanded"][0]["content"] == "discord carried source payload"

    def test_expand_retained_depth_zero_summary_recovers_original_session_sources(self, engine):
        engine._config.new_session_retain_depth = -1
        engine.on_session_start("old-expand-d0", platform="discord", context_length=200000)
        old_store_id = engine._store.append(
            "old-expand-d0",
            {"role": "user", "content": "depth zero carried source payload"},
            token_estimate=9,
            source="discord",
        )
        retained_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-expand-d0",
            depth=0,
            summary="depth zero carried rollover summary",
            token_count=7,
            source_token_count=9,
            source_ids=[old_store_id],
            source_type="messages",
            created_at=time.time(),
        ))

        moved = engine.rollover_session(
            "old-expand-d0",
            "new-expand-d0",
            previous_messages=[],
            platform="discord",
            context_length=200000,
        )
        expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": retained_node_id}))

        assert moved == 1
        assert expanded["expanded"][0]["session_id"] == "old-expand-d0"
        assert expanded["expanded"][0]["from_current_session"] is False
        assert expanded["expanded"][0]["content"] == "depth zero carried source payload"

    def test_rollover_session_compression_boundary_keeps_depth_zero_nodes(self, engine):
        engine._config.new_session_retain_depth = 2
        engine.on_session_start("compress-rollover-old", platform="telegram", context_length=200000)
        store_id = engine._store.append(
            "compress-rollover-old",
            {"role": "user", "content": "compression rollover keeps depth zero"},
            token_estimate=13,
            source="telegram",
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="compress-rollover-old",
            depth=0,
            summary="compression rollover depth zero summary",
            token_count=5,
            source_token_count=13,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = store_id
        old_conversation_id = engine._conversation_id

        moved = engine.rollover_session(
            "compress-rollover-old",
            "compress-rollover-new",
            previous_messages=[],
            boundary_reason="compression",
            platform="telegram",
            context_length=200000,
        )

        assert moved == 1
        assert engine._session_id == "compress-rollover-new"
        assert engine._conversation_id == old_conversation_id
        assert engine._store.get_session_count("compress-rollover-old") == 1
        assert engine._store.get_session_count("compress-rollover-new") == 0
        assert engine._dag.get_session_nodes("compress-rollover-old") == []
        new_nodes = engine._dag.get_session_nodes("compress-rollover-new")
        assert [node.node_id for node in new_nodes] == [node_id]
        expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))
        assert expanded["expanded"][0]["content"] == "compression rollover keeps depth zero"

    def test_rollover_session_compression_boundary_respects_disabled_carry_over(self, engine):
        engine._config.new_session_retain_depth = 2
        engine.on_session_start("compress-no-carry-old", platform="telegram", context_length=200000)
        store_id = engine._store.append(
            "compress-no-carry-old",
            {"role": "user", "content": "do not leak compression carry over"},
            token_estimate=13,
            source="telegram",
        )
        retained_node_id = engine._dag.add_node(SummaryNode(
            session_id="compress-no-carry-old",
            depth=2,
            summary="do not leak retained summary",
            token_count=5,
            source_token_count=13,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = store_id
        old_conversation_id = engine._conversation_id

        moved = engine.rollover_session(
            "compress-no-carry-old",
            "compress-no-carry-new",
            previous_messages=[],
            carry_over_context=False,
            boundary_reason="compression",
            platform="telegram",
            context_length=200000,
        )

        assert moved == 0
        assert engine._session_id == "compress-no-carry-new"
        assert engine._conversation_id == old_conversation_id
        assert engine._store.get_session_count("compress-no-carry-old") == 1
        assert engine._store.get_session_count("compress-no-carry-new") == 0
        assert [node.node_id for node in engine._dag.get_session_nodes("compress-no-carry-old")] == [retained_node_id]
        assert engine._dag.get_session_nodes("compress-no-carry-new") == []
        result = json.loads(engine.handle_tool_call(
            "lcm_grep",
            {"query": "leak", "session_scope": "current", "sort": "relevance", "limit": 10},
        ))
        assert result["total_results"] == 0

    def test_rollover_session_skips_carry_over_when_old_session_is_not_bound(self, engine):
        engine._config.new_session_retain_depth = 2

        engine.on_session_start("attacker-current", platform="cli", context_length=200000)
        engine._dag.add_node(SummaryNode(
            session_id="victim-session",
            depth=2,
            summary="victim summary",
            token_count=100,
            source_token_count=200,
            source_ids=[],
            source_type="messages",
            created_at=time.time(),
        ))

        moved = engine.rollover_session(
            "victim-session",
            "attacker-new",
            previous_messages=[],
            platform="cli",
            context_length=200000,
        )

        assert moved == 0
        assert len(engine._dag.get_session_nodes("victim-session")) == 1
        assert engine._dag.get_session_nodes("attacker-new") == []
        assert engine._session_id == "attacker-new"

    def test_compression_boundary_skip_uses_new_session_cursor_for_fresh_messages(self, engine):
        """Unproven boundary skips must not trust stale cursor/frontier state.

        If the bound session cannot be proven to be the carry-over source, the
        new session must persist its own fresh messages rather than inheriting a
        cursor that makes them look already ingested.
        """
        engine.on_session_start("session-a", platform="telegram", context_length=200000)
        engine._last_compacted_store_id = 42
        engine._ingest_cursor = 3

        engine.on_session_start(
            "session-b",
            old_session_id="session-c",
            boundary_reason="compression",
            platform="telegram",
            context_length=200000,
        )

        fresh_messages = [
            {"role": "user", "content": "fresh session-b question"},
            {"role": "assistant", "content": "fresh session-b answer"},
        ]
        replay = engine._ingest_messages(fresh_messages)

        assert replay == fresh_messages
        assert engine._session_id == "session-b"
        assert engine._last_compacted_store_id == 0
        assert engine._ingest_cursor == len(fresh_messages)
        assert engine._store.get_session_count("session-b") == len(fresh_messages)
        assert [row["content"] for row in engine._store.get_session_messages("session-b")] == [
            "fresh session-b question",
            "fresh session-b answer",
        ]

    def test_compression_boundary_skip_preflight_cooldown_is_lossless(self, engine):
        """Preflight should ingest fresh messages but not request compression during cooldown."""
        engine.on_session_start("session-a", platform="telegram", context_length=200000)
        engine._last_compacted_store_id = 42
        engine._ingest_cursor = 3
        engine.on_session_start(
            "session-b",
            old_session_id="session-c",
            boundary_reason="compression",
            platform="telegram",
            context_length=200000,
        )
        engine.threshold_tokens = 1
        engine._config.leaf_chunk_tokens = 1
        engine._config.dynamic_leaf_chunk_enabled = False
        engine._config.fresh_tail_count = 1

        fresh_messages = [
            {"role": "user", "content": f"fresh preflight payload {idx}"}
            for idx in range(6)
        ]

        assert engine.should_compress_preflight(fresh_messages) is False
        assert engine._store.get_session_count("session-b") == len(fresh_messages)
        assert engine._ingest_cursor == len(fresh_messages)

    def test_compression_boundary_skip_preflight_cooldown_blocks_replay_diff(self, engine, monkeypatch):
        engine.on_session_start("session-a", platform="telegram", context_length=200000)
        engine.on_session_start(
            "session-b",
            old_session_id="session-c",
            boundary_reason="compression",
            platform="telegram",
            context_length=200000,
        )
        messages = [{"role": "assistant", "content": "raw assistant output"}]

        def replay_diff(_messages):
            return [{"role": "assistant", "content": "sanitized assistant output"}]

        monkeypatch.setattr(engine, "_ingest_messages", replay_diff)

        assert engine.should_compress_preflight(messages) is False

    def test_compression_cooldown_prevents_cascade_after_boundary_skip(self, engine):
        engine.on_session_start("session-a", platform="telegram", context_length=200000)
        engine.threshold_tokens = 100000
        engine._last_boundary_skip_time = time.monotonic()

        assert not engine.should_compress(200000)

    def test_compression_boundary_continues_logical_session_without_resetting_state(self, engine):
        engine.on_session_start("old-session", platform="telegram", context_length=200000)
        store_id = engine._store.append(
            "old-session",
            {"role": "user", "content": "important pre-rollover context"},
            token_estimate=17,
            source="telegram",
        )
        engine._dag.add_node(SummaryNode(
            session_id="old-session",
            depth=0,
            summary="pre-rollover summary",
            token_count=5,
            source_token_count=17,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 1
        engine.last_prompt_tokens = 1000
        engine.last_completion_tokens = 50
        engine.last_total_tokens = 1050
        engine._last_compacted_store_id = store_id
        engine._ingest_cursor = 2
        old_conversation_id = engine._conversation_id

        engine.on_session_start(
            "new-session",
            boundary_reason="compression",
            old_session_id="old-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-session"
        assert engine._conversation_id == old_conversation_id
        assert engine.compression_count == 1
        assert engine.last_prompt_tokens == 1000
        assert engine.last_completion_tokens == 50
        assert engine.last_total_tokens == 1050
        assert engine._last_compacted_store_id == store_id
        # No commit proof crossed the boundary: the child reconciles (#484 round 2).
        assert engine._ingest_cursor == 0 and engine._ingest_cursor_needs_reconcile
        assert engine._store.get_session_count("old-session") == 1
        assert engine._store.get_session_count("new-session") == 0
        assert engine._dag.get_session_nodes("old-session") == []
        new_nodes = engine._dag.get_session_nodes("new-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].summary == "pre-rollover summary"

        status = engine.get_status()
        assert status["store_messages"] == 0
        assert status["dag_nodes"] == 1
        assert status["compression_count"] == 1
        assert status["lifecycle"]["current_session_id"] == "new-session"
        assert status["lifecycle"]["last_finalized_session_id"] == "old-session"
        assert status["lifecycle"]["current_frontier_store_id"] == store_id
        assert status["lifecycle"]["last_finalized_frontier_store_id"] == store_id
        assert status["lifecycle"]["last_rollover_at"] is not None
        assert status["lifecycle"]["last_reset_at"] is None

    def test_compression_boundary_uses_bound_lcm_source_when_host_old_session_differs(self, engine):
        engine.on_session_start("lcm-source", platform="telegram", context_length=200000)
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "important LCM-bound context"},
            token_estimate=17,
            source="telegram",
        )
        stale_host_store_id = engine._store.append(
            "old-hermes-session",
            {"role": "user", "content": "unrelated stale host context"},
            token_estimate=11,
            source="telegram",
        )
        source_node_id = engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="LCM-bound summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        stale_host_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-hermes-session",
            depth=0,
            summary="stale host summary should not move",
            token_count=5,
            source_token_count=11,
            source_ids=[stale_host_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 2
        engine.last_prompt_tokens = 1000
        engine.last_completion_tokens = 50
        engine.last_total_tokens = 1050
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        old_conversation_id = engine._conversation_id

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-hermes-session"
        assert engine._conversation_id == old_conversation_id
        assert engine.compression_count == 2
        assert engine.last_prompt_tokens == 1000
        assert engine.last_completion_tokens == 50
        assert engine.last_total_tokens == 1050
        assert engine._last_compacted_store_id == source_store_id
        # No commit proof crossed the boundary: the child reconciles (#484 round 2).
        assert engine._ingest_cursor == 0 and engine._ingest_cursor_needs_reconcile
        assert engine._store.get_session_count("lcm-source") == 1
        assert engine._store.get_session_count("new-hermes-session") == 0
        assert engine._store.get_session_count("old-hermes-session") == 1
        assert engine._dag.get_session_nodes("lcm-source") == []
        new_nodes = engine._dag.get_session_nodes("new-hermes-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].node_id == source_node_id
        assert new_nodes[0].summary == "LCM-bound summary"
        stale_host_node = engine._dag.get_node(stale_host_node_id)
        assert stale_host_node is not None
        assert stale_host_node.session_id == "old-hermes-session"

        status = engine.get_status()
        assert status["store_messages"] == 0
        assert status["dag_nodes"] == 1
        assert status["compression_count"] == 2
        expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": source_node_id}))
        assert expanded["expanded"][0]["content"] == "important LCM-bound context"

    def test_compression_boundary_prefers_active_bound_source_over_stale_finalized_host(self, engine):
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="shared-conversation",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "active bound context must move"},
            token_estimate=17,
            source="telegram",
        )
        stale_host_store_id = engine._store.append(
            "old-hermes-session",
            {"role": "user", "content": "stale finalized host context must stay put"},
            token_estimate=11,
            source="telegram",
        )
        source_node_id = engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="active bound summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        stale_host_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-hermes-session",
            depth=0,
            summary="stale finalized host summary should not move",
            token_count=5,
            source_token_count=11,
            source_ids=[stale_host_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._lifecycle.record_rollover(
            "shared-conversation",
            old_session_id="old-hermes-session",
            new_session_id="lcm-source",
            finalized_frontier_store_id=stale_host_store_id,
        )
        engine._lifecycle.advance_frontier(
            "shared-conversation",
            "lcm-source",
            source_store_id,
        )
        lifecycle_before = engine._lifecycle.get_by_conversation("shared-conversation")
        assert lifecycle_before is not None
        assert lifecycle_before.current_session_id == "lcm-source"
        assert lifecycle_before.last_finalized_session_id == "old-hermes-session"
        engine.compression_count = 2
        engine.last_prompt_tokens = 1000
        engine.last_completion_tokens = 50
        engine.last_total_tokens = 1050
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-hermes-session"
        assert engine._conversation_id == "shared-conversation"
        assert engine._store.get_session_count("lcm-source") == 1
        assert engine._store.get_session_count("new-hermes-session") == 0
        assert engine._store.get_session_count("old-hermes-session") == 1
        assert engine._dag.get_session_nodes("lcm-source") == []
        new_nodes = engine._dag.get_session_nodes("new-hermes-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].node_id == source_node_id
        stale_host_node = engine._dag.get_node(stale_host_node_id)
        assert stale_host_node is not None
        assert stale_host_node.session_id == "old-hermes-session"
        expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": source_node_id}))
        assert expanded["expanded"][0]["content"] == "active bound context must move"

    def test_compression_boundary_uses_finalized_bound_lcm_source_when_host_old_session_differs(self, engine):
        engine.on_session_start("lcm-source", platform="telegram", context_length=200000)
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "finalized LCM-bound context"},
            token_estimate=17,
            source="telegram",
        )
        stale_host_store_id = engine._store.append(
            "old-hermes-session",
            {"role": "user", "content": "unrelated stale host context"},
            token_estimate=11,
            source="telegram",
        )
        source_node_id = engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="finalized LCM-bound summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        stale_host_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-hermes-session",
            depth=0,
            summary="stale host summary should not move",
            token_count=5,
            source_token_count=11,
            source_ids=[stale_host_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 2
        engine.last_prompt_tokens = 1000
        engine.last_completion_tokens = 50
        engine.last_total_tokens = 1050
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        old_conversation_id = engine._conversation_id
        engine._lifecycle.finalize_session(
            old_conversation_id,
            "lcm-source",
            frontier_store_id=source_store_id,
        )
        finalized = engine._lifecycle.get_by_conversation(old_conversation_id)
        assert finalized is not None
        assert finalized.current_session_id is None
        assert finalized.last_finalized_session_id == "lcm-source"
        # Prove the rollover restores the finalized lifecycle frontier, not only
        # the engine's in-memory compacted marker.
        engine._last_compacted_store_id = 0

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-hermes-session"
        assert engine._conversation_id == old_conversation_id
        assert engine.compression_count == 2
        assert engine.last_prompt_tokens == 1000
        assert engine.last_completion_tokens == 50
        assert engine.last_total_tokens == 1050
        assert engine._last_compacted_store_id == source_store_id
        # No commit proof crossed the boundary: the child reconciles (#484 round 2).
        assert engine._ingest_cursor == 0 and engine._ingest_cursor_needs_reconcile
        assert engine._store.get_session_count("lcm-source") == 1
        assert engine._store.get_session_count("new-hermes-session") == 0
        assert engine._store.get_session_count("old-hermes-session") == 1
        assert engine._dag.get_session_nodes("lcm-source") == []
        new_nodes = engine._dag.get_session_nodes("new-hermes-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].node_id == source_node_id
        stale_host_node = engine._dag.get_node(stale_host_node_id)
        assert stale_host_node is not None
        assert stale_host_node.session_id == "old-hermes-session"
        lifecycle = engine._lifecycle.get_by_conversation(old_conversation_id)
        assert lifecycle is not None
        assert lifecycle.current_session_id == "new-hermes-session"
        assert lifecycle.last_finalized_session_id == "lcm-source"
        assert lifecycle.current_frontier_store_id == source_store_id
        assert lifecycle.last_finalized_frontier_store_id == source_store_id
        expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": source_node_id}))
        assert expanded["expanded"][0]["content"] == "finalized LCM-bound context"

    def test_compression_boundary_rejects_bound_source_for_explicit_conversation_mismatch(self, engine):
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="conversation-a",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "conversation A context"},
            token_estimate=17,
            source="telegram",
        )
        stale_host_store_id = engine._store.append(
            "old-hermes-session",
            {"role": "user", "content": "unrelated stale host context"},
            token_estimate=11,
            source="telegram",
        )
        source_node_id = engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="conversation A summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        stale_host_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-hermes-session",
            depth=0,
            summary="stale host summary should not move",
            token_count=5,
            source_token_count=11,
            source_ids=[stale_host_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 2
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        engine._lifecycle.finalize_session(
            "conversation-a",
            "lcm-source",
            frontier_store_id=source_store_id,
        )

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",
            conversation_id="conversation-b",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-hermes-session"
        assert engine._conversation_id == "conversation-b"
        assert engine.compression_count == 0
        assert engine._last_compacted_store_id == 0
        assert engine._ingest_cursor == 0
        assert engine._store.get_session_count("lcm-source") == 1
        assert engine._store.get_session_count("new-hermes-session") == 0
        assert engine._store.get_session_count("old-hermes-session") == 1
        source_node = engine._dag.get_node(source_node_id)
        assert source_node is not None
        assert source_node.session_id == "lcm-source"
        stale_host_node = engine._dag.get_node(stale_host_node_id)
        assert stale_host_node is not None
        assert stale_host_node.session_id == "old-hermes-session"
        conversation_a = engine._lifecycle.get_by_conversation("conversation-a")
        assert conversation_a is not None
        assert conversation_a.current_session_id is None
        assert conversation_a.last_finalized_session_id == "lcm-source"
        conversation_b = engine._lifecycle.get_by_conversation("conversation-b")
        assert conversation_b is not None
        assert conversation_b.current_session_id == "new-hermes-session"

    # ── Sibling-chain fallback tests (PR #242, zero-DAG host) ──

    def test_compression_boundary_sibling_chain_zero_dag_host_positive(
        self, engine,
    ):
        """Active bound sibling with zero-DAG host — fallback activates."""
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="conversation-a",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "sibling chain context must move"},
            token_estimate=17,
            source="telegram",
        )
        engine._store.append(
            "old-hermes-session",
            {"role": "user", "content": "stale host context must stay"},
            token_estimate=11,
            source="telegram",
        )
        source_node_id = engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="sibling chain summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        # Stale host gets zero DAG nodes — the trigger condition
        engine.compression_count = 3
        engine.last_prompt_tokens = 1000
        engine.last_completion_tokens = 50
        engine.last_total_tokens = 1050
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        engine._lifecycle.record_rollover(
            "conversation-a",
            old_session_id="old-hermes-session",
            new_session_id="lcm-source",
            finalized_frontier_store_id=0,
        )
        engine._lifecycle.advance_frontier(
            "conversation-a",
            "lcm-source",
            source_store_id,
        )
        # Verify active bound with parent=old-hermes-session
        conv_a = engine._lifecycle.get_by_conversation("conversation-a")
        assert conv_a is not None
        assert conv_a.current_session_id == "lcm-source"
        assert conv_a.last_finalized_session_id == "old-hermes-session"

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",
            conversation_id="conversation-b",
            platform="telegram",
            context_length=200000,
        )

        # Fallback activated — nodes transferred
        assert engine._session_id == "new-hermes-session"
        assert engine._conversation_id == "conversation-a"  # source wins
        assert engine.compression_count == 3
        assert engine._last_compacted_store_id == source_store_id
        # No commit proof crossed the boundary: the child reconciles (#484 round 2).
        assert engine._ingest_cursor == 0 and engine._ingest_cursor_needs_reconcile
        assert engine._store.get_session_count("lcm-source") == 1
        assert engine._store.get_session_count("new-hermes-session") == 0
        assert engine._store.get_session_count("old-hermes-session") == 1
        assert engine._dag.get_session_nodes("lcm-source") == []
        new_nodes = engine._dag.get_session_nodes("new-hermes-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].node_id == source_node_id
        # Stale host message stays put (zero DAG means no node to check)
        assert engine._store.get_session_count("old-hermes-session") == 1
        # Content verifiable
        expanded = json.loads(
            engine.handle_tool_call("lcm_expand", {"node_id": source_node_id}),
        )
        assert expanded["expanded"][0]["content"] == "sibling chain context must move"

    def test_compression_boundary_sibling_chain_bound_no_dag_negative(
        self, engine,
    ):
        """Bound source has no DAG — fallback deactivated."""
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="conversation-a",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "context should not transfer"},
            token_estimate=17,
            source="telegram",
        )
        # Zero DAG for bound session
        engine.compression_count = 3
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        engine._lifecycle.finalize_session(
            "conversation-a",
            "lcm-source",
            frontier_store_id=source_store_id,
        )

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",  # zero DAG
            conversation_id="conversation-b",
            platform="telegram",
            context_length=200000,
        )

        # Fallback rejected — bound_has_summary_nodes guard failed
        assert engine.compression_count == 0  # reset
        assert engine._last_compacted_store_id == 0
        assert engine._store.get_session_count("lcm-source") == 1
        assert engine._store.get_session_count("new-hermes-session") == 0

    def test_compression_boundary_sibling_chain_parent_mismatch_negative(
        self, engine,
    ):
        """Bound session has different parent — fallback deactivated."""
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="conversation-a",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "parent mismatch context"},
            token_estimate=17,
            source="telegram",
        )
        engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="parent mismatch summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 3
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        # Finalize with a DIFFERENT parent from old_session_id
        engine._lifecycle.finalize_session(
            "conversation-a",
            "lcm-source",
            frontier_store_id=source_store_id,
        )
        # Override last_finalized — it will stay as "lcm-source", not
        # matching "old-hermes-session" that we will pass as old_session_id
        conv_a = engine._lifecycle.get_by_conversation("conversation-a")
        assert conv_a.last_finalized_session_id == "lcm-source"

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="other-parent-session",  # different parent
            conversation_id="conversation-b",
            platform="telegram",
            context_length=200000,
        )

        # Fallback rejected — bound_shares_parent_with_host guard failed
        assert engine.compression_count == 0
        assert engine._last_compacted_store_id == 0
        assert engine._store.get_session_count("lcm-source") == 1

    def test_compression_boundary_sibling_chain_host_has_dag_negative(
        self, engine,
    ):
        """Host old_session_id has DAG — falls to host-authoritative path."""
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="conversation-a",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "bound source context"},
            token_estimate=17,
            source="telegram",
        )
        host_store_id = engine._store.append(
            "old-hermes-session",
            {"role": "user", "content": "host DAG must survive"},
            token_estimate=11,
            source="telegram",
        )
        engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="bound summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        host_node_id = engine._dag.add_node(SummaryNode(
            session_id="old-hermes-session",
            depth=0,
            summary="host summary must stay",
            token_count=5,
            source_token_count=11,
            source_ids=[host_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 3
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        engine._lifecycle.finalize_session(
            "conversation-a",
            "lcm-source",
            frontier_store_id=source_store_id,
        )

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",  # HAS DAG
            conversation_id="conversation-b",
            platform="telegram",
            context_length=200000,
        )

        # Sibling-chain fallback deactivated — host_has_no_dag guard failed
        # Host DAG stays put
        host_node = engine._dag.get_node(host_node_id)
        assert host_node is not None
        assert host_node.session_id == "old-hermes-session"
        # Bound session NOT transferred
        assert engine._store.get_session_count("lcm-source") == 1

    def test_compression_boundary_sibling_chain_active_source_different_conv_id(
        self, engine,
    ):
        """Active bound sibling with explicit conversation_id mismatch —
        fallback activates despite mismatched conversations."""
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="conversation-x",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "active sibling with diff conv"},
            token_estimate=17,
            source="telegram",
        )
        source_node_id = engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="active sibling with diff conv summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 3
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        engine._lifecycle.record_rollover(
            "conversation-x",
            old_session_id="old-hermes-session",
            new_session_id="lcm-source",
            finalized_frontier_store_id=0,
        )
        engine._lifecycle.advance_frontier(
            "conversation-x",
            "lcm-source",
            source_store_id,
        )
        # Verify active bound with parent=old-hermes-session
        conv_x = engine._lifecycle.get_by_conversation("conversation-x")
        assert conv_x.current_session_id == "lcm-source"
        assert conv_x.last_finalized_session_id == "old-hermes-session"

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",
            conversation_id="conversation-y",
            platform="telegram",
            context_length=200000,
        )

        # Fallback activated — nodes transferred, source conv wins
        assert engine._conversation_id == "conversation-x"
        assert engine.compression_count == 3
        assert engine._store.get_session_count("lcm-source") == 1
        assert engine._store.get_session_count("new-hermes-session") == 0
        assert engine._dag.get_session_nodes("lcm-source") == []
        new_nodes = engine._dag.get_session_nodes("new-hermes-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].node_id == source_node_id

    def test_compression_boundary_sibling_chain_source_none_kwargs_fallback(
        self, engine,
    ):
        """source_state is None — conversation_id falls to kwargs."""
        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",  # zero DAG, no bound
            conversation_id="conversation-c",
            platform="telegram",
            context_length=200000,
        )

        # All guards fail — source_state is None
        # conversation_id falls through: kwargs → self._conversation_id
        assert engine._conversation_id == "conversation-c"

    def test_compression_boundary_sibling_chain_conversation_id_regression(
        self, engine,
    ):
        """Sibling-chain fallback → conversation_id from bound session,
        NOT kwargs. Regression test for the bug stephenschoettler found."""
        engine.on_session_start(
            "lcm-source",
            platform="telegram",
            context_length=200000,
            conversation_id="conversation-a",
        )
        source_store_id = engine._store.append(
            "lcm-source",
            {"role": "user", "content": "regression test context"},
            token_estimate=17,
            source="telegram",
        )
        source_node_id = engine._dag.add_node(SummaryNode(
            session_id="lcm-source",
            depth=0,
            summary="regression test summary",
            token_count=5,
            source_token_count=17,
            source_ids=[source_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine.compression_count = 3
        engine._last_compacted_store_id = source_store_id
        engine._ingest_cursor = 2
        engine._lifecycle.record_rollover(
            "conversation-a",
            old_session_id="old-hermes-session",
            new_session_id="lcm-source",
            finalized_frontier_store_id=0,
        )
        engine._lifecycle.advance_frontier(
            "conversation-a",
            "lcm-source",
            source_store_id,
        )

        engine.on_session_start(
            "new-hermes-session",
            boundary_reason="compression",
            old_session_id="old-hermes-session",
            conversation_id="conversation-b",
            platform="telegram",
            context_length=200000,
        )

        # KEY ASSERTION: conversation_id = "conversation-a" (source wins)
        # NOT "conversation-b" (which kwargs would produce pre-fix)
        assert engine._conversation_id == "conversation-a"
        assert engine._session_id == "new-hermes-session"
        assert engine.compression_count == 3
        # Verify the lifecycle under conversation-a is aware of the new session
        conv_a = engine._lifecycle.get_by_conversation("conversation-a")
        assert conv_a is not None
        assert conv_a.current_session_id == "new-hermes-session"
        # conversation-b not created — session bound to conversation-a
        conv_b = engine._lifecycle.get_by_conversation("conversation-b")
        assert conv_b is None
        # Nodes moved correctly
        assert engine._dag.get_session_nodes("lcm-source") == []
        new_nodes = engine._dag.get_session_nodes("new-hermes-session")
        assert len(new_nodes) == 1
        assert new_nodes[0].node_id == source_node_id

    def test_compression_boundary_prefers_host_old_session_when_bound_session_drifted(self, engine):
        engine.on_session_start(
            "foreground-old",
            conversation_id="foreground-conversation",
            platform="telegram",
            context_length=200000,
        )
        store_id = engine._store.append(
            "foreground-old",
            {"role": "user", "content": "foreground DAG must survive drift"},
            token_estimate=17,
            source="telegram",
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="foreground-old",
            depth=0,
            summary="foreground summary before drift",
            token_count=5,
            source_token_count=17,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = store_id
        engine._lifecycle.advance_frontier(
            "foreground-conversation",
            "foreground-old",
            store_id,
        )

        # A short-lived session binds after the foreground compaction. It has no
        # DAG nodes and is not the host-authoritative session that compressed.
        engine.on_session_start(
            "short-lived-auxiliary",
            conversation_id="auxiliary-conversation",
            platform="cron",
            context_length=200000,
        )
        assert engine._session_id == "short-lived-auxiliary"
        assert engine._dag.get_session_nodes("short-lived-auxiliary") == []

        engine.on_session_start(
            "foreground-new",
            boundary_reason="compression",
            old_session_id="foreground-old",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "foreground-new"
        assert engine._conversation_id == "foreground-conversation"
        assert engine._store.get_session_count("foreground-old") == 1
        assert engine._store.get_session_count("foreground-new") == 0
        assert engine._dag.get_session_nodes("foreground-old") == []
        new_nodes = engine._dag.get_session_nodes("foreground-new")
        assert [node.node_id for node in new_nodes] == [node_id]
        assert engine._last_compacted_store_id == store_id
        lifecycle = engine._lifecycle.get_by_conversation("foreground-conversation")
        assert lifecycle is not None
        assert lifecycle.current_session_id == "foreground-new"
        assert lifecycle.last_finalized_session_id == "foreground-old"
        assert lifecycle.current_frontier_store_id == store_id
        expanded = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))
        assert expanded["expanded"][0]["content"] == "foreground DAG must survive drift"

    def test_compression_boundary_scopes_frontier_to_host_old_session_when_bound_session_drifted(self, engine):
        engine.on_session_start(
            "foreground-old",
            conversation_id="foreground-conversation",
            platform="telegram",
            context_length=200000,
        )
        fg_store_id = engine._store.append(
            "foreground-old",
            {"role": "user", "content": "foreground frontier must stay scoped"},
            token_estimate=17,
            source="telegram",
        )
        fg_node_id = engine._dag.add_node(SummaryNode(
            session_id="foreground-old",
            depth=0,
            summary="foreground scoped frontier summary",
            token_count=5,
            source_token_count=17,
            source_ids=[fg_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = fg_store_id
        engine._lifecycle.advance_frontier(
            "foreground-conversation",
            "foreground-old",
            fg_store_id,
        )

        engine.on_session_start(
            "aux-old",
            conversation_id="aux-conversation",
            platform="cron",
            context_length=200000,
        )
        aux_store_id = engine._store.append(
            "aux-old",
            {"role": "user", "content": "auxiliary frontier must not leak"},
            token_estimate=13,
            source="cron",
        )
        aux_node_id = engine._dag.add_node(SummaryNode(
            session_id="aux-old",
            depth=0,
            summary="auxiliary summary should stay put",
            token_count=5,
            source_token_count=13,
            source_ids=[aux_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        assert aux_store_id > fg_store_id
        engine._last_compacted_store_id = aux_store_id
        engine._lifecycle.advance_frontier(
            "aux-conversation",
            "aux-old",
            aux_store_id,
        )

        engine.on_session_start(
            "foreground-new",
            boundary_reason="compression",
            old_session_id="foreground-old",
            platform="telegram",
            context_length=200000,
        )

        lifecycle = engine._lifecycle.get_by_conversation("foreground-conversation")
        assert lifecycle is not None
        assert lifecycle.current_session_id == "foreground-new"
        assert lifecycle.last_finalized_session_id == "foreground-old"
        assert lifecycle.current_frontier_store_id == fg_store_id
        assert lifecycle.last_finalized_frontier_store_id == fg_store_id
        assert engine._last_compacted_store_id == fg_store_id
        assert engine._store.get_session_count("foreground-old") == 1
        assert engine._store.get_session_count("foreground-new") == 0
        assert engine._store.get_session_count("aux-old") == 1
        assert engine._dag.get_session_nodes("foreground-old") == []
        new_nodes = engine._dag.get_session_nodes("foreground-new")
        assert [node.node_id for node in new_nodes] == [fg_node_id]
        aux_node = engine._dag.get_node(aux_node_id)
        assert aux_node is not None
        assert aux_node.session_id == "aux-old"
        aux_state = engine._lifecycle.get_by_conversation("aux-conversation")
        assert aux_state is not None
        assert aux_state.current_frontier_store_id == aux_store_id

    def test_compression_boundary_uses_conversation_row_when_auxiliary_rows_reference_host_id(self, engine):
        engine.on_session_start(
            "foreground-active",
            conversation_id="host-conversation",
            platform="telegram",
            context_length=200000,
        )
        fg_store_id = engine._store.append(
            "foreground-active",
            {"role": "user", "content": "foreground conversation row must win"},
            token_estimate=17,
            source="telegram",
        )
        fg_node_id = engine._dag.add_node(SummaryNode(
            session_id="foreground-active",
            depth=0,
            summary="foreground active summary",
            token_count=5,
            source_token_count=17,
            source_ids=[fg_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = fg_store_id
        engine._lifecycle.advance_frontier(
            "host-conversation",
            "foreground-active",
            fg_store_id,
        )

        # Auxiliary lifecycle rows can reference the host conversation id as a
        # finalized session. They must not outrank the real conversation row
        # just because their updated_at is newer.
        engine._lifecycle.record_rollover(
            "auxiliary-conversation",
            old_session_id="host-conversation",
            new_session_id="drifted-auxiliary",
            finalized_frontier_store_id=99,
        )
        engine.on_session_start(
            "drifted-auxiliary",
            conversation_id="auxiliary-conversation",
            platform="cron",
            context_length=200000,
        )
        wrong_state = engine._lifecycle.get_by_session("host-conversation")
        assert wrong_state is not None
        assert wrong_state.conversation_id == "auxiliary-conversation"

        engine.on_session_start(
            "foreground-new",
            boundary_reason="compression",
            old_session_id="host-conversation",
            platform="telegram",
            context_length=200000,
        )

        lifecycle = engine._lifecycle.get_by_conversation("host-conversation")
        assert lifecycle is not None
        assert lifecycle.current_session_id == "foreground-new"
        assert lifecycle.last_finalized_session_id == "foreground-active"
        assert lifecycle.current_frontier_store_id == fg_store_id
        assert lifecycle.last_finalized_frontier_store_id == fg_store_id
        assert engine._store.get_session_count("foreground-active") == 1
        assert engine._store.get_session_count("foreground-new") == 0
        assert engine._dag.get_session_nodes("foreground-active") == []
        new_nodes = engine._dag.get_session_nodes("foreground-new")
        assert [node.node_id for node in new_nodes] == [fg_node_id]
        aux_state = engine._lifecycle.get_by_conversation("auxiliary-conversation")
        assert aux_state is not None
        assert aux_state.current_session_id == "drifted-auxiliary"
        assert aux_state.last_finalized_session_id == "host-conversation"

    def test_compression_boundary_mismatch_resets_session_scoped_state(self, engine):
        engine.on_session_start("bound-session", platform="telegram", context_length=200000)
        engine.compression_count = 3
        engine.last_prompt_tokens = 900
        engine.last_completion_tokens = 12
        engine.last_total_tokens = 912
        engine._last_compacted_store_id = 42
        engine._ingest_cursor = 7
        old_conversation_id = engine._conversation_id

        engine.on_session_start(
            "new-session",
            boundary_reason="compression",
            old_session_id="different-old-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "new-session"
        assert engine._conversation_id != old_conversation_id
        assert engine.compression_count == 0
        assert engine.last_prompt_tokens == 0
        assert engine.last_completion_tokens == 0
        assert engine.last_total_tokens == 0
        assert engine._last_compacted_store_id == 0
        assert engine._ingest_cursor == 0

    def test_compression_boundary_preserves_externalized_payload_session_metadata(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_compression_externalized.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=200,
        )
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
        engine.on_session_start("old-session", platform="telegram", context_length=200000)

        content = "RESULT:\n" + ("abcdef" * 2000)
        engine._serialize_messages([
            {"role": "tool", "tool_call_id": "call_big", "content": content}
        ])
        payload_file = next((tmp_path / "hermes" / "lcm-large-outputs").glob("*.json"))
        placeholder = (
            "[Externalized tool output: tool_call_id=call_big; "
            f"chars={len(content)}; bytes={len(content.encode('utf-8'))}; ref={payload_file.name}]"
        )
        store_id = engine._store.append(
            "old-session",
            {"role": "tool", "tool_call_id": "call_big", "content": placeholder},
            token_estimate=17,
            source="telegram",
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="old-session",
            depth=0,
            summary="Externalized tool-output summary",
            token_count=10,
            source_token_count=17,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        assert json.loads(payload_file.read_text())["session_id"] == "old-session"

        engine.on_session_start(
            "new-session",
            boundary_reason="compression",
            old_session_id="old-session",
            platform="telegram",
            context_length=200000,
        )

        assert json.loads(payload_file.read_text())["session_id"] == "old-session"
        result = json.loads(engine.handle_tool_call("lcm_expand", {"node_id": node_id}))
        assert result["expanded"][0]["externalized"]["session_id"] == "old-session"
        assert result["expanded"][0]["externalized"]["tool_call_id"] == "call_big"

    def test_rollover_session_records_durable_lifecycle_state_idempotently(self, engine):
        engine._config.new_session_retain_depth = 2
        from hermes_lcm.dag import SummaryNode
        import time

        engine.on_session_start("s1", platform="cli", context_length=200000)
        for depth in range(3):
            engine._dag.add_node(SummaryNode(
                session_id="s1", depth=depth,
                summary=f"seed d{depth}", token_count=100,
                source_token_count=500, source_ids=[],
                source_type="messages", created_at=time.time(),
            ))

        moved = engine.rollover_session("s1", "s2", previous_messages=[], platform="cli", context_length=200000)
        assert moved == 1

        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state is not None
        assert state.current_session_id == "s2"
        assert state.last_finalized_session_id == "s1"

        moved_repeat = engine.rollover_session("s1", "s2", previous_messages=[], platform="cli", context_length=200000)
        assert moved_repeat == 0

        state_repeat = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state_repeat is not None
        assert state_repeat.current_session_id == "s2"
        assert state_repeat.last_finalized_session_id == "s1"
        assert engine._lifecycle.row_count() == 1

    def test_legacy_reset_then_start_finalizes_old_lifecycle_before_new_bind(self, engine):
        engine.on_session_start("legacy-old", platform="cli", context_length=200000)
        store_id = engine._store.append(
            "legacy-old",
            {"role": "user", "content": "legacy host context before /new"},
            token_estimate=17,
            source="cli",
        )
        engine._last_compacted_store_id = store_id
        old_conversation_id = engine._conversation_id

        # Older Hermes hosts may not call rollover_session(...) yet. They can
        # still call the older lifecycle pair: reset current state, then bind a
        # fresh session. LCM must not leave the old conversation marked current.
        engine.on_session_reset()
        engine.on_session_start("legacy-new", platform="cli", context_length=200000)

        old_state = engine._lifecycle.get_by_conversation(old_conversation_id)
        assert old_state is not None
        assert old_state.current_session_id is None
        assert old_state.last_finalized_session_id == "legacy-old"
        assert old_state.last_finalized_frontier_store_id == store_id
        assert old_state.last_reset_at is not None

        new_state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert new_state is not None
        assert new_state.conversation_id == "legacy-new"
        assert new_state.current_session_id == "legacy-new"
        assert new_state.last_finalized_session_id is None
        assert engine._lifecycle.row_count() == 2

    def test_same_session_reset_keeps_lifecycle_current_and_allows_future_frontier_updates(self, engine):
        engine.on_session_start("same-session", platform="cli", context_length=200000)
        first_store_id = engine._store.append(
            "same-session",
            {"role": "user", "content": "before reset"},
            token_estimate=7,
            source="cli",
        )
        engine._last_compacted_store_id = first_store_id

        engine.on_session_reset()
        after_reset = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert after_reset is not None
        assert after_reset.current_session_id == "same-session"
        assert after_reset.last_finalized_session_id is None
        assert after_reset.last_reset_at is not None

        second_store_id = engine._store.append(
            "same-session",
            {"role": "assistant", "content": "after reset"},
            token_estimate=9,
            source="cli",
        )
        engine._last_compacted_store_id = second_store_id
        engine._persist_frontier_marker()

        after_frontier = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert after_frontier is not None
        assert after_frontier.current_session_id == "same-session"
        assert after_frontier.current_frontier_store_id == second_store_id

    def test_same_session_reset_then_later_new_session_preserves_latest_frontier(self, engine):
        engine.on_session_start("same-then-new", platform="cli", context_length=200000)
        first_store_id = engine._store.append(
            "same-then-new",
            {"role": "user", "content": "before reset"},
            token_estimate=7,
            source="cli",
        )
        engine._last_compacted_store_id = first_store_id
        old_conversation_id = engine._conversation_id

        engine.on_session_reset()
        second_store_id = engine._store.append(
            "same-then-new",
            {"role": "assistant", "content": "same session continued after reset"},
            token_estimate=11,
            source="cli",
        )
        engine._last_compacted_store_id = second_store_id
        engine._persist_frontier_marker()

        engine.on_session_start("eventual-new", platform="cli", context_length=200000)

        old_state = engine._lifecycle.get_by_conversation(old_conversation_id)
        assert old_state is not None
        assert old_state.current_session_id is None
        assert old_state.last_finalized_session_id == "same-then-new"
        assert old_state.last_finalized_frontier_store_id == second_store_id
        assert engine._pending_reset_session_id == ""

    def test_reset_before_compression_boundary_does_not_leave_stale_pending_reset(self, engine):
        engine.on_session_start("compress-old", platform="telegram", context_length=200000)
        store_id = engine._store.append(
            "compress-old",
            {"role": "user", "content": "compression boundary after reset"},
            token_estimate=13,
            source="telegram",
        )
        engine._dag.add_node(SummaryNode(
            session_id="compress-old",
            depth=0,
            summary="compression summary",
            token_count=5,
            source_token_count=13,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = store_id

        engine.on_session_reset()
        assert engine._pending_reset_session_id == "compress-old"

        engine.on_session_start(
            "compress-new",
            boundary_reason="compression",
            old_session_id="compress-old",
            platform="telegram",
            context_length=200000,
        )

        assert engine._pending_reset_session_id == ""
        status = engine.get_status()
        assert status["lifecycle"]["current_session_id"] == "compress-new"
        assert status["lifecycle"]["last_finalized_session_id"] == "compress-old"
        assert status["lifecycle"]["last_finalized_frontier_store_id"] == store_id

    def test_reset_before_compression_boundary_mismatch_finalizes_pending_old_session(self, engine):
        engine.on_session_start("bound-after-reset", platform="telegram", context_length=200000)
        store_id = engine._store.append(
            "bound-after-reset",
            {"role": "user", "content": "pending reset before mismatch"},
            token_estimate=13,
            source="telegram",
        )
        engine._dag.add_node(SummaryNode(
            session_id="bound-after-reset",
            depth=0,
            summary="will be pruned on reset",
            token_count=5,
            source_token_count=13,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = store_id
        old_conversation_id = engine._conversation_id

        engine.on_session_reset()
        assert engine._pending_reset_session_id == "bound-after-reset"

        engine.on_session_start(
            "new-after-mismatch",
            boundary_reason="compression",
            old_session_id="stale-host-session",
            platform="telegram",
            context_length=200000,
        )

        old_state = engine._lifecycle.get_by_conversation(old_conversation_id)
        assert old_state is not None
        assert old_state.current_session_id is None
        assert old_state.last_finalized_session_id == "bound-after-reset"
        assert old_state.last_finalized_frontier_store_id == store_id
        assert engine._pending_reset_session_id == ""

    def test_on_session_start_recovers_durable_lifecycle_state_after_restart(self, engine, monkeypatch):
        engine.on_session_start("active-session", platform="cli", context_length=200000)
        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("durable summary", 1),
        )

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "alpha " * 80},
            {"role": "assistant", "content": "beta " * 80},
            {"role": "user", "content": "gamma " * 80},
            {"role": "assistant", "content": "delta " * 80},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta"},
        ]
        engine.compress(messages)
        old_conversation_id = engine._conversation_id
        old_frontier = engine._last_compacted_store_id
        assert old_frontier > 0

        restarted = LCMEngine(config=engine._config)
        restarted.on_session_start("active-session", platform="cli", context_length=200000)

        assert restarted._conversation_id == old_conversation_id
        assert restarted._last_compacted_store_id == old_frontier
        recovered = restarted._lifecycle.get_by_conversation(old_conversation_id)
        assert recovered is not None
        assert recovered.current_session_id == "active-session"

    def test_bind_lifecycle_gc_prunes_empty_rows_above_threshold(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_lifecycle.db"),
            empty_lifecycle_gc_enabled=True,
            empty_lifecycle_gc_threshold=1,
        )
        engine = LCMEngine(config=config)
        try:
            # Create stale orphan rows by binding sessions with no data.
            for i in range(5):
                engine._lifecycle.bind_session(f"orphan-{i}")
            stale_ts = time.time() - (25 * 3600)
            engine._lifecycle._conn.execute(
                """
                UPDATE lcm_lifecycle_state
                SET current_bound_at = ?, updated_at = ?
                """,
                (stale_ts, stale_ts),
            )
            engine._lifecycle._conn.commit()
            assert engine._lifecycle.row_count() == 5

            # Bind to a new session — should trigger GC since threshold(1) < 5.
            engine.on_session_start("live-session", platform="cli", context_length=200000)
            # All 5 stale empty rows should be pruned, leaving only the live one.
            assert engine._lifecycle.row_count() == 1
            state = engine._lifecycle.get_by_conversation("live-session")
            assert state is not None
            assert state.current_session_id == "live-session"
        finally:
            engine.shutdown()

    def test_bind_lifecycle_gc_preserves_recent_empty_active_session_from_other_engine(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_active_empty.db"),
            empty_lifecycle_gc_enabled=True,
            empty_lifecycle_gc_threshold=1,
        )
        engine_a = LCMEngine(config=config)
        engine_b = LCMEngine(config=config)
        try:
            engine_a.on_session_start("active-other", platform="cli", context_length=200000)
            assert engine_a._lifecycle.get_by_session("active-other") is not None

            # A second engine sharing the same DB may start before the first
            # engine has ingested its first message. Startup GC must not treat
            # that recently-bound empty row as an orphan.
            engine_b.on_session_start("gc-trigger", platform="cli", context_length=200000)

            assert engine_a._lifecycle.get_by_session("active-other") is not None
            assert engine_b._lifecycle.get_by_session("active-other") is not None

            engine_a._ingest_messages([{"role": "user", "content": "first persisted message"}])
            assert engine_a._store.get_session_count("active-other") == 1
            assert engine_a._lifecycle.get_by_session("active-other") is not None
        finally:
            engine_a.shutdown()
            engine_b.shutdown()

    def test_bind_lifecycle_gc_skips_when_below_threshold(self, tmp_path, monkeypatch):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_gc_below_threshold.db"),
            empty_lifecycle_gc_enabled=True,
            empty_lifecycle_gc_threshold=10,
        )
        engine = LCMEngine(config=config)

        # Create 3 orphan rows — below threshold of 10
        for i in range(3):
            engine._lifecycle.bind_session(f"orphan-{i}")

        engine.on_session_start("live-session", platform="cli", context_length=200000)
        # Should NOT prune because 4 < threshold(10)
        assert engine._lifecycle.row_count() == 4
        engine.shutdown()

    def test_frontier_marker_only_advances_after_successful_leaf_compaction(self, engine, monkeypatch):
        engine.on_session_start("frontier-session", platform="cli", context_length=200000)
        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("frontier summary", 1),
        )

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "alpha " * 80},
            {"role": "assistant", "content": "beta " * 80},
            {"role": "user", "content": "gamma " * 80},
            {"role": "assistant", "content": "delta " * 80},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta"},
        ]
        engine.compress(messages)

        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state is not None
        assert state.current_frontier_store_id == engine._last_compacted_store_id
        frontier_before_failure = state.current_frontier_store_id

        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: (_ for _ in ()).throw(TimeoutError("summary timed out")),
        )
        failing_messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta " * 80},
            {"role": "user", "content": "eta " * 80},
            {"role": "assistant", "content": "theta " * 80},
            {"role": "user", "content": "iota " * 80},
            {"role": "assistant", "content": "kappa"},
        ]
        with pytest.raises(TimeoutError):
            engine.compress(failing_messages)

        after_failure = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert after_failure is not None
        assert after_failure.current_frontier_store_id == frontier_before_failure

    def test_rollover_resets_active_frontier_but_preserves_last_finalized_frontier(self, engine, monkeypatch):
        engine.on_session_start("frontier-old", platform="cli", context_length=200000)
        monkeypatch.setattr(
            lcm_engine,
            "summarize_with_escalation",
            lambda **kwargs: ("rollover summary", 1),
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "alpha " * 80},
            {"role": "assistant", "content": "beta " * 80},
            {"role": "user", "content": "gamma " * 80},
            {"role": "assistant", "content": "delta " * 80},
            {"role": "user", "content": "epsilon " * 80},
            {"role": "assistant", "content": "zeta"},
        ]
        engine.compress(messages)
        old_frontier = engine._last_compacted_store_id

        engine.rollover_session("frontier-old", "frontier-new", previous_messages=[], platform="cli", context_length=200000)

        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state is not None
        assert state.current_session_id == "frontier-new"
        assert state.current_frontier_store_id == 0
        assert state.last_finalized_session_id == "frontier-old"
        assert state.last_finalized_frontier_store_id == old_frontier
        assert state.last_rollover_at is not None
        assert state.last_reset_at is not None
