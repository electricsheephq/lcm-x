"""LCM engine integration tests: auxiliary context."""

import gc
import sqlite3
import sys
import threading
import time
from types import ModuleType



from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

from .engine_helpers import (
    TestSessionRollover as _SessionRolloverHelpers,
)


class TestSessionRollover:
    HostAgentFrame = _SessionRolloverHelpers.__dict__["HostAgentFrame"]
    _start_host_child = _SessionRolloverHelpers.__dict__["_start_host_child"]

    def test_auxiliary_child_worker_thread_stays_stateless_without_parent_thread_leak(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_worker_thread.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = self._start_host_child(
            engine,
            hermes_home,
            "delegate-child-session",
            "foreground-session",
        )

        assert engine._thread_context_session_id() == ""
        engine.should_compress_preflight([
            {"role": "user", "content": "foreground parent thread must still persist"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("delegate-child-session") == 0

        errors = []

        def run_child_worker():
            try:
                assert child.should_compress_preflight(engine, [
                    {"role": "user", "content": "delegate child worker must stay stateless"},
                ]) is False
                assert child.compress(engine, [
                    {"role": "user", "content": "delegate child compression must bypass"},
                ]) == [
                    {"role": "user", "content": "delegate child compression must bypass"},
                ]
                child.update_from_response(engine, {
                    "prompt_tokens": 999,
                    "completion_tokens": 1,
                    "total_tokens": 1000,
                })
                child.on_session_end(engine, [
                    {"role": "assistant", "content": "delegate child end must not persist"},
                ])
            except Exception as exc:  # pragma: no cover - assertion helper
                errors.append(exc)

        worker = threading.Thread(target=run_child_worker)
        worker.start()
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert errors == []
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("delegate-child-session") == 0
        assert engine.last_prompt_tokens == 0
        assert not engine._thread_context_stateless()

    def test_auxiliary_child_end_on_clean_parent_thread_does_not_poison_foreground(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_clean_parent_end.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")

        engine.on_session_end(
            "background-review-session",
            [{"role": "assistant", "content": "background child end from parent thread"}],
        )

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        engine.should_compress_preflight([
            {"role": "user", "content": "foreground must persist after child end"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("background-review-session") == 0

    def test_matching_parent_frame_foreground_branch_rebinds_normally(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-branch-session', 'foreground-session', 2.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        class ForegroundBranchFrame:
            def __init__(self):
                self.session_id = "foreground-branch-session"
                self._parent_session_id = "foreground-session"
                self.enabled_toolsets = ["terminal", "file"]
                self.log_prefix = ""
                self._delegate_depth = 0

            def on_session_start(self, lcm_engine: LCMEngine) -> None:
                lcm_engine.on_session_start(
                    self.session_id,
                    hermes_home=str(hermes_home),
                    platform="tui",
                    context_length=200000,
                )

            def should_compress_preflight(self, lcm_engine: LCMEngine, messages):
                return lcm_engine.should_compress_preflight(messages)

            def update_model(self, lcm_engine: LCMEngine, context_length: int) -> None:
                lcm_engine.update_model("foreground-branch-model", context_length)

        config = LCMConfig(database_path=str(tmp_path / "lcm_matching_foreground_branch.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        branch = ForegroundBranchFrame()
        branch.on_session_start(engine)

        assert engine._session_id == "foreground-branch-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("foreground-branch-session")
        branch.update_model(engine, 300000)
        assert engine.context_length == 300000
        assert engine.threshold_tokens == int(300000 * config.context_threshold)
        branch.should_compress_preflight(engine, [
            {"role": "user", "content": "foreground branch must persist"},
        ])
        assert engine._store.get_session_count("foreground-branch-session") == 1
        assert engine._store.get_session_count("foreground-session") == 0

    def test_auxiliary_child_update_from_response_does_not_mutate_foreground_metrics(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_usage.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        foreground_usage = {
            "prompt_tokens": 12345,
            "completion_tokens": 10,
            "total_tokens": 12355,
            "input_tokens": 12000,
            "output_tokens": 355,
            "cache_read_tokens": 6000,
            "cache_write_tokens": 500,
            "reasoning_tokens": 42,
        }
        engine.update_from_response(foreground_usage)
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.update_from_response(engine, {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "input_tokens": 3,
            "output_tokens": 4,
            "cache_read_tokens": 5,
            "cache_write_tokens": 6,
            "reasoning_tokens": 7,
        })

        assert engine.last_prompt_tokens == 12345
        assert engine.last_completion_tokens == 10
        assert engine.last_total_tokens == 12355
        assert engine.last_input_tokens == 12000
        assert engine.last_output_tokens == 355
        assert engine.last_cache_read_tokens == 6000
        assert engine.last_cache_write_tokens == 500
        assert engine.last_reasoning_tokens == 42
        assert engine.cache_metrics_available is True

        engine.update_from_response({
            "prompt_tokens": 222,
            "completion_tokens": 3,
            "total_tokens": 225,
            "cache_read_tokens": 11,
            "cache_write_tokens": 12,
        })
        assert engine.last_prompt_tokens == 222
        assert engine.last_total_tokens == 225
        assert engine.last_cache_read_tokens == 11
        assert engine.last_cache_write_tokens == 12

    def test_stale_auxiliary_thread_marker_clears_on_next_normal_session_start(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-session', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('next-normal-session', NULL, 3.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_aux_marker.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )

        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._session_id == "foreground-session"

        engine.on_session_start(
            "next-normal-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )

        assert engine._thread_context_session_id() == ""
        assert engine._session_id == "next-normal-session"
        messages = [
            {"role": "user", "content": "normal foreground ingestion returns"},
        ]
        engine.should_compress_preflight(messages)

        assert engine._store.get_session_count("next-normal-session") == 1
        assert engine._store.get_session_count("background-review-session") == 0

    def test_stale_auxiliary_thread_marker_clears_before_compression_boundary(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-session', 'foreground-session', 2.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_aux_boundary.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        foreground_store_id = engine._store.append(
            "foreground-session",
            {"role": "user", "content": "foreground context crosses boundary"},
            token_estimate=17,
            source="telegram",
        )
        foreground_node_id = engine._dag.add_node(SummaryNode(
            session_id="foreground-session",
            depth=0,
            summary="foreground boundary summary",
            token_count=5,
            source_token_count=17,
            source_ids=[foreground_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = foreground_store_id
        foreground_conversation_id = engine._conversation_id

        self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )

        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._session_id == "foreground-session"

        engine.on_session_start(
            "foreground-continuation",
            boundary_reason="compression",
            old_session_id="foreground-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._session_id == "foreground-continuation"
        assert engine._conversation_id == foreground_conversation_id
        assert [
            node.node_id for node in engine._dag.get_session_nodes("foreground-continuation")
        ] == [foreground_node_id]

        messages = [
            {"role": "user", "content": "post-boundary foreground ingestion returns"},
        ]
        engine.should_compress_preflight(messages)

        background_messages = [
            {"role": "user", "content": "late background review must still not persist"},
        ]
        engine.on_session_end("background-review-session", background_messages)

        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("foreground-continuation") == 1
        assert engine._store.get_session_count("background-review-session") == 0
        assert not engine._thread_context_has_auxiliary_session("background-review-session")

    def test_auxiliary_compression_boundary_does_not_reassign_foreground_state(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-session', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-continuation', 'background-review-session', 3.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_boundary.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        foreground_store_id = engine._store.append(
            "foreground-session",
            {"role": "user", "content": "foreground must not move into child continuation"},
            token_estimate=13,
            source="telegram",
        )
        foreground_node_id = engine._dag.add_node(SummaryNode(
            session_id="foreground-session",
            depth=0,
            summary="foreground must stay foreground",
            token_count=5,
            source_token_count=13,
            source_ids=[foreground_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = foreground_store_id
        foreground_conversation_id = engine._conversation_id

        self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        assert engine._session_id == "foreground-session"
        assert engine._thread_context_has_auxiliary_session("background-review-session")

        engine.on_session_start(
            "background-review-continuation",
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "foreground-session"
        assert engine._conversation_id == foreground_conversation_id
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("background-review-continuation") == 0
        assert [
            node.node_id for node in engine._dag.get_session_nodes("foreground-session")
        ] == [foreground_node_id]
        assert engine._dag.get_session_nodes("background-review-continuation") == []

        engine.on_session_end(
            "background-review-session",
            [{"role": "user", "content": "old child end must not persist"}],
        )
        engine.on_session_end(
            "background-review-continuation",
            [{"role": "user", "content": "new child end must not persist"}],
        )

        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")

    def test_stale_generated_auxiliary_boundary_cannot_reassign_reused_foreground(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_stale_aux_reused_fg_boundary.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            conversation_id="foreground-conversation",
            context_length=200000,
        )
        stale_auxiliary = self._start_host_child(
            engine,
            hermes_home,
            "reused-session",
            "foreground-session",
        )
        stale_auxiliary.on_session_end(engine, [])

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="normal-conversation",
            context_length=200000,
        )
        foreground_store_id = engine._store.append(
            "reused-session",
            {"role": "user", "content": "normal foreground state must stay put"},
            token_estimate=11,
            source="cli",
            conversation_id="normal-conversation",
        )
        foreground_node_id = engine._dag.add_node(SummaryNode(
            session_id="reused-session",
            depth=0,
            summary="normal reused foreground summary",
            token_count=5,
            source_token_count=11,
            source_ids=[foreground_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = foreground_store_id

        stale_auxiliary.on_compression_boundary(engine, "stale-continuation")

        assert engine._session_id == "reused-session"
        assert engine._conversation_id == "normal-conversation"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("stale-continuation")
        assert [
            node.node_id for node in engine._dag.get_session_nodes("reused-session")
        ] == [foreground_node_id]
        assert engine._dag.get_session_nodes("stale-continuation") == []
        assert engine._store.get_session_count("reused-session") == 1
        assert engine._store.get_session_count("stale-continuation") == 0
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-session")

    def test_direct_end_clears_active_generated_auxiliary_with_usage(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_generated_aux_direct_end.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.update_from_response(engine, {"prompt_tokens": 45})
        assert child.should_compress(engine)

        engine.on_session_end("background-review-session", [])

        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}
        assert engine._store.get_session_count("background-review-session") == 0

    def test_direct_end_clears_restarted_generated_auxiliary_after_clean_end(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_generated_aux_direct_end.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        first = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        first.update_from_response(engine, {"prompt_tokens": 45})
        assert first.should_compress(engine)
        first.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")

        replacement = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        replacement.update_from_response(engine, {"prompt_tokens": 25})
        assert replacement.should_compress(engine)

        engine.on_session_end("background-review-session", [])

        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert not replacement.should_compress(engine)
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}
        assert engine._store.get_session_count("background-review-session") == 0

    def test_direct_end_allows_same_generated_auxiliary_object_to_restart(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_same_generated_aux_restart.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.update_from_response(engine, {"prompt_tokens": 25})
        assert child.should_compress(engine)
        child.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")

        child.on_session_start(engine)
        child.update_from_response(engine, {"prompt_tokens": 25})
        assert child.should_compress(engine)
        child.on_session_end(engine, [])

        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}
        assert engine._store.get_session_count("background-review-session") == 0

    def test_late_retired_generated_auxiliary_restart_does_not_replace_active_generation(
        self,
        tmp_path,
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_late_retired_aux_restart.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        stale_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        stale_generation = engine._auxiliary_session_generations["background-review-session"]
        stale_child.update_from_response(engine, {"prompt_tokens": 25})
        stale_child.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")

        replacement = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        replacement_generation = engine._auxiliary_session_generations["background-review-session"]
        assert replacement_generation != stale_generation
        replacement.update_from_response(engine, {"prompt_tokens": 25})
        assert replacement.should_compress(engine)

        stale_child.on_session_start(engine)
        stale_child.update_from_response(engine, {"prompt_tokens": 35})

        assert engine._auxiliary_session_generations == {
            "background-review-session": replacement_generation
        }
        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 25}
        assert replacement.should_compress(engine)
        replacement.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._store.get_session_count("background-review-session") == 0

    def test_retired_auxiliary_generation_cannot_adopt_generationless_reuse(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_retired_aux_generationless.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        stale_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        stale_generation = engine._auxiliary_session_generations["background-review-session"]

        engine.on_session_start(
            "background-review-session",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        assert stale_generation in engine._auxiliary_retired_session_generations[
            "background-review-session"
        ]
        assert not engine._thread_context_has_auxiliary_session("background-review-session")

        engine._mark_thread_context_stateless("background-review-session")
        stale_child.on_session_start(engine)
        stale_child.update_from_response(engine, {"prompt_tokens": 35})

        assert engine._auxiliary_session_generations == {}
        assert engine._auxiliary_last_prompt_tokens == {}
        assert not stale_child.should_compress(engine)
        assert engine._store.get_session_count("background-review-session") == 0

    def test_late_reused_auxiliary_end_without_normal_prefix_does_not_persist(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_late_reused_aux_no_prefix.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        self._start_host_child(
            engine,
            hermes_home,
            "reused-session",
            "foreground-session",
        )

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="normal-conversation",
            context_length=1_000,
        )
        assert engine._session_id == "reused-session"
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-session")
        assert engine._store.get_session_count("reused-session") == 0

        engine.on_session_start(
            "next-foreground-session",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="next-conversation",
            context_length=1_000,
        )
        engine.on_session_end(
            "reused-session",
            [{"role": "user", "content": "late old auxiliary must not persist"}],
        )

        assert engine._store.get_session_count("reused-session") == 0
        assert engine._store.get_session_count("next-foreground-session") == 0

    def test_auxiliary_compression_boundary_retires_old_child_if_old_end_is_missing(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-session', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-continuation', 'background-review-session', 3.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_handoff_missing_old_end.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.should_compress_preflight(engine, [
            {"role": "user", "content": "old child work must stay stateless"},
        ])
        assert engine._thread_context_has_auxiliary_session("background-review-session")

        engine.on_session_start(
            "background-review-continuation",
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")
        continuation = self.HostAgentFrame(
            "background-review-continuation",
            "background-review-session",
            hermes_home,
        )
        continuation.should_compress_preflight(engine, [
            {"role": "user", "content": "new child work must stay stateless"},
        ])
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

        continuation.on_session_end(engine, [
            {"role": "user", "content": "new child end must not persist"},
        ])

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        engine.should_compress_preflight([
            {"role": "user", "content": "foreground persists after child compression handoff"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_compression_boundary_after_child_end_stays_auxiliary(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-session', 'foreground-session', 2.0, 3.0);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-continuation', 'background-review-session', 4.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_late_boundary.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        foreground_store_id = engine._store.append(
            "foreground-session",
            {"role": "user", "content": "foreground must survive late auxiliary boundary"},
            token_estimate=13,
            source="telegram",
        )
        foreground_node_id = engine._dag.add_node(SummaryNode(
            session_id="foreground-session",
            depth=0,
            summary="foreground must not move after child end",
            token_count=5,
            source_token_count=13,
            source_ids=[foreground_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = foreground_store_id
        foreground_conversation_id = engine._conversation_id

        # The child was seen as auxiliary while live, then ended before its
        # compression continuation starts. A clean foreground thread must stay
        # clean; the continuation is stateless only when executed through the
        # auxiliary child frame/lineage.
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.on_session_end(engine, [
            {"role": "user", "content": "ended child must not persist"},
        ])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        engine.on_session_start(
            "background-review-continuation",
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "foreground-session"
        assert engine._conversation_id == foreground_conversation_id
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")
        continuation = self.HostAgentFrame(
            "background-review-continuation",
            "background-review-session",
            hermes_home,
        )
        continuation.should_compress_preflight(engine, [
            {"role": "user", "content": "continuation child must not persist"},
        ])
        assert engine._thread_context_session_id() == ""
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0
        assert [
            node.node_id for node in engine._dag.get_session_nodes("foreground-session")
        ] == [foreground_node_id]
        assert engine._dag.get_session_nodes("background-review-continuation") == []

    def test_multiple_auxiliary_child_sessions_are_tracked_independently(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-a', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-b', 'foreground-session', 3.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_multiple_aux.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        self._start_host_child(
            engine,
            hermes_home,
            "background-review-a",
            "foreground-session",
        )
        b = self._start_host_child(
            engine,
            hermes_home,
            "background-review-b",
            "foreground-session",
        )

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-a")
        assert engine._thread_context_has_auxiliary_session("background-review-b")
        assert b.thread_context_session_id(engine) == "background-review-b"
        b.should_compress_preflight(engine, [
            {"role": "user", "content": "child b worker must be stateless"},
        ])
        assert engine._thread_context_session_id() == ""

        engine.on_session_end(
            "background-review-a",
            [{"role": "user", "content": "first background review must not persist"}],
        )

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("background-review-a")
        assert engine._thread_context_has_auxiliary_session("background-review-b")
        assert engine._store.get_session_count("background-review-a") == 0

        b.on_session_end(engine, [
            {"role": "user", "content": "second background review must not persist"},
        ])

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert not engine._thread_context_has_auxiliary_session("background-review-b")
        assert engine._store.get_session_count("background-review-a") == 0
        assert engine._store.get_session_count("background-review-b") == 0
        engine.should_compress_preflight([
            {"role": "user", "content": "foreground persists after auxiliary children end"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1

    def test_nested_auxiliary_child_end_restores_previous_thread_marker(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-a', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-b', 'foreground-session', 3.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_nested_aux.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        a = self._start_host_child(
            engine,
            hermes_home,
            "background-review-a",
            "foreground-session",
        )
        b = self._start_host_child(
            engine,
            hermes_home,
            "background-review-b",
            "foreground-session",
        )
        assert a.thread_context_session_id(engine) == "background-review-a"
        a.should_compress_preflight(engine, [
            {"role": "user", "content": "child a worker must be stateless"},
        ])
        assert b.thread_context_session_id(engine) == "background-review-b"
        b.should_compress_preflight(engine, [
            {"role": "user", "content": "child b worker must be stateless"},
        ])

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        b.on_session_end(engine, [
            {"role": "user", "content": "nested child b must not persist"},
        ])

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert engine._thread_context_has_auxiliary_session("background-review-a")
        assert not engine._thread_context_has_auxiliary_session("background-review-b")

        a.should_compress_preflight(engine, [
            {"role": "user", "content": "continuing child a must still be stateless"},
        ])

        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-a") == 0
        assert engine._store.get_session_count("background-review-b") == 0

        a.on_session_end(engine, [
            {"role": "user", "content": "nested child a must not persist"},
        ])

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert not engine._thread_context_has_auxiliary_session("background-review-a")
        engine.should_compress_preflight([
            {"role": "user", "content": "foreground persists after nested children end"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("background-review-a") == 0
        assert engine._store.get_session_count("background-review-b") == 0

    def test_auxiliary_descendant_session_does_not_rebind_shared_engine(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-a', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-followup', 'background-review-a', 3.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_descendant.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        parent = self._start_host_child(
            engine,
            hermes_home,
            "background-review-a",
            "foreground-session",
        )
        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-a")
        assert parent.thread_context_session_id(engine) == "background-review-a"
        parent.should_compress_preflight(engine, [
            {"role": "user", "content": "parent auxiliary must be stateless"},
        ])
        assert engine._thread_context_session_id() == ""

        engine.on_session_start(
            "background-review-followup",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-a")
        assert engine._thread_context_has_auxiliary_session("background-review-followup")
        followup = self.HostAgentFrame(
            "background-review-followup",
            "background-review-a",
            hermes_home,
        )
        followup.should_compress_preflight(engine, [
            {"role": "user", "content": "descendant auxiliary must stay stateless"},
        ])
        assert followup.thread_context_session_id(engine) == "background-review-followup"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-a") == 0
        assert engine._store.get_session_count("background-review-followup") == 0

        followup.on_session_end(engine, [
            {"role": "user", "content": "followup end must not persist"},
        ])

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert engine._thread_context_has_auxiliary_session("background-review-a")
        assert not engine._thread_context_has_auxiliary_session("background-review-followup")

        parent.on_session_end(engine, [
            {"role": "user", "content": "parent auxiliary end must not persist"},
        ])

        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert not engine._thread_context_has_auxiliary_session("background-review-a")
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-a") == 0
        assert engine._store.get_session_count("background-review-followup") == 0

    def test_auxiliary_descendant_after_parent_end_stays_auxiliary(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-a', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-followup', 'background-review-a', 4.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_descendant_late.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        self._start_host_child(
            engine,
            hermes_home,
            "background-review-a",
            "foreground-session",
        )
        engine.on_session_end(
            "background-review-a",
            [{"role": "user", "content": "parent auxiliary ended first"}],
        )
        conn = sqlite3.connect(state_db)
        conn.execute(
            "UPDATE sessions SET ended_at = 3.0 WHERE id = 'background-review-a'"
        )
        conn.commit()
        conn.close()

        assert not engine._thread_context_has_auxiliary_session("background-review-a")
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        engine.on_session_start(
            "background-review-followup",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-followup")
        followup = self.HostAgentFrame(
            "background-review-followup",
            "background-review-a",
            hermes_home,
        )
        followup.should_compress_preflight(engine, [
            {"role": "user", "content": "late descendant auxiliary must stay stateless"},
        ])
        assert followup.thread_context_session_id(engine) == "background-review-followup"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-a") == 0
        assert engine._store.get_session_count("background-review-followup") == 0

    def test_auxiliary_child_end_is_ignored_across_threads(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        state_db = hermes_home / "state.db"
        conn = sqlite3.connect(state_db)
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('foreground-session', NULL, 1.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('background-review-session', 'foreground-session', 2.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_cross_thread_aux.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.should_compress_preflight(engine, [
            {"role": "user", "content": "child worker must be stateless before cross-thread end"},
        ])

        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        errors = []

        def end_auxiliary_child():
            try:
                engine.on_session_end(
                    "background-review-session",
                    [{"role": "user", "content": "cross-thread child end must not persist"}],
                )
            except Exception as exc:  # pragma: no cover - assertion helper
                errors.append(exc)

        thread = threading.Thread(target=end_auxiliary_child)
        thread.start()
        thread.join(timeout=5)

        assert not thread.is_alive()
        assert errors == []
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._session_id == "foreground-session"
        engine.should_compress_preflight([
            {"role": "user", "content": "foreground persists after cross-thread auxiliary end"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1

    def test_same_thread_late_auxiliary_callback_after_end_stays_stateless(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_same_thread_late_aux.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.should_compress_preflight(engine, [
            {"role": "user", "content": "child worker must become stateless"},
        ])
        assert child.thread_context_session_id(engine) == "background-review-session"
        assert child.thread_context_stateless(engine)
        assert engine._thread_context_session_id() == ""

        child.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_stateless()

        child.should_compress_preflight(engine, [
            {"role": "user", "content": "same-thread late child callback must not persist"},
        ])
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

        engine.should_compress_preflight([
            {"role": "user", "content": "bare foreground callback persists after child end"},
        ])
        assert engine._store.get_session_count("foreground-session") == 1

        engine.on_session_start(
            "foreground-followup-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        assert engine._thread_context_session_id() == ""
        engine.should_compress_preflight([
            {"role": "user", "content": "foreground followup persists after marker clear"},
        ])
        assert engine._store.get_session_count("foreground-followup-session") == 1

    def test_same_thread_late_auxiliary_usage_after_end_is_not_reused(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_late_aux_usage_after_end.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.update_from_response(engine, {"prompt_tokens": 25})
        assert child.should_compress(engine)

        child.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._auxiliary_last_prompt_tokens == {}

        child.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}
        assert "background-review-session" not in engine._auxiliary_direct_end_guard_session_ids

        replacement = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        assert not replacement.should_compress(engine)
        engine.on_session_end("background-review-session", [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_late_auxiliary_usage_from_old_generation_does_not_pollute_reused_id(
        self,
        tmp_path,
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_usage_generation_reuse.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.on_session_end(engine, [])
        replacement = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        assert not replacement.should_compress(engine)

        old_child.update_from_response(engine, {"prompt_tokens": 25})

        assert engine._auxiliary_last_prompt_tokens == {}
        assert not replacement.should_compress(engine)
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_auxiliary_reused_generation_clears_usage_and_ignores_old_end(
        self,
        tmp_path,
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generation_replace_before_end.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 25})
        assert old_child.should_compress(engine)

        replacement = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        assert engine._auxiliary_last_prompt_tokens == {}
        assert not replacement.should_compress(engine)

        old_child.on_session_end(engine, [])
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert not replacement.should_compress(engine)

        replacement.update_from_response(engine, {"prompt_tokens": 25})
        assert replacement.should_compress(engine)
        engine.on_session_end("background-review-session", [])
        assert not engine._thread_context_has_auxiliary_session("background-review-session")
        assert not replacement.should_compress(engine)
        old_messages = [{"role": "user", "content": "old frame short payload"}]
        assert count_messages_tokens(old_messages) < engine.threshold_tokens
        assert not old_child.should_compress(engine)
        assert old_child.compress(engine, old_messages) == old_messages
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_auxiliary_compress_zero_current_tokens_uses_cached_usage(self, tmp_path, monkeypatch):
        native_calls = []

        class FakeContextCompressor:
            def __init__(self, **kwargs):
                self.compression_count = 0
                self._last_compress_aborted = False
                self._last_summary_error = None

            def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
                native_calls.append(current_tokens)
                self.compression_count += 1
                return [{"role": "user", "content": "native compacted auxiliary"}]

        agent_module = sys.modules.get("agent") or ModuleType("agent")
        if not hasattr(agent_module, "__path__"):
            agent_module.__path__ = []
        compressor_module = ModuleType("agent.context_compressor")
        setattr(compressor_module, "ContextCompressor", FakeContextCompressor)
        monkeypatch.setitem(sys.modules, "agent", agent_module)
        monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)

        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_zero_current_tokens.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.model = "test-model"
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.update_from_response(engine, {"prompt_tokens": 30})
        messages = [{"role": "user", "content": "tiny"}]

        result = child.compress(engine, messages, current_tokens=0)

        assert native_calls == [30]
        assert result == [{"role": "user", "content": "native compacted auxiliary"}]
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_auxiliary_no_arg_should_compress_without_usage_ignores_foreground_tokens(
        self,
        tmp_path,
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_no_foreground_fallback.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        engine.update_from_response({"prompt_tokens": 25})
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )

        assert engine.last_prompt_tokens == 25
        assert engine._auxiliary_last_prompt_tokens == {}
        assert not child.should_compress(engine)
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_stack_marked_auxiliary_no_arg_calls_use_generation_usage_without_caller_frame(
        self,
        tmp_path,
        monkeypatch,
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_stack_marked_generation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        child.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_session_generations["background-review-session"]
        stack = engine._thread_context_auxiliary_stack()
        stack[:] = ["background-review-session"]
        engine._thread_context.current_auxiliary_session_id = "background-review-session"
        assert engine._thread_context_session_id() == "background-review-session"

        calls = []

        def fake_bypassed_compress(messages, *, current_tokens=None, focus_topic=None, force=False):
            calls.append((current_tokens, focus_topic, force))
            return list(messages)

        monkeypatch.setattr(engine, "_compress_lcm_bypassed_session", fake_bypassed_compress)
        messages = [{"role": "user", "content": "short bypass payload"}]

        assert count_messages_tokens(messages) < engine.threshold_tokens
        assert engine.should_compress()
        assert engine.compress(messages) == messages
        assert calls == [(25, None, False)]
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_stack_marked_auxiliary_update_uses_generation_usage_without_caller_frame(
        self,
        tmp_path,
    ):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_stack_marked_update.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        assert engine._auxiliary_session_generations["background-review-session"]
        stack = engine._thread_context_auxiliary_stack()
        stack[:] = ["background-review-session"]
        engine._thread_context.current_auxiliary_session_id = "background-review-session"
        assert engine._thread_context_session_id() == "background-review-session"

        engine.update_from_response({"prompt_tokens": 25})

        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 25}
        assert child.should_compress(engine)
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_auxiliary_handoff_accepts_new_continuation_generation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_handoff_generation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 15})

        engine.on_session_start(
            "background-review-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=1_000,
        )
        continuation = self.HostAgentFrame(
            "background-review-continuation",
            "background-review-session",
            hermes_home,
        )
        assert engine._auxiliary_last_prompt_tokens == {}
        assert not continuation.should_compress(engine)

        continuation.update_from_response(engine, {"prompt_tokens": 25})

        assert engine._auxiliary_last_prompt_tokens == {"background-review-continuation": 25}
        assert continuation.should_compress(engine)
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_handoff_preserves_same_active_continuation_generation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_same_generation_handoff.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 15})
        continuation = self._start_host_child(
            engine,
            hermes_home,
            "background-review-continuation",
            "background-review-session",
        )
        continuation.update_from_response(engine, {"prompt_tokens": 10})

        continuation.on_compression_boundary_from(
            engine,
            old_session_id="background-review-session",
        )
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")

        continuation.update_from_response(engine, {"prompt_tokens": 25})

        assert engine._auxiliary_last_prompt_tokens == {"background-review-continuation": 25}
        assert continuation.should_compress(engine)
        engine.on_session_end("background-review-continuation", [])
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_handoff_generationless_boundary_preserves_active_continuation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generationless_active_handoff.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 15})
        continuation = self._start_host_child(
            engine,
            hermes_home,
            "background-review-continuation",
            "background-review-session",
        )
        continuation.update_from_response(engine, {"prompt_tokens": 10})
        active_generation = engine._auxiliary_session_generations["background-review-continuation"]

        engine.on_session_start(
            "background-review-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=1_000,
        )

        assert engine._auxiliary_session_generations == {
            "background-review-continuation": active_generation
        }
        assert active_generation not in engine._auxiliary_retired_session_generations.get(
            "background-review-continuation",
            set(),
        )
        assert "background-review-continuation" not in engine._auxiliary_direct_end_guard_session_ids
        continuation.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_last_prompt_tokens == {"background-review-continuation": 25}
        assert continuation.should_compress(engine)
        engine.on_session_end("background-review-continuation", [])
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert engine._auxiliary_session_generations == {}
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_handoff_retired_displaced_generation_rejects_stale_usage(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_retired_displaced_generation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        parent = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        parent.update_from_response(engine, {"prompt_tokens": 15})
        stale_continuation = self._start_host_child(
            engine,
            hermes_home,
            "background-review-continuation",
            "background-review-session",
        )
        stale_continuation.update_from_response(engine, {"prompt_tokens": 10})
        replacement_continuation = self._start_host_child(
            engine,
            hermes_home,
            "background-review-continuation",
            "background-review-session",
        )
        replacement_continuation.update_from_response(engine, {"prompt_tokens": 11})

        engine.on_session_start(
            "background-review-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=1_000,
        )
        stale_continuation.update_from_response(engine, {"prompt_tokens": 99})

        assert engine._auxiliary_last_prompt_tokens == {}
        assert not stale_continuation.should_compress(engine)
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_retired_generation_boundary_does_not_reactivate_stale_continuation(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_retired_generation_boundary_reject.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        stale_continuation = self._start_host_child(
            engine,
            hermes_home,
            "background-review-continuation",
            "background-review-session",
        )
        stale_generation = engine._auxiliary_session_generations["background-review-continuation"]
        stale_continuation.on_session_end(engine, [])
        assert engine._auxiliary_generation_is_retired(
            "background-review-continuation",
            stale_generation,
        )

        stale_continuation.on_compression_boundary_from(
            engine,
            old_session_id="background-review-session",
            context_length=1_000,
        )
        stale_continuation.update_from_response(engine, {"prompt_tokens": 99})

        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert engine._auxiliary_last_prompt_tokens == {}
        assert not stale_continuation.should_compress(engine)
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_handoff_retires_old_generation_before_generationless_reuse(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_retired_old_generation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 10})

        engine.on_session_start(
            "background-review-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=1_000,
        )
        engine._mark_thread_context_stateless("background-review-session")
        old_child.update_from_response(engine, {"prompt_tokens": 25})

        assert engine._auxiliary_last_prompt_tokens == {}
        assert not old_child.should_compress(engine)
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_generation_token_is_not_reusable_object_id(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generation_token.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        token = engine._auxiliary_session_generations["background-review-session"]
        assert token != id(child)

        engine._auxiliary_retired_session_generations["background-review-session"] = {id(child)}
        child.update_from_response(engine, {"prompt_tokens": 25})

        assert engine._auxiliary_last_prompt_tokens == {"background-review-session": 25}
        assert child.should_compress(engine)
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_generationless_reused_handoff_cleans_up_after_usage(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generationless_handoff_cleanup.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session-1",
            "foreground-session",
        )
        old_child.on_compression_boundary(
            engine,
            "background-review-continuation",
            context_length=1_000,
        )
        engine.on_session_end("background-review-continuation", [])
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")

        self._start_host_child(
            engine,
            hermes_home,
            "background-review-session-2",
            "foreground-session",
        )
        engine._handoff_auxiliary_session(
            "background-review-session-2",
            "background-review-continuation",
        )
        assert "background-review-continuation" not in engine._auxiliary_direct_end_guard_session_ids
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")

        engine.on_session_end("background-review-continuation", [])

        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert engine._auxiliary_last_prompt_tokens == {}
        assert "background-review-continuation" not in engine._auxiliary_direct_end_guard_session_ids
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_clean_generated_continuation_reuse_does_not_guard_direct_end(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_clean_generated_reuse.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.on_compression_boundary(
            engine,
            "background-review-continuation",
            context_length=1_000,
        )
        engine.on_session_end("background-review-continuation", [])
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")

        next_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        continuation = self.HostAgentFrame(
            "background-review-continuation",
            "background-review-session",
            hermes_home,
        )
        continuation.on_compression_boundary_from(
            engine,
            "background-review-session",
            context_length=1_000,
        )
        assert next_child.session_id == "background-review-session"
        assert "background-review-continuation" not in engine._auxiliary_direct_end_guard_session_ids
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")

        engine.on_session_end("background-review-continuation", [])

        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert engine._auxiliary_session_generations == {}
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._store.get_session_count("background-review-continuation") == 0

    def test_auxiliary_generation_token_uses_identity_not_equality(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generation_identity.db"))
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))

        class EqualFrame:
            def __eq__(self, other):
                return isinstance(other, EqualFrame)

            def __hash__(self):
                return 1

        first = EqualFrame()
        second = EqualFrame()
        assert first == second
        assert first is not second

        first_token = engine._auxiliary_generation_token_for(first)
        second_token = engine._auxiliary_generation_token_for(second)

        assert first_token != second_token
        assert engine._auxiliary_generation_token_for(first) == first_token
        assert engine._auxiliary_generation_token_for(second) == second_token

    def test_auxiliary_generation_token_handles_nonweakref_slotted_callers(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generation_slotted.db"))
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))

        class SlottedFrame:
            __slots__ = ()

        frame = SlottedFrame()
        token = engine._auxiliary_generation_token_for(frame)

        assert token
        assert engine._auxiliary_generation_token_for(frame) == token

    def test_auxiliary_generation_token_prunes_dead_weakref_callers(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generation_weakref_prune.db"))
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))

        class WeakrefableFrame:
            pass

        frame = WeakrefableFrame()
        object_id = id(frame)
        token = engine._auxiliary_generation_token_for(frame)
        assert engine._auxiliary_generation_tokens[object_id][1] == token

        del frame
        gc.collect()

        assert object_id not in engine._auxiliary_generation_tokens

    def test_auxiliary_generation_token_handles_callable_nonweakref_slotted_callers(self, tmp_path):
        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_generation_callable_slotted.db"))
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))

        class CallableSlottedFrame:
            __slots__ = ()

            def __call__(self):
                return "not-self"

        frame = CallableSlottedFrame()
        token = engine._auxiliary_generation_token_for(frame)

        assert token
        assert engine._auxiliary_generation_token_for(frame) == token

    def test_auxiliary_handoff_reused_continuation_clears_stale_state(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_handoff_reused_continuation.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine.threshold_tokens = 20

        old_child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        old_child.update_from_response(engine, {"prompt_tokens": 15})

        stale_continuation = self._start_host_child(
            engine,
            hermes_home,
            "background-review-continuation",
            "background-review-session",
        )
        stale_continuation.update_from_response(engine, {"prompt_tokens": 45})
        engine._auxiliary_direct_end_guard_session_ids.add("background-review-continuation")
        assert engine._auxiliary_last_prompt_tokens["background-review-continuation"] == 45
        engine.on_session_start(
            "background-review-continuation",
            hermes_home=str(hermes_home),
            boundary_reason="compression",
            old_session_id="background-review-session",
            platform="telegram",
            context_length=1_000,
        )

        continuation = self.HostAgentFrame(
            "background-review-continuation",
            "background-review-session",
            hermes_home,
        )
        assert engine._auxiliary_last_prompt_tokens == {}
        assert "background-review-continuation" in engine._auxiliary_direct_end_guard_session_ids
        assert engine._auxiliary_handoff_parent_session_ids == {
            "background-review-continuation": "background-review-session"
        }
        stale_continuation.on_session_end(engine, [])
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")
        stale_continuation.update_from_response(engine, {"prompt_tokens": 30})
        assert engine._auxiliary_last_prompt_tokens == {}
        assert engine._auxiliary_session_generations == {}
        engine.on_session_end("background-review-continuation", [])
        assert engine._thread_context_has_auxiliary_session("background-review-continuation")
        continuation.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._auxiliary_last_prompt_tokens == {"background-review-continuation": 25}
        assert continuation.should_compress(engine)
        continuation.on_session_end(engine, [])
        assert not engine._thread_context_has_auxiliary_session("background-review-continuation")
        assert "background-review-continuation" not in engine._auxiliary_direct_end_guard_session_ids
        assert "background-review-continuation" not in engine._auxiliary_handoff_parent_session_ids
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("background-review-continuation") == 0
