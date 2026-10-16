"""LCM engine integration tests: auxiliary identity."""

import sqlite3
import threading
import time

import pytest


from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

from .engine_helpers import (
    TestSessionRollover as _SessionRolloverHelpers,
)


class TestSessionRollover:
    HostAgentFrame = _SessionRolloverHelpers.__dict__["HostAgentFrame"]
    _start_host_child = _SessionRolloverHelpers.__dict__["_start_host_child"]

    def test_live_auxiliary_child_session_does_not_rebind_shared_engine(self, tmp_path):
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

        config = LCMConfig(database_path=str(tmp_path / "lcm_auxiliary_child.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        foreground_store_id = engine._store.append(
            "foreground-session",
            {"role": "user", "content": "foreground context must stay bound"},
            token_estimate=17,
            source="telegram",
        )
        foreground_node_id = engine._dag.add_node(SummaryNode(
            session_id="foreground-session",
            depth=0,
            summary="foreground summary",
            token_count=5,
            source_token_count=17,
            source_ids=[foreground_store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = foreground_store_id
        foreground_conversation_id = engine._conversation_id

        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )
        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        assert engine._conversation_id == foreground_conversation_id
        assert engine._lifecycle.get_by_conversation(
            foreground_conversation_id
        ).current_session_id == "foreground-session"

        background_messages = [
            {"role": "user", "content": "background review must not enter LCM"},
            {"role": "assistant", "content": "Nothing to save."},
        ]
        assert child.should_compress_preflight(engine, background_messages) is False
        assert child.compress(engine, background_messages) == background_messages
        child.on_session_end(engine, background_messages)

        assert engine._store.get_session_count("background-review-session") == 0
        assert engine._store.get_session_count("foreground-session") == 1
        assert [
            node.node_id for node in engine._dag.get_session_nodes("foreground-session")
        ] == [foreground_node_id]

        engine.on_session_start(
            "foreground-continuation",
            boundary_reason="compression",
            old_session_id="foreground-session",
            platform="telegram",
            context_length=200000,
        )

        assert engine._session_id == "foreground-continuation"
        assert engine._conversation_id == foreground_conversation_id
        assert engine._store.get_session_count("foreground-session") == 1
        assert engine._store.get_session_count("foreground-continuation") == 0
        assert engine._dag.get_session_nodes("foreground-session") == []
        assert [
            node.node_id for node in engine._dag.get_session_nodes("foreground-continuation")
        ] == [foreground_node_id]

    def test_auxiliary_child_with_explicit_parent_id_and_aux_frame_does_not_need_state_db_row(self, tmp_path):
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
            """
        )
        conn.commit()
        conn.close()

        class ExplicitParentAuxFrame:
            def __init__(self, session_id: str, parent_session_id: str):
                self.session_id = session_id
                self._parent_session_id = parent_session_id
                self.enabled_toolsets = ["memory", "skills"]
                self.log_prefix = ""

            def on_session_start(self, lcm_engine: LCMEngine) -> None:
                lcm_engine.on_session_start(
                    self.session_id,
                    hermes_home=str(hermes_home),
                    platform="telegram",
                    context_length=200000,
                    parent_session_id=self._parent_session_id,
                )

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_explicit_parent.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = ExplicitParentAuxFrame("background-review-session", "foreground-session")
        child.on_session_start(engine)

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        self.HostAgentFrame("background-review-session", "foreground-session", hermes_home).should_compress_preflight(engine, [
            {"role": "user", "content": "explicit parent aux-frame child must stay stateless"},
        ])
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_auxiliary_child_parent_id_can_be_inferred_from_host_agent_frame(self, tmp_path):
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
            """
        )
        conn.commit()
        conn.close()

        class HostAgentFrame:
            def __init__(self, session_id: str, parent_session_id: str):
                self.session_id = session_id
                self._parent_session_id = parent_session_id
                self.enabled_toolsets = ["memory", "skills"]
                self.log_prefix = ""

            def notify_context_engine(self, lcm_engine: LCMEngine) -> None:
                lcm_engine.on_session_start(
                    self.session_id,
                    hermes_home=str(hermes_home),
                    platform="telegram",
                    context_length=200000,
                )

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_frame_parent.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = HostAgentFrame(
            session_id="background-review-session",
            parent_session_id="foreground-session",
        )
        child.notify_context_engine(engine)

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        self.HostAgentFrame(
            "background-review-session",
            "foreground-session",
            hermes_home,
        ).should_compress_preflight(engine, [
            {"role": "user", "content": "frame parent child must stay stateless"},
        ])
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("background-review-session") == 0

    def test_auxiliary_child_parent_frame_is_honored_even_on_fresh_engine(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_fresh_engine.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        child = self._start_host_child(
            engine,
            hermes_home,
            "background-review-session",
            "foreground-session",
        )

        assert engine._session_id == ""
        assert engine._thread_context_session_id() == ""
        assert engine._thread_context_has_auxiliary_session("background-review-session")
        child.should_compress_preflight(engine, [
            {"role": "user", "content": "fresh-engine child must stay stateless"},
        ])
        assert engine._store.get_session_count("background-review-session") == 0

    def test_delegate_depth_only_parent_frame_is_auxiliary(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        class DelegateDepthOnlyFrame:
            def __init__(self, session_id: str, parent_session_id: str):
                self.session_id = session_id
                self._parent_session_id = parent_session_id
                self.enabled_toolsets = ["terminal", "file"]
                self.log_prefix = ""
                self._delegate_depth = 1

            def on_session_start(self, lcm_engine: LCMEngine) -> None:
                lcm_engine.on_session_start(
                    self.session_id,
                    hermes_home=str(hermes_home),
                    platform="telegram",
                    context_length=200000,
                )

            def should_compress_preflight(self, lcm_engine: LCMEngine, messages):
                return lcm_engine.should_compress_preflight(messages)

        config = LCMConfig(database_path=str(tmp_path / "lcm_delegate_depth_only.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = DelegateDepthOnlyFrame("delegate-child-session", "foreground-session")
        child.on_session_start(engine)

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_has_auxiliary_session("delegate-child-session")
        assert child.should_compress_preflight(engine, [
            {"role": "user", "content": "delegate-depth child must stay stateless"},
        ]) is False
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("delegate-child-session") == 0

    def test_auxiliary_frame_detection_survives_wrapper_depth(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        class DeepWrappedChildFrame:
            def __init__(self, session_id: str, parent_session_id: str):
                self.session_id = session_id
                self._parent_session_id = parent_session_id
                self.enabled_toolsets = ["terminal", "file"]
                self.log_prefix = "[subagent-deep] "
                self._delegate_depth = 0

            def on_session_start(self, lcm_engine: LCMEngine) -> None:
                def wrapped(depth: int) -> None:
                    if depth <= 0:
                        lcm_engine.on_session_start(
                            self.session_id,
                            hermes_home=str(hermes_home),
                            platform="telegram",
                            context_length=200000,
                        )
                        return
                    wrapped(depth - 1)

                wrapped(20)

            def should_compress_preflight(self, lcm_engine: LCMEngine, messages):
                def wrapped(depth: int):
                    if depth <= 0:
                        return lcm_engine.should_compress_preflight(messages)
                    return wrapped(depth - 1)

                return wrapped(20)

        config = LCMConfig(database_path=str(tmp_path / "lcm_deep_wrapper_aux.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        child = DeepWrappedChildFrame("deep-child-session", "foreground-session")
        child.on_session_start(engine)

        assert engine._session_id == "foreground-session"
        assert engine._thread_context_has_auxiliary_session("deep-child-session")
        assert child.should_compress_preflight(engine, [
            {"role": "user", "content": "deep wrapper child must stay stateless"},
        ]) is False
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("deep-child-session") == 0

    def test_unrelated_parent_frame_does_not_make_foreground_branch_auxiliary(self, tmp_path):
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

        class UnrelatedFrame:
            session_id = "some-other-session"
            _parent_session_id = "foreground-session"

            def notify_context_engine(self, lcm_engine: LCMEngine) -> None:
                lcm_engine.on_session_start(
                    "foreground-branch-session",
                    hermes_home=str(hermes_home),
                    platform="tui",
                    context_length=200000,
                )

        config = LCMConfig(database_path=str(tmp_path / "lcm_unrelated_frame.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
        )
        UnrelatedFrame().notify_context_engine(engine)

        assert engine._session_id == "foreground-branch-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("foreground-branch-session")

    def test_auxiliary_child_model_update_does_not_mutate_foreground_threshold(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        class HostAgentFrame:
            def __init__(self, session_id: str, parent_session_id: str):
                self.session_id = session_id
                self._parent_session_id = parent_session_id
                self.enabled_toolsets = ["memory", "skills"]
                self.log_prefix = ""

            def update_context_engine(self, lcm_engine: LCMEngine) -> None:
                lcm_engine.update_model("tiny-child-model", 1000)

        config = LCMConfig(database_path=str(tmp_path / "lcm_aux_update_model.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        engine.update_model("foreground-model", 200000)
        assert engine.context_length == 200000
        assert engine.threshold_tokens == int(200000 * config.context_threshold)

        HostAgentFrame(
            session_id="background-review-session",
            parent_session_id="foreground-session",
        ).update_context_engine(engine)

        assert engine.context_length == 200000
        assert engine.threshold_tokens == int(200000 * config.context_threshold)

    def test_state_db_only_child_session_can_rebind_as_foreground_branch(self, tmp_path):
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

        config = LCMConfig(database_path=str(tmp_path / "lcm_foreground_branch.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
        )
        engine.on_session_start(
            "foreground-branch-session",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
        )

        assert engine._session_id == "foreground-branch-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("foreground-branch-session")

        engine.should_compress_preflight([
            {"role": "user", "content": "foreground branch should persist normally"},
        ])

        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("foreground-branch-session") == 1

    def test_explicit_parent_id_state_db_child_can_rebind_as_foreground_branch(self, tmp_path):
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

        config = LCMConfig(database_path=str(tmp_path / "lcm_explicit_foreground_branch.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        engine.on_session_start(
            "foreground-branch-session",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
            parent_session_id="foreground-session",
        )

        assert engine._session_id == "foreground-branch-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("foreground-branch-session")
        engine.should_compress_preflight([
            {"role": "user", "content": "explicit parent foreground branch should persist"},
        ])
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("foreground-branch-session") == 1

    def test_active_auxiliary_reused_as_foreground_with_parent_metadata_rebinds(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_active_aux_reused_foreground.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
        )
        aux_child = self._start_host_child(
            engine,
            hermes_home,
            "foreground-branch-session",
            "foreground-session",
        )
        aux_child.update_from_response(engine, {"prompt_tokens": 25})
        assert engine._thread_context_has_auxiliary_session("foreground-branch-session")

        engine.on_session_start(
            "foreground-branch-session",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
            parent_session_id="foreground-session",
        )

        assert engine._session_id == "foreground-branch-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("foreground-branch-session")
        engine.should_compress_preflight([
            {"role": "user", "content": "active aux id reused as foreground persists normally"},
        ])
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("foreground-branch-session") == 1

    def test_explicit_parent_id_without_aux_frame_or_live_row_rebinds_foreground_branch(self, tmp_path):
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
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_explicit_missing_row_branch.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        engine.on_session_start(
            "foreground-branch-session",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
            parent_session_id="foreground-session",
        )

        assert engine._session_id == "foreground-branch-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("foreground-branch-session")
        engine.should_compress_preflight([
            {"role": "user", "content": "explicit parent missing row branch should persist"},
        ])
        assert engine._store.get_session_count("foreground-session") == 0
        assert engine._store.get_session_count("foreground-branch-session") == 1

    def test_auxiliary_lineage_does_not_poison_reused_root_session_id(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_aux_id.db"))
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
            "reused-session",
            "foreground-session",
        )
        engine.on_session_end("reused-session", [])
        assert engine._has_auxiliary_lineage_session("reused-session")

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=200000,
        )

        assert engine._session_id == "reused-session"
        assert engine._thread_context_session_id() == ""
        assert not engine._thread_context_has_auxiliary_session("reused-session")
        messages = [
            {"role": "user", "content": "reused root session should persist normally"},
        ]
        engine.should_compress_preflight(messages)
        assert engine._store.get_session_count("reused-session") == 1

        messages.append({"role": "assistant", "content": "reused root session final message"})
        engine.on_session_end(
            "reused-session",
            messages,
        )
        assert engine._store.get_session_count("reused-session") == 2
        state = engine._lifecycle.get_by_conversation(engine._conversation_id)
        assert state.last_finalized_session_id == "reused-session"

    def test_auxiliary_lineage_does_not_poison_reused_root_inside_non_aux_parent_frame(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_aux_id_frame.db"))
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
            "reused-session",
            "foreground-session",
        )
        engine.on_session_end("reused-session", [])
        assert engine._has_auxiliary_lineage_session("reused-session")

        class ForegroundFrame:
            def __init__(self):
                self.session_id = "reused-session"
                self._parent_session_id = "foreground-session"
                self.enabled_toolsets = ["terminal", "file"]
                self.log_prefix = ""
                self._delegate_depth = 0

            def should_compress_preflight(self, lcm_engine: LCMEngine, messages):
                return lcm_engine.should_compress_preflight(messages)

            def on_session_end(self, lcm_engine: LCMEngine, messages) -> None:
                lcm_engine.on_session_end(self.session_id, messages)

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=200000,
        )
        frame = ForegroundFrame()
        messages = [
            {"role": "user", "content": "reused root frame should persist normally"},
        ]
        frame.should_compress_preflight(engine, messages)
        assert engine._thread_context_session_id() == ""
        assert engine._store.get_session_count("reused-session") == 1
        messages.append({"role": "assistant", "content": "reused root frame final"})
        frame.on_session_end(engine, messages)
        assert engine._store.get_session_count("reused-session") == 2

    def test_auxiliary_lineage_does_not_poison_reused_root_parent_branches(self, tmp_path):
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
            VALUES ('reused-parent', NULL, 10.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('reused-child-state', 'reused-parent', 11.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_aux_parent_branch.db"))
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
            "reused-parent",
            "foreground-session",
        )
        assert engine._has_auxiliary_lineage_session("reused-parent")
        assert engine._thread_context_has_auxiliary_session("reused-parent")

        # If that id later becomes a real foreground/root session, the old
        # auxiliary lineage must not poison real child branches under it.
        engine.on_session_start(
            "reused-parent",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=200000,
        )
        assert engine._session_id == "reused-parent"
        assert not engine._thread_context_has_auxiliary_session("reused-parent")

        engine.on_session_start(
            "reused-child-state",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
        )
        assert engine._session_id == "reused-child-state"
        assert not engine._thread_context_has_auxiliary_session("reused-child-state")
        engine.should_compress_preflight([
            {"role": "user", "content": "state-db child of reused parent persists"},
        ])
        assert engine._store.get_session_count("reused-child-state") == 1

        engine.on_session_start(
            "reused-parent",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=200000,
        )
        engine.on_session_start(
            "reused-child-explicit",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=200000,
            parent_session_id="reused-parent",
        )
        assert engine._session_id == "reused-child-explicit"
        assert not engine._thread_context_has_auxiliary_session("reused-child-explicit")
        engine.should_compress_preflight([
            {"role": "user", "content": "explicit child of reused parent persists"},
        ])
        assert engine._store.get_session_count("reused-child-explicit") == 1

    def test_guarded_auxiliary_reuse_does_not_poison_later_foreground_children(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_guarded_reuse_foreground.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        old_child = self._start_host_child(
            engine,
            hermes_home,
            "reused-parent",
            "foreground-session",
        )
        replacement = self._start_host_child(
            engine,
            hermes_home,
            "reused-parent",
            "foreground-session",
        )
        replacement.update_from_response(engine, {"prompt_tokens": 25})
        engine.on_session_end("reused-parent", [])
        assert not engine._thread_context_has_auxiliary_session("reused-parent")
        assert not old_child.should_compress(engine)

        engine.on_session_start(
            "reused-parent",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        assert engine._session_id == "reused-parent"
        assert engine._has_auxiliary_lineage_session("reused-parent")
        assert not engine._thread_context_has_auxiliary_session("reused-parent")
        old_child.should_compress_preflight(engine, [
            {"role": "user", "content": "stale old child must stay stateless"},
        ])
        assert engine._store.get_session_count("reused-parent") == 0

        engine.on_session_start(
            "other-foreground-session",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        old_child.on_compression_boundary(
            engine,
            "stale-reused-parent-continuation",
            context_length=1_000,
        )
        assert "reused-parent" in engine._auxiliary_foreground_reused_session_ids

        stale_descendant = self._start_host_child(
            engine,
            hermes_home,
            "stale-reused-parent-descendant",
            "reused-parent",
            context_length=1_000,
        )
        assert engine._session_id == "other-foreground-session"
        assert engine._thread_context_has_auxiliary_session("stale-reused-parent-descendant")
        stale_descendant.should_compress_preflight(engine, [
            {"role": "user", "content": "stale in-process descendant must stay stateless"},
        ])
        assert engine._store.get_session_count("stale-reused-parent-descendant") == 0
        assert engine._store.get_session_count("other-foreground-session") == 0

        engine.on_session_start(
            "reused-child-explicit",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=1_000,
            parent_session_id="reused-parent",
        )
        assert engine._session_id == "reused-child-explicit"
        assert not engine._thread_context_has_auxiliary_session("reused-child-explicit")
        engine.should_compress_preflight([
            {"role": "user", "content": "guarded reuse child persists"},
        ])
        assert engine._store.get_session_count("reused-child-explicit") == 1

        engine.on_session_start(
            "reused-child-sibling",
            hermes_home=str(hermes_home),
            platform="tui",
            context_length=1_000,
            parent_session_id="reused-parent",
        )
        assert engine._session_id == "reused-child-sibling"
        assert not engine._thread_context_has_auxiliary_session("reused-child-sibling")
        engine.should_compress_preflight([
            {"role": "user", "content": "guarded reuse sibling persists"},
        ])
        assert engine._store.get_session_count("reused-child-sibling") == 1

    def test_stateless_auxiliary_rebind_preserves_parent_lineage(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_stateless_aux_rebind_lineage.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        child = self._start_host_child(
            engine,
            hermes_home,
            "reused-auxiliary",
            "foreground-session",
        )
        child.on_session_end(engine, [])
        assert engine._has_auxiliary_lineage_session("reused-auxiliary")
        assert not engine._auxiliary_lineage_suppressed_as_foreground("reused-auxiliary")

        engine.on_session_start(
            "reused-auxiliary",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="normal-conversation",
            context_length=1_000,
        )
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-auxiliary")
        normal_prefix = [{"role": "user", "content": "normal prefix before stateless rebind"}]
        engine.ingest(normal_prefix)
        assert engine._store.get_session_count("reused-auxiliary") == 1

        engine.on_session_start(
            "reused-auxiliary",
            hermes_home=str(hermes_home),
            platform="stateless",
            context_length=1_000,
        )
        assert engine._session_stateless
        assert not engine._thread_context_has_auxiliary_session("reused-auxiliary")
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-auxiliary")

        engine.on_session_start(
            "explicit-child-of-bound-stateless-rebind",
            hermes_home=str(hermes_home),
            platform="cli",
            parent_session_id="reused-auxiliary",
            context_length=1_000,
        )
        assert engine._session_id == "reused-auxiliary"
        assert engine._thread_context_has_auxiliary_session(
            "explicit-child-of-bound-stateless-rebind"
        )
        engine.should_compress_preflight([
            {"role": "user", "content": "bound stateless child must stay stateless"},
        ])
        assert engine._store.get_session_count("explicit-child-of-bound-stateless-rebind") == 0
        assert engine._store.get_session_count("reused-auxiliary") == 1
        engine.on_session_end("explicit-child-of-bound-stateless-rebind", [])
        assert not engine._thread_context_has_auxiliary_session(
            "explicit-child-of-bound-stateless-rebind"
        )

        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        engine.on_session_start(
            "explicit-child-of-stateless-rebind",
            hermes_home=str(hermes_home),
            platform="cli",
            parent_session_id="reused-auxiliary",
            context_length=1_000,
        )
        assert engine._session_id == "foreground-2"
        assert engine._thread_context_has_auxiliary_session("explicit-child-of-stateless-rebind")
        engine.should_compress_preflight([
            {"role": "user", "content": "child of stateless side channel must stay stateless"},
        ])
        assert engine._store.get_session_count("explicit-child-of-stateless-rebind") == 0
        assert engine._store.get_session_count("foreground-2") == 0

        engine.on_session_end(
            "reused-auxiliary",
            normal_prefix + [
                {"role": "assistant", "content": "normal final after stateless rebind"},
            ],
        )
        rows = engine._store.get_range("reused-auxiliary")
        assert [row["content"] for row in rows] == [
            "normal prefix before stateless rebind",
            "normal final after stateless rebind",
        ]

    def test_auxiliary_lineage_does_not_block_reused_root_compression_boundary(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_aux_boundary.db"))
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
            "reused-session",
            "foreground-session",
        )
        engine.on_session_end("reused-session", [])
        assert engine._has_auxiliary_lineage_session("reused-session")

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=200000,
        )
        store_id = engine._store.append(
            "reused-session",
            {"role": "user", "content": "reused root compression source"},
            token_estimate=5,
            source="cli",
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="reused-session",
            depth=0,
            summary="reused root node should move",
            token_count=4,
            source_token_count=5,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        conversation_id = engine._conversation_id
        engine._last_compacted_store_id = store_id

        engine.on_session_start(
            "reused-continuation",
            boundary_reason="compression",
            old_session_id="reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=200000,
        )

        assert engine._session_id == "reused-continuation"
        assert engine._conversation_id == conversation_id
        assert engine._thread_context_session_id() == ""
        assert engine._store.get_session_count("reused-session") == 1
        assert engine._store.get_session_count("reused-continuation") == 0
        assert engine._dag.get_session_nodes("reused-session") == []
        assert [
            node.node_id for node in engine._dag.get_session_nodes("reused-continuation")
        ] == [node_id]

    def test_side_channel_child_preserves_own_foreground_reuse_suffix(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_side_channel_child_own_reuse.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        stateless_parent = self._start_host_child(
            engine,
            hermes_home,
            "stateless-parent",
            "foreground-session",
        )
        stateless_parent.on_session_end(engine, [])
        engine.on_session_start(
            "stateless-parent",
            hermes_home=str(hermes_home),
            platform="stateless",
            context_length=1_000,
        )
        assert engine._lcm_session_last_bypassed["stateless-parent"]

        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        reused_child = self._start_host_child(
            engine,
            hermes_home,
            "reused-child",
            "foreground-2",
        )
        reused_child.on_session_end(engine, [])
        engine.on_session_start(
            "reused-child",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="reused-child-normal",
            context_length=1_000,
        )
        normal_prefix = [{"role": "user", "content": "reused child normal prefix"}]
        engine.ingest(normal_prefix)
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-child")

        engine.on_session_start(
            "foreground-3",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        side_channel_child = self.HostAgentFrame(
            "reused-child",
            "stateless-parent",
            hermes_home,
        )
        side_channel_child.on_session_start(
            engine,
            parent_session_id="stateless-parent",
        )
        assert engine.current_session_id == "foreground-3"
        assert engine._thread_context_has_auxiliary_session("reused-child")
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-child")
        side_channel_child.should_compress_preflight(engine, [
            {"role": "user", "content": "side-channel child payload"},
        ])
        assert engine._store.get_session_count("reused-child") == 1
        assert engine._store.get_session_count("foreground-3") == 0

        side_channel_end = normal_prefix + [
            {"role": "assistant", "content": "side-channel final must not append"},
        ]
        errors = []

        def end_side_channel_from_other_thread():
            try:
                engine.on_session_end("reused-child", side_channel_end)
            except Exception as exc:  # pragma: no cover - surfaced by assertion below
                errors.append(exc)

        end_thread = threading.Thread(target=end_side_channel_from_other_thread)
        end_thread.start()
        end_thread.join()
        assert errors == []
        assert not engine._thread_context_has_auxiliary_session("reused-child")
        rows = engine._store.get_range("reused-child")
        assert [row["content"] for row in rows] == ["reused child normal prefix"]
        engine.on_session_end("reused-child", side_channel_end)
        rows = engine._store.get_range("reused-child")
        assert [row["content"] for row in rows] == ["reused child normal prefix"]

        engine.on_session_end(
            "reused-child",
            normal_prefix + [
                {"role": "assistant", "content": "reused child normal final"},
            ],
        )
        rows = engine._store.get_range("reused-child")
        assert [row["content"] for row in rows] == [
            "reused child normal prefix",
            "reused child normal final",
        ]

    def test_state_db_child_of_bypassed_reused_parent_stays_auxiliary(self, tmp_path):
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
            VALUES ('stateless-parent', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('state-db-child', 'stateless-parent', 3.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_state_db_bypassed_child.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine._auxiliary_lineage_session_ids.add("stateless-parent")
        engine._auxiliary_foreground_reused_session_ids.add("stateless-parent")
        engine._lcm_session_last_bypassed["stateless-parent"] = True

        engine.on_session_start(
            "state-db-child",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )

        assert engine.current_session_id == "foreground-session"
        assert engine._thread_context_has_auxiliary_session("state-db-child")
        assert engine._store.get_session_count("state-db-child") == 0
        assert engine.should_compress_preflight([
            {"role": "user", "content": "state db child must stay stateless"},
        ]) is False
        assert engine._store.get_session_count("state-db-child") == 0

    def test_state_db_grandchild_of_bypassed_reused_parent_stays_auxiliary(self, tmp_path):
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
            VALUES ('stateless-parent', 'foreground-session', 2.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('intermediate-child', 'stateless-parent', 3.0, NULL);
            INSERT INTO sessions(id, parent_session_id, started_at, ended_at)
            VALUES ('state-db-grandchild', 'intermediate-child', 4.0, NULL);
            """
        )
        conn.commit()
        conn.close()

        config = LCMConfig(database_path=str(tmp_path / "lcm_state_db_bypassed_grandchild.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        engine._auxiliary_lineage_session_ids.add("stateless-parent")
        engine._auxiliary_foreground_reused_session_ids.add("stateless-parent")
        engine._lcm_session_last_bypassed["stateless-parent"] = True

        engine.on_session_start(
            "state-db-grandchild",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )

        assert engine.current_session_id == "foreground-session"
        assert engine._thread_context_has_auxiliary_session("state-db-grandchild")
        assert engine._store.get_session_count("state-db-grandchild") == 0
        assert engine.should_compress_preflight([
            {"role": "user", "content": "state db grandchild must stay stateless"},
        ]) is False
        assert engine._store.get_session_count("state-db-grandchild") == 0

    def test_side_channel_child_marks_prior_normal_id_as_foreground_reuse(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_side_channel_child_prior_normal.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        stateless_parent = self._start_host_child(
            engine,
            hermes_home,
            "stateless-parent",
            "foreground-session",
        )
        stateless_parent.on_session_end(engine, [])
        engine.on_session_start(
            "stateless-parent",
            hermes_home=str(hermes_home),
            platform="stateless",
            context_length=1_000,
        )

        engine.on_session_start(
            "prior-normal-child",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="prior-normal-child-conversation",
            context_length=1_000,
        )
        normal_prefix = [{"role": "user", "content": "prior normal child prefix"}]
        engine.ingest(normal_prefix)
        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        engine.on_session_start(
            "prior-normal-child",
            hermes_home=str(hermes_home),
            platform="cli",
            parent_session_id="stateless-parent",
            context_length=1_000,
        )
        assert engine._auxiliary_lineage_suppressed_as_foreground("prior-normal-child")

        side_channel_end = normal_prefix + [
            {"role": "assistant", "content": "prior normal side-channel final"},
        ]
        engine.on_session_end("prior-normal-child", side_channel_end)
        assert not engine._thread_context_has_auxiliary_session("prior-normal-child")
        assert [row["content"] for row in engine._store.get_range("prior-normal-child")] == [
            "prior normal child prefix",
        ]

        engine.on_session_end(
            "prior-normal-child",
            normal_prefix + [
                {"role": "assistant", "content": "prior normal child final"},
            ],
        )
        assert [row["content"] for row in engine._store.get_range("prior-normal-child")] == [
            "prior normal child prefix",
            "prior normal child final",
        ]

    def test_inferred_side_channel_child_preserves_prior_normal_suffix(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_inferred_side_channel_child_prior_normal.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        stateless_parent = self._start_host_child(
            engine,
            hermes_home,
            "stateless-parent",
            "foreground-session",
        )
        stateless_parent.on_session_end(engine, [])
        engine.on_session_start(
            "stateless-parent",
            hermes_home=str(hermes_home),
            platform="stateless",
            context_length=1_000,
        )

        engine.on_session_start(
            "prior-normal-child",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="prior-normal-child-conversation",
            context_length=1_000,
        )
        normal_prefix = [{"role": "user", "content": "inferred prior normal prefix"}]
        engine.ingest(normal_prefix)
        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        side_channel_child = self.HostAgentFrame(
            "prior-normal-child",
            "stateless-parent",
            hermes_home,
        )
        side_channel_child.on_session_start(engine)
        assert engine._auxiliary_lineage_suppressed_as_foreground("prior-normal-child")
        side_channel_child.on_session_end(
            engine,
            normal_prefix + [
                {"role": "assistant", "content": "inferred side-channel final"},
            ],
        )
        assert [row["content"] for row in engine._store.get_range("prior-normal-child")] == [
            "inferred prior normal prefix",
        ]

        engine.on_session_end(
            "prior-normal-child",
            normal_prefix + [
                {"role": "assistant", "content": "inferred prior normal final"},
            ],
        )
        assert [row["content"] for row in engine._store.get_range("prior-normal-child")] == [
            "inferred prior normal prefix",
            "inferred prior normal final",
        ]

    def test_side_channel_prefix_only_end_fails_closed_for_ambiguous_suffix(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(
            database_path=str(tmp_path / "lcm_side_channel_prefix_only_end.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=1_000,
        )
        stateless_parent = self._start_host_child(
            engine,
            hermes_home,
            "stateless-parent",
            "foreground-session",
        )
        stateless_parent.on_session_end(engine, [])
        engine.on_session_start(
            "stateless-parent",
            hermes_home=str(hermes_home),
            platform="stateless",
            context_length=1_000,
        )

        engine.on_session_start(
            "reused-child",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="reused-child-normal",
            context_length=1_000,
        )
        normal_prefix = [{"role": "user", "content": "prefix-only normal prefix"}]
        engine.ingest(normal_prefix)
        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            context_length=1_000,
        )
        side_channel_child = self.HostAgentFrame(
            "reused-child",
            "stateless-parent",
            hermes_home,
        )
        side_channel_child.on_session_start(
            engine,
            parent_session_id="stateless-parent",
        )
        side_channel_child.on_session_end(engine, normal_prefix)

        engine.on_session_end(
            "reused-child",
            normal_prefix + [
                {"role": "assistant", "content": "ambiguous side-channel final must not append"},
            ],
        )
        assert [row["content"] for row in engine._store.get_range("reused-child")] == [
            "prefix-only normal prefix",
        ]

    def test_off_current_reused_auxiliary_root_end_preserves_normal_suffix(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_aux_off_current_end.db"))
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
            "reused-session",
            "foreground-session",
        )
        engine.on_session_end("reused-session", [])
        assert engine._has_auxiliary_lineage_session("reused-session")

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="normal-conversation",
            context_length=200000,
        )
        normal_prefix = [{"role": "user", "content": "normal prefix"}]
        engine.ingest(normal_prefix)
        assert engine._store.get_session_count("reused-session") == 1

        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="foreground-conversation",
            context_length=200000,
        )
        engine.on_session_end(
            "reused-session",
            normal_prefix + [{"role": "assistant", "content": "normal final"}],
        )

        rows = engine._store.get_range("reused-session")
        assert [row["content"] for row in rows] == ["normal prefix", "normal final"]
        normal_state = engine._lifecycle.get_by_conversation("normal-conversation")
        assert normal_state is not None
        assert normal_state.last_finalized_session_id == "reused-session"
        assert engine.current_session_id == "foreground-2"

    def test_stale_auxiliary_boundary_preserves_off_current_reused_normal_suffix(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_aux_stale_boundary.db"))
        engine = LCMEngine(config=config, hermes_home=str(hermes_home))
        engine.on_session_start(
            "foreground-session",
            hermes_home=str(hermes_home),
            platform="telegram",
            context_length=200000,
        )
        stale_child = self._start_host_child(
            engine,
            hermes_home,
            "reused-session",
            "foreground-session",
        )
        stale_child.on_session_end(engine, [])
        assert engine._has_auxiliary_lineage_session("reused-session")

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="normal-conversation",
            context_length=200000,
        )
        normal_prefix = [{"role": "user", "content": "normal prefix"}]
        engine.ingest(normal_prefix)
        reused_node_id = engine._dag.add_node(SummaryNode(
            session_id="reused-session",
            depth=0,
            summary="reused normal node must not move",
            token_count=5,
            source_token_count=5,
            source_type="messages",
            created_at=time.time(),
        ))
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-session")

        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="foreground-conversation",
            context_length=200000,
        )
        new_frame_stale_continuation = self.HostAgentFrame(
            "stale-continuation",
            "reused-session",
            hermes_home,
        )
        new_frame_stale_continuation.on_compression_boundary_from(
            engine,
            old_session_id="reused-session",
        )
        assert engine.current_session_id == "foreground-2"
        assert engine._dag.get_node(reused_node_id).session_id == "reused-session"
        assert engine._dag.get_session_nodes("stale-continuation") == []
        assert engine._thread_context_has_auxiliary_session("stale-continuation")
        stale_child.on_compression_boundary(engine, "stale-continuation")
        assert engine._auxiliary_lineage_suppressed_as_foreground("reused-session")

        engine.on_session_end(
            "reused-session",
            normal_prefix + [{"role": "assistant", "content": "normal final"}],
        )

        rows = engine._store.get_range("reused-session")
        assert [row["content"] for row in rows] == ["normal prefix", "normal final"]
        normal_state = engine._lifecycle.get_by_conversation("normal-conversation")
        assert normal_state is not None
        assert normal_state.last_finalized_session_id == "reused-session"
        assert engine.current_session_id == "foreground-2"

    def test_off_current_reused_auxiliary_root_boundary_keeps_normal_carryover(self, tmp_path):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()

        config = LCMConfig(database_path=str(tmp_path / "lcm_reused_aux_off_current_boundary.db"))
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
            "reused-session",
            "foreground-session",
        )
        engine.on_session_end("reused-session", [])
        assert engine._has_auxiliary_lineage_session("reused-session")

        engine.on_session_start(
            "reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="normal-conversation",
            context_length=200000,
        )
        store_id = engine._store.append(
            "reused-session",
            {"role": "user", "content": "normal node source"},
            token_estimate=5,
            source="cli",
            conversation_id="normal-conversation",
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="reused-session",
            depth=0,
            summary="normal reused root node should move",
            token_count=4,
            source_token_count=5,
            source_ids=[store_id],
            source_type="messages",
            created_at=time.time(),
        ))
        engine._last_compacted_store_id = store_id

        engine.on_session_start(
            "foreground-2",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="foreground-conversation",
            context_length=200000,
        )
        engine.on_session_start(
            "reused-continuation",
            boundary_reason="compression",
            old_session_id="reused-session",
            hermes_home=str(hermes_home),
            platform="cli",
            conversation_id="normal-conversation",
            context_length=200000,
        )

        assert engine._session_id == "reused-continuation"
        assert engine._conversation_id == "normal-conversation"
        assert not engine._thread_context_has_auxiliary_session("reused-continuation")
        assert engine._dag.get_session_nodes("reused-session") == []
        assert [
            node.node_id for node in engine._dag.get_session_nodes("reused-continuation")
        ] == [node_id]

    def test_off_current_suffix_only_late_bypass_end_after_normal_rebound_stays_stateless(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_late_bypass_suffix_after_normal.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            engine.on_session_start(
                "reused-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            bypass_messages = [
                {"role": "user", "content": "ignored prefix that must stay out"},
            ]
            engine.ingest(bypass_messages)
            engine.on_session_end("reused-session", bypass_messages)
            assert engine._store.get_session_count("reused-session") == 0

            engine.on_session_start(
                "reused-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            normal_messages = [
                {"role": "user", "content": "normal rebound prefix"},
            ]
            engine.ingest(normal_messages)
            assert engine._store.get_session_count("reused-session") == 1

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_suffix_only_bypass_end = [
                {"role": "assistant", "content": "late ignored suffix must not persist"},
            ]

            engine.on_session_end("reused-session", late_suffix_only_bypass_end)

            assert engine._store.get_session_count("reused-session") == 1
            assert engine._store.get_session_count("foreground-session") == 0
            assert engine._dag.get_session_node_count("reused-session") == 0
            assert engine._dag.get_session_node_count("foreground-session") == 0
            assert engine.current_session_id == "foreground-session"
            assert engine.current_conversation_id == "foreground-conversation"
        finally:
            engine.shutdown()

    @pytest.mark.parametrize(
        ("bypass_session_id", "config_kwargs"),
        [
            ("ignored:late", {"ignore_session_patterns": ["ignored:*"]}),
            ("stateless:late", {"stateless_session_patterns": ["stateless:*"]}),
        ],
    )
    def test_direct_off_current_bypass_session_end_does_not_write_or_finalize_foreground(
        self,
        tmp_path,
        bypass_session_id,
        config_kwargs,
    ):
        payload_dir = tmp_path / f"payloads-{bypass_session_id.replace(':', '-')}"
        config = LCMConfig(
            database_path=str(tmp_path / f"direct-{bypass_session_id.replace(':', '-')}.db"),
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=64,
            large_output_externalization_path=str(payload_dir),
            **config_kwargs,
        )
        engine = LCMEngine(config=config)
        try:
            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_bypass_end = [
                {"role": "user", "content": "should not store " + ("x" * 200)},
            ]

            engine.on_session_end(bypass_session_id, late_bypass_end)

            assert engine._store.get_session_count("foreground-session") == 0
            assert engine._store.get_session_count(bypass_session_id) == 0
            assert engine._dag.get_session_node_count("foreground-session") == 0
            assert engine._dag.get_session_node_count(bypass_session_id) == 0
            assert not payload_dir.exists() or list(payload_dir.glob("*.json")) == []
            assert engine.current_session_id == "foreground-session"
            assert engine.current_conversation_id == "foreground-conversation"
            foreground_state = engine._lifecycle.get_by_conversation("foreground-conversation")
            assert foreground_state is not None
            assert foreground_state.current_session_id == "foreground-session"
            assert foreground_state.last_finalized_session_id != bypass_session_id
            assert engine._lifecycle.get_by_session(bypass_session_id) is None
        finally:
            engine.shutdown()

    def test_suffix_only_late_bypass_end_does_not_externalize_when_skipped(self, tmp_path):
        payload_dir = tmp_path / "payloads"
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_late_bypass_no_externalized_payload.db"),
            stateless_session_patterns=["stateless"],
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=64,
            large_output_externalization_path=str(payload_dir),
        )
        engine = LCMEngine(config=config)
        try:
            engine.on_session_start(
                "reused-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest([{"role": "user", "content": "bypass seed"}])

            engine.on_session_start(
                "reused-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            normal_messages = [
                {"role": "user", "content": "normal rebound prefix"},
            ]
            engine.ingest(normal_messages)

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_suffix_only_bypass_end = [
                {"role": "assistant", "content": "late bypass payload " + ("x" * 200)},
            ]

            engine.on_session_end("reused-session", late_suffix_only_bypass_end)

            assert engine._store.get_session_count("reused-session") == 1
            assert not payload_dir.exists() or list(payload_dir.glob("*.json")) == []
        finally:
            engine.shutdown()

    def test_shared_opener_bypass_and_normal_ambiguity_does_not_append_bypass_suffix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_shared_opener_bypass_normal_ambiguity.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared_opener = {"role": "user", "content": "shared opener"}
            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest([shared_opener])

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest([shared_opener])

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_bypass_end = [
                shared_opener,
                {"role": "assistant", "content": "bypass suffix must not append"},
            ]

            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            assert [row["content"] for row in normal_rows] == ["shared opener"]
            assert all(row["content"] != "bypass suffix must not append" for row in rows)
        finally:
            engine.shutdown()

    def test_multiple_bypass_prefixes_do_not_append_older_late_bypass_suffix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_multiple_bypass_prefixes.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared_opener = {"role": "user", "content": "shared opener A"}
            newer_bypass_opener = {"role": "user", "content": "newer bypass opener B"}

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation-a",
                context_length=200000,
            )
            engine.ingest([shared_opener])

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation-b",
                context_length=200000,
            )
            engine.ingest([newer_bypass_opener])

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest([shared_opener])

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_older_bypass_end = [
                shared_opener,
                {"role": "assistant", "content": "older bypass suffix must not append"},
            ]

            engine.on_session_end("shared-session", late_older_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            assert [row["content"] for row in normal_rows] == ["shared opener A"]
            assert all(row["content"] != "older bypass suffix must not append" for row in rows)
        finally:
            engine.shutdown()

    def test_off_current_store_mismatch_does_not_append_from_recorded_prefix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_off_current_store_mismatch.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            normal_messages = [
                {"role": "user", "content": f"A{i}"}
                for i in range(9)
            ]
            divergent_end = [
                *normal_messages[:8],
                {"role": "assistant", "content": "B8 must not append"},
                {"role": "assistant", "content": "new divergent suffix must not append"},
            ]

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest(normal_messages)

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )

            engine.on_session_end("shared-session", divergent_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            assert [row["content"] for row in normal_rows] == [f"A{i}" for i in range(9)]
            assert all(row["content"] != "B8 must not append" for row in rows)
            assert all(row["content"] != "new divergent suffix must not append" for row in rows)
        finally:
            engine.shutdown()

    def test_off_current_recorded_prefix_appends_when_store_normalizes_equivalent_row(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_off_current_recorded_prefix_normalized.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            normalized_prefix = {
                "role": "assistant",
                "content": "stored row normalizes empty tool_calls",
                "tool_calls": [],
            }
            normal_suffix = {"role": "user", "content": "normal suffix must append"}

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest([normalized_prefix])

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            engine.on_session_end("shared-session", [normalized_prefix, normal_suffix])

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            assert [row["content"] for row in normal_rows] == [
                "stored row normalizes empty tool_calls",
                "normal suffix must append",
            ]
            assert all(row["conversation_id"] != "bypass-conversation" for row in rows)
        finally:
            engine.shutdown()

    def test_off_current_store_prefix_appends_normalized_empty_tool_calls_beyond_recorded_limit(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_off_current_long_normalized_prefix.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            normal_prefix = [
                {
                    "role": "assistant",
                    "content": "long prefix stores empty tool_calls as NULL",
                    "tool_calls": [],
                },
                *(
                    {"role": "user", "content": f"long normalized prefix {idx}"}
                    for idx in range(8)
                ),
            ]
            normal_suffix = {"role": "assistant", "content": "long normalized suffix must append"}

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest(normal_prefix)

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            engine.on_session_end("shared-session", [*normal_prefix, normal_suffix])

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            assert [row["content"] for row in normal_rows] == [
                "long prefix stores empty tool_calls as NULL",
                *(f"long normalized prefix {idx}" for idx in range(8)),
                "long normalized suffix must append",
            ]
            assert normal_rows[0]["tool_calls"] is None
            assert all(row["conversation_id"] != "bypass-conversation" for row in rows)
        finally:
            engine.shutdown()

    @pytest.mark.parametrize("bypass_tool_calls", [[], {}])
    def test_off_current_empty_tool_calls_bypass_tie_does_not_append_suffix(
        self,
        tmp_path,
        bypass_tool_calls,
    ):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_off_current_empty_tool_calls_bypass_tie.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            bypass_prefix = {
                "role": "assistant",
                "content": "shared empty tool_calls opener",
                "tool_calls": bypass_tool_calls,
            }
            normal_prefix = {
                "role": "assistant",
                "content": "shared empty tool_calls opener",
            }
            late_bypass_end = [
                {
                    "role": "assistant",
                    "content": "shared empty tool_calls opener",
                },
                {
                    "role": "assistant",
                    "content": "empty tool_calls bypass suffix must not append",
                },
            ]

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest([bypass_prefix])

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest([normal_prefix])

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            bypass_rows = [row for row in rows if row["conversation_id"] == "bypass-conversation"]
            assert [row["content"] for row in normal_rows] == ["shared empty tool_calls opener"]
            assert normal_rows[0]["tool_calls"] is None
            assert bypass_rows == []
            assert all(
                row["content"] != "empty tool_calls bypass suffix must not append"
                for row in rows
            )
        finally:
            engine.shutdown()

    def test_off_current_truncated_bypass_prefix_does_not_append_ambiguous_suffix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_off_current_truncated_bypass_prefix.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared_prefix = [
                {"role": "user", "content": f"shared prefix {i}"}
                for i in range(9)
            ]

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest(shared_prefix)

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest(shared_prefix)

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_bypass_end = [
                *shared_prefix,
                {"role": "assistant", "content": "truncated bypass suffix must not append"},
            ]

            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            bypass_rows = [row for row in rows if row["conversation_id"] == "bypass-conversation"]
            assert [row["content"] for row in normal_rows] == [
                f"shared prefix {i}"
                for i in range(9)
            ]
            assert bypass_rows == []
            assert all(row["content"] != "truncated bypass suffix must not append" for row in rows)
        finally:
            engine.shutdown()

    def test_off_current_duplicate_short_bypass_snapshot_keeps_truncated_ambiguity(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_off_current_duplicate_short_bypass_prefix.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared8 = [
                {"role": "user", "content": f"duplicate shared prefix {i}"}
                for i in range(8)
            ]
            shared9 = [
                *shared8,
                {"role": "assistant", "content": "duplicate shared ninth"},
            ]

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="long-bypass-conversation",
                context_length=200000,
            )
            engine.ingest(shared9)
            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="short-bypass-conversation",
                context_length=200000,
            )
            engine.ingest(shared8)

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest(shared9)

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_bypass_end = [
                *shared9,
                {"role": "assistant", "content": "duplicate truncated bypass suffix must not append"},
            ]

            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            bypass_rows = [
                row
                for row in rows
                if row["conversation_id"] in {"long-bypass-conversation", "short-bypass-conversation"}
            ]
            assert [row["content"] for row in normal_rows] == [
                *(f"duplicate shared prefix {i}" for i in range(8)),
                "duplicate shared ninth",
            ]
            assert bypass_rows == []
            assert all(
                row["content"] != "duplicate truncated bypass suffix must not append"
                for row in rows
            )
        finally:
            engine.shutdown()

    @pytest.mark.parametrize("bypass_probe", ["preflight", "compress"])
    def test_shared_opener_bypass_probe_ambiguity_does_not_append_bypass_suffix(self, tmp_path, bypass_probe):
        config = LCMConfig(
            database_path=str(tmp_path / f"lcm_shared_opener_bypass_{bypass_probe}_ambiguity.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared_opener = {"role": "user", "content": f"shared opener from {bypass_probe}"}
            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            if bypass_probe == "preflight":
                engine.should_compress_preflight([shared_opener])
            else:
                engine.compress([shared_opener], force=True)

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest([shared_opener])

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            late_bypass_end = [
                shared_opener,
                {"role": "assistant", "content": f"{bypass_probe} bypass suffix must not append"},
            ]

            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            assert [row["content"] for row in normal_rows] == [f"shared opener from {bypass_probe}"]
            assert all(row["content"] != f"{bypass_probe} bypass suffix must not append" for row in rows)
            assert engine._store.get_session_count("foreground-session") == 0
        finally:
            engine.shutdown()

    def test_current_normal_reuse_with_tied_bypass_prefix_stores_final_suffix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_current_normal_tied_bypass_prefix.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared_opener = {"role": "user", "content": "shared opener reused as normal"}
            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest([shared_opener])
            engine.on_session_end("shared-session", [shared_opener])
            assert engine._store.get_session_count("shared-session") == 0

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            normal_prefix = [shared_opener]
            engine.ingest(normal_prefix)

            normal_end = normal_prefix + [
                {"role": "assistant", "content": "normal final suffix after tied prefix"},
            ]
            engine.on_session_end("shared-session", normal_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            bypass_rows = [row for row in rows if row["conversation_id"] == "bypass-conversation"]
            assert [row["content"] for row in normal_rows] == [
                "shared opener reused as normal",
                "normal final suffix after tied prefix",
            ]
            assert bypass_rows == []

            normal_state = engine._lifecycle.get_by_conversation("normal-conversation")
            assert normal_state is not None
            assert normal_state.current_session_id is None
            assert normal_state.last_finalized_session_id == "shared-session"
        finally:
            engine.shutdown()

    def test_current_normal_reuse_with_truncated_bypass_prefix_does_not_append_ambiguous_suffix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_current_normal_truncated_bypass_prefix.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared_prefix = [
                {"role": "user", "content": f"current shared prefix {i}"}
                for i in range(9)
            ]

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest(shared_prefix)
            engine.on_session_end("shared-session", shared_prefix)
            assert engine._store.get_session_count("shared-session") == 0

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest(shared_prefix)

            late_bypass_end = [
                *shared_prefix,
                {"role": "assistant", "content": "current truncated bypass suffix must not persist"},
            ]
            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            bypass_rows = [row for row in rows if row["conversation_id"] == "bypass-conversation"]
            assert [row["content"] for row in normal_rows] == [
                f"current shared prefix {i}"
                for i in range(9)
            ]
            assert bypass_rows == []
            assert all(row["content"] != "current truncated bypass suffix must not persist" for row in rows)
        finally:
            engine.shutdown()

    def test_current_duplicate_short_bypass_snapshot_keeps_truncated_ambiguity(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_current_duplicate_short_bypass_prefix.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            shared8 = [
                {"role": "user", "content": f"current duplicate shared prefix {i}"}
                for i in range(8)
            ]
            long_bypass = [
                *shared8,
                {"role": "assistant", "content": "current duplicate long bypass ninth"},
            ]

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="long-bypass-conversation",
                context_length=200000,
            )
            engine.ingest(long_bypass)
            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="short-bypass-conversation",
                context_length=200000,
            )
            engine.ingest(shared8)

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest(shared8)

            late_bypass_end = [
                *shared8,
                {"role": "assistant", "content": "current duplicate truncated suffix must not persist"},
            ]
            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            bypass_rows = [
                row
                for row in rows
                if row["conversation_id"] in {"long-bypass-conversation", "short-bypass-conversation"}
            ]
            assert [row["content"] for row in normal_rows] == [
                f"current duplicate shared prefix {i}"
                for i in range(8)
            ]
            assert bypass_rows == []
            assert all(
                row["content"] != "current duplicate truncated suffix must not persist"
                for row in rows
            )
        finally:
            engine.shutdown()

    def test_current_normal_reuse_does_not_tie_distinct_bypass_tool_call_prefix(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_current_normal_distinct_tool_call_prefix.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            bypass_prefix = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "bypass_call",
                            "type": "function",
                            "function": {"name": "side_channel", "arguments": "{}"},
                        }
                    ],
                }
            ]
            normal_prefix = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "normal_call",
                            "type": "function",
                            "function": {"name": "foreground", "arguments": "{}"},
                        }
                    ],
                }
            ]

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest(bypass_prefix)
            engine.on_session_end("shared-session", bypass_prefix)

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            engine.ingest(normal_prefix)

            late_bypass_end = bypass_prefix + [
                {"role": "assistant", "content": "late bypass suffix must not persist"},
            ]
            engine.on_session_end("shared-session", late_bypass_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            bypass_rows = [row for row in rows if row["conversation_id"] == "bypass-conversation"]
            assert [row["content"] for row in normal_rows] == [""]
            assert normal_rows[0]["tool_calls"][0]["id"] == "normal_call"
            assert bypass_rows == []
            assert all(row["content"] != "late bypass suffix must not persist" for row in rows)
        finally:
            engine.shutdown()

    def test_normal_final_after_later_stateless_rebind_uses_normal_source(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_later_stateless_rebind_normal_source.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            normal_prefix = [
                {"role": "user", "content": "normal prefix before stateless rebind"},
            ]
            engine.ingest(normal_prefix)

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="stateless-conversation",
                context_length=200000,
            )
            engine.ingest([{"role": "user", "content": "stateless same-id turn"}])

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            normal_end = normal_prefix + [
                {"role": "assistant", "content": "normal final after stateless rebind"},
            ]

            engine.on_session_end("shared-session", normal_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            final_rows = [
                row
                for row in normal_rows
                if row["content"] == "normal final after stateless rebind"
            ]
            stateless_rows = [row for row in rows if row["conversation_id"] == "stateless-conversation"]
            assert len(final_rows) == 1
            assert final_rows[0]["source"] == "cli"
            assert stateless_rows == []
        finally:
            engine.shutdown()

    def test_reused_normal_session_id_scopes_off_current_dedupe_to_current_conversation(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_reused_normal_id_current_conversation.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="bypass-conversation",
                context_length=200000,
            )
            engine.ingest([{"role": "user", "content": "bypass seed"}])

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="old-normal-conversation",
                context_length=200000,
            )
            old_messages = [
                {"role": "user", "content": "old conversation prefix"},
                {"role": "assistant", "content": "old conversation answer"},
            ]
            engine.ingest(old_messages)
            engine.on_session_end("shared-session", old_messages)

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="current-normal-conversation",
                context_length=200000,
            )
            current_prefix = [
                {"role": "user", "content": "current conversation prefix"},
            ]
            engine.ingest(current_prefix)

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            current_end = current_prefix + [
                {"role": "assistant", "content": "current conversation final"},
            ]

            engine.on_session_end("shared-session", current_end)

            rows = engine._store.get_session_messages("shared-session")
            old_rows = [row for row in rows if row["conversation_id"] == "old-normal-conversation"]
            current_rows = [row for row in rows if row["conversation_id"] == "current-normal-conversation"]
            bypass_rows = [row for row in rows if row["conversation_id"] == "bypass-conversation"]

            assert [row["content"] for row in old_rows] == [
                "old conversation prefix",
                "old conversation answer",
            ]
            assert [row["content"] for row in current_rows] == [
                "current conversation prefix",
                "current conversation final",
            ]
            assert bypass_rows == []
            current_state = engine._lifecycle.get_by_conversation("current-normal-conversation")
            old_state = engine._lifecycle.get_by_conversation("old-normal-conversation")
            assert current_state is not None
            assert old_state is not None
            assert current_state.last_finalized_session_id == "shared-session"
            assert old_state.last_finalized_session_id == "shared-session"
        finally:
            engine.shutdown()

    def test_later_bypass_bind_does_not_replace_normal_conversation_for_off_current_finalization(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_later_bypass_preserves_normal_conversation.db"),
            stateless_session_patterns=["stateless"],
        )
        engine = LCMEngine(config=config)
        try:
            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="initial-bypass-conversation",
                context_length=200000,
            )
            engine.ingest([{"role": "user", "content": "initial bypass seed"}])

            engine.on_session_start(
                "shared-session",
                platform="cli",
                conversation_id="normal-conversation",
                context_length=200000,
            )
            normal_prefix = [
                {"role": "user", "content": "normal prefix before later bypass"},
            ]
            engine.ingest(normal_prefix)

            engine.on_session_start(
                "shared-session",
                platform="stateless",
                conversation_id="later-bypass-conversation",
                context_length=200000,
            )
            engine.ingest([{"role": "user", "content": "later bypass must not own final"}])

            engine.on_session_start(
                "foreground-session",
                platform="cli",
                conversation_id="foreground-conversation",
                context_length=200000,
            )
            normal_end = normal_prefix + [
                {"role": "assistant", "content": "normal final after later bypass"},
            ]

            engine.on_session_end("shared-session", normal_end)

            rows = engine._store.get_session_messages("shared-session")
            normal_rows = [row for row in rows if row["conversation_id"] == "normal-conversation"]
            later_bypass_rows = [row for row in rows if row["conversation_id"] == "later-bypass-conversation"]
            assert [row["content"] for row in normal_rows] == [
                "normal prefix before later bypass",
                "normal final after later bypass",
            ]
            assert later_bypass_rows == []

            normal_state = engine._lifecycle.get_by_conversation("normal-conversation")
            bypass_state = engine._lifecycle.get_by_conversation("later-bypass-conversation")
            assert normal_state is not None
            assert normal_state.last_finalized_session_id == "shared-session"
            assert bypass_state is not None
            assert bypass_state.last_finalized_session_id != "shared-session"
        finally:
            engine.shutdown()
