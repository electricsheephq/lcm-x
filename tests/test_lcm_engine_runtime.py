"""LCM engine integration tests: runtime."""

import json
import logging
import sqlite3
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.tools as lcm_tools

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.engine_registry import (
    ActiveEngineUseStatus,
    use_active_lcm_engine,
)

from .engine_helpers import (
    engine as engine,
    _FakeTimeoutRegexEngine,
)


def test_delegated_child_clone_keeps_host_model_and_window(tmp_path):
    prototype = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "child-clone.db")))
    clone = prototype.clone_for_agent()

    class ChildAgent:
        session_id = "child"
        _parent_session_id = "parent"
        log_prefix = "[subagent-1] "

        def start(self):
            clone.update_model("gpt-5.5", 272_000, provider="openai-codex", api_mode="responses")
            clone.on_session_start(
                self.session_id, platform="subagent", model="gpt-5.5",
                context_length=clone.context_length,
            )
            assert clone.model == "gpt-5.5"
            assert clone.context_length == 272_000
            assert clone.provider == "openai-codex"
            assert clone.api_mode == "responses"
            assert clone._session_id == ""
            self.check_threshold()
            clone.update_model("fallback-model", 128_000, provider="openrouter")
            assert clone.model == "fallback-model"
            assert clone.context_length == 128_000
            assert clone.provider == "openrouter"
            assert clone._session_id == ""
            self.check_threshold()

        def check_threshold(self):
            assert clone.threshold_tokens > 0
            assert clone._bypasses_lcm_context_management()
            assert clone.should_compress(clone.threshold_tokens)
            assert not clone.should_compress(clone.threshold_tokens - 1)

    try:
        ChildAgent().start()
    finally:
        clone.shutdown()
        prototype.shutdown()



@pytest.mark.parametrize("make_clone", ["clone_for_agent", "deepcopy"])
def test_delegated_child_clone_of_bound_parent_accepts_child_updates(tmp_path, make_clone):
    import copy

    parent = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "bound-parent.db")))
    parent.update_model("gpt-5.5", 272_000, provider="openai-codex", api_mode="responses")
    parent.on_session_start("parent", platform="cli", model="gpt-5.5", context_length=272_000)
    assert parent._session_id == "parent"
    parent_before = (parent.model, parent.context_length, parent.threshold_tokens, parent._session_id)

    clone = parent.clone_for_agent() if make_clone == "clone_for_agent" else copy.deepcopy(parent)

    class ChildAgent:
        # The update runs inside a delegated child's frame, so the bound-engine guard is armed: a clone that kept
        # the parent's session binding would ignore this update and keep a zero threshold.
        session_id = "child"
        _parent_session_id = "parent"
        log_prefix = "[subagent-1] "

        def start(self):
            clone.update_model("child-model", 128_000, provider="openrouter")

    try:
        assert clone is not parent
        assert clone._session_id == ""
        ChildAgent().start()
        assert clone.model == "child-model"
        assert clone.context_length == 128_000
        assert clone.threshold_tokens > 0
        assert clone.should_compress(clone.threshold_tokens)
        assert not clone.should_compress(clone.threshold_tokens - 1)
        assert (parent.model, parent.context_length, parent.threshold_tokens, parent._session_id) == parent_before
    finally:
        clone.shutdown()
        parent.shutdown()



def test_subagent_child_model_update_does_not_mutate_foreground_threshold(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "shared-child.db"))
    instance = LCMEngine(config=config)

    class ChildAgent:
        session_id = "child"
        _parent_session_id = "foreground-session"
        log_prefix = "[subagent-1] "

        def update_context_engine(self):
            instance.update_model("tiny-child-model", 1_000)

    try:
        instance.on_session_start("foreground-session", platform="telegram", context_length=200_000)
        instance.update_model("foreground-model", 200_000)
        ChildAgent().update_context_engine()
        assert instance._session_id == "foreground-session"
        assert instance.model == "foreground-model"
        assert instance.context_length == 200_000
        assert instance.threshold_tokens == int(200_000 * config.context_threshold)
    finally:
        instance.shutdown()



@pytest.mark.parametrize("summary_model", ["", "aux-summary"])
def test_bypass_host_fallback_never_sends_unknown_model(tmp_path, monkeypatch, caplog, summary_model):
    constructors = []
    updates = []

    class FakeContextCompressor:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            self.compression_count = 0

        def update_model(self, **kwargs):
            updates.append(kwargs)

        def compress(self, messages, **kwargs):
            self.compression_count += 1
            return [{"role": "system", "content": "native compacted"}]

    compressor_module = ModuleType("agent.context_compressor")
    compressor_module.ContextCompressor = FakeContextCompressor
    monkeypatch.setitem(sys.modules, "agent.context_compressor", compressor_module)
    config = LCMConfig(database_path=str(tmp_path / "unresolved-model.db"), summary_model=summary_model)
    instance = LCMEngine(config=config)
    messages = [{"role": "user", "content": f"message {i}"} for i in range(10)]
    try:
        instance._set_context_length(1_000, source="test")
        instance._mark_thread_context_stateless("aux:sid")
        with caplog.at_level(logging.INFO):
            result = instance.compress(messages, force=True)
            instance.compress(messages, force=True)
        if not summary_model:
            assert constructors == []
            assert updates == []
            assert any("[Context omitted:" in row.get("content", "") for row in result)
            assert caplog.text.count("host_fallback_unresolved_model") == 1
        else:
            assert [call["model"] for call in constructors] == ["aux-summary"]
            assert updates and all(call["model"] == "aux-summary" for call in updates)
            assert result == [{"role": "system", "content": "native compacted"}]
        assert all(call["model"] != "unknown" for call in constructors + updates)
    finally:
        instance.shutdown()



def test_shutdown_closes_lifecycle_store(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "shutdown-lifecycle.db"))
    engine = LCMEngine(config=config)

    engine.shutdown()

    assert engine._lifecycle._conn is None



def test_stable_active_use_blocks_rebind_and_keeps_ingest_in_original_session(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "stable-active-use.db"))
    engine = LCMEngine(config=config)
    use_started = threading.Event()
    release_use = threading.Event()
    rebind_started = threading.Event()
    rebind_finished = threading.Event()
    result_holder = {}
    try:
        engine.on_session_start(
            "session-a",
            platform="discord",
            conversation_id="conversation-a",
        )

        def use_session_a():
            def ingest_while_stable(selected):
                use_started.set()
                assert release_use.wait(timeout=2)
                selected.ingest([{"role": "user", "content": "stable payload for A"}])

            result_holder["use"] = use_active_lcm_engine(
                ingest_while_stable,
                session_id="session-a",
                conversation_id="conversation-a",
            )

        def rebind_to_b():
            rebind_started.set()
            engine.on_session_start(
                "session-b",
                platform="discord",
                conversation_id="conversation-b",
            )
            rebind_finished.set()

        use_thread = threading.Thread(target=use_session_a)
        rebind_thread = threading.Thread(target=rebind_to_b)
        use_thread.start()
        assert use_started.wait(timeout=2)
        rebind_thread.start()
        assert rebind_started.wait(timeout=2)
        assert not rebind_finished.wait(timeout=0.05)
        assert engine.bound_session_id == "session-a"

        release_use.set()
        use_thread.join(timeout=2)
        rebind_thread.join(timeout=2)
        assert not use_thread.is_alive()
        assert not rebind_thread.is_alive()
        assert result_holder["use"].status is ActiveEngineUseStatus.USED
        assert engine.bound_session_id == "session-b"

        rows = engine._store.search("stable payload")
        assert len(rows) == 1
        assert rows[0]["session_id"] == "session-a"
        assert rows[0]["conversation_id"] == "conversation-a"

        late_a = use_active_lcm_engine(
            lambda selected: selected.ingest(
                [{"role": "user", "content": "late stale payload for A"}]
            ),
            session_id="session-a",
            conversation_id="conversation-a",
        )
        assert late_a.status is ActiveEngineUseStatus.ENGINE_NOT_RESIDENT
        assert engine._store.search("late stale payload") == []

        engine.on_session_end(
            "session-a",
            [
                {"role": "user", "content": "stable payload for A"},
                {"role": "assistant", "content": "late lifecycle payload for A"},
            ],
        )
        lifecycle_rows = engine._store.search("late lifecycle payload")
        assert len(lifecycle_rows) == 1
        assert lifecycle_rows[0]["session_id"] == "session-a"
        assert lifecycle_rows[0]["conversation_id"] == "conversation-a"
        assert all(
            row["session_id"] != "session-b"
            for row in lifecycle_rows
        )

        def reject_reentrant_rebind(selected):
            with pytest.raises(RuntimeError, match="during stable engine use"):
                selected.on_session_start(
                    "session-c",
                    conversation_id="conversation-c",
                )
            return selected.bound_session_id

        reentrant = use_active_lcm_engine(
            reject_reentrant_rebind,
            session_id="session-b",
            conversation_id="conversation-b",
        )
        assert reentrant.status is ActiveEngineUseStatus.USED
        assert reentrant.value == "session-b"
        assert engine.bound_session_id == "session-b"
    finally:
        release_use.set()
        engine.shutdown()



def test_shutdown_waits_for_stable_use_and_is_terminal_and_idempotent(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "stable-shutdown.db"))
    engine = LCMEngine(config=config)
    operation_started = threading.Event()
    release_operation = threading.Event()
    shutdown_finished = threading.Event()
    try:
        engine.on_session_start("session-a", conversation_id="conversation-a")

        def hold_engine(selected):
            operation_started.set()
            assert release_operation.wait(timeout=2)

        use_thread = threading.Thread(
            target=lambda: use_active_lcm_engine(
                hold_engine,
                session_id="session-a",
                conversation_id="conversation-a",
            )
        )
        shutdown_thread = threading.Thread(
            target=lambda: (engine.shutdown(), shutdown_finished.set())
        )
        use_thread.start()
        assert operation_started.wait(timeout=2)
        shutdown_thread.start()
        assert not shutdown_finished.wait(timeout=0.05)

        release_operation.set()
        use_thread.join(timeout=2)
        shutdown_thread.join(timeout=2)
        assert shutdown_finished.is_set()
        engine.shutdown()

        closed = engine._run_stably(lambda selected: selected)
        assert closed.status is ActiveEngineUseStatus.CLOSED_ENGINE
    finally:
        release_operation.set()
        engine.shutdown()



def test_shutdown_attempts_all_cleanup_and_remains_retryable_after_failure(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "retryable-shutdown.db"))
    engine = LCMEngine(config=config)
    attempts = 0

    def flaky_close():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("synthetic cleanup failure")

    engine._adaptive_retrieval = SimpleNamespace(close=flaky_close)
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        engine.shutdown()

    assert not engine._stable_use_closed
    assert engine._store._conn is None
    assert engine._dag._conn is None
    assert engine._lifecycle._conn is None

    engine.shutdown()
    assert attempts == 2
    assert engine._stable_use_closed



def test_assertion_store_is_default_off_and_closes_when_enabled(tmp_path):
    disabled_db = tmp_path / "assertions-disabled.db"
    disabled = LCMEngine(config=LCMConfig(database_path=str(disabled_db)))
    try:
        assert disabled._assertions is None
        assert disabled._store._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'lcm_assertion%'"
        ).fetchone()[0] == 0
    finally:
        disabled.shutdown()

    enabled_db = tmp_path / "assertions-enabled.db"
    enabled = LCMEngine(
        config=LCMConfig(
            database_path=str(enabled_db),
            assertions_enabled=True,
        )
    )
    assertion_store = enabled._assertions
    assert assertion_store is not None
    assert assertion_store.db_path == enabled._store.db_path == enabled_db

    enabled.shutdown()

    assert assertion_store._conn is None



def test_discord_short_turn_ingest_preserves_conversation_id(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "discord-lanes.db"))
    engine = LCMEngine(config=config)
    try:
        conversation_id = "agent:main:discord:thread:1520890589762031776:1520890589762031776"
        engine.on_session_start(
            "discord-session-1",
            platform="discord",
            conversation_id=conversation_id,
            context_length=200_000,
        )

        engine.ingest([
            {"role": "user", "content": "needle from discord topic"},
            {"role": "assistant", "content": "topic answer"},
        ])

        rows = engine._store.search(
            "needle",
            source="discord",
            conversation_id=conversation_id,
        )
        assert len(rows) == 1
        assert rows[0]["session_id"] == "discord-session-1"
        assert rows[0]["source"] == "discord"
        assert rows[0]["conversation_id"] == conversation_id
    finally:
        engine.shutdown()



def test_webui_like_local_short_turn_ingest_preserves_lane_metadata(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "webui-local.db"))
    engine = LCMEngine(config=config)
    try:
        conversation_id = "agent:main:local:webui:session-42"
        engine.on_session_start(
            "webui-session-42",
            platform="local",
            conversation_id=conversation_id,
            model="local-test-model",
            provider="local",
            context_length=128_000,
        )

        engine.ingest([
            {"role": "user", "content": "needle from local webui lane"},
            {"role": "assistant", "content": "local lane answer"},
        ])
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        rows = engine._store.search(
            "needle",
            source="local",
            conversation_id=conversation_id,
        )

        assert len(rows) == 1
        assert rows[0]["session_id"] == "webui-session-42"
        assert rows[0]["source"] == "local"
        assert rows[0]["conversation_id"] == conversation_id
        assert engine.current_session_platform == "local"
        assert status["runtime_identity"]["conversation_id"] == conversation_id
    finally:
        engine.shutdown()



def test_engine_deallocation_releases_sqlite_fds_without_gc(tmp_path):
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.exists():
        pytest.skip("fd count is Linux-specific")
    before = len(list(fd_dir.iterdir()))

    for idx in range(5):
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"engine-{idx}.db")))
        del engine

    after = len(list(fd_dir.iterdir()))
    assert after <= before + 2



def test_reused_engine_rebinds_storage_when_hermes_home_changes(tmp_path):
    """Plugin-side guard for Hermes hosts that reuse one engine across profiles."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    config = LCMConfig(database_path="")
    engine = LCMEngine(config=config, hermes_home=str(home_a))
    try:
        engine.context_length = 200000
        engine.threshold_tokens = int(200000 * config.context_threshold)

        engine.on_session_start(
            "session-a",
            hermes_home=str(home_a),
            platform="cli",
            context_length=200000,
        )
        assert Path(engine._store.db_path) == home_a / "lcm.db"
        engine._ingest_messages([{"role": "user", "content": "message from profile a"}])
        assert engine._store.get_session_count("session-a") == 1

        engine.on_session_start(
            "session-b",
            hermes_home=str(home_b),
            platform="cli",
            context_length=200000,
        )
        assert Path(engine._store.db_path) == home_b / "lcm.db"
        assert engine._session_id == "session-b"
        assert engine._store.get_session_count("session-a") == 0
        engine._ingest_messages([{"role": "user", "content": "message from profile b"}])
        assert engine._store.get_session_count("session-b") == 1

        with sqlite3.connect(home_a / "lcm.db") as conn_a:
            rows_a = conn_a.execute("SELECT session_id, content FROM messages").fetchall()
        with sqlite3.connect(home_b / "lcm.db") as conn_b:
            rows_b = conn_b.execute("SELECT session_id, content FROM messages").fetchall()

        assert rows_a == [("session-a", "message from profile a")]
        assert rows_b == [("session-b", "message from profile b")]
    finally:
        engine.shutdown()



def test_assertion_store_rebinds_with_profile_home(tmp_path):
    home_a = tmp_path / "assertion-profile-a"
    home_b = tmp_path / "assertion-profile-b"
    config = LCMConfig(database_path="", assertions_enabled=True)
    engine = LCMEngine(config=config, hermes_home=str(home_a))
    try:
        first_store = engine._assertions
        assert first_store is not None
        assert first_store.db_path == engine._store.db_path == home_a / "lcm.db"

        engine.on_session_start(
            "session-b",
            hermes_home=str(home_b),
            platform="cli",
            context_length=200_000,
        )

        assert first_store._conn is None
        assert engine._assertions is not None
        assert engine._assertions.db_path == engine._store.db_path == home_b / "lcm.db"
        for db_path in (home_a / "lcm.db", home_b / "lcm.db"):
            with sqlite3.connect(db_path) as conn:
                assert conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type='table' AND name='lcm_assertions'"
                ).fetchone()[0] == 1
    finally:
        engine.shutdown()



def test_profile_rebind_clears_old_auxiliary_session_state(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    config = LCMConfig(database_path="")
    engine = LCMEngine(config=config, hermes_home=str(home_a))
    try:
        engine.on_session_start(
            "session-a",
            hermes_home=str(home_a),
            platform="cli",
            context_length=200000,
        )
        engine._mark_thread_context_stateless("old-profile-aux")
        assert engine._has_auxiliary_lineage_session("old-profile-aux")
        assert engine._thread_context_stateless()

        engine.on_session_start(
            "session-b",
            hermes_home=str(home_b),
            platform="cli",
            parent_session_id="old-profile-aux",
            context_length=200000,
        )

        assert engine._session_id == "session-b"
        assert not engine._has_auxiliary_lineage_session("old-profile-aux")
        assert not engine._thread_context_stateless()
    finally:
        engine.shutdown()



def test_config_database_path_profile_rebind_updates_externalization_home(tmp_path, monkeypatch):
    # #1055: force externalization for this fixture independently of the bound window.
    monkeypatch.setattr("hermes_lcm.ingest_protection._non_tool_text_over_floor", lambda text, config, **_: bool(text))
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    config = LCMConfig(
        database_path=str(tmp_path / "shared-lcm.db"),
        large_output_externalization_enabled=True,
        large_output_externalization_threshold_chars=10,
    )
    engine = LCMEngine(config=config, hermes_home=str(home_a))
    try:
        engine.on_session_start(
            "session-a",
            hermes_home=str(home_a),
            platform="cli",
            context_length=200000,
        )
        engine._ingest_messages([{"role": "assistant", "content": "profile-a " + "A" * 32}])
        assert len(list((home_a / "lcm-large-outputs").glob("*.json"))) == 1

        engine.on_session_start(
            "session-b",
            hermes_home=str(home_b),
            platform="cli",
            context_length=200000,
        )
        assert engine._store._hermes_home == str(home_b)
        engine._ingest_messages([{"role": "assistant", "content": "profile-b " + "B" * 32}])

        assert len(list((home_a / "lcm-large-outputs").glob("*.json"))) == 1
        assert len(list((home_b / "lcm-large-outputs").glob("*.json"))) == 1
    finally:
        engine.shutdown()



def test_lcm_tool_status_reports_lifecycle_fragmentation_summary(engine, tmp_path):
    engine._hermes_home = str(tmp_path / "hermes_home")
    state_db = tmp_path / "hermes_home" / "state.db"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    state_conn = sqlite3.connect(state_db)
    state_conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY);
        INSERT INTO sessions(id) VALUES ('covered-session');
        """
    )
    state_conn.commit()
    state_conn.close()
    engine._store.append("covered-session", {"role": "user", "content": "covered"}, source="cli")
    engine._lifecycle._conn.execute(
        """INSERT INTO lcm_lifecycle_state
           (conversation_id, current_session_id, last_finalized_session_id, current_frontier_store_id, last_finalized_frontier_store_id, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("conv-stale", "missing-current", None, 0, 0, 1.0),
    )
    engine._lifecycle._conn.commit()

    payload = json.loads(lcm_tools.lcm_status({}, engine=engine))

    assert payload["lifecycle_fragmentation"]["read_only"] is True
    assert payload["lifecycle_fragmentation"]["lifecycle_rows"] == 1
    assert payload["lifecycle_fragmentation"]["lifecycle_current_missing_in_lcm_any"] == 1
    assert payload["lifecycle_fragmentation"]["lifecycle_current_missing_in_state"] == 1



def test_lcm_tool_status_includes_optional_cache_usage_metrics(engine):
    engine.update_from_response({
        "prompt_tokens": 1050,
        "completion_tokens": 120,
        "total_tokens": 1170,
        "input_tokens": 600,
        "output_tokens": 120,
        "cache_read_tokens": 400,
        "cache_write_tokens": 50,
        "reasoning_tokens": 30,
    })

    payload = json.loads(lcm_tools.lcm_status({}, engine=engine))

    assert payload["cache_metrics_available"] is True
    assert payload["last_input_tokens"] == 600
    assert payload["last_output_tokens"] == 120
    assert payload["last_cache_read_tokens"] == 400
    assert payload["last_cache_write_tokens"] == 50
    assert payload["last_reasoning_tokens"] == 30
    assert payload["cache_read_ratio"] == 0.381
    assert payload["last_compression_status"] == "idle"
    assert payload["last_compression_noop_reason"] == ""
    assert payload["runtime_identity"]["plugin_name"] == "hermes-lcm-x"
    assert payload["runtime_identity"]["database_path_source"] == "config.database_path"
    assert payload["config"]["summary_timeout_ms"] == 120_000



def test_compress_exception_sets_terminal_error_status_and_reraises_same_exception(
    engine,
    monkeypatch,
):
    original = RuntimeError("forced compaction failure")
    engine._last_compression_status = "noop"
    engine._last_compression_noop_reason = "stale no-op reason"

    def fail_ingest(_messages):
        raise original

    monkeypatch.setattr(engine, "_ingest_messages", fail_ingest)
    with pytest.raises(RuntimeError) as caught:
        engine.compress([{"role": "user", "content": "trigger"}])

    assert caught.value is original
    assert engine.last_compression_status == "error"
    assert engine.last_compression_noop_reason == ""
    assert engine.last_compression_was_noop is False



def test_update_model_updates_runtime_metadata_and_context_window(engine):
    engine.update_model(
        model="deepseek-v4-flash",
        context_length=1_000_000,
        base_url="https://opencode.ai/zen/go",
        api_key="test-key",
        provider="opencode-go",
        api_mode="anthropic_messages",
    )

    assert engine.model == "deepseek-v4-flash"
    assert engine.base_url == "https://opencode.ai/zen/go"
    assert engine.api_key == "test-key"
    assert engine.provider == "opencode-go"
    assert engine.api_mode == "anthropic_messages"
    assert engine.context_length == 1_000_000
    assert engine.threshold_tokens == int(1_000_000 * engine._config.context_threshold)



def test_codex_gpt55_uses_route_cap_and_hermes_autoraise_threshold(tmp_path):
    config = LCMConfig(
        context_threshold=0.68,
        database_path=str(tmp_path / "codex-gpt55.db"),
    )
    config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 272_000
        assert engine.effective_context_length_cap == 272_000
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"
        assert engine.context_threshold == 0.85
        assert engine.threshold_tokens == int(272_000 * 0.85)

        status = engine.get_status()
        assert status["raw_context_length"] == 400_000
        assert status["context_length"] == 272_000
        assert status["configured_context_threshold"] == 0.68
        assert status["context_threshold"] == 0.85
        assert status["context_threshold_source"] == "codex_gpt55_autoraise"
        assert status["context_threshold_autoraised"] == {"from": 0.68, "to": 0.85}
    finally:
        engine.shutdown()



def test_codex_gpt56_variants_share_272k_oauth_cap(engine):
    for model in ("gpt-5.6-sol", "gpt-5.6-luna"):
        engine.update_model(
            model=model,
            provider="openai-codex",
            context_length=372_000,
        )

        assert engine.raw_context_length == 372_000
        assert engine.context_length == 272_000
        assert engine.effective_context_length_cap == 272_000
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"



def test_codex_oauth_context_cap_keeps_explicit_lcm_threshold(tmp_path):
    config = LCMConfig(
        context_threshold=0.68,
        database_path=str(tmp_path / "codex-explicit-threshold.db"),
    )
    config.config_sources["context_threshold"] = "env:LCM_CONTEXT_THRESHOLD"
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.5-2026-04-23",
            provider="openai-codex",
            context_length=400_000,
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 272_000
        assert engine.context_threshold == 0.68
        assert engine.threshold_tokens == int(272_000 * 0.68)
        assert engine._effective_assembly_token_cap() is None

        status = engine.get_status()
        assert status["context_threshold_source"] == "env:LCM_CONTEXT_THRESHOLD"
        assert status["context_threshold_autoraised"] is None
    finally:
        engine.shutdown()



def test_codex_oauth_context_cap_keeps_lcm_config_yaml_threshold_override(tmp_path):
    config = LCMConfig(
        context_threshold=0.62,
        database_path=str(tmp_path / "codex-lcm-yaml-threshold.db"),
    )
    config.config_sources["context_threshold"] = "config_yaml:lcm.context_threshold"
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 272_000
        assert engine.context_threshold == 0.62
        assert engine.threshold_tokens == int(272_000 * 0.62)
        assert engine._context_threshold_source == "config_yaml:lcm.context_threshold"
        assert engine._context_threshold_autoraised is None
    finally:
        engine.shutdown()



def test_codex_gpt55_autoraise_can_be_disabled(tmp_path):
    config = LCMConfig(
        context_threshold=0.68,
        codex_gpt55_autoraise_enabled=False,
        database_path=str(tmp_path / "codex-autoraise-disabled.db"),
    )
    config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 272_000
        assert engine.context_threshold == 0.68
        assert engine.threshold_tokens == int(272_000 * 0.68)
        assert engine._context_threshold_source == "config_yaml:compression.threshold"
        assert engine._context_threshold_autoraised is None
    finally:
        engine.shutdown()



def test_codex_oauth_context_cap_applies_without_gpt55_threshold_magic(tmp_path):
    config = LCMConfig(
        context_threshold=0.68,
        database_path=str(tmp_path / "codex-spark-cap.db"),
    )
    config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.3-codex-spark",
            provider="openai-codex",
            context_length=400_000,
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 128_000
        assert engine.effective_context_length_cap == 128_000
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"
        assert engine.context_threshold == 0.68
        assert engine.threshold_tokens == int(128_000 * 0.68)
        assert engine._context_threshold_source == "config_yaml:compression.threshold"
        assert engine._context_threshold_autoraised is None
    finally:
        engine.shutdown()



def test_codex_oauth_context_cap_constrains_reserve_based_assembly_cap(tmp_path):
    config = LCMConfig(
        context_threshold=0.85,
        database_path=str(tmp_path / "codex-assembly-cap.db"),
        max_assembly_tokens=700_000,
        reserve_tokens_floor=24_000,
    )
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.5-pro",
            provider="openai-codex",
            context_length=400_000,
        )

        assert engine.context_length == 272_000
        assert engine._effective_assembly_token_cap() == 248_000
        assert engine.threshold_tokens == int(272_000 * 0.85)
    finally:
        engine.shutdown()



def test_threshold_tokens_respects_lower_max_assembly_cap_for_host_preflight(tmp_path):
    config = LCMConfig(
        context_threshold=0.75,
        database_path=str(tmp_path / "codex-low-assembly-cap.db"),
        max_assembly_tokens=96_000,
    )
    config.config_sources["context_threshold"] = "env:LCM_CONTEXT_THRESHOLD"
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
        )

        assert engine.context_length == 272_000
        assert int(272_000 * 0.75) == 204_000
        assert engine._effective_assembly_token_cap() == 96_000
        assert engine.threshold_tokens == 96_000
        assert engine.should_compress(96_000)
    finally:
        engine.shutdown()



def test_threshold_tokens_respects_lower_reserve_based_cap_for_host_preflight(tmp_path):
    config = LCMConfig(
        context_threshold=0.90,
        database_path=str(tmp_path / "reserve-low-assembly-cap.db"),
        reserve_tokens_floor=30_000,
    )
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-test",
            provider="openai",
            context_length=100_000,
        )

        assert engine._effective_assembly_token_cap() == 70_000
        assert engine.threshold_tokens == 70_000
        assert engine.should_compress(70_000)
    finally:
        engine.shutdown()



def test_non_codex_gpt55_keeps_host_context_window(engine):
    engine.update_model(
        model="gpt-5.5",
        provider="openai",
        context_length=400_000,
    )

    assert engine.raw_context_length == 400_000
    assert engine.context_length == 400_000
    assert engine.effective_context_length_cap is None
    assert engine.threshold_tokens == int(400_000 * engine._config.context_threshold)



def test_session_start_does_not_overwrite_update_model_context_length_with_stale_metadata(engine):
    engine.update_model(
        model="deepseek-v4-flash",
        context_length=1_000_000,
        provider="opencode-go",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="deepseek-v4-flash",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "deepseek-v4-flash"
    assert engine.context_length == 1_000_000
    assert engine.threshold_tokens == int(1_000_000 * engine._config.context_threshold)



def test_session_start_accepts_raw_context_length_for_capped_codex_runtime(tmp_path, caplog):
    config = LCMConfig(
        context_threshold=0.68,
        database_path=str(tmp_path / "codex-session-start.db"),
    )
    config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
    engine = LCMEngine(config=config)
    try:
        engine.update_model(
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
        )

        caplog.set_level(logging.WARNING)
        engine.on_session_start(
            "telegram:chat-1:session-2",
            platform="telegram",
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
            conversation_id="telegram:chat-1",
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 272_000
        assert engine.threshold_tokens == int(272_000 * 0.85)
        assert "ignored stale session-start context_length" not in caplog.text
    finally:
        engine.shutdown()



def test_session_start_only_uses_incoming_route_for_codex_gpt55_cap(tmp_path):
    config = LCMConfig(
        context_threshold=0.68,
        database_path=str(tmp_path / "codex-session-start-only.db"),
    )
    config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
    engine = LCMEngine(config=config)
    try:
        engine.on_session_start(
            "telegram:chat-1:session-1",
            platform="telegram",
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
            conversation_id="telegram:chat-1",
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 272_000
        assert engine.context_threshold == 0.85
        assert engine.threshold_tokens == int(272_000 * 0.85)
        assert engine.effective_context_length_reason == "codex_oauth_context_cap"

        engine.on_session_start(
            "telegram:chat-2:session-1",
            platform="telegram",
            model="gpt-5.5",
            provider="openai",
            context_length=400_000,
            conversation_id="telegram:chat-2",
        )

        assert engine.raw_context_length == 400_000
        assert engine.context_length == 400_000
        assert engine.context_threshold == 0.68
        assert engine.threshold_tokens == int(400_000 * 0.68)
        assert engine.effective_context_length_cap is None
        assert engine.effective_context_length_reason == ""
    finally:
        engine.shutdown()



def test_lcm_status_surfaces_capped_context_and_effective_threshold(tmp_path):
    config = LCMConfig(
        context_threshold=0.68,
        database_path=str(tmp_path / "codex-status.db"),
    )
    config.config_sources["context_threshold"] = "config_yaml:compression.threshold"
    engine = LCMEngine(config=config)
    try:
        engine._session_id = "codex-status-session"
        engine.update_model(
            model="gpt-5.5",
            provider="openai-codex",
            context_length=400_000,
        )

        payload = json.loads(lcm_tools.lcm_status({}, engine=engine))

        assert payload["raw_context_length"] == 400_000
        assert payload["context_length"] == 272_000
        assert payload["effective_context_length_cap"] == 272_000
        assert payload["effective_context_length_reason"] == "codex_oauth_context_cap"
        assert payload["configured_context_threshold"] == 0.68
        assert payload["context_threshold"] == 0.85
        assert payload["context_threshold_source"] == "codex_gpt55_autoraise"
        assert payload["context_threshold_autoraised"] == {"from": 0.68, "to": 0.85}
    finally:
        engine.shutdown()



def test_lcm_status_surfaces_configured_reasoning_efforts(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "reasoning-status.db"),
        summary_reasoning_effort="high",
        expansion_reasoning_effort="low",
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("reasoning-status", platform="cli", context_length=200_000)
    try:
        payload = json.loads(lcm_tools.lcm_status({}, engine=engine))
        routing = payload["config"]
        assert routing["summary_reasoning_effort"] == "high"
        assert routing["expansion_reasoning_effort"] == "low"
    finally:
        engine.shutdown()



def test_session_start_does_not_overwrite_update_model_with_stale_runtime_identity(engine):
    engine.update_model(
        model="deepseek-v4-flash",
        context_length=1_000_000,
        provider="opencode-go",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="minimax-m2.7",
        provider="minimax",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "deepseek-v4-flash"
    assert engine.provider == "opencode-go"
    assert engine.context_length == 1_000_000
    assert engine.threshold_tokens == int(1_000_000 * engine._config.context_threshold)



def test_session_start_does_not_overwrite_update_model_identity_when_context_length_matches(engine):
    engine.update_model(
        model="new-model-same-window",
        context_length=204_800,
        base_url="https://new.example/v1",
        api_key="new-key",
        provider="new-provider",
        api_mode="chat_completions",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="old-model-same-window",
        base_url="https://old.example/v1",
        api_key="old-key",
        provider="old-provider",
        api_mode="anthropic_messages",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "new-model-same-window"
    assert engine.base_url == "https://new.example/v1"
    assert engine.api_key == "new-key"
    assert engine.provider == "new-provider"
    assert engine.api_mode == "chat_completions"
    assert engine.context_length == 204_800
    assert engine.threshold_tokens == int(204_800 * engine._config.context_threshold)



def test_session_start_does_not_clear_or_repopulate_update_model_identity_when_optional_fields_are_empty(engine):
    engine.update_model(
        model="new-model-same-window",
        context_length=204_800,
        base_url="https://new.example/v1",
        api_key="new-key",
        provider="new-provider",
        api_mode="chat_completions",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="new-model-same-window",
        base_url="",
        api_key="",
        provider="new-provider",
        api_mode="chat_completions",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.base_url == "https://new.example/v1"
    assert engine.api_key == "new-key"

    engine.update_model(
        model="empty-endpoint-model",
        context_length=204_800,
        base_url="",
        api_key="",
        provider="new-provider",
        api_mode="chat_completions",
    )

    engine.on_session_start(
        "telegram:chat-1:session-3",
        platform="telegram",
        model="empty-endpoint-model",
        base_url="https://old.example/v1",
        api_key="old-key",
        provider="new-provider",
        api_mode="chat_completions",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.base_url == ""
    assert engine.api_key == ""



def test_session_start_can_initialize_context_length_without_update_model(engine):
    engine.model = ""
    engine.context_length = 0
    engine.threshold_tokens = 0
    engine._context_length_source = ""

    engine.on_session_start(
        "telegram:chat-1:session-1",
        platform="telegram",
        model="minimax-m2.7",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "minimax-m2.7"
    assert engine.context_length == 204_800
    assert engine.threshold_tokens == int(204_800 * engine._config.context_threshold)



def test_session_start_clears_previous_session_context_window_when_new_window_is_missing(engine):
    engine.on_session_start(
        "telegram:chat-1:session-1",
        platform="telegram",
        model="known-window-model",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.context_length == 204_800
    assert engine.threshold_tokens > 0

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="unknown-window-model",
        context_length=0,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "unknown-window-model"
    assert engine.context_length == 0
    assert engine.threshold_tokens == 0
    assert engine._context_length_source == "session_start"
    assert not engine.should_compress(1_000_000)
    assert not engine._critical_budget_pressure_reached(observed_tokens=1_000_000)



def test_missing_session_start_context_length_does_not_clear_authoritative_update_model_window(engine):
    engine.update_model(
        model="resolver-window-model",
        context_length=1_000_000,
        provider="resolver-provider",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="resolver-window-model",
        provider="resolver-provider",
        context_length=0,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "resolver-window-model"
    assert engine.provider == "resolver-provider"
    assert engine.context_length == 1_000_000
    assert engine.threshold_tokens == int(1_000_000 * engine._config.context_threshold)
    assert engine._context_length_source == "update_model"



def test_missing_session_start_context_length_preserves_update_model_window_with_blank_optional_fields(engine):
    engine.update_model(
        model="resolver-window-model",
        context_length=1_000_000,
        base_url="https://resolver.example/v1",
        api_key="resolver-key",
        provider="resolver-provider",
        api_mode="chat_completions",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="resolver-window-model",
        base_url="",
        api_key="",
        provider="resolver-provider",
        api_mode="chat_completions",
        context_length=0,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "resolver-window-model"
    assert engine.base_url == "https://resolver.example/v1"
    assert engine.api_key == "resolver-key"
    assert engine.provider == "resolver-provider"
    assert engine.api_mode == "chat_completions"
    assert engine.context_length == 1_000_000
    assert engine.threshold_tokens == int(1_000_000 * engine._config.context_threshold)
    assert engine._context_length_source == "update_model"



def test_missing_session_start_context_length_clears_stale_update_model_window_for_new_runtime(engine):
    engine.update_model(
        model="previous-resolver-model",
        context_length=1_000_000,
        provider="previous-provider",
    )
    engine.on_session_start(
        "telegram:chat-1:session-1",
        platform="telegram",
        model="previous-resolver-model",
        provider="previous-provider",
        context_length=1_000_000,
        conversation_id="telegram:chat-1",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="session-only-model",
        provider="session-provider",
        context_length=0,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "session-only-model"
    assert engine.provider == "session-provider"
    assert engine.context_length == 0
    assert engine.threshold_tokens == 0
    assert engine._context_length_source == "session_start"



def test_update_model_zero_window_ignores_stale_positive_session_metadata(engine):
    engine.on_session_start(
        "telegram:chat-1:session-1",
        platform="telegram",
        model="previous-window-model",
        provider="previous-provider",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )
    engine.update_model(
        model="unknown-window-model",
        context_length=0,
        provider="unknown-provider",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="previous-window-model",
        provider="previous-provider",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "unknown-window-model"
    assert engine.provider == "unknown-provider"
    assert engine.context_length == 0
    assert engine.threshold_tokens == 0
    assert engine._context_length_source == "update_model"



def test_update_model_zero_window_ignores_stale_zero_session_metadata(engine):
    engine.on_session_start(
        "telegram:chat-1:session-1",
        platform="telegram",
        model="previous-window-model",
        provider="previous-provider",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )
    engine.update_model(
        model="unknown-window-model",
        context_length=0,
        provider="unknown-provider",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="previous-window-model",
        provider="previous-provider",
        context_length=0,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "unknown-window-model"
    assert engine.provider == "unknown-provider"
    assert engine.context_length == 0
    assert engine.threshold_tokens == 0
    assert engine._context_length_source == "update_model"



def test_positive_session_start_context_length_replaces_consumed_update_model_window_for_new_runtime(engine):
    engine.update_model(
        model="previous-resolver-model",
        context_length=1_000_000,
        provider="previous-provider",
    )
    engine.on_session_start(
        "telegram:chat-1:session-1",
        platform="telegram",
        model="previous-resolver-model",
        provider="previous-provider",
        context_length=1_000_000,
        conversation_id="telegram:chat-1",
    )

    engine.on_session_start(
        "telegram:chat-1:session-2",
        platform="telegram",
        model="session-only-model",
        provider="session-provider",
        context_length=204_800,
        conversation_id="telegram:chat-1",
    )

    assert engine.model == "session-only-model"
    assert engine.provider == "session-provider"
    assert engine.context_length == 204_800
    assert engine.threshold_tokens == int(204_800 * engine._config.context_threshold)
    assert engine._context_length_source == "session_start"



def test_lcm_tool_status_forwards_filter_config_to_agent_surface(tmp_path, monkeypatch):
    from hermes_lcm import message_patterns as message_patterns_mod

    monkeypatch.setattr(message_patterns_mod, "_regex_engine", _FakeTimeoutRegexEngine)

    config = LCMConfig(
        database_path=str(tmp_path / "tool-status-filter-config.db"),
        ignore_session_patterns=["cron:*"],
        stateless_session_patterns=["debug:*"],
        ignore_message_patterns=["^Cronjob Response:"],
        ignore_session_patterns_source="env",
        stateless_session_patterns_source="env",
        ignore_message_patterns_source="env",
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("chat-1", platform="telegram", context_length=200000)
    engine._ingest_messages([{"role": "user", "content": "Cronjob Response: heartbeat"}])

    payload = json.loads(lcm_tools.lcm_status({}, engine=engine))

    assert payload["session_filters"] == {
        "ignored": False,
        "stateless": False,
        "ignore_session_patterns": ["cron:*"],
        "ignore_session_patterns_source": "env",
        "stateless_session_patterns": ["debug:*"],
        "stateless_session_patterns_source": "env",
        "ignore_message_patterns": ["^Cronjob Response:"],
        "ignore_message_patterns_source": "env",
        "ignored_message_count": 1,
        "side_channel_active": False,
    }



def test_lcm_tool_status_reports_runtime_identity_before_session_binding(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "unbound-tool-status.db"))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes-home"))

    payload = json.loads(lcm_tools.lcm_status({}, engine=engine))

    assert payload["error"] == "No active session"
    assert payload["runtime_identity"]["plugin_name"] == "hermes-lcm-x"
    assert payload["runtime_identity"]["session_bound"] is False
    assert payload["runtime_identity"]["database_path_source"] == "config.database_path"




def test_get_status_exposes_runtime_identity_for_loaded_plugin_tree(tmp_path):
    db_path = tmp_path / "identity.db"
    hermes_home = tmp_path / "hermes-home"
    config = LCMConfig(database_path=str(db_path))
    engine = LCMEngine(config=config, hermes_home=str(hermes_home))
    engine.on_session_start(
        "telegram:chat-1:session-1",
        platform="telegram",
        context_length=200000,
        conversation_id="telegram:chat-1",
    )

    status = engine.get_status()
    identity = status["runtime_identity"]
    repo_root = Path(__file__).resolve().parent.parent

    assert identity["engine"] == "lcm-x"
    assert identity["plugin_name"] == "hermes-lcm-x"
    assert identity["plugin_version"] == "0.27.0"
    assert Path(identity["plugin_path"]) == repo_root
    assert Path(identity["module_path"]).name == "engine.py"
    assert Path(identity["database_path"]) == db_path
    assert identity["database_path_source"] == "config.database_path"
    assert identity["hermes_home"] == str(hermes_home)
    assert identity["session_id"] == "telegram:chat-1:session-1"
    assert identity["session_platform"] == "telegram"
    assert identity["conversation_id"] == "telegram:chat-1"
    assert identity["lifecycle_current_session_id"] == "telegram:chat-1:session-1"
    assert identity["lifecycle_last_finalized_session_id"] == ""





def test_plugin_metadata_refreshes_when_manifest_changes(tmp_path, monkeypatch):
    import hermes_lcm.runtime_identity as identity_mod

    repo_root = Path(identity_mod.__file__).resolve().parent
    original = (repo_root / "plugin.yaml").read_text(encoding="utf-8")
    manifest = tmp_path / "plugin.yaml"
    manifest.write_text(original, encoding="utf-8")

    monkeypatch.setattr(identity_mod, "_PLUGIN_ROOT", tmp_path)
    monkeypatch.setattr(identity_mod, "_PLUGIN_METADATA", None)

    initial = identity_mod._plugin_metadata()
    assert initial["name"] == "hermes-lcm-x"
    assert initial["version"] == "0.27.0"

    updated = original.replace('version: "0.27.0"', 'version: "9.9.9-test"')
    if updated == original:
        updated = original.replace('version: 0.27.0', 'version: 9.9.9-test')
    assert updated != original

    manifest.write_text(updated, encoding="utf-8")
    refreshed = identity_mod._plugin_metadata()
    assert refreshed == {"name": "hermes-lcm-x", "version": "9.9.9-test"}

    manifest.unlink()
    fallback = identity_mod._plugin_metadata()
    assert fallback == {"name": "hermes-lcm-x", "version": "9.9.9-test"}



def test_plugin_metadata_defaults_when_manifest_missing_before_first_read(tmp_path, monkeypatch):
    import hermes_lcm.runtime_identity as identity_mod

    repo_root = Path(identity_mod.__file__).resolve().parent
    manifest = tmp_path / "plugin.yaml"
    manifest.write_text((repo_root / "plugin.yaml").read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(identity_mod, "_PLUGIN_ROOT", tmp_path)
    monkeypatch.setattr(identity_mod, "_PLUGIN_METADATA", None)

    manifest.unlink()
    metadata = identity_mod._plugin_metadata()
    assert metadata == {"name": "hermes-lcm-x", "version": "unknown"}



def test_lcm_doctor_json_includes_runtime_identity(engine):
    payload = json.loads(engine.handle_tool_call("lcm_doctor", {}))

    assert payload["runtime_identity"]["plugin_name"] == "hermes-lcm-x"
    assert payload["runtime_identity"]["plugin_version"] == "0.27.0"
    assert "plugin_git_commit" in payload["runtime_identity"]



def test_lcm_doctor_warns_on_extreme_summary_compression_ratios(engine):
    engine._dag.add_node(SummaryNode(
        session_id="test-session",
        depth=0,
        summary="tiny",
        token_count=100,
        source_token_count=180_000,
        source_ids=[],
        source_type="messages",
        created_at=1.0,
    ))

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {}))
    check = next(item for item in payload["checks"] if item["check"] == "summary_quality")

    assert check["status"] == "warn"
    assert check["detail"]["session_id"] == "test-session"
    assert check["detail"]["extreme_ratio_nodes"] == 1
    assert check["detail"]["tiny_large_source_nodes"] == 1
    assert check["detail"]["worst_nodes"][0]["node_id"] == 1
    assert check["detail"]["worst_nodes"][0]["compression_ratio"] == 1800.0



def test_lcm_doctor_summary_quality_ignores_other_sessions(engine):
    engine._dag.add_node(SummaryNode(
        session_id="other-session",
        depth=0,
        summary="tiny",
        token_count=100,
        source_token_count=180_000,
        source_ids=[],
        source_type="messages",
        created_at=1.0,
    ))
    engine._dag.add_node(SummaryNode(
        session_id="test-session",
        depth=0,
        summary="healthy enough",
        token_count=1_000,
        source_token_count=20_000,
        source_ids=[],
        source_type="messages",
        created_at=2.0,
    ))

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {}))
    check = next(item for item in payload["checks"] if item["check"] == "summary_quality")

    assert check["status"] == "pass"
    assert check["detail"]["session_id"] == "test-session"
    assert check["detail"]["total_nodes"] == 1
    assert check["detail"]["extreme_ratio_nodes"] == 0
    assert check["detail"]["tiny_large_source_nodes"] == 0
    assert check["detail"]["worst_nodes"][0]["session_id"] == "test-session"


def test_lcm_doctor_summary_quality_flags_zero_token_large_source(engine):
    engine._dag.add_node(SummaryNode(
        session_id="test-session",
        depth=0,
        summary="",
        token_count=0,
        source_token_count=180_000,
        source_ids=[],
        source_type="messages",
        created_at=1.0,
    ))

    payload = json.loads(engine.handle_tool_call("lcm_doctor", {}))
    check = next(item for item in payload["checks"] if item["check"] == "summary_quality")

    assert check["status"] == "warn"
    assert check["detail"]["tiny_large_source_nodes"] == 1
    assert check["detail"]["extreme_ratio_nodes"] == 0
    assert check["detail"]["worst_nodes"][0]["token_count"] == 0
    assert check["detail"]["worst_nodes"][0]["compression_ratio"] is None



class TestConfigCleanup:
    """Tests for issue #2c follow-up — expansion path is now separate from summary-only config."""

    def test_has_expansion_model(self):
        config = LCMConfig()
        assert hasattr(config, "expansion_model")
        assert config.expansion_model == ""

    def test_has_summary_timeout_ms(self):
        config = LCMConfig()
        assert hasattr(config, "summary_timeout_ms")
        assert config.summary_timeout_ms == 120_000

    def test_has_expansion_timeout_ms(self):
        config = LCMConfig()
        assert hasattr(config, "expansion_timeout_ms")
        assert config.expansion_timeout_ms == 120_000



class TestEngineTools:

    def test_handle_status_returns_session_overview(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "hello world"})
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="greeting summary",
                token_count=10,
                source_token_count=50,
                source_ids=[1],
                source_type="messages",
                created_at=0,
            )
        )
        engine.compression_count = 3
        engine.context_length = 128000
        engine.threshold_tokens = 96000
        engine.last_prompt_tokens = 40000

        result = json.loads(engine.handle_tool_call("lcm_status", {}))

        assert result["session_id"] == "test-session"
        assert result["compression_count"] == 3
        assert result["context_length"] == 128000
        assert result["store"]["messages"] == 1
        assert result["dag"]["total_nodes"] == 1
        assert "d0" in result["dag"]["depths"]
        assert result["config"]["fresh_tail_count"] == engine._config.fresh_tail_count
        assert result["session_filters"]["ignored"] is False
        assert result["source_lineage"]["messages_total"] == 1
        assert result["source_lineage"]["normalized_unknown_messages"] == 1
        assert result["source_lineage"]["legacy_blank_source_messages"] == 0

    def test_handle_status_exposes_structured_preset_suggestion(self, engine, monkeypatch):
        monkeypatch.setenv("LCM_FRESH_TAIL_COUNT", "abc")
        engine.context_length = 272000

        result = json.loads(engine.handle_tool_call("lcm_status", {}))
        preset = result["preset_suggestion"]

        assert preset["read_only"] is True
        assert preset["runtime_mutation"] is False
        assert preset["suggested_preset"]["name"] == "codex_gpt_long_context"
        assert preset["suggested_preset"]["family"] == "GPT/Codex long-context"
        assert preset["match_confidence"] == "context-only"
        assert preset["provenance"]["benchmark_version"] == "2"
        assert preset["invalid_overrides"]["fresh_tail_count"] == {
            "env": "LCM_FRESH_TAIL_COUNT",
            "value": "abc",
            "runtime_value": engine._config.fresh_tail_count,
            "preset_value": 24,
        }
        assert {
            "field": "fresh_tail_count",
            "env": "LCM_FRESH_TAIL_COUNT",
            "action": "replace_invalid",
            "invalid_value": "abc",
            "current_value": engine._config.fresh_tail_count,
            "preset_value": 24,
        } in preset["dry_run_delta"]

    def test_handle_status_shows_compression_ratio(self, engine):
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="short",
                token_count=10,
                source_token_count=100,
                source_ids=[],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert result["dag"]["compression_ratio"] == "10.0:1"

    def test_handle_doctor_returns_healthy(self, engine):
        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        check_names = [c["check"] for c in result["checks"]]
        assert "database_integrity" in check_names
        assert "messages_fts_integrity" in check_names
        assert "nodes_fts_integrity" in check_names
        assert "fts_index_sync" in check_names
        assert "orphaned_dag_nodes" in check_names
        assert "config_validation" in check_names
        assert all(c["status"] == "pass" for c in result["checks"])

    def test_handle_doctor_reports_fts_integrity_failures_separately(self, engine, monkeypatch):
        def fake_fts_integrity(_conn, spec):
            if spec.table_name == "nodes_fts":
                return {
                    "status": "fail",
                    "detail": "malformed inverted index for FTS5 table main.nodes_fts",
                }
            return {"status": "pass", "detail": "ok"}

        monkeypatch.setattr(lcm_tools, "check_external_content_fts_integrity", fake_fts_integrity)

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        checks = {check["check"]: check for check in result["checks"]}
        assert result["overall"] == "unhealthy"
        assert checks["database_integrity"]["status"] == "pass"
        assert checks["messages_fts_integrity"]["status"] == "pass"
        assert checks["nodes_fts_integrity"] == {
            "check": "nodes_fts_integrity",
            "status": "fail",
            "detail": "malformed inverted index for FTS5 table main.nodes_fts",
        }

    def test_handle_doctor_treats_legacy_blank_source_rows_as_healthy(self, engine):
        for source in (None, "", "   ", "\t\n"):
            engine._store._conn.execute(
                """INSERT INTO messages
                   (session_id, source, role, content, tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("legacy-session", source, "user", "legacy blank source", None, None, None, 1.0, 5, 0),
            )
        engine._store._conn.commit()

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        lineage_check = next(c for c in result["checks"] if c["check"] == "source_lineage_hygiene")
        assert lineage_check["status"] == "pass"
        assert lineage_check["detail"]["legacy_blank_source_messages"] == 4
        assert lineage_check["detail"]["effective_unknown_messages"] == 4

    def test_handle_doctor_reports_lifecycle_fragmentation_without_mutating(self, engine, tmp_path):
        engine._hermes_home = str(tmp_path / "hermes_home")
        state_db = tmp_path / "hermes_home" / "state.db"
        state_db.parent.mkdir(parents=True, exist_ok=True)
        state_conn = sqlite3.connect(state_db)
        state_conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY);
            INSERT INTO sessions(id) VALUES ('current-with-message');
            INSERT INTO sessions(id) VALUES ('state-only');
            """
        )
        state_conn.commit()
        state_conn.close()
        engine._store.append("current-with-message", {"role": "user", "content": "covered"}, source="cli")
        engine._dag.add_node(
            SummaryNode(
                session_id="node-missing-in-state",
                depth=0,
                summary="summary-only coverage",
                token_count=5,
                source_token_count=5,
                source_ids=[],
                source_type="messages",
                created_at=1.0,
            )
        )
        engine._lifecycle._conn.execute(
            """INSERT INTO lcm_lifecycle_state
               (conversation_id, current_session_id, last_finalized_session_id, current_frontier_store_id, last_finalized_frontier_store_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("conv-current", "current-with-message", "node-missing-in-state", 0, 0, 1.0),
        )
        engine._lifecycle._conn.execute(
            """INSERT INTO lcm_lifecycle_state
               (conversation_id, current_session_id, last_finalized_session_id, current_frontier_store_id, last_finalized_frontier_store_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("conv-stale", "missing-current", "missing-final", 0, 0, 1.0),
        )
        engine._lifecycle._conn.commit()

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        lifecycle_check = next(c for c in result["checks"] if c["check"] == "lifecycle_fragmentation")
        assert lifecycle_check["status"] == "warn"
        assert lifecycle_check["detail"]["lifecycle_rows"] == 2
        assert lifecycle_check["detail"]["empty_lifecycle_rows"] == 1
        assert lifecycle_check["detail"]["lifecycle_current_missing_in_lcm_any"] == 1
        assert lifecycle_check["detail"]["lifecycle_current_missing_in_state"] == 1
        assert lifecycle_check["detail"]["lcm_node_sessions_missing_in_state"] == 1
        assert lifecycle_check["detail"]["state_sessions_missing_in_lcm_any"] == 1
        assert lifecycle_check["detail"]["read_only"] is True
        assert engine._lifecycle.row_count() == 2

    def test_handle_doctor_keeps_retained_history_lifecycle_drift_healthy(self, engine, tmp_path):
        engine._hermes_home = str(tmp_path / "hermes_home")
        state_db = tmp_path / "hermes_home" / "state.db"
        state_db.parent.mkdir(parents=True, exist_ok=True)
        state_conn = sqlite3.connect(state_db)
        state_conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY);
            INSERT INTO sessions(id) VALUES ('state-only');
            """
        )
        state_conn.commit()
        state_conn.close()
        engine._store.append("live-current", {"role": "user", "content": "covered"}, source="cli")
        engine._store.append("live-finalized", {"role": "user", "content": "covered finalized"}, source="cli")
        engine._store.append("message-history", {"role": "user", "content": "retained"}, source="cli")
        engine._dag.add_node(
            SummaryNode(
                session_id="node-history",
                depth=0,
                summary="retained summary",
                token_count=5,
                source_token_count=5,
                source_ids=[],
                source_type="messages",
                created_at=1.0,
            )
        )
        engine._lifecycle._conn.execute(
            """INSERT INTO lcm_lifecycle_state
               (conversation_id, current_session_id, last_finalized_session_id, current_frontier_store_id, last_finalized_frontier_store_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("conv-retained", "live-current", "stale-finalized", 0, 0, 1.0),
        )
        engine._lifecycle._conn.execute(
            """INSERT INTO lcm_lifecycle_state
               (conversation_id, current_session_id, last_finalized_session_id, current_frontier_store_id, last_finalized_frontier_store_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("conv-retained-current", "stale-current", "live-finalized", 0, 0, 1.0),
        )
        engine._lifecycle._conn.commit()

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        lifecycle_check = next(c for c in result["checks"] if c["check"] == "lifecycle_fragmentation")
        assert lifecycle_check["status"] == "pass"
        detail = lifecycle_check["detail"]
        assert detail["empty_lifecycle_rows"] == 0
        assert detail["lifecycle_current_missing_in_lcm_any"] == 1
        assert detail["lifecycle_last_finalized_missing_in_lcm_any"] == 1
        assert detail["lcm_message_sessions_missing_in_state"] == 3
        assert detail["lcm_node_sessions_missing_in_state"] == 1
        assert detail["state_sessions_missing_in_lcm_any"] == 1
        assert detail["classification"]["status"] == "warn"
        assert any(
            item["name"] == "stale_lifecycle_current" and item["severity"] == "warn"
            for item in detail["classification"]["categories"]
        )
        assert any(
            item["name"] == "stale_lifecycle_finalized" and item["severity"] == "warn"
            for item in detail["classification"]["categories"]
        )
        assert not result["guidance"]

    def test_handle_doctor_warns_when_existing_state_db_is_unreadable(self, engine, tmp_path):
        engine._hermes_home = str(tmp_path / "hermes_home")
        state_db = tmp_path / "hermes_home" / "state.db"
        state_db.parent.mkdir(parents=True, exist_ok=True)
        state_db.write_text("not sqlite")

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "warnings"
        lifecycle_check = next(c for c in result["checks"] if c["check"] == "lifecycle_fragmentation")
        assert lifecycle_check["status"] == "warn"
        assert lifecycle_check["detail"]["state_db_checked"] is True
        assert lifecycle_check["detail"]["state_db_error"]
        assert lifecycle_check["detail"]["read_only"] is True

    def test_handle_doctor_warns_on_message_session_without_lifecycle_current(self, engine):
        engine.on_session_start("current-session", platform="cli", context_length=200000)
        engine._store.append("current-session", {"role": "user", "content": "covered"}, source="cli")
        engine._store.append("message-only-session", {"role": "user", "content": "missing lifecycle"}, source="cli")

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        lifecycle_check = next(c for c in result["checks"] if c["check"] == "lifecycle_fragmentation")
        assert lifecycle_check["status"] == "pass"
        assert lifecycle_check["detail"]["message_sessions_without_lifecycle_current"] == 1
        assert lifecycle_check["detail"]["message_sessions_without_lifecycle_reference"] == 1
        assert lifecycle_check["detail"]["empty_lifecycle_rows"] == 0
        assert lifecycle_check["detail"]["read_only"] is True

    def test_handle_doctor_does_not_warn_on_last_finalized_message_session(self, engine):
        engine.on_session_start(
            "current-session",
            platform="cli",
            context_length=200000,
            conversation_id="conversation",
        )
        engine._store.append("previous-session", {"role": "user", "content": "previous"}, source="cli")
        engine._store.append("current-session", {"role": "user", "content": "current"}, source="cli")
        engine._lifecycle.record_rollover(
            "conversation",
            old_session_id="previous-session",
            new_session_id="current-session",
        )

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        lifecycle_check = next(c for c in result["checks"] if c["check"] == "lifecycle_fragmentation")
        assert lifecycle_check["status"] == "pass"
        assert lifecycle_check["detail"]["message_sessions_without_lifecycle_current"] == 1
        assert lifecycle_check["detail"]["message_sessions_without_lifecycle_reference"] == 0
        assert lifecycle_check["detail"]["read_only"] is True

    def test_handle_doctor_warns_on_node_session_without_lifecycle_reference(self, engine):
        engine.on_session_start("current-session", platform="cli", context_length=200000)
        engine._store.append("current-session", {"role": "user", "content": "covered"}, source="cli")
        engine._dag.add_node(
            SummaryNode(
                session_id="node-only-session",
                depth=0,
                summary="missing lifecycle reference",
                token_count=5,
                source_token_count=5,
                source_ids=[],
                source_type="messages",
                created_at=1.0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "healthy"
        lifecycle_check = next(c for c in result["checks"] if c["check"] == "lifecycle_fragmentation")
        assert lifecycle_check["status"] == "pass"
        assert lifecycle_check["detail"]["node_sessions_without_lifecycle_reference"] == 1
        assert lifecycle_check["detail"]["empty_lifecycle_rows"] == 0
        assert lifecycle_check["detail"]["read_only"] is True

    def test_handle_doctor_warns_on_bad_config(self, tmp_path):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm_doctor.db"),
            fresh_tail_count=1,
            context_threshold=0.99,
            condensation_fanin=1,
        )
        engine = LCMEngine(config=config)
        engine._session_id = "test-session"

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert result["overall"] == "warnings"
        config_check = next(c for c in result["checks"] if c["check"] == "config_validation")
        assert config_check["status"] == "warn"
        assert len(config_check["detail"]) == 3  # three warnings

    def test_handle_doctor_detects_orphaned_nodes(self, engine):
        # Add a node referencing a store_id that doesn't exist
        engine._dag.add_node(
            SummaryNode(
                session_id="test-session",
                depth=0,
                summary="orphan",
                token_count=10,
                source_token_count=50,
                source_ids=[99999],
                source_type="messages",
                created_at=0,
            )
        )

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        orphan_check = next(c for c in result["checks"] if c["check"] == "orphaned_dag_nodes")
        assert orphan_check["status"] == "warn"

    def test_handle_doctor_fts_sync_is_session_scoped(self, engine):
        engine._store.append("test-session", {"role": "user", "content": "session A"})
        engine._store.append("other-session", {"role": "user", "content": "session B 1"})
        engine._store.append("other-session", {"role": "assistant", "content": "session B 2"})

        result = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        fts_check = next(c for c in result["checks"] if c["check"] == "fts_index_sync")
        assert fts_check["status"] == "pass"
        assert fts_check["detail"] == "1 session FTS rows, 1 session messages"
