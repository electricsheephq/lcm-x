"""Tests for host capability detection before registering lcm_* tools."""

import builtins
import importlib.util
import logging
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_host_capability_storage(tmp_path, monkeypatch):
    """Keep every plugin registration on a per-test SQLite database."""
    hermes_home = tmp_path / "hermes-home"
    database_path = hermes_home / "lcm.db"
    for name in (
        "HERMES_PROFILE",
        "LCM_HERMES_BASE_DIR",
        "LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(database_path))
    return database_path


EXPECTED_LCM_TOOLS = {
    "lcm_grep",
    "lcm_recall",
    "lcm_query_state",
    "lcm_compute",
    "lcm_compile_evidence",
    "lcm_evidence_pack",
    "lcm_retrieve",
    "lcm_recent",
    "lcm_load_session",
    "lcm_describe",
    "lcm_expand",
    "lcm_expand_query",
    "lcm_status",
    "lcm_inspect",
    "lcm_doctor",
}


def _load_plugin_module(name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        name, str(repo_root / "__init__.py"), submodule_search_locations=[str(repo_root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestHostCapabilityDetection:
    """Verify explicit host capability detection for registered lcm_* tools."""

    def test_returns_false_when_ctx_lacks_capability(self):
        module = _load_plugin_module("hermes_lcm_cap_no_attr")

        class _Ctx:
            pass

        assert module._host_forwards_registered_tool_messages(_Ctx()) is False

    def test_returns_false_when_capability_is_false(self):
        module = _load_plugin_module("hermes_lcm_cap_false")

        class _Ctx:
            context_engine_tool_handlers_receive_messages = False

        assert module._host_forwards_registered_tool_messages(_Ctx()) is False

    def test_returns_true_when_capability_is_true(self):
        module = _load_plugin_module("hermes_lcm_cap_true")

        class _Ctx:
            context_engine_tool_handlers_receive_messages = True

        assert module._host_forwards_registered_tool_messages(_Ctx()) is True

    def test_supports_callable_capability(self):
        module = _load_plugin_module("hermes_lcm_cap_callable")

        class _Ctx:
            def context_engine_tool_handlers_receive_messages(self):
                return True

        assert module._host_forwards_registered_tool_messages(_Ctx()) is True

    def test_callable_capability_failure_fails_closed(self):
        module = _load_plugin_module("hermes_lcm_cap_callable_raises")

        class _Ctx:
            def context_engine_tool_handlers_receive_messages(self):
                raise RuntimeError("host capability unavailable")

        assert module._host_forwards_registered_tool_messages(_Ctx()) is False


class TestRegistrationGating:
    """Verify register() skips ctx.register_tool unless messages forwarding is explicit."""

    def test_registration_uses_per_test_database(self, _isolate_host_capability_storage):
        module = _load_plugin_module("hermes_lcm_gating_storage")

        class _Ctx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

        ctx = _Ctx()
        module.register(ctx)
        try:
            identity = ctx.engine.get_status()["runtime_identity"]
            assert Path(identity["database_path"]) == _isolate_host_capability_storage
            assert identity["database_path_source"] == "config.database_path"
        finally:
            ctx.engine.shutdown()

    def test_skips_register_tool_without_explicit_message_forwarding(self):
        module = _load_plugin_module("hermes_lcm_gating_skip")
        registered_tools = []

        class _CtxNoForwarding:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                registered_tools.append(name)

        ctx = _CtxNoForwarding()
        module.register(ctx)

        assert ctx.engine is not None
        assert ctx.engine.name == "lcm"
        assert registered_tools == []
        assert EXPECTED_LCM_TOOLS.issubset(
            {schema["name"] for schema in ctx.engine.get_tool_schemas()}
        )

    def test_registers_tools_when_host_explicitly_supports_message_forwarding(self):
        module = _load_plugin_module("hermes_lcm_gating_register")
        registered_tools = []

        class _CtxWithForwarding:
            context_engine_tool_handlers_receive_messages = True

            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                registered_tools.append(name)

        ctx = _CtxWithForwarding()
        module.register(ctx)

        assert ctx.engine is not None
        assert set(registered_tools) == EXPECTED_LCM_TOOLS

    def test_existing_context_engine_path_still_loads_without_register_tool(self):
        module = _load_plugin_module("hermes_lcm_gating_no_register_tool")

        class _Ctx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

        ctx = _Ctx()
        module.register(ctx)
        assert ctx.engine is not None
        assert ctx.engine.name == "lcm"

    def test_logs_hermes_home_import_failure_and_uses_env_fallback(
        self, monkeypatch, tmp_path, caplog
    ):
        module = _load_plugin_module("hermes_lcm_home_import_failure")
        fallback_home = tmp_path / "hermes-home"
        monkeypatch.setenv("HERMES_HOME", str(fallback_home))
        original_import = builtins.__import__

        def _fail_hermes_cli_config(name, *args, **kwargs):
            if name == "hermes_cli.config":
                raise ImportError("simulated hermes_cli.config failure")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail_hermes_cli_config)

        class _Ctx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

        with caplog.at_level(logging.WARNING, logger=module.logger.name):
            ctx = _Ctx()
            module.register(ctx)

        assert ctx.engine is not None
        assert ctx.engine._hermes_home == str(fallback_home)
        assert any(
            record.levelno == logging.WARNING
            and "could not import get_hermes_home" in record.message
            and "simulated hermes_cli.config failure" in record.message
            for record in caplog.records
        )


class TestHermesAgentRegression:
    """Regression: Hermes Agent-shaped hosts must not shadow native LCM routing."""

    def test_hermes_agent_shaped_host_uses_context_engine_path(self):
        module = _load_plugin_module("hermes_lcm_hermes_agent_regression")
        registered_via_tool = []
        registered_via_engine = []

        class _HermesAgentCtx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine
                registered_via_engine.extend(
                    s["name"] for s in engine.get_tool_schemas()
                )

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                registered_via_tool.append(name)

        ctx = _HermesAgentCtx()
        module.register(ctx)

        assert ctx.engine is not None
        assert registered_via_tool == []
        assert set(registered_via_engine) == EXPECTED_LCM_TOOLS

    def test_messages_forwarded_through_context_engine_path(self):
        module = _load_plugin_module("hermes_lcm_messages_forward_regression")

        class _HermesAgentCtx:
            def __init__(self):
                self.engine = None

            def register_context_engine(self, engine):
                self.engine = engine

            def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
                raise AssertionError("Hermes Agent-shaped host must not register lcm_* tools")

        ctx = _HermesAgentCtx()
        module.register(ctx)
        assert ctx.engine is not None

        test_messages = [{"role": "user", "content": "test context"}]
        result = ctx.engine.handle_tool_call(
            "lcm_status", {}, messages=test_messages
        )

        assert isinstance(result, str)
        assert len(result) > 0
