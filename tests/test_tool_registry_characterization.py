"""Freeze tool exposure bytes, registration metadata, and late-bound dispatch."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

import hermes_lcm.tools as tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_BYTES = (Path(__file__).with_name("tool_registry_schema_bytes.json")).read_bytes().rstrip(b"\n")
FROZEN_SCHEMAS = json.loads(SCHEMA_BYTES)
NAMES = [schema["name"] for schema in FROZEN_SCHEMAS]
EMOJIS = ["🔍", "🧠", "🧾", "🧮", "🧷", "📦", "🧭", "🕒", "📋", "📊", "🔎", "❓", "💚", "🧭", "🏥"]


@pytest.fixture(scope="module")
def plugin():
    name = "lcm_tool_registry_characterization"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                sys.modules.pop(key)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("LCM_", "HERMES_")) and name != "HERMES_DISABLE_LAZY_INSTALLS":
            monkeypatch.delenv(name)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")


@pytest.mark.parametrize("disabled_count", [0, 1, 3, 15])
@pytest.mark.parametrize("flags", [None, (False, False), (True, False), (False, True), (True, True)])
def test_exposure_bytes_and_registration_sequence(plugin, monkeypatch, flags, disabled_count):
    if flags is not None:
        monkeypatch.setenv("LCM_ASSERTIONS_ENABLED", str(flags[0]).lower())
        monkeypatch.setenv("LCM_ADAPTIVE_RETRIEVAL_ENABLED", str(flags[1]).lower())
    if disabled_count:
        monkeypatch.setenv("LCM_DISABLED_TOOLS", ",".join(NAMES[:disabled_count]))
    assertions, adaptive = flags or (False, False)
    omitted = set(NAMES[:disabled_count])
    if not assertions:
        omitted.add("lcm_query_state")
    if not adaptive:
        omitted.add("lcm_retrieve")
    expected = [schema for schema in FROZEN_SCHEMAS if schema["name"] not in omitted]
    calls = []

    class Context:
        context_engine_tool_handlers_receive_messages = True
        engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            assert callable(handler)
            calls.append((name, toolset, schema, description, emoji))

    ctx = Context()
    try:
        plugin.register(ctx)
        actual = ctx.engine.get_tool_schemas()
        assert json.dumps(actual).encode() == json.dumps(expected).encode()
        if not omitted:
            assert json.dumps(actual).encode() == SCHEMA_BYTES
        schema_module = sys.modules[plugin.__name__ + ".schemas"]
        expected_calls = [
            (name, "context_engine", getattr(schema_module, name.upper()),
             schema["description"], emoji)
            for name, schema, emoji in zip(NAMES, FROZEN_SCHEMAS, EMOJIS)
            if name not in omitted
        ]
        assert calls == expected_calls
        for call, expected_call, engine_schema in zip(calls, expected_calls, actual):
            assert call[2] is expected_call[2] is engine_schema
        assert ctx.engine._omitted_tool_names() == omitted
    finally:
        if ctx.engine is not None:
            ctx.engine.shutdown()


@pytest.mark.parametrize("enabled", [False, True])
def test_dispatch_resolves_every_handler_after_construction(tmp_path, monkeypatch, enabled):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "dispatch.db"),
        assertions_enabled=enabled, adaptive_retrieval_enabled=enabled,
    ))
    calls = []

    def recorder(name):
        def handler(args, *, engine):
            calls.append((name, args, engine))
            return "handled:" + name
        return handler

    try:
        for name in NAMES:
            monkeypatch.setattr(tools, name, recorder(name))
        for name in NAMES:
            args = {"sentinel": name}
            assert engine.handle_tool_call(name, args) == "handled:" + name
            assert calls[-1] == (name, args, engine)
        monkeypatch.setattr(tools, "lcm_grep", recorder("rebound"))
        assert engine.handle_tool_call("lcm_grep", {}) == "handled:rebound"
        assert engine.handle_tool_call("lcm_missing", {}) == '{"error": "Unknown LCM tool: lcm_missing"}'
        monkeypatch.setenv("LCM_DISABLED_TOOLS", "lcm_grep,lcm_missing")
        engine._session_id = "session"
        ingested = []
        monkeypatch.setattr(engine, "_ingest_messages", lambda messages: ingested.extend(messages))
        before = list(calls)
        for name in ("lcm_grep", "lcm_missing"):
            assert engine.handle_tool_call(name, {}, messages=[{"role": "user", "content": "x"}]) == (
                '{"error": "LCM tool ' + name + ' is disabled via LCM_DISABLED_TOOLS"}'
            )
        assert calls == before
        assert ingested == []
    finally:
        engine.shutdown()
