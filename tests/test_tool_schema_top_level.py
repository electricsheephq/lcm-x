"""Tool parameter schemas must be accepted by strict provider routes (#550).

Anthropic's tool API rejects a tool whose ``input_schema`` has ``allOf``,
``anyOf``, ``oneOf``, ``enum`` or ``not`` at the top level, and one rejected
tool fails the whole request. Conditional rules belong in property
descriptions and in the handler, not in top-level combinators.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.store import MessageStore
from hermes_lcm.tools import lcm_compile_evidence

FORBIDDEN_TOP_LEVEL_KEYS = ("allOf", "anyOf", "oneOf", "enum", "not")


def _top_level_violations(schemas):
    violations = []
    for schema in schemas:
        parameters = schema.get("parameters") or {}
        for key in FORBIDDEN_TOP_LEVEL_KEYS:
            if key in parameters:
                violations.append(f"{schema.get('name')}:{key}")
    return violations


@pytest.fixture
def exposed_tool_schemas(tmp_path, monkeypatch):
    """Both exposure paths: plugin-registry tools and engine.get_tool_schemas()."""
    for name in (
        "HERMES_PROFILE",
        "LCM_HERMES_BASE_DIR",
        "LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH",
        "LCM_DISABLED_TOOLS",
    ):
        monkeypatch.delenv(name, raising=False)
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("LCM_DATABASE_PATH", str(hermes_home / "lcm.db"))

    repo_root = Path(__file__).resolve().parent.parent
    module_name = "hermes_lcm_schema_top_level"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    registered = []

    class _Ctx:
        context_engine_tool_handlers_receive_messages = True

        def __init__(self):
            self.engine = None

        def register_context_engine(self, engine):
            self.engine = engine

        def register_tool(self, name, toolset, schema, handler, description="", emoji=""):
            registered.append(schema)

    ctx = _Ctx()
    module.register(ctx)
    assert ctx.engine is not None
    return registered, ctx.engine.get_tool_schemas()


def test_checker_flags_a_top_level_all_of():
    synthetic = {
        "name": "lcm_synthetic",
        "parameters": {"type": "object", "properties": {}, "allOf": [{}]},
    }
    assert _top_level_violations([synthetic]) == ["lcm_synthetic:allOf"]


def test_no_exposed_tool_schema_has_a_top_level_combinator(exposed_tool_schemas):
    registered, engine_schemas = exposed_tool_schemas
    registered_names = {schema["name"] for schema in registered}
    engine_names = {schema["name"] for schema in engine_schemas}
    assert "lcm_compile_evidence" in registered_names
    assert registered_names == engine_names
    assert _top_level_violations(registered) == []
    assert _top_level_violations(engine_schemas) == []


@pytest.mark.parametrize("mode", ["proposal", None])
def test_missing_proposal_in_proposal_mode_is_selector_schema_invalid(tmp_path, mode):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    store = MessageStore(config.database_path, ingest_protection_config=config)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _assertions=None,
        _session_occurrence_dates={},
    )
    content = "You need 15 points to redeem the reward."
    store_id = store.append("session-a", {"role": "user", "content": content})
    args = {
        "question": "How many points do I need to redeem the reward?",
        "baseline_refs": [
            {"exact_ref": f"lcm:{store_id}:0-{len(content)}", "quote": content}
        ],
    }
    if mode is not None:
        args["mode"] = mode
    try:
        payload = json.loads(lcm_compile_evidence(args, engine=engine))
    finally:
        store.close()

    assert payload["reason_code"] == "selector_schema_invalid"
