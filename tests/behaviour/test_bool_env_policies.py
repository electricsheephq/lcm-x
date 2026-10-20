"""Legacy boolean policies, including readers loaded outside a package."""

import importlib.util
from pathlib import Path

import pytest

from hermes_lcm import config, db_bootstrap, identity_anchor
from hermes_lcm.engine import LCMEngine
from bench.instruments.reliability import plugin_tree


ROOT = Path(__file__).resolve().parents[2]


def load_plugin(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_boolean_edges(monkeypatch):
    """Pin whitespace, non-spellings, AST extraction and standalone loading."""
    plugins = [load_plugin(name) for name in ("bool_env_plugin", "hermes_lcm.bool_env_plugin")]
    for raw in (" \t\n", "t", "f", "enabled", "disabled", "2", "  OFF\n", " Yes\t"):
        on = raw == " Yes\t"
        off = raw == "  OFF\n"
        for default in (False, True):
            monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", raw)
            assert config._parse_bool_env("LCM_ENABLE_SLASH_COMMAND", default) is (
                True if on else False if off else default
            )
            for plugin in plugins:
                assert plugin._env_flag_enabled("LCM_ENABLE_SLASH_COMMAND", default) is on
        for key, reader in (
            ("LCM_IDENTITY_ANCHOR", identity_anchor.identity_anchor_enabled),
            ("LCM_FTS_INTEGRITY_BACKGROUND", db_bootstrap._background_integrity_enabled),
        ):
            monkeypatch.setenv(key, raw)
            assert reader() is not off
        assert plugin_tree.parse_bool(ROOT, "LCM_NATIVE_RECOVERY", raw)[0] is on
        monkeypatch.setenv("LCM_DISABLED_TOOLS", raw)
        expected = {raw.strip()} if raw.strip() else set()
        for plugin in plugins:
            assert plugin._disabled_tool_names() == expected
        assert LCMEngine._disabled_tool_names() == expected

    # This standalone live-provider authorization guard is deliberately exact,
    # and rejects before loading any plugin modules or constructing a provider.
    spec = importlib.util.spec_from_file_location("bool_env_eval", ROOT / "scripts/eval_retrieval_recall.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *a, **kw: pytest.fail("loaded plugin"))

    class Allowed(Exception):
        pass

    def reached_config():
        raise Allowed

    monkeypatch.setattr(config.LCMConfig, "from_env", reached_config)
    for raw in ("", " ", "true", "TRUE", "yes", "on", " 1 ", "garbage", "0"):
        monkeypatch.setenv("LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER", raw)
        with pytest.raises(RuntimeError, match="requires LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER=1"):
            module._load_live_provider()
    monkeypatch.setenv("LCM_RECALL_EVAL_ALLOW_LIVE_PROVIDER", "1")
    with pytest.raises(Allowed):
        module._load_live_provider()


# Frozen key -> (unknown/blank policy, recognized true tokens, recognized false
# tokens). None means keep the supplied default, including the privacy default.
TRUE_VALUES = ("1", "true", "yes", "on")
FALSE_VALUES = ("0", "false", "no", "off")
BOOLEAN_POLICIES = {
    **dict.fromkeys("""
        LCM_FRESH_TAIL_PRESSURE_YIELD_ENABLED LCM_DYNAMIC_LEAF_CHUNK_ENABLED
        LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED LCM_DEFERRED_MAINTENANCE_ENABLED
        LCM_THRESHOLD_FULL_SWEEP_ENABLED LCM_EXTRACTION_ENABLED LCM_ASSERTIONS_ENABLED
        LCM_QUERY_VIEWS_ENABLED LCM_ADAPTIVE_RETRIEVAL_ENABLED LCM_ASSERTION_EXTRACTION_ENABLED
        LCM_SENSITIVE_PATTERNS_ENABLED LCM_EMBEDDING_PRIVACY_ENABLED
        LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED
        LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED LCM_NATIVE_RECOVERY LCM_SURVIVAL_FIT
        LCM_EMBEDDINGS_ENABLED LCM_RERANK_ENABLED LCM_RECALL_REFERENCE_STRICT LCM_RECALL_FTS_ANCHOR
        LCM_PROACTIVE_RECALL_ENABLED LCM_PREANSWER_EVIDENCE_ENABLED LCM_SELECTIVE_COMPILER_ENABLED
        LCM_EMBEDDING_BINARY_PRESCREEN LCM_DOCTOR_CLEAN_APPLY_ENABLED LCM_EMPTY_LIFECYCLE_GC_ENABLED
        LCM_TEMPORAL_ROLLUPS_ENABLED LCM_BACKGROUND_LEAF_PREP_ENABLED
    """.split(), (None, TRUE_VALUES, FALSE_VALUES)),
    "LCM_ENABLE_SLASH_COMMAND": (False, TRUE_VALUES, ()),
    "LCM_IDENTITY_ANCHOR": (True, (), FALSE_VALUES),
    "LCM_FTS_INTEGRITY_BACKGROUND": (True, (), FALSE_VALUES),
}

# Raw readers are scalar/path/credential readers, not boolean policy escapes.
# A new inline env reader must be classified rather than bypassing the table.
RAW_READERS = {
    "__init__.py": {"_session_context_value", "register"},
    "command.py": {"_env_float", "_embedding_backfill_batch_size", "_preset_suggest_text"},
    "config.py": {
        "_parse_int_env", "_parse_float_env", "_parse_bool_env", "_parse_str_env",
        "_disabled_tool_names", "host_message_uid_mode", "_parse_int_env_with_source",
        "_parse_float_env_with_source", "_hermes_config_path", "_lcm_config_string_with_source",
        "_warn_env_range", "from_env", "containment_base_dir",
    },
    "db_bootstrap.py": {"_integrity_check_interval_hours"},
    "embedding_provider.py": {"probe_provider_availability", "_request", "rerank", "__init__"},
    "engine.py": {"_set_context_length"},
}


def test_boolean_policy_inventory():
    """Derive declarations from the field registry and AST of actual callers."""
    import ast
    import inspect

    signature = inspect.signature(config._parse_bool_env)
    policy_names = ("unknown", "true_values", "false_values")
    defaults = {name: signature.parameters[name].default for name in policy_names}
    assert config._PARSER_BY_TYPE[bool] is config._parse_bool_env
    derived = {spec.env_key: tuple(defaults.values()) for spec in config.ENV_FIELD_SPECS if spec.py_type is bool}
    call_sites = set()
    raw_readers = {}
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        parser_names = {"_parse_bool_env"} | {
            alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for alias in node.names if alias.name == "_parse_bool_env"
        }
        for function in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
            if any(isinstance(node, ast.Call) and ast.unparse(node.func) in ("os.environ.get", "os.getenv", "lcm_env_get")
                   for node in ast.walk(function)):
                raw_readers.setdefault(path.name, set()).add(function.name)
        for call in calls:
            called = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
            if called not in parser_names:
                continue
            policy = defaults | {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
            assert policy.keys() == defaults.keys()
            call_sites.add((path.name, ast.unparse(call.args[0])))
            if path.name == "__init__.py":
                keys = [ast.literal_eval(node.args[0]) for node in calls
                        if isinstance(node.func, ast.Name) and node.func.id == "_env_flag_enabled"]
            elif isinstance(call.args[0], ast.Name):
                keys = [getattr(db_bootstrap, call.args[0].id)]
            else:
                keys = [ast.literal_eval(call.args[0])]
            for key in keys:
                assert key not in derived
                derived[key] = tuple(policy[name] for name in policy_names)
    assert call_sites == {
        ("__init__.py", "name"), ("identity_anchor.py", "'LCM_IDENTITY_ANCHOR'"),
        ("db_bootstrap.py", "BACKGROUND_INTEGRITY_ENV"),
    }
    assert derived == BOOLEAN_POLICIES
    assert raw_readers == RAW_READERS
