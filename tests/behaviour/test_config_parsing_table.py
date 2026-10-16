"""Characterize current parsing, not a proposed common parsing policy.

Columns are unset, empty, true, false, 1, 0, yes, no, on, off, TRUE,
space-padded true, garbage, -1, 0.5, and a huge integer. Rows are grouped
by the production reader. Expectations are frozen data, never generated
from the reader under test. No network, model, or user configuration is used.
"""

import ast
import importlib.util
import os
from pathlib import Path
import re

import pytest

from hermes_lcm import command, config, db_bootstrap, diagnostics
from hermes_lcm import embedding_provider, externalize, identity_anchor, presets
from hermes_lcm.engine import LCMEngine


ROOT = Path(__file__).resolve().parents[2]
HUGE = 10**100
VALUES = (None, "", "true", "false", "1", "0", "yes", "no", "on", "off",
          "TRUE", " true ", "garbage", "-1", "0.5", str(HUGE))
VALUE_IDS = ("unset", "empty", "true", "false", "1", "0", "yes", "no", "on",
             "off", "uppercase", "padded", "garbage", "negative", "fraction", "huge")
D = object()  # The pinned default in this row, not the current production default.
INT = (D, D, D, D, 1, 0, D, D, D, D, D, D, D, -1, D, HUGE)
FLOAT = (D, D, D, D, 1.0, 0.0, D, D, D, D, D, D, D, -1.0, 0.5, 1e100)
BOOL = (D, D, True, False, True, False, True, False, True, False, True, True, D, D, D, D)
FLAG = (D, False, True, False, True, False, True, False, True, False, True, True, False, False, False, False)
SWITCH = (True, True, True, False, True, False, True, False, True, False, True, True, True, True, True, True)
STRING = (D, *VALUES[1:])

# Frozen ENV_FIELD_SPECS rows: field, environment key, parser, default.
CONFIG_ROWS = (
    # config._parse_bool_env -> LCMConfig.from_env (construction time).
    ('fresh_tail_pressure_yield_enabled', 'LCM_FRESH_TAIL_PRESSURE_YIELD_ENABLED', 'bool', True),
    ('dynamic_leaf_chunk_enabled', 'LCM_DYNAMIC_LEAF_CHUNK_ENABLED', 'bool', False),
    ('cache_friendly_condensation_enabled', 'LCM_CACHE_FRIENDLY_CONDENSATION_ENABLED', 'bool', False),
    ('deferred_maintenance_enabled', 'LCM_DEFERRED_MAINTENANCE_ENABLED', 'bool', False),
    ('threshold_full_sweep_enabled', 'LCM_THRESHOLD_FULL_SWEEP_ENABLED', 'bool', True),
    ('extraction_enabled', 'LCM_EXTRACTION_ENABLED', 'bool', False),
    ('assertions_enabled', 'LCM_ASSERTIONS_ENABLED', 'bool', False),
    ('query_views_enabled', 'LCM_QUERY_VIEWS_ENABLED', 'bool', False),
    ('adaptive_retrieval_enabled', 'LCM_ADAPTIVE_RETRIEVAL_ENABLED', 'bool', False),
    ('assertion_extraction_enabled', 'LCM_ASSERTION_EXTRACTION_ENABLED', 'bool', False),
    ('sensitive_patterns_enabled', 'LCM_SENSITIVE_PATTERNS_ENABLED', 'bool', False),
    ('embedding_privacy_enabled', 'LCM_EMBEDDING_PRIVACY_ENABLED', 'bool', None),
    ('large_output_externalization_enabled', 'LCM_LARGE_OUTPUT_EXTERNALIZATION_ENABLED', 'bool', True),
    ('large_output_active_replay_stubbing_enabled', 'LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUBBING_ENABLED', 'bool', True),
    ('large_output_transcript_gc_enabled', 'LCM_LARGE_OUTPUT_TRANSCRIPT_GC_ENABLED', 'bool', False),
    ('native_recovery', 'LCM_NATIVE_RECOVERY', 'bool', False),
    ('survival_fit', 'LCM_SURVIVAL_FIT', 'bool', True),
    ('embeddings_enabled', 'LCM_EMBEDDINGS_ENABLED', 'bool', False),
    ('rerank_enabled', 'LCM_RERANK_ENABLED', 'bool', False),
    ('recall_reference_strict', 'LCM_RECALL_REFERENCE_STRICT', 'bool', True),
    ('recall_fts_anchor', 'LCM_RECALL_FTS_ANCHOR', 'bool', True),
    ('proactive_recall_enabled', 'LCM_PROACTIVE_RECALL_ENABLED', 'bool', False),
    ('preanswer_evidence_enabled', 'LCM_PREANSWER_EVIDENCE_ENABLED', 'bool', False),
    ('selective_compiler_enabled', 'LCM_SELECTIVE_COMPILER_ENABLED', 'bool', False),
    ('embedding_binary_prescreen', 'LCM_EMBEDDING_BINARY_PRESCREEN', 'bool', False),
    ('doctor_clean_apply_enabled', 'LCM_DOCTOR_CLEAN_APPLY_ENABLED', 'bool', False),
    ('empty_lifecycle_gc_enabled', 'LCM_EMPTY_LIFECYCLE_GC_ENABLED', 'bool', True),
    ('temporal_rollups_enabled', 'LCM_TEMPORAL_ROLLUPS_ENABLED', 'bool', True),
    # config._parse_float_env -> LCMConfig.from_env (construction time).
    ('critical_budget_pressure_ratio', 'LCM_CRITICAL_BUDGET_PRESSURE_RATIO', 'float', 0.0),
    ('foreground_soft_seconds', 'LCM_FOREGROUND_SOFT_SECONDS', 'float', 60.0),
    ('foreground_hard_seconds', 'LCM_FOREGROUND_HARD_SECONDS', 'float', 0.0),
    ('l2_budget_ratio', 'LCM_L2_BUDGET_RATIO', 'float', 0.5),
    ('assertion_extraction_timeout_seconds', 'LCM_ASSERTION_EXTRACTION_TIMEOUT_SECONDS', 'float', 30.0),
    ('survival_reserve', 'LCM_SURVIVAL_RESERVE', 'float', 0.15),
    ('rerank_margin', 'LCM_RERANK_MARGIN', 'float', 0.0),
    ('recall_scan_budget_s', 'LCM_RECALL_SCAN_BUDGET_S', 'float', 0.0),
    ('proactive_recall_min_score', 'LCM_PROACTIVE_RECALL_MIN_SCORE', 'float', 0.01),
    ('embedding_query_timeout_s', 'LCM_EMBEDDING_QUERY_TIMEOUT_S', 'float', 3.0),
    ('recall_query_timeout_s', 'LCM_RECALL_QUERY_TIMEOUT_S', 'float', 8.0),
    ('embedding_backfill_timeout_s', 'LCM_EMBEDDING_BACKFILL_TIMEOUT_S', 'float', 120.0),
    ('embedding_query_spend_window_seconds', 'LCM_EMBEDDING_QUERY_SPEND_WINDOW_SECONDS', 'float', 60.0),
    ('embedding_query_spend_backoff_seconds', 'LCM_EMBEDDING_QUERY_SPEND_BACKOFF_SECONDS', 'float', 60.0),
    # config._parse_float_env_with_source -> LCMConfig.from_env (construction time).
    ('leaf_target_ratio', 'LCM_LEAF_TARGET_RATIO', 'float', 0.2),
    ('context_threshold', 'LCM_CONTEXT_THRESHOLD', 'float', 0.75),
    ('summary_spend_window_seconds', 'LCM_SUMMARY_SPEND_WINDOW_SECONDS', 'float', 600.0),
    ('summary_spend_backoff_seconds', 'LCM_SUMMARY_SPEND_BACKOFF_SECONDS', 'float', 1800.0),
    # config._parse_int_env -> LCMConfig.from_env (construction time).
    ('fresh_tail_pressure_yield_min_observations', 'LCM_FRESH_TAIL_PRESSURE_YIELD_MIN_OBSERVATIONS', 'int', 3),
    ('incremental_max_depth', 'LCM_INCREMENTAL_MAX_DEPTH', 'int', 3),
    ('condensation_fanin', 'LCM_CONDENSATION_FANIN', 'int', 4),
    ('dynamic_leaf_chunk_max', 'LCM_DYNAMIC_LEAF_CHUNK_MAX', 'int', 40000),
    ('cache_friendly_min_debt_groups', 'LCM_CACHE_FRIENDLY_MIN_DEBT_GROUPS', 'int', 2),
    ('deferred_maintenance_max_passes', 'LCM_DEFERRED_MAINTENANCE_MAX_PASSES', 'int', 4),
    ('summary_prefix_target_tokens', 'LCM_SUMMARY_PREFIX_TARGET_TOKENS', 'int', 0),
    ('l3_truncate_tokens', 'LCM_L3_TRUNCATE_TOKENS', 'int', 512),
    ('max_assembly_tokens', 'LCM_MAX_ASSEMBLY_TOKENS', 'int', 0),
    ('reserve_tokens_floor', 'LCM_RESERVE_TOKENS_FLOOR', 'int', 0),
    ('assertion_extraction_max_sources_per_pass', 'LCM_ASSERTION_EXTRACTION_MAX_SOURCES_PER_PASS', 'int', 4),
    ('large_output_externalization_threshold_chars', 'LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS', 'int', 12000),
    ('large_output_active_replay_stub_threshold_tokens', 'LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_THRESHOLD_TOKENS', 'int', 10000),
    ('large_output_active_replay_stub_aged_threshold_tokens', 'LCM_LARGE_OUTPUT_ACTIVE_REPLAY_STUB_AGED_THRESHOLD_TOKENS', 'int', 2000),
    ('summary_circuit_breaker_failure_threshold', 'LCM_SUMMARY_CIRCUIT_BREAKER_FAILURE_THRESHOLD', 'int', 2),
    ('summary_circuit_breaker_cooldown_seconds', 'LCM_SUMMARY_CIRCUIT_BREAKER_COOLDOWN_SECONDS', 'int', 300),
    ('summary_circuit_breaker_rejection_threshold', 'LCM_SUMMARY_CIRCUIT_BREAKER_REJECTION_THRESHOLD', 'int', 6),
    ('expansion_context_tokens', 'LCM_EXPANSION_CONTEXT_TOKENS', 'int', 32000),
    ('expansion_timeout_ms', 'LCM_EXPANSION_TIMEOUT_MS', 'int', 120000),
    ('rerank_window_limit', 'LCM_RERANK_WINDOW_LIMIT', 'int', 0),
    ('recall_scan_rows', 'LCM_RECALL_SCAN_ROWS', 'int', 25000),
    ('recall_scan_max_rows', 'LCM_RECALL_SCAN_MAX_ROWS', 'int', 0),
    ('proactive_recall_budget_tokens', 'LCM_PROACTIVE_RECALL_BUDGET_TOKENS', 'int', 500),
    ('embedding_bounded_scan_rows', 'LCM_EMBEDDING_BOUNDED_SCAN_ROWS', 'int', 2000),
    ('embedding_store_dim', 'LCM_EMBEDDING_STORE_DIM', 'int', 0),
    ('knn_prescreen_multiplier', 'LCM_KNN_PRESCREEN_MULTIPLIER', 'int', 4),
    ('embedding_max_batch_items', 'LCM_EMBEDDING_MAX_BATCH_ITEMS', 'int', 1000),
    ('embedding_query_spend_max_calls', 'LCM_EMBEDDING_QUERY_SPEND_MAX_CALLS', 'int', 600),
    ('new_session_retain_depth', 'LCM_NEW_SESSION_RETAIN_DEPTH', 'int', 2),
    ('empty_lifecycle_gc_threshold', 'LCM_EMPTY_LIFECYCLE_GC_THRESHOLD', 'int', 200),
    ('rollup_daily_target_tokens', 'LCM_ROLLUP_DAILY_TARGET_TOKENS', 'int', 5000),
    ('rollup_daily_max_tokens', 'LCM_ROLLUP_DAILY_MAX_TOKENS', 'int', 15000),
    ('rollup_aggregate_max_tokens', 'LCM_ROLLUP_AGGREGATE_MAX_TOKENS', 'int', 20000),
    ('rollup_builds_per_pass', 'LCM_ROLLUP_BUILDS_PER_PASS', 'int', 2),
    ('rollup_maintenance_budget_ms', 'LCM_ROLLUP_MAINTENANCE_BUDGET_MS', 'int', 5000),
    # config._parse_int_env_with_source -> LCMConfig.from_env (construction time).
    ('fresh_tail_count', 'LCM_FRESH_TAIL_COUNT', 'int', 24),
    ('fresh_tail_max_tokens', 'LCM_FRESH_TAIL_MAX_TOKENS', 'int', 24000),
    ('leaf_chunk_tokens', 'LCM_LEAF_CHUNK_TOKENS', 'int', 24000),
    ('leaf_target_min_tokens', 'LCM_LEAF_TARGET_MIN_TOKENS', 'int', 2000),
    ('leaf_target_max_tokens', 'LCM_LEAF_TARGET_MAX_TOKENS', 'int', 2000),
    ('summary_prompt_version', 'LCM_SUMMARY_PROMPT_VERSION', 'int', 1),
    ('summary_spend_max_calls', 'LCM_SUMMARY_SPEND_MAX_CALLS', 'int', 24),
    ('summary_timeout_ms', 'LCM_SUMMARY_TIMEOUT_MS', 'int', 120000),
    # config._parse_str_env -> LCMConfig.from_env (construction time).
    ('custom_instructions', 'LCM_CUSTOM_INSTRUCTIONS', 'str', ''),
    ('extraction_model', 'LCM_EXTRACTION_MODEL', 'str', ''),
    ('extraction_output_path', 'LCM_EXTRACTION_OUTPUT_PATH', 'str', ''),
    ('assertion_extraction_model', 'LCM_ASSERTION_EXTRACTION_MODEL', 'str', ''),
    ('large_output_externalization_path', 'LCM_LARGE_OUTPUT_EXTERNALIZATION_PATH', 'str', ''),
    ('summary_model', 'LCM_SUMMARY_MODEL', 'str', ''),
    ('expansion_model', 'LCM_EXPANSION_MODEL', 'str', ''),
    ('database_path', 'LCM_DATABASE_PATH', 'str', ''),
    ('rerank_model', 'LCM_RERANK_MODEL', 'str', 'rerank-2.5-lite'),
    ('proactive_recall_provider', 'LCM_PROACTIVE_RECALL_PROVIDER', 'str', ''),
    ('preanswer_evidence_mode', 'LCM_PREANSWER_EVIDENCE_MODE', 'str', ''),
    ('selective_compiler_model', 'LCM_SELECTIVE_COMPILER_MODEL', 'str', ''),
    ('embedding_storage_dtype', 'LCM_EMBEDDING_STORAGE_DTYPE', 'str', 'float32'),
    ('embedding_provider', 'LCM_EMBEDDING_PROVIDER', 'str', ''),
    ('embedding_model', 'LCM_EMBEDDING_MODEL', 'str', ''),
    ('embedding_content_policy', 'LCM_EMBED_CONTENT_POLICY', 'str', 'conversational'),
    ('ollama_base_url', 'LCM_OLLAMA_BASE_URL', 'str', 'http://localhost:11434'),
    ('embedding_base_url', 'LCM_EMBEDDING_BASE_URL', 'str', ''),
    ('embedding_api_key_env', 'LCM_EMBEDDING_API_KEY_ENV', 'str', 'LCM_EMBEDDING_API_KEY'),
)

EXTRA_CONFIG_ROWS = (
    ("model_thresholds", "LCM_MODEL_THRESHOLDS", "mapping", {}),
    ("summary_reasoning_effort", "LCM_SUMMARY_REASONING_EFFORT", "effort", ""),
    ("expansion_reasoning_effort", "LCM_EXPANSION_REASONING_EFFORT", "effort", ""),
    ("sensitive_patterns", "LCM_SENSITIVE_PATTERNS", "list", ["api_key", "bearer_token", "password_assignment", "private_key"]),
    ("summary_fallback_models", "LCM_SUMMARY_FALLBACK_MODELS", "list", []),
    ("empty_lifecycle_gc_max_age_hours", "LCM_EMPTY_LIFECYCLE_GC_MAX_AGE_HOURS", "float", 24.0),
    ("recall_arm_weights", "LCM_RECALL_ARM_WEIGHTS", "mapping", {"fts": 0.5, "summary": 1.0, "chunk": 1.0}),
    ("ignore_session_patterns", "LCM_IGNORE_SESSION_PATTERNS", "list", []),
    ("stateless_session_patterns", "LCM_STATELESS_SESSION_PATTERNS", "list", []),
    ("ignore_message_patterns", "LCM_IGNORE_MESSAGE_PATTERNS", "list", []),
)

# from_env has validation after the primitive reader for these five fields.
SPECIAL = {
    "fresh_tail_max_tokens": (D, D, D, D, 1, 0, D, D, D, D, D, D, D, 0, D, HUGE),
    "leaf_target_ratio": (D, D, D, D, 1.0, D, D, D, D, D, D, D, D, D, 0.5, D),
    "leaf_target_min_tokens": (D, D, D, D, 1, D, D, D, D, D, D, D, D, D, D, D),
    "leaf_target_max_tokens": (D, D, D, D, D, D, D, D, D, D, D, D, D, D, D, HUGE),
    "summary_prompt_version": (1,) * 16,
}


def derive_environment_keys():
    """Derivation helper: ENV_FIELD_SPECS plus exact LCM_* string literals.

    This is the AST equivalent of searching root *.py for LCM_[A-Z0-9_]+,
    excluding comments, docstrings, and constant identifiers (e.g. LCM_GREP).
    It includes indirect reads through named constants and the default API-key
    name. A new literal forces a coverage decision instead of auto-adding a test.
    """
    keys = {spec.env_key for spec in config.ENV_FIELD_SPECS}
    for path in ROOT.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and re.fullmatch(r"LCM_[A-Z0-9_]+", node.value)):
                keys.add(node.value)
    return keys


def set_value(monkeypatch, key, value):
    if value is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, value)


def resolve(column, index, default):
    value = column[index]
    return default if value is D else value


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("LCM_") or key == "SILICONFLOW_API_KEY":
            monkeypatch.delenv(key)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {})
    monkeypatch.setattr(config, "_host_message_uid_mode_warned", False)
    monkeypatch.setattr(config, "_sqlite_mmap_size_warned", False)


@pytest.fixture(scope="module")
def plugin():
    # conftest deliberately leaves hermes_lcm.__init__ unexecuted. Load a
    # separate real module without replacing that shared package or registering.
    spec = importlib.util.spec_from_file_location("config_table_plugin", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("field,key,kind,default", CONFIG_ROWS + EXTRA_CONFIG_ROWS,
                         ids=[row[1] for row in CONFIG_ROWS + EXTRA_CONFIG_ROWS])
def test_from_env_construction_table(monkeypatch, field, key, kind, default):
    before = config.LCMConfig.from_env()
    assert getattr(before, field) == default
    for index, raw in enumerate(VALUES):
        set_value(monkeypatch, key, raw)
        if kind == "list":
            expected = default if raw is None else ([raw.strip()] if raw.strip() else [])
        elif kind in {"mapping", "effort"}:
            expected = default
        else:
            column = SPECIAL.get(field, {"int": INT, "float": FLOAT, "bool": BOOL, "str": STRING}[kind])
            expected = resolve(column, index, default)
        current = config.LCMConfig.from_env()
        assert getattr(current, field) == expected, (key, VALUE_IDS[index])
        # Changing env never alters an existing config; a new construction reads it.
        assert getattr(before, field) == default, (key, VALUE_IDS[index])
        set_value(monkeypatch, key, None)
        assert getattr(current, field) == expected
        assert getattr(config.LCMConfig.from_env(), field) == default


# Standalone readers resolve again on every invocation, including after a
# config object exists. Defaults here are literals, independent of production.
DYNAMIC_ROWS = (
    ("LCM_HOST_MESSAGE_UID", config.host_message_uid_mode,
     ("shadow",) * 8 + ("on", "off") + ("shadow",) * 6),
    ("LCM_SQLITE_MMAP_SIZE", config.sqlite_mmap_size,
     (268435456,) * 4 + (1, 0) + (268435456,) * 9 + (HUGE,)),
    ("LCM_IDENTITY_ANCHOR", identity_anchor.identity_anchor_enabled, SWITCH),
    ("LCM_FTS_INTEGRITY_BACKGROUND", db_bootstrap._background_integrity_enabled, SWITCH),
    ("LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS", db_bootstrap._integrity_check_interval_hours,
     tuple(24.0 if value is D else value for value in FLOAT)),
    ("LCM_EMBEDDING_BACKFILL_LEASE_TTL_S", command._embedding_backfill_lease_ttl_s,
     (600.0,) * 4 + (1.0, 600.0) + (600.0,) * 8 + (0.5, 1e100)),
    ("LCM_EMBEDDING_BACKFILL_HEARTBEAT_S", command._embedding_backfill_heartbeat_s,
     (60.0,) * 4 + (1.0, 0.0) + (60.0,) * 8 + (0.5, 1e100)),
    ("LCM_EMBEDDING_BACKFILL_BUDGET_S", command._embedding_backfill_budget_s,
     (0.0,) * 4 + (1.0, 0.0) + (0.0,) * 8 + (0.5, 1e100)),
    ("LCM_EMBEDDING_BACKFILL_BATCH_SIZE", command._embedding_backfill_batch_size,
     (32,) * 4 + (1, 32) + (32,) * 9 + (1000,)),
)


@pytest.mark.parametrize("key,reader,expected", DYNAMIC_ROWS, ids=[row[1].__name__ for row in DYNAMIC_ROWS])
def test_call_time_readers(monkeypatch, key, reader, expected):
    config.LCMConfig.from_env()
    for index, raw in enumerate(VALUES):
        set_value(monkeypatch, key, raw)
        assert reader() == expected[index], (key, VALUE_IDS[index])
        set_value(monkeypatch, key, None)
        assert reader() == expected[0]


@pytest.mark.parametrize("default", (False, True))
def test_boolean_reader_disagreement(monkeypatch, plugin, default):
    key = "LCM_ENABLE_SLASH_COMMAND"
    for index, raw in enumerate(VALUES):
        set_value(monkeypatch, key, raw)
        assert plugin._env_flag_enabled(key, default) is resolve(FLAG, index, default)
        assert config._parse_bool_env(key, default) is resolve(BOOL, index, default)


def test_disabled_tools_call_time(monkeypatch, plugin):
    for raw in (*VALUES, " lcm_grep, lcm_recall, lcm_grep, , "):
        set_value(monkeypatch, "LCM_DISABLED_TOOLS", raw)
        expected = set() if not raw else {raw.strip()}
        if raw and "," in raw:
            expected = {"lcm_grep", "lcm_recall"}
        assert plugin._disabled_tool_names() == expected
        assert LCMEngine._disabled_tool_names() == expected


@pytest.mark.parametrize("field,key", tuple(presets._FIELD_ENV.items()))
def test_preset_readers_call_time(monkeypatch, field, key):
    before = config.LCMConfig.from_env()
    default = getattr(before, field)
    for index, raw in enumerate(VALUES):
        set_value(monkeypatch, key, raw)
        column = FLOAT if field == "context_threshold" else INT
        valid = column[index] is not D
        assert presets.explicit_operator_overrides() == ({field: key} if valid else {})
        assert presets.invalid_operator_overrides() == ({field: key} if raw is not None and not valid else {})
        if valid:
            assert presets._parse_override_value(field, raw) == column[index]
        assert getattr(before, field) == default


def test_engine_and_containment_call_time(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = config.LCMConfig(database_path=str(tmp_path / "table.db"),
                           context_threshold=0.5, max_assembly_tokens=0,
                           embeddings_enabled=False, temporal_rollups_enabled=False)
    engine = LCMEngine(config=cfg, hermes_home=str(tmp_path / "home"))
    try:
        for index, raw in enumerate(VALUES):
            set_value(monkeypatch, "LCM_ABSOLUTE_THRESHOLD_TOKENS", raw)
            engine._set_context_length(1000, source="table", model="table", provider="custom")
            expected = 1 if index == 4 else HUGE if index == 15 else 500
            assert engine.threshold_tokens == expected, VALUE_IDS[index]
            set_value(monkeypatch, "LCM_ABSOLUTE_THRESHOLD_TOKENS", None)
            engine._set_context_length(1000, source="table", model="table", provider="custom")
            assert engine.threshold_tokens == 500

        for raw in VALUES:
            set_value(monkeypatch, "LCM_HERMES_BASE_DIR", raw)
            expected_base = tmp_path / raw if raw else None
            assert engine._get_allowed_hermes_base() == expected_base
            # Both externalization branches and diagnostics re-read the same
            # key after construction; nonempty text is a literal path, not bool.
            base = expected_base or tmp_path
            assert diagnostics._enforce_state_db_containment(base / "state.db", description="table") == base / "state.db"
            for configured in (False, True):
                cfg.large_output_externalization_path = str(base / "payloads") if configured else ""
                result = externalize.get_large_output_storage_dir(cfg, str(base), create=False)
                assert result == base / ("payloads" if configured else externalize.DEFAULT_LARGE_OUTPUT_DIRNAME)
                if raw:
                    cfg.large_output_externalization_path = str(tmp_path / "outside") if configured else ""
                    with pytest.raises(ValueError, match="not within allowed base"):
                        externalize.get_large_output_storage_dir(cfg, str(tmp_path), create=False)
            if raw:
                with pytest.raises(ValueError, match="not within allowed base"):
                    diagnostics._enforce_state_db_containment(tmp_path / "outside", description="table")
        assert not (tmp_path / "payloads").exists()
    finally:
        engine.shutdown()


def test_embedding_key_call_time(monkeypatch):
    observed = []

    def transport(**kwargs):
        observed.append(kwargs["headers"]["Authorization"])
        return embedding_provider.HttpResponse(200, {}, b'{"data":[{"index":0,"embedding":[1.0,0.0]}]}')

    provider = embedding_provider.OpenAICompatibleProvider(
        "table-model", base_url="https://example.invalid/v1", transport=transport,
    )
    cfg = config.LCMConfig(embeddings_enabled=True, embedding_provider="openai-compatible",
                           embedding_model="table-model", embedding_base_url="https://example.invalid/v1")
    for raw in VALUES:
        set_value(monkeypatch, "LCM_EMBEDDING_API_KEY", raw)
        expected = raw.strip() if raw else ""
        assert embedding_provider.probe_provider_availability(cfg)["available"] is bool(expected)
        observed.clear()
        if expected:
            assert provider._request(["synthetic text"]) == [[1.0, 0.0]]
            assert observed == [f"Bearer {expected}"]
        else:
            with pytest.raises(embedding_provider.EmbeddingProviderError, match="is not set"):
                provider._request(["synthetic text"])
            assert observed == []


def test_key_inventory_is_complete():
    scalar = {(spec.name, spec.env_key, spec.py_type.__name__) for spec in config.ENV_FIELD_SPECS}
    assert scalar == {row[:3] for row in CONFIG_ROWS}
    covered = {row[1] for row in CONFIG_ROWS + EXTRA_CONFIG_ROWS}
    covered.update(row[0] for row in DYNAMIC_ROWS)
    covered.update({"LCM_ENABLE_SLASH_COMMAND", "LCM_DISABLED_TOOLS",
                    "LCM_ABSOLUTE_THRESHOLD_TOKENS", "LCM_HERMES_BASE_DIR", "LCM_EMBEDDING_API_KEY"})
    assert derive_environment_keys() == covered


@pytest.mark.parametrize("key,raw,field,expected,warns", (
    ("LCM_SURVIVAL_FIT", "garbage", "survival_fit", True, False),
    ("LCM_CONTEXT_THRESHOLD", "-1", "context_threshold", -1.0, False),
    ("LCM_SURVIVAL_RESERVE", "-1", "survival_reserve", -1.0, False),
    ("LCM_LEAF_CHUNK_TOKENS", "-1", "leaf_chunk_tokens", -1, False),
    ("LCM_FRESH_TAIL_COUNT", "garbage", "fresh_tail_count", 24, True),
    ("LCM_SUMMARY_TIMEOUT_MS", "0", "summary_timeout_ms", 0, False),
    ("LCM_MAX_ASSEMBLY_TOKENS", "garbage", "max_assembly_tokens", 0, False),
    ("LCM_EMBEDDING_PROVIDER", "garbage", "embedding_provider", "garbage", False),
    ("LCM_PREANSWER_EVIDENCE_MODE", "garbage", "preanswer_evidence_mode", "garbage", False),
    ("LCM_EMBEDDING_STORAGE_DTYPE", "garbage", "embedding_storage_dtype", "garbage", False),
))
def test_config_warning_presence(monkeypatch, caplog, key, raw, field, expected, warns):
    set_value(monkeypatch, key, raw)
    current = config.LCMConfig.from_env()
    assert getattr(current, field) == expected
    assert bool(current.config_source_warnings) is warns
    if warns:
        assert key in current.config_source_warnings[0]
    else:
        assert not caplog.records
