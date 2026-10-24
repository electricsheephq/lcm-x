"""S2-R1 probes: loader-only provenance, copies, keys and fleet precedence."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by name
import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from hermes_lcm import config as config_module, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.embedding_maintenance import _local_model_key
from hermes_lcm.ingest_policy import embedding_prefix_suffix, embedding_privacy_revision
from hermes_lcm.leaf_prep import route_fingerprint
from tests.invariants.test_issue_1603_profile_config import host, load_profile  # noqa: F401
from tests.test_issue_1186_smallhost import DefaultOnConfig
from tests.test_lcm_recall import recall_engine  # noqa: F401
from tests.harness import engine_context


@pytest.mark.parametrize("raw", ["", "true", "false", "1", "0", "garbage"])
@pytest.mark.parametrize("enabled", [None, "true", "false"])
def test_p1_internal_flag_not_settable(monkeypatch, raw, enabled):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED_EXPLICIT", raw)
    if enabled is not None:
        monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", enabled)
    monkeypatch.setattr(config_module, "_load_hermes_config_yaml", lambda: {
        "lcm": {"embeddings_enabled_explicit": enabled is None}})
    config = DefaultOnConfig.from_env()
    assert config.embeddings_enabled_explicit is (enabled is not None)
    assert all(spec.name != "embeddings_enabled_explicit" for spec in config_module.ENV_FIELD_SPECS)


@pytest.mark.parametrize("raw,explicit,value", [
    ("", False, True), ("   ", False, True), ("invalid", False, True),
    ("true", True, True), ("false", True, False),
    ("1", True, True), ("0", True, False),
    (" YES ", True, True), ("off", True, False),
])
def test_p2_only_valid_value_is_explicit(monkeypatch, raw, explicit, value):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", raw)
    config = DefaultOnConfig.from_env()
    assert config.embeddings_enabled_explicit is explicit
    assert config.embeddings_enabled is value
    # Provenance must not add, suppress or rewrite the existing parser warning.
    warnings = []
    config_module._parse_bool_env("LCM_EMBEDDINGS_ENABLED", True, warnings=warnings)
    assert [w for w in config.config_source_warnings if "LCM_EMBEDDINGS_ENABLED=" in w] == warnings


@pytest.mark.parametrize("explicit", [False, True])
def test_p3_copies_preserve_provenance(recall_engine, monkeypatch, explicit):
    config = DefaultOnConfig.from_env()
    config.embeddings_enabled_explicit = explicit
    for copied in (copy.copy(config), copy.deepcopy(config), replace(config, embedding_model="override")):
        assert copied.embeddings_enabled_explicit is explicit
    recall_engine._config = config
    seen = []
    monkeypatch.setattr(tools, "resolve_provider", lambda c: seen.append(c) or object())
    tools._resolve_recall_provider(recall_engine, provider_override="ollama")
    monkeypatch.setattr(tools, "default_chunk_model", lambda *a: "chunk-model")
    tools._resolve_recall_chunk_provider(recall_engine, SimpleNamespace(provider_id="ollama", model_id="summary"))
    assert len(seen) == 2
    assert all(c is not config and c.embeddings_enabled_explicit is explicit for c in seen)
    assert (config.embedding_provider, config.embedding_model) == ("fastembed", "BAAI/bge-small-en-v1.5")


@pytest.mark.parametrize("explicit", [False, True])
def test_p3_profile_copy_preserves_provenance(host, load_profile, tmp_path, explicit):
    mapping = {"LCM_EMBEDDINGS_ENABLED": "false"} if explicit else {}
    with host.bind(tmp_path, mapping):
        _, engine = load_profile()
        clone = engine.clone_for_agent()
    try:
        assert engine._config.embeddings_enabled_explicit is explicit
        assert clone._config.embeddings_enabled_explicit is explicit
    finally:
        clone.shutdown()


def test_p4_provenance_does_not_change_runtime_keys(tmp_path, monkeypatch):
    with engine_context(tmp_path) as engine:
        _assert_runtime_keys(engine, monkeypatch)


def _assert_runtime_keys(recall_engine, monkeypatch):
    config = DefaultOnConfig.from_env()
    assert asdict(config)["embeddings_enabled_explicit"] is False
    recall_engine._config = config
    monkeypatch.setattr(recall_engine, "_bypasses_lcm_context_management", lambda: False)
    recall_engine._session_id = "probe-session"
    recall_engine._conversation_id = "probe-conversation"
    recall_engine._no_progress_replay = ("probe-session", "probe-conversation", "probe-replay")
    recall_engine._no_progress_observed_tokens = 100
    def keys():
        return (route_fingerprint(config), recall_engine._no_progress_fingerprint(),
                _local_model_key(config), embedding_privacy_revision(config),
                embedding_prefix_suffix(config, "ollama"))
    before = keys()
    assert before[1] is not None
    config.embeddings_enabled_explicit = True
    assert keys() == before
    # Provider cache keys are checked by calling both resolvers across the flag change.
    resolved = []
    monkeypatch.setattr(tools, "resolve_provider", lambda c: resolved.append(c) or object())
    first = tools._resolve_recall_provider(recall_engine, provider_override="ollama")
    config.embeddings_enabled_explicit = False
    assert tools._resolve_recall_provider(recall_engine, provider_override="ollama") is first
    assert len(resolved) == 1


@pytest.mark.parametrize("cpus", [0.5, 1, 2, 4, 64])
def test_p5_fleet_env_never_adapted(recall_engine, monkeypatch, cpus):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "true")
    recall_engine._config = DefaultOnConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True, "detail": "synthetic backend"})
    called = []
    monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *a, **k: called.append(True))
    result = tools.lcm_recall({"query": "cobalt"}, engine=recall_engine)
    assert called
    assert "cpu_limited" not in result


def test_p5_explicit_off_inert(monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    config = LCMConfig.from_env()
    assert config.embeddings_enabled is False
    assert config.embeddings_enabled_explicit is True
    monkeypatch.setattr(tools, "effective_cpus", lambda: pytest.fail("OFF must not probe CPU"))
    assert tools.default_semantic_cpu_limited(config) is False
