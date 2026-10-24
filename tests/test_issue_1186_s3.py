"""Default-on slice: provider-scoped identity and truthful local readiness."""
# ruff: noqa: F811 -- shared pytest fixture
import json
import sqlite3
import shlex
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_lcm import command, config as config_module, embedding_maintenance, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.diagnostics import doctor_guidance_for_check
from hermes_lcm.embedding_provider import ProviderUnavailable, resolve_provider
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
from tests.test_lcm_recall import recall_engine  # noqa: F401
from tests.harness import engine_context


MODEL = "BAAI/bge-small-en-v1.5"


@pytest.mark.parametrize("profile,profile_home", [(None, False), ("work", False), (None, True)])
def test_missing_companion_matches_installer_command(tmp_path, monkeypatch, profile, profile_home):
    from tests.test_issue_1186_dependency_companion import run_resolving_installer
    from hermes_lcm.embedding_provider import probe_provider_availability
    result, home, target, _ = run_resolving_installer(
        tmp_path, profile=profile, profile_home=profile_home, custom=False,
    )
    assert result.returncode == 0
    monkeypatch.setenv("HERMES_HOME", str(target if profile_home else home))
    if profile:
        monkeypatch.setenv("HERMES_PROFILE", profile)
    else:
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda *a: None)
    probe = probe_provider_availability(LCMConfig(embedding_provider="fastembed", embedding_model=MODEL))
    expected = next(line.split("run: ", 1)[1] for line in result.stdout.splitlines()
                    if line.startswith("If the companion is disabled, run:"))
    assert expected in probe["detail"]
    assert shlex.split(expected)[-2:] == ["enable", "hermes-lcm-x-local-embeddings"]


def test_shipped_default_and_provider_scoped_model():
    config = LCMConfig.from_env()
    assert config.embeddings_enabled is True
    assert config.embedding_provider == "fastembed"
    assert config.embedding_model == ""
    assert resolve_provider(config).model_id == MODEL


@pytest.mark.parametrize("provider", ["voyage", "ollama", "openai-compatible"])
def test_cloud_without_model_rejected(provider):
    with pytest.raises(ProviderUnavailable, match="must both be set"):
        resolve_provider(LCMConfig(embedding_provider=provider))


def test_proactive_cloud_without_model_rejected():
    engine = SimpleNamespace(_config=LCMConfig(proactive_recall_provider="voyage"))
    with pytest.raises(ProviderUnavailable, match="must both be set"):
        tools._resolve_recall_provider(engine, provider_override="voyage")


def test_explicit_fastembed_model_wins(monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "explicit-local-model")
    config = LCMConfig.from_env()
    assert resolve_provider(config).model_id == "explicit-local-model"
    assert config_module.effective_embedding_model("voyage", "explicit-cloud") == "explicit-cloud"


def test_default_warmup_records_identical_effective_identity(tmp_path, monkeypatch):
    config = LCMConfig(database_path=str(tmp_path / "identity.db"))
    # The real resolver selects the model; only model execution is substituted.
    from hermes_lcm.embedding_provider import FastembedProvider
    monkeypatch.setattr(FastembedProvider, "warmup", lambda self: None)
    monkeypatch.setattr(FastembedProvider, "_ensure_local", lambda self: None)
    monkeypatch.setattr(FastembedProvider, "embed_documents", lambda self, texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(FastembedProvider, "embed_query", lambda self, text: [1.0, 0.0])
    identities = []
    for selected in (config, replace(config, embedding_model=MODEL)):
        result = command._embedding_register_profiles(selected, selected.database_path, allow_download=False)
        assert isinstance(result, dict), result
        store = VectorStore(selected.database_path, config=selected)
        try:
            identity = store.capture_identity(MODEL, provider="fastembed")
            identities.append(identity)
            assert identity == EmbeddingIdentity.canonical("fastembed", MODEL, "", 2, "float32", "little", "summary")
        finally:
            store.close()
    assert identities[0] == identities[1]
    assert identities[0].identity_hash.encode() == identities[1].identity_hash.encode()
    assert embedding_maintenance._local_model_key(config)[0] == MODEL


@pytest.mark.parametrize("cpus", [1, 8])
@pytest.mark.parametrize("state", ["preparing", "present"])
@pytest.mark.parametrize("coverage_error", [False, True])
def test_doctor_readiness_on_every_host(recall_engine, monkeypatch, cpus, state, coverage_error):
    engine = recall_engine
    engine._config = LCMConfig.from_env()
    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus)
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: (state, ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "provider": "fastembed", "model": MODEL, "available": True, "detail": "importable",
    })
    if coverage_error:
        def unreadable(*args):
            raise sqlite3.OperationalError("synthetic coverage read failure")
        monkeypatch.setattr(command, "_embedding_read_connection", unreadable)
    check = tools._embedding_provider_health_check(engine)
    guidance = doctor_guidance_for_check(check)
    if coverage_error:
        assert check["status"] == "warn"
        assert "coverage" in guidance["operator_action"] and "SQLite" in guidance["operator_action"]
        assert not check["detail"].get("informational", False)
    else:
        assert "chunks" in check["detail"]["readiness"]
        assert check["detail"].get("informational") is True
        if state == "preparing":
            assert check["detail"]["readiness"].startswith("model: preparing;")
            assert "prepar" in guidance["operator_action"]
    if guidance:
        assert "configured but not runnable" not in guidance["operator_action"]


@pytest.mark.parametrize("include", ["all", "summaries"])
def test_partial_local_index_uses_full_text(recall_engine, monkeypatch, include):
    engine = recall_engine
    engine._config = LCMConfig(embedding_provider="fastembed", embedding_model=MODEL)
    target = engine._store.append("old-session", {"role": "user", "content": "cobalt preserved memory"})
    provider = SimpleNamespace(provider_id="fastembed", model_id=MODEL)
    monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *a, **k: provider)
    monkeypatch.setattr(tools, "_lcm_recall_catchup_reason", lambda *a: "summary semantic index catching up (10 rows not yet embedded)")
    monkeypatch.setattr(tools, "_lcm_grep_embed_query", lambda *a, **k: pytest.fail("partial local index must use full text"))
    result = json.loads(tools.lcm_recall({"query": "cobalt", "include": include}, engine=engine))
    assert result["degraded"] is True and "catching up" in result["degraded_reason"]
    assert result["provenance"]["coverage"]["fts"] == "ok"
    assert any(hit.get("store_id") == target for hit in result["hits"])


@pytest.mark.parametrize("state", ["preparing", "present"])
def test_informational_state_does_not_raise_doctor_status(tmp_path, monkeypatch, state):
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: (state, ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "provider": "fastembed", "model": MODEL, "available": True, "detail": "importable",
    })
    with engine_context(tmp_path, window=40_000, temporal_rollups_enabled=False) as engine:
        engine._config.embeddings_enabled_explicit = False
        text = command._doctor_text(engine)
        assert "status: issues-found" not in text
        assert "embedding_readiness:" in text
        check = next(c for c in json.loads(tools.lcm_doctor({}, engine=engine))["checks"]
                     if c["check"] == "embedding_provider_health")
        assert check["detail"]["informational"] is True


def test_small_host_prefix_parity_with_binding_store(tmp_path, monkeypatch):
    from tests.test_issue_930_prefix_bound import add_nodes, assemble, visible_node_ids
    from hermes_lcm import engine as engine_module, survival_fit, tokens
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(engine_module, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tools, "effective_cpus", lambda: 1)
    with engine_context(tmp_path, window=40_000, temporal_rollups_enabled=False,
                        proactive_recall_enabled=True, proactive_recall_budget_tokens=500) as engine:
        engine._config.embeddings_enabled_explicit = False
        monkeypatch.setattr(engine, "_build_proactive_recall_message", lambda *a, **k: None)
        ids = add_nodes(engine, 1, 100, token_count=400)
        tail = [{"role": "user", "content": "current question"}]
        engine._config.embeddings_enabled = False
        off = assemble(engine, tail)
        assert 0 < len(visible_node_ids(off)) < len(ids)
        engine._config.embeddings_enabled = True
        assert json.dumps(assemble(engine, tail), sort_keys=True) == json.dumps(off, sort_keys=True)
        engine._config.proactive_recall_provider = "fastembed"
        assert len(visible_node_ids(assemble(engine, tail))) < len(visible_node_ids(off))


@pytest.mark.parametrize("state", ["missing", "preparing"])
def test_local_readiness_preserves_full_fts_budget(recall_engine, monkeypatch, state):
    engine = recall_engine
    engine._config = LCMConfig(embedding_provider="fastembed", embedding_model=MODEL)
    engine._store.append("old", {"role": "user", "content": "cobalt memory"})
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: ("preparing", "") if state == "preparing" else None)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "available": state != "missing", "detail": "missing dependency" if state == "missing" else "importable"})
    monkeypatch.setattr(tools, "_lcm_recall_has_usable_vector_corpus", lambda *a, **k: pytest.fail("unready local recall must not reserve the semantic budget"))
    result = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=engine))
    assert result["provenance"]["coverage"]["fts"] == "ok"
    assert result["degraded"] is True
