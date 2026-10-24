"""R3: consent survives the local default; unknown readiness stays full text."""
# ruff: noqa: F811 -- shared pytest fixture
import json
import socket
import sqlite3
import urllib.error
import urllib.request
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_lcm import command, config as config_module, embedding_maintenance, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.embedding_provider import FastembedProvider, ProviderUnavailable, resolve_provider
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import VectorStore
from tests.test_lcm_recall import recall_engine  # noqa: F401

MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture
def no_egress(monkeypatch):
    requests = []

    def refuse(*args, **kwargs):
        raise OSError("synthetic test refuses sockets")

    def record(request, *args, **kwargs):
        requests.append({"url": request.full_url, "bytes": len(request.data or b"")})
        raise urllib.error.URLError("synthetic test refuses HTTP")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", record)
    return requests


@pytest.mark.parametrize("provider", ["voyage", "openai-compatible", "ollama", "siliconflow"])
@pytest.mark.parametrize("warmed", [False, True])
def test_cloud_unset_enable_is_inert(tmp_path, monkeypatch, no_egress, provider, warmed):
    from hermes_lcm import engine as engine_module

    monkeypatch.delenv("LCM_EMBEDDINGS_ENABLED", raising=False)
    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", provider)
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "fixture-model")
    monkeypatch.setenv("LCM_EMBEDDING_BASE_URL", "https://fixture.invalid/v1")
    for key in ("VOYAGE_API_KEY", "LCM_EMBEDDING_API_KEY", "SILICONFLOW_API_KEY"):
        monkeypatch.setenv(key, "synthetic-fixture-key")
    config = LCMConfig.from_env()
    config.database_path = str(tmp_path / "cloud.db")
    config.temporal_rollups_enabled = False
    # Real profile registration and identity; only the historical warmup transport
    # is substituted. Restore the real provider before all unset-flag operations.
    if warmed:
        with monkeypatch.context() as warm:
            real_resolve = command.resolve_provider

            def warm_provider(selected):
                resolved = real_resolve(selected)
                resolved.embed_query = lambda text: [1.0, 0.0]
                return resolved

            warm.setattr(command, "resolve_provider", warm_provider)
            result = command._embedding_register_profiles(
                replace(config, embeddings_enabled=True), config.database_path, allow_download=False,
            )
            assert isinstance(result, dict), result
    queued = []
    monkeypatch.setattr(engine_module._EMBEDDING_MAINTENANCE_SCHEDULER, "schedule",
                        lambda key, callback, **kwargs: queued.append(callback))
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        target = engine._store.append("old", {"role": "user", "content": "cobalt preserved memory"})
        engine._dag.add_node(SummaryNode(session_id="old", depth=0, summary="cobalt summary",
                                        created_at=1.0, latest_at=1.0))
        engine.on_session_start("current", platform="cli", context_length=40_000)
        for callback in queued:
            callback()
        recall = json.loads(tools.lcm_recall({"query": "cobalt"}, engine=engine))
        tools.lcm_grep({"query": "cobalt", "mode": "semantic"}, engine=engine)
        health = tools._embedding_provider_health_check(engine)
        print(json.dumps({"provider": provider, "warmed": warmed,
                          "enabled": config.embeddings_enabled, "scheduled": len(queued),
                          "requests": no_egress, "doctor": health}, sort_keys=True))
        assert config.embeddings_enabled is False
        assert no_egress == [] and queued == []
        assert recall["provenance"]["coverage"]["summary"] == "disabled"
        assert any(hit.get("store_id") == target for hit in recall["hits"])
        if warmed:
            assert health["status"] == "warn"
            assert health["detail"]["reason"] == (
                "embeddings are disabled (LCM_EMBEDDINGS_ENABLED=false) but this "
                "database still has an active summary embedding profile; stored "
                "vectors are no longer being maintained and will go stale"
            )
        else:
            assert health["status"] == "pass"
            assert health["detail"] == "semantic embeddings are disabled (LCM_EMBEDDINGS_ENABLED=false)"
    finally:
        engine.shutdown()


def test_runtime_model_only_retains_missing_provider(monkeypatch):
    monkeypatch.delenv("LCM_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("LCM_EMBEDDINGS_ENABLED", raising=False)
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", MODEL)
    config = LCMConfig.from_env()
    assert config.embeddings_enabled is False
    assert config.embedding_provider == ""
    with pytest.raises(ProviderUnavailable, match="must both be set"):
        resolve_provider(config)


@pytest.mark.parametrize("raw", ["disabled", "garbage", "", "  "])
def test_invalid_enable_is_off_with_warning(monkeypatch, raw):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", raw)
    config = LCMConfig.from_env()
    assert config.embeddings_enabled is False
    if raw.strip():  # Base deliberately emitted no invalid-value warning for blanks.
        assert any("LCM_EMBEDDINGS_ENABLED=" in item and "using False" in item
                   for item in config.config_source_warnings)


@pytest.mark.parametrize("provider", ["fastembed", "voyage", "ollama", "openai-compatible", "siliconflow"])
@pytest.mark.parametrize("flag,expected", [("true", True), ("off", False)])
def test_explicit_enable_values_still_win(monkeypatch, provider, flag, expected):
    monkeypatch.setenv("LCM_EMBEDDING_PROVIDER", provider)
    monkeypatch.setenv("LCM_EMBEDDING_MODEL", "fixture-model")
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", flag)
    selected = LCMConfig.from_env()
    assert selected.embeddings_enabled is expected
    assert selected.embeddings_enabled_explicit is True
    assert selected.embedding_provider == provider and selected.embedding_model == "fixture-model"


def test_ignored_yaml_doctor_names_honoured_key(tmp_path, monkeypatch):
    from tests.harness import engine_context

    monkeypatch.setattr(config_module, "_load_hermes_config_yaml", lambda: {"lcm": {"embeddings_enabled": False}})
    with engine_context(tmp_path, window=40_000, temporal_rollups_enabled=False) as engine:
        engine._config = LCMConfig.from_env()
        payload = json.loads(tools.lcm_doctor({}, engine=engine))
        check = next(check for check in payload["checks"] if check["check"] == "config_validation")
        notes = [note for note in check["detail"] if "lcm.embeddings_enabled" in note]
        assert len(notes) == 1
        assert "LCM_EMBEDDINGS_ENABLED" in notes[0] and "ignored" in notes[0]
        assert "LCM_EMBEDDINGS_ENABLED" in command._doctor_text(engine)


@pytest.mark.parametrize("include", ["all", "summaries"])
def test_absent_prepare_state_schedules_without_turn_model_load(recall_engine, monkeypatch, include):
    engine = recall_engine
    engine._config = LCMConfig(embedding_provider="fastembed", embedding_model=MODEL)
    target = engine._store.append("old", {"role": "user", "content": "cobalt preserved memory"})
    scheduled = []
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: None)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True})
    monkeypatch.setattr(FastembedProvider, "_ensure_local", lambda self: pytest.fail("turn model construction"))
    monkeypatch.setattr(tools, "_lcm_grep_embed_query", lambda *a, **k: pytest.fail("turn semantic query"))
    with monkeypatch.context() as prepare_patch:
        prepare_patch.setattr(embedding_maintenance, "start_local_model_prepare",
                              lambda path, config, **kwargs: scheduled.append((path, config)) or False)
        result = json.loads(tools.lcm_recall({"query": "cobalt", "include": include}, engine=engine))
    assert len(scheduled) == 1 and str(scheduled[0][0]) == str(engine._store.db_path)
    assert result["degraded"] is True and "local model preparing" in result["degraded_reason"]
    assert result["provenance"]["coverage"]["fts"] == "ok"
    assert result["provenance"]["coverage"]["summary"] == "none"
    assert any(hit.get("store_id") == target for hit in result["hits"])


@pytest.mark.parametrize("include", ["all", "summaries"])
@pytest.mark.parametrize("receipt", [False, True])
def test_partial_index_never_certified_without_receipt(recall_engine, monkeypatch, include, receipt):
    engine = recall_engine
    engine._config = LCMConfig(embedding_provider="fastembed", embedding_model=MODEL)
    target = engine._store.append("old", {"role": "user", "content": "cobalt preserved memory"})
    nodes = [engine._dag.add_node(SummaryNode(session_id="old", depth=0, summary="cobalt summary",
                                            created_at=float(i), latest_at=float(i))) for i in range(3)]
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile(MODEL, "fastembed", 2)
        identity = store.capture_identity(MODEL, provider="fastembed")
        store.record_embedding(str(nodes[0]), "summary", MODEL, [1.0, 0.0], identity=identity)
    finally:
        store.close()
    if receipt:
        command._record_embedding_catchup(engine._store.db_path, "summary", identity.identity_hash, 2)
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: ("present", ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True})
    monkeypatch.setattr(tools, "_lcm_grep_embed_query", lambda *a, **k: pytest.fail("partial index semantic query"))
    statements = []
    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    result = json.loads(tools.lcm_recall({"query": "cobalt", "include": include}, engine=engine))
    assert result["degraded"] is True and "catching up" in result["degraded_reason"]
    assert result["provenance"]["coverage"]["summary"] == "none"
    assert result["provenance"]["coverage"]["fts"] == "ok"
    assert any(hit.get("store_id") == target for hit in result["hits"])
    assert not any("count(" in sql.lower() and "lcm_embedding" in sql.lower() for sql in statements)


def test_present_model_with_complete_receipt_keeps_semantic_path(recall_engine, monkeypatch):
    engine = recall_engine
    engine._config = LCMConfig(embedding_provider="fastembed", embedding_model=MODEL)
    engine._store.append("old", {"role": "user", "content": "cobalt memory"})
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile(MODEL, "fastembed", 2)
        identity = store.capture_identity(MODEL, provider="fastembed")
    finally:
        store.close()
    command._record_embedding_catchup(engine._store.db_path, "summary", identity.identity_hash, 0)
    provider = SimpleNamespace(provider_id="fastembed", model_id=MODEL)
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: ("present", ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True})
    monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *a, **k: provider)
    monkeypatch.setattr(tools, "_lcm_grep_embed_query", lambda *a, **k: [1.0, 0.0])
    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", lambda *a, **k: ([], "full", 0, 0, []))
    result = json.loads(tools.lcm_recall({"query": "cobalt", "include": "summaries"}, engine=engine))
    assert result["degraded"] is False
    assert result["provenance"]["coverage"]["summary"] == "full"
