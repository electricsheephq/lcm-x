"""R5: serializable consent, coherent reservation, and real producer readiness."""
# ruff: noqa: F811 -- imported fixtures are injected by pytest
import copy
import importlib.util
import json
import pickle
import sqlite3
import sys
import threading
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import embedding_maintenance as maintenance, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.diagnostics import doctor_guidance_for_check
from tests.test_issue_1186_s3_r4 import (  # noqa: F401
    MARK, REAL_PREPARE, _engine, _seed, transport,
)

FORMS = ["same", "replace", "replace_noop", "replace_change", "asdict", "json", "pickle", "copy", "deepcopy"]


def _copy(config, form):
    if form.startswith("replace"):
        return replace(config, **({"proactive_recall_budget_tokens": config.proactive_recall_budget_tokens}
                                 if form == "replace_noop" else
                                 {"proactive_recall_budget_tokens": 400} if form == "replace_change" else {}))
    if form in {"asdict", "json"}:
        data = asdict(config)
        return LCMConfig(**(json.loads(json.dumps(data)) if form == "json" else data))
    if form == "pickle":
        return pickle.loads(pickle.dumps(config))
    return getattr(copy, form)(config) if form != "same" else config


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("provider,model,warmed", [("voyage", "voyage-3", True), ("ollama", "nomic-embed-text", False)])
def test_copy_forms_preserve_consent_egress(tmp_path, monkeypatch, transport, form, enabled, provider, model, warmed):
    from hermes_lcm import command
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-fixture-key")
    warm = LCMConfig(embeddings_enabled=True, embedding_provider=provider, embedding_model=model)
    engine = _engine(tmp_path, warm)
    try:
        _seed(engine)
        if warmed:
            assert isinstance(command._embedding_register_profiles(warm, warm.database_path, allow_download=False), dict)
            assert "error" not in str(maintenance.run_incremental_embedding_pass(warm.database_path, warm))
        config = LCMConfig(embedding_provider="fastembed", embedding_model=model,
                           proactive_recall_enabled=True, proactive_recall_provider=provider,
                           **({} if enabled is None else {"embeddings_enabled": enabled}))
        engine._config = _copy(config, form)
        sent, sockets = transport
        sent.clear()
        engine._assemble_context(None, [{"role": "user", "content": MARK}])
        assert engine._config._embeddings_enabled_supplied is (enabled is not None)
        assert sockets == []
        assert any(row["carries_mark"] for row in sent) is (enabled is True)
        if enabled is not True:
            assert sent == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("form", FORMS)
def test_env_copy_forms_preserve_provenance(monkeypatch, enabled, form):
    if enabled is not None:
        monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", str(enabled).lower())
    else:
        monkeypatch.delenv("LCM_EMBEDDINGS_ENABLED", raising=False)
    config = LCMConfig.from_env()
    cloned = _copy(config, form)
    assert cloned.embeddings_enabled == config.embeddings_enabled
    assert cloned.embeddings_enabled_explicit == config.embeddings_enabled_explicit
    assert cloned._embeddings_enabled_supplied == config._embeddings_enabled_supplied


@pytest.mark.parametrize("cpus", [1, 8])
def test_inert_override_preserves_assembled_bytes(tmp_path, monkeypatch, transport, cpus):
    from hermes_lcm.dag import SummaryNode
    monkeypatch.setattr(tools, "effective_cpus", lambda: cpus)
    config = LCMConfig.from_env()
    config.proactive_recall_enabled = False
    engine = _engine(tmp_path, config)
    try:
        engine.on_session_start("new", context_length=12000)
        for i in range(200):
            engine._dag.add_node(SummaryNode(session_id="new", depth=1,
                summary=f"memory {i} " + "detailed fact " * 60, token_count=180, created_at=float(i)))
        tail = [{"role": "user", "content": MARK}]
        baseline = engine._assemble_context(None, tail, persist=False)
        config.proactive_recall_enabled = True
        config.proactive_recall_provider = "voyage"
        inert = engine._assemble_context(None, tail, persist=False)
        assert inert == baseline
        assert transport == ([], [])
    finally:
        engine.shutdown()


@pytest.mark.parametrize("provider", ["", "   ", None])
@pytest.mark.parametrize("model", ["", "fixture-model"])
def test_blank_constructor_matches_omitted(provider, model):
    assert asdict(LCMConfig(embedding_provider=provider, embedding_model=model)) == asdict(LCMConfig(embedding_model=model))


def test_held_guidance_uses_own_remedy():
    remedy = "store's profile belongs to voyage; restore that provider's settings with LCM_EMBEDDINGS_ENABLED=true, or switch to the local model with /lcm embed warmup"
    result = doctor_guidance_for_check({"check": "embedding_provider_health", "status": "warn",
        "detail": {"available": True, "reason": remedy, "readiness": "model: preparing;"}})
    assert result["operator_action"] == remedy
    assert "companion" not in result["operator_action"]


def test_doctor_names_inert_override(tmp_path, monkeypatch):
    config = LCMConfig(proactive_recall_enabled=True, proactive_recall_provider="voyage")
    engine = _engine(tmp_path, config)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {
        "available": True, "provider": "fastembed", "model": "test", "detail": "configured"})
    try:
        result = tools._embedding_provider_health_check(engine)
        assert "voyage" in result["detail"]["proactive_recall_note"]
        assert "LCM_EMBEDDINGS_ENABLED=true" in result["detail"]["proactive_recall_note"]
    finally:
        engine.shutdown()


def test_prepare_start_failure_logs_unlocked_and_backs_off(tmp_path, monkeypatch):
    config = LCMConfig()
    starts, locks = [], []
    clock = [1000.0]
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: clock[0])
    def start(thread):
        starts.append(thread.name)
        raise RuntimeError("synthetic thread refusal")
    def warning(*args, **kwargs):
        acquired = maintenance._STATE_LOCK.acquire(blocking=False)
        locks.append(acquired)
        if acquired:
            maintenance._STATE_LOCK.release()
    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(maintenance.logger, "warning", warning)
    assert REAL_PREPARE(tmp_path / "lcm.db", config) is False
    assert locks == [True]
    assert maintenance.local_model_state(config) is None
    assert REAL_PREPARE(tmp_path / "lcm.db", config) is False
    assert len(starts) == 1
    assert list(maintenance._PREPARE_RETRY.values()) == [(1, 1600.0)]
    clock[0] = 1601.0
    assert REAL_PREPARE(tmp_path / "lcm.db", config) is False
    assert len(starts) == 2


@pytest.fixture
def fake_fastembed(monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            pass
        def embed(self, texts, **kwargs):
            return [[1.0] + [0.0] * 383 for _ in texts]
        query_embed = embed
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=Model,
        __spec__=importlib.util.spec_from_loader("fastembed", loader=None)))


def _assert_health(payload):
    assert payload["degraded"] is False, payload
    assert payload["provenance"]["coverage"]["summary"] == "full", payload
    assert payload["provenance"]["coverage"]["chunk"] == "full", payload


@pytest.mark.parametrize("producer", ["bridge", "longmemeval"])
@pytest.mark.parametrize("dedicated_cache", [False, True])
def test_local_producer_has_real_vectors_and_full_recall(tmp_path, monkeypatch, transport, fake_fastembed, producer, dedicated_cache):
    from benchmarking import longmemeval as lme
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("HERMES_LCM_REPO", str(root))
    monkeypatch.setenv("HERMES_MB_WORKDIR", str(tmp_path / "bridge"))
    if dedicated_cache:
        monkeypatch.setenv("LCM_LONGMEMEVAL_FASTEMBED_CACHE", str(tmp_path / "dedicated-cache"))
    monkeypatch.setattr(lme, "_cleanup_question_db", lambda *args, **kwargs: None)
    captured = []
    real_recall = tools.lcm_recall
    def recall(*args, **kwargs):
        raw = real_recall(*args, **kwargs)
        captured.append(json.loads(raw))
        return raw
    monkeypatch.setattr(tools, "lcm_recall", recall)
    session = [{"role": "user", "content": "I learn cello with Marta"},
               {"role": "assistant", "content": "Marta teaches cello"}]
    if producer == "bridge":
        spec = importlib.util.spec_from_file_location("r5bridge", root / "benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py")
        module = importlib.util.module_from_spec(spec)
        original_stdout = sys.stdout
        try:
            spec.loader.exec_module(module)
        finally:
            sys.stdout = original_stdout
        bridge = module.Bridge()
        bridge.initialize({})
        bridge.ingest({"containerTag": "synthetic", "session": {"sessionId": "old", "messages": session}})
        bridge.search({"containerTag": "synthetic", "query": "cello", "limit": 5})
        db = bridge._db_path("synthetic")
    else:
        question = lme.parse_question({"question_id": "synthetic", "question_type": "single-session-user",
            "question": "cello", "haystack_session_ids": ["old"], "haystack_sessions": [session],
            "answer_session_ids": ["old"]})
        provider = lme.resolve_harness_provider("fastembed", "BAAI/bge-small-en-v1.5", use_embed_cache=False)
        lme.evaluate_question(question, provider, provider_name="fastembed", tmp_dir=tmp_path,
                              embeddings_enabled=True, embeddings_enabled_explicit=True, top_k=5)
        db = next(tmp_path.glob("synthetic*.db"))
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM lcm_embedding_vectors").fetchone()[0] > 0
        assert conn.execute("SELECT count(*) FROM lcm_chunk_vectors").fetchone()[0] > 0
    assert captured
    _assert_health(captured[-1])
    assert transport == ([], [])


def _reset_to_omitted(config, form):
    if form == "replace_none":
        return replace(config, embeddings_enabled=None)
    return LCMConfig(**{**asdict(config), "embeddings_enabled": None})


@pytest.mark.parametrize("form", ["replace_none", "asdict_none"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("provider,model,warmed", [("voyage", "voyage-3", True), ("ollama", "nomic-embed-text", False)])
def test_reset_to_omitted_drops_consent(tmp_path, monkeypatch, transport, form, enabled, provider, model, warmed):
    from hermes_lcm import command
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-fixture-key")
    warm = LCMConfig(embeddings_enabled=True, embedding_provider=provider, embedding_model=model)
    engine = _engine(tmp_path, warm)
    try:
        _seed(engine)
        if warmed:
            assert isinstance(command._embedding_register_profiles(warm, warm.database_path, allow_download=False), dict)
            assert "error" not in str(maintenance.run_incremental_embedding_pass(warm.database_path, warm))
        fields = dict(embedding_provider="fastembed", embedding_model=model,
                      proactive_recall_enabled=True, proactive_recall_provider=provider)
        cloned = _reset_to_omitted(LCMConfig(**fields, embeddings_enabled=enabled), form)
        engine._config = cloned
        sent, sockets = transport
        sent.clear()
        engine._assemble_context(None, [{"role": "user", "content": MARK}])
        assert sockets == []
        assert sent == []
        assert asdict(cloned) == asdict(LCMConfig(**fields))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("form", ["replace_none", "asdict_none"])
@pytest.mark.parametrize("enabled", ["false", "true"])
def test_env_reset_to_omitted_drops_consent(monkeypatch, form, enabled):
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", enabled)
    config = LCMConfig.from_env()
    assert config._embeddings_enabled_supplied is True
    cloned = _reset_to_omitted(config, form)
    assert cloned._embeddings_enabled_supplied is False
    assert cloned.embeddings_enabled == LCMConfig(embedding_provider=config.embedding_provider,
                                                  embedding_model=config.embedding_model).embeddings_enabled
