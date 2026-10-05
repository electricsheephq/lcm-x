"""#387: convert privacy refusals only at the public tool boundary."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import ingest_protection, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import (
    EmbeddingPrivacyPolicyError,
    embedding_privacy_revision,
)
from hermes_lcm.vector_store import VectorStore


QUERY = "Constellation anchor"
PRIVACY_CODE = "embedding_privacy_policy"


class MockProvider:
    provider_id = "voyage"
    model_id = "mock-model"
    dim = 2

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def privacy_engine(tmp_path, monkeypatch):
    monkeypatch.delenv("LCM_DISABLED_TOOLS", raising=False)
    provider = MockProvider()
    monkeypatch.setattr(tools, "resolve_provider", lambda _config: provider)
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=2,
        embeddings_enabled=True,
        embedding_provider=provider.provider_id,
        embedding_model=provider.model_id,
        embedding_query_timeout_s=2.0,
        sensitive_patterns_enabled=True,
        proactive_recall_enabled=True,
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start("session-current", platform="discord", conversation_id="c:1")
    engine._store.append_batch(
        "session-current", [{"role": "user", "content": QUERY}],
        source="discord", conversation_id="c:1",
    )
    now = time.time()
    node = engine._dag.add_node(SummaryNode(
        session_id="session-other", depth=0, summary=QUERY,
        token_count=20, source_token_count=40, source_ids=[],
        source_type="messages", created_at=now, earliest_at=now, latest_at=now,
    ))
    store = VectorStore(engine._store.db_path, config=config)
    try:
        store.register_profile(
            provider.model_id, provider.provider_id, provider.dim,
            revision=embedding_privacy_revision(config) + "-stale",
        )
        store.register_profile(
            provider.model_id, provider.provider_id, provider.dim,
            revision=embedding_privacy_revision(config) + "-stale", task="chunk",
        )
        identity = store.capture_identity(provider.model_id, provider=provider.provider_id)
        store.record_embedding(str(node), "summary", provider.model_id, [1.0, 0.0], identity=identity)
    finally:
        store.close()
    try:
        yield engine, provider
    finally:
        engine.shutdown()


def test_tool_call_recall_returns_privacy_envelope_not_raise(privacy_engine):
    engine, _provider = privacy_engine
    raw = engine.handle_tool_call("lcm_recall", {"query": QUERY})
    assert isinstance(raw, str)
    payload = json.loads(raw)
    assert payload["error_code"] == PRIVACY_CODE
    assert payload["tool"] == "lcm_recall"
    assert "/lcm embed warmup" in payload["error"]
    assert payload["remediation"]
    assert "hits" not in payload


def test_direct_lcm_recall_still_raises_privacy_error(privacy_engine):
    engine, _provider = privacy_engine
    with pytest.raises(EmbeddingPrivacyPolicyError, match="registered vector identity"):
        tools.lcm_recall({"query": QUERY}, engine=engine)


def test_proactive_recall_counts_real_privacy_error(privacy_engine, caplog):
    engine, _provider = privacy_engine
    assert engine._build_proactive_recall_message(
        [{"role": "user", "content": QUERY}], "user", set(),
    ) is None
    assert engine._proactive_recall_privacy_error_count == 1
    status = json.loads(engine.handle_tool_call("lcm_status", {}))
    assert status["proactive_recall"]["privacy_policy_errors"] == 1
    assert "registered vector identity" in caplog.text


def test_tool_call_other_exceptions_still_propagate(privacy_engine, monkeypatch):
    engine, _provider = privacy_engine

    def fail(args, *, engine):
        raise ValueError("boom")

    monkeypatch.setattr(tools, "lcm_recall", fail)
    with pytest.raises(ValueError, match="boom"):
        engine.handle_tool_call("lcm_recall", {"query": QUERY})


def _grep_privacy_degrade(engine, mode):
    payload = json.loads(engine.handle_tool_call("lcm_grep", {"query": QUERY, "mode": mode}))
    assert payload["degraded_to_fts"] is True
    assert payload["degraded_code"] == PRIVACY_CODE
    assert payload["degraded_reason"] == (
        "query embedding failed: cloud embedding privacy policy differs from registered vector "
        "identity; run `/lcm embed warmup` before semantic retrieval"
    )
    assert payload["results"]
    hit = payload["results"][0]
    assert hit["type"] == "message"
    assert hit["session_id"] == "session-current"
    assert hit["snippet"].replace(">>>", "").replace("<<<", "") == QUERY


def test_grep_semantic_privacy_degrade_carries_code(privacy_engine):
    _grep_privacy_degrade(privacy_engine[0], "semantic")


def test_grep_hybrid_privacy_degrade_carries_code(privacy_engine):
    _grep_privacy_degrade(privacy_engine[0], "hybrid")


def test_grep_transient_embed_failure_has_no_privacy_code(privacy_engine, monkeypatch):
    engine, provider = privacy_engine
    store = VectorStore(engine._store.db_path, config=engine._config)
    try:
        store.register_profile(
            provider.model_id, provider.provider_id, provider.dim,
            revision=embedding_privacy_revision(engine._config),
        )
    finally:
        store.close()

    def fail(text):
        raise RuntimeError("net")

    monkeypatch.setattr(provider, "embed_query", fail)
    payload = json.loads(engine.handle_tool_call("lcm_grep", {"query": QUERY, "mode": "semantic"}))
    assert payload["degraded_to_fts"] is True
    assert "degraded_code" not in payload
    assert payload["degraded_reason"] == "query embedding failed: net"


def _load_module(path, name, *, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if package:
        module.__package__ = package
    spec.loader.exec_module(module)
    return module


def _envelope():
    return json.dumps({"error": "privacy refused", "error_code": PRIVACY_CODE})


def test_answer_ready_baseline_reraises_privacy_envelope():
    plugin = _load_module(Path(__file__).resolve().parents[1] / "__init__.py", "issue387_plugin", package="hermes_lcm")
    stub = SimpleNamespace(handle_tool_call=lambda *_args: _envelope())
    with pytest.raises(EmbeddingPrivacyPolicyError, match="privacy refused"):
        plugin._answer_ready_baseline(stub, "q", {})
    stub.handle_tool_call = lambda *_args: json.dumps({"error": "other"})
    assert plugin._answer_ready_baseline(stub, "q", {}) == ()


def test_pre_llm_context_privacy_refusal_injects_nothing(privacy_engine, caplog):
    engine, _provider = privacy_engine
    engine._config.preanswer_evidence_enabled = True
    engine._config.preanswer_evidence_mode = "requirements_v1"
    plugin = _load_module(Path(__file__).resolve().parents[1] / "__init__.py", "issue387_hook", package="hermes_lcm")
    assert plugin._pre_llm_context(engine, {"user_message": "What is the Constellation anchor?"}) is None
    assert "pre-answer evidence failed open" in caplog.text
    assert "registered vector identity" in caplog.text


class MisconfigurationEngine:
    def __init__(self, behavior="envelope"):
        self._config = SimpleNamespace()
        self._proactive_recall_privacy_error_count = 0
        self.behavior = behavior
        self.closed = False

    def handle_tool_call(self, name, args):
        if name == "lcm_status":
            return json.dumps({"proactive_recall": {"privacy_policy_errors": self._proactive_recall_privacy_error_count}})
        assert name == "lcm_recall"
        if self.behavior == "raise":
            raise EmbeddingPrivacyPolicyError("privacy refused")
        if self.behavior == "hits":
            return json.dumps({"hits": [{"snippet": QUERY}]})
        return _envelope()

    def _build_proactive_recall_message(self, *args):
        self._proactive_recall_privacy_error_count += 1
        return None

    def shutdown(self):
        self.closed = True


def _misconfiguration_harness(monkeypatch, behavior="envelope"):
    path = Path(__file__).resolve().parents[1] / "bench/instruments/release_gauntlet/phase_a_tool_matrix.py"
    phase_a = _load_module(path, "issue387_phase_a")
    engine = MisconfigurationEngine(behavior)
    monkeypatch.setattr(phase_a, "_engine", lambda *_args, **_kwargs: engine)
    monkeypatch.setattr(phase_a, "_seed", lambda *_args: None)
    return phase_a, engine


def test_phase_a_misconfiguration_accepts_envelope(tmp_path, monkeypatch):
    phase_a, engine = _misconfiguration_harness(monkeypatch)
    phase_a._misconfiguration({"ingest_protection": ingest_protection}, tmp_path)
    assert engine.closed
    assert engine._proactive_recall_privacy_error_count == 1


@pytest.mark.parametrize("behavior", ["raise", "hits"])
def test_phase_a_misconfiguration_rejects_raise_and_hits(tmp_path, monkeypatch, behavior):
    phase_a, engine = _misconfiguration_harness(monkeypatch, behavior)
    with pytest.raises((EmbeddingPrivacyPolicyError, AssertionError)):
        phase_a._misconfiguration({"ingest_protection": ingest_protection}, tmp_path)
    assert engine.closed
