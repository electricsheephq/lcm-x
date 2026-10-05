"""#387/#882: stale cloud identity preserves FTS and doctor stays inert."""
from __future__ import annotations

import json
import socket
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.embedding_provider as providers
import hermes_lcm.ingest_protection as privacy
import hermes_lcm.tools as tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
from hermes_lcm.store import MessageStore

REASON = (
    "embedding_identity_stale: stored vectors use an older embedding privacy "
    "revision; run /lcm embed warmup, then /lcm embed backfill --apply"
)
TEXT = "Zebrawood orchard has a beagle puppy named Biscuit."


class CloudProvider:
    provider_id = "voyage"

    def __init__(self, model_id):
        self.model_id = model_id
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0]


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "stale.db"),
        embeddings_enabled=True,
        embedding_provider="voyage",
        embedding_model="voyage-4-large",
        sensitive_patterns=["api_key", "private_key"],
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config, _store=store, _dag=dag,
        current_session_id="session-a", _hermes_home=str(tmp_path),
    )
    engine.store_id = store.append("session-a", {"role": "user", "content": TEXT})
    engine.node_id = dag.add_node(SummaryNode(
        session_id="session-a", depth=0, summary=TEXT,
        source_ids=[engine.store_id], source_type="messages",
        created_at=time.time(), latest_at=time.time(),
    ))
    try:
        yield engine
    finally:
        dag.close()
        store.close()


def _seed(engine, state, *, task="summary", provider="voyage", model=None):
    model = model or ("voyage-4-large" if task == "summary" else "voyage-context-4")
    revision = privacy.embedding_privacy_revision(engine._config)
    if state == "old":
        revision = revision.replace("privacy:v3:", "privacy:v2:", 1)
    vs = VectorStore(engine._store.db_path, config=engine._config)
    try:
        if state == "missing":
            return
        vs.register_profile(model, provider, 2, revision=revision, task=task)
        identity = EmbeddingIdentity.canonical(provider, model, revision, 2, "float32", "little", task)
        if task == "summary":
            vs.record_embedding(str(engine.node_id), "summary", model, [1.0, 0.0], identity=identity)
        else:
            vs.record_chunk_embedding(
                f"{engine.store_id}:0", model, [1.0, 0.0],
                store_id=engine.store_id, chunk_index=0, char_start=0,
                char_end=len(TEXT), token_estimate=10, identity=identity,
            )

    finally:
        vs.close()


def _providers(monkeypatch, *, provider_id="voyage"):
    instances = {}

    def resolve(config):
        provider = instances.setdefault(config.embedding_model, CloudProvider(config.embedding_model))
        provider.provider_id = provider_id
        return provider

    monkeypatch.setattr(tools, "resolve_provider", resolve)
    return instances


def _preflight(engine, *, summary=True, chunk=True):
    return tools._lcm_recall_has_usable_vector_corpus(
        engine, run_summary=summary, run_chunk=chunk,
        provider_name=engine._config.embedding_provider,
        model_name=engine._config.embedding_model,
        excluded_session_ids=set(), deadline=time.monotonic() + 8,
    )


@pytest.mark.parametrize("state", ["old", "missing"])
@pytest.mark.parametrize("include", ["all", "verbatim"])
def test_stale_recall_keeps_fts_without_embed_or_vector_reads(engine, monkeypatch, state, include):
    _seed(engine, state)
    _seed(engine, state, task="chunk")
    instances = _providers(monkeypatch)

    def no_scan(*args, **kwargs):
        pytest.fail("stale arm read stored vectors")

    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", no_scan)
    monkeypatch.setattr(tools, "_lcm_recall_chunk_arm", no_scan)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": include}, engine=engine))
    assert engine.store_id in [hit["store_id"] for hit in payload["hits"]]
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == REASON
    coverage = payload["provenance"]["coverage"]
    assert coverage["fts"] == "ok"
    assert coverage["chunk"] == "none"
    if include == "all":
        assert coverage["summary"] == "none"
    assert sum(len(provider.queries) for provider in instances.values()) == 0
    assert payload["metrics"]["embedding_query_calls"] == 0


def test_policy_invalid_still_raises_base_error(engine, monkeypatch):
    _seed(engine, "matching")
    _seed(engine, "matching", task="chunk")
    instances = _providers(monkeypatch)
    engine._config.sensitive_patterns = ["api_key", "unknown_sensitive_pattern"]
    # Invalid policy is still handled by the later query guard, not preflight.
    assert _preflight(engine) is True
    with pytest.raises(privacy.EmbeddingPrivacyPolicyError, match="unknown") as raised:
        tools.lcm_recall({"query": "Zebrawood"}, engine=engine)
    assert type(raised.value) is privacy.EmbeddingPrivacyPolicyError
    assert sum(len(provider.queries) for provider in instances.values()) == 0


@pytest.mark.parametrize("provider_id", ["voyage", "ollama"])
def test_matching_cloud_and_local_keep_vector_arms(engine, monkeypatch, provider_id):
    engine._config.embedding_provider = provider_id
    _seed(engine, "matching", provider=provider_id)
    chunk_model = providers.default_chunk_model(provider_id, engine._config.embedding_model)
    _seed(engine, "matching", provider=provider_id, model=chunk_model, task="chunk")
    if provider_id == "ollama":
        engine._config.sensitive_patterns = ["unknown_sensitive_pattern"]
    instances = _providers(monkeypatch, provider_id=provider_id)
    assert _preflight(engine) is True
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood"}, engine=engine))
    assert payload["degraded"] is False
    assert payload["provenance"]["coverage"] == {"fts": "ok", "summary": "full", "chunk": "full"}
    assert sum(len(provider.queries) for provider in instances.values()) == (2 if provider_id == "voyage" else 1)


@pytest.mark.parametrize("task", ["summary", "chunk"])
@pytest.mark.parametrize("state", ["old", "missing", "archived"])
def test_stale_preflight_is_not_usable(engine, task, state):
    _seed(engine, "matching" if state == "archived" else state, task=task)
    if state == "archived":
        engine._dag.connection.execute("UPDATE lcm_embedding_profile SET active=0, archived_at=1")
        engine._dag.connection.commit()
    assert _preflight(engine, summary=task == "summary", chunk=task == "chunk") is False


def test_stale_preflight_keeps_full_fts_deadline(engine, monkeypatch):
    _seed(engine, "old")
    _seed(engine, "old", task="chunk")
    _providers(monkeypatch)
    deadlines = {}
    preflight = tools._lcm_recall_has_usable_vector_corpus
    fts = tools._lcm_recall_fts_arm

    def capture_preflight(*args, **kwargs):
        deadlines["request"] = kwargs["deadline"]
        return preflight(*args, **kwargs)

    def capture_fts(*args, **kwargs):
        deadlines["fts"] = kwargs["deadline"]
        return fts(*args, **kwargs)

    monkeypatch.setattr(tools, "_lcm_recall_has_usable_vector_corpus", capture_preflight)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", capture_fts)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood"}, engine=engine))
    assert payload["hits"]
    assert deadlines["fts"] == deadlines["request"]


@pytest.mark.parametrize("state", ["old", "missing", "matching", "invalid", "local", "off", "off-profile"])
def test_doctor_identity_health_is_inert(engine, monkeypatch, state):
    if state not in {"missing", "off"}:
        _seed(engine, "old" if state == "old" else "matching")
    if state == "local":
        engine._config.embedding_provider = "ollama"
        engine._config.sensitive_patterns = ["unknown_sensitive_pattern"]
    if state.startswith("off"):
        engine._config.embeddings_enabled = False
    if state == "invalid":
        engine._config.sensitive_patterns = ["api_key", "unknown_sensitive_pattern"]
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-test-key")
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        raise AssertionError("doctor constructed a provider, loaded a model or used the network")

    monkeypatch.setattr(tools, "resolve_provider", forbidden)
    monkeypatch.setattr(providers, "resolve_provider", forbidden)
    monkeypatch.setattr(providers, "_load_fastembed", forbidden)
    monkeypatch.setattr(providers.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    check = tools._embedding_provider_health_check(engine)
    assert forbidden_calls == []
    if state in {"old", "missing"}:
        assert check["status"] == "warn"
        detail = check["detail"]
        assert detail["embedding_identity_stale"] is True
        assert detail["reason"] == REASON
        assert detail["impact"] == "lcm_recall serves full-text only until the profile is re-registered"
        assert detail["remedy"] == "/lcm embed warmup, then /lcm embed backfill --apply"
    elif state == "invalid":
        assert check["status"] == "warn"
        assert "unknown_sensitive_pattern" in check["detail"]["reason"]
        assert "embedding_identity_stale" not in check["detail"]
    elif state == "off-profile":
        assert check["status"] == "warn"
        assert check["detail"]["embeddings_enabled"] is False
        assert "embedding_identity_stale" not in check["detail"]
    else:
        assert check["status"] == "pass"


@pytest.mark.parametrize("state", ["old", "missing"])
def test_semantic_grep_stale_degrades_to_fts(engine, monkeypatch, state):
    _seed(engine, state)
    instances = _providers(monkeypatch)
    payload = json.loads(tools.lcm_grep({
        "query": "Zebrawood", "mode": "semantic", "session_scope": "current",
    }, engine=engine))
    assert payload["degraded_to_fts"] is True
    assert payload["degraded_reason"] == REASON
    assert engine.store_id in [hit.get("store_id") for hit in payload["results"]]
    assert sum(len(provider.queries) for provider in instances.values()) == 0


@pytest.mark.parametrize("arm", ["summary", "chunk"])
def test_arm_stale_errors_degrade_but_policy_errors_raise(engine, monkeypatch, arm):
    _seed(engine, "matching")
    _seed(engine, "matching", task="chunk")
    _providers(monkeypatch)

    def stale(*args, **kwargs):
        raise privacy.EmbeddingIdentityStaleError("identity changed before scan")

    monkeypatch.setattr(tools, f"_lcm_recall_{arm}_arm", stale)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood"}, engine=engine))
    assert payload["hits"]
    assert payload["degraded_reason"] == REASON
    assert payload["provenance"]["coverage"][arm] == "none"

    def invalid(*args, **kwargs):
        raise privacy.EmbeddingPrivacyPolicyError("dispatch policy invalid")

    monkeypatch.setattr(tools, f"_lcm_recall_{arm}_arm", invalid)
    with pytest.raises(privacy.EmbeddingPrivacyPolicyError, match="dispatch policy invalid"):
        tools.lcm_recall({"query": "Zebrawood"}, engine=engine)


@pytest.mark.parametrize("state", ["old", "missing"])
def test_shared_cloud_query_checks_chunk_identity_before_scan(engine, monkeypatch, state):
    # Voyage context models and OpenAI-compatible models can share one embed.
    # A healthy summary profile must not bypass the chunk profile's identity.
    model = "voyage-context-4"
    engine._config.embedding_model = model
    _seed(engine, "matching", model=model)
    _seed(engine, state, model=model, task="chunk")
    instances = _providers(monkeypatch)

    def no_scan(*args, **kwargs):
        pytest.fail("shared query bypassed stale chunk identity")

    monkeypatch.setattr(tools, "_lcm_recall_chunk_arm", no_scan)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood"}, engine=engine))
    assert payload["hits"]
    assert payload["degraded_reason"] == REASON
    assert payload["provenance"]["coverage"]["chunk"] == "none"
    assert len(instances[model].queries) == 1  # Healthy summary arm only.
