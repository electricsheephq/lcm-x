"""#387/#882: stale cloud identity preserves FTS and doctor stays inert."""
from __future__ import annotations

import json
import socket
import sqlite3
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.embedding_provider as providers
import hermes_lcm.command as command_mod
import hermes_lcm.ingest_protection as privacy
import hermes_lcm.tools as tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
from hermes_lcm.store import MessageStore

REASON = (
    "embedding_identity_stale: stored vectors use an older embedding privacy "
    "revision; run /lcm embed warmup, then /lcm embed backfill --apply"
)
REPEAT_GUIDANCE = (
    "; repeat the backfill apply only while remaining is above 0 and each run lowers it, "
    "but stop and inspect its warnings and details on status: partial, status: error, "
    "status: failed, or if remaining stops falling"
)
TEXT = "Zebrawood orchard has a beagle puppy named Biscuit."


@pytest.fixture
def diagnostic_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "doctor.db"),
        embeddings_enabled=True,
        embedding_provider="voyage",
        embedding_model="voyage-4-large",
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine._session_id = "session-a"
    engine.context_length = 200000
    engine.store_id = engine._store.append("session-a", {"role": "user", "content": TEXT})
    engine.node_id = engine._dag.add_node(SummaryNode(
        session_id="session-a", depth=0, summary=TEXT,
        source_ids=[engine.store_id], source_type="messages", created_at=1, latest_at=1,
    ))
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("stale_tasks", "remedy"),
    [
        (["summary"], "/lcm embed warmup, then /lcm embed backfill --apply"),
        (["chunk"], "/lcm embed warmup, then /lcm embed backfill --corpus chunks --apply --confirm-raw-text"),
        (["summary", "chunk"], "/lcm embed warmup, then /lcm embed backfill --corpus both --apply --confirm-raw-text"),
    ],
)
def test_doctor_stale_corpus_remedy_on_every_surface(diagnostic_engine, monkeypatch, stale_tasks, remedy):
    engine = diagnostic_engine
    _seed(engine, "old" if "summary" in stale_tasks else "matching")
    _seed(engine, "old" if "chunk" in stale_tasks else "matching", task="chunk")
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("doctor attempted to construct an embedding provider")

    monkeypatch.setattr(tools, "resolve_provider", forbidden)
    monkeypatch.setattr(providers, "resolve_provider", forbidden)
    standalone = tools._embedding_provider_health_check(engine)
    payload = json.loads(tools.lcm_doctor({}, engine=engine))
    emitted = next(check for check in payload["checks"] if check["check"] == "embedding_provider_health")
    assert emitted == standalone
    assert emitted["detail"]["stale_tasks"] == stale_tasks
    assert emitted["detail"]["remedy"] == remedy
    reason = (
        "embedding_identity_stale: stored vectors use an older embedding privacy "
        f"revision; run {remedy}"
    )
    if "chunk" in stale_tasks:
        reason += (
            "; --confirm-raw-text acknowledges sending raw message text to the "
            "'voyage' cloud embedding provider"
        )
    reason += REPEAT_GUIDANCE
    assert emitted["detail"]["reason"] == reason
    text = handle_lcm_command("doctor", engine)
    assert f"embedding_provider_health: warn ({reason})" in text


@pytest.mark.parametrize("stale_tasks", [["chunk"], ["summary", "chunk"]])
def test_named_doctor_backfill_rebuilds_stale_chunks(engine, monkeypatch, stale_tasks):
    from tests.test_chunk_backfill import FakeProvider
    import hermes_lcm.chunking as chunking

    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(str(text)))
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))

    store_ids = []
    node_ids = []
    for index in range(2):
        if index:
            engine.store_id = engine._store.append("session-a", {"role": "user", "content": TEXT})
            engine.node_id = engine._dag.add_node(SummaryNode(
                session_id="session-a", depth=0, summary=TEXT,
                source_ids=[engine.store_id], source_type="messages", created_at=1, latest_at=1,
            ))
        _seed(engine, "old" if "summary" in stale_tasks else "matching")
        _seed(engine, "old", task="chunk")
        store_ids.append(engine.store_id)
        node_ids.append(str(engine.node_id))

    parse_options = command_mod._embedding_backfill_options

    def one_row_page(tokens):
        options = parse_options(tokens)
        assert not isinstance(options, str)
        assert options[1] == 200
        return (options[0], 1, *options[2:])

    monkeypatch.setattr(command_mod, "_embedding_backfill_options", one_row_page)
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-test-key")
    instances = {}

    def resolve(config, **kwargs):
        if config.embedding_model not in instances:
            provider = FakeProvider()
            provider.provider_id = "voyage"
            provider.model_id = config.embedding_model
            provider.embed_query = lambda _text: [1.0, 0.0]
            instances[config.embedding_model] = provider
        return instances[config.embedding_model]

    monkeypatch.setattr(command_mod, "resolve_provider", resolve)
    detail = tools._embedding_provider_health_check(engine)["detail"]
    warmup, backfill = detail["remedy"].split(", then ")
    assert "status: ready" in handle_lcm_command(warmup.removeprefix("/lcm "), engine)
    # Execute the emitted arguments unchanged; the handler receives the text after /lcm.
    for page in range(2):
        result = handle_lcm_command(backfill.removeprefix("/lcm "), engine)
        assert "status: complete" in result
        remaining = [int(line.split(": ")[1]) for line in result.splitlines() if line.startswith("remaining: ")]
        assert remaining == [1 - page] * len(stale_tasks)
        conn = sqlite3.connect(engine._store.db_path)
        try:
            current_chunks = conn.execute(
                "SELECT v.chunk_id FROM lcm_chunk_vectors v "
                "JOIN lcm_embedding_profile p ON p.identity_hash = v.identity_hash "
                "WHERE p.active = 1 AND p.task = 'chunk' AND p.revision = ? ORDER BY v.chunk_id",
                (privacy.embedding_privacy_revision(engine._config),),
            ).fetchall()
            assert len(current_chunks) == page + 1
            if page == 1:
                assert current_chunks == sorted((f"{store_id}:0",) for store_id in store_ids)
                assert current_chunks == conn.execute(
                    "SELECT DISTINCT chunk_id FROM lcm_chunk_meta ORDER BY chunk_id"
                ).fetchall()
            assert conn.execute("SELECT store_id, content FROM messages ORDER BY store_id").fetchall() == [
                (store_id, TEXT) for store_id in store_ids
            ]
            if "summary" in stale_tasks:
                current_summaries = conn.execute(
                    "SELECT v.embedded_id FROM lcm_embedding_vectors v "
                    "JOIN lcm_embedding_profile p ON p.identity_hash = v.identity_hash "
                    "WHERE p.active = 1 AND p.task = 'summary' AND p.revision = ? ORDER BY v.embedded_id",
                    (privacy.embedding_privacy_revision(engine._config),),
                ).fetchall()
                assert len(current_summaries) == page + 1
                if page == 1:
                    assert current_summaries == sorted((node_id,) for node_id in node_ids)
        finally:
            conn.close()
    assert instances["voyage-context-4"].calls == [[TEXT], [TEXT]]


def test_doctor_guidance_stops_when_backfill_leaves_overcap_rows_pending(engine, monkeypatch):
    from tests.test_embedding_backfill import FakeProvider

    _seed(engine, "old")
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-test-key")
    reason = tools._embedding_provider_health_check(engine)["detail"]["reason"]
    _seed(engine, "matching")
    node_id = engine._dag.add_node(SummaryNode(
        session_id="session-a", depth=0, summary=TEXT,
        source_ids=[engine.store_id], source_type="messages", created_at=2, latest_at=2,
    ))
    provider = FakeProvider()
    provider.provider_id = "voyage"
    provider.model_id = engine._config.embedding_model

    def skip_overcap(texts):
        provider.calls.append(list(texts))
        provider.last_skipped_documents = list(range(len(texts)))
        return []

    provider.embed_documents = skip_overcap
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _config, **_kw: provider)
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))
    for _ in range(2):
        result = handle_lcm_command("embed backfill --apply", engine)
        assert "status: partial" in result
        assert "embedded: 0" in result
        assert "skipped_overcap: 1" in result
        assert "remaining: 1" in result
        assert f"skipped_detail: node_id={node_id} reason=provider_document_token_cap" in result
    assert provider.calls == [[TEXT], [TEXT]]
    assert "only while remaining is above 0 and each run lowers it" in reason
    assert "stop and inspect its warnings and details on status: partial, status: error" in reason
    assert "status: failed, or if remaining stops falling" in reason


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
        assert detail["reason"] == REASON + REPEAT_GUIDANCE
        assert detail["impact"] == "lcm_recall skips the stale semantic arm until the profile is re-registered"
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


def test_doctor_reports_unavailable_provider_before_stale_identity(engine, monkeypatch):
    # No profile AND no credential: warmup cannot succeed until the provider is available, so the
    # availability blocker is reported, not the stale-identity remedy (#884 review).
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    check = tools._embedding_provider_health_check(engine)
    assert check["status"] == "warn"
    assert check["detail"]["available"] is False
    assert "embedding_identity_stale" not in check["detail"]
    assert "semantic retrieval is degraded" in check["detail"]["impact"]


@pytest.mark.parametrize("state", ["old", "missing"])
def test_stale_summaries_only_recall_falls_back_to_fts(engine, monkeypatch, state):
    _seed(engine, state)
    instances = _providers(monkeypatch)

    def no_scan(*args, **kwargs):
        pytest.fail("stale arm read stored vectors")

    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", no_scan)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine))
    assert engine.store_id in [hit["store_id"] for hit in payload["hits"]]
    assert payload["degraded"] is True
    assert payload["degraded_reason"] == REASON
    coverage = payload["provenance"]["coverage"]
    assert coverage["fts"] == "ok"
    assert coverage["summary"] == "none"
    assert sum(len(provider.queries) for provider in instances.values()) == 0


@pytest.mark.parametrize(
    ("stage", "failure", "reason"),
    [
        ("resolve", None, "embedding provider is not configured"),
        ("query", tools.VoyageError, "query embedding failed: synthetic outage"),
        ("resolve", TimeoutError, None),
        ("query", TimeoutError, None),
        ("resolve", RuntimeError, "embedding provider unavailable: synthetic outage"),
        ("query", RuntimeError, "embedding provider unavailable: synthetic outage"),
        ("scan", TimeoutError, None),
        ("scan", RuntimeError, "summary arm failed: synthetic outage"),
    ],
    ids=["unconfigured", "voyage-error", "resolution-timeout", "query-timeout",
         "resolution-error", "query-error", "scan-timeout", "scan-error"],
)
def test_unavailable_summaries_only_recall_falls_back_to_fts(engine, monkeypatch, stage, failure, reason):
    _seed(engine, "matching")
    _providers(monkeypatch)
    calls = []
    fts = tools._lcm_recall_fts_arm

    def capture_fts(*args, **kwargs):
        calls.append(kwargs["deadline"])
        return fts(*args, **kwargs)

    def unavailable(*args, **kwargs):
        # Simulate a timeout without consuming the remaining request budget.
        if failure is tools.VoyageError:
            raise failure("unavailable", "synthetic outage")
        if failure is not None:
            raise failure("synthetic outage")
        return None

    target = {"resolve": "resolve_provider", "query": "_lcm_grep_embed_query",
              "scan": "_lcm_recall_summary_arm"}[stage]
    monkeypatch.setattr(tools, target, unavailable)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", capture_fts)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine))
    assert engine.store_id in [hit["store_id"] for hit in payload["hits"]]
    assert len(calls) == 1
    coverage = payload["provenance"]["coverage"]
    assert coverage == ({"summary": "none", "fts": "ok"} if stage == "scan" else {"fts": "ok"})
    assert payload["degraded"] is (reason is not None)
    assert payload.get("degraded_reason") == reason
    assert payload.get("timeout", False) is (failure is TimeoutError)


@pytest.mark.parametrize("stage", ["query", "scan"])
def test_summaries_only_fallback_gets_the_whole_remaining_budget(engine, monkeypatch, stage):
    # The semantic arm has already failed, so the fallback does not keep the cap reserved for it.
    _seed(engine, "matching")
    _providers(monkeypatch)
    now = [1000.0]
    monkeypatch.setattr(tools.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tools, "_lcm_recall_has_usable_vector_corpus", lambda *args, **kwargs: True)
    deadlines = []
    fts = tools._lcm_recall_fts_arm

    def capture_fts(*args, **kwargs):
        deadlines.append(kwargs["deadline"])
        return fts(*args, **kwargs)

    def unavailable(*args, **kwargs):
        raise RuntimeError("synthetic outage")

    monkeypatch.setattr(tools, "_lcm_grep_embed_query" if stage == "query" else "_lcm_recall_summary_arm", unavailable)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", capture_fts)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine))
    assert engine.store_id in [hit["store_id"] for hit in payload["hits"]]
    assert deadlines == [pytest.approx(1000.0 + engine._config.recall_query_timeout_s)]


def test_summaries_only_recall_without_summary_vectors_falls_back_to_fts(engine, monkeypatch):
    # A new embeddings-on store has no summary vectors until background embedding runs (#1014).
    _seed(engine, "matching")
    _providers(monkeypatch)
    calls = []
    fts = tools._lcm_recall_fts_arm

    def capture_fts(*args, **kwargs):
        calls.append(kwargs["deadline"])
        return fts(*args, **kwargs)

    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", lambda *args, **kwargs: ([], "none", 0, 0, []))
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", capture_fts)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine))
    assert engine.store_id in [hit["store_id"] for hit in payload["hits"]]
    assert len(calls) == 1
    assert payload["provenance"]["coverage"] == {"summary": "none", "fts": "ok"}
    assert payload["degraded_reason"] == "summary vectors are unavailable"


@pytest.mark.parametrize("stage", ["query", "scan"])
def test_summaries_only_privacy_error_still_raises(engine, monkeypatch, stage):
    _seed(engine, "matching")
    _providers(monkeypatch)

    def invalid(*args, **kwargs):
        raise privacy.EmbeddingPrivacyPolicyError("synthetic invalid policy")

    def no_fts(*args, **kwargs):
        pytest.fail("privacy-policy error ran the full-text fallback")

    target = "_lcm_grep_embed_query" if stage == "query" else "_lcm_recall_summary_arm"
    monkeypatch.setattr(tools, target, invalid)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", no_fts)
    with pytest.raises(privacy.EmbeddingPrivacyPolicyError, match="synthetic invalid policy"):
        tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine)


@pytest.mark.parametrize("stage", ["resolve", "query", "scan"])
@pytest.mark.parametrize("failure", [TimeoutError, RuntimeError])
def test_summaries_only_failure_after_deadline_runs_no_fts(engine, monkeypatch, stage, failure):
    _seed(engine, "matching")
    _providers(monkeypatch)
    now = [time.monotonic()]
    monkeypatch.setattr(tools.time, "monotonic", lambda: now[0])

    def expired(*args, **kwargs):
        now[0] += engine._config.recall_query_timeout_s + 1
        raise failure("synthetic outage")

    def no_fts(*args, **kwargs):
        pytest.fail("full-text fallback started after the request deadline")

    target = {"resolve": "resolve_provider", "query": "_lcm_grep_embed_query",
              "scan": "_lcm_recall_summary_arm"}[stage]
    monkeypatch.setattr(tools, target, expired)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", no_fts)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine))
    assert "fts" not in payload["provenance"]["coverage"]
    assert payload["hits"] == []
    assert payload.get("timeout", False) is (failure is TimeoutError)


def test_matching_summaries_only_recall_runs_no_fts_arm(engine, monkeypatch):
    _seed(engine, "matching")
    _providers(monkeypatch)

    def no_fts(*args, **kwargs):
        pytest.fail("summaries-only recall with a usable identity ran the FTS arm")

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", no_fts)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine))
    assert "fts" not in payload["provenance"]["coverage"]
    assert payload["provenance"]["coverage"]["summary"] == "full"
    assert payload["hits"]
    assert payload["degraded"] is False


@pytest.mark.parametrize(
    ("summary", "chunk", "expected"),
    [
        ("matching", "old", ["chunk"]),
        ("old", "old", ["summary", "chunk"]),
        ("matching", "matching", None),
        ("matching", None, None),
    ],
)
def test_doctor_reports_a_stale_chunk_identity_when_chunk_vectors_exist(engine, monkeypatch, summary, chunk, expected):
    _seed(engine, summary)
    if chunk is not None:
        _seed(engine, chunk, task="chunk")
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        raise AssertionError("doctor constructed a provider")

    monkeypatch.setattr(tools, "resolve_provider", forbidden)
    monkeypatch.setattr(providers, "resolve_provider", forbidden)
    check = tools._embedding_provider_health_check(engine)
    if expected is None:
        assert check["status"] == "pass"
    else:
        assert check["status"] == "warn"
        assert check["detail"]["stale_tasks"] == expected
        corpus = "both" if "summary" in expected else "chunks"
        assert check["detail"]["reason"] == (
            REASON.replace("backfill --apply", f"backfill --corpus {corpus} --apply --confirm-raw-text")
            + "; --confirm-raw-text acknowledges sending raw message text to the "
            "'voyage' cloud embedding provider"
            + REPEAT_GUIDANCE
        )


@pytest.mark.parametrize("orphan", ["vector", "source"])
def test_doctor_ignores_stale_chunk_metadata_recall_cannot_use(engine, monkeypatch, orphan):
    _seed(engine, "matching")
    _seed(engine, "old", task="chunk")
    conn = sqlite3.connect(engine._store.db_path)
    try:
        if orphan == "vector":
            conn.execute("DELETE FROM lcm_chunk_vectors")
        else:
            conn.execute("UPDATE lcm_chunk_meta SET store_id = store_id + 1000000")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-test-key")
    check = tools._embedding_provider_health_check(engine)
    assert check["status"] == "pass"


def test_summary_scan_stale_race_falls_back_to_fts_for_summaries_only(engine, monkeypatch):
    _seed(engine, "matching")
    _providers(monkeypatch)

    def drifted(*args, **kwargs):
        raise privacy.EmbeddingIdentityStaleError("identity changed before the scan")

    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", drifted)
    payload = json.loads(tools.lcm_recall({"query": "Zebrawood", "include": "summaries"}, engine=engine))
    assert engine.store_id in [hit["store_id"] for hit in payload["hits"]]
    coverage = payload["provenance"]["coverage"]
    assert coverage["fts"] == "ok"
    assert coverage["summary"] == "none"
    assert payload["degraded_reason"] == REASON
