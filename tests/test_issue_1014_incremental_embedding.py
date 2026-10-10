"""#1014: new summaries and chunks get vectors in background maintenance.

Before #1014 only `/lcm embed backfill` wrote vectors, so every summary or chunk
published after a manual backfill stayed unembedded. These tests drive the real
engine and drain its incremental embedding scheduler; no manual backfill runs
unless a test says so.
"""

from __future__ import annotations

import gc
import json
import io
import logging
import sqlite3
import threading
import time
import urllib.error
import weakref
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.embedding_maintenance as maintenance_mod
import hermes_lcm.embedding_provider as provider_mod
import hermes_lcm.engine as lcm_engine
import hermes_lcm.tools as lcm_tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.embedding_provider import EmbeddedDocumentBatch, ProviderUnavailable
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import embedding_privacy_revision
from hermes_lcm.vector_store import VectorStore

SESSION = "incremental-session"
FILLER = " ".join(f"detail{index}" for index in range(60))


class FakeProvider:
    """Deterministic offline provider; every document gets the same vector."""

    def __init__(self, provider_id: str, model_id: str):
        self.provider_id = provider_id
        self.model_id = model_id
        self.dim = 2
        self.documents: list[str] = []
        self.last_skipped_documents: list[int] = []
        self.last_usage_tokens = 1

    def embed_document_batches(self, texts, *, before_dispatch):
        indexes = tuple(range(len(texts)))
        before_dispatch(indexes)
        self.documents.extend(texts)
        yield EmbeddedDocumentBatch(indexes, tuple((1.0, 0.0) for _ in texts))

    def embed_query(self, text):
        return [1.0, 0.0]


class UnavailableProvider(FakeProvider):
    """Fails the way FastEmbed does when its model cannot load: before dispatch."""

    def embed_document_batches(self, texts, *, before_dispatch):
        raise ProviderUnavailable("FastEmbed is not installed")
        yield  # pragma: no cover


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "hermes-home"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


@pytest.fixture
def providers(monkeypatch):
    made: list[FakeProvider] = []

    def resolve(config, **_kwargs):
        provider = FakeProvider(config.embedding_provider, config.embedding_model)
        made.append(provider)
        return provider

    monkeypatch.setattr(command_mod, "resolve_provider", resolve)
    return made


def _make_engine(tmp_path, home, *, enabled=True, provider="ollama", register=True):
    config = LCMConfig(
        database_path=str(tmp_path / "incremental.db"),
        fresh_tail_count=4,
        leaf_chunk_tokens=1,
        embeddings_enabled=enabled,
        embedding_provider=provider,
        embedding_model="model-a",
    )
    engine = LCMEngine(config=config, hermes_home=str(home))
    if enabled and register:
        revision = embedding_privacy_revision(config) if provider == "voyage" else ""
        store = VectorStore(config.database_path, config=config)
        try:
            store.register_profile("model-a", provider, 2, revision=revision)
            store.register_profile("model-a", provider, 2, revision=revision, task="chunk")
        finally:
            store.close()
    return engine


def _drive_until_leaf(engine, monkeypatch):
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **_kwargs: ("Zebra migration ledger summary.\nExpand for details about: zebras", 1),
    )
    engine.on_session_start(
        SESSION, platform="cli", conversation_id="incremental-conversation",
        context_length=200000,
    )
    messages = [{"role": "system", "content": "You are concise."}]
    for turn in range(6):
        messages.append({"role": "user", "content": f"user turn {turn} {FILLER}"})
        messages.append({"role": "assistant", "content": f"assistant turn {turn} {FILLER}"})
        messages = engine.compress(messages)
        if _leaf_ids(engine):
            return
    raise AssertionError("no leaf summary was published")


def _leaf_ids(engine) -> list[str]:
    return sorted(
        str(node.node_id)
        for node in engine._dag.get_session_nodes(SESSION, limit=1000)
        if node.depth == 0
    )


def _rows(engine, sql) -> list[tuple]:
    conn = sqlite3.connect(engine._config.database_path)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _summary_vector_ids(engine) -> list[str]:
    return sorted(
        str(row[0]) for row in _rows(
            engine,
            "SELECT embedded_id FROM lcm_embedding_meta WHERE embedded_kind = 'summary'",
        )
    )


def _chunk_vector_count(engine) -> int:
    try:
        return int(_rows(engine, "SELECT COUNT(*) FROM lcm_chunk_meta")[0][0])
    except sqlite3.OperationalError:
        return 0


def _inflight_count(engine) -> int:
    try:
        return int(_rows(engine, "SELECT COUNT(*) FROM lcm_embedding_backfill_inflight")[0][0])
    except sqlite3.OperationalError:
        return 0


def _drain(engine):
    assert engine.drain_embedding_maintenance(timeout=30)


def _add_leaf(engine, text):
    node = SummaryNode(session_id=SESSION, depth=0, summary=text, created_at=1.0, latest_at=1.0)
    node_id = engine._dag.add_node(node)
    engine._invalidate_rollups_for_published_node(engine._dag.get_node(node_id))
    return str(node_id)


# #1063 exercises the real FastembedProvider cached-only construction path.
@pytest.fixture
def local_fastembed(monkeypatch):
    calls = []
    state = {"cached": True}

    class Model:
        def query_embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        embed = query_embed

    def construct(self, *, allow_download):
        calls.append(allow_download)
        if not state["cached"]:
            raise provider_mod.ProviderNotWarmedUp("offline fake model is not cached")
        return Model()

    monkeypatch.setattr(provider_mod.FastembedProvider, "_construct", construct)
    monkeypatch.setattr(
        provider_mod, "probe_provider_availability",
        lambda _config: {"available": True, "detail": "offline fake"},
    )
    return calls, state


def _local_engine(tmp_path, home, provider="fastembed"):
    engine = _make_engine(tmp_path, home, provider=provider, register=False)
    engine._config.embedding_model = "BAAI/bge-small-en-v1.5"
    return engine


def _profiles(engine):
    try:
        return _rows(engine, "SELECT task, active, identity_hash FROM lcm_embedding_profile")
    except sqlite3.OperationalError:
        return []


@pytest.mark.parametrize("provider", ["fastembed", "fast-embed"])
def test_1063_a_first_pass_registers_and_embeds(tmp_path, home, local_fastembed, provider):
    engine = _local_engine(tmp_path, home, provider)
    try:
        node_id = str(engine._dag.add_node(SummaryNode(
            session_id=SESSION, depth=0, summary="first cached local summary",
            created_at=1.0, latest_at=1.0,
        )))
        engine._store.append(SESSION, {"role": "user", "content": FILLER})
        outcome = maintenance_mod.run_incremental_embedding_pass(
            engine._config.database_path, engine._config,
        )
        print(f"A: outcome={outcome}; profiles={len(_profiles(engine))}")
        assert "summaries=complete:1" in outcome
        assert _summary_vector_ids(engine) == [node_id]
        assert _chunk_vector_count(engine) > 0
        assert sorted((task, active) for task, active, _ in _profiles(engine)) == [
            ("chunk", 1), ("summary", 1),
        ]
        assert local_fastembed[0] and not any(local_fastembed[0])
    finally:
        engine._close_storage()


def test_1063_b_uncached_warns_once_retries_and_preserves_ingest(
    tmp_path, home, local_fastembed, monkeypatch, caplog,
):
    calls, state = local_fastembed
    state["cached"] = False
    monkeypatch.setattr(maintenance_mod, "_FAILURE_WARNED", False)
    engine = _local_engine(tmp_path, home)
    try:
        with caplog.at_level(logging.DEBUG, logger=maintenance_mod.__name__):
            engine.on_session_start(SESSION, platform="cli", context_length=200000)
            _drain(engine)
            engine._store.append(SESSION, {"role": "user", "content": FILLER})
            node_id = _add_leaf(engine, "uncached local summary")
            _drain(engine)
        last, _ = maintenance_mod.incremental_embedding_state(engine._config.database_path)
        print(f"B: outcome={last[1]}; construct_allow_download={calls}")
        assert last[1] == "auto_register_unavailable"
        assert calls == [False, False]
        assert _profiles(engine) == []
        assert [r.levelno for r in caplog.records if r.name == maintenance_mod.__name__] == [
            logging.WARNING, logging.DEBUG,
        ]
        assert _rows(engine, "SELECT COUNT(*) FROM messages")[0][0] > 0
        assert "auto_register_unavailable" in handle_lcm_command("status", engine)
        state["cached"] = True
        engine._schedule_embedding_maintenance()
        _drain(engine)
        assert _summary_vector_ids(engine) == [node_id]
        assert not any(calls)
    finally:
        engine._close_storage()


@pytest.mark.parametrize("provider", ["voyage", "ollama"])
def test_1063_c_other_providers_still_require_warmup(
    tmp_path, home, monkeypatch, provider,
):
    monkeypatch.setenv("VOYAGE_API_KEY", "offline-placeholder")

    def forbidden(*args, **kwargs):
        pytest.fail("no-profile non-FastEmbed pass must not call a provider")

    monkeypatch.setattr(command_mod, "resolve_provider", forbidden)
    monkeypatch.setattr(provider_mod, "probe_provider_availability", forbidden)
    engine = _make_engine(tmp_path, home, provider=provider, register=False)
    try:
        assert maintenance_mod.run_incremental_embedding_pass(
            engine._config.database_path, engine._config,
        ) == "no_profile"
        assert _profiles(engine) == []
    finally:
        engine._close_storage()


WARMUP_1063_READY = (
    "LCM embedding warmup\nstatus: ready\ndownload: ready (about 130 MB)\n"
    "provider: fastembed\nmodel: BAAI/bge-small-en-v1.5\ndim: 2\n"
    "chunk_model: BAAI/bge-small-en-v1.5\nchunk_dim: 2\n"
    "chunk_probe: shared summary probe\ndtype: float32\n"
    "privacy_revision: (local)\ncost_note: local model; no per-call API charge"
)
WARMUP_1063_ERROR = (
    "LCM embedding warmup\nstatus: error\nerror: offline fake model is not cached"
)


@pytest.mark.parametrize("cached", [True, False])
def test_1063_d_warmup_bytes_match_base(tmp_path, home, local_fastembed, cached):
    calls, state = local_fastembed
    state["cached"] = cached
    engine = _local_engine(tmp_path, home)
    try:
        text = command_mod._embedding_warmup_text(engine)
        print(f"D cached={cached}: {json.dumps(text)}")
        assert text == (WARMUP_1063_READY if cached else WARMUP_1063_ERROR)
        assert calls == ([False] if cached else [False, True])
        if cached:
            assert engine._lcm_embedding_provider_cache[1].provider_id == "fastembed"
    finally:
        engine._close_storage()


@pytest.mark.parametrize("auto_first", [True, False])
def test_1063_e_manual_and_auto_share_identities(tmp_path, home, local_fastembed, auto_first):
    engine = _local_engine(tmp_path, home)
    try:
        def auto():
            outcome = maintenance_mod.run_incremental_embedding_pass(
                engine._config.database_path, engine._config,
            )
            assert outcome != "no_profile"

        def manual():
            assert command_mod._embedding_warmup_text(engine) == WARMUP_1063_READY

        (auto if auto_first else manual)()
        first = _profiles(engine)
        (manual if auto_first else auto)()
        assert _profiles(engine) == first
        assert sorted((task, active) for task, active, _ in first) == [
            ("chunk", 1), ("summary", 1),
        ]
        assert len({identity for _, _, identity in first}) == 2
    finally:
        engine._close_storage()


def test_a_new_leaf_and_chunks_get_vectors_without_manual_backfill(
    tmp_path, home, providers, monkeypatch
):
    engine = _make_engine(tmp_path, home)
    try:
        _drive_until_leaf(engine, monkeypatch)
        _drain(engine)

        leaves = _leaf_ids(engine)
        assert leaves and _summary_vector_ids(engine) == leaves
        assert _chunk_vector_count(engine) > 0
        status = handle_lcm_command("status", engine)
        assert "embedding_backlog_summaries: 0" in status
        assert "embedding_backlog_chunks: 0" in status
        assert "summaries=complete" in status and "chunks=complete" in status
    finally:
        engine.shutdown()


@pytest.mark.parametrize("legacy_inflight", [False, True])
def test_scheduling_prepares_schema_before_the_background_pass(
    tmp_path, home, providers, monkeypatch, legacy_inflight
):
    engine = _make_engine(tmp_path, home)
    release_pass = threading.Event()
    real_pass = lcm_engine.run_incremental_embedding_pass

    def held_pass(*args, **kwargs):
        assert release_pass.wait(30)
        return real_pass(*args, **kwargs)

    monkeypatch.setattr(lcm_engine, "run_incremental_embedding_pass", held_pass)
    try:
        assert not _rows(engine, "SELECT name FROM sqlite_master WHERE name IN "
                         "('lcm_chunk_meta', 'lcm_embedding_backfill_inflight')")
        if legacy_inflight:
            with sqlite3.connect(engine._config.database_path) as conn:
                conn.execute(
                    "CREATE TABLE lcm_embedding_backfill_inflight ("
                    "embedded_id TEXT, identity_hash TEXT, lease_id TEXT, "
                    "generation INTEGER, claimed_at REAL, "
                    "PRIMARY KEY (embedded_id, identity_hash))"
                )
                conn.execute(
                    "INSERT INTO lcm_embedding_backfill_inflight "
                    "VALUES ('old-id', 'old-identity', 'old-lease', 1, 1)"
                )
        node_id = str(engine._dag.add_node(SummaryNode(
            session_id=SESSION, depth=0, summary="schema preparation leaf",
            created_at=1.0, latest_at=1.0,
        )))
        engine._store.append(SESSION, {"role": "user", "content": FILLER})
        engine._schedule_embedding_maintenance()
        scheduled_version = _rows(engine, "PRAGMA schema_version")[0][0]
        release_pass.set()
        _drain(engine)
        assert _rows(engine, "PRAGMA schema_version")[0][0] == scheduled_version
        assert _summary_vector_ids(engine) == [node_id]
        assert _chunk_vector_count(engine) > 0
        if legacy_inflight:
            assert _rows(engine, "SELECT embedded_id, identity_hash, state FROM "
                         "lcm_embedding_backfill_inflight") == [
                ("old-id", "old-identity", "uncertain")
            ]
    finally:
        release_pass.set()
        engine.shutdown()


def test_schema_preparation_retries_and_caches_per_database_and_chunk_policy(
    tmp_path, home, monkeypatch, caplog
):
    engine = _make_engine(tmp_path, home, provider="voyage", register=False)
    scheduled = []
    ensured_paths = []
    real_ensure = command_mod._ensure_inflight_table

    def ensure(conn):
        ensured_paths.append(conn.execute("PRAGMA database_list").fetchone()[2])
        if len(ensured_paths) == 1:
            raise sqlite3.OperationalError("schema setup unavailable")
        real_ensure(conn)

    monkeypatch.setattr(command_mod, "_ensure_inflight_table", ensure)
    monkeypatch.setattr(
        lcm_engine._EMBEDDING_MAINTENANCE_SCHEDULER, "schedule",
        lambda *args, **kwargs: scheduled.append(args),
    )
    try:
        engine._schedule_embedding_maintenance()
        assert scheduled == []
        assert "could not schedule background incremental embedding" in caplog.text
        engine._schedule_embedding_maintenance()
        engine._schedule_embedding_maintenance()
        assert len(ensured_paths) == 2 and len(scheduled) == 2
        assert _rows(engine, "SELECT name FROM sqlite_master "
                     "WHERE name='lcm_embedding_profile'")
        assert not _rows(engine, "SELECT name FROM sqlite_master WHERE name='lcm_chunk_meta'")
        engine._config.embedding_provider = "ollama"
        engine._schedule_embedding_maintenance()
        assert len(ensured_paths) == 3
        assert _rows(engine, "SELECT name FROM sqlite_master WHERE name='lcm_chunk_meta'")
        next_path = tmp_path / "next.db"
        monkeypatch.setattr(engine._dag, "db_path", next_path)
        engine._schedule_embedding_maintenance()
        assert len(ensured_paths) == 4 and len(scheduled) == 4
        assert ensured_paths[-1] == str(next_path)
    finally:
        engine.shutdown()


def test_schema_preparation_is_shared_by_engines(tmp_path, home, monkeypatch):
    engine = _make_engine(tmp_path, home, register=False)
    other = _make_engine(tmp_path, home, register=False)
    ensured = []
    real_ensure = command_mod._ensure_inflight_table

    def ensure(conn):
        ensured.append(conn.execute("PRAGMA database_list").fetchone()[2])
        real_ensure(conn)

    monkeypatch.setattr(command_mod, "_ensure_inflight_table", ensure)
    monkeypatch.setattr(
        lcm_engine._EMBEDDING_MAINTENANCE_SCHEDULER, "schedule",
        lambda *args, **kwargs: None,
    )
    try:
        engine._schedule_embedding_maintenance()
        other._schedule_embedding_maintenance()
        assert ensured == [str(tmp_path / "incremental.db")]
        other._config.ollama_base_url = "http://remote.example:11434"
        other._schedule_embedding_maintenance()
        assert len(ensured) == 2  # same database, different chunk policy
        monkeypatch.setattr(other._dag, "db_path", tmp_path / "other.db")
        other._schedule_embedding_maintenance()
        assert ensured[-1] == str(tmp_path / "other.db") and len(ensured) == 3
    finally:
        other.shutdown()
        engine.shutdown()


@pytest.mark.parametrize("host,loopback", [
    ("localhost", True), ("127.0.0.2", True), ("[::1]", True),
    ("remote.example", False), ("192.0.2.1", False),
])
@pytest.mark.parametrize("http_error", [False, True])
def test_http_transport_bypasses_proxy_only_for_loopback(monkeypatch, host, loopback, http_error):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9999")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    routes = []
    url = f"http://{host}:11434/api/embed"
    body = b'{"result":"offline fixture"}'
    headers = {"Content-Type": "application/json"}

    def open_request(request, *, timeout, route):
        routes.append(route)
        assert request.method == "POST" and timeout == 3.0
        assert json.loads(request.data) == {"input": "synthetic message"}
        if http_error:
            raise urllib.error.HTTPError(url, 429, "busy", headers, io.BytesIO(body))
        response = io.BytesIO(body)
        response.status = 200
        response.headers = headers
        return response

    def build_opener(handler):
        assert isinstance(handler, provider_mod.urllib.request.ProxyHandler)
        assert handler.proxies == {}
        return SimpleNamespace(open=lambda request, **kwargs: open_request(
            request, route="direct", **kwargs,
        ))

    monkeypatch.setattr(provider_mod.urllib.request, "build_opener", build_opener)
    monkeypatch.setattr(provider_mod.urllib.request, "urlopen", lambda request, **kwargs:
                        open_request(request, route="environment proxy", **kwargs))
    result = provider_mod._default_http_transport(
        url=url, payload={"input": "synthetic message"}, headers=headers, timeout=3.0,
    )
    assert routes == ["direct" if loopback else "environment proxy"]
    assert result.status == (429 if http_error else 200)
    assert result.body == body and result.headers == headers
    config = LCMConfig(embedding_provider="ollama", ollama_base_url=url)
    assert maintenance_mod._automatic_chunks_allowed(config, command_mod) is loopback


def test_b_held_manual_lease_skips_then_next_pass_completes_once(
    tmp_path, home, providers
):
    engine = _make_engine(tmp_path, home)
    try:
        store = VectorStore(engine._config.database_path, config=engine._config)
        try:
            command_mod._ensure_inflight_table(store.connection)
            lease = command_mod._acquire_embedding_backfill_lease(
                store.connection, ttl_s=600.0, heartbeat_s=60.0
            )
            assert lease is not None
            node_id = _add_leaf(engine, "leaf published while a manual backfill runs")
            _drain(engine)
            last, _failures = maintenance_mod.incremental_embedding_state(
                engine._config.database_path
            )
            assert last[1] == "lease_held"
            assert _summary_vector_ids(engine) == []
            assert all(not provider.documents for provider in providers)
            lease.release()
        finally:
            store.close()

        engine._schedule_embedding_maintenance()
        _drain(engine)
        engine._schedule_embedding_maintenance()
        _drain(engine)

        assert _summary_vector_ids(engine) == [node_id]
        sent = [doc for provider in providers for doc in provider.documents]
        assert sent.count("leaf published while a manual backfill runs") == 1
        assert _inflight_count(engine) == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("failure", ["probe", "resolve", "pre_dispatch"])
def test_c_unavailable_provider_never_raises_counts_and_recovers(
    tmp_path, home, providers, monkeypatch, caplog, failure
):
    engine = _make_engine(tmp_path, home)
    monkeypatch.setattr(maintenance_mod, "_FAILURE_WARNED", False)
    down = {"value": True}
    real_probe = provider_mod.probe_provider_availability
    real_resolve = command_mod.resolve_provider

    def probe(config):
        if down["value"] and failure == "probe":
            return {"available": False, "detail": "FastEmbed is not installed"}
        return real_probe(config)

    def resolve(config, **kwargs):
        if down["value"] and failure == "resolve":
            raise ProviderUnavailable("provider is down")
        if down["value"] and failure == "pre_dispatch":
            return UnavailableProvider(config.embedding_provider, config.embedding_model)
        return real_resolve(config, **kwargs)

    monkeypatch.setattr(provider_mod, "probe_provider_availability", probe)
    monkeypatch.setattr(command_mod, "resolve_provider", resolve)
    try:
        _, before = maintenance_mod.incremental_embedding_state(engine._config.database_path)
        with caplog.at_level(logging.DEBUG, logger=maintenance_mod.__name__):
            node_id = _add_leaf(engine, "leaf published while the provider is down")
            _drain(engine)
            engine._schedule_embedding_maintenance()
            _drain(engine)
        _, after = maintenance_mod.incremental_embedding_state(engine._config.database_path)
        assert after == before + 2
        levels = [r.levelno for r in caplog.records if r.name == maintenance_mod.__name__]
        assert levels == [logging.WARNING, logging.DEBUG]
        assert _summary_vector_ids(engine) == []
        # Nothing reached a provider, so nothing is left uncertain.
        assert _inflight_count(engine) == 0

        down["value"] = False
        engine._schedule_embedding_maintenance()
        _drain(engine)
        assert _summary_vector_ids(engine) == [node_id]
    finally:
        engine.shutdown()


def test_d_cloud_provider_embeds_summaries_but_never_chunks(
    tmp_path, home, providers, monkeypatch
):
    monkeypatch.setattr(
        provider_mod,
        "probe_provider_availability",
        lambda _config: {"available": True, "detail": "offline test"},
    )
    chunk_runs = []
    real_chunk_run = command_mod._chunk_backfill_run
    monkeypatch.setattr(
        command_mod,
        "_chunk_backfill_run",
        lambda *args, **kwargs: chunk_runs.append(kwargs) or real_chunk_run(*args, **kwargs),
    )
    engine = _make_engine(tmp_path, home, provider="voyage")
    try:
        _drive_until_leaf(engine, monkeypatch)
        _drain(engine)

        assert _summary_vector_ids(engine) == _leaf_ids(engine) != []
        assert _chunk_vector_count(engine) == 0
        assert chunk_runs == []
        assert "embedding_backlog_chunks: 0" not in handle_lcm_command("status", engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("enabled", [False, True])
def test_e_embeddings_off_never_schedules_a_pass(
    tmp_path, home, providers, monkeypatch, enabled
):
    scheduled = []
    monkeypatch.setattr(
        lcm_engine._EMBEDDING_MAINTENANCE_SCHEDULER,
        "schedule",
        lambda key, job, **kwargs: scheduled.append(key) or True,
    )
    engine = _make_engine(tmp_path, home, enabled=enabled)
    try:
        _drive_until_leaf(engine, monkeypatch)
        # Positive control: the same drive does schedule once embeddings are on.
        assert bool(scheduled) is enabled
        assert ("embedding_backlog" in handle_lcm_command("status", engine)) is enabled
    finally:
        engine.shutdown()


def test_g_recall_finds_a_summary_created_after_the_last_manual_backfill(
    tmp_path, home, providers, monkeypatch
):
    engine = _make_engine(tmp_path, home)
    try:
        report = handle_lcm_command("embed backfill --corpus both --apply", engine)
        assert "status: complete" in report
        _drive_until_leaf(engine, monkeypatch)
        _drain(engine)
        leaf = _leaf_ids(engine)[0]

        summary_arm_hits = []
        real_arm = lcm_tools._lcm_recall_summary_arm

        def spy(*args, **kwargs):
            result = real_arm(*args, **kwargs)
            summary_arm_hits.extend(str(hit["node_id"]) for hit in result[0])
            return result

        monkeypatch.setattr(lcm_tools, "_lcm_recall_summary_arm", spy)
        monkeypatch.setattr(
            lcm_tools,
            "resolve_provider",
            lambda config, **_kw: FakeProvider(config.embedding_provider, config.embedding_model),
        )
        payload = json.loads(lcm_tools.lcm_recall(
            {"query": "qwertyuiop", "include": "summaries", "scope_bias": 0.0, "limit": 5},
            engine=engine,
        ))
        assert leaf in summary_arm_hits
        assert leaf in [str(hit.get("node_id")) for hit in payload["hits"]]
    finally:
        engine.shutdown()


def test_status_backlog_names_uncertain_rows_the_pass_cannot_retry(tmp_path, home, providers):
    engine = _make_engine(tmp_path, home)
    try:
        store = VectorStore(engine._config.database_path, config=engine._config)
        try:
            conn = store.connection
            command_mod._ensure_inflight_table(conn)
            identity = str(conn.execute(
                "SELECT identity_hash FROM lcm_embedding_profile "
                "WHERE active = 1 AND task != 'chunk'"
            ).fetchone()[0])
            node_id = engine._dag.add_node(SummaryNode(
                session_id=SESSION, depth=0, summary="dispatched before a crash",
                created_at=1.0, latest_at=1.0,
            ))
            conn.execute(
                "INSERT INTO lcm_embedding_backfill_inflight(embedded_id, identity_hash, "
                "lease_id, generation, claimed_at, state, request_id, updated_at, last_error) "
                "VALUES (?, ?, 'prior', 1, 1, 'uncertain', 'prior-request', 1, 'unknown')",
                (str(node_id), identity),
            )
            conn.commit()
        finally:
            store.close()

        engine._schedule_embedding_maintenance()
        _drain(engine)

        assert _summary_vector_ids(engine) == []
        assert "embedding_backlog_summaries: 0 (uncertain 1)" in handle_lcm_command(
            "status", engine
        )
    finally:
        engine.shutdown()


def test_embedding_worker_is_named_for_embeddings_not_rollups(caplog):
    scheduler = lcm_engine._RollupMaintenanceScheduler(
        max_pending_jobs=1, kind="incremental embedding",
        thread_name="lcm-embedding-maintenance",
    )
    assert lcm_engine._EMBEDDING_MAINTENANCE_SCHEDULER._thread_name == "lcm-embedding-maintenance"
    assert lcm_engine._EMBEDDING_MAINTENANCE_SCHEDULER._kind == "incremental embedding"
    assert lcm_engine._ROLLUP_MAINTENANCE_SCHEDULER._thread_name == "lcm-rollup-maintenance"
    started, release = threading.Event(), threading.Event()
    try:
        assert scheduler.schedule(("db", "embeddings"), lambda: (started.set(), release.wait()))
        assert started.wait(5)  # the first job is active, so the queue is empty
        assert scheduler.schedule(("db-2", "embeddings"), lambda: None)
        with caplog.at_level(logging.WARNING, logger=lcm_engine.__name__):
            assert not scheduler.schedule(("db-3", "embeddings"), lambda: None)
        full = [r.getMessage() for r in caplog.records if "queue is full" in r.getMessage()]
        assert full and full[0].startswith("LCM incremental embedding maintenance queue is full")
        assert scheduler._worker.name == "lcm-embedding-maintenance"
    finally:
        release.set()
        scheduler.drain({("db", "embeddings"), ("db-2", "embeddings")}, timeout=5)


def test_pass_peeks_the_query_breaker_without_clearing_a_fresh_cooldown(
    tmp_path, home, providers
):
    from hermes_lcm.embedding_provider import EmbeddingCircuitBreaker

    engine = _make_engine(tmp_path, home)
    try:
        db_path = engine._config.database_path
        # Open: the pass skips and the cooldown is untouched.
        breaker = EmbeddingCircuitBreaker(failure_threshold=1, cooldown_seconds=400.0)
        breaker.record_failure()
        opened = (breaker._open_until, breaker._failures)
        assert maintenance_mod.run_incremental_embedding_pass(
            db_path, engine._config, breaker=breaker
        ) == "circuit_open"
        assert (breaker._open_until, breaker._failures) == opened

        # Expired: the pass runs, but a read-only peek leaves the state for the
        # query path to reset, so it cannot erase a cooldown a concurrent query
        # failure sets in between.
        breaker._open_until = 1.0
        assert not breaker.is_open()
        maintenance_mod.run_incremental_embedding_pass(db_path, engine._config, breaker=breaker)
        assert (breaker._open_until, breaker._failures) == (1.0, opened[1])
    finally:
        engine.shutdown()


def test_condensed_publication_does_not_schedule_a_pass(tmp_path, home, monkeypatch):
    engine = _make_engine(tmp_path, home)
    scheduled = []
    monkeypatch.setattr(
        engine, "_schedule_embedding_maintenance", lambda: scheduled.append(True)
    )
    try:
        engine._invalidate_rollups_for_published_node(
            SummaryNode(session_id=SESSION, depth=1, summary="condensed", created_at=1.0)
        )
        assert scheduled == []
        engine._invalidate_rollups_for_published_node(
            SummaryNode(session_id=SESSION, depth=0, summary="leaf", created_at=1.0)
        )
        assert scheduled == [True]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("base_url", "chunks_run"),
    [
        ("http://10.1.2.3:11434", False),
        ("https://ollama.example.com", False),
        ("http://127.0.0.2:11434/", True),
        ("http://[::1]:11434", True),
        ("http://localhost:11434", True),
    ],
)
def test_remote_ollama_gets_summaries_but_never_automatic_chunks(
    tmp_path, home, providers, monkeypatch, base_url, chunks_run
):
    chunk_runs = []
    real_chunk_run = command_mod._chunk_backfill_run
    monkeypatch.setattr(
        command_mod,
        "_chunk_backfill_run",
        lambda *args, **kwargs: chunk_runs.append(kwargs) or real_chunk_run(*args, **kwargs),
    )
    engine = _make_engine(tmp_path, home)
    engine._config.ollama_base_url = base_url
    try:
        _drive_until_leaf(engine, monkeypatch)
        _drain(engine)

        assert _summary_vector_ids(engine) == _leaf_ids(engine) != []
        assert bool(chunk_runs) is chunks_run
        assert (_chunk_vector_count(engine) > 0) is chunks_run
    finally:
        engine.shutdown()


def test_a_summary_failure_does_not_starve_the_chunk_corpus(
    tmp_path, home, providers, monkeypatch
):
    monkeypatch.setattr(
        command_mod,
        "_embedding_backfill_summary_run",
        lambda *_args, **_kwargs: {
            "status": "failed", "error": None, "failed": [("1", "provider_error:bad row")],
            "selected": 1, "privacy_withheld": 0, "skipped": [], "embedded": 0,
            "stop_reason": None,
        },
    )
    engine = _make_engine(tmp_path, home)
    try:
        for index in range(3):
            engine._store.append(SESSION, {"role": "user", "content": f"message {index} {FILLER}"})
        _, before = maintenance_mod.incremental_embedding_state(engine._config.database_path)
        outcome = maintenance_mod.run_incremental_embedding_pass(
            engine._config.database_path, engine._config
        )
        _, after = maintenance_mod.incremental_embedding_state(engine._config.database_path)

        assert outcome.startswith("summaries=failed chunks=complete:")
        assert _chunk_vector_count(engine) >= 3
        assert after == before + 1
    finally:
        engine.shutdown()


def test_automatic_chunk_discovery_is_bounded_by_new_messages_not_the_store(
    tmp_path, home, providers, monkeypatch
):
    engine = _make_engine(tmp_path, home)
    db_path = engine._config.database_path
    calls = []
    real_chunk_message = command_mod.chunk_message
    monkeypatch.setattr(
        command_mod,
        "chunk_message",
        lambda *args, **kwargs: calls.append(args[0]) or real_chunk_message(*args, **kwargs),
    )
    try:
        for index in range(600):
            engine._store.append(SESSION, {"role": "user", "content": f"old {index} {FILLER}"})
        batch = command_mod._embedding_backfill_batch_size(
            engine._config.embedding_max_batch_items
        )

        # A fresh store: one pass reads about one batch of messages, oldest first.
        calls.clear()
        maintenance_mod.run_incremental_embedding_pass(db_path, engine._config)
        assert _chunk_vector_count(engine) == batch
        assert len(calls) <= batch + 2

        # Steady state: the whole history is embedded; three new messages cost
        # about three chunker calls, not one per stored message.
        report = handle_lcm_command("embed backfill --corpus chunks --apply --limit 10000", engine)
        assert "status: complete" in report
        for index in range(3):
            engine._store.append(SESSION, {"role": "user", "content": f"new {index} {FILLER}"})
        calls.clear()
        maintenance_mod.run_incremental_embedding_pass(db_path, engine._config)
        assert _chunk_vector_count(engine) == 603
        assert len(calls) <= 5
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "spelling", ["fastembed", "fast-embed", " FastEmbed ", "Fast-Embed", "OLLAMA"]
)
def test_every_accepted_local_provider_spelling_allows_automatic_chunks(spelling):
    config = LCMConfig(embedding_provider=spelling, embedding_model="model-a")
    resolved = provider_mod.resolve_provider(config)
    assert resolved.provider_id in {"fastembed", "ollama"}
    assert maintenance_mod._automatic_chunks_allowed(config, command_mod)


def test_1063_f_slow_cached_load_is_outside_the_query_deadline(
    tmp_path, home, local_fastembed, monkeypatch,
):
    calls, _state = local_fastembed
    construct = provider_mod.FastembedProvider._construct

    def slow_construct(self, *, allow_download):
        time.sleep(0.3)  # a cold page cache can make the first ONNX load slow
        return construct(self, allow_download=allow_download)

    monkeypatch.setattr(provider_mod.FastembedProvider, "_construct", slow_construct)
    engine = _local_engine(tmp_path, home)
    engine._config.embedding_query_timeout_s = 0.1
    try:
        engine._dag.add_node(SummaryNode(
            session_id=SESSION, depth=0, summary="slow cached local summary",
            created_at=1.0, latest_at=1.0,
        ))
        outcome = maintenance_mod.run_incremental_embedding_pass(
            engine._config.database_path, engine._config,
        )
        print(f"F: outcome={outcome}; construct_allow_download={calls}")
        assert sorted(task for task, _active, _ in _profiles(engine)) == ["chunk", "summary"]
        assert not any(calls)
    finally:
        engine._close_storage()


def test_1063_g_registration_provider_released_before_backfill(
    tmp_path, home, local_fastembed, monkeypatch,
):
    register = command_mod._embedding_register_profiles
    backfill = command_mod._embedding_backfill_summary_run
    refs, alive_at_backfill = [], []

    def tracking_register(*args, **kwargs):
        result = register(*args, **kwargs)
        if not isinstance(result, str):
            refs.append(weakref.ref(result["provider"]))
        return result

    def tracking_backfill(*args, **kwargs):
        gc.collect()
        alive_at_backfill.append([ref() is not None for ref in refs])
        return backfill(*args, **kwargs)

    monkeypatch.setattr(command_mod, "_embedding_register_profiles", tracking_register)
    monkeypatch.setattr(command_mod, "_embedding_backfill_summary_run", tracking_backfill)
    engine = _local_engine(tmp_path, home)
    try:
        engine._dag.add_node(SummaryNode(
            session_id=SESSION, depth=0, summary="released provider summary",
            created_at=1.0, latest_at=1.0,
        ))
        maintenance_mod.run_incremental_embedding_pass(
            engine._config.database_path, engine._config,
        )
        print(f"G: registration provider alive at backfill={alive_at_backfill}")
        assert refs and alive_at_backfill == [[False]]
    finally:
        engine._close_storage()
