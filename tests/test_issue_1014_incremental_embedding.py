"""#1014: new summaries and chunks get vectors in background maintenance.

Before #1014 only `/lcm embed backfill` wrote vectors, so every summary or chunk
published after a manual backfill stayed unembedded. These tests drive the real
engine and drain its incremental embedding scheduler; no manual backfill runs
unless a test says so.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading

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
