"""Default-OFF precursor contracts; deterministic offline LOCAL models only."""

import contextvars
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance
from hermes_lcm import embedding_provider as providers, engine as engines, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.dag import SummaryNode


@pytest.fixture
def make_engine(tmp_path):
    made = []

    def make(name="off", **kwargs):
        config = LCMConfig(database_path=str(tmp_path / f"{name}.db"),
                           temporal_rollups_enabled=False, **kwargs)
        engine = LCMEngine(config=config, hermes_home=str(tmp_path / name))
        engine._session_id = name
        engine.context_length = 200000
        made.append(engine)
        return engine

    yield make
    for engine in made:
        engine.shutdown()


def local_config():
    return dict(embeddings_enabled=True, embedding_provider="fastembed", embedding_model="fixture-local")


def test_default_off_identity(make_engine, monkeypatch):
    engine = make_engine()
    calls = []
    monkeypatch.setattr(providers, "_load_fastembed", lambda: calls.append("model"))
    monkeypatch.setattr(engines._EMBEDDING_MAINTENANCE_SCHEDULER, "schedule",
                        lambda *_a, **_k: calls.append("enqueue"))
    before = tools._embedding_provider_health_check(engine)
    assert before == {"check": "embedding_provider_health", "status": "pass",
                      "detail": "semantic embeddings are disabled (LCM_EMBEDDINGS_ENABLED=false)"}
    engine._store.append("off", {"role": "user", "content": "precursor canary"})

    def health():
        result = json.loads(engine.handle_tool_call("lcm_recall", {"query": "canary", "include": "all"}))
        return {key: result[key] for key in ("degraded", "degraded_reason", "provenance")}

    recall_before = health()
    doctor_before = command.handle_lcm_command("doctor", engine)
    # New helper must itself stay inert for the existing default configuration.
    assert maintenance.start_local_model_prepare(engine._store.db_path, engine._config) is False
    engine._schedule_embedding_maintenance()
    assert engine._config.embedding_provider == engine._config.embedding_model == ""
    assert not engine._config.embeddings_enabled
    assert maintenance._PREPARE_THREAD is None
    assert not calls
    assert tools._embedding_provider_health_check(engine) == before
    assert health() == recall_before
    assert command.handle_lcm_command("doctor", engine) == doctor_before
    assert "embedding_readiness:" not in doctor_before


@pytest.mark.parametrize("retire_a", [False, True])
def test_profile_capture_completion_and_retirement(make_engine, monkeypatch, retire_a):
    profile = contextvars.ContextVar("scheduling-profile", default="unbound")
    entered, release = threading.Event(), threading.Event()
    prepared, passes, enqueued = [], [], []
    scheduler = engines._RollupMaintenanceScheduler(kind="incremental embedding")
    monkeypatch.setattr(engines, "_EMBEDDING_MAINTENANCE_SCHEDULER", scheduler)
    original_schedule = scheduler.schedule

    def schedule(key, job, **kwargs):
        enqueued.append((Path(key[0]).name, profile.get()))
        return original_schedule(key, job, **kwargs)

    def prepare(config, db, **kwargs):
        prepared.append((Path(db).name, profile.get(), kwargs["allow_download"]))
        entered.set()
        assert release.wait(5)
        return {}

    monkeypatch.setattr(scheduler, "schedule", schedule)
    monkeypatch.setattr(command, "_embedding_register_profiles", prepare)
    monkeypatch.setattr(engines, "run_incremental_embedding_pass",
                        lambda db, *_a, **_k: passes.append((Path(db).name, profile.get())))
    a, b = make_engine("a", **local_config()), make_engine("b", **local_config())
    try:
        token = profile.set("A")
        a._schedule_embedding_maintenance()
        profile.reset(token)
        assert entered.wait(5), "explicit LOCAL never started background preparation"
        token = profile.set("B")
        b._schedule_embedding_maintenance()
        profile.reset(token)
        assert a.drain_embedding_maintenance(5) and b.drain_embedding_maintenance(5)
        assert prepared == [("a.db", "A", True)]
        passes.clear()
        enqueued.clear()
        if retire_a:
            a._retire_embedding_maintenance(wait=True)
        release.set()
        maintenance._PREPARE_THREAD.join(5)
        assert not maintenance._PREPARE_THREAD.is_alive()
        assert b.drain_embedding_maintenance(5)
        expected = [("b.db", "B")] if retire_a else [("a.db", "A"), ("b.db", "B")]
        assert a.drain_embedding_maintenance(5)
        assert enqueued == expected
        assert passes == expected
    finally:
        release.set()
        thread = getattr(maintenance, "_PREPARE_THREAD", None)
        if thread is not None:
            thread.join(5)


def test_local_profiles_share_one_model_per_key(make_engine, monkeypatch, tmp_path):
    constructions = []
    gate = threading.Barrier(2)

    class Model:
        def __init__(self, **kwargs):
            constructions.append(kwargs)

        def query_embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        embed = query_embed

    monkeypatch.setattr(providers, "_load_fastembed", lambda: Model)
    a, b = make_engine("a", **local_config()), make_engine("b", **local_config())
    local_providers = [providers.resolve_provider(e._config) for e in (a, b)]
    vectors = []

    def query(provider):
        gate.wait(5)
        vectors.append(provider.embed_query("canary"))

    threads = [threading.Thread(target=query, args=(provider,)) for provider in local_providers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert len(constructions) == 1, "same LOCAL model/cache/thread key constructed twice"
    assert constructions[0]["threads"] == 2
    assert local_providers[0]._model is local_providers[1]._model
    providers.FastembedProvider("different", cache_dir=tmp_path).embed_query("canary")
    providers.FastembedProvider("fixture-local", cache_dir=tmp_path).embed_query("canary")
    monkeypatch.setattr(providers, "_FASTEMBED_THREADS", 1)
    providers.FastembedProvider("fixture-local", cache_dir=tmp_path).embed_query("canary")
    assert len(constructions) == 4


def test_uncertainty_retry_is_fastembed_only(make_engine, monkeypatch):
    calls = []
    monkeypatch.setattr(command, "_embedding_current_profile", lambda _conn: {"identity_hash": "fixture"})
    monkeypatch.setattr(command, "_chunk_current_profile", lambda _conn: {"identity_hash": "fixture"})
    monkeypatch.setattr(providers, "probe_provider_availability", lambda _c: {"available": True})

    def backfill(config, _db, **kwargs):
        calls.append((config.embedding_provider, kwargs["retry_uncertain"]))
        return {"status": "complete", "selected": 0, "privacy_withheld": 0, "skipped": [],
                "embedded": 0, "stop_reason": "", "error": None, "failed": []}

    monkeypatch.setattr(command, "_embedding_backfill_summary_run", backfill)
    monkeypatch.setattr(command, "_chunk_backfill_run", backfill)
    for provider in ("fastembed", "fast-embed", "voyage", "ollama"):
        engine = make_engine(provider, embeddings_enabled=True, embedding_provider=provider,
                             embedding_model="fixture-local", ollama_base_url="https://remote.invalid")
        maintenance.run_incremental_embedding_pass(engine._store.db_path, engine._config)
    assert calls == [("fastembed", False), ("fastembed", True)] * 2 + [
        ("fast-embed", False), ("fast-embed", True)] * 2 + [("voyage", False), ("ollama", False)]


def test_explicit_local_prepare_keeps_recall_answering(make_engine, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    engine = make_engine("local", **local_config())
    engine._store.append("local", {"role": "user", "content": "precursor canary " + "detail " * 60})
    engine._dag.add_node(SummaryNode(session_id="local", depth=0, summary="precursor canary summary",
                                    created_at=1.0, latest_at=1.0))
    calls = []

    class Model:
        def __init__(self, *, local_files_only, **kwargs):
            calls.append((threading.current_thread().name, local_files_only))
            if local_files_only:
                raise RuntimeError("offline model is not cached yet")
            entered.set()
            assert release.wait(5)

        def query_embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        embed = query_embed

    monkeypatch.setattr(providers, "_load_fastembed", lambda: Model)
    probe = lambda _c: {"provider": "fastembed", "model": "fixture-local",  # noqa: E731
                        "available": True, "detail": "offline fixture"}
    monkeypatch.setattr(tools, "probe_provider_availability", probe)
    monkeypatch.setattr(providers, "probe_provider_availability", probe)
    try:
        engine._schedule_embedding_maintenance()
        assert entered.wait(5), "explicit LOCAL did not start preparation"
        assert engine.drain_embedding_maintenance(5)
        assert maintenance.local_model_state(engine._config) == ("preparing", "")
        text = command.handle_lcm_command("doctor", engine)
        assert "embedding_readiness: model: preparing; coverage: 0/1 summaries" in text
        result = json.loads(engine.handle_tool_call("lcm_recall", {"query": "canary", "include": "all"}))
        assert result["hits"] and result["degraded"]
        assert "preparing" in result["degraded_reason"]
        assert calls == [("lcm-embedding-prepare", True), ("lcm-embedding-prepare", False)]
        release.set()
        maintenance._PREPARE_THREAD.join(5)
        assert not maintenance._PREPARE_THREAD.is_alive()
        assert engine.drain_embedding_maintenance(5)
        with sqlite3.connect(engine._store.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_meta").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM lcm_chunk_meta").fetchone()[0] > 0
        assert "embedding_readiness: model: present; coverage: 1/1 summaries" in command.handle_lcm_command("doctor", engine)
        result = json.loads(engine.handle_tool_call("lcm_recall", {"query": "canary", "include": "all"}))
        assert result["hits"] and not result["degraded"]
        assert len(calls) == 2
    finally:
        engine._retire_embedding_maintenance(wait=True)
        release.set()
        thread = getattr(maintenance, "_PREPARE_THREAD", None)
        if thread is not None:
            thread.join(5)


def test_retire_snapshots_owner_before_concurrent_rearm(make_engine, monkeypatch):
    engine = make_engine("race")
    old_owner = engine._embedding_maintenance_owner
    retired = []
    monkeypatch.setattr(engines._EMBEDDING_MAINTENANCE_SCHEDULER, "retire_owner",
                        lambda owner, timeout=0: retired.append(owner))
    real_lock = engine._embedding_maintenance_lock

    class RearmOnRelease:
        """Runs one rearm right after retire releases the lock: the race window."""
        fired = False

        def __enter__(self):
            real_lock.acquire()
            return self

        def __exit__(self, *exc):
            real_lock.release()
            if not RearmOnRelease.fired:
                RearmOnRelease.fired = True
                engine._rearm_embedding_maintenance()
            return False

    engine._embedding_maintenance_lock = RearmOnRelease()
    try:
        engine._retire_embedding_maintenance(wait=False)
    finally:
        engine._embedding_maintenance_lock = real_lock
    assert retired == [old_owner]
    assert engine._embedding_maintenance_owner is not old_owner
    assert engine._embedding_maintenance_retired is False


def test_doctor_reports_preparing_while_cached_probe_fails(make_engine, monkeypatch):
    engine = make_engine("preparing", **local_config())
    key = maintenance._local_model_key(engine._config)
    assert key
    monkeypatch.setitem(maintenance._PREPARE, key, ("preparing", ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda _c: {
        "available": False, "provider": "fastembed", "model": "fixture-local", "probed": True,
        "detail": "model files are not cached"})
    check = tools._embedding_provider_health_check(engine)
    assert check["status"] == "warn"
    assert check["detail"]["readiness"].startswith("model: preparing;")
    assert "prepares in the background" in check["detail"]["impact"]


def test_prepare_keeps_every_owner_callback_for_one_database(monkeypatch, tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "shared.db"), **local_config())
    ran, release = [], threading.Event()

    def register(_config, _db, **kwargs):
        assert release.wait(5)
        return {}

    monkeypatch.setattr(command, "_embedding_register_profiles", register)
    owner_a, owner_b = object(), object()
    assert maintenance.start_local_model_prepare(config.database_path, config,
                                                 on_ready=lambda: ran.append("a"), owner=owner_a)
    # A second engine on the same database, e.g. after a rearm, must not displace the first owner.
    assert not maintenance.start_local_model_prepare(config.database_path, config,
                                                     on_ready=lambda: ran.append("b"), owner=owner_b)
    release.set()
    maintenance._PREPARE_THREAD.join(5)
    assert sorted(ran) == ["a", "b"]


@pytest.mark.parametrize(("model", "expected"), [
    ("other-local", "not prepared (run /lcm embed warmup)"), ("fixture-local", "present")])
def test_doctor_present_needs_the_configured_local_model(make_engine, monkeypatch, model, expected):
    from hermes_lcm.vector_store import VectorStore

    engine = make_engine("stale-profile", **local_config())
    vectors = VectorStore(engine._store.db_path, config=engine._config)
    try:
        vectors.register_profile(model, "fastembed", 2)
    finally:
        vectors.close()
    monkeypatch.setattr(tools, "probe_provider_availability", lambda _c: {
        "available": True, "provider": "fastembed", "model": "fixture-local", "probed": True, "detail": "fixture"})
    readiness = tools._embedding_provider_health_check(engine)["detail"]["readiness"]
    assert readiness.startswith(f"model: {expected};")


def test_process_present_state_does_not_vouch_for_another_database(make_engine, monkeypatch):
    from hermes_lcm.vector_store import VectorStore

    prepared, other = make_engine("db-a", **local_config()), make_engine("db-b", **local_config())
    vectors = VectorStore(prepared._store.db_path, config=prepared._config)
    try:
        vectors.register_profile("fixture-local", "fastembed", 2)
    finally:
        vectors.close()
    other._dag.add_node(SummaryNode(session_id="db-b", depth=0, summary="unembedded summary",
                                    created_at=1.0, latest_at=1.0))
    monkeypatch.setitem(maintenance._PREPARE, maintenance._local_model_key(other._config), ("present", ""))
    monkeypatch.setattr(tools, "probe_provider_availability", lambda _c: {
        "available": True, "provider": "fastembed", "model": "fixture-local", "probed": True, "detail": "fixture"})
    assert tools._embedding_provider_health_check(prepared)["detail"]["readiness"].startswith("model: present;")
    readiness = tools._embedding_provider_health_check(other)["detail"]["readiness"]
    assert readiness == "model: not prepared (run /lcm embed warmup); coverage: 0/1 summaries"
