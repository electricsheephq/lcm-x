"""#1330: session end and shutdown retire only this engine's embedding work."""

import logging
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import hermes_lcm.engine as engine_mod
import hermes_lcm.command as command_mod
import hermes_lcm.embedding_maintenance as maintenance_mod
import hermes_lcm.embedding_provider as provider_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import VectorStore
from hermes_lcm.chunking import chunk_message


@pytest.fixture(autouse=True)
def reset_exit_stop():
    signal = getattr(maintenance_mod, "_STOP_EMBEDDING_MAINTENANCE", None)
    if signal is not None:
        signal.clear()
    yield
    if signal is not None:
        signal.clear()


def _seed_batches(path, corpus):
    config = LCMConfig(
        database_path=str(path), embeddings_enabled=True,
        embedding_provider="fastembed", embedding_model="test-model",
        embedding_max_batch_items=40, temporal_rollups_enabled=False,
    )
    engine = LCMEngine(config=config, hermes_home=str(path.parent / "home"))
    store = VectorStore(path, config=config)
    store.register_profile("test-model", "fastembed", 2)
    store.register_profile("test-model", "fastembed", 2, task="chunk")
    store.close()
    for index in range(40):
        if corpus == "summaries":
            engine._dag.add_node(SummaryNode(
                session_id="batch-test", depth=0, summary=f"summary {index}",
                created_at=float(index), latest_at=float(index),
            ))
        elif corpus == "chunks":
            engine._store.append("batch-test", {
                "role": "user", "content": f"message {index} " + "detail " * 60,
            })
    engine._close_storage()
    return config


class _SlowBatches:
    provider_id = "fastembed"
    model_id = "test-model"
    last_skipped_documents = []

    def __init__(self, started, marker=None):
        self.started = started
        self.marker = marker
        self.completed = 0
        self.delay = 0.3

    def embed_document_batches(self, texts, *, before_dispatch):
        for offset in range(0, len(texts), 2):
            indexes = tuple(range(offset, min(offset + 2, len(texts))))
            before_dispatch(indexes)
            self.started.set()
            if self.delay:
                # Hold the first batch until quiesce requests a stop, even if
                # the test thread is descheduled after observing started.
                assert maintenance_mod._STOP_EMBEDDING_MAINTENANCE.wait(30)
                time.sleep(self.delay)
            self.completed += 1
            if self.marker is not None:
                self.marker.write_text(str(self.completed))
            yield provider_mod.EmbeddedDocumentBatch(indexes, tuple((1.0, 0.0) for _ in indexes))


@pytest.mark.parametrize("corpus", ["summaries", "chunks"])
def test_quiesce_stops_between_provider_batches_and_resumes(tmp_path, scheduler, monkeypatch, caplog, corpus):
    monkeypatch.setenv("LCM_EMBEDDING_BACKFILL_BATCH_SIZE", "40")
    config = _seed_batches(tmp_path / "batches.db", corpus)
    started = threading.Event()
    provider = _SlowBatches(started)
    monkeypatch.setattr(command_mod, "resolve_provider", lambda *_a, **_k: provider)
    monkeypatch.setattr(provider_mod, "probe_provider_availability", lambda _c: {"available": True})
    key = (config.database_path, "embeddings")
    outcomes = []

    def run():
        outcomes.append(maintenance_mod.run_incremental_embedding_pass(config.database_path, config))

    try:
        with caplog.at_level(logging.INFO):
            assert scheduler.schedule(key, run)
            assert started.wait(30)
            assert scheduler.quiesce(timeout=30)
        assert provider.completed == 1
        assert outcomes == ["stopped"]
        stopped = [r for r in caplog.records if "embedding maintenance stopped" in r.getMessage()]
        assert len(stopped) == 1
        assert f"database={config.database_path}" in stopped[0].getMessage()
        assert "scope=embeddings" in stopped[0].getMessage()
        assert not any(r.levelno >= logging.WARNING for r in caplog.records)
        table = "lcm_embedding_meta" if corpus == "summaries" else "lcm_chunk_meta"
        with sqlite3.connect(config.database_path) as conn:
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 2
            assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight").fetchone()[0] == 0
        maintenance_mod._STOP_EMBEDDING_MAINTENANCE.clear()
        provider.delay = 0
        run()
        assert provider.completed == 20  # Already published rows are not embedded again.
        with sqlite3.connect(config.database_path) as conn:
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 40
    finally:
        provider.delay = 0
        assert scheduler.drain({key}, timeout=30)


def test_session_end_does_not_signal_process_exit(make_engine, monkeypatch):
    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", lambda *_a, **_k: None)
    engine = make_engine()
    _bind(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    engine.on_session_end("session", [])
    signal = getattr(maintenance_mod, "_STOP_EMBEDDING_MAINTENANCE", None)
    assert signal is None or not signal.is_set()


def test_interrupted_multichunk_message_resumes(tmp_path, scheduler, monkeypatch):
    config = _seed_batches(tmp_path / "multichunk.db", "empty")
    config.embedding_max_batch_items = 2
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    text = "A factual sentence with several details. " * 400
    row = engine._store.append("batch-test", {"role": "user", "content": text})
    expected = len(chunk_message(row, "user", text, policy="conversational"))
    assert expected > 2
    engine._close_storage()
    started = threading.Event()
    provider = _SlowBatches(started)
    monkeypatch.setattr(command_mod, "resolve_provider", lambda *_a, **_k: provider)
    monkeypatch.setattr(provider_mod, "probe_provider_availability", lambda _c: {"available": True})
    key = (config.database_path, "embeddings")
    def run():
        return maintenance_mod.run_incremental_embedding_pass(config.database_path, config)
    try:
        assert scheduler.schedule(key, run)
        assert started.wait(30)
        assert scheduler.quiesce(timeout=30)
        assert provider.completed == 1
        maintenance_mod._STOP_EMBEDDING_MAINTENANCE.clear()
        provider.delay = 0
        run()
        with sqlite3.connect(config.database_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM lcm_chunk_meta").fetchone()[0] == expected
            assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight").fetchone()[0] == 0
    finally:
        provider.delay = 0
        assert scheduler.drain({key}, timeout=30)


@pytest.fixture
def scheduler(monkeypatch):
    scheduler = engine_mod._RollupMaintenanceScheduler(
        kind="incremental embedding", thread_name="lcm-embedding-maintenance",
    )
    monkeypatch.setattr(engine_mod, "_EMBEDDING_MAINTENANCE_SCHEDULER", scheduler)
    return scheduler


@pytest.fixture
def make_engine(tmp_path, scheduler):
    engines = []

    def make(name="shared"):
        engine = LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / f"{name}.db"),
                embeddings_enabled=True,
                temporal_rollups_enabled=False,
            ),
            hermes_home=str(tmp_path / "home"),
        )
        engines.append(engine)
        return engine

    yield make
    for engine in engines:
        assert engine.drain_embedding_maintenance(timeout=5)
        engine.shutdown()


def _bind(engine, session="session"):
    engine.on_session_start(session, platform="cli", context_length=200000)


def _publish_leaf(engine):
    node = SummaryNode(
        session_id=engine._session_id, depth=0, summary="retirement test leaf",
        created_at=1.0, latest_at=1.0,
    )
    engine._dag.add_node(node)
    engine._invalidate_rollups_for_published_node(node)


@contextmanager
def _blocked_worker(scheduler):
    started, release = threading.Event(), threading.Event()
    key = ("other-database", "embeddings")
    try:
        assert scheduler.schedule(key, lambda: (started.set(), release.wait(10)))
        assert started.wait(5)
        yield release
    finally:
        release.set()
        assert scheduler.drain({key}, timeout=5)


def test_session_end_does_not_wait_and_cancels_follow_up(make_engine, scheduler, monkeypatch, caplog):
    active, started, unexpected = threading.Event(), threading.Event(), threading.Event()
    calls = []
    release = threading.Event()

    def run(*_args, **_kwargs):
        calls.append(True)
        active.set()
        started.set()
        if len(calls) > 2:
            unexpected.set()
        try:
            if len(calls) == 2:
                release.wait(10)
        finally:
            active.clear()

    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", run)
    engine = make_engine()
    try:
        _bind(engine)
        assert engine.drain_embedding_maintenance(timeout=5)
        started.clear()
        _publish_leaf(engine)
        assert started.wait(5)
        _publish_leaf(engine)  # Request the active key's single follow-up.
        before = time.monotonic()
        engine.on_session_end("session", [])
        assert time.monotonic() - before < 0.5
        assert active.is_set()  # Session rotation must not wait for this pass.
        assert not any("timeout" in record.getMessage().lower() for record in caplog.records)
        _publish_leaf(engine)  # Retirement also suppresses new scheduling.
        assert engine.drain_embedding_maintenance(timeout=0)
        release.set()
        assert scheduler.drain({engine._rollup_maintenance_key("embeddings")}, timeout=5)
        assert not active.is_set()
        assert not unexpected.wait(0.5)
        assert len(calls) == 2
        assert engine.drain_embedding_maintenance(timeout=0)
    finally:
        release.set()


def test_session_end_drops_sole_owners_queued_pass(make_engine, scheduler, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", lambda *_a, **_k: ran.set())
    engine = make_engine()
    key = engine._rollup_maintenance_key("embeddings")
    exclusive_key = ("operator-database", "embeddings")
    assert scheduler.try_acquire_exclusive(exclusive_key)
    try:
        with _blocked_worker(scheduler) as release:
            _bind(engine)
            assert key in scheduler._queued_keys
            engine.on_session_end("session", [])
            assert key not in scheduler._queued_keys
            assert engine.drain_embedding_maintenance(timeout=0)
            assert exclusive_key in scheduler._exclusive_keys
            release.set()
            assert scheduler.drain({key}, timeout=5)
            assert not ran.is_set()
    finally:
        scheduler.release_exclusive(exclusive_key)


def test_session_end_preserves_other_owners_queued_pass(make_engine, scheduler, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", lambda *_a, **_k: ran.set())
    first, second = make_engine(), make_engine()
    key = first._rollup_maintenance_key("embeddings")
    with _blocked_worker(scheduler) as release:
        _bind(first, "first")
        _bind(second, "second")
        assert scheduler._key_owners[key] == {
            first._embedding_maintenance_owner, second._embedding_maintenance_owner,
        }
        first.on_session_end("first", [])
        assert first.drain_embedding_maintenance(timeout=0)
        assert scheduler._key_owners[key] == {second._embedding_maintenance_owner}
        assert key in scheduler._queued_keys
        release.set()
        assert second.drain_embedding_maintenance(timeout=5)
        assert ran.is_set()


def test_new_bind_rearms_embedding_maintenance(make_engine, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", lambda *_a, **_k: ran.set())
    engine = make_engine()
    _bind(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    engine.on_session_end("session", [])
    ran.clear()
    _publish_leaf(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    assert not ran.is_set()

    _bind(engine, "next-session")
    assert engine.drain_embedding_maintenance(timeout=5)
    assert ran.is_set()  # Bind itself schedules again.
    ran.clear()
    _publish_leaf(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    assert ran.is_set()


@pytest.mark.parametrize("short_pass", [False, True], ids=["timeout", "finished"])
def test_quiesce_bounds_wait_drops_pending_and_closes(scheduler, caplog, short_pass):
    started, release = threading.Event(), threading.Event()
    pending_ran = threading.Event()
    key = ("embedding-database", "embeddings")
    queued_key = ("queued-database", "embeddings")
    owner = object()

    def run():
        started.set()
        release.wait(0.3 if short_pass else 10)

    try:
        assert scheduler.schedule(key, run, owner=owner)
        assert started.wait(5)
        assert scheduler.schedule(key, pending_ran.set, owner=owner)
        assert scheduler.schedule(queued_key, pending_ran.set, owner=owner)
        with caplog.at_level(logging.WARNING, logger=engine_mod.__name__):
            before = time.monotonic()
            result = scheduler.quiesce(timeout=1 if short_pass else 0.1)
            elapsed = time.monotonic() - before
        assert result is short_pass
        assert (0.2 if short_pass else 0.08) <= elapsed < (1 if short_pass else 0.5)
        assert any(
            "timeout" in record.getMessage().lower()
            and f"database={key[0]}" in record.getMessage()
            and f"scope={key[1]}" in record.getMessage()
            for record in caplog.records
        ) is (not short_pass)
        assert not scheduler._queued_keys
        assert scheduler._follow_up_key is None
        with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
            assert not scheduler.schedule(queued_key, pending_ran.set)
            assert not scheduler.schedule(queued_key, pending_ran.set)
        assert sum("scheduler is closed" in record.getMessage() for record in caplog.records) == 1
        assert (key in scheduler._active_keys) is (not short_pass)
    finally:
        release.set()
        assert scheduler.drain({key, queued_key}, timeout=5)
    assert scheduler.drain_owner(owner, timeout=0)
    assert not pending_ran.is_set()


def test_shutdown_retires_before_closing_stores(make_engine, monkeypatch):
    active, started, release = threading.Event(), threading.Event(), threading.Event()

    def run(*_args, **_kwargs):
        active.set()
        started.set()
        try:
            release.wait(0.3)
        finally:
            active.clear()

    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", run)
    engine = make_engine()
    active_at_close = []
    close = engine._store.close

    def close_store():
        active_at_close.append(active.is_set())
        close()

    monkeypatch.setattr(engine._store, "close", close_store)
    try:
        _bind(engine)
        assert started.wait(5)
        before = time.monotonic()
        engine.shutdown()
        assert 0.2 <= time.monotonic() - before < 2.5
        assert active_at_close == [False]
        assert not active.is_set()
        assert engine.drain_embedding_maintenance(timeout=0)
    finally:
        release.set()


def test_process_exit_waits_for_active_embedding_pass(tmp_path):
    marker = tmp_path / "embedding-finished"
    script = """
import importlib.util
import os
import sys
import threading
from pathlib import Path

repo, marker = sys.argv[1:]
sys.path.insert(0, repo)
spec = importlib.util.spec_from_file_location(
    "hermes_lcm", Path(repo) / "__init__.py", submodule_search_locations=[repo],
)
package = importlib.util.module_from_spec(spec)
sys.modules["hermes_lcm"] = package
from hermes_lcm import engine, command, embedding_maintenance, embedding_provider
from tests.test_issue_1330_retire_embedding_maintenance import _seed_batches, _SlowBatches

os.environ["LCM_EMBEDDING_BACKFILL_BATCH_SIZE"] = "40"
started = threading.Event()
config = _seed_batches(Path(marker).parent / "exit.db", "chunks")
provider = _SlowBatches(started, Path(marker))
command.resolve_provider = lambda *_a, **_k: provider
embedding_provider.probe_provider_availability = lambda _c: {"available": True}
def job():
    embedding_maintenance.run_incremental_embedding_pass(config.database_path, config)
    Path(marker + ".pass").write_text(str(provider.completed))

assert engine._EMBEDDING_MAINTENANCE_SCHEDULER.schedule(("exit-test", "embeddings"), job)
assert started.wait(30)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1]), str(marker)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text() == "1"  # Written by the completed batch, not pass cleanup.
    assert Path(str(marker) + ".pass").read_text() == "1"


def test_in_place_compression_rearms_embedding_maintenance(make_engine, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", lambda *_a, **_k: ran.set())
    engine = make_engine()
    _bind(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    engine.on_session_end("session", [])
    engine.on_session_start(
        "session", boundary_reason="compression", old_session_id="session",
        platform="cli", context_length=200000,
    )
    assert engine._lifecycle.get_by_conversation(engine._conversation_id).current_session_id == "session"
    assert engine.drain_embedding_maintenance(timeout=5)
    ran.clear()
    _publish_leaf(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    assert ran.is_set()


def test_unadopted_compaction_rebind_rearms_embedding_maintenance(make_engine, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", lambda *_a, **_k: ran.set())
    engine = make_engine()
    _bind(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    engine.on_session_end("session", [])
    state = engine._lifecycle.get_by_conversation(engine._conversation_id)
    assert state.current_session_id is None
    assert state.last_finalized_session_id == "session"
    engine._rebind_after_unadopted_compaction_commit()
    assert engine._lifecycle.get_by_conversation(engine._conversation_id).current_session_id == "session"
    ran.clear()
    _publish_leaf(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    assert ran.is_set()


@pytest.mark.parametrize("continuation", ["in-place", "unadopted"])
def test_continuation_reschedules_retired_leaf_without_publication(
    make_engine, scheduler, monkeypatch, continuation,
):
    passes = []
    engine = make_engine()

    def run(*_args, **_kwargs):
        passes.append([
            node.summary for node in engine._dag.get_session_nodes("session")
        ])

    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", run)
    _bind(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    passes.clear()
    key = engine._rollup_maintenance_key("embeddings")
    with _blocked_worker(scheduler) as release:
        _publish_leaf(engine)
        assert key in scheduler._queued_keys
        engine.on_session_end("session", [])
        assert engine._embedding_maintenance_retired
        assert key not in scheduler._queued_keys
        if continuation == "in-place":
            engine.on_session_start(
                "session", boundary_reason="compression", old_session_id="session",
                platform="cli", context_length=200000,
            )
        else:
            engine._rebind_after_unadopted_compaction_commit()
        assert not engine._embedding_maintenance_retired
        release.set()
        assert engine.drain_embedding_maintenance(timeout=5)
        assert passes == [["retirement test leaf"]]


@pytest.mark.parametrize("ended_session", ["auxiliary", "stale", "same-id-bypass"])
def test_nonforeground_end_preserves_embedding_maintenance(
    make_engine, scheduler, monkeypatch, ended_session,
):
    ran = threading.Event()
    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", lambda *_a, **_k: ran.set())
    engine = make_engine()
    key = engine._rollup_maintenance_key("embeddings")
    with _blocked_worker(scheduler) as release:
        _bind(engine)
        if ended_session == "auxiliary":
            class ChildAgent:
                session_id = "auxiliary"
                _parent_session_id = "session"
                log_prefix = "[subagent-test] "
                _delegate_depth = 1

                def start(self):
                    engine.on_session_start(self.session_id, platform="subagent", context_length=200000)

            ChildAgent().start()
            assert engine._has_auxiliary_lineage_session("auxiliary")
        elif ended_session == "same-id-bypass":
            # A bypass callback can reuse the foreground id without ending it.
            monkeypatch.setattr(engine, "_ended_session_directly_bypasses_lcm", lambda _sid: True)
            ended_session = "session"
        assert engine._session_id == "session"
        assert key in scheduler._queued_keys
        engine.on_session_end(ended_session, [])
        assert not engine._embedding_maintenance_retired
        assert key in scheduler._queued_keys
        assert engine._embedding_maintenance_owner in scheduler._key_owners[key]
        release.set()
        assert engine.drain_embedding_maintenance(timeout=5)
        assert ran.is_set()
    ran.clear()
    _publish_leaf(engine)
    assert engine.drain_embedding_maintenance(timeout=5)
    assert ran.is_set()


def test_session_end_then_shutdown_waits_before_closing_stores(make_engine, scheduler, monkeypatch):
    active, started, release = threading.Event(), threading.Event(), threading.Event()

    def run(*_args, **_kwargs):
        active.set()
        started.set()
        try:
            release.wait(0.4)
        finally:
            active.clear()

    monkeypatch.setattr(engine_mod, "run_incremental_embedding_pass", run)
    engine = make_engine()
    active_at_close = []
    close = engine._store.close

    def close_store():
        active_at_close.append(active.is_set())
        close()

    monkeypatch.setattr(engine._store, "close", close_store)
    try:
        _bind(engine)
        assert started.wait(5)
        before = time.monotonic()
        engine.on_session_end("session", [])
        assert time.monotonic() - before < 0.5
        assert active.is_set()
        before = time.monotonic()
        engine.shutdown()
        assert 0.2 <= time.monotonic() - before < 2.5
        assert active_at_close == [False]
        assert not active.is_set()
        assert not scheduler._retired_active_keys
    finally:
        release.set()
