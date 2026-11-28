"""Main-only shutdown checkpoints; the rc2 T1-T4 file stays unchanged."""

import sqlite3
import threading
from contextlib import closing, contextmanager
from types import SimpleNamespace

import pytest

from tests.test_issue_1931_delegate_children import Agent, DELEGATE, child_view, host as _host
from tests.test_issue_2055_emission_writes import emissions, emission_views, prepare_path
from tests.test_issue_2055_ingest_ledger_writes import engines as _engines

host = _host
engines = _engines


def _set_closing(engine):
    with engine._compaction_gate:
        engine._compaction_closing = True


@contextmanager
def _race(engine, monkeypatch, *releases):
    """Observe the real shutdown drain and always release/join synthetic workers."""
    waiting, finished = threading.Event(), threading.Event()
    workers, errors = [], []
    wait_for = engine._compaction_gate.wait_for
    monkeypatch.setenv("LCM_SHUTDOWN_COMPACTION_WAIT_S", "10")

    def observed_wait(predicate, timeout=None):
        waiting.set()
        return wait_for(predicate, timeout)

    monkeypatch.setattr(engine._compaction_gate, "wait_for", observed_wait)

    def launch(call):
        def run():
            try:
                call()
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run, daemon=True)
        workers.append(worker)
        worker.start()
        return worker

    def shutdown():
        try:
            engine.shutdown()
        finally:
            finished.set()

    try:
        yield SimpleNamespace(launch=launch, shutdown=shutdown, waiting=waiting,
                              finished=finished, errors=errors)
    finally:
        for release in releases:
            release.set()
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive(), "shutdown race worker did not exit"
        engine.shutdown()


def _assert_closed(engine):
    assert engine._stable_use_closed
    assert engine._compaction_inflight == 0
    assert all(service._conn is None for service in (engine._store, engine._dag, engine._lifecycle))


def test_closing_before_commit_proof_writes_no_emissions(engines, monkeypatch):
    """A1: a pre-proof stop skips the main-only exit writer in shadow mode."""
    engine = engines(mode="shadow")
    messages = prepare_path(engine, monkeypatch)
    fit = engine._survival_fit

    def close_before_proof(*args, **kwargs):
        result = fit(*args, **kwargs)
        _set_closing(engine)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("pre-proof closing reached proof/exit bookkeeping")

    monkeypatch.setattr(engine, "_survival_fit", close_before_proof)
    monkeypatch.setattr(engine, "_record_compress_commit_proof", forbidden)
    monkeypatch.setattr(engine, "_compress_exit_write", forbidden)
    assert engine.compress(messages, current_tokens=100, force=True) is messages
    assert emissions(engine) == emission_views(engine) == []


def test_closing_after_commit_proof_keeps_exit_write_counted(engines, monkeypatch):
    """A2: post-proof closing keeps the returned vector and its shadow emission."""
    engine = engines(mode="shadow")
    messages = prepare_path(engine, monkeypatch)
    proof_done, exit_done = threading.Event(), threading.Event()
    release_proof, release_exit = threading.Event(), threading.Event()
    captured, compressed, written = [], [], []
    record, write = engine._record_compress_commit_proof, engine._compress_exit_write

    def close_after_proof(*args, **kwargs):
        capture = record(*args, **kwargs)
        assert capture[1], "test must exercise a real fresh emission descriptor"
        captured.append(capture)
        _set_closing(engine)
        proof_done.set()
        assert release_proof.wait(10), "proof was never released"
        return capture

    def observed_exit(returned, capture):
        assert engine._compaction_is_closing()
        assert engine._compaction_inflight == 1
        assert all(service._conn is not None for service in (engine._store, engine._dag, engine._lifecycle))
        write(returned, capture)
        written.extend(emissions(engine))
        assert written and emission_views(engine)
        exit_done.set()
        assert release_exit.wait(10), "exit writer was never released"

    monkeypatch.setattr(engine, "_record_compress_commit_proof", close_after_proof)
    monkeypatch.setattr(engine, "_compress_exit_write", observed_exit)
    with _race(engine, monkeypatch, release_proof, release_exit) as race:
        compression = race.launch(lambda: compressed.append(engine.compress(messages, current_tokens=100, force=True)))
        assert proof_done.wait(5)
        shutdown = race.launch(race.shutdown)
        assert race.waiting.wait(5)
        assert not race.finished.is_set()
        release_proof.set()
        assert exit_done.wait(5)
        assert not race.finished.is_set(), "shutdown did not wait for the exit writer"
        release_exit.set()
        compression.join(5)
        shutdown.join(5)
        assert not race.errors
        assert race.finished.is_set()
        assert compressed[0] is captured[0][0] and compressed[0] is not messages
        _assert_closed(engine)
        with closing(sqlite3.connect(engine._store.db_path)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM ledger_emissions").fetchone()[0] == len(written)


@pytest.mark.parametrize("helper_raises", [False, True])
def test_shutdown_waits_for_delegate_summarizer_and_preserves_child_rows(host, monkeypatch, helper_raises):
    """B: native summarisation is counted, closing is an identity return, and stored child rows survive."""
    load, clone = host
    ctx, _ = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    messages = child_view(3)
    started, release = threading.Event(), threading.Event()
    compressed, late_writes = [], []

    def summarize(rows, **kwargs):
        started.set()
        assert release.wait(10), "delegate summarizer was never released"
        return rows[-2:]

    compressor = SimpleNamespace(compress=summarize, compression_count=0)
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: compressor)
    monkeypatch.setattr(engine, "_bypass_compaction_target_tokens", lambda **kwargs: 100_000)
    bypass = engine._compress_lcm_bypassed_session

    def observed_bypass(*args, **kwargs):
        result = bypass(*args, **kwargs)
        if helper_raises:
            raise RuntimeError("synthetic bypass helper failure during closing")
        return result

    monkeypatch.setattr(engine, "_compress_lcm_bypassed_session", observed_bypass)

    def trace(sql):
        if engine._compaction_is_closing() and threading.current_thread() is compression:
            if sql.lstrip().split()[0].upper() in {"BEGIN", "SAVEPOINT", "INSERT", "UPDATE", "DELETE", "CREATE", "REPLACE"}:
                late_writes.append(sql)

    with _race(engine, monkeypatch, release) as race:
        compression = race.launch(lambda: compressed.append(child.compress(engine, messages)))
        assert started.wait(5)
        before = engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
        assert engine._store.get_session_count("child") == len(messages)
        assert engine._store._conn.execute(
            "SELECT parent_session_id FROM session_lineage WHERE child_session_id='child'",
        ).fetchone() == ("parent",)
        engine._store._conn.set_trace_callback(trace)
        shutdown = race.launch(race.shutdown)
        assert race.waiting.wait(5)
        assert not race.finished.is_set()
        assert all(service._conn is not None for service in (engine._store, engine._dag, engine._lifecycle))
        release.set()
        compression.join(5)
        shutdown.join(5)
        assert not race.errors
        assert race.finished.is_set()
        assert compressed[0] is messages
        assert not late_writes
        _assert_closed(engine)
        with closing(sqlite3.connect(engine._store.db_path)) as conn:
            assert conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == before
            assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_delegate_store_lock_wait_rechecks_closing(host, monkeypatch):
    """B: no child-row transaction starts if shutdown won during the store-lock wait."""
    load, clone = host
    ctx, module = load()
    engine = clone(ctx)
    engine.on_session_start("parent", platform="cli", context_length=200_000)
    child = Agent("child", **DELEGATE)
    child.start(engine)
    messages = child_view(1)
    lock = module("aux_session")._DELEGATE_STORE_LOCK
    entered, release = threading.Event(), threading.Event()
    compressed = []

    class ObservedLock:
        def __enter__(self):
            entered.set()
            assert release.wait(10), "child-store lock was never released"
            return lock.__enter__()

        def __exit__(self, *args):
            return lock.__exit__(*args)

    monkeypatch.setattr(module("aux_session"), "_DELEGATE_STORE_LOCK", ObservedLock())

    def forbidden(*args, **kwargs):
        pytest.fail("closing delegate pass reached its row store")

    monkeypatch.setattr(engine, "_store_delegate_child_turn", forbidden)
    with _race(engine, monkeypatch, release) as race:
        compression = race.launch(lambda: compressed.append(child.compress(engine, messages)))
        assert entered.wait(5)
        shutdown = race.launch(race.shutdown)
        assert race.waiting.wait(5)
        assert not race.finished.is_set()
        release.set()
        compression.join(5)
        shutdown.join(5)
        assert not race.errors
        assert race.finished.is_set()
        assert compressed[0] is messages
        _assert_closed(engine)
