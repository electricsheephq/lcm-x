"""Deterministic shutdown/compaction ordering, using isolated synthetic stores."""

import logging
import sqlite3
import threading
import time
from contextlib import closing
from types import SimpleNamespace

import pytest

from hermes_lcm.engine_registry import ActiveEngineUseStatus, use_active_lcm_engine
from hermes_lcm.tokens import count_messages_tokens
from tests.harness import make_engine, pin_char_counter


def _engine(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    engine = make_engine(
        tmp_path, window=200000, session_id="shutdown-race",
        conversation_id="shutdown-race-conversation", embeddings_enabled=False,
        fresh_tail_count=2, leaf_chunk_tokens=120, threshold_full_sweep_enabled=False,
    )
    engine.threshold_tokens = 1
    return engine


def _messages():
    return [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"FACT-{i} " + (f"detail-{i} " * 60)}
        for i in range(10)
    ]


def _assert_durable_rows(engine, before):
    # Read the on-disk store independently, including after engine shutdown.
    with closing(sqlite3.connect(engine._store.db_path)) as conn:
        after = conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
        assert after == before, "compaction must preserve every original row (I2)"
        assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        node_ids = {row[0] for row in conn.execute("SELECT node_id FROM summary_nodes")}
    # The DAG shares the database; SQLite integrity includes its committed tables.
    return node_ids


@pytest.fixture
def blocked_pass(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch)
    messages = _messages()
    engine.ingest(messages)
    before = engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
    started, release = threading.Event(), threading.Event()
    shutdown_waiting, shutdown_finished = threading.Event(), threading.Event()
    workers, errors, compressed = [], [], []

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

    def summarize(chunk, **_kwargs):
        started.set()
        assert release.wait(10), "summarizer was never released"
        return chunk, count_messages_tokens(chunk), "Earlier facts retained.", 1, 0

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    finish_budget = engine._finish_foreground_budget

    def finish_with_live_connections(*args, **kwargs):
        assert all(service._conn is not None for service in (engine._store, engine._dag, engine._lifecycle))
        if hasattr(engine, "_compaction_inflight"):
            assert engine._compaction_inflight > 0, "counted region ended before compress finalizers"
        return finish_budget(*args, **kwargs)

    monkeypatch.setattr(engine, "_finish_foreground_budget", finish_with_live_connections)
    # Observe the real drain without manufacturing the production condition on
    # the base tree. There, shutdown's completion signals the same rendezvous,
    # making T1 fail on the behavioural assertion below, without a live SQL race.
    gate = getattr(engine, "_compaction_gate", None)
    if gate is not None:
        original_wait = gate.wait_for

        def observed_wait(predicate, timeout=None):
            shutdown_waiting.set()
            return original_wait(predicate, timeout)

        monkeypatch.setattr(gate, "wait_for", observed_wait)

    def shutdown():
        try:
            engine.shutdown()
        finally:
            shutdown_finished.set()
            shutdown_waiting.set()

    compression = launch(lambda: compressed.append(engine.compress(messages, force=True)))
    try:
        assert started.wait(5), "compress never reached its summarizer"
        yield SimpleNamespace(
            engine=engine, messages=messages, before=before, release=release,
            waiting=shutdown_waiting, finished=shutdown_finished, errors=errors,
            compressed=compressed, compression=compression, launch=launch, shutdown=shutdown,
        )
    finally:
        release.set()
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive(), "race worker did not exit"
        engine.shutdown()


def test_shutdown_waits_for_blocked_compress_and_preserves_rows(blocked_pass):
    """T1 / red proof: base shutdown finishes before the blocked pass is released."""
    race = blocked_pass
    shutdown = race.launch(race.shutdown)
    assert race.waiting.wait(5)
    assert not race.finished.is_set(), "shutdown closed connections under running compress"
    assert all(service._conn is not None for service in (
        race.engine._store, race.engine._dag, race.engine._lifecycle,
    ))
    race.release.set()
    race.compression.join(5)
    shutdown.join(5)
    assert not race.errors
    assert race.finished.is_set()
    assert not race.compression.is_alive()
    assert race.compressed[0] is race.messages
    assert race.engine._stable_use_closed
    assert race.engine._compaction_inflight == 0
    assert all(service._conn is None for service in (
        race.engine._store, race.engine._dag, race.engine._lifecycle,
    ))
    assert _assert_durable_rows(race.engine, race.before) == set()


def test_shutdown_timeout_keeps_connections_open_and_retry_closes(blocked_pass, monkeypatch, caplog):
    """T2: a timed-out drain leaves committed rows and live connections intact."""
    race = blocked_pass
    monkeypatch.setenv("LCM_SHUTDOWN_COMPACTION_WAIT_S", "0.2")
    caplog.set_level(logging.WARNING, logger="hermes_lcm.engine")
    started = time.monotonic()
    shutdown = race.launch(race.shutdown)
    shutdown.join(3)
    elapsed = time.monotonic() - started
    assert race.finished.is_set()
    assert 0.15 <= elapsed < 3
    assert not race.release.is_set()
    assert not race.engine._stable_use_closed
    for service in (race.engine._store, race.engine._dag, race.engine._lifecycle):
        assert service._conn.execute("SELECT 1").fetchone()[0] == 1
    warnings = [record.getMessage() for record in caplog.records
                if "shutdown: compaction still running" in record.getMessage()]
    assert warnings == ["shutdown: compaction still running after 0.2s; connections left open"]
    race.release.set()
    race.compression.join(5)
    assert not race.compression.is_alive()
    assert not race.errors
    assert race.compressed[0] is race.messages
    assert _assert_durable_rows(race.engine, race.before) == set()
    # A summary returning after closing stops before publication (Design 2).
    # Timeout still leaves connections usable for commits.
    assert race.engine._store.write_metadata_json(["shutdown-timeout-probe"], '"committed"')
    assert race.engine._store._conn.execute(
        "SELECT value FROM metadata WHERE key = 'shutdown-timeout-probe'",
    ).fetchone()[0] == '"committed"'
    race.engine.shutdown()
    assert race.engine._stable_use_closed
    assert all(service._conn is None for service in (
        race.engine._store, race.engine._dag, race.engine._lifecycle,
    ))


def test_timeout_allows_inflight_atomic_publication_to_commit(blocked_pass, monkeypatch):
    """T2's commit case: an admitted DAG transaction completes after timeout (I14)."""
    race = blocked_pass
    monkeypatch.setenv("LCM_SHUTDOWN_COMPACTION_WAIT_S", "0.2")
    publication_started, release_publication = threading.Event(), threading.Event()
    original_add = race.engine._dag.add_node

    def blocked_add(node, **kwargs):
        stage = kwargs["before_commit"]

        def before_commit(conn, node_id):
            stage(conn, node_id)
            publication_started.set()
            assert release_publication.wait(10), "publication was never released"

        return original_add(node, **{**kwargs, "before_commit": before_commit})

    monkeypatch.setattr(race.engine._dag, "add_node", blocked_add)
    try:
        race.release.set()
        assert publication_started.wait(5), "leaf never entered its atomic publication"
        shutdown = race.launch(race.shutdown)
        shutdown.join(3)
        assert race.finished.is_set()
        assert not release_publication.is_set()
        assert not race.engine._stable_use_closed
        assert all(service._conn is not None for service in (
            race.engine._store, race.engine._dag, race.engine._lifecycle,
        ))
        release_publication.set()
        race.compression.join(5)
        assert not race.compression.is_alive()
        assert not race.errors
        assert race.compressed[0] is race.messages
        nodes = _assert_durable_rows(race.engine, race.before)
        assert len(nodes) == 1
        lifecycle = race.engine._lifecycle.get_by_conversation("shutdown-race-conversation")
        assert lifecycle.current_frontier_store_id > 0
        race.engine.shutdown()
        assert race.engine._stable_use_closed
        assert _assert_durable_rows(race.engine, race.before) == nodes
    finally:
        release_publication.set()


def test_closing_and_closed_compress_are_identity_noops(tmp_path, monkeypatch):
    """T3: closing admission performs no writes or late-auxiliary reclassification."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _messages()
    engine.ingest(messages)
    before = tuple(service._conn.total_changes for service in (
        engine._store, engine._dag, engine._lifecycle,
    ))

    def forbidden():
        pytest.fail("closing compress entered its body")

    monkeypatch.setattr(engine, "_maybe_reclassify_late_auxiliary_before_compaction_write", forbidden)
    try:
        with engine._compaction_gate:
            engine._compaction_closing = True
        assert engine.compress(messages, force=True) is messages
        assert engine._compaction_inflight == 0
        assert tuple(service._conn.total_changes for service in (
            engine._store, engine._dag, engine._lifecycle,
        )) == before
        engine.shutdown()
        engine.shutdown()
        assert engine.compress(messages, bypass_cooldown=True) is messages
        assert all(service._conn is None for service in (engine._store, engine._dag, engine._lifecycle))
    finally:
        engine.shutdown()


def test_closing_exception_rolls_back_the_open_store_write(tmp_path, monkeypatch):
    """An error after closing begins still rolls back the store's open write (#582)."""
    engine = _engine(tmp_path, monkeypatch)
    messages = _messages()
    engine.ingest(messages)
    before = engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
    engine._store._conn.execute("CREATE TEMP TABLE partial_2101 (value TEXT)")
    assert not engine._store._conn.in_transaction
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", lambda chunk, **_kwargs: (
        chunk, count_messages_tokens(chunk), "Earlier facts retained.", 1, 0))

    def fail_inside_open_write(*_args, **_kwargs):
        engine._store._conn.execute("INSERT INTO partial_2101 VALUES ('partial')")
        assert engine._store._conn.in_transaction
        with engine._compaction_gate:
            engine._compaction_closing = True
        raise RuntimeError("write failed after shutdown began")

    monkeypatch.setattr(engine, "_record_compress_commit_proof", fail_inside_open_write)
    try:
        assert engine.compress(messages, force=True) is messages
        assert engine._compaction_inflight == 0
        assert not engine._store._conn.in_transaction, "the partial write stayed open"
        assert engine._store._conn.execute("SELECT COUNT(*) FROM partial_2101").fetchone() == (0,)
    finally:
        engine.shutdown()
    _assert_durable_rows(engine, before)

def test_blocked_compress_does_not_take_stable_use_lease(blocked_pass):
    """T4: the pre_llm_call lease route and post-call ingest remain concurrent."""
    race = blocked_pass
    results = []

    def use_engine():
        kwargs = {"session_id": "shutdown-race", "conversation_id": "shutdown-race-conversation"}
        # pre_llm_call uses this registry route into _run_stably.
        results.append(use_active_lcm_engine(
            lambda selected: selected._store.get_session_count("shutdown-race"), **kwargs,
        ))
        results.append(use_active_lcm_engine(
            lambda selected: selected.ingest([
                *race.messages, {"role": "user", "content": "Concurrent post-call ingest"},
            ]), **kwargs,
        ))

    worker = race.launch(use_engine)
    worker.join(3)
    assert not worker.is_alive(), "stable-use caller waited for the compaction summarizer"
    assert not race.release.is_set()
    assert not race.errors
    assert len(results) == 2
    assert all(result.status is ActiveEngineUseStatus.USED for result in results)
    assert results[0].value == len(race.messages)
    assert race.engine._store.get_session_count("shutdown-race") == len(race.messages) + 1
    race.release.set()
    race.compression.join(5)
    assert not race.compression.is_alive()
    assert not race.errors
