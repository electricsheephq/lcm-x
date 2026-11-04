"""P1e PR-B: raw cold tails preserve final messages and session ownership (I1/I2)."""

import copy
import logging
import sqlite3

import pytest

from hermes_lcm import maintenance_scheduler
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore
from tests.invariants.test_p1e_decided_tails import contents, history, queue, replay
from tests.test_issue_2055_ingest_ledger_writes import count, engines as engines, sql_rows
from tests.test_host_uid_shadow import _m
from tests.test_host_uid_emit_carry import _ContainsPattern


def queue_cold(engine, monkeypatch, view=None, *, read="get_session_count", error=sqlite3.OperationalError,
               reconcile=True):
    view = history() if view is None else view
    engine.ingest(copy.deepcopy(view[:2]))
    engine._ingest_cursor_needs_reconcile = reconcile

    def failed(*args, **kwargs):
        raise error("database is locked" if issubclass(error, sqlite3.Error) else "injected frozen read failure")

    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, read, failed)
            engine.on_session_end("S", view)
    finally:
        locker.rollback()
        locker.close()
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    assert tail["kind"] == "cold" and tail["view"] == view
    assert engine in maintenance_scheduler._PENDING_TAIL_ENGINES
    return tail


@pytest.mark.parametrize("error", [sqlite3.OperationalError, ValueError])
def test_1805_locked_count_read_queues_cold_tail(engines, monkeypatch, caplog, error):
    engine = engines()
    view = history()
    tail = queue_cold(engine, monkeypatch, view, error=error)
    assert tail["view"] is not view and tail["view"][0] is not view[0]
    assert count(engine, "tail.queued.cold." + error.__name__) == 1
    assert "retry pending" in caplog.text and error.__name__ in caplog.text
    view[-1]["content"] = "host mutation"
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


@pytest.mark.parametrize("reconcile", [False, True])
def test_1805_locked_tail_read_queues_cold_tail(engines, monkeypatch, reconcile):
    engine = engines()
    queue_cold(engine, monkeypatch, read="get_session_tail", reconcile=reconcile)
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


def test_dedupe_hit_queues_no_cold_tail(engines):
    engine = engines()
    queue(engine)
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", history())
        assert len(engine._pending_session_tails) == 1
        assert engine._pending_session_tails[0]["kind"] == "warm"
        assert count(engine, "tail.queued.cold.OperationalError") == 0
    finally:
        locker.rollback()
        locker.close()


def test_second_end_after_rolled_back_append_queues_cold(engines, monkeypatch):
    engine = engines()
    engine.ingest(history()[:2])
    append, attempts = engine._append_ingest_batch, []

    def busy_once(batch):
        attempts.append(True)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        return append(batch)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_ingest_batch", busy_once)
        engine.on_session_end("S", history())
    assert attempts == [True, True] and not engine._pending_session_tails
    # The first attempt reached append, but its warm queue rolled back.
    assert engine.__dict__.get("_frozen_tail_reached") is True
    view = [*history(), _m("user", "E", 5., "E")]
    queue_cold(engine, monkeypatch, view)
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D", "E"]


def test_cold_tail_retried_after_other_ingests(engines, monkeypatch):
    engine = engines()
    queue_cold(engine, monkeypatch)
    engine.ingest(history()[:3])
    replay(engine, [_m("user", "other", 9.)], session="other", conversation="other")
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert not engine._pending_session_tails


@pytest.mark.parametrize("case", ["unreplayed", "covered", "longer_store", "mismatch_kept"])
def test_cold_off_current_s3(engines, monkeypatch, caplog, case):
    engine = engines()
    tail = queue_cold(engine, monkeypatch)
    if case in {"covered", "longer_store"}:
        replay(engine, history()[2:])
    if case == "longer_store":
        replay(engine, [_m("user", "new", 8.)])
    if case == "mismatch_kept":
        engine._store._conn.execute("UPDATE messages SET content='mismatch' WHERE store_id=1")
        engine._store._conn.commit()
    engine._session_id = "next"
    before_views = sql_rows(engine, "ingest_views")
    append = engine._append_off_current_session_end_suffix

    def owned(*args, **kwargs):
        assert engine._store._transaction_owner.owns_connection(engine._store._conn)
        return append(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_append_off_current_session_end_suffix", owned)
        patch.setattr(engine._lifecycle, "finalize_session", lambda *a, **k: pytest.fail("retry finalized"))
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
    if case == "mismatch_kept":
        assert engine._pending_session_tails == [tail]
        assert contents(engine) == ["mismatch", "B"]
        assert count(engine, "tail.cold_unmatched") == 2
        assert sum("cold tail prefix mismatch" in r.getMessage() for r in caplog.records) == 1
    else:
        assert contents(engine) == ["A", "B", "C", "D"] + (["new"] if case == "longer_store" else [])
        assert not engine._pending_session_tails
    assert sql_rows(engine, "ingest_views") == before_views


@pytest.mark.parametrize("route", ["sibling", "unbound_store", "different_conversation"])
def test_sibling_cold_uses_s3(engines, monkeypatch, tmp_path, route):
    engine = engines()
    tail = queue_cold(engine, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_messages", lambda *a, **k: pytest.fail("cold current route"))
        if route == "sibling":
            sibling = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
            try:
                sibling.on_session_start("next", platform="cli")
            finally:
                sibling.shutdown()
        elif route == "different_conversation":
            engine._conversation_id = "other"
            engine._retry_pending_session_tails()
        else:
            rebound = MessageStore(tmp_path / "rebound.db")
            try:
                patch.setattr(engine, "_store", rebound)
                engine._retry_pending_session_tails()
                assert engine._pending_session_tails == [tail]
                assert count(engine, "tail.cold_store_unbound") == 1
                assert not rebound.get_session_messages("S")
            finally:
                rebound.close()
    if route != "unbound_store":
        assert contents(engine) == ["A", "B", "C", "D"]
        assert not engine._pending_session_tails


def test_cold_retry_busy_never_nests(engines, monkeypatch):
    engine = engines()
    tail = queue_cold(engine, monkeypatch)
    engine._session_end_flush_active = True
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine._retry_pending_session_tails(exiting=True)
        assert engine._pending_session_tails == [tail]
        assert not engine._session_end_flush_active
        assert count(engine, "tail.retry.deferred") == 1
        assert count(engine, "tail.unflushed") == len(history())
    finally:
        locker.rollback()
        locker.close()
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]


def test_poisoned_tail_does_not_block(engines, monkeypatch, caplog):
    engine = engines()
    tail = queue(engine)
    poison = {**tail, "messages": [_m("user", "poison")]}
    engine._pending_session_tails.insert(0, poison)
    match = engine._match_pending_tail

    def failed(candidate):
        if candidate is poison:
            raise ValueError("poisoned tail")
        return match(candidate)

    with monkeypatch.context() as patch, caplog.at_level(logging.WARNING):
        patch.setattr(engine, "_match_pending_tail", failed)
        engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D"]
    assert engine._pending_session_tails == [poison]
    assert count(engine, "tail.retry_failed") == 1
    assert "ValueError" in caplog.text
    engine._pending_session_tails.clear()  # Synthetic malformed input is not a recoverable host view.


def test_cold_current_writes_no_view(engines, monkeypatch):
    engine = engines()
    engine._config.ignore_message_patterns = ["^ignore$"]
    engine._compiled_ignore_message_patterns = [_ContainsPattern("ignore")]
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["api_key"]
    view = [*history()[:2], _m("user", "ignore", 3.),
            _m("assistant", "api_key=sk-user1234567890abcdef", 4.)]
    queue_cold(engine, monkeypatch, view)
    before_views = sql_rows(engine, "ingest_views")
    with monkeypatch.context() as patch:
        patch.setattr(engine._lifecycle, "finalize_session", lambda *a, **k: pytest.fail("retry finalized"))
        engine._retry_pending_session_tails()
    assert sql_rows(engine, "ingest_views") == before_views
    assert len(sql_rows(engine, "occurrence_bindings")) == 3
    assert contents(engine)[:2] == ["A", "B"]
    assert len(contents(engine)) == 3 and contents(engine)[-1] != view[-1]["content"]
    assert count(engine, "views_written") == len(before_views)
    assert not engine._pending_session_tails


@pytest.mark.parametrize("error", [KeyboardInterrupt, ValueError])
def test_committed_uid_disagreement_tail_removed_after_counter_interrupt(engines, monkeypatch, error):
    engine = engines()
    queue(engine)
    replay(engine, [{**row, "message_uid": "foreign-" + row["message_uid"]} for row in history()[2:]])
    counter, interrupted = engine._ledger_count, []

    def after_commit(key, *args, **kwargs):
        if key == "ledger.tail.retry.replay_rows" and len(contents(engine)) == 6 and not interrupted:
            interrupted.append(True)
            raise error("after COMMIT")
        return counter(key, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ledger_count", after_commit)
        engine._retry_pending_session_tails()
    assert interrupted == [True]
    engine._retry_pending_session_tails()
    assert contents(engine) == ["A", "B", "C", "D", "C", "D"]
    assert not engine._pending_session_tails
