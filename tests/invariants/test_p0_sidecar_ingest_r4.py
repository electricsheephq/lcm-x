"""T1-604c sidecar recovery when session-end ingest cannot open a writer."""

import sqlite3
from contextlib import ExitStack, closing, contextmanager

import pytest

from hermes_lcm import db_bootstrap
from tests.invariants.test_p0_sidecar_ingest import BLOCK, SESSION, _carrier, _engine, _user
from tests.invariants.test_p0_sidecar_ingest import host_uid as host_uid
from tests.invariants.test_p0_sidecar_ingest_r2 import COMPANION, CONVERSATION, _block, _host_sql_rows, _rebind
from tests.test_compression_boundary import _compacted_engine


@contextmanager
def _sqlite_lock(engine, begin="BEGIN IMMEDIATE"):
    with closing(sqlite3.connect(engine._store.db_path, timeout=0, isolation_level=None)) as locker:
        locker.execute(begin)
        yield


def _watch_busy(call, errors, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except sqlite3.OperationalError as exc:
        errors.append(exc.sqlite_errorcode)
        raise


def _assert_companion(engine, block, session=SESSION):
    rows = engine._store.get_session_messages("lcm-sidecar:" + session)
    assert [row["content"] for row in rows] == [block]


@pytest.mark.parametrize("route", ["suffix-append", "store-prefix", "recorded-prefix"])
def test_off_current_end_retries_sidecar_after_real_lock(tmp_path, monkeypatch, route):
    if route != "suffix-append":
        # WAL readers do not conflict with a held writer. A rollback-journal
        # fixture lets the actual prefix SELECT encounter an exclusive lock.
        monkeypatch.setattr(db_bootstrap, "_execute_wal_conversion_with_lock_retry", lambda conn: None)
    block = _block()
    carrier = _carrier()
    late = [dict(carrier, api_content=block), {"role": "assistant", "content": "Synthetic answer."},
            _user("Synthetic follow-up.")]
    with _engine(tmp_path) as engine:
        engine.ingest([carrier])
        before = _host_sql_rows(engine)
        _rebind(engine, "foreground", "synthetic-other-conversation")
        store = engine._store
        store.connection.execute("PRAGMA busy_timeout=0")
        errors = []
        with monkeypatch.context() as patch, ExitStack() as locks:
            if route == "suffix-append":
                append = engine._append_off_current_session_end_suffix
                patch.setattr(engine, "_append_off_current_session_end_suffix",
                              lambda *a, **kw: _watch_busy(append, errors, *a, **kw))
                locks.enter_context(_sqlite_lock(engine))
            else:
                assert store.connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
                get_range = store.get_range
                target_limit = len(late) if route == "store-prefix" else len(before) + 1

                def locked_probe(session, *args, **kwargs):
                    if session == SESSION and kwargs.get("limit") == target_limit:
                        locks.enter_context(_sqlite_lock(engine, "BEGIN EXCLUSIVE"))
                        return _watch_busy(get_range, errors, session, *args, **kwargs)
                    return get_range(session, *args, **kwargs)

                patch.setattr(store, "get_range", locked_probe)
            engine.on_session_end(SESSION, late)
        assert errors == [sqlite3.SQLITE_BUSY]
        assert _host_sql_rows(engine) == before
        assert not _host_sql_rows(engine, COMPANION)
        assert any(tail["view"] == late and tail["conversation_id"] == CONVERSATION
                   for tail in engine._pending_session_tails)
        _rebind(engine, "retry", "synthetic-retry-conversation")
        _rebind(engine, "retry-again", "synthetic-retry-conversation")
        _assert_companion(engine, block)
        durable = _host_sql_rows(engine)
        assert durable[:1] == before and len(durable) == len(late)
        assert not _host_sql_rows(engine, "foreground")
        assert not engine._pending_session_tails


def test_end_retries_sidecar_after_refused_compaction_restore_busy(tmp_path, monkeypatch):
    engine, original, _ = _compacted_engine(tmp_path, monkeypatch)
    try:
        snapshot = engine._compress_refusal
        assert snapshot["metadata"]
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_compaction_closing", True)
            engine.record_rejected_compaction()
        assert engine._compress_refusal is snapshot and snapshot["refused"]
        block = _block()
        late = [dict(original[0], api_content=block), *original[1:]]
        before = _host_sql_rows(engine, "S0")
        restore = engine._restore_refused_compaction
        errors = []
        with monkeypatch.context() as patch, _sqlite_lock(engine):
            patch.setattr(engine, "_restore_refused_compaction",
                          lambda *a, **kw: _watch_busy(restore, errors, *a, **kw))
            engine.on_session_end("S0", late)
        assert errors == [sqlite3.SQLITE_BUSY]
        assert engine._compress_refusal is snapshot
        assert _host_sql_rows(engine, "S0") == before
        assert not _host_sql_rows(engine, "lcm-sidecar:S0")
        engine._retry_pending_session_tails()
        engine._retry_pending_session_tails()
        _assert_companion(engine, block, "S0")
        assert _host_sql_rows(engine, "S0") == before
        assert not engine._pending_session_tails
    finally:
        engine.shutdown()


@pytest.mark.parametrize("new_row", [False, True], ids=["frozen-replay", "warm-tail"])
def test_current_end_admission_busy_retries_sidecar(tmp_path, monkeypatch, new_row):
    block = _block()
    carrier = _carrier()
    late = [dict(carrier, api_content=block)]
    if new_row:
        late.append({"role": "assistant", "content": "Synthetic final reply."})
    with _engine(tmp_path) as engine:
        engine.ingest([carrier])
        before = _host_sql_rows(engine)
        transaction = engine._ingest_messages_transaction
        frozen = []

        def read_only_attempt(*args, **kwargs):
            frozen.append(engine._ingest_read_only.sqlite_errorcode)
            return transaction(*args, **kwargs)

        with monkeypatch.context() as patch, _sqlite_lock(engine):
            patch.setattr(engine, "_ingest_messages_transaction", read_only_attempt)
            engine.on_session_end(SESSION, late)
        assert frozen == [sqlite3.SQLITE_BUSY]
        assert [tail["kind"] for tail in engine._pending_session_tails if not tail.get("sidecars_only")] == (
            ["warm"] if new_row else [])
        assert _host_sql_rows(engine) == before
        assert not _host_sql_rows(engine, COMPANION)
        _rebind(engine, "retry", "synthetic-retry-conversation")
        _rebind(engine, "retry-again", "synthetic-retry-conversation")
        _assert_companion(engine, block)
        durable = _host_sql_rows(engine)
        assert durable[:1] == before and len(durable) == len(late)
        assert not engine._pending_session_tails


def test_full_and_sidecars_only_cold_retries_store_one_copy(tmp_path, monkeypatch):
    carrier = _carrier()
    late = [dict(carrier, api_content=BLOCK), {"role": "assistant", "content": "Synthetic final reply."}]
    with _engine(tmp_path) as engine:
        engine.ingest([carrier])
        with monkeypatch.context() as patch, _sqlite_lock(engine):
            def busy_read(*args, **kwargs):
                # Model an earlier frozen-read failure using SQLite's actual admission error.
                raise engine._ingest_read_only

            patch.setattr(engine, "_ingest_messages_transaction", busy_read)
            engine.on_session_end(SESSION, late)
        assert not _host_sql_rows(engine, COMPANION)
        tails = list(engine._pending_session_tails)
        recover = engine._persist_ingest_sidecars
        attempts = []

        def observed_recovery(session, conversation, source, view, *args, **kwargs):
            attempts.append((view, kwargs.get("retry_store")))
            return recover(session, conversation, source, view, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(engine, "_persist_ingest_sidecars", observed_recovery)
            engine._retry_pending_session_tails()
            engine._retry_pending_session_tails()
        _assert_companion(engine, BLOCK)
        assert [tail["kind"] for tail in tails] == ["cold", "cold"]
        assert attempts == [(late, None), (late, engine._store)]
        assert len(_host_sql_rows(engine)) == len(late)
        assert not engine._pending_session_tails


@pytest.mark.parametrize("hook", [False, True], ids=["genuine-block", "then-hook-quote"])
def test_carrier_delimiter_inside_steer_does_not_cut_block(tmp_path, hook):
    block = _block("let's go\n\nuse staging")
    quoted = _block(message_id="synthetic-unseen-quote")
    api = block + (f"\n\ngo\n\n<lcm-answer-brief>{quoted}</lcm-answer-brief>" if hook else "")
    with _engine(tmp_path) as engine:
        view = [_user("go", api_content=api)]
        engine.ingest(view)
        engine.ingest(view)
        _assert_companion(engine, block)
