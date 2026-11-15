"""Round 3e: cancellation and implicit admission (I1/I2)."""
import sqlite3

import pytest

from hermes_lcm.store import MessageStore
from tests.test_2293_r3d_regressions import _stored, _warm


@pytest.mark.parametrize('exit_type', [KeyboardInterrupt, SystemExit])
def test_cancellation_before_virtual_scope_sync(tmp_path, monkeypatch, exit_type):
    results = []
    for lazy in (False, True):
        engine, view = _warm(tmp_path / str(lazy), lazy)
        store = engine._store
        before = _stored(engine)
        original_sync, original_begin, original_commit = (MessageStore._sync_ingest_prelock,
            MessageStore._ingest_prelock_before_begin, MessageStore._commit)
        fired, cancelled = [], []
        changes = store._conn.total_changes

        def foreign():
            other = sqlite3.connect(store.db_path)
            try:
                other.execute("INSERT INTO metadata VALUES ('r3e-foreign', 'x')")
                other.commit()
            finally:
                other.close()
            fired.append(True)

        def sync(self):
            result = original_sync(self)
            if self is store and lazy and self._ingest_is_prelock() and not self._conn.in_transaction and not fired:
                foreign()
            return result

        def begin(self):
            if self is store and not lazy and not fired:
                foreign()
            return original_begin(self)

        def commit(self):
            if self is store and not cancelled and self._conn.total_changes > changes:
                state = self._ingest_prelock_state
                if lazy:
                    assert state.phase == 'prelock' and state.scopes
                    assert any(not scope['materialized'] for scope in state.scopes)
                assert self._conn.in_transaction
                cancelled.append(True)
                raise exit_type()
            return original_commit(self)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(MessageStore, '_sync_ingest_prelock', sync)
                patch.setattr(MessageStore, '_ingest_prelock_before_begin', begin)
                patch.setattr(MessageStore, '_commit', commit)
                with pytest.raises(exit_type):
                    engine.ingest(view)
            assert fired and cancelled
            assert _stored(engine) == before
        finally:
            engine.shutdown()
        cold, _ = _warm_reopen(tmp_path / str(lazy), lazy)
        try:
            results.append(_stored(cold))
            assert results[-1] == before
            assert cold._store._conn.execute("SELECT key, value FROM metadata WHERE key='r3e-foreign'").fetchall() == [
                ('r3e-foreign', 'x')]
        finally:
            cold.shutdown()
    assert results[0] == results[1]


def _warm_reopen(root, lazy):
    from tests.test_2293_prelock import _engine
    return _engine(root, prelock=lazy), None


@pytest.mark.parametrize('release_for_retry', [False, True])
def test_failed_implicit_admission_preserves_rows(tmp_path, monkeypatch, release_for_retry):
    engine, view = _warm(tmp_path / 'lazy', True)
    store = engine._store
    conn = store._conn
    conn.execute('PRAGMA busy_timeout=400')
    holder = sqlite3.connect(store.db_path, timeout=0, isolation_level=None)
    holder.execute('BEGIN IMMEDIATE')
    calls, timeouts = [], []
    transaction = engine._ingest_messages_transaction
    seam = MessageStore._ingest_prelock_before_begin
    failed = MessageStore._ingest_prelock_write_failed
    before = _stored(engine)

    def raw(*args, **kwargs):
        calls.append(True)
        try:
            conn.execute("INSERT INTO metadata VALUES ('r3e-implicit', 'durable')")
        except sqlite3.Error as error:
            assert error.sqlite_errorcode == sqlite3.SQLITE_BUSY
            assert holder.in_transaction and not conn.in_transaction
            timeouts.append(True)
            store._ingest_prelock_write_failed(error)
            raise
        return transaction(*args, **kwargs)

    def begin(self):
        if self is store:
            if release_for_retry:
                holder.rollback()
        return seam(self)

    def poison(self, error):
        result = failed(self, error)
        if self is store:
            assert self._ingest_prelock_state.poisoned
        return result

    try:
        with monkeypatch.context() as patch:
            patch.setattr(engine, '_ingest_messages_transaction', raw)
            patch.setattr(MessageStore, '_ingest_prelock_before_begin', begin)
            patch.setattr(MessageStore, '_ingest_prelock_write_failed', poison)
            engine.ingest(view)
        assert timeouts == [True]
        if release_for_retry:
            assert len(calls) == 2
            assert conn.execute("SELECT value FROM metadata WHERE key='r3e-implicit'").fetchall() == [('durable',)]
            assert [row[3] for row in _stored(engine)] == [view[i]['content'] for i in (0, 1, 3, 4)]
        else:
            assert len(calls) == 1
            assert _stored(engine) == before
            assert conn.execute("SELECT value FROM metadata WHERE key='r3e-implicit'").fetchall() == []
    finally:
        holder.rollback()
        holder.close()
        engine.shutdown()
