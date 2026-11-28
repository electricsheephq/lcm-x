"""Failed restart BEGIN preserves the eager host contract and retry choice (I1)."""
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_lcm.sqlite_util import _IngestReclassify
from hermes_lcm.store import MessageStore
from tests.test_2293_prelock import SESSION, _a, _engine, _u


def _warm(root, lazy):
    root.mkdir()
    engine = _engine(root, prelock=lazy)
    engine._store._conn.execute('PRAGMA busy_timeout=40')
    engine._compiled_ignore_message_patterns = [SimpleNamespace(search=lambda text, **kw: 'IGNORE' in text)]
    base = [_u('U0 first', 1000), _a('R0 reply', 1000.5)]
    engine.ingest(base)
    return engine, base + [_a('IGNORE me', 1001), _u('U1 new', 1001.2), _a('R1 new', 1001.5)]


def _foreign_before_implicit_write(patch, engine, root):
    original = MessageStore._sync_ingest_prelock
    fired = []

    def sync(store):
        result = original(store)
        if store is engine._store and store._ingest_is_prelock() and not store._conn.in_transaction and not fired:
            with sqlite3.connect(root / 'lcm.db') as foreign:
                foreign.execute("INSERT INTO metadata(key, value) VALUES('foreign-writer', 'x')")
            foreign.close()
            fired.append(True)
        return result

    patch.setattr(MessageStore, '_sync_ingest_prelock', sync)
    return fired


def _stored(engine):
    return [(r['store_id'], r['session_id'], r['role'], r['content'], r.get('tool_call_id'))
            for r in engine._store.get_session_messages(SESSION)]


@pytest.mark.parametrize('failure', ['busy', 'nonbusy'])
@pytest.mark.parametrize('entry', ['ingest', 'session_end'])
def test_failed_restart_preserves_eager_rows_and_never_escapes(tmp_path, monkeypatch, entry, failure):
    """Review interleaving: foreign commit, implicit write in a scope, failed restart.

    The non-busy variant fails once before encountering the same held writer.
    Eager/main's pinned rows are the original pair, plus the final pair at end.
    """
    results = []
    for lazy in (False, True):
        root = tmp_path / str(lazy)
        engine, view = _warm(root, lazy)
        holders, calls, errors = [], [], []
        with monkeypatch.context() as patch:
            patch.setattr('hermes_lcm.engine._SESSION_END_BUSY_TIMEOUT_MS', 40)
            fired = _foreign_before_implicit_write(patch, engine, root) if lazy else []
            original = MessageStore._ingest_prelock_before_begin

            def seam(store):
                if store is engine._store:
                    if not holders:
                        holder = sqlite3.connect(root / 'lcm.db', isolation_level=None)
                        holder.execute('BEGIN IMMEDIATE')
                        holders.append(holder)
                    calls.append(True)
                    if failure == 'nonbusy' and len(calls) == 1:
                        raise sqlite3.OperationalError('injected disk I/O error during restart BEGIN')
                return original(store)

            patch.setattr(MessageStore, '_ingest_prelock_before_begin', seam)
            try:
                if entry == 'session_end':
                    engine.on_session_end(SESSION, view)
                else:
                    engine.ingest(view)
            except BaseException as error:
                errors.append(error)
            finally:
                for holder in holders:
                    holder.rollback()
                    holder.close()
        engine.shutdown()
        cold = _engine(root, prelock=lazy)
        try:
            results.append(_stored(cold))
        finally:
            cold.shutdown()
        assert calls and bool(fired) == lazy
        assert not errors, [(type(e).__name__, str(e)) for e in errors]
    expected = [view[i] for i in ([0, 1, 3, 4] if entry == 'session_end' else [0, 1])]
    assert results[0] == results[1]
    assert results[0] == [(i + 1, SESSION, m['role'], m['content'], None) for i, m in enumerate(expected)]


@pytest.mark.parametrize('failure', ['busy', 'nonbusy'])
def test_restart_retry_uses_current_acquisition_and_timeout(tmp_path, monkeypatch, failure):
    engine, view = _warm(tmp_path / 'lazy', True)
    root = tmp_path / 'lazy'
    holders, calls, errors = [], [], []
    attempts = []
    original_atomic = engine._store.ingest_atomic
    original_transaction = engine._ingest_messages_transaction

    def atomic(**kwargs):
        attempts.append((kwargs['prelock'], kwargs['admission_retry']))
        return original_atomic(**kwargs)

    def expose_begin_error(*args, **kwargs):
        # Isolate F2 from F1 on the red tree using the reviewer's F1 emulation.
        # The public-entry regression above independently requires the real fix.
        try:
            return original_transaction(*args, **kwargs)
        except _IngestReclassify as error:
            state = engine._store._ingest_prelock_state
            if state is not None and state.begin_failed is not None:
                raise state.begin_failed from error
            raise

    with monkeypatch.context() as patch:
        fired = _foreign_before_implicit_write(patch, engine, root)
        patch.setattr(engine._store, 'ingest_atomic', atomic)
        patch.setattr(engine, '_ingest_messages_transaction', expose_begin_error)
        original = MessageStore._ingest_prelock_before_begin

        def seam(store):
            if store is engine._store:
                calls.append((store._ingest_is_prelock(), store._conn.execute('PRAGMA busy_timeout').fetchone()[0]))
                if len(calls) == 1:
                    state = store._ingest_prelock_state
                    assert state.timing['first_acquired_at'] is not None  # an implicit write held the lock
                    assert not store._conn.in_transaction  # restart rolled it back before this BEGIN
                    if failure == 'nonbusy':
                        raise sqlite3.OperationalError('injected disk I/O error during restart BEGIN')
                    holder = sqlite3.connect(root / 'lcm.db', isolation_level=None)
                    holder.execute('BEGIN IMMEDIATE')
                    holders.append(holder)
            return original(store)

        patch.setattr(MessageStore, '_ingest_prelock_before_begin', seam)
        try:
            engine._ingest_messages(view)
        except sqlite3.OperationalError as error:
            errors.append(error)
        finally:
            for holder in holders:
                holder.rollback()
                holder.close()
    try:
        assert fired == [True]
        assert attempts == [(True, False), (False, True) if failure == 'busy' else (True, False)]
        # The successful non-busy retry uses implicit BEGIN; only the failed restart is explicit.
        assert calls == ([(True, 40), (False, 0)] if failure == 'busy' else [(True, 40)])
        assert bool(errors) == (failure == 'busy')
        assert engine._store.get_session_count(SESSION) == (2 if failure == 'busy' else 4)
    finally:
        engine.shutdown()
