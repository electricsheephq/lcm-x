"""Transaction completion callbacks are generic, isolated from canonical outcomes."""
from contextlib import closing
import sqlite3
import threading

import pytest

from hermes_lcm.sqlite_session import TransactionOwner


@pytest.mark.parametrize('rollback', [False, True])
@pytest.mark.parametrize('raising', [False, True])
@pytest.mark.parametrize('outer_lock', [False, True])
def test_callbacks_run_once_after_outcome_and_errors_are_counted(rollback, raising, outer_lock, caplog):
    lock = threading.RLock()
    owner = TransactionOwner(lock)
    seen = []
    with closing(sqlite3.connect(':memory:')) as conn:
        conn.execute('CREATE TABLE example(value)')
        def callback():
            seen.append((conn.in_transaction, owner.owns(conn), conn.execute('SELECT count(*) FROM example').fetchone()[0]))
            if raising:
                raise RuntimeError('callback failed')
        from contextlib import nullcontext
        with lock if outer_lock else nullcontext():
            try:
                with owner.atomic(conn):
                    owner.on_commit.append(callback if not rollback else lambda: pytest.fail('wrong commit callback'))
                    owner.on_rollback.append(callback if rollback else lambda: pytest.fail('wrong rollback callback'))
                    with owner.atomic(conn):
                        conn.execute('INSERT INTO example VALUES (1)')
                    if rollback:
                        raise ValueError('canonical failure')
            except ValueError as exc:
                assert rollback and str(exc) == 'canonical failure'
        assert seen == [(False, False, 0 if rollback else 1)]
        assert owner.callback_failures == int(raising)
        assert owner.on_commit == owner.on_rollback == []
        with owner.atomic(conn):
            pass
        assert len(seen) == 1
        if raising:
            assert 'SQLite completion callback failed' in caplog.text


def test_failed_rollback_does_not_run_callbacks_for_an_unknown_outcome():
    class FailedRollback(sqlite3.Connection):
        fail_rollback = True

        def rollback(self):
            if self.fail_rollback:
                raise sqlite3.OperationalError('rollback failed')
            super().rollback()
    with closing(sqlite3.connect(':memory:', factory=FailedRollback)) as conn:
        conn.execute('CREATE TABLE example(value)')
        owner = TransactionOwner(threading.RLock())
        seen = []
        with pytest.raises(ValueError, match='original failure'):
            with owner.atomic(conn):
                conn.execute('INSERT INTO example VALUES (1)')
                owner.on_rollback.append(lambda: seen.append(conn.in_transaction))
                raise ValueError('original failure')
        assert conn.in_transaction
        assert seen == [], 'an uncertain rollback must retain purge intents'
        assert owner.rollback_failures == 1
        assert owner.on_commit == owner.on_rollback == []
        conn.fail_rollback = False
        conn.rollback()
