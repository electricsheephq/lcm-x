"""#897: uncommitted shadow DDL must not poison the schema readiness cache."""

import sqlite3

import pytest

from hermes_lcm.store import MessageStore


class _FailFirstCommit:
    """Delegate to a real connection, failing its first schema-creating commit."""

    def __init__(self, real):
        self._real = real
        self.commits = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def commit(self):
        self.commits += 1
        if self.commits == 1:
            raise sqlite3.OperationalError("injected schema commit failure")
        return self._real.commit()


def test_uncommitted_table_read_does_not_prevent_recovery_after_commit_failure(tmp_path, monkeypatch):
    store = MessageStore(tmp_path / "lcm.db")
    real = store._conn
    faulty = _FailFirstCommit(real)
    write = store._host_uid_write
    observed_readiness = []

    def read_inside_creating_write(create):
        def create_then_read(conn):
            create(conn)
            assert conn.in_transaction
            # Same-thread read while _host_uid_write holds the store's RLock:
            # the table is visible here, but its existence is not durable yet.
            assert store._host_uid_table_exists()
            observed_readiness.append(getattr(store, "_host_uid_schema_ready", False))

        return write(create_then_read)

    try:
        assert not store._host_uid_table_exists()
        with monkeypatch.context() as patch:
            patch.setattr(store, "_conn", faulty)
            patch.setattr(store, "_host_uid_write", read_inside_creating_write)
            with pytest.raises(sqlite3.OperationalError, match="injected schema commit failure"):
                store.add_host_uid_bindings("synthetic-lineage", [(1, "synthetic-uid", "canonical", "stored_new")])

        assert faulty.commits == 1
        assert observed_readiness == [False]
        assert not real.in_transaction
        assert not getattr(store, "_host_uid_schema_ready", False)
        assert real.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='host_uid_bindings'").fetchone() is None
        assert store.host_uid_gate() == []
        assert store.count_host_uid_bindings() is None

        # Recovery uses the very same store; no process restart or manual flag reset.
        store.add_host_uid_bindings("synthetic-lineage", [(1, "synthetic-uid", "canonical", "stored_new")])
        assert store._host_uid_schema_ready
        assert not real.in_transaction
        assert store.host_uid_bindings_for("synthetic-lineage", ["synthetic-uid"]) == {
            "synthetic-uid": [(1, "canonical")],
        }
        store.record_host_uid_checks("synthetic-lineage", [("synthetic-uid", 1, True)])
        assert store.host_uid_gate() == [(1, 1)]
        assert store.count_host_uid_bindings() == 1
    finally:
        store.close()


def test_existing_committed_table_read_sets_ready_flag(tmp_path):
    path = tmp_path / "lcm.db"
    store = MessageStore(path)
    try:
        store.add_host_uid_bindings("synthetic-lineage", [(1, "synthetic-uid", "canonical", "stored_new")])
    finally:
        store.close()

    reopened = MessageStore(path)
    try:
        assert not getattr(reopened, "_host_uid_schema_ready", False)
        assert reopened._host_uid_table_exists()
        assert reopened._host_uid_schema_ready
        assert reopened.count_host_uid_bindings() == 1
    finally:
        reopened.close()
