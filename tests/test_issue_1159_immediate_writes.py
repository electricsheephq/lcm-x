"""#1159 DDL-rollback write waits and #1112 transient FTS schema checks."""

import sqlite3
import threading
import time

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore, build_message_fts_spec


@pytest.mark.parametrize("kind", ["store", "dag"])
def test_ddl_rollback_write_waits_for_other_writer(tmp_path, kind):
    path = tmp_path / "write-wait.db"
    owner = MessageStore(path) if kind == "store" else SummaryDAG(path)
    conn = owner.connection

    def write(content):
        if kind == "store":
            return owner._append_protected_batch(
                "session", [{"role": "user", "content": content}]
            )[0]
        return owner.add_node(SummaryNode(session_id="session", summary=content))

    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    timer = None
    trace = []
    try:
        warm_id = write("warm cached insert")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE rolled_back_schema(value)")
        conn.rollback()
        conn.set_trace_callback(trace.append)
        holder.execute("BEGIN IMMEDIATE")
        timer = threading.Timer(0.3, holder.commit)
        timer.start()
        start = time.monotonic()
        try:
            stored_id = write("waited insert")
        except sqlite3.OperationalError as exc:
            pytest.fail(
                f"write failed after {time.monotonic() - start:.6f}s: "
                f"code={exc.sqlite_errorcode} {exc}"
            )
        elapsed = time.monotonic() - start
        assert elapsed >= 0.2, (elapsed, trace)
        assert stored_id > warm_id
        stored = owner.get(stored_id) if kind == "store" else owner.get_node(stored_id)
        assert (stored["content"] if kind == "store" else stored.summary) == "waited insert"
        assert [sql for sql in trace if sql.startswith("BEGIN")] == ["BEGIN IMMEDIATE"]
        assert not conn.in_transaction
    finally:
        if timer is not None:
            timer.join()
        holder.rollback()
        holder.close()
        owner.close()


class _ConnectionProxy:
    def __init__(self, conn):
        self.conn = conn

    def __getattr__(self, name):
        return getattr(self.conn, name)


@pytest.mark.parametrize("schema_kind", ["code", "vtable_text"])
@pytest.mark.parametrize("failures", [1, 2])
def test_structural_schema_error_rechecks_once(tmp_path, monkeypatch, schema_kind, failures):
    store = MessageStore(tmp_path / "schema-recheck.db")
    proxy = _ConnectionProxy(store.connection)
    attempts = 0

    def execute(sql, *args):
        nonlocal attempts
        if sql.startswith("PRAGMA table_info"):
            attempts += 1
            if attempts <= failures:
                exc = sqlite3.DatabaseError(
                    "database schema has changed" if schema_kind == "code"
                    else "vtable constructor failed: messages_fts"
                )
                if schema_kind == "code":
                    exc.sqlite_errorcode = sqlite3.SQLITE_SCHEMA
                raise exc
        return store.connection.execute(sql, *args)

    monkeypatch.setattr(proxy, "execute", execute)
    trace = []
    store.connection.set_trace_callback(trace.append)
    try:
        assert db_bootstrap._fts_needs_rebuild_structural(
            proxy, build_message_fts_spec()
        ) is (failures == 2)
        assert attempts == 2
        assert not store.connection.in_transaction
        assert not any(sql.startswith("BEGIN") for sql in trace)
    finally:
        store.close()


def test_structural_lock_error_precedes_vtable_retry(tmp_path, monkeypatch):
    store = MessageStore(tmp_path / "lock-precedence.db")
    proxy = _ConnectionProxy(store.connection)
    attempts = 0

    def execute(sql, *args):
        nonlocal attempts
        if sql.startswith("PRAGMA table_info"):
            attempts += 1
            exc = sqlite3.OperationalError("vtable constructor failed: messages_fts")
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exc
        return store.connection.execute(sql, *args)

    monkeypatch.setattr(proxy, "execute", execute)
    try:
        with pytest.raises(sqlite3.OperationalError, match="vtable constructor failed"):
            db_bootstrap._fts_needs_rebuild_structural(proxy, build_message_fts_spec())
        assert attempts == 1
    finally:
        store.close()
