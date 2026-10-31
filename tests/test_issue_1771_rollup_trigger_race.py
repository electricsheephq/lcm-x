from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from hermes_lcm import db_bootstrap


SUMMARY_SCHEMA = """
CREATE TABLE summary_nodes(
    node_id INTEGER, session_id TEXT, depth INTEGER, summary TEXT,
    token_count INTEGER, source_token_count INTEGER, source_ids TEXT,
    source_type TEXT, created_at REAL, earliest_at REAL, latest_at REAL,
    expand_hint TEXT
)
"""
TRIGGER_QUERY = (
    "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
    "AND name LIKE 'lcm_rollup_node_%' ORDER BY name"
)


def _prepared(conn):
    conn.execute(SUMMARY_SCHEMA)
    db_bootstrap.ensure_temporal_rollup_tables(conn)
    return conn.execute(TRIGGER_QUERY).fetchall()


@pytest.mark.parametrize("initial", ["fresh", "mismatch"])
@pytest.mark.parametrize("interleave", [False, True], ids=["natural", "forced"])
def test_two_connection_initialization_200_times(tmp_path, initial, interleave):
    with sqlite3.connect(":memory:") as reference:
        expected = _prepared(reference)
    errors = []
    wrong_sql = []
    for iteration in range(200):
        path = tmp_path / f"{initial}-{iteration}.db"
        with sqlite3.connect(path) as conn:
            conn.execute(SUMMARY_SCHEMA)
            if initial == "mismatch":
                db_bootstrap.ensure_temporal_rollup_tables(conn)
                conn.execute("DROP TRIGGER lcm_rollup_node_update")
                conn.execute(
                    "CREATE TRIGGER lcm_rollup_node_update "
                    "AFTER UPDATE ON summary_nodes BEGIN SELECT 1; END"
                )
        barrier = threading.Barrier(2, timeout=10)
        read_barrier = threading.Barrier(2, timeout=10)

        class InterleavedConnection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                result = super().execute(sql, parameters)
                if (
                    interleave and not self.in_transaction
                    and sql == "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?"
                    and parameters == ("lcm_rollup_node_update",)
                ):
                    # Force both unprotected readers to consume the same stale
                    # definition; a write transaction prevents this interleaving.
                    row = result.fetchone()
                    if row is None or row[0] != dict(expected)["lcm_rollup_node_update"]:
                        read_barrier.wait()
                    return SimpleNamespace(fetchone=lambda: row)
                return result

        def initialize():
            with sqlite3.connect(path, timeout=10, factory=InterleavedConnection) as conn:
                barrier.wait()
                db_bootstrap.ensure_temporal_rollup_tables(conn)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(initialize) for _ in range(2)]
            errors.extend(
                (iteration, str(error))
                for future in futures
                if (error := future.exception(timeout=20)) is not None
            )
        with sqlite3.connect(path) as conn:
            if conn.execute(TRIGGER_QUERY).fetchall() != expected:
                wrong_sql.append(iteration)
    assert errors == [], errors
    assert wrong_sql == [], wrong_sql


@pytest.mark.parametrize("finish", ["commit", "rollback"])
@pytest.mark.parametrize("begin", ["BEGIN", "BEGIN IMMEDIATE"])
def test_trigger_reconcile_preserves_caller_transaction(begin, finish):
    conn = sqlite3.connect(":memory:")
    try:
        expected = _prepared(conn)
        conn.execute("DROP TRIGGER lcm_rollup_node_update")
        old_sql = (
            "CREATE TRIGGER lcm_rollup_node_update "
            "AFTER UPDATE ON summary_nodes BEGIN SELECT 1; END"
        )
        conn.execute(old_sql)
        conn.execute("CREATE TABLE caller_work(value)")
        conn.commit()
        conn.execute(begin)
        conn.execute("INSERT INTO caller_work VALUES ('uncommitted')")
        trace = []
        conn.set_trace_callback(trace.append)
        db_bootstrap.ensure_temporal_rollup_invalidation_triggers(conn)
        conn.set_trace_callback(None)
        assert conn.in_transaction
        assert not any(sql.startswith(("BEGIN", "COMMIT")) for sql in trace)
        assert conn.execute(TRIGGER_QUERY).fetchall() == expected
        getattr(conn, finish)()
        assert not conn.in_transaction
        assert conn.execute("SELECT * FROM caller_work").fetchall() == (
            [("uncommitted",)] if finish == "commit" else []
        )
        if finish == "rollback":
            assert conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='lcm_rollup_node_update'"
            ).fetchone() == (old_sql,)
    finally:
        conn.close()


@pytest.mark.parametrize("outer", [False, True])
def test_failed_reconcile_rolls_back_only_its_own_changes(outer):
    class FailingConnection(sqlite3.Connection):
        fail = False

        def execute(self, sql, parameters=()):
            if self.fail and "CREATE TRIGGER lcm_rollup_node_update" in sql:
                raise sqlite3.OperationalError("injected trigger create failure")
            return super().execute(sql, parameters)

    conn = sqlite3.connect(":memory:", factory=FailingConnection)
    try:
        _prepared(conn)
        conn.execute("DROP TRIGGER lcm_rollup_node_update")
        conn.execute(
            "CREATE TRIGGER lcm_rollup_node_update "
            "AFTER UPDATE ON summary_nodes BEGIN SELECT 1; END"
        )
        old = conn.execute(TRIGGER_QUERY).fetchall()
        conn.execute("CREATE TABLE caller_work(value)")
        conn.commit()
        if outer:
            conn.execute("INSERT INTO caller_work VALUES ('uncommitted')")
        conn.fail = True
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            db_bootstrap.ensure_temporal_rollup_invalidation_triggers(conn)
        assert conn.in_transaction == outer
        assert conn.execute(TRIGGER_QUERY).fetchall() == old
        assert conn.execute("SELECT * FROM caller_work").fetchall() == (
            [("uncommitted",)] if outer else []
        )
    finally:
        conn.close()


def test_verifier_temp_database_owns_and_finishes_transaction(monkeypatch):
    original = db_bootstrap.ensure_temporal_rollup_invalidation_triggers
    observed = []

    def observe(conn):
        before = conn.in_transaction
        trace = []
        conn.set_trace_callback(trace.append)
        original(conn)
        conn.set_trace_callback(None)
        observed.append((before, conn.in_transaction, trace))

    conn = sqlite3.connect(":memory:")
    try:
        _prepared(conn)
        monkeypatch.setattr(
            db_bootstrap, "ensure_temporal_rollup_invalidation_triggers", observe
        )
        assert db_bootstrap.verify_temporal_rollup_schema(conn) == []
        assert len(observed) == 1
        before, after, trace = observed[0]
        assert (before, after) == (False, False)
        assert [sql for sql in trace if sql.startswith(("BEGIN", "COMMIT"))] == [
            "BEGIN IMMEDIATE", "COMMIT",
        ]
        assert not conn.in_transaction
    finally:
        conn.close()


@pytest.mark.parametrize("repair_interval", [False, True])
def test_table_initializer_passes_through_transaction_state(monkeypatch, repair_interval):
    conn = sqlite3.connect(":memory:")
    try:
        _prepared(conn)
        if repair_interval:
            conn.execute(
                "INSERT INTO lcm_rollup_invalidations"
                "(scope, covered_start, covered_end, operation) "
                "VALUES ('session', 2, 1, 'insert')"
            )
            conn.commit()
        original = db_bootstrap.ensure_temporal_rollup_invalidation_triggers
        observed = []

        def observe(connection):
            before = connection.in_transaction
            original(connection)
            observed.append((before, connection.in_transaction))

        monkeypatch.setattr(
            db_bootstrap, "ensure_temporal_rollup_invalidation_triggers", observe
        )
        db_bootstrap.ensure_temporal_rollup_tables(conn)
        assert observed == [(repair_interval, repair_interval)]
        if repair_interval:
            assert conn.execute(
                "SELECT covered_start, covered_end FROM lcm_rollup_invalidations"
            ).fetchall() == [(1, 2)]
            conn.rollback()
            assert conn.execute(
                "SELECT covered_start, covered_end FROM lcm_rollup_invalidations"
            ).fetchall() == [(2, 1)]
    finally:
        conn.close()
