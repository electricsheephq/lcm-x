from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from hermes_lcm import db_bootstrap


@pytest.mark.parametrize("name", [
    "idx_lcm_rollups_ready_period",
    "idx_lcm_rollups_failed_aggregate",
])
@pytest.mark.parametrize("initial", ["absent", "changed", "changed_after_create"])
def test_concurrent_rollup_bootstrap(tmp_path, name, initial):
    path = tmp_path / "rollups.db"
    with sqlite3.connect(path) as conn:
        # Obtain the exact required SQL, then remove the feature tables so the
        # absent case really bootstraps one fresh store from two connections.
        db_bootstrap.ensure_temporal_rollup_tables(conn)
        expected = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
        ).fetchone()[0]
        if initial == "absent":
            for table in db_bootstrap.REQUIRED_TEMPORAL_ROLLUP_TABLES:
                conn.execute(f"DROP TABLE {table}")
        else:
            conn.execute(f"DROP INDEX {name}")
            conn.execute(f"CREATE INDEX {name} ON lcm_rollups(scope)")

    read_barrier = threading.Barrier(2, timeout=10)
    create_barrier = threading.Barrier(2, timeout=10)
    first_created = threading.Event()

    def bootstrap(worker):
        class InterleavedConnection(sqlite3.Connection):
            observed = False

            def execute(self, sql, parameters=()):
                if (
                    not self.observed
                    and sql == "SELECT sql FROM sqlite_master WHERE type='index' AND name=?"
                    and parameters == (name,)
                ):
                    self.observed = True
                    # Fully consume the read before waiting; neither connection
                    # holds a cursor read lock while its peer executes DDL.
                    row = super().execute(sql, parameters).fetchone()
                    read_barrier.wait()
                    if initial == "changed_after_create" and worker == 1:
                        assert first_created.wait(10)
                    return SimpleNamespace(fetchone=lambda: row)
                if sql.startswith(f"CREATE INDEX {name} "):
                    if initial != "changed_after_create":
                        # Both stale DROPs must finish before either CREATE.
                        create_barrier.wait()
                    result = super().execute(sql, parameters)
                    first_created.set()
                    return result
                return super().execute(sql, parameters)

        conn = sqlite3.connect(path, factory=InterleavedConnection)
        try:
            db_bootstrap.ensure_temporal_rollup_tables(conn)
            conn.commit()
        except BaseException:
            read_barrier.abort()
            create_barrier.abort()
            first_created.set()
            raise
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(bootstrap, worker) for worker in range(2)]
        # Collect both errors so an aborted peer cannot hide the SQLite error.
        errors = [future.exception(timeout=15) for future in futures]
    assert errors == [None, None], [str(error) for error in errors]
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name=?", (name,)
        ).fetchone()[0] == expected
        assert db_bootstrap.verify_temporal_rollup_schema(conn) == []


@pytest.mark.parametrize("competing_sql", [
    "CREATE INDEX idx_lcm_rollups_ready_period ON lcm_rollups(scope)",
    "CREATE INDEX idx_lcm_rollups_ready_period "
    "ON lcm_rollups(scope, period_kind, period_start DESC) WHERE status = 'ready'",
])
def test_concurrent_creator_requires_matching_definition(tmp_path, competing_sql):
    class CompetingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("CREATE INDEX idx_lcm_rollups_ready_period "):
                super().execute(competing_sql)
            return super().execute(sql, parameters)

    conn = sqlite3.connect(tmp_path / "rollups.db", factory=CompetingConnection)
    try:
        if "DESC" in competing_sql:
            db_bootstrap.ensure_temporal_rollup_tables(conn)
        else:
            with pytest.raises(sqlite3.OperationalError, match="already exists"):
                db_bootstrap.ensure_temporal_rollup_tables(conn)
    finally:
        conn.close()


def test_unrelated_create_error_is_not_swallowed(tmp_path):
    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith("CREATE INDEX idx_lcm_rollups_ready_period "):
                super().execute(sql)
                raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, parameters)

    conn = sqlite3.connect(tmp_path / "rollups.db", factory=FailingConnection)
    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            db_bootstrap.ensure_temporal_rollup_tables(conn)
    finally:
        conn.close()
