"""The additive access_scope columns, and what they must NOT cost.

This is the slice that touches every existing database, so the tests here are
mostly negative: a stock install gains nullable columns and nothing else, an
upgrade pays for them once, and a store that never enabled Teams does not pay
an eleven-table scan on every bind for the privilege.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

import pytest

from hermes_lcm import db_bootstrap, scope_storage
from hermes_lcm.access_policy import FailClosedPolicy, TrustedOwnerPolicy, policy_for_engine
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.scope_storage import (
    SCOPE_BEARING_TABLES,
    SCOPE_DOCTOR_CHECK,
    TEAMS_NEVER_ENABLED_METADATA_KEY,
    bind_startup_teams_state,
    clear_never_enabled_cache,
    doctor_scope_check,
    persist_teams_enabled,
    read_never_enabled_cache,
    resolve_startup_teams_state,
    teams_enabled,
    verify_scope_storage,
)
from hermes_lcm.store import MessageStore

#: Everything a fresh install materializes. The opt-in families (rollups,
#: embeddings, chunks, assertions, query views, trajectories, and the Teams
#: catalog) are deliberately absent: this slice adds COLUMNS, and a new table
#: on a stock install would make the database unreadable by a base build.
_STOCK_TABLES = frozenset({
    "messages",
    "metadata",
    "summary_nodes",
    "lcm_lifecycle_state",
    "lcm_migration_state",
    "messages_fts",
    "nodes_fts",
    "sqlite_sequence",
})


class _Engine:
    """Stands in for a freshly bound engine with no context accessor wired."""


def _open_stock(db_path) -> tuple[MessageStore, SummaryDAG]:
    """Open the two components that own the core tables, in production order."""

    return MessageStore(str(db_path)), SummaryDAG(db_path)


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        # fts5 shadow tables belong to their virtual table, not to this contract.
        if not str(row[0]).endswith(
            ("_data", "_idx", "_docsize", "_config", "_content")
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]


def _scope_marker_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name = ?",
        (scope_storage.SCOPE_MIGRATION_STEP,),
    ).fetchone() is not None


# -- Inertness -------------------------------------------------------------


def test_a_fresh_stock_database_gains_columns_and_nothing_else(tmp_path):
    """No tables, no Teams markers -- the ratified bar for a dormant feature."""

    db_path = tmp_path / "stock.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        assert _tables(conn) == _STOCK_TABLES
        for table in ("messages", "summary_nodes"):
            columns = _columns(conn, table)
            assert columns[-1] == "access_scope", table
            # Nullable, no default, not part of any key: an ALTER on a
            # customer-sized table has to be a metadata-only operation.
            info = {
                str(row[1]): row
                for row in conn.execute(f"PRAGMA table_info({table})")
            }["access_scope"]
            assert (info[2], info[3], info[4], info[5]) == ("TEXT", 0, None, 0)

        metadata_keys = {
            str(row[0]) for row in conn.execute("SELECT key FROM metadata")
        }
        assert not [key for key in metadata_keys if "teams" in key.lower()]
        # The only new migration step is the one that names the columns, and it
        # is not spelled as a Teams marker either.
        steps = {
            str(row[0])
            for row in conn.execute("SELECT step_name FROM lcm_migration_state")
        }
        assert scope_storage.SCOPE_MIGRATION_STEP in steps
        assert not [step for step in steps if "teams" in step.lower()]
    finally:
        dag.close()
        store.close()


def test_the_migration_step_is_defined_once(tmp_path):
    """db_bootstrap imports the name; it does not spell a second one."""

    assert not hasattr(db_bootstrap, "SCOPE_MIGRATION_STEP")
    assert scope_storage.SCOPE_MIGRATION_STEP == "scope_v1"


def test_reopening_a_stock_database_changes_nothing(tmp_path):
    """The migration walk is idempotent: same schema, same markers, twice."""

    db_path = tmp_path / "idempotent.db"
    store, dag = _open_stock(db_path)
    conn = store.connection
    first = sorted(
        (str(row[0]), str(row[1] or ""))
        for row in conn.execute("SELECT name, sql FROM sqlite_master")
    )
    dag.close()
    store.close()

    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        second = sorted(
            (str(row[0]), str(row[1] or ""))
            for row in conn.execute("SELECT name, sql FROM sqlite_master")
        )
        assert second == first
    finally:
        dag.close()
        store.close()


def test_summary_dag_first_defers_the_scope_marker_until_messages_exist(tmp_path):
    """A partial core initializer must not make the later core sweep a no-op."""

    db_path = tmp_path / "dag-first.db"
    dag = SummaryDAG(db_path)
    store = None
    try:
        assert not _scope_marker_exists(dag.connection)
        store = MessageStore(db_path)
        conn = store.connection
        assert "access_scope" in _columns(conn, "messages")
        assert _scope_marker_exists(conn)
        report = verify_scope_storage(conn, teams_enabled=False)
        assert "missing access_scope column" not in str(report)
    finally:
        if store is not None:
            store.close()
        dag.close()


def test_lifecycle_first_defers_the_scope_marker_until_both_core_tables_exist(tmp_path):
    """Lifecycle bootstrap may precede both core scope-bearing stores."""

    db_path = tmp_path / "lifecycle-first.db"
    lifecycle = LifecycleStateStore(db_path)
    store = None
    dag = None
    try:
        assert not _scope_marker_exists(lifecycle.connection)
        store = MessageStore(db_path)
        assert not _scope_marker_exists(store.connection)
        dag = SummaryDAG(db_path)
        conn = store.connection
        assert "access_scope" in _columns(conn, "messages")
        assert "access_scope" in _columns(conn, "summary_nodes")
        assert _scope_marker_exists(conn)
        report = verify_scope_storage(conn, teams_enabled=False)
        assert "missing access_scope column" not in str(report)
    finally:
        if dag is not None:
            dag.close()
        if store is not None:
            store.close()
        lifecycle.close()


def test_a_pre_scope_database_is_migrated_once(tmp_path):
    """The upgrade path: columns ALTERed in, then a second open is a no-op."""

    db_path = tmp_path / "legacy.db"
    store, dag = _open_stock(db_path)
    store.append("legacy-session", {"role": "user", "content": "hello"})
    dag.add_node(SummaryNode(session_id="legacy-session", summary="hi", created_at=1.0))
    conn = store.connection
    # Rewind to the pre-slice shape: no columns, no marker.
    for table in ("messages", "summary_nodes"):
        conn.execute(f"ALTER TABLE {table} DROP COLUMN access_scope")
    conn.execute(
        "DELETE FROM lcm_migration_state WHERE step_name = ?",
        (scope_storage.SCOPE_MIGRATION_STEP,),
    )
    conn.commit()
    dag.close()
    store.close()

    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        assert _columns(conn, "messages")[-1] == "access_scope"
        assert _columns(conn, "summary_nodes")[-1] == "access_scope"
        # Existing rows survive, unstamped.
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE access_scope IS NULL"
        ).fetchone()[0] == 1
        altered = sorted(
            (str(row[0]), str(row[1] or ""))
            for row in conn.execute("SELECT name, sql FROM sqlite_master")
        )
    finally:
        dag.close()
        store.close()

    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        assert sorted(
            (str(row[0]), str(row[1] or ""))
            for row in conn.execute("SELECT name, sql FROM sqlite_master")
        ) == altered
    finally:
        dag.close()
        store.close()


def test_a_created_rollup_table_matches_a_migrated_one(tmp_path):
    """One logical schema, one column order.

    ``ensure_temporal_rollup_tables`` creates the column and ALTER adds it to an
    older table; SQLite appends on ALTER, so a mid-table position in the CREATE
    would give the same feature two different PRAGMA shapes.
    """

    created = sqlite3.connect(":memory:")
    migrated = sqlite3.connect(":memory:")
    try:
        db_bootstrap.ensure_temporal_rollup_tables(created)
        db_bootstrap.ensure_temporal_rollup_tables(migrated)
        for table in ("lcm_rollups", "lcm_rollup_invalidations", "lcm_rollup_state"):
            migrated.execute(f"ALTER TABLE {table} DROP COLUMN access_scope")
        db_bootstrap.ensure_temporal_rollup_tables(migrated)
        for table in ("lcm_rollups", "lcm_rollup_invalidations", "lcm_rollup_state"):
            assert _columns(created, table) == _columns(migrated, table), table
            assert _columns(created, table)[-1] == "access_scope", table
    finally:
        created.close()
        migrated.close()


def test_rollup_invalidation_triggers_install_on_a_stock_database(tmp_path):
    """The triggers resolve ``new.access_scope`` when they are CREATED.

    A deferred column does not make them inert -- it makes them fail to install,
    which takes the whole rollup feature down on a stock store.
    """

    db_path = tmp_path / "rollups.db"
    store, dag = _open_stock(db_path)
    rollups = RollupStore(db_path)
    try:
        node_id = dag.add_node(
            SummaryNode(session_id="s1", summary="hello", created_at=1.0)
        )
        conn = rollups.connection
        row = conn.execute(
            "SELECT node_id, access_scope FROM lcm_rollup_invalidations "
            "WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        # The invalidation is recorded exactly as before, carrying the NULL a
        # never-enabled store has for every row.
        assert row is not None
        assert row["access_scope"] is None
    finally:
        rollups.close()
        dag.close()
        store.close()


# -- F2: the startup cost --------------------------------------------------


def _count_stamp_probes(conn: sqlite3.Connection, work) -> int:
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        work()
    finally:
        conn.set_trace_callback(None)
    return len([sql for sql in seen if "access_scope IS NOT NULL" in sql])


def test_a_never_enabled_store_probes_once_and_never_again(tmp_path):
    """The regression this fix exists for.

    Without the cached answer every bind scans up to eleven tables looking for
    a stamp that a never-enabled store will never have -- unindexed, and on the
    two tables that hold the whole corpus.
    """

    db_path = tmp_path / "never-enabled.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        first = _count_stamp_probes(conn, lambda: bind_startup_teams_state(conn))
        assert first > 0
        assert read_never_enabled_cache(conn) is True

        second = _count_stamp_probes(conn, lambda: bind_startup_teams_state(conn))
        assert second == 0
        assert bind_startup_teams_state(conn) == (False, "never-enabled")
    finally:
        dag.close()
        store.close()


def test_engine_bind_publishes_cache_and_fails_closed_after_a_stamp(tmp_path):
    """The production bind owns both the cache and the engine policy flag."""

    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    db_path = tmp_path / "engine-bind.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    engine._store.append("s1", {"role": "user", "content": "hello"})
    engine.shutdown()

    reopened = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    try:
        assert teams_enabled(reopened) is False
        assert isinstance(policy_for_engine(reopened), TrustedOwnerPolicy)
        assert reopened._store.connection.execute(
            "SELECT COUNT(*) FROM metadata WHERE key = ?",
            (TEAMS_NEVER_ENABLED_METADATA_KEY,),
        ).fetchone()[0] == 1
    finally:
        reopened.shutdown()

    with sqlite3.connect(db_path) as writer:
        clear_never_enabled_cache(writer)
        writer.execute("UPDATE messages SET access_scope = 'principal-a'")
        writer.commit()

    stamped = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    try:
        assert teams_enabled(stamped) is True
        assert isinstance(policy_for_engine(stamped), FailClosedPolicy)
        assert read_never_enabled_cache(stamped._store.connection) is False
    finally:
        stamped.shutdown()


def test_read_only_never_enabled_cache_publish_degrades_startup(tmp_path, caplog):
    """A read-only database cannot make optional cache publication fatal."""

    if os.geteuid() == 0:
        pytest.skip("chmod-based read-only regression is not enforceable as root")

    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    db_path = tmp_path / "read-only-never-enabled.db"
    engine = LCMEngine(config=LCMConfig(database_path=str(db_path)))
    try:
        clear_never_enabled_cache(engine._store.connection)
        assert read_never_enabled_cache(engine._store.connection) is False
    finally:
        engine.shutdown()

    original_mode = db_path.stat().st_mode
    db_path.chmod(original_mode & ~0o222)
    try:
        with caplog.at_level(logging.DEBUG, logger=scope_storage.logger.name):
            reopened = LCMEngine(config=LCMConfig(database_path=str(db_path)))
        try:
            assert teams_enabled(reopened) is False
            assert bind_startup_teams_state(reopened._store.connection) == (
                False,
                "never-enabled",
            )
            assert reopened._store.connection.execute(
                "SELECT COUNT(*) FROM metadata WHERE key = ?",
                (TEAMS_NEVER_ENABLED_METADATA_KEY,),
            ).fetchone()[0] == 0
        finally:
            reopened.shutdown()

        cache_skips = [
            record
            for record in caplog.records
            if record.levelno == logging.DEBUG
            and "Skipping Teams never-enabled cache" in record.getMessage()
        ]
        assert cache_skips
        assert any("readonly" in record.getMessage().lower() for record in cache_skips)
    finally:
        db_path.chmod(original_mode)


def test_cache_publication_and_stamp_commit_are_serialized(tmp_path):
    """Neither ordering can leave a published cache over a committed stamp."""

    db_path = tmp_path / "interleaving.db"
    store, dag = _open_stock(db_path)
    store.append("s1", {"role": "user", "content": "hello"})
    store.close()
    dag.close()

    writer = sqlite3.connect(db_path, timeout=0.01)
    binder = sqlite3.connect(db_path, timeout=0.01)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE messages SET access_scope = 'principal-a'")
        assert bind_startup_teams_state(binder) == (False, "never-enabled")
        assert read_never_enabled_cache(binder) is False

        writer.commit()
        assert bind_startup_teams_state(binder) == (
            True,
            "stamped-without-marker",
        )

        writer.execute("UPDATE messages SET access_scope = NULL")
        writer.commit()
        assert bind_startup_teams_state(binder) == (False, "never-enabled")
        assert read_never_enabled_cache(binder) is True

        assert scope_storage._backfill_session_table(
            writer,
            "messages",
            lambda session_id: "principal-a",
            batch_size=1,
        ) == 1
        assert read_never_enabled_cache(binder) is False
    finally:
        binder.close()
        writer.close()


def test_profile_rebind_refreshes_the_teams_flag_in_both_directions(tmp_path):
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine

    never_home = tmp_path / "never"
    enabled_home = tmp_path / "enabled"
    enabled_store, enabled_dag = _open_stock(enabled_home / "lcm.db")
    persist_teams_enabled(enabled_store.connection, True)
    enabled_dag.close()
    enabled_store.close()

    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=str(never_home))
    try:
        assert teams_enabled(engine) is False
        engine.on_session_start(
            "enabled-session",
            hermes_home=str(enabled_home),
            platform="cli",
            context_length=200_000,
        )
        assert teams_enabled(engine) is True

        engine.on_session_start(
            "never-session",
            hermes_home=str(never_home),
            platform="cli",
            context_length=200_000,
        )
        assert teams_enabled(engine) is False
    finally:
        engine.shutdown()


def test_the_cached_answer_is_not_an_operator_decision(tmp_path):
    """It must not be readable as "an operator disabled Teams".

    A disable is authority: it stops `resolve_startup_teams_state` from looking
    at the store at all. An observation is not, and writing one in the other's
    spelling is how two different absences come to read alike.
    """

    db_path = tmp_path / "cache-key.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        bind_startup_teams_state(conn)
        assert (
            TEAMS_NEVER_ENABLED_METADATA_KEY
            != scope_storage.TEAMS_ENABLED_METADATA_KEY
        )
        assert scope_storage.read_persisted_teams_marker(conn) == "absent"
        assert scope_storage.read_persisted_teams_enabled(conn) is None
    finally:
        dag.close()
        store.close()


def test_an_enable_clears_the_cached_answer_before_stamping(tmp_path):
    """Otherwise an enable that dies partway reads as a store with no stamps.

    That is the permissive-policy-on-scoped-data failure the startup resolver
    exists to prevent, reintroduced by an optimization.
    """

    db_path = tmp_path / "aborted-enable.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        store.append("s1", {"role": "user", "content": "hello"})
        bind_startup_teams_state(conn)
        assert read_never_enabled_cache(conn) is True

        with pytest.raises(scope_storage.ScopeBackfillIncompleteError):
            # One unresolvable owner is all it takes to abort partway.
            scope_storage.backfill_scopes(conn, lambda session_id: None)

        assert read_never_enabled_cache(conn) is False
        assert resolve_startup_teams_state(conn) == (False, "never-enabled")
    finally:
        dag.close()
        store.close()


def test_recording_a_decision_drops_the_cached_answer(tmp_path):
    """A decision outranks the cache, and a REVOKED decision must not fall back
    onto a cached observation older than the stamps it is about."""

    db_path = tmp_path / "decided.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        bind_startup_teams_state(conn)
        assert read_never_enabled_cache(conn) is True
        persist_teams_enabled(conn, False)
        assert read_never_enabled_cache(conn) is False
    finally:
        dag.close()
        store.close()


# -- F3: the stray stamp ---------------------------------------------------


def _stamp_one_row(store: MessageStore) -> None:
    """What an importer does: a row that already carries an owner."""

    conn = store.connection
    store.append("imported-session", {"role": "user", "content": "hello"})
    conn.execute("UPDATE messages SET access_scope = 'principal-a'")
    conn.commit()


def test_a_stray_stamp_still_fails_closed(tmp_path):
    """The negative control. This behavior is deliberate and stays.

    A stamp with no recorded decision is indistinguishable from an enable that
    died partway, and the direction to fail in when the store might be scoped
    is closed -- even though the cost is that the store refuses all work.
    """

    db_path = tmp_path / "stray.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        _stamp_one_row(store)
        enabled, reason = resolve_startup_teams_state(conn)
        assert (enabled, reason) == (True, "stamped-without-marker")

        engine = _Engine()
        if enabled:
            scope_storage.mark_teams_enabled(engine)
        assert isinstance(policy_for_engine(engine), FailClosedPolicy)
    finally:
        dag.close()
        store.close()


def test_the_doctor_names_the_stray_stamp_and_its_repair(tmp_path):
    """A store that refuses all work has to say why, and what to do about it."""

    db_path = tmp_path / "stray-doctor.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        _stamp_one_row(store)
        check = doctor_scope_check(conn)
        assert check["check"] == SCOPE_DOCTOR_CHECK
        assert check["status"] == "fail"
        detail = check["detail"]
        assert detail["startup_state"] == "stamped-without-marker"
        assert detail["stamped_tables"] == {"messages": 1}
        repair = str(detail["repair"])
        # Both origins, because the repair differs between them.
        assert "setup_teams_scope" in repair
        assert "persist_teams_enabled(conn, True)" in repair
        assert "persist_teams_enabled(conn, False)" in repair
        assert "import_lossless_claw" in repair
    finally:
        dag.close()
        store.close()


def test_the_doctor_is_green_on_a_never_enabled_store(tmp_path):
    """The other negative control: dormant Teams is not a defect."""

    db_path = tmp_path / "clean-doctor.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        store.append("s1", {"role": "user", "content": "hello"})
        dag.add_node(SummaryNode(session_id="s1", summary="hi", created_at=1.0))
        check = doctor_scope_check(conn)
        assert check["status"] == "pass"
        assert check["detail"]["startup_state"] == "never-enabled"
        assert check["detail"]["stamped_tables"] == {}

        engine = _Engine()
        enabled, _ = resolve_startup_teams_state(conn)
        if enabled:
            scope_storage.mark_teams_enabled(engine)
        assert isinstance(policy_for_engine(engine), TrustedOwnerPolicy)
    finally:
        dag.close()
        store.close()


def test_the_doctor_reports_a_cache_that_outlived_its_fact(tmp_path):
    """The backstop for the F2 optimization.

    `verify_scope_storage` counts stamps from the rows themselves, so a cached
    never-enabled answer that the data contradicts is reported rather than
    believed -- the cache can only ever cost a doctor failure, never a silent
    permissive policy.
    """

    db_path = tmp_path / "stale-cache.db"
    store, dag = _open_stock(db_path)
    try:
        conn = store.connection
        bind_startup_teams_state(conn)
        # Simulate a stamp writer that did not clear the cache.
        store.append("imported-session", {"role": "user", "content": "hello"})
        conn.execute("UPDATE messages SET access_scope = 'principal-a'")
        conn.commit()
        assert read_never_enabled_cache(conn) is True

        check = doctor_scope_check(conn)
        assert check["status"] == "fail"
        assert "clear_never_enabled_cache" in str(check["detail"]["repair"])
    finally:
        dag.close()
        store.close()


def test_the_check_reaches_lcm_doctor_with_its_guidance(tmp_path):
    """The check is only useful where an operator actually looks.

    Pinned end to end -- tool output AND the guidance list -- because a check
    that is built but never appended, or appended under a name the guidance
    table does not know, is silent in exactly the state it exists to report.
    """

    from hermes_lcm.config import LCMConfig
    from hermes_lcm.engine import LCMEngine
    from hermes_lcm import tools as lcm_tools

    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    try:
        engine._store.append("imported-session", {"role": "user", "content": "hello"})
        clear_never_enabled_cache(engine._store.connection)
        engine._store.connection.execute("UPDATE messages SET access_scope = 'principal-a'")
        engine._store.connection.commit()

        report = json.loads(lcm_tools.lcm_doctor({}, engine=engine))
        check = next(
            item for item in report["checks"] if item["check"] == SCOPE_DOCTOR_CHECK
        )
        assert check["status"] == "fail"
        guidance = next(
            item
            for item in report["guidance"]
            if item["check"] == SCOPE_DOCTOR_CHECK
        )
        assert "setup_teams_scope" in guidance["operator_action"]
    finally:
        engine.shutdown()


def test_every_scope_bearing_table_the_module_names_is_migrated(tmp_path):
    """The migration and the table list must not disagree.

    A table named in `SCOPE_BEARING_TABLES` but never migrated is unstamped
    forever; the doctor reports it as defective and the enable cannot fix it.
    """

    db_path = tmp_path / "all-tables.db"
    store, dag = _open_stock(db_path)
    rollups = RollupStore(db_path)
    conn = store.connection
    db_bootstrap.ensure_embedding_tables(conn)
    db_bootstrap.ensure_chunk_tables(conn)
    try:
        for table in SCOPE_BEARING_TABLES:
            assert "access_scope" in _columns(conn, table), table
    finally:
        rollups.close()
        dag.close()
        store.close()


def test_lazy_tables_are_target_repaired_after_the_core_marker(tmp_path):
    """Lazy families bypass the completed core marker for their own tables."""

    db_path = tmp_path / "lazy-scope.db"
    store, dag = _open_stock(db_path)
    conn = store.connection
    lazy_groups = (
        (
            db_bootstrap.ensure_embedding_tables,
            ("lcm_embedding_meta", "lcm_embedding_vectors", "lcm_embedding_binary"),
        ),
        (
            db_bootstrap.ensure_chunk_tables,
            ("lcm_chunk_meta", "lcm_chunk_vectors", "lcm_chunk_binary"),
        ),
    )
    try:
        assert conn.execute(
            "SELECT 1 FROM lcm_migration_state WHERE step_name = ?",
            (scope_storage.SCOPE_MIGRATION_STEP,),
        ).fetchone() is not None
        assert not any(_tables(conn) & set(tables) for _, tables in lazy_groups)

        for initializer, tables in lazy_groups:
            initializer(conn)
            for table in tables:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN access_scope")
            conn.commit()

            initializer(conn)
            assert all("access_scope" in _columns(conn, table) for table in tables)
            assert "missing access_scope column" not in str(
                verify_scope_storage(conn, teams_enabled=False)
            )

            schema_version = conn.execute("PRAGMA schema_version").fetchone()[0]
            initializer(conn)
            assert conn.execute("PRAGMA schema_version").fetchone()[0] == schema_version
    finally:
        dag.close()
        store.close()
