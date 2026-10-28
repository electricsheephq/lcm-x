"""#2055: forward and rollback compatibility before ledger writers land (I5)."""

import hashlib
import sqlite3

from tests.compat.helpers import PREVIOUS_GA, READERS, open_engine, run_old


TABLES = {"ledger_meta", "occurrence_bindings", "ledger_emissions", "stamp_observations",
          "ingest_views", "lineage_marks"}
VIEW = [{"role": "user", "content": "Synthetic ledger request.", "timestamp": 1.0},
        {"role": "assistant", "content": "Synthetic ledger response.", "timestamp": 2.0}]
OLD_TURN = r'''
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    view = [{"role": "user", "content": "Synthetic ledger request.", "timestamp": 1.0},
            {"role": "assistant", "content": "Synthetic ledger response.", "timestamp": 2.0}]
    before = engine._store.get_session_messages("S")
    if mode == "rollback":
        assert len(before) == 2
        assert [row["content"] for row in before] == [row["content"] for row in view]
        view.append({"role": "user", "content": "Synthetic rollback turn.", "timestamp": 3.0})
    engine.ingest(view)
    rows = engine._store.get_session_messages("S")
    assert len(rows) == len(view)
    assert [row["content"] for row in rows] == [row["content"] for row in view]
    print(json.dumps({"rows": [[r["store_id"], r["role"], r["content"]] for r in rows]}))
finally:
    engine.shutdown()
'''


def _snapshot(conn, tables):
    """Counts and content hashes of every pre-existing table, excluding the new marker."""
    result = {}
    for table in tables:
        where = " WHERE step_name != 'ingest_ledger_v1'" if table == "lcm_migration_state" else ""
        rows = sorted(repr(tuple(row)) for row in conn.execute(f'SELECT * FROM "{table}"{where}'))
        result[table] = (len(rows), hashlib.sha256(repr(rows).encode()).hexdigest())
    return result


def test_previous_ga_store_upgrades_without_row_changes(tmp_path, old_tree, host_root):
    report = run_old(old_tree(PREVIOUS_GA, READERS), tmp_path, host_root, OLD_TURN, mode="seed")
    engine = open_engine(tmp_path)
    try:
        store = engine._store
        rows = store.get_session_messages("S")
        assert [[r["store_id"], r["role"], r["content"]] for r in rows] == report["rows"]
        conn = store.connection
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        assert not tables & TABLES
        before = _snapshot(conn, tables)
        version = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        store.ensure_ingest_ledger()
        assert _snapshot(conn, tables) == before
        assert conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone() == version
        assert conn.execute("SELECT 1 FROM lcm_migration_state WHERE step_name='ingest_ledger_v1'").fetchone()
        assert store.ingest_ledger_state() == "ready"
    finally:
        engine.shutdown()


def test_previous_ga_reads_and_ingests_with_ledger_tables(tmp_path, old_tree, host_root):
    engine = open_engine(tmp_path)
    try:
        engine.ingest(VIEW)
        assert len(engine._store.get_session_messages("S")) == 2
        engine._store.ensure_ingest_ledger()
        salt = engine._store.ingest_ledger_salt()
    finally:
        engine.shutdown()
    report = run_old(old_tree(PREVIOUS_GA, READERS), tmp_path, host_root, OLD_TURN, mode="rollback")
    assert len(report["rows"]) == 3
    with sqlite3.connect(tmp_path / "lcm.db") as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert TABLES <= tables
        assert conn.execute("SELECT value FROM ledger_meta WHERE key='digest_salt'").fetchone() == (salt,)
