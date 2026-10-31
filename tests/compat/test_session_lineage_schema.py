"""#1931: the additive ``session_lineage`` table (named step session_lineage_v1), forward and rollback (I5)."""

import hashlib
import sqlite3

from tests.compat.helpers import PREVIOUS_GA, READERS, open_engine, run_old

VIEW = [{"role": "user", "content": "Synthetic lineage request.", "timestamp": 1.0},
        {"role": "assistant", "content": "Synthetic lineage response.", "timestamp": 2.0}]
OLD_TURN = r'''
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    view = [{"role": "user", "content": "Synthetic lineage request.", "timestamp": 1.0},
            {"role": "assistant", "content": "Synthetic lineage response.", "timestamp": 2.0}]
    if mode == "rollback":
        assert [row["content"] for row in engine._store.get_session_messages("S")] == [m["content"] for m in view]
        assert [row["content"] for row in engine._store.get_session_messages("child")] == ["Child row."]
        view.append({"role": "user", "content": "Synthetic rollback turn.", "timestamp": 3.0})
    engine.ingest(view)
    rows = engine._store.get_session_messages("S")
    assert [row["content"] for row in rows] == [row["content"] for row in view]
    print(json.dumps({"rows": [[r["store_id"], r["role"], r["content"]] for r in rows]}))
finally:
    engine.shutdown()
'''


def _snapshot(conn, tables):
    result = {}
    for table in tables:
        where = " WHERE step_name != 'session_lineage_v1'" if table == "lcm_migration_state" else ""
        rows = sorted(repr(tuple(row)) for row in conn.execute(f'SELECT * FROM "{table}"{where}'))
        result[table] = (len(rows), hashlib.sha256(repr(rows).encode()).hexdigest())
    return result


def test_previous_ga_store_gains_lineage_without_row_changes(tmp_path, old_tree, host_root):
    report = run_old(old_tree(PREVIOUS_GA, READERS), tmp_path, host_root, OLD_TURN, mode="seed")
    engine = open_engine(tmp_path)
    try:
        store, conn = engine._store, engine._store.connection
        assert [[r["store_id"], r["role"], r["content"]] for r in store.get_session_messages("S")] == report["rows"]
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        assert "session_lineage" not in tables
        before = _snapshot(conn, tables)
        version = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        store.record_session_lineage("child", "S", source="subagent_start")
        store.record_session_lineage("child", "other", source="parent_frame")  # the first link wins
        assert _snapshot(conn, tables) == before
        assert conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone() == version
        assert conn.execute("SELECT 1 FROM lcm_migration_state WHERE step_name='session_lineage_v1'").fetchone()
        assert store.session_lineage_parents(["child", "S"]) == {"child": "S"}
        assert store.session_lineage_descendants("S") == ["child"]
    finally:
        engine.shutdown()


def test_previous_ga_reads_and_ingests_with_lineage_table(tmp_path, old_tree, host_root):
    engine = open_engine(tmp_path)
    try:
        engine.ingest(VIEW)
        engine._store.record_session_lineage("child", "S", source="subagent_start")
        engine._store.append_batch("child", [{"role": "assistant", "content": "Child row."}], conversation_id="child")
    finally:
        engine.shutdown()
    report = run_old(old_tree(PREVIOUS_GA, READERS), tmp_path, host_root, OLD_TURN, mode="rollback")
    assert len(report["rows"]) == 3
    with sqlite3.connect(tmp_path / "lcm.db") as conn:
        assert conn.execute("SELECT child_session_id, parent_session_id FROM session_lineage").fetchall() == [
            ("child", "S")]
