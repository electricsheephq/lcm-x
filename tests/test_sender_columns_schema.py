"""S0 additive shape, write-once isolation, and old-store classifier contracts."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.query_view_store import _ensure_query_view_schema
from hermes_lcm.store import MessageStore


def columns(conn):
    return list(conn.execute("PRAGMA table_info(messages)"))


def legacy(path, *, teams=False):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            source TEXT DEFAULT '', conversation_id TEXT DEFAULT '', role TEXT NOT NULL,
            content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
            timestamp REAL NOT NULL, token_estimate INTEGER DEFAULT 0, pinned INTEGER DEFAULT 0,
            ingested_at REAL, observed_at REAL, observed_at_source TEXT);
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO messages(session_id, role, content, timestamp, ingested_at)
        VALUES ('S', 'user', 'synthetic historical row', 1, 1);
    """)
    if teams:
        conn.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
    before = [r[1] for r in columns(conn)]
    conn.commit()
    conn.close()
    return before


def derived_snapshot(conn):
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
              if r[0].startswith(("messages_fts", "lcm_assertion", "lcm_query_corpus"))]
    result = {name: sorted(conn.execute(f'SELECT * FROM "{name}"').fetchall(), key=repr) for name in tables}
    result["markers"] = list(conn.execute(
        "SELECT * FROM metadata WHERE key LIKE 'chunk_gapfill_scan:%' "
        "OR key LIKE 'chunk_policy_revision_through:%' ORDER BY key"))
    return result


def prepare_derived(store):
    conn = store.connection
    db_bootstrap.ensure_chunk_tables(conn)
    db_bootstrap.ensure_assertion_tables(conn)
    _ensure_query_view_schema(conn)
    conn.execute("INSERT OR REPLACE INTO metadata VALUES ('chunk_gapfill_scan:S', '7')")
    conn.execute("INSERT OR REPLACE INTO metadata VALUES ('chunk_policy_revision_through:S', '8')")
    conn.commit()


@pytest.mark.parametrize("teams_first", [False, True], ids=["s0-then-teams", "teams-then-s0"])
def test_migration_shape_and_idempotence(tmp_path, teams_first):
    path = tmp_path / "legacy.db"
    before = legacy(path, teams=teams_first)
    store = MessageStore(path)
    try:
        conn = store.connection
        info = columns(conn)
        names = [r[1] for r in info]
        assert names[-2:] == ["sender_id", "sender_source"]
        assert [n for n in names if n not in {"sender_id", "sender_source"}] == before
        assert all(r[2:5] == ("TEXT", 0, None) for r in info[-2:])
        assert db_bootstrap.SCHEMA_VERSION == 5
        assert list(conn.execute("SELECT sender_id, sender_source FROM messages")) == [(None, None)]
        if not teams_first:
            conn.execute("ALTER TABLE messages ADD COLUMN access_scope TEXT")
            conn.commit()
        first = list(conn.execute("SELECT * FROM sqlite_master ORDER BY name"))
    finally:
        store.close()
    reopened = MessageStore(path)
    try:
        assert list(reopened.connection.execute("SELECT * FROM sqlite_master ORDER BY name")) == first
    finally:
        reopened.close()


def test_fresh_shape_and_concurrent_ensure(tmp_path):
    store = MessageStore(tmp_path / "fresh.db")
    try:
        assert [r[1] for r in columns(store.connection)][-2:] == ["sender_id", "sender_source"]
        def ensure(_):
            connection = sqlite3.connect(store.db_path, timeout=5)
            try:
                proxy = object.__new__(MessageStore)
                proxy._conn = connection
                proxy._ensure_time_contract_columns()
                connection.commit()
            finally:
                connection.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(ensure, range(2)))
    finally:
        store.close()


def test_sender_update_changes_one_cell_pair_without_trigger_effects(tmp_path):
    path = tmp_path / "update.db"
    legacy(path)
    store = MessageStore(path)
    try:
        prepare_derived(store)
        conn = store.connection
        before = derived_snapshot(conn)
        changes = conn.total_changes
        assert store.set_sender_if_null(1, "telegram:synthetic", "gateway_principal", ["S"]) == 1
        assert conn.total_changes == changes + 1
        assert derived_snapshot(conn) == before
        assert store.set_sender_if_null(1, "telegram:other", "gateway_principal", ["S"]) == 0
        assert conn.total_changes == changes + 1
        assert store.set_sender_if_null(1, "telegram:other", "gateway_principal", ["elsewhere"]) == 0
    finally:
        store.close()


def test_trigger_census_and_reader_projection(tmp_path):
    store = MessageStore(tmp_path / "census.db")
    try:
        prepare_derived(store)
        assert {"sender_id", "sender_source"} <= {r[1] for r in columns(store.connection)}
        updates = [r[0].upper() for r in store.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name='messages'")
            if "UPDATE" in r[0].split(" ON messages")[0].upper()]
        assert len(updates) >= 5
        for sql in updates:
            clause = sql.split(" ON MESSAGES")[0]
            assert "UPDATE OF " in clause
            assert "SENDER_ID" not in clause and "SENDER_SOURCE" not in clause
        sid = store.append("S", {"role": "user", "content": "synthetic"})
        assert store.get_message_senders([sid]) == {sid: (None, None)}
        assert "sender_id" not in store.get(sid)
    finally:
        store.close()


def test_sender_update_joins_enclosing_transaction(tmp_path):
    store = MessageStore(tmp_path / "enclosing.db")
    try:
        sid = store.append("S", {"role": "user", "content": "synthetic"})
        conn = store.connection
        conn.execute("BEGIN")
        assert store.set_sender_if_null(sid, "telegram:synthetic", "gateway_principal", ["S"]) == 1
        assert conn.in_transaction
        conn.rollback()
        assert store.get_message_senders([sid]) == {sid: (None, None)}
    finally:
        store.close()


def test_classifier_tolerates_only_sender_columns(tmp_path):
    dag = SummaryDAG(tmp_path / "classify.db")
    store = MessageStore(tmp_path / "classify.db")
    try:
        assert {"sender_id", "sender_source"} <= {r[1] for r in columns(store.connection)}
        conn = store.connection
        db_bootstrap.set_schema_version(conn, 6)
        conn.commit()
        assert db_bootstrap.classify_version_mismatch(conn) == "interim_stamp"
        conn.execute("ALTER TABLE messages ADD COLUMN unknown_future_column TEXT")
        assert db_bootstrap.classify_version_mismatch(conn) == "genuinely_newer"
        assert not {"sender_id", "sender_source"} & set(db_bootstrap._V5_CORE_OPTIONAL_COLUMNS["messages"])
    finally:
        store.close()
        dag.close()
