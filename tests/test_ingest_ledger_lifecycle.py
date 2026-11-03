"""P1a carry-ins: missing-salt repair and both explicit message deletion paths.

No ledger API imports: these regressions must reach their assertions on P1a.
"""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from hermes_lcm.command import _delete_clean_candidates_atomically
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore

TABLES = ("occurrence_bindings", "stamp_observations", "ingest_views", "ledger_emissions", "lineage_marks")


@pytest.fixture
def store(tmp_path):
    instance = MessageStore(tmp_path / "lifecycle.db")
    try:
        yield instance
    finally:
        instance.close()


def seed(store):
    store.ensure_ingest_ledger()
    store._ensure_identity_anchor_schema()
    ids = [store.append(session, {"role": "user", "content": session}) for session in ("S", "T", "U")]
    with store.atomic():
        for number, (store_id, lineage) in enumerate(zip(ids, ("L", "L", "Q")), 1):
            occ_id = store.connection.execute(
                "INSERT INTO occurrence_bindings(lineage,ordinal,store_id,identity_digest,disposition,view_seq) "
                "VALUES (?,?,?,?, 'stored', 0)", (lineage, number, store_id, b"d" * 16)).lastrowid
            store.connection.execute("INSERT INTO stamp_observations VALUES (?,?,?,'alt_stamp',0,'ledger')",
                                     (lineage, occ_id, str(number)))
            store.connection.execute("INSERT INTO ledger_emissions(lineage,view_seq,output_index,descriptor_digest,kind,source_store_ids,state) "
                                     "VALUES (?,0,?,?,'summary',?,'adopted')", (lineage, number, b"e" * 16, json.dumps([store_id])))
        for lineage in ("L", "Q"):
            store.connection.execute("INSERT INTO ingest_views VALUES (?,0,'ingest',NULL,?,?)", (lineage, b"vector", b"chain"))
            store.connection.execute("INSERT INTO lineage_marks(lineage,max_store_id) VALUES (?,?)", (lineage, max(ids)))
        # An emission may refer to forgotten sources without any own binding.
        store.connection.execute("INSERT INTO ledger_emissions(lineage,view_seq,output_index,descriptor_digest,kind,source_store_ids,state) "
                                 "VALUES ('E',0,0,?,'summary',?,'adopted')", (b"e" * 16, json.dumps(ids[:2])))
        store.connection.execute("INSERT INTO ingest_views VALUES ('E',0,'emission',NULL,?,?)", (b"vector", b"chain"))
        store.connection.execute("INSERT INTO message_relations(store_id,kind,related_store_id) VALUES (?,'witness',?)", ids[:2])
    return ids


def snapshot(conn, tables):
    return {table: conn.execute(f"SELECT * FROM {table}").fetchall() for table in tables}


def test_salt_missing_resets_only_ledger_and_increments_epoch(store, caplog):
    seed(store)
    conn, salt = store.connection, store.ingest_ledger_salt()
    nonledger = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                 if row[0] not in (*TABLES, "ledger_meta", "sqlite_sequence")]
    before = snapshot(conn, nonledger)
    for epoch in (1, 2):
        with store.atomic():
            conn.execute("DELETE FROM ledger_meta WHERE key='digest_salt'")
        store.ensure_ingest_ledger()
        assert snapshot(conn, nonledger) == before
        assert all(not rows for rows in snapshot(conn, TABLES).values())
        assert store.ingest_ledger_salt() != salt
        assert conn.execute("SELECT value FROM ledger_meta WHERE key='salt_epoch'").fetchone()[0] == epoch
        if epoch == 1:
            with store.atomic():
                conn.execute("INSERT INTO lineage_marks(lineage) VALUES ('L')")
    assert "salt_missing" in caplog.text and all(table in caplog.text for table in TABLES)


def test_salt_present_untouched_and_empty_missing_salt(store):
    # Regression guard: both salt-present and empty-ledger setup already work on P1a.
    seed(store)
    before = snapshot(store.connection, (*TABLES, "ledger_meta"))
    store.ensure_ingest_ledger()
    assert snapshot(store.connection, (*TABLES, "ledger_meta")) == before
    with store.atomic():
        for table in TABLES:
            store.connection.execute(f"DELETE FROM {table}")
        store.connection.execute("DELETE FROM ledger_meta WHERE key='digest_salt'")
    store.ensure_ingest_ledger()
    assert len(store.ingest_ledger_salt()) == 32
    assert not store.connection.execute("SELECT 1 FROM ledger_meta WHERE key='salt_epoch'").fetchone()


def test_salt_repair_failure_rolls_back_all_ledger_content(store):
    seed(store)
    conn = store.connection
    with store.atomic():
        conn.execute("DELETE FROM ledger_meta WHERE key='digest_salt'")
        conn.execute("CREATE TRIGGER refuse_ledger_reset BEFORE DELETE ON lineage_marks "
                     "BEGIN SELECT RAISE(ABORT, 'synthetic reset failure'); END")
    before = snapshot(conn, (*TABLES, "ledger_meta", "messages", "message_relations"))
    with pytest.raises(sqlite3.IntegrityError, match="synthetic reset failure"):
        store.ensure_ingest_ledger()
    assert snapshot(conn, (*TABLES, "ledger_meta", "messages", "message_relations")) == before
    assert not conn.in_transaction


def purge(store):
    SummaryDAG(store.db_path).close()
    # MessageStore's bootstrap owns the lifecycle table and its cleanup indexes.
    return _delete_clean_candidates_atomically(SimpleNamespace(_store=store, _session_id="T"), {"S", "T"})["messages_deleted"]


@pytest.mark.parametrize("path", ["forget", "purge"])
@pytest.mark.parametrize("ledger_present", [False, True])
def test_forget_and_purge_scoped(store, path, ledger_present):
    if ledger_present:
        ids = seed(store)
        marks = snapshot(store.connection, ("lineage_marks",))
    else:
        # Regression guard: explicit deletion must also work on pre-ledger stores.
        ids = [store.append(session, {"role": "user", "content": session}) for session in ("S", "T", "U")]
    deleted = store.delete_session_messages("S") if path == "forget" else purge(store)
    assert deleted == 1
    assert [row[0] for row in store.connection.execute("SELECT store_id FROM messages ORDER BY store_id")] == ids[1:]
    if ledger_present:
        assert [row[0] for row in store.connection.execute("SELECT store_id FROM occurrence_bindings ORDER BY store_id")] == ids[1:]
        assert [row[0] for row in store.connection.execute("SELECT stamp FROM stamp_observations ORDER BY stamp")] == ["2", "3"]
        assert [row[0] for row in store.connection.execute("SELECT lineage FROM ingest_views")] == ["Q"]
        assert [json.loads(row[0]) for row in store.connection.execute("SELECT source_store_ids FROM ledger_emissions ORDER BY output_index")] == [[ids[1]], [ids[2]]]
        assert snapshot(store.connection, ("lineage_marks",)) == marks
    else:
        assert store.ingest_ledger_state() == "absent"


@pytest.mark.parametrize("path", ["forget", "purge"])
def test_deletion_failure_rolls_back_ledger_with_messages(store, path):
    seed(store)
    tables = (*TABLES, "messages", "message_relations")
    before = snapshot(store.connection, tables)
    with store.atomic():
        store.connection.execute("CREATE TRIGGER refuse_message_delete BEFORE DELETE ON messages "
                                 "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="synthetic failure"):
        store.delete_session_messages("S") if path == "forget" else purge(store)
    assert snapshot(store.connection, tables) == before


@pytest.mark.parametrize("path", ["forget", "purge"])
def test_emission_links_and_malformed_sources(store, path):
    deleted_id = seed(store)[0]
    conn = store.connection
    occ_id = conn.execute("SELECT occ_id FROM occurrence_bindings WHERE store_id=?", (deleted_id,)).fetchone()[0]
    with store.atomic():
        for index, lineage, sources, next_occ, recovered in (
                (10, "Q", "{malformed", None, None), (11, "N", "[]", occ_id, None),
                (12, "R", "[]", None, deleted_id)):
            conn.execute("INSERT INTO ledger_emissions(lineage,view_seq,output_index,descriptor_digest,kind,source_store_ids,state,next_occ_id,recovered_store_id) "
                         "VALUES (?,0,?,?,'summary',?,'adopted',?,?)", (lineage, index, b"e" * 16, sources, next_occ, recovered))
            if lineage != "Q":
                conn.execute("INSERT INTO ingest_views VALUES (?,0,'emission',NULL,?,?)", (lineage, b"v", b"c"))
    store.delete_session_messages("S") if path == "forget" else purge(store)
    assert conn.execute("SELECT output_index FROM ledger_emissions WHERE output_index>=10").fetchall() == [(10,)]
    assert conn.execute("SELECT lineage FROM ingest_views").fetchall() == [("Q",)]
