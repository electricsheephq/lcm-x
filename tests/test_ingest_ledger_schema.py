"""#2055 P1a: additive ledger schema, salt, and digest; no ingest writers."""

import hashlib
import sqlite3

import pytest

from hermes_lcm import db_bootstrap, store as store_module
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore, ledger_digest


@pytest.fixture
def store(tmp_path):
    instance = MessageStore(tmp_path / "lcm.db")
    try:
        yield instance
    finally:
        instance.close()


def test_idempotent_metadata_and_states(store):
    assert store.ingest_ledger_state() == "absent"
    assert store.ingest_ledger_salt() is None
    store.ensure_ingest_ledger()
    salt = store.ingest_ledger_salt()
    assert isinstance(salt, bytes) and len(salt) == 32
    assert store.ingest_ledger_state() == "ready"
    assert dict(store.connection.execute("SELECT key, value FROM ledger_meta")) == {
        "digest_salt": salt, "ledger_format": 1, "view_seq": 0,
    }
    store.ensure_ingest_ledger()
    assert store.ingest_ledger_salt() == salt
    with store.atomic():
        store.connection.execute("DELETE FROM ledger_meta WHERE key='digest_salt'")
    assert store.ingest_ledger_salt() is None
    assert store.ingest_ledger_state() == "salt_missing"


def test_fresh_stores_have_distinct_salts(store, tmp_path):
    other = MessageStore(tmp_path / "other.db")
    try:
        store.ensure_ingest_ledger()
        other.ensure_ingest_ledger()
        assert len(other.ingest_ledger_salt()) == 32
        assert store.ingest_ledger_salt() != other.ingest_ledger_salt()
    finally:
        other.close()


def test_occurrence_ordinal_uniqueness(store):
    store.ensure_ingest_ledger()
    sql = ("INSERT INTO occurrence_bindings(lineage, ordinal, identity_digest, disposition, view_seq) "
           "VALUES ('L', 1, ?, ?, 0)")
    with store.atomic():
        store.connection.execute(sql, (b"digest", "stored"))
    for disposition in ("stored", "migrated", "ambiguous"):
        with pytest.raises(sqlite3.IntegrityError), store.atomic():
            store.connection.execute(sql, (b"digest", disposition))
    with store.atomic():
        store.connection.execute(sql, (b"digest", "composite_part"))
        store.connection.execute(sql, (b"digest", "split_part"))
    assert store.connection.execute("SELECT count(*) FROM occurrence_bindings").fetchone()[0] == 3


@pytest.mark.parametrize("sql, args", [
    ("INSERT INTO occurrence_bindings(lineage, identity_digest, disposition, view_seq) VALUES ('L', ?, ?, 0)",
     (b"digest", "unknown")),
    ("INSERT INTO ledger_emissions(lineage, view_seq, output_index, descriptor_digest, kind, state) "
     "VALUES ('L', 0, 0, ?, ?, ?)", (b"digest", "unknown", "proposed")),
    ("INSERT INTO ledger_emissions(lineage, view_seq, output_index, descriptor_digest, kind, state) "
     "VALUES ('L', 0, 0, ?, ?, ?)", (b"digest", "summary", "unknown")),
])
def test_checked_values(store, sql, args):
    store.ensure_ingest_ledger()
    with pytest.raises(sqlite3.IntegrityError), store.atomic():
        store.connection.execute(sql, args)


def test_digest_encoding():
    salt = b"s" * 32
    parts = (None, b"raw", "é", -1)
    encoded = (b"n" + (0).to_bytes(8, "big")
               + b"b" + (3).to_bytes(8, "big") + b"raw"
               + b"s" + (2).to_bytes(8, "big") + "é".encode("utf-8")
               + b"i" + (8).to_bytes(8, "big") + (-1).to_bytes(8, "big", signed=True))
    expected = hashlib.blake2b(encoded, key=salt, digest_size=16).digest()
    assert ledger_digest(salt, *parts) == expected == ledger_digest(salt, *parts)
    assert ledger_digest(b"t" * 32, *parts) != expected
    assert ledger_digest(salt, "ab", "c") != ledger_digest(salt, "a", "bc")
    assert ledger_digest(salt, None) != ledger_digest(salt, "")
    assert ledger_digest(salt, b"1") != ledger_digest(salt, "1") != ledger_digest(salt, 1)


def test_interim_stamp_classification_unchanged(store):
    dag = SummaryDAG(store.db_path)
    dag.close()
    with store.atomic():
        db_bootstrap.set_schema_version(store.connection, db_bootstrap.SCHEMA_VERSION + 1)
    before = db_bootstrap.classify_version_mismatch(store.connection)
    assert before == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    store.ensure_ingest_ledger()
    assert db_bootstrap.classify_version_mismatch(store.connection) == before


def test_failed_creation_does_not_cache_uncommitted_schema(store, monkeypatch):
    mark = store_module.mark_migration_step_complete

    def fail(conn, step):
        assert store.ingest_ledger_state() == "salt_missing"  # re-entrant DDL read
        mark(conn, step)
        raise sqlite3.OperationalError("synthetic migration failure")

    monkeypatch.setattr(store_module, "mark_migration_step_complete", fail)
    with pytest.raises(sqlite3.OperationalError, match="synthetic"):
        store.ensure_ingest_ledger()
    assert store.ingest_ledger_state() == "absent"
    assert not store.connection.in_transaction
    assert not store.connection.execute(
        "SELECT 1 FROM lcm_migration_state WHERE step_name='ingest_ledger_v1'"
    ).fetchone()
    monkeypatch.setattr(store_module, "mark_migration_step_complete", mark)
    store.ensure_ingest_ledger()
    assert store.ingest_ledger_state() == "ready"
