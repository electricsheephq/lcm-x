"""Backup-first duplicate cleanup on synthetic stores."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import pytest

from hermes_lcm import ingest_ledger, maintenance
from hermes_lcm.command import _ensure_inflight_table, handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore


CONTENT = "café " * 1600
HASH = sha256(CONTENT.encode()).hexdigest()
SCAN = "doctor dedupe"
APPLY = "doctor dedupe apply"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    config = LCMConfig(
        database_path=str(tmp_path / "synthetic.db"), fresh_tail_count=2,
        embeddings_enabled=False, temporal_rollups_enabled=False,
    )
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "host"))
    instance._session_id = "session"
    instance._conversation_id = "conversation"
    instance._lifecycle.bind_session("session", conversation_id="conversation")
    try:
        yield instance
    finally:
        instance.shutdown()


def append(engine, content=CONTENT, *, session="session", conversation="conversation"):
    return engine._store.append(
        session, {"role": "assistant", "content": content}, conversation_id=conversation,
    )


def cover(engine, ids):
    engine._dag.add_node(SummaryNode(session_id="session", summary="Earlier facts", source_ids=ids))


def bind(engine, store_id):
    engine._store.add_host_uid_bindings("test-lineage", [(store_id, f"uid-{store_id}", "canonical", "agree_new")])


def append_tail(engine, conversation="conversation"):
    ids = [append(engine, f"fresh row {i}", conversation=conversation) for i in range(2)]
    bind(engine, ids[-1])
    return ids


def seed_copies(engine):
    covered = [append(engine) for _ in range(14)]
    cover(engine, covered)
    engine._lifecycle.advance_frontier("conversation", "session", covered[-1])
    removable = [append(engine) for _ in range(11)]
    return covered, removable, append_tail(engine)


def ids(conn, table, column="store_id"):
    return {row[0] for row in conn.execute(f"SELECT {column} FROM {table}")}


def seed_ledger(store, lineage, messages):
    store.ensure_ingest_ledger()
    conn = store.connection
    salt = store.ingest_ledger_salt()
    entries, stamps, occurrences = [], [], []
    with store.atomic():
        for index, store_id in enumerate(messages, 1):
            digest = store_id.to_bytes(16, "big")
            stamp = str(index * 100)
            occurrence = conn.execute(
                "INSERT INTO occurrence_bindings(lineage,ordinal,store_id,identity_digest,stamp,disposition,view_seq) "
                "VALUES (?,?,?,?,?,'stored',1)", (lineage, index, store_id, digest, stamp),
            ).lastrowid
            conn.execute(
                "INSERT INTO stamp_observations(lineage,occ_id,stamp,kind,view_seq) VALUES (?,?,?,'alt_stamp',1)",
                (lineage, occurrence, stamp),
            )
            source_ids, next_occ_id, recovered_store_id = [
                (json.dumps([store_id]), None, None), ("[]", occurrence, None), ("[]", None, store_id),
            ][index % 3]
            conn.execute(
                "INSERT INTO ledger_emissions(lineage,view_seq,output_index,descriptor_digest,kind,"
                "source_store_ids,state,next_occ_id,recovered_store_id) VALUES (?,1,?,?,'summary',?,'adopted',?,?)",
                (lineage, index, digest, source_ids, next_occ_id, recovered_store_id),
            )
            entries.append(ingest_ledger.ViewEntry(digest, None, None, occurrence))
            stamps.append(stamp)
            occurrences.append(occurrence)
        ingest_ledger.write_view(conn, salt, lineage, "ingest", None, entries)
    return occurrences, {
        "lineage": lineage, "entries": entries, "stamps": stamps,
        "chain": ingest_ledger.prefix_chain(salt, entries),
    }


def cold_cursor(conn, attempt):
    return ingest_ledger.cold_cursor_candidate(
        conn, session_id="session", attempt=attempt, ignored=[False] * len(attempt["entries"]),
    )


def seed_chunks(store, messages):
    vectors = VectorStore(store.db_path)
    try:
        identity = EmbeddingIdentity.canonical("synthetic", "test-model", "", 2, "float32", "little", "chunk")
        profile = vectors.register_profile("test-model", "synthetic", 2, task="chunk")
        for store_id in messages:
            vectors.record_chunk_embedding(
                f"{store_id}:0", "test-model", [1.0, 0.0], store_id=store_id, chunk_index=0,
                char_start=0, char_end=6, token_estimate=1, identity=identity,
            )
        _ensure_inflight_table(vectors.connection)
        vectors.connection.executemany(
            "INSERT INTO lcm_embedding_backfill_inflight(embedded_id,identity_hash) VALUES (?,?)",
            [(f"{store_id}:0", profile) for store_id in messages],
        )
        vectors.connection.commit()
    finally:
        vectors.close()


def test_scan_reports_twenty_five_copies_without_changing_store(engine):
    _, removable, _ = seed_copies(engine)
    conn = engine._store.connection
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    paths = [Path(f"{engine._store.db_path}{suffix}") for suffix in ("", "-wal")]
    before = [path.read_bytes() for path in paths]

    result = handle_lcm_command(SCAN, engine)

    assert result.startswith("LCM doctor dedupe\n")
    assert f"{HASH[:12]} length=8000 copies=25 kept=14 removable=11 unproven=0" in result
    assert "covered=14" in result
    assert f"removable_ids={','.join(map(str, removable))}" in result
    assert CONTENT not in result
    assert [path.read_bytes() for path in paths] == before
    assert not engine.backup_dir().exists()


def test_apply_cleans_dependents_and_keeps_cold_cursor_proof(engine):
    covered, removable, tail = seed_copies(engine)
    store = engine._store
    conn = store.connection
    gone_occ, gone_attempt = seed_ledger(store, "deleted", removable)
    live_occ, live_attempt = seed_ledger(store, "retained", tail)
    seed_chunks(store, [*removable, covered[0]])
    version_sql = "SELECT data_version FROM lcm_embedding_profile"
    version = conn.execute(version_sql).fetchone()[0]
    timestamps = dict(conn.execute("SELECT store_id,timestamp FROM messages"))
    assert cold_cursor(conn, gone_attempt)[0]["k"] == 11
    assert cold_cursor(conn, live_attempt)[0]["k"] == 2
    fts_tables = ("messages_fts_docsize", "messages_fts_stem_docsize")
    for table in fts_tables:
        assert set(removable) <= ids(conn, table, "id")

    result = handle_lcm_command(APPLY, engine)

    assert result.startswith("LCM doctor dedupe apply\n")
    assert "deleted: 11" in result and CONTENT not in result
    backup = next(line.removeprefix("backup_path: ") for line in result.splitlines() if line.startswith("backup_path: "))
    with closing(sqlite3.connect(Path(backup).as_uri() + "?mode=ro", uri=True)) as saved:
        assert ids(saved, "messages") == set(timestamps)
    assert ids(conn, "messages") == set(covered + tail)
    for table in fts_tables:
        assert set(covered) <= ids(conn, table, "id")
        assert not ids(conn, table, "id").intersection(removable)
    assert ids(conn, "occurrence_bindings") == set(tail)
    assert ids(conn, "stamp_observations", "occ_id") == set(live_occ)
    assert not ids(conn, "stamp_observations", "occ_id").intersection(gone_occ)
    assert ids(conn, "ledger_emissions", "lineage") == {"retained"}
    proof, counts = cold_cursor(conn, gone_attempt)
    assert proof["k"] == 0 and counts["stop.missing_row"] == 0
    proof, counts = cold_cursor(conn, live_attempt)
    assert proof["k"] == 2 and counts["stop.missing_row"] == 0
    assert dict(conn.execute("SELECT store_id,archived FROM lcm_chunk_meta")) == {
        covered[0]: 0, **dict.fromkeys(removable, 1),
    }
    assert conn.execute(version_sql).fetchone()[0] == version + 1
    assert ids(conn, "lcm_embedding_backfill_inflight", "embedded_id") == {f"{covered[0]}:0"}
    audit = {key: json.loads(value) for key, value in conn.execute(
        "SELECT key,value FROM metadata WHERE key GLOB 'doctor_dedupe:*'",
    )}
    assert set(audit) == {f"doctor_dedupe:{store_id}" for store_id in removable}
    for store_id in removable:
        record = audit[f"doctor_dedupe:{store_id}"]
        assert record.pop("deleted_at") >= timestamps[store_id]
        assert record == dict(
            session_id="session", conversation_id="conversation", role="assistant",
            content_sha256=HASH, content_length=8000, timestamp=timestamps[store_id],
            kept_store_id=covered[0],
        )


@pytest.mark.parametrize(("guard", "reason"), [
    ("short", "below minimum length"),
    ("no-covered", "no covered copy"),
    ("uid-bound", "host uid bound"),
    ("too-new", "unproven (no later host uid binding)"),
    ("uid-off", "unproven (host uid mode off)"),
    ("tail", "fresh tail"),
    ("only", "only row in session"),
    ("relation-from", "message relation"),
    ("relation-to", "message relation"),
    ("frontier", "at or below frontier"),
    ("other-conversation", "no covered copy"),
    ("different-role", "no covered copy"),
    ("different-tool-call-id", "no covered copy"),
    ("different-tool-calls", "no covered copy"),
    ("different-tool-name", "no covered copy"),
])
def test_apply_guards_keep_rows(engine, monkeypatch, guard, reason):
    content = CONTENT[:-1] if guard == "short" else CONTENT
    conn = engine._store.connection
    if guard == "other-conversation":
        cover(engine, [append(engine, content, conversation="other")])
    first = append(engine, content)
    mismatch_columns = {
        "different-role": ("role", "tool"),
        "different-tool-call-id": ("tool_call_id", "call"),
        "different-tool-calls": ("tool_calls", "[]"),
        "different-tool-name": ("tool_name", "read"),
    }
    if guard in mismatch_columns:
        column, value = mismatch_columns[guard]
        with engine._store.atomic():
            conn.execute(f"UPDATE messages SET {column}=? WHERE store_id=?", (value, first))
    if guard not in {"no-covered", "other-conversation"}:
        cover(engine, [first])
    if guard == "too-new":
        append_tail(engine)
    candidate = append(engine, content, session="single" if guard == "only" else "session")
    if guard == "only":
        engine._config.fresh_tail_count = 0
    if guard == "uid-bound":
        bind(engine, candidate)
    if guard == "uid-off":
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    if guard.startswith("relation-"):
        ends = (candidate, first) if guard == "relation-from" else (first, candidate)
        engine._store.add_message_relations([[(ends[0], "witness", ends[1], 0, None)]])
    if guard == "frontier":
        engine._lifecycle.advance_frontier("conversation", "session", candidate)
    if guard == "tail":
        bind(engine, append(engine, "fresh row"))
    elif guard == "too-new":
        append(engine, "fresh row")
        append(engine, "another fresh row")
        append_tail(engine, "other")
    else:
        append_tail(engine)
    if guard in mismatch_columns:
        append(engine, content)  # Make the uncovered rows a group without the nonidentical covered row.
        append_tail(engine)
    before = ids(conn, "messages")

    result = handle_lcm_command(APPLY, engine)

    assert result.startswith("LCM doctor dedupe apply\n")
    assert "deleted: 0" in result
    assert f"{reason}=" in result
    assert ids(conn, "messages") == before


def test_apply_is_idempotent(engine):
    seed_copies(engine)
    result = handle_lcm_command(APPLY, engine)
    assert result.startswith("LCM doctor dedupe apply\n")
    assert "deleted: 11" in result
    before = {p: p.read_bytes() for p in engine.backup_dir().glob("*.sqlite3")}
    assert "deleted: 0" in handle_lcm_command(APPLY, engine)
    assert {p: p.read_bytes() for p in engine.backup_dir().glob("*.sqlite3")} == before


def test_apply_aborts_on_failed_backup(engine, monkeypatch):
    covered, removable, tail = seed_copies(engine)

    def fail(*args):
        raise OSError("synthetic backup failure")

    monkeypatch.setattr(maintenance, "_backup_pair", fail)
    result = handle_lcm_command(APPLY, engine)

    assert result.startswith("LCM doctor dedupe apply\n")
    assert "backup failed" in result and "deleted: 0" in result
    assert ids(engine._store.connection, "messages") == set(covered + removable + tail)


def test_apply_rechecks_after_backup(engine, monkeypatch):
    covered, removable, tail = seed_copies(engine)
    backup_pair = maintenance._backup_pair
    added = []

    def back_up_then_bind(store, path):
        backup_pair(store, path)
        bind(engine, removable[0])
        cover(engine, [removable[1]])
        added.append(append(engine))
        added.extend(append_tail(engine))

    monkeypatch.setattr(maintenance, "_backup_pair", back_up_then_bind)
    result = handle_lcm_command(APPLY, engine)

    assert result.startswith("LCM doctor dedupe apply\n")
    assert "deleted: 9" in result
    assert ids(engine._store.connection, "messages") == set(covered + removable[:2] + tail + added)


def test_apply_rolls_back_on_delete_count_mismatch(engine):
    covered, removable, tail = seed_copies(engine)
    conn = engine._store.connection
    with engine._store.atomic():
        conn.execute(f"""
            CREATE TRIGGER ignore_synthetic_delete BEFORE DELETE ON messages
            WHEN old.store_id = {removable[-1]} BEGIN SELECT RAISE(IGNORE); END
        """)

    result = handle_lcm_command(APPLY, engine)

    assert result.startswith("LCM doctor dedupe apply\n")
    assert "status: error" in result and "deleted: 0" in result
    assert ids(conn, "messages") == set(covered + removable + tail)
    assert not conn.execute("SELECT 1 FROM metadata WHERE key GLOB 'doctor_dedupe:*'").fetchone()


def test_store_complete_walk_skips_deleted_positions(engine, caplog):
    covered, removable, tail = seed_copies(engine)
    result = handle_lcm_command(APPLY, engine)
    assert result.startswith("LCM doctor dedupe apply\n")
    assert "deleted: 11" in result

    with caplog.at_level("INFO", logger="hermes_lcm.store_complete"):
        leaf = engine._store_complete_input([], {}, {}, [], [], covered[-1], [], 10_000)
    assert [store_id for _row, claims in leaf for store_id in claims] == tail
    assert not any("contiguity hole" in record.message for record in caplog.records)
    assert not ids(engine._store.connection, "messages").intersection(removable)
