"""Backup-first duplicate cleanup on synthetic stores."""

from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import ingest_ledger, maintenance
from hermes_lcm.command import _ensure_inflight_table, handle_lcm_command
from hermes_lcm.compaction import _CompressionState, _LeafState
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.entity_store import EntityStore
from hermes_lcm.ingest_protection import protect_message_for_ingest
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
from tests.test_host_uid_shadow import _state_db


CONTENT = "café " * 1600
HASH = sha256(CONTENT.encode()).hexdigest()
TODO = "[Your active task list was preserved across context compression]\n- [>] 1. Review (in_progress)"
SCAN = "doctor dedupe"
APPLY = "doctor dedupe apply"
APPLY_USERS = f"{APPLY} users"


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    config = LCMConfig(
        database_path=str(tmp_path / "synthetic.db"), fresh_tail_count=2,
        embeddings_enabled=False, temporal_rollups_enabled=False,
    )
    config.doctor_dedupe_apply_enabled = True
    instance = LCMEngine(config=config, hermes_home=str(tmp_path / "host"))
    instance._session_id = "session"
    instance._conversation_id = "conversation"
    instance._lifecycle.bind_session("session", conversation_id="conversation")
    host_list(instance, lambda _, repair_alternation=False: [])
    try:
        yield instance
    finally:
        instance.shutdown()


def host_list(engine, read):
    engine.bind_session_state(SimpleNamespace(get_messages_as_conversation=read), "session")


def append(engine, content=CONTENT, *, session="session", conversation="conversation", role="assistant"):
    store_id = engine._store.append(
        session, {"role": role, "content": content}, conversation_id=conversation,
    )
    with engine._store.atomic():
        engine._store.connection.execute("UPDATE messages SET timestamp=1 WHERE store_id=?", (store_id,))
    return store_id


def cover(engine, ids, source_type="messages"):
    engine._dag.add_node(SummaryNode(session_id="session", summary="Earlier facts", source_ids=ids,
                                     source_type=source_type))


def bind(engine, store_id):
    engine._store.add_host_uid_bindings("test-lineage", [(store_id, f"uid-{store_id}", "canonical", "agree_new")])


def append_tail(engine, conversation="conversation"):
    ids = [append(engine, f"fresh row {i}", conversation=conversation) for i in range(2)]
    bind(engine, ids[-1])
    return ids


def seed_copies(engine, content=CONTENT):
    covered = [append(engine, content) for _ in range(14)]
    cover(engine, covered)
    engine._lifecycle.advance_frontier("conversation", "session", covered[-1])
    removable = [append(engine, content) for _ in range(11)]
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


@pytest.mark.parametrize("mapped", [False, True], ids=["absent", "mapping-veto"])
def test_scan_reports_twenty_five_copies_without_changing_store(engine, monkeypatch, mapped):
    steer = "[OUT-OF-BAND USER MESSAGE]Synthetic steer[/OUT-OF-BAND USER MESSAGE]"
    content = f"Lead\n\n{steer}{CONTENT}" if mapped else CONTENT
    covered, removable, _ = seed_copies(engine, content)
    mapped_ids = []
    if mapped:
        # The mapper strips the delivered steer block from rows with a tool call ID.
        with engine._store.atomic():
            engine._store.connection.execute("UPDATE messages SET tool_call_id='call' WHERE content=?", (content,))
        view = [{"role": "assistant", "content": f"Lead\n\n{CONTENT}".rstrip(), "tool_call_id": "call"}]
        host_list(engine, lambda _, repair_alternation=False: deepcopy(view))
        engine._last_compacted_store_id = covered[-1]
        real_map = engine._get_store_id_map_for_messages

        def record_mapping(messages):
            result = real_map(messages)
            mapped_ids.extend(result.values())
            return result

        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", record_mapping)
    conn = engine._store.connection
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    paths = [Path(f"{engine._store.db_path}{suffix}") for suffix in ("", "-wal")]
    before = [path.read_bytes() for path in paths]

    result = handle_lcm_command(SCAN, engine)

    assert [path.read_bytes() for path in paths] == before
    assert result.startswith("LCM doctor dedupe\n")
    if mapped:
        assert mapped_ids == [removable[0], removable[0]]
        assert "copies=25 kept=15 removable=10 unproven=0" in result
        assert "host listed=1" in result
        removable = removable[1:]
    else:
        assert f"{HASH[:12]} length=8000 copies=25 kept=14 removable=11 unproven=0" in result
    assert "covered=14" in result
    assert f"removable_ids={','.join(map(str, removable))}" in result
    assert CONTENT not in result
    assert not engine.backup_dir().exists()


def test_dedupe_scan_preserves_compress_placeholder_counts(engine):
    seed_copies(engine)
    previous = {("tool", "synthetic placeholder", "call", "read"): 2}
    engine._current_compress_placeholder_identity_counts = previous.copy()

    assert "removable=11" in handle_lcm_command(SCAN, engine)
    assert engine._current_compress_placeholder_identity_counts == previous


def test_dedupe_is_unproven_while_a_compress_pass_is_in_flight(engine):
    seed_copies(engine)
    assert "removable=11" in handle_lcm_command(SCAN, engine)
    engine._compress_occurrences = {}
    try:
        text = handle_lcm_command(SCAN, engine)
    finally:
        engine._compress_occurrences = None
    assert "removable=0" in text
    assert "host list unavailable" in text


def test_apply_cleans_dependents_and_keeps_cold_cursor_proof(engine):
    covered, removable, tail = seed_copies(engine)
    store = engine._store
    conn = store.connection
    gone_occ, gone_attempt = seed_ledger(store, "deleted", removable)
    live_occ, live_attempt = seed_ledger(store, "retained", tail)
    seed_chunks(store, [*removable, covered[0]])
    version_sql = "SELECT data_version FROM lcm_embedding_profile"
    version = conn.execute(version_sql).fetchone()[0]
    with store.atomic():
        conn.executemany("UPDATE messages SET sender_id='synthetic', sender_source='test', observed_at=12, source='cli' "
                         "WHERE store_id=?", [(store_id,) for store_id in removable])
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
        contents = dict(saved.execute("SELECT store_id, CAST(content AS BLOB) FROM messages"))
    assert ids(conn, "messages") == set(covered + tail)
    kept_content = conn.execute("SELECT CAST(content AS BLOB) FROM messages WHERE store_id=?", (covered[0],)).fetchone()[0]
    assert all(contents[store_id] == kept_content == CONTENT.encode() for store_id in removable)
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
            kept_store_id=covered[0], sender_id="synthetic", sender_source="test", observed_at=12, source="cli",
        )


@pytest.mark.parametrize(("guard", "reason"), [
    ("short", "groups=0"),
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
    ("different-content", "no covered copy"),
    ("too-young", "below minimum age"),
    ("later-twin", "no earlier covered copy"),
    ("node-id-collision", "no covered copy"),
    ("pinned", "pinned"),
])
def test_apply_guards_keep_rows(engine, monkeypatch, guard, reason):
    content = CONTENT[:-1] if guard == "short" else CONTENT
    conn = engine._store.connection
    if guard == "other-conversation":
        cover(engine, [append(engine, content, conversation="other")])
    first = append(engine, content[:-1] + "!" if guard == "different-content" else content)
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
    if guard not in {"no-covered", "other-conversation", "later-twin"}:
        cover(engine, [first], "nodes" if guard == "node-id-collision" else "messages")
    if guard == "too-new":
        append_tail(engine)
    candidate = append(engine, content, session="single" if guard == "only" else "session")
    if guard == "later-twin":
        cover(engine, [append(engine, content)])
    if guard == "too-young":
        with engine._store.atomic():
            conn.execute("UPDATE messages SET timestamp=? WHERE store_id=?", (time.time() - 1800, candidate))
    if guard == "pinned":
        with engine._store.atomic():
            conn.execute("UPDATE messages SET pinned=1 WHERE store_id=?", (candidate,))
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
    if guard in mismatch_columns or guard == "different-content":
        append(engine, content)  # Make the uncovered rows a group without the nonidentical covered row.
        append_tail(engine)
    before = ids(conn, "messages")

    result = handle_lcm_command(APPLY, engine)

    assert result.startswith("LCM doctor dedupe apply\n")
    assert "deleted: 0" in result
    assert (reason if guard == "short" else f"{reason}=") in result
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


def start_live_session(engine, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    host = Path(engine._hermes_home)
    host.mkdir(exist_ok=True)
    _state_db(host, [("session", None, None)])
    engine.on_session_start("session", platform="cli", context_length=200_000, conversation_id="conversation")
    historical = {"role": "assistant", "content": CONTENT, "timestamp": 20.0}
    live = [
        {"role": "user", "content": "Next question", "timestamp": 30.0},
        {"role": "assistant", "content": "Next answer", "timestamp": 40.0, "message_uid": "answer"},
        {"role": "user", "content": "Follow-up question", "timestamp": 50.0},
    ]
    engine.ingest(deepcopy([dict(historical, timestamp=10.0, message_uid="kept"), historical, *live]))
    store = engine._store
    kept, candidate, _, answer, _ = sorted(ids(store.connection, "messages"))
    assert ids(store.connection, "host_uid_bindings") == {kept, answer}
    cover(engine, [kept])
    engine._lifecycle.advance_frontier("conversation", "session", kept)
    engine._last_compacted_store_id = kept
    with store.atomic():
        store.connection.execute("UPDATE messages SET timestamp=1 WHERE store_id=?", (candidate,))
    return historical, live, kept, candidate


def prepare_leaf(engine, view):
    engine._compress_run.store_ids_by_message_id = engine._get_store_id_map_for_messages(view)
    state = _CompressionState(working_messages=view)
    leaf = _LeafState(to_compact=view[:1])
    action, _ = engine._prepare_leaf_input(state, leaf, set(), lambda _: 0)
    return action, state.noop_reason


def test_dedupe_host_visible_copy_is_protected(engine, monkeypatch):
    historical, live, _, candidate = start_live_session(engine, monkeypatch)
    view = [historical, *live]
    host_list(engine, lambda _, repair_alternation=False: deepcopy(view))
    assert engine._get_store_id_map_for_messages(view)[id(historical)] == candidate
    assert prepare_leaf(engine, view) == ("next", "")
    scan = handle_lcm_command(SCAN, engine)
    result = handle_lcm_command(APPLY, engine)
    assert "removable=0" in scan and "host listed=1" in scan
    assert "deleted: 0" in result and "host listed=1" in result
    assert candidate in ids(engine._store.connection, "messages")


def test_dedupe_deletes_unlisted_copy_with_empty_and_short_host_rows(engine):
    hidden = f"The operation returned ok. {CONTENT}"
    kept = append(engine, hidden)
    long_reply = "File contents follow. " * 400
    covered_reply = append(engine, long_reply)
    cover(engine, [kept, covered_reply])
    candidate = append(engine, hidden)
    listed = [
        {"role": "user", "content": "Read the file."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "read-1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ]},
        {"role": "tool", "content": "File contents", "tool_call_id": "read-1", "tool_name": "read"},
        {"role": "assistant", "content": long_reply},
        {"role": "user", "content": "Thanks."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "What did the file say?"},
    ]
    store = engine._store
    listed_ids = [store.append("session", message, conversation_id="conversation") for message in listed]
    with store.atomic():
        store.connection.execute("UPDATE messages SET timestamp=1 WHERE content=?", (long_reply,))
    bind(engine, listed_ids[-1])
    host_list(engine, lambda _, repair_alternation=False: deepcopy(listed))

    result = handle_lcm_command(APPLY, engine)

    assert f"removable_ids={candidate}" in result and "deleted: 1" in result
    assert "host listed=1" in result
    assert ids(store.connection, "messages") == {kept, covered_reply, *listed_ids}


@pytest.mark.parametrize("shape", [
    "whitespace", "internal-whitespace", "middle-block", "host-scaffolding", "stored-scaffolding",
    "contained", "short-contained", "reverse-contained",
])
def test_dedupe_transformed_host_text_is_protected(engine, shape):
    block = "\n[synthetic block stripped by the host loader]\n"
    fragment = f"{'x' * 127} {'y' * 128}"
    if shape in {"whitespace", "internal-whitespace"}:
        stored = CONTENT
    elif shape == "middle-block":
        stored = f"{CONTENT[:2500]}{block}{CONTENT[2500:]}"
    elif shape == "stored-scaffolding":
        stored = f"Lead\n\n{TODO}\n\n{CONTENT}"
    elif shape in {"contained", "short-contained"}:
        stored = f"Lead\n\n{fragment}\n\n{CONTENT}"
    else:
        stored = f"Lead\n\n{CONTENT}"
    cover(engine, [append(engine, stored)])
    candidate = append(engine, stored)
    append_tail(engine)

    def read(_, repair_alternation=False):
        host = stored.strip()
        if shape == "internal-whitespace":
            host = host.replace("café café", "café \t\ncafé", 1)
        elif shape == "middle-block":
            host = host.replace(block, "")
        elif shape == "host-scaffolding":
            host = f"Lead\n\n{TODO}\n\n{CONTENT}"
        elif shape == "stored-scaffolding":
            host = f"Lead\n\n{CONTENT}"
        elif shape in {"contained", "short-contained"}:
            host = fragment if shape == "contained" else fragment[:-1]
            host = host.replace(" ", " \t\n " * 4)
        elif shape == "reverse-contained":
            host = f"Different lead\n\n{stored}\n\nTrailing text"
        # Different tool-call metadata isolates the text veto from replay identity mapping.
        return [{"role": "assistant", "content": host, "tool_call_id": "synthetic-host-call"}]

    host_list(engine, read)

    result = handle_lcm_command(APPLY, engine)
    remaining = ids(engine._store.connection, "messages")
    if shape == "short-contained":
        assert "deleted: 1" in result and candidate not in remaining
    else:
        assert "deleted: 0" in result and "host listed=1" in result
        assert candidate in remaining


def test_dedupe_redacts_raw_host_text_for_comparison(engine):
    engine._config.sensitive_patterns_enabled = True
    engine._config.sensitive_patterns = ["password_assignment"]
    raw = {"role": "assistant", "content": f"password=synthetic-fixture-value\n{CONTENT}"}
    protected = protect_message_for_ingest(raw, engine._config)["content"]
    assert "[LCM sensitive redaction:" in protected
    cover(engine, [append(engine, protected)])
    candidate = append(engine, protected)
    append_tail(engine)
    host_list(engine, lambda _, repair_alternation=False: [dict(raw)])

    result = handle_lcm_command(APPLY, engine)
    assert "deleted: 0" in result and "host listed=1" in result
    assert candidate in ids(engine._store.connection, "messages")


@pytest.mark.parametrize("marker", [
    "[LCM sensitive redaction: name=password_assignment; chars=10; bytes=10]",
    "[Externalized LCM ingest payload: synthetic fixture]",
])
def test_dedupe_unmatched_protected_content_is_unproven(engine, marker):
    content = f"{marker}\n{CONTENT}"
    cover(engine, [append(engine, content)])
    candidate = append(engine, content)
    append_tail(engine)

    result = handle_lcm_command(APPLY, engine)
    assert "deleted: 0" in result and "unproven (protected content)=1" in result
    assert "removable=0 unproven=1" in result
    assert candidate in ids(engine._store.connection, "messages")


def test_dedupe_non_text_content_is_kept(engine):
    kept, candidate = append(engine), append(engine)
    cover(engine, [kept])
    conn = engine._store.connection
    with engine._store.atomic():
        conn.execute("UPDATE messages SET content=CAST(content AS BLOB) WHERE store_id IN (?,?)", (kept, candidate))
    append_tail(engine)

    result = handle_lcm_command(APPLY, engine)
    assert "deleted: 0" in result and "unproven (non-text content)=1" in result
    assert candidate in ids(conn, "messages")


def test_dedupe_protects_merged_users_from_repaired_host_list(engine):
    parts = [f"Part {i}: " + "x" * 200 for i in range(40)]
    merged = "\n\n".join(parts)
    cover(engine, [append(engine, merged, role="user")])
    candidate = append(engine, merged, role="user")
    append_tail(engine)
    reads = []
    show_merged = True

    def read(_, repair_alternation=False):
        reads.append(repair_alternation)
        texts = [merged] if repair_alternation and show_merged else parts
        return [{"role": "user", "content": text} for text in texts]

    host_list(engine, read)
    result = handle_lcm_command(APPLY_USERS, engine)
    assert "deleted: 0" in result and "host listed=1" in result
    assert reads == [False, True]
    assert candidate in ids(engine._store.connection, "messages")

    show_merged = False
    assert "deleted: 1" in handle_lcm_command(APPLY_USERS, engine)
    assert candidate not in ids(engine._store.connection, "messages")


def test_dedupe_requires_host_reader_with_alternation_repair(engine):
    _, removable, _ = seed_copies(engine)
    host_list(engine, lambda _: [])

    result = handle_lcm_command(APPLY, engine)
    assert "deleted: 0" in result and "unproven (host list unavailable)=11" in result
    assert set(removable) <= ids(engine._store.connection, "messages")


@pytest.mark.parametrize("read_fails", [False, True])
def test_dedupe_host_list_unavailable(engine, read_fails):
    _, removable, _ = seed_copies(engine)
    if read_fails:
        def unreadable(_, repair_alternation=False):
            raise OSError("synthetic host read failure")
        host_list(engine, unreadable)
    else:
        engine._host_session_db = None
    for command in (SCAN, APPLY):
        result = handle_lcm_command(command, engine)
        assert "removable=0" in result and "unproven (host list unavailable)=11" in result
    assert set(removable) <= ids(engine._store.connection, "messages")


@pytest.mark.parametrize("read_fails", [False, True])
def test_dedupe_host_list_is_reread_after_backup(engine, monkeypatch, read_fails):
    _, removable, _ = seed_copies(engine)
    view = []
    reads = []

    def read(session_id, repair_alternation=False):
        reads.append((session_id, repair_alternation))
        if view and read_fails:
            raise OSError("synthetic host read failure")
        return deepcopy(view)

    host_list(engine, read)
    backup_pair = maintenance._backup_pair

    def back_up_then_list_copy(store, path):
        backup_pair(store, path)
        view.append({"role": "assistant", "content": CONTENT})

    monkeypatch.setattr(maintenance, "_backup_pair", back_up_then_list_copy)
    result = handle_lcm_command(APPLY, engine)
    assert "deleted: 0" in result and "recheck_skipped: 11" in result
    if read_fails:
        assert "recheck: unproven (host list unavailable)" in result
    assert reads == [("session", False), ("session", True), ("session", False)] + (
        [] if read_fails else [("session", True)]
    )
    assert set(removable) <= ids(engine._store.connection, "messages")


@pytest.mark.parametrize("stale_view", [False, True], ids=["shortened-view", "dangling-view"])
def test_dedupe_hidden_copy_preserves_same_lineage_cursor_and_leaf(engine, monkeypatch, caplog, stale_view):
    historical, view, kept, candidate = start_live_session(engine, monkeypatch)
    host_list(engine, lambda _, repair_alternation=False: deepcopy(view))
    store = engine._store
    conn = store.connection
    before = ids(conn, "messages") - {candidate}
    lineage = conn.execute("SELECT lineage FROM occurrence_bindings WHERE store_id=?", (candidate,)).fetchone()[0]
    assert ids(conn, "occurrence_bindings") - {kept, candidate} == before - {kept}
    assert ids(conn, "occurrence_bindings", "lineage") == {lineage}
    # Model a retained view from either before or after the host dropped the candidate.
    original = ingest_ledger.newest_view(conn, lineage, "ingest")
    entries = ingest_ledger.decode_entries(original["entries"])[1 if stale_view else 2:]
    proof_view = [historical, *view] if stale_view else view
    if stale_view:
        occurrence = conn.execute("SELECT occ_id FROM occurrence_bindings WHERE store_id=?", (candidate,)).fetchone()
        assert entries[0].ref == occurrence[0]
    salt = store.ingest_ledger_salt()
    with store.atomic():
        ingest_ledger.write_view(conn, salt, lineage, "ingest", None, entries)
    views = list(conn.execute("SELECT * FROM ingest_views ORDER BY view_seq"))
    attempt = dict(lineage=lineage, entries=entries, stamps=[str(m["timestamp"]) for m in proof_view],
                   chain=ingest_ledger.prefix_chain(salt, entries))
    assert cold_cursor(conn, attempt)[0]["k"] == len(proof_view)

    assert "deleted: 1" in handle_lcm_command(APPLY, engine)
    assert ids(conn, "messages") == before
    views_after = list(conn.execute("SELECT * FROM ingest_views ORDER BY view_seq"))
    cold = LCMEngine(config=engine._config, hermes_home=engine._hermes_home)
    try:
        cold.on_session_start("session", platform="cli", context_length=200_000, conversation_id="conversation")
        proof, counts = cold_cursor(cold._store.connection, attempt)
        assert counts["stop.missing_row"] == 0
        if stale_view:
            assert proof["k"] == 0 and counts["stop.no_anchor"] == 1
        else:
            assert proof["k"] == len(view)
        cold.ingest(deepcopy(view))
        assert ids(cold._store.connection, "messages") == before
        assert cold._store_complete_frontier() == kept < candidate
        assert cold._get_store_id_map_for_messages(view)[id(view[0])] == min(before - {kept})
        with caplog.at_level("INFO", logger="hermes_lcm.store_complete"):
            assert prepare_leaf(cold, view) == ("next", "")
        assert not any("contiguity hole" in record.message for record in caplog.records)
    finally:
        cold.shutdown()
    assert views_after == views


def test_dedupe_users_require_explicit_apply_and_minimum_age(engine):
    kept = append(engine, role="user")
    cover(engine, [kept])
    old = append(engine, role="user")
    young = engine._store.append("session", {"role": "user", "content": CONTENT}, conversation_id="conversation")
    append_tail(engine)
    conn = engine._store.connection
    assert "needs `apply users`=1" in handle_lcm_command(SCAN, engine)
    assert "deleted: 0" in handle_lcm_command(APPLY, engine)
    assert {kept, old, young} <= ids(conn, "messages")
    assert "deleted: 1" in handle_lcm_command(APPLY_USERS, engine)
    assert {kept, young} <= ids(conn, "messages") and old not in ids(conn, "messages")


@pytest.mark.parametrize("enabled_command", [APPLY, APPLY_USERS])
def test_dedupe_flag_defaults_off_and_enables_both_forms(engine, monkeypatch, enabled_command):
    monkeypatch.delenv("LCM_DOCTOR_DEDUPE_APPLY_ENABLED", raising=False)
    _, removable, _ = seed_copies(engine)
    engine._config = LCMConfig.from_env()
    engine._config.fresh_tail_count = 2
    assert "removable=11" in handle_lcm_command(SCAN, engine)
    for command in (APPLY, APPLY_USERS):
        result = handle_lcm_command(command, engine)
        assert "LCM_DOCTOR_DEDUPE_APPLY_ENABLED" in result and "deleted: 0" in result
    assert set(removable) <= ids(engine._store.connection, "messages")
    assert not engine.backup_dir().exists()
    monkeypatch.setenv("LCM_DOCTOR_DEDUPE_APPLY_ENABLED", "true")
    engine._config = LCMConfig.from_env()
    engine._config.fresh_tail_count = 2
    assert "deleted: 11" in handle_lcm_command(enabled_command, engine)


@pytest.mark.parametrize(("command", "role"), [(APPLY, "assistant"), (APPLY_USERS, "user")])
def test_dedupe_only_applies_to_the_bound_session(engine, command, role):
    kept = append(engine, role=role)
    cover(engine, [kept])
    other = append(engine, session="other-session", role=role)
    for i in range(2):
        append(engine, f"other session tail {i}", session="other-session")
    hidden_content = "separate text " * 620
    cover(engine, [append(engine, hidden_content)])
    hidden = append(engine, hidden_content)
    append_tail(engine)
    reads = []

    def read(session, repair_alternation=False):
        reads.append(session)
        return []

    host_list(engine, read)
    assert "other session (not handled)=1" in handle_lcm_command(SCAN, engine)
    result = handle_lcm_command(command, engine)
    assert "other session (not handled)=1" in result and "deleted: 1" in result
    assert reads == ["session"] * 6
    remaining = ids(engine._store.connection, "messages")
    assert {other, kept} <= remaining and hidden not in remaining


def test_dedupe_keeps_rows_with_live_assertion_sources(engine):
    _, removable, _ = seed_copies(engine)
    conn = engine._store.connection
    conn.execute(
        "CREATE TABLE IF NOT EXISTS lcm_assertion_sources (source_store_id INTEGER NOT NULL, "
        "invalidated_at REAL, invalidation_reason TEXT)"
    )
    with engine._store.atomic():
        conn.execute("INSERT INTO lcm_assertion_sources(source_store_id) VALUES (?)", (removable[0],))
        conn.execute(
            "INSERT INTO lcm_assertion_sources(source_store_id, invalidated_at, invalidation_reason) "
            "VALUES (?, 1.0, 'superseded')", (removable[1],))
    text = handle_lcm_command(SCAN, engine)
    assert "live assertion source=1" in text and "removable=10" in text
    assert "deleted: 10" in handle_lcm_command(APPLY, engine)
    assert removable[0] in ids(conn, "messages") and removable[1] not in ids(conn, "messages")


def test_dedupe_keeps_entity_and_note_citations(engine):
    kept, removable, _ = seed_copies(engine)
    entities = EntityStore(engine._store.db_path, profile_key="synthetic")
    try:
        entity = entities.create_entity("Synthetic entity")
        entities.add_alias(entity, "Synthetic alias", source_store_ids=[removable[0]])
        with entities.transaction():
            entities.add_key("synthetic key", entity, "alias_match", [removable[1]])
        note = entities.add_note("Synthetic note", cites=[removable[2]])
        entities.add_note("Synthetic retraction", kind="retraction", cites=[kept[0]],
                          retracts_store_ids=[removable[3]])
        assert "deleted: 7" in handle_lcm_command(APPLY, engine)
        assert set(removable[:4]) <= ids(engine._store.connection, "messages")
        assert entities.get_note(note) is not None
    finally:
        entities.close()
