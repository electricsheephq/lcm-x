"""Cloud-shaped catch-up regressions; all provider traffic is synthetic."""
import json
import sqlite3
import time

from hermes_lcm import command
from tests.test_issue_1652_catchup_disclosure import (
    backfill, engine as engine, recall, record,
)
from tests.test_issue_1536_chunk_revision import (
    LONG, add_rows, counts, metadata, store as store,
)
from tests.test_issue_1543_coverage_marker import certify, marker, older_walk


def test_completion_cannot_clear_an_append_after_remaining_scan(engine, monkeypatch):
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    backfill(engine, "chunk", limit=100)
    original = command._record_embedding_catchup
    appended = []

    def append_before_write(*args, **kwargs):
        if args[1] == "chunk":
            appended.append(engine._store.append("seed", {
                "role": "user", "content": "cobalt arrived between scan and write",
            }))
            assert record(engine, "chunk")["remaining"] is True
        return original(*args, **kwargs)

    monkeypatch.setattr(command, "_record_embedding_catchup", append_before_write)
    backfill(engine, "chunk", limit=100)
    assert appended
    assert record(engine, "chunk")["remaining"], "completion overwrote the newer ingest mark"
    assert recall(engine, "chunk")["degraded"] is True


def test_manual_override_keeps_configured_full_backlog(engine):
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    engine._config.embedding_content_policy = "full"
    for role in ("system", "tool"):
        engine._store.append("seed", {"role": role, "content": "cobalt configured full " * 60})
    report = command._embedding_backfill_text(
        ["--apply", "--corpus", "chunks", "--policy", "conversational"], engine,
    )
    assert "error:" not in report, report
    assert record(engine, "chunk")["remaining"] > 0
    assert recall(engine, "chunk")["degraded"] is True
    backfill(engine, "chunk", limit=100)
    assert record(engine, "chunk")["remaining"] == 0
    assert recall(engine, "chunk")["degraded"] is False


def seed_uncertain(engine, sources=60):
    """Several thousand uncertain rows, each source a large synthetic message."""
    engine._config.large_output_externalization_enabled = False
    backfill(engine, "chunk", limit=100)
    rows = []
    with sqlite3.connect(engine._store.db_path) as conn:
        identity = conn.execute("SELECT identity_hash FROM lcm_embedding_profile "
                                "WHERE active=1 AND task='chunk'").fetchone()[0]
    for _ in range(sources):
        sid = engine._store.append("bulk", {"role": "user", "content": "cobalt sentence. " * 6000})
        chunks = command.chunk_message(sid, "user", engine._store.get(sid)["content"],
                                       policy="conversational")
        rows.extend((c.chunk_id, identity, time.time() - 3600) for c in chunks)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.executemany("INSERT INTO lcm_embedding_backfill_inflight "
                         "(embedded_id,identity_hash,lease_id,generation,claimed_at,state,updated_at) "
                         "VALUES(?,?,'expired',1,?,'uncertain',0)", rows)
    return identity, len(rows)


def test_inflight_walk_is_capped_per_pass(engine, monkeypatch):
    identity, total = seed_uncertain(engine)
    assert total >= 3000
    walked = []
    original = command._embedding_read_connection

    class Counted:
        def __init__(self, conn):
            self.conn = conn
            conn.create_function("visit_inflight", 1, lambda value: walked.append(value) or value)

        def execute(self, sql, parameters=()):
            if sql.startswith("SELECT embedded_id, state, lease_id, generation FROM"):
                sql = sql.replace("SELECT embedded_id,", "SELECT visit_inflight(embedded_id),", 1)
            return self.conn.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.conn, name)

    monkeypatch.setattr(command, "_embedding_read_connection", lambda *a: Counted(original(*a)))
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a, **kw: (0, [], {}))
    result = command._chunk_backfill_run(
        engine._config, engine._store.db_path, apply=True, limit=1,
        retry_uncertain=False, policy="conversational", maintenance_slice=1,
    )
    assert result["error"] is None, result
    assert len(walked) <= 256, f"walked {len(walked)} in-flight rows after one bounded pass"
    assert record(engine, "chunk")["remaining"] or record(engine, "chunk")["operator_retry"]
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight "
                            "WHERE identity_hash=?", (identity,)).fetchone()[0] == total


def test_cloud_low_band_advances_despite_upper_arrivals(store, monkeypatch):
    config, vectors, identity, texts = store
    config.ollama_base_url = "http://192.0.2.1:11434"
    add_rows(store, [(i, "user", LONG) for i in range(1, 9)], {6, 7, 8})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (6, 8))
    older_walk(store, {1, 2, 3, 4, 5})
    monkeypatch.setattr(command, "_CHUNK_HISTORY_SCAN_MESSAGES", 2)
    monkeypatch.setattr(command, "chunk_message", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("metadata validation must not chunk text")))
    for call in range(6):
        high = 9 + call * 2
        # A missing upper row precedes the newly published frontier on every call.
        add_rows(store, [(high, "user", LONG), (high + 1, "user", LONG)], set())
        # Publish without using the patched command chunker.
        from tests.test_issue_1536_chunk_revision import chunking
        for chunk in chunking.chunk_message(high + 1, "user", LONG, policy="full"):
            vectors.record_chunk_embedding(chunk.chunk_id, "model-a", [1., 0.],
                store_id=chunk.store_id, chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity)
        command._validate_cloud_chunk_metadata(config, config.database_path)
        proof = marker(store)
        assert all(sid in counts(store) for sid in range(proof["from"], proof["through"] + 1))
        if proof["from"] == 1:
            break
    else:
        raise AssertionError(f"low band starved for six calls: {marker(store)}")
    assert not texts
    assert json.loads(metadata(store, "chunk_gapfill_scan"))["deferred_low"]
