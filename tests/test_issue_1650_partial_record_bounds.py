"""Partial-message intents stay cheap and leave room for new turns (#1650)."""
import dataclasses
import json
import sqlite3
import threading
from collections import Counter

import pytest

from hermes_lcm import command
from tests.test_issue_1014_incremental_embedding import FakeProvider
from tests.test_issue_1586_local_pacing import local_store as local_store


def _record(db):
    with sqlite3.connect(db) as conn:
        identity = conn.execute(
            "SELECT identity_hash FROM lcm_embedding_profile WHERE task='chunk' AND active=1"
        ).fetchone()[0]
        saved = conn.execute("SELECT value FROM metadata WHERE key=?",
                             (f"chunk_partial_messages:{identity}",)).fetchone()
    return identity, json.loads(saved[0]) if saved else []


def _run(config, db, **kwargs):
    return command._chunk_backfill_run(config, db, apply=True, limit=32,
                                       retry_uncertain=False, policy="", **kwargs)


@pytest.mark.parametrize("slice_s", [0.0, 1.0])
def test_interrupted_large_manual_record_cost(local_store, monkeypatch, slice_s):
    # Adapted from the independent 3,000-message cost reproduction.
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=32)
    db = engine._store.db_path
    for i in range(3000):
        engine._store.append("seed", {"role": "user", "content": f"message {i} words " * 8})
    stop = threading.Event()

    class StopAfterFirst(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            yield from super().embed_document_batches(texts, before_dispatch=before_dispatch)
            stop.set()

    real = command.chunk_message
    calls = Counter()

    def counting(store_id, *args, **kwargs):
        calls[store_id] += 1
        return real(store_id, *args, **kwargs)

    monkeypatch.setattr(command, "chunk_message", counting)
    monkeypatch.setattr(command, "resolve_provider", lambda c, **k: StopAfterFirst(c.embedding_provider, c.embedding_model))
    manual = command._chunk_backfill_run(config, db, apply=True, limit=100000,
                                         retry_uncertain=False, policy="", stop_event=stop)
    assert manual["embedded"] == 32 and manual["stop_reason"] == "process_exit"
    assert len(calls) == 3000 and max(calls.values()) == 1
    _, saved_after_manual = _record(db)
    calls.clear()
    monkeypatch.setattr(command, "resolve_provider", lambda c, **k: FakeProvider(c.embedding_provider, c.embedding_model))
    result = _run(config, db, incremental=True, maintenance_slice=slice_s)
    _, saved = _record(db)
    print(f"COST slice={slice_s} chunk_message_calls={sum(calls.values())} "
          f"manual_record={len(saved_after_manual)} final_record={len(saved)}")
    assert result["embedded"] == 32, result
    assert sum(calls.values()) <= 2 * 32 + 64
    assert saved_after_manual == saved == []  # All published messages are whole.


@pytest.mark.parametrize("slice_s", [0.0, 1.0])
@pytest.mark.parametrize("partial", [False, True])
def test_new_turn_progresses_with_400_recorded_messages(local_store, monkeypatch, slice_s, partial):
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=32)
    db = engine._store.db_path
    for i in range(400):
        engine._store.append("seed", {"role": "user", "content": f"old message {i} sentence. " * 190})
    command._chunk_backfill_run(config, db, apply=True, limit=100000 if partial else 32,
                                retry_uncertain=False, policy="")
    identity, _ = _record(db)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT store_id, role, content FROM messages").fetchall()
    if partial:
        # Model an interrupted run with 400 genuinely partial messages.
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id IN "
                         "(SELECT chunk_id FROM lcm_chunk_meta WHERE chunk_index>0)")
            conn.execute("DELETE FROM lcm_chunk_meta WHERE chunk_index>0")
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                     (f"chunk_partial_messages:{identity}", json.dumps([r[0] for r in rows])))
    engine._store.append("seed", {"role": "user", "content": "brand new turn " * 8})
    real = command.chunk_message
    calls = Counter()

    def counting(store_id, *args, **kwargs):
        calls[store_id] += 1
        return real(store_id, *args, **kwargs)

    monkeypatch.setattr(command, "chunk_message", counting)
    result = _run(config, db, incremental=True, maintenance_slice=slice_s)
    with sqlite3.connect(db) as conn:
        new = conn.execute("SELECT MAX(store_id) FROM messages").fetchone()[0]
        published = conn.execute("SELECT 1 FROM lcm_chunk_meta WHERE store_id=?", (new,)).fetchone()
    print(f"PRIORITY slice={slice_s} partial={partial} new_turn_passes={1 if published else '>1'} "
          f"chunk_message_calls={sum(calls.values())}")
    assert published, result
    if partial:
        assert all(count <= 1 for sid, count in calls.items() if sid != new), calls
        assert result["embedded"] <= 17  # At most half the limit is recorded chunks, plus the new turn.
        assert result["remaining"] > 0 and result["status"] != "complete"


@pytest.mark.parametrize("value", ["not json", "null"])
def test_malformed_partial_record_runs_and_reports(local_store, caplog, value):
    engine, _ = local_store
    db = engine._store.db_path
    _run(engine._config, db)
    identity, _ = _record(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
                     (f"chunk_partial_messages:{identity}", value))
        assert command._chunk_backlog_estimate(conn, identity, "conversational") == "0"
        assert command._chunk_backlog_estimate(conn, identity, "conversational") == "0"
    result = _run(engine._config, db, incremental=True)
    assert result["status"] == "complete" and not result["error"], result
    assert _record(db)[1] == []
    assert sum("partial-message record" in record.message for record in caplog.records) == 1
