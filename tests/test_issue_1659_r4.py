"""r4: resumed cycles and policy overrides cannot publish false completion."""
import sqlite3
import time

import pytest

from hermes_lcm import command
from tests.test_issue_1652_catchup_disclosure import (
    backfill, engine as engine, recall, record,
)
from tests.test_issue_1659_inflight_cycle import seed_filtered_inflight


def test_row_inserted_before_resumed_cursor_restarts_cycle(engine, monkeypatch):
    identity, ordered = seed_filtered_inflight(engine)
    arrival = ordered[0]
    # Reserve a source whose in-flight row will arrive only after page one.
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("DELETE FROM lcm_embedding_backfill_inflight WHERE embedded_id=? "
                     "AND identity_hash=?", (arrival, identity))
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a, **kw: (0, [], {}))
    backfill(engine, "chunk")
    cursor = record(engine, "chunk")["inflight_cursor"]
    assert arrival < cursor
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE messages SET role='user' WHERE store_id=?",
                     (int(arrival.split(":")[0]),))
        conn.execute("INSERT INTO lcm_embedding_backfill_inflight "
                     "(embedded_id,identity_hash,lease_id,generation,claimed_at,state,updated_at) "
                     "VALUES(?,?,'expired',1,?,'uncertain',?)",
                     (arrival, identity, time.time() - 3600, time.time()))
    for _ in range(2):
        backfill(engine, "chunk")
    state = record(engine, "chunk")
    assert state["inflight_cursor"] == "", state
    assert state["remaining"], "new actionable row behind cursor was certified complete"
    assert recall(engine, "chunk")["degraded"] is True
    # The restarted cycle must actually visit the newly inserted row.
    backfill(engine, "chunk")
    assert record(engine, "chunk")["operator_retry"] > 0


def test_earlier_uncertainty_survives_filtered_suffix_and_completion(engine, monkeypatch):
    seed_filtered_inflight(engine, actionable=True)
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a, **kw: (0, [], {}))
    results = [backfill(engine, "chunk") for _ in range(3)]
    state = record(engine, "chunk")
    assert state["inflight_cursor"] == "", state
    assert results[0]["uncertain"] == 1
    assert state["operator_retry"] == 1, state
    assert all(result["uncertain"] == 1 for result in results), results
    assert all(result["status"] != "complete" for result in results), results


@pytest.mark.parametrize("configured,override,role,repetitions", [
    ("conversational", "full", "user", 14000),
    ("full", "heads", "tool", 1200),
])
def test_configured_verification_rechunks_override_cache(
    engine, configured, override, role, repetitions,
):
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    engine._config.large_output_externalization_enabled = False
    engine._config.embedding_content_policy = configured
    backfill(engine, "chunk", limit=100)
    text = "cobalt sentence. " * repetitions
    sid = engine._store.append("override", {"role": role, "content": text})
    expected = {c.chunk_id for c in command.chunk_message(sid, role, text, policy=configured)}
    selected = {c.chunk_id for c in command.chunk_message(sid, role, text, policy=override)}
    assert expected - selected, "fixture must have configured-policy windows omitted by override"
    result = command._chunk_backfill_run(
        engine._config, engine._store.db_path, apply=True, limit=100,
        retry_uncertain=False, policy=override,
    )
    assert result["error"] is None, result
    with sqlite3.connect(engine._store.db_path) as conn:
        published = {row[0] for row in conn.execute("SELECT chunk_id FROM lcm_chunk_meta")}
    assert selected <= published
    assert expected - published
    assert record(engine, "chunk")["remaining"] > 0
    assert recall(engine, "chunk")["degraded"] is True
