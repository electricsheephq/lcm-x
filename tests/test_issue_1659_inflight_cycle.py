"""r3: bounded in-flight pages must finish a conservative, ordered cycle."""
import json
import math
import sqlite3
import time

import pytest

from hermes_lcm import command
from tests.test_issue_1652_catchup_disclosure import (
    backfill, engine as engine, recall, record,
)


def seed_filtered_inflight(engine, count=600, actionable=False):
    engine._config.ollama_base_url = "http://192.0.2.1:11434"
    backfill(engine, "chunk", limit=100)
    ids = engine._store.append_batch("filtered", [
        {"role": "tool", "content": "cobalt filtered tool result"} for _ in range(count)
    ])
    with sqlite3.connect(engine._store.db_path) as conn:
        identity = conn.execute("SELECT identity_hash FROM lcm_embedding_profile "
                                "WHERE active=1 AND task='chunk'").fetchone()[0]
        ordered = sorted(f"{sid}:0" for sid in ids)
        if actionable:
            conn.execute("UPDATE messages SET role='user' WHERE store_id=?",
                         (int(ordered[0].split(":")[0]),))
        conn.executemany("INSERT INTO lcm_embedding_backfill_inflight "
                         "(embedded_id,identity_hash,lease_id,generation,claimed_at,state,updated_at) "
                         "VALUES(?,?,'expired',1,?,'uncertain',0)",
                         [(cid, identity, time.time() - 3600) for cid in ordered])
    return identity, ordered


@pytest.mark.parametrize("actionable", [False, True])
def test_resumed_cycle_completes_without_losing_earlier_actionable_rows(engine, monkeypatch, actionable):
    identity, ordered = seed_filtered_inflight(engine, actionable=actionable)
    # Isolate in-flight accounting from discovery, as in the existing count tests.
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a, **kw: (0, [], {}))
    states = []
    for _ in range(math.ceil(len(ordered) / 256) + 1):
        result = backfill(engine, "chunk")
        assert result["error"] is None, result
        state = record(engine, "chunk")
        states.append(state)
        if not state.get("inflight_cursor"):
            break
        assert state["remaining"], states
    assert states[0]["inflight_cursor"] == ordered[255]
    assert states[1]["inflight_cursor"] == ordered[511]
    assert states[-1]["inflight_cursor"] == "", states
    assert bool(states[-1]["remaining"]) is actionable, states
    if actionable:
        assert states[0].get("inflight_actionable_seen") is True, states
        assert states[1].get("inflight_actionable_seen") is True, states
    assert states[-1].get("inflight_actionable_seen") is False, states
    assert recall(engine, "chunk")["degraded"] is actionable
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT embedded_id FROM lcm_embedding_backfill_inflight "
                            "WHERE identity_hash=? ORDER BY embedded_id", (identity,)).fetchall() == [
                                (cid,) for cid in ordered]


def inflight_query(engine, monkeypatch):
    queries = []
    original = command._embedding_read_connection

    class Observed:
        def __init__(self, conn):
            self.conn = conn

        def execute(self, sql, parameters=()):
            if sql.startswith("SELECT embedded_id, state, lease_id, generation FROM"):
                queries.append((sql, parameters))
            return self.conn.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.conn, name)

    monkeypatch.setattr(command, "_embedding_read_connection", lambda *a: Observed(original(*a)))
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a, **kw: (0, [], {}))
    backfill(engine, "chunk")
    assert len(queries) == 1
    return queries[0]


def test_capped_inflight_query_uses_primary_key_without_sort(engine, monkeypatch):
    seed_filtered_inflight(engine)
    sql, parameters = inflight_query(engine, monkeypatch)
    with sqlite3.connect(engine._store.db_path) as conn:
        plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, parameters)]
    assert not any("TEMP B-TREE" in detail for detail in plan), plan
    assert any("sqlite_autoindex_lcm_embedding_backfill_inflight_1" in detail
               and "embedded_id>?" in detail for detail in plan), plan


def test_deadline_cut_preserves_unconsumed_cursor_and_cycle_flag(engine, monkeypatch):
    identity, ordered = seed_filtered_inflight(engine, actionable=True)
    monkeypatch.setattr(command, "_chunk_pending_rows", lambda *a, **kw: (0, [], {}))
    backfill(engine, "chunk")
    saved = record(engine, "chunk")
    state = {}
    remaining, _, _ = command._chunk_backfill_remaining(
        engine._store.db_path, identity, "conversational", 0, 0,
        deadline=0, scan_state=state,
    )
    assert remaining
    assert state["inflight_cursor"] == ordered[255]
    assert state.get("inflight_actionable_seen") is True
    command._record_embedding_catchup(engine._store.db_path, "chunk", identity,
                                      remaining, scan_state=state)
    persisted = record(engine, "chunk")
    assert persisted["inflight_cursor"] == saved["inflight_cursor"]
    assert json.loads(json.dumps(persisted))["inflight_actionable_seen"] is True
