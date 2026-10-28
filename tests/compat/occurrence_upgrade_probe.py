"""Probe 4f: REAL upgrade direction. PHASE=write on the BASE tree (old writer) builds a carrier-head compaction
(#1116 Host, tail 8) that the host adopts. PHASE=read on BASE or HEAD (each on an identical copy at the SAME path)
cold-binds, reports whether the old-writer snapshot attaches, then ingests the adopted list with the retained
user rows re-stamped (a new host occurrence, as in the matrix 'retry') plus one new turn; dumps stored rows.
"""
import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine
import tests.test_issue_1116_stub_first_exit_commit_proof as sf

HOME, OUT, PHASE = Path(os.environ["P4F_HOME"]), Path(os.environ["P4F_OUT"]), os.environ["P4F_PHASE"]
HEAD = os.environ.get("P4F_HEAD", "carrier")
FLOW = os.environ.get("P4F_FLOW", "commit")  # "drive": the #1116 (f) flow, then over-threshold passes, adopted


@pytest.fixture(autouse=True)
def stub(monkeypatch, caplog):
    calls = [0]

    def summarize(**kwargs):
        calls[0] += 1
        return f"Summary {calls[0]}: " + "lorem ipsum " * 200 + "\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    caplog.set_level(logging.INFO, logger="hermes_lcm")


def test_phase(caplog):
    if PHASE == "write":
        HOME.mkdir(parents=True)
        host = sf.Host(HOME, sf.TAIL_COUNT[HEAD])
        try:
            sf.committed_session(host, HEAD)
            if FLOW == "drive":
                sf.grow_over_threshold(host)
                sf.stub_first_result(host, caplog)  # returned, never adopted
                host.clock += 0.5
                host.live.append({"role": "user", "content": "next turn: " + host.words(25), "timestamp": host.clock})
                host.engine.ingest(deepcopy(host.live))
                host.drive(caplog)
            state = {"live": host.live, "clock": host.clock, "n_turns": host.n_turns}
            (HOME / "host-state.json").write_text(json.dumps(state))
        finally:
            host.engine.shutdown()
        return
    state = json.loads((HOME / "host-state.json").read_text())
    if os.environ.get("P4F_ADDLEN") == "1":  # control: the same valid snapshot in the NEW writer's format
        import sqlite3
        conn = sqlite3.connect(HOME / "lcm.db")
        key = f"host_occurrence_snapshot:{sf.SID}"
        snap = json.loads(conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0])
        snap["output_length"] = len(snap["output_metadata"])
        conn.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps(snap, sort_keys=True), key))
        conn.commit()
        conn.close()
    restamp = os.environ.get("P4F_RESTAMP", "all")
    host = sf.Host.__new__(sf.Host)
    host.tmp_path, host.tail_count = HOME, sf.TAIL_COUNT[HEAD]
    host.rng, host.clock, host.n_turns = random.Random(4242), state["clock"], state["n_turns"]
    host.live, host.sid = state["live"], sf.SID
    host.engine = host._bind()
    try:
        payload = host.engine._durable_commit_proof_payload()
        rec = {"proof": payload is not None, "attached": "output_metadata" in (payload or {})}
        snap = host.engine._store.read_metadata_json(host.engine._replay_snapshot_metadata_key("host_occurrence_snapshot"))
        rec["snapshot_keys"] = sorted(snap) if isinstance(snap, dict) else None
        before = host.engine._store.get_session_messages(sf.SID)
        live = deepcopy(host.live)
        retried = []
        users = [row for row in live if row.get("role") == "user" and "timestamp" in row
                 and not str(row["content"]).lstrip().startswith("[")]
        chosen = {"all": users, "last": users[-1:], "none": []}[restamp]
        for row in chosen:  # a retained real user row re-delivered with a new host stamp (a new occurrence)
            if True:
                host.clock += 0.5
                row["timestamp"] = host.clock
                retried.append(row["content"][:30])
        host.live = live
        host.turn()  # appends one new turn and ingests the whole list
        after = host.engine._store.get_session_messages(sf.SID)
        rec.update(retried=retried, rows_before=len(before), rows_after=len(after),
                   added=[(r["role"], r["content"][:30]) for r in after[len(before):]],
                   ingest_failures=host.engine._ingest_failure_count)
        OUT.write_text(json.dumps(rec, indent=1))
    finally:
        host.engine.shutdown()
