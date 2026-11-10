"""Probe: a predecessor write that failed in this process, seen from a fresh per-agent clone (clone_for_agent /
deepcopy, as Hermes builds per AIAgent). The PR claims this falls back to main; the flag lives on one instance."""
from __future__ import annotations

import copy
import json
import logging
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 40


def _u(text, ts):
    return {"role": "user", "content": text + PAD, "timestamp": float(ts)}


def _a(text, ts):
    return {"role": "assistant", "content": text + PAD, "timestamp": float(ts)}


def _call(call_id, ts):
    return {"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "notes.txt"}'}}]}


def _result(call_id, text, ts):
    return {"role": "tool", "content": text, "tool_call_id": call_id, "tool_name": "read_file", "timestamp": float(ts)}


def _key(row):
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    ids = ",".join(sorted(str(call.get("id")) for call in calls))
    return (row.get("role"), str(row.get("content") or "").strip(), row.get("tool_call_id") or "", ids)


@pytest.mark.parametrize("statedb", ["none", "missing-child-row", "verified"])
@pytest.mark.parametrize("mode", ["same-engine", "deepcopy-clone"])
def test_failed_predecessor_write_falls_back_in_any_engine_of_the_process(tmp_path, monkeypatch, caplog, mode, statedb):
    import sqlite3
    import hermes_lcm.engine as lcm_engine

    if statedb == "missing-child-row":
        with sqlite3.connect(tmp_path / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
            conn.execute("INSERT INTO sessions VALUES ('P', NULL, 'compression')")
    if statedb == "verified":
        with sqlite3.connect(tmp_path / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT)")
            conn.executemany("INSERT INTO sessions VALUES (?,?,?)", [("P", None, "compression"), ("C", "P", None)])

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: (
        "Earlier turns.\nExpand for details about: turns", 1))
    config = LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1, database_path=str(tmp_path / "lcm.db"),
                       large_output_externalization_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))  # no state.db
    engine.on_session_start("P", platform="cli", context_length=200_000, conversation_id="conv")
    clone = None
    try:
        live = [{"role": "system", "content": "stable system prompt"}]
        for i in range(6):
            live += [_u(f"T{i} prompt", 10 * i), _a(f"R{i} reply", 10 * i + 1)]
        live += [_u("TQ tool prompt", 100), _call("call_p", 101), _result("call_p", "Pfile text", 102),
                 _a("RQ after call", 103)]
        engine.ingest(live)
        out = engine.compress(live)
        engine.on_session_end("P", list(live))
        write = engine._store.write_metadata_json

        def write_metadata(keys, *args, **kwargs):
            if "rotation_predecessor_session:C" in keys:
                raise OSError("synthetic predecessor write failure")
            return write(keys, *args, **kwargs)

        monkeypatch.setattr(engine._store, "write_metadata_json", write_metadata)
        engine.on_session_start("C", boundary_reason="compression", old_session_id="P", platform="cli",
                                context_length=200_000, conversation_id="conv")
        first = [*out, _u("N1 prompt", 500), _a("N1 reply", 501)]
        engine.ingest(first)
        runner = engine
        if mode == "deepcopy-clone":
            # Hermes deep-copies the registered engine per AIAgent (engine.__deepcopy__ -> clone_for_agent).
            clone = copy.deepcopy(engine)
            clone.on_session_start("C", platform="cli", context_length=200_000, conversation_id="conv")
            clone.ingest(first)  # resend: a replay
            runner = clone
        print("flag on runner:", getattr(runner, "_rotation_lineage_unrecorded", None))
        before = Counter(_key(r) for r in runner._store.get_session_messages("C") if r["role"] != "system")
        ins = next((i for i, m in enumerate(out) if m["role"] == "user" and "timestamp" in m), 1)
        changed = [*out[:ins], _u("X inserted prompt", 450), *out[ins:], *first[-2:],
                   _u("N2 prompt", 600), _a("N2 reply", 601)]
        caplog.clear()
        with caplog.at_level(logging.INFO):
            runner.ingest(changed)
        after = Counter(_key(r) for r in runner._store.get_session_messages("C") if r["role"] != "system")
        added = after - before
        added_roles = Counter((key[0], key[2], key[3]) for key in added.elements())
        print("added total:", sum(added.values()), "call rows:", added_roles[("assistant", "", "call_p")],
              "result rows:", added_roles[("tool", "call_p", "")])
        print("log:", [r.getMessage()[:90] for r in caplog.records if "identity-anchor" in r.getMessage()])
        assert not before - after
        # main never stores P's carried call/result in C (they are positional replays)
        assert added_roles[("assistant", "", "call_p")] == added_roles[("tool", "call_p", "")] == 0
    finally:
        if clone is not None:
            clone.shutdown()
        engine.shutdown()
