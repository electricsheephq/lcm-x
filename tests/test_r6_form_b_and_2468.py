"""Form (b) (r6 ACP session): class A drop: class A drop at turn end, then a later ingest of a longer view.

Shape: (1) a compacted-style history is stored; (2) turn start: new prompt P60 at its own stamp; (3) turn end: the host
moved P60 into the in-flight prompt P14's slot on P14's stamp and the call row took P60's old index (class A); (4) the
engine binds again (a new engine after a restart, or a rebind after another session ran) and the host sends its FULL
history in true order (P14 and P60 on their own stamps, the tool turn, its reply) plus a new prompt P70.
Strict: the drop turn stores the call row once; the later ingest stores only P70 (no copy of the tool result or of any
user row); nothing is lost. Synthetic strings only.
"""
from __future__ import annotations

import logging
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig  # noqa: E402
from hermes_lcm.engine import LCMEngine  # noqa: E402

SESSION = "s-formb"
OTHER = "s-formb-other"
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path, session=SESSION) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(session, platform="acp", context_length=1_000_000)
    return engine


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
    import json
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    ids = ",".join(sorted(str(c.get("id")) for c in calls))
    return (row.get("role"), str(row.get("content") or "").strip(), row.get("tool_call_id") or "", ids)


def _stored(engine, session=SESSION):
    return Counter(_key(r) for r in engine._store.get_session_messages(session) if r.get("role") != "system")


def _views():
    history = [_u("P1 first prompt", 100), _a("A1 first reply", 101),
               _u("P2 second prompt", 110), _a("A2 second reply", 111),
               _u("P14 in-flight prompt", 120), _a("A14 reply", 121),
               _call("call_old", 130), _result("call_old", "old file text", 131), _a("A15 after the old call", 132)]
    slot = 4
    turn_start = [*history, _u("P60 new prompt", 200)]  # own stamp first (r6: 336 on 1934's own stamp)
    turn_end = list(history)
    turn_end[slot] = _u("P60 new prompt", 120)  # moved into P14's slot on P14's stamp (r6: 337 on 1932's stamp)
    tool_turn = [_call("call_new", 201), _result("call_new", "new file text", 202), _a("A60 after the new call", 203)]
    turn_end += tool_turn
    # The longer view: the host's full history in true order, then a new prompt (r6: incoming=55, 1940).
    longer = [*history, _u("P60 new prompt", 200), *tool_turn, _u("P70 later prompt", 300)]
    return history, turn_start, turn_end, longer, tool_turn


def _run(tmp_path, mode, caplog):
    history, turn_start, turn_end, longer, (call, result, reply) = _views()
    engine = _engine(tmp_path)
    out = {}
    try:
        engine.ingest(history)
        engine.ingest(turn_start)
        before = _stored(engine)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.ingest(turn_end)
        out["audit_fired"] = any("before the cursor" in r.getMessage() for r in caplog.records)
        mid = _stored(engine)
        added = mid - before
        out["drop_turn"] = {"call": added[_key(call)], "result": added[_key(result)], "reply": added[_key(reply)]}
        out["drop_turn_cursor"] = engine._ingest_cursor
        if mode == "restart":
            engine.shutdown()
            engine = _engine(tmp_path)
        else:  # another session runs in the same engine, then this one is bound again
            engine.on_session_start(OTHER, platform="acp", context_length=1_000_000)
            engine.ingest([_u("O1 other prompt", 250), _a("O1 other reply", 251)])
            engine.on_session_start(SESSION, platform="acp", context_length=1_000_000)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
            engine.ingest(longer)
        out["later_lines"] = [m.split("session=")[0] + " ".join(t for t in m.split() if t.startswith(("cursor=", "incoming=", "reason=")))
                              for m in (r.getMessage() for r in caplog.records)
                              if "identity-anchor recognised" in m or "reconciled ingest cursor" in m]
        after = _stored(engine)
        later = after - mid
        users = {_key(r) for r in longer if r["role"] == "user"} - {_key(longer[-1])}
        out["later"] = {"call": later[_key(call)], "result": later[_key(result)],
                        "user_copies": sum(later[k] for k in users), "new_prompt": later[_key(longer[-1])],
                        "other": sum(v for k, v in later.items()
                                     if k not in users | {_key(call), _key(result), _key(longer[-1])})}
        out["final_call_rows"] = after[_key(call)]
        out["final_result_rows"] = after[_key(result)]
        ids = [r["store_id"] for r in engine._store.get_session_messages(SESSION) if _key(r) == _key(result)]
        out["result_ids"] = ids
        out["result_relations"] = {i: [tuple(x) for x in engine._store._conn.execute(
            "SELECT kind, related_store_id FROM message_relations WHERE store_id=? OR related_store_id=?", (i, i))]
            for i in ids}
        out["lost"] = {k: v for k, v in (Counter({_key(r): 1 for r in longer}) - after).items()}
        return out
    finally:
        engine.shutdown()


@pytest.mark.parametrize("mode", ["restart", "rebind"])
def test_form_b_drop_then_full_history_stores_each_row_once(tmp_path, mode, caplog):
    out = _run(tmp_path, mode, caplog)
    assert out["audit_fired"], "setup: the prefix audit must fire on the drop turn"
    assert out["drop_turn"]["call"] == 1, out
    assert out["later"]["result"] == 0 and out["later"]["user_copies"] == 0, out
    assert out["later"]["new_prompt"] == 1 and out["later"]["call"] == 0 and out["later"]["other"] == 0, out
    assert out["final_call_rows"] == 1 and out["final_result_rows"] == 1 and not out["lost"], out


def _presence_window_exceeded(monkeypatch, engine):
    from hermes_lcm import identity_anchor
    monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1)


def _presence_read_fails(monkeypatch, engine):
    def boom(*_a, **_k):
        raise OSError("injected presence read failure")
    monkeypatch.setattr(engine, "_rotation_predecessor_session_ids", boom)


@pytest.mark.xfail(strict=True, reason="#2468: when a call row was not stored at its turn (past the presence window or "
                   "after a failed presence read), a later full-history ingest stores the call and a second, unlinked "
                   "copy of its tool result")
@pytest.mark.parametrize("cause", ["window", "read_failure"])
def test_2468_a_late_call_does_not_store_an_unlinked_second_result(tmp_path, cause, monkeypatch, caplog):
    history, turn_start, turn_end, longer, (call, result, reply) = _views()
    engine = _engine(tmp_path)
    try:
        engine.ingest(history)
        engine.ingest(turn_start)
        with monkeypatch.context() as patch:
            (_presence_window_exceeded if cause == "window" else _presence_read_fails)(patch, engine)
            engine.ingest(turn_end)
        if _stored(engine)[_key(call)] != 0:
            pytest.fail("setup: the call row must stay unstored at the drop turn")
        engine.shutdown()
        engine = _engine(tmp_path)
        engine.ingest(longer)
        assert _stored(engine)[_key(call)] == 1, "setup: the later ingest stores the call once"
        ids = [r["store_id"] for r in engine._store.get_session_messages(SESSION) if _key(r) == _key(result)]
        linked = {int(a) for a, in engine._store._conn.execute(
            "SELECT store_id FROM message_relations WHERE kind = 'possible_replay_of' AND related_store_id = ?",
            (ids[0],))} if ids else set()
        assert len(ids) == 1 or set(ids[1:]) <= linked, ids
    finally:
        engine.shutdown()
