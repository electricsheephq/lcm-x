"""#2466 review round 3 (T1, T3): a stored call's result below the cursor is stored; a result apart from its call never leaves the call alone.

Adapted from the release manager's verification probes."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-r3-split"
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path, **overrides):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False, **overrides)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _u(text, ts=None):
    row = {"role": "user", "content": text + PAD}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _a(text, ts=None):
    row = {"role": "assistant", "content": text + PAD}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _call(ids, ts=None):
    ids = [ids] if isinstance(ids, str) else ids
    row = {"role": "assistant", "content": "", "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": i})}}
        for i in ids]}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _res(call_id, text, ts=None):
    row = {"role": "tool", "content": text, "tool_call_id": call_id, "tool_name": "read_file"}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _key(row):
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    ids = ",".join(sorted(str(c.get("id")) for c in calls))
    return (row.get("role"), str(row.get("content") or "").strip()[:30], row.get("tool_call_id") or "", ids)


def _stored(engine):
    return [r for r in engine._store.get_session_messages(SESSION) if r.get("role") != "system"]


def _pairs(engine):
    rows = _stored(engine)
    call_ids = Counter()
    result_ids = Counter()
    for r in rows:
        calls = r.get("tool_calls") or []
        calls = json.loads(calls) if isinstance(calls, str) else calls
        for c in calls:
            call_ids[str(c.get("id"))] += 1
        if r.get("role") == "tool":
            result_ids[str(r.get("tool_call_id"))] += 1
    return call_ids, result_ids


def _history():
    return [_u("P1 first prompt", 100), _a("A1 first reply", 101),
            _u("P2 second prompt", 110), _a("A2 second reply", 111),
            _u("P14 in-flight prompt", 120), _a("A14 reply", 121),
            _call("call_old", 130), _res("call_old", "old file text", 131), _a("A15 after the old call", 132)]


SLOT = 4


def test_nonadjacent_result_never_leaves_its_new_call_alone(tmp_path):
    """The host moved a prompt into P14's slot (class A shift). The new turn's call and its result are both before
    the cursor, but a row sits between them (here a steer-like user row). Correct: call_new stored iff its result is."""
    engine = _engine(tmp_path)
    try:
        history = _history()
        engine.ingest(history)
        turn_start = [*history, _u("P60 new prompt", 200), _a("interim one", 199.5), _a("interim two", 199.6)]
        engine.ingest(turn_start)
        before = Counter(_key(r) for r in _stored(engine))
        turn_end = list(history)
        turn_end[SLOT] = _u("P60 new prompt", 120)
        turn_end += [_call("call_new", 201), _u("steer: be brief"), _res("call_new", "new file text", 202),
                     _a("A60 after the new call", 203)]
        cursor = len(turn_start)
        assert turn_end.index(next(r for r in turn_end if r.get("tool_call_id") == "call_new")) < cursor
        engine.ingest(turn_end)
        after = Counter(_key(r) for r in _stored(engine))
        added = after - before
        print("ADDED", dict(added))
        calls, results = _pairs(engine)
        print("CALLS", dict(calls), "RESULTS", dict(results))
        assert calls["call_new"] == results["call_new"], ("split pair", dict(calls), dict(results))
    finally:
        engine.shutdown()


def test_nonadjacent_presence_unit(tmp_path):
    """Same rule on the presence function directly: [call c1, user X, result c1], nothing stored."""
    engine = _engine(tmp_path)
    try:
        view = [_call("c1", 1), _u("X between", 2), _res("c1", "r", 3)]
        found = engine._identity_anchor_unaccounted(
            view, view, len(view), 0, lambda i: engine._message_replay_identity(view[i]), {}, True,
            set(), {"replayed": set()}, {}, (), [])
        print("FOUND", found)
        assert (0 in found) == (2 in found), found
    finally:
        engine.shutdown()


@pytest.mark.parametrize("same_object", [False, True], ids=["copy", "same-object"])
@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "unstamped"])
def test_dangling_call_stored_then_its_result_before_the_cursor_is_stored(tmp_path, stamped, same_object):
    """Turn start stored the prompt and a dangling call; the host then moved the prompt into P14's slot, so at turn
    end the call's result is before the cursor. The call is stored, so its (class A) result must be stored once."""
    engine = _engine(tmp_path)
    try:
        history = _history()
        engine.ingest(history)
        call = _call("call_new", 201 if stamped else None)
        turn_start = [*history, _u("P60 new prompt", 200), call]
        engine.ingest(turn_start)
        before = Counter(_key(r) for r in _stored(engine))
        assert before[_key(call)] == 1
        turn_end = list(history)
        turn_end[SLOT] = _u("P60 new prompt", 120)
        result = _res("call_new", "new file text", 202 if stamped else None)
        turn_end += [call if same_object else dict(call), result, _a("A60 after the new call", 203)]
        assert turn_end.index(result) < len(turn_start)
        engine.ingest(turn_end)
        after = Counter(_key(r) for r in _stored(engine))
        print("ADDED", dict(after - before))
        assert after[_key(call)] == 1
        assert after[_key(result)] == 1, ("result of a stored call lost", dict(after - before))
    finally:
        engine.shutdown()
