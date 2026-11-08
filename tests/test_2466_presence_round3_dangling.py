"""#2466 review round 3 (T1): dangling calls stored at an earlier ingest, then their results below the cursor.

Adapted from the release manager's verification probes."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-r31"
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _u(text, ts):
    return {"role": "user", "content": text + PAD, "timestamp": float(ts)}


def _a(text, ts):
    return {"role": "assistant", "content": text + PAD, "timestamp": float(ts)}


def _call(ids, ts):
    ids = [ids] if isinstance(ids, str) else ids
    return {"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": i})}} for i in ids]}


def _res(call_id, text, ts):
    return {"role": "tool", "content": text, "tool_call_id": call_id, "tool_name": "read_file", "timestamp": float(ts)}


def _key(row):
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    ids = ",".join(sorted(str(c.get("id")) for c in calls))
    return (row.get("role"), str(row.get("content") or "").strip()[:60], row.get("tool_call_id") or "", ids)


def _stored(engine):
    return [r for r in engine._store.get_session_messages(SESSION) if r.get("role") != "system"]


def _history():
    return [_u("P1 first prompt", 100), _a("A1 first reply", 101),
            _u("P2 second prompt", 110), _a("A2 second reply", 111),
            _u("P14 in-flight prompt", 120), _a("A14 reply", 121),
            _call("call_old", 130), _res("call_old", "old file text", 131), _a("A15 after the old call", 132)]


SLOT = 4


def _run(tmp_path, turn_start_tail, turn_end_tail, own_first):
    """turn-start view: history + P60 at the end (+ turn_start_tail rows stored then); turn-end view: P60 moved into
    P14's slot (the class A move) + turn_end_tail. Returns (added Counter, before, after, turn_end)."""
    engine = _engine(tmp_path)
    try:
        history = _history()
        start_stamp, moved_stamp = (200, 120) if own_first else (120, 200)
        turn_start = [*history, _u("P60 new prompt", start_stamp), *turn_start_tail]
        turn_end = list(history)
        turn_end[SLOT] = _u("P60 new prompt", moved_stamp)
        turn_end += turn_end_tail
        engine.ingest(history)
        engine.ingest(turn_start)
        before = Counter(_key(r) for r in _stored(engine))
        engine.ingest(turn_end)
        after = Counter(_key(r) for r in _stored(engine))
        engine.ingest(turn_end)
        again = Counter(_key(r) for r in _stored(engine))
        return after - before, before, after, again, turn_end, len(turn_start)
    finally:
        engine.shutdown()


def _pairs_ok(after, ids):
    return {i: (sum(c for k, c in after.items() if i in k[3].split(",")), sum(c for k, c in after.items() if k[2] == i))
            for i in ids}


@pytest.mark.parametrize("own_first", [True, False], ids=["own-first", "carried-first"])
def test_dangling_call_stored_at_turn_start_result_lands_below_cursor(tmp_path, own_first):
    """Hermes persisted the dangling call at turn start (stamped 201). At turn end the class A move shifts the list,
    so the call AND its new result are both below the cursor. The stored call is matched by its stamp (R1)."""
    call, res, reply = _call("call_new", 201), _res("call_new", "new file text", 202), _a("A60 after the new call", 203)
    added, before, after, again, view, cursor = _run(tmp_path, [call], [call, res, reply], own_first)
    print("CURSOR", cursor, "RES_IDX", view.index(res), "ADDED", dict(added), "PAIRS", _pairs_ok(after, ["call_new"]))
    assert view.index(res) < cursor
    assert after[_key(call)] == 1
    assert after[_key(res)] == 1, ("tool result of a stored call lost", dict(added))
    assert after[_key(reply)] == 1
    assert again == after


@pytest.mark.parametrize("own_first", [True, False], ids=["own-first", "carried-first"])
def test_two_call_row_stored_with_first_result_second_result_below_cursor(tmp_path, own_first):
    """A two-call row (call_a, call_b) and call_a's result stored at turn start; call_b's result then lands below
    the cursor."""
    call = _call(["call_a", "call_b"], 201)
    ra, rb, reply = _res("call_a", "a text", 202), _res("call_b", "b text", 203), _a("A60 after both", 204)
    added, before, after, again, view, cursor = _run(tmp_path, [call, ra], [call, ra, rb, reply], own_first)
    print("CURSOR", cursor, "RB_IDX", view.index(rb), "ADDED", dict(added), "PAIRS", _pairs_ok(after, ["call_a", "call_b"]))
    assert view.index(rb) < cursor
    assert after[_key(call)] == 1 and after[_key(ra)] == 1
    assert after[_key(rb)] == 1, ("second result of a stored two-call row lost", dict(added))
    assert again == after


@pytest.mark.parametrize("own_first", [True, False], ids=["own-first", "carried-first"])
def test_two_moved_pairs_next_to_each_other(tmp_path, own_first):
    """Control: two new pairs below the cursor, neither stored before (V2-like with two groups)."""
    interim = [_a("interim one", 199.1), _a("interim two", 199.2), _a("interim three", 199.3)]
    c1, r1, c2, r2 = _call("call_1", 201), _res("call_1", "one", 202), _call("call_2", 203), _res("call_2", "two", 204)
    reply = _a("A60 after both pairs", 205)
    added, before, after, again, view, cursor = _run(tmp_path, interim, [c1, r1, c2, r2, reply], own_first)
    print("CURSOR", cursor, "ADDED", dict(added), "PAIRS", _pairs_ok(after, ["call_1", "call_2"]))
    for row in (c1, r1, c2, r2, reply):
        assert after[_key(row)] == 1, (_key(row), dict(added))


@pytest.mark.parametrize("own_first", [True, False], ids=["own-first", "carried-first"])
def test_stored_pair_then_second_pair_below_cursor_text_between(tmp_path, own_first):
    """Control: a stored call/result pair at turn start, a text row, then a new pair below the cursor."""
    c1, r1 = _call("call_1", 201), _res("call_1", "one", 202)
    note = _a("note between groups", 203)
    c2, r2, reply = _call("call_2", 204), _res("call_2", "two", 205), _a("A60 final", 206)
    added, before, after, again, view, cursor = _run(tmp_path, [c1, r1, _a("interim", 202.5)], [c1, r1, note, c2, r2, reply], own_first)
    print("CURSOR", cursor, "ADDED", dict(added), "PAIRS", _pairs_ok(after, ["call_1", "call_2"]))
    for row in (c1, r1, note, c2, r2, reply):
        assert after[_key(row)] == 1, (_key(row), dict(added))


@pytest.mark.parametrize("own_first", [True, False], ids=["own-first", "carried-first"])
@pytest.mark.parametrize("partial", [False, True], ids=["dangling-call", "one-of-two-results"])
def test_interrupted_turn_default_config_result_below_cursor(tmp_path, monkeypatch, own_first, partial):
    """Default LCMConfig (externalization on). The turn-start view is persisted by the host's stopped-turn seam
    (on_turn_complete(interrupted=True), #1623): the user interrupted while the tool ran, so the stored view ends
    in a dangling call (or a two-call row with only its first result). The turn-end view carries the class A move."""
    from agent.context_engine import ContextEngine
    monkeypatch.setattr(ContextEngine, "on_turn_complete", lambda *a, **k: None, raising=False)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    try:
        history = _history()
        start_stamp, moved_stamp = (200, 120) if own_first else (120, 200)
        if partial:
            call = _call(["call_a", "call_b"], 201)
            done, late = [_res("call_a", "a text", 202)], _res("call_b", "b text", 203)
        else:
            call, done, late = _call("call_new", 201), [], _res("call_new", "new file text", 202)
        reply = _a("A60 after the call", 204)
        turn_start = [*history, _u("P60 new prompt", start_stamp), call, *done]
        engine.ingest(history)
        engine.on_turn_complete(json.loads(json.dumps(turn_start)), interrupted=True)
        before = Counter(_key(r) for r in _stored(engine))
        if before[_key(call)] != 1:
            pytest.fail(f"setup: the stopped turn must store the call {dict(before)}")
        turn_end = list(history)
        turn_end[SLOT] = _u("P60 new prompt", moved_stamp)
        turn_end += [call, *done, late, reply]
        cursor = len(turn_start)
        assert turn_end.index(late) < cursor
        engine.ingest(turn_end)
        after = Counter(_key(r) for r in _stored(engine))
        added = after - before
        print("CURSOR", cursor, "LATE_IDX", turn_end.index(late), "ADDED", dict(added))
        assert after[_key(call)] == 1
        assert after[_key(late)] == 1, ("result of a stored call lost", dict(added))
        assert after[_key(reply)] == 1
    finally:
        engine.shutdown()
