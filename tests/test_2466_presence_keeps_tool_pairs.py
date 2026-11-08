"""Pairing probes for #2466 (read-only skeptic). Full-identity keys: role, content, call id, call arguments."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-pairing-probe"
PAD = " alpha beta gamma delta" * 40


def _engine(tmp_path, **overrides):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=False, **overrides)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    return engine


def _u(text, ts):
    return {"role": "user", "content": text + PAD, "timestamp": float(ts)}


def _a(text, ts):
    return {"role": "assistant", "content": text + PAD, "timestamp": float(ts)}


def _stored(engine):
    return [row for row in engine._store.get_session_messages(SESSION) if row.get("role") != "system"]


def _call(call_id, args, ts=None):
    row = {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": args}}]}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _res(call_id, text, ts=None):
    row = {"role": "tool", "content": text, "tool_call_id": call_id, "tool_name": "read_file"}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _full(row):
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    args = tuple(sorted((c.get("id"), c["function"]["arguments"]) for c in calls))
    return (row.get("role"), str(row.get("content") or "").strip()[:40], row.get("tool_call_id") or "", args)


def _report(engine):
    rows = _stored(engine)
    return [_full(r) for r in rows]


def test_p1_reused_id_dangling_call_accounts_new_call_but_result_is_stored(tmp_path):
    """Turn-start stored a dangling call (call_0, args A). The host replaced the turn; the model re-issued call_0
    with other arguments and its result, both now before the cursor. Expect: resB stored only if callB stored."""
    engine = _engine(tmp_path)
    try:
        history = [_u("P1 first", 100), _a("A1 reply", 101), _u("P2 second", 110), _a("A2 reply", 111)]
        u_old = _u("poll the file", 120)
        interim = _a("let me look", 121)
        call_a = _call("call_0", '{"path": "a.txt"}', 122)
        v1 = [*history, u_old, interim, call_a]
        engine.ingest(history)
        engine.ingest(v1)
        before = Counter(_report(engine))
        u_new = _u("poll the other file", 130)
        call_b = _call("call_0", '{"path": "b.txt"}', 131)
        res_b = _res("call_0", "B file text", 132)
        after_reply = _a("A after B", 133)
        v2 = [*history, u_new, call_b, res_b, after_reply]
        assert len(v2) == len(v1) + 1
        engine.ingest(v2)
        after = Counter(_report(engine))
        added = after - before
        print("ADDED", dict(added))
        assert added[_full(res_b)] == 0 or added[_full(call_b)] == 1, (
            "result of call_0 (args b.txt) stored without its own call", dict(added))
    finally:
        engine.shutdown()


def test_p5_unstamped_reused_id_pair_inserted_before_a_stored_pair(tmp_path):
    """Unstamped tool rows (host in-memory rows). A stored pair call_0/B; the host's list now holds a NEW pair
    call_0/C before it, all before the cursor. Expect: nothing stored twice; C's pair stored only together."""
    engine = _engine(tmp_path)
    try:
        u1 = _u("P1 first", 100)
        call_b, res_b = _call("call_0", '{"path": "b.txt"}'), _res("call_0", "B file text")
        tail = [_a("A1 reply", 103), _u("P2 second", 110), _a("A2 reply", 111), _u("P3 third", 120), _a("A3", 121)]
        v1 = [u1, call_b, res_b, *tail]
        engine.ingest(v1)
        before = Counter(_report(engine))
        call_c, res_c = _call("call_0", '{"path": "c.txt"}'), _res("call_0", "C file text")
        u1n = _u("P1 replaced first", 99)
        v2 = [u1n, call_c, res_c, call_b, res_b, *tail[:-2], _u("P4 new", 200)]
        engine.ingest(v2)
        after = Counter(_report(engine))
        added = after - before
        print("ADDED", dict(added))
        twice = {k: c for k, c in after.items() if c > 1}
        assert not twice, ("stored twice", twice, dict(added))
        assert (added[_full(res_c)] == 0) or added[_full(call_c)] == 1, ("result C without its call", dict(added))
    finally:
        engine.shutdown()


def test_p2_reused_id_split_into_a_two_call_row_leaves_a_call_without_its_result(tmp_path):
    """Unstamped tool rows. Stored single call call_0/A + result; the host's list now holds, in its slot, a new
    two-call row (call_0, call_1) with both results. The id-only result key lets resA account for the new call_0 result."""
    engine = _engine(tmp_path)
    try:
        u1 = _u("P1 first", 100)
        call_a, res_a = _call("call_0", '{"path": "a.txt"}'), _res("call_0", "A file text")
        tail = [_a("A1 reply", 103), _u("P2 second", 110), _a("A2 reply", 111)]
        v1 = [u1, call_a, res_a, _a("note", 102), *tail]
        engine.ingest(v1)
        before = Counter(_report(engine))
        two = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}},
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "c.txt"}'}}]}
        r0, r1 = _res("call_0", "B file text"), _res("call_1", "C file text")
        v2 = [_u("P1 replaced", 99), two, r0, r1, *tail, _u("P3 new", 200)]
        engine.ingest(v2)
        added = Counter(_report(engine)) - before
        print("ADDED", dict(added))
        assert added[_full(two)] == 0 or (added[_full(r0)] == 1 and added[_full(r1)] == 1), dict(added)
    finally:
        engine.shutdown()


def test_reused_id_pairs_are_never_split(tmp_path):
    # Astra (delta 6): stored (A, X); view (A, Y), (B, X) under one id. A new call is stored only with its own
    # results, and a new result only with its own call: Y (its call A is stored by payload) is stored; B is not,
    # since its result X is already stored (main's result for B).
    from tests.test_anchor_unstored_before_cursor import _engine, _presence, SESSION

    def call(path):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_x", "type": "function", "function": {"name": "read_file", "arguments": path}}]}

    def result(text):
        return {"role": "tool", "content": text, "tool_call_id": "call_x", "tool_name": "read_file"}

    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [call("A"), result("X")])
        assert _presence(engine, [call("A"), result("Y"), call("B"), result("X")]) == [1]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["before", "after", "before_stamped", "before_unique_id"])
def test_a_new_pair_next_to_a_stored_pair_is_stored_once_in_any_order(tmp_path, shape):
    """A new pair (reused id, or a fresh one) placed before or after a stored pair, stamped or not: each row is
    stored once, and the stored pair is never stored a second time (#2466 review, round 2 p5 order variants)."""
    engine = _engine(tmp_path)
    try:
        ts = (lambda t: t) if "stamped" in shape else (lambda t: None)
        stored_call, stored_result = _call("call_0", '{"path": "b.txt"}', ts(101)), _res("call_0", "B file text", ts(102))
        tail = [_a("A1 reply", 103), _u("P2 second", 110), _a("A2 reply", 111), _u("P3 third", 120), _a("A3", 121)]
        engine.ingest([_u("P1 first", 100), stored_call, stored_result, *tail])
        before = Counter(_report(engine))
        new_id = "call_9" if "unique" in shape else "call_0"
        new_call, new_result = _call(new_id, '{"path": "c.txt"}', ts(98)), _res(new_id, "C file text", ts(98.5))
        pair = ([new_call, new_result, stored_call, stored_result] if shape.startswith("before")
                else [stored_call, stored_result, new_call, new_result])
        engine.ingest([_u("P1 replaced first", 99), *pair, *tail[:-2], _u("P4 new", 200)])
        after = Counter(_report(engine))
        added = after - before
        assert not {key: count for key, count in after.items() if count > 1}
        assert added[_full(new_call)] == 1 and added[_full(new_result)] == 1
    finally:
        engine.shutdown()


def test_parallel_calls_in_separate_rows_keep_their_own_results(tmp_path):
    """[call A, call B, result A, result B]: each result belongs to the call row carrying its id, so all four are
    stored and no call is stored without its result (#2466 review, delta 7)."""
    engine = _engine(tmp_path)
    try:
        call_a, call_b = _call("call_a", '{"path": "a"}'), _call("call_b", '{"path": "b"}')
        view = [call_a, call_b, _res("call_a", "A text"), _res("call_b", "B text")]
        assert engine._identity_anchor_unaccounted(
            view, view, len(view), 0, lambda i: engine._message_replay_identity(view[i]), {}, True,
            set(), {"replayed": set()}, {}, (), []) == [0, 1, 2, 3]
    finally:
        engine.shutdown()
