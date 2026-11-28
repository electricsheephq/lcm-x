"""#2466 review round 3 (T2, T4): segments straddling the cursor and reused ids are never stored twice.

Adapted from the release manager's verification probes."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SESSION = "s-r32-probe"
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


def _call(cid, args, ts=None):
    row = {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "read_file", "arguments": args}}]}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _res(cid, text, ts=None):
    row = {"role": "tool", "content": text, "tool_call_id": cid, "tool_name": "read_file"}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _stored(engine):
    return [r for r in engine._store.get_session_messages(SESSION) if r.get("role") != "system"]


def _full(row):
    calls = row.get("tool_calls") or []
    calls = json.loads(calls) if isinstance(calls, str) else calls
    args = tuple(sorted((c.get("id"), c["function"]["arguments"]) for c in calls))
    return (row.get("role"), str(row.get("content") or "").strip()[:30], row.get("tool_call_id") or "", args)


def _dump(engine, tag):
    for r in _stored(engine):
        print(tag, r.get("store_id"), _full(r))


@pytest.mark.parametrize("stamped", [False, True], ids=["unstamped-tools", "stamped-tools"])
def test_e1_reused_id_same_output_new_pair_moved_in_front(tmp_path, stamped):
    """Stored pair call_x(A)/X. The host's list now holds a NEW pair call_x(B)/X (same output text) in front of
    it, below the cursor. Correct: A's result is never stored a second time; X is stored again only with call B."""
    ts = (lambda t: t) if stamped else (lambda t: None)
    e = _engine(tmp_path)
    try:
        u1 = _u("P1 first", 100)
        call_a, res_a = _call("call_x", '{"path": "a.txt"}', ts(101)), _res("call_x", "written", ts(102))
        tail = [_a("A1 reply", 103), _u("P2 second", 110), _a("A2 reply", 111), _u("P3 third", 120), _a("A3", 121)]
        e.ingest([u1, call_a, res_a, *tail])
        before = Counter(_full(r) for r in _stored(e))
        call_b, res_b = _call("call_x", '{"path": "b.txt"}', ts(98)), _res("call_x", "written", ts(98.5))
        v2 = [_u("P1 replaced first", 99), call_b, res_b, call_a, res_a, *tail[:-2], _u("P4 new", 200)]
        e.ingest(v2)
        _dump(e, "ROW")
        added = Counter(_full(r) for r in _stored(e)) - before
        print("ADDED", __import__("os").environ.get("PYTEST_CURRENT_TEST","").split(" ")[0].split("::")[-1], dict(added))
        assert added[_full(res_a)] <= added[_full(call_b)], ("A's result stored again without call B", dict(added))
    finally:
        e.shutdown()


def _two_call(c1, c2, ts=None):
    row = {"role": "assistant", "content": "", "tool_calls": [
        {"id": c1, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "1.txt"}'}},
        {"id": c2, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "2.txt"}'}}]}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


@pytest.mark.parametrize("call_stamp", ["restamped", "unstamped", "same"])
@pytest.mark.parametrize("tool_stamps", [True, pytest.param(False, marks=pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#2467: unstamped results of a segment straddling the cursor are stored again, as on main"))],
    ids=["results-stamped", "results-unstamped"])
def test_e2_unique_ids_straddling_segment_with_a_lost_result(tmp_path, call_stamp, tool_stamps):
    """Unique ids. The store lost result R1 earlier (main's class A loss; simulated by removing it). The host now
    shows a replaced prompt and an inserted steer row, so C and R1 sit below the cursor and R2 past it.
    Correct: R2 (stored) is never stored again; nothing is stored twice."""
    ts = (lambda t: t) if tool_stamps else (lambda t: None)
    e = _engine(tmp_path)
    try:
        H = [_u("P1 first", 100), _a("A1 reply", 101), _u("P2 second", 110), _a("A2 reply", 111)]
        u3 = _u("P3 third", 120)
        C = _two_call("c1", "c2", 121)
        R1, R2 = _res("c1", "one text", ts(122)), _res("c2", "two text", ts(123))
        e.ingest(H)
        e.ingest([*H, u3, C, R2, _a("interim note", 124)])  # R1 never stored (an earlier loss)
        before = Counter(_full(r) for r in _stored(e))
        C2 = dict(C)
        if call_stamp == "restamped":
            C2["timestamp"] = 121.5
        elif call_stamp == "unstamped":
            C2.pop("timestamp")
        steer = {"role": "user", "content": "steer: also check 3.txt" + PAD}
        v2 = [*H, _u("P3 replaced third", 119), steer, C2, R1, R2, _a("A3 final", 130)]
        e.ingest(v2)
        _dump(e, "ROW")
        after = Counter(_full(r) for r in _stored(e))
        added = after - before
        print("ADDED", __import__("os").environ.get("PYTEST_CURRENT_TEST","").split(" ")[0].split("::")[-1], dict(added))
        twice = {k: c for k, c in after.items() if c > 1}
        assert not twice, twice
    finally:
        e.shutdown()


@pytest.mark.parametrize("calls", [1, 2], ids=["one-call", "two-calls"])
def test_e3_stamped_dangling_call_whose_result_lands_below_the_cursor_is_stored_once(tmp_path, calls):
    """Default Hermes shape, unique ids, every row stamped. Turn start stored a dangling call C (stamped). At turn
    end the host moved the new prompt into the in-flight prompt's slot (on that prompt's stamp) and the list in front
    shrank by one, so C's first result sits below the cursor. Correct: every result stored exactly once."""
    e = _engine(tmp_path)
    try:
        H = [_u("P1 first", 100), _a("A1 reply", 101), _u("P2 second", 110), _a("A2 reply", 111)]
        p14, a14 = _u("P14 in-flight prompt", 120), _a("A14 reply", 121)
        p60, interim = _u("P60 new prompt", 200), _a("interim note", 201)
        C = _two_call("c1", "c2", 202) if calls == 2 else _call("c1", '{"path": "1.txt"}', 202)
        e.ingest(H)
        v1 = [*H, p14, a14, p60, interim, C]
        e.ingest(v1)
        cursor = len(v1)
        before = Counter(_full(r) for r in _stored(e))
        R1 = _res("c1", "one text", 203)
        results = [R1, _res("c2", "two text", 204)] if calls == 2 else [R1]
        v2 = [*H, _u("P60 new prompt", 120), a14, interim, C, *results, _a("A60 final", 205)]
        if v2.index(R1) >= cursor:
            pytest.fail("setup: R1 must sit below the cursor")
        e.ingest(v2)
        _dump(e, "ROW")
        after = Counter(_full(r) for r in _stored(e))
        added = after - before
        print("ADDED", __import__("os").environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0].split("::")[-1], dict(added))
        for r in results:
            assert after[_full(r)] == 1, ("result count", _full(r)[2], after[_full(r)])
    finally:
        e.shutdown()
