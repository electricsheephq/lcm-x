"""Named limit (#3022, release ruling T1-511): a call that a no-trip pass left unstored behind its stored result (the
blocked segment) is witnessed from that pass's host list, so a later tripped pass keeps it unstored. The base loses it
the same way; the v0.30.2 fix keeps the base's result. The repair in #3022 restores it, and this cell then passes.
The cell is standalone so that it runs unchanged on the base and on the head (a differential pin: same counts)."""

from __future__ import annotations

from collections import Counter

import pytest

from hermes_lcm import identity_anchor
from tests.test_anchor_unstored_before_cursor import (  # noqa: F401 (autouse fixture)
    _a, _engine, _fresh_presence_notices, _key, _result, _stored, _u,
)


def _call(call_id, ts) -> dict:
    return {"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x.txt"}'}}]}


def _run_i2_then_trip(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
    result_x = _result("call_i2", "result stored without its call", 1004)
    engine.ingest([u0, a0, u1, result_x])
    call_x = _call("call_i2", 1003)
    view = [u0, a0, _u("T1 user turn, edited", 1002), call_x, result_x, _a("reply", 1600)]
    engine.ingest(list(view))  # no trip: the blocked segment leaves call_x unstored (I2)
    monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)
    engine.ingest([*view[:2], _u("T1 user turn, edited again", 1002), *view[3:], _a("next", 1700)])
    return engine, Counter(_key(row) for row in _stored(engine)), (u0, a0, u1, result_x, call_x)


def test_a_blocked_call_then_trip_copies_nothing(tmp_path, monkeypatch):
    """The part both trees keep: the stored result is never copied, and the text rows are stored once."""
    engine, stored, (u0, a0, u1, result_x, _call_x) = _run_i2_then_trip(tmp_path, monkeypatch)
    try:
        assert stored[_key(result_x)] == 1
        for row in (u0, a0, u1):
            assert stored[_key(row)] == 1
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#3022: a call the blocked segment left unstored stays unstored after a later trip "
                          "(the base does the same; the repair restores it)")
def test_a_blocked_call_stays_unstored_after_a_later_trip_named_limit(tmp_path, monkeypatch):
    engine, stored, (_u0, _a0, _u1, _result_x, call_x) = _run_i2_then_trip(tmp_path, monkeypatch)
    try:
        assert stored[_key(call_x)] == 1, "#3022: the call is restored"
    finally:
        engine.shutdown()
