"""Topic-9 loss fix (T1-484): route G for real (I5), the whole-interval scan (I1) and the trip discriminator (cell 3).

Expectations are fixed before any run (red-first):
- I5 and I1: every fresh call and result row below the cursor is stored exactly once; nothing earlier is lost.
- Cell 3: on a tripped window, only the fresh rows (and the genuinely edited row) are stored; unchanged stored call,
  result and unstamped-reply rows between the edit and the cursor get zero copies.
"""

from __future__ import annotations

from collections import Counter

import pytest

from hermes_lcm import identity_anchor
from tests.test_anchor_unstored_before_cursor import (  # noqa: F401 (autouse fixture)
    PAD, _a, _engine, _fresh_presence_notices, _key, _result, _stored, _u,
)


def _call(call_id, ts) -> dict:
    return {"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "x.txt"}'}}]}


class _IgnoreMe:
    """A timeout-aware ignore pattern (the optional `regex` package is not a test dependency)."""
    pattern = "IGNORE-ME"

    def search(self, text, timeout=None):
        return "IGNORE-ME" in text or None


def _spy_unaccounted(monkeypatch, engine):
    seen = []
    original = engine._identity_anchor_unaccounted

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append((list(args[-1]), result))  # (the initial missed list, the scan's return)
        return result

    monkeypatch.setattr(engine, "_identity_anchor_unaccounted", spy)
    return seen


def _counts(engine) -> Counter:
    return Counter(_key(row) for row in _stored(engine))


def test_route_g_trip_stores_the_fresh_reply_and_results(tmp_path, monkeypatch):
    """I5: an edited prompt that cannot decompose into stored rows is an initial miss; the scan trips."""
    engine = _engine(tmp_path)
    try:
        history = []
        for index in range(3):
            history += [_u(f"T{index} user turn", 1000 + index), _a(f"reply to T{index}", 1100 + index)]
        prompt_a, prompt_b = _u("A: first part", 2000), _u("B: second part", 2001)
        engine.ingest([*history, prompt_a, prompt_b])
        # The host replaces A with an edit that is not A plus B, and drops B.
        prompt_a["content"] = "A rewritten by the host: unrelated text" + PAD
        m1, r1 = _call("call_g1", 2002), _result("call_g1", "result of g1", 2003)
        m2 = _a("answer after g1", 2004)
        seen = _spy_unaccounted(monkeypatch, engine)
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)
        engine.ingest([*history, prompt_a, m1, r1, m2])
        assert seen and seen[0][0], "route G needs a nonempty initial missed list"
        assert seen[0][1] is None, "the presence scan must trip"
        stored = _counts(engine)
        for row in (prompt_a, m1, r1, m2):
            assert stored[_key(row)] == 1
        for row in history:
            assert stored[_key(row)] == 1
    finally:
        engine.shutdown()


def test_whole_interval_scan_before_the_rewind(tmp_path, monkeypatch):
    """I1: audit_from=1, a fresh call and result at 1-2, the first stamped miss at 3, then a trip."""
    engine = _engine(tmp_path)
    try:
        u0, a0 = _u("T0 user turn", 1000), _a("reply to T0", 1001)
        u1, a1 = _u("T1 user turn", 1002), _a("reply to T1", 1003)
        engine.ingest([u0, a0, u1, a1])
        fresh_call, fresh_result = _call("call_i1", 1500), _result("call_i1", "fresh result", 1501)
        edited = _u("T1 user turn, edited by the host", 1002)
        reply = _a("a new reply", 1600)
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)
        engine.ingest([u0, fresh_call, fresh_result, edited, reply])
        stored = _counts(engine)
        assert stored[_key(fresh_call)] == 1
        assert stored[_key(fresh_result)] == 1
        assert stored[_key(edited)] == 1
        assert stored[_key(reply)] == 1
        for row in (u0, a0, u1, a1):  # nothing earlier is lost
            assert stored[_key(row)] == 1
    finally:
        engine.shutdown()


def test_trip_discriminator_stores_only_fresh_rows(tmp_path, monkeypatch):
    """Cell 3: a tripped window and an early prefix edit must not copy unchanged stored rows."""
    engine = _engine(tmp_path)
    try:
        u0, a0 = _u("T0 user turn", 1000), _a("reply to T0", 1001)
        u1 = _u("T1 user turn", 1002)
        call1, result1 = _call("call_d1", 1003), _result("call_d1", "result d1", 1004)
        unstamped = {"role": "assistant", "content": "an unstamped reply" + PAD}
        u2, a2 = _u("T2 user turn", 1006), _a("reply to T2", 1007)
        stored_view = [u0, a0, u1, call1, result1, unstamped, u2, a2]
        engine.ingest(list(stored_view))
        before = _counts(engine)
        edited = _u("T1 user turn, edited by the host well below the cursor", 1002)
        u3 = _u("T3 user turn", 2000)
        call2, result2 = _call("call_d2", 2001), _result("call_d2", "result d2", 2002)
        a3 = _a("reply to T3", 2003)
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)
        engine.ingest([u0, a0, edited, call1, result1, unstamped, u2, a2, u3, call2, result2, a3])
        added = _counts(engine) - before
        assert added == Counter(_key(row) for row in (edited, u3, call2, result2, a3)), added
    finally:
        engine.shutdown()


@pytest.mark.parametrize("window", [1, None], ids=["trip", "no-trip"])
def test_a_stored_calls_unstored_result_slid_below_the_cursor_is_stored(tmp_path, monkeypatch, window):
    """RM probe: the call and only its first result are stored; a host merge then slides the second result below
    the cursor. Expected before any run: the second result is stored exactly once (a new result of a stored call)."""
    engine = _engine(tmp_path)
    try:
        history = [_u("T0 user turn", 1000), _a("reply to T0", 1001)]
        prompt_a, prompt_b = _u("A: first part", 2000), _u("B: second part", 2001)
        m1 = {"role": "assistant", "content": "Reading two files.", "timestamp": 2002.0, "tool_calls": [
            {"id": "call_p_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
            {"id": "call_p_b", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b"}'}}]}
        r_a, r_b = _result("call_p_a", "result a", 2003), _result("call_p_b", "result b", 2004)
        engine.ingest([*history, prompt_a, prompt_b, m1, r_a])  # the call and only its first result are stored
        original_a = prompt_a["content"]
        prompt_a["content"] = original_a + "\n\n" + prompt_b["content"]
        prompt_a["_merged_turn_prefix"] = original_a
        m2 = _a("answer", 2005)
        if window is not None:
            monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", window, raising=False)
        engine.ingest([*history, prompt_a, m1, r_a, r_b, m2])  # r_b now sits below the old cursor
        stored = _counts(engine)
        assert stored[_key(r_b)] == 1, "the stored call's second result must be stored once"
        for row in (*history, m1, r_a, m2):
            assert stored[_key(row)] == 1
    finally:
        engine.shutdown()


def test_trip_never_stores_an_orphan_result_of_an_ignored_call(tmp_path, monkeypatch):
    """I7: an ignored fresh call and its non-ignored result below the cursor: groups stay whole on a trip."""
    engine = _engine(tmp_path)
    engine._compiled_ignore_message_patterns = [_IgnoreMe()]  # as tests/test_ingest_protection.py does
    try:
        u0, a0 = _u("T0 user turn", 1000), _a("reply to T0", 1001)
        u1, a1 = _u("T1 user turn", 1002), _a("reply to T1", 1003)
        engine.ingest([u0, a0, u1, a1])
        ignored_call = {**_call("call_ign", 1500), "content": "IGNORE-ME"}
        its_result = _result("call_ign", "result of the ignored call", 1501)
        edited = _u("T1 user turn, edited by the host", 1002)
        reply = _a("a new reply", 1600)
        monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)
        engine.ingest([u0, ignored_call, its_result, edited, reply])
        stored = _counts(engine)
        assert stored[_key(its_result)] == 0, "a result is never stored without its call"
        assert stored[_key(ignored_call)] == 0
        assert stored[_key(edited)] == 1
        for row in (u0, a0, u1, a1):
            assert stored[_key(row)] == 1
    finally:
        engine.shutdown()
