"""Topic-9 loss fix, ruling T1-486 cells (F2, F3, F8), the I2 limit and the rewind cell. Expectations fixed before
any run: every fresh call and result row below the cursor is stored exactly once; a rewritten row's new bytes are
stored; with no valid witness (F8) unproven rows fail toward storing."""

from __future__ import annotations

from collections import Counter

import pytest

from hermes_lcm import identity_anchor
from tests.test_anchor_unstored_before_cursor import (  # noqa: F401 (autouse fixture)
    PAD, _a, _engine, _fresh_presence_notices, _key, _result, _stored, _u,
)
from tests.test_topic9_loss_clause import _call


def _counts(engine) -> Counter:
    return Counter(_key(row) for row in _stored(engine))


def _trip(monkeypatch):
    monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)


def test_f2_an_unexplained_rewrite_past_the_trip_cursor_is_stored(tmp_path, monkeypatch):
    """F2: a fresh call at 1, an in-place user rewrite at 4, the first stamped miss at 6, then a trip."""
    engine = _engine(tmp_path)
    try:
        rows = [_u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002), _a("reply to T1", 1003),
                _u("T2 user turn", 1004), _a("reply to T2", 1005), _u("T3 user turn", 1006), _a("reply to T3", 1007)]
        engine.ingest(list(rows))
        u0, _a0, _u1, a1, u2, a2, _u3, a3 = rows
        call_new, result_new = _call("call_f2", 1500), _result("call_f2", "fresh f2 result", 1501)
        original_u2 = u2["content"]
        u2["content"] = "T2 user turn, rewritten in place by the host" + PAD  # the same dict object: an R6 rewrite
        u3_edited = _u("T3 user turn, edited", 1006)
        reply = _a("a new reply", 1600)
        _trip(monkeypatch)
        engine.ingest([u0, call_new, result_new, a1, u2, a2, u3_edited, a3, reply])
        stored = _counts(engine)
        assert stored[_key(call_new)] == 1 and stored[_key(result_new)] == 1
        assert stored[_key(u2)] == 1, "the rewrite's new bytes must be stored"
        assert stored[_key({**u2, "content": original_u2})] == 1, "the original stays stored"
        assert stored[_key(u3_edited)] == 1 and stored[_key(reply)] == 1
    finally:
        engine.shutdown()


def test_f3_a_nested_id_mutated_after_the_ingest_never_witnesses(tmp_path, monkeypatch):
    """F3: the host mutates a shared nested tool_call id after the ingest finished; the frozen witness keeps the
    id the ingest saw, so the new id is not credited."""
    engine = _engine(tmp_path)
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        call_a, result_a = _call("call_f3_a", 1003), _result("call_f3_a", "result a", 1004)
        engine.ingest([u0, a0, u1, call_a, result_a])
        call_a["tool_calls"][0]["id"] = "call_f3_b"  # shared nested state mutated after the ingest finished
        call_b = _call("call_f3_b", 1003)
        result_b = _result("call_f3_b", "result b", 1004)
        u1_edited = _u("T1 user turn, edited", 1002)
        _trip(monkeypatch)
        engine.ingest([u0, a0, u1_edited, call_b, result_b, _a("reply", 1600)])
        stored = _counts(engine)
        assert stored[_key(call_b)] == 1 and stored[_key(result_b)] == 1
    finally:
        engine.shutdown()


def test_f8_no_witness_from_another_session_rows_fail_toward_storing(tmp_path, monkeypatch):
    """F8: a witness frozen by another session never counts; unproven stored rows are stored again (a duplicate,
    never a loss)."""
    engine = _engine(tmp_path)
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        call_a, result_a = _call("call_f8", 1003), _result("call_f8", "result f8", 1004)
        a1 = _a("reply to T1", 1005)
        engine.ingest([u0, a0, u1, call_a, result_a, a1])
        session, keys = engine._trip_witness
        engine._trip_witness = ("another-session", keys)
        u1_edited = _u("T1 user turn, edited", 1002)
        fresh_call, fresh_result = _call("call_f8_new", 1500), _result("call_f8_new", "fresh", 1501)
        _trip(monkeypatch)
        engine.ingest([u0, a0, u1_edited, call_a, result_a, fresh_call, fresh_result, _a("reply", 1600)])
        stored = _counts(engine)
        assert stored[_key(fresh_call)] == 1 and stored[_key(fresh_result)] == 1
        assert stored[_key(call_a)] >= 1 and stored[_key(result_a)] >= 1  # never lost; a copy is allowed here
    finally:
        engine.shutdown()


@pytest.mark.parametrize("window", [
    pytest.param(1, id="trip"),
    pytest.param(None, id="no-trip", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="blocked segment (I2): with no trip, the presence scan's segment guard accounts an absent call below "
               "the cursor whose result, at the cursor, is already stored; the next patch's backfill restores it")),
])
def test_i2_absent_call_whose_result_is_already_stored(tmp_path, monkeypatch, window):
    """I2 probe: an absent call below the cursor whose result, at the cursor, is already stored."""
    engine = _engine(tmp_path)
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        result_x = _result("call_i2", "result stored without its call", 1004)
        engine.ingest([u0, a0, u1, result_x])
        call_x = _call("call_i2", 1003)
        if window is not None:
            _trip(monkeypatch)
        engine.ingest([u0, a0, _u("T1 user turn, edited", 1002), call_x, result_x, _a("reply", 1600)])
        assert _counts(engine)[_key(call_x)] == 1
    finally:
        engine.shutdown()


def test_cell4_a_rewrite_below_a_fresh_call_keeps_the_fresh_call(tmp_path, monkeypatch):
    """Cell 4: an in-place rewrite below a fresh call row; the fresh call and its result are stored once."""
    engine = _engine(tmp_path)
    try:
        rows = [_u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002), _a("reply to T1", 1003),
                _u("T2 user turn", 1004), _a("reply to T2", 1005)]
        engine.ingest(list(rows))
        u0, a0, u1, _a1, _u2, a2 = rows
        u1["content"] = "T1 user turn, rewritten in place" + PAD
        call_new, result_new = _call("call_c4", 1500), _result("call_c4", "fresh c4", 1501)
        engine.ingest([u0, a0, u1, call_new, result_new, a2, _a("reply", 1600)])
        stored = _counts(engine)
        assert stored[_key(call_new)] == 1 and stored[_key(result_new)] == 1
        assert stored[_key(u1)] == 1
    finally:
        engine.shutdown()
