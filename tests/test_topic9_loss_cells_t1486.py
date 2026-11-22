"""Topic-9 loss fix, ruling T1-486 cells (F2, F3, F4, F6, F8), the I2 limit, the rewind cell, the trip-forced
differential, the busy retry and the cold start (trip clause line 22).

Expectations fixed before any run: every fresh call and result row below the cursor is stored exactly once; a
rewritten row's new bytes are stored; with no valid witness unproven rows fail toward storing (copies, never a
loss). Stamped stored rows are proven by the stamp pre-match, so the cells that exercise the witness hold
unstamped stored pairs, and each trip cell asserts that the presence scan tripped."""

from __future__ import annotations

import sqlite3
from collections import Counter

import pytest

from hermes_lcm import identity_anchor
from tests.test_anchor_unstored_before_cursor import (  # noqa: F401 (autouse fixture)
    PAD, _a, _engine, _fresh_presence_notices, _key, _result, _stored, _u,
)
from tests.test_topic9_loss_clause import _call, _IgnoreMe


def _counts(engine) -> Counter:
    return Counter(_key(row) for row in _stored(engine))


def _trip(monkeypatch):
    monkeypatch.setattr(identity_anchor, "_PRESENCE_WINDOW_ROWS", 1, raising=False)


def _unstamped(row):
    row.pop("timestamp", None)
    return row


def _trips(monkeypatch, engine) -> list:
    trips, real = [], engine._identity_anchor_trip_unproven

    def spy(*args, **kwargs):
        found = real(*args, **kwargs)
        trips.append(found)
        return found

    monkeypatch.setattr(engine, "_identity_anchor_trip_unproven", spy)
    return trips


def test_f2_an_unexplained_rewrite_past_the_trip_cursor_is_stored(tmp_path, monkeypatch):
    """F2: a fresh call at 1, an in-place rewrite at 4 of an unstamped user row (a steer note: the audit skips it, so
    only the R6 rewind can save it), the first stamped miss at 6, then a trip."""
    engine = _engine(tmp_path)
    try:
        reply_row = {"role": "user", "content": "a steer note" + PAD}
        rows = [_u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002), _a("reply to T1", 1003),
                reply_row, _a("reply to T2", 1005), _u("T3 user turn", 1006), _a("reply to T3", 1007)]
        engine.ingest(list(rows))
        u0, _a0, _u1, a1, _r, a2, _u3, a3 = rows
        call_new, result_new = _call("call_f2", 1001), _result("call_f2", "fresh f2 result", 1002)
        original = reply_row["content"]
        reply_row["content"] = "continue with the plan" + PAD  # the same host object, rewritten in place: R6
        u3_edited = _u("T3 user turn, edited", 1006)
        reply = _a("a new reply", 1600)
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        engine.ingest([u0, call_new, result_new, a1, reply_row, a2, u3_edited, a3, reply])
        assert trips, "setup: the presence scan tripped"
        stored = _counts(engine)
        assert stored[_key(call_new)] == 1 and stored[_key(result_new)] == 1
        assert stored[_key(reply_row)] == 1, "the rewrite's new bytes must be stored"
        assert stored[_key({**reply_row, "content": original})] == 1, "the original stays stored"
        assert stored[_key(u3_edited)] == 1 and stored[_key(reply)] == 1
    finally:
        engine.shutdown()


def test_f3_a_nested_id_mutated_after_the_ingest_never_witnesses(tmp_path, monkeypatch):
    """F3: the host mutates a shared nested tool_call id after the ingest finished. The frozen witness keeps the id
    the ingest saw, so the call under its new id is stored."""
    engine = _engine(tmp_path)
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        call = _unstamped(_call("call_f3_a", 1003))
        result_a = _unstamped(_result("call_f3_a", "result a", 1004))
        a1 = _a("reply to T1", 1005)
        engine.ingest([u0, a0, u1, call, result_a, a1])
        call["tool_calls"][0]["id"] = "call_f3_b"  # shared nested state mutated after the ingest finished
        result_b = _unstamped(_result("call_f3_b", "result b", 1004))
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        engine.ingest([u0, a0, u1, call, result_b, a1, _a("reply", 1600)])
        assert trips, "setup: the presence scan tripped"
        stored = _counts(engine)
        assert stored[_key(call)] == 1 and stored[_key(result_b)] == 1
    finally:
        engine.shutdown()


def test_f4_a_generated_placeholder_never_witnesses(tmp_path, monkeypatch):
    """F4: the host echoes LCM's placeholder for an ignored result; the writer never stores it. When the real
    result later takes its slot below the cursor, the placeholder must not witness it."""
    engine = _engine(tmp_path)
    engine._compiled_ignore_message_patterns = [_IgnoreMe()]
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        call = _unstamped(_call("call_f4", 1003))
        ignored = _unstamped(_result("call_f4", "IGNORE-ME raw output", 1004))
        a1 = _a("reply to T1", 1005)
        engine.ingest([u0, a0, u1, call, ignored, a1])
        replay = list(engine._last_active_replay_messages)
        placeholder = replay[4]
        assert str(placeholder.get("content", "")).startswith("[LCM active replay placeholder:"), placeholder
        engine.ingest([*replay, _u("T2 user turn", 1006)])  # the host echoes the active list, placeholder included
        assert not any("[LCM active replay placeholder:" in str(r["content"]) for r in _stored(engine)), \
            "setup: the placeholder is never stored"
        real = _unstamped(_result("call_f4", "the real output", 1004))
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        view = [*replay, _u("T2 user turn", 1006), _a("reply", 1600)]
        view[4] = real
        engine.ingest(view)
        assert trips, "setup: the presence scan tripped"
        assert _counts(engine)[_key(real)] == 1
    finally:
        engine.shutdown()


def test_f6_an_ambiguous_id_keeps_mains_result(tmp_path, monkeypatch):
    """F6: a fresh call reuses the id of a stored unstamped call and sits before it in the audit interval. The id is
    ambiguous, so the trip neither stores nor witnesses either occurrence (main's result): the reuse cannot spend the
    old occurrence's witness, so the old call is not copied. The unambiguous fresh pair is still stored."""
    engine = _engine(tmp_path)
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        old_call = _unstamped(_call("call_x", 1003))
        a1 = _a("reply to T1", 1005)
        engine.ingest([u0, a0, u1, old_call, a1])
        reused = _unstamped(_call("call_x", 1001))
        fresh_call, fresh_result = _call("call_f6", 1002), _result("call_f6", "fresh", 1500)
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        engine.ingest([u0, reused, fresh_call, old_call, a1, fresh_result, _a("reply", 1600)])
        assert trips, "setup: the presence scan tripped"
        assert 1 not in trips[0] and 3 not in trips[0], trips
        stored = _counts(engine)
        assert stored[_key(old_call)] == 1, "the old call is not copied"
        assert stored[_key(fresh_call)] == 1 and stored[_key(fresh_result)] == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("owner", ["own", "foreign"])
def test_f8_only_this_sessions_witness_counts(tmp_path, monkeypatch, owner):
    """F8: the witness counts only when this session froze it. With a foreign witness the unstamped stored pair
    fails toward storing (a second copy, never a loss); with its own witness it keeps its credit."""
    engine = _engine(tmp_path)
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        call = _unstamped(_call("call_f8", 1003))
        result = _unstamped(_result("call_f8", "result f8", 1004))
        a1 = _a("reply to T1", 1005)
        engine.ingest([u0, a0, u1, call, result, a1])
        if owner == "foreign":
            engine._trip_witness = ("another-session", engine._trip_witness[1])
        # The host drops two text rows and a fresh pair takes their slots, so the stored pair stays below the cursor.
        fresh_call, fresh_result = _call("call_f8_new", 1001), _result("call_f8_new", "fresh", 1002)
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        engine.ingest([u0, fresh_call, fresh_result, call, result, a1, _a("reply", 1600)])
        assert trips, "setup: the presence scan tripped"
        stored = _counts(engine)
        assert stored[_key(fresh_call)] == 1 and stored[_key(fresh_result)] == 1
        expected = 1 if owner == "own" else 2
        assert (stored[_key(call)], stored[_key(result)]) == (expected, expected)
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
        trips = _trips(monkeypatch, engine)
        if window is not None:
            _trip(monkeypatch)
        engine.ingest([u0, a0, _u("T1 user turn, edited", 1002), call_x, result_x, _a("reply", 1600)])
        if window is not None:
            assert trips, "setup: the presence scan tripped"
        counts = _counts(engine)
        assert counts[_key(result_x)] == 1, "restoring the call never stores its stored result again"
        assert counts[_key(call_x)] == 1
    finally:
        engine.shutdown()


def test_cell4_a_rewrite_below_a_fresh_call_keeps_the_fresh_call(tmp_path, monkeypatch):
    """Cell 4 (no trip): an in-place rewrite below a fresh call row; the fresh call and its result are stored once,
    and so are the rewrite's new bytes."""
    engine = _engine(tmp_path)
    try:
        rows = [_u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002), _a("reply to T1", 1003),
                _u("T2 user turn", 1004), _a("reply to T2", 1005)]
        engine.ingest(list(rows))
        u0, a0, u1, _a1, _u2, a2 = rows
        u1["content"] = "T1 user turn, rewritten in place" + PAD
        call_new, result_new = _call("call_c4", 1500), _result("call_c4", "fresh c4", 1501)
        trips = _trips(monkeypatch, engine)
        engine.ingest([u0, a0, u1, call_new, result_new, a2, _a("reply", 1600)])
        assert not trips, "setup: no trip"
        stored = _counts(engine)
        assert stored[_key(call_new)] == 1 and stored[_key(result_new)] == 1
        assert stored[_key(u1)] == 1
    finally:
        engine.shutdown()


def _tool_history() -> list:
    rows = [_u("T0 user turn", 1000), _a("reply to T0", 1001)]
    for n in range(3):
        rows += [_u(f"T{n + 1} user turn", 1010 + 10 * n), _unstamped(_call(f"call_d{n}", 1011 + 10 * n)),
                 _unstamped(_result(f"call_d{n}", f"result d{n}", 1012 + 10 * n)), _a(f"reply d{n}", 1013 + 10 * n)]
    return rows


@pytest.mark.parametrize("cause", ["window", "read_failure"])
def test_trip_forced_on_an_unchanged_list_adds_only_the_new_rows(tmp_path, monkeypatch, cause):
    """Trip-forced differential: every unstamped call and result row the previous ingest held is witnessed, so a
    forced trip stores zero copies, only the fresh rows."""
    engine = _engine(tmp_path)
    try:
        rows = _tool_history()
        engine.ingest(list(rows))
        before = _counts(engine)
        if cause == "window":
            _trip(monkeypatch)
        else:
            def boom(*_a, **_k):
                raise OSError("injected presence read failure")
            monkeypatch.setattr(engine, "_rotation_predecessor_session_ids", boom)
        # The host drops two stored text rows and a fresh pair takes their slots, so every stored pair stays below
        # the cursor; the fresh pair needs the presence window, so the scan trips beside three stored pairs.
        fresh_call, fresh_result = _call("call_dfresh", 1003), _result("call_dfresh", "fresh", 1004)
        reply = _a("a new reply", 1600)
        trips = _trips(monkeypatch, engine)
        engine.ingest([rows[0], fresh_call, fresh_result, *rows[3:], reply])
        assert len(trips) == 1, "setup: the presence check tripped"
        added = _counts(engine) - before
        assert added == Counter({_key(fresh_call): 1, _key(fresh_result): 1, _key(reply): 1}), added
        assert not before - _counts(engine), "the dropped rows stay stored"
    finally:
        engine.shutdown()


def test_a_busy_retry_keeps_the_witness(tmp_path, monkeypatch):
    """A busy error rolls the first attempt back; the retry still has the previous ingest's witness, so the trip
    stores no copy of the unchanged unstamped pairs."""
    engine = _engine(tmp_path)
    try:
        rows = _tool_history()
        engine.ingest(list(rows))
        before = _counts(engine)
        _trip(monkeypatch)
        real, calls = engine._identity_anchor_trip_unproven, []

        def busy_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            return real(*args, **kwargs)

        monkeypatch.setattr(engine, "_identity_anchor_trip_unproven", busy_once)
        fresh_call, fresh_result = _call("call_busy", 1003), _result("call_busy", "fresh", 1004)
        reply = _a("a new reply", 1600)
        engine.ingest([rows[0], fresh_call, fresh_result, *rows[3:], reply])
        assert len(calls) == 2, "setup: the first attempt hit the busy error and the retry tripped again"
        added = _counts(engine) - before
        assert added == Counter({_key(fresh_call): 1, _key(fresh_result): 1, _key(reply): 1}), added
    finally:
        engine.shutdown()


def test_cold_start_on_the_same_store_and_list_stores_nothing_new(tmp_path, monkeypatch):
    """Trip clause line 22: a new engine instance on the same store, handed the same list, stores nothing new."""
    rows = _tool_history()
    engine = _engine(tmp_path)
    try:
        engine.ingest(list(rows))
        before = _counts(engine)
    finally:
        engine.shutdown()
    _trip(monkeypatch)
    engine = _engine(tmp_path)
    try:
        engine.ingest(list(rows))
        assert _counts(engine) == before
    finally:
        engine.shutdown()


def test_r1_an_ignored_call_seen_again_below_the_scan_start_never_witnesses(tmp_path, monkeypatch):
    """Review r1 (Astra): an ignored call ingested twice sits below the next pass's scan start; it must not
    witness the storable version of the same invocation, nor authorize its result alone."""
    engine = _engine(tmp_path)
    engine._compiled_ignore_message_patterns = [_IgnoreMe()]
    try:
        u0, a0, u1, a1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002), \
            _a("reply to T1", 1004)
        ignored = _unstamped({**_call("call_r1", 1003), "content": "IGNORE-ME"})
        engine.ingest([u0, a0, u1, ignored, a1])
        u2 = _u("T2 user turn", 1005)
        engine.ingest([u0, a0, u1, ignored, a1, u2])  # the same raw list again: the call is now below scan start
        real = _unstamped(_call("call_r1", 1003))
        result = _unstamped(_result("call_r1", "its result", 1004))
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        engine.ingest([u0, a0, u1, real, result, u2, _a("reply", 1600)])  # the result takes a1's slot
        assert trips, "setup: the presence scan tripped"
        stored = _counts(engine)
        assert stored[_key(real)] == 1 and stored[_key(result)] == 1
    finally:
        engine.shutdown()


def test_r1_a_placeholder_echoed_as_text_parts_never_witnesses(tmp_path, monkeypatch):
    """Review r1 (Astra): the host echoes LCM's placeholder as a list of text parts; it still never witnesses."""
    engine = _engine(tmp_path)
    engine._compiled_ignore_message_patterns = [_IgnoreMe()]
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        call = _unstamped(_call("call_r1p", 1003))
        ignored = _unstamped(_result("call_r1p", "IGNORE-ME raw output", 1004))
        a1 = _a("reply to T1", 1005)
        engine.ingest([u0, a0, u1, call, ignored, a1])
        replay = list(engine._last_active_replay_messages)
        marker = replay[4]["content"]
        assert str(marker).startswith("[LCM active replay placeholder:"), marker
        echoed = {**replay[4], "content": [{"type": "text", "text": marker}]}
        engine.ingest([*replay[:4], echoed, replay[5], _u("T2 user turn", 1006)])
        real = _unstamped(_result("call_r1p", "the real output", 1004))
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        engine.ingest([*replay[:4], real, replay[5], _u("T2 user turn", 1006), _a("reply", 1600)])
        assert trips, "setup: the presence scan tripped"
        assert _counts(engine)[_key(real)] == 1
    finally:
        engine.shutdown()


def test_r1_a_repeated_result_id_never_copies_the_stored_result(tmp_path, monkeypatch):
    """Review r1 (Astra): a new result reusing a stored result's id, placed before it, must not spend its witness
    and copy it. A repeated result id is ambiguous and keeps main's result (a named limit)."""
    engine = _engine(tmp_path)
    try:
        rows = _tool_history()
        engine.ingest(list(rows))
        before = _counts(engine)
        again = _unstamped(_result("call_d1", "a second result for d1", 1015))
        fresh_call, fresh_result = _call("call_rr", 1003), _result("call_rr", "fresh", 1004)
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        # The host drops two stored text rows (fresh pair) and one more (the repeated result) before d1's pair.
        engine.ingest([rows[0], fresh_call, fresh_result, *rows[3:6], again, *rows[7:], _a("reply", 1600)])
        assert trips, "setup: the presence scan tripped"
        stored = _counts(engine)
        assert stored[_key(rows[8])] == 1, "the stored d1 result is not copied"
        assert stored[_key(fresh_call)] == 1 and stored[_key(fresh_result)] == 1
        assert not before - stored
    finally:
        engine.shutdown()


def test_r1_a_real_output_that_starts_like_a_placeholder_still_witnesses(tmp_path, monkeypatch):
    """Review r1 (Astra): a genuine stored tool output that begins with the placeholder prefix is not a placeholder;
    on a trip it is witnessed, not copied."""
    engine = _engine(tmp_path)
    try:
        rows = _tool_history()
        rows[8] = _unstamped(_result("call_d1", "[LCM active replay placeholder: example]", 1022))
        engine.ingest(list(rows))
        before = _counts(engine)
        fresh_call, fresh_result = _call("call_rp", 1003), _result("call_rp", "fresh", 1004)
        _trip(monkeypatch)
        trips = _trips(monkeypatch, engine)
        engine.ingest([rows[0], fresh_call, fresh_result, *rows[3:], _a("reply", 1600)])
        assert trips, "setup: the presence scan tripped"
        added = _counts(engine) - before
        assert added[_key(rows[8])] == 0, added
    finally:
        engine.shutdown()


@pytest.mark.parametrize("window", [1, None], ids=["trip", "no-trip"])
def test_r1_a_pass_whose_audit_fell_open_never_witnesses_its_prefix(tmp_path, monkeypatch, window):
    """Review r1 (Claude): the topic-9 merge turn's pre-match raises on both attempts, so the ingest falls open and
    never stores the slid reply. Its witness must not hold that reply, so the next ingest stores it once."""
    engine = _engine(tmp_path)
    try:
        hist = [_u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002), _a("reply to T1", 1003)]
        prompt_a, prompt_b = _u("A: first part", 2000), _u("B: second part", 2001)
        engine.ingest([*hist, prompt_a, prompt_b])
        m1 = {"role": "assistant", "content": "Let me check both files.", "tool_calls": [
            {"id": "call_fo_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
            {"id": "call_fo_b", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b"}'}}]}
        r1a = _unstamped(_result("call_fo_a", "contents a", 2003))
        r1b = _unstamped(_result("call_fo_b", "contents b", 2004))
        m2 = _a("answer", 2005)
        original = prompt_a["content"]
        prompt_a["content"] = original + "\n\n" + prompt_b["content"]
        prompt_a["_merged_turn_prefix"] = original
        with monkeypatch.context() as patch:
            def boom(*_a, **_k):
                raise RuntimeError("injected pre-match fault")
            patch.setattr(engine, "_identity_anchor_prematch", boom)
            try:
                engine.ingest([*hist, prompt_a, m1, r1a, r1b, m2])
            except Exception:  # noqa: BLE001 (the fall-open path may surface the fault; the next ingest decides)
                pass
        witness = (engine.__dict__.get("_trip_witness") or (None, {}))[1]
        assert not witness.get(("call", ("call_fo_a", "call_fo_b"))), "the fallen-open pass never witnesses m1"
        if window is not None:
            _trip(monkeypatch)
        engine.ingest([*hist, prompt_a, m1, r1a, r1b, m2, _a("next answer", 2100)])
        stored = _counts(engine)
        for row in (m1, r1a, r1b, m2):
            assert stored[_key(row)] >= 1, row
        assert stored[_key(m1)] == 1
    finally:
        engine.shutdown()


def test_r2_a_list_valued_result_id_never_breaks_the_audit(tmp_path, monkeypatch):
    """Review r2 (Astra): a replayed tail result whose tool_call_id is a list (stringified the same as a stored string
    id) must not raise in the audit; the missed user edit below the cursor is still stored."""
    engine = _engine(tmp_path)
    try:
        u0, a0, u1 = _u("T0 user turn", 1000), _a("reply to T0", 1001), _u("T1 user turn", 1002)
        stored_result = _result("['x']", "result text", 1003)
        engine.ingest([u0, a0, u1, stored_result])
        edited = _u("T1 user turn, edited by the host", 1002)
        listed = {**stored_result, "tool_call_id": ["x"]}
        # A new text row before it moves the stamp-matched result to the old cursor (a replayed tail row).
        engine.ingest([u0, a0, edited, _a("a new reply before the result", 1002.5), listed, _a("reply", 1600)])
        assert _counts(engine)[_key(edited)] == 1
    finally:
        engine.shutdown()
