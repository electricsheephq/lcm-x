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
    # Astra (delta 6): stored (A, X); view (A, Y), (B, X) under one id. The id is reused (its calls and results
    # differ), so all its rows keep main's result: nothing is stored, so no pair is split (#2172).
    from tests.test_anchor_unstored_before_cursor import _engine, _presence, SESSION

    def call(path):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_x", "type": "function", "function": {"name": "read_file", "arguments": path}}]}

    def result(text):
        return {"role": "tool", "content": text, "tool_call_id": "call_x", "tool_name": "read_file"}

    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [call("A"), result("X")])
        assert _presence(engine, [call("A"), result("Y"), call("B"), result("X")]) == []
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
        # A fresh id is stored once; a reused id keeps main's result, which stores neither row (#2172).
        expected = 1 if "unique" in shape else 0
        assert added[_full(new_call)] == added[_full(new_result)] == expected
    finally:
        engine.shutdown()


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2172: a reused id keeps main's result, so its new pair is not stored (#2466 review)")
def test_a_new_pair_reusing_a_stored_id_is_stored_once(tmp_path):
    engine = _engine(tmp_path)
    try:
        stored_call, stored_result = _call("call_0", '{"path": "b.txt"}'), _res("call_0", "B file text")
        tail = [_a("A1 reply", 103), _u("P2 second", 110), _a("A2 reply", 111), _u("P3 third", 120), _a("A3", 121)]
        engine.ingest([_u("P1 first", 100), stored_call, stored_result, *tail])
        before = Counter(_report(engine))
        new_call, new_result = _call("call_0", '{"path": "c.txt"}'), _res("call_0", "C file text")
        engine.ingest([_u("P1 replaced first", 99), new_call, new_result, stored_call, stored_result, *tail[:-2],
                       _u("P4 new", 200)])
        added = Counter(_report(engine)) - before
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


@pytest.mark.parametrize("stored_text,shown_text", [
    ("[Externalized tool output: read_file; ref=out.json]", "the full output " * 500),
    ("the full output " * 500, "[Old tool output cleared to save context space]")],
    ids=["externalized-in-store", "pruned-by-host"])
def test_one_result_stored_in_another_form_is_still_one_invocation(tmp_path, stored_text, shown_text):
    """The store can hold a result in another form than the host shows: an externalized placeholder, or the full
    text the host has since pruned. It is the same invocation, so its call, missing from the store, is stored
    (class A), and the result is not stored again (#2466 review)."""
    from tests.test_anchor_unstored_before_cursor import _engine, _presence, SESSION

    engine = _engine(tmp_path)
    try:
        call = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_big", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}]}
        engine._store.append_batch(SESSION, [{"role": "tool", "tool_call_id": "call_big", "tool_name": "read_file",
                                              "content": stored_text}])
        shown = {"role": "tool", "tool_call_id": "call_big", "tool_name": "read_file", "content": shown_text}
        assert _presence(engine, [call, shown]) == [0]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("stored_args,shown_args", [('{"path": "a", "n": 1}', '{"n":1,"path":"a"}'),
                                                    ('{"path": "a"}', {"path": "a"})],
                         ids=["key-order-and-spacing", "string-or-object"])
def test_one_call_serialized_differently_is_still_one_invocation(tmp_path, stored_args, shown_args):
    """A stored call whose arguments the host now serializes differently is the same invocation: its result, missing
    from the store, is stored (class A) (#2466 review)."""
    from tests.test_anchor_unstored_before_cursor import _engine, _presence, SESSION

    def call(args):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_s", "type": "function", "function": {"name": "read_file", "arguments": args}}]}

    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [call(stored_args)])
        result = {"role": "tool", "tool_call_id": "call_s", "tool_name": "read_file", "content": "file text"}
        assert _presence(engine, [call(shown_args), result]) == [1]
    finally:
        engine.shutdown()


def test_past_the_parse_budget_a_call_serialized_differently_keeps_mains_result(tmp_path, monkeypatch):
    """Arguments are parsed only within a byte budget (the wait bar). Past it, an id whose stored and shown arguments
    differ as text reads as reused, so its rows keep main's result: nothing is added (#2466 review, delta 9)."""
    from hermes_lcm import identity_anchor
    from tests.test_anchor_unstored_before_cursor import _engine, _presence, SESSION

    def call(args):
        return {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_s", "type": "function", "function": {"name": "read_file", "arguments": args}}]}

    monkeypatch.setattr(identity_anchor, "_PRESENCE_PARSE_BYTES", 0)
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [call('{"path": "a", "n": 1}')])
        result = {"role": "tool", "tool_call_id": "call_s", "tool_name": "read_file", "content": "file text"}
        assert _presence(engine, [call('{"n":1,"path":"a"}'), result]) == []
        assert _presence(engine, [call('{"path": "a", "n": 1}'), result]) == [1]  # same text: no parse needed
    finally:
        engine.shutdown()


@pytest.mark.parametrize("form", ["externalized", "host-pruned"])
@pytest.mark.parametrize("own_first", [True, False], ids=["own-stamp-first", "carried-stamp-first"])
def test_class_a_moved_pair_is_stored_once_when_a_stored_output_has_another_form(tmp_path, form, own_first):
    """Class A V1 at default settings with an earlier tool output stored in another form than the host shows: LCM
    externalized it (a large output), or the host has since pruned it to a stub. The moved turn's call and result are
    each stored once, and nothing else is stored (#2466 review)."""
    from tests.test_anchor_unstored_before_cursor import SESSION, _key, _moved_turn, _stored

    big = "old file line\n" * 1_500  # past the default 12,000-character externalization threshold
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), large_output_externalization_enabled=True)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", context_length=1_000_000)
    try:
        history, turn_start, turn_end = _moved_turn(own_first)
        old = next(i for i, row in enumerate(history) if row.get("tool_call_id") == "call_old")
        for view in (history, turn_start, turn_end):
            view[old] = {**view[old], "content": big}
        if form == "host-pruned":
            turn_end[old] = {**turn_end[old], "content": "[Old tool output cleared to save context space]"}
        engine.ingest(history)
        if form == "externalized":
            stored_old = next(r for r in _stored(engine) if r.get("tool_call_id") == "call_old")
            assert len(stored_old["content"]) < len(big)  # setup: the stored output is a placeholder
        engine.ingest(turn_start)
        before = Counter(_key(row) for row in _stored(engine))
        engine.ingest(turn_end)
        added = Counter(_key(row) for row in _stored(engine)) - before
        new_call = next(r for r in turn_end if (r.get("tool_calls") or [{}])[0].get("id") == "call_new")
        new_result = next(r for r in turn_end if r.get("tool_call_id") == "call_new")
        assert added[_key(new_call)] == 1 and added[_key(new_result)] == 1
        assert not any(key[2] == "call_old" or key[3] == "call_old" for key in added)
        # Nothing stored twice except the moved prompt at its other stamp (#2421, unchanged here).
        twice = {key for key, count in Counter(_key(row) for row in _stored(engine)).items() if count > 1}
        assert twice <= {_key(turn_end[4])}, twice
    finally:
        engine.shutdown()


def test_a_call_whose_id_is_a_question_mark_is_not_stored_again(tmp_path):
    """A valid id "?" must not collide with the unreadable-call marker: a stored pre-ledger pair under it is found,
    so its view copy is not stored again (#2466 review, delta 8)."""
    from tests.test_anchor_unstored_before_cursor import _engine, _presence, SESSION

    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "?", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "?", "content": "done"}
    engine = _engine(tmp_path)
    try:
        engine._store.append_batch(SESSION, [call, result])
        assert _presence(engine, [call, result]) == []
    finally:
        engine.shutdown()


def test_an_ignored_result_past_the_cursor_never_authorizes_its_call(tmp_path, monkeypatch):
    """The tail ingest drops a result the ignore patterns match, so it cannot complete its call's pair (#2466 review,
    delta 8): the call keeps main's result."""
    from tests.test_anchor_unstored_before_cursor import _engine

    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(engine, "_matches_ignore_message_patterns",
                            lambda msg: str(msg.get("content") or "").startswith("IGNORE"))
        call = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
        view = [{"role": "user", "content": "new prompt"}, call, {"role": "tool", "tool_call_id": "c",
                                                                  "content": "IGNORE result"}]
        found = engine._identity_anchor_unaccounted(
            view, view, 2, 0, lambda i: engine._message_replay_identity(view[i]), {}, True,
            set(), {"replayed": set()}, {}, (), [0])
        assert found == []
    finally:
        engine.shutdown()
