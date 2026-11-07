"""I1/I2: compacted history cannot prove a fresh reused-id tool occurrence."""

from collections import Counter
from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored
from tests.test_compression_boundary import _turn
from tests.test_issue_671_stub_first_exit import tool_pair


def _replay_links(engine):
    conn = engine._store._conn
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone():
        return {}
    return dict(conn.execute("SELECT store_id, related_store_id FROM message_relations "
                             "WHERE kind='possible_replay_of'"))


def _key(row):
    return row["role"], row.get("content"), row.get("tool_call_id"), row.get("tool_calls")


@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_cold_cursor_between_call_and_result_stores_no_silent_copy(build, monkeypatch, ledger):
    """#2163: a cold bind whose cursor lands between a stored call and its stored result.

    The soak saw the identity anchor stop there; the cursor is pinned to that boundary. The result
    at the cursor is a replay of the stored one: it is either not stored again or, when stored,
    linked ``possible_replay_of`` to the stored copy (visible, never silent). The new turn is stored.
    """
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    cursor = len(host) - 1
    assert host[cursor - 1]["role"] == "assistant" and host[cursor]["role"] == "tool"
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: cursor)
    new = {"role": "user", "content": "next"}
    engine.ingest(deepcopy(host) + [new])
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [_key(row) for row in added if row["role"] == "user"] == [_key(new)]
    links = _replay_links(engine)
    by_id = {row["store_id"]: row for row in before}
    silent = [row["store_id"] for row in added if row["role"] != "user"
              and _key(by_id.get(links.get(row["store_id"]), {"role": None})) != _key(row)]
    assert silent == []


def _cold_boundary(build, monkeypatch, stored_rows, replay, cursor=None):
    """Store ``stored_rows``, restart cold, pin the reconcile cursor (unless None) and replay; the added rows."""
    engine = build()
    engine.ingest(deepcopy(stored_rows))
    engine.shutdown()
    engine = build()
    before = stored(engine)
    if cursor is not None:
        monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: cursor)
    engine.ingest(deepcopy(replay))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    return [(row["role"], row.get("content"), row.get("tool_call_id")) for row in after[len(before):]]


def test_reused_id_result_after_a_dangling_stored_call_is_stored(build, monkeypatch):
    """The call's latest stored copy has no result after it, so an identical older result proves nothing."""
    call, result = tool_pair("call_0", "OK", tool_name="poll")
    head = [row for i in range(2) for row in _turn(i)] + [{"role": "user", "content": "poll"}, call, result,
                                                          {"role": "user", "content": "poll again"}, call]
    added = _cold_boundary(build, monkeypatch, head, head + [result, {"role": "user", "content": "next"}], len(head))
    assert added == [("tool", "OK", "call_0"), ("user", "next", None)]


def test_two_calls_with_one_stored_result_suppress_only_that_result(build, monkeypatch):
    call = {"role": "assistant", "content": "running tools", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "lookup", "arguments": "{}"}} for cid in ("a", "b")]}
    first, second = ({"role": "tool", "tool_call_id": cid, "content": "result " + cid} for cid in ("a", "b"))
    head = [row for i in range(2) for row in _turn(i)] + [{"role": "user", "content": "look"}, call]
    added = _cold_boundary(build, monkeypatch, head + [first],
                           head + [first, second, {"role": "user", "content": "next"}], len(head))
    assert added == [("tool", "result b", "b"), ("user", "next", None)]


def test_cold_compaction_cursor_keeps_changed_result_timestamp(build, monkeypatch):
    """A real durable-proof rewind (no cursor stub) to a result whose host stamp changed: it is new."""
    plain_fixture = shapes.fixture

    def stamped_fixture(*, uids=False):
        rows = plain_fixture(uids=uids)
        rows[-2]["timestamp"], rows[-1]["timestamp"] = 11, 12
        return rows

    monkeypatch.setattr(shapes, "fixture", stamped_fixture)
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    previous = deepcopy(host)
    result_index = len(host) - 1
    host[result_index]["timestamp"] = 22
    assert engine._host_occurrence_changed(host[result_index], previous[result_index])
    scans = []
    helper = engine._replayed_tool_segment_indexes_after_cursor

    def observe(messages, cursor, **kwargs):
        selected = helper(messages, cursor, **kwargs)
        scans.append((cursor, selected))
        return selected

    monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", observe)
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    assert scans and scans[0][0] == result_index
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [(row["role"], row.get("tool_call_id"), row.get("observed_at")) for row in added] == [("tool", "c", 22)]


def _distinct_pair():
    return tool_pair("d", "distinct output", tool_name="lookup")


def test_a_result_of_an_earlier_dangling_call_is_not_taken_for_a_later_stored_pair(build, monkeypatch):
    """Reused ids: the store holds a dangling call, then a later identical call with its result. The host now shows
    the first call's result. Its own call has no stored result, so it is stored (no stub: the natural cursor)."""
    call, result = tool_pair("c", "OK", tool_name="poll")
    head = [{"role": "user", "content": "start"}, *_distinct_pair(), call]
    store = [*head, {"role": "user", "content": "again"}, call, result]
    host = [*head, result, {"role": "user", "content": "again"}, call, result, {"role": "user", "content": "next"}]
    added = _cold_boundary(build, monkeypatch, store, host)
    assert ("tool", "OK", "c") in added and ("user", "next", None) in added, added


def test_a_ledger_binding_outside_the_tail_proves_nothing(build):
    engine = build()
    call, result = tool_pair("c", "OK", tool_name="poll")
    engine.ingest([{"role": "user", "content": "start"}, call, result])
    tail = engine._store.get_session_tail(engine._session_id, limit=10)
    counts = Counter(engine._message_replay_identity(row, stored_row=True) for row in tail)
    host = [call, deepcopy(result), {"role": "user", "content": "next"}]
    assert engine._replayed_results_at_cursor(host, 1, tail, set(), {0: 10**9}, counts) == set()
    # Control: the same call bound to its stored copy in the tail proves the result.
    bound = next(row["store_id"] for row in tail if row["role"] == "assistant")
    assert engine._replayed_results_at_cursor(host, 1, tail, set(), {0: bound}, counts) == {1}


def test_a_cleared_output_placeholder_is_not_evidence(build, monkeypatch):
    """#1385: a placeholder borrows the stored payload's identity; it is no proof that this result was stored."""
    call = tool_pair("c", "x", tool_name="read_file")[0]
    stored_result = {"role": "tool", "tool_call_id": "c", "content": "x" * 300}
    cleared = {"role": "tool", "tool_call_id": "c", "content": _HERMES_PRUNED_TOOL_PLACEHOLDER}
    head = [{"role": "user", "content": "start"}, *_distinct_pair(), call]
    added = _cold_boundary(build, monkeypatch, [*head, stored_result],
                           [*head, cleared, {"role": "user", "content": "next"}])
    assert ("tool", _HERMES_PRUNED_TOOL_PLACEHOLDER, "c") in added, added


def test_a_later_identical_stored_pair_does_not_prove_an_earlier_dangling_call(build, monkeypatch):
    """The cursor's call matches an earlier stored copy with no result; a later identical call and result is in the
    store. Two stored copies of the call are ambiguous, so the host's result is stored (pinned at the boundary)."""
    call, result = tool_pair("call_0", "OK", tool_name="poll")
    first = [row for i in range(2) for row in _turn(i)] + [{"role": "user", "content": "poll"}, call]
    engine = build()
    engine.ingest(deepcopy(first))
    engine.ingest(deepcopy([*first, {"role": "user", "content": "poll again"}, call, result]))
    engine.shutdown()
    engine = build()
    before = stored(engine)
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: len(first))
    engine.ingest(deepcopy([*first, result, {"role": "user", "content": "next"}]))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    added = [(row["role"], row.get("content"), row.get("tool_call_id")) for row in after[len(before):]]
    assert added == [("tool", "OK", "call_0"), ("user", "next", None)], added


def test_a_call_whose_stamp_differs_from_its_stored_copy_proves_nothing(build):
    """Review round 2: the only copy of the call in the tail is a different occurrence when the stamps disagree."""
    engine = build()
    call, result = tool_pair("c", "OK", tool_name="poll")
    engine.ingest([{"role": "user", "content": "start", "timestamp": 1.0},
                   {**call, "timestamp": 2.0}, {**result, "timestamp": 3.0}])
    tail = engine._store.get_session_tail(engine._session_id, limit=10)
    counts = Counter(engine._message_replay_identity(row, stored_row=True) for row in tail)
    host = [{**call, "timestamp": 1.0}, deepcopy(result), {"role": "user", "content": "next"}]
    assert engine._replayed_results_at_cursor(host, 1, tail, set(), {}, Counter(counts)) == set()
    # Control: the same call with its stored stamp is the stored occurrence.
    host[0]["timestamp"] = 2.0
    assert engine._replayed_results_at_cursor(host, 1, tail, set(), {}, Counter(counts)) == {1}


def test_a_vetoed_call_keeps_its_stored_result_from_proving_a_later_call(build):
    """Review round 3: A and B reuse id ``c`` with other arguments. The host's A is another occurrence (its stamp
    differs), so the cursor result is stored; A's stored result must not then prove B's new result as a replay."""
    engine = build()
    call_a, result = tool_pair("c", "OK", tool_name="poll")
    call_b = deepcopy(call_a)
    call_b["tool_calls"][0]["function"]["arguments"] = '{"again": true}'
    start = {"role": "user", "content": "start", "timestamp": 0.5}
    engine.ingest([start, {**call_a, "timestamp": 2.0}, deepcopy(result), {**call_b, "timestamp": 4.0}])
    rows = engine._store.get_session_messages(engine._session_id)
    stored_a = next(row["store_id"] for row in rows if row["role"] == "assistant")
    host = [start, {**call_a, "timestamp": 1.0}, deepcopy(result), {**call_b, "timestamp": 4.0}, deepcopy(result),
            {"role": "user", "content": "next"}]
    assert engine._replayed_tool_segment_indexes_after_cursor(host, 2, consumed_pairs={1: stored_a}) == set()
    # Control: with A's stamp agreeing, A's result is the replay and B's new result still is not.
    host[1]["timestamp"] = 2.0
    assert engine._replayed_tool_segment_indexes_after_cursor(host, 2, consumed_pairs={1: stored_a}) == {2}


def _poll(ids, args, stamp):
    return {"role": "assistant", "content": "", "timestamp": stamp, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "poll", "arguments": args}} for i in ids]}


def test_a_result_proves_replay_only_for_the_call_copy_it_follows(build):
    """Review round 4: the store holds A(c)·X, B(c,d)·Y, C(d). The host's A is another occurrence (its stamp
    differs), so the X after it is new. B's stored segment owns one X and one Y; the host repeats X and Y after B and
    after C. Only the (call, result) pairs the tail still owns prove a segment replayed; C's Y is not owned by C."""
    engine = build()
    x = {"role": "tool", "tool_call_id": "c", "content": "X"}
    y = {"role": "tool", "tool_call_id": "d", "content": "Y"}
    start = {"role": "user", "content": "start", "timestamp": 0.5}
    engine.ingest(deepcopy([start, _poll(["c"], '{"a":1}', 2.0), x, _poll(["c", "d"], '{"b":1}', 4.0), y,
                            _poll(["d"], '{"c":1}', 5.0)]))
    rows = engine._store.get_session_messages(engine._session_id)
    stored_a = next(row["store_id"] for row in rows if row["role"] == "assistant")
    host = [start, _poll(["c"], '{"a":1}', 1.0), x, _poll(["c", "d"], '{"b":1}', 4.0), x, y,
            _poll(["d"], '{"c":1}', 5.0), y, {"role": "user", "content": "next"}]
    suppressed = engine._replayed_tool_segment_indexes_after_cursor(deepcopy(host), 2, consumed_pairs={1: stored_a})
    assert 7 not in suppressed, suppressed  # C's result is not owned by any stored C pair: it is stored


def _shape_row(row):
    return (*_key(row), row.get("observed_at", row.get("timestamp")))


def _pooled_shape(build, monkeypatch, source, host, cursor, bindings):
    """Ingest ``source``, restart, pin the resume point and the ledger's bindings (host index -> stored row index),
    ingest ``host``, and return its whole stored list and the first scan's replay indexes."""
    engine = build()
    engine.ingest(deepcopy(source))
    rows = engine._store.get_session_messages(engine._session_id)
    expected = [_shape_row(row) for row in source]
    _expect_stored_shape([_shape_row(row) for row in rows], expected, expected, "setup")
    engine.shutdown()
    engine = build()
    try:
        pairs = {at: rows[stored] for at, stored in bindings.items()}
        monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *args, **kwargs: cursor)
        monkeypatch.setattr(engine, "_ledger_cold_cursor", lambda *args, **kwargs: dict(k=cursor, pairs=pairs))
        scans, scan = [], engine._replayed_tool_segment_indexes_after_cursor

        def tracked(messages, at, **kwargs):
            replayed = scan(messages, at, **kwargs)
            scans.append(set(replayed))
            return replayed
        monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", tracked)
        engine.ingest(deepcopy(host))
        if engine._ingest_failure_count:
            pytest.fail(f"setup: the ingest failed: {engine._last_ingest_error}")
        if not scans:
            pytest.fail("setup: the cold ingest never ran the tool-segment scan")
        return [_shape_row(row) for row in engine._store.get_session_messages(engine._session_id)], scans[0]
    finally:
        engine.shutdown()


def _expect_2377_shape(observed, fixed, known, new, replayed):
    _expect_stored_shape(observed, [_shape_row(row) for row in fixed], [_shape_row(row) for row in known],
                         f"#2377 known shape: host index {new} taken for a replay: {replayed}")


_U = {"role": "user", "content": "start", "timestamp": 0.5}
_V = {"role": "user", "content": "next", "timestamp": 10.0}


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2377: a vetoed occurrence's ownership stays spendable by another occurrence")
def test_issue_2377_vetoed_ownership_stays_spendable(build, monkeypatch):
    """Store A@2(c)·X, B@3(c)·X, A@4(c) dangling; the host's A@1 is another occurrence. The new X at host index 6 is
    the dangling A@4's result: no single stored occurrence proves it, so it is not a replay."""
    x = {"role": "tool", "tool_call_id": "c", "content": "X"}
    y = {"role": "tool", "tool_call_id": "c", "content": "Y"}
    a = lambda stamp: _poll(["c"], '{"a":1}', stamp)  # noqa: E731
    b = lambda stamp: _poll(["c"], '{"b":1}', stamp)  # noqa: E731
    source = [_U, a(2.0), x, b(3.0), x, a(4.0)]
    host = [_U, a(1.0), x, b(3.0), y, a(4.0), x, _V]
    observed, replayed = _pooled_shape(build, monkeypatch, source, host, cursor=2, bindings={1: 1})
    _expect_2377_shape(observed, [*source, *host[2:]], [*source, *host[2:5], host[-1]], 6, replayed)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2377: partial parallel occurrences pool their pairs into a false replay proof")
def test_issue_2377_partial_parallel_occurrences_pool(build, monkeypatch):
    """P(c,d)@2·X and P(c,d)@4·D are stored; neither occurrence holds both results. The host's first segment
    P@2·X·D is not proven by one occurrence, so its D (host index 3) is not a replay."""
    x = {"role": "tool", "tool_call_id": "c", "content": "X"}
    d = {"role": "tool", "tool_call_id": "d", "content": "D"}
    call = _poll(["c", "d"], "{}", 2.0)
    later = {**call, "timestamp": 4.0}
    source = [_U, call, x, later, d]
    host = [_U, call, x, d, later, x, d, _V]
    observed, replayed = _pooled_shape(build, monkeypatch, source, host, cursor=1, bindings={})
    _expect_2377_shape(observed, [*source, *host[1:]], [*source, *host[4:]], 3, replayed)


def _expect_stored_shape(observed, fixed, known, message, error_type=AssertionError):
    """Accept only the complete fixed list or the complete named-limit list, in store order."""
    if observed == fixed:
        return
    if observed == known:
        raise error_type(message)
    differences = []
    for label, expected in (("fixed", fixed), ("known", known)):
        index = next((i for i, (actual, wanted) in enumerate(zip(observed, expected)) if actual != wanted),
                     min(len(observed), len(expected)))
        differences.append(f"{label}: first difference={index}, lengths={len(observed)}/{len(expected)}")
    pytest.fail("unexpected stored shape; " + "; ".join(differences))


@pytest.mark.parametrize("case", ["known", "fixed", "doubled", "swapped"])
def test_stored_shape_comparison(case):
    fixed = [("user", "first"), ("assistant", "second")]
    known = [fixed[0]]
    observed = {"known": known, "fixed": fixed, "doubled": [fixed[0], *fixed], "swapped": fixed[::-1]}[case]
    if case == "known":
        with pytest.raises(AssertionError, match="synthetic known shape"):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    elif case == "fixed":
        _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    else:
        with pytest.raises(pytest.fail.Exception, match="first difference=.*lengths="):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
