"""I1/I2: compacted history cannot prove a fresh reused-id tool occurrence."""

from collections import Counter
from copy import deepcopy

import pytest

from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored
from tests.test_compression_boundary import _turn
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("dangling", [
    pytest.param(False, id="full-pair", marks=pytest.mark.xfail(strict=True, reason=(
        "#2172: a new call and result that repeat a pair compacted out of the view are taken as a replay"))),
    pytest.param(True, id="dangling-call"),
])
def test_compacted_away_identical_pair_is_not_proof(build, monkeypatch, dangling):
    def fixture(*, uids=False):
        rows = [row for i in range(2) for row in _turn(i)]
        rows += [{"role": "user", "content": "poll early"},
                 *tool_pair("call_0", "OK", tool_name="poll")]
        rows += [row for i in range(2, 12) for row in _turn(i)]
        for cid, text, stamp in (("older", "older lookup", 8),
                                 ("middle", "middle lookup", 9), ("c", "repeat", 10)):
            rows += [{"role": "user", "content": text, "timestamp": stamp},
                     *tool_pair(cid, shapes.LONG + cid, tool_name="lookup")]
        return rows

    monkeypatch.setattr(shapes, "fixture", fixture)
    engine, host = compact(build, mode="cold")
    assert not any(row.get("tool_call_id") == "call_0" for row in host)
    call, result = tool_pair("call_0", "OK", tool_name="poll")
    if dangling:
        host = deepcopy(host) + [{"role": "user", "content": "poll again"}, call]
        engine.ingest(deepcopy(host))
        engine.shutdown()
        engine = build()
        new = [result, {"role": "user", "content": "next"}]
    else:
        new = [{"role": "user", "content": "poll again"}, call, result,
               {"role": "user", "content": "next"}]
    before = stored(engine)
    engine.ingest(deepcopy(host) + new)
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [(row["role"], row["content"]) for row in added] == [
        (row["role"], row["content"]) for row in new
    ]
    assert added[-2]["tool_call_id"] == "call_0"


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
