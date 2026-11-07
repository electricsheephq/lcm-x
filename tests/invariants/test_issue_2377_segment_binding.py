"""I1/I2: one stored tool occurrence, never an identity pool, proves a replay."""

from copy import deepcopy
import json

import pytest

from tests.invariants.test_issue_2163_compacted_cursor import (
    _cold_boundary, _poll, _pooled_shape, _U, _V, build as build,
)

pytestmark = pytest.mark.issue(2377)
X = {"role": "tool", "tool_call_id": "c", "content": "X"}
D = {"role": "tool", "tool_call_id": "d", "content": "D"}


def test_three_identical_parallel_calls_keep_the_new_occurrence(build, monkeypatch):
    """The first of three P(c,d) occurrences is new; only the two stamped copies bind."""
    p = lambda stamp: _poll(["c", "d"], "{}", stamp)  # noqa: E731
    source = [_U, p(2), X, D, p(4), X, D]
    host = [_U, p(1), X, D, p(2), X, D, p(4), X, D, _V]
    replayed = _pooled_shape(build, monkeypatch, source, host, 1, {})
    assert replayed == {4, 5, 6, 7, 8, 9}


def test_stamped_mismatch_disqualifies_the_only_segment(build, monkeypatch):
    call = _poll(["c", "d"], "{}", 2)
    replayed = _pooled_shape(build, monkeypatch, [_U, call, X, D],
                             [_U, {**call, "timestamp": 3}, X, D, _V], 1, {})
    assert replayed == set()


def test_unstamped_order_spends_only_two_of_three_identical_segments(build, monkeypatch):
    call = _poll(["c", "d"], "{}", None)
    source = [_U, call, X, D, call, X, D]
    host = [*source, call, X, D, _V]
    replayed = _pooled_shape(build, monkeypatch, source, host, 1, {})
    assert replayed == {1, 2, 3, 4, 5, 6}


def test_ambiguous_unstamped_results_are_stored(build, monkeypatch):
    call = _poll(["c"], "{}", None)
    other = {**X, "content": "different result"}
    source = [_U, call, X, call, other]
    added = _cold_boundary(build, monkeypatch, source, [_U, call, X, _V], 1)
    assert added == [("assistant", "", None), ("tool", "X", "c"), ("user", "next", None)]


@pytest.mark.parametrize("carry", ["neither", "call-only", "result-only", "both", "after-proof"])
def test_compacted_out_copy_cannot_bind(build, carry):
    """Use a real durable v2 proof; both call and result must remain in its carry set."""
    engine = build()
    call = _poll(["c"], "{}", 2)
    engine.ingest(deepcopy([_U, call, X]))
    rows = engine._store.get_session_messages(engine._session_id)
    call_id, result_id = (row["store_id"] for row in rows[1:])
    ranges = {
        "neither": [], "call-only": [["S", call_id - 1, call_id]],
        "result-only": [["S", result_id - 1, result_id]],
        "both": [["S", call_id - 1, result_id]], "after-proof": [],
    }
    proof = {"version": 2, "hermes_home": engine._hermes_home,
             "conversation_id": engine._conversation_id, "created_at": 10**12,
             "last_store_id": rows[0]["store_id"] if carry == "after-proof" else result_id,
             "carry_ranges": ranges[carry]}
    engine._store.write_metadata_json(["compaction_commit_proof:S"], json.dumps(proof))
    assert engine._durable_commit_proof_payload() is not None
    host = [_U, call, X, _V]
    expected = {1, 2} if carry in ("both", "after-proof") else set()
    assert engine._replayed_tool_segment_indexes_after_cursor(host, 1) == expected
