"""I1/I2: one stored tool occurrence, never an identity pool, proves a replay."""

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
