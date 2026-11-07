"""I1/I2: one stored tool occurrence, never an identity pool, proves a replay."""

from copy import deepcopy

import pytest

from tests.invariants.test_issue_2163_compacted_cursor import (
    _cold_boundary, _poll, _pooled_shape, _replay_links, _U, _V, build as build,
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


def test_full_result_match_beats_other_unstamped_results(build, monkeypatch):
    call = _poll(["c"], "{}", None)
    other = {**X, "content": "different result"}
    source = [_U, call, X, call, other]
    added = _cold_boundary(build, monkeypatch, source, [_U, call, X, _V], 1)
    assert added == [("user", "next", None)]


def _ingest_shape(build, monkeypatch, source, host, cursor=1, pairs=None, relation_fault=False):
    engine = build()
    engine.ingest(deepcopy(source))
    before = engine._store.get_session_messages(engine._session_id)
    assert len(before) == len(source)
    engine.shutdown()
    engine = build()
    if relation_fault:
        write = engine._store.add_message_relations

        def faulted(groups):
            assert engine._store._transaction_owner.owns_connection(engine._store._conn)
            result = write(groups)
            if any(row[1] == "possible_replay_of" for group in groups for row in group):
                raise RuntimeError("synthetic relation failure after write")
            return result
        monkeypatch.setattr(engine._store, "add_message_relations", faulted)
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: cursor)
    monkeypatch.setattr(engine, "_ledger_cold_cursor", lambda *a, **kw: dict(
        k=cursor, pairs={idx: before[at] for idx, at in (pairs or {}).items()}))
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = engine._store.get_session_messages(engine._session_id)
    assert after[:len(before)] == before
    return engine, before, after[len(before):]


@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_ambiguous_unstamped_results_are_stored(build, monkeypatch, ledger):
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    call = _poll(["c", "d"], "{}", None)
    engine, before, added = _ingest_shape(
        build, monkeypatch, [_U, call, X, call, D], [_U, call, X, D, _V])
    assert [(r["role"], r.get("tool_call_id")) for r in added] == [
        ("assistant", None), ("tool", "c"), ("tool", "d"), ("user", None)]
    links = _replay_links(engine)
    assert links == {added[i]["store_id"]: before[j]["store_id"] for i, j in ((0, 1), (1, 2), (2, 4))}


def test_ambiguous_copies_spend_link_targets_once_in_store_order(build, monkeypatch):
    call = _poll(["c", "d"], "{}", None)
    engine, before, added = _ingest_shape(
        build, monkeypatch, [_U, call, X, call, D], [_U, call, X, D, call, X, D, _V])
    assert len(added) == 7
    # Each target and each copy is used once, in candidate store order.
    assert _replay_links(engine) == {added[i]["store_id"]: before[j]["store_id"]
                                    for i, j in ((0, 1), (1, 2), (2, 4), (3, 3))}
    pairs = list(engine._store._conn.execute(
        "SELECT store_id,related_store_id FROM message_relations WHERE kind='possible_replay_of'"))
    assert len(pairs) == len({p[0] for p in pairs}) == len({p[1] for p in pairs})


def test_ambiguous_link_failure_keeps_ingest_and_rolls_back_links(build, monkeypatch):
    call = _poll(["c", "d"], "{}", None)
    engine, before, added = _ingest_shape(
        build, monkeypatch, [_U, call, X, call, D], [_U, call, X, D, _V], relation_fault=True)
    assert [(r["role"], r["content"]) for r in added] == [
        ("assistant", ""), ("tool", "X"), ("tool", "D"), ("user", "next")]
    assert _replay_links(engine) == {}


def test_stamp_veto_stores_and_links_identity_equal_rows(build, monkeypatch):
    call = _poll(["c"], "{}", 2)
    engine, before, added = _ingest_shape(
        build, monkeypatch, [_U, call, X], [_U, {**call, "timestamp": 3}, X, _V])
    assert [(r["role"], r["content"]) for r in added] == [("assistant", ""), ("tool", "X"), ("user", "next")]
    assert _replay_links(engine) == {added[i]["store_id"]: before[j]["store_id"] for i, j in ((0, 1), (1, 2))}


@pytest.mark.parametrize("shape", ["vetoed", "partial"])
def test_issue_2377_preserved_rows_have_replay_links(build, monkeypatch, shape):
    a = lambda stamp: _poll(["c"], '{"a":1}', stamp)  # noqa: E731
    b = _poll(["c"], '{"b":1}', 3)
    if shape == "vetoed":
        source = [_U, a(2), X, b, X, a(4)]
        host = [_U, a(1), X, b, {**X, "content": "Y"}, a(4), X, _V]
        cursor, pairs = 2, {1: 1}
        expected = ((0, 2), (1, 3))
        expected_rows = [host[i] for i in (2, 3, 4, 6, 7)]
    else:
        call = _poll(["c", "d"], "{}", 2)
        later = {**call, "timestamp": 4}
        source = [_U, call, X, later, D]
        host = [_U, call, X, D, later, X, D, _V]
        cursor, pairs = 1, {}
        expected = ((0, 1), (1, 2), (2, 4), (3, 3))
        expected_rows = host[cursor:]
    engine, before, added = _ingest_shape(build, monkeypatch, source, host, cursor, pairs)
    assert [(r["role"], r["content"], r.get("tool_call_id")) for r in added] == [
        (r["role"], r["content"], r.get("tool_call_id")) for r in expected_rows]
    assert _replay_links(engine) == {added[i]["store_id"]: before[j]["store_id"] for i, j in expected}
