"""Approved #2377 design v7 §2/§4: I1/I2/I8, expected behavior on main.

All acceptance rows exercise the implemented proof and attribution contracts.
"""
import sqlite3

import pytest

from hermes_lcm.dag import SummaryNode
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER as PH
from tests.invariants._ingest_shapes_2377 import D, U, V, X, call, ingest_shape, relations
from tests.invariants.test_issue_2163_compacted_cursor import build as build
from tests.test_host_uid_emit_carry import _ContainsPattern

pytestmark = pytest.mark.issue(2377)


def test_A1_vetoed_ownership(build, monkeypatch):
    a = lambda stamp: call(stamp, args='{"a":1}')  # noqa: E731
    b = call(3, args='{"b":1}')
    host = [U, a(1), X, b, dict(X, content="Y"), a(4), X, V]
    run = ingest_shape(build, monkeypatch, [U, a(2), X, b, X, a(4)], host, 2, {1: 1})
    run.check({2, 4, 6, 7})


def test_A2_partial_parallel(build, monkeypatch):
    p, q = call(2, ("c", "d")), call(4, ("c", "d"))
    host = [U, p, X, D, q, X, D, V]
    run = ingest_shape(build, monkeypatch, [U, p, X, q, D], host)
    run.check({3, 5, 7})


def summary_claim(engine, rows, *, condensed=False, partial=False):
    leaf = SummaryNode(session_id="S", summary="Earlier tool occurrence", source_ids=[
        rows[i]["store_id"] for i in ((2,) if partial else (1, 2))])
    engine._dag.add_node(leaf)
    if condensed:
        parent = SummaryNode(session_id="S", depth=1, summary="Condensed earlier facts",
                             source_ids=[leaf.node_id], source_type="nodes")
        engine._dag.add_node(parent)
        assert parent.depth >= 1 and parent.source_ids == [leaf.node_id]


def test_A3_summary_owns_full_pair(build, monkeypatch):
    host = [U, call(), X, V]  # Old pair absent; this pair is a NEW occurrence.
    run = ingest_shape(build, monkeypatch, host[:-1], host, prepare=summary_claim)
    run.check({1, 2, 3})


def test_A3b_condensed_summary_claim(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X], [U, call(), X, V],
                       prepare=lambda e, r: summary_claim(e, r, condensed=True))
    run.check({1, 2, 3})


def test_A4_result_stamps(build, monkeypatch):
    run = ingest_shape(build, monkeypatch,
                       [U, call(), dict(X, timestamp=2), call(), dict(X, timestamp=4)],
                       [U, call(), dict(X, timestamp=4), V])
    run.check({3})
    assert run.skips == {1, 2}


def test_A5_changed_prefix_result(build, monkeypatch):
    a = lambda stamp: call(stamp, ("c", "d"))  # noqa: E731
    source = [U, a(None), X, D, a(2), call(3, ("c", "d"), '{"b":1}'), X, D]
    host = [U, a(None), X, dict(D, content="Y"), a(1), X, D, V]
    run = ingest_shape(build, monkeypatch, source, host, 3)
    run.check({3, 4, 5, 6, 7})


def test_A6_astra_floor(build, monkeypatch, record_property):
    a = lambda stamp: call(stamp, ("c", "d"))  # noqa: E731
    host = [U, a(None), X, dict(D, content="Y"), a(2), X, D, V]
    run = ingest_shape(build, monkeypatch, [U, a(None), dict(X, content="Q"), D, a(2), X, D], host, 2)
    record_property("A6_outcome", run.floor({4, 5, 6}))
    run.check({2, 3, 7})


def test_A7_claude_floor(build, monkeypatch, record_property):
    a = call(None, ("c", "d", "e"))
    e = dict(X, tool_call_id="e", content="E")
    y, f, w = dict(D, content="Y"), dict(e, content="F"), dict(e, content="W")
    run = ingest_shape(build, monkeypatch, [U, a, X, D, e, a, X, y, f],
                       [U, a, X, y, w, a, X, y, f, V], 3)
    record_property("A7_outcome", run.floor({5, 6, 7, 8}))
    run.check({3, 4, 9})


def test_A8_mixed_stamp_matching(build, monkeypatch):
    k = dict(X, content="k")
    run = ingest_shape(build, monkeypatch, [U, call(), k, dict(k, timestamp=2)],
                       [U, call(), k, dict(k, timestamp=3), V])
    run.check({4})
    # A perfect assignment exists (@3 -> unstamped, unstamped -> @2), but
    # main's ceiling excludes the first result: retain it with attribution.
    assert 2 in run.inserted
    run.floor({1, 2, 3})


A9_CASES = [
    "three-parallel",
    "stamp-mismatch",
    "unstamped-order", "full-match",
    "ambiguous-shadow",
    "ambiguous-off",
    "two-ambiguous",
    "earlier-same",
    "earlier-unstamped",
    "earlier-new",
    "stamp-veto",
    "prefix-new", "2163-partial",
    "2163-vetoed",
    "consumed-twin",
    "rule-b",
    "vetoed",
    "partial",
]


@pytest.mark.parametrize("shape", A9_CASES)
def test_A9_parked_shapes(build, monkeypatch, shape):
    """18 independently selected r1-r4 shapes; v7 links replace r5 silent fallbacks."""
    p, q = call(2, ("c", "d")), call(4, ("c", "d"))
    a = call(None, ("c", "d"))
    prepare, bindings, cursor = None, {}, 1
    skips = None
    if shape == "three-parallel":
        source, host = [U, p, X, D, q, X, D], [U, dict(p, timestamp=1), X, D, p, X, D, q, X, D, V]
        new = {1, 2, 3, 10}
    elif shape == "stamp-mismatch":
        source, host, new = [U, p, X, D], [U, dict(p, timestamp=3), X, D, V], {1, 2, 3, 4}
    elif shape == "unstamped-order":
        source = [U, a, X, D, a, X, D]
        host, new, skips = [*source, a, X, D, V], {7, 8, 9, 10}, {1, 2, 3, 4, 5, 6}
    elif shape == "full-match":
        a = call()
        source, host, new = [U, a, X, a, dict(X, content="other")], [U, a, X, V], {3}
        skips = {1, 2}
    elif shape.startswith("ambiguous") or shape == "two-ambiguous":
        # The two host results are known replay rows from different partial segments.
        source = [U, a, X, a, D]
        host = [U, a, X, D, V] if shape != "two-ambiguous" else [U, a, X, D, a, X, D, V]
        new = {len(host) - 1}
        if shape == "two-ambiguous":
            source = [U, a, X, a, D, a, X, a, D]
        if shape == "ambiguous-off":
            monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    elif shape.startswith("earlier-"):
        stamp = {"earlier-same": 10, "earlier-unstamped": None, "earlier-new": 20}[shape]
        source = [U, a, dict(X, timestamp=10), a, D, call(None, args='{"b":1}'), dict(X, timestamp=10)]
        host, new = [U, a, dict(X, timestamp=stamp), D, V], {4}
        def prepare(e, rows):
            e._store.add_message_relations([[(rows[6]["store_id"], "possible_replay_of",
                                             rows[2]["store_id"], None, None)]])
    elif shape == "stamp-veto":
        source, host, new = [U, call(2), X], [U, call(3), X, V], {1, 2, 3}
    elif shape == "prefix-new":
        source = [U, call(), X, call(2), call(3, args='{"b":1}'), X]
        host, cursor, new = [U, call(), X, call(1), X, V], 3, {3, 4, 5}
    elif shape == "2163-partial":
        source, host, cursor, bindings, new = [U, p, X, D], [U, p, X, D, V], 2, {1: 1}, {4}
    elif shape in {"2163-vetoed", "vetoed"}:
        a = lambda stamp: call(stamp, args='{"a":1}')  # noqa: E731
        b = call(3, args='{"b":1}')
        source = [U, a(2), X, b, X, a(4)]
        host = [U, a(1), X, b, dict(X, content="Y"), a(4), X, V]
        cursor, bindings, new = 2, {1: 1}, {2, 4, 6, 7}
    elif shape == "consumed-twin":
        source = [U, call(2), X, call(4), X]
        host, cursor, bindings, new = [U, call(2), X, call(4), X, V], 3, {1: 1, 2: 2}, {5}
        skips = {3, 4}
    elif shape == "rule-b":
        # #2055 t2 rule b: an out-of-tail binding still reserves its prefix copy.
        source = [U, call(2), X, *[dict(U, content=f"pad {i}") for i in range(65)], call(4), X]
        host, cursor, bindings, new = [U, call(2), X, call(4), X, V], 3, {1: 1, 2: 2}, {5}
    else:
        source, host, new = [U, p, X, q, D], [U, p, X, D, q, X, D, V], {3, 5, 7}
    run = ingest_shape(build, monkeypatch, source, host, cursor, bindings, prepare=prepare)
    run.check(new)
    if shape.startswith("ambiguous") or shape == "two-ambiguous" or shape.startswith("earlier-"):
        assert set(range(1, len(host) - 1)) <= run.inserted.keys(), "no complete occurrence proves this segment"
    if skips is not None:
        assert run.skips == skips


@pytest.mark.parametrize("cursor", [1, 3, 5])
def test_A10_common_unstamped_replay(build, monkeypatch, cursor):
    source = [U, call(), X, call(), X]
    run = ingest_shape(build, monkeypatch, source, [*source, call(), X, V], cursor)
    run.check({5, 6, 7})
    assert not ({i for i in range(cursor, 5)} - run.skips)


@pytest.mark.parametrize("cursor", [1, 3, 5])
def test_A11_stamp_blind_replay(build, monkeypatch, cursor):
    source = [U, call(2), dict(X, timestamp=2), call(4), dict(X, timestamp=4)]
    host = [U, call(), X, call(), X, call(), X, V]
    run = ingest_shape(build, monkeypatch, source, host, cursor)
    run.check({5, 6, 7})
    assert not ({i for i in range(cursor, 5)} - run.skips)


def test_A11b_mixed_view_copy_rate(build, monkeypatch, record_property):
    source = [U, call(2), dict(X, timestamp=2), call(4), dict(X, timestamp=4)]
    host = [U, call(), X, call(), X, call(6), dict(X, timestamp=6), V]
    run = ingest_shape(build, monkeypatch, source, host)
    record_property("linked_copy_rate", run.linked_copy_rate({1, 2, 3, 4}))
    run.check({5, 6, 7})


def test_A12_shadow_no_double_claim(build, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    source = [U, call(), X, call(), X]
    run = ingest_shape(build, monkeypatch, source, [*source, call(), X, V], 3, {1: 1, 2: 2})
    run.check({5, 6, 7})
    assert {3, 4} <= run.skips


def test_A13_placeholder_partial_occurrences(build, monkeypatch):
    a = call(None, ("c", "d"))
    long = dict(X, content="long output " * 50)
    source = [U, a, long, D, a, long, a, D, call(None, args='{"b":1}'), dict(long, content="Y" * 600)]
    host = [U, a, dict(X, content=PH), D, a, long, D, V]
    run = ingest_shape(build, monkeypatch, source, host, 4)
    run.check({4, 5, 6, 7})


def test_A14_changed_prefix(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X, call(2)],
                       [U, call(), dict(X, content="Y"), call(1), X, V], 3)
    run.check({3, 4, 5})


def test_A15_result_only_binding(build, monkeypatch):
    a = lambda stamp: call(stamp, ("c", "d"))  # noqa: E731
    run = ingest_shape(build, monkeypatch, [U, a(None), X, D, a(2), X, D],
                       [U, a(None), X, a(1), D, V], 3, {2: 2})
    run.check({3, 4, 5})


def test_A16_literal_placeholder_is_new(build, monkeypatch):
    source = [U, call(), dict(X, content="long output " * 50)]
    host = [*source, call(3), dict(X, content=PH), V]
    run = ingest_shape(build, monkeypatch, source, host, 3)
    run.check({3, 4, 5})


def test_A17_scorer_key_rejects_false_targets(build):
    engine = build()
    targets = [dict(X, store_id=1, content="different"), dict(X, store_id=2, tool_call_id="other")]
    # Proposed pure planner: (batch-position, target-store-id) pairs.
    assert engine._plan_tool_replay_links([X], targets, seeds=[]) == []


def test_A18_transient_relation_failure_rolls_back(build, monkeypatch):
    engine = build()
    engine.ingest([U, call(), X])
    before = engine._store.get_session_messages("S")
    before_links = relations(engine)
    # Planner is the hook that guarantees this ingest attempts a relation write.
    monkeypatch.setattr(engine, "_plan_tool_replay_links", lambda *a, **kw: [(0, before[1]["store_id"]),
                                                                           (1, before[2]["store_id"])])
    write = engine._store.add_message_relations
    def busy(groups):
        write(groups)
        raise sqlite3.OperationalError("database is locked")
    with monkeypatch.context() as fault:
        fault.setattr(engine._store, "add_message_relations", busy)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            engine._ingest_messages([U, call(), X, call(4), dict(X, timestamp=4)])
    assert engine._store.get_session_messages("S") == before
    assert relations(engine) == before_links
    engine._ingest_messages([U, call(), X, call(4), dict(X, timestamp=4)])
    assert len(engine._store.get_session_messages("S")) == len(before) + 2
    assert len(relations(engine)) == len(before_links) + 2


def test_A18b_computation_failure_marks_entire_batch(build, monkeypatch):
    engine = build()
    engine.ingest([U, call(), X])
    before = engine._store.get_session_messages("S")
    def broken(*args, **kwargs):
        partial = [(0, before[1]["store_id"])]
        assert partial  # A partially computed plan must be discarded on failure.
        raise ValueError("synthetic planner failure after partial computation")
    monkeypatch.setattr(engine, "_plan_tool_replay_links", broken)
    engine.ingest([U, call(), X, call(4), dict(X, timestamp=4), V])
    assert engine._ingest_failure_count == 0
    added = engine._store.get_session_messages("S")[len(before):]
    assert len(added) == 3
    assert {(s, k, t) for s, k, t in relations(engine) if s in {r["store_id"] for r in added}} == {
        (r["store_id"], "link_pending", None) for r in added}


def test_A18c_read_only_missing_relations(build, monkeypatch, caplog):
    engine = build()
    engine.ingest([U, call(), X])
    # SQLite denies creation of the relation table, as in read-only session-end.
    engine._store._conn.execute("DROP TABLE IF EXISTS message_relations")
    conn = engine._store._conn
    conn.commit()
    def deny_schema(action, arg1, arg2, db, trigger):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TABLE and arg1 == "message_relations" else sqlite3.SQLITE_OK
    conn.set_authorizer(deny_schema)
    try:
        engine.on_session_end("S", [U, call(), X, call(4), dict(X, timestamp=4)])
        failures = engine._last_tool_link_failures
        assert len(engine._store.get_session_messages("S")) == 5
        assert len(failures) == 2
        assert all(f["source_store_id"] and f["target_store_id"] and f["reason"] for f in failures)
        assert all(str(f["source_store_id"]) in caplog.text and str(f["target_store_id"]) in caplog.text
                   for f in failures)
    finally:
        conn.set_authorizer(None)



@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2419: a warm tool-first view after [U, A, X] keeps main's positional "
                          "cursor, so its new rows are never stored (pre-existing on main)")
def test_warm_tool_first_view_is_stored(build):
    engine = build()
    engine.ingest([U, call(), X])
    before = len(engine._store.get_session_messages("S"))
    b = call(4, ids=("b",), args='{"b":1}')
    y = {"role": "tool", "tool_call_id": "b", "content": "Y", "timestamp": 4}
    engine.ingest([b, y])
    assert engine._ingest_failure_count == 0
    assert len(engine._store.get_session_messages("S")) == before + 2

@pytest.mark.parametrize("twin", ["ignored", "borrowed"])
def test_A19_ignored_borrowed_twins(build, monkeypatch, twin):
    a = call(None, ("c", "d"))
    long = dict(X, content="long output " * 50)
    other = dict(long, content="IGNORE") if twin == "ignored" else dict(X, content=PH)
    def configure(e):
        e._compiled_ignore_message_patterns = [_ContainsPattern("IGNORE")]
    source = [U, a, long, a, D]
    host = [U, a, other, D, a, long, D, V]
    run = ingest_shape(build, monkeypatch, source, host, 4, configure=configure)
    run.check({4, 5, 6, 7})


@pytest.mark.parametrize("cursor,bindings", [(2, {1: 1}), (3, {1: 1, 2: 2})],
                         ids=["cursor-helper-first", "ledger-first"])
def test_A20_reservation_counted_once(build, monkeypatch, cursor, bindings):
    source = [U, call(), X, call(), X]
    run = ingest_shape(build, monkeypatch, source, [*source, V], cursor, bindings)
    run.check({5})
    assert {3, 4} <= run.skips


@pytest.mark.parametrize("mode", ["lossy-redaction", "persisted-output"])
def test_A21_ceiling_exclusions(build, monkeypatch, mode):
    from tests.test_issue_10_persisted_output_spillover import _marker
    from hermes_lcm.ingest_persisted import _has_lossy_sensitive_redaction
    def configure(e):
        e._config.sensitive_patterns_enabled = True
        e._config.sensitive_patterns = ["password_assignment"]
    original = dict(X, content="password=abcdef" if mode == "lossy-redaction" else _marker("/missing/2377"))
    result = dict(original, content="password=ghijkl") if mode == "lossy-redaction" else original
    run = ingest_shape(build, monkeypatch, [U, call(), original], [U, call(4), result, V], configure=configure)
    if mode == "lossy-redaction":
        assert _has_lossy_sensitive_redaction(run.before[-1]["content"])
    run.check({1, 2, 3})
    assert not run.skips


@pytest.mark.parametrize("classes,stamped_call", [(64, False), (65, False), (64, True), (65, True)],
                         ids=["64-call-stamps", "65-call-stamps", "64-result-stamps", "65-result-stamps"])
def test_A22_cap_boundary(build, monkeypatch, classes, stamped_call, record_property):
    source = [U]
    for i in range(classes):
        source += [call(2 if stamped_call else i + 2), dict(X, timestamp=i + 2)]
    # Stamped-call mode remains stamp-sensitive because its call carries @2.
    # Unstamped-call mode has a stamped host suffix of the SAME identity.
    host = [U, call(2 if stamped_call else None), X, call(999), dict(X, timestamp=999),
            *[dict(V, content=f"tail padding {i}") for i in range(40)]]
    assert len(host) * 4 >= len(source), "the whole class population must be in the scanned tail"
    run = ingest_shape(build, monkeypatch, source, host)
    record_property("stored_copy_rate", len(set(run.inserted) & {1, 2}) / 2)
    record_property("linked_copy_rate", run.linked_copy_rate({1, 2}))
    run.check(set(range(3, len(host))))
    assert not ({1, 2} & run.skips), "multiple compatible classes cannot prove a skip"
    assert {1, 2} <= run.inserted.keys()


@pytest.mark.parametrize("classes", [64, 65])
@pytest.mark.parametrize("stamped_call", [False, True])
def test_A22_cap_counter(build, monkeypatch, classes, stamped_call):
    source = [U]
    for i in range(classes):
        source += [call(2 if stamped_call else i + 2), dict(X, timestamp=i + 2)]
    host = [U, call(2 if stamped_call else None), X, call(999), dict(X, timestamp=999),
            *[dict(V, content=f"tail padding {i}") for i in range(40)]]
    run = ingest_shape(build, monkeypatch, source, host)
    run.check(set(range(3, len(host))))
    assert run.engine._last_tool_class_cap_count == int(classes > 64)


def test_A23_frozen_ceiling_subset(build, monkeypatch):
    p, q = call(2, ("c", "d")), call(4, ("c", "d"))
    run = ingest_shape(build, monkeypatch, [U, p, X, q, D], [U, p, X, D, q, X, D, V])
    run.check({3, 5, 7})
    main_skips, v7_skips = run.engine._last_tool_skip_sets
    assert set(v7_skips) < set(main_skips)  # Unsafe pooled skips were removed.


def test_A24_stored_suffix_claim(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X], [U, call(), dict(X, content="Y"), call(), X, V])
    run.check({3, 4, 5})


def test_A25_restamped_resume_links(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(2), dict(X, timestamp=2)],
                       [U, call(4), dict(X, timestamp=4), V])
    assert {1, 2} <= run.inserted.keys()
    run.check({3})


def test_A26_partial_summary_claim(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(), X], [U, call(), X, V],
                       prepare=lambda e, r: summary_claim(e, r, partial=True))
    run.check({1, 2, 3})


def test_A27_borrow_spends_capacity(build, monkeypatch):
    long = dict(X, content="long output " * 50)
    host = [U, call(), dict(X, content=PH), call(), long,
            call(None, ("f0",)), dict(X, tool_call_id="f0", content="following 0"),
            call(None, ("f1",)), dict(X, tool_call_id="f1", content="following 1"), V]
    def configure(e):
        # Same durable-metadata seam as #2055's borrowed_retry_view regression.
        monkeypatch.setattr(e, "_durable_commit_proof_payload", lambda: {"output_metadata": []})
    run = ingest_shape(build, monkeypatch, [U, call(), long], host, configure=configure)
    assert {1, 2} <= run.skips, "fixture must exercise the proven #1540 borrow"
    run.check({3, 4, 5, 6, 7, 8, 9})


def test_A28_prior_ingest_target_is_last_priority(build):
    engine = build()
    targets = [dict(X, store_id=1, observed_at=30), dict(X, store_id=2, observed_at=10)]
    copies = [X, dict(X, timestamp=20)]
    assert engine._plan_tool_replay_links(copies, targets, seeds=[(99, "possible_replay_of", 2)]) == [(0, 1), (1, 2)]



@pytest.mark.parametrize("targets, seeds", [
    ([(1, 2.0), (2, 9.0)], [(2, 1)]),  # Astra r3: the fresh wrong-stamp target
    ([(1, 2.0), (3, None)], [(3, 1)]),  # Claude C1: the fresh unstamped target
    ([(1, 2.0), (5, 5.0)], [(99, 1)]),  # Claude C2b: a fresh other-stamp target
], ids=["astra-wrong-stamp", "c1-unstamped", "c2b-other-stamp"])
def test_A28c_own_equal_stamp_original_outranks_fresh_targets(build, targets, seeds):
    """An earlier ingest's use never demotes a copy's own equal-stamp original below other targets."""
    from tests.invariants.test_issue_2377_v7_oracle import _row
    rows = [_row(sid, stamp) for sid, stamp in targets]
    copies = [dict(_row(None), timestamp=2.0)]
    plan = build()._plan_tool_replay_links(
        copies, rows, seeds=[(s, "possible_replay_of", t) for s, t in seeds])
    assert plan == [(0, 1)]


def test_A28b_equal_stamp_replays_beat_prior_targets(build):
    """Astra F2: prior-used targets must not lose their equal-stamp replays to other copies."""
    from tests.invariants.test_issue_2377_v7_oracle import _row, _run
    targets = [_row(1, 2.0), _row(2, 2.0), _row(3), _row(4)]
    seeds = [(3, "possible_replay_of", 1), (4, "possible_replay_of", 2)]
    stamps = [None, None, 9.0, 2.0, 2.0]  # host 0-2 are new, host 3-4 replay the @2 originals
    copies = [dict(_row(None), timestamp=s) for s in stamps]
    plan = build()._plan_tool_replay_links(copies, targets, seeds=seeds)
    added = [_row(11 + i, s) for i, s in enumerate(stamps)]
    links = [(s, t) for s, _k, t in seeds] + [(11 + pos, t) for pos, t in plan]
    run = _run(targets, added, {i: 11 + i for i in range(5)}, links=links)
    assert run.violations({0, 1, 2}) == dict(missing=[], silent=[], invalid=[])


def test_A29_restamped_rewritten_original(build, monkeypatch):
    run = ingest_shape(build, monkeypatch, [U, call(2), dict(X, timestamp=2)],
                       [U, call(4), dict(X, content="Y", timestamp=4), call(), X, V])
    run.check({3, 4, 5})


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="named residual (exactly-once-ingest §5): two different tool pairs that share call "
                          "text and stamp; main skips the same new pair")
def test_A29b_stamp_collision_residual(build, monkeypatch):
    """Astra F1: the stored A@2·X@2 comes back re-stamped @4 and rewritten (Z), next to the real A@4·Y@4.
    Both host segments claim only the @4 class, so the free @2·X member is spent on the new A·X."""
    source = [U, call(2), dict(X, timestamp=2), call(4), dict(X, content="Y", timestamp=4)]
    host = [U, call(4), dict(X, content="Z", timestamp=4), call(4), dict(X, content="Y", timestamp=4),
            call(), X, V]
    run = ingest_shape(build, monkeypatch, source, host)
    run.check({5, 6, 7})


def test_A30_stamped_run_prefix_claim(build, monkeypatch):
    source = [U, call(2), dict(X, timestamp=2), call(4), dict(X, timestamp=4)]
    run = ingest_shape(build, monkeypatch, source, [*source, V], 3)
    run.check({5})
    assert {3, 4} <= run.skips
