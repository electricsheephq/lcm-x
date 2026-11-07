"""Pin the product scorer key to the independent release oracle."""
import json
import sqlite3

import pytest

from hermes_lcm.ingest_ledger import tool_replay_scorer_key
from hermes_lcm.reconcile import _commit_proof_identity_digest, _proof_user_identity
from tests.invariants._ingest_shapes_2377 import U, V, X, call, ingest_shape, key, relations
from tests.invariants.test_issue_2163_compacted_cursor import build as build


@pytest.mark.parametrize("row", [X, call(), dict(X, tool_name="lookup"),
                                      dict(call(), tool_calls=json.dumps(call()["tool_calls"]))],
                         ids=["result", "call", "named-result", "serialized-call"])
def test_product_key_matches_independent_oracle(row):
    assert tool_replay_scorer_key(row, conversation="conversation") == key(row)


@pytest.mark.parametrize("occurrences", [64, 128])
def test_distinct_result_stamps_do_not_normalize_every_class_per_segment(build, monkeypatch, occurrences):
    normalized = 0

    def configure(engine):
        original = engine._message_replay_identity

        def counted(*args, **kwargs):
            nonlocal normalized
            normalized += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, "_message_replay_identity", counted)

    source = [U, *[row for stamp in range(2, occurrences + 2) for row in (call(), dict(X, timestamp=stamp))]]
    run = ingest_shape(build, monkeypatch, source, [*source, V], configure=configure)
    run.check({len(source)})
    assert run.skips == set(range(1, len(source)))
    assert normalized <= 64 * occurrences, "incompatible stamped classes must not cause quadratic identity work"


def test_unstamped_claim_has_no_equal_stamp_evidence_in_a_mixed_view(build, monkeypatch):
    source = [U, call(2), X, call(), dict(X, content="earlier alternate result")]
    host = [U, call(), dict(X, content="earlier alternate result"),
            call(), dict(X, content="rewritten result"), call(), X,
            call(6), dict(X, content="fresh stamped result"), V]
    run = ingest_shape(build, monkeypatch, source, host)
    # Both originals precede these genuinely new rows. Missing call stamps
    # cannot narrow ownership to just the stored unstamped class.
    run.check(set(range(5, len(host))))


def test_lineage_writer_target_is_already_spent_in_this_ingest(build, monkeypatch):
    engine = build()
    engine.ingest([U, call(), X])
    target = engine._store.get_session_messages("S")[-1]["store_id"]
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_ingest_ledger_attempt", {"link_targets_used": {target}}, raising=False)
        assert engine._prepare_tool_replay_links([X], 64) == []
    assert engine._prepare_tool_replay_links([X], 64) == [(0, target)]


@pytest.mark.parametrize("ledger", ["shadow", "off"])
def test_nontransient_partial_write_keeps_rows_and_logs_each_link(build, monkeypatch, caplog, ledger):
    monkeypatch.setenv("LCM_INGEST_LEDGER", ledger)
    engine = build()
    original = [U, call(2), dict(X, timestamp=2)]
    engine.ingest(original)
    write = engine._store.add_message_relations
    calls = 0
    metadata = {idx: ("S", dict(timestamp=2, identity_sha256=_commit_proof_identity_digest(
        _proof_user_identity(engine._message_replay_identity(original[idx]))))) for idx in (1, 2)}
    restamp = engine._link_restamped_occurrences

    def competing_writer(*args, **kwargs):
        # The snapshot matcher supplies exact old occurrences. Exercise the
        # real later writer, including when ledger instrumentation is off.
        args = list(args)
        args[2] = metadata
        return restamp(*args, **kwargs)

    def failed(groups):
        nonlocal calls
        if any(row[1] == "possible_replay_of" for group in groups for row in group):
            calls += 1
            if calls == 1:
                write(groups[:1])
                raise ValueError("synthetic deterministic write fault")
        return write(groups)

    monkeypatch.setattr(engine._store, "add_message_relations", failed)
    monkeypatch.setattr(engine, "_link_restamped_occurrences", competing_writer)
    engine.ingest([U, call(4), dict(X, timestamp=4), V])
    assert engine._store.get_session_count("S") == 6
    assert relations(engine) == []  # The partial link rolled back, the rows committed.
    assert engine._ingest_failure_count == 0
    assert len(engine._last_tool_link_failures) == 2
    assert engine._ingest_ledger_counters["ledger.tool_link_failures"] == 2
    assert calls == 1
    assert sum(r.message.startswith("LCM tool replay link failed ") for r in caplog.records) == 2


def test_pending_counters_publish_only_committed_rows(build, monkeypatch):
    engine = build()
    engine.ingest([U, call(), X])

    def failed(*args, **kwargs):
        raise ValueError("synthetic deterministic planner fault")

    monkeypatch.setattr(engine, "_plan_tool_replay_links", failed)
    engine.ingest([U, call(), X, call(4), dict(X, timestamp=4), V])
    assert engine._last_tool_link_pending_count == 3
    assert engine._ingest_ledger_counters["ledger.tool_link_pending"] == 3
    assert len([r for r in relations(engine) if r[1] == "link_pending"]) == 3
    engine.ingest([U, call(), X, call(4), dict(X, timestamp=4), V, dict(V, timestamp=1001)])
    assert engine._last_tool_link_pending_count == 0
    assert engine._ingest_ledger_counters["ledger.tool_link_pending"] == 3


def test_computation_cancellation_rolls_back_before_retry(build, monkeypatch):
    engine = build()
    engine.ingest([U, call(), X])
    before = engine._store.get_session_messages("S")

    def cancelled(*args, **kwargs):
        error = sqlite3.OperationalError("interrupted")
        error.sqlite_errorcode = sqlite3.SQLITE_INTERRUPT
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_plan_tool_replay_links", cancelled)
        with pytest.raises(sqlite3.OperationalError, match="interrupted"):
            engine._ingest_messages([U, call(), X, call(4), dict(X, timestamp=4)])
    assert engine._store.get_session_messages("S") == before
    assert relations(engine) == []
    engine._ingest_messages([U, call(), X, call(4), dict(X, timestamp=4)])
    assert engine._store.get_session_count("S") == 5
    assert len(relations(engine)) == 2


_RECURSIVE_SUMMARY_WALK = """
    WITH RECURSIVE walk(kind,id) AS (
        SELECT n.source_type, CAST(j.value AS INTEGER)
        FROM summary_nodes n, json_each(n.source_ids) j
        UNION SELECT n.source_type, CAST(j.value AS INTEGER)
        FROM summary_nodes n JOIN walk w ON w.kind='nodes' AND n.node_id=w.id
        JOIN json_each(n.source_ids) j
    ) SELECT DISTINCT id FROM walk WHERE kind='messages'"""


def test_summary_coverage_matches_the_recursive_walk(build):
    """Every node is a root, so the flat read of message sources equals the transitive walk."""
    from hermes_lcm.dag import SummaryNode
    from hermes_lcm.reconcile import _summary_covered_message_ids

    engine = build()
    engine.ingest([U, call(), X, call(2), dict(X, timestamp=2), V])
    ids = [r["store_id"] for r in engine._store.get_session_messages("S")]
    leaves = []
    for chunk in (ids[0:2], ids[2:4], [ids[4], 10**6]):  # one source id has no message row
        leaf = SummaryNode(session_id="S", summary="leaf", source_ids=chunk)
        engine._dag.add_node(leaf)
        leaves.append(leaf.node_id)
    node = SummaryNode(session_id="S", depth=1, summary="d1", source_ids=leaves[:2], source_type="nodes")
    engine._dag.add_node(node)
    for depth in range(2, 9):  # a deep condensed chain, each level also naming a missing node
        node = SummaryNode(session_id="S", depth=depth, summary=f"d{depth}",
                           source_ids=[node.node_id, 10**6 + depth], source_type="nodes")
        engine._dag.add_node(node)
    conn = engine._store._conn
    expected = {row[0] for row in conn.execute(_RECURSIVE_SUMMARY_WALK)}
    assert expected == {*ids[:5], 10**6}
    assert _summary_covered_message_ids(conn) == expected


@pytest.mark.parametrize("stored_id,links", [
    ("' ' || conversation_id || ' '", True),  # a padded id for the same conversation still matches
    ("NULL", False),  # a legacy row with no conversation is not a wildcard for the active one
    ("''", False),
], ids=["padded", "null", "empty"])
def test_blank_stored_conversation_is_not_the_active_one(build, stored_id, links):
    engine = build()
    assert engine._conversation_id
    engine.ingest([U, call(), X])
    targets = [r["store_id"] for r in engine._store.get_session_messages("S")[1:]]
    conn = engine._store._conn
    conn.execute(f"UPDATE messages SET conversation_id = {stored_id} WHERE session_id = 'S'")
    conn.commit()
    engine.ingest([U, call(), X, call(), X])
    linked = {target for _source, kind, target in relations(engine) if kind == "possible_replay_of"}
    assert (linked == set(targets)) if links else not (linked & set(targets))
