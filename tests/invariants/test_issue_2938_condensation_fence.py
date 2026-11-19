"""I2/I14: condensation publishes only from live, unclaimed sources."""
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from threading import Event

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.dag import SummaryDAG, SummaryNode
from tests.test_issue_2851_rebind_cursor import _view
from tests.test_issue_581_582_store_complete_survival import _engine, _frontier, _rows, _turn

WAIT = 60
REJECTION_REASONS = ("source_missing", "source_wrong_depth", "source_already_claimed")


def _digest(text):
    return sha256(text.encode()).hexdigest()


def _summarize(*, text, depth, **kwargs):
    return f"Depth {depth}: {_digest(text)[:16]}.\nExpand for details about: turns", 1


def _new_engine(tmp_path, *, sweep, **config):
    return _engine(tmp_path, embeddings_enabled=False, temporal_rollups_enabled=False,
                   threshold_full_sweep_enabled=sweep, **config)


def _nodes(engine):
    return sorted(engine._dag.get_session_nodes("S", limit=1000), key=lambda node: node.node_id)


def _provenance(engine):
    return [tuple(row) for row in engine._dag._conn.execute(
        "SELECT node_id, escalation_level, model FROM summary_node_provenance ORDER BY node_id"
    )]


@pytest.mark.parametrize(("hold_point", "fresh_engine"), [
    ("summary", True), ("prewrite", True), ("summary", False),
])
def test_concurrent_compress(tmp_path, monkeypatch, hold_point, fresh_engine):
    held, release = Event(), Event()
    first = _new_engine(tmp_path, sweep=True)
    second = first
    rows = _view(12)

    def hold():
        if not held.is_set():
            held.set()
            assert release.wait(WAIT), "held compression was never released"

    def held_summarize(**kwargs):
        if kwargs["depth"] > 0:
            hold()
        return _summarize(**kwargs)

    add_node = first._dag.add_node

    def held_add_node(node, **kwargs):
        if node.source_type == "nodes":
            hold()
        return add_node(node, **kwargs)

    monkeypatch.setattr(engine_module, "summarize_with_escalation", _summarize)
    try:
        if fresh_engine:
            second = first.clone_for_agent()
            second.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
        seed_result = first.compress(deepcopy(rows), current_tokens=24000)
        assert [node.depth for node in _nodes(first)] == [0] * 12
        if hold_point == "summary":
            monkeypatch.setattr(engine_module, "summarize_with_escalation", held_summarize)
        else:
            monkeypatch.setattr(first._dag, "add_node", held_add_node)
        retry_input = deepcopy(rows)
        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(first.compress, deepcopy(rows), current_tokens=24000)
            try:
                assert held.wait(WAIT), "compression did not reach the selected hold point"
                retry = pool.submit(second.compress, retry_input, current_tokens=24000, bypass_cooldown=True)
                retry_result = retry.result(timeout=WAIT)
                if not fresh_engine:
                    assert retry_result is retry_input
                    assert first._compress_reentry_refusals == 1
                    assert [node.depth for node in _nodes(first)] == [0] * 12
            finally:
                release.set()
            running_result = running.result(timeout=WAIT)

        nodes = _nodes(first)
        claims = Counter((node.depth, source) for node in nodes if node.source_type == "nodes"
                         for source in node.source_ids)
        overlaps = {claim: count for claim, count in claims.items() if count > 1}
        assert not overlaps
        assert [claims[(1, source)] for source in range(1, 5)] == [1] * 4
        assert len(_rows(first)) == 48
        assert set(source for _, source in claims) <= {node.node_id for node in nodes}
        assert [row[0] for row in _provenance(first)] == [node.node_id for node in nodes]
        if fresh_engine:
            assert first.last_compression_status != "error"
            assert first._condensation_publication_rejections == {"source_already_claimed": 1}
        else:
            assert first.last_compression_status == "compacted"
            assert _solo_snapshot(first, [seed_result, running_result]) == PROBE_SOLO
    finally:
        if fresh_engine and second is not first:
            second.shutdown()
        first.shutdown()


def test_concurrent_routine_compress(tmp_path, monkeypatch):
    held, release = Event(), Event()
    first = _new_engine(tmp_path, sweep=False, condensation_fanin=2, incremental_max_depth=0)
    second = None
    stalled_rows = _view(8)
    later_rows = _view(11)

    def held_summarize(**kwargs):
        if kwargs["depth"] > 0 and not held.is_set():
            held.set()
            assert release.wait(WAIT), "routine compression was never released"
        return _summarize(**kwargs)

    monkeypatch.setattr(engine_module, "summarize_with_escalation", _summarize)
    try:
        for turns in range(4, 8):
            first.compress(deepcopy(stalled_rows[: 4 * turns]), current_tokens=24000, bypass_cooldown=True)
        leaves = [node.node_id for node in first._dag.get_uncondensed_at_depth("S", 0)]
        assert len(leaves) == 4  # Two full fan-in groups before either engine starts.
        first._config.incremental_max_depth = 1
        monkeypatch.setattr(engine_module, "summarize_with_escalation", held_summarize)

        with ThreadPoolExecutor(max_workers=2) as pool:
            running = pool.submit(first.compress, stalled_rows, current_tokens=24000, bypass_cooldown=True)
            try:
                assert held.wait(WAIT), "routine compression did not reach condensation"
                # Bind after the held engine's leaf commits and advance the raw input.
                second = first.clone_for_agent()
                second.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
                retry = pool.submit(second.compress, later_rows, current_tokens=24000, bypass_cooldown=True)
                retry.result(timeout=WAIT)
            finally:
                release.set()
            running.result(timeout=WAIT)

        nodes = _nodes(first)
        parents = [node.source_ids for node in nodes if node.depth == 1]
        assert parents == [leaves[:2]]
        assert {node.node_id for node in first._summary_frontier_nodes()} >= set(leaves[2:])
        assert first._condensation_publication_rejections == {"source_already_claimed": 1}
        assert first.last_compression_status == second.last_compression_status == "compacted"
        assert len(_rows(first)) == len(later_rows)
        assert [row[0] for row in _provenance(first)] == [node.node_id for node in nodes]
    finally:
        if second is not None:
            second.shutdown()
        first.shutdown()


@pytest.mark.parametrize("reason", REJECTION_REASONS)
def test_post_leaf_rejection_reports_partial_sweep(tmp_path, monkeypatch, reason):
    from hermes_lcm.dag import CondensationPublicationRejected

    engine = _new_engine(tmp_path, sweep=True)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _summarize)
    validate = SummaryDAG.validate_condensation_sources
    calls = 0

    def reject_after_first(conn, node, node_id):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise CondensationPublicationRejected(reason)
        validate(conn, node, node_id)

    try:
        engine.compress(_view(12), current_tokens=24000)
        monkeypatch.setattr(engine._dag, "validate_condensation_sources", reject_after_first)
        rows = _view(13)
        engine.compress(deepcopy(rows), current_tokens=24000, bypass_cooldown=True)
        engine.compress(rows, current_tokens=24000, bypass_cooldown=True)
        sweep = engine._last_threshold_full_sweep
        assert sweep["leaf_passes"] == 2
        assert sweep["condensation_passes"] == 0
        assert sweep["stop_reason"] == reason
        assert engine._condensation_publication_rejections == {reason: 3}
        assert sweep["status"] == "partial"
    finally:
        engine.shutdown()


def _snapshot(engine):
    nodes = _nodes(engine)
    return {
        "depth_counts": dict(Counter(node.depth for node in nodes)),
        "nodes_and_edges_sha256": _digest(json.dumps([
            (n.node_id, n.depth, _digest(n.summary), n.source_type, n.source_ids) for n in nodes
        ])),
        "summary_frontier": [n.node_id for n in engine._summary_frontier_nodes()],
        "raw_frontier": _frontier(engine),
    }


def _solo_snapshot(engine, results):
    return {**_snapshot(engine), "returned_lists_sha256": _digest(json.dumps(results, sort_keys=True))}


# Characterization of the two synthetic shapes on origin/main at 0ff64e73.
PROBE_SOLO = {
    "depth_counts": {0: 21, 1: 3},
    "nodes_and_edges_sha256": "4b85ab389d2404b8495f25514bf03ff5d2ec7be461cc4b546edfe5f5d2a4e1f6",
    "summary_frontier": [16, 17, 18, 19, 20, 21, 22, 23, 24, 13, 14, 15],
    "raw_frontier": 41,
    "returned_lists_sha256": "6f004799485728cd5da33776ff539584895932d9d6ab5132710213dd6f725d3c",
}
ROUTINE_SOLO = {
    "depth_counts": {0: 6, 1: 3, 2: 1},
    "nodes_and_edges_sha256": "33334cd89d2563e7a7430d5af9815f37c2a3f1b0b8d1abe6229b7117acfaf3d7",
    "summary_frontier": [10, 7],
    "raw_frontier": 26,
    "returned_lists_sha256": "3c55a239534e6db231126ea7c461e8cca0b23445443d5c9551f1753817b6f432",
}


@pytest.mark.parametrize("shape", ["probe", "routine"])
def test_solo_output(tmp_path, monkeypatch, shape):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", _summarize)
    probe = shape == "probe"
    config = {} if probe else {"condensation_fanin": 2, "incremental_max_depth": 4}
    engine = _new_engine(tmp_path, sweep=probe, **config)
    try:
        if probe:
            results = [engine.compress(_view(12), current_tokens=24000) for _ in range(2)]
        else:
            rows = [row for i in range(14) for row in _turn(f"ordinary-{i}", 1000.0 + i * 10)]
            for i, row in enumerate(rows):
                row.update(message_uid=f"ordinary-{i}", timestamp=1000.0 + i * 2)
            results = [
                engine.compress(deepcopy(rows[: 2 * turns]), current_tokens=24000, bypass_cooldown=True)
                for turns in range(4, 15)
            ]
        snapshot = _solo_snapshot(engine, results)
        assert snapshot == (PROBE_SOLO if probe else ROUTINE_SOLO), json.dumps(snapshot)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("reason", REJECTION_REASONS)
def test_rollback_complete(tmp_path, monkeypatch, reason):
    rejected_text = "rejectedcondensation"
    fts_match = "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ?"
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: (rejected_text, 1))
    engine = _new_engine(tmp_path, sweep=False, condensation_fanin=2, incremental_max_depth=1)
    dag = engine._dag
    for index in range(2):
        dag.add_node(SummaryNode(session_id="S", summary=f"leaf {index}", token_count=100))
    source_id = _nodes(engine)[-1].node_id
    add_node = dag.add_node
    before = {}

    def change_source(node, **kwargs):
        conn = dag._conn
        with conn:
            if reason == "source_missing":
                conn.execute("DELETE FROM summary_nodes WHERE node_id = ?", (source_id,))
            elif reason == "source_wrong_depth":
                conn.execute("UPDATE summary_nodes SET depth = 1 WHERE node_id = ?", (source_id,))
            else:
                parent = SummaryNode(session_id="S", depth=1, summary="existing parent", token_count=1,
                                     source_type="nodes", source_ids=[source_id])
                add_node(parent, escalation_level=1)
        before.update(snapshot=_snapshot(engine), provenance=_provenance(engine))
        if (validate := kwargs.get("before_commit")) is not None:
            def observe_validation(publication_conn, node_id):
                assert publication_conn is conn and publication_conn.in_transaction
                assert _provenance(engine)[-1][0] == node_id
                assert conn.execute(fts_match, (rejected_text,)).fetchall() == [(node_id,)]
                before["validated_inside_transaction"] = True
                validate(publication_conn, node_id)

            kwargs["before_commit"] = observe_validation
        return add_node(node, **kwargs)

    monkeypatch.setattr(dag, "add_node", change_source)
    try:
        assert engine._maybe_condense() == 0
        assert engine._condensation_publication_rejections == {reason: 1}
        assert before["validated_inside_transaction"]
        assert _snapshot(engine) == before["snapshot"]
        assert _provenance(engine) == before["provenance"]
        assert dag._conn.execute(fts_match, (rejected_text,)).fetchall() == []
        assert not dag._conn.in_transaction
    finally:
        engine.shutdown()


def test_claimant_query_uses_session_depth_index(tmp_path):
    dag = SummaryDAG(tmp_path / "lcm.db")
    conn = dag._conn
    try:
        leaf_id = dag.add_node(SummaryNode(session_id="S", summary="leaf", token_count=2))
        dag.add_node(SummaryNode(session_id="other", depth=1, summary="unrelated", token_count=1,
                                 source_type="nodes", source_ids=[leaf_id + 100]))
        parent = SummaryNode(session_id="S", depth=1, summary="parent", token_count=1,
                             source_type="nodes", source_ids=[leaf_id])
        statements = []
        conn.set_trace_callback(statements.append)
        try:
            dag.add_node(parent, before_commit=lambda c, n: dag.validate_condensation_sources(c, parent, n))
        finally:
            conn.set_trace_callback(None)
        query, = [stmt for stmt in statements if "FROM summary_nodes AS parent" in stmt]
        plan = [row[-1] for row in conn.execute("EXPLAIN QUERY PLAN " + query)]
        assert any(step.startswith("SEARCH parent USING INDEX idx_nodes_session_depth ")
                   and "(session_id=? AND depth=?)" in step for step in plan), plan
    finally:
        dag.close()


def test_fresh_engine_ingest_lock_propagates_on_recovery(tmp_path):
    """On exhausted ingest admission, a bypass-cooldown attempt can raise."""
    first = _new_engine(tmp_path, sweep=True)
    second = first.clone_for_agent()
    try:
        second.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
        second._store.connection.execute("PRAGMA busy_timeout=100")
        rows = _view(12)
        with first._store.ingest_atomic():
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                second.compress(rows, current_tokens=24000, bypass_cooldown=True)
        assert second.last_compression_status == "error"
        assert rows == _view(12)
        assert not second._store.connection.in_transaction
        assert not _rows(first)
        assert not _nodes(first)
    finally:
        second.shutdown()
        first.shutdown()
