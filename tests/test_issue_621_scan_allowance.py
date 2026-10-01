"""#621: excluded prefixes do not spend the first tool group's scan allowance."""

import logging
from collections import Counter

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.store_complete as store_complete
from hermes_lcm.dag import SummaryNode
from tests.test_issue_581_582_store_complete_survival import (
    _covered,
    _engine,
    _frontier,
    _rows,
    _run_scan_cap_leaves,
    _turn,
    _unmatched_calls,
)


@pytest.fixture
def summaries(monkeypatch):
    captured = []

    def summarize(**kwargs):
        captured.append(kwargs["text"])
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return captured


@pytest.mark.parametrize("cap,prefix,calls", [(2000, 1999, 6001), (10, 9, 31), (10, 9, 30), (10, 0, 31)],
                         ids=["product-failure", "scaled-failure", "covered-control", "no-prefix-control"])
def test_first_group_allowance_starts_at_call(tmp_path, monkeypatch, caplog, summaries, cap, prefix, calls):
    call_id, first, covered, lines, serialized = _run_scan_cap_leaves(
        tmp_path, monkeypatch, caplog, cap, prefix + 1, calls=calls, covered_before=bool(prefix))
    assert set(range(call_id, call_id + calls + 1)) <= first
    assert set(range(call_id, call_id + calls + 1)) <= covered
    assert not any("admitted as read" in line for line in lines)
    assert not any(_unmatched_calls(messages) for messages in serialized)


def _covered_prefix(engine, count):
    for i in range(count):
        engine._store.append("S", {"role": "user", "content": f"covered {i}", "timestamp": float(i)},
                             conversation_id="conv")
    ids = [int(row["store_id"]) for row in _rows(engine)]
    engine._dag.add_node(SummaryNode(session_id="OTHER", summary="covered prefix", token_count=1,
                                     source_token_count=1, source_ids=ids))
    return ids


def test_covered_run_above_frontier_publishes_one_leaf(tmp_path, monkeypatch, summaries):
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", 10)
    engine = _engine(tmp_path, leaf_chunk_tokens=500_000)
    try:
        prefix = _covered_prefix(engine, 11)
        hidden, tail = _turn("uncovered", 100.0), _turn("fresh", 200.0)
        engine.ingest([*hidden, *tail])
        before = _rows(engine)
        uncovered = {int(row["store_id"]) for row in before if row["content"] in {m["content"] for m in hidden}}
        assert _frontier(engine) == 0
        engine.compress(list(tail))
        assert engine._last_compression_status == "compacted", engine._last_compression_noop_reason
        assert len(engine._dag.get_session_nodes("S")) == 1
        assert set(_covered(engine)) == uncovered
        assert not set(prefix) & set(_covered(engine))
        assert _frontier(engine) == max(uncovered)
        every = Counter(int(row[0]) for row in engine._store.connection.execute(
            "SELECT source.value FROM summary_nodes AS node, json_each(node.source_ids) AS source "
            "WHERE node.source_type = 'messages'").fetchall())
        assert all(every[store_id] == 1 for store_id in [*prefix, *uncovered])
        assert _rows(engine) == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("kind", ["covered", "ignored", "passive", "system"])
@pytest.mark.parametrize("bounded", [False, True], ids=["reaches-uncovered", "bounded-noop"])
def test_excluded_pages_are_bounded(tmp_path, monkeypatch, summaries, caplog, kind, bounded):
    cap = 10
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", cap)
    engine = _engine(tmp_path)
    try:
        count = cap * (store_complete._SCAN_EXTEND + 2) if bounded else cap + 1
        prefix = _covered_prefix(engine, count) if kind == "covered" else [
            engine._store.append("S", {"role": "system" if kind == "system" else "user",
                                       "content": f"excluded {i}"}, conversation_id="conv")
            for i in range(count)]
        if kind == "ignored":
            monkeypatch.setattr(engine, "_matches_ignore_message_patterns",
                                lambda row, **k: row.get("content", "").startswith("excluded"))
        uncovered = None if bounded else engine._store.append(
            "S", {"role": "user", "content": "uncovered turn"}, conversation_id="conv")
        reads = []
        real = engine._store.get_range

        def get_range(*args, **kwargs):
            reads.append(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(engine._store, "get_range", get_range)
        with caplog.at_level(logging.WARNING):
            result = engine._store_complete_input([], {}, {}, [], [], 0, [], 500_000,
                                                   accounted_ids=prefix if kind == "passive" else ())
        assert result == [] if bounded else [ids for _message, ids in result] == [[uncovered]]
        assert len(reads) == (store_complete._SCAN_EXTEND if bounded else 2)
        assert all(read["limit"] == cap for read in reads)
        assert [read["start_id"] for read in reads] == list(range(1, 1 + cap * len(reads), cap))
        assert _frontier(engine) == 0
        assert not _covered(engine)
        assert not any("admitted as read" in record.getMessage() for record in caplog.records)
    finally:
        engine.shutdown()
