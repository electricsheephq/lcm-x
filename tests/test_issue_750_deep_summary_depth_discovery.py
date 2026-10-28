"""#750: depth discovery must see every depth, not the first 1000 DAG rows.

``_assemble_context`` and ``_maybe_condense`` (``incremental_max_depth=-1``) discovered depths from
``SummaryDAG.get_session_nodes()``, whose default ``ORDER BY depth, created_at LIMIT 1000`` fills with
depth-0 rows first. Nodes go through the real ``SummaryDAG.add_node``; only the summariser is stubbed.
"""
from __future__ import annotations

import logging
import re

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine

FANIN = 4
PART = re.compile(r"\(d(\d+), node (\d+)\)")
PAD = " alpha beta gamma delta" * 30
SYSTEM = {"role": "system", "content": "system prompt"}
TAIL = [{"role": "user", "content": "latest question"}, {"role": "assistant", "content": "latest answer"}]


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def build(**overrides):
        settings = dict(database_path=str(tmp_path / f"lcm-{len(engines)}.db"), fresh_tail_count=2)
        settings.update(overrides)
        engine = LCMEngine(config=LCMConfig(**settings), hermes_home=str(tmp_path / "hermes"))
        engine.on_session_start("S", conversation_id="conv", context_length=200_000)
        engines.append(engine)
        return engine

    yield build
    for engine in engines:
        engine.shutdown()


def _stub_summaries(monkeypatch) -> list[int]:
    calls: list[int] = []

    def summarize(**kwargs):
        depth = int(kwargs.get("depth", 0))
        calls.append(depth)
        return ("Earlier turns." if depth == 0 else "Merged arc.") + "\nExpand for details about: arc", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _node(session_id, depth, created_at, source_ids, source_type, text="node"):
    return SummaryNode(session_id=session_id, depth=depth, summary=text, token_count=5,
                       source_token_count=500 if depth == 0 else 20, source_ids=source_ids,
                       source_type=source_type, created_at=created_at)


def _build_tree(dag, leaves: int, top_depth: int, recent: int = 0) -> dict:
    """The DAG condensation builds with fanin 4: ``leaves`` leaves condensed up to ``top_depth``,
    then ``recent`` uncondensed leaves."""
    clock = iter(range(1, 1_000_000))
    ids: dict = {0: [dag.add_node(_node("S", 0, float(next(clock)), [], "messages", f"leaf {i}"))
                     for i in range(leaves)]}
    for depth in range(1, top_depth + 1):
        children = ids[depth - 1]
        ids[depth] = [dag.add_node(_node("S", depth, float(next(clock)), children[s:s + FANIN], "nodes",
                                         f"d{depth} arc {s // FANIN}"))
                      for s in range(0, len(children) - FANIN + 1, FANIN)]
    ids["recent"] = [dag.add_node(_node("S", 0, float(next(clock)), [], "messages", f"recent {i}"))
                     for i in range(recent)]
    return ids


def _rendered(messages) -> set[tuple[int, int]]:
    return {(int(d), int(n)) for m in messages for d, n in PART.findall(str(m.get("content")))}


def test_get_session_depths_is_unbounded_and_session_scoped(tmp_path):
    dag = SummaryDAG(tmp_path / "dag.db")
    try:
        assert dag.get_session_depths("S") == []
        for i in range(1200):
            dag.add_node(_node("S", 0, float(i + 1), [], "messages"))
        dag.add_node(_node("S", 5, 5000.0, [], "nodes"))
        dag.add_node(_node("other", 7, 6000.0, [], "nodes"))
        assert dag.get_session_depths("S") == [0, 5]
        assert dag.get_session_depths("other") == [7]
    finally:
        dag.close()


@pytest.mark.parametrize("leaves", [64, 760, 768, 1024])
def test_assembly_renders_top_depth_beyond_the_first_1000_rows(make_engine, leaves):
    engine = make_engine()  # defaults: condensation_fanin=4, incremental_max_depth=3
    ids = _build_tree(engine._dag, leaves, top_depth=3)
    expected = {(3, n) for n in ids[3]}  # d3 is never condensed at the default depth: it carries all history

    rendered = _rendered(engine._assemble_context(SYSTEM, TAIL))

    assert expected and expected <= rendered, (leaves, sorted({d for d, _ in rendered}), len(rendered))


def test_assembly_keeps_session_history_next_to_recent_leaves(make_engine):
    engine = make_engine()
    ids = _build_tree(engine._dag, 1024, top_depth=3, recent=3)

    rendered = _rendered(engine._assemble_context(SYSTEM, TAIL))

    assert {(0, n) for n in ids["recent"]} <= rendered
    assert {(3, n) for n in ids[3]} <= rendered, sorted({d for d, _ in rendered})


@pytest.mark.parametrize("leaves", [64, 1024])
def test_compress_output_keeps_top_depth(make_engine, monkeypatch, caplog, leaves):
    engine = make_engine(leaf_chunk_tokens=400, context_threshold=0.001,
                         threshold_full_sweep_enabled=False, max_assembly_tokens=100_000)
    _stub_summaries(monkeypatch)
    ids = _build_tree(engine._dag, leaves, top_depth=3)
    view = [SYSTEM, *[row for i in range(6) for row in (
        {"role": "user", "content": f"[T{i}] user turn{PAD}", "timestamp": 10.0 * (i + 1)},
        {"role": "assistant", "content": f"reply to T{i}{PAD}"})]]
    engine.ingest(view)

    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        out = engine.compress(view, current_tokens=engine.threshold_tokens + 1)

    rendered = _rendered(out)
    assert {(3, n) for n in ids[3]} <= rendered, (leaves, sorted({d for d, _ in rendered}), len(rendered))


def test_unlimited_depth_condensation_sees_every_depth(make_engine, monkeypatch):
    engine = make_engine(incremental_max_depth=-1)
    ids = _build_tree(engine._dag, 1000, top_depth=1)  # 1000 condensed leaves, 250 uncondensed d1 nodes
    calls = _stub_summaries(monkeypatch)

    passes = engine._maybe_condense()

    assert passes >= 1 and engine._dag.get_session_nodes("S", depth=2, limit=100_000), (passes, calls, len(ids[1]))
    assert set(calls) == {2}  # d1 groups condensed into d2; depth 0 has nothing uncondensed
