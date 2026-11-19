"""#2869: above 480,000-token windows the default summary-prefix target follows the window, clamped by the mean size of
this session's depth-0 summaries, so a level stays under the view's 100 uncondensed summaries (dag.py limit=100).

target = max(today's default, min(5% of the window, 80 x the mean depth-0 summary tokens, the threshold cap)), on
automatic sweeps only. At or below 480,000, forced and overflow passes, and explicit settings keep today's target.
The hole cell walks the grid runner on CAND_TREE (env LCMX_GRID_CAND, default this tree); the window-only target of the
superseded change is reproduced in the runner process by a mean that never clamps, and, with env LCMX_2869_UNCLAMPED
set to a tree of that change, on that tree too.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from tests import test_catchup_budget_grid as grid

pytestmark = pytest.mark.issue(2869)

MESSAGES = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"FACT-{i} " + "alpha beta gamma delta " * 44}
            for i in range(48)] + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i}"}
                                   for i in range(24)]
AGGREGATE = "SUM(LENGTH(summary))"


@pytest.fixture
def open_engine(tmp_path, monkeypatch):
    engines = []

    def open_at(window, nodes=0, mean=0, session="S", **config):
        home = tmp_path / f"h{len(engines)}"
        engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"), **config), hermes_home=str(home))
        engine.on_session_start(session, platform="cli", context_length=window)
        for _ in range(nodes):  # summary text of exactly ``mean`` tokens at 4 ASCII chars a token
            engine._dag.add_node(SummaryNode(session_id=session, depth=0, summary="x" * (4 * mean), token_count=0,
                                             source_token_count=5, source_ids=[], created_at=time.time(),
                                             expand_hint="turns"))
        monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", lambda chunk, focus_topic=None, deadline=None: (
            chunk, 100, "Earlier facts.\nExpand for details about: facts", 1, 0))
        engines.append(engine)
        return engine

    yield open_at
    for engine in engines:
        engine.shutdown()


def _target(engine, monkeypatch, force=False, current=None):
    """(sweep target, sweep active, overflow) of the first compress pass."""
    seen, prepare = [], engine._prepare_compression_sweep

    def wrapped(state, *args, **kwargs):
        out = prepare(state, *args, **kwargs)
        seen.append((state.sweep_target_tokens, state.threshold_full_sweep_active, state.force_overflow))
        return out

    monkeypatch.setattr(engine, "_prepare_compression_sweep", wrapped)
    engine.compress(MESSAGES, current_tokens=current or engine.threshold_tokens, force=force)
    return seen[0]


def _today(engine):
    return min(int(engine._config.leaf_chunk_tokens), int(engine.threshold_tokens * 0.25))


def test_benefit_1m_mean_1590_takes_the_window_share(open_engine, monkeypatch):
    engine = open_engine(1_000_000, nodes=10, mean=1_590)
    assert _target(engine, monkeypatch) == (50_000, True, False) and _today(engine) == 24_000


@pytest.mark.parametrize("window, mean, expected", [(1_000_000, 400, 32_000), (2_000_000, 800, 64_000)])
def test_clamp_80_mean_summaries(open_engine, monkeypatch, window, mean, expected):
    engine = open_engine(window, nodes=10, mean=mean)
    assert _target(engine, monkeypatch) == (expected, True, False)


def test_clamp_never_below_todays_default(open_engine, monkeypatch):
    engine = open_engine(1_000_000, nodes=10, mean=100)  # 80 x 100 = 8,000 < 24,000
    assert _target(engine, monkeypatch)[0] == _today(engine) == 24_000


def test_empty_store_keeps_todays_default(open_engine, monkeypatch):
    engine = open_engine(1_000_000)
    assert _target(engine, monkeypatch) == (_today(engine), True, False)


def test_other_sessions_summaries_do_not_count(open_engine, monkeypatch):
    engine = open_engine(1_000_000, nodes=10, mean=1_590, session="other")
    engine.on_session_start("S", platform="cli", context_length=1_000_000)
    assert _target(engine, monkeypatch)[0] == _today(engine)


@pytest.mark.parametrize("window", [64_000, 272_000, 480_000])
def test_unchanged_at_or_below_480k(open_engine, monkeypatch, window):
    engine = open_engine(window, nodes=10, mean=1_590)
    assert _target(engine, monkeypatch)[0] == _today(engine)


def test_unchanged_forced(open_engine, monkeypatch):
    engine = open_engine(1_000_000, nodes=10, mean=1_590)
    assert _target(engine, monkeypatch, force=True)[0] == _today(engine)


def test_unchanged_overflow(open_engine, monkeypatch):
    engine = open_engine(1_000_000, nodes=10, mean=1_590, max_assembly_tokens=1_000_000)
    target, _active, overflow = _target(engine, monkeypatch, current=1_200_000)
    assert overflow and target == _today(engine)


@pytest.mark.parametrize("config, expected", [({"summary_prefix_target_tokens": 30_000}, 30_000),
                                              ({"summary_prefix_target_ratio": 0.01}, 10_000)])
def test_unchanged_explicit_settings(open_engine, monkeypatch, config, expected):
    engine = open_engine(1_000_000, nodes=10, mean=1_590, **config)
    assert _target(engine, monkeypatch)[0] == expected


def test_one_indexed_aggregate_per_sweep(open_engine, monkeypatch):
    engine = open_engine(1_000_000, nodes=10, mean=1_590)
    statements = []
    engine._dag._conn.set_trace_callback(statements.append)
    try:
        _target(engine, monkeypatch)
    finally:
        engine._dag._conn.set_trace_callback(None)
    aggregate = [s for s in statements if AGGREGATE in s]
    assert len(aggregate) == 1, aggregate
    plan = " ".join(str(row[-1]) for row in engine._dag._conn.execute(
        "EXPLAIN QUERY PLAN SELECT COUNT(*), COALESCE(SUM(LENGTH(summary)), 0) FROM summary_nodes "
        "WHERE session_id = ? AND depth = 0", ("S",)))
    assert re.search(r"SEARCH summary_nodes USING (COVERING )?INDEX idx_nodes_session_depth\w* \(session_id=\? AND depth=\?\)",
                     plan), plan


# -- the hole: the view shows only the newest 100 uncondensed summaries per level ---------------------------------------

HOLE = {"group": "D", "id": "d-hole-1m", "window": 1_000_000, "pre": 0, "estimate": None, "call": 2, "delta": 0,
        "finalize": 32, "chunk": 1_500, "row_tokens": 250, "raw_tokens": 120_000, "over_tokens": 10_000,
        "growth_tokens": 20_000, "passes": 20, "turn_gap": 60, "prefix": "none", "summary_tokens": 300,
        "settings": {"dynamic_leaf_chunk_enabled": False}}
UNCLAMPED = "mods['dag'].SummaryDAG.mean_depth0_summary_tokens = lambda self, session_id: 10 ** 9\n"


def _depth0_peak(run) -> int:
    return max(r["uncondensed"].get("0", 0) for r in run["by_cell"][HOLE["id"]])


@pytest.fixture(scope="module")
def hole(tmp_path_factory):
    out = tmp_path_factory.mktemp("issue-2869")
    grid.PATCHES.setdefault("unclamped_2869", UNCLAMPED)
    runs = {"clamped": grid._run(grid.CAND_TREE, [HOLE], out, "clamped"),
            "unclamped": grid._run(grid.CAND_TREE, [HOLE], out, "unclamped", patch="unclamped_2869")}
    if os.environ.get("LCMX_2869_UNCLAMPED"):
        runs["window_only_tree"] = grid._run(Path(os.environ["LCMX_2869_UNCLAMPED"]).resolve(), [HOLE], out, "tree")
    return runs


def test_hole_premise_the_window_only_target_passes_100(hole):
    assert _depth0_peak(hole["unclamped"]) > grid.DEPTH_LIMIT, _depth0_peak(hole["unclamped"])
    if "window_only_tree" in hole:
        assert _depth0_peak(hole["window_only_tree"]) > grid.DEPTH_LIMIT, _depth0_peak(hole["window_only_tree"])


def test_hole_the_clamped_target_stays_under_100(hole):
    records = hole["clamped"]["by_cell"][HOLE["id"]]
    assert len(records) == HOLE["passes"] and sum(r["leaves"] for r in records) > grid.DEPTH_LIMIT
    assert _depth0_peak(hole["clamped"]) <= grid.DEPTH_LIMIT, _depth0_peak(hole["clamped"])
