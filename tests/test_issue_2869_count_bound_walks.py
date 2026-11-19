"""#2869 raised mode, walked with the grid runner and checked at EVERY condensation check, not only at a final peak.

Raised mode (above 480,000-token windows, automatic sweeps, no explicit target, background leaf preparation off, not a
recovery pass): a check condenses iff the frontier is over today's target AND (over the raised target OR more than 80
nodes are live). The harness patch logs every sweep start, every check that stops condensing, every condensation and
every abnormal stop, and the cells assert, from that log:
- P1: every condensation starts with the frontier over today's target (main would condense there too);
- P2: a check that stops condensing has at most 80 live nodes, or the frontier at or under today's target;
- P3: a sweep start after a normal stop with at most 80 live holds at most 80 plus the leaves stored since (<= 12);
- the view (100 a level) is never exceeded, and no row leaves the list without a pointer (A0 0).
K1: the skew drift shapes (big leaves first, then 300-token leaves). K6: the hole cell, against the window-only target of
the superseded change, reproduced here by a live bound that never binds (and on that tree when env LCMX_2869_UNCLAMPED
names it). The runner walks CAND_TREE (env LCMX_GRID_CAND, default this tree).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests import test_catchup_budget_grid as grid

pytestmark = pytest.mark.issue(2869)

BASE_CELL = {"group": "D", "pre": 0, "estimate": None, "call": 2, "delta": 0, "finalize": 32, "chunk": 1_500,
             "row_tokens": 250, "raw_tokens": 120_000, "over_tokens": 10_000, "growth_tokens": 20_000, "turn_gap": 60,
             "prefix": "none", "summary_tokens": 300, "settings": {"dynamic_leaf_chunk_enabled": False}}
HOLE = {**BASE_CELL, "id": "d-hole-1m", "window": 1_000_000, "passes": 20}
SKEWS = {  # cell id: (window, passes, big leaves, their size); every later leaf and every condensation is 300 tokens
    "k1-40x1000": (1_000_000, 30, 40, 1_000), "k1-44x1100": (1_000_000, 30, 44, 1_100),
    "k1-40x1250": (1_000_000, 30, 40, 1_250), "k1-2m-70x1200": (2_000_000, 40, 70, 1_200)}
MAX_LIVE, SWEEP_LEAVES = 80, 12

SKEW_PATCH = """
import sys
_M = sys.modules["__main__"]
BIG_N, BIG, SMALL = {big_n}, {big}, 300
_Base = _M.Provider
class _Skew(_Base):
    leaf_calls = 0
    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        if "user turn" in prompt:
            _Skew.leaf_calls += 1
            self.summary = _M.sized_summary(BIG if _Skew.leaf_calls <= BIG_N else SMALL)
        else:
            self.summary = _M.sized_summary(SMALL)
        return _Base.__call__(self, prompt, max_tokens, model, timeout, reasoning_effort)
_M.Provider = _Skew
"""
HARNESS = """
import json as _json
_LOG = {log!r}
_E = mods["engine"].LCMEngine
_frontier, _prep, _due = _E._summary_frontier_tokens, _E._prepare_compression_sweep, _E._sweep_condensation_due
_pub, _cond, _run = _E._publish_leaf_node, _E._condense_summary_nodes, _E._run_threshold_sweep_condensation
_S = {{"since": 0, "last": None, "target": 0, "raised": 0}}
def _log(**event):
    with open(_LOG, "a") as handle:
        handle.write(_json.dumps(event) + "\\n")
def _p(self, state, *a, **k):
    out = _prep(self, state, *a, **k)
    if state.threshold_full_sweep_active:
        tokens, live = _frontier(self, with_count=True)
        _S["target"], _S["raised"] = state.sweep_target_tokens, getattr(state.budget, "sweep_raise_tokens", 0)
        depth0 = len(self._dag.get_uncondensed_at_depth(self._session_id, 0, limit=10 ** 6))
        _log(ev="start", tokens=tokens, live=live, depth0=depth0, since=_S["since"], last=_S["last"],
             target=_S["target"], raised=_S["raised"])
    return out
def _d(self, target, raise_tokens):
    due = _due(self, target, raise_tokens)
    tokens, live = _frontier(self, with_count=True)
    if due:
        _S["last"] = "pending"
    else:
        _S["since"], _S["last"] = 0, ("live" if live <= {max_live} else "floor")
        _log(ev="stop", tokens=tokens, live=live, target=target)
    return due
def _u(self, *a, **k):
    out = _pub(self, *a, **k)
    _S["since"] += 1
    return out
def _c(self, *a, **k):
    tokens, live = _frontier(self, with_count=True)
    _log(ev="condense", tokens=tokens, live=live, target=_S["target"], raised=_S["raised"])
    return _cond(self, *a, **k)
def _r(self, *a, **k):
    out = _run(self, *a, **k)
    if out[1] != "summary_prefix_target_reached":
        _S["last"] = "abnormal"
        _log(ev="abnormal", reason=out[1])
    return out
(_E._prepare_compression_sweep, _E._sweep_condensation_due, _E._publish_leaf_node, _E._condense_summary_nodes,
 _E._run_threshold_sweep_condensation) = _p, _d, _u, _c, _r
"""
WINDOW_ONLY = "mods['engine'].RAISED_MAX_LIVE = 10 ** 9  # the live bound never binds: the window-only target\n"


def _skew_cell(cell_id):
    window, passes, _n, _size = SKEWS[cell_id]
    return {**BASE_CELL, "id": cell_id, "window": window, "passes": passes}


@pytest.fixture(scope="module")
def walks(tmp_path_factory):
    out = tmp_path_factory.mktemp("issue-2869-walks")
    runs = {}
    extra = grid._extra_paths()
    with pytest.MonkeyPatch.context() as patch:
        if not any((Path(entry) / "agent" / "context_engine.py").is_file() for entry in extra):
            # CI writes the host stub at the repo root, inside the tree the runner may not import from: hand it a copy.
            import agent.context_engine as host
            shutil.copytree(Path(host.__file__).parent, out / "host" / "agent")
            patch.setattr(grid, "_extra_paths", lambda: [*extra, str(out / "host")])
        for name, cell, prefix in ([(cell_id, _skew_cell(cell_id), SKEW_PATCH.format(big_n=SKEWS[cell_id][2],
                                                                                    big=SKEWS[cell_id][3]))
                                    for cell_id in SKEWS] + [("hole", HOLE, "")]):
            log = out / f"{name}.jsonl"
            key = f"k2869_{name}"
            patch.setitem(grid.PATCHES, key, prefix + HARNESS.format(log=str(log), max_live=MAX_LIVE))
            run = grid._run(grid.CAND_TREE, [cell], out, name, patch=key)
            events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            runs[name] = (run["by_cell"][cell["id"]], events)
        patch.setitem(grid.PATCHES, "k2869_window_only", WINDOW_ONLY)
        runs["window_only"] = (grid._run(grid.CAND_TREE, [HOLE], out, "window-only",
                                         patch="k2869_window_only")["by_cell"][HOLE["id"]], [])
        if os.environ.get("LCMX_2869_UNCLAMPED"):
            runs["window_only_tree"] = (grid._run(Path(os.environ["LCMX_2869_UNCLAMPED"]).resolve(), [HOLE], out,
                                                  "tree")["by_cell"][HOLE["id"]], [])
    return runs


def _peak(records) -> int:
    return max(n for r in records for n in r["uncondensed"].values())


def _check_every_check(records, events, name):
    raised = [e for e in events if e["ev"] == "start" and e["raised"]]
    assert raised, f"{name}: no raised-mode sweep"
    p1 = [e for e in events if e["ev"] == "condense" and e["raised"] and not e["tokens"] > e["target"]]
    assert not p1, f"{name}: P1, a condensation main would not start: {p1[:3]}"
    p2 = [e for e in events if e["ev"] == "stop" and not (e["live"] <= MAX_LIVE or e["tokens"] <= e["target"])]
    assert not p2, f"{name}: P2: {p2[:3]}"
    p3 = [e for e in raised if e["last"] == "live" and not (e["live"] <= MAX_LIVE + e["since"]
                                                            and e["since"] <= SWEEP_LEAVES)]
    assert not p3, f"{name}: P3, sweep starts: {p3[:3]}"
    assert any(e["last"] == "live" for e in raised), f"{name}: P3 never applied"
    assert _peak(records) <= grid.DEPTH_LIMIT, f"{name}: a level held {_peak(records)} uncondensed summaries"
    a0 = sum(r["loss"]["missing_count"] for r in records if r["loss"])
    assert a0 == 0, f"{name}: A0 {a0}"


@pytest.mark.parametrize("name", list(SKEWS))
def test_k1_skew_drift_bounded_at_every_check(walks, name):
    records, events = walks[name]
    assert len(records) == SKEWS[name][1]
    _check_every_check(records, events, name)
    starts = [e["depth0"] for e in events if e["ev"] == "start" and e["raised"] and e["last"] == "live"]
    assert max(starts) <= MAX_LIVE + SWEEP_LEAVES, starts


def test_k6_hole_window_only_target_passes_100(walks):
    assert _peak(walks["window_only"][0]) > grid.DEPTH_LIMIT, _peak(walks["window_only"][0])
    if "window_only_tree" in walks:
        assert _peak(walks["window_only_tree"][0]) > grid.DEPTH_LIMIT, _peak(walks["window_only_tree"][0])


def test_k6_hole_raised_mode_stays_at_92(walks):
    records, events = walks["hole"]
    assert len(records) == HOLE["passes"] and sum(r["leaves"] for r in records) > grid.DEPTH_LIMIT
    _check_every_check(records, events, "hole")
    assert _peak(records) <= MAX_LIVE + SWEEP_LEAVES, _peak(records)
