"""Two-tree catch-up budget grid: never less progress than the base, plus a per-pass loss oracle.

tests/behaviour/catchup_grid_runner.py walks every cell below on BASE_TREE, then on CAND_TREE, one isolated
process after the other (``-I``: each imports the engine only from its own tree). The clock is simulated; the
stand-in summariser honours its timeout. The assertions read the per-pass records:

A0 loss with no pointer and A0b rows behind a survival-fit notice (both trees; the known base cells are pinned
xfail(strict) "issue 2877"; the candidate has no A0 loss and no more A0b rows than the base), A1 never less (G1, G2,
S, including hold arming), A2 walls (both trees), A3 equal
records at windows <= 480,000 (L), A4 commits per drained token (G2, and the condensation-then-leaf cell), A5 the
hang cell, A6 a bounded summary area (S, the steady cells). A4's strict cells are xfail(strict=True) while CAND_TREE
has no catch-up constant.

BASE_TREE: env LCMX_GRID_BASE, a separate copy of the base commit (for example a git archive); with it unset (CI has
no base tree) every two-tree test skips, and the runner and the single-tree helpers still import. CAND_TREE: env
LCMX_GRID_CAND (default this tree). LCMX_GRID_OUT: a directory for
the run outputs (default the pytest temp dir).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "behaviour" / "catchup_grid_runner.py"
BASE_TREE = Path(os.environ["LCMX_GRID_BASE"]).resolve() if os.environ.get("LCMX_GRID_BASE") else None
NO_BASE = "two-tree grid: set LCMX_GRID_BASE to a copy of the base commit's tree (CI has none)"
CAND_TREE = Path(os.environ.get("LCMX_GRID_CAND", str(HERE.parent))).resolve()
RUN_TIMEOUT = 900  # seconds per runner process
WALL_PLUGIN, WALL_HOST, HOST_COMMIT = 250.0, 270.0, 17.0
CATCHUP_HARD, BASE_HARD = 210.0, 120.0
STRICT = "needs the catch-up change"

# -- cells ---------------------------------------------------------------------------------------------------------

WINDOWS, PRE, ESTIMATE = [600_000, 1_000_000], [0, 37, 59, 88, 95], [None, 20, 44, 58, 115]
CALL, DELTA = [10, 40, 45, 115, "hang"], [0, 15]
PREFIX = ["under", "1-2x", "over2x", "over2x_nogroup"]
G1_FIXED = {"group": "G1", "passes": 10, "raw_tokens": 200_000, "turn_gap": 60, "finalize": 32, "chunk": 24_000}
G1_GROWTH, G1_OVER = 0.03, 0.1  # of the window: tokens added between passes, and the first prompt over the trigger
A6_FROM, A6_TO = 40, 60  # passes 41-60 (1-based)


def _g1_load(window: int) -> dict:
    return {"growth_tokens": int(G1_GROWTH * window), "over_tokens": int(G1_OVER * window)}


def _g1_cells() -> list[dict]:
    """Strength-2 covering array: OA(25, 6, 5, 2) over GF(5), the 2- and 4-level factors folded by ``mod``.

    Rows (i, j); columns i, j, i+j, i+2j, i+3j, i+4j (mod 5) take every pair of values once, and a surjective fold
    of a column keeps every pair covered. The full product is 2,000 cells (see the summary for its estimated time).
    """
    cells = []
    for i in range(5):
        for j in range(5):
            window = WINDOWS[(i + 2 * j) % 5 % 2]
            cells.append({**G1_FIXED, **_g1_load(window), "id": f"g1-{i}{j}", "pre": PRE[i], "estimate": ESTIMATE[j],
                          "call": CALL[(i + j) % 5], "window": window,
                          "delta": DELTA[(i + 3 * j) % 5 % 2], "prefix": PREFIX[(i + 4 * j) % 5 % 4]})
    cells += [
        # a later call cut at the cap (the first stores a leaf, the second hangs), then the next turn's pass
        {**G1_FIXED, **_g1_load(1_000_000), "id": "g1-cut-then-next", "window": 1_000_000, "pre": 0, "estimate": 20, "call": [10, "hang"],
         "delta": 0, "prefix": "under"},
        # the user's wait: every call after the first of a pass hangs, on every pass
        {**G1_FIXED, **_g1_load(1_000_000), "id": "g1-hang-every-pass", "window": 1_000_000, "pre": 37, "estimate": 44,
         "call": [40, "hang"], "delta": 0, "prefix": "under"},
        # prefix over target: a 44 s condensation, then a 40 s leaf the base's soft target refuses
        {**G1_FIXED, **_g1_load(1_000_000), "id": "g1-cond-then-leaf", "window": 1_000_000, "pre": 37, "estimate": None, "call": [44, 40],
         "delta": 0, "prefix": "over2x"},
    ]
    return cells


def _steady_cells() -> list[dict]:
    """A6: 60 passes with turns between them, growth a little over the base's one leaf a pass, sized summaries."""
    steady = {"group": "S", "passes": A6_TO, "raw_tokens": 150_000, "over_tokens": 20_000, "growth_tokens": 26_000,
              "turn_gap": 60, "finalize": 32, "chunk": 24_000, "summary_tokens": 1_500, "window": 1_000_000}
    return [
        {**steady, "id": "s-1m-l40", "pre": 37, "estimate": 44, "call": 40, "delta": 0, "prefix": "none"},
        {**steady, "id": "s-1m-l45-d15", "pre": 37, "estimate": 44, "call": 45, "delta": 15, "prefix": "none"},
        {**steady, "id": "s-600k-l40-prefix", "window": 600_000, "pre": 37, "estimate": 44, "call": 40, "delta": 0,
         "prefix": "1-2x"},
        {**steady, "id": "s-1m-cond-then-leaf", "pre": 37, "estimate": None, "call": [44, 40], "delta": 0,
         "prefix": "over2x"},
    ]


def _g2_cells() -> list[dict]:
    cells = []
    for backlog in (100_000, 200_000, 320_000):
        for chunk in (24_000, 113_000):
            for delta in (0, 15):
                for call in (40, 45):
                    cells.append({"group": "G2", "id": f"g2-b{backlog // 1000}k-c{chunk // 1000}k-d{delta}-l{call}",
                                  "window": 1_000_000, "pre": 37, "estimate": call, "call": call, "delta": delta,
                                  "finalize": 32, "chunk": chunk, "raw_tokens": backlog + 40_000,
                                  "over_tokens": backlog, "passes": 40, "until_drained": True, "growth_turns": 0,
                                  "turn_gap": 60, "prefix": "none", "settings": {"dynamic_leaf_chunk_enabled": False},
                                  "strict": backlog == 320_000 and chunk == 24_000 and delta == 0})
    # nine tiny leaves drain the backlog: the base stores one per pass
    cells.append({"group": "G2", "id": "g2-nine-tiny-leaves", "window": 1_000_000, "pre": 37, "estimate": 40,
                  "call": 40, "delta": 0, "finalize": 32, "chunk": 4_000, "raw_tokens": 40_000 + 31_000,
                  "over_tokens": 31_000, "passes": 40, "until_drained": True, "growth_turns": 0, "turn_gap": 60,
                  "prefix": "none", "settings": {"dynamic_leaf_chunk_enabled": False}, "strict": True})
    return cells


def _l_cells() -> list[dict]:
    cells = []
    for window in (64_000, 200_000, 272_000, 480_000):
        for setting in ("default", "explicit"):
            for kind in ("auto", "forced", "overflow"):
                for depth in (0.0, 0.3):
                    cells.append({
                        "group": "L", "id": f"l-w{window // 1000}k-{setting}-{kind}-depth{'on' if depth else 'off'}",
                        "window": window, "pre": 37, "estimate": 44, "call": 40, "delta": 0, "finalize": 32,
                        "chunk": min(24_000, window // 16), "row_tokens": min(500, window // 400),
                        "raw_tokens": int(window * 0.4), "over_tokens": int(window * 0.1), "passes": 3,
                        "growth_turns": 2, "turn_gap": 60, "prefix": "none", "kind": kind, "depth_ratio": depth,
                        "settings": {"foreground_soft_seconds": 90.0, "foreground_hard_seconds": 150.0}
                        if setting == "explicit" else {}})
    return cells


DEPTH_LIMIT = 100  # the assembly shows only the newest 100 uncondensed summaries per level (dag.py default limit)
DEPTH_TARGET = 130
LIMIT_100 = "issue 2883: the assembled view shows only the newest 100 uncondensed summaries per level"


def _depth_cells() -> list[dict]:
    """Round 7: one cell whose depth 0 passes the assembly's 100-per-level limit. A tiny chunk and fast calls store up
    to 12 leaves a pass; an explicit, very large summary prefix target keeps condensation from running."""
    return [{"group": "D", "id": "d-depth0-past-100", "window": 200_000, "pre": 0, "estimate": None, "call": 2,
             "delta": 0, "finalize": 32, "chunk": 1_500, "row_tokens": 250, "raw_tokens": 120_000,
             "over_tokens": 10_000, "growth_tokens": 20_000, "passes": 16, "turn_gap": 60, "prefix": "none",
             "settings": {"dynamic_leaf_chunk_enabled": False, "summary_prefix_target_tokens": 50_000_000}}]


ISSUE_2896 = "issue 2896"
NO_SYSTEM_ORIGINALS = ["s-1m-l40", "s-1m-l45-d15", "s-600k-l40-prefix", "s-1m-cond-then-leaf",
                       "g1-04", "g1-13", "g1-22", "g1-30", "g1-31", "g1-40", "g1-44", "d-depth0-past-100"]


def _no_system_twins(cells: list[dict]) -> list[dict]:
    """Issue 2896: twins whose host view has no leading system row (the reference host's shape), so a notice main
    writes only into a system row has nowhere to go. Group N: the A0, A0b, A2 and duplicate tests run on them; the
    group tests of their originals (A1, A6, the depth premise) do not."""
    by_id = {cell["id"]: cell for cell in cells}
    return [{**by_id[cell], "group": "N", "id": f"{cell}-nosys", "no_system_row": True, "twin_of": cell}
            for cell in NO_SYSTEM_ORIGINALS]


CELLS = _g1_cells() + _g2_cells() + _l_cells() + _steady_cells() + _depth_cells()
CELLS += _no_system_twins(CELLS)
BY_ID = {cell["id"]: cell for cell in CELLS}
G1 = [cell["id"] for cell in CELLS if cell["group"] == "G1"]
G2 = [cell["id"] for cell in CELLS if cell["group"] == "G2"]
L = [cell["id"] for cell in CELLS if cell["group"] == "L"]
S = [cell["id"] for cell in CELLS if cell["group"] == "S"]
N = [cell["id"] for cell in CELLS if cell["group"] == "N"]
# Classification on main b289240a (grid v2): every twin loses rows with no pointer in the returned list (A0), because
# the survival-fit notice goes only into a system row the view does not have. So both trees' A0 tests are pinned.
KNOWN_2896 = list(N)  # twins main fails on the base test (A0 or A0b)
# Issue 2901: leaf selection charges a glued carrier at its remainder, so these twins no longer stall into the fit.
FIXED_2901 = ["s-1m-l40-nosys", "s-1m-l45-d15-nosys", "s-1m-cond-then-leaf-nosys"]
KNOWN_2896_A0 = [cell for cell in N if cell not in FIXED_2901]  # twins with A0 rows on the candidate
# Issue 2901 moves these twins' trajectories: on some passes the candidate shows rows out of the list where the base did
# not, while its totals stay at most the base's. Only the per-pass clause is pinned; the totals test is not.
PER_PASS_2901 = ["s-600k-l40-prefix-nosys", "d-depth0-past-100-nosys"]
PER_PASS_2901_REASON = ("issues 2896, 904, 2883: the candidate's own later survival fit (notice not delivered without a "
                        "system row), a kept leaf not rendered under the hidden-only rule, and depth-0 nodes beyond "
                        "the per-level limit")
D = [cell["id"] for cell in CELLS if cell["group"] == "D"]
ALL = [cell["id"] for cell in CELLS]
NEGATIVE_CELLS = ["g1-00", "g1-11"]
GATE_SHARE_TARGET = 0.8  # G1 passes that reached the gate: triggered, or at or over the trigger and refused by a hold
ISSUE_2877 = "issue 2877"
# Known base cells (round 3 classification): rows leave the list behind a survival-fit notice (A0b) in all of them; in
# s-600k-l40-prefix a later fit's notice no longer names 37 of the first fit's rows, which no leaf covers yet (A0).
KNOWN_2877_BASE = ["g1-03", "g1-04", "g1-12", "g1-13", "g1-20", "g1-21", "g1-22", "g1-30", "g1-31", "g1-33", "g1-34",
                   "g1-40", "g1-41", "g1-42", "g1-43", "g1-44", "s-600k-l40-prefix"]
KNOWN_2877_A0: list[str] = []  # #2877 (N1): the notice names every range still out of view; the candidate pin is off
ARM_CELLS = S + [cell for cell in KNOWN_2877_BASE if cell.startswith("g1-")]  # steady cells, G1 cells with A0b
ARM_PRE_SECONDS = 5

# The negative variants, executed inside the CAND runner after its engine import (``mods`` holds its modules).
PATCHES = {
    # A1 must catch it: the first summariser call the candidate would start is refused.
    "refuses_one_call": (
        "_admit, _seen = escalation.ForegroundBudget.admit, []\n"
        "def _refuse_once(self, route_key, _admit=_admit, _seen=_seen):\n"
        "    left = _admit(self, route_key); _seen.append(1)\n"
        "    if len(_seen) == 1: raise self.stop()\n"
        "    return left\n"
        "escalation.ForegroundBudget.admit = _refuse_once\n"),
    # A0 must catch it: the first changed list a pass returns loses its last row.
    "drops_one_row": (
        "_compress, _done = engine.LCMEngine.compress, []\n"
        "def _drop_once(self, messages, *args, _compress=_compress, _done=_done, **kwargs):\n"
        "    out = _compress(self, messages, *args, **kwargs)\n"
        "    if not _done and out is not messages: _done.append(1); out = out[:-1]\n"
        "    return out\n"
        "engine.LCMEngine.compress = _drop_once\n"),
    # A0 must catch it: the first returned summary head of a stored leaf is removed, so its rows are neither
    # covered nor returned.
    "summary_head_drop": (
        "_compress, _done = engine.LCMEngine.compress, []\n"
        "def _leaf_head(self, row):\n"
        "    ids = self._verified_lcm_summary_prefix(row['content'])[1] if row.get('role') == 'user' and isinstance(row.get('content'), str) else []\n"
        "    return any(n is not None and n.source_type == 'messages' and n.source_ids for n in map(self._dag.get_node, ids))\n"
        "def _drop_head(self, messages, *args, _compress=_compress, _done=_done, **kwargs):\n"
        "    out = _compress(self, messages, *args, **kwargs)\n"
        "    at = None if _done else next((i for i, row in enumerate(out) if _leaf_head(self, row)), None)\n"
        "    if at is not None: _done.append(1); out = out[:at] + out[at + 1:]\n"
        "    return out\n"
        "engine.LCMEngine.compress = _drop_head\n"),
    # A0 must catch it: the first returned tool call row is removed and its result is returned without it.
    "broken_pair": (
        "_compress, _done = engine.LCMEngine.compress, []\n"
        "def _break_pair(self, messages, *args, _compress=_compress, _done=_done, **kwargs):\n"
        "    out = _compress(self, messages, *args, **kwargs)\n"
        "    at = None if _done or out is messages else next((i for i, row in enumerate(out) if row.get('tool_calls')), None)\n"
        "    if at is not None: _done.append(1); out = out[:at] + out[at + 1:]\n"
        "    return out\n"
        "engine.LCMEngine.compress = _break_pair\n"),
}
# Measurement arm (b), BASE only: after two consecutive cut pre-leaf condensations, the next over-target pass gives the
# condensation the pass's whole grant (leaf_reserve = 0 for that pass). A pre-leaf call is one made with a leaf reserve.
LIVENESS_PATCH = (
    "_run_condensation = engine.LCMEngine._run_threshold_sweep_condensation\n"
    "def _liveness(self, *args, _run=_run_condensation, **kwargs):\n"
    "    budget = self._foreground_budget\n"
    "    if budget is None or budget.leaf_reserve <= 0:\n"
    "        return _run(self, *args, **kwargs)\n"
    "    streak, granted = self.__dict__.get('_arm_cut_streak', 0), False\n"
    "    if streak >= 2:\n"
    "        budget.leaf_reserve, granted = 0.0, True\n"
    "        self.__dict__['_arm_grants'] = self.__dict__.get('_arm_grants', 0) + 1\n"
    "    ended = budget.last_call_ended\n"
    "    passes, reason = _run(self, *args, **kwargs)\n"
    "    cut = passes == 0 and reason == 'time_budget_exhausted' and budget.last_call_ended != ended\n"
    "    self.__dict__['_arm_cut_streak'] = (0 if granted else streak) + 1 if cut else 0\n"
    "    return passes, reason\n"
    "engine.LCMEngine._run_threshold_sweep_condensation = _liveness\n")
NEGATIVE_EXPECT = {"refuses_one_call": ("A1", "calls started"), "drops_one_row": ("A0", "neither covered nor returned"),
                   "summary_head_drop": ("A0", "neither covered nor returned"), "broken_pair": ("A0", "unpaired")}

# -- running the two trees -----------------------------------------------------------------------------------------


def _extra_paths() -> list[str]:
    """The host stub and dependency directories of this process, never either tree."""
    found = []
    for entry in sys.path:
        path = Path(entry).resolve() if entry else None
        if path is None or not path.is_dir() or any(path == tree or tree in path.parents
                                                   for tree in (BASE_TREE, CAND_TREE) if tree is not None):
            continue
        if entry.endswith("site-packages") or (path / "agent" / "context_engine.py").is_file():
            found.append(str(path))
    return found


def _run(tree: Path, cells: list[dict], out_dir: Path, name: str, patch: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path, out_path = out_dir / f"{name}-cells.json", out_dir / f"{name}-records.json"
    cells_path.write_text(json.dumps(cells, indent=1))
    command = [sys.executable, "-I", "-B", str(RUNNER), "--tree", str(tree), "--cells", str(cells_path),
               "--out", str(out_path), "--workdir", str(out_dir / f"{name}-work")]
    for entry in _extra_paths():
        command += ["--syspath", entry]
    if patch:
        patch_path = out_dir / f"{name}-patch.py"
        patch_path.write_text(PATCHES[patch])
        command += ["--patch", str(patch_path)]
    home = out_dir / f"{name}-home"
    home.mkdir(exist_ok=True)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "HERMES_HOME": str(home / "hermes"), "LANG": "C.UTF-8",
           "PYTHONHASHSEED": "0"}
    began = time.perf_counter()
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    seconds = time.perf_counter() - began
    (out_dir / f"{name}-stderr.txt").write_text(done.stderr)
    assert done.returncode == 0, f"runner on {tree} exited {done.returncode}: {done.stderr[-2000:]}"
    output = json.loads(out_path.read_text())
    header = output["header"]
    assert Path(header["tree"]) == tree
    assert Path(header["home"]).is_relative_to(out_dir) and Path(header["hermes_home"]).is_relative_to(out_dir)
    assert header["modules"] and all(Path(module).is_relative_to(tree) for module in header["modules"]), (
        f"runner on {tree} loaded engine modules from another tree")
    output["seconds"] = seconds
    output["by_cell"] = {}
    for record in output["records"]:
        output["by_cell"].setdefault(record["cell"], []).append(record)
    return output


def _trigger_share(run: dict, key: str = "triggered") -> dict:
    share = {}
    for group in ("G1", "G2", "L", "S", "D", "N"):
        records = [r for cell in CELLS if cell["group"] == group for r in run["by_cell"][cell["id"]]]
        share[group] = f"{sum(bool(r[key]) for r in records)}/{len(records)}"
    return share


def _a0_totals(records: list[dict]) -> dict:
    """A0 (no pointer in the returned list) and A0b (named by a pointer in the returned list), summed over passes."""
    return {"A0": sum(r["loss"]["missing_count"] for r in records if r["loss"]), "A0b": sum(a0b_rows(records))}


def _uncondensed_max(records: list[dict]) -> dict:
    """The largest uncondensed summary count per level over a cell's passes."""
    peak: dict = {}
    for record in records:
        for depth, count in record["uncondensed"].items():
            peak[depth] = max(peak.get(depth, 0), count)
    return peak


def _content_vs_id(run: dict) -> dict:
    """Cells and passes where the content-aware and the id-based A0/A0b figures differ."""
    found = {}
    for cell in ALL:
        for r in run["by_cell"][cell]:
            loss = r.get("loss")
            if loss and (loss["missing_count"] != loss["missing_count_by_id"]
                         or loss["behind_notice_count"] != loss["behind_notice_count_by_id"]):
                found.setdefault(cell, []).append({
                    "pass": r["pass"], "a0_content": loss["missing_count"], "a0_id": loss["missing_count_by_id"],
                    "a0b_content": loss["behind_notice_count"], "a0b_id": loss["behind_notice_count_by_id"]})
    return found


def a0b_rows(records: list[dict]) -> list[int]:
    """A0b per pass: stored rows out of the list whose store ids a survival-fit pointer names."""
    return [record["loss"]["behind_notice_count"] if record.get("loss") else 0 for record in records]


def _slope(values: list[float]) -> float:
    n = len(values)
    mean_x, mean_y = (n - 1) / 2, sum(values) / n
    return sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values)) / sum((x - mean_x) ** 2 for x in range(n))


def _a6(records: list[dict]) -> dict:
    """Max summary area, and whether it grows over passes 41-60: a positive least-squares slope whose total over the
    window (slope x 19 passes) exceeds one leaf summary (the stand-in's leaf summary, in tokens)."""
    window = [r["summary_area"] for r in records[A6_FROM:A6_TO]]
    slope = _slope(window) if len(window) == A6_TO - A6_FROM else 0.0
    total, leaf = slope * (len(window) - 1), records[0]["leaf_summary_tokens"]
    return {"max_summary_area": max(r["summary_area"] for r in records), "passes": len(records),
            "slope_41_60": round(slope, 3), "total_41_60": round(total, 1), "one_leaf_summary": leaf,
            "grows": bool(slope > 0 and total > leaf)}


@pytest.fixture(scope="session")
def grid_dir(tmp_path_factory) -> Path:
    configured = os.environ.get("LCMX_GRID_OUT")
    return Path(configured) if configured else tmp_path_factory.mktemp("catchup-grid")


@pytest.fixture(scope="session")
def grid(grid_dir):
    if BASE_TREE is None:
        pytest.skip(NO_BASE)
    base = _run(BASE_TREE, CELLS, grid_dir, "base")  # one tree after the other, never in parallel
    cand = _run(CAND_TREE, CELLS, grid_dir, "cand")
    g1_seconds = [base["cell_seconds"][cell] for cell in G1]
    summary = {
        "cells": len(CELLS), "g1_cells": len(G1), "g2_cells": len(G2), "l_cells": len(L),
        "passes_base": len(base["records"]), "passes_cand": len(cand["records"]),
        "runner_seconds": {"base": round(base["seconds"], 1), "cand": round(cand["seconds"], 1)},
        "g1_full_product_cells": 2_000,
        "g1_full_product_estimated_seconds_two_trees": round(2 * 2_000 * sum(g1_seconds) / len(g1_seconds), 0),
        "cand_catchup_constant": cand["header"]["catchup_constant"],
        "trigger_share": {tree: _trigger_share(run) for tree, run in (("base", base), ("cand", cand))},
        "gate_share": {tree: _trigger_share(run, "gate_reached") for tree, run in (("base", base), ("cand", cand))},
        "uncondensed_max_per_level": {tree: {cell: _uncondensed_max(run["by_cell"][cell]) for cell in ALL}
                                      for tree, run in (("base", base), ("cand", cand))},
        "unshown_rows_max": {tree: {cell: max((r["loss"]["unshown_rows"] for r in run["by_cell"][cell] if r["loss"]),
                                              default=0) for cell in D}
                             for tree, run in (("base", base), ("cand", cand))},
        "no_system_twins": {cell: {tree: {"twin": _a0_totals(run["by_cell"][cell]),
                                          "original": _a0_totals(run["by_cell"][BY_ID[cell]["twin_of"]])}
                                   for tree, run in (("base", base), ("cand", cand))} for cell in N},
        "duplicate_rows_max": {tree: {cell: max(r["duplicate_rows"] for r in run["by_cell"][cell]) for cell in ALL
                                      if max(r["duplicate_rows"] for r in run["by_cell"][cell])}
                               for tree, run in (("base", base), ("cand", cand))},
        "duplicate_rows_total_max": {tree: sum(max(r["duplicate_rows"] for r in run["by_cell"][cell]) for cell in ALL)
                                     for tree, run in (("base", base), ("cand", cand))},
        "content_vs_id": {tree: _content_vs_id(run) for tree, run in (("base", base), ("cand", cand))},
        "a0b_rows_total": {tree: sum(sum(a0b_rows(run["by_cell"][cell])) for cell in ALL)
                           for tree, run in (("base", base), ("cand", cand))},
        "a0b_rows_by_cell": {cell: {tree: sum(a0b_rows(run["by_cell"][cell])) for tree, run in
                                    (("base", base), ("cand", cand))}
                             for cell in ALL if sum(a0b_rows(base["by_cell"][cell])) or sum(a0b_rows(cand["by_cell"][cell]))},
        "a6": {cell: {tree: _a6(run["by_cell"][cell]) for tree, run in (("base", base), ("cand", cand))} for cell in S},
        "a0_base_findings": {cell: {"passes": [r["pass"] for r in base["by_cell"][cell] if a0_violations([r])],
                                    "rows": sum(r["loss"]["missing_count"] for r in base["by_cell"][cell] if r["loss"]),
                                    "first": a0_violations(base["by_cell"][cell])[0]}
                             for cell in ALL if a0_violations(base["by_cell"][cell])},
    }
    (grid_dir / "grid-summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\ncatch-up grid: {summary}")
    return {"base": base, "cand": cand, "summary": summary}


@pytest.fixture
def strict_gain(grid, request):
    """A4's strict cells: xfail(strict=True) until CAND_TREE carries the catch-up constant."""
    if not grid["cand"]["header"]["catchup_constant"]:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=STRICT))


# -- assertions ----------------------------------------------------------------------------------------------------


def a0_violations(records: list[dict]) -> list[str]:
    found = []
    for record in records:
        loss = record.get("loss")
        if loss is None:  # the host's gate skipped the pass: no compress ran
            continue
        where = f"pass {record['pass']}"
        if loss["missing_count"]:
            found.append(f"{where}: {loss['missing_count']} stored row(s) neither covered nor returned, "
                         f"first {loss['missing'][:2]}")
        if loss["unpaired"]:
            found.append(f"{where}: tool call/result unpaired {loss['unpaired'][:2]}")
        if loss["noop_not_identical"]:
            found.append(f"{where}: a pass that did no work returned an equal copy, not its input")
    return found


def a1_violations(base: list[dict], cand: list[dict]) -> list[str]:
    """Per pass index: a base call needs a candidate call unless the candidate's own prompt is under its trigger; and
    the candidate's cumulative leaves and cumulative tokens drained are each at least the base's."""
    found = []
    for b, c in zip(base, cand):
        if b["calls_started"] > 0 and not (c["calls_started"] > 0 or c["prompt_before"] < c["trigger"]):
            found.append(f"pass {b['pass']}: calls started 0 with the prompt {c['prompt_before']} at or over the "
                         f"trigger {c['trigger']}, base {b['calls_started']}")
        if c["leaves_cum"] < b["leaves_cum"]:
            found.append(f"pass {b['pass']}: cumulative leaves {c['leaves_cum']} < base {b['leaves_cum']}")
        # T1-285 / T1-293: drained = tokens removed by stored leaves and condensations, never a survival-fit cut or notice
        if c["summary_drained_cum"] < b["summary_drained_cum"]:
            found.append(f"pass {b['pass']}: cumulative drained {c['summary_drained_cum']} < base "
                         f"{b['summary_drained_cum']}")
        if c["hold_armed"] and not b["hold_armed"]:
            found.append(f"pass {b['pass']}: the candidate armed a hold the base did not")
    return found


def a2_violations(records: list[dict]) -> list[str]:
    return [f"pass {r['pass']}: wall {r['wall']} s (bounds {WALL_PLUGIN} s, {WALL_HOST} s with a {HOST_COMMIT} s "
            f"host commit)" for r in records
            if r["triggered"] and (r["wall"] > WALL_PLUGIN or r["wall"] + HOST_COMMIT > WALL_HOST)]


def _commits(records: list[dict]) -> tuple[int, int]:
    """(commits, whole-prompt tokens drained) of one walk; a commit is a pass that stored a node."""
    ran = [r for r in records if r["triggered"]]
    commits = sum(1 for r in ran if r["leaves"] or r["condensations"])
    drained = ran[0]["prompt_before"] - ran[-1]["prompt_after"] if ran else 0
    return commits, drained


@pytest.mark.parametrize("tree", ["base", "cand"])
@pytest.mark.parametrize("cell", ALL)
def test_a0_loss(grid, cell, tree, request):
    """A0, a row that leaves the list with no pointer. On the base, also no A0b row (issue 2877 pins the known cells,
    so a fix on main turns them red until the pin is removed)."""
    records = grid[tree]["by_cell"][cell]
    if tree == "base" and cell in KNOWN_2877_BASE or tree == "cand" and cell in KNOWN_2877_A0:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=ISSUE_2877))
    if cell in D and not (tree == "cand" and cell == "d-depth0-past-100"):  # T1-293: N1 names the rows the limit hides
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=LIMIT_100))
    if cell in KNOWN_2896 and (tree == "base" or cell in KNOWN_2896_A0):
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=ISSUE_2896))
    found = a0_violations(records)
    if tree == "base":
        found += [f"pass {r['pass']}: {n} row(s) behind a survival-fit notice (A0b)"
                  for r, n in zip(records, a0b_rows(records)) if n]
    label = "BASE finding (main already loses this)" if tree == "base" else "candidate failure"
    assert not found, f"A0 {label} in cell {cell}: " + "; ".join(found)


@pytest.mark.parametrize("cell", D)
def test_depth_cell_premise(grid, cell):
    """Not under the pin: depth 0 reaches the target, no condensation runs, and both trees walk every pass."""
    for tree in ("base", "cand"):
        records = grid[tree]["by_cell"][cell]
        peak = _uncondensed_max(records).get("0", 0)
        assert len(records) == BY_ID[cell]["passes"] and peak >= DEPTH_TARGET, f"{tree} {cell}: depth 0 peak {peak}"
        assert not any(r["condensations"] for r in records), f"{tree} {cell}: a condensation ran"


@pytest.mark.parametrize("tree", ["base", "cand"])
@pytest.mark.parametrize("cell", D)
def test_depth_cell_every_row_shown(grid, cell, tree, request):
    """Rows neither in the list nor under a summary shown in it, on every pass (pinned on main: LIMIT_100)."""
    request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=LIMIT_100))
    unshown = [(r["pass"], r["loss"]["unshown_rows"], r["uncondensed"].get("0", 0))
               for r in grid[tree]["by_cell"][cell] if r["loss"] and r["loss"]["unshown_rows"]]
    assert not unshown, f"{tree} {cell}: (pass, rows with no shown summary, depth-0 uncondensed) {unshown[:5]}"


@pytest.mark.parametrize("tree", ["base", "cand"])
@pytest.mark.parametrize("cell", ALL)
def test_no_duplicate_content_rows(grid, cell, tree):
    """No cell restarts or replays a session, so no stored row should repeat another's role and content."""
    counts = [(r["pass"], r["duplicate_rows"]) for r in grid[tree]["by_cell"][cell] if r["duplicate_rows"]]
    assert not counts, f"{tree} cell {cell}: duplicate-content stored rows (pass, count) {counts[:5]}"


@pytest.mark.parametrize("cell", ALL)
def test_a0_candidate_no_worse_than_the_base(grid, cell, request):
    """Per pass (#2877, ruling T1-275): the candidate's A0 rows are at most the base's, and its A0 + A0b rows are at most
    the base's (N1 moves rows with no pointer behind its notice)."""
    if cell in PER_PASS_2901:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=PER_PASS_2901_REASON))
    base, cand = grid["base"]["by_cell"][cell], grid["cand"]["by_cell"][cell]
    for b, c, nb, nc in zip(base, cand, a0b_rows(base), a0b_rows(cand)):
        lost = [r["loss"]["missing_count"] if r["loss"] else 0 for r in (b, c)]
        assert lost[1] <= lost[0], f"A0 in cell {cell} pass {b['pass']}: candidate {lost[1]} rows, base {lost[0]}"
        assert lost[1] + nc <= lost[0] + nb, (f"A0 + A0b in cell {cell} pass {b['pass']}: candidate {lost[1]} + {nc}, "
                                              f"base {lost[0]} + {nb}")


@pytest.mark.parametrize("cell", PER_PASS_2901)
def test_a0_candidate_totals_no_worse_than_the_base(grid, cell):
    """Not under the per-pass pin: summed over passes, the candidate's A0 and A0b rows are at most the base's."""
    base, cand = grid["base"]["by_cell"][cell], grid["cand"]["by_cell"][cell]
    a0 = {tree: sum(r["loss"]["missing_count"] for r in records if r["loss"])
          for tree, records in (("base", base), ("cand", cand))}
    assert a0["cand"] <= a0["base"], f"A0 total in cell {cell}: candidate {a0['cand']}, base {a0['base']}"
    assert sum(a0b_rows(cand)) <= sum(a0b_rows(base)), f"A0b total in cell {cell}"


def test_g1_gate_share(grid):
    for tree in ("base", "cand"):
        records = [r for cell in G1 for r in grid[tree]["by_cell"][cell]]
        share = sum(r["gate_reached"] for r in records) / len(records)
        assert share >= GATE_SHARE_TARGET, f"{tree}: {share:.2f} of G1 passes reached the gate"


@pytest.mark.parametrize("cell", G1 + G2 + S)
def test_a1_never_less_progress(grid, cell):
    found = a1_violations(grid["base"]["by_cell"][cell], grid["cand"]["by_cell"][cell])
    assert not found, f"A1 in cell {cell}: " + "; ".join(found)


@pytest.mark.parametrize("tree", ["base", "cand"])
@pytest.mark.parametrize("cell", ALL)
def test_a2_walls(grid, cell, tree):
    found = a2_violations(grid[tree]["by_cell"][cell])
    assert not found, f"A2 on {tree} in cell {cell}: " + "; ".join(found)


@pytest.mark.parametrize("cell", L)
def test_a3_small_windows_match_the_base(grid, cell):
    strip = lambda records: [{k: v for k, v in r.items() if k != "tree"} for r in records]  # noqa: E731
    base, cand = strip(grid["base"]["by_cell"][cell]), strip(grid["cand"]["by_cell"][cell])
    assert len(cand) == len(base), f"A3 in cell {cell}: {len(cand)} passes, base {len(base)}"
    for b, c in zip(base, cand):
        diff = {k: (b.get(k), c.get(k)) for k in b.keys() | c.keys() if b.get(k) != c.get(k)}
        assert not diff, f"A3 in cell {cell} pass {b['pass']}: (base, candidate) {diff}"


@pytest.mark.parametrize("cell", [c for c in G2 if not BY_ID[c]["strict"]])
def test_a4_commits_per_drained_token(grid, cell):
    (base_commits, base_drained), (cand_commits, cand_drained) = (
        _commits(grid[tree]["by_cell"][cell]) for tree in ("base", "cand"))
    assert base_commits > 0, f"A4 in cell {cell}: the base walk committed nothing"
    assert cand_commits <= base_commits, f"A4 in cell {cell}: commits {cand_commits} > base {base_commits}"
    assert cand_commits and cand_drained / cand_commits >= base_drained / base_commits, (
        f"A4 in cell {cell}: drained per commit {cand_drained}/{cand_commits} < base {base_drained}/{base_commits}")


@pytest.mark.parametrize("cell", [c for c in G2 if BY_ID[c]["strict"]])
def test_a4_strict_cell_premise_on_the_base(grid, cell):
    """Not under the xfail: the strict cells must still show the base committing once per drained leaf."""
    records = grid["base"]["by_cell"][cell]
    commits, drained = _commits(records)
    assert commits > 1 and drained > 0 and records[-1]["triggered"], f"cell {cell} premise: {commits}, {drained}"
    assert all(r["leaves"] == 1 for r in records if r["triggered"]), f"cell {cell}: the base stored != 1 leaf a pass"


@pytest.mark.parametrize("cell", [c for c in G2 if BY_ID[c]["strict"]])
def test_a4_strict_fewer_commits(grid, cell, strict_gain):
    (base_commits, base_drained), (cand_commits, cand_drained) = (
        _commits(grid[tree]["by_cell"][cell]) for tree in ("base", "cand"))
    assert cand_commits < base_commits, f"A4 strict in cell {cell}: commits {cand_commits}, base {base_commits}"
    assert cand_drained / cand_commits >= base_drained / base_commits


def test_a4_condensation_then_leaf_premise_on_the_base(grid):
    """Not under the xfail: the base condenses, then its soft target refuses the leaf."""
    base = grid["base"]["by_cell"]["g1-cond-then-leaf"][0]
    assert base["call_kinds"] == ["condensation"] and base["condensations"] == 1 and base["leaves"] == 0, (
        f"the cell no longer shows the base refusing the leaf after its condensation: {base}")


def test_a4_strict_condensation_then_leaf(grid, strict_gain):
    cand = grid["cand"]["by_cell"]["g1-cond-then-leaf"][0]
    assert cand["leaves"] == 1, f"A4 strict in cell g1-cond-then-leaf: the candidate stored {cand['leaves']} leaves"


def test_a5_hang_cell_wait(grid):
    base, cand = (grid[tree]["by_cell"]["g1-hang-every-pass"] for tree in ("base", "cand"))
    for b, c in zip(base, cand):
        assert c["wall"] - b["wall"] <= CATCHUP_HARD - BASE_HARD, (
            f"A5 pass {b['pass']}: the candidate added {c['wall'] - b['wall']:.1f} s over the base")
    assert not a2_violations(cand) and not a2_violations(base)


@pytest.mark.parametrize("cell", S)
def test_a6_bounded_summary_area(grid, cell):
    base, cand = (_a6(grid[tree]["by_cell"][cell]) for tree in ("base", "cand"))
    assert base["passes"] == cand["passes"] == A6_TO, f"A6 in cell {cell}: walks of {base['passes']}, {cand['passes']}"
    assert not (cand["grows"] and not base["grows"]), f"A6 in cell {cell}: candidate {cand}, base {base}"


@pytest.mark.parametrize("variant", list(NEGATIVE_EXPECT))
def test_negative_variant_is_caught(grid, grid_dir, variant):
    """The assertions are live: each variant fails the named assertion in the first cell, and only there."""
    cells = [BY_ID[cell] for cell in NEGATIVE_CELLS]
    bad = _run(CAND_TREE, cells, grid_dir, f"negative-{variant}", patch=variant)
    assert bad["header"]["patched"]
    check, text = NEGATIVE_EXPECT[variant]
    if check == "A1":
        caught = {cell: a1_violations(grid["base"]["by_cell"][cell], bad["by_cell"][cell]) for cell in NEGATIVE_CELLS}
    else:
        caught = {cell: a0_violations(bad["by_cell"][cell]) for cell in NEGATIVE_CELLS}
    assert any(text in found for found in caught[NEGATIVE_CELLS[0]]), caught
    assert not any(caught[cell] for cell in NEGATIVE_CELLS[1:]), f"the variant changed more than one cell: {caught}"
    (grid_dir / f"negative-{variant}-caught.json").write_text(json.dumps(caught, indent=1))


ARM_FIELDS = ("condensations stored", "peak summary area", "leaves published", "tokens drained",
              "rows behind a notice (A0b)", "A0 rows lost", "largest pass wall (s)", "passes over 250 s",
              "holds armed", "passes refused by a hold", "passes that stored no leaf", "sum of pass walls (s)")
# Issue 2877/2878 context, BASE only, settings and monkeypatches inside the harness. (c) stands in for design B with
# settings alone: soft 0, hard 210 (more generous to the first call than B, whose first call keeps hard 120).
ARM_SOFT0_HARD210 = {"foreground_soft_seconds": 0.0, "foreground_hard_seconds": 210.0}

# The no-leaf pass arms two holds on main: the #608 sweep hold and the #651 no-progress hold (both 600 s). Ending only
# the first changes nothing (the second still refuses), so the hold options end both, the way a stored leaf does
# (compaction.py: "a stored leaf ends the hold"), keeping any hung-route strike and circuit.
_HOLD_HELPERS = (
    "def _holding(self):\n"
    "    return self._sweep_budget_hold_active() or self._no_progress_hold_active(managed_scope=True)\n"
    "def _end_holds(self):\n"
    "    self._sweep_budget_hold_until, self._sweep_budget_hold_turn_paced = 0.0, False\n"
    "    self._no_progress_hold = None\n"
    "    self._clear_persisted_no_progress_hold()\n")
# H1: the holds end early once the prompt has grown a share s of the way from the trigger to the window.
H1_PATCH = _HOLD_HELPERS + (
    "_should = engine.LCMEngine.should_compress\n"
    "def _h1(self, prompt_tokens=None, _should=_should):\n"
    "    tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens\n"
    "    if (tokens is not None and 0 < self.threshold_tokens < self.context_length and _holding(self)\n"
    "            and tokens >= self.threshold_tokens + {share} * (self.context_length - self.threshold_tokens)):\n"
    "        _end_holds(self)\n"
    "    return _should(self, prompt_tokens)\n"
    "engine.LCMEngine.should_compress = _h1\n")
# H2: the holds last N turns that reach the gate (or their 600 s, whichever ends first).
# H3 (on (c), whose hard 210 already gives the first call the catch-up bound): holds armed by a pass whose calls were
# cut only by the plugin's own budget (no hang; the route allowed, with no error class) refuse one gated turn, not 600 s.
# (budget.summary_failure is set by a budget cut too, so it cannot tell the cases apart.)
TURN_HOLD_PATCH = _HOLD_HELPERS + (
    "_should, _finish = engine.LCMEngine.should_compress, engine.LCMEngine._finish_foreground_budget\n"
    "def _note_budget(self, budget, *args, _finish=_finish, **kwargs):\n"
    "    self.__dict__['_arm_budget'] = budget\n"
    "    return _finish(self, budget, *args, **kwargs)\n"
    "def _turn_hold(self, prompt_tokens=None, _should=_should):\n"
    "    tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens\n"
    "    if tokens is not None and 0 < self.threshold_tokens <= tokens and _holding(self):\n"
    "        key = (self._sweep_budget_hold_until, self._no_progress_hold)\n"
    "        held, refused, turns = self.__dict__.get('_arm_hold', (None, 0, None))\n"
    "        if held != key:\n"
    "            refused, turns = 0, {turns}\n"
    "            if {budget_cut_only}:\n"
    "                budget = self.__dict__.get('_arm_budget')\n"
    "                cut = (budget is not None and not budget.progress and budget.first_call_started is not None\n"
    "                       and budget.stop_reason == 'time_budget_exhausted' and budget.hung_route is None\n"
    "                       and self._summary_route_available() and self._summary_circuit_breaker.route_status(\n"
    "                           [self._primary_summary_route()]).get('last_error_class') is None)\n"
    "                turns = 1 if cut else None\n"
    "        if turns is not None and refused >= turns:\n"
    "            _end_holds(self)\n"
    "        self.__dict__['_arm_hold'] = ((self._sweep_budget_hold_until, self._no_progress_hold), refused + 1, turns)\n"
    "    return _should(self, prompt_tokens)\n"
    "engine.LCMEngine.should_compress, engine.LCMEngine._finish_foreground_budget = _turn_hold, _note_budget\n")
# N1: a cumulative notice. It names every store-id range still out of view (earlier fits' ranges not yet under a node,
# merged with this fit's) with the exact call that loads it; at most N1_MAX_RANGES, the rest pointed to lcm_grep.
N1_MAX_RANGES = 8
N1_PATCH = (
    "import re as _re\n"
    "_ctx, _compress = {}, engine.LCMEngine.compress\n"
    "def _bind(self, *args, _compress=_compress, **kwargs):\n"
    "    _ctx['engine'] = self\n"
    "    return _compress(self, *args, **kwargs)\n"
    "engine.LCMEngine.compress = _bind\n"
    "class _Cumulative:\n"
    "    def format(self, n, first, last):\n"
    "        eng = _ctx.get('engine')\n"
    "        if eng is None or first == '-':\n"
    "            return _old.format(n=n, first=first, last=last)\n"
    "        covered = {int(i) for node in eng._dag.get_session_nodes(eng._session_id, limit=10_000_000)\n"
    "                   if node.source_type == 'messages' for i in node.source_ids}\n"
    "        prior = eng.__dict__.get('_arm_n1_ranges', [])\n"
    "        ids = sorted({i for a, b in prior for i in range(a, b + 1) if i not in covered} | set(range(int(first), int(last) + 1)))\n"
    "        ranges = []\n"
    "        for i in ids:\n"
    "            if ranges and ranges[-1][1] == i - 1: ranges[-1][1] = i\n"
    "            else: ranges.append([i, i])\n"
    "        eng.__dict__['_arm_n1_ranges'] = [tuple(r) for r in ranges]\n"
    "        calls = '; '.join(f'store ids {a}..{b}: lcm_load_session(session_id=\"{eng._session_id}\", after_store_id={a - 1}, limit=200)'\n"
    "                          for a, b in ranges[:MAX])\n"
    "        rest = f'; {len(ranges) - MAX} further ranges: lcm_grep' if len(ranges) > MAX else ''\n"
    "        return (f'[LCM survival fit: {len(ids)} earlier messages are stored verbatim but not in live context; '\n"
    "                f'load each range as written, paging with next_cursor: {calls}{rest}.]')\n"
    "_old = survival_fit._NOTICE\n"
    "survival_fit._NOTICE = _Cumulative()\n"
    "_n1 = r'\\[LCM survival fit: \\d+ earlier messages are stored verbatim but not in live context; load each range as written[^\\]]*\\]'\n"
    "survival_fit._NOTICE_RE = _re.compile(r'(?:\\n\\n)?(?:' + survival_fit._NOTICE_RE.pattern[len(r'(?:\\n\\n)?'):] + '|' + _n1 + ')')\n"
).replace("MAX", str(N1_MAX_RANGES))
ARMS = {  # name: (cell overrides, settings overrides, patch source)
    "base": ({}, {}, ""),
    "(c) soft 0, hard 210": ({}, ARM_SOFT0_HARD210, ""),
    "(c)+H1 s=0.25": ({}, ARM_SOFT0_HARD210, H1_PATCH.format(share=0.25)),
    "(c)+H1 s=0.5": ({}, ARM_SOFT0_HARD210, H1_PATCH.format(share=0.5)),
    "(c)+H2 N=2": ({}, ARM_SOFT0_HARD210, TURN_HOLD_PATCH.format(turns=2, budget_cut_only=False)),
    "(c)+H3": ({}, ARM_SOFT0_HARD210, TURN_HOLD_PATCH.format(turns=None, budget_cut_only=True)),
    "(c)+N1": ({}, ARM_SOFT0_HARD210, N1_PATCH),
}
DIAGNOSE_ARM = "(c) soft 0, hard 210"


def _arm_row(records: list[dict]) -> tuple:
    ran = [r for r in records if r["triggered"]]
    walls = [r["wall"] for r in ran]
    return (sum(r["condensations"] for r in records), max(r["summary_area"] for r in records),
            records[-1]["leaves_cum"], records[-1]["drained_cum"], sum(a0b_rows(records)),
            sum(r["loss"]["missing_count"] for r in records if r["loss"]), max(walls, default=0.0),
            sum(1 for wall in walls if wall > WALL_PLUGIN), sum(r["hold_armed"] for r in records),
            sum(1 for r in records if r["gate_reached"] and not r["triggered"]),
            sum(1 for r in ran if not r["leaves"]), round(sum(walls), 1))


def _first_a0b(records: list[dict]):
    return next((r["pass"] for r, n in zip(records, a0b_rows(records)) if n), None)


def _diagnose(cell: dict, records: list[dict]) -> dict:
    """Why the first pass that stored no leaf stored none: its stop reason, the pre-call work, the first call's grant
    and whether that call was cut, and the route's state after the pass."""
    record = next((r for r in records if r["triggered"] and not r["leaves"]), None)
    if record is None:
        return {"cell": cell["id"], "first_no_leaf_pass": None}
    return {"cell": cell["id"], "first_no_leaf_pass": record["pass"], "stop_reason": record["stop_reason"],
            "pre_call_work_s": cell["pre"], "stand_in_call": cell["call"], "first_call": (record["call_kinds"] or [None])[0],
            "first_call_grant_s": (record["call_timeouts"] or [None])[0],
            "first_call_cut": (record["call_cut_flags"] or [None])[0], "calls": len(record["call_kinds"]),
            "route_after": record["circuit"], "hold_armed": record["hold_armed"], "wall_s": record["wall"]}


@pytest.mark.skipif(os.environ.get("LCMX_GRID_ARMS") != "1", reason="measurement arms: set LCMX_GRID_ARMS=1")
@pytest.mark.skipif(BASE_TREE is None, reason=NO_BASE)
def test_measurement_arms_on_the_base(grid_dir):
    """A measurement over ARM_CELLS on the BASE tree, one runner process per arm (about 25 minutes, so off unless
    LCMX_GRID_ARMS=1): it asserts only that each arm ran every cell. Writes arms.md and arms.json in the run dir."""
    runs = {}
    for index, (name, (overrides, settings, patch)) in enumerate(ARMS.items()):
        cells = [{**BY_ID[cell], **overrides, "settings": {**BY_ID[cell].get("settings", {}), **settings}}
                 for cell in ARM_CELLS]
        if patch:
            PATCHES[f"arm-{index}"] = patch
        runs[name] = _run(BASE_TREE, cells, grid_dir, f"arm-{index}", patch=f"arm-{index}" if patch else "")
        assert set(runs[name]["by_cell"]) == set(ARM_CELLS), f"arm {name} did not run every cell"
    rows = {name: {cell: _arm_row(run["by_cell"][cell]) for cell in ARM_CELLS} for name, run in runs.items()}
    diagnosed = [_diagnose(BY_ID[cell], runs[DIAGNOSE_ARM]["by_cell"][cell]) for cell in ARM_CELLS
                 if cell in G1 and sum(a0b_rows(runs[DIAGNOSE_ARM]["by_cell"][cell]))]
    head = "| {} | arm | " + " | ".join(ARM_FIELDS) + " |"
    rule = "|---|---|" + "---|" * len(ARM_FIELDS)
    fmt = lambda row: " | ".join(f"{v:,}" if isinstance(v, int) else f"{v:,.1f}" for v in row)  # noqa: E731
    lines = [f"Diagnostic under {DIAGNOSE_ARM}: G1 cells with A0b, first pass that stored no leaf", "",
             "| cell | pass | stop reason | pre-call work (s) | stand-in call | first call | grant (s) | cut | calls "
             "| route after |", "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {d['cell']} | {d['first_no_leaf_pass']} | {d.get('stop_reason')} | {d.get('pre_call_work_s')} | "
              f"{d.get('stand_in_call')} | {d.get('first_call')} | {d.get('first_call_grant_s')} | "
              f"{d.get('first_call_cut')} | {d.get('calls')} | {(d.get('route_after') or {}).get('state')}/"
              f"{(d.get('route_after') or {}).get('last_error_class')} |" for d in diagnosed]
    lines += ["", head.format("all 20 arm cells"), rule]
    for name in ARMS:
        values = list(zip(*rows[name].values()))
        total = [max(v) if i in (1, 6) else round(sum(v), 1) if i == 11 else sum(v) for i, v in enumerate(values)]
        lines.append(f"| total (peak and wall: max) | {name} | {fmt(total)} |")
    lines += ["", "First pass with A0b > 0 (0-based; none = never), steady cells:", ""]
    for cell in S:
        firsts = {name: _first_a0b(runs[name]["by_cell"][cell]) for name in ARMS}
        lines.append(f"- {cell}: " + "; ".join(f"{name} {'none' if v is None else v}" for name, v in firsts.items()))
    lines += ["", head.format("cell"), rule]
    lines += [f"| {cell} | {name} | {fmt(rows[name][cell])} |" for cell in ARM_CELLS for name in ARMS]
    (grid_dir / "arms.md").write_text("\n".join(lines) + "\n")
    (grid_dir / "arms.json").write_text(json.dumps({
        "fields": ARM_FIELDS, "rows": rows, "diagnostic": diagnosed, "patches": {k: v[2] for k, v in ARMS.items()},
        "first_a0b_steady": {cell: {name: _first_a0b(runs[name]["by_cell"][cell]) for name in ARMS} for cell in S}},
        indent=1))
