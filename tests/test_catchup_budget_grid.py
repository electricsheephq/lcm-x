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
RUN_TIMEOUT = int(os.environ.get("LCMX_GRID_RUN_TIMEOUT", "900"))  # seconds per runner process
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


def _reference_cells() -> list[dict]:
    """T1-285, issue 2878: shipped budgets with the reference agent's timing (37 s pre-call work, a 44 s condensation, a
    38 s leaf, the route estimate 44 s) and a prefix over target: the condensation and a leaf land in the same pass."""
    return [{**G1_FIXED, **_g1_load(1_000_000), "group": "R", "id": "r-shipped-cond-then-leaf", "window": 1_000_000,
             "pre": 37, "estimate": 44, "call": [44, 38], "delta": 0, "prefix": "over2x"}] + [
        # T1-289's former boundary cells (pre + condensation + leaf estimates of 189 s and 192 s against 210 - 20 s),
        # kept as regression cells: the pre-leaf leaf reserve is never released, so both keep the base's reserve
        {**G1_FIXED, **_g1_load(1_000_000), "group": "R", "id": f"r-release-{side}", "window": 1_000_000, "pre": pre,
         "estimate": 70, "call": [60, 60], "delta": 0, "prefix": "over2x"} for side, pre in (("inside", 49), ("outside", 52))]


_G120 = {**G1_FIXED, **_g1_load(600_000), "window": 600_000, "pre": 59, "estimate": None, "call": 45, "delta": 0,
         "prefix": "over2x"}
_R1M = {**G1_FIXED, **_g1_load(1_000_000), "window": 1_000_000, "delta": 0, "prefix": "over2x"}


def _regression_cells() -> list[dict]:
    """Review fix round 1 (group V), each red on e529d499 (the released pre-leaf leaf reserve) and green after:
    the reviewer's hardened cells, the verifier's starvation shape (1) at 60 passes, its route-recovery shapes (5) and
    its seven controls. ``call_kind`` and ``from_pass`` are the runner's per-kind latency and per-pass overrides."""
    kind = lambda cond, leaf: {"call_kind": {"condensation": cond, "leaf": leaf}}  # noqa: E731
    cells = [
        {**_G120, "id": "h-g120-params-call60-25", "call": [60, 25]},
        {**_R1M, "id": "h-pre37-est44-hang-30", "pre": 37, "estimate": 44, "call": ["hang", 30]},
        {**_R1M, "id": "h-pre37-est44-call80-38", "pre": 37, "estimate": 44, "call": [80, 38]},
        {**_R1M, "id": "x-hangcond-fastleaf-pre10", "pre": 10, "estimate": 30, "call": ["hang", 10]},
        {**_R1M, "id": "x-slowcond-fastleaf", "pre": 37, "estimate": 44, "call": [100, 30]},
        {**_G120, "id": "v1-starve-c70-l20-60p", "passes": 60, **kind(70, 20)},
        {**_G120, "id": "v-ctl-c70-l70", "passes": 24, **kind(70, 70)},
        {**_G120, "id": "v-ctl-pre37-c45-l45", "passes": 24, "pre": 37},
        {**_G120, "id": "v-ctl-c40-l20-gap700", "passes": 8, "turn_gap": 700, **kind(40, 20)},
        {**_G120, "id": "v-ctl-c42-l20-gap700", "passes": 8, "turn_gap": 700, **kind(42, 20)},
        {**_G120, "id": "v-ctl-c45-l35", "passes": 24, **kind(45, 35)},
        {**_G120, "id": "v-ctl-pre37-c70-l20", "passes": 24, "pre": 37, **kind(70, 20)},
        {**_G120, "id": "v-ctl-pre37-c70-l45", "passes": 24, "pre": 37, **kind(70, 45)},
    ]
    cells = [{**cell, "group": "V"} for cell in cells]
    return cells + [{**_G120, "group": "VR", "id": f"v5-recover-at-{at}", "passes": 30,
                     "from_pass": [{"pass": at, "pre": 5, "call": 5}]} for at in (9, 10, 11, 12)]


def _mark_cells() -> list[dict]:
    """Group M. (4): a depth target of 0.49 (M 490k) a prompt 1k over the trigger, so the pre-leaf condensation alone
    brings it under M: no extension leaf may follow. (3): a 400k trigger on a 1M window (M = 400k - 100k = 300k), where
    leaf_prep.levels has no low-water mark: the catch-up must engage."""
    return [{**_R1M, "group": "M", "id": "m4-depth-stale", "passes": 3, "pre": 37, "call": 40, "over_tokens": 1_000,
             "depth_ratio": 0.49},
            {**_R1M, "group": "M", "id": "m3-trigger-400k", "passes": 3, "pre": 37, "estimate": 40, "call": 40,
             "prefix": "none", "settings": {"context_threshold": 0.4}}]


CELLS = (_g1_cells() + _g2_cells() + _l_cells() + _steady_cells() + _depth_cells() + _reference_cells()
         + _regression_cells() + _mark_cells())
CELLS += _no_system_twins(CELLS)
BY_ID = {cell["id"]: cell for cell in CELLS}
G1 = [cell["id"] for cell in CELLS if cell["group"] == "G1"]
G2 = [cell["id"] for cell in CELLS if cell["group"] == "G2"]
L = [cell["id"] for cell in CELLS if cell["group"] == "L"]
S = [cell["id"] for cell in CELLS if cell["group"] == "S"]
N = [cell["id"] for cell in CELLS if cell["group"] == "N"]
# Classification on main b289240a (grid v2): every twin loses rows with no pointer in the returned list (A0), because
# the survival-fit notice goes only into a system row the view does not have. So both trees' A0 tests are pinned.
# Part 2's base already carries the issue-2901 fix (merge of fa4cad11 and main 2c75f52b), so the three fixed twins lose
# nothing on the base either: their base pin is lifted (each was a strict XPASS on that base).
KNOWN_2896 = [cell for cell in N if cell not in ["s-1m-l40-nosys", "s-1m-l45-d15-nosys", "s-1m-cond-then-leaf-nosys"]]
# Issue 2901: leaf selection charges a glued carrier at its remainder, so these twins no longer stall into the fit.
FIXED_2901 = ["s-1m-l40-nosys", "s-1m-l45-d15-nosys", "s-1m-cond-then-leaf-nosys"]
KNOWN_2896_A0 = [cell for cell in N if cell not in FIXED_2901]  # twins with A0 rows on the candidate
# Part 2: the candidate loses no row in s-600k-l40-prefix-nosys (A0 0 against the base's 2,080), so its candidate pin
# and its per-pass pin are lifted (both were strict XPASSes).
KNOWN_2896_A0 = [cell for cell in KNOWN_2896_A0 if cell != "s-600k-l40-prefix-nosys"]
# Issue 2901 moves these twins' trajectories: on some passes the candidate shows rows out of the list where the base did
# not, while its totals stay at most the base's. Only the per-pass clause is pinned; the totals test is not.
PER_PASS_2901 = ["d-depth0-past-100-nosys"]  # Part 2 lifts s-600k-l40-prefix-nosys (see KNOWN_2896_A0)
PER_PASS_2901_REASON = ("issues 2896, 904, 2883: the candidate's own later survival fit (notice not delivered without a "
                        "system row), a kept leaf not rendered under the hidden-only rule, and depth-0 nodes beyond "
                        "the per-level limit")
D = [cell["id"] for cell in CELLS if cell["group"] == "D"]
V = [cell["id"] for cell in CELLS if cell["group"] == "V"]
VR = [cell["id"] for cell in CELLS if cell["group"] == "VR"]
ALL = [cell["id"] for cell in CELLS]
NEGATIVE_CELLS = ["g1-00", "g1-11"]
GATE_SHARE_TARGET = 0.8  # G1 passes that reached the gate: triggered, or at or over the trigger and refused by a hold
ISSUE_2877 = "issue 2877"
# Known base cells (round 3 classification): rows leave the list behind a survival-fit notice (A0b) in all of them; in
# s-600k-l40-prefix a later fit's notice no longer names 37 of the first fit's rows, which no leaf covers yet (A0).
KNOWN_2877_BASE = ["g1-03", "g1-04", "g1-12", "g1-13", "g1-20", "g1-21", "g1-22", "g1-30", "g1-31", "g1-33", "g1-34",
                   "g1-40", "g1-41", "g1-42", "g1-43", "g1-44", "s-600k-l40-prefix",
                   "r-release-inside", "r-release-outside",  # T1-289's former boundary cells: the base fits at pass 9
                   # round 2 (6): the V/VR cells where the base puts rows behind a fit notice (v-ctl-c70-l70 also loses
                   # 512 rows that a later fit's notice no longer names, A0)
                   "v-ctl-c45-l35", "v-ctl-c70-l70", "v-ctl-pre37-c70-l45",
                   "v5-recover-at-9", "v5-recover-at-10", "v5-recover-at-11", "v5-recover-at-12"]
# The candidate's A0 pins under issue 2877: v-ctl-c70-l70 loses the same 512 rows as the base (B reproduces the base's
# fits there and never worsens them; test_a0_candidate_no_worse_than_the_base stays unpinned).
KNOWN_2877_A0 = ["v-ctl-c70-l70"]
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
    """A4's strict cells: live gates of the catch-up gain (their xfail markers were removed with the change)."""
    assert grid["cand"]["header"]["catchup_constant"], "CAND_TREE has no catch-up constant"


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
    """Per pass index: a base call needs a candidate call unless the candidate's own prompt is under its trigger, and a
    hold is compared cumulatively (T1-289, below). T1-285: cumulative leaves and drained tokens (removed by stored
    leaves and condensations only, never by a survival-fit cut) are compared with a one-pass lag (the candidate at pass
    k against the base at pass k - 1), and at the end of the walk the candidate has at least the base's totals."""
    found = []
    for index, (b, c) in enumerate(zip(base, cand)):
        # T1-295: a base call at pass k needs a candidate call at k or k + 1, unless the candidate is under its trigger
        later = cand[index + 1]["calls_started"] if index + 1 < len(cand) else 0
        if b["calls_started"] > 0 and not (c["calls_started"] > 0 or later > 0 or c["prompt_before"] < c["trigger"]):
            found.append(f"pass {b['pass']}: calls started 0 (and 0 at the next pass) with the prompt "
                         f"{c['prompt_before']} at or over the trigger {c['trigger']}, base {b['calls_started']}")
        for key, label in (("leaves_cum", "leaves"), ("summary_drained_cum", "drained")):
            if index and c[key] < base[index - 1][key]:
                found.append(f"pass {b['pass']}: cumulative {label} {c[key]} < base {base[index - 1][key]} at pass "
                             f"{base[index - 1]['pass']}")
        # T1-289: holds compared cumulatively, the candidate at pass k against the base at pass k + 1, excused where
        # the candidate's own prompt is under its trigger
        cand_holds = sum(r["hold_armed"] for r in cand[:index + 1])
        base_holds = sum(r["hold_armed"] for r in base[:index + 2])
        if cand_holds > base_holds and c["prompt_before"] >= c["trigger"]:
            found.append(f"pass {b['pass']}: the candidate armed {cand_holds} holds, the base {base_holds} by pass "
                         f"{min(index + 1, len(base) - 1)}")
    if sum(r["hold_armed"] for r in cand) > sum(r["hold_armed"] for r in base):
        found.append(f"end of walk: the candidate armed {sum(r['hold_armed'] for r in cand)} holds, the base "
                     f"{sum(r['hold_armed'] for r in base)}")
    # T1-295: the end of the walk needs the base's leaves, and its drained tokens or a final whole prompt no larger
    # than the base's (with A0 holding: no candidate row lost without a pointer)
    if base and cand and cand[-1]["leaves_cum"] < base[-1]["leaves_cum"]:
        found.append(f"end of walk: cumulative leaves {cand[-1]['leaves_cum']} < base {base[-1]['leaves_cum']}")
    smaller = cand[-1]["prompt_after"] <= base[-1]["prompt_after"] and not a0_violations(cand) if base and cand else 0
    if base and cand and cand[-1]["summary_drained_cum"] < base[-1]["summary_drained_cum"] and not smaller:
        found.append(f"end of walk: cumulative drained {cand[-1]['summary_drained_cum']} < base "
                     f"{base[-1]['summary_drained_cum']} and final prompt {cand[-1]['prompt_after']} > base "
                     f"{base[-1]['prompt_after']}")
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
    if cell in (KNOWN_2877_BASE if tree == "base" else KNOWN_2877_A0):
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=ISSUE_2877))
    if cell in D:
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
    """Per pass: the candidate's A0b rows, and (in the pinned A0 cells) its A0 rows, are at most the base's."""
    if cell in PER_PASS_2901:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=PER_PASS_2901_REASON))
    base, cand = grid["base"]["by_cell"][cell], grid["cand"]["by_cell"][cell]
    for b, c, nb, nc in zip(base, cand, a0b_rows(base), a0b_rows(cand)):
        assert nc <= nb, f"A0b in cell {cell} pass {b['pass']}: candidate {nc} rows behind a notice, base {nb}"
        lost = [r["loss"]["missing_count"] if r["loss"] else 0 for r in (b, c)]
        assert lost[1] <= lost[0], f"A0 in cell {cell} pass {b['pass']}: candidate {lost[1]} rows, base {lost[0]}"


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
    assert cand["leaves"] >= 1, f"A4 strict in cell g1-cond-then-leaf: the candidate stored {cand['leaves']} leaves"


A5_ADDED = 0.85 * 120.0 + 15.0  # an extension call's cap at the configured 120 s timeout, plus 15 s bookkeeping: 117 s


def test_a5_hang_cell_wait(grid):
    base, cand = (grid[tree]["by_cell"]["g1-hang-every-pass"] for tree in ("base", "cand"))
    for b, c in zip(base, cand):
        assert c["wall"] - b["wall"] <= A5_ADDED, (
            f"A5 pass {b['pass']}: the candidate added {c['wall'] - b['wall']:.1f} s over the base")
    assert not a2_violations(cand) and not a2_violations(base)


# A host whose stream deadline acts as a no-progress window: it ignores the call's timeout and ends a call that is still
# running at the installed deadline with the host's deadline error (executed inside the CAND runner).
PATCHES["host_window"] = (
    "import sys as _sys\n"
    "escalation._host_stream_deadline_seam = lambda: True\n"
    "def _host_window(self, prompt, max_tokens, model='', timeout=None, reasoning_effort=''):\n"
    "    took, deadline = self.seconds(self.started), escalation._summary_call.stream_deadline\n"
    "    self.started += 1\n"
    "    self.timeouts.append(None if timeout is None else round(float(timeout), 3))\n"
    "    self.kinds.append('leaf' if 'user turn' in prompt else 'condensation')\n"
    "    ends = None if took is None else self.clock.offset + took\n"
    "    cut = ends is None or (deadline is not None and ends > deadline)\n"
    "    self.cut_flags.append(cut)\n"
    "    if cut:\n"
    "        self.cut += 1\n"
    "        self.clock.offset = deadline if deadline is not None else self.clock.offset + 600.0\n"
    "        raise TimeoutError('host compression deadline exceeded')\n"
    "    self.clock.offset = ends\n"
    "    return self.summary\n"
    "_sys.modules['__main__'].Provider.__call__ = _host_window\n")


@pytest.mark.skipif(BASE_TREE is None, reason=NO_BASE)
def test_extension_call_on_a_no_progress_window_host_is_never_a_hang(grid_dir):
    """An extension-only call that hangs on a host whose deadline is a no-progress window ends at its own capped
    timeout (the host deadline it installs), so it is a budget cut: no hang, no strike, the circuit stays closed."""
    cell = {**BY_ID["g1-hang-every-pass"], "id": "g1-host-window"}
    records = _run(CAND_TREE, [cell], grid_dir, "host-window", patch="host_window")["by_cell"][cell["id"]]
    cut = [r["pass"] for r in records if any(r["call_cut_flags"][1:])]
    assert cut, f"no extension call was cut by the host deadline: {records}"
    hung = [r["pass"] for r in records if r["circuit"]["state"] != "closed"
            or r["circuit"]["last_error_class"] == "hung" or r["holds"]["hung_strike"]]
    assert not hung, f"an extension call was classified as a hang in passes {hung}"
    assert not a2_violations(records)


def _base_sweep_target(window: int) -> int:
    """The base's default summary-prefix target at context_threshold 0.5 (PR 2876: 5% of the window above 480,000,
    capped at a quarter of the trigger): 50,000 on 1M, 30,000 on 600k."""
    trigger = window // 2
    return max(min(24_000, trigger // 4), min(int(0.05 * window), trigger // 4))


# Round 2 (5): B's area per stored leaf is compared with the base's by integer cross-products, with this many tokens of
# total area for the rounding of the per-leaf summary count (s-1m-l45-d15 is 0.6 tokens over the exact proportion).
AREA_TOLERANCE = 2


@pytest.mark.parametrize("cell", S)
def test_steady_cells_condense_and_keep_their_leaves(grid, cell):
    """A6 restated (the RM's ruling on review round 1): per pass the candidate stores at least the base's leaves (a pass
    where its own prompt is under its trigger is excused), a pass that condenses still stores a leaf, no pass passes
    250 s; and over the walk its summary area per stored leaf (peak area / leaves stored, within AREA_TOLERANCE tokens
    of total area), its end prompt (against the base's last non-fit prompt), its mean prompt, and its A0 and A0b rows
    are each at most the base's. The 60,000 / 36,000 figure (1.2 x the base's sweep target) is reported as a target
    only. Boundedness is test_a6_bounded_summary_area."""
    base, cand = (grid[tree]["by_cell"][cell] for tree in ("base", "cand"))
    fewer = [c["pass"] for b, c in zip(base, cand) if c["leaves"] < b["leaves"] and c["prompt_before"] >= c["trigger"]]
    assert not fewer, f"{cell}: fewer leaves than the base in passes {fewer}"
    no_leaf = [c["pass"] for c in cand if c["condensations"] and not c["leaves"]]
    assert not no_leaf, f"{cell}: a condensation without a leaf in passes {no_leaf}"
    assert not a2_violations(cand), a2_violations(cand)
    (area_b, leaves_b), (area_c, leaves_c) = (
        (max(r["summary_area"] for r in records), max(1, records[-1]["leaves_cum"])) for records in (base, cand))
    # area_c / leaves_c <= area_b / leaves_b, in integers, with AREA_TOLERANCE tokens of total area for rounding
    assert area_c * leaves_b <= area_b * leaves_c + AREA_TOLERANCE * leaves_b, (
        f"{cell}: summary area per stored leaf: candidate {area_c} / {leaves_c}, base {area_b} / {leaves_b}")
    # T1-327: the end prompt is compared with the base's prompt on its last pass that was not itself a fit pass (one that
    # moved more rows behind a notice than the pass before: a lossy fit cut is not the base doing better), and the mean
    # prompt over the walk must also be at most the base's
    behind = a0b_rows(base)
    non_fit = [r for i, r in enumerate(base) if behind[i] <= (behind[i - 1] if i else 0)]
    base_end = (non_fit or base)[-1]["prompt_after"]
    assert cand[-1]["prompt_after"] <= base_end, f"{cell}: end prompt {cand[-1]['prompt_after']} > base {base_end}"
    mean = {tree: sum(r["prompt_after"] for r in records) / len(records) for tree, records in (("base", base),
                                                                                                ("cand", cand))}
    assert mean["cand"] <= mean["base"], f"{cell}: mean prompt over the walk {mean}"
    lost = {tree: sum(r["loss"]["missing_count"] for r in records if r["loss"]) for tree, records in (("base", base),
                                                                                                    ("cand", cand))}
    assert lost["cand"] <= lost["base"], f"{cell}: A0 rows {lost}"
    assert sum(a0b_rows(cand)) <= sum(a0b_rows(base)), f"{cell}: A0b rows {sum(a0b_rows(cand))} > {sum(a0b_rows(base))}"


@pytest.mark.parametrize("cell", V)
def test_regression_cells_never_behind_the_base(grid, cell):
    """On every pass, no lag: cumulative leaves and cumulative leaves + condensations at least the base's, rows behind a
    notice (A0b) at most the base's, and no hung strike the base lacks. A1's excuse (the RM's ruling on round 1): a
    count shortfall is excused at a pass where the candidate's prompt is under its trigger AND at or below the base's."""
    base, cand = (grid[tree]["by_cell"][cell] for tree in ("base", "cand"))
    assert len(base) == len(cand), f"{cell}: walks of {len(base)} and {len(cand)} passes"
    done = {"base": 0, "cand": 0}
    for b, c, nb, nc in zip(base, cand, a0b_rows(base), a0b_rows(cand)):
        done["base"] += b["leaves"] + b["condensations"]
        done["cand"] += c["leaves"] + c["condensations"]
        excused = c["prompt_before"] < c["trigger"] and c["prompt_before"] <= b["prompt_before"]
        assert excused or c["leaves_cum"] >= b["leaves_cum"], (
            f"{cell} pass {b['pass']}: leaves {c['leaves_cum']} < {b['leaves_cum']}")
        assert excused or done["cand"] >= done["base"], f"{cell} pass {b['pass']}: leaves + condensations {done}"
        assert nc <= nb, f"{cell} pass {b['pass']}: A0b {nc} > base {nb}"
        assert not (c["holds"]["hung_strike"] and not b["holds"]["hung_strike"]), f"{cell} pass {b['pass']}: hung strike"


@pytest.mark.parametrize("cell", VR)
def test_recovery_cells_bounded_lag(grid, cell):
    """Route recovery (the RM's restatement): never behind by more than one firing's publications (the base's largest
    one-pass count of leaves + condensations), never behind for longer than one hold period (600 s of turn gaps,
    counted in passes) except where A1's excuse holds (the candidate under its trigger and at or below the base's
    prompt), the same or a better rate over whole firing cycles (see ``rate``), rows behind a notice at most the base's,
    and no hung strike the base lacks."""
    base, cand = (grid[tree]["by_cell"][cell] for tree in ("base", "cand"))
    assert len(base) == len(cand), f"{cell}: walks of {len(base)} and {len(cand)} passes"
    firing = max(r["leaves"] + r["condensations"] for r in base)
    hold_passes = -(-600 // int(BY_ID[cell]["turn_gap"]))
    done, behind_for = {"base": 0, "cand": 0}, 0
    for b, c, nb, nc in zip(base, cand, a0b_rows(base), a0b_rows(cand)):
        done["base"] += b["leaves"] + b["condensations"]
        done["cand"] += c["leaves"] + c["condensations"]
        deficit = done["base"] - done["cand"]
        assert deficit <= firing, f"{cell} pass {b['pass']}: behind by {deficit} > one firing ({firing})"
        excused = c["prompt_before"] < c["trigger"] and c["prompt_before"] <= b["prompt_before"]
        behind_for = behind_for + 1 if deficit > 0 and not excused else 0
        assert behind_for <= hold_passes, f"{cell} pass {b['pass']}: behind for {behind_for} passes > {hold_passes}"
        assert nc <= nb, f"{cell} pass {b['pass']}: A0b {nc} > base {nb}"
        assert not (c["holds"]["hung_strike"] and not b["holds"]["hung_strike"]), f"{cell} pass {b['pass']}: hung strike"

    def rate(records):
        """Round 2 (4) and round 3 (3), T1-324: publications per pass over whole firing cycles, from the first firing
        to the last, with half of the first and half of the last firing's count inside that span (each borders a cycle
        outside it). The open cycle after the last firing is left out, unless it has run past one hold period (the
        tree stopped firing), when it counts whole, to the end of the walk. F4: an open tail of at least hold_passes
        idle passes counts as stopped."""
        fired = [(r["pass"], r["leaves"] + r["condensations"]) for r in records if r["leaves"] + r["condensations"]]
        if not fired:
            return 0.0
        if len(fired) == 1 or len(records) - 1 - fired[-1][0] >= hold_passes:
            return sum(n for _, n in fired) / (len(records) - fired[0][0])
        inside = fired[0][1] / 2 + sum(n for _, n in fired[1:-1]) + fired[-1][1] / 2
        return inside / (fired[-1][0] - fired[0][0])
    assert rate(cand) >= rate(base), f"{cell}: rate {rate(cand):.3f} < base {rate(base):.3f}"
    # T1-337 (a): eligible work left unfired. Once the candidate has gone a hold period or more without a firing, a pass
    # whose prompt is at or over its trigger is work it no longer does (a large early burst cannot buy that off)
    last = max((r["pass"] for r in cand if r["leaves"] + r["condensations"]), default=-1)
    starved = [r["pass"] for r in cand if r["pass"] - last >= hold_passes and r["prompt_before"] >= r["trigger"]]
    assert not starved, f"{cell}: eligible work, no firing for a hold period or more, at passes {starved}"


def test_stale_estimate_no_extension_leaf_below_the_mark(grid):
    """(4): once the pre-leaf condensation brings the prompt under M, only the base admits (pass 0 matches the base)."""
    base, cand = (grid[tree]["by_cell"]["m4-depth-stale"][0] for tree in ("base", "cand"))
    assert base["condensations"] >= 1, f"premise: the base condenses first: {base}"
    assert (cand["leaves"], cand["calls_started"]) == (base["leaves"], base["calls_started"]), (cand, base)


def test_mark_per_rule_4_engages_at_a_400k_trigger(grid):
    """(3): M = threshold - max(2 x chunk, 0.1 x window) engages the catch-up where leaf_prep.levels has no mark."""
    base, cand = (grid[tree]["by_cell"]["m3-trigger-400k"][0] for tree in ("base", "cand"))
    assert cand["leaves"] > base["leaves"], f"candidate {cand['leaves']} leaves, base {base['leaves']}"


# (2): an extension-only call on a host with the stream-deadline seam. The pass's second call (the extension call) is
# either masked at its own deadline as a non-timeout error (MASK), or returns late, 11 s past it (LATE). Round 2 (2):
# PAUSE seconds pass between installing the stream deadline and sampling the call's start (a scheduling stall).
_EXT_HOST = (
    "import sys as _sys\n"
    "escalation._host_stream_deadline_seam = lambda: setattr(clock, 'offset', clock.offset + PAUSE) or True\n"
    "_P = _sys.modules['__main__'].Provider\n"
    "_orig = _P.__call__\n"
    "def _ext(self, prompt, max_tokens, model='', timeout=None, reasoning_effort=''):\n"
    "    if self.started == 0:\n"
    "        return _orig(self, prompt, max_tokens, model=model, timeout=timeout, reasoning_effort=reasoning_effort)\n"
    "    deadline = escalation._summary_call.stream_deadline\n"
    "    self.started += 1; self.cut += 1\n"
    "    self.timeouts.append(None if timeout is None else round(float(timeout), 3))\n"
    "    self.kinds.append('leaf' if 'user turn' in prompt else 'condensation'); self.cut_flags.append(True)\n"
    "    self.clock.offset = deadline + LATE\n"
    "    raise (TimeoutError('timed out') if LATE else RuntimeError('invalid argument: stream closed'))\n"
    "_P.__call__ = _ext\n")
PATCHES["ext_masked"] = _EXT_HOST.replace("LATE", "0").replace("PAUSE", "0")
PATCHES["ext_late"] = _EXT_HOST.replace("LATE", "11").replace("PAUSE", "0")
PATCHES["ext_masked_pause_1_1"] = _EXT_HOST.replace("LATE", "0").replace("PAUSE", "1.1")
PATCHES["ext_masked_pause_2"] = _EXT_HOST.replace("LATE", "0").replace("PAUSE", "2.0")


@pytest.mark.skipif(BASE_TREE is None, reason=NO_BASE)
@pytest.mark.parametrize("variant", ["ext_masked", "ext_late", "ext_masked_pause_1_1", "ext_masked_pause_2"])
def test_extension_call_classified_against_its_own_deadline(grid_dir, variant):
    """(2): (a) a timeout the host masks at the extension call's own deadline charges no route failure; (b) a return
    past its own timeout opens no circuit and marks no hang. Round 2 (2): (a) holds after a pause between installing
    the deadline and the call's start, because the call is classified against the deadline it installed."""
    cell = {**BY_ID["g1-hang-every-pass"], "id": f"e-{variant}", "passes": 3}
    records = _run(CAND_TREE, [cell], grid_dir, variant, patch=variant)["by_cell"][cell["id"]]
    assert any(any(r["call_cut_flags"][1:]) for r in records), f"no extension call ran: {records}"
    bad = [(r["pass"], r["circuit"]) for r in records if r["circuit"]["state"] != "closed"
           or r["circuit"]["last_error_class"] in ("provider_failure", "hung") or r["holds"]["hung_strike"]]
    assert not bad, f"{variant}: the extension call was charged to the route: {bad}"


# Round 2 (1): every summary-frontier scan costs 0.2 s on both trees. The base admits pass 0's leaf at 39.9 s (59.9 s
# with its 20 s estimate, under the 60 s soft target); B must not spend a scan of its own before that admission.
PATCHES["frontier_cost"] = (
    "_ft = engine.LCMEngine._summary_frontier_tokens\n"
    "def _costly(self, _ft=_ft):\n"
    "    clock.offset += 0.2\n"
    "    return _ft(self)\n"
    "engine.LCMEngine._summary_frontier_tokens = _costly\n")


@pytest.mark.skipif(BASE_TREE is None, reason=NO_BASE)
def test_no_candidate_only_frontier_scan_before_the_leaf(grid_dir):
    """(1): a pre-leaf condensation brings B's estimate under M (depth 0.49), so only the base admit applies; with a
    charged scan the candidate still admits every call the base admits, pass by pass."""
    cell = {**_R1M, "group": "M", "id": "m1-frontier-scan", "passes": 4, "over_tokens": 1_000, "pre": 28.9,
            "estimate": 20, "call": 20, "call_kind": {"condensation": 10, "leaf": 20}, "depth_ratio": 0.49}
    base, cand = (_run(tree, [cell], grid_dir, f"frontier-{name}", patch="frontier_cost")["by_cell"][cell["id"]]
                  for name, tree in (("base", BASE_TREE), ("cand", CAND_TREE)))
    assert base[0]["leaves"] >= 1 and base[0]["condensations"] >= 1, "premise: the base condenses, then stores a leaf"
    for b, c in zip(base, cand):
        assert c["leaves_cum"] >= b["leaves_cum"], f"pass {b['pass']}: leaves {c['leaves_cum']} < {b['leaves_cum']}"


# Round 2 (9) and round 3: a summariser that fails the calls ``FAIL_IF`` selects with a genuine provider error, alike
# on both trees. FAIL_IF sees ``n`` (the call's number in the walk), ``p`` (the pass), ``idx`` (the call's index in the
# pass), ``ext`` (an extension-only call), ``fb`` (a call on the fallback route) and ``kind`` (a condensation while
# _condense_summary_nodes runs, else a leaf); a failure ends at
# ``END``. ``p`` starts at -2: reset_pass runs in the Provider's constructor and again before pass 0. Per call the
# patch tags ``call_cut_flags``: a failure is "ext_fail" or "fail" (base-admitted), prefixed
# "fb_" on the fallback route, and a fallback call that succeeds is "fb_call". At the end of each pass it adds
# "pass:summary_failure" when the budget's sticky summary failure is set, and "pass:fb_open" while the fallback's
# circuit is not closed. ``SETUP`` runs first.
_FAILING = (
    "import sys as _sys\n"
    "SETUP\n"
    "_P = _sys.modules['__main__'].Provider\n"
    "_orig, _reset, _admit = _P.__call__, _P.reset_pass, escalation.ForegroundBudget.admit\n"
    "_st = {'end': -1e9, 'n': 0, 'p': -2, 'ext': False, 'b': None, 'prov': None, 'cond': False}\n"
    "_cs = engine.LCMEngine._condense_summary_nodes\n"
    "def _cs_seen(self, *args, _cs=_cs, **kwargs):\n"
    "    _st['cond'] = True\n"
    "    try:\n"
    "        return _cs(self, *args, **kwargs)\n"
    "    finally:\n"
    "        _st['cond'] = False\n"
    "engine.LCMEngine._condense_summary_nodes = _cs_seen\n"
    "def _reset_seen(self, _reset=_reset):\n"
    "    _st['p'] += 1\n"
    "    return _reset(self)\n"
    "_P.reset_pass = _reset_seen\n"
    "def _admit_seen(self, route_key, _admit=_admit):\n"
    "    left = _admit(self, route_key)\n"
    "    _st['ext'], _st['b'] = bool(getattr(self, 'extended', False)), self\n"
    "    return left\n"
    "escalation.ForegroundBudget.admit = _admit_seen\n"
    "def _failing(self, prompt, max_tokens, model='', timeout=None, reasoning_effort=''):\n"
    "    _st['n'] += 1\n"
    "    _st['prov'] = self\n"
    "    n, p, idx, ext, fb = _st['n'], _st['p'], self.started, _st['ext'], model == 'fb-model'\n"
    "    kind = 'condensation' if _st['cond'] else 'leaf'\n"
    "    if FAIL_IF:\n"
    "        self.started += 1\n"
    "        self.timeouts.append(None if timeout is None else round(float(timeout), 3))\n"
    "        self.kinds.append(kind)\n"
    "        self.cut_flags.append(('fb_' if fb else '') + ('ext_fail' if ext else 'fail'))\n"
    "        self.clock.offset = END\n"
    "        _st['end'] = self.clock.offset\n"
    "        raise RuntimeError(ERROR)\n"
    "    try:\n"
    "        return _orig(self, prompt, max_tokens, model=model, timeout=timeout, reasoning_effort=reasoning_effort)\n"
    "    finally:\n"
    "        _st['end'] = self.clock.offset\n"
    "        if fb:\n"
    "            self.cut_flags[-1] = 'fb_call'\n"
    "_P.__call__ = _failing\n"
    "_finish = engine.LCMEngine._finish_foreground_budget\n"
    "def _finish_seen(self, budget, *args, _finish=_finish, **kwargs):\n"
    "    if _st['prov'] is not None and budget.summary_failure:\n"
    "        _st['prov'].cut_flags.append('pass:summary_failure')\n"
    "    if _st['prov'] is not None and self._summary_circuit_breaker.route_status(['fb-model']).get('state') not in (\n"
    "            None, 'closed'):\n"
    "        _st['prov'].cut_flags.append('pass:fb_open')\n"
    "    return _finish(self, budget, *args, **kwargs)\n"
    "engine.LCMEngine._finish_foreground_budget = _finish_seen\n")


def _failing(fail_if: str, error: str, end: str = "self.clock.offset + 3.0", setup: str = "") -> str:
    return (_FAILING.replace("FAIL_IF", fail_if).replace("ERROR", repr(error)).replace("END", end)
            .replace("SETUP", setup))


_LIMIT = "self.clock.offset - _st['end'] < 10.0"  # a call that starts within 10 s of the previous call's end
# (1)/(5): an enclosing host deadline 40 s into the pass closes any call still running then (the stream closed); the
# fallback route has a 15 s estimate, so the base would admit it after the closed extension call
_HOST40 = "_st['b'].t0 + 40.0"
_FB15 = ("_fe = escalation.ForegroundEstimates.__init__\n"
         "def _fe15(self, *a, _fe=_fe, **k):\n"
         "    _fe(self, *a, **k)\n"
         "    self.record_call('fb-model', 15.0)\n"
         "escalation.ForegroundEstimates.__init__ = _fe15")
# (7): twelve depth-0 nodes of 7,000 tokens, so one pre-leaf condensation leaves the frontier over the 50,000 target and
# the post-leaf condensation runs under the catch-up
_SEED12 = ("_runner = _sys.modules['__main__']\n"
           "def _seed12(mods, engine, cell, target):\n"
           "    for index in range(12):\n"
           "        engine._dag.add_node(mods['dag'].SummaryNode(\n"
           "            session_id=engine._session_id, depth=0, summary=_runner.sized_summary(7_000), token_count=7_000,\n"
           "            source_token_count=14_000, source_ids=[], source_type='messages', created_at=float(index + 1)))\n"
           "_runner._seed_prefix = _seed12")
PATCHES["burst"] = _failing(_LIMIT, "Error code: 429 - rate limit exceeded")
PATCHES["burst_once"] = _failing(f"n <= 3 and {_LIMIT}", "Error code: 429 - rate limit exceeded")
# the control: every third call fails, whatever its index in the pass. Its circuit stays closed on both trees; the
# candidate's per-pass counts move, because the catch-up ends at its first failed extension call
PATCHES["flaky"] = _failing("n % 3 == 0", "HTTP 500 upstream", end="self.clock.offset + 5.0")
PATCHES["host40"] = _failing(f"self.clock.offset + self.seconds(idx) > {_HOST40}", "stream closed",
                             end=f"max(self.clock.offset, {_HOST40})", setup=_FB15)
PATCHES["fb_after1"] = _failing("idx >= 1", "HTTP 500 upstream", end="self.clock.offset + 2.0")
PATCHES["fb_down"] = _failing("(p < 3 and idx >= 1) or (p >= 3 and not fb)", "HTTP 500 upstream",
                              end="self.clock.offset + 2.0")
PATCHES["ext_condensation"] = _failing("ext and kind == 'condensation'", "HTTP 500 upstream",
                                       end="self.clock.offset + 2.0", setup=_SEED12)
_FB = {"settings": {"summary_fallback_models": ["fb-model"]}}
_FAILING_CELLS = {
    "burst": {**_R1M, "id": "f-burst", "passes": 14, "pre": 37, "estimate": 44, "call": 40, "prefix": "under"},
    "burst_once": {**_R1M, "id": "f-burst-once", "passes": 20, "pre": 37, "estimate": 44, "call": 40, "prefix": "under"},
    "flaky": {"group": "S", "id": "f-flaky", "passes": 12, "raw_tokens": 150_000, "over_tokens": 20_000,
              "growth_tokens": 26_000, "turn_gap": 60, "finalize": 32, "chunk": 24_000, "summary_tokens": 1_500,
              "window": 1_000_000, "pre": 37, "estimate": 44, "call": 40, "delta": 0, "prefix": "none"},
    "host40": {**_R1M, **_FB, "id": "f-host40", "passes": 3, "pre": 5, "estimate": 44, "call": [25, 40],
               "prefix": "under"},
    "fb_after1": {**_R1M, **_FB, "id": "f-fb-after1", "passes": 8, "pre": 5, "estimate": 50, "call": 10,
                  "prefix": "under"},
    "fb_down": {**_R1M, **_FB, "id": "f-fb-down", "passes": 10, "pre": 5, "estimate": 50, "call": 10,
                "prefix": "under"},
    "ext_condensation": {**_R1M, "id": "f-ext-condensation", "passes": 3, "pre": 5, "estimate": 40, "call": 40,
                         "raw_tokens": 24_000, "prefix": "none"},
}


@pytest.mark.skipif(BASE_TREE is None, reason=NO_BASE)
@pytest.mark.parametrize("variant", list(_FAILING_CELLS))
def test_extension_call_failure_counts_nothing(grid_dir, variant):
    """(9), round 3 (1, 5, 6, 7): a genuine provider failure of an extension-only call charges no route, ends the
    route chain (no fallback call follows it in the pass) and leaves the budget's sticky summary failure as the base
    leaves it; base-admitted calls are charged as before. Per pass: the candidate's cumulative leaves at least the
    base's (A1's excuse: its prompt under its trigger and at or below the base's), at most one failed extension call,
    no call of any route after one (only the pass: diagnostics follow it), no open circuit (primary or fallback) the base does not have, and a summary failure
    only with a base-admitted failure. With the primary route down (fb_down) the prompt is also never above the base's
    (with the same excuse)."""
    cell = _FAILING_CELLS[variant]
    base, cand = (_run(tree, [cell], grid_dir, f"{variant}-{name}", patch=variant)["by_cell"][cell["id"]]
                  for name, tree in (("base", BASE_TREE), ("cand", CAND_TREE)))
    assert any("ext_fail" in r["call_cut_flags"] for r in cand), f"premise: no extension-only call failed: {cand}"
    for b, c in zip(base, cand):
        where, flags = f"{variant} pass {b['pass']}", c["call_cut_flags"]
        excused = c["prompt_before"] < c["trigger"] and c["prompt_before"] <= b["prompt_before"]
        assert excused or c["leaves_cum"] >= b["leaves_cum"], f"{where}: leaves {c['leaves_cum']} < {b['leaves_cum']}"
        assert c["circuit"]["state"] == "closed" or b["circuit"]["state"] != "closed", (
            f"{where}: circuit {c['circuit']}, base {b['circuit']}")
        assert flags.count("ext_fail") + flags.count("fb_ext_fail") <= 1, f"{where}: {flags}"
        failed = [i for i, flag in enumerate(flags) if flag in ("ext_fail", "fb_ext_fail")]
        after = [flag for flag in flags[failed[0] + 1:] if not str(flag).startswith("pass:")] if failed else []
        assert not after, f"{where}: a call after the extension-only failure: {flags}"
        assert "pass:fb_open" not in flags or "pass:fb_open" in b["call_cut_flags"], f"{where}: fallback circuit"
        assert "pass:summary_failure" not in flags or {"fail", "fb_fail"} & set(flags), (
            f"{where}: the sticky summary failure names an extension-only failure: {flags}")
        if variant == "fb_down":
            assert excused or c["prompt_after"] <= b["prompt_after"], (
                f"{where}: prompt {c['prompt_after']} > base {b['prompt_after']}")


# A6 boundedness, pinned BY CELL NAME on both trees (the RM's ruling): the summary area grows without bound in every
# steady cell on the base too, because the pre-leaf condensation's grant (about 19-23 s against a 44 s call) never lets
# a condensation land. Pre-existing; the pin covers this boundedness clause only.
ISSUE_2878 = "issue 2878: the starved pre-leaf condensation grant; the summary area grows on both trees"
KNOWN_2878 = ["s-1m-l40", "s-1m-l45-d15", "s-600k-l40-prefix", "s-1m-cond-then-leaf"]


@pytest.mark.parametrize("tree", ["base", "cand"])
@pytest.mark.parametrize("cell", S)
def test_a6_bounded_summary_area(grid, cell, tree, request):
    a6 = _a6(grid[tree]["by_cell"][cell])
    assert a6["passes"] == A6_TO, f"A6 in cell {cell}: a walk of {a6['passes']} passes"
    if cell in KNOWN_2878:
        request.applymarker(pytest.mark.xfail(strict=True, raises=AssertionError, reason=ISSUE_2878))
    assert not a6["grows"], f"A6 on {tree} in cell {cell}: {a6}"


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
    "B (candidate tree)": ({}, {}, ""),  # the catch-up change itself: CAND_TREE with the base's settings
}
B_ARM = "B (candidate tree)"
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
        runs[name] = _run(CAND_TREE if name == B_ARM else BASE_TREE, cells, grid_dir, f"arm-{index}", patch=f"arm-{index}" if patch else "")
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
