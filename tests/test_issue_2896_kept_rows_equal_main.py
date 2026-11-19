"""#2896 C1 and C11: on a list with no system row, the fit keeps exactly main's rows and only adds the block on top.

Two trees, like tests/test_catchup_budget_grid.py: LCMX_GRID_BASE is a copy of the base commit, this tree (or
LCMX_GRID_CAND) the candidate; with no base the tests skip (CI has none). tests/behaviour/notice_fit_sweep.py runs the
same synthetic cases on each tree in its own isolated process; the records are compared row by row, never by count:
  C1  the band sweep and the direct-fit sweep (carrier, carrier_sep2, two_carriers, twofits): the returned rows (with
      notice blocks left out), the summary nodes shown, the dropped count and the carrier emission equal main's at
      every cap, on both fits of twofits; the candidate returns at most one block, main none.
  C11 a randomised differential over every shape, pasted blocks and ignore patterns: the fit's rows and the store's
      rows after the host's replay (as returned, merged, a new turn, session end) and in a cold process equal main's.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_catchup_budget_grid import BASE_TREE, CAND_TREE, NO_BASE, RUN_TIMEOUT, _extra_paths

SWEEP = Path(__file__).resolve().parent / "behaviour" / "notice_fit_sweep.py"

pytestmark = pytest.mark.skipif(BASE_TREE is None, reason=NO_BASE)


def _sweep(tree: Path, sweep: str, out_dir: Path) -> dict:
    name = f"{'base' if tree == BASE_TREE else 'cand'}-{sweep}"
    out, work, home = out_dir / f"{name}.json", out_dir / f"{name}-work", out_dir / f"{name}-home"
    work.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-I", "-B", str(SWEEP), "--tree", str(tree), "--sweep", sweep, "--out", str(out),
               "--workdir", str(work)]
    for entry in _extra_paths():
        command += ["--syspath", entry]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "HERMES_HOME": str(home / "hermes"), "LANG": "C.UTF-8",
           "PYTHONHASHSEED": "0"}
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT)
    (out_dir / f"{name}-stderr.txt").write_text(done.stderr)
    assert done.returncode == 0, f"sweep {sweep} on {tree} exited {done.returncode}: {done.stderr[-2000:]}"
    output = json.loads(out.read_text())
    assert Path(output["tree"]) == tree and output["modules"]
    assert all(Path(module).is_relative_to(tree) for module in output["modules"]), "a sweep loaded another tree"
    return {record["key"]: record for record in output["records"]}


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> Path:
    configured = os.environ.get("LCMX_GRID_OUT")
    path = Path(configured) / "notice-fit-sweep" if configured else tmp_path_factory.mktemp("notice-fit-sweep")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fit_differences(key: str, base: dict, cand: dict) -> list[str]:
    found = []
    for field in ("fitted", "rows", "nodes", "dropped"):
        if base[field] != cand[field]:
            found.append(f"{key}: {field} differ")
    if sum(base["blocks"]):
        found.append(f"{key}: main returned a block")
    if sum(cand["blocks"]) > 1:
        found.append(f"{key}: {sum(cand['blocks'])} blocks")
    # the carrier emission is main's; a head carrier that takes the block is registered where main has none
    if (base["emission_row"], base["emission_span"]) != (cand["emission_row"], cand["emission_span"]) and not (
            base["emission_row"] is None and cand["emission_row"] is not None
            and cand["blocks"][cand["emission_row"]] == 1):
        found.append(f"{key}: emission differs (main {base['emission_row']}, candidate {cand['emission_row']})")
    return found


def _compare(base: dict, cand: dict) -> list[str]:
    assert set(base) == set(cand) and base, "the two sweeps ran different cases"
    found = []
    for key in sorted(base):
        b, c = base[key], cand[key]
        if "rows" in b:
            found += _fit_differences(key, b, c)
            if "second" in b or "second" in c:
                found += (_fit_differences(f"{key}/second", b["second"], c["second"])
                          if "second" in b and "second" in c else [f"{key}: one tree ran no second fit"])
            continue
        found += _fit_differences(f"{key}/fit", b["fit"], c["fit"])
        if "second" in b or "second" in c:
            found += (_fit_differences(f"{key}/second", b["second"], c["second"])
                      if "second" in b and "second" in c else [f"{key}: one tree ran no second fit"])
        for stage in b["stages"]:
            if b["stages"][stage] != c["stages"][stage]:
                found.append(f"{key}: the store after {stage} differs ({len(b['stages'][stage])} rows on main, "
                             f"{len(c['stages'][stage])} on the candidate)")
        for counts in ("counts_warm", "counts_cold"):
            if b[counts] != c[counts]:
                found.append(f"{key}: {counts} main {b[counts]}, candidate {c[counts]}")
    return found


@pytest.mark.parametrize("sweep", ["band", "d1"])
def test_c1_kept_rows_equal_main_on_every_cap(out_dir, sweep):
    found = _compare(_sweep(BASE_TREE, sweep, out_dir), _sweep(CAND_TREE, sweep, out_dir))
    assert not found, f"C1 {sweep}: {len(found)} difference(s): " + "; ".join(found[:20])


def test_c11_randomised_no_system_differential(out_dir):
    found = _compare(_sweep(BASE_TREE, "random", out_dir), _sweep(CAND_TREE, "random", out_dir))
    assert not found, f"C11: {len(found)} difference(s): " + "; ".join(found[:20])


def test_c1_c9_report_the_notice_gain(out_dir):
    """A report, not a bar (C9's rule): how many caps the candidate places the notice on."""
    for sweep in ("band", "d1"):
        path = out_dir / f"cand-{sweep}.json"
        if path.is_file():
            records = json.loads(path.read_text())["records"]
            fitted = [r for r in records if r["fitted"]]
            placed = sum(1 for r in fitted if sum(r["blocks"]))
            print(f"notice-gain {sweep}: fitted {len(fitted)}, block placed {placed}, not placed {len(fitted) - placed}")
