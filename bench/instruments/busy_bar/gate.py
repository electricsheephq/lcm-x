"""Compare busy operations; mismatched exposure cannot pass this bar."""

from __future__ import annotations

import argparse
import json
from collections import Counter
import math
from pathlib import Path
from typing import Any

if __package__:
    from .cells import CELLS
else:
    from cells import CELLS


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def comparison(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: list[str] = []
    tables: dict[str, Any] = {}
    exposure: list[Counter] = []
    for name, runs in (("baseline", baseline), ("candidate", candidate)):
        counts: Counter = Counter()
        run_ids: set = set()
        table = {cell: {owner: {"operations": 0, "busy": 0} for owner in ("A", "B", "total")}
                 for cell in CELLS}
        for run in runs:
            cell, number = run["cell"], run["run"]
            if cell not in CELLS:
                reasons.append(f"{name}: unknown cell {cell}")
                continue
            identity = (cell, number)
            if identity in run_ids:
                reasons.append(f"{name}: duplicate run {identity}")
            run_ids.add(identity)
            for setting in ("k", "b_view_rows", "busy_timeout_ms", "seed", "window_ms"):
                counts[(cell, number, setting, run[setting])] += 1
            counts[(cell, number, "offsets_ms", tuple(run["offsets_ms"]))] += 1
            if run.get("errors"):
                reasons.append(f"{name}: {cell} run {number} has driver errors")
            if run.get("inconclusive") or any(record.get("inconclusive") for record in run["operations"]):
                reasons.append(f"{name}: {cell} run {number} is inconclusive (non-busy raise)")
            for record in run["operations"]:
                owner, op, busy = record["engine"], record["op"], record["busy"]
                if owner not in ("A", "B") or op not in ("ingest", "compress", "session-end") or type(busy) is not bool:
                    reasons.append(f"{name}: invalid operation in {cell} run {number}")
                    continue
                counts[(cell, number, owner, op)] += 1
                for bucket in (owner, "total"):
                    table[cell][bucket]["operations"] += 1
                    table[cell][bucket]["busy"] += int(busy)
        missing = set(CELLS) - {cell for cell, _ in run_ids}
        if missing:
            reasons.append(f"{name}: missing cells {', '.join(sorted(missing))}")
        tables[name] = table
        exposure.append(counts)
    if exposure[0] != exposure[1]:
        differing = sorted({key[0] for key in exposure[0].keys() | exposure[1].keys()
                            if exposure[0][key] != exposure[1][key]})
        reasons.append("operation counts, run coverage or schedules differ: " + ", ".join(differing))
    failed = [cell for cell in CELLS
              if tables["candidate"][cell]["total"]["busy"] >
              math.ceil(1.25 * tables["baseline"][cell]["total"]["busy"]) + 1]
    return {"verdict": "INCONCLUSIVE" if reasons else "GATE FAIL" if failed else "GATE PASS",
            "reasons": reasons, "failed_cells": failed, "tables": tables}


def extension_cells(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[str]:
    """Only the prescribed first twenty production runs decide whether to extend."""
    selected = []
    for cell in CELLS:
        pair = [[run for run in rows if run["cell"] == cell and run["run"] < 20]
                for rows in (baseline, candidate)]
        if not all({run["run"] for run in runs} == set(range(20)) for runs in pair):
            continue  # R<20 is a local smoke, not the gate session.
        busy = [sum(record["busy"] for run in runs for record in run["operations"]) for runs in pair]
        if busy[0] < 8 and busy[1] > busy[0]:
            selected.append(cell)
    return selected


def evaluate(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]],
             control_baseline: list[dict[str, Any]] | None = None,
             control_candidate: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result = comparison(baseline, candidate)
    reasons = result["reasons"]
    extensions = extension_cells(baseline, candidate)
    result["extension_cells"] = extensions
    result["initial_tables"] = comparison([run for run in baseline if run["run"] < 20],
                                           [run for run in candidate if run["run"] < 20])["tables"]
    for cell in extensions:
        for name, runs in (("baseline", baseline), ("candidate", candidate)):
            if {run["run"] for run in runs if run["cell"] == cell} != set(range(80)):
                reasons.append(f"{name}: {cell} requires 60 additional runs (80 total)")
    if any(run["busy_timeout_ms"] is not None for run in baseline + candidate):
        reasons.append("gating ledgers must use production budgets (no override)")
    if control_baseline is None or control_candidate is None:
        reasons.append("positive control ledgers required for both trees")
    else:
        control = comparison(control_baseline, control_candidate)
        control["verdict"] = "CONTROL ONLY"
        result["control"] = control
        reasons.extend("control: " + reason for reason in control["reasons"])
        for name, prod, scaled in (("baseline", baseline, control_baseline),
                                    ("candidate", candidate, control_candidate)):
            if any(run["busy_timeout_ms"] != 100 for run in scaled):
                reasons.append(f"{name}: positive control must use 100 ms")
            # Control has the same attempted operations, seeds and arrivals; only its budget differs.
            normalized = [{**run, "busy_timeout_ms": None} for run in scaled]
            # The control covers the first 20 only; production extensions never extend it.
            initial = [run for run in prod if run["run"] < 20]
            reasons.extend(f"{name} production/control: " + reason
                           for reason in comparison(initial, normalized)["reasons"])
            for op in ("ingest", "compress", "session-end"):
                busy = sum(record["busy"] for run in scaled for record in run["operations"] if record["op"] == op)
                if busy == 0:
                    reasons.append(f"control: {name} has 0 busy {op} operations; schedule did not exercise the class")
    result["verdict"] = ("INCONCLUSIVE" if reasons else
                         "GATE FAIL" if result["failed_cells"] else "GATE PASS")
    return result


def report(result: dict[str, Any]) -> str:
    lines = ["cell engine baseline_operations baseline_busy candidate_operations candidate_busy"]
    totals = {name: {owner: Counter() for owner in ("A", "B", "total")}
              for name in ("baseline", "candidate")}
    for cell in CELLS:
        for owner in ("A", "B", "total"):
            base = result["tables"]["baseline"][cell][owner]
            cand = result["tables"]["candidate"][cell][owner]
            lines.append(f"{cell} {owner} {base['operations']} {base['busy']} {cand['operations']} {cand['busy']}")
            for name in totals:
                totals[name][owner].update(result["tables"][name][cell][owner])
    for owner in ("A", "B", "total"):
        base, cand = totals["baseline"][owner], totals["candidate"][owner]
        lines.append(f"TOTAL {owner} {base['operations']} {base['busy']} {cand['operations']} {cand['busy']}")
    if "control" in result:
        for cell in result["extension_cells"]:
            base = result["initial_tables"]["baseline"][cell]["total"]["busy"]
            cand = result["initial_tables"]["candidate"][cell]["total"]["busy"]
            final_base = result["tables"]["baseline"][cell]["total"]["busy"]
            final_cand = result["tables"]["candidate"][cell]["total"]["busy"]
            lines.append(f"EXTENSION {cell}: 20-run busy {base}/{cand}; final busy {final_base}/{final_cand}")
        lines.append("POSITIVE CONTROL (100 ms; never gating)")
        lines.extend(report(result["control"]).splitlines()[:-1])
    lines.append(result["verdict"])
    lines.extend(result["reasons"])
    if result["verdict"] == "GATE FAIL":
        lines.append("candidate exceeds ceil(1.25 * baseline) + 1: " + ", ".join(result["failed_cells"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--control-baseline", type=Path)
    parser.add_argument("--control-candidate", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(read(args.baseline), read(args.candidate),
                          read(args.control_baseline) if args.control_baseline else None,
                          read(args.control_candidate) if args.control_candidate else None)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"INCONCLUSIVE: invalid ledger ({type(exc).__name__})")
        raise SystemExit(2) from exc
    print(report(result), end="")
    raise SystemExit({"GATE PASS": 0, "GATE FAIL": 1, "INCONCLUSIVE": 2}[result["verdict"]])


if __name__ == "__main__":
    main()
