#!/usr/bin/env python3
"""1096-H T/P comparison; D7 bars and retention are reported, never gated.

Exit 0 PASS, 1 FAIL, 2 INVALID. Inputs are complete pmw_read JSON receipts.
"""
import argparse
import json
import math
from pathlib import Path


def arm(path, variant, sha):
    cell = json.loads(path.with_name("cell.json").read_text())
    if cell["product_sha"] != sha or cell["fault"] != variant:
        raise ValueError(f"{path}: cell product_sha or fault differs from declaration")
    identity = {key: cell[key] for key in ("seeds", "kit_sha", "arm")}
    data = json.loads(path.read_text())
    verdict = data["verdict"]
    if verdict["result"] not in ("PASS", "FAIL") or verdict["complete"] is not True:
        raise ValueError(f"{path.name}: INVALID or incomplete reader")
    bars = verdict["bars"]
    if len(bars) != 6 or any(type(v) is not bool for v in bars.values()):
        raise ValueError("missing D7 bars")
    if len(data["pooled"]) != 1:
        raise ValueError("expected one pooled arm per input")
    name, metrics = next(iter(data["pooled"].items()))
    if name != cell["arm"]:
        raise ValueError("pooled arm differs from cell arm")
    # pmw_read.read() uses one repeat per declared seed: arm|d<seed>-r1.
    if set(data["per_run"]) != {f"{name}|d{seed}-r1" for seed in cell["seeds"]}:
        raise ValueError("per_run ids differ from cell seeds and arm")
    material = {}
    for seed in cell["seeds"]:
        summary = path.parent / "lcmx-runs" / name / f"seed-{seed}" / f"d{seed}-r1" / "summary.json"
        digest = json.loads(summary.read_text())["material_sha256"]
        if not isinstance(digest, str) or not digest.strip():
            raise ValueError(f"{summary}: missing or empty material_sha256")
        material[str(seed)] = digest
    identity["material_sha256"] = material
    for key in ("runs", "faulted_calls", "long_waits", "max_long_waits_in_one_turn", "dropped_rows", "survival_fits"):
        value = metrics[key]
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid {key}")
    for key in ("c_per_100", "waited_turn_share", "mean_turn_wait_s", "max_turn_wait_s", "facts_304_mean"):
        value = metrics[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"missing or invalid {key}")
    runs = metrics["runs"]
    if not runs or len(data["per_run"]) != runs or metrics["fault_mode"] != variant:
        raise ValueError("incomplete runs or wrong pooled fault_mode")
    counts = []
    for run in data["per_run"].values():
        if run["status"] != "DONE" or run["fault_mode"] != variant:
            raise ValueError("incomplete run or wrong fault_mode")
        count = run["faulted_calls"]
        if type(count) is not int or count < 1 or (variant == "T" and count != 1):
            raise ValueError("fault did not fire as declared per run")
        counts.append(count)
        for key in ("long_waits", "max_long_waits_in_one_turn"):
            if type(run[key]) is not int or run[key] < 0:
                raise ValueError(f"invalid per-run {key}")
    if sum(counts) != metrics["faulted_calls"]:
        raise ValueError("pooled faulted_calls disagrees with per-run counts")
    return {"arm": name, "d7": verdict, **metrics}, data["per_run"], identity


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--candidate-sha", required=True)
    ap.add_argument("--variant", choices=("T", "P"), required=True)
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {"variant": args.variant}
    try:
        base, base_runs, base_identity = arm(args.base, args.variant, args.base_sha)
        candidate, candidate_runs, candidate_identity = arm(args.candidate, args.variant, args.candidate_sha)
        if base_identity != candidate_identity:
            raise ValueError("cell seeds, kit_sha, arm or material_sha256 differ")
        if base_runs.keys() != candidate_runs.keys():
            raise ValueError("per_run key sets differ")
        out.update(base=base, candidate=candidate, material_sha256=base_identity["material_sha256"])
        out["per_run"] = {run_id: {side: {key: run[key] for key in ("long_waits", "max_long_waits_in_one_turn")}
                                   for side, run in (("base", base_runs[run_id]), ("candidate", candidate_runs[run_id]))}
                          for run_id in base_runs}
        passed = (candidate["dropped_rows"] == 0 and candidate["long_waits"] <= base["long_waits"]
                  if args.variant == "T" else
                  all(row["candidate"]["long_waits"] <= row["base"]["long_waits"] + 1
                      and row["candidate"]["max_long_waits_in_one_turn"] <= 1 for row in out["per_run"].values()))
        out["result"] = "PASS" if passed else "FAIL"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        out.update(result="INVALID", error=str(exc))
    args.json.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[out["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
