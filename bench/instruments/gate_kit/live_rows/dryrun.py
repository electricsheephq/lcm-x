"""The producer's dry run: build the kit store, qualify it, run every cell on a candidate tree, and write verdicts.

    python -B -m bench.instruments.gate_kit.live_rows.dryrun --baseline V0300_TREE --candidate TREE \
        --scratch NEW_SCRATCH_DIR --out NEW_RECEIPT_DIR [--load 2] [--skip-candidate]

Qualification (#2868, D9's dry-run rule, the L1 planner rule):
- the v0.30.0 tree at its defaults holds the kit's store longer than D9's bar on this host;
- the v0.30.0 planner exceeds the frozen window count on the planner fixture (stopped at the count plus one).
The candidate's cells then run one process each; `verdicts.json` holds D9-D12's raw verdicts and the L1 checks'.
Every receipt lands in --out; bulk (the store, one working copy) stays in --scratch. Nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from bench.instruments.gate_kit.live_rows import rt, verdicts  # noqa: E402

PACKAGE = "bench.instruments.gate_kit.live_rows"


def module(name: str, *args: str, log: Path) -> int:
    with log.open("a") as handle:
        handle.write(f"\n## {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {name} {' '.join(args)}\n")
        handle.flush()
        return subprocess.call([sys.executable, "-B", "-m", f"{PACKAGE}.{name}", *args], stdout=handle,
                               stderr=subprocess.STDOUT, cwd=str(rt.REPO))


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="the v0.30.0 tree (builds and qualifies)")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load", type=int, default=2, help="requirement 2's load processes (recorded arms)")
    parser.add_argument("--skip-candidate", action="store_true", help="qualification only")
    args = parser.parse_args(argv)
    args.scratch.mkdir(parents=True, exist_ok=False)
    args.out.mkdir(parents=True, exist_ok=False)
    kit = rt.load_kit()
    log = args.out / "dryrun.log"
    build, work = args.scratch / "build", args.scratch / "work"
    base, cand = str(args.baseline.resolve()), str(args.candidate.resolve())
    started = time.time()
    steps: dict[str, int] = {}
    steps["build"] = module("build", "--tree", base, "--out", str(build), log=log)
    (args.out / "build-manifest.json").write_text((build / "build-manifest.json").read_text())

    def cell(name: str, tree: str, cell_name: str, *extra: str) -> dict[str, Any] | None:
        out = args.out / f"{name}.json"
        steps[name] = module("run", "--tree", tree, "--build", str(build), "--cell", cell_name, "--work", str(work),
                             "--out", str(out), *extra, log=log)
        return load(out)

    frozen = kit["frozen"]
    qualify_d9 = cell("qualify-d9-v0300-default", base, "d9", "--arm", "default")
    planner_bound = str(frozen["planner_windows"] or 66048)
    steps["qualify-planner"] = module("l1_planner", "--tree", base, "--work", str(args.scratch / "planner-v0300"),
                                      "--max-windows", planner_bound, "--max-statements",
                                      str(frozen["planner_statements"] or 1000),
                                      "--out", str(args.out / "qualify-planner-v0300.json"), log=log)
    qualify_planner = load(args.out / "qualify-planner-v0300.json")
    qualification = {
        "d9_store": bool(qualify_d9 and qualify_d9.get("qualifies") and
                         qualify_d9["longest_hold_s"] > kit["bars"]["d9_hold_s"]),
        "d9_v0300_hold_s": qualify_d9 and qualify_d9.get("longest_hold_s"),
        "planner_fixture": bool(qualify_planner and qualify_planner["window_budget_exceeded"]),
        "planner_v0300_windows": qualify_planner and qualify_planner.get("windows"),
        "planner_bound": int(planner_bound),
    }
    qualification["passed"] = qualification["d9_store"] and qualification["planner_fixture"]
    result: dict[str, Any] = {"qualification": qualification, "kit_frozen": frozen}
    if not args.skip_candidate:
        pinned = load(args.out / "build-manifest.json")["generated"]["target_list_rows"]
        d9 = {arm: cell(f"d9-{arm}", cand, "d9", "--arm", arm) for arm in ("uid_on", "default")}
        d10 = {arm: cell(f"d10-{arm}", cand, "d10", "--arm", arm) for arm in ("uid_on", "default")}
        cell("d10-uid_on-worst-case", cand, "d10", "--arm", "uid_on", "--worst-case")
        d11 = {arm: {budget: cell(f"d11-{arm}-{budget}", cand, "d11", "--arm", arm, "--budget", budget)
                     for budget in kit["budgets"]} for arm in ("uid_on", "default")}
        # Recorded variants (requirements 2-5); none decides a row.
        cell("req2-d9-uid_on-load", cand, "d9", "--arm", "uid_on", "--load", str(args.load))
        cell("req2-d10-uid_on-load", cand, "d10", "--arm", "uid_on", "--load", str(args.load))
        cell("req3-kill-uid_on", cand, "kill-at-first-call", "--arm", "uid_on")
        cell("req3-after-kill-d11-uid_on", cand, "d11", "--arm", "uid_on", "--keep-copy")
        cell("req4-turns-uid_on", cand, "turns", "--arm", "uid_on")
        cell("req5-timeout-hold-uid_on", cand, "timeout-hold", "--arm", "uid_on")
        steps["l1-switch"] = module("l1_switch", "--tree", cand, "--work", str(args.scratch / "switch"),
                                    "--out", str(args.out / "l1-switch.json"), log=log)
        planner_args = ["--max-windows", str(frozen["planner_windows"])] if frozen["planner_windows"] else []
        planner_args += ["--max-statements", str(frozen["planner_statements"])] if frozen["planner_statements"] else []
        steps["l1-planner"] = module("l1_planner", "--tree", cand, "--work", str(args.scratch / "planner"),
                                     *planner_args, "--out", str(args.out / "l1-planner.json"), log=log)
        rows_pinned = frozen["target_rows_examined"] or pinned
        result["verdicts"] = {
            "D9": verdicts.d9(d9, kit["bars"], rows_pinned), "D10": verdicts.d10(d10, kit["bars"], rows_pinned),
            "D11": verdicts.d11(d11), "D12": verdicts.d12(d11),
            "L1-switch": (load(args.out / "l1-switch.json") or {}).get("verdict", "UNRUN"),
            "L1-planner": (load(args.out / "l1-planner.json") or {}).get("verdict", "UNRUN"),
        }
        if not qualification["passed"]:
            result["note"] = "the producer has not passed its dry run: D9-D12 and both L1 checks stay UNRUN"
    result.update({"steps": steps, "wall_s": round(time.time() - started, 1),
                   "candidate_identity": rt.driver.tree_identity(args.candidate.resolve()),
                   "baseline_identity": rt.driver.tree_identity(args.baseline.resolve())})
    (args.out / "verdicts.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({"qualification": qualification,
                      "verdicts": {k: (v["outcome"] if isinstance(v, dict) else v)
                                   for k, v in result.get("verdicts", {}).items()}}))
    return 0 if qualification["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
