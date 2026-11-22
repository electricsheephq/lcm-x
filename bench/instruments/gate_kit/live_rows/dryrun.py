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
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from contextlib import closing
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


FROZEN = ("target_rows_examined", "planner_windows", "planner_statements")
MINIMUMS = ("rows", "conversations", "tool_pairs", "externalized_payloads")


def stored_counts(build: Path, manifest: dict[str, Any]) -> dict[str, int]:
    """The built store's counts the D9 row fixes against the reference copy (read-only)."""
    with closing(sqlite3.connect(rt.ro_uri(build / "store" / "lcm.db"), uri=True)) as conn:
        rows, conversations, tools = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT conversation_id), SUM(role = 'tool') FROM messages").fetchone()
    return {"rows": rows, "conversations": conversations, "tool_pairs": int(tools or 0),
            "externalized_payloads": int(manifest.get("externalized_files") or 0)}


def qualify(kit: dict[str, Any], manifest: dict[str, Any], counts: dict[str, int], baseline: dict[str, Any],
            d9: dict[str, Any] | None, planner: dict[str, Any] | None) -> dict[str, Any]:
    """D9's dry-run rule and the planner rule, plus the kit's own preconditions: frozen counts, the pinned
    baseline tree, a qualifying receipt, a clean lock watch, and counts no smaller than the reference copy's."""
    frozen, minimums, reasons = kit.get("frozen") or {}, kit.get("minimums") or {}, []
    missing = [name for name in FROZEN if not frozen.get(name)]
    if missing:
        reasons.append(f"kit counts not frozen: {missing}")
    short = {name: (counts.get(name), minimums.get(name)) for name in MINIMUMS
             if minimums.get(name) is None or counts.get(name, 0) < minimums[name]}
    if short:
        reasons.append(f"store counts below the reference minimums (or no minimum pinned): {short}")
    pinned = kit.get("baseline") or {}
    if not pinned or {k: baseline.get(k) for k in pinned} != pinned:
        reasons.append("the baseline tree is not the pinned v0.30.0 tree")
    if manifest.get("scale") != 1.0:
        reasons.append("the store is scaled")
    hold = d9.get("longest_hold_s") if d9 else None
    if not d9 or not d9.get("qualifies") or d9.get("lock_unexpected") or d9.get("cell_error"):
        reasons.append("the v0.30.0 D9 receipt is missing, non-qualifying, or its lock watch saw unclassified answers")
    elif hold is None or hold <= kit["bars"]["d9_hold_s"]:
        reasons.append(f"v0.30.0 at defaults held {hold} s, not longer than {kit['bars']['d9_hold_s']} s")
    if not planner or not planner.get("window_budget_exceeded"):
        reasons.append("the v0.30.0 planner did not exceed the frozen window count")
    return {"passed": not reasons, "reasons": reasons, "d9_v0300_hold_s": hold, "store_counts": counts,
            "minimums": minimums, "planner_v0300_windows": planner and planner.get("windows"),
            "planner_bound": frozen.get("planner_windows")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="the v0.30.0 tree (builds and qualifies)")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load", type=int, default=2, help="requirement 2's load processes (recorded arms)")
    parser.add_argument("--skip-candidate", action="store_true", help="qualification only")
    parser.add_argument("--kit", type=Path, default=rt.KIT, help="the kit the store is built from")
    args = parser.parse_args(argv)
    # Absolute before any child runs: the children run in the repository, the caller may not.
    for name in ("baseline", "candidate", "scratch", "out", "kit"):
        setattr(args, name, getattr(args, name).absolute())
    args.scratch.mkdir(parents=True, exist_ok=False)
    args.out.mkdir(parents=True, exist_ok=False)
    log = args.out / "dryrun.log"
    build, work = args.scratch / "build", args.scratch / "work"
    base, cand = str(args.baseline.resolve()), str(args.candidate.resolve())
    started = time.time()
    steps: dict[str, int] = {}
    steps["build"] = module("build", "--tree", base, "--kit", str(args.kit.resolve()), "--out", str(build),
                            "--work", str(work), log=log)
    (args.out / "build-manifest.json").write_text((build / "build-manifest.json").read_text())
    kit = json.loads((build / "build-manifest.json").read_text())["kit_body"]  # bars and counts the store was built with

    def cell(name: str, tree: str, cell_name: str, *extra: str) -> dict[str, Any] | None:
        out = args.out / f"{name}.json"
        steps[name] = module("run", "--tree", tree, "--build", str(build), "--cell", cell_name, "--work", str(work),
                             "--out", str(out), *extra, log=log)
        return load(out)

    manifest = json.loads((build / "build-manifest.json").read_text())
    frozen = kit.get("frozen") or {}
    qualify_d9 = cell("qualify-d9-v0300-default", base, "d9", "--arm", "default")
    planner_args = ["--max-windows", str(frozen["planner_windows"])] if frozen.get("planner_windows") else []
    planner_args += ["--max-statements", str(frozen["planner_statements"])] if frozen.get("planner_statements") else []
    if planner_args:
        steps["qualify-planner"] = module("l1_planner", "--tree", base, "--work", str(args.scratch / "planner-v0300"),
                                          *planner_args, "--out", str(args.out / "qualify-planner-v0300.json"),
                                          log=log)
    qualification = qualify(kit, manifest, stored_counts(build, manifest),
                            rt.driver.tree_identity(args.baseline), qualify_d9,
                            load(args.out / "qualify-planner-v0300.json"))
    result: dict[str, Any] = {"qualification": qualification, "kit_frozen": frozen,
                              "census_disclosure": kit.get("census_disclosure")}
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
        steps["l1-planner"] = module("l1_planner", "--tree", cand, "--work", str(args.scratch / "planner"),
                                     *planner_args, "--out", str(args.out / "l1-planner.json"), log=log)
        rows_pinned = frozen.get("target_rows_examined") or pinned
        raw = {
            "D9": verdicts.d9(d9, kit["bars"], rows_pinned), "D10": verdicts.d10(d10, kit["bars"], rows_pinned),
            "D11": verdicts.d11(d11), "D12": verdicts.d12(d11),
            "L1-switch": (load(args.out / "l1-switch.json") or {}).get("verdict", "UNRUN"),
            "L1-planner": (load(args.out / "l1-planner.json") or {}).get("verdict", "UNRUN"),
        }
        if qualification["passed"]:
            result["verdicts"] = raw
        else:  # the rows stay UNRUN until the dry run passes; the raw outcomes are kept for diagnosis only
            result["verdicts"] = {name: "UNRUN" for name in raw}
            result["raw_unqualified"] = raw
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
