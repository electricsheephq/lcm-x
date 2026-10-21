#!/usr/bin/env python3
"""Run the six C1 engine-direct cells; reuse only verified completed cells.

All paths are explicit CLI inputs. Source and provider accounts are never copied.
Summary/reader calls share c1_budget's lock and persistent counters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PRODUCT_SHA = "77b2b4f795fb6a4f84c55abf6d0441da4469971e"
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--product", type=Path, required=True)
    ap.add_argument("--material", type=Path, required=True)
    ap.add_argument("--hermes-src", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["LCMX-fleet-rc3", "H"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    head = subprocess.check_output(["git", "-C", str(args.product), "rev-parse", "HEAD"], text=True).strip()
    if head != PRODUCT_SHA:
        raise SystemExit("C1 STOP: product SHA mismatch")
    host_sha = json.loads((args.hermes_src.parent / "HOST.json").read_text())["sha"]
    logs = args.evidence / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tmp = args.scratch / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TRACK_S_OUT=str(args.scratch), S2_RUN_ROOT=str(args.scratch / "runs"),
               TRACK_S_MATERIAL=str(args.material), S2_PRODUCT_WORKTREE=str(args.product), S2_PRODUCT_SHA=head,
               HERMES_SRC=str(args.hermes_src), TMPDIR=str(tmp), PYTHONDONTWRITEBYTECODE="1",
               C1_BUDGET_ROOT=str(args.evidence / "throttle"),
               CODEX_HOME=os.environ.get("CODEX_HOME", str(Path.home() / ".codex")),
               S2_ARM_PINS=json.dumps({"H": dict(worktree=str(args.hermes_src), sha=host_sha)}))
    for arm in args.arms:
        for seed in args.seeds:
            if shutil.disk_usage(args.scratch).free < 60 * 1024**3:
                raise SystemExit("C1 STOP: free disk below 60 GB")
            sdir = args.material / f"seed-{seed}"
            material_sha = digest(sdir / "material.manifest.json")
            man = json.loads((sdir / "material.manifest.json").read_text())
            cps = list(json.loads((sdir / "deep_metadata.json").read_text())["checkpoint_answers"])
            cps = sorted(cps, key=int)
            name = f"s2-{arm}-d{seed}-r1"
            run = args.scratch / "runs" / arm / f"seed-{seed}" / f"d{seed}-r1"
            wall = logs / f"{name}.log.wall"
            if (run / "summary.json").exists():
                summary = json.loads((run / "summary.json").read_text())
                if summary.get("status") == "DONE" and summary.get("material_sha256") == material_sha and (
                        summary.get("worktree_head") == head if arm.startswith("LCMX") else
                        summary.get("host_sha") == host_sha and summary.get("worktree_head") == host_sha) and all(
                        (run / f"cp-{cp}" / "summary.json").exists() for cp in cps):
                    print(f"SKIP completed {arm} seed-{seed}", flush=True)
                    continue
                raise SystemExit(f"C1 STOP: incomplete/mismatched existing run {arm} seed-{seed}; preserve for diagnosis")
            if run.exists() and any(run.iterdir()):
                raise SystemExit(f"C1 STOP: partial existing run {arm} seed-{seed}; preserve for diagnosis")
            tail = ["--seed", str(seed), "--run", f"d{seed}-r1", "--lane", "codex-sol",
                    "--reader", "sol-low", "--checkpoints", ",".join(cps)]
            cmd = (["bash", str(HERE / "launch_h.sh")] if arm == "H" else
                   [sys.executable, "-B", str(HERE / "run_s_lcmx.py"), "--arm", arm, "--turn-end-notify"]) + tail
            wall.write_text(f"start {time.time()}\n")
            print(f"RUN {arm} seed-{seed}; material={material_sha}; free={shutil.disk_usage(args.scratch).free // 1024**3}GB", flush=True)
            with (logs / f"{name}.log").open("w") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            with wall.open("a") as receipt:
                receipt.write(f"exit {rc} end {time.time()}\n")
            if rc:
                raise SystemExit(f"C1 run failed: {arm} seed-{seed}; exit {rc}; inspect log")
            summary = json.loads((run / "summary.json").read_text())
            if arm.startswith("LCMX"):
                depth = max((node["depth"] for event in summary["events"] for node in event.get("nodes", [])), default=-1)
                if depth < 2:
                    stop = dict(status="STOP", reason="DAG below three generations", arm=arm, seed=seed,
                                max_depth=depth, generations=depth + 1, material_sha256=material_sha,
                                scale=man.get("deep_scale"), product_sha=head)
                    (args.evidence / "depth-stop.json").write_text(json.dumps(stop, indent=2) + "\n")
                    raise SystemExit("C1 STOP: DAG below three generations; report, do not extend beyond 4x")
    print("C1 six-cell execution complete", flush=True)


if __name__ == "__main__":
    main()
