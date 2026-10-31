#!/usr/bin/env python3
"""G1 OFF replay and G2 paired think-time instrument. Exit 0 pass/smoke, 1 fail, 2 invalid, 3 projected >3h."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import SETTINGS, TURNS, VERSION, digest, dump, run_cell, workload
from scoring import at_least, score_cell

BASE = "e6d58e715b12a1b04869ecd8b15aee8881010dcd"
HERE = Path(__file__).resolve().parent


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def instrument_hash():
    return digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))})


def protocol(args):
    return dict(version=VERSION, instrument_sha256=instrument_hash(), turns=TURNS, window=60000,
                settings=SETTINGS, gap_seconds=args.gap_seconds, real_route=args.real_route,
                summary_model=args.summary_model, zero_gap_idle_sync=args.gap_seconds == 0, jobs=args.jobs,
                workload_hashes={str(s): digest(workload(s)) for s in args.seeds},
                host_sha=git(args.hermes_source, "rev-parse", "HEAD"),
                host_context_sha256=hashlib.sha256((args.hermes_source / "agent/context_engine.py").read_bytes()).hexdigest())


def load_reference(path):
    ref = json.loads(path.read_text())
    meta = json.loads((path.parent / "run.json").read_text())
    summary = json.loads((path.parent / "summary.json").read_text())
    expected = dict(pooled_on=summary["pooled_on"], protocol=meta["protocol"], seeds=meta["seeds"],
                    per_seed={str(s): summary["cells"][f"on-{s}"]["promotion"] for s in meta["seeds"]})
    if any(ref[k] != value for k, value in expected.items()):
        raise ValueError("reference/card-covered files mismatch")
    return ref


def summarize(root):
    meta = json.loads((root / "run.json").read_text())
    if (root / "projection.json").exists():
        return dict(verdict="NOT_RUN_OVER_3H", bars={}, pooled_on=dict(numerator=0, denominator=0, rate=None),
                    projection=json.loads((root / "projection.json").read_text()),
                    not_proven="reference rate not measured; no REFERENCE.json exists; G2 release gate remains incomplete")
    cells, raw_cells = {}, {}
    for seed in meta["seeds"]:
        for arm in ("off", "on"):
            key = f"{arm}-{seed}"
            raw = raw_cells[key] = json.loads((root / key / "raw.json").read_text())
            if raw["workload_sha"] != meta["protocol"]["workload_hashes"][str(seed)]:
                raise ValueError("workload hash mismatch")
            if len(raw["events"]) != TURNS or len(raw["probes"]) != 2 * TURNS + 1:
                raise ValueError("missing events/probes")
            if any(e["gap_s"] is None or e["gap_s"] < meta["protocol"]["gap_seconds"]
                   for e in raw["events"][1:]):
                raise ValueError("missing/short real gap")
            cells[key] = score_cell(raw)
    on = [cells[f"on-{s}"]["promotion"] for s in meta["seeds"]]
    numerator, denominator = sum(r["numerator"] for r in on), sum(r["denominator"] for r in on)
    pooled = dict(numerator=numerator, denominator=denominator, rate=numerator / denominator if denominator else None)
    bars = dict(integrity=all(c["integrity"] for c in cells.values()),
                off_quiet=all(cells[f"off-{s}"]["off_quiet"] for s in meta["seeds"]),
                off_foreground=all(cells[f"off-{s}"]["activity"]["foreground_summary_calls"] > 0 and
                                   cells[f"off-{s}"]["promotion"]["denominator"] > 0 for s in meta["seeds"]),
                on_observability=all(cells[f"on-{s}"]["positive_control"] for s in meta["seeds"]))
    if meta["runner"] == "background-prep-thinktime-on-off" and not meta["smoke"] and not meta["freeze_reference"]:
        ref = json.loads((root / "reference-input.json").read_text())
        if ref["product_sha"] != BASE or ref["seeds"] != [1, 2, 3] or ref["protocol"] != meta["protocol"]:
            raise ValueError("reference identity/protocol mismatch")
        bars["promotion_vs_reference"] = at_least(pooled, ref["pooled_on"])
    deltas = {}
    for s in meta["seeds"]:
        a, b = cells[f"off-{s}"]["wait"], cells[f"on-{s}"]["wait"]
        deltas[str(s)] = {k: b[k] - a[k] if a[k] is not None and b[k] is not None else None
                          for k in ("p90_s", "waited_turn_share")}
    return dict(verdict="FAIL" if not all(bars.values()) else "WIRING_ONLY" if meta["smoke"] else "PASS",
                bars=bars, pooled_on=pooled, cells=cells, paired_wait_delta_on_minus_off=deltas,
                wait_boundary="ingest + threshold compress, user arrival to model dispatch; excludes response/prep/gap",
                not_proven="live Hermes latency, semantic recall, rc/GA readiness; no Track S material used")


def card(args, root, summary):
    tool = [sys.executable, str(args.card_tool)]

    def call(*words):
        subprocess.run(tool + list(map(str, words)), check=True, stdout=subprocess.DEVNULL)

    path_flags = {"--out": "<out>", "--product-source": "<product>", "--hermes-source": "<hermes>",
                  "--reference": "<reference>", "--card-tool": "<card-tool>", "--rescore": "<out>"}
    recorded, pending = [], None
    for word in sys.argv[1:]:
        flag, equals, _ = word.partition("=")
        if pending:
            recorded.append(pending)
            pending = None
        elif flag in path_flags:
            recorded.append(flag + "=" + path_flags[flag] if equals else flag)
            pending = None if equals else path_flags[flag]
        else:
            recorded.append(word)
    command = shlex.join(["python3", "bench/instruments/compaction_probe/bgprep/run.py", *recorded])
    rescore = "python3 instrument/run.py --rescore ."
    call("init", root, "--project", "lcm-x", "--kind", "acceptance", "--owner-lane", "bgprep-instrument",
         "--code-dir", args.product_source, "--claim-class", "advisory", "--command", command)
    lines = ["# " + args.runner, "", "| bar | result | receipt |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v} | summary.json bars.{k} |" for k, v in summary["bars"].items()]
    lines += ["", f"Verdict: {summary['verdict']}", "", f"Pooled ON: {summary['pooled_on']}", "",
              "Per-seed rates and wait denominators: summary.json cells; raw: <arm>-<seed>/raw.json.",
              "", "Re-score from this run root: `" + rescore + "`", "",
              "Not proven: " + summary["not_proven"], "", "T1: unpublished (release-manager handoff).",
              "T2: incomplete, local retained evidence; no publication authorized."]
    (root / "RESULT.md").write_text("\n".join(lines) + "\n")
    paths = [p.name for p in root.iterdir() if p.name not in ("RESULT-MANIFEST.json", "RESULT.md")]
    call("add", root, *sorted(paths), "--role", "per-item")
    inputs = (["projection.json"] if summary["verdict"] == "NOT_RUN_OVER_3H" else
              [f"{a}-{s}/raw.json" for s in args.seeds for a in ("off", "on")])
    call("rescore", root, "--command", rescore, "--input", "run.json", "instrument/", *inputs)
    if summary["pooled_on"]["rate"] is not None:
        call("headline", root, "--benchmark", args.runner, "--metric", "pooled_on_promotion_rate",
             "--value", summary["pooled_on"]["rate"], "--n", summary["pooled_on"]["denominator"],
             "--source", "summary.json")
    call("status", root, "settled", "--owner-verdict", "Retain all results locally; publication is release-manager owned.")
    call("finalize", root)
    call("check", root, "--strict")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", nargs="?", choices=("background-prep-off-replay", "background-prep-thinktime-on-off"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--product-source", type=Path, default=HERE.parents[3])
    parser.add_argument("--product-sha")
    parser.add_argument("--hermes-source", type=Path)
    parser.add_argument("--gap-seconds", type=float, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--real-route", action="store_true", help="Use product provider via inherited auth; no live config copied")
    parser.add_argument("--summary-model", default="")
    parser.add_argument("--smoke", action="store_true", help="Wiring only; cannot freeze reference or satisfy G2 rate gate")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--freeze-reference", action="store_true")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Cells run concurrently (separate processes and stores); part of the protocol")
    parser.add_argument("--rescore", type=Path, help="Read immutable raw snapshots and print score, without product/host imports")
    parser.add_argument("--card-tool", type=Path, default=Path.home() / ".claude/scripts/bench-card.py")
    parser.add_argument("--cell", nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--host-sha", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python >=3.11 required (same minimum as product CI)")
    if args.rescore:
        score = summarize(args.rescore)
        print(json.dumps(score, indent=2))
        return 1 if score["verdict"] == "FAIL" else 0
    if not all((args.runner, args.out, args.hermes_source, args.product_sha)):
        parser.error("runner, --out, --hermes-source and --product-sha required")
    args.out, args.product_source, args.hermes_source = (p.resolve() for p in (args.out, args.product_source, args.hermes_source))
    if any(args.out.is_relative_to(source) for source in (args.product_source, args.hermes_source, HERE)):
        parser.error("output must be outside product, host and instrument source trees")
    if git(args.product_source, "rev-parse", "HEAD") != args.product_sha:
        raise ValueError("product HEAD does not match pinned SHA")
    if args.cell and git(args.hermes_source, "rev-parse", "HEAD") != args.host_sha:
        raise ValueError("host HEAD does not match pinned SHA")
    for name, source in (("product", args.product_source), ("host", args.hermes_source)):
        if git(source, "status", "--porcelain", "--untracked-files=normal"):
            parser.error(f"{name} checkout is dirty")
    if args.cell:
        run_cell(args, args.out, args.cell[0], int(args.cell[1]))
        return 0
    if not math.isfinite(args.gap_seconds) or args.gap_seconds < 0 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("finite nonnegative gap and distinct seeds required")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.smoke and args.freeze_reference:
        parser.error("a smoke cannot be a reference")
    if args.freeze_reference and args.runner != "background-prep-thinktime-on-off":
        parser.error("only G2 can freeze the reference")
    if args.runner.endswith("thinktime-on-off") and not args.smoke:
        if args.seeds != [1, 2, 3] or args.gap_seconds != 60:
            parser.error("gating/reference run requires seeds 1 2 3 and real 60s gaps")
        if not args.freeze_reference and not args.reference:
            parser.error("G2 needs --reference (or baseline --freeze-reference)")
    if args.freeze_reference and args.product_sha != BASE:
        raise ValueError("reference must run at the approved main e6d58e71")
    if args.real_route and not args.summary_model:
        parser.error("--real-route requires explicit --summary-model")
    current_protocol = protocol(args)
    if args.reference:
        ref = load_reference(args.reference)
        if ref["product_sha"] != BASE or ref["seeds"] != [1, 2, 3] or ref["protocol"] != current_protocol:
            raise ValueError("reference identity/protocol mismatch (before replay)")
    if args.out.exists():
        raise ValueError("output exists; choose a fresh run root, never overwrite evidence")
    args.out.mkdir(parents=True)
    shutil.copytree(HERE, args.out / "instrument", ignore=shutil.ignore_patterns("__pycache__"))
    meta = dict(runner=args.runner, seeds=args.seeds, smoke=args.smoke, freeze_reference=args.freeze_reference,
                product_sha=args.product_sha, protocol=current_protocol,
                host_sha=current_protocol["host_sha"])
    dump(args.out / "run.json", meta)
    if args.reference:
        shutil.copyfile(args.reference, args.out / "reference-input.json")
    # Gap time alone bounds wall time from below: each wave of `jobs` concurrent cells waits TURNS - 1 gaps.
    lower = (TURNS - 1) * args.gap_seconds * len(args.seeds)
    wall = (TURNS - 1) * args.gap_seconds * math.ceil(2 * len(args.seeds) / args.jobs)
    if args.freeze_reference and wall > 3 * 3600:
        dump(args.out / "projection.json", dict(status="NOT_RUN_OVER_3H", on_only_gap_lower_bound_s=lower,
            paired_gap_lower_bound_s=2 * lower, wall_gap_lower_bound_s=wall, jobs=args.jobs,
            turns=TURNS, seeds=args.seeds, reason="gaps alone exceed limit; no REFERENCE.json fabricated"))
        score = summarize(args.out)
        dump(args.out / "summary.json", score)
        card(args, args.out, score)
        print(f"NOT RUN: wall gaps >= {wall}s at --jobs {args.jobs}; serial paired gaps >= {2 * lower}s")
        return 3
    started = time.monotonic()
    cells = []
    for seed in args.seeds:
        for arm in ("off", "on"):
            root = args.out / f"{arm}-{seed}"
            root.mkdir()
            cmd = [sys.executable, "-B", str(args.out / "instrument" / "run.py"), args.runner, "--out", str(root),
                   "--product-source", str(args.product_source), "--product-sha", args.product_sha,
                   "--hermes-source", str(args.hermes_source), "--gap-seconds", str(args.gap_seconds),
                   "--host-sha", current_protocol["host_sha"],
                   "--cell", arm, str(seed), "--summary-model", args.summary_model]
            if args.real_route:
                cmd.append("--real-route")
            cells.append((root, cmd))

    def run_one(cell):
        root, cmd = cell
        with (root / "engine.log").open("w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=log,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for result in [pool.submit(run_one, cell) for cell in cells]:
            result.result()
    meta["wall_s"] = time.monotonic() - started
    dump(args.out / "run.json", meta)
    summary = summarize(args.out)
    dump(args.out / "summary.json", summary)
    card(args, args.out, summary)
    if args.freeze_reference and summary["verdict"] == "PASS":
        dump(args.out / "REFERENCE.json", dict(product_sha=BASE, seeds=args.seeds, protocol=meta["protocol"],
             pooled_on=summary["pooled_on"], wall_s=meta["wall_s"],
             per_seed={str(s): summary["cells"][f"on-{s}"]["promotion"] for s in args.seeds}))
        (args.out / "REFERENCE.json").chmod(0o444)
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))
    return 1 if summary["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
