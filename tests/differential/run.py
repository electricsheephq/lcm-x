"""Opt-in CLI; archive identities, replay inputs and every red are retained."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .ACCEPTED_DELTAS import ACCEPTED_DELTAS
from .cases import exact_cases, generated_cases, projection_cases
from .oracles import compare
from .trees import BASELINE_CONFIG, BASELINES, DEFAULT_BASELINE, ROOT, PrerequisiteUnavailable, check_host, materialize, resolve


def candidate_identity(root):
    """Name the executed checkout, including tracked and untracked input bytes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout

    sha = git("rev-parse", "HEAD").decode().strip()
    tracked = git("diff", "HEAD", "--name-only", "-z").split(b"\0")
    untracked = sorted(path for path in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if path)
    files = sorted({os.fsdecode(path) for path in tracked + untracked if path})
    if not files:
        return dict(candidate=sha)
    digest = hashlib.sha256(git("diff", "HEAD"))
    for name in untracked:
        path = Path(root) / os.fsdecode(name)
        digest.update(os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes())
    return dict(candidate=f"{sha}+dirty", candidate_dirty=dict(files=files, diff_sha256=digest.hexdigest()))


def run(cases, out, baseline=DEFAULT_BASELINE, candidate="worktree", seed=1227, baseline_config=None):
    if os.environ.get("LCM_DIFFERENTIAL") != "1":
        raise RuntimeError("set LCM_DIFFERENTIAL=1 to opt in")
    baseline_ref, baseline_sha = baseline, resolve(baseline)
    baseline = next((tag for tag, sha in BASELINES.items() if sha == baseline_sha), baseline)
    baseline_config = baseline_config or os.environ.get("LCM_DIFFERENTIAL_BASELINE_CONFIG") or BASELINE_CONFIG.get(baseline, "bare")
    if baseline_config not in {"bare", "managed"}:
        raise ValueError("baseline_config must be bare or managed")
    check_host()
    identity = candidate_identity(ROOT) if candidate == "worktree" else dict(candidate=resolve(candidate))
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "trees"
    trees = {"ga": materialize(baseline, cache),
             "candidate": ROOT if candidate == "worktree" else materialize(candidate, cache)}
    summary = dict(baseline=baseline_sha, baseline_ref=baseline_ref,
                   baseline_tag=baseline if baseline in BASELINES else None, **identity,
                   baseline_config=baseline_config, seed=seed, cases=len(cases), counts={}, delta_absorption={}, stale_deltas=[], unexplained=[],
                   accepted=[], coverage={}, timings={}, missing_seams={}, absolute_over_window=[],
                   inherited_user_marks=dict(cases=0, marks=0))
    counts, absorbed = Counter(), {delta.id: set() for delta in ACCEPTED_DELTAS}
    for layer in sorted({case["layer"] for case in cases}):
        sample = [case for case in cases if case["layer"] == layer]
        case_path = out / f"{layer}-cases.jsonl"
        case_path.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in sample))
        results = {}
        for label, tree in trees.items():
            result_path = out / f"{layer}-{label}.jsonl"
            started = time.perf_counter()
            with (out / f"{layer}-{label}.log").open("w") as log:
                subprocess.run(["nice", "-n", "19", sys.executable, str(Path(__file__).with_name("driver.py")),
                                "--tree", str(tree), "--side", label,
                                "--baseline-config", baseline_config if label == "ga" else "bare", "--layer", layer, "--cases", str(case_path), "--out", str(result_path)],
                               check=True, stdout=log, stderr=log, cwd=ROOT)
            records = [json.loads(line) for line in result_path.read_text().splitlines()]
            assert [record["id"] for record in records] == [case["id"] for case in sample], "incomplete driver output"
            results[label] = records
            summary["timings"][f"{layer}/{label}"] = dict(wall_seconds=time.perf_counter() - started,
                                                          mean_case_seconds=sum(r["seconds"] for r in records) / len(records))
            summary["missing_seams"][f"{layer}/{label}"] = sorted({seam for record in records for seam in record["missing_seams"]})
        for case, ga, cand in zip(sample, results["ga"], results["candidate"]):
            if layer == "projection" and baseline != "v0.26.2" and "error" not in ga and "error" not in cand:
                inherited = set(cand["user_marks"]) & set(ga["user_marks"])
                summary["inherited_user_marks"]["cases"] += bool(inherited)
                summary["inherited_user_marks"]["marks"] += len(inherited)
            if layer == "public" and "error" not in ga and "error" not in cand:
                summary["coverage"][case["id"]] = dict(ga=ga["uncovered_rows"], candidate=cand["uncovered_rows"])
            for failure in compare(case, ga, cand, baseline=baseline):
                key = f"{layer}/{case['family']}/{failure['oracle']}"
                counts[key] += 1
                record = dict(case_id=case["id"], layer=layer, **failure)
                if failure["delta"]:
                    absorbed[failure["delta"]].add(case["id"])
                    summary["accepted"].append(record)
                else:
                    summary["unexplained"].append({**record, "case": case})
            if layer == "public" and "error" not in cand:
                for i, call in enumerate(cand["calls"]):
                    if call["tokens"] > case["window"]:
                        summary["absolute_over_window"].append(dict(case_id=case["id"], call=i, tokens=call["tokens"], window=case["window"]))
    summary["counts"] = dict(sorted(counts.items()))
    summary["delta_absorption"] = {key: len(value) for key, value in absorbed.items()}
    summary["stale_deltas"] = [key for key, value in absorbed.items() if not value]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Differential report", f"Baseline: `{summary['baseline']}`; candidate: `{summary['candidate']}`.",
             f"Baseline ref: `{baseline_ref}`; canonical tag: `{summary['baseline_tag']}`.",
             f"Baseline config: {baseline_config}.", f"Cases: {len(cases)}; seed: {seed}; unexplained events: {len(summary['unexplained'])}.",
             f"Inherited user marks: {summary['inherited_user_marks']['cases']} cases; {summary['inherited_user_marks']['marks']} marks.",
             "", "## Counts (layer / family / oracle)", "", "```json", json.dumps(summary["counts"], indent=2), "```",
             "", f"Delta absorption (cases): {summary['delta_absorption']}",
             f"Zero-absorption deltas (stale on a full run): {summary['stale_deltas']}",
             f"Timings: {summary['timings']}", f"Missing seams: {summary['missing_seams']}", "", "## Unexplained", ""]
    for record in summary["unexplained"]:
        lines += [f"- {record['case_id']}: {record['oracle']} — {record['detail']}",
                  "```json", json.dumps(record["case"], ensure_ascii=False), "```"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=os.environ.get("LCM_DIFFERENTIAL_BASELINE", DEFAULT_BASELINE))
    parser.add_argument("--baseline-config", choices=["bare", "managed"],
                        default=os.environ.get("LCM_DIFFERENTIAL_BASELINE_CONFIG"))
    parser.add_argument("--candidate", default=os.environ.get("LCM_DIFFERENTIAL_CANDIDATE", "worktree"))
    parser.add_argument("--layers", default="public,projection")
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1227)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", help="Comma-separated exact case IDs; omits generated cases")
    args = parser.parse_args()
    if os.environ.get("LCM_DIFFERENTIAL") != "1":
        parser.error("set LCM_DIFFERENTIAL=1 to opt in")
    sample = []
    layers = args.layers.split(",")
    if not set(layers) <= {"public", "projection"}:
        parser.error("layers must be public and/or projection")
    if "public" in layers:
        sample += exact_cases() + generated_cases(args.cases, args.seed)
    if "projection" in layers:
        sample += projection_cases(args.cases, args.seed)
    if args.only:
        sample = [case for case in sample if case["id"] in args.only.split(",")]
        if {case["id"] for case in sample} != set(args.only.split(",")):
            parser.error("unknown case ID")
    try:
        summary = run(sample, args.out, args.baseline, args.candidate, args.seed, args.baseline_config)
    except PrerequisiteUnavailable as exc:
        print(f"{'FAIL' if os.environ.get('CI') else 'SKIP'}: {exc}", file=sys.stderr)
        return 1 if os.environ.get("CI") else 0
    print(f"{args.out / 'summary.json'}: {len(summary['unexplained'])} unexplained events")
    return bool(summary["unexplained"])


if __name__ == "__main__":
    sys.exit(main())
