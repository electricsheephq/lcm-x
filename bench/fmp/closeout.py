"""Immutable scoring inputs and the installed result-card writer; no publication."""
import importlib.util
import os
import shlex
import uuid
from pathlib import Path


def card_tool():
    path = os.environ.get("FMP_RESULT_CARD_TOOL")
    if not path or not Path(path).is_file():
        raise ValueError("set FMP_RESULT_CARD_TOOL to the installed bench-card.py")
    spec = importlib.util.spec_from_file_location("fmp_bench_card", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(runs, output):
    card_tool()  # Refuse a score that cannot produce its mandatory envelope.
    if (output / "report.json").exists():
        raise ValueError("score output already exists; use a new directory")
    root = output / "score-inputs" / uuid.uuid4().hex
    paths = []
    for i, run in enumerate(map(Path, runs)):
        target = root / str(i)
        target.mkdir(parents=True)
        for name in ("run-manifest.json", "judged.jsonl"):
            (target / name).write_bytes((run / name).read_bytes())
        paths.append(target)
    return paths


def result_card(output, report, allow_partial, pair_across_commits, snapshots):
    tool = card_tool()
    inputs = [str(p.relative_to(output)) for p in snapshots]
    command = shlex.join(["python", "-m", "bench.fmp", "score", "--split-label", "rescore",
        "--store-root", "unused", "--output", "RESCORE_OUTPUT", "--runs", *inputs,
        *(["--allow-partial"] if allow_partial else []),
        *(["--pair-across-commits"] if pair_across_commits else [])])

    def invoke(*args):
        if tool.execute(tool.parser().parse_args([args[0], str(output), *args[1:]])):
            raise ValueError("result card validation failed")

    invoke("init", "--project", "lcm-x", "--kind", "benchmark", "--owner-lane", "fmp",
           "--code-dir", str(Path(__file__).resolve().parents[2]), "--command", command,
           "--claim-class", "advisory", "--local-raw", "false")
    lines = ["# FMP score", "", "| Arm | Accuracy | N | Receipt |", "|---|---:|---:|---|"]
    for arm, summary in report["arms"].items():
        lines.append(f"| {arm} | {summary['primary']} | {summary['scored_n']} | report.json |")
        if summary["primary"] is not None:
            invoke("headline", "--benchmark", "FMP/" + arm, "--metric", "accuracy",
                   "--value", str(summary["primary"]), "--n", str(summary["scored_n"]), "--source", "report.json")
    (output / "RESULT.md").write_text("\n".join(lines) + "\n\nRe-score from this run directory with "
        "FMP_RESULT_CARD_TOOL set and PYTHONPATH pointing to the recorded harness checkout:\n```sh\n" + command +
        "\n```\n\nNot proven: publication, release quality, runtime or customer readiness.\nT1: unpublished. T2: unpublished.\n")
    invoke("add", "report.json", "report.md", "adjudicate.jsonl", "RESULT.md", "--role", "headline")
    invoke("rescore", "--command", command, "--input", *inputs)
    invoke("add", *inputs, "--role", "per-item")
    invoke("finalize")
    invoke("check")
    invoke("status", "settled")
