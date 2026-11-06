"""Interleave fresh baseline/candidate processes, serially, run by run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

if __package__:
    from . import cells, driver, gate
else:
    import cells
    import driver
    import gate


def collect(baseline_tree: Path, candidate_tree: Path, outdir: Path, runs: int,
            k: int, b_view_rows: int, seed: int, window_ms: float,
            busy_timeout_ms: int | None, work_root: Path,
            selected_cells: tuple[str, ...] = cells.CELLS, start_run: int = 0) -> dict[str, list]:
    outdir.mkdir(parents=True, exist_ok=False)
    driver_path = Path(__file__).with_name("driver.py")
    collected: dict[str, list] = {"baseline": [], "candidate": []}
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (outdir / "baseline.jsonl").open("x") as baseline, (outdir / "candidate.jsonl").open("x") as candidate:
        outputs = {"baseline": baseline, "candidate": candidate}
        for number in range(start_run, start_run + runs):
            for cell in selected_cells:
                pair = [("baseline", baseline_tree), ("candidate", candidate_tree)]
                if number % 2:
                    pair.reverse()
                for name, tree in pair:
                    prefix = f"{number}-{cell}-{name}"
                    output = outdir / f"{prefix}.jsonl"
                    command = [sys.executable, "-B", str(driver_path), str(tree.resolve()), cell,
                               str(output), str(work_root / prefix),
                               "--k", str(k), "--b-view-rows", str(b_view_rows),
                               "--seed", str(seed + number), "--window-ms", str(window_ms)]
                    if busy_timeout_ms is not None:
                        command.extend(["--busy-timeout-ms", str(busy_timeout_ms)])
                    with (outdir / f"{prefix}.log").open("x") as log:
                        completed = subprocess.run(command, env=env, stdout=log, stderr=log, timeout=300)
                    if completed.returncode not in (0, 2) or not output.exists():
                        raise RuntimeError(f"driver failed: {prefix}, exit {completed.returncode}")
                    rows = gate.read(output)
                    if len(rows) != 1:
                        raise RuntimeError(f"expected one run: {prefix}")
                    row = rows[0]
                    row["run"] = number
                    collected[name].append(row)
                    outputs[name].write(json.dumps(row) + "\n")
                    outputs[name].flush()
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_tree", type=Path)
    parser.add_argument("candidate_tree", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--control-outdir", type=Path)
    parser.add_argument("--work-root", type=Path,
                        default=Path(tempfile.gettempdir()) / ("busy-bar-" + uuid.uuid4().hex),
                        help="fresh scratch root for regenerable stores")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--k", type=int, default=driver.DEFAULT_K)
    parser.add_argument("--b-view-rows", type=int, default=driver.DEFAULT_B_VIEW_ROWS)
    parser.add_argument("--seed", type=int, default=driver.DEFAULT_SEED)
    parser.add_argument("--window-ms", type=float, default=driver.DEFAULT_WINDOW_MS)
    args = parser.parse_args()
    if min(args.runs, args.k, args.b_view_rows) < 1 or args.window_ms < 0:
        parser.error("runs, k and b-view-rows must be positive; window-ms must be nonnegative")
    control_dir = args.control_outdir or args.outdir.with_name(args.outdir.name + "-control")
    if args.outdir.exists() or control_dir.exists() or args.work_root.exists():
        parser.error("output and scratch directories must be fresh")
    prod = collect(args.baseline_tree, args.candidate_tree, args.outdir, args.runs,
                   args.k, args.b_view_rows, args.seed, args.window_ms, None, args.work_root / "production")
    extensions = gate.extension_cells(prod["baseline"], prod["candidate"]) if args.runs == 20 else []
    if extensions:
        extra = collect(args.baseline_tree, args.candidate_tree, args.outdir / "extension", 60,
                        args.k, args.b_view_rows, args.seed, args.window_ms, None,
                        args.work_root / "extension", tuple(extensions), start_run=20)
        for name in prod:
            prod[name].extend(extra[name])
            with (args.outdir / f"{name}.jsonl").open("a") as output:
                for row in extra[name]:
                    output.write(json.dumps(row) + "\n")
    control = collect(args.baseline_tree, args.candidate_tree, control_dir, args.runs,
                      args.k, args.b_view_rows, args.seed, args.window_ms, 100, args.work_root / "control")
    result = gate.evaluate(prod["baseline"], prod["candidate"], control["baseline"], control["candidate"])
    rendered = gate.report(result)
    (args.outdir / "verdict.txt").write_text(rendered)
    (args.outdir / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    (control_dir / "verdict.txt").write_text("POSITIVE CONTROL ONLY; never gating\n" +
                                             gate.report(result["control"]))
    print(rendered, end="")
    raise SystemExit({"GATE PASS": 0, "GATE FAIL": 1, "INCONCLUSIVE": 2}[result["verdict"]])


if __name__ == "__main__":
    main()
