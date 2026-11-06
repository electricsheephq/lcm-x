"""Relative WAL write-lock probe using an external read-only F_GETLK monitor."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import uuid
import time
from pathlib import Path
from typing import Any

if __package__:
    from . import cells, driver, monitor
else:
    import cells
    import driver
    import monitor

WRITE_WORDS = {"INSERT", "UPDATE", "DELETE", "REPLACE"}

# Design r3.7 (§5): the 2,000-new-row hold ratio is reported; it fails only above this
# regression tripwire against v0.29.2 (r3.6's 1.25 bar no longer applies).
TRIPWIRE = 2.25
GATE_RUNS = 10
GATE_ROWS = 2000


def windows(events: list[tuple[float, int, str]], begin: float, end: float) -> list[dict[str, Any]]:
    """Extract per-connection top-level windows; SAVEPOINT/RELEASE are neutral."""
    opened: dict[int, float] = {}
    found: list[dict[str, Any]] = []

    def close(conn: int, at: float, open_at_end: bool = False) -> None:
        start = opened.pop(conn)
        found.append({"conn": conn, "start_ms": (start - begin) * 1000,
                      "hold_ms": (at - start) * 1000, "open_at_end": open_at_end})

    for at, conn, word in events:
        if not begin <= at <= end:
            continue
        if conn not in opened:
            if word == "BEGIN" or word in WRITE_WORDS:
                opened[conn] = at
        elif word in {"COMMIT", "ROLLBACK", "END"}:
            close(conn, at)
    for conn in list(opened):
        close(conn, end, True)
    return found


def probe(tree: Path, work: Path, rows: int, trace: bool = False) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=False)
    driver.isolated_environment(work)
    events: list[tuple[float, int, str]] = []
    original_connect = sqlite3.connect
    connections = 0

    def traced_connect(*args: Any, **kwargs: Any) -> Any:
        nonlocal connections
        conn = original_connect(*args, **kwargs)
        index = connections
        connections += 1

        def callback(statement: str) -> None:
            parts = statement.lstrip().split(None, 1)
            events.append((time.monotonic(), index, parts[0].upper() if parts else ""))

        conn.set_trace_callback(callback)
        return conn

    if trace:
        sqlite3.connect = traced_connect
    engine = observer = None
    try:
        config_module, engine_module = driver.load_tree(tree)
        driver.offline_setup(engine_module)
        config = config_module.LCMConfig(
            database_path=str(work / "lcm.db"), embeddings_enabled=False,
            extraction_enabled=False, assertion_extraction_enabled=False,
            background_leaf_prep_enabled=False, fresh_tail_count=6, leaf_chunk_tokens=400,
            large_output_externalization_path=str(work / "externalized"),
        )
        engine = engine_module.LCMEngine(config=config, hermes_home=str(work / "home"))
        cells.start(engine, "B")
        journal = engine._store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        if journal.lower() != "wal":
            raise RuntimeError(f"lock monitor requires WAL, got {journal}")
        observer = monitor.Monitor(work / "lcm.db")
        results = []
        # Each all-new bounded view is distinct, but belongs to the same session.
        views = [("n", position + 1, cells.writer_view(position, rows)) for position in range(6)]
        views.append(("t", 1, views[-1][2] + cells.turn(6 * rows + 1, "B")))
        for shape, position, host_view in views:
            events.clear()
            stored_before = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            failures_before = getattr(engine, "_consecutive_ingest_failures", 0)
            begin = time.monotonic_ns()
            engine.ingest(host_view)
            end = time.monotonic_ns()
            # ingest() records internal failures without raising; a failed or short ingest would
            # hold the lock briefly and could pass the tripwire, so it fails the probe instead.
            stored = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] - stored_before
            expected = len(host_view) if shape == "n" else len(host_view) - len(views[-2][2])
            failures = getattr(engine, "_consecutive_ingest_failures", 0) - failures_before
            if stored != expected or failures:
                raise RuntimeError(f"{shape} call {position} stored {stored} rows, expected {expected}, "
                                   "or recorded an ingest failure")
            call = {"shape": shape, "position": position, "view_rows": len(host_view),
                    "rows_stored": stored, "ingest_failures": failures,
                    "begin_ns": begin, "end_ns": end, "call_ms": (end - begin) / 1e6}
            if trace:
                call["trace"] = events.copy()
                call["trace_windows"] = windows(events, begin / 1e9, end / 1e9)
            results.append(call)
        # Ensure a post-call observation even for the last very short call.
        time.sleep(2 * monitor.POLL_NS / 1e9)
        observed = observer.stop()
        observer = None
        samples = observed["samples"]
        if samples[0][0] > results[0]["begin_ns"] or samples[-1][0] < results[-1]["end_ns"]:
            raise RuntimeError("monitor samples do not cover all calls")
        for call in results:
            found = monitor.held_periods(samples, call["begin_ns"], call["end_ns"])
            call["windows"] = [{"start_ns": start, "end_ns": end,
                                "hold_ms": (end - start) / 1e6} for start, end in found]
            call["longest_ms"] = max(((end - start) / 1e6 for start, end in found), default=0)
            call["first_start_ms"] = (found[0][0] - call["begin_ns"]) / 1e6 if found else None
        return {"tree": str(tree), "tree_identity": driver.tree_identity(tree), "run_id": uuid.uuid4().hex,
                "rows": rows, "journal_mode": journal,
                "effective_defaults": driver.effective_defaults(config_module),
                "trace_diagnostic": trace, "monitor": observed, "calls": results}
    finally:
        try:
            if observer is not None:
                observer.stop()
        finally:
            try:
                if engine is not None:
                    engine.shutdown()
            finally:
                sqlite3.connect = original_connect


def unsupported(run: dict[str, Any]) -> str | None:
    """Why a run is not a monitored probe of full, failure-free views, if it is not (#2301 review)."""
    calls = run["calls"]
    if any(call.get("view_rows") != run.get("rows") or call.get("rows_stored") != run.get("rows")
           for call in calls if call["shape"] == "n"):
        return "a shape-n call did not present and store its full view"
    if any(call.get("ingest_failures") != 0 for call in calls):
        return "a call recorded an ingest failure or none was recorded"
    monitor_result = run.get("monitor") or {}
    samples = monitor_result.get("samples") or []
    begins, ends = [call.get("begin_ns") for call in calls], [call.get("end_ns") for call in calls]
    if (not samples or "intervals" not in monitor_result or None in begins + ends
            or samples[0][0] > min(begins) or samples[-1][0] < max(ends)):
        return "the monitor did not complete over every call"
    return None


def summarize(collected: dict[str, list], baseline_identity_override: bool = False) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    statistics_by_tree: dict[str, dict[str, float]] = {}
    for name, runs in collected.items():
        tables[name] = {}
        if not runs:
            raise ValueError(f"no runs: {name}")
        for run in runs:
            positions = [call["position"] for call in run["calls"] if call["shape"] == "n"]
            if positions != list(range(1, 7)) or run["journal_mode"].lower() != "wal":
                raise ValueError(f"invalid six-call/WAL run: {name}")
        for shape in ("n", "t"):
            calls = [call for run in runs for call in run["calls"] if call["shape"] == shape]
            if not calls:
                raise ValueError(f"missing calls: {name} shape {shape}")
            longest = sorted(call["longest_ms"] for call in calls)
            starts = [call["first_start_ms"] for call in calls if call["first_start_ms"] is not None]
            tables[name][shape] = {
                "calls": len(calls), "median_longest_ms": statistics.median(longest),
                "p90_longest_ms": longest[math.ceil(0.9 * len(longest)) - 1],
                "median_first_start_ms": statistics.median(starts) if starts else None,
                "median_call_ms": statistics.median(call["call_ms"] for call in calls),
            }
        n_calls = [call for run in runs for call in run["calls"] if call["shape"] == "n"]
        statistics_by_tree[name] = {
            "all-calls": tables[name]["n"]["median_longest_ms"],
            **{f"position-{position}": statistics.median(
                call["longest_ms"] for call in n_calls if call["position"] == position)
               for position in range(1, 7)},
            "run-longest": statistics.median(max(call["longest_ms"] for call in run["calls"]
                                                 if call["shape"] == "n") for run in runs),
        }
    if len(collected["baseline"]) != len(collected["candidate"]):
        raise ValueError("unequal lock-probe run counts")
    failed = [key for key, value in statistics_by_tree["candidate"].items()
              if value > TRIPWIRE * statistics_by_tree["baseline"][key]]
    diagnostic = any(run.get("trace_diagnostic") for runs in collected.values() for run in runs)
    # A blind monitor must never pass: every 2,000-new-row call holds the write lock. NaN and inf
    # compare false against the tripwire, so only finite positive holds can be compared at all.
    def unobserved(run: dict[str, Any]) -> int:
        return sum(1 for call in run["calls"] if call["shape"] == "n"
                   and not (isinstance(call["longest_ms"], (int, float)) and math.isfinite(call["longest_ms"])
                            and call["longest_ms"] > 0))

    reasons = [f"{name}: {sum(map(unobserved, runs))} shape-n calls with no observed hold (or a nonfinite one)"
               for name, runs in collected.items() if any(map(unobserved, runs))]
    reasons += [f"{name} statistic {key} is {value}" for name, values in statistics_by_tree.items()
                for key, value in values.items() if not (math.isfinite(value) and value > 0)]
    reasons += [f"{name} run {number}: {why}" for name, runs in collected.items()
                for number, run in enumerate(runs) if (why := unsupported(run))]
    run_ids = [run.get("run_id") for runs in collected.values() for run in runs]
    if None in run_ids or len(set(run_ids)) != len(run_ids):
        reasons.append("run ids are missing or repeated; each counted run must be a distinct probe")
    reasons += driver.identity_reasons(collected, baseline_identity_override)
    verdict = ("DIAGNOSTIC ONLY" if diagnostic else "INCONCLUSIVE" if reasons else
               "LOCK-TRIPWIRE FAIL" if failed else "LOCK-TRIPWIRE PASS")
    if verdict.startswith("LOCK-TRIPWIRE") and len(collected["baseline"]) != GATE_RUNS:
        verdict = verdict.replace("TRIPWIRE", "SMOKE")  # The gate session is ten runs per tree.
        reasons.append(f"{len(collected['baseline'])} runs per tree; a gate session needs {GATE_RUNS}")
    rows = sorted({run.get("rows") for runs in collected.values() for run in runs} - {GATE_ROWS})
    if verdict.startswith("LOCK-TRIPWIRE") and rows:
        verdict = verdict.replace("TRIPWIRE", "SMOKE")  # The gate session uses 2,000-new-row views.
        reasons.append(f"rows={rows}; a gate session uses {GATE_ROWS}")
    if verdict.startswith("LOCK-TRIPWIRE") and baseline_identity_override:
        verdict = verdict.replace("TRIPWIRE", "SMOKE")
        reasons.append("baseline identity override: a smoke, never a gate verdict")
    return {"tables": tables, "statistics": statistics_by_tree, "failed_statistics": failed,
            "reasons": reasons, "verdict": verdict}


def report(result: dict[str, Any]) -> str:
    lines = ["tree shape calls median_longest_ms p90_longest_ms median_first_start_ms median_call_ms"]
    for name, shapes in result["tables"].items():
        for shape, row in shapes.items():
            values = " ".join(f"{row[key]:.3f}" if row[key] is not None else "unobserved" for key in (
                "median_longest_ms", "p90_longest_ms", "median_first_start_ms", "median_call_ms"))
            lines.append(f"{name} {shape} {row['calls']} {values}")
    lines.append("statistic baseline_ms candidate_ms tripwire_ms ratio")
    for key, base in result["statistics"]["baseline"].items():
        candidate = result["statistics"]["candidate"][key]
        lines.append(f"{key} {base:.3f} {candidate:.3f} {TRIPWIRE * base:.3f} {candidate / base if base else float('nan'):.3f}")
    lines.append(result["verdict"])
    lines.extend(result.get("reasons", ()))
    if result["failed_statistics"]:
        lines.append("exceeded: " + ", ".join(result["failed_statistics"]))
    lines.append("Shape t is reported only; its absolute bar comes from harness counters.")
    return "\n".join(lines) + "\n"


def main() -> None:
    # Private child mode keeps each tree isolated in its own interpreter.
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        tree, output, work, rows, trace = sys.argv[2:]
        result = probe(Path(tree).resolve(), Path(work).resolve(), int(rows), trace == "True")
        with Path(output).open("x") as handle:
            json.dump(result, handle)
            handle.write("\n")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_tree", type=Path)
    parser.add_argument("candidate_tree", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--trace", action="store_true", help="diagnostic only; disables verdict")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--rows", type=int, default=GATE_ROWS)
    parser.add_argument("--work-root", type=Path,
                        default=Path(tempfile.gettempdir()) / ("busy-bar-holds-" + uuid.uuid4().hex))
    parser.add_argument("--baseline-identity-override", action="store_true",
                        help="accept a baseline other than v0.29.2 GA; the verdict is a smoke")
    args = parser.parse_args()
    if min(args.runs, args.rows) < 1:
        parser.error("runs and rows must be positive")
    if args.work_root.exists():
        parser.error("scratch directory must be fresh")
    args.outdir.mkdir(parents=True, exist_ok=False)
    collected: dict[str, list] = {"baseline": [], "candidate": []}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for number in range(args.runs):
        pair = [("baseline", args.baseline_tree), ("candidate", args.candidate_tree)]
        if number % 2:
            pair.reverse()
        for name, tree in pair:
            prefix = f"{number}-{name}"
            output = args.outdir / f"{prefix}.json"
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--probe",
                       str(tree.resolve()), str(output.resolve()),
                       str((args.work_root / prefix).resolve()), str(args.rows), str(args.trace)]
            with (args.outdir / f"{prefix}.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=log, check=True, timeout=120)
            collected[name].append(json.loads(output.read_text()))
    result = summarize(collected, args.baseline_identity_override)
    (args.outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rendered = report(result)
    (args.outdir / "verdict.txt").write_text(rendered)
    print(rendered, end="")
    raise SystemExit({"LOCK-TRIPWIRE PASS": 0, "DIAGNOSTIC ONLY": 0, "LOCK-TRIPWIRE FAIL": 1,
                      "INCONCLUSIVE": 2}.get(result["verdict"], 3))


if __name__ == "__main__":
    main()
