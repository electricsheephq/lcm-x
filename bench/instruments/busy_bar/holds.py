"""Relative write-window probe: fresh serial processes, keyword-only SQLite traces."""

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
    from . import cells, driver
else:
    import cells
    import driver

WRITE_WORDS = {"INSERT", "UPDATE", "DELETE", "REPLACE"}


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


def probe(tree: Path, work: Path, rows: int) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=False)
    for key in list(os.environ):
        if key.startswith(("LCM_", "HERMES_")):
            del os.environ[key]
    os.environ.update(HERMES_HOME=str(work / "isolated-home"), LCM_INGEST_LEDGER="shadow",
                      LCM_IDENTITY_ANCHOR="on", LCM_HOST_MESSAGE_UID="off")
    events: list[tuple[float, int, str]] = []
    original_connect = sqlite3.connect
    connections = 0

    def traced_connect(*args: Any, **kwargs: Any) -> Any:
        nonlocal connections
        conn = original_connect(*args, **kwargs)
        index = connections
        connections += 1

        def trace(statement: str) -> None:
            parts = statement.lstrip().split(None, 1)
            # SQL text is discarded here, never retained in events or output.
            events.append((time.monotonic(), index, parts[0].upper() if parts else ""))

        conn.set_trace_callback(trace)
        return conn

    sqlite3.connect = traced_connect
    engine = None
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
        cells.start(engine, "A")
        view = cells.writer_view(0, rows)
        results = []
        for shape, host_view in (("n", view), ("t", view + cells.turn(rows + 1, "B"))):
            events.clear()
            begin = time.monotonic()
            engine.ingest(host_view)
            end = time.monotonic()
            found = windows(events, begin, end)
            results.append({"shape": shape, "view_rows": len(host_view),
                            "call_ms": (end - begin) * 1000, "windows": found,
                            "longest_ms": max((window["hold_ms"] for window in found), default=0),
                            "first_start_ms": min((window["start_ms"] for window in found), default=None),
                            "trace": events.copy()})
        return {"tree": str(tree), "rows": rows, "connections": connections, "calls": results}
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            sqlite3.connect = original_connect


def summarize(collected: dict[str, list]) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    for name, runs in collected.items():
        tables[name] = {}
        for shape in ("n", "t"):
            calls = [call for run in runs for call in run["calls"] if call["shape"] == shape]
            if not calls or any(call["first_start_ms"] is None for call in calls):
                raise ValueError(f"missing write windows: {name} shape {shape}")
            longest = sorted(call["longest_ms"] for call in calls)
            tables[name][shape] = {
                "calls": len(calls), "median_longest_ms": statistics.median(longest),
                "p90_longest_ms": longest[math.ceil(0.9 * len(longest)) - 1],
                "median_first_start_ms": statistics.median(call["first_start_ms"] for call in calls),
                "median_call_ms": statistics.median(call["call_ms"] for call in calls),
            }
    base = tables["baseline"]["n"]["median_longest_ms"]
    candidate = tables["candidate"]["n"]["median_longest_ms"]
    return {"tables": tables, "verdict": "LOCK-RELATIVE PASS" if candidate <= 1.25 * base else "LOCK-RELATIVE FAIL"}


def report(result: dict[str, Any]) -> str:
    lines = ["tree shape calls median_longest_ms p90_longest_ms median_first_start_ms median_call_ms"]
    for name, shapes in result["tables"].items():
        for shape, row in shapes.items():
            values = " ".join(f"{row[key]:.3f}" for key in (
                "median_longest_ms", "p90_longest_ms", "median_first_start_ms", "median_call_ms"))
            lines.append(f"{name} {shape} {row['calls']} {values}")
    base = result["tables"]["baseline"]["n"]["median_longest_ms"]
    candidate = result["tables"]["candidate"]["n"]["median_longest_ms"]
    lines.append(f"{result['verdict']}: baseline={base:.3f} ms candidate={candidate:.3f} ms limit={1.25 * base:.3f} ms")
    lines.append("Shape t is reported only; its absolute bar comes from PR1 counters.")
    return "\n".join(lines) + "\n"


def main() -> None:
    # Private child mode keeps each tree isolated in its own interpreter.
    if len(sys.argv) > 1 and sys.argv[1] == "--probe":
        tree, output, work, rows = sys.argv[2:]
        result = probe(Path(tree).resolve(), Path(work).resolve(), int(rows))
        with Path(output).open("x") as handle:
            json.dump(result, handle)
            handle.write("\n")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_tree", type=Path)
    parser.add_argument("candidate_tree", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--work-root", type=Path,
                        default=Path(tempfile.gettempdir()) / ("busy-bar-holds-" + uuid.uuid4().hex))
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
                       str((args.work_root / prefix).resolve()), str(args.rows)]
            with (args.outdir / f"{prefix}.log").open("x") as log:
                subprocess.run(command, env=env, stdout=log, stderr=log, check=True, timeout=120)
            collected[name].append(json.loads(output.read_text()))
    result = summarize(collected)
    (args.outdir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rendered = report(result)
    (args.outdir / "verdict.txt").write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["verdict"] == "LOCK-RELATIVE PASS" else 1)


if __name__ == "__main__":
    main()
