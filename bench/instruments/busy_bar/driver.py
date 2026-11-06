"""Run one tree by path with two public-API engines sharing a fresh store."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import random
import sqlite3
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

if __package__:
    from . import cells
else:
    import cells

DEFAULT_K = 2
DEFAULT_B_VIEW_ROWS = 2000
DEFAULT_BUSY_TIMEOUT_MS = None
DEFAULT_SEED = 2055
DEFAULT_WINDOW_MS = 400
COUNTERS = {"compress": "ledger.emission_busy", "session-end": "ledger.tail.busy_at_end"}


def busy_signal(message: str) -> str | None:
    if "SQLite write lock recovery exhausted" in message:
        return "write-lock-recovery"
    if "LCM ingest failed" in message and any(word in message.lower() for word in ("busy", "locked")):
        return "ingest-busy"
    if "skipped due to SQLite lock" in message:
        return "skipped-lock"
    return None


class Capture(logging.Handler):
    """Root records retain emitting thread, monotonic time, and full message."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append({"thread": record.thread, "time": time.monotonic(),
                             "message": record.getMessage()})


def count_operation(records: list[dict[str, Any]], thread: int, begin: float, end: float,
                    op: str, before: int, after: int, candidate: bool) -> dict[str, Any]:
    signals = sorted({signal for record in records
                      if record["thread"] == thread and begin <= record["time"] <= end
                      if (signal := busy_signal(record["message"]))})
    counter = COUNTERS.get(op)
    if candidate and counter and after > before:
        signals.append(counter)
    return {"busy": bool(signals), "signals": signals}


class Operations:
    def __init__(self, capture: Capture, candidate: bool) -> None:
        self.capture = capture
        self.candidate = candidate
        self.records: list[dict[str, Any]] = []

    def __call__(self, engine: Any, owner: str, op: str, call: Callable[[], Any],
                 view_rows: int) -> Any:
        counter = COUNTERS.get(op, "")

        def read() -> int:
            return int(getattr(engine, "_ingest_ledger_counters", {}).get(counter, 0)) if self.candidate else 0

        before = read()
        thread, begin = threading.get_ident(), time.monotonic()
        error = message = None
        try:
            return call()
        except Exception as exc:
            error = type(exc).__name__
            if isinstance(exc, sqlite3.Error):
                message = str(exc)[:200]
            return None
        finally:
            end, after = time.monotonic(), read()
            counted = count_operation(self.capture.records.copy(), thread, begin, end,
                                      op, before, after, self.candidate)
            if message and any(word in message.lower() for word in ("locked", "busy")):
                counted["busy"] = True
                counted["signals"].append("sqlite-exception")
            self.records.append({"op": op, "engine": owner, "thread": thread,
                                 "begin": begin, "end": end, "view_rows": view_rows,
                                 "counter_before": before, "counter_after": after,
                                 "error": error, "sqlite_message": message, **counted})


def load_tree(tree: Path) -> tuple[Any, Any]:
    # Same package registration as occurrence_fuzz.driver.load_tree; one tree/process.
    sys.path.insert(0, str(tree))
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm", tree / "__init__.py", submodule_search_locations=[str(tree)]
    )
    assert spec is not None
    sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
    return importlib.import_module("hermes_lcm.config"), importlib.import_module("hermes_lcm.engine")


def offline_setup(engine_module: Any) -> None:
    for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
        if hasattr(engine_module.LCMEngine, name):
            setattr(engine_module.LCMEngine, name, lambda *args, **kwargs: None)
    # Retain the real public compression pipeline, replacing only its model adapter.
    engine_module.summarize_with_escalation = lambda *args, **kwargs: (
        "Busy-bar deterministic summary of earlier turns. Expand for details: earlier turns.", 1
    )

    def deny_network(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("busy-bar is offline: network access forbidden")

    socket.socket.connect = deny_network
    socket.create_connection = deny_network


class FixedContention:
    """One barrier-released batch of exactly K B operations per A operation."""

    def __init__(self, k: int, write: Callable[[int, int], None],
                 seed: int = DEFAULT_SEED, window_ms: float = DEFAULT_WINDOW_MS) -> None:
        self.k, self.write = k, write
        self.random = random.Random(seed)
        self.window_ms = window_ms
        self.offsets: list[float] = []
        self.start = threading.Barrier(2, timeout=120)
        self.done = threading.Barrier(2, timeout=120)
        self.stopping = False
        self.error: BaseException | None = None
        self.worker = threading.Thread(target=self._worker, name="busy-bar-B")
        self.worker.start()

    def _worker(self) -> None:
        batch = 0
        try:
            while True:
                self.start.wait()
                if self.stopping:
                    return
                for index in range(self.k):
                    self.write(batch, index)
                self.done.wait()
                batch += 1
        except BaseException as exc:
            self.error = exc
            self.start.abort()
            self.done.abort()

    def __call__(self, call: Callable[[], Any]) -> Any:
        offset = self.random.random() * self.window_ms
        self.offsets.append(offset)
        self.start.wait()
        try:
            time.sleep(offset / 1000)
            return call()
        finally:
            self.done.wait()

    def close(self) -> None:
        self.stopping = True
        try:
            self.start.wait()
        finally:
            self.worker.join()
        if self.error is not None:
            raise RuntimeError("B operation failed") from self.error


def run(cell: str, directory: Path, config_module: Any, engine_module: Any,
        capture: Capture, k: int = DEFAULT_K, b_view_rows: int = DEFAULT_B_VIEW_ROWS,
        busy_timeout_ms: int | None = DEFAULT_BUSY_TIMEOUT_MS,
        seed: int = DEFAULT_SEED, window_ms: float = DEFAULT_WINDOW_MS) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    capture.records.clear()
    config = config_module.LCMConfig(
        database_path=str(directory / "lcm.db"), embeddings_enabled=False,
        extraction_enabled=False, assertion_extraction_enabled=False,
        background_leaf_prep_enabled=False, fresh_tail_count=6, leaf_chunk_tokens=400,
        large_output_externalization_path=str(directory / "externalized"),
    )
    candidate = hasattr(engine_module.LCMEngine, "_ledger_count")
    operations = Operations(capture, candidate)
    errors: list[str] = []
    timeouts: list[dict[str, Any]] = []
    ticks: list[dict[str, Any]] = []

    def new_engine(owner: str) -> Any:
        engine = engine_module.LCMEngine(config=config, hermes_home=str(directory / "home"))
        cells.start(engine, owner)  # Opens the lazy store before contention.
        conn = engine._store._conn
        if busy_timeout_ms is not None:
            conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        actual = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        if busy_timeout_ms is not None and actual != busy_timeout_ms:
            raise RuntimeError("busy timeout was not applied")
        timeouts.append({"engine": owner, "busy_timeout_ms": actual})
        return engine

    engine_a = new_engine("A")
    ready, finish = threading.Event(), threading.Event()
    engine_b = None

    # Create/shutdown B on the same thread that executes its public operations.
    def writer_lifecycle(batch: int, index: int) -> None:
        number = batch * k + index
        view = cells.writer_view(number, b_view_rows)
        ticks.append({"a_operation": batch, "b_index": index, "view_rows": len(view)})
        operations(engine_b, "B", "ingest", lambda: engine_b.ingest(view), len(view))

    # FixedContention owns the driver thread; initialize before its first barrier.
    class Writer(FixedContention):
        def _worker(self) -> None:
            nonlocal engine_b
            try:
                engine_b = new_engine("B")
                ready.set()
                super()._worker()
            except BaseException as exc:
                self.error = exc
                self.start.abort()
                self.done.abort()
            finally:
                ready.set()
                # A's last batch is complete before shutdown releases B's store.
                finish.wait()
                if engine_b is not None:
                    try:
                        engine_b.shutdown()
                    except Exception as exc:
                        errors.append(f"B shutdown:{type(exc).__name__}")

    contention = Writer(k, writer_lifecycle, seed, window_ms)
    ready.wait()

    def restart() -> Any:
        nonlocal engine_a
        engine_a.shutdown()
        engine_a = new_engine("A")
        return engine_a

    def operation(*args: Any) -> Any:
        return contention(lambda: operations(*args))

    try:
        if contention.error is not None:
            raise RuntimeError("B initialization failed")
        cells.sequence(cell, engine_a, restart, operation)
    except Exception as exc:
        errors.append(f"A:{type(exc).__name__}")
    finally:
        finish.set()
        try:
            contention.close()
        except Exception as exc:
            errors.append(f"B:{type(exc).__name__}")
        try:
            engine_a.shutdown()
        except Exception as exc:
            errors.append(f"A shutdown:{type(exc).__name__}")
    records = sorted(operations.records, key=lambda record: record["begin"])
    totals = {owner: {"operations": sum(r["engine"] == owner for r in records),
                      "busy": sum(r["engine"] == owner and r["busy"] for r in records)}
              for owner in ("A", "B")}
    totals["total"] = {key: sum(totals[owner][key] for owner in ("A", "B"))
                       for key in ("operations", "busy")}
    if (totals["A"]["operations"] != cells.A_OPERATIONS[cell]
            or totals["B"]["operations"] != k * cells.A_OPERATIONS[cell]):
        errors.append("fixed exposure violated: driver bug")
    return {"cell": cell, "k": k, "b_view_rows": b_view_rows,
            "busy_timeout_ms": busy_timeout_ms, "applied_timeouts": timeouts,
            "seed": seed, "window_ms": window_ms, "offsets_ms": contention.offsets,
            "candidate_counters": candidate, "b_batches": ticks,
            "b_view_sizes": [tick["view_rows"] for tick in ticks],
            "operations": records, "totals": totals, "errors": errors,
            "logs": capture.records.copy()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path)
    parser.add_argument("cell", choices=cells.CELLS)
    parser.add_argument("output", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--b-view-rows", type=int, default=DEFAULT_B_VIEW_ROWS)
    parser.add_argument("--busy-timeout-ms", type=int, default=DEFAULT_BUSY_TIMEOUT_MS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window-ms", type=float, default=DEFAULT_WINDOW_MS)
    args = parser.parse_args()
    if min(args.runs, args.k, args.b_view_rows) < 1 or args.window_ms < 0:
        parser.error("runs, k and b-view-rows must be positive; window-ms must be nonnegative")
    if args.busy_timeout_ms is not None and args.busy_timeout_ms < 0:
        parser.error("busy-timeout-ms must be nonnegative")
    # Bind every home/config lookup before importing the target, never the real profile.
    for key in list(os.environ):
        if key.startswith(("LCM_", "HERMES_")):
            del os.environ[key]
    os.environ.update(HERMES_HOME=str(args.work.resolve() / "isolated-home"),
                      LCM_INGEST_LEDGER="shadow", LCM_IDENTITY_ANCHOR="on", LCM_HOST_MESSAGE_UID="off")
    config_module, engine_module = load_tree(args.tree.resolve())
    for method in ("ingest", "compress", "on_session_start", "on_session_end", "shutdown"):
        if not callable(getattr(engine_module.LCMEngine, method, None)):
            raise RuntimeError(f"needed public method missing: {method}")
    offline_setup(engine_module)
    capture = Capture()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(capture)
    failed = False
    try:
        with args.output.open("x") as output:
            for number in range(args.runs):
                result = run(args.cell, args.work / f"run-{number}", config_module,
                             engine_module, capture, args.k, args.b_view_rows, args.busy_timeout_ms,
                             args.seed + number, args.window_ms)
                result["run"] = number
                output.write(json.dumps(result) + "\n")
                output.flush()
                failed |= bool(result["errors"])
    finally:
        root.removeHandler(capture)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
