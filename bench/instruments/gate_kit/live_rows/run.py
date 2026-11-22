"""Run one live-rows cell on one tree, in its own process, and write its receipt.

    python -B -m bench.instruments.gate_kit.live_rows.run --tree TREE --build BUILD_DIR --cell d9 \
        --arm uid_on --budget default --work WORK_DIR --out RECEIPT.json

The store is copied from BUILD_DIR into WORK_DIR/current (one working copy, overwritten by each run). The receipt
holds counts, ids, hashes, seconds and the product's fixed log lines; never message content.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import hashlib
import json
import fcntl
import os
import sqlite3
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from bench.instruments.gate_kit.live_rows import build as builder  # noqa: E402
from bench.instruments.gate_kit.live_rows import lockwatch, rt, shape  # noqa: E402
from bench.instruments.gate_kit.live_rows.host import auxiliary_client as aux  # noqa: E402

CELLS = ("d9", "d10", "d11", "selftest-cut", "timeout-hold", "turns", "kill-at-first-call")
PROFILE = rt.PROFILE  # diagnostic only: --profile writes the first pass's cProfile stats
CONTROL = rt.CONTROL  # instrument self-test: "short-pass" ingests only half of the list
# One shared dict each in `rt`: under `python -m ...run` this file is `__main__`, and `cells` imports a second copy.


def claim(work: Path) -> int:
    """Hold an exclusive lock on WORK for this process: a second run on the same working copy refuses."""
    work.mkdir(parents=True, exist_ok=True)
    fd = os.open(work / ".run-lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run holds {work}; use a different --work") from None
    return fd


def fresh_copy(build: Path, work: Path) -> Path:
    """Overwrite WORK/current (and WORK/home/state.db) with the build's store, payloads and host store."""
    return builder.restore(build, work)


def key(role: Any, content: Any, tool_call_id: Any) -> str:
    """Tool results by call id (an externalized result is stored as a reference), other rows by role and content."""
    value = ["tool", tool_call_id] if role == "tool" and tool_call_id else [role, content]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def digest(row: dict[str, Any]) -> str:
    return key(row.get("role"), row.get("content"), row.get("tool_call_id"))


def rows_examined(db: Path, conversation: str, view: list[dict[str, Any]]) -> int:
    """Host-list rows stored in the target conversation, matched as a multiset (`key`)."""
    stored = Counter(key(role, content, call) for _sid, _session, role, content, call in rt.stored_rows(db, conversation))
    found = 0
    for row in view:
        wanted = digest(row)
        if stored[wanted] > 0:
            stored[wanted] -= 1
            found += 1
    return found


def pressure(kit: dict[str, Any], view: list[dict[str, Any]]) -> int:
    """The host figure handed to compress(): the list's own tokens (the kit's window puts it over the threshold)."""
    return sum(len(str(r.get("content") or "")) for r in view) // shape.CHARS_PER_TOKEN


def timed(record: list[dict[str, Any]], name: str, call: Any) -> Any:
    entry = {"call": name, "begin_ns": time.monotonic_ns()}
    record.append(entry)
    try:
        return call()
    except Exception as error:  # noqa: BLE001 - class and frames only
        entry["exception"] = type(error).__name__
        if isinstance(error, sqlite3.Error):  # SQLite's own text: no message content
            entry["sqlite_message"] = str(error)[:200]
        entry["frames"] = [f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                           for f in traceback.extract_tb(error.__traceback__)][-12:]
        raise
    finally:
        entry["end_ns"] = time.monotonic_ns()
        entry["seconds"] = round((entry["end_ns"] - entry["begin_ns"]) / 1e9, 3)


TXN: dict[str, Any] = {}  # diagnostic only: --trace-txn records write windows per connection and thread


def trace_transactions() -> list[tuple[float, int, str, str]]:
    """Patch sqlite3.connect so every later connection logs (time, connection, thread name, first SQL word)."""
    events: list[tuple[float, int, str, str]] = []
    original, counter = sqlite3.connect, {"n": 0}

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = original(*args, **kwargs)
        index = counter["n"]
        counter["n"] += 1

        def callback(statement: str) -> None:
            word = statement.lstrip().split(None, 1)[0].upper() if statement.strip() else ""
            if word in ("BEGIN", "COMMIT", "ROLLBACK", "END", "INSERT", "UPDATE", "DELETE", "REPLACE", "SAVEPOINT",
                        "RELEASE"):
                events.append((time.monotonic(), index, threading.current_thread().name, word))

        conn.set_trace_callback(callback)
        return conn

    sqlite3.connect = connect  # type: ignore[assignment]
    return events


def write_windows(events: list[tuple[float, int, str, str]]) -> list[dict[str, Any]]:
    """Top-level write windows per connection: from BEGIN (or a first write) to COMMIT/ROLLBACK/END."""
    opened: dict[int, tuple[float, str, int]] = {}
    found = []
    for at, conn, thread, word in events:
        if conn not in opened:
            if word == "BEGIN" or word in ("INSERT", "UPDATE", "DELETE", "REPLACE"):
                opened[conn] = (at, thread, 0)
        elif word in ("COMMIT", "ROLLBACK", "END"):
            start, owner, writes = opened.pop(conn)
            found.append({"conn": conn, "thread": owner, "seconds": round(at - start, 3), "writes": writes})
        elif word in ("INSERT", "UPDATE", "DELETE", "REPLACE"):
            start, owner, writes = opened[conn]
            opened[conn] = (start, owner, writes + 1)
    return sorted(found, key=lambda w: -w["seconds"])[:8]


def completed(result: Any, compress: dict[str, Any], reasons: list[str], kit: dict[str, Any]) -> bool:
    """The pass reached its end: compress returned a list, raised nothing, logged a stop, and the stop is not one
    that cuts the pass short (an error inside the product, an exhausted budget; the kit's `short_stop_reasons`)."""
    short = set(kit["bars"].get("short_stop_reasons") or SHORT_STOPS)
    return (isinstance(result, list) and "exception" not in compress and bool(reasons)
            and not any(reason in short for reason in reasons))


SHORT_STOPS = ("error", "time_budget_exhausted", "pass_budget_exhausted", "leaf_summary_error",
               "summary_route_unavailable", "summary_result_rejected")


def watch_cursor(engine: Any) -> dict[str, Any]:
    """Rows the pass itself examined: the engine's ingest cursor after each of its ingests during the pass (the
    host-list position the store has reached). A count the pass moves, so a short pass reads short."""
    seen: dict[str, Any] = {"max_cursor": None, "ingests": 0, "hooked": False}
    original = getattr(engine, "_ingest_messages", None)
    if not callable(original) or not hasattr(engine, "_ingest_cursor"):
        return seen
    seen["hooked"] = True

    def counted(messages: Any, *args: Any, **kwargs: Any) -> Any:
        if CONTROL.get("mode") == "short-pass" and isinstance(messages, list):
            messages = messages[: len(messages) // 2]
        try:
            return original(messages, *args, **kwargs)
        finally:
            seen["ingests"] += 1
            cursor = getattr(engine, "_ingest_cursor", None)
            if isinstance(cursor, int):
                seen["max_cursor"] = max(cursor, seen["max_cursor"] or 0)

    engine._ingest_messages = counted
    return seen


def first_pass(engine_module: Any, config: Any, kit: dict[str, Any], home: Path, db: Path,
               view: list[dict[str, Any]], calls: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]] | None]:
    """Restart (a new engine and its session start), then the first compaction pass on the host list."""
    try:
        engine = timed(calls, "restart", lambda: rt.engine(engine_module, config, kit, home))
    except Exception as error:  # noqa: BLE001 - the restart raised: the receipt says so (never a silent UNRUN)
        raise CellError("restart", error) from error
    engine._live_rows_cursor = watch_cursor(engine)
    result = None
    profiler = cProfile.Profile() if PROFILE.get("path") else None
    try:
        if profiler:
            profiler.enable()
        result = timed(calls, "compress", lambda: engine.compress(copy.deepcopy(view),
                                                                  current_tokens=pressure(kit, view)))
    except Exception:  # noqa: BLE001 - recorded in `calls`
        pass
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(PROFILE["path"])
    return engine, result


CellError = rt.CellError


def cell_d9(args: argparse.Namespace, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
            view: list[dict[str, Any]]) -> dict[str, Any]:
    config_module, engine_module, lines = modules
    config = rt.config_for(config_module, db)
    calls: list[dict[str, Any]] = []
    reader = rt.open_reader(db)  # creates the shared-memory file the watcher probes; never writes
    watcher = lockwatch.Watcher(db)
    begin = time.monotonic_ns()
    engine, result = first_pass(engine_module, config, kit, args.work / "home", db, view, calls)
    end = time.monotonic_ns()
    observed = watcher.stop()
    reader.close()
    try:
        engine.shutdown()
    except Exception as error:  # noqa: BLE001
        calls.append({"call": "shutdown", "exception": type(error).__name__})
    compress = next(c for c in calls if c["call"] == "compress")
    pid = None  # one process: every holder is this run
    reasons = [r for r in _stop_reasons(lines)]
    return {
        "effective": rt.effective(config),
        "calls": calls,
        "longest_hold_s": lockwatch.longest_s(observed["transitions"], begin, end, pid),
        "longest_hold_in_pass_s": lockwatch.longest_s(observed["transitions"], compress["begin_ns"],
                                                       compress["end_ns"], pid),
        "holds_over_1s": [round((e - s) / 1e9, 3) for s, e in lockwatch.holds(observed["transitions"], begin, end)
                          if e - s > 1e9],
        "lock_watch": {k: observed.get(k) for k in ("samples", "poll_ns", "largest_gap_ms", "unexpected_answers")},
        "pass_completed": completed(result, compress, reasons, kit), "stop_reasons": reasons,
        "exception": compress.get("exception") or ("compress failed inside the product" if "error" in reasons
                                                    else None),
        "lock_unexpected": observed.get("unexpected_answers"),
        "list_rows_in": len(view), "list_rows_out": len(result) if isinstance(result, list) else None,
        "rows_examined": engine._live_rows_cursor["max_cursor"],
        "cursor_watch": engine._live_rows_cursor,
        "rows_stored_matching_list": rows_examined(db, kit["target"]["conversation_id"], view),
        "summary_calls": [{k: c.get(k) for k in ("input_tokens", "latency_s", "granted_s", "slept_s", "outcome")}
                          for c in aux.CALLS],
        "lines": [{"t_s": round(line["t"] - begin / 1e9, 3), "message": line["message"]} for line in lines.lines],
    }


def _stop_reasons(lines: Any) -> list[str]:
    from bench.instruments.gate_kit.live_rows.cells import stop_reasons
    main = threading.main_thread().ident
    return stop_reasons([line["message"] for line in lines.lines if line.get("thread") in (None, main)])


def cell_selftest_cut(args: argparse.Namespace, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
                      view: list[dict[str, Any]]) -> dict[str, Any]:
    """Shows the stand-in honouring its budget: every call is longer than the time it is given, so each is cut."""
    aux.configure(force_cut=True)
    receipt = cell_d9(args, kit, modules, db, view)
    cuts = [c for c in aux.CALLS if c.get("outcome") == "cut"]
    receipt["cut_calls"] = len(cuts)
    receipt["cut_slept_within_granted"] = all(
        c["granted_s"] is None or c["slept_s"] <= c["granted_s"] + 0.5 for c in cuts)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cell", choices=CELLS, required=True)
    parser.add_argument("--arm", default="default")
    parser.add_argument("--budget", default="default")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=None, help="diagnostic: cProfile stats of the first pass")
    parser.add_argument("--control", choices=("none", "short-pass"), default="none",
                        help="instrument self-test: short-pass ingests half the list, so the row must read short")
    parser.add_argument("--trace-txn", action="store_true", help="diagnostic: write windows by connection and thread")
    parser.add_argument("--worst-case", action="store_true", help="D10's recorded arm: threads never sleep")
    parser.add_argument("--load", type=int, default=0, help="requirement 2: N CPU-bound processes during the cell")
    parser.add_argument("--keep-copy", action="store_true",
                        help="requirement 3: reuse WORK/current as the killed run left it (no fresh copy)")
    parser.add_argument("--stand-in-floor", type=float, default=None,
                        help="smoke only: override the kit's latency floor (the receipt says so; never qualifies)")
    args = parser.parse_args(argv)
    if args.profile:
        PROFILE["path"] = str(args.profile)
    CONTROL["mode"] = args.control
    manifest = json.loads((args.build / "build-manifest.json").read_text())
    # The store carries its kit (already scaled); older builds fall back to the kit file.
    kit = manifest.get("kit_body") or builder.scaled(rt.load_kit(), manifest["scale"])
    generated = shape.build(kit)
    if generated["input_hash"] != manifest["input_hash"]:
        raise SystemExit("the kit's input hash does not match the build's; rebuild the store")
    env = {**kit["arms"][args.arm], **kit["budgets"][args.budget]}
    args.work = rt.plain_dir(args.work)
    lock = claim(args.work)  # noqa: F841 - held until the process exits
    rt.isolate(args.work / "home", env)
    modules = rt.load(args.tree, kit)
    events = trace_transactions() if args.trace_txn else None
    if args.stand_in_floor is not None:
        aux.configure(floor_s=args.stand_in_floor)
    if args.keep_copy:
        # A killed run leaves its WAL: SQLite recovers it at the next open, as a restart does. The copy must be
        # this build's (the marker restore wrote), never another build's left at the same path.
        db = args.work / "current" / "lcm.db"
        if builder.marker(args.work) != manifest["store_sha256"]:
            raise SystemExit("--keep-copy: the working copy is not this build's store")
    else:
        db = fresh_copy(args.build, args.work)
    from bench.instruments.gate_kit.live_rows import cells
    handler = {"d9": cell_d9, "selftest-cut": cell_selftest_cut, "d10": cells.cell_d10, "d11": cells.cell_d11,
               "timeout-hold": cells.cell_timeout_hold, "turns": cells.cell_turns,
               "kill-at-first-call": cells.cell_kill_at_first_call}[args.cell]
    started = time.time()
    load = cells.Load(args.load) if args.load else None
    try:
        receipt = handler(args, kit, modules, db, copy.deepcopy(generated["restart_list"]))
    except rt.CellError as error:  # one class in `rt`, whichever copy of this module raised it
        receipt = {"cell_error": {"step": error.step, "class": error.error_class}, "exception": error.error_class,
                   "exceptions": [error.error_class], "pass_completed": False}
    finally:
        load_pids = load.stop() if load else []
    receipt["load"] = {"processes": args.load, "pids": load_pids}
    if events is not None:
        receipt["write_windows"] = write_windows(events)
    receipt["keep_copy"] = bool(args.keep_copy)
    receipt["stand_in"] = dict(aux.SETTINGS)
    # Ruling T1-538: every receipt names the latency law and D11's budget mapping as kit fields, and discloses
    # the census stand-in copy the kit's counts were fitted to.
    receipt["kit_fields"] = {"summary_route": kit.get("summary_route"), "budgets": kit.get("budgets"),
                             "budget_used": {args.budget: kit["budgets"].get(args.budget)}}
    receipt["census_disclosure"] = kit.get("census_disclosure")
    receipt["control"] = args.control
    receipt["lineage_path_matches"] = manifest.get("work") == str(args.work.resolve())  # both resolved
    receipt["copy_marker_matches"] = builder.marker(args.work) == manifest["store_sha256"]
    receipt["qualifies"] = (manifest["scale"] == 1.0 and args.stand_in_floor is None and not args.profile
                            and args.control == "none" and not args.trace_txn and not args.keep_copy
                            and receipt["lineage_path_matches"]
                            and receipt["copy_marker_matches"])
    receipt.update({"cell": args.cell, "arm": args.arm, "budget": args.budget, "env": env,
                    "worst_case": bool(getattr(args, "worst_case", False)),
                    "threads": sorted({t.name for t in threading.enumerate()}),
                    "tree_identity": rt.driver.tree_identity(args.tree.resolve()),
                    "input_hash": manifest["input_hash"], "store_sha256": manifest["store_sha256"],
                    "scale": manifest["scale"], "started_epoch": started, "threads_alive": threading.active_count()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=True)
    print(json.dumps({k: receipt.get(k) for k in ("cell", "arm", "budget", "longest_hold_s", "pass_completed",
                                                   "exception", "rows_examined", "list_rows_in", "list_rows_out")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
