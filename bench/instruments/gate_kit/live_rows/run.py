"""Run one live-rows cell on one tree, in its own process, and write its receipt.

    python -B -m bench.instruments.gate_kit.live_rows.run --tree TREE --build BUILD_DIR --cell d9 \
        --arm uid_on --budget default --work WORK_DIR --out RECEIPT.json

The store is copied from BUILD_DIR into WORK_DIR/current (one working copy, overwritten by each run). The receipt
holds counts, ids, hashes, seconds and the product's fixed log lines; never message content.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
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

CELLS = ("d9", "d10", "d11", "selftest-cut")


def fresh_copy(build: Path, work: Path) -> Path:
    """Overwrite WORK/current with the build's store; stale WAL and shared-memory files are removed first."""
    current = work / "current"
    current.mkdir(parents=True, exist_ok=True)
    for name in ("lcm.db-wal", "lcm.db-shm"):
        (current / name).unlink(missing_ok=True)
    shutil.copyfile(build / "store" / "lcm.db", current / "lcm.db")
    source = build / "store" / "externalized"
    if source.exists():
        shutil.copytree(source, current / "externalized", dirs_exist_ok=True)
    return current / "lcm.db"


def digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps([row.get("role"), row.get("content")], sort_keys=True).encode()).hexdigest()


def rows_examined(db: Path, conversation: str, view: list[dict[str, Any]]) -> int:
    """Host-list rows whose role and content are stored in the target conversation (multiset match)."""
    stored = Counter(hashlib.sha256(json.dumps([role, content], sort_keys=True).encode()).hexdigest()
                     for _sid, _session, role, content in rt.stored_rows(db, conversation))
    found = 0
    for row in view:
        key = digest(row)
        if stored[key] > 0:
            stored[key] -= 1
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
        entry["frames"] = [f"{Path(f.filename).name}:{f.lineno}:{f.name}"
                           for f in traceback.extract_tb(error.__traceback__)][-12:]
        raise
    finally:
        entry["end_ns"] = time.monotonic_ns()
        entry["seconds"] = round((entry["end_ns"] - entry["begin_ns"]) / 1e9, 3)


def first_pass(engine_module: Any, config: Any, kit: dict[str, Any], home: Path, db: Path,
               view: list[dict[str, Any]], calls: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]] | None]:
    """Restart (a new engine and its session start), then the first compaction pass on the host list."""
    engine = timed(calls, "restart", lambda: rt.engine(engine_module, config, kit, home))
    result = None
    try:
        result = timed(calls, "compress", lambda: engine.compress(copy.deepcopy(view),
                                                                  current_tokens=pressure(kit, view)))
    except Exception:  # noqa: BLE001 - recorded in `calls`
        pass
    return engine, result


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
    return {
        "effective": rt.effective(config),
        "calls": calls,
        "longest_hold_s": lockwatch.longest_s(observed["transitions"], begin, end, pid),
        "longest_hold_in_pass_s": lockwatch.longest_s(observed["transitions"], compress["begin_ns"],
                                                       compress["end_ns"], pid),
        "holds_over_1s": [round((e - s) / 1e9, 3) for s, e in lockwatch.holds(observed["transitions"], begin, end)
                          if e - s > 1e9],
        "lock_watch": {k: observed[k] for k in ("samples", "poll_ns", "largest_gap_ms")},
        "pass_completed": result is not None and "exception" not in compress,
        "exception": compress.get("exception"),
        "list_rows_in": len(view), "list_rows_out": len(result) if isinstance(result, list) else None,
        "rows_examined": rows_examined(db, kit["target"]["conversation_id"], view),
        "summary_calls": [{k: c.get(k) for k in ("input_tokens", "latency_s", "granted_s", "slept_s", "outcome")}
                          for c in aux.CALLS],
        "lines": [{"t_s": round(line["t"] - begin / 1e9, 3), "message": line["message"]} for line in lines.lines],
    }


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
    args = parser.parse_args(argv)
    manifest = json.loads((args.build / "build-manifest.json").read_text())
    kit = builder.scaled(rt.load_kit(), manifest["scale"])
    generated = shape.build(kit)
    if generated["input_hash"] != manifest["input_hash"]:
        raise SystemExit("the kit's input hash does not match the build's; rebuild the store")
    env = {**kit["arms"][args.arm], **kit["budgets"][args.budget]}
    rt.isolate(args.work / "home", env)
    modules = rt.load(args.tree, kit)
    db = fresh_copy(args.build, args.work)
    handler = {"d9": cell_d9, "selftest-cut": cell_selftest_cut}[args.cell]
    started = time.time()
    receipt = handler(args, kit, modules, db, copy.deepcopy(generated["target_list"]))
    receipt.update({"cell": args.cell, "arm": args.arm, "budget": args.budget, "env": env,
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
