"""The multi-pass and contention cells: D10, D11 (with D12's passes), and #2868's recorded variants.

Each runs in its own process through run.py. Receipts carry counts, ids, hashes and seconds; never content.
"""

from __future__ import annotations

import copy
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from bench.instruments.gate_kit.live_rows import lockwatch, rt
from bench.instruments.gate_kit.live_rows import run as base
from bench.instruments.gate_kit.live_rows.host import auxiliary_client as aux


def summary_nodes(db: Path) -> int:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        return int(conn.execute("SELECT COALESCE(MAX(node_id), 0) FROM summary_nodes").fetchone()[0])


def stored_identity(db: Path, conversation: str) -> list[tuple[int, str]]:
    return [(sid, base.key(role, content, call)) for sid, _s, role, content, call in rt.stored_rows(db, conversation)]


def compare_stored(before: list[tuple[int, str]], after: list[tuple[int, str]]) -> dict[str, Any]:
    """By identity and order, not by count: every earlier row keeps its id, value and place; nothing is added."""
    old, new = dict(before), dict(after)
    lost = [sid for sid in old if sid not in new]
    changed = [sid for sid in old if sid in new and new[sid] != old[sid]]
    added = [sid for sid in new if sid not in old]
    order_old = [sid for sid, _ in before if sid in new]
    order_new = [sid for sid, _ in after if sid in old]
    return {"lost": len(lost), "changed": len(changed), "duplicated": len(added),
            "reordered": order_old != order_new, "lost_ids": lost[:20], "added_ids": added[:20]}


class Holds:
    """Counts hold arming through the engine's own hook (present in v0.30.0 and later) and its log line."""

    def __init__(self, engine: Any) -> None:
        self.count = 0
        original = getattr(engine, "_start_sweep_budget_hold", None)
        self.hooked = callable(original)
        if self.hooked:
            def counted(*args: Any, **kwargs: Any) -> Any:
                self.count += 1
                return original(*args, **kwargs)
            engine._start_sweep_budget_hold = counted


def one_pass(engine: Any, db: Path, kit: dict[str, Any], view: list[dict[str, Any]], holds: Holds,
             lines: Any, conversation: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    snapshot = copy.deepcopy(view)  # by value, before the call (D12)
    stored_before, nodes_before, held_before = stored_identity(db, conversation), summary_nodes(db), holds.count
    mark, calls_before = len(lines.lines), len(aux.CALLS)
    began = time.monotonic()
    record: dict[str, Any] = {"rows_in": len(view)}
    try:
        out = engine.compress(copy.deepcopy(view), current_tokens=base.pressure(kit, view))
    except Exception as error:  # noqa: BLE001 - class only
        record["exception"] = type(error).__name__
        out = view
    record["seconds"] = round(time.monotonic() - began, 3)
    out = out if isinstance(out, list) else view
    new_lines = [line["message"] for line in lines.lines[mark:]]
    record.update({
        "rows_out": len(out), "published": summary_nodes(db) > nodes_before,
        "hold_armed": holds.count > held_before or any(m.startswith("LCM automatic compaction held")
                                                       for m in new_lines),
        "stored_check": compare_stored(stored_before, stored_identity(db, conversation)),
        "returned_equals_input": out == snapshot,
        "summary_calls": [{k: c.get(k) for k in ("input_tokens", "granted_s", "deadline_left", "slept_s", "outcome")}
                          for c in aux.CALLS[calls_before:]],
        "stop_lines": [m[:240] for m in new_lines if m.startswith(("LCM compaction stop", "LCM automatic"))],
    })
    return out, record


def cell_d11(args: Any, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
             view: list[dict[str, Any]]) -> dict[str, Any]:
    """D11's eight passes at this run's budget, then D12's further passes (at most three) until one publishes none."""
    config_module, engine_module, lines = modules
    config = rt.config_for(config_module, db)
    conversation = kit["target"]["conversation_id"]
    engine = rt.engine(engine_module, config, kit, args.work / "home")
    holds = Holds(engine)
    passes, extra = [], []
    rows_before = len(view)
    for _ in range(kit["bars"]["d11_passes"]):
        view, record = one_pass(engine, db, kit, view, holds, lines, conversation)
        passes.append(record)
    for _ in range(kit["bars"]["d12_extra_passes"]):
        view, record = one_pass(engine, db, kit, view, holds, lines, conversation)
        extra.append(record)
        if not record["published"]:
            break
    engine.shutdown()
    return {"effective": rt.effective(config), "rows_before": rows_before, "passes": passes, "extra_passes": extra,
            "hold_hook": holds.hooked, "rows_examined": base.rows_examined(db, conversation, view)}


def cell_timeout_hold(args: Any, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
                      view: list[dict[str, Any]]) -> dict[str, Any]:
    """Requirement 5, recorded: every call of the first pass is cut; does that arm a hold that refuses the next?"""
    config_module, engine_module, lines = modules
    config = rt.config_for(config_module, db)
    conversation = kit["target"]["conversation_id"]
    engine = rt.engine(engine_module, config, kit, args.work / "home")
    holds = Holds(engine)
    aux.configure(force_cut=True)
    view, first = one_pass(engine, db, kit, view, holds, lines, conversation)
    aux.configure(force_cut=False)
    view, second = one_pass(engine, db, kit, view, holds, lines, conversation)
    engine.shutdown()
    return {"effective": rt.effective(config), "passes": [first, second], "hold_hook": holds.hooked,
            "first_published": first["published"], "first_armed_hold": first["hold_armed"],
            "second_attempted_a_call": bool(second["summary_calls"]), "second_published": second["published"]}


def host_turn(engine: Any, pre: list[dict[str, Any]], post: list[dict[str, Any]], kit: dict[str, Any]) -> dict[str, Any]:
    """One host turn in the host's call order; the seconds before the model call are LCM-X's (requirement 4)."""
    tokens = base.pressure(kit, pre)
    began = time.monotonic()
    compressed = None
    if engine.should_compress(tokens):
        engine.should_compress_preflight(pre)
        compressed = engine.compress(copy.deepcopy(pre), current_tokens=tokens)
    engine.should_compress(tokens)
    before_model = time.monotonic() - began
    usage = {"prompt_tokens": tokens, "completion_tokens": 64, "total_tokens": tokens + 64}
    engine.update_from_response(dict(usage))
    engine.ingest(copy.deepcopy(post))
    if hasattr(engine, "on_turn_complete"):
        try:
            engine.on_turn_complete(copy.deepcopy(post), usage=dict(usage), turn_id="t", task_id="t",
                                    api_call_count=1, interrupted=False, failed=False, turn_exit_reason="completed")
        except TypeError:
            engine.on_turn_complete(copy.deepcopy(post))
    return {"before_model_s": round(before_model, 3), "compressed": compressed is not None,
            "after_model_s": round(time.monotonic() - began - before_model, 3), "rows": len(pre)}


def cell_turns(args: Any, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
               view: list[dict[str, Any]]) -> dict[str, Any]:
    """Requirement 4, recorded: LCM-X seconds before the model call per turn, before and after the first publish."""
    config_module, engine_module, lines = modules
    config = rt.config_for(config_module, db)
    engine = rt.engine(engine_module, config, kit, args.work / "home")
    turns, published_at = [], None
    nodes = summary_nodes(db)
    for number in range(int(kit["bars"].get("turn_count", 6))):
        user = {"role": "user", "content": f"live-rows turn {number} user", "timestamp": time.time()}
        reply = {"role": "assistant", "content": f"live-rows turn {number} reply", "timestamp": time.time()}
        pre, post = view + [user], view + [user, reply]
        record = host_turn(engine, pre, post, kit)
        record["after_first_publish"] = published_at is not None
        turns.append(record)
        if published_at is None and summary_nodes(db) > nodes:
            published_at = number
        view = post
    engine.shutdown()
    return {"effective": rt.effective(config), "turns": turns, "first_publish_turn": published_at}


def busy_thread(stop: threading.Event, work: Any, records: list[dict[str, Any]], pause_s: float) -> None:
    while not stop.is_set():
        began = time.monotonic_ns()
        error = None
        try:
            work()
        except Exception as exc:  # noqa: BLE001 - class only
            error = type(exc).__name__
        ended = time.monotonic_ns()
        records.append({"begin_ns": began, "end_ns": ended, "seconds": round((ended - began) / 1e9, 4), "error": error})
        if pause_s:
            stop.wait(pause_s)


def cell_d10(args: Any, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
             view: list[dict[str, Any]]) -> dict[str, Any]:
    """D9's restart and first pass while a turn-path thread and a compaction thread work on other sessions of the
    same store. Each starts its next call within the kit's pause (under 1 s) of the previous return; the
    worst-case arm (`--worst-case`) never sleeps and is capped in wall time."""
    config_module, engine_module, lines = modules
    config = rt.config_for(config_module, db)
    home = args.work / "home"
    generated = base.shape.build(kit)
    turn_view = copy.deepcopy(generated["others"][0])
    compact_view = copy.deepcopy(generated["others"][1 % len(generated["others"])])
    turn_engine = rt.engine(engine_module, config, kit, home, "other0-s", "other0-conv")
    compact_engine = rt.engine(engine_module, config, kit, home, "other1-s", "other1-conv")
    pause = 0.0 if args.worst_case else float(kit["bars"].get("d10_pause_s", 0.5))
    counter = {"n": 0}

    def turn_call() -> None:  # the turn path's calls, in the kit's order: preflight, then a store write
        counter["n"] += 1
        row = {"role": "user", "content": f"live-rows busy turn {counter['n']}", "timestamp": time.time()}
        turn_view.append(row)
        turn_engine.should_compress_preflight(copy.deepcopy(turn_view))
        turn_engine.ingest(copy.deepcopy(turn_view))

    def compaction_call() -> None:
        compact_engine.compress(copy.deepcopy(compact_view), current_tokens=1)

    stop = threading.Event()
    turn_records: list[dict[str, Any]] = []
    compaction_records: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    reader = rt.open_reader(db)
    watcher = lockwatch.Watcher(db)
    threads = [threading.Thread(target=busy_thread, args=(stop, turn_call, turn_records, pause), daemon=True),
               threading.Thread(target=busy_thread, args=(stop, compaction_call, compaction_records, pause),
                                daemon=True)]
    begin = time.monotonic_ns()
    for thread in threads:
        thread.start()
    cap = float(kit["bars"].get("d10_worst_cap_s", 1500))
    timer = threading.Timer(cap, stop.set) if args.worst_case else None
    if timer:
        timer.start()
    engine, result = base.first_pass(engine_module, config, kit, home, db, view, calls)
    stop.set()
    for thread in threads:
        thread.join(timeout=120)
    if timer:
        timer.cancel()
    end = time.monotonic_ns()
    observed = watcher.stop()
    reader.close()
    for item in (engine, turn_engine, compact_engine):
        try:
            item.shutdown()
        except Exception:  # noqa: BLE001
            pass
    compress = next(c for c in calls if c["call"] == "compress")
    windows = lockwatch.holds(observed["transitions"], compress["begin_ns"], compress["end_ns"])
    hold = max(windows, key=lambda w: w[1] - w[0], default=(compress["begin_ns"], compress["begin_ns"]))

    def during(record: dict[str, Any]) -> bool:
        return hold[0] <= record["begin_ns"] <= hold[1]

    for record in compaction_records:
        record["during_hold"] = during(record)
        record["return_s"] = record["seconds"]
    for record in turn_records:
        record["wait_s"] = record["seconds"]
    exceptions = [r["error"] for r in turn_records + compaction_records if r["error"]]
    if compress.get("exception"):
        exceptions.append(compress["exception"])
    return {
        "effective": rt.effective(config), "worst_case": bool(args.worst_case), "pause_s": pause, "calls": calls,
        "lock_watch": {k: observed.get(k) for k in ("samples", "poll_ns", "largest_gap_ms", "unexpected_answers")},
        "longest_hold_in_pass_s": (hold[1] - hold[0]) / 1e9, "longest_hold_s": lockwatch.longest_s(
            observed["transitions"], begin, end),
        "pass_completed": result is not None and "exception" not in compress, "exceptions": exceptions,
        "rows_examined": engine._live_rows_cursor["max_cursor"],
        "rows_stored_matching_list": base.rows_examined(db, kit["target"]["conversation_id"], view),
        "started_during_hold": {"turn": sum(map(during, turn_records)),
                                "compaction": sum(map(during, compaction_records))},
        "turn_calls": [{k: r[k] for k in ("seconds", "wait_s", "error")} | {"during_hold": during(r)}
                       for r in turn_records],
        "compaction_calls": [{k: r[k] for k in ("seconds", "return_s", "during_hold", "error")}
                             for r in compaction_records],
        "summary_calls": [{k: c.get(k) for k in ("input_tokens", "granted_s", "deadline_left", "slept_s", "outcome")}
                          for c in aux.CALLS],
    }


def cell_kill_at_first_call(args: Any, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
                            view: list[dict[str, Any]]) -> dict[str, Any]:
    """Requirement 3, first half: restart, begin the first pass, and stop the process at its first summary call."""
    def kill(_record: dict[str, Any]) -> None:
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    aux.HOOKS["enter"] = kill
    config_module, engine_module, _lines = modules
    config = rt.config_for(config_module, db)
    engine = rt.engine(engine_module, config, kit, args.work / "home")
    engine.compress(copy.deepcopy(view), current_tokens=base.pressure(kit, view))
    return {"killed": False, "note": "the pass ended before any summary call; nothing was stopped"}


class Load:
    """Requirement 2: CPU-bound processes loading the machine for the cell's duration, stopped by exact PID."""

    def __init__(self, count: int) -> None:
        self.processes = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(count)]

    def stop(self) -> list[int]:
        pids = [p.pid for p in self.processes]
        for process in self.processes:
            process.kill()
            process.wait(timeout=30)
        return pids
