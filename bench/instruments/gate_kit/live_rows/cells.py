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


def start(engine_module: Any, config: Any, kit: dict[str, Any], home: Path, *names: str) -> Any:
    """An engine start inside a cell; a raise becomes `rt.CellError`, which run.main turns into a receipt."""
    try:
        return rt.engine(engine_module, config, kit, home, *names)
    except Exception as error:  # noqa: BLE001
        raise rt.CellError("engine start", error) from error


def summary_nodes(db: Path) -> int:
    with closing(sqlite3.connect(rt.ro_uri(db), uri=True)) as conn:
        return int(conn.execute("SELECT COALESCE(MAX(node_id), 0) FROM summary_nodes").fetchone()[0])


def stored_identity(db: Path, conversation: str | None = None) -> list[tuple[int, Any]]:
    """The whole store's rows (ids are immutable keys; the value digest covers owner, session and content)."""
    return rt.stored_identity(db)


def compare_stored(before: list[tuple[int, Any]], after: list[tuple[int, Any]]) -> dict[str, Any]:
    """By identity and order across the whole store, not by count: every earlier row keeps its id, owner and value
    (a changed value is a lost row), earlier values keep their order, and nothing is added (an added row whose
    value an earlier row holds is a copy)."""
    old, new = dict(before), dict(after)
    lost = [sid for sid in old if sid not in new]
    changed = [sid for sid in old if sid in new and new[sid] != old[sid]]
    added = [sid for sid in new if sid not in old]
    earlier_values = set(old.values())
    copies = [sid for sid in added if new[sid] in earlier_values]
    kept = [sid for sid, _ in before if sid in new]
    reordered = bool(changed) and sorted(old[sid] for sid in changed) == sorted(new[sid] for sid in changed)
    return {"lost": len(lost) + len(changed), "removed": len(lost), "changed": len(changed),
            "duplicated": len(added), "added_copies": len(copies), "reordered": reordered,
            "kept_in_order": [sid for sid, _ in after if sid in old] == kept,
            "lost_ids": (lost + changed)[:20], "added_ids": added[:20]}


SHORT_STOPS_DEFAULT = ("error", "time_budget_exhausted", "pass_budget_exhausted", "leaf_summary_error",
                       "summary_route_unavailable", "summary_result_rejected")


def stop_reasons(messages: list[str]) -> list[str]:
    found = []
    for message in messages:
        if message.startswith("LCM compaction stop: reason="):
            found.append(message.split("reason=", 1)[1].split()[0])
        elif message.startswith("LCM compress failed"):
            found.append("error")
    return found


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
    out: Any = None
    try:
        out = engine.compress(copy.deepcopy(view), current_tokens=base.pressure(kit, view))
    except Exception as error:  # noqa: BLE001 - class only
        record["exception"] = type(error).__name__
    record["seconds"] = round(time.monotonic() - began, 3)
    # The host receives a list only when compress returns one; otherwise the pass is never scored as unchanged,
    # and the next pass is given the host's previous list.
    record["returned_list"] = isinstance(out, list)
    new_lines = [line["message"] for line in lines.lines[mark:]]
    record["stop_reasons"] = stop_reasons(new_lines)
    returned = out if isinstance(out, list) else None
    out = returned if returned is not None else view
    record.update({
        "rows_out": len(out) if returned is not None else None, "published": summary_nodes(db) > nodes_before,
        "hold_armed": holds.count > held_before or any(m.startswith("LCM automatic compaction held")
                                                       for m in new_lines),
        "stored_check": compare_stored(stored_before, stored_identity(db, conversation)),
        "returned_equals_input": returned is not None and returned == snapshot,
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
    engine = start(engine_module, config, kit, args.work / "home")
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
    engine = start(engine_module, config, kit, args.work / "home")
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
    engine.should_compress_preflight(pre)  # the host calls preflight every turn
    if engine.should_compress(tokens):
        compressed = engine.compress(copy.deepcopy(pre), current_tokens=tokens)
    if isinstance(compressed, list):  # the host adopts the compressed list; the model's reply follows it
        post = compressed + post[len(pre):]
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
    note = getattr(engine, "note_turn_complete", None)
    if callable(note):  # the host's post-LLM-call hook ends turn-paced holds through this notification
        note()
    return {"returned": compressed if isinstance(compressed, list) else None,
            "before_model_s": round(before_model, 3), "compressed": compressed is not None,
            "after_model_s": round(time.monotonic() - began - before_model, 3), "rows": len(pre)}


def cell_turns(args: Any, kit: dict[str, Any], modules: tuple[Any, Any, Any], db: Path,
               view: list[dict[str, Any]]) -> dict[str, Any]:
    """Requirement 4, recorded: LCM-X seconds before the model call per turn, before and after the first publish."""
    config_module, engine_module, lines = modules
    config = rt.config_for(config_module, db)
    engine = start(engine_module, config, kit, args.work / "home")
    turns, published_at = [], None
    nodes = summary_nodes(db)
    for number in range(int(kit["bars"].get("turn_count", 6))):
        user = {"role": "user", "content": f"live-rows turn {number} user", "timestamp": time.time()}
        reply = {"role": "assistant", "content": f"live-rows turn {number} reply", "timestamp": time.time()}
        pre, post = view + [user], view + [user, reply]
        record = host_turn(engine, pre, post, kit)
        if record.get("returned") is not None:  # the host adopts a compressed list
            post = record.pop("returned") + [reply]
        record.pop("returned", None)
        record["after_first_publish"] = published_at is not None
        turns.append(record)
        if published_at is None and summary_nodes(db) > nodes:
            published_at = number
        view = post
    engine.shutdown()
    return {"effective": rt.effective(config), "turns": turns, "first_publish_turn": published_at}


def busy_thread(stop: threading.Event, works: list[tuple[str, Any]], records: list[dict[str, Any]],
                pause_s: float) -> None:
    """The kit's calls in order, each its own record, registered before it starts (an unfinished call is seen)."""
    while not stop.is_set():
        for name, work in works:
            if stop.is_set():
                break
            record: dict[str, Any] = {"call": name, "begin_ns": time.monotonic_ns(), "end_ns": None, "error": None}
            records.append(record)
            try:
                work()
            except Exception as exc:  # noqa: BLE001 - class only
                record["error"] = type(exc).__name__
            record["end_ns"] = time.monotonic_ns()
            if pause_s:
                stop.wait(pause_s)


def finish_records(records: list[dict[str, Any]], at_ns: int) -> None:
    """Seconds per call, unrounded; a call still running when the cell ended runs to `at_ns` and is marked."""
    for record in records:
        record["unfinished"] = record["end_ns"] is None
        record["seconds"] = ((record["end_ns"] or at_ns) - record["begin_ns"]) / 1e9


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
    turn_engine = start(engine_module, config, kit, home, "other0-s", "other0-conv")
    compact_engine = start(engine_module, config, kit, home, "other1-s", "other1-conv")
    pause = 0.0 if args.worst_case else float(kit["bars"].get("d10_pause_s", 0.5))
    counter = {"n": 0}

    def turn_preflight() -> None:  # the turn path's calls, in the kit's order: preflight, then a store write
        counter["n"] += 1
        turn_view.append({"role": "user", "content": f"live-rows busy turn {counter['n']}", "timestamp": time.time()})
        turn_engine.should_compress_preflight(copy.deepcopy(turn_view))

    def turn_ingest() -> None:
        turn_engine.ingest(copy.deepcopy(turn_view))

    def compaction_call() -> None:
        compact_engine.compress(copy.deepcopy(compact_view), current_tokens=1)

    stop = threading.Event()
    turn_records: list[dict[str, Any]] = []
    compaction_records: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    reader = rt.open_reader(db)
    watcher = lockwatch.Watcher(db)
    threads = [threading.Thread(target=busy_thread, args=(stop, [("preflight", turn_preflight), ("ingest", turn_ingest)],
                                                          turn_records, pause), daemon=True),
               threading.Thread(target=busy_thread, args=(stop, [("compress", compaction_call)], compaction_records,
                                                          pause), daemon=True)]
    begin = time.monotonic_ns()
    for thread in threads:
        thread.start()
    cap = float(kit["bars"].get("d10_worst_cap_s", 1500))
    timer = threading.Timer(cap, stop.set) if args.worst_case else None
    if timer:
        timer.start()
    try:
        engine, result = base.first_pass(engine_module, config, kit, home, db, view, calls)
    except rt.CellError:  # the restart raised: stop the threads, the timer and the watcher, then report it
        stop.set()
        if timer:
            timer.cancel()
        for thread in threads:
            thread.join(timeout=120)
        watcher.stop()
        reader.close()
        raise
    stop.set()
    for thread in threads:
        thread.join(timeout=120)
    if timer:
        timer.cancel()
    end = time.monotonic_ns()
    finish_records(turn_records, end)
    finish_records(compaction_records, end)
    observed = watcher.stop()
    reader.close()
    for item in (engine, turn_engine, compact_engine):
        try:
            item.shutdown()
        except Exception:  # noqa: BLE001
            pass
    compress = next(c for c in calls if c["call"] == "compress")
    # The hold window spans the restart and the pass, as D9's does (a cold start may hold in either).
    restart = next(c for c in calls if c["call"] == "restart")
    windows = lockwatch.holds(observed["transitions"], restart["begin_ns"], compress["end_ns"])
    hold = max(windows, key=lambda w: w[1] - w[0], default=(compress["begin_ns"], compress["begin_ns"]))

    def during(record: dict[str, Any]) -> bool:  # any write-lock hold inside the restart and pass
        return any(start <= record["begin_ns"] <= stop for start, stop in windows)

    for record in compaction_records:
        record["during_hold"] = during(record)
        record["return_s"] = record["seconds"]
    for record in turn_records:
        record["wait_s"] = record["seconds"]
    exceptions = [r["error"] for r in turn_records + compaction_records if r["error"]]
    if compress.get("exception"):
        exceptions.append(compress["exception"])
    reasons = stop_reasons([line["message"] for line in lines.lines if line["thread"] == threading.main_thread().ident])
    if "error" in reasons:
        exceptions.append("compress failed inside the product (reason=error)")
    # The busy compaction thread's own swallowed errors (a returned fallback list is still an error: bar 4).
    busy_errors = [line for line in lines.lines if line["thread"] != threading.main_thread().ident
                   and line["t"] * 1e9 <= end and "error" in stop_reasons([line["message"]])]
    if busy_errors:
        exceptions.append(f"{len(busy_errors)} busy-thread compress failures inside the product (reason=error)")
    return {
        "effective": rt.effective(config), "worst_case": bool(args.worst_case), "pause_s": pause, "calls": calls,
        "lock_watch": {k: observed.get(k) for k in ("samples", "poll_ns", "largest_gap_ms", "unexpected_answers")},
        "longest_hold_in_pass_s": (hold[1] - hold[0]) / 1e9, "longest_hold_s": lockwatch.longest_s(
            observed["transitions"], begin, end),
        "pass_completed": base.completed(result, compress, reasons, kit), "stop_reasons": reasons,
        "exceptions": exceptions,
        "rows_examined": engine._live_rows_cursor_pass["max_cursor"],
        "rows_stored_matching_list": base.rows_examined(db, kit["target"]["conversation_id"], view),
        "started_during_hold": {"turn": sum(map(during, turn_records)),
                                "compaction": sum(map(during, compaction_records))},
        "turn_calls": [{k: r[k] for k in ("call", "seconds", "wait_s", "error", "unfinished")}
                       | {"during_hold": during(r)} for r in turn_records],
        "compaction_calls": [{k: r[k] for k in ("call", "seconds", "return_s", "during_hold", "error", "unfinished")}
                             for r in compaction_records],
        "lock_unexpected": observed.get("unexpected_answers"),
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
    engine = start(engine_module, config, kit, args.work / "home")
    engine.compress(copy.deepcopy(view), current_tokens=base.pressure(kit, view))
    return {"killed": False, "note": "the pass ended before any summary call; nothing was stopped"}


class Load:
    """Requirement 2: CPU-bound processes loading the machine for the cell's duration, stopped by exact PID."""

    def __init__(self, count: int) -> None:
        self.processes: list[subprocess.Popen] = []
        try:
            for _ in range(count):
                self.processes.append(subprocess.Popen([sys.executable, "-c", "while True: pass"]))
        except Exception:
            self.stop()  # never leave a started load process behind
            raise

    def stop(self) -> list[int]:
        pids = [p.pid for p in self.processes]
        for process in self.processes:
            process.kill()
            process.wait(timeout=30)
        return pids
