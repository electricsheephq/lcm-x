"""L1 check: a session switch that lands while a compaction pass waits for its summary (#2851).

    python -B -m bench.instruments.gate_kit.live_rows.l1_switch --tree TREE --work NEW_DIR --out RECEIPT.json

Session A holds a tool-heavy view (`--plant-copy` adds a second stored copy of its rows, as #2851's reproduction
does; off by default, since it sends A's pass to a no-publish stop). The stand-in summariser holds the pass's first summary
call open; meanwhile the engine is bound to a new, empty session B; then the call is released. B's first row and
then a grown view for B are ingested.

Rows are recorded before the run: A's stored rows (copies included) and B's planned rows. Verdict:
- INCONCLUSIVE when the trace does not show the switch inside the pass (switch start after the call's entry and
  switch end before the call's return);
- PASS when A's stored rows are exactly the recorded ones (same ids, same order, nothing added) and B's rows are
  each stored once, in B's session, in order;
- FAIL otherwise.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from bench.instruments.gate_kit.live_rows import rt  # noqa: E402
from bench.instruments.gate_kit.live_rows.host import auxiliary_client as aux  # noqa: E402
from bench.instruments.gate_kit.live_rows.run import key  # noqa: E402

WINDOW = 32_000


def a_view(turns: int = 40) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number in range(turns):
        stamp = 2_000_000.0 + 40.0 * number
        call = f"call_switch_{number:04d}"
        rows += [
            {"role": "user", "content": f"switch A turn {number} " + "alpha beta " * 120, "timestamp": stamp,
             "message_uid": f"a-{number}-u"},
            {"role": "assistant", "content": f"switch A plan {number}", "timestamp": stamp + 1,
             "message_uid": f"a-{number}-c",
             "tool_calls": [{"id": call, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call, "tool_name": "terminal", "timestamp": stamp + 2,
             "message_uid": f"a-{number}-t", "content": f"switch A result {number} " + "gamma delta " * 150},
            {"role": "assistant", "content": f"switch A reply {number} " + "epsilon " * 60, "timestamp": stamp + 3,
             "message_uid": f"a-{number}-r"},
        ]
    return rows


def b_view(count: int) -> list[dict[str, Any]]:
    """B's first row is an unstamped user row: one of the first-row shapes #2851 lost (a stamped one survived)."""
    rows = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"switch B row {i}",
             "timestamp": 3_000_000.0 + i, "message_uid": f"b-{i}"} for i in range(count)]
    rows[0].pop("timestamp")
    return rows


def session_rows(db: Path, session: str) -> list[tuple[int, str]]:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT store_id, role, content, tool_call_id FROM messages WHERE session_id = ? "
                            "ORDER BY store_id", (session,)).fetchall()
    return [(sid, key(role, content, call)) for sid, role, content, call in rows]


def plant_copy(db: Path, session: str) -> int:
    """Append a second stored copy of the session's rows (every column but the id), in order."""
    with closing(sqlite3.connect(db)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(messages)") if row[1] != "store_id"]
        names = ", ".join(columns)
        with conn:
            cursor = conn.execute(f"INSERT INTO messages ({names}) SELECT {names} FROM messages "
                                  "WHERE session_id = ? ORDER BY store_id", (session,))
        return cursor.rowcount


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hold-timeout", type=float, default=120.0)
    parser.add_argument("--plant-copy", action="store_true",
                        help="diagnostic: a second stored copy of A's rows (#2851's staging); off by default")
    parser.add_argument("--control", choices=("none", "drop-first-b-row", "switch-after-pass"), default="none",
                        help="instrument self-test: a planted loss must FAIL, a late switch must be INCONCLUSIVE")
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=False)
    kit = rt.load_kit()
    rt.isolate(args.work / "home")
    config_module, engine_module, lines = rt.load(args.tree, kit)
    aux.configure(floor_s=0.2, rate_tps=1e9)
    db = args.work / "store" / "lcm.db"
    db.parent.mkdir()
    config = rt.config_for(config_module, db)
    platform = kit["target"]["platform"]
    engine = engine_module.LCMEngine(config=config, hermes_home=str(args.work / "home"))
    engine.on_session_start("switch-A", platform=platform, context_length=WINDOW, conversation_id="switch-A-conv")
    view = a_view()
    engine.ingest(copy.deepcopy(view))
    planted = plant_copy(db, "switch-A") if args.plant_copy else 0
    recorded_a = session_rows(db, "switch-A")
    planned_b = b_view(9)

    entered, release = threading.Event(), threading.Event()
    trace: dict[str, float] = {}

    def hold(record: dict[str, Any]) -> None:
        if not entered.is_set():
            trace["call_enter"] = time.monotonic()
            entered.set()
            release.wait(args.hold_timeout)

    aux.HOOKS["enter"] = hold
    outcome: dict[str, Any] = {}

    def run_pass() -> None:
        trace["pass_begin"] = time.monotonic()
        try:
            engine.compress(copy.deepcopy(view), current_tokens=int(WINDOW * 0.95))
        except Exception as error:  # noqa: BLE001 - class only
            outcome["exception"] = type(error).__name__
        trace["pass_end"] = time.monotonic()

    worker = threading.Thread(target=run_pass, name="live-rows-switch-pass")
    worker.start()
    switched = entered.wait(60)
    if args.control == "switch-after-pass":
        release.set()
        worker.join(args.hold_timeout + 60)
    if switched:
        trace["switch_begin"] = time.monotonic()
        engine.on_session_start("switch-B", platform=platform, context_length=WINDOW, conversation_id="switch-B-conv")
        trace["switch_end"] = time.monotonic()
    release.set()
    worker.join(args.hold_timeout + 60)
    calls = [c for c in aux.CALLS if "t_end" in c]
    trace["call_exit"] = calls[0]["t_end"] if calls else float("nan")
    aux.HOOKS.clear()
    if args.control == "drop-first-b-row":  # the loss #2851 produced, planted by the instrument
        engine.ingest(copy.deepcopy(planned_b[1:]))
    else:
        engine.ingest(copy.deepcopy(planned_b[:1]))
        engine.ingest(copy.deepcopy(planned_b))
    engine.shutdown()

    stored_a, stored_b = session_rows(db, "switch-A"), session_rows(db, "switch-B")
    inside = (switched and trace["call_enter"] <= trace["switch_begin"] and trace["switch_end"] <= trace["call_exit"]
              and trace["pass_begin"] <= trace["call_enter"] and trace["call_exit"] <= trace["pass_end"])
    a_ok = stored_a == recorded_a
    b_ok = [k for _sid, k in stored_b] == [key(r["role"], r["content"], r.get("tool_call_id")) for r in planned_b]
    verdict = "INCONCLUSIVE" if not inside else ("PASS" if a_ok and b_ok else "FAIL")
    receipt = {
        "verdict": verdict, "switch_inside_pass": inside, "control": args.control,
        "trace_s": {k: round(v - trace.get("pass_begin", v), 3) for k, v in trace.items() if v == v},
        "a_recorded_rows": len(recorded_a), "a_planted_copies": planted, "a_stored_rows": len(stored_a),
        "a_unchanged": a_ok, "b_planned_rows": len(planned_b), "b_stored_rows": len(stored_b), "b_in_order_once": b_ok,
        "pass_exception": outcome.get("exception"), "summary_calls": len(aux.CALLS),
        "lines": [line["message"][:240] for line in lines.lines if line["message"].startswith("LCM compaction stop")],
        "tree_identity": rt.driver.tree_identity(args.tree.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=True)
    print(json.dumps({k: receipt[k] for k in ("verdict", "switch_inside_pass", "a_unchanged", "b_in_order_once",
                                               "a_stored_rows", "a_recorded_rows", "b_stored_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
