"""L1 check: the ledger's ordinal planner plans and applies one insertion into a packed 20,000-label session.

    python -B -m bench.instruments.gate_kit.live_rows.l1_planner --tree TREE --work NEW_DIR --out RECEIPT.json \
        [--max-windows N] [--max-statements M]

The session is ingested through the tree's public ingest. Its lineage is then packed to the #2840 reporter shape
(20,014 labels; a run 5,458 wide packed below the minimum gap, with a gap of 1 at its middle) by rewriting the
ordinals only. After a restart, one host row is inserted at the middle of the packed run and the view is ingested.

Counted, never timed (#2840):
- candidate windows examined: each `step >= MIN_GAP` test inside `plan_ordinals`. The module's MIN_GAP is swapped
  for an equal int whose reflected comparison counts, so trees with and without the bounded search are counted the
  same way;
- statements executed: every SQL statement this process's connections run during the insertion's ingest.
Past `--max-windows`, the counter raises and the planner stops: the unbounded v0.30.0 search cannot stall the run.
Then the insertion must be committed and read back in order: the lineage's stored rows, by ordinal, equal the view.
"""

from __future__ import annotations

import argparse
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

LABELS, RUN = 20014, 5458
DEFAULT_ABORT = 5_000_000


class WindowBudgetExceeded(RuntimeError):
    pass


def reporter_labels(step: int, min_gap: int) -> tuple[list[int], int]:
    """The reporter shape of tests/test_issue_2840_plan_ordinals_respace.py; returns labels and the insert index."""
    labels = [step * (i + 1) for i in range(4)]
    for i in range(RUN):
        labels.append(labels[-1] + (1 if i == RUN // 2 else min_gap - 1))
    labels += [labels[-1] + step * (i + 1) for i in range(LABELS - len(labels))]
    return labels, 4 + RUN // 2


def counting_gap(value: int, counter: dict[str, int], limit: int) -> int:
    class Gap(int):
        def __le__(self, other: Any) -> bool:  # `step >= MIN_GAP` runs this reflected method first
            if counter["active"]:
                counter["windows"] += 1
                if counter["windows"] > limit:
                    counter["aborted"] = True
                    raise WindowBudgetExceeded(f"candidate windows passed {limit}")
            return int(self) <= other

        def __ge__(self, other: Any) -> bool:
            return int(self) >= other

    return Gap(value)


def view_rows(count: int) -> list[dict[str, Any]]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"planner fixture row {i}",
             "timestamp": 1_000_000.0 + 10.0 * i} for i in range(count)]


def lineage_rows(db: Path, session: str) -> list[tuple[int, int, str]]:
    with closing(sqlite3.connect(rt.ro_uri(db), uri=True)) as conn:
        return conn.execute(
            "SELECT b.ordinal, b.store_id, m.content FROM occurrence_bindings b JOIN messages m "
            "ON m.store_id = b.store_id WHERE m.session_id = ? AND b.ordinal IS NOT NULL "
            "AND b.disposition IN ('stored','migrated','ambiguous') ORDER BY b.ordinal", (session,)).fetchall()


def pack(db: Path, session: str, labels: list[int]) -> str:
    """Rewrite the session's ordinals (in their current order) to `labels`; two phases keep the unique index."""
    with closing(sqlite3.connect(db)) as conn:
        rows = conn.execute(
            "SELECT b.occ_id, b.lineage FROM occurrence_bindings b JOIN messages m ON m.store_id = b.store_id "
            "WHERE m.session_id = ? AND b.ordinal IS NOT NULL ORDER BY b.ordinal", (session,)).fetchall()
        lineages = {lineage for _occ, lineage in rows}
        if len(rows) != len(labels) or len(lineages) != 1:
            raise RuntimeError(f"fixture has {len(rows)} labels in {len(lineages)} lineages, expected "
                               f"{len(labels)} in one")
        with conn:
            conn.executemany("UPDATE occurrence_bindings SET ordinal = ? WHERE occ_id = ?",
                             [(-occ, occ) for occ, _ in rows])
            conn.executemany("UPDATE occurrence_bindings SET ordinal = ? WHERE occ_id = ?",
                             [(label, occ) for (occ, _), label in zip(rows, labels)])
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return lineages.pop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-windows", type=int, default=None, help="frozen kit count (judged)")
    parser.add_argument("--max-statements", type=int, default=None, help="frozen kit count (judged)")
    args = parser.parse_args(argv)
    args.work.mkdir(parents=True, exist_ok=False)
    kit = rt.load_kit()
    rt.isolate(args.work / "home")
    config_module, engine_module, lines = rt.load(args.tree, kit)
    ledger = sys.modules["hermes_lcm.ingest_ledger"] if "hermes_lcm.ingest_ledger" in sys.modules else \
        __import__("hermes_lcm.ingest_ledger", fromlist=["x"])
    db = args.work / "store" / "lcm.db"
    db.parent.mkdir()
    config = rt.config_for(config_module, db)
    session, conversation = "planner-fixture", "planner-fixture-conv"
    receipt: dict[str, Any] = {"labels": LABELS, "run": RUN, "tree_identity": rt.driver.tree_identity(args.tree.resolve()),
                               "max_windows": args.max_windows, "max_statements": args.max_statements}
    view = view_rows(LABELS)
    started = time.monotonic()
    engine = rt.engine(engine_module, config, kit, args.work / "home", session, conversation)
    engine.ingest(view)
    engine.shutdown()
    receipt["fixture_ingest_s"] = round(time.monotonic() - started, 2)
    labels, middle = reporter_labels(ledger.ORDINAL_STEP, ledger.MIN_GAP)
    receipt["lineage"] = pack(db, session, labels)
    inserted = {"role": "user", "content": "planner fixture inserted row", "timestamp": 1_000_000.0 + 10.0 * middle - 5}
    new_view = view[:middle] + [inserted] + view[middle:]

    counter = {"windows": 0, "active": False, "aborted": False, "step_calls": 0}
    limit = args.max_windows if args.max_windows is not None else DEFAULT_ABORT
    original_gap, original_plan = ledger.MIN_GAP, ledger.plan_ordinals
    ledger.MIN_GAP = counting_gap(original_gap, counter, limit)
    plans: list[dict[str, Any]] = []

    def counted_plan(*a: Any, **k: Any) -> Any:
        counter["active"] = True
        before = counter["windows"]
        try:
            return original_plan(*a, **k)
        finally:
            counter["active"] = False
            plans.append({"labels": len(a[0]) if a else None, "windows": counter["windows"] - before})

    ledger.plan_ordinals = counted_plan
    # Main also has `_window_step`: every candidate window, the whole-lineage fallback included (its `step >= 1`
    # does not pass through MIN_GAP), is one call. Counted when present; v0.30.0 has only the reflected compare.
    original_step = getattr(ledger, "_window_step", None)
    if callable(original_step):
        def counted_step(*a: Any, **k: Any) -> Any:
            if counter["active"]:
                counter["step_calls"] += 1
                if counter["step_calls"] > limit:
                    counter["aborted"] = True
                    raise WindowBudgetExceeded(f"candidate windows passed {limit}")
            return original_step(*a, **k)

        ledger._window_step = counted_step
    statements: list[int] = []
    owner = threading.get_ident()
    original_connect = sqlite3.connect
    tracing = {"on": False}

    def traced_connect(*a: Any, **k: Any) -> sqlite3.Connection:
        conn = original_connect(*a, **k)
        conn.set_trace_callback(lambda sql: statements.append(1)
                                if tracing["on"] and threading.get_ident() == owner else None)
        return conn

    sqlite3.connect = traced_connect
    engine = rt.engine(engine_module, config, kit, args.work / "home", session, conversation)
    error = None
    tracing["on"] = True
    began = time.monotonic()
    try:
        engine.ingest(new_view)
    except Exception as exc:  # noqa: BLE001 - class only
        error = type(exc).__name__
    finally:
        tracing["on"] = False
        receipt["insert_ingest_s"] = round(time.monotonic() - began, 3)
        sqlite3.connect = original_connect
        ledger.MIN_GAP, ledger.plan_ordinals = original_gap, original_plan
        if callable(original_step):
            ledger._window_step = original_step
        engine.shutdown()
    stored = lineage_rows(db, session)
    contents = [content for _o, _s, content in stored]
    windows = max(counter["windows"], counter["step_calls"])
    receipt.update({
        "windows": windows, "windows_reflected": counter["windows"], "window_step_calls": counter["step_calls"],
        "window_budget_exceeded": counter["aborted"] or (args.max_windows is not None and windows > args.max_windows),
        "plans": plans,
        "statements": len(statements), "exception": error,
        "respace_lines": [line["message"] for line in lines.lines if "respace" in line["message"]],
        "stored_in_order": contents == [row["content"] for row in new_view],
        "inserted_stored_once": contents.count(inserted["content"]) == 1,
        "lineage_rows": len(stored),
    })
    within = ((args.max_windows is None or windows <= args.max_windows)
              and (args.max_statements is None or len(statements) <= args.max_statements))
    if args.max_windows is None or args.max_statements is None:
        receipt["verdict"] = "UNRUN"  # not green: the check is judged only against the kit's frozen counts
    elif not plans or not windows:  # the counter never attached (a renamed planner): no evidence either way
        receipt["verdict"] = "INCONCLUSIVE"
    else:
        receipt["verdict"] = "PASS" if (within and not counter["aborted"] and receipt["stored_in_order"]
                                        and receipt["inserted_stored_once"]) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(receipt, handle, indent=1, sort_keys=True)
    print(json.dumps({k: receipt[k] for k in ("windows", "window_budget_exceeded", "statements", "exception",
                                               "stored_in_order", "inserted_stored_once", "verdict",
                                               "fixture_ingest_s", "insert_ingest_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
