"""Bars D1-D3 (``drain/hidden-backlog/*``): does the host's message list drain while stored raw rows the host no
longer holds (a hidden backlog) wait above the frontier? Declared before any run, not tuned.

Input: the observer's per-call ``counters["compactions"]`` in each phase-*.json (observer/rel_observer.py
``list_counts``): ``in`` / ``out`` = the lengths of the list handed to ``compress`` and of the list it returned,
``host_rows_summarized`` = the input rows a leaf replaced (by identity), ``secs`` = the call's wall time. A phase-2
compaction is a non-final call at turn >= ``drain.phase2_turn`` whose status committed (compacted / host_native).

D1: by the SECOND phase-2 compaction, one has ``out < in`` (the host list shrinks).
D2: every phase-2 compaction has ``host_rows_summarized >= 1`` or ``out < in``; a raised call fails this bar.
D3: no phase-2 compaction that replaced no host row ran longer than the host's pre-turn hold (``hold_seconds``).
Each bar is INCONCLUSIVE when phase 2 reached fewer than 2 compactions, or when a cell that asks for the Fixture B
host-forget step (``drain.forget_host_rows``) has no ``fixture.jsonl`` record that it archived at least one row.
Phase 2 also needs a recorded boundary by ``phase2_turn``: the fired clean-exit fault for a cell that has one (every
in-place cell, and Fixture B in both modes), else a host session change in the observer's prompt/end transcript. D2 errors still fail below two commits.
D4 (a count, never a failure): the phase-2 compactions with ``out == in`` (the plateau). Reported with it: the
pass on which the host list first shrank and, per pass, the stored rows newly covered that were not host rows
(``rows_covered`` minus ``host_rows_summarized``; the host count also carries a row or two the pass rewrote rather
than summarized, so the per-pass figure is approximate) with its running total, to set against the fixture's
forgotten rows.
"""
from __future__ import annotations

import json
from pathlib import Path

COMMITTED = ("compacted", "host_native")


def calls(phases: list[dict]) -> list[dict]:
    return [dict(c, phase=p.get("phase")) for p in phases for c in (p.get("counters") or {}).get("compactions") or []]


def fixture(cell_dir: Path | None) -> list[dict]:
    path = Path(cell_dir) / "fixture.jsonl" if cell_dir else None
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path and path.exists() else []


def exits_cleanly(cell: dict) -> bool:
    """The cell's boundary is a clean host exit (every in-place drain cell; Fixture B in both modes)."""
    return bool(cell.get("in_place")) or any(f.get("kind") == "clean_exit_before_turn" for f in cell.get("faults") or [])


def boundary(cell: dict, cell_dir: Path | None, first: int) -> bool:
    """The runner's fired clean-exit fault when the cell has one, otherwise the observer's actual host session
    transition (a rotation cell without the fault); never the turn count alone."""
    name = "faults-fired.jsonl" if exits_cleanly(cell) else "transcript.jsonl"
    path = Path(cell_dir) / name if cell_dir else None
    records = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path and path.exists() else []
    if exits_cleanly(cell):
        return any(r.get("kind") == "clean_exit_before_turn" and 0 < (r.get("turn") or 0) <= first for r in records)
    sessions = [r["session"] for r in records if r.get("event") in ("host_prompt", "turn_end") and r.get("session")
                and r.get("session_prefix", "T") == "T" and 0 < (r.get("turn") or 0) <= first]
    return any(a != b for a, b in zip(sessions, sessions[1:]))


def plateau(done: list[dict]) -> dict:
    """D4 and the drain profile of the phase-2 compactions."""
    flat = [c for c in done if c["out"] is not None and c["out"] == c["in"]]
    shrink = next((i for i, c in enumerate(done, 1) if c["out"] is not None and c["out"] < c["in"]), None)
    per_pass = [None if c.get("rows_covered") is None else c["rows_covered"] - (c["host_rows_summarized"] or 0)
                for c in done]
    running, total = [], 0
    for x in per_pass:
        total += max(x or 0, 0)
        running.append(total)
    return {"plateau_compactions": len(flat), "plateau_turns": [c["turn"] for c in flat],
            "first_shrink_pass": shrink, "hidden_rows_covered_per_pass": per_pass, "hidden_rows_covered_running": running}


def score(cell: dict, phases: list[dict], cell_dir: Path | None = None) -> tuple[dict, dict, dict]:
    """(failed, inconclusive, numbers) for the D bars."""
    spec = cell.get("drain") or {}
    first, hold = int(spec.get("phase2_turn", 1)), float(spec.get("hold_seconds", 10.0))
    every = calls(phases)
    crossed = boundary(cell, cell_dir, first)
    phase2 = [c for c in every if not c.get("final") and (c.get("turn") or 0) >= first] if crossed else []
    done = [c for c in phase2 if c.get("status") in COMMITTED]
    numbers = {"phase2_turn": first, "hold_seconds": hold, "calls": len(every),
               "phase1_compactions": [c for c in every if not c.get("final") and (c.get("turn") or 0) < first
                                      and c.get("status") in COMMITTED],
               "phase2_compactions": done, "phase2_other_calls": [c for c in phase2 if c not in done]}
    numbers["D4"] = plateau(done)
    failed, inconclusive = {}, {}
    if not crossed:
        why = "phase 2 has no recorded clean exit" if exits_cleanly(cell) else "phase 2 has no recorded host rotation/session change"
        return failed, {b: f"{why} at or before turn {first}" for b in ("D1", "D2", "D3")}, numbers
    errors = [{k: c.get(k) for k in ("turn", "error", "secs")} for c in phase2
              if c.get("error") or c.get("status") == "error"]
    if errors:  # decided before any INCONCLUSIVE return: an errored phase-2 call always fails D2
        failed["D2"] = {"errored_calls": errors}
    if spec.get("forget_host_rows"):
        numbers["fixture"] = fixture(cell_dir)
        if not any(f.get("kind") == "forget_host_rows" and f.get("rows") for f in numbers["fixture"]):
            return failed, {b: "Fixture B: the host-forget step archived no row" for b in ("D1", "D2", "D3")
                            if b not in failed}, numbers
    if len(done) < 2:
        why = f"phase 2 reached {len(done)} compaction(s) (< 2)" if every else "no compaction counters recorded"
        return failed, {b: why for b in ("D1", "D2", "D3") if b not in failed}, numbers
    shrink = [c for c in done[:2] if c["out"] is not None and c["out"] < c["in"]]
    numbers["D1"] = {"first_two": [{k: c[k] for k in ("turn", "in", "out", "host_rows_summarized")} for c in done[:2]]}
    if not shrink:
        failed["D1"] = numbers["D1"]
    stuck = [c for c in done if not ((c["host_rows_summarized"] or 0) >= 1 or (c["out"] is not None and c["out"] < c["in"]))]
    numbers["D2"] = {"compactions": len(done), "no_host_row_and_no_shrink": len(stuck),
                     "turns": [c["turn"] for c in stuck]}
    if errors:
        numbers["D2"]["errored_calls"] = errors
    if stuck or errors:
        failed["D2"] = numbers["D2"]
    slow = [c for c in done if not c["host_rows_summarized"] and (c.get("secs") or 0) > hold]
    numbers["D3"] = {"no_host_row_compactions": sum(1 for c in done if not c["host_rows_summarized"]),
                     "max_secs_no_host_row": max((c.get("secs") or 0 for c in done if not c["host_rows_summarized"]),
                                                 default=None),
                     "over_hold": [{k: c.get(k) for k in ("turn", "secs")} for c in slow]}
    if slow:
        failed["D3"] = numbers["D3"]
    return failed, inconclusive, numbers
