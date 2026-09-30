"""Bars D1-D3 (``drain/hidden-backlog/*``): does the host's message list drain while stored raw rows the host no
longer holds (a hidden backlog) wait above the frontier? Declared before any run, not tuned.

Input: the observer's per-call ``counters["compactions"]`` in each phase-*.json (observer/rel_observer.py
``list_counts``): ``in`` / ``out`` = the lengths of the list handed to ``compress`` and of the list it returned,
``host_rows_summarized`` = the input rows a leaf replaced (by identity), ``secs`` = the call's wall time. A phase-2
compaction is a non-final call at turn >= ``drain.phase2_turn`` whose status committed (compacted / host_native).

D1: by the SECOND phase-2 compaction, one has ``out < in`` (the host list shrinks).
D2: every phase-2 compaction has ``host_rows_summarized >= 1`` or ``out < in``.
D3: no phase-2 compaction that replaced no host row ran longer than the host's pre-turn hold (``hold_seconds``).
Each bar is INCONCLUSIVE when phase 2 reached fewer than 2 compactions.
"""
from __future__ import annotations

COMMITTED = ("compacted", "host_native")


def calls(phases: list[dict]) -> list[dict]:
    return [dict(c, phase=p.get("phase")) for p in phases for c in (p.get("counters") or {}).get("compactions") or []]


def score(cell: dict, phases: list[dict]) -> tuple[dict, dict, dict]:
    """(failed, inconclusive, numbers) for the D bars."""
    spec = cell.get("drain") or {}
    first, hold = int(spec.get("phase2_turn", 1)), float(spec.get("hold_seconds", 10.0))
    every = calls(phases)
    phase2 = [c for c in every if not c.get("final") and (c.get("turn") or 0) >= first]
    done = [c for c in phase2 if c.get("status") in COMMITTED]
    numbers = {"phase2_turn": first, "hold_seconds": hold, "calls": len(every),
               "phase1_compactions": [c for c in every if not c.get("final") and (c.get("turn") or 0) < first
                                      and c.get("status") in COMMITTED],
               "phase2_compactions": done, "phase2_other_calls": [c for c in phase2 if c not in done]}
    failed, inconclusive = {}, {}
    if len(done) < 2:
        why = f"phase 2 reached {len(done)} compaction(s) (< 2)" if every else "no compaction counters recorded"
        return failed, {b: why for b in ("D1", "D2", "D3")}, numbers
    shrink = [c for c in done[:2] if c["out"] is not None and c["out"] < c["in"]]
    numbers["D1"] = {"first_two": [{k: c[k] for k in ("turn", "in", "out", "host_rows_summarized")} for c in done[:2]]}
    if not shrink:
        failed["D1"] = numbers["D1"]
    stuck = [c for c in done if not ((c["host_rows_summarized"] or 0) >= 1 or (c["out"] is not None and c["out"] < c["in"]))]
    numbers["D2"] = {"compactions": len(done), "no_host_row_and_no_shrink": len(stuck),
                     "turns": [c["turn"] for c in stuck]}
    if stuck:
        failed["D2"] = numbers["D2"]
    slow = [c for c in done if not c["host_rows_summarized"] and (c.get("secs") or 0) > hold]
    numbers["D3"] = {"no_host_row_compactions": sum(1 for c in done if not c["host_rows_summarized"]),
                     "max_secs_no_host_row": max((c.get("secs") or 0 for c in done if not c["host_rows_summarized"]),
                                                 default=None),
                     "over_hold": [{k: c.get(k) for k in ("turn", "secs")} for c in slow]}
    if slow:
        failed["D3"] = numbers["D3"]
    return failed, inconclusive, numbers
