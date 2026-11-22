"""Raw verdicts for D9-D12 from receipts, applied as the rows in bench/specs/RELEASE-READINESS-V1.md state them.

Outcomes: PASS, FAIL, INCONCLUSIVE (not green), UNRUN (not green). A recorded arm never decides a row. The added
variants of #2868 (contention, interrupted first turn, per-turn wait, timeout with no publish) are recorded only.

Arms: `uid_on` is the opt-in mode `LCM_HOST_MESSAGE_UID=on` (judged where the row says so), `default` the shipped
default. Bars come from the kit (`bars`).
"""

from __future__ import annotations

from typing import Any

PASS, FAIL, INCONCLUSIVE, UNRUN = "PASS", "FAIL", "INCONCLUSIVE", "UNRUN"
JUDGED, RECORDED = "uid_on", "default"
BUDGETS = ("default", "60s", "120s")
GAP_MS = 1000.0  # a sampling gap over this could hide or stretch a hold: the run is INCONCLUSIVE


def mismatch(receipt: dict[str, Any], cell: str, arm: str, budget: str | None = None) -> str | None:
    """A receipt decides only the cell, arm, budget and variant it says it ran (never by its file name)."""
    if receipt.get("cell") != cell or receipt.get("arm") != arm:
        return f"receipt is {receipt.get('cell')}/{receipt.get('arm')}, not {cell}/{arm}"
    if budget is not None and receipt.get("budget") != budget:
        return f"receipt budget is {receipt.get('budget')}, not {budget}"
    if (receipt.get("worst_case") or receipt.get("control", "none") != "none"
            or (receipt.get("load") or {}).get("processes") or receipt.get("keep_copy")):
        return "a recorded variant (worst case, control, load or a kept copy) never decides a row"
    if not receipt.get("qualifies", False):
        return "the receipt does not qualify (scaled store, stand-in override, profile, control or copy mismatch)"
    return None


def combine(outcomes: list[str]) -> str:
    """FAIL beats INCONCLUSIVE beats UNRUN beats PASS; an empty list is UNRUN."""
    if not outcomes:
        return UNRUN
    for outcome in (FAIL, INCONCLUSIVE, UNRUN):
        if outcome in outcomes:
            return outcome
    return PASS


def unexpected(receipt: dict[str, Any]) -> int:
    return int(receipt.get("lock_unexpected") or (receipt.get("lock_watch") or {}).get("unexpected_answers") or 0)


def d9(receipts: dict[str, dict[str, Any] | None], bars: dict[str, Any], pinned_rows: int) -> dict[str, Any]:
    """Judged arm: the first pass completes without an exception and holds <= bar; a pass that raises, does not
    reach its end, or examines fewer rows than pinned is INCONCLUSIVE. The default arm is recorded."""
    judged = receipts.get(JUDGED)
    reasons: list[str] = []
    wrong = mismatch(judged, "d9", JUDGED, "default") if judged is not None else None
    if judged is None:
        outcome = UNRUN
        reasons.append("no receipt for the judged arm")
    elif wrong:
        outcome = INCONCLUSIVE
        reasons.append(wrong)
    elif (judged.get("lock_watch") or {}).get("largest_gap_ms", 0) > GAP_MS:
        outcome = INCONCLUSIVE
        reasons.append("the lock watcher's largest sampling gap could hide a hold over the bar")
    elif unexpected(judged):
        outcome = INCONCLUSIVE
        reasons.append(f"the lock watcher saw {unexpected(judged)} answers it could not classify")
    elif judged.get("exception") or not judged.get("pass_completed"):
        outcome = INCONCLUSIVE
        reasons.append("the first pass raised or did not reach its end")
    elif (judged.get("rows_examined") or 0) < pinned_rows:
        outcome = INCONCLUSIVE
        reasons.append(f"examined {judged.get('rows_examined')} rows, fewer than the pinned {pinned_rows}")
    else:
        hold = judged["longest_hold_s"]
        outcome = PASS if hold <= bars["d9_hold_s"] else FAIL
        reasons.append(f"longest hold {hold:.3f} s against {bars['d9_hold_s']} s")
    return {"row": "D9", "outcome": outcome, "reasons": reasons, "judged_arm": JUDGED,
            "recorded": {arm: _hold_summary(r) for arm, r in receipts.items() if arm != JUDGED}}


def _hold_summary(receipt: dict[str, Any] | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    return {k: receipt.get(k) for k in ("longest_hold_s", "pass_completed", "exception", "rows_examined")}


def d10_run(receipt: dict[str, Any] | None, arm: str, bars: dict[str, Any], pinned_rows: int) -> dict[str, Any]:
    """One run in one arm. Bars 2-4 in both arms; bar 1 (hold) judged in the opt-in arm, recorded in the default."""
    if receipt is None:
        return {"arm": arm, "outcome": UNRUN, "reasons": ["no receipt"]}
    wrong = mismatch(receipt, "d10", arm, "default")
    if receipt.get("exceptions") and wrong is None:  # an exception fails bar 4 in either arm, crash receipts included
        return {"arm": arm, "outcome": FAIL, "reasons": ["an exception: bar 4 fails"]}
    if wrong:
        return {"arm": arm, "outcome": INCONCLUSIVE, "reasons": [wrong]}
    if receipt.get("exceptions"):
        return {"arm": arm, "outcome": FAIL, "reasons": ["an exception: bar 4 fails"]}
    reasons: list[str] = []
    started = receipt.get("started_during_hold") or {}
    if not receipt.get("pass_completed"):
        reasons.append("the pass ended before completing the whole conversation")
    if (receipt.get("rows_examined") or 0) < pinned_rows:
        reasons.append(f"examined {receipt.get('rows_examined')} rows, fewer than the pinned {pinned_rows}")
    if not started.get("turn") or not started.get("compaction"):
        reasons.append("no turn-path call or no further compaction call started during the hold")
    if (receipt.get("lock_watch") or {}).get("largest_gap_ms", 0) > GAP_MS:
        reasons.append("the lock watcher's largest sampling gap could hide a hold")
    if unexpected(receipt):
        reasons.append(f"the lock watcher saw {unexpected(receipt)} answers it could not classify")
    if reasons:
        return {"arm": arm, "outcome": INCONCLUSIVE, "reasons": reasons}
    failed: list[str] = []
    hold = receipt["longest_hold_in_pass_s"]
    if arm == JUDGED and hold > bars["d10_hold_s"]:
        failed.append(f"bar 1: hold {hold:.3f} s over {bars['d10_hold_s']} s")
    slow = [c for c in receipt.get("compaction_calls", []) if c.get("during_hold")
            and (c.get("unfinished") or c.get("return_s", 0) >= bars["d10_compaction_return_s"])]
    if slow:
        failed.append(f"bar 2: {len(slow)} compaction calls started during the hold returned in "
                      f"{bars['d10_compaction_return_s']} s or more")
    waits = [c for c in receipt.get("turn_calls", []) if c.get("wait_s", 0) > bars["d10_turn_wait_s"]]
    if waits:
        failed.append(f"bar 3: {len(waits)} turn-path calls waited more than {bars['d10_turn_wait_s']} s")
    recorded = {"hold_s": hold} if arm != JUDGED else {}
    return {"arm": arm, "outcome": FAIL if failed else PASS, "reasons": failed or ["bars met"], **recorded}


def d10(receipts: dict[str, dict[str, Any] | None], bars: dict[str, Any], pinned_rows: int) -> dict[str, Any]:
    runs = [d10_run(receipts.get(arm), arm, bars, pinned_rows) for arm in (JUDGED, RECORDED)]
    return {"row": "D10", "outcome": combine([run["outcome"] for run in runs]), "runs": runs}


def d11_budget(receipt: dict[str, Any] | None, arm: str = JUDGED, budget: str | None = None) -> dict[str, Any]:
    """One budget: shorter at the end, every publishing pass shrinks, no hold, nothing lost or duplicated."""
    if receipt is None:
        return {"outcome": UNRUN, "reasons": ["no receipt"]}
    wrong = mismatch(receipt, "d11", arm, budget)
    if wrong:
        return {"outcome": INCONCLUSIVE, "reasons": [wrong]}
    passes = receipt.get("passes") or []
    if receipt.get("cell_error") or (receipt.get("exception") and not passes):
        return {"outcome": INCONCLUSIVE, "reasons": [f"the sequence raised {receipt.get('exception')}"]}
    broken = [n for n, p in enumerate(passes, 1) if p.get("exception") or p.get("returned_list") is not True
              or "stop_reasons" not in p or "error" in (p.get("stop_reasons") or [])]  # no evidence is not success
    failed: list[str] = []
    for number, item in enumerate(passes, 1):  # a measured loss or copy fails whether or not anything published
        check = item.get("stored_check") or {}
        if check.get("lost") or check.get("changed") or check.get("duplicated") or check.get("reordered"):
            failed.append(f"pass {number}: stored rows lost {check.get('lost')} (changed {check.get('changed')}), "
                          f"duplicated {check.get('duplicated')}, reordered {check.get('reordered')}")
    if not any(p.get("published") for p in passes):
        rows = [receipt.get("rows_before")] + [p.get("rows_out") for p in passes]
        return ({"outcome": FAIL, "reasons": failed, "rows": rows} if failed else
                {"outcome": INCONCLUSIVE, "reasons": ["no pass published a summary"], "rows": rows})
    last = next((p["rows_out"] for p in reversed(passes) if p.get("rows_out") is not None), None)
    if last is None or last >= receipt["rows_before"]:
        failed.append("the list is not shorter at the end than at the start")
    for number, item in enumerate(passes, 1):
        if item.get("published") and (item.get("rows_out") is None or item["rows_out"] >= item["rows_in"]):
            failed.append(f"pass {number} published and returned {item['rows_out']} rows for {item['rows_in']}")
        if item.get("hold_armed"):
            failed.append(f"pass {number} armed a compaction hold")
    rows = [receipt.get("rows_before")] + [p.get("rows_out") for p in passes]
    if failed:
        return {"outcome": FAIL, "reasons": failed, "rows": rows}
    if broken:
        return {"outcome": INCONCLUSIVE, "reasons": [f"passes {broken} raised or returned no list"], "rows": rows}
    if len(passes) < 8:
        return {"outcome": INCONCLUSIVE, "reasons": [f"only {len(passes)} of 8 passes ran"], "rows": rows}
    return {"outcome": PASS, "reasons": ["bars met"], "rows": rows}


def d11(receipts: dict[str, dict[str, dict[str, Any] | None]]) -> dict[str, Any]:
    """`receipts[arm][budget]`. The judged arm decides; the default arm is recorded."""
    given = receipts.get(JUDGED) or {}
    judged = {budget: d11_budget(given.get(budget), JUDGED, budget) for budget in BUDGETS}
    recorded = {budget: d11_budget((receipts.get(RECORDED) or {}).get(budget), RECORDED, budget) for budget in BUDGETS}
    outcome = combine([v["outcome"] for v in judged.values()])
    return {"row": "D11", "outcome": outcome, "judged": judged, "recorded": recorded}


def d12(receipts: dict[str, dict[str, dict[str, Any] | None]]) -> dict[str, Any]:
    """Both arms, every budget: every no-publish pass returns its input unchanged; each needs at least one."""
    cells: dict[str, Any] = {}
    for arm in (JUDGED, RECORDED):
        for budget in BUDGETS:
            receipt = (receipts.get(arm) or {}).get(budget)
            if receipt is None:
                cells[f"{arm}/{budget}"] = {"outcome": UNRUN}
                continue
            wrong = mismatch(receipt, "d11", arm, budget)
            if wrong:
                cells[f"{arm}/{budget}"] = {"outcome": INCONCLUSIVE, "reason": wrong}
                continue
            every = (receipt.get("passes") or []) + (receipt.get("extra_passes") or [])
            # A pass that raised or returned no list gave the host nothing to compare: never an unchanged pass.
            broken = [p for p in every if p.get("exception") or p.get("returned_list") is not True
                      or "stop_reasons" not in p or "error" in (p.get("stop_reasons") or [])]
            judged = [p for p in every if not p.get("published") and p not in broken]
            changed = [p for p in judged if not p.get("returned_equals_input")]
            if changed:
                cells[f"{arm}/{budget}"] = {"outcome": FAIL, "passes": len(judged), "changed": len(changed)}
            elif broken or receipt.get("cell_error"):
                cells[f"{arm}/{budget}"] = {"outcome": INCONCLUSIVE, "passes": len(judged), "broken": len(broken)}
            elif not judged:
                cells[f"{arm}/{budget}"] = {"outcome": INCONCLUSIVE, "passes": 0}
            else:
                cells[f"{arm}/{budget}"] = {"outcome": PASS, "passes": len(judged)}
    outcomes = [c["outcome"] for c in cells.values()]
    return {"row": "D12", "outcome": combine(outcomes), "cells": cells}
