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


def combine(outcomes: list[str]) -> str:
    """FAIL beats INCONCLUSIVE beats UNRUN beats PASS; an empty list is UNRUN."""
    if not outcomes:
        return UNRUN
    for outcome in (FAIL, INCONCLUSIVE, UNRUN):
        if outcome in outcomes:
            return outcome
    return PASS


def d9(receipts: dict[str, dict[str, Any] | None], bars: dict[str, Any], pinned_rows: int) -> dict[str, Any]:
    """Judged arm: the first pass completes without an exception and holds <= bar; a pass that raises, does not
    reach its end, or examines fewer rows than pinned is INCONCLUSIVE. The default arm is recorded."""
    judged = receipts.get(JUDGED)
    reasons: list[str] = []
    if judged is None:
        outcome = UNRUN
        reasons.append("no receipt for the judged arm")
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
    if reasons:
        return {"arm": arm, "outcome": INCONCLUSIVE, "reasons": reasons}
    failed: list[str] = []
    hold = receipt["longest_hold_in_pass_s"]
    if arm == JUDGED and hold > bars["d10_hold_s"]:
        failed.append(f"bar 1: hold {hold:.3f} s over {bars['d10_hold_s']} s")
    slow = [c for c in receipt.get("compaction_calls", []) if c.get("during_hold")
            and c.get("return_s", 0) >= bars["d10_compaction_return_s"]]
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


def d11_budget(receipt: dict[str, Any] | None) -> dict[str, Any]:
    """One budget in the judged arm: shorter at the end, every publishing pass shrinks, no hold, nothing lost."""
    if receipt is None:
        return {"outcome": UNRUN, "reasons": ["no receipt"]}
    passes = receipt.get("passes") or []
    if receipt.get("exception") and not passes:
        return {"outcome": INCONCLUSIVE, "reasons": [f"the sequence raised {receipt['exception']}"]}
    if not any(p.get("published") for p in passes):
        return {"outcome": INCONCLUSIVE, "reasons": ["no pass published a summary"],
                "rows": [receipt.get("rows_before")] + [p.get("rows_out") for p in passes]}
    failed: list[str] = []
    if not passes or passes[-1]["rows_out"] >= receipt["rows_before"]:
        failed.append("the list is not shorter at the end than at the start")
    for number, item in enumerate(passes, 1):
        if item.get("published") and item["rows_out"] >= item["rows_in"]:
            failed.append(f"pass {number} published and returned {item['rows_out']} rows for {item['rows_in']}")
        if item.get("hold_armed"):
            failed.append(f"pass {number} armed a compaction hold")
        check = item.get("stored_check") or {}
        if check.get("lost") or check.get("duplicated") or check.get("reordered"):
            failed.append(f"pass {number}: stored rows lost {check.get('lost')}, duplicated "
                          f"{check.get('duplicated')}, reordered {check.get('reordered')}")
    if len(passes) < 8 and not failed:
        return {"outcome": INCONCLUSIVE, "reasons": [f"only {len(passes)} of 8 passes ran"]}
    return {"outcome": FAIL if failed else PASS, "reasons": failed or ["bars met"],
            "rows": [receipt["rows_before"]] + [p["rows_out"] for p in passes]}


def d11(receipts: dict[str, dict[str, dict[str, Any] | None]]) -> dict[str, Any]:
    """`receipts[arm][budget]`. The judged arm decides; the default arm is recorded."""
    judged = {budget: d11_budget(r) for budget, r in (receipts.get(JUDGED) or {}).items()}
    recorded = {budget: d11_budget(r) for budget, r in (receipts.get(RECORDED) or {}).items()}
    outcome = combine([v["outcome"] for v in judged.values()]) if len(judged) == 3 else (
        combine([v["outcome"] for v in judged.values()] + [UNRUN]))
    return {"row": "D11", "outcome": outcome, "judged": judged, "recorded": recorded}


def d12(receipts: dict[str, dict[str, dict[str, Any] | None]]) -> dict[str, Any]:
    """Both arms, every budget: every no-publish pass returns its input unchanged; each needs at least one."""
    cells: dict[str, Any] = {}
    for arm in (JUDGED, RECORDED):
        for budget, receipt in (receipts.get(arm) or {}).items():
            if receipt is None:
                cells[f"{arm}/{budget}"] = {"outcome": UNRUN}
                continue
            judged = [p for p in (receipt.get("passes") or []) + (receipt.get("extra_passes") or [])
                      if not p.get("published")]
            changed = [p for p in judged if not p.get("returned_equals_input")]
            if changed:
                cells[f"{arm}/{budget}"] = {"outcome": FAIL, "passes": len(judged), "changed": len(changed)}
            elif not judged:
                cells[f"{arm}/{budget}"] = {"outcome": INCONCLUSIVE, "passes": 0}
            else:
                cells[f"{arm}/{budget}"] = {"outcome": PASS, "passes": len(judged)}
    outcomes = [c["outcome"] for c in cells.values()]
    if len(cells) < 6:
        outcomes.append(UNRUN)
    return {"row": "D12", "outcome": combine(outcomes), "cells": cells}
