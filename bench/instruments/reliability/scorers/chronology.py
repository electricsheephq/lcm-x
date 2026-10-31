"""Chronology (PR #567 T16): a REPORTED, non-gating metric. Per session lineage, the ``[Xnn]`` turn tags of stored
user rows should be non-decreasing in store_id order; each row whose smallest tag is below the largest tag already
stored in its lineage is listed with both store_ids. Not a bar: a legitimate crash-recovery tail append can reorder
rows. Promote it only after its behaviour on the #436 build has been seen.
"""
from __future__ import annotations

import re

TAG = re.compile(r"\[([A-Z])(\d{2,3})\] user turn")


def report(stored_rows, group) -> dict:
    """``stored_rows``: (store_id, session_id, role, content) in store_id order; ``group``: session id -> lineage."""
    high, violations, checked = {}, [], 0
    for store_id, session_id, role, content in sorted(stored_rows, key=lambda r: r[0]):
        tags = TAG.findall(content or "") if role == "user" else []
        if not tags:
            continue
        checked += 1
        for prefix in sorted({p for p, _n in tags}):
            nums = [int(n) for p, n in tags if p == prefix]
            key = (group(session_id), prefix)
            top = high.get(key)
            if top and min(nums) < top[0]:
                violations.append({"lineage": key[0], "store_id": store_id, "tag": f"{prefix}{min(nums):02d}",
                                   "after_tag": f"{prefix}{top[0]:02d}", "after_store_id": top[1]})
            if not top or max(nums) >= top[0]:
                high[key] = (max(nums), store_id)
    return {"user_rows_checked": checked, "violations": len(violations), "examples": violations[:10]}
