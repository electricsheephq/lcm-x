"""Resolve B1/B2 rows with the product's placeholder predicate and payload reader."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import externalize
from config import LCMConfig


def resolve(rows: list[tuple], home: Path) -> tuple[list[tuple], dict, set]:
    config = LCMConfig()  # The harness forbids path-valued overrides; never read the operator's env/home.
    resolved, ids, losses = [], set(), []
    for row in rows:
        if not externalize.is_externalized_placeholder(row[3]):
            resolved.append(row)
            continue
        ref = externalize.extract_externalized_ref(row[3])
        reason = None
        try:
            path = externalize.get_large_output_storage_dir(config, hermes_home=str(home), create=False) / ref
            # The product reader defaults an absent content field to ''. Check presence separately.
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or "content" not in raw:
                reason = "no_content_field"
            elif not isinstance(raw["content"], str):
                reason = "invalid_content"
            else:
                payload = externalize.load_externalized_payload(ref, config=config, hermes_home=str(home))
                if payload is None:
                    reason = "unreadable_payload"
        except FileNotFoundError:
            reason = "missing_payload_file"
        except (ValueError, OSError, TypeError, AttributeError):
            reason = "unreadable_payload"
        if reason:
            losses.append({"store_id": row[0], "reason": reason})
        else:
            resolved.append((*row[:3], payload["content"], *row[4:]))
            ids.add(row[0])
    numbers = {"resolved": len(ids), "unresolved": len(losses), "deficit_rows": len(losses),
               "reasons": dict(Counter(e["reason"] for e in losses)),
               "store_ids": [e["store_id"] for e in losses][:10], "deficits": losses[:10]}
    return resolved, numbers, ids


def mismatches(numbers: dict, rows: list[tuple], ids: set, per: dict, group, expected_tools, loose, host) -> None:
    """Attribute wrong resolved bytes to externalized losses; ordinary B2 accounting still applies."""
    from . import multiset, tool_calls

    want = {g: {(r, multiset.h(t)) for r, t in items if multiset.norm(t)} for g, (items, _) in per.items()}
    for row in rows:
        sid, session, role, content = row[:4]
        if sid not in ids:
            continue
        g = group(session)
        if role == "tool":
            keys = tool_calls.stored_keys([row], group)
            bad = any(k not in expected_tools and k not in loose for k in keys)
        else:
            key = (role, multiset.h(content))
            licensed = role == "user" and (host or {}).get(g, {}).get("keys", {}).get(key, {}).get("n", 0)
            bad = bool(multiset.norm(content)) and key not in want[g] and not licensed
        if bad:
            numbers["deficit_rows"] += 1
            numbers["reasons"]["content_mismatch"] = numbers["reasons"].get("content_mismatch", 0) + 1
            if len(numbers["store_ids"]) < 10:
                numbers["store_ids"].append(sid)
                numbers["deficits"].append({"store_id": sid, "reason": "content_mismatch", "sha256": multiset.h(content)})
