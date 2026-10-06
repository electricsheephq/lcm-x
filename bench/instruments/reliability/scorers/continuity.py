"""First provider request after each committed R2 compaction; diagnostic only.

Tag presence is literal anywhere in the request, not proof of an operative
instruction. Missing observations are unknown, never a successful check.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

NOTE = "This conversation uses Lossless Context Management (LCM)"
USER = re.compile(r"\[([A-Z]\d{2,3})\]")
REPLY = re.compile(r"reply to ([A-Z]\d{2,3}):")
CHECKS = ("F1", "F2", "F3", "F4")


def markers(messages: list[dict], *, anchor=None, previous=None, current=None) -> dict:
    """No request content escapes: only booleans, synthetic tags and role/counts."""
    users, replies, roles = set(), set(), []
    for m in messages:
        text = m.get("content") or ""
        u, r = set(USER.findall(text)), set(REPLY.findall(text))
        users.update(u)
        replies.update(r)
        roles.append({"role": m.get("role"), "user_tags": len(u), "reply_tags": len(r)})
    return {"F1": any(m.get("role") == "system" and NOTE in (m.get("content") or "") for m in messages),
            "F2": anchor in users if anchor else None, "F3": previous in replies if previous else None,
            "F4": current in users if current else None,
            "user_tags": sorted(users), "reply_tags": sorted(replies), "message_roles": roles,
            "role_counts": dict(Counter(m.get("role") for m in messages))}


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def score(cell: dict, requests: list[dict], observations: list[dict]) -> dict:
    """Join by receive/commit time, including within-turn loops and restarted phases.

    Held/closed records repeat a request id: the earliest receive time defines
    the request even if a response arrives later. Completed replies come from
    observer turn ends, not provider emits that may never reach the host.
    """
    by_id = {}
    for r in requests:
        if r.get("role") == "main" and isinstance(r.get("ts"), (int, float)):
            by_id.setdefault(r["rid"], r)
    main = sorted(by_id.values(), key=lambda r: (r["ts"], r["rid"]))
    completed = [n for n in observations if n.get("kind") == "turn_end" and n.get("turn_kind") != "final"
                 and not n.get("failed") and not n.get("interrupted") and n.get("ts") is not None]
    totals = {f: {k: 0 for k in ("present", "absent", "unknown", "not_applicable")} for f in CHECKS}
    rows = []
    for c in observations:
        if c.get("kind") not in ("compaction_committed", "survival_fit_committed"):
            continue
        ts = c.get("ts")
        req = next((r for r in main if ts is not None and r["ts"] > ts), None)
        prev = max((n for n in completed if ts is not None and n["ts"] < ts),
                   key=lambda n: n["ts"], default={}).get("reply_tag")
        data = (req or {}).get("continuity") or {}
        current = (req or {}).get("current_user_tag")
        sole = bool((cell.get("continuity") or {}).get("sole_user"))
        row = {"phase": c.get("phase"), "turn": c.get("turn"), "kind": c.get("compaction_kind") or
               ("forced" if c.get("final") else "in-place" if cell.get("in_place") else "rotation"),
               "status": "observed" if req and data else "no request" if req is None else "no markers",
               "request_id": (req or {}).get("rid"), "request_turn": (req or {}).get("turn"),
               "previous_reply_tag": prev, "current_user_tag": current,
               "survival_fit": c.get("survival_fit", False), "fit_reason": c.get("fit_reason"),
               "previous_reply_in_compacted_span": prev in c["compacted_reply_tags"]
               if prev and "compacted_reply_tags" in c else None,
               "F1": data.get("F1"),
               "F2": "not applicable" if not sole else "T01" in data["user_tags"] if "user_tags" in data else None,
               "F3": prev in data["reply_tags"] if prev and "reply_tags" in data else None,
               "F4": current in data["user_tags"] if current and "user_tags" in data else None}
        for f in CHECKS:
            value = row[f]
            key = "present" if value is True else "absent" if value is False else \
                  "not_applicable" if value == "not applicable" or (f == "F3" and prev is None) else "unknown"
            totals[f][key] += 1
        rows.append(row)
    return {"rows": rows, "totals": totals, "compactions": len(rows),
            "no_request": sum(r["status"] == "no request" for r in rows),
            "survival_fits": sum(r["kind"] == "survival fit" or r["survival_fit"] for r in rows),
            "scenario": cell.get("continuity") or {},
            "scenario_observed": len([r for r in rows if r["kind"] != "forced"]) >= 2 if
            (cell.get("continuity") or {}).get("sole_user") else
            any(r["kind"] == "survival fit" or r["survival_fit"] for r in rows) if
            (cell.get("continuity") or {}).get("require_survival_fit") else
            any(r["kind"] == "forced" and r["status"] == "observed" for r in rows) if
            (cell.get("continuity") or {}).get("forced_followup") else None}


def score_dir(cell: dict, directory: Path) -> dict:
    return score(cell, read(directory / "provider-requests.jsonl"), read(directory / "observer.jsonl"))
