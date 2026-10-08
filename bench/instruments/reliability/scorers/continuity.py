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
# #659 probes: the host's todo fold header (tools/todo_tool.py TODO_INJECTION_HEADER; LCM-X protects the same row).
TODO_HEADER = "[Your active task list was preserved across context compression]"


def carrier(m: dict, at: int, carriers) -> str:
    """The class of the row that holds a nonce at offset ``at``: system / user_raw / summary_or_head (the plugin's own
    part header or preserved prefix, plugin_tree.carrier_markers) / todo_fold (after the fold header) / tool_result /
    assistant. Without the plugin's markers a summary row reads as user_raw."""
    role, text = m.get("role"), m.get("content") or ""
    if role in ("system", "tool"):
        return "system" if role == "system" else "tool_result"
    fold = text.find(TODO_HEADER)
    if 0 <= fold <= at:
        return "todo_fold"
    header, prefixes = carriers or (None, ())
    if (header and header.search(text)) or text.lstrip().startswith(tuple(p for p in prefixes if p != TODO_HEADER)):
        return "summary_or_head"
    return "assistant" if role == "assistant" else "user_raw"


def nonce_markers(messages: list[dict], nonces, tool_args=(), carriers=None) -> dict:
    """Per probe nonce: presence, count and carrier classes in this request (synthetic nonces only, no text)."""
    out = {}
    for n in nonces:
        hits = Counter()
        for m in messages:
            for found in re.finditer(re.escape(n), m.get("content") or ""):
                hits[carrier(m, found.start(), carriers)] += 1
        hits["tool_args"] += sum(a.count(n) for a in tool_args)
        hits = {k: v for k, v in hits.items() if v}
        out[n] = {"present": bool(hits), "count": sum(hits.values()), "carriers": hits}
    return out


def markers(messages: list[dict], *, anchor=None, previous=None, current=None, nonces=(), tool_args=(),
            carriers=None) -> dict:
    """No request content escapes: only booleans, synthetic tags and role/counts."""
    users, replies, roles = set(), set(), []
    for m in messages:
        text = m.get("content") or ""
        u, r = set(USER.findall(text)), set(REPLY.findall(text))
        users.update(u)
        replies.update(r)
        roles.append({"role": m.get("role"), "user_tags": len(u), "reply_tags": len(r)})
    last_user = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
    return {"F1": any(m.get("role") == "system" and NOTE in (m.get("content") or "") for m in messages),
            "F2": anchor in users if anchor else None, "F3": previous in replies if previous else None,
            "F4": current in users if current else None,
            "current_user_projected": last_user.startswith("[LCM survival fit:"),
            "user_tags": sorted(users), "reply_tags": sorted(replies), "message_roles": roles,
            "role_counts": dict(Counter(m.get("role") for m in messages)),
            **({"nonces": nonce_markers(messages, nonces, tool_args, carriers)} if nonces else {})}


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
    rows, pairs = [], []
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
               "current_user_projected": data.get("current_user_projected"),
               "survival_fit": c.get("survival_fit", False), "fit_reason": c.get("fit_reason"),
               "previous_reply_in_compacted_span": prev in c["compacted_reply_tags"]
               if prev and "compacted_reply_tags" in c else None,
               "F1": data.get("F1"),
               "F2": "not applicable" if not sole else "T01" in data["user_tags"] if "user_tags" in data else None,
               "F3": prev in data["reply_tags"] if prev and "reply_tags" in data else None,
               "F4": data.get("F4") if "F4" in data else
               current in data["user_tags"] if current and "user_tags" in data else None}
        for f in CHECKS:
            value = row[f]
            key = "present" if value is True else "absent" if value is False else \
                  "not_applicable" if value == "not applicable" or (f == "F3" and prev is None) else "unknown"
            totals[f][key] += 1
        rows.append(row)
        pairs.append((c, req, row))
    result = {"rows": rows, "totals": totals, "compactions": len(rows),
            "no_request": sum(r["status"] == "no request" for r in rows),
            "survival_fits": sum(r["kind"] == "survival fit" or r["survival_fit"] for r in rows),
            "scenario": cell.get("continuity") or {},
            "scenario_observed": len([r for r in rows if r["kind"] != "forced"]) >= 2 if
            (cell.get("continuity") or {}).get("sole_user") else
            any(r["kind"] == "survival fit" or r["survival_fit"] for r in rows) if
            (cell.get("continuity") or {}).get("require_survival_fit") else
            any(r["kind"] == "forced" and r["status"] == "observed" for r in rows) if
            (cell.get("continuity") or {}).get("forced_followup") else None}
    if (cell.get("continuity") or {}).get("probes"):
        seen = [c for c, _, r in pairs if r["kind"] != "forced" and r["status"] == "observed"]
        result["scenario_observed"] = len(seen) >= 2
        if (cell["continuity"].get("soul") or {}).get("rewrite"):  # P1: the new line is tested only by a later commit
            rewrite = next((o["ts"] for o in observations if o.get("kind") == "probe_rewrite"), None)
            result["scenario_observed"] = result["scenario_observed"] and rewrite is not None and any(
                c.get("ts") is not None and c["ts"] >= rewrite for c in seen)
        result["probes"] = score_probes(cell, pairs, main, observations)
    return result


def probe_check(spec: dict, nonces: dict) -> bool:
    hit = nonces.get(spec["nonce"]) or {}
    n = (hit.get("carriers") or {}).get(spec["carrier"], 0) if spec.get("carrier") else hit.get("count", 0)
    return (n == 1 if spec.get("once") else n > 0) if spec["expect"] == "present" else n == 0


def score_probes(cell: dict, pairs: list, main: list[dict], observations: list[dict]) -> dict:
    """#659, recorded only: per probe, every compaction event (the row's first main request) checks each applicable
    nonce spec (``from_turn``/``to_turn``; ``window`` before/after the SOUL.md rewrite, before when it never ran). No request or no markers is unknown,
    never met. ``host-instruction-rebuild`` checks every main request between the rewrite and the next commit: the
    cached prompt must still carry the old line and not yet the new one."""
    scen = cell["continuity"]
    rewrite = next((o["ts"] for o in observations if o.get("kind") == "probe_rewrite"), None)
    out = {}
    for spec in scen["probes"]:
        out.setdefault(spec["probe"], {"events": [], "met": 0, "missed": 0, "unknown": 0, "not_applicable": 0})
    for c, req, row in pairs:
        nonces = ((req or {}).get("continuity") or {}).get("nonces")
        turn = (req or {}).get("turn") or c.get("turn") or 0
        window = None if c.get("ts") is None else "after" if rewrite is not None and c["ts"] >= rewrite else "before"
        for name, p in out.items():
            specs = [s for s in scen["probes"] if s["probe"] == name and s.get("from_turn", 1) <= turn <= s.get("to_turn", turn)
                     and s.get("window") in (None, window)]
            if not specs:
                p["not_applicable"] += 1
                continue
            ok = None if nonces is None else all(probe_check(s, nonces) for s in specs)
            p["met" if ok else "missed" if ok is False else "unknown"] += 1
            p["events"].append({"turn": row["turn"], "request_turn": turn, "kind": row["kind"], "status": row["status"],
                                "ok": ok, "nonces": {} if nonces is None else {s["nonce"]: {
                                    "expect": s["expect"], "present": (nonces.get(s["nonce"]) or {}).get("present"),
                                    "carriers": (nonces.get(s["nonce"]) or {}).get("carriers")} for s in specs}})
    soul = scen.get("soul") or {}
    if rewrite is not None and soul.get("rewrite"):
        until = min((o["ts"] for o in observations if o.get("kind") in ("compaction_committed", "survival_fit_committed")
                     and o.get("ts") is not None and o["ts"] >= rewrite), default=float("inf"))
        old = {"nonce": soul["nonce"], "expect": "present", "carrier": "system", "once": True}
        new = {"nonce": soul["rewrite"]["nonce"], "expect": "absent"}
        between = [r for r in main if rewrite < r["ts"] < until]
        checks = [None if (n := (r.get("continuity") or {}).get("nonces")) is None else
                  probe_check(old, n) and probe_check(new, n) for r in between]
        out["host-instruction-rebuild"] = {"requests": len(between), "met": checks.count(True),
                                           "missed": checks.count(False), "unknown": checks.count(None)}
    return out


def score_dir(cell: dict, directory: Path) -> dict:
    return score(cell, read(directory / "provider-requests.jsonl"), read(directory / "observer.jsonl"))
