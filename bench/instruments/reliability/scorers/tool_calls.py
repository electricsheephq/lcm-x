"""Tool-call fidelity: every planned call is bound to a real host dispatch and to durable rows in the DB copy.

The probe logs, independently of the host result, each call the scripted provider issued (``tool_issue``: id,
name, args), each real host execution at the cited dispatch hook (``tool_dispatch``: model_tools
``handle_function_call`` with the call id, or the context engine's ``handle_tool_call``, bound by name + args),
and each tool result as the next provider request carried it (``tool_seen``: id, sha256). Keys:
``(lineage, "call", id, name, canonical args)`` for an assistant tool-call entry and ``(lineage, "tool", id,
sha256(content))`` for a tool result row; stored and expected keys must match as a multiset within each session
lineage, like user/assistant rows.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter


def canon(args) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args
    return json.dumps(args, sort_keys=True)


def underlying(name: str, args):
    """The host's ``tool_call`` bridge (tools/tool_search.py ``resolve_underlying_call``): the dispatch hooks see the
    underlying tool while the stored call keeps the bridge. One entry, ``{name, arguments}`` or ``{calls: [entry]}``."""
    if name != "tool_call":
        return name, args
    try:
        a = json.loads(args) if isinstance(args, str) else args
        calls = a.get("calls")
        entry = a if calls is None else calls[0] if isinstance(calls, list) and len(calls) == 1 else None
        inner = entry.get("arguments")
        inner = json.loads(inner) if isinstance(inner, str) and inner.strip() else inner or {}
        return (entry["name"], inner) if entry.get("name") else (name, args)
    except (AttributeError, TypeError, ValueError):
        return name, args


def completed(a: dict) -> bool:
    end = a.get("end") or {}
    return bool(a["ended"]) and not end.get("failed") and end.get("kind") != "cancel"


def bind(atts: list[dict], lineage_of=lambda a: "chat") -> dict:
    """Per completed attempt: gaps (dispatch never happened -> UNSUPPORTED), failures (wrong/failed/incomplete/unseen
    result, unplanned dispatch -> FAIL) and the expected keys. Calls of unfinished attempts may be stored at most once."""
    gaps, failures, expected, loose = [], [], Counter(), set()
    for a in atts:
        issues, g = a.get("tool_issues", []), lineage_of(a)
        if not completed(a):
            for c in issues:
                loose |= {(g, "call", c["id"], c["name"], canon(c["args"]))}
                loose |= {k for k in [(g, "tool", c["id"], s["sha"]) for s in a.get("tool_seen", []) if s["id"] == c["id"]]}
            continue
        free = [d for d in a.get("tool_dispatch", [])]
        seen = {s["id"]: s for s in a.get("tool_seen", [])}
        for c in issues:
            name, args = underlying(c["name"], c["args"])
            d = next((d for d in free if d.get("id") == c["id"]), None) or \
                next((d for d in free if d.get("id") is None and d["name"] == name and canon(d["args"]) == canon(args)), None)
            if d is None:
                gaps.append(f"{a['tag']}: planned {c['name']} ({c['id']}) was never dispatched by the host")
                continue
            free.remove(d)
            if d["name"] != name or canon(d["args"]) != canon(args):
                failures.append(f"{a['tag']}: {c['id']} planned {c['name']} but the host ran {d['name']}")
            elif not d.get("ok"):
                failures.append(f"{a['tag']}: {c['name']} ({c['id']}) failed: {d.get('detail')}")
            elif d.get("chars", 0) < (c.get("expect") or {}).get("min_chars", 0):
                failures.append(f"{a['tag']}: {c['name']} result {d.get('chars')} chars < {c['expect']['min_chars']}")
            expected[(g, "call", c["id"], c["name"], canon(c["args"]))] += 1
            if c["id"] in seen:
                expected[(g, "tool", c["id"], seen[c["id"]]["sha"])] += 1
            else:
                failures.append(f"{a['tag']}: the result of {c['id']} never reached the next provider request")
        failures += [f"{a['tag']}: unplanned host dispatch of {d['name']}" for d in free]
    return {"gaps": gaps, "failures": failures, "expected": expected, "loose": loose}


def stored_keys(rows, lineage=lambda sid: "chat") -> Counter:
    """rows: (store_id, session_id, role, content, tool_calls, tool_call_id). An assistant row whose ``tool_calls``
    is not a JSON list of objects is one ``(lineage, "malformed_tool_calls", store_id)`` key: never expected, so it is
    B2 surplus, never silently "no calls". Only SQL NULL means no calls: ``''`` is malformed text."""
    keys = Counter()
    for store_id, session, role, content, calls, call_id in rows:
        g = lineage(session)
        if role == "tool":
            keys[(g, "tool", call_id, hashlib.sha256((content or "").encode()).hexdigest())] += 1
        elif role == "assistant" and calls is not None:
            try:
                entries = json.loads(calls)
            except ValueError:
                entries = None
            if not isinstance(entries, list) or not all(isinstance(c, dict) for c in entries):
                keys[(g, "malformed_tool_calls", store_id)] += 1
                continue
            for c in entries:
                fn = c.get("function") or {}
                keys[(g, "call", c.get("id"), fn.get("name"), canon(fn.get("arguments") or "{}"))] += 1
    return keys


def compare(expected: Counter, loose: set, stored: Counter) -> dict:
    missing = {k: n - stored[k] for k, n in expected.items() if stored[k] < n}
    surplus = {k: stored[k] - expected[k] - (1 if k in loose else 0) for k in stored
               if stored[k] > expected[k] + (1 if k in loose else 0)}
    return {"tool_items": sum(expected.values()), "tool_missing_rows": sum(missing.values()),
            "tool_surplus_rows": sum(surplus.values()),
            "tool_missing": [list(k[:4]) for k in list(missing)[:5]], "tool_surplus": [list(k[:4]) for k in list(surplus)[:5]]}
