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
import sqlite3
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


# Hosts without agent/turn_preflight.py's re-anchor after a post-tool compaction (0.21.5 adds it): upstream v2026.9.14
# (0.21.3) and v2026.9.21 (0.21.4). Only there does bind() license a dangling-tail notice.
HOSTS_WITHOUT_POST_TOOL_REANCHOR = frozenset({"345cd2b057a452236de401d3534b8502a7465e8d",
                                              "d337b736aa1e8ebecfab043842d13e4a2d2f48a3"})


def _dispatches(a, gaps, failures):
    """Validate each dispatch once; yield matched calls in issue order, then report extras."""
    free = list(a.get("tool_dispatch", []))
    for c in a.get("tool_issues", []):
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
        yield c, d
    failures += [f"{a['tag']}: unplanned host dispatch of {d['name']}" for d in free]


def bind(atts: list[dict], lineage_of=lambda a: "chat", *, license_wire_notices: bool = False) -> dict:
    """Per completed attempt: gaps (dispatch never happened -> UNSUPPORTED), failures (wrong/failed/incomplete/unseen
    result, unplanned dispatch -> FAIL) and the expected keys. Calls of unfinished attempts may be stored at most once.

    ``wire_notices``: floor-0.21.3 (345cd2b0) does not re-anchor the turn after a post-tool compaction, so its request
    copy replaces this turn's live results with its own dangling-tail notice (agent/replay_cleanup.py) while the host
    persists the dispatched result. Only for a completed dispatch with a ``result_sha256`` that the probe saw so
    flagged (``host_wire_notice``) is the expected key the dispatch's result, and only with ``license_wire_notices``
    (a host in HOSTS_WITHOUT_POST_TOOL_REANCHOR); each such call is recorded. The licence does not depend on the
    result's semantic validity (``ok``, #1098): a failed result is its own failure above."""
    gaps, failures, expected, loose, wire_notices = [], [], Counter(), set(), []
    for a in atts:
        issues, g = a.get("tool_issues", []), lineage_of(a)
        if not completed(a):
            for c in issues:
                loose |= {(g, "call", c["id"], c["name"], canon(c["args"]))}
                loose |= {k for k in [(g, "tool", c["id"], s["sha"]) for s in a.get("tool_seen", []) if s["id"] == c["id"]]}
            continue
        seen = {s["id"]: s for s in a.get("tool_seen", [])}
        for c, d in _dispatches(a, gaps, failures):
            expected[(g, "call", c["id"], c["name"], canon(c["args"]))] += 1
            if license_wire_notices and c["id"] in seen and seen[c["id"]].get("host_wire_notice") \
                    and d.get("result_sha256"):
                expected[(g, "tool", c["id"], d["result_sha256"])] += 1
                wire_notices.append({"id": c["id"], "notice_sha256": seen[c["id"]]["sha"], "result_sha256": d["result_sha256"]})
            elif c["id"] in seen:
                expected[(g, "tool", c["id"], seen[c["id"]]["sha"])] += 1
            else:
                failures.append(f"{a['tag']}: the result of {c['id']} never reached the next provider request")
    return {"gaps": gaps, "failures": failures, "expected": expected, "loose": loose, "wire_notices": wire_notices}


def stored_keys(rows, lineage=lambda sid: "chat", *, store_ids=None) -> Counter:
    """rows: (store_id, session_id, role, content, tool_calls, tool_call_id). An assistant row whose ``tool_calls``
    is not a JSON list of objects is one ``(lineage, "malformed_tool_calls", store_id)`` key: never expected, so it is
    B2 surplus, never silently "no calls". Only SQL NULL means no calls: ``''`` is malformed text."""
    keys = Counter()

    def record(key, store_id):
        keys[key] += 1
        if store_ids is not None:
            store_ids.setdefault(key, []).append(store_id)

    for store_id, session, role, content, calls, call_id in rows:
        g = lineage(session)
        if role == "tool":
            record((g, "tool", call_id, hashlib.sha256((content or "").encode()).hexdigest()), store_id)
        elif role == "assistant" and calls is not None:
            try:
                entries = json.loads(calls)
            except ValueError:
                entries = None
            if not isinstance(entries, list) or not all(isinstance(c, dict) for c in entries):
                record((g, "malformed_tool_calls", store_id), store_id)
                continue
            for c in entries:
                fn = c.get("function") or {}
                record((g, "call", c.get("id"), fn.get("name"), canon(fn.get("arguments") or "{}")), store_id)
    return keys


def host_held(atts, bound, state, group, turns, lineage_of):
    """Validate dispatches and require host-held keys of interrupted held-fault tool turns, once per distinct key."""
    selected = [a for a in atts if a["prefix"] == "T" and not a["ended"] and a["turn"] in turns]
    if not selected:
        return None
    for a in selected:
        # Exhaust dispatch validation even when the host DB is unavailable.
        list(_dispatches(a, bound["gaps"], bound["failures"]))
    try:
        con = sqlite3.connect(state.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT id, session_id, role, content, tool_calls, tool_call_id FROM messages "
                               "WHERE role = 'tool' OR (role = 'assistant' AND tool_calls IS NOT NULL)").fetchall()
        finally:
            con.close()
    except sqlite3.Error as exc:
        return {"unavailable": "state.db unreadable: " + type(exc).__name__}
    keys, records = stored_keys(rows, group), []
    for a in selected:
        g = lineage_of(a)
        for c in a["tool_issues"]:
            bound["loose"] = {k for k in bound["loose"] if not (k[0] == g and k[2] == c["id"])}
            for k in keys:
                if k[:3] not in ((g, "call", c["id"]), (g, "tool", c["id"])):
                    continue
                bound["expected"][k] = 1
                records.append(dict(tag=a["tag"], session=g, id=c["id"], kind=k[1],
                                    key=hashlib.sha256(json.dumps(k[1:]).encode()).hexdigest()[:12]))
    return {"rows": len(records), "records": records}


def compare(expected: Counter, loose: set, stored: Counter, *, store_ids=None, surplus_keys=None) -> dict:
    missing = {k: n - stored[k] for k, n in expected.items() if stored[k] < n}
    surplus = {k: stored[k] - expected[k] - (1 if k in loose else 0) for k in stored
               if stored[k] > expected[k] + (1 if k in loose else 0)}
    if surplus_keys is not None:
        for key, n in surplus.items():
            role = "tool" if key[1] == "tool" else "assistant"
            digest = hashlib.sha256(json.dumps(key[1:], sort_keys=True).encode()).hexdigest()
            surplus_keys.append(((role, digest), n, list(store_ids[key])))
    return {"tool_items": sum(expected.values()), "tool_missing_rows": sum(missing.values()),
            "tool_surplus_rows": sum(surplus.values()),
            "tool_missing": [list(k[:4]) for k in sorted(missing, key=json.dumps)[:5]],
            "tool_surplus": [list(k[:4]) for k in sorted(surplus, key=json.dumps)[:5]]}
