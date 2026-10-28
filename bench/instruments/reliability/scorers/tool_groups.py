"""#559 bar: no message-sourced summary covers part of a tool group (the invariant of
tests/test_issue_559_tool_group_boundary.py). A group is an assistant row with ``tool_calls`` plus the
result rows of the same session that answer those call ids."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict


def split_groups(db_path) -> dict:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute("select store_id, session_id, role, tool_calls, tool_call_id from messages order by store_id").fetchall()
        nodes = db.execute("select node_id, source_ids from summary_nodes where source_type = 'messages'").fetchall()
    finally:
        db.close()
    results = defaultdict(list)
    for store_id, session, role, _calls, call_id in rows:
        if role == "tool" and call_id:
            results[(session, call_id)].append(store_id)
    groups = []
    for store_id, session, role, calls, _cid in rows:
        try:
            ids = [c.get("id") for c in json.loads(calls or "[]") if isinstance(c, dict)] if role == "assistant" else []
        except ValueError:
            ids = []
        if ids:
            groups.append({store_id, *(r for i in ids for r in results.get((session, i), []))})
    split = []
    for node_id, source_ids in nodes:
        try:
            covered = {int(x) for x in json.loads(source_ids or "[]") if str(x).lstrip("-").isdigit()}
        except ValueError:
            continue
        for g in groups:
            if 0 < len(g & covered) < len(g):
                split.append({"node_id": node_id, "group": sorted(g), "covered": sorted(g & covered)})
    return {"groups": len(groups), "summaries": len(nodes), "split_groups": len(split),
            "splits": sorted(split, key=lambda r: (r["node_id"], r["group"]))[:20]}
