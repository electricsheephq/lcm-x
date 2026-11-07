"""B10 (#1931): delegated (delegate_task) children, scored from the host's state.db and LCM-X's store.

A delegated child is a host session whose ``source`` is ``subagent`` (the child's platform, tools/delegate_tool.py
``_build_child_agent``; LCM-X aux_session.py ``_DELEGATE_SOURCE``) and whose parent is not a subagent session. It
roots its own lineage, so B1/B2 score its rows apart from the chat lineage; its own compression continuations still
join it. LCM-X records the parent link in lcm.db ``session_lineage`` (store.py ``record_session_lineage``, written by
aux_session.py ``DelegateChildMixin``). B10 fails on: a host child count other than the scripted delegate count; a
child with no stored rows; a child whose ``session_lineage`` row is missing or names another parent than the host's;
a child-tagged row stored in another lineage; a row stored under a child whose content is a chat-lineage row (the
parent's prompts and scripted replies) other than the child's own goal. A scripted delegate whose child never ran or
did not complete, or a child the host did not record as a subagent session, is a scenario gap (UNSUPPORTED).
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

from . import multiset, tool_calls

SOURCE = "subagent"
CHILD_TAG = r"\[(D\d{2,3})\] user turn"  # probe.py CHILD_TAG
CHILD_REPLY = r"reply to (D\d{2,3})\b"


def host_sessions(state: Path) -> dict:
    """state.db session id -> (parent_session_id, source); {} without a state.db, source None without the column."""
    if not state.exists():
        return {}
    con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("pragma table_info(sessions)")}
        return {sid: (parent, source) for sid, parent, source in con.execute(
            "select id, parent_session_id, " + ("source" if "source" in cols else "NULL") + " from sessions")}
    finally:
        con.close()


def is_child_root(sid: str, sessions: dict) -> bool:
    parent, source = sessions.get(sid, (None, None))
    return bool(parent) and source == SOURCE and sessions.get(parent, (None, None))[1] != SOURCE


def links(lcm: Path) -> dict:
    """lcm.db ``session_lineage``: child -> (parent, source); {} before the first link creates the table."""
    con = sqlite3.connect(f"file:{lcm}?mode=ro", uri=True)
    try:
        if not con.execute("select 1 from sqlite_master where type='table' and name='session_lineage'").fetchone():
            return {}
        return {c: (p, s) for c, p, s in con.execute(
            "select child_session_id, parent_session_id, source from session_lineage")}
    finally:
        con.close()


def child_attempts(events: list[dict]) -> list[dict]:
    """The observer's ``child_user_sent`` / ``child_turn_end`` and the provider's ``child_emit``, as bars.attempts
    attempts with prefix "D" and the child's host session."""
    out, open_ = [], {}
    for e in events:
        if e["event"] == "child_user_sent" and e.get("tag"):
            content = e.get("content") or ""
            a = {"tag": e["tag"], "prefix": "D", "content": content, "persist": content, "held": None, "reply": None,
                 "user_tags": {}, "ended": False, "turn": e.get("turn"), "notices": [], "emitted": [],
                 "tool_issues": [], "tool_dispatch": [], "tool_seen": [], "session": e.get("session")}
            out.append(a)
            open_[e["tag"]] = a
        elif e["event"] == "child_emit" and e.get("tag") in open_:
            open_[e["tag"]]["emitted"].append(e["text"])
        elif e["event"] == "child_turn_end" and e.get("tag") in open_:
            a = open_.pop(e["tag"])
            a.update(held=e.get("held"), ended=True, end=e, user_tags={a["tag"]: 1} if e.get("held") else {})
            a["reply"] = a["emitted"][-1] if tool_calls.completed(a) and a["emitted"] else None
    return out


def score(cell: dict, events: list[dict], atts: list[dict], per: dict, full: list, db_dir: Path, group):
    """(numbers, failed or None, gaps). ``per``: lineage -> (expected items, stored rows) from bars.score."""
    sessions = host_sessions(db_dir / "state.db")
    roots = {sid: v[0] for sid, v in sessions.items() if is_child_root(sid, sessions)}
    kids = [a for a in atts if a["prefix"] == "D"]
    planned = sum(len(g["turns"]) for g in cell.get("delegate_plan", []))
    issued = [m.group(1) for e in events if e["event"] == "tool_issue" and e.get("name") == "delegate_task"
              for m in [re.search(CHILD_TAG, str((e.get("args") or {}).get("goal", "")))] if m]
    gaps = [] if len(issued) == planned else [f"B10: {planned} delegates scripted, {len(issued)} issued"]
    failed = {}
    for ctag in issued:
        ran = [a for a in kids if a["tag"] == ctag and tool_calls.completed(a) and a.get("session")]
        # A child that started and ended failed is a FAIL (it may be LCM-X failing inside the child); one that never
        # started, never ended or was cancelled leaves the scenario unproven.
        if not ran and any(a["tag"] == ctag and a["ended"] and (a.get("end") or {}).get("failed") for a in kids):
            failed.setdefault("children_failed", []).append(ctag)
        elif not ran:
            gaps.append(f"B10: {ctag}: the host never ran or completed the delegated child")
        gaps += [f"B10: {ctag}: host session {a['session']} is not a subagent child in state.db"
                 for a in ran if a["session"] not in roots]
    linked = links(db_dir / "lcm.db")
    child_lineages = {group(sid) for sid in roots}
    stored = Counter(group(r[1]) for r in full)
    numbers = {"scripted": planned, "issued": len(issued), "host_children": len(roots), "children": {}}
    if len(roots) != len(issued):
        failed["child_count_mismatch"] = {"issued": len(issued), "host_children": len(roots)}
    for sid, parent in sorted(roots.items()):
        link = linked.get(sid)
        numbers["children"][sid] = {"host_parent": parent, "stored_rows": stored[group(sid)],
                                    "link": list(link) if link else None}
        if not stored[group(sid)]:
            failed.setdefault("children_without_rows", []).append(sid)
        if not link or link[0] != parent:
            failed.setdefault("children_without_parent_link", []).append(sid)
    misplaced = [r[0] for r in full if group(r[1]) not in child_lineages and r[2] in ("user", "assistant")
                 and (re.search(CHILD_TAG, r[3] or "") or re.search(CHILD_REPLY, r[3] or ""))]
    goals = {multiset.h(a["held"] if a["held"] is not None else a["persist"]) for a in kids}
    parent_keys = {multiset.h(text) for _role, text in per.get("chat", ([], []))[0] if multiset.norm(text)} - goals
    copies = [r[0] for r in full if group(r[1]) in child_lineages and multiset.norm(r[3])
              and multiset.h(r[3]) in parent_keys]
    numbers.update(child_rows_outside_child_lineages=len(misplaced), parent_rows_copied_into_children=len(copies))
    if misplaced:
        failed["child_rows_outside_child_lineages"] = misplaced[:10]
    if copies:
        failed["parent_rows_copied_into_children"] = copies[:10]
    return numbers, (dict(failed, **{k: numbers[k] for k in ("scripted", "issued", "host_children")}) if failed else None), gaps
