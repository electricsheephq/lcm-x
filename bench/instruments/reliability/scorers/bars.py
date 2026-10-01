"""Bars B1-B8 over one finished cell: its DB copy (db/lcm.db), transcript.jsonl and phase-*.json.

The expected transcript is what the host HELD for each attempt (the probe records the user row the host
kept after its persist override and consecutive-user merge), so the bars compare LCM's store with the
host, never with LCM itself. An attempt with no stored reply whose prompt the next attempt of the same
session folded into a composite row (and no other held row still carries its tag) is superseded by that
composite: that is the host merge (agent/agent_runtime_helpers.py ``_merge_consecutive_users``).

B1/B2 are scored per session lineage: the chat lineage is S0 and its compression children (state.db
``parent_session_id``), each cron fire is its own lineage. A row stored under the wrong lineage is a loss in
one and a surplus in the other. A user-row surplus the host's own state.db holds at the same multiplicity is
licensed and reported (``host_parity_licensed``, scorers/host_parity.py). A cell that cannot prove its scenario
ran (tool dispatch, native attempts, tool groups) is UNSUPPORTED, never PASS.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from . import chronology, host_parity, multiset, summary, tool_calls, tool_groups

ALL_BARS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")


def _candidate_phases(cell: dict, phases: list[dict]) -> list[dict]:
    """A native-on-off cell's (``from_ref``) candidate phases: every phase after the plugin switch."""
    post = phases[next((i + 1 for i, p in enumerate(phases) if p.get("exit") == "plugin_switch"), len(phases)):]
    if not post:
        raise ValueError("native-on-off cell has no candidate phase after plugin_switch (never scored as 0)")
    return post


def counted(cell: dict, phases: list[dict], key: str) -> int:
    """A log count over the cell's phases; a native-on-off cell (``from_ref``) counts EVERY event of the
    candidate's phases, before its first publication too (the D2 bar: 0 conflicts on the migration path);
    the older ref's phase is not the candidate's. See :func:`pre_publication` for the diagnostic split."""
    if not cell.get("from_ref"):
        return sum(p.get("log_counts", {}).get(key, 0) for p in phases)
    return sum(p.get("log_counts", {}).get(key, 0) for p in _candidate_phases(cell, phases))


def pre_publication(cell: dict, phases: list[dict], key: str) -> int | None:
    """Diagnostic (native-on-off): the candidate's ``key`` events before its first publication, from the
    probe's ``log_counts_after_commit``; None when no candidate phase records it."""
    post = _candidate_phases(cell, phases)
    if not any("log_counts_after_commit" in p for p in post):
        return None
    first = next((i for i, p in enumerate(post) if p.get("log_counts_after_commit") is not None), None)
    total = sum(p.get("log_counts", {}).get(key, 0) for p in post)
    if first is None:
        return total
    after = sum((p["log_counts_after_commit"] if i == first else p.get("log_counts", {})).get(key, 0)
                for i, p in enumerate(post) if i >= first)
    return total - after


def load(cell_dir: Path):
    events = [json.loads(x) for x in (cell_dir / "transcript.jsonl").read_text().splitlines() if x.strip()]
    phases = [json.loads(p.read_text()) for p in sorted(cell_dir.glob("phase-*.json"), key=lambda p: (len(p.name), p.name))]
    return events, phases


def attempts(events: list[dict]) -> list[dict]:
    """Attempts with what the host held (user row, notices) and, independently, what the provider emitted and
    which tools the host really ran."""
    out, open_, last = [], {}, None
    for e in events:
        if e["event"] in ("user_sent", "retry"):
            a = {"tag": e["tag"], "prefix": e.get("session_prefix", "T"), "content": e["content"],
                 "persist": e["persist"] if e.get("persist") is not None else e["content"],
                 "held": None, "reply": None, "user_tags": {}, "ended": False, "turn": e.get("turn"), "notices": [],
                 "emitted": [], "tool_issues": [], "tool_dispatch": [], "tool_seen": []}
            out.append(a)
            open_[e["tag"]] = last = a
        elif e["event"] == "emit" and e.get("tag") in open_:
            open_[e["tag"]]["emitted"].append(e["text"])
        elif e["event"] in ("tool_issue", "tool_dispatch") and e.get("tag") in open_:
            open_[e["tag"]][e["event"] if e["event"] == "tool_dispatch" else "tool_issues"].append(e)
        elif e["event"] == "tool_seen" and last is not None and not last["ended"]:
            last["tool_seen"].append(e)
        elif e["event"] == "turn_end" and e["tag"] in open_:
            a = open_.pop(e["tag"])
            a.update(held=a["persist"] if e.get("held_same") else e.get("held"), user_tags=e.get("user_tags") or {},
                     ended=True, end=e, notices=e.get("host_notices") or [])
            a["reply"] = a["emitted"][-1] if tool_calls.completed(a) and a["emitted"] else None
    return out


def lineage(cell_dir: Path, root: str = "S0"):
    """store session id -> lineage: "chat" for ``root`` (R1: S0; R2: the ACP session id) and its compression
    descendants, else the root session."""
    parents, state = {}, cell_dir / "db" / "state.db"
    if state.exists():
        con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
        try:
            parents = dict(con.execute("select id, parent_session_id from sessions"))
        finally:
            con.close()

    def group(sid):
        seen = set()
        while parents.get(sid) and sid not in seen:
            seen.add(sid)
            sid = parents[sid]
        return "chat" if sid == root else sid
    return group


def attempt_group(a: dict, group) -> str:
    return "chat" if a["prefix"] == "T" else group((a.get("end") or {}).get("session") or f"cron_job_{int(a['tag'][1:]):02d}")


def continue_positions(seq: list[tuple[str, str]], reply_pat: str) -> Counter:
    """For every 'continue' user row, the reply tag of the row right after it (None if not a reply)."""
    out = Counter()
    for i, (role, text) in enumerate(seq):
        if role == "user" and multiset.norm(text) == "continue":
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            m = re.search(reply_pat, nxt[1] or "") if nxt and nxt[0] == "assistant" else None
            out[m.group(1) if m else None] += 1
    return out


def scenario_gaps(cell: dict, events: list[dict], atts: list[dict], tg: dict, bound: dict) -> list[str]:
    """What the cell was meant to exercise and did not: each gap makes the cell UNSUPPORTED."""
    gaps = [f"{a['tag']}: completed but the provider never returned its scripted reply" for a in atts
            if tool_calls.completed(a) and not a["emitted"]] + bound["gaps"]
    if cell.get("tool_plan"):
        for a in atts:
            if a["prefix"] != "T" or not tool_calls.completed(a):
                continue
            want = sum(len(g["calls"]) for g in cell["tool_plan"] if isinstance(g["turns"], list) and a["turn"] in g["turns"])
            if want and len(a["tool_issues"]) != want:
                gaps.append(f"{a['tag']}: {want} tool calls planned, {len(a['tool_issues'])} issued")
        if "B6" in (cell.get("bars") or ALL_BARS) and not tg["groups"]:
            gaps.append("B6: no tool-call group was stored, so the split check had nothing to check")
    if cell.get("native_recovery"):
        tried = sum(e.get("native_attempts", 0) for e in events if e["event"] == "turn_end") + \
            sum(1 for e in events if e["event"] == "compaction" and e.get("compression_status") == "host_native")
        if not tried:
            gaps.append("native cell with zero native recovery attempts")
    return gaps


def expected_items(atts: list[dict], notices=()) -> list[tuple[str, str]]:
    """Held user rows; exactly one scripted reply per completed non-cancel attempt (from the provider log, so a
    reply the host dropped is a deficit); the host's cited failed-turn copy only on a verified interrupted attempt."""
    items = []
    for i, a in enumerate(atts):
        text = a["held"] if a["held"] is not None else a["persist"]
        nxt = next((b for b in atts[i + 1:] if b["prefix"] == a["prefix"]), None)
        cand = multiset.norm(text)
        folded = (nxt is not None and not a["reply"] and nxt["held"] is not None and cand
                  and cand in multiset.norm(nxt["held"]) and cand != multiset.norm(nxt["held"])
                  and nxt["user_tags"].get(a["tag"], 1) <= 1)
        if not folded:
            items.append(("user", text))
        if a["reply"]:
            items.append(("assistant", a["reply"]))
        if (a.get("end") or {}).get("interrupted"):
            items += [("assistant", x) for x in a["notices"] if x.strip() in notices]
    return items


def source_ids(src) -> list:
    """A summary node's ``source_ids`` JSON list; unparseable or non-list values are an empty source."""
    try:
        ids = json.loads(src) if src else []
    except ValueError:
        return []
    return ids if isinstance(ids, list) else []


def tag_counts(texts, pattern):
    counts = Counter()
    for text in texts:
        for tag in set(re.findall(pattern, text or "")):
            counts[tag] += 1
    return counts


def score(cell: dict, cell_dir: Path) -> dict:
    events, phases = load(cell_dir)
    db = cell_dir / "db" / "lcm.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        full = con.execute("select store_id, session_id, role, content, tool_calls, tool_call_id from messages"
                           " order by store_id").fetchall()
        stored = [r[:4] for r in full]
        nodes = con.execute("select node_id, session_id, source_ids from summary_nodes where source_type = 'messages'").fetchall()
    finally:
        con.close()
    atts = attempts(events)
    group = lineage(cell_dir, cell.get("chat_root", "S0"))
    groups = sorted({attempt_group(a, group) for a in atts} | {group(sid) for _s, sid, _r, _c in stored})
    notices = {x for p in phases for x in p.get("failed_turn_notices") or []}
    plugin = cell.get("plugin") or (json.loads((cell_dir / "cell.json").read_text()).get("plugin")
                                    if (cell_dir / "cell.json").exists() else None) or {}
    host, host_why = host_parity.load(cell_dir / "db" / "state.db", group, plugin.get("tree"))
    per = {g: (expected_items([a for a in atts if attempt_group(a, group) == g], notices),
               [r for r in stored if group(r[1]) == g]) for g in groups}
    applicable = [b for b in cell.get("bars") or ALL_BARS
                  if (b != "B6" or cell.get("tool_plan")) and (b != "B7" or cell.get("native_recovery"))
                  and (b != "B5" or cell.get("min_compactions", 5) > 0)]
    failed, numbers = {}, {}

    user_pat, reply_pat = r"\[([A-Z]\d{2,3})\] user turn", r"reply to ([A-Z]\d{2,3})\b"
    b1, b1_tags, b1_per, b2_parts, b1_lic = {}, [0, 0], {}, {}, []
    for g, (items, rows) in per.items():
        label = "" if g == "chat" else f"{g}:"
        want_u = tag_counts([t for r, t in items if r == "user"], user_pat)
        want_a = tag_counts([t for r, t in items if r == "assistant"], reply_pat)
        have_u = tag_counts([c for _s, _sid, r, c in rows if r == "user"], user_pat)
        have_a = tag_counts([c for _s, _sid, r, c in rows if r == "assistant"], reply_pat)
        hg = None if host is None else host.get(g, {})
        mism = {}
        for k in set(want_u) | set(have_u):
            ids = [s for s, _sid, r, c in rows if r == "user" and k in re.findall(user_pat, c or "")] \
                if have_u[k] > want_u[k] else []
            lic = host_parity.b1_licence(k, want_u[k], have_u[k], hg, ids)
            b1_lic += [dict(lic, session=g)] if lic else []
            if want_u[k] + (lic or {}).get("licensed", 0) != have_u[k]:
                mism[f"{label}{k}"] = {"expected": want_u[k], "stored": have_u[k]}
        mism.update({f"reply {label}{k}": {"expected": want_a[k], "stored": have_a[k]}
                     for k in set(want_a) | set(have_a) if want_a[k] != have_a[k]})
        want_c = continue_positions(items, reply_pat)
        have_c = continue_positions([(r, c) for _s, _sid, r, c in rows if r in ("user", "assistant")], reply_pat)
        if want_c != have_c:
            mism[f"{label}continue"] = {"expected_replies_after": dict(want_c), "stored_replies_after": dict(have_c)}
        b1.update(mism)
        b1_tags[0] += len(want_u)
        b1_tags[1] += len(want_a)
        b1_per[g] = len(mism)
        b2_parts[g] = multiset.score(items, rows, None if host is None else host.get(g, {}).get("keys", {}))
    numbers["B1"] = {"user_tags": b1_tags[0], "reply_tags": b1_tags[1], "mismatched": len(b1), "per_session": b1_per,
                     "host_parity_licensed": host_parity.summary(b1_lic, host_why)}
    if b1:
        failed["B1"] = dict(sorted(b1.items())[:30])
    keys = ("expected_items", "missing_keys", "deficit_rows", "duplicated_keys", "surplus_rows",
            "stored_rows_not_expected", "split_keys")
    numbers["B2"] = {k: sum(m[k] for m in b2_parts.values()) for k in keys}
    numbers["B2"]["per_session"] = {g: m["verdict"] for g, m in b2_parts.items()}
    numbers["B2"]["host_parity_licensed"] = host_parity.summary(
        [dict(r, session=g) for g, m in b2_parts.items() for r in m["host_parity_licensed"]], host_why)
    bound = tool_calls.bind(atts, lambda a: attempt_group(a, group))
    tools = tool_calls.compare(bound["expected"], bound["loose"], tool_calls.stored_keys(full, group))
    numbers["B2"].update(tools)
    if tools["tool_missing_rows"] or tools["tool_surplus_rows"]:
        failed["B2"] = dict(numbers["B2"])
    if any(m["verdict"] != "PASS" for m in b2_parts.values()):
        bad = {g: m for g, m in b2_parts.items() if m["verdict"] != "PASS"}
        failed["B2"] = {**numbers["B2"], "missing": [dict(e, session=g) for g, m in bad.items() for e in m["missing"]][:5],
                        "duplicated": [dict(e, session=g) for g, m in bad.items() for e in m["duplicated"]][:5],
                        "extra": [dict(e, session=g) for g, m in bad.items() for e in m["extra"]][:5]}
    conflicts = counted(cell, phases, "publication_invariant_conflict")
    numbers["B3"] = {"publication_invariant_conflict": conflicts}
    if conflicts:
        failed["B3"] = numbers["B3"]
    fits = counted(cell, phases, "survival_fit")  # #582: a fit is a compaction miss
    exit_fits = counted(cell, phases, "exit_fit")  # diagnostic only: exit headroom does not fail B8
    skipped = counted(cell, phases, "exit_fit_skipped")
    numbers["B8"] = {"survival_fit": fits, **({"exit_fit": exit_fits} if exit_fits else {}),
                     **({"exit_fit_skipped": skipped} if skipped else {})}
    if fits:
        failed["B8"] = numbers["B8"]
    if cell.get("from_ref"):  # diagnostic only: how many of those came before the candidate first published
        numbers["pre_publication_counts"] = {key: pre_publication(cell, phases, key)
                                             for key in ("publication_invariant_conflict", "survival_fit")}
    failed_turns = [t for p in phases for t in p.get("counters", {}).get("failed", [])]
    final = next((p["final_check"] for p in reversed(phases) if "final_check" in p), None)
    numbers["B4"] = {"failed_turns": failed_turns, "final_check": final}
    inconclusive = {}
    outcome = (final or {}).get("outcome", "published" if (final or {}).get("published") else "failed")
    if failed_turns or (cell.get("final_compaction_check", True) and outcome == "failed"):
        failed["B4"] = numbers["B4"]
    elif cell.get("final_compaction_check", True) and outcome == "inconclusive":
        inconclusive["B4"] = f"forced compaction ended {final.get('engine_status')!r} ({final.get('noop_reason')!r}) " \
                             f"after {len(final.get('attempts', []))} attempt(s) of {final.get('entry')}; backlog checks " \
                             f"{[c.get('turns_since_pass') for c in final.get('backlog_checks') or []]}"
    grow = summary.growth(events, sum(p.get("compactions_logged", 0) for p in phases), cell.get("min_compactions", 5))
    session_of = {r[0]: r[1] for r in full}
    crossing, empty = [], []
    for nid, sid, src in nodes:  # a summary must cover >= 1 stored row, and only rows of its own session lineage
        ids = source_ids(src)
        if not any(type(i) is int and i in session_of for i in ids):  # a JSON true is not store id 1
            empty.append(nid)  # counted toward depth-0 growth, yet it summarises no stored message
        if any(group(session_of.get(i, f"missing:{i}") if type(i) is int else f"invalid:{i}") != group(sid)
               for i in ids):
            crossing.append(nid)
    grow["cross_lineage_nodes"], grow["empty_source_nodes"] = crossing[:10], empty[:10]
    numbers["B5"] = grow
    if not grow["ok"] or crossing or empty:
        failed["B5"] = {k: v for k, v in grow.items() if k != "depth0_sequence"} | {"depth0_tail": grow["depth0_sequence"][-8:]}
    tg = tool_groups.split_groups(db)
    hooked = all("orphan_hook" not in p for p in phases)
    orphans = sum(p.get("counters", {}).get("orphan_drops", 0) if hooked else p.get("log_counts", {}).get("orphan_log", 0)
                  for p in phases)
    numbers["B6"] = {"groups": tg["groups"], "split_groups": tg["split_groups"], "host_orphan_drops": orphans,
                     "tool_results": sum(1 for e in events if e["event"] == "tool_result")}
    numbers["B6"]["tool_execution_failures"] = len(bound["failures"])
    if tg["split_groups"] or orphans or bound["failures"]:
        failed["B6"] = {**numbers["B6"], "splits": tg["splits"][:3], "tool_failures": bound["failures"][:5]}
    native = {"native_unusable": sum(p.get("log_counts", {}).get("native_unusable", 0) for p in phases),
              "summary_generation_aborted": sum(p.get("log_counts", {}).get("summary_generation_aborted", 0) for p in phases),
              "max_native_attempts_per_turn": max((p.get("counters", {}).get("native_max", 0) for p in phases), default=0)}
    numbers["B7"] = native
    if native["native_unusable"] or native["summary_generation_aborted"] or native["max_native_attempts_per_turn"] > 1:
        failed["B7"] = native
    failed = {b: v for b, v in failed.items() if b in applicable}
    numbers["diagnostic"] = {
        "log_counts": {k: sum(p.get("log_counts", {}).get(k, 0) for p in phases)
                       for k in ("resident_engine_conflict", "skipped_ingest_resident_conflict", "recorded_replaced")},
        "phases": len(phases), "session_count": phases[-1].get("session_count") if phases else None,
        "lcm_tool_calls": sum(p.get("counters", {}).get("lcm_tool_calls", 0) for p in phases),
        "summary_nodes": summary.nodes_report(db), "chronology": chronology.report(stored, group)}
    inconclusive = {b: v for b, v in inconclusive.items() if b in applicable}
    numbers["diagnostic"]["native_rejections"] = dict(Counter(e.get("rejection") for e in events
                                                              if e["event"] == "compaction" and e.get("rejection")))
    gaps = scenario_gaps(cell, events, atts, tg, bound)
    if gaps:
        return {"verdict": "UNSUPPORTED", "reason": "scenario not proven: " + "; ".join(gaps[:5]),
                "applicable_bars": applicable, "failed_bars": failed, "inconclusive_bars": inconclusive, "numbers": numbers}
    return {"verdict": "FAIL" if failed else "INCONCLUSIVE" if inconclusive else "PASS", "applicable_bars": applicable,
            "failed_bars": failed, "inconclusive_bars": inconclusive, "numbers": numbers}
