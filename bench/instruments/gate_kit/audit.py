"""Gate-kit audit: every count of the v0.30.0 "nothing lost" gate (docs/exactly-once-ingest.md section 5, "Gate
(v0.30.0)" and "The driven run") from one driven run's Hermes state.db, LCM-X store and logs. Read-only.

    python -m bench.instruments.gate_kit.audit --state-db <state.db> --lcm-db <lcm.db> [--log <file> ...]
        [--since <ISO-8601 or epoch>] [--events <events.jsonl>] --out <result.json>

Both databases are opened with ``mode=ro`` URIs and never written; logs are only read. The output holds counts, rates,
booleans and ids (session and store ids) only: message content, tool arguments and results are compared in memory.

Expected rows come from the host's own state.db, per lineage (a session plus its compression children), with the
reliability harness's host-count rule (scorers/host_parity.py): a key's count is the most ACTIVE rows one session of
the lineage holds, else 1 when the lineage holds it only on inactive rows, because Hermes re-issues every durable row
on each rotation copy and in-place generation. Stored rows are scored against them by the harness's own B2
instruments: multiset.score (user/assistant rows), tool_calls.stored_keys/compare (tool calls and results),
externalized.resolve (externalized payloads) and multiset.split_surplus (visible versus silent surplus).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.instruments.reliability import plugin_tree  # noqa: E402
from bench.instruments.reliability.scorers import externalized, multiset, tool_calls  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FLOORS = {"turns": 200, "delegated_children": 10, "compactions": 20, "compactions_then_restart": 5, "restarts": 10,
          "restarts_cold": 5, "restarts_warm": 5, "undo_resend": 10, "tool_calls": 50, "tool_call_repeats": 10,
          "host_sources": 2}
BAR = 0.01  # the 1% bar of the exempted surplus and of M3 (section 5)
DELEGATE_SOURCE = "subagent"  # aux_session.py:805 _DELEGATE_SOURCE: a delegate child's state.db sessions.source
# Log lines. Hermes formats "<asctime> <LEVEL>[ [<session>]] <logger>: <message>" (hermes_logging.py _LOG_FORMAT).
LOG_HEAD = re.compile(r"^(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)(?:[,.]\d+)?\s+(?:- \S+ - )?[A-Z]+(?: \[([^\]\s]+)\])?")
COMPACTION = "LCM compaction #"  # compaction.py:2910, one line per published LCM pass
# aux_session.py logs no line when it classes a session AUXILIARY (_delegate_child_link, aux_session.py:1017-1022);
# the only per-session "auxiliary" classification line that never covers a delegate child is engine.py:2770.
AUXILIARY = re.compile(r"LCM reclassified session (\S+) as auxiliary at first ingest")
ROUTE = re.compile(r"LCM ingest route: session=(?P<session>\S+) route=(?P<route>\S+) cursor=(?P<cursor>-?\d+) "
                   r"reason=.*? stored=(?P<a>\d+)-(?P<b>\d+) rows=(?P<rows>\d+) replay_pairs=(?P<pairs>\d+) "
                   r"stripped=(?P<stripped>\d+)")
# Recorder route names (normalised: lower case, '-' and ' ' to '_'). The recorder PR fixes the spelling; keep in step.
ROUTE_2305 = "persisted_ambiguous_delta"  # the cursor-0 persisted batch (#2305)
ROUTE_RESTART_REPLAY, ROUTE_COLD_TAIL, ROUTE_EXIT_DRAIN = "restart_replay", "cold_tail_retry", "exit_drain"
ROUTE_WARM_CURSOR0, ROUTE_COMPOSITE = "warm_cursor0", "composite"
EVENT_KINDS = ("restart_cold", "restart_warm", "undo_resend", "delegate_prompt", "compress_request")


def ro(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} does not exist")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def columns(con, table: str) -> set:
    return {r[1] for r in con.execute(f"pragma table_info({table})")}


def has_table(con, name: str) -> bool:
    return con.execute("select 1 from sqlite_master where type='table' and name=?", (name,)).fetchone() is not None


def parse_since(value):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()  # naive = UTC


def opt(cols: set, name: str, default: str) -> str:
    return name if name in cols else f"{default} as {name}"


def read_host(path: Path):
    with ro(path) as con:
        sc, mc = columns(con, "sessions"), columns(con, "messages")
        sessions = {r[0]: dict(zip(("source", "parent", "started", "ended", "end_reason"), r[1:])) for r in con.execute(
            "select id, " + ", ".join(opt(sc, c, d) for c, d in (("source", "''"), ("parent_session_id", "NULL"),
                                                                    ("started_at", "0"), ("ended_at", "NULL"),
                                                                    ("end_reason", "NULL"))) + " from sessions")}
        rows = con.execute("select id, session_id, role, content, " + ", ".join(opt(mc, c, d) for c, d in (
            ("tool_calls", "NULL"), ("tool_call_id", "NULL"), ("active", "1"), ("_compressed_summary", "0"))) +
            " from messages order by id").fetchall()
    return sessions, rows


def compression_child(sid: str, sessions: dict) -> bool:
    """A rotation child: its parent ended with end_reason 'compression' (agent/conversation_compression.py) on the same
    source, and it started no earlier than one second before that end (a delegate child starts while its parent runs)."""
    s = sessions[sid]
    p = sessions.get(s["parent"] or "")
    return bool(p and p["end_reason"] == "compression" and p["source"] == s["source"]
                and (p["ended"] is None or (s["started"] or 0) >= p["ended"] - 1))


def lineage_of(sessions: dict):
    def group(sid):
        seen = set()
        while sid in sessions and compression_child(sid, sessions) and sid not in seen:
            seen.add(sid)
            sid = sessions[sid]["parent"]
        return sid
    return group


def host_expected(rows, group, markers):
    """Per lineage: (role, text) items and the tool-key Counter, by the host-count rule (scorers/host_parity.py)."""
    header, prefixes = markers
    active, inactive, texts, excluded = defaultdict(Counter), defaultdict(set), {}, Counter()
    for rid, sid, role, content, calls, call_id, is_active, summary in rows:
        g = group(sid)
        if summary:
            continue  # the host's own compression summary is not conversation
        keys = Counter()
        if role in ("user", "assistant") and multiset.norm(content or ""):
            head = (content or "").lstrip()
            if role == "user" and (header.match(head) or head.startswith(prefixes)):
                excluded[g] += 1  # an LCM generated carrier the host persisted: LCM's own output, never expected
            else:
                key = (role, multiset.h(content))
                texts[key] = content
                keys[key] += 1
        if role in ("tool", "assistant"):
            keys.update(k for k in tool_calls.stored_keys([(rid, sid, role, content, calls, call_id)], lambda _s: g)
                        if k[1] != "malformed_tool_calls")
        for key, n in keys.items():
            if is_active == 1:
                active[(g, sid)][key] += n
            else:
                inactive[g].add(key)
    count = defaultdict(Counter)
    for (g, _sid), keys in active.items():
        for key, n in keys.items():
            count[g][key] = max(count[g][key], n)
    for g, keys in inactive.items():
        for key in keys:
            count[g][key] = count[g][key] or 1
    out = {}
    for g, keys in count.items():
        items = [(k[0], texts[k]) for k, n in sorted(keys.items(), key=repr) if k[0] in ("user", "assistant")
                 for _ in range(n)]
        out[g] = (items, Counter({k: n for k, n in keys.items() if k[0] not in ("user", "assistant")}))
    return out, excluded


def read_store(path: Path):
    with ro(path) as con:
        mc = columns(con, "messages")
        rows = con.execute("select store_id, session_id, role, content, tool_calls, tool_call_id, " +
                           ", ".join(opt(mc, c, "NULL") for c in ("observed_at", "ingested_at")) +
                           " from messages order by store_id").fetchall()
        # store.py:1321-1329 message_relations(store_id, kind, related_store_id, ...); store_id is the row the
        # relation is on: engine.py:7486 writes (copy, 'possible_replay_of', target), engine.py:7547 (row, 'generated_unproven').
        relations = con.execute("select store_id, kind, related_store_id from message_relations").fetchall() \
            if has_table(con, "message_relations") else []
        uids = {r[0] for r in con.execute("select store_id from host_uid_bindings where kind = 'canonical'")} \
            if has_table(con, "host_uid_bindings") else None  # store.py:1477-1481
        lineage = {r[0]: (r[1], r[2]) for r in con.execute(
            "select child_session_id, parent_session_id, source from session_lineage")} \
            if has_table(con, "session_lineage") else {}  # store.py:1597-1598
        leaves = Counter(r[0] for r in con.execute(
            "select session_id from summary_nodes where depth = 0 and source_type = 'messages'")) \
            if has_table(con, "summary_nodes") else Counter()  # dag.py:208-221
    return rows, relations, uids, lineage, leaves


def read_logs(paths):
    out = {"compactions": [], "auxiliary": set(), "ingests": []}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            head = LOG_HEAD.match(line)
            sid = head.group(2) if head else None
            at = time.mktime(time.strptime(head.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S")) if head else None
            if COMPACTION in line:
                out["compactions"].append((sid, at))
            if m := AUXILIARY.search(line):
                out["auxiliary"].add(m.group(1))
            if m := ROUTE.search(line):
                out["ingests"].append({"session": m["session"], "route": m["route"].lower().replace("-", "_").replace(
                    " ", "_"), "cursor": int(m["cursor"]), "range": (int(m["a"]), int(m["b"])), "rows": int(m["rows"]),
                    "pairs": int(m["pairs"]), "stripped": int(m["stripped"])})
    return out


def read_events(path):
    if path is None:
        return None
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [e for e in events if isinstance(e, dict) and e.get("kind") in EVENT_KINDS]


def attribute_silent_surplus(silent, ingests, observed):
    """Section 5 (r3.8) "Surplus and other limits": attribute each silent surplus row to #1674, #2203, #849 or #1961
    ONLY from the one recorder line whose session matches and whose stored range holds the row; zero or several such
    lines leave the row unattributed (ambiguity fails as silent surplus).
    - #1674: a restart-replay route with cursor > 0; the copy has observed_at NULL and an earlier copy of the same key
      in the same session has observed_at set. A cursor-0 persisted batch (#2305) is never #1674.
    - #2203: a cold-tail retry or exit-drain route, and the copy has observed_at NULL.
    - #849: the composite-path route.
    - #1961: a warm cursor-0 route; the carrier-merge condition is not checkable here, so the row is reported as
      "needs manual check" and stays silent surplus for the gate.
    Not checked here (the store does not hold it): the view-order conditions of #1674 and #2203, and #849's
    "next prompt also stored as its own row"; the receipt checks them by hand.
    ``silent``: [(store_id, session, ids holding the same key)]; ``observed``: store_id -> observed_at."""
    out = {"#1674": [], "#2203": [], "#849": [], "#1961_needs_manual_check": [], "unattributed": []}
    for sid, session, ids in silent:
        lines = [i for i in ingests if i["session"] == session and i["range"][0] <= sid <= i["range"][1]]
        limit = "unattributed"
        if len(lines) == 1:
            route, cursor, unstamped = lines[0]["route"], lines[0]["cursor"], observed.get(sid) is None
            if route == ROUTE_RESTART_REPLAY and cursor > 0 and unstamped and any(
                    other < sid and observed.get(other) is not None for other in ids):
                limit = "#1674"
            elif route in (ROUTE_COLD_TAIL, ROUTE_EXIT_DRAIN) and unstamped:
                limit = "#2203"
            elif route == ROUTE_COMPOSITE:
                limit = "#849"
            elif route == ROUTE_WARM_CURSOR0 and cursor == 0:
                limit = "#1961_needs_manual_check"
        out[limit].append(sid)
    return {k: sorted(v) for k, v in out.items()}


def score_lineage(g, items, tools, rows, relations):
    """B2's instruments on one lineage; returns numbers and the silent rows [(store_id, ids of its key)]."""
    groups = []
    stored4 = [r[:4] for r in rows]
    ms = multiset.score(items, stored4, None, surplus_keys=groups)
    ids = {}
    tc = tool_calls.compare(tools, set(), tool_calls.stored_keys([r[:6] for r in rows], lambda _s: g, store_ids=ids),
                            store_ids=ids, surplus_keys=groups)
    visible, silent, truncated = 0, [], 0
    for group in groups:  # split_surplus matches within one key, so one call per key loses nothing
        split = multiset.split_surplus([group], relations)
        visible += split["visible_surplus_rows"]
        found = [e["store_id"] for e in split["silent_surplus_examples"]]
        truncated += split["silent_surplus_rows"] - len(found)  # over 10 in one key: left unattributable
        silent += [(sid, group[2]) for sid in found]
    return {"expected_rows": len(items) + sum(tools.values()), "stored_rows": len(rows),
            "missing_rows": ms["deficit_rows"] + tc["tool_missing_rows"], "split_assistant_keys": ms["split_keys"],
            "surplus_rows": ms["surplus_rows"] + tc["tool_surplus_rows"], "visible_surplus_rows": visible,
            "silent_surplus_rows": len(silent) + truncated, "silent_unlisted_rows": truncated}, silent


def audit(state_db: Path, lcm_db: Path, logs=(), since=None, events_path=None) -> dict:
    sessions, host_rows = read_host(state_db)
    stored, relations, uids, store_lineage, leaves = read_store(lcm_db)
    log = read_logs(logs)
    events = read_events(events_path)
    group = lineage_of(sessions)
    aux = set(log["auxiliary"])
    in_window = {s for s, v in sessions.items() if since is None or (v["started"] or 0) >= since}
    roots = {group(s) for s in in_window if group(s) in in_window and group(s) not in aux}
    expected, carriers = host_expected([r for r in host_rows if group(r[1]) in roots], group,
                                       plugin_tree.carrier_markers(REPO))
    resolved, ext, _ids = externalized.resolve([r[:6] for r in stored], state_db.parent)
    extra = {r[0]: r for r in stored}
    by_lineage = defaultdict(list)
    for r in resolved:
        g = group(r[1])
        unknown = r[1] not in sessions and (since is None or (extra[r[0]][7] or 0) >= since)  # rows under no host session
        if g in roots or (unknown and g not in aux):
            by_lineage[g].append((*r, extra[r[0]][6]))
    lost = Counter(group(extra[s][1]) for s in ext["store_ids"])
    parity, silent_rows = {}, []
    for g in sorted(roots | set(by_lineage)):
        items, tools = expected.get(g, ([], Counter()))
        numbers, silent = score_lineage(g, items, tools, by_lineage[g], relations)
        numbers["stored_rows"] += lost[g]
        numbers.update(externalized_unresolved=lost[g], host_carrier_rows_excluded=carriers[g],
                       source=sessions.get(g, {}).get("source"), host_session=g in sessions)
        parity[g] = numbers
        silent_rows += [(sid, extra[sid][1], ids) for sid, ids in silent]

    # #1931 buckets: ended children that are not rotation children, started in the window, not AUXILIARY. A bucket B
    # child is reported but does not gate, so its lineage stays out of the gate totals below.
    buckets = {"A": [], "B": []}
    for s in sorted(in_window - aux):
        v = sessions[s]
        if v["parent"] and v["ended"] is not None and not compression_child(s, sessions):
            buckets["A" if v["source"] == DELEGATE_SOURCE else "B"].append(s)
    for g, p in parity.items():
        p["gates"] = g not in buckets["B"]
    keys = ("expected_rows", "stored_rows", "missing_rows", "split_assistant_keys", "surplus_rows",
            "visible_surplus_rows", "silent_surplus_rows", "silent_unlisted_rows", "externalized_unresolved")
    totals = {k: sum(p[k] for p in parity.values() if p["gates"]) for k in keys}
    silent_rows = [r for r in silent_rows if group(r[1]) not in buckets["B"]]
    zero = sorted(g for g, p in parity.items()
                  if p["gates"] and p["host_session"] and p["expected_rows"] and not p["stored_rows"])
    per_session = Counter(r[1] for r in stored)
    window_ids = {r[0] for rows in by_lineage.values() for r in rows} | set(ext["store_ids"]) & {
        r[0] for r in stored if group(r[1]) in parity}
    denominator = len(window_ids)

    # Coverage floors (section 5, "The driven run").
    turns = sum(1 for g in roots for role, _t in expected.get(g, ([],))[0] if role == "user")
    calls = Counter((k[3], k[4]) for g in roots for k, n in expected.get(g, ([], Counter()))[1].items()
                    if k[1] == "call" for _ in range(n))
    children = {s for s in in_window if sessions[s]["parent"] and sessions[s]["source"] == DELEGATE_SOURCE
                and not compression_child(s, sessions)}
    children |= {c for c, (_p, src) in store_lineage.items() if src != "compression" and (c not in sessions or c in in_window)}
    kinds = Counter(e["kind"] for e in events) if events is not None else None
    compactions = len(log["compactions"]) if logs else "unknown"
    then_restart = "unknown"
    if logs and events is not None:
        restarts = [(group(e.get("session") or ""), e.get("t") or 0) for e in events if e["kind"].startswith("restart")]
        then_restart = sum(1 for sid, at in log["compactions"] if sid and at is not None
                           and any(g == group(sid) and t > at for g, t in restarts))
    value = {"turns": turns, "delegated_children": len(children), "compactions": compactions,
             "compactions_then_restart": then_restart, "tool_calls": sum(calls.values()),
             "tool_call_repeats": sum(n - 1 for n in calls.values()),
             "host_sources": len({sessions[s]["source"] for s in in_window})}
    for name, kind in (("restarts_cold", "restart_cold"), ("restarts_warm", "restart_warm"), ("undo_resend", "undo_resend")):
        value[name] = kinds[kind] if kinds is not None else "unknown"
    value["restarts"] = value["restarts_cold"] + value["restarts_warm"] if kinds is not None else "unknown"
    coverage = {k: {"value": value[k], "floor": f, "met": isinstance(value[k], int) and value[k] >= f}
                for k, f in FLOORS.items()}
    coverage_basis = {"compactions": "log lines 'LCM compaction #' (compaction.py:2910)" if logs else "no log given",
                      "store_depth0_message_leaves": sum(n for s, n in leaves.items() if group(s) in parity),
                      "restarts_undo": "driver events" if events is not None else "no events given"}

    # Uid coverage per host source (store rows in the window, canonical host_uid_bindings rows).
    uid = defaultdict(lambda: {"rows_stored": 0, "rows_with_canonical_uid": 0})
    for sid in window_ids:
        src = sessions.get(extra[sid][1], {}).get("source") or "unknown"
        uid[src]["rows_stored"] += 1
        uid[src]["rows_with_canonical_uid"] += int(uids is not None and sid in uids)

    # M3: rows carrying generated_unproven, or the store_id side of a possible_replay_of pair.
    doubt = sorted({s for s, kind, _o in relations if kind in ("possible_replay_of", "generated_unproven")} & window_ids)
    m3 = {"numerator": len(doubt), "denominator": denominator, "store_ids": doubt,
          "rate": len(doubt) / denominator if denominator else 0.0}

    ingests = log["ingests"]
    storing = [i for i in ingests if i["rows"] > 0]
    pairs = sum(i["pairs"] for i in ingests)
    recorder = {"present": bool(ingests), **({"status": "recorder absent"} if not ingests else {}),
                "ingests_per_route": dict(sorted(Counter(i["route"] for i in ingests).items())),
                "storing_ingests": len(storing),
                "rate_2305": sum(i["route"] == ROUTE_2305 for i in storing) / len(storing) if storing else None,
                "stripped_links": sum(i["stripped"] for i in ingests), "replay_pairs": pairs,
                "stripped_per_replay_pair": sum(i["stripped"] for i in ingests) / pairs if pairs else None}

    observed = {r[0]: r[6] for r in stored}
    attributed = attribute_silent_surplus(sorted(silent_rows, key=lambda r: r[0]), ingests, observed)
    exempt = sum(len(attributed[k]) for k in ("#1674", "#2203", "#849"))
    unattributed = totals["silent_surplus_rows"] - exempt
    exempted = {"per_limit": {k: len(v) for k, v in attributed.items()},
                "store_ids": {k: v for k, v in attributed.items()}, "exempted_rows": exempt,
                "unattributed_silent_rows": unattributed, "stored_rows": denominator,
                "fraction": exempt / denominator if denominator else 0.0}
    exempted["met"] = exempted["fraction"] <= BAR

    issue_1931 = {}
    for name, kids in buckets.items():
        p = [parity.get(k, {}) for k in kids]
        linked = [k for k in kids if store_lineage.get(k, ("", ""))[0] == sessions[k]["parent"]]
        issue_1931[name] = {"n": len(kids), "children": kids, "zero_row_children": sorted(
            k for k, q in zip(kids, p) if not q.get("stored_rows")), "missing_rows": sum(q.get("missing_rows", 0) for q in p),
            "silent_surplus_rows": sum(q.get("silent_surplus_rows", 0) for q in p), "lineage_present": len(linked)}
    a = issue_1931["A"]
    issue_1931["pass_A"] = (a["missing_rows"] == 0 and a["silent_surplus_rows"] == 0 and a["lineage_present"] == a["n"]
                            and not a["zero_row_children"] and a["n"] >= 5)
    issue_1931["auxiliary_excluded"] = sorted(aux)

    bars = {"zero_row_sessions": {"value": len(zero), "pass": not zero},
            "exact_content_deficits": {"value": totals["missing_rows"] + totals["split_assistant_keys"],
                                       "pass": not (totals["missing_rows"] + totals["split_assistant_keys"])},
            "unattributed_silent_surplus": {"value": unattributed, "pass": unattributed == 0},
            "exempted_surplus_fraction": {"value": exempted["fraction"], "pass": exempted["met"]},
            "m3_rate": {"value": m3["rate"], "pass": m3["rate"] <= BAR},
            "coverage_floors": {"value": sorted(k for k, c in coverage.items() if not c["met"]),
                                "pass": all(c["met"] for c in coverage.values())},
            "issue_1931_pass_A": {"value": issue_1931["pass_A"], "pass": issue_1931["pass_A"]}}
    data_bars = [b for b in bars if b != "coverage_floors"]
    verdict = "GATE FAIL" if not all(bars[b]["pass"] for b in data_bars) else \
        "INCOMPLETE" if not bars["coverage_floors"]["pass"] or not ingests else "GATE PASS"
    return {"instrument": "gate-kit-audit-v1", "claim_class": "instrument",
            "inputs": {"since": since, "logs": len(logs), "events": events is not None},
            "parity": {"totals": totals, "lineages": parity, "zero_row_lineages": zero,
                       "sessions": {s: {"lineage": group(s), "source": v["source"], "stored_rows": per_session[s]}
                                    for s, v in sessions.items() if group(s) in parity}},
            "coverage": coverage, "coverage_basis": coverage_basis,
            "uid_coverage": {"table": "present" if uids is not None else "table absent", "per_source": dict(uid)},
            "m3": m3, "recorder": recorder, "exempted_surplus": exempted, "issue_1931": issue_1931,
            "gate": {"bars": bars, "verdict": verdict}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-db", type=Path, required=True)
    ap.add_argument("--lcm-db", type=Path, required=True)
    ap.add_argument("--log", type=Path, action="append", default=[])
    ap.add_argument("--since")
    ap.add_argument("--events", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        result = audit(a.state_db, a.lcm_db, a.log, parse_since(a.since), a.events)
    except (OSError, sqlite3.Error, ValueError) as exc:
        ap.exit(2, f"audit failed: {type(exc).__name__}: {exc}\n"[:400])
    a.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    t, gate = result["parity"]["totals"], result["gate"]
    print(f"verdict: {gate['verdict']}")
    print(f"parity: expected={t['expected_rows']} stored={t['stored_rows']} missing={t['missing_rows']} "
          f"silent={t['silent_surplus_rows']} visible={t['visible_surplus_rows']}")
    print(f"m3: {result['m3']['numerator']}/{result['m3']['denominator']}  recorder: "
          f"{'present' if result['recorder']['present'] else 'absent'}  #1931 A: {result['issue_1931']['pass_A']}")
    print("failed bars: " + (", ".join(sorted(b for b, v in gate["bars"].items() if not v["pass"])) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
