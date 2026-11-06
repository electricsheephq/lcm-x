"""Gate-kit audit: every count of the v0.30.0 "nothing lost" gate (docs/exactly-once-ingest.md section 5, "Gate
(v0.30.0)" and "The driven run") from one driven run's Hermes state.db, LCM-X store and logs. Read-only.

    python -m bench.instruments.gate_kit.audit --state-db <state.db> --lcm-db <lcm.db> [--log <file> ...]
        [--since <ISO-8601 or epoch>] [--events <events.jsonl>] --out <result.json>

Run it after the run has stopped. Both databases are opened with ``mode=ro&immutable=1`` URIs: never written, and no
-wal or -shm file is created beside them. Logs are only read. The output holds counts, rates, booleans and ids
(session and store ids) only: message content, tool arguments and results are compared in memory.

Expected rows come from the host's own state.db, per lineage (a session plus its compression children), counted the
way the host's own lineage display counts them (``host_expected``). Stored rows are scored against them by the
reliability harness's B2 instruments: multiset.score (user/assistant rows), tool_calls.stored_keys/compare (tool calls
and results), externalized.resolve (externalized payloads) and multiset.split_surplus (visible versus silent surplus).
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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
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
DELEGATE_SOURCE = "subagent"  # aux_session.py:807 _DELEGATE_SOURCE: a delegate child's state.db sessions.source
NOT_ENTRY_SOURCES = frozenset({DELEGATE_SOURCE, "tool"})  # child sessions, not host entry points
# Log lines, in either Hermes format (hermes_logging.py _LOG_FORMAT and _LOG_FORMAT_VERBOSE):
# "<asctime> <LEVEL>[ [<session>]] <logger>: <message>" or "<asctime> - <logger> - <LEVEL>[ [<session>]] - <message>".
# A pattern counts only in the message of a line whose logger is LCM-X's (a module of the hermes_lcm package).
LOG_LINE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)(?:[,.]\d+)? (?:"
                      r"(?P<level>[A-Z]+)(?: \[(?P<sid>[^\]\s]+)\])? (?P<logger>[\w.]+): |"
                      r"- (?P<logger2>[\w.]+) - (?P<level2>[A-Z]+)(?: \[(?P<sid2>[^\]\s]+)\])? - )(?P<msg>.*)$")
LCM_LOGGER = re.compile(r"(?:^|\.)hermes_lcm(?:_x)?(?:\.|$)")
COMPACTION = re.compile(r"LCM compaction #\d+:")  # compaction.py:2910, one line per published LCM pass
# A session LCM classes AUXILIARY (stores nothing): engine.py:2770's late reclassification, and the #2042 session
# decision line. aux_session.py:1015-1021 (_delegate_child_link) classes the #1931 AUXILIARY case without a line.
RECLASSIFIED = re.compile(r"LCM reclassified session (\S+) as auxiliary at first ingest")
DECISION = re.compile(r"LCM session decision: session=(\S+) .*?\bdecision=auxiliary\b(?:.*?\breason=(\S+))?")
# A decision=auxiliary line excludes a session only for these reasons: a frame_marker decision is logged for an
# unproven child, which is the #1931 bypass itself, so that session stays scored (listed as auxiliary_unproven).
AUX_PROVEN_REASONS = frozenset({"background_review", "toolsets_memory_only"})
# The #2305 route recorder (engine.py _log_ingest_route): one line per ingest that stores rows. The reason is free
# text (spaces, parentheses), so it is matched lazily; route spaces become underscores (normalised again here).
ROUTE = re.compile(r"^LCM ingest route: session=(?P<session>\S+) route=(?P<route>.+?) cursor=(?P<cursor>-?\d+) "
                   r"reason=(?P<reason>.*?) stored=(?P<a>\d+)-(?P<b>\d+) rows=(?P<rows>\d+) replay_pairs=(?P<pairs>\d+) "
                   r"stripped=(?P<stripped>\d+)")
# Route tokens: the reconciliation action of a re-bound engine's first ingest (reconcile.py: advanced cursor,
# persisted batch, skipped batch), the warm default ``foreground``/``append``, and the tail routes: ``cold`` (a retried
# cold tail through the ingest), ``S3`` (the S3 tail retry), ``session_end`` and ``warm_tail``.
ROUTE_2305 = "persisted_batch"  # at cursor 0: reconcile.py "persisted ambiguous delta", the #2305 shape
# The pending-tail retry logs its match rule as the reason (engine.py _match_pending_tail): only these mark a drain.
TAIL_RULES = frozenset({"digest", "off", "no_ledger", "salt_changed", "read_failed"})
EVENT_KINDS = ("restart_cold", "restart_warm", "undo_resend", "delegate_prompt", "compress_request")
RESTARTS = ("restart_cold", "restart_warm")


def ro(path: Path) -> sqlite3.Connection:
    """Read-only and immutable: SQLite neither writes nor creates a -wal/-shm file (the run must have stopped).
    An immutable open does not read a -wal file either, so a non-empty one is refused: its pages would be missed."""
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} does not exist")
    wal = Path(f"{path}-wal")
    if wal.is_file() and wal.stat().st_size:
        raise ValueError(f"{path.name} has a non-empty -wal file: stop the run and checkpoint the database first")
    journal = Path(f"{path}-journal")
    if journal.is_file() and journal.stat().st_size:  # a hot rollback journal: an interrupted write not rolled back
        raise ValueError(f"{path.name} has a non-empty -journal file: open it once with its owner to roll it back")
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


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
        displayed = "(active = 1 OR compacted = 1)" if "compacted" in mc else "1"
        rows = con.execute("select id, session_id, role, content, " + ", ".join(opt(mc, c, d) for c, d in (
            ("tool_calls", "NULL"), ("tool_call_id", "NULL"), ("_compressed_summary", "0"), ("timestamp", "NULL"),
            ("tool_name", "NULL"))) + f", {displayed} from messages order by id").fetchall()
    return sessions, rows, "compacted" in mc


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


def display_key(role, content, ts, call_id, calls, tool_name):
    """The host's display identity (hermes_state_messages.py _display_dedupe_key, hermes_state_identity.py
    _stable_tool_key): a tool-calling assistant row by its call ids, any other row by its content; both with the row's
    timestamp. A carried handoff copy takes its original's timestamp (_carry_parent_timestamps), so it collapses with
    it, while a genuine repeat has a timestamp of its own. The host's user-handoff normalisation
    (split_user_originated_turn) is not repeated here: LCM-X's carriers are excluded by marker instead."""
    if role == "assistant" and calls:
        try:
            ids = tuple(c.get("id") or c.get("call_id") for c in json.loads(calls))
        except (ValueError, TypeError, AttributeError):
            ids = ()
        if ids and all(ids):
            return ("assistant", None, ts, call_id, ids, tool_name)
    # Content compares edge-stripped, as B2 keys it (multiset.norm): copies that differ only at the edges are one.
    return (role, multiset.norm(content) if role in ("user", "assistant") else content, ts, call_id, calls, tool_name)


def host_expected(rows, group, markers):
    """Per lineage: (role, text) items and the tool-key Counter, counted as the host's lineage display counts rows.

    Hermes keeps a rotation parent's originals in the closed parent and gives the child only the compressed handoff
    (hermes_state_compression.py _publish_compression_child_on_conn); an in-place compaction archives the displayed
    history (active = 0, compacted = 1) and rewinds the originals of the carried tail (active = 0, compacted = 0)
    (hermes_state_messages.py). The displayed rows are ``active = 1 OR compacted = 1``, and the lineage display collapses
    rows by display identity (``display_key``) across segments. One insert batch stamps all its rows with one time
    (hermes_state_messages.py _insert_message_rows), so two identical sends in one batch share an identity: an
    identity's count is therefore the most displayed rows any one segment (session) holds of it, and a key's expected
    count is the sum of those over the distinct identities of the lineage. A carried copy is not counted twice, and a
    double within a segment still counts twice. The host's summary row (``_compressed_summary``) and user rows that
    begin with LCM-X's own generated-carrier markers are not expected."""
    header, prefixes = markers
    per_segment, first = Counter(), {}
    for row in rows:
        rid, sid, role, content, calls, call_id, summary, ts, tool_name, displayed = row
        if summary or not displayed:
            continue
        identity = (group(sid), display_key(role, content, ts, call_id, calls, tool_name))
        per_segment[(identity, sid)] += 1
        first.setdefault(identity, row)
    copies = Counter()
    for (identity, _sid), n in per_segment.items():
        copies[identity] = max(copies[identity], n)
    count, texts, excluded = defaultdict(Counter), {}, Counter()
    for identity, n in copies.items():
        g = identity[0]
        rid, sid, role, content, calls, call_id = first[identity][:6]
        if role in ("user", "assistant") and multiset.norm(content or ""):
            head = (content or "").lstrip()
            if role == "user" and (header.match(head) or head.startswith(prefixes)):
                excluded[g] += n  # an LCM generated carrier the host persisted: LCM's own output, never expected
            else:
                key = (role, multiset.h(content))
                texts[key] = content
                count[g][key] += n
        if role in ("tool", "assistant"):
            for k in tool_calls.stored_keys([(rid, sid, role, content, calls, call_id)], lambda _s: g):
                if k[1] != "malformed_tool_calls":
                    count[g][k] += n
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


def log_time(stamp: str, tz):
    """A log asctime as epoch seconds, read in ``tz`` (a zoneinfo name), or in this machine's zone when None."""
    stamp = stamp.replace("T", " ")
    if tz is None:
        return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(tz)).timestamp()


def read_logs(paths, tz=None):
    """LCM-X's own log lines: compactions (deduplicated across files), AUXILIARY lines and recorder lines."""
    out = {"compactions": set(), "auxiliary": defaultdict(set), "ingests": [], "untagged_compactions": 0}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            m = LOG_LINE.match(line)
            if not m or not LCM_LOGGER.search(m["logger"] or m["logger2"]):
                continue
            sid, msg, at = m["sid"] or m["sid2"], m["msg"], log_time(m["ts"], tz)
            if COMPACTION.match(msg):
                if sid:
                    out["compactions"].add((sid, at, msg))  # a line logged to two files counts once
                else:
                    out["untagged_compactions"] += 1
            if hit := RECLASSIFIED.match(msg):
                out["auxiliary"][hit.group(1)].add("first_ingest_reclassification")
            if hit := DECISION.match(msg):
                out["auxiliary"][hit.group(1)].add(hit.group(2) or "")
            if hit := ROUTE.match(msg):
                out["ingests"].append({"session": hit["session"], "route": hit["route"].strip().replace(" ", "_"),
                                       "reason": hit["reason"].strip(), "cursor": int(hit["cursor"]), "at": at,
                                       "range": (int(hit["a"]), int(hit["b"])), "rows": int(hit["rows"]),
                                       "pairs": int(hit["pairs"]), "stripped": int(hit["stripped"])})
    return out


def auxiliary_sessions(lines: dict) -> tuple[set, set]:
    """(excluded, unproven). A session is excluded as AUXILIARY when the first-ingest reclassification (engine.py:2770)
    names it, or a decision=auxiliary line names it with a proven reason (AUX_PROVEN_REASONS), whatever its source.
    Any other decision=auxiliary session (a frame_marker reason, say) stays scored and is listed as unproven."""
    excluded = {s for s, reasons in lines.items()
                if "first_ingest_reclassification" in reasons or reasons & AUX_PROVEN_REASONS}
    return excluded, set(lines) - excluded


def read_events(path):
    if path is None:
        return None
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [e for e in events if isinstance(e, dict) and e.get("kind") in EVENT_KINDS]


def attribute_silent_surplus(silent, ingests, observed, user_rows, restarts):
    """Section 5 (r3.8) "Surplus and other limits": attribute each silent surplus row to #1674, #2203, #849 or #1961
    ONLY from the one recorder line whose session matches and whose stored range holds the row; zero or several such
    lines leave the row unattributed (ambiguity fails as silent surplus). Only #1674 and #2203 are exempted here;
    #849 and #1961 rows are listed for a hand check and stay silent surplus for the gate. Line numbers cite the
    recorder PR's engine.py (``_log_ingest_route`` at 7140-7160).
    ``silent``: [(store_id, session, ids holding the same key)]; ``observed``: store_id -> observed_at;
    ``user_rows``: store_id -> (session, text) for stored user rows (compared in memory only); ``restarts``: lineage ->
    driver restart times (restart_cold, restart_warm), or None without --events.
    An exempted row is still a hand check: the verdict is then GATE PASS PENDING HAND CHECK, never GATE PASS."""
    out = {"#1674": [], "#2203": [], "#1674_needs_manual_check": [], "#849_needs_manual_check": [],
           "#1961_needs_manual_check": [], "unattributed": []}
    prompts = defaultdict(list)  # session -> [(store_id, edge-stripped user text)]
    for other, (s, t) in user_rows.items():
        if multiset.norm(t):
            prompts[s].append((other, multiset.norm(t)))
    for sid, session, ids in silent:
        lines = [i for i in ingests if i["session"] == session and i["range"][0] <= sid <= i["range"][1]]
        limit = "unattributed"
        if len(lines) == 1:
            route, cursor, reason = lines[0]["route"], lines[0]["cursor"], lines[0]["reason"]
            unstamped = observed.get(sid) is None
            text = multiset.norm(user_rows.get(sid, (None, ""))[1])
            if route == ROUTE_2305 and cursor == 0:
                pass  # #2305's cursor-0 persisted batch: never exempted, never #1674 (section 5)
            # #1674 (restart replay): the first ingest of a re-bound engine reconciles, and the recorder logs the
            # reconciliation action as the route (engine.py:8218); "advanced cursor" past a replayed prefix
            # (reconcile.py, every replayed-prefix reason) with cursor > 0. The copy is unstamped and an earlier copy
            # of its key in the session is stamped; section 5 says the stamp condition alone does not qualify, so a
            # driver restart of the session must also come before the recorder line (none known without --events:
            # hand check). Not checked: the view-order condition.
            elif route == "advanced_cursor" and cursor > 0 and unstamped and any(
                    other < sid and observed.get(other) is not None for other in ids):
                if restarts is None or lines[0]["at"] is None:
                    limit = "#1674_needs_manual_check"
                elif any(t < lines[0]["at"] for t in restarts.get(session, ())):
                    limit = "#1674"
            # #2203: a cold-start retry of a queued tail, through the ingest (route "cold", engine.py:7141) or the S3
            # retry (route "S3", engine.py:11794); or the exit drain, route "session_end" logged by the pending-tail
            # retry with exiting=True (engine.py:11844), told apart from an ordinary session-end flush (also
            # "session_end", engine.py:7142) by its reason, the tail match rule. The copy is unstamped. Not checked:
            # that the view differs from the stored rows by no host insertion or reordering. "warm_tail" (a retry
            # that is not the exit drain) is neither section 5 shape, so it is not attributed.
            elif unstamped and (route in ("cold", "S3") or (route == "session_end" and reason in TAIL_RULES)):
                limit = "#2203"
            # #849 (a composite of an unanswered prompt and the next prompt across a restart): the recorder has no
            # composite-path route, so only the row's shape is checked here, in memory: a user row whose text is
            # another prompt plus "\n\n" plus a user row stored apart in the same session.
            elif text and any(other != sid and len(text) > len(t) + 2 and text.endswith("\n\n" + t)
                              for other, t in prompts[session]):
                limit = "#849_needs_manual_check"
            # #1961 (a warm re-ingest after the host merged LCM's carrier): a warm engine does not reconcile, so the
            # recorder logs the default route "foreground" (engine.py:8218); at cursor 0. The carrier-merge condition
            # needs the host view, so the row is listed for a hand check.
            elif route == "foreground" and cursor == 0:
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


def audit(state_db: Path, lcm_db: Path, logs=(), since=None, events_path=None, log_tz=None) -> dict:
    sessions, host_rows, has_compacted = read_host(state_db)
    stored, relations, uids, store_lineage, leaves = read_store(lcm_db)
    log = read_logs(logs, log_tz)
    events = read_events(events_path)
    group = lineage_of(sessions)
    aux, aux_unproven = auxiliary_sessions(log["auxiliary"])
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

    # Whole-run parity: every scored lineage, #1931 bucket B children included (only PASS(A) is bucket A's own bar).
    keys = ("expected_rows", "stored_rows", "missing_rows", "split_assistant_keys", "surplus_rows",
            "visible_surplus_rows", "silent_surplus_rows", "silent_unlisted_rows", "externalized_unresolved")
    totals = {k: sum(p[k] for p in parity.values()) for k in keys}
    zero = sorted(g for g, p in parity.items() if p["host_session"] and p["expected_rows"] and not p["stored_rows"])
    per_session = Counter(r[1] for r in stored)
    window_ids = {r[0] for rows in by_lineage.values() for r in rows} | set(ext["store_ids"]) & {
        r[0] for r in stored if group(r[1]) in parity}
    denominator = len(window_ids)

    # #1931 buckets: ended children that are not rotation children, started in the window, not AUXILIARY.
    buckets = {"A": [], "B": []}
    for s in sorted(in_window - aux):
        v = sessions[s]
        if v["parent"] and v["ended"] is not None and not compression_child(s, sessions):
            buckets["A" if v["source"] == DELEGATE_SOURCE else "B"].append(s)

    # Coverage floors (section 5, "The driven run").
    turns = sum(1 for g in roots for role, _t in expected.get(g, ([],))[0] if role == "user")
    calls = Counter((k[3], k[4]) for g in roots for k, n in expected.get(g, ([], Counter()))[1].items()
                    if k[1] == "call" for _ in range(n))
    kinds = Counter(e["kind"] for e in events) if events is not None else None
    # Compaction lines: deduplicated, tagged with a scored session, and inside the window when one is given.
    compaction_lines = sorted((sid, at) for sid, at, _msg in log["compactions"]
                              if group(sid) in parity and (since is None or at >= since))
    restarts = None
    if events is not None:
        restarts = defaultdict(list)
        for e in events:
            if e["kind"] in RESTARTS:
                restarts[group(str(e.get("session") or ""))].append(float(e.get("t") or 0))
    then_restart = "unknown" if not logs or restarts is None else sum(
        1 for sid, at in compaction_lines if any(t > at for t in restarts.get(group(sid), ())))
    value = {"turns": turns, "delegated_children": sum(1 for k in buckets["A"] if k in parity),
             "compactions": len(compaction_lines) if logs else "unknown", "compactions_then_restart": then_restart,
             "tool_calls": sum(calls.values()), "tool_call_repeats": sum(n - 1 for n in calls.values()),
             "host_sources": len({sessions[s]["source"] for s in in_window} - NOT_ENTRY_SOURCES)}
    for name, kind in (("restarts_cold", "restart_cold"), ("restarts_warm", "restart_warm"), ("undo_resend", "undo_resend")):
        value[name] = kinds[kind] if kinds is not None else "unknown"
    value["restarts"] = value["restarts_cold"] + value["restarts_warm"] if kinds is not None else "unknown"
    coverage = {k: {"value": value[k], "floor": f, "met": isinstance(value[k], int) and value[k] >= f}
                for k, f in FLOORS.items()}
    coverage_basis = {"compactions": "log lines 'LCM compaction #' of LCM-X's logger, tagged with a scored session, "
                                     "deduplicated (compaction.py:2910)" if logs else "no log given",
                      "untagged_compaction_lines": log["untagged_compactions"],
                      "store_depth0_message_leaves": sum(n for s, n in leaves.items() if group(s) in parity),
                      "delegated_children": "ended, scored #1931 bucket A children",
                      "host_sources": "distinct sessions.source in the window, without subagent and tool",
                      "restarts_undo": "driver events" if events is not None else "no events given",
                      "expected_rows": "host display identity per lineage" if has_compacted else
                                       "host display identity per lineage (no compacted column: every row displayed)"}

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
                "rate_2305": sum(i["route"] == ROUTE_2305 and i["cursor"] == 0 for i in storing) / len(storing)
                if storing else None,
                "stripped_links": sum(i["stripped"] for i in ingests), "replay_pairs": pairs,
                "stripped_per_replay_pair": sum(i["stripped"] for i in ingests) / pairs if pairs else None}

    observed = {r[0]: r[6] for r in stored}
    user_rows = {r[0]: (r[1], r[3]) for r in resolved if r[2] == "user"}
    session_restarts = None if restarts is None else {s: restarts.get(group(s), []) for s in sessions}
    attributed = attribute_silent_surplus(sorted(silent_rows, key=lambda r: r[0]), ingests, observed, user_rows,
                                          session_restarts)
    exempt = sum(len(attributed[k]) for k in ("#1674", "#2203"))
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
    issue_1931["auxiliary_unproven"] = sorted(aux_unproven)

    deficits = totals["missing_rows"] + totals["split_assistant_keys"]
    bars = {"zero_row_sessions": {"value": len(zero), "pass": not zero},
            "exact_content_deficits": {"value": deficits, "pass": not deficits},
            "externalized_unresolved": {"value": totals["externalized_unresolved"],
                                        "pass": not totals["externalized_unresolved"]},
            "unattributed_silent_surplus": {"value": unattributed, "pass": unattributed == 0},
            "exempted_surplus_fraction": {"value": exempted["fraction"], "pass": exempted["met"]},
            "m3_rate": {"value": m3["rate"], "pass": m3["rate"] <= BAR},
            "coverage_floors": {"value": sorted(k for k, c in coverage.items() if not c["met"]),
                                "pass": all(c["met"] for c in coverage.values())},
            "issue_1931_pass_A": {"value": issue_1931["pass_A"], "pass": issue_1931["pass_A"]}}
    data_bars = [b for b in bars if b != "coverage_floors"]
    verdict = "GATE FAIL" if not all(bars[b]["pass"] for b in data_bars) else \
        "INCOMPLETE" if not bars["coverage_floors"]["pass"] or not ingests else \
        "GATE PASS PENDING HAND CHECK" if exempt else "GATE PASS"
    return {"instrument": "gate-kit-audit-v1", "claim_class": "instrument",
            "inputs": {"since": since, "logs": len(logs), "events": events is not None, "log_tz": log_tz},
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
    ap.add_argument("--log-tz", help="zoneinfo name the log times are in (default: this machine's local zone)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        if a.log_tz:
            ZoneInfo(a.log_tz)  # an unknown zone is an input error, before any database is opened
        result = audit(a.state_db, a.lcm_db, a.log, parse_since(a.since), a.events, a.log_tz)
    except (OSError, sqlite3.Error, ValueError, ZoneInfoNotFoundError) as exc:
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
