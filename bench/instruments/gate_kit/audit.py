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
import os
import re
import sqlite3
import sys
import tempfile
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
# Log lines, in the Hermes formats (hermes_logging.py _LOG_FORMAT and _LOG_FORMAT_VERBOSE):
# "<asctime> <LEVEL>[ [<session>]] <logger>: <message>" or "<asctime> - <logger> - <LEVEL>[ [<session>]] - <message>";
# and the ACP host's stderr format, which has no session field (acp_adapter/entry.py _setup_logging):
# "<asctime> [<LEVEL>] <logger>: <message>".
# A pattern counts only in the message of a line whose logger is LCM-X's (a module of the hermes_lcm package).
LOG_LINE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)(?:[,.](?P<frac>\d+))? (?:"
                      r"(?P<level>[A-Z]+)(?: \[(?P<sid>[^\]\s]+)\])? (?P<logger>[\w.]+): |"
                      r"- (?P<logger2>[\w.]+) - (?P<level2>[A-Z]+)(?: \[(?P<sid2>[^\]\s]+)\])? - |"
                      r"\[(?P<level3>[A-Z]+)\] (?P<logger3>[\w.]+): )(?P<msg>.*)$")
MESSAGE_SESSION = re.compile(r"\bsession=([^\s,;)]+)")
# A compaction line without a session is tied to the driver's compress_request for a session when the line falls
# within this window around the request (seconds before, after): some log formats have one-second resolution, and
# the compaction follows the request after the summary model call.
COMPRESS_REQUEST_WINDOW = (2.0, 120.0)
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
# A route line's optional ``link_pending=<n>`` field (#2377): the rows the ingest marked link_pending.
ROUTE_LINK_PENDING = re.compile(r"\blink_pending=(\d+)\b")
# #2377's logged intended link (final format): one WARNING per intended link that failed, logged after commit, with no
# content. The pending marker has a NULL target, so a pending_write_failed line may name target_store_id=None.
LINK_FAILED = "LCM tool replay link failed"
LINK_LINE = re.compile(r"LCM tool replay link failed source_store_id=(\d+) target_store_id=(?:\d+ "
                       r"reason=(?:relation_write_failed|pending_write_failed)|None reason=pending_write_failed)")
# #2377: link computation raised for one ingest; every row of that batch carries a link_pending relation instead.
LINK_COMPUTATION_FAILED = re.compile(r"LCM tool replay computation failed reason=\w+")
LINK_STORE_SLACK = 1.0  # seconds: a log stamp of one-second resolution can read just before the row's ingested_at
# Route tokens: the reconciliation action of a re-bound engine's first ingest (reconcile.py: advanced cursor,
# persisted batch, skipped batch), the warm default ``foreground``/``append``, and the tail routes: ``cold`` (a retried
# cold tail through the ingest), ``S3`` (the S3 tail retry), ``session_end`` and ``warm_tail``.
ROUTE_2305 = "persisted_batch"  # at cursor 0: reconcile.py "persisted ambiguous delta", the #2305 shape
# The pending-tail retry logs its match rule as the reason (engine.py _match_pending_tail): only these mark a drain.
TAIL_RULES = frozenset({"digest", "off", "no_ledger", "salt_changed", "read_failed"})
EVENT_KINDS = ("restart_cold", "restart_cold_shared", "restart_warm", "undo_resend", "delegate_prompt",
               "compress_request", "compress_request_failed")
RESTARTS = ("restart_cold", "restart_cold_shared", "restart_warm")


def ro(path: Path) -> sqlite3.Connection:
    """Read-only and immutable: SQLite neither writes nor creates a -wal/-shm file (the run must have stopped).
    An immutable open does not read a -wal file either, so a non-empty one is refused: its pages would be missed."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"{Path(path).name} does not exist")
    path = Path(path).resolve()  # SQLite opens the link's target, so its -wal and -journal sit beside the target
    wal = Path(f"{path}-wal")
    if wal.is_file() and wal.stat().st_size:
        raise ValueError(f"{path.name} has a non-empty -wal file: stop the run and checkpoint the database first")
    journal = Path(f"{path}-journal")
    if journal.is_file() and journal.stat().st_size:  # a hot rollback journal: an interrupted write not rolled back
        raise ValueError(f"{path.name} has a non-empty -journal file: open it once with its owner to roll it back")
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)


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


def read_sessions(con):
    sc = columns(con, "sessions")
    return {r[0]: dict(zip(("source", "parent", "started", "ended", "end_reason", "rewind_count"), r[1:]))
            for r in con.execute("select id, " + ", ".join(opt(sc, c, d) for c, d in (
                ("source", "''"), ("parent_session_id", "NULL"), ("started_at", "0"), ("ended_at", "NULL"),
                ("end_reason", "NULL"), ("rewind_count", "NULL"))) + " from sessions")}


def read_host(path: Path):
    with ro(path) as con:
        sessions, mc = read_sessions(con), columns(con, "messages")
        # Displayed: active, or archived history. Undone: soft-deleted by a rewind (hermes_state_messages.py
        # rewind_to_message sets active = 0 and leaves compacted = 0); without the compacted column none is known.
        displayed = "(active = 1 OR compacted = 1)" if "compacted" in mc else "1"
        undone = "(active = 0 AND compacted = 0)" if {"active", "compacted"} <= mc else "0"
        rows = con.execute("select id, session_id, role, content, " + ", ".join(opt(mc, c, d) for c, d in (
            ("tool_calls", "NULL"), ("tool_call_id", "NULL"), ("_compressed_summary", "0"), ("timestamp", "NULL"),
            ("tool_name", "NULL"))) + f", {displayed}, {undone}, " +
                           ", ".join(opt(mc, c, "0") for c in ("active", "compacted")) +
                           " from messages order by id").fetchall()
    return sessions, rows, "compacted" in mc


def compression_child(sid: str, sessions: dict) -> bool:
    """A rotation child: its parent ended with end_reason 'compression' (agent/conversation_compression.py) on the same
    source, and it started no earlier than one second before that end (a delegate child starts while its parent runs)."""
    s = sessions[sid]
    p = sessions.get(s["parent"] or "")
    return bool(p and p["end_reason"] == "compression" and p["source"] == s["source"]
                and (p["ended"] is None or (s["started"] or 0) >= p["ended"] - 1))


def eligible_child(sid: str, sessions: dict) -> bool:
    """An ended, non-rotation child; callers also restrict source and scored sessions."""
    s = sessions[sid]
    return bool(s["parent"] and s["ended"] is not None and not compression_child(sid, sessions))


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
    timestamp. Rotation handoffs take their original's timestamp (_carry_parent_timestamps); in-place copies keep it
    because the host wrote the stamp back onto the message it later re-sends (excluded by ``in_place_carried_ids``). The host's user-handoff normalisation
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


def carrier_remainder(content, header, nodes, same_lineage):
    """A host user row's text with a leading LCM summary carrier taken off, as LCM-X reads it (engine.py
    _generated_context_carrier_remainder; the stored row is the user's own row, which LCM-X matches by this remainder,
    reconcile.py _merged_pair_row): hosts that repair role alternation merge LCM's user-role summary with the next
    user row (``prev + "\\n\\n" + next``). Each part ``[<Label> Summary (dN, node N)]\\n<summary>\\n[Expand for
    details: <hint>]`` must match its node in the store byte for byte, as LCM-X's verifier requires (engine.py
    _verified_lcm_summary_prefix: dag.get_node, the node's depth, the rendered part): ``nodes`` holds summary_nodes
    (dag.py:208) as node_id -> (session_id, depth, summary, expand_hint). The node's session must be in the row's
    lineage (``same_lineage``; LCM-X checks its bound session, and a rotation moves the nodes to the newest child).
    Parts joined by ``\\n\\n---\\n\\n``, then ``\\n\\n---\\n\\n`` or ``\\n\\n`` and the row. Returns the content
    unchanged when any part does not verify (the row is then ordinary user content), "" for a carrier with no row."""
    text, pos, parts = (content or "").lstrip(), 0, 0
    while hit := header.match(text, pos):
        node = nodes.get(int(hit.group(2)))
        if node is None or node[1] != int(hit.group(1)) or not same_lineage(node[0]):
            return content
        label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(node[1], f"Depth-{node[1]}")
        part = f"[{label} Summary (d{node[1]}, node {hit.group(2)})]\n{node[2]}\n[Expand for details: {node[3] or ''}]"
        if not text.startswith(part, pos):
            return content
        pos, parts = pos + len(part), parts + 1
        if not (text.startswith("\n\n---\n\n", pos) and header.match(text, pos + 7)):
            break
        pos += 7
    if not parts:
        return content
    if not text[pos:].strip():
        return ""
    for separator in ("\n\n---\n\n", "\n\n"):
        if text.startswith(separator, pos):
            return text[pos + len(separator):]
    return content


def in_place_carried_ids(rows, strip, prefixes):
    """Same-session copies at EVERY in-place compaction, in host id order, even if later archived.

    A compaction archives the active rows (compacted = 1) and inserts its compressed list after a boundary: the host's
    ``_compressed_summary`` row or LCM's summary carrier (a user row whose verified strip changes it); a run of boundary
    rows with no displayed non-carrier row between them is one compaction (this host pin flags no summary row, and LCM's
    carrier opens the block). Its pool is what it archived: the displayed non-carrier archived rows between the
    previous compaction and this one, earlier copies included (a chained compaction copies copies). A displayed
    non-carrier row after the boundary, up to the next compaction, is a copy when its full display_key (role,
    content or call ids, and the exact stamp) matches an unused pool row: a copy keeps the original's stamp (the host
    wrote it back onto the message), and each pool row stands for one copy. Order is not required (the host can
    reorder or drop rows of the list). A new row has a stamp of its own, so a resend or repeat is never a copy; a copy
    the host re-stamps stays expected (a false alarm, never a hidden loss). User carriers (verified strip changes,
    empty text or LCM prefixes) are left to the existing carrier logic. No compacted evidence means no deduction.
    """
    sessions = defaultdict(list)
    for row in rows:
        if row[9] or row[6]:
            sessions[row[1]].append(row)

    def carrier(row):
        if row[2] != "user":
            return False
        text = strip(row[3], row[1])
        return text != row[3] or not (text or "").strip() or (text or "").lstrip().startswith(prefixes)

    def boundary(row):
        return bool(row[9]) and (bool(row[6]) or (row[2] == "user" and strip(row[3], row[1]) != row[3]))

    def key(row):
        _rid, sid, role, content, calls, call_id = row[:6]
        return display_key(role, strip(content, sid) if role == "user" else content, row[7], call_id, calls, row[8])

    carried = set()
    for segment in sessions.values():
        segment.sort(key=lambda r: r[0])
        window, pool, opened = [], Counter(), False
        for row in segment:
            if boundary(row):
                if not opened:  # a new compaction: it archived the window since the previous one
                    pool = Counter(key(r) for r in window if r[12] and r[7] is not None)
                    window, opened = [], True
                continue
            if not row[9] or carrier(row):
                continue
            opened = False
            window.append(row)
            k = key(row)
            if row[7] is not None and pool[k] > 0:
                pool[k] -= 1
                carried.add(row[0])
    return carried


def host_expected(rows, group, markers, strip, scored=None):
    """Per lineage: (role, text) items and the tool-key Counter, counted as the host's lineage display counts rows.

    Hermes keeps a rotation parent's originals in the closed parent and gives the child only the compressed handoff
    (hermes_state_compression.py _publish_compression_child_on_conn); an in-place compaction archives the displayed
    history (active = 0, compacted = 1). Some hosts rewind the carried originals (active = 0, compacted = 0);
    others keep both originals and active copies displayed. ``in_place_carried_ids`` excludes those copies by
    their exact stamp against what each compaction archived, not by global text deduplication. The displayed rows are
    ``active = 1 OR compacted = 1``, and the lineage display collapses
    rows by display identity (``display_key``) across segments. The host stamps each row of an insert batch 1e-6 s
    after the one before it (hermes_state_messages.py _insert_message_rows), so two identical sends have identities of
    their own; an identity's count is the most displayed rows any one segment (session) holds of it, and a key's
    expected count is the sum of those over the distinct identities of the lineage. A carried copy is not counted
    twice, and a double within a segment still counts twice. The host's summary row (``_compressed_summary``) is not
    expected, nor is a user row that is only an LCM-X generated carrier; a user row that begins with LCM's summary
    carrier verified in the store and goes on with the user's own text is that text (``strip``): one occurrence with
    a row of the same text and timestamp. A merged row of a text the lineage also holds at another timestamp may be
    the host's merged copy of that row (a rotation handoff stamps it anew) or a resend: it is not counted, and it is
    listed for a hand check (``merged_carrier_repeat``); a text held only by merged rows counts its first. A row that begins
    with one of LCM-X's prefix carriers (objective, task list, overflow recovery) is not expected: where such a carrier
    ends is not marked (``todo_expected`` expects a stored copy of a displayed todo row). ``scored``: the sessions in the audit window; only their rows are expected, and an identity's
    copies in an unscored segment of the lineage (a row carried from a parent that started before the window) are the
    parent's rows: the scored segments expect only the occurrences beyond the most any unscored segment holds.
    Returns (expected per lineage, carrier rows excluded per lineage, hand checks)."""
    _header, prefixes = markers
    per_segment, first, plain, merged = Counter(), {}, set(), set()
    merged_rows, earlier_texts = defaultdict(list), defaultdict(list)  # for the hand-check detectors only
    carried = in_place_carried_ids(rows, strip, prefixes)
    for row in rows:
        rid, sid, role, content, calls, call_id, summary, ts, tool_name, displayed = row[:10]
        if summary or not displayed or rid in carried:
            continue
        text = strip(content, sid) if role == "user" else content
        identity = (group(sid), display_key(role, text, ts, call_id, calls, tool_name))
        first.setdefault(identity, (row, text))
        if text == content:
            plain.add(identity)
            per_segment[(identity, sid)] += 1
        else:  # merged rows of one text and time are one occurrence, and the same one as a plain row of it
            merged.add((identity, sid))
            merged_rows[(identity, sid)].append(rid)
        if scored is not None and sid not in scored and role == "user" and multiset.norm(text or ""):
            earlier_texts[(identity[0], multiset.norm(text))].append(rid)
    for slot in merged - set(per_segment):
        per_segment[slot] = 1
    copies, earlier = Counter(), Counter()
    for (identity, sid), n in per_segment.items():
        held = earlier if scored is not None and sid not in scored else copies
        held[identity] = max(held[identity], n)
    copies.subtract(earlier)  # per occurrence: a child that holds more copies than its earlier parent expects the rest
    by_text = defaultdict(list)  # (lineage, user text) -> its identities, plain ones first, then by timestamp
    for identity in sorted((i for i, n in copies.items() if n > 0), key=lambda i: (i not in plain, i[1][2] or 0)):
        if identity[1][0] == "user" and multiset.norm(first[identity][1] or ""):
            by_text[(identity[0], multiset.norm(first[identity][1]))].append(identity)
    hand_checks = []
    # Detectors: they only add hand checks, never change a count. Two merged rows of one session with one text and
    # one time are one occurrence above; and a merged row in the window whose text an earlier, unscored segment holds
    # may be a resend that the window's subtraction does not see.
    for (identity, sid), ids in sorted(merged_rows.items(), key=lambda m: m[1]):
        if not multiset.norm(first[identity][1] or ""):
            continue  # a carrier with no user text behind it
        if len(ids) > 1:
            hand_checks.append({"reason": "merged_carrier_same_stamp_multiplicity", "lineage": identity[0],
                                "host_ids": sorted(ids)})
        earlier_ids = earlier_texts.get((identity[0], multiset.norm(first[identity][1] or "")))
        if scored is not None and sid in scored and earlier_ids:
            hand_checks.append({"reason": "merged_carrier_window_repeat", "lineage": identity[0],
                                "host_ids": sorted(earlier_ids + ids)})
    for (g, _text), identities in by_text.items():
        dropped = [i for i in identities[1:] if i not in plain]
        if dropped:
            hand_checks.append({"reason": "merged_carrier_repeat", "lineage": g,
                                "host_ids": sorted(first[i][0][0] for i in dropped)})
            for identity in dropped:
                copies[identity] = 0
    count, texts, excluded = defaultdict(Counter), {}, Counter()
    for identity, n in copies.items():
        g = identity[0]
        (rid, sid, role, content, calls, call_id), text = first[identity][0][:6], first[identity][1]
        if n <= 0:
            continue
        if role in ("user", "assistant") and multiset.norm(content or ""):
            if role == "user" and (not multiset.norm(text) or text.lstrip().startswith(prefixes)):
                excluded[g] += n  # an LCM generated carrier the host persisted: LCM's own output, never expected
            else:
                key = (role, multiset.h(text))
                texts[key] = text
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
    return out, excluded, hand_checks


# #2087: Hermes's empty-response recovery pair (host commit 8cfaaa8e). After tool results with an empty reply, the
# loop appends an assistant row "(empty)" and a user row holding _EMPTY_TOOL_RESPONSE_NUDGE (agent/turn_empty_response.py:
# 208-215, the text at agent/conversation_loop.py:900-903), both flagged _empty_recovery_synthetic, a flag the host's
# persistence skips (agent/session_persistence.py:32-40): the host never saves them, so a stored copy is not expected.
EMPTY_RECOVERY_REPLY = "(empty)"
EMPTY_RECOVERY_NUDGE = ("You just executed tool calls but returned an empty response. Please process the tool "
                        "results above and continue with the task.")


def empty_recovery_rows(rows, host_rows=(), group=lambda session: session, strip=lambda content, session: content):
    """Stored copies of the host's empty-response recovery pair (#2087), matched by structure and exact host text: in
    one session's store order, a tool row, then an assistant row whose content is exactly ``EMPTY_RECOVERY_REPLY`` with
    no tool calls, then a user row whose content is exactly ``EMPTY_RECOVERY_NUDGE``. A user row with the nudge's text
    and no such pair in front of it is a real turn and stays scored. The host never saves the flagged pair, so a pair in a
    lineage (``group``: a compression rotation joins its parent) whose host rows hold a saved row with either text is
    ambiguous: it may be a copy of a real reply or turn, or the host's scaffolding standing in for a lost real row. A
    saved user row is read behind its verified summary carrier (``strip``), as parity reads it. It
    is not excluded (it stays scored) and it is returned apart, for a hand check. ``rows``: every stored row of the
    audited sessions, (store_id, session, role, content, tool_calls, ...), so adjacency is the store's own order;
    ``host_rows``: (id, session, role, content, ...). Returns (matched store ids, ambiguous store ids)."""
    saved = {group(r[1]) for r in host_rows
             if (r[2], strip(r[3], r[1]) if r[2] == "user" else r[3]) in {("assistant", EMPTY_RECOVERY_REPLY),
                                                                         ("user", EMPTY_RECOVERY_NUDGE)}}
    by_session = defaultdict(list)
    for r in sorted(rows):
        by_session[r[1]].append(r)
    matched, ambiguous = set(), set()
    for session, seq in by_session.items():
        for before, reply, nudge in zip(seq, seq[1:], seq[2:]):
            if before[2] == "tool" and reply[2] == "assistant" and reply[3] == EMPTY_RECOVERY_REPLY and not reply[4] \
                    and nudge[2] == "user" and nudge[3] == EMPTY_RECOVERY_NUDGE:
                (ambiguous if group(session) in saved else matched).update({reply[0], nudge[0]})
    return matched, ambiguous


def todo_expected(host_rows, stored_rows, group, header):
    """Host-displayed todo rows a stored user row copies (#2377): a stored user row whose content equals, byte for
    byte, a displayed host user row that starts with the todo injection header (``header``), of the same lineage and
    at the same stamp (observed_at equals the host timestamp), is expected; each displayed occurrence matches one
    stored row. Every other todo row stays out of the expected rows (``host_expected``), and a displayed one with no
    stored copy is no deficit. ``stored_rows``: (store_id, session, role, content, tool_calls, tool_call_id,
    observed_at). Returns {lineage: [store id of each matched row]}: the stamp-equal row is the kept copy, taken out
    of scoring with its host occurrence, so an earlier copy of the same text stays surplus."""
    shown = {(group(r[1]), r[3], float(r[7])) for r in host_rows if r[2] == "user" and r[9] and not r[6]
             and r[7] is not None and (r[3] or "").startswith(header)}
    out = defaultdict(list)
    for _store_id, session, role, content, _calls, _call_id, observed in sorted(stored_rows):
        slot = (group(session), content, float(observed)) if role == "user" and observed is not None else None
        if slot in shown:
            shown.discard(slot)
            out[slot[0]].append(_store_id)
    return out


def read_store(path: Path):
    with ro(path) as con:
        mc = columns(con, "messages")
        rows = con.execute("select store_id, session_id, role, content, tool_calls, tool_call_id, " +
                           ", ".join(opt(mc, c, "NULL") for c in ("observed_at", "ingested_at")) +
                           " from messages order by store_id").fetchall()
        # store.py:1321-1329 message_relations(store_id, kind, related_store_id, ...); store_id is the row the
        # relation is on: engine.py:7486 writes (copy, 'possible_replay_of', target), engine.py:7547 (row, 'generated_unproven');
        # composite_of segments carry their ordinal (multiset.composite_witness).
        relations = con.execute("select store_id, kind, related_store_id, " + opt(
            columns(con, "message_relations"), "ordinal", "NULL") + " from message_relations").fetchall() \
            if has_table(con, "message_relations") else []
        uids = {r[0] for r in con.execute("select store_id from host_uid_bindings where kind = 'canonical'")} \
            if has_table(con, "host_uid_bindings") else None  # store.py:1477-1481
        lineage = {r[0]: (r[1], r[2]) for r in con.execute(
            "select child_session_id, parent_session_id, source from session_lineage")} \
            if has_table(con, "session_lineage") else {}  # store.py:1597-1598
        leaves = Counter(r[0] for r in con.execute(
            "select session_id from summary_nodes where depth = 0 and source_type = 'messages'")) \
            if has_table(con, "summary_nodes") else Counter()  # dag.py:208-221
        nodes = {r[0]: r[1:] for r in con.execute(
            "select node_id, session_id, depth, summary, expand_hint from summary_nodes")} \
            if has_table(con, "summary_nodes") else {}
    return rows, relations, uids, lineage, leaves, nodes


def log_time(stamp: str, tz, frac=None):
    """A log asctime as epoch seconds, read in ``tz`` (a zoneinfo name), or in this machine's zone when None; ``frac``:
    the digits after the seconds' comma or point (Hermes logs milliseconds), kept so sub-second order survives."""
    stamp, sub = stamp.replace("T", " "), float(f"0.{frac}") if frac else 0.0
    if tz is None:
        return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S")) + sub
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(tz)).timestamp() + sub


def logged_link(msg):
    """#2377's logged intended link: the source store id of a well-formed line (integer ids, a known reason code),
    False for a malformed one or an unknown reason (no credit, counted as unparsed), None for any other message."""
    if not msg.startswith(LINK_FAILED):
        return None
    hit = LINK_LINE.fullmatch(msg)
    return int(hit.group(1)) if hit else False


def logged_link_credits(links, extra, since):
    """#2377's logged intended links: the store ids they credit, and the well-formed lines out of scope. A line credits
    its source row only when it is a WARNING, its session field names the row's session (exactly, as the route lines'
    session is matched), and its time is in the run window (at or after ``since``) and not before the row was stored
    (its ingested_at, less LINK_STORE_SLACK). ``links``: (source, session, at, level) of each well-formed line."""
    credited, out = set(), 0
    for source, session, at, level in links:
        row = extra.get(source)
        if row is not None and level == "WARNING" and session == row[1] and (since is None or at >= since) and (
                row[7] is None or at >= row[7] - LINK_STORE_SLACK):
            credited.add(source)
        else:
            out += 1
    return credited, out


def read_logs(paths, tz=None):
    """LCM-X's own log lines: compactions (deduplicated across files), AUXILIARY lines, recorder lines and logged
    intended links (``logged_link``)."""
    out, seen_ingests = {"compactions": set(), "auxiliary": defaultdict(set), "ingests": [], "links": set(),
                         "links_unparsed": set(), "link_computation_failed": set()}, set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            m = LOG_LINE.match(line)
            if not m or not LCM_LOGGER.search(m["logger"] or m["logger2"] or m["logger3"]):
                continue
            sid, msg, at = m["sid"] or m["sid2"], m["msg"], log_time(m["ts"], tz, m["frac"])
            if COMPACTION.match(msg):
                in_message = MESSAGE_SESSION.search(msg)
                # A line logged to two files counts once; no session yet means none in the line (ACP stderr).
                out["compactions"].add((sid or (in_message.group(1) if in_message else None), at, msg))
            if hit := RECLASSIFIED.match(msg):
                out["auxiliary"][hit.group(1)].add("first_ingest_reclassification")
            if hit := DECISION.match(msg):
                out["auxiliary"][hit.group(1)].add(hit.group(2) or "")
            if hit := ROUTE.match(msg):
                record = {"session": hit["session"], "route": hit["route"].strip().replace(" ", "_"),
                          "reason": hit["reason"].strip(), "cursor": int(hit["cursor"]), "at": at,
                          "range": (int(hit["a"]), int(hit["b"])), "rows": int(hit["rows"]),
                          "pairs": int(hit["pairs"]), "stripped": int(hit["stripped"]),
                          "link_pending": int(pending.group(1)) if (
                              pending := ROUTE_LINK_PENDING.search(msg, hit.end())) else None}
                identity = tuple(sorted(record.items()))
                if identity not in seen_ingests:  # a line logged to two files is one ingest
                    seen_ingests.add(identity)
                    out["ingests"].append(record)
            if LINK_COMPUTATION_FAILED.fullmatch(msg):  # a line logged to two files counts once
                out["link_computation_failed"].add((at, msg))
            if (link := logged_link(msg)) is False:  # a line logged to two files counts once
                out["links_unparsed"].add((at, msg))
            elif link is not None:
                out["links"].add((link, sid, at, m["level"] or m["level2"] or m["level3"]))
    # ACP stderr omits the session: prefer a matching session-bearing file record, while keeping distinct sessions.
    session_records = {(at, msg) for sid, at, msg in out["compactions"] if sid is not None}
    out["compactions"] = {(sid, at, msg) for sid, at, msg in out["compactions"]
                          if sid is not None or (at, msg) not in session_records}
    return out


def row_key(role, content, calls, call_id):
    """A row's stored identity: role, edge-stripped content, tool_call_id and its tool-call keys."""
    return role, multiset.norm(content or ""), call_id, tuple(sorted(
        map(repr, tool_calls.stored_keys([(0, "", role, content, calls, call_id)]))))


def host_row_key(row, strip):
    """``row_key`` of a host row, a user row's verified summary carrier taken off (``strip``)."""
    _rid, sid, role, content, calls, call_id = row[:6]
    return row_key(role, strip(content, sid) if role == "user" else content, calls, call_id)


def undo_retained_rows(stored_rows, host_rows, strip):
    """Stored rows that are a host undo's retained occurrence: neither expected nor surplus.

    A TUI undo soft-deletes the undone rows in the host (hermes_state_messages.py rewind_to_message: active = 0, with
    compacted left 0), so they are not displayed and not expected; LCM-X keeps the copy it stored and stores the
    resend as a new occurrence. A stored row is undo_retained when the host holds an undone row (active = 0 and
    compacted = 0) of the same session and the same row key, and the stored row's observed_at equals that host row's
    timestamp, which proves it is that occurrence. Each undone host row excuses at most one stored copy; a stored row
    with NULL observed_at (an unstamped host) is never excused. An undone row with the same session, key and timestamp
    as a displayed row excuses nothing: an in-place compaction (and replace_messages archive_dropped, and
    deactivate_message) leaves the original inactive next to a live clone that takes its timestamp, and the one stored
    copy is that displayed row's. ``host_rows``: the host's rows (index 9 displayed, 10 undone);
    ``stored_rows``: (store_id, session, role, content, tool_calls, tool_call_id, observed_at)."""
    pool, live = Counter(), set()
    for row in host_rows:
        if row[7] is not None and (row[9] or row[10]):
            slot = (row[1], host_row_key(row, strip), float(row[7]))
            if row[9]:
                live.add(slot)
            else:
                pool[slot] += 1
    for slot in live:
        pool.pop(slot, None)
    excused = []
    for store_id, session, role, content, calls, call_id, observed in sorted(stored_rows):
        slot = (session, row_key(role, content, calls, call_id), float(observed)) if observed is not None else None
        if slot is not None and pool[slot] > 0:
            pool[slot] -= 1
            excused.append(store_id)
    return excused


def attribute_compactions(lines, events):
    """Give each compaction line without a session the session of the nearest unused driver compress_request whose
    time it falls within (COMPRESS_REQUEST_WINDOW); a line left without one keeps None (it counts toward compactions,
    never toward compactions_then_restart). Returns the lines and how many stayed without a session."""
    requests, pending = [], {}
    for e in events or ():
        sid = str(e.get("session") or "")
        if e["kind"] == "compress_request":
            pending[sid] = len(requests)
            requests.append((float(e.get("t") or 0), sid))
        elif e["kind"] == "compress_request_failed" and sid in pending:
            requests[pending.pop(sid)] = None
    requests = sorted(r for r in requests if r is not None)
    used, out = set(), []
    for sid, at in lines:
        if sid is None and at is not None:
            near = [(abs(at - t), i) for i, (t, _s) in enumerate(requests) if i not in used
                    and t - COMPRESS_REQUEST_WINDOW[0] <= at <= t + COMPRESS_REQUEST_WINDOW[1]]
            if near:
                i = min(near)[1]
                used.add(i)
                sid = requests[i][1]
        out.append((sid, at))
    return out, sum(1 for sid, _at in out if sid is None)


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


def stamp_swaps(stored_rows, host_rows, strip):
    """Hand-check detector (it changes no count): parity matches rows by content, not time. For a row key the host
    holds more than once in a session (displayed or undone rows, a clone with its original's key and time once), when
    every stored copy of that key in the session is stamped and their observed_at values are not contained in the
    host's timestamps of the key (as multisets), one occurrence may stand in for another (an undone row kept twice
    while its resend is lost, say). Unstamped copies are not judged here. Returns the hand checks, with the key's
    store ids. ``stored_rows``: (store_id, session, role, content, tool_calls, tool_call_id, observed_at)."""
    host = defaultdict(set)
    for row in host_rows:
        if not row[6] and (row[9] or row[10]) and row[7] is not None:
            host[(row[1], host_row_key(row, strip))].add(float(row[7]))
    stored = defaultdict(list)
    for store_id, session, role, content, calls, call_id, observed in stored_rows:
        stored[(session, row_key(role, content, calls, call_id))].append((store_id, observed))
    out = []
    for slot, copies in sorted(stored.items(), key=lambda c: min(c[1])):
        times = host.get(slot, set())
        if len(times) > 1 and all(o is not None for _i, o in copies) and (
                Counter(float(o) for _i, o in copies) - Counter(times)):
            out.append({"reason": "stamp_swap", "session": slot[0], "store_ids": sorted(i for i, _o in copies)})
    return out


def view_order(host_rows, stored_rows, strip):
    """Section 5's view-order condition of #1674 and #2203, as far as the store and the host establish it: returns
    ``in_order(session, copy)``: "ordered", "ambiguous" or "unplaced" for the session's stored rows before store id
    ``copy``. The host's rows are every row it keeps (displayed or undone, not its summary row) in id order, a carried
    clone that repeats an earlier row's key and timestamp dropped; the view an ingest saw is a prefix of them. Each
    stored row takes one host row of its key, each host row at most once, in order, with no exception: "unplaced"
    when no such matching exists (a host insertion stored before the copy, as a #2305 insert drained through a retry
    route, a reordering, or an earlier re-stored copy). "ambiguous" when repeated text allows more than one matching
    (the earliest and the latest matching up to the same last host row differ); else "ordered".
    ``stored_rows``: (store_id, session, role, content, tool_calls, tool_call_id), content resolved."""
    host, seen = defaultdict(list), set()
    for row in host_rows:
        if row[6] or not (row[9] or row[10]):
            continue
        slot = (row[1], host_row_key(row, strip), row[7])
        if slot not in seen:
            seen.add(slot)
            host[row[1]].append(slot[1])
    stored = defaultdict(list)
    for store_id, session, role, content, calls, call_id in sorted(stored_rows):
        stored[session].append((store_id, row_key(role, content, calls, call_id)))

    def in_order(session, copy):
        keys, prefix, earliest = host.get(session, []), [k for i, k in stored[session] if i < copy], []
        for key in prefix:
            if key not in keys[earliest[-1] + 1 if earliest else 0:]:
                return "unplaced"
            earliest.append(keys.index(key, earliest[-1] + 1 if earliest else 0))
        latest, pos = [], earliest[-1] if earliest else -1
        for key in reversed(prefix):
            pos = max(i for i in range(pos + 1) if keys[i] == key)
            latest.append(pos)
            pos -= 1
        return "ordered" if earliest == latest[::-1] else "ambiguous"
    return in_order


def attribute_silent_surplus(silent, ingests, observed, user_rows, restarts, in_order):
    """Section 5 (r3.8) "Surplus and other limits": attribute each silent surplus row to #1674, #2203, #849 or #1961
    ONLY from the one recorder line whose session matches and whose stored range holds the row; zero or several such
    lines leave the row unattributed (ambiguity fails as silent surplus). Only #1674 and #2203 are exempted here;
    #849 and #1961 rows are listed for a hand check and stay silent surplus for the gate. Line numbers cite the
    recorder PR's engine.py (``_log_ingest_route`` at 7140-7160).
    ``silent``: [(store_id, session, [(store_id, session)] of the copies of its key)]; ``observed``: store_id -> observed_at;
    ``user_rows``: store_id -> (session, text) for stored user rows (compared in memory only); ``restarts``: lineage ->
    driver restart times (restart_cold, restart_cold_shared, restart_warm), or None without --events; ``in_order``: ``view_order``'s check.
    #1674 and #2203 need the view-order condition (no host insertion or reordering) on the session's stored rows
    before the copy: "unplaced" leaves the row unattributed; "ambiguous" (repeated text) lists it under
    ``order_ambiguous``, a hand check, not an exemption. #2203 also needs an earlier copy of the key in the session.
    An exempted row is still a hand check: the verdict is then GATE PASS PENDING HAND CHECK, never GATE PASS."""
    out = {"#1674": [], "#2203": [], "order_ambiguous": [], "#1674_needs_manual_check": [],
           "#849_needs_manual_check": [], "#1961_needs_manual_check": [], "unattributed": []}
    prompts = defaultdict(list)  # session -> [(store_id, edge-stripped user text)]
    for other, (s, t) in user_rows.items():
        if multiset.norm(t):
            prompts[s].append((other, multiset.norm(t)))
    for sid, session, ids in silent:  # ids: [(store_id, session)] of every stored copy of the row's key
        lines = [i for i in ingests if i["session"] == session and i["range"][0] <= sid <= i["range"][1]]
        limit = "unattributed"
        if len(lines) == 1:
            route, cursor, reason = lines[0]["route"], lines[0]["cursor"], lines[0]["reason"]
            # #1674 and #2203 both need an unstamped copy and the view-order condition (``in_order``).
            order = in_order(session, sid) if observed.get(sid) is None else "unplaced"
            eligible, held = order != "unplaced", "#1674" if order == "ordered" else "order_ambiguous"
            text = multiset.norm(user_rows.get(sid, (None, ""))[1])
            if route == ROUTE_2305 and cursor == 0:
                pass  # #2305's cursor-0 persisted batch: never exempted, never #1674 (section 5)
            # #1674 (restart replay): the first ingest of a re-bound engine reconciles, and the recorder logs the
            # reconciliation action as the route (engine.py:8218); "advanced cursor" past a replayed prefix
            # (reconcile.py, every replayed-prefix reason) with cursor > 0. The copy is unstamped and an earlier copy
            # of its key in the session is stamped; section 5 says the stamp condition alone does not qualify, so a
            # driver restart of the session must also come before the recorder line (none known without --events:
            # hand check). The view's rows before the copy follow the stored rows in order (``eligible``).
            elif route == "advanced_cursor" and cursor > 0 and eligible and any(
                    other < sid and other_session == session and observed.get(other) is not None
                    for other, other_session in ids):
                if restarts is None or lines[0]["at"] is None:
                    limit = "#1674_needs_manual_check"
                elif any(t < lines[0]["at"] for t in restarts.get(session, ())):
                    limit = held
            # #2203: a cold-start retry of a queued tail, through the ingest (route "cold", engine.py:7141) or the S3
            # retry (route "S3", engine.py:11794); or the exit drain, route "session_end" logged by the pending-tail
            # retry with exiting=True (engine.py:11844), told apart from an ordinary session-end flush (also
            # "session_end", engine.py:7142) by its reason, the tail match rule. The copy is unstamped, and the view
            # differs from the stored rows by no host insertion or reordering (``eligible``). "warm_tail" (a retry
            # that is not the exit drain) is neither section 5 shape, so it is not attributed.
            elif eligible and any(other < sid and other_session == session for other, other_session in ids) and (
                    route in ("cold", "S3") or (route == "session_end" and reason in TAIL_RULES)):
                limit = "#2203" if order == "ordered" else "order_ambiguous"
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


def score_lineage(g, items, tools, rows, relations, stored=None, forms=None):
    """B2's instruments on one lineage; returns numbers and the silent rows [(store_id, ids of its key)].

    ``stored`` maps store_id -> (session, role, content) as stored: the product links replay copies on stored
    bytes, so a link is checked against stored content, not resolved content (as in the B2 scorer). ``forms``:
    ``multiset.attribution_forms`` of the whole store, computed once."""
    groups = []
    stored4 = [r[:4] for r in rows]
    ms = multiset.score(items, stored4, None, surplus_keys=groups)
    ids = {}
    tc = tool_calls.compare(tools, set(), tool_calls.stored_keys([r[:6] for r in rows], lambda _s: g, store_ids=ids),
                            store_ids=ids, surplus_keys=groups)
    visible, silent, truncated, credited = 0, [], 0, Counter()
    for group in groups:  # split_surplus matches within one key, so one call per key loses nothing
        split = multiset.split_surplus([group], relations, stored, forms)
        visible += split["visible_surplus_rows"]
        credited.update({k: split[k] for k in forms or ()})
        found = [e["store_id"] for e in split["silent_surplus_examples"]]
        truncated += split["silent_surplus_rows"] - len(found)  # over 10 in one key: left unattributable
        silent += [(sid, group[2]) for sid in found]
    return {"expected_rows": len(items) + sum(tools.values()), "stored_rows": len(rows),
            "missing_rows": ms["deficit_rows"] + tc["tool_missing_rows"], "split_assistant_keys": ms["split_keys"],
            "surplus_rows": ms["surplus_rows"] + tc["tool_surplus_rows"], "visible_surplus_rows": visible,
            "silent_surplus_rows": len(silent) + truncated, "silent_unlisted_rows": truncated,
            **{k: credited[k] for k in forms or ()}}, silent


def audit(state_db: Path, lcm_db: Path, logs=(), since=None, events_path=None, log_tz=None) -> dict:
    sessions, host_rows, has_compacted = read_host(state_db)
    stored, relations, uids, store_lineage, leaves, nodes = read_store(lcm_db)
    log = read_logs(logs, log_tz)
    events = read_events(events_path)
    group = lineage_of(sessions)
    aux, aux_unproven = auxiliary_sessions(log["auxiliary"])
    in_window = {s for s, v in sessions.items() if since is None or (v["started"] or 0) >= since}
    # A lineage is scored when any member started in the window; only its in-window members (``scored``) count, so a
    # rotation child in the window is scored even when its parent started before it, without the parent's rows.
    scored = in_window - aux
    roots = {group(s) for s in scored}
    markers = plugin_tree.carrier_markers(REPO)

    def strip(content, sid):  # a user row's text behind a summary carrier its store verifies (``carrier_remainder``)
        return carrier_remainder(content, markers[0], nodes, lambda s: group(s) == group(sid))
    expected, carriers, hand_checks = host_expected([r for r in host_rows if group(r[1]) in roots], group, markers,
                                                    strip, scored)
    # Payloads sit beside the state.db the link points to, not beside a symlink to it.
    resolved, ext, _ids = externalized.resolve([r[:6] for r in stored], Path(state_db).resolve().parent)
    extra = {r[0]: r for r in stored}
    stored_content = {r[0]: (r[1], r[2], r[3]) for r in stored}

    def audited(store_id):  # a scored session's row, or a row under no host session stored in the window
        session = extra[store_id][1]
        return session in scored or (session not in sessions and session not in aux
                                     and (since is None or (extra[store_id][7] or 0) >= since))

    by_lineage = defaultdict(list)
    for r in resolved:
        if audited(r[0]):
            by_lineage[group(r[1])].append((*r, extra[r[0]][6]))
    lost_ids = [s for s in ext["store_ids"] if audited(s)]  # unreadable payloads: kept, even with no readable row
    retained = set(undo_retained_rows([r for rows in by_lineage.values() for r in rows],
                                      [r for r in host_rows if r[1] in scored], strip))
    hand_checks += stamp_swaps([r for rows in by_lineage.values() for r in rows],
                               [r for r in host_rows if r[1] in scored], strip)
    for g, rows in by_lineage.items():  # kept rows of an undone occurrence: out of parity, surplus and M3's numerator
        by_lineage[g] = [r for r in rows if r[0] not in retained]
    # Matched on every stored row of the audited sessions, before any is set aside, so a removed row never makes two
    # rows adjacent; an undo's retained row stays the undo's.
    recovery, ambiguous_recovery = empty_recovery_rows([r for r in stored if audited(r[0])],
                                                       [r for r in host_rows if group(r[1]) in roots], group, strip)
    recovery, ambiguous_recovery = recovery - retained, ambiguous_recovery - retained
    if ambiguous_recovery:  # scored, but parity alone cannot tell a copy of a real row from scaffolding for a lost one
        hand_checks.append({"reason": "host_empty_recovery_ambiguous", "store_ids": sorted(ambiguous_recovery)})
    for g, rows in by_lineage.items():  # the host's recovery scaffolding (#2087): out of parity, surplus and M3's numerator
        by_lineage[g] = [r for r in rows if r[0] not in recovery]
    todo = todo_expected([r for r in host_rows if r[1] in scored], [r for rows in by_lineage.values() for r in rows],
                         group, multiset.todo_header())
    kept_todo = {i for ids in todo.values() for i in ids}
    for g, rows in by_lineage.items():  # a matched todo row and its host occurrence: expected and stored, not scored
        by_lineage[g] = [r for r in rows if r[0] not in kept_todo]
    lost = Counter(group(extra[s][1]) for s in lost_ids)
    logged, links_out_of_scope = logged_link_credits(log["links"], extra, since)
    forms = multiset.attribution_forms(relations, stored_content, logged)
    parity, silent_rows = {}, []
    for g in sorted(roots | set(by_lineage) | set(lost)):
        items, tools = expected.get(g, ([], Counter()))
        numbers, silent = score_lineage(g, items, tools, by_lineage[g], relations, stored_content, forms)
        numbers["stored_rows"] += lost[g] + len(todo[g])
        numbers["expected_rows"] += len(todo[g])
        numbers.update(externalized_unresolved=lost[g], host_carrier_rows_excluded=carriers[g],
                       host_todo_rows_expected=len(todo[g]),
                       source=sessions.get(g, {}).get("source"), host_session=g in sessions)
        parity[g] = numbers
        silent_rows += [(sid, extra[sid][1], [(i, extra[i][1]) for i in ids]) for sid, ids in silent]

    # Whole-run parity: every scored lineage, #1931 bucket B children included (only PASS(A) is bucket A's own bar).
    keys = ("expected_rows", "stored_rows", "missing_rows", "split_assistant_keys", "surplus_rows",
            "visible_surplus_rows", "silent_surplus_rows", "silent_unlisted_rows", "externalized_unresolved",
            "host_todo_rows_expected", *forms)
    totals = {k: sum(p[k] for p in parity.values()) for k in keys}
    zero = sorted(g for g, p in parity.items() if p["host_session"] and p["expected_rows"] and not p["stored_rows"])
    per_session = Counter(r[1] for r in stored)
    window_ids = {r[0] for rows in by_lineage.values() for r in rows} | set(lost_ids) | retained | kept_todo | recovery
    audited_sessions = scored | {extra[i][1] for i in window_ids}
    denominator = len(window_ids)

    # #1931 buckets: ended children that are not rotation children, started in the window, not AUXILIARY.
    buckets = {"A": [], "B": []}
    for s in sorted(scored):
        v = sessions[s]
        if eligible_child(s, sessions):
            buckets["A" if v["source"] == DELEGATE_SOURCE else "B"].append(s)

    # Coverage floors (section 5, "The driven run").
    turns = sum(1 for g in parity for role, _t in expected.get(g, ([],))[0] if role == "user")
    calls = Counter((k[3], k[4]) for g in parity for k, n in expected.get(g, ([], Counter()))[1].items()
                    if k[1] == "call" for _ in range(n))
    if events is not None:  # driver evidence counts only for audited sessions, inside the window
        events = [e for e in events if str(e.get("session") or "") in audited_sessions
                  and (since is None or float(e.get("t") or 0) >= since)]
    kinds = Counter(e["kind"] for e in events) if events is not None else None
    # Compaction lines: deduplicated, inside the window when one is given, of an audited session or of none.
    compaction_lines = sorted(((sid, at) for sid, at, _msg in log["compactions"]
                               if (sid is None or sid in audited_sessions) and (since is None or at >= since)),
                              key=lambda c: (c[1], c[0] or ""))
    compaction_lines, unattributed_compactions = attribute_compactions(compaction_lines, events)
    restarts = None
    if events is not None:
        restarts = defaultdict(list)
        for e in events:
            if e["kind"] in RESTARTS:
                restarts[group(str(e.get("session") or ""))].append(float(e.get("t") or 0))
    then_restart = "unknown" if not logs or restarts is None else sum(
        1 for sid, at in compaction_lines if sid and any(t > at for t in restarts.get(group(sid), ())))
    value = {"turns": turns, "delegated_children": sum(1 for k in buckets["A"] if k in parity),
             "compactions": len(compaction_lines) if logs else "unknown", "compactions_then_restart": then_restart,
             "tool_calls": sum(calls.values()), "tool_call_repeats": sum(n - 1 for n in calls.values()),
             "host_sources": len({sessions[s]["source"] for s in scored} - NOT_ENTRY_SOURCES)}
    for name, kind in (("restarts_cold", "restart_cold"), ("restarts_warm", "restart_warm"), ("undo_resend", "undo_resend")):
        value[name] = kinds[kind] if kinds is not None else "unknown"
    value["restarts"] = value["restarts_cold"] + value["restarts_warm"] if kinds is not None else "unknown"
    coverage = {k: {"value": value[k], "floor": f, "met": isinstance(value[k], int) and value[k] >= f}
                for k, f in FLOORS.items()}
    coverage_basis = {"compactions": "log lines 'LCM compaction #' of LCM-X's logger, deduplicated (compaction.py:2910);"
                                     " a line without a session takes its message's session=, else the nearest"
                                     " driver compress_request, else counts only here" if logs else "no log given",
                      "compactions_without_session": unattributed_compactions,
                      "compress_request_window_s": list(COMPRESS_REQUEST_WINDOW),
                      "store_depth0_message_leaves": sum(n for s, n in leaves.items() if group(s) in parity),
                      "delegated_children": "ended, scored #1931 bucket A children",
                      "host_sources": "distinct scored sessions.source in the window, without subagent and tool",
                      "restarts_undo": "driver events" if events is not None else "no events given",
                      "expected_rows": "host display identity per lineage" if has_compacted else
                                       "host display identity per lineage (no compacted column: every row displayed)"}

    # Uid coverage per host source (#2237 ruling A): per distinct message, with copies reported apart. A copy is an
    # unstamped row (no observed_at) with the role, tool call id, tool calls and content of an earlier stored row of
    # its session; every other row is a distinct message. A message is covered when any of its stored rows has a
    # canonical host_uid_bindings row (a uid binds one row of a lineage, so its copies stay unbound).
    uid = defaultdict(lambda: {"rows_stored": 0, "rows_with_canonical_uid": 0, "messages": 0,
                               "messages_with_canonical_uid": 0, "copy_rows": 0, "copy_rows_with_canonical_uid": 0})
    message_of, bound_messages, first = {}, set(), {}
    for r in sorted(stored):  # every stored row, so a copy of a row stored before the window is still a copy
        key = (r[1], r[2], r[5], r[4], r[3])
        message_of[r[0]] = first[key] if r[6] is None and key in first else r[0]
        first.setdefault(key, r[0])
        if uids is not None and r[0] in uids and message_of[r[0]] == r[0]:
            bound_messages.add(r[0])  # a copy's own binding covers no message: it counts in copy_rows_with_canonical_uid
    for sid in window_ids:
        src = sessions.get(extra[sid][1], {}).get("source") or "unknown"
        bound = int(uids is not None and sid in uids)
        uid[src]["rows_stored"] += 1
        uid[src]["rows_with_canonical_uid"] += bound
        if message_of[sid] == sid:
            uid[src]["messages"] += 1
            uid[src]["messages_with_canonical_uid"] += int(sid in bound_messages)
        else:
            uid[src]["copy_rows"] += 1
            uid[src]["copy_rows_with_canonical_uid"] += bound

    # M3: rows carrying generated_unproven, or the store_id side of a possible_replay_of pair.
    # An undo_retained row leaves M3 for a possible_replay_of only: a generated_unproven row is LCM's own output.
    # A recovery row (the full pair rule: host provenance plus adjacency) leaves M3 for either relation: the host wrote
    # it, so neither label is a doubt about a host row. The rows it takes out are counted and named, never silent.
    labelled = ({s for s, kind, *_rest in relations if kind == "possible_replay_of"} - retained
                | {s for s, kind, *_rest in relations if kind == "generated_unproven"}) & window_ids
    doubt, recovery_doubt = sorted(labelled - recovery), sorted(labelled & recovery)
    m3 = {"numerator": len(doubt), "denominator": denominator, "store_ids": doubt,
          "rate": len(doubt) / denominator if denominator else 0.0,
          "recovery_rows_excluded": len(recovery_doubt), "recovery_store_ids_excluded": recovery_doubt}

    ingests = [i for i in log["ingests"] if i["session"] in audited_sessions and (since is None or i["at"] >= since)]
    storing = [i for i in ingests if i["rows"] > 0]
    pairs = sum(i["pairs"] for i in ingests)
    recorder = {"present": bool(ingests), **({"status": "recorder absent"} if not ingests else {}),
                "ingests_per_route": dict(sorted(Counter(i["route"] for i in ingests).items())),
                "storing_ingests": len(storing),
                "rate_2305": sum(i["route"] == ROUTE_2305 and i["cursor"] == 0 for i in storing) / len(storing)
                if storing else None,
                "stripped_links": sum(i["stripped"] for i in ingests), "replay_pairs": pairs,
                "stripped_per_replay_pair": sum(i["stripped"] for i in ingests) / pairs if pairs else None,
                # #2377 cross-checks: route lines' link_pending field (None when no line has it), malformed links.
                "link_pending_logged": sum(i["link_pending"] for i in ingests if i["link_pending"] is not None)
                if any(i["link_pending"] is not None for i in ingests) else None,
                "logged_link_unparsed": len(log["links_unparsed"]), "logged_link_out_of_scope": links_out_of_scope,
                "link_computation_failed": len(log["link_computation_failed"])}

    observed = {r[0]: r[6] for r in stored}
    user_rows = {r[0]: (r[1], r[3]) for r in resolved if r[2] == "user"}
    session_restarts = None if restarts is None else {s: restarts.get(group(s), []) for s in sessions}
    # The proven recovery rows are out of parity, so they stay out of the view-order condition too (#2087).
    attributed = attribute_silent_surplus(sorted(silent_rows, key=lambda r: r[0]), ingests, observed, user_rows,
                                          session_restarts, view_order(host_rows, [r for r in resolved if r[0] not in recovery], strip))
    # Silent surplus counts copies per key, so a row holding two keys (content and a tool call) is listed once per
    # key; the gate's unattributed count stays in copies, and the row-based fraction counts distinct store ids.
    # An order_ambiguous row is held for a hand check: not unattributed, and counted in the fraction like an exemption.
    exempt_copies = sum(len(attributed[k]) for k in ("#1674", "#2203", "order_ambiguous"))
    exempt_ids = set(attributed["#1674"]) | set(attributed["#2203"])
    held_ids = exempt_ids | set(attributed["order_ambiguous"])
    exempt = len(exempt_ids)
    unattributed = totals["silent_surplus_rows"] - exempt_copies
    exempted = {"per_limit": {k: len(set(v)) for k, v in attributed.items()},
                "store_ids": {k: sorted(set(v)) for k, v in attributed.items()}, "exempted_rows": exempt,
                "unattributed_silent_rows": unattributed, "unattributed_silent_rows_unit": "key copies",
                # Distinct store rows among them; rows past split_surplus's 10-example list have no id, each counted.
                "unattributed_silent_distinct_rows": len({r[0] for r in silent_rows} - held_ids)
                + totals["silent_unlisted_rows"], "stored_rows": denominator,
                "fraction": len(held_ids) / denominator if denominator else 0.0}
    exempted["met"] = exempted["fraction"] <= BAR
    for limit, reason in (("#1674", "exempted_1674"), ("#2203", "exempted_2203"), ("order_ambiguous", "view_order_ambiguous")):
        if attributed[limit]:
            hand_checks.append({"reason": reason, "store_ids": sorted(set(attributed[limit]))})

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
    marked = sorted(forms["link_pending_rows"] & window_ids)
    if marked:  # #2377: link computation raised for these rows, so a credited copy never reads as a clean pass
        hand_checks.append({"reason": "link_pending_marked", "store_ids": marked})
    failed_links = sorted(set(logged) & window_ids)
    if failed_links:  # the relation write or the pending mark failed: logged, and held for a hand check
        hand_checks.append({"reason": "logged_link_failed", "store_ids": failed_links})
    data_bars = [b for b in bars if b != "coverage_floors"]
    verdict = "GATE FAIL" if not all(bars[b]["pass"] for b in data_bars) else \
        "INCOMPLETE" if not bars["coverage_floors"]["pass"] or not ingests else \
        "GATE PASS PENDING HAND CHECK" if hand_checks else "GATE PASS"
    return {"instrument": "gate-kit-audit-v1", "claim_class": "instrument",
            "inputs": {"since": since, "logs": len(logs), "events": events is not None, "log_tz": log_tz},
            "parity": {"totals": totals, "lineages": parity, "zero_row_lineages": zero,
                       "link_pending_rows_total": len(forms["link_pending_rows"] & window_ids),
                       "sessions": {s: {"lineage": group(s), "source": v["source"], "stored_rows": per_session[s]}
                                    for s, v in sessions.items() if group(s) in parity}},
            "coverage": coverage, "coverage_basis": coverage_basis,
            "uid_coverage": {"table": "present" if uids is not None else "table absent", "per_source": dict(uid)},
            "m3": m3, "recorder": recorder,
            "host_empty_recovery": {"rows": len(recovery), "store_ids": sorted(recovery),
                                    "ambiguous_store_ids": sorted(ambiguous_recovery)},
            "undo_retained": {"rows": len(retained), "store_ids": sorted(retained),
                              "host_rewind_count": {s: sessions[s]["rewind_count"] for s in sorted(scored)
                                                    if sessions[s]["rewind_count"]}}, "exempted_surplus": exempted, "issue_1931": issue_1931,
            "gate": {"bars": bars, "verdict": verdict, "hand_checks": hand_checks}}


def out_refusal(out: Path, inputs, databases):
    """Why ``--out`` may not be written, or None: it is (or resolves to, or is a hard link of) an input, or it sits
    in or under a database's directory (including the audited home's externalized payloads)."""
    target = out.resolve()
    for path in inputs:
        if path is not None and Path(path).exists() and (
                target == Path(path).resolve() or out.exists() and os.path.samefile(out, path)):
            return f"--out is the input {Path(path).name}"
    for db in databases:
        if (target.is_relative_to(Path(db).resolve().parent)
                or out.absolute().is_relative_to(Path(db).absolute().parent)):
            return f"--out is under the directory of {Path(db).name}: write the result elsewhere"
    return None


def write_out(out: Path, text: str):
    """Write through a temporary file in --out's directory, then rename it over --out (a symlink there is replaced,
    never followed)."""
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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
    inputs = [a.state_db, a.lcm_db, a.events, *a.log]
    # Protect payload targets too, including files symlinked out of the audited home.
    inputs.extend((a.state_db.resolve().parent / externalized.PAYLOAD_DIRNAME).glob("*"))
    refusal = out_refusal(a.out, inputs, [a.state_db, a.lcm_db])
    if refusal:
        ap.exit(2, f"audit refused: {refusal}\n")
    try:
        if a.log_tz:
            ZoneInfo(a.log_tz)  # an unknown zone is an input error, before any database is opened
        result = audit(a.state_db, a.lcm_db, a.log, parse_since(a.since), a.events, a.log_tz)
        write_out(a.out, json.dumps(result, indent=1, sort_keys=True) + "\n")
    except (OSError, sqlite3.Error, ValueError, ZoneInfoNotFoundError) as exc:
        ap.exit(2, f"audit failed: {type(exc).__name__}: {exc}\n"[:400])
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
