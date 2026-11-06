"""Conservative forget-request detection and read-only delivery policy.

Rows remain lossless; this module only classifies or reorders read results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
import sqlite3
import time

_PREFIX = r"(?:^|[.!?;:\n])\s*(?:(?:please|kindly|can you|could you|would you|I want you to|I'd like you to|I need you to|just|and)\s+)?"
_PATTERNS = tuple(re.compile(pattern, re.I) for pattern in (
    _PREFIX + r"forget\s+(?:that I|that I'm|that I've|that my|what I (?:said|told you|mentioned|shared)|about my|my|I|I'm|I've)\b",
    _PREFIX + r"forget\s+(?:the|this|that)\s+(?:detail|preference|fact|info|information|note)\s+(?:about|that|on|regarding)\b",
    _PREFIX + r"(?:don't|do not|stop|no longer)\s+(?:remember|store|retain|memorize|keep in mind)\w*\s+(?:that I|what I|my|about me|this about me)\b",
    r"\b(?:erase|delete|remove|wipe|purge|discard)\b[^.!?\n]{0,40}\bfrom (?:your )?memory\b",
    _PREFIX + r"(?:disregard|ignore) what I (?:said|told you|mentioned|shared) about\b",
    r"\bI don't want you to (?:remember|store|retain|know) (?:that|this|my|about)\b",
    r"\bI'd rather you (?:not|didn't) (?:remember|store|retain|know) (?:that|this|my|about)\b",
))
_NON_REQUEST_TEXT = re.compile(r'```[\s\S]*?(?:```|\Z)|`[^`\n]*(?:`|$)|^\s*>[^\n]*|"[^"]*(?:"|\Z)|“[^”]*(?:”|\Z)|‘[^’]*(?:’|\Z)', re.M)
_CONTRACTIONS = frozenset("""
    i'm i've i'd i'll it's let's that's what's who's he's she's there's here's
    where's when's how's don't doesn't didn't can't couldn't won't wouldn't
    shouldn't isn't aren't wasn't weren't haven't hasn't hadn't we're we've
    we'd we'll you're you've you'd you'll they're they've they'd they'll
    mustn't needn't mightn't shan't ain't
""".split())
_STOPWORDS = frozenset("""
    a an the i me my mine you your yours we our us they their he she his her it its
    is am are was were be been being do does did don t not no never longer to of
    in on at for from with as by and or but if then than so this that these those
    what everything any please kindly can could would want like need just forget
    about remember keep store save retain track record bring up mention reference
    factor use disregard erase delete remove wipe scratch discard purge said
    mentioned told shared preference preferences detail details info information
    history data earlier previous memory rather didn know
""".split())
_KEYWORDS = ('forget', 'remember', 'disregard', 'erase', 'delete', 'remove', 'wipe',
             'scratch', 'discard', 'purge', 'rather you', 'want you to', 'memory', 'memorize', 'retain',
             'store', 'ignore', 'know')
# Every detection pattern has at least one of these terms or phrases (the index has no stemming).
_FTS_QUERY = ('forget OR remember OR remembering OR memory OR memorize OR retain OR disregard '
              'OR "store my" OR "store that" OR "store this" OR "know my" OR "know that" '
              'OR "know this" OR "know about" OR "ignore what"')


def request_text(text: str) -> str:
    """Use the same unquoted prose for detection and coverage detail words."""
    return _NON_REQUEST_TEXT.sub(" ", text).replace("’", "'").replace("‘", "'")


def is_forget_request(text: str, role: str = "user") -> bool:
    text = _NON_REQUEST_TEXT.sub(" ", text).replace("’", "'").replace("‘", "'")
    return role.lower() == "user" and any(p.search(text) for p in _PATTERNS)


def forget_content_words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("’", "'").replace("‘", "'"))
    words = {t.removesuffix("'s") for t in tokens if t not in _CONTRACTIONS}
    return {w for w in words if len(w) >= 3 and w not in _STOPWORDS}


def covers(row: dict, forget: dict) -> bool:
    """Inputs carry store scope and the forget row's two predecessors."""
    if row.get("scope") != forget.get("scope"):
        return False
    # Tool calls and tool results are never withheld or annotated, so a
    # call/result group is never split (precision first).
    if str(row.get("role") or "").lower() == "tool" or row.get("tool_calls"):
        return False
    earlier = row["store_id"] < forget["store_id"]
    if row.get("observed_at") is not None and forget.get("observed_at") is not None:
        earlier = earlier and row["observed_at"] <= forget["observed_at"]
    words = forget_content_words(request_text(forget.get("content") or ""))
    if not earlier or len(words) < 2:
        return False
    shared = len(words & forget_content_words(row.get("content") or ""))
    adjacent = (row["session_id"] == forget["session_id"]
                and row["store_id"] in forget.get("previous_ids", ()))
    # A question never states the detail to be forgotten.
    if str(row.get("content") or "").rstrip().endswith("?"):
        return False
    return adjacent and shared / len(words) >= .3


def event_time(row: dict) -> str | None:
    value = row.get("observed_at")
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def read_forget_rows(db_path, store_ids, *, allowed=None, excluded=(), deadline, subagent_source="subagent"):
    """One private, deadline-bounded snapshot; at most 200 prefiltered USER rows."""
    if time.monotonic() >= deadline:
        return [], {}, "deadline"
    conn = None
    try:
        conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True,
                               timeout=max(.001, deadline - time.monotonic()))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        conn.execute("BEGIN")
        where, params = ["lower(m.role) = 'user'", "coalesce(m.source, '') != ?"], [subagent_source]
        for sessions, operator in ((allowed, "IN"), (excluded, "NOT IN")):
            if sessions is not None:
                sessions = list(sessions)
                where.append(f"m.session_id {operator} ({','.join('?' for _ in sessions)})")
                params.extend(sessions)
        source = "messages m"
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts'").fetchone():
            source = "messages_fts JOIN messages m ON m.store_id=messages_fts.rowid"
            where.append("messages_fts MATCH ?")
            params.append(_FTS_QUERY)
        else:
            where.append("(" + " OR ".join("instr(lower(m.content), ?) > 0" for _ in _KEYWORDS) + ")")
            params.extend(_KEYWORDS)
        ids = list(dict.fromkeys(store_ids))
        rows = conn.execute(
            f"WITH requests AS (SELECT m.store_id FROM {source} WHERE "
            + " AND ".join(where)
            + " ORDER BY coalesce(m.observed_at, m.timestamp) DESC, m.store_id DESC LIMIT 200), "
            "selected AS (SELECT store_id FROM requests UNION SELECT store_id FROM messages "
            f"WHERE store_id IN ({','.join('?' for _ in ids)})) "
            "SELECT m.*, m.store_id IN (SELECT store_id FROM requests) AS potential_forget, "
            "(SELECT json_group_array(store_id) FROM (SELECT store_id FROM messages p "
            "WHERE p.session_id=m.session_id AND p.store_id<m.store_id ORDER BY store_id DESC LIMIT 2)) "
            "AS previous_ids FROM messages m JOIN selected USING(store_id) "
            "ORDER BY coalesce(m.observed_at, m.timestamp) DESC, m.store_id DESC",
            [*params, *ids],
        ).fetchall()
        candidates = {r["store_id"]: {**dict(r), "previous_ids": json.loads(r["previous_ids"])} for r in rows}
        forgets = [r for r in candidates.values() if r["potential_forget"]
                   and is_forget_request(r["content"] or "", r["role"])]
        if time.monotonic() >= deadline:
            return [], {}, "deadline"
        return forgets, candidates, None
    except (sqlite3.Error, OSError):
        return [], {}, "deadline" if time.monotonic() >= deadline else "read_error"
    finally:
        if conn is not None:
            conn.close()


def covering_forget(row, forgets):
    return next((f for f in forgets if covers(row, f)), None)


def withhold(ordered, rows, forgets, limit, current_session):
    """Keep withheld entries below the window, and return their delivery identities."""
    safe, withheld, receipts = [], [], []
    used = set()
    window_ids = {e["hit"].get("store_id") for e in ordered[:limit]
                  if e["hit"].get("kind") != "summary"}
    for entry in ordered:
        hit = entry["hit"]
        ident = hit.get("store_id") if hit.get("kind") != "summary" else None
        row = rows.get(ident)
        forget = covering_forget(row, forgets) if row else None
        if forget is not None:
            withheld.append(entry)
            if ident in window_ids:
                receipts.append(dict(store_id=ident, forget_store_id=forget["store_id"],
                                     forget_event_time=event_time(forget)))
            fid = forget["store_id"]
            # A forget turn can itself be superseded by a later request.
            if fid not in used and fid not in window_ids and covering_forget(forget, forgets) is None:
                replacement = next((e for e in ordered if e["hit"].get("kind") != "summary"
                                    and e["hit"].get("store_id") == fid), None)
                if replacement is None:
                    replacement = {**entry, "ranks": {}, "hit": dict(
                        kind="message_excerpt", store_id=fid, session_id=forget["session_id"],
                        source=forget.get("source") or "", role=forget["role"],
                        timestamp=forget.get("timestamp") or 0, content_offset=0,
                        snippet=forget["content"][:300], expand_hint=f"lcm_expand(store_id={fid})",
                        from_current_session=forget["session_id"] == current_session)}
                safe.append(replacement)
                used.add(fid)
        elif ident is None or ident not in used:
            safe.append(entry)
            if ident is not None:
                used.add(ident)
    return safe + withheld, len(safe), receipts


def filter_session_payload(payload, rows, forgets, current_session, *, seen_refs=()):
    """Session expansion may introduce covered neighbours beyond ranked seeds."""
    hits, receipts, affected_sessions = [], [], set()
    for hit in payload["hits"]:
        row = rows.get(hit.get("store_id"))
        f = covering_forget(row, forgets) if row else None
        if f is None:
            hits.append(hit)
        else:
            affected_sessions.add(hit["session_id"])
            receipts.append(dict(store_id=hit["store_id"], forget_store_id=f["store_id"],
                                 forget_event_time=event_time(f)))
    if not receipts:
        return []
    payload.update(hits=hits, total_results=len(hits))
    delivery = payload["provenance"]["delivery"]
    delivery.update(message_count=len(hits), session_count=len({h["session_id"] for h in hits}))
    for meta in delivery["sessions"]:
        if meta["session_id"] not in affected_sessions:
            continue
        ids = [h["store_id"] for h in hits if h["session_id"] == meta["session_id"]]
        meta.update(complete=False, omission_reason="forget_requested",
                    returned_id_ranges=[[i, i] for i in ids])
    if "delta" in payload:
        refs = [h["exact_ref"] for h in hits if h.get("exact_ref") and h["exact_ref"] not in seen_refs]
        payload["delta"].update(novel_refs=refs, novel_ref_count=len(refs), progress=bool(refs),
                                termination_reason=None if refs else "no_novel_exact_ref")
    return receipts
