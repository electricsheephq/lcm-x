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

_PREFIX = r"(?:^|[.!?;:\n])\s*[\"'“‘]*\s*(?:(?:please|kindly|can you|could you|would you|I want you to|I'd like you to|I need you to|just|and)\s+)?"
_PATTERNS = tuple(re.compile(pattern, re.I) for pattern in (
    _PREFIX + r"forget\s+(?:that|about|what|my|the|this|it|I|everything|any)\b",
    _PREFIX + r"(?:don't|do not|stop|no longer)\s+(?:remember|keep|store|save|retain|track|record)\w*\s+(?:that|my|the|this|it|what|any)\b",
    _PREFIX + r"(?:don't|do not|stop|no longer|never)\s+(?:bring up|mention|reference|factor in|use)\s+(?:that|my|this|it|what I)\b",
    r"\b(?:disregard|erase|delete|remove|wipe|scratch|discard|purge)\b[\s\S]{0,40}?(?:I said|I mentioned|I told you|I shared|that I|what I|my preference|that detail|that info|my history|my data|earlier|previous|from your memory)\b",
    r"\b(?:I'd rather you (?:not|didn't)|I don't want you to)\s+(?:remember|keep|store|know|bring up|mention|use)\b",
))
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
             'scratch', 'discard', 'purge', 'rather you', 'want you to')


def is_forget_request(text: str, role: str = "user") -> bool:
    return role.lower() == "user" and any(p.search(text) for p in _PATTERNS)


def forget_content_words(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.lower())) - _STOPWORDS


def covers(row: dict, forget: dict) -> bool:
    """Inputs carry store scope, session order and the forget row's two predecessors."""
    if row.get("scope") != forget.get("scope"):
        return False
    if row["session_id"] == forget["session_id"]:
        earlier = row["store_id"] < forget["store_id"]
    elif row.get("observed_at") is not None and forget.get("observed_at") is not None:
        earlier = row["observed_at"] < forget["observed_at"]
    else:
        earlier = row.get("session_order", float("inf")) < forget.get("session_order", 0)
    words = forget_content_words(forget.get("content") or "")
    if not earlier or len(words) < 2:
        return False
    shared = len(words & forget_content_words(row.get("content") or ""))
    adjacent = (row["session_id"] == forget["session_id"]
                and row["store_id"] in forget.get("previous_ids", ()))
    return (adjacent and shared / len(words) >= .3) or (shared >= 2 and shared / len(words) >= .5)


def event_time(row: dict) -> str | None:
    value = row.get("observed_at")
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if value else None


def read_forget_rows(db_path, store_ids, *, allowed=None, excluded=(), deadline):
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
        where, params = ["lower(m.role) = 'user'"], []
        for sessions, operator in ((allowed, "IN"), (excluded, "NOT IN")):
            if sessions is not None:
                sessions = list(sessions)
                where.append(f"m.session_id {operator} ({','.join('?' for _ in sessions)})")
                params.extend(sessions)
        where.append("(" + " OR ".join("instr(lower(m.content), ?) > 0" for _ in _KEYWORDS) + ")")
        params.extend(_KEYWORDS)
        ids = list(dict.fromkeys(store_ids))
        rows = conn.execute(
            "WITH requests AS (SELECT m.store_id FROM messages m WHERE "
            + " AND ".join(where)
            + " ORDER BY coalesce(m.observed_at, m.timestamp) DESC, m.store_id DESC LIMIT 200), "
            "selected AS (SELECT store_id FROM requests UNION SELECT store_id FROM messages "
            f"WHERE store_id IN ({','.join('?' for _ in ids)})) "
            "SELECT m.*, (SELECT min(s.store_id) FROM messages s WHERE s.session_id=m.session_id) "
            "AS session_order, m.store_id IN (SELECT store_id FROM requests) AS potential_forget, "
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


def filter_session_payload(payload, rows, forgets, current_session):
    """Session expansion may introduce covered neighbours beyond ranked seeds."""
    entries = [{"hit": h, "ranks": {}, "_final_score": h["score"]} for h in payload["hits"]]
    ranked, safe, receipts = withhold(entries, rows, forgets, len(entries), current_session)
    by_id = {f["store_id"]: f for f in forgets}
    hits = []
    for entry in ranked[:safe]:
        hit = entry["hit"]
        if "content" not in hit and hit.get("store_id") in by_id:
            f = by_id[hit["store_id"]]
            text = f["content"]
            hit.update(content=text, content_chars=len(text), content_returned_chars=len(text),
                       content_truncated=False, evidence_span=dict(char_start=0, char_end=len(text)),
                       score=round(entry["_final_score"], 6), arms=[], delivery_origin="forget_request",
                       exact_ref=f"lcm:{f['store_id']}:0-{len(text)}")
            stamp = event_time(f)
            if stamp:
                hit["event_time"] = stamp
                hit["event_time_source"] = f.get("observed_at_source")
        hits.append(hit)
    if not receipts:
        return []
    payload.update(hits=hits, total_results=len(hits))
    delivery = payload["provenance"]["delivery"]
    delivery.update(message_count=len(hits), session_count=len({h["session_id"] for h in hits}))
    for meta in delivery["sessions"]:
        ids = [h["store_id"] for h in hits if h["session_id"] == meta["session_id"]]
        meta.update(complete=False, omission_reason="forget_requested",
                    returned_id_ranges=[[i, i] for i in ids])
    if "delta" in payload:
        refs = [h["exact_ref"] for h in hits if h.get("exact_ref")]
        payload["delta"].update(novel_refs=refs, novel_ref_count=len(refs), progress=bool(refs))
    return receipts
