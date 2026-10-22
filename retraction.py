"""Default-off, read-only retraction lookup and deterministic delivery arms."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time

from .store import _normalize_observed_at

# Caller-composed FTS literals only; unicode61 does not stem these variants.
RETRACTION_PHRASES = (
    "forget", "forgetting", "forgotten", "disregard", "disregarding", "ignoring", "ignore what", "ignore my", "stop using",
    "stop remembering", "remove from memory", "erase from memory",
)
STOPWORDS = frozenset("""a an the i me my mine you your we our it its is are was were
be been being to of for from in on at as and or but that this these those what
said told mentioned earlier previous previously just about please do did with
statement information detail details remember memory anymore""".split())
OVERLAP_TAU = 0.75
_REQUEST = re.compile(
    r"^(?:(?:please|kindly)\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|"
    r"would\s+you\s+mind\s+|i\s+(?:want|need)\s+you\s+to\s+)?"
    r"(?:forget(?:ting)?|disregard(?:ing)?|ignor(?:e|ing)|stop\s+(?:using|remembering)|"
    r"(?:remove|erase)\s+from\s+memory)\s+(.+)$", re.I)


def request_object(text):
    """Return a request's object (empty for deictic), or None for non-requests."""
    text = " ".join(text.split()).strip()
    forgotten = re.fullmatch(r"i\s+(?:want|need)\s+(.+?)\s+(?:to\s+be\s+)?forgotten[.!?]*", text, re.I)
    if forgotten:
        text = "forget " + forgotten[1]
    match = _REQUEST.match(text)
    if not match or re.search(r"\b(?:if|hypothetically|told|says|said to|tell me|talk about)\b", text, re.I):
        return None
    obj = match[1].strip(" .!?;")
    if re.match(r"(?:to\b|about\s+it\b)", obj, re.I):
        return None
    obj = re.sub(r"^(?:(?:what|everything|anything)\s+i\s+(?:just\s+)?(?:said|told you|mentioned)|"
                 r"my\s+(?:(?:earlier|previous)\s+)?(?:statement|information|details?))\s*(?:about\s+)?", "", obj, flags=re.I)
    if obj.lower() in {"that", "this", "it", "all that", "what i just said"}:
        obj = ""
    return obj.lower()


def _words(text):
    words = re.findall(r"[^\W_]+", text.lower())
    return {re.sub(r"(?:ing|ed|es|s)$", "", w) if len(w) > 4 else w
            for w in words if w not in STOPWORDS}


def _ids(hit):
    return {int(s) for s in [hit.get("store_id"), *hit.get("duplicate_store_ids", [])] if s is not None}


def find_retractions(engine, entries, *, excluded_session_ids, deadline):
    """Return (R row, delivered indices) groups using at most PRAGMA + SELECT."""
    candidates = [(i, e["hit"]) for i, e in enumerate(entries)
                  if e["hit"].get("kind") == "message_excerpt" and e["hit"].get("store_id") is not None
                  and e["hit"].get("role") in {"user", "assistant"}]
    if not candidates:
        return [], None
    if time.monotonic() >= deadline:
        return [], "deadline"
    ids = sorted({sid for _, h in candidates for sid in _ids(h)})
    lower = min(max(_ids(h)) for _, h in candidates)
    conn = None
    expired = [False]

    def interrupt():
        expired[0] = time.monotonic() >= deadline
        return int(expired[0])

    try:
        conn = sqlite3.connect(f"{Path(engine._store.db_path).resolve().as_uri()}?mode=ro",
                               uri=True, timeout=max(.001, deadline - time.monotonic()))
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(interrupt, 1000)
        conn.row_factory = sqlite3.Row
        columns = "m.store_id, m.session_id, m.role, m.content, m.observed_at, m.timestamp"
        rows = conn.execute(
            f"SELECT * FROM (SELECT {columns}, (SELECT MAX(p.store_id) FROM messages p "
            "WHERE p.session_id=m.session_id AND p.role='user' AND p.store_id<m.store_id) AS prev_user_store_id "
            "FROM messages_fts JOIN messages m ON m.store_id=messages_fts.rowid "
            "WHERE messages_fts MATCH ? AND m.role='user' AND m.store_id>? "
            "AND m.session_id NOT IN (SELECT value FROM json_each(?)) ORDER BY m.store_id DESC LIMIT 200) "
            f"UNION ALL SELECT * FROM (SELECT {columns}, NULL AS prev_user_store_id FROM messages m "
            "WHERE m.store_id IN (SELECT value FROM json_each(?)))",
            (' OR '.join('"' + p.replace('"', '""') + '"' for p in RETRACTION_PHRASES),
             lower, json.dumps(list(excluded_session_ids)), json.dumps(ids)),
        ).fetchall()
    except (sqlite3.Error, OSError):
        return [], "deadline" if expired[0] or time.monotonic() >= deadline else "read_error"
    finally:
        if conn is not None:
            conn.close()
    if time.monotonic() >= deadline:
        return [], "deadline"
    originals = {r["store_id"]: dict(r) for r in rows if r["store_id"] in ids}
    requests = {}
    for row in rows:
        if row["role"] == "user" and row["store_id"] > lower:
            requests.setdefault(row["store_id"], dict(row))
    groups, assigned = [], set()
    for request in sorted(requests.values(), key=lambda r: r["store_id"], reverse=True):
        obj = request_object(request["content"])
        if obj is None:
            continue
        linked = []
        for index, hit in candidates:
            copies = [originals[sid] for sid in _ids(hit) if sid in originals]
            row = originals.get(hit["store_id"])
            if (index in assigned or row is None or request["store_id"] <= max(_ids(hit))
                    or (row["role"] == "user" and request_object(row["content"]) is not None)):
                continue
            observed = [_normalize_observed_at(r["observed_at"]) for r in copies]
            observed = [t for t in observed if t is not None]
            rt = _normalize_observed_at(request["observed_at"])
            if rt is not None and observed and rt < max(observed):
                continue
            if obj:
                words = _words(obj)
                matches = bool(words) and len(words & _words(row["content"])) / len(words) >= OVERLAP_TAU
            else:
                prev = request["prev_user_store_id"]
                matches = prev in _ids(hit) or (prev is not None and row["role"] == "assistant" and row["session_id"] == request["session_id"]
                                               and prev < row["store_id"] < request["store_id"])
            if matches:
                linked.append(index)
        linked.sort(key=lambda i: max(_ids(entries[i]["hit"])), reverse=True)
        if linked:
            assigned.update(linked)
            linked = linked[:3]
            groups.append((request, linked))
    return groups, None


def apply_retractions(hits, links, *, mode, response_chars, response_char_cap):
    """Pure bounded transformation; preserve order and never add a hit."""
    delivered = {sid for h in hits for sid in _ids(h)}
    replacements, removed, added, truncated = {}, set(), 0, False
    for request, indices in links:
        already = request["store_id"] in delivered
        if mode == "codeliver" and already:
            continue
        index = max(indices, key=lambda i: max(_ids(hits[i])))
        old = hits[index]
        text = " ".join(request["content"].split())
        text = text[:600] + ("…" if len(text) > 600 else "")
        observed = _normalize_observed_at(request.get("observed_at"))
        event = datetime.fromtimestamp(observed, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if observed is not None else None
        if mode == "codeliver":
            when = f"Later, on {event}," if event else "Later,"
            addition = {"annotations": [f'{when} the user said: "{text}"']}
            replacement = {**old, **addition}
        else:
            note = "[Withdrawn: the user later asked to forget an earlier statement here."
            note += "]" if already else f' The user said: "{text}"]'
            addition = dict(kind="retraction", snippet=note, score=old["score"],
                            from_current_session=old.get("from_current_session", False),
                            session_id=request["session_id"], timestamp=request["timestamp"],
                            withdrawn_store_id=old["store_id"], cites=[request["store_id"]])
            if event:
                addition["event_time"] = event
            replacement = addition
        size = len(json.dumps(addition, ensure_ascii=False))
        if added + size > 2000 or response_chars + added + size > response_char_cap:
            truncated = True
            continue
        added += size
        replacements[index] = replacement
        if mode == "suppress":
            removed.update(i for i in indices if i != index)
    return [replacements.get(i, h) for i, h in enumerate(hits) if i not in removed], truncated
