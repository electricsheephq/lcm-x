"""Explicit, backup-first cleanup of covered message copies."""

from collections import Counter
from contextlib import closing
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import time

from .config import host_message_uid_mode
from .maintenance import backup_database
from .store import (
    delete_chunk_inflight,
    delete_host_uid_bindings,
    delete_ingest_ledger_rows,
    delete_message_relations,
)
from .vector_store import VectorStore


MIN_DEDUPE_CONTENT_LENGTH = 8000
_UID_OFF = "unproven (host uid mode off)"
_UID_MISSING = "unproven (no later host uid binding)"


def _scan(engine, conn, conversation_id):
    """Group by the full stored values, using the hash for display and audit."""
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name IN ('host_uid_bindings', 'message_relations')"
    )}
    uid_off = host_message_uid_mode() == "off"
    newest_bound = 0
    bound = "0"
    if not uid_off and "host_uid_bindings" in tables:
        bound = "EXISTS (SELECT 1 FROM host_uid_bindings u WHERE u.store_id = d.store_id)"
        newest_bound = conn.execute(
            f"SELECT MAX(d.store_id) FROM messages d WHERE d.conversation_id = ? AND {bound}",
            (conversation_id,),
        ).fetchone()[0] or 0
    related = "0"
    if "message_relations" in tables:
        related = (
            "EXISTS (SELECT 1 FROM message_relations r "
            "WHERE r.store_id = d.store_id OR r.related_store_id = d.store_id)"
        )
    frontier = engine._store_complete_frontier()
    tail = max(0, int(engine._config.fresh_tail_count))
    rows = conn.execute(
        f"""
        WITH grouped AS (
            SELECT store_id, session_id, role, content, timestamp,
                   MIN(store_id) OVER identity AS group_id, COUNT(*) OVER identity AS copies
            FROM messages
            WHERE conversation_id = :conversation_id
            WINDOW identity AS (PARTITION BY role, content, tool_call_id, tool_calls, tool_name)
        ), covered AS (
            SELECT DISTINCT CAST(source.value AS INTEGER) AS store_id
            FROM summary_nodes AS node, json_each(node.source_ids) AS source
            WHERE node.source_type = 'messages'
        ), duplicates AS (
            SELECT grouped.*, covered.store_id IS NOT NULL AS covered,
                   MIN(covered.store_id) OVER (PARTITION BY group_id) AS kept_id
            FROM grouped LEFT JOIN covered USING (store_id)
            WHERE copies > 1
        )
        SELECT group_id, store_id, session_id, role, timestamp, copies, kept_id, COALESCE(length(content), 0),
               CASE WHEN store_id = group_id THEN CAST(COALESCE(content, '') AS BLOB) END,
               CASE
                   WHEN covered THEN 'covered'
                   WHEN COALESCE(length(content), 0) < :minimum THEN 'below minimum length'
                   WHEN kept_id IS NULL THEN 'no covered copy'
                   WHEN store_id <= :frontier THEN 'at or below frontier'
                   WHEN :uid_off THEN :off_reason
                   WHEN {bound} THEN 'host uid bound'
                   WHEN store_id >= :newest_bound THEN :missing_reason
                   WHEN {related} THEN 'message relation'
                   WHEN NOT EXISTS (
                       SELECT 1 FROM messages m WHERE m.session_id = d.session_id AND m.store_id <> d.store_id
                   ) THEN 'only row in session'
                   WHEN :tail > 0 AND NOT EXISTS (
                       SELECT 1 FROM messages m WHERE m.session_id = d.session_id AND m.store_id > d.store_id
                       LIMIT 1 OFFSET :tail_offset
                   ) THEN 'fresh tail'
               END AS reason
        FROM duplicates d
        ORDER BY group_id, store_id
        """,
        dict(conversation_id=conversation_id, minimum=MIN_DEDUPE_CONTENT_LENGTH, frontier=frontier,
             uid_off=uid_off, off_reason=_UID_OFF, newest_bound=newest_bound, missing_reason=_UID_MISSING,
             tail=tail, tail_offset=max(0, tail - 1)),
    )
    lines, removable = [], {}
    copies_total = unproven_total = 0
    for _, group in groupby(rows, key=lambda row: row[0]):
        reasons = Counter()
        group_removable = []
        for _, store_id, session_id, role, timestamp, copies, kept_id, length, content, reason in group:
            if content is not None:
                digest = sha256(content).hexdigest()
            if reason is not None:
                reasons[reason] += 1
            else:
                group_removable.append(str(store_id))
                removable[store_id] = dict(
                    session_id=session_id, conversation_id=conversation_id, role=role,
                    content_sha256=digest, content_length=length, timestamp=timestamp, kept_store_id=kept_id,
                )
        kept = sum(reasons.values())
        unproven = reasons[_UID_OFF] + reasons[_UID_MISSING]
        detail = [f"{reason}={count}" for reason, count in reasons.items()]
        if group_removable:
            detail.append(f"removable_ids={','.join(group_removable)}")
        lines.append(
            f"{digest[:12]} length={length} copies={copies} kept={kept} "
            f"removable={len(group_removable)} unproven={unproven} ({', '.join(detail)})"
        )
        copies_total += copies
        unproven_total += unproven
    lines.append(
        f"totals: groups={len(lines)} copies={copies_total} kept={copies_total - len(removable)} "
        f"removable={len(removable)} unproven={unproven_total}"
    )
    return lines, removable


def _delete_selected(engine, conversation_id, selected):
    store = engine._store
    conn = store.connection
    with store.atomic(immediate=True):
        if engine._conversation_id != conversation_id:
            return 0
        _, rechecked = _scan(engine, conn, conversation_id)
        ids = sorted(selected.keys() & rechecked.keys())
        if not ids:
            return 0
        deleted_at = time.time()
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ((f"doctor_dedupe:{store_id}", json.dumps(dict(rechecked[store_id], deleted_at=deleted_at)))
             for store_id in ids),
        )
        # One bound JSON array supplies the exact integer ids without a variable limit.
        selection = "SELECT value FROM json_each(?)"
        args = (json.dumps(ids),)
        delete_ingest_ledger_rows(conn, selection, args)
        delete_message_relations(conn, selection, args)
        delete_host_uid_bindings(conn, selection, args)
        delete_chunk_inflight(conn, selection, args)
        for offset in range(0, len(ids), 256):
            VectorStore.archive_chunks_for_messages_on_connection(conn, ids[offset:offset + 256])
        deleted = conn.execute(f"DELETE FROM messages WHERE store_id IN ({selection})", args).rowcount
        if deleted != len(ids):
            raise sqlite3.IntegrityError("dedupe delete count did not match selected rows")
    return deleted


def _doctor_dedupe_text(engine, *, apply=False):
    title = "LCM doctor dedupe apply" if apply else "LCM doctor dedupe"
    conversation_id = engine._conversation_id
    if not conversation_id:
        lines = [title, "status: no current conversation"]
        return "\n".join([*lines, "deleted: 0"] if apply else lines)

    lines = [title]
    try:
        uri = Path(engine._store.db_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("BEGIN")
            groups, selected = _scan(engine, conn, conversation_id)
        lines.extend(groups)
        if not apply:
            return "\n".join(lines)
        if not selected:
            return "\n".join([*lines, "backup_path: (none)", "deleted: 0"])
        try:
            backup = backup_database(engine)
        except (OSError, sqlite3.Error) as exc:
            backup = {"ok": False, "error": str(exc)}
        if not backup["ok"]:
            return "\n".join([*lines, "status: error", f"error: backup failed: {backup['error']}", "deleted: 0"])
        lines.append(f"backup_path: {backup['backup_path']}")
        deleted = _delete_selected(engine, conversation_id, selected)
        if deleted != len(selected):
            lines.append(f"recheck_skipped: {len(selected) - deleted}")
        lines.append(f"deleted: {deleted}")
    except sqlite3.Error as exc:
        lines.extend(("status: error", f"error: {exc}"))
        if apply:
            lines.append("deleted: 0")
    return "\n".join(lines)


def _doctor_dedupe_apply_text(engine):
    return _doctor_dedupe_text(engine, apply=True)
