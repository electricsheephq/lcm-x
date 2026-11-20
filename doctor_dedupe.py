"""Explicit, backup-first cleanup of covered message copies."""

from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import time

from .config import host_message_uid_mode
from .ingest_content import _EXTERNALIZED_PLACEHOLDER_PREFIX
from .ingest_policy import _SENSITIVE_PLACEHOLDER_PREFIX
from .ingest_protection import redact_sensitive_value
from .maintenance import backup_database
from .message_content import normalize_content_value
from .reconcile import _strip_volatile_host_scaffolding
from .store import (
    delete_chunk_inflight,
    delete_host_uid_bindings,
    delete_ingest_ledger_message_refs,
    delete_message_relations,
)
from .vector_store import VectorStore


MIN_DEDUPE_CONTENT_LENGTH = 8000
MIN_DEDUPE_AGE_SECONDS = 3600
_HOST_PREFIX_LENGTH = 2000
_MIN_CONTAINED_HOST_LENGTH = 256
_UID_OFF = "unproven (host uid mode off)"
_UID_MISSING = "unproven (no later host uid binding)"
_HOST_UNAVAILABLE = "unproven (host list unavailable)"
_NON_TEXT = "unproven (non-text content)"
_PROTECTED_CONTENT = "unproven (protected content)"
_HOST_LISTED = "host listed"
_CITATIONS = (
    ("lcm_entity_aliases", "source_store_ids"),
    ("lcm_entity_keys", "source_store_ids"),
    ("lcm_memory_notes", "cites"),
    ("lcm_memory_notes", "retracts_store_ids"),
)


def _normalize_whitespace(content):
    return " ".join(content.split())


def _host_evidence(engine, session_id):
    """Collect mapped ids and same-role text forms from both host views."""
    if not session_id:
        return None
    mapped_ids = set()
    forms_by_role = defaultdict(set)
    try:
        read = engine._host_session_db.get_messages_as_conversation
        for options in ({}, {"repair_alternation": True}):
            messages = read(session_id, **options)
            if not isinstance(messages, list):
                return None
            saved = getattr(engine, "_current_compress_placeholder_identity_counts", None)
            try:
                mapping = engine._get_store_id_map_for_messages(messages)
            finally:
                engine._current_compress_placeholder_identity_counts = saved
            mapped_ids.update(mapping.values())
            for message in messages:
                role, content = message["role"], message.get("content")
                if not isinstance(role, str):
                    return None
                if content is None:
                    continue
                redacted = redact_sensitive_value(content, engine._config, parse_json_strings=False)
                for value in (content, redacted):
                    text = normalize_content_value(value)
                    for form in (text, _strip_volatile_host_scaffolding(text)):
                        if normalized := _normalize_whitespace(form):
                            forms_by_role[role].add(normalized)
    except Exception:  # Failed reads or mapping cannot prove absence.
        return None
    return mapped_ids, forms_by_role


def _content_veto(content, host_forms):
    if not isinstance(content, str):
        return _NON_TEXT
    for text in (content, _strip_volatile_host_scaffolding(content)):
        normalized = _normalize_whitespace(text)
        prefix = normalized[:_HOST_PREFIX_LENGTH]
        if any(
            prefix == form[:_HOST_PREFIX_LENGTH]
            or normalized in form
            or (len(form) >= _MIN_CONTAINED_HOST_LENGTH and form in normalized)
            for form in host_forms
        ):
            return _HOST_LISTED
    if _SENSITIVE_PLACEHOLDER_PREFIX in content or _EXTERNALIZED_PLACEHOLDER_PREFIX in content:
        return _PROTECTED_CONTENT
    return None


def _scan(engine, conn, conversation_id, session_id, *, users, selected=None):
    """Group by the full stored values, using the hash for display and audit."""
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name IN ('host_uid_bindings', 'message_relations', "
        "'lcm_entity_aliases', 'lcm_entity_keys', 'lcm_memory_notes')"
    )}
    citations = " UNION ".join(
        f"SELECT CAST(c.value AS INTEGER) FROM {table} t, json_each(t.{column}) c"
        for table, column in _CITATIONS if table in tables
    )
    cited = f"d.store_id IN ({citations})" if citations else "0"
    # A selected row only needs its original covered witness. A lost witness skips the row.
    messages_from = "messages"
    scope_ids = None
    if selected is not None:
        scope_ids = json.dumps(sorted(set(selected) | {row["kept_store_id"] for row in selected.values()}))
        messages_from = "json_each(:scope_ids) AS scope CROSS JOIN messages ON messages.store_id = scope.value"
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
            SELECT store_id, session_id, role, content, timestamp, pinned,
                   MIN(store_id) OVER identity AS group_id, COUNT(*) OVER identity AS copies
            FROM {messages_from}
            WHERE conversation_id = :conversation_id AND length(content) >= :minimum
            WINDOW identity AS (PARTITION BY role, content, tool_call_id, tool_calls, tool_name)
        ), covered AS (
            SELECT DISTINCT CAST(source.value AS INTEGER) AS store_id
            FROM summary_nodes AS node, json_each(node.source_ids) AS source
            WHERE node.source_type = 'messages'
              AND CAST(source.value AS INTEGER) IN (SELECT store_id FROM grouped WHERE copies > 1)
        ), duplicates AS (
            SELECT grouped.*, covered.store_id IS NOT NULL AS covered,
                   MIN(covered.store_id) OVER (PARTITION BY group_id) AS kept_id
            FROM grouped LEFT JOIN covered USING (store_id)
            WHERE copies > 1
        )
        SELECT group_id, store_id, session_id, role, timestamp, copies, kept_id, length(content),
               CASE WHEN store_id = group_id THEN content END,
               CASE
                   WHEN covered THEN 'covered'
                   WHEN kept_id IS NULL THEN 'no covered copy'
                   WHEN kept_id >= store_id THEN 'no earlier covered copy'
                   WHEN store_id <= :frontier THEN 'at or below frontier'
                   WHEN NOT (typeof(timestamp) IN ('integer', 'real') AND timestamp <= :cutoff)
                        THEN 'below minimum age'
                   WHEN pinned THEN 'pinned'
                   WHEN :uid_off THEN :off_reason
                   WHEN {bound} THEN 'host uid bound'
                   WHEN store_id >= :newest_bound THEN :missing_reason
                   WHEN {related} THEN 'message relation'
                   WHEN {cited} THEN 'entity or note citation'
                   WHEN NOT EXISTS (
                       SELECT 1 FROM messages m WHERE m.session_id = d.session_id AND m.store_id <> d.store_id
                   ) THEN 'only row in session'
                   WHEN :tail > 0 AND NOT EXISTS (
                       SELECT 1 FROM messages m WHERE m.session_id = d.session_id AND m.store_id > d.store_id
                       LIMIT 1 OFFSET :tail_offset
                   ) THEN 'fresh tail'
                   WHEN d.session_id IS NOT :session_id THEN 'other session (not handled)'
               END AS reason
        FROM duplicates d
        ORDER BY group_id, store_id
        """,
        dict(conversation_id=conversation_id, session_id=session_id, minimum=MIN_DEDUPE_CONTENT_LENGTH, frontier=frontier,
             uid_off=uid_off, off_reason=_UID_OFF, newest_bound=newest_bound, missing_reason=_UID_MISSING,
             tail=tail, tail_offset=max(0, tail - 1), scope_ids=scope_ids,
             cutoff=time.time() - MIN_DEDUPE_AGE_SECONDS),
    ).fetchall()
    host = _host_evidence(engine, session_id) if any(row[-1] is None for row in rows) else (set(), {})
    mapped_ids, host_forms = host if host is not None else (set(), {})
    lines, removable = [], {}
    copies_total = unproven_total = 0
    for group_id, group in groupby(rows, key=lambda row: row[0]):
        reasons = Counter()
        group_removable = []
        for _, store_id, row_session, role, timestamp, copies, kept_id, length, content, reason in group:
            if store_id == group_id:
                digest = sha256(content.encode()).hexdigest() if isinstance(content, str) else "n/a"
                content_reason = _content_veto(content, host_forms.get(role, ()))
            if reason is None:
                if host is None:
                    reason = _HOST_UNAVAILABLE
                elif store_id in mapped_ids:
                    reason = _HOST_LISTED
                elif content_reason is not None:
                    reason = content_reason
                elif role == "user" and not users:
                    reason = "needs `apply users`"
            if reason is not None:
                reasons[reason] += 1
            else:
                group_removable.append(str(store_id))
                removable[store_id] = dict(
                    session_id=row_session, conversation_id=conversation_id, role=role,
                    content_sha256=digest, content_length=length, timestamp=timestamp, kept_store_id=kept_id,
                )
        kept = sum(reasons.values())
        unproven = sum(reasons[reason] for reason in (_UID_OFF, _UID_MISSING, _HOST_UNAVAILABLE,
                                                      _NON_TEXT, _PROTECTED_CONTENT))
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
    return lines, removable, host is not None


def _delete_selected(engine, conversation_id, session_id, selected, *, users):
    store = engine._store
    conn = store.connection
    with store.atomic(immediate=True):
        if (engine._conversation_id, engine._session_id) != (conversation_id, session_id):
            return 0, None
        _, rechecked, host_available = _scan(engine, conn, conversation_id, session_id, users=users, selected=selected)
        if not host_available:
            return 0, _HOST_UNAVAILABLE
        if (engine._conversation_id, engine._session_id) != (conversation_id, session_id):
            return 0, None
        ids = sorted(selected.keys() & rechecked.keys())
        if not ids:
            return 0, None
        # One bound JSON array supplies the exact integer ids without a variable limit.
        selection = "SELECT value FROM json_each(?)"
        args = (json.dumps(ids),)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        audit_fields = sorted(columns & {"sender_id", "sender_source", "observed_at", "source"})
        if audit_fields:
            for store_id, *values in conn.execute(
                f"SELECT store_id, {', '.join(audit_fields)} FROM messages WHERE store_id IN ({selection})", args,
            ):
                rechecked[store_id].update(zip(audit_fields, values))
        deleted_at = time.time()
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ((f"doctor_dedupe:{store_id}", json.dumps(dict(rechecked[store_id], deleted_at=deleted_at)))
             for store_id in ids),
        )
        delete_ingest_ledger_message_refs(conn, selection, args)
        delete_message_relations(conn, selection, args)
        delete_host_uid_bindings(conn, selection, args)
        delete_chunk_inflight(conn, selection, args)
        for offset in range(0, len(ids), 256):
            VectorStore.archive_chunks_for_messages_on_connection(conn, ids[offset:offset + 256])
        deleted = conn.execute(f"DELETE FROM messages WHERE store_id IN ({selection})", args).rowcount
        if deleted != len(ids):
            raise sqlite3.IntegrityError("dedupe delete count did not match selected rows")
    return deleted, None


def _doctor_dedupe_text(engine, *, apply=False, users=False):
    title = "LCM doctor dedupe apply" if apply else "LCM doctor dedupe"
    if users:
        title += " users"
    if apply and not getattr(getattr(engine, "_config", None), "doctor_dedupe_apply_enabled", False):
        return "\n".join((title, "status: disabled", "deleted: 0",
                          "note: set LCM_DOCTOR_DEDUPE_APPLY_ENABLED=true in trusted operator environments"))
    conversation_id, session_id = engine._conversation_id, engine._session_id
    if not conversation_id:
        lines = [title, "status: no current conversation"]
        return "\n".join([*lines, "deleted: 0"] if apply else lines)

    lines = [title]
    try:
        uri = Path(engine._store.db_path).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("BEGIN")
            groups, selected, _ = _scan(engine, conn, conversation_id, session_id, users=users)
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
        deleted, recheck_status = _delete_selected(engine, conversation_id, session_id, selected, users=users)
        if recheck_status:
            lines.append(f"recheck: {recheck_status}")
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


def _doctor_dedupe_apply_users_text(engine):
    return _doctor_dedupe_text(engine, apply=True, users=True)
