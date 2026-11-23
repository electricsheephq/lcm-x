"""Recognise a stored copy of LCM-X's own summary head (v0.30.3 hotfix).

A stored user row is a known head copy when every leading summary part verifies byte for byte against this
session's summary nodes, it carries the ``generated_unproven`` relation, and any carrier remainder is already stored
as an earlier user row. Such a row stays in the store unchanged; assembly and compaction input leave it out.
Read-only: nothing here writes.
"""

import json
import sqlite3
from types import SimpleNamespace


def _read_head_copy_candidates(conn, sql, args):
    try:
        return conn.execute(sql, args)
    except sqlite3.OperationalError as exc:
        if str(exc) != "no such table: message_relations":
            raise
        return None  # Older stores may never have created the optional relation table.


def is_known_head_copy(conn, row) -> bool:
    """Prove this stored user occurrence is fully represented by earlier LCM context."""
    content = row.get("content")
    if (conn is None or row.get("role") != "user" or not isinstance(content, str) or not content.startswith("[")
            or "[Expand for details:" not in content or not row.get("session_id") or not row.get("store_id")
            or any(row.get(key) for key in ("tool_calls", "tool_call_id", "tool_name"))):
        return False
    store_id, session_id = row["store_id"], row["session_id"]
    relation = _read_head_copy_candidates(
        conn, "SELECT 1 FROM message_relations WHERE store_id=? AND kind='generated_unproven'", (store_id,)
    )
    if relation is None or relation.fetchone() is None:
        return False

    from .engine import LCMEngine

    def get_node(node_id):
        node = conn.execute(
            "SELECT session_id, depth, summary, expand_hint FROM summary_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        return SimpleNamespace(node_id=node_id, session_id=node[0], depth=node[1],
                               summary=node[2], expand_hint=node[3] or "") if node else None

    def owns_node(owner):
        return owner == session_id or conn.execute(
            "WITH RECURSIVE lineage(owner, hops) AS (VALUES (?, 0) UNION ALL "
            "SELECT CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$') END, hops+1 "
            "FROM lineage JOIN metadata m ON m.key='rotation_predecessor_session:' || owner WHERE hops<32) "
            "SELECT 1 FROM lineage WHERE owner=? LIMIT 1", (owner, session_id),
        ).fetchone() is not None

    end, _ = LCMEngine._verify_summary_prefix(content, get_node, owns_node)
    if end is None or end == len(content):
        return end is not None
    suffix = content[end:]
    for separator in ("\n\n---\n\n", "\n\n"):
        if suffix.startswith(separator):
            remainder = suffix[len(separator):]
            return bool(remainder.strip() and conn.execute(
                "SELECT 1 FROM messages WHERE session_id=? AND store_id<? AND role='user' "
                "AND COALESCE(conversation_id, '') IN ('', ?) AND content=? LIMIT 1",
                (session_id, store_id, row.get("conversation_id") or "", remainder),
            ).fetchone())
    return False


def known_head_copy_ids(conn, session_id=None, *, store_ids=None) -> set[int]:
    if conn is None or store_ids == []:
        return set()
    session_filter = "" if session_id is None else "AND m.session_id=?"
    id_filter = "" if store_ids is None else "AND r.store_id IN (SELECT value FROM json_each(?))"
    args = ([] if session_id is None else [session_id]) + ([] if store_ids is None else [json.dumps(store_ids)])
    candidates = _read_head_copy_candidates(
        conn,
        "SELECT DISTINCT m.* FROM message_relations r JOIN messages m USING (store_id) "
        f"WHERE r.kind='generated_unproven' AND m.role='user' AND m.content GLOB '[[]*' {session_filter} {id_filter}", args
    )
    if candidates is None:
        return set()
    columns = [column[0] for column in candidates.description]
    return {row["store_id"] for raw in candidates if is_known_head_copy(conn, row := dict(zip(columns, raw)))}
