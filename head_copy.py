"""Recognise a stored copy of LCM-X's own summary head (v0.30.3 hotfix).

A stored user row is a known head copy when every leading summary part verifies byte for byte against this
session's summary nodes, it carries the ``generated_unproven`` relation, and any carrier remainder binds to
an earlier user occurrence outside those nodes' own source coverage. The store retains the original row.
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

    end, node_ids = LCMEngine._verify_summary_prefix(content, get_node, owns_node)
    if end is None or end == len(content):
        return end is not None
    suffix = content[end:]
    for separator in ("\n\n---\n\n", "\n\n"):
        if suffix.startswith(separator):
            remainder = suffix[len(separator):]
            return bool(remainder.strip() and _carrier_remainder_is_replay(conn, row, remainder, node_ids))
    return False


def _carrier_remainder_is_replay(conn, row, remainder, node_ids):
    stamp = row.get("observed_at")
    matches = dict(conn.execute(
        """SELECT store_id, observed_at=? FROM messages
           WHERE session_id=? AND store_id<? AND role='user'
             AND COALESCE(conversation_id, '') IN ('', ?) AND content=?""",
        (stamp, row["session_id"], row["store_id"], row.get("conversation_id") or "", remainder),
    ))
    if not matches:
        return False

    uncovered = conn.execute(
        """WITH RECURSIVE sources(kind, id) AS MATERIALIZED (
               SELECT 'nodes', value FROM json_each(?)
               UNION
               SELECT n.source_type, source.value FROM sources s JOIN summary_nodes n ON n.node_id=s.id,
                   json_each(n.source_ids) source WHERE s.kind='nodes'
           ), matches(id) AS MATERIALIZED (
               SELECT value FROM json_each(?)
           ), covered(id) AS MATERIALIZED (
               SELECT id FROM sources WHERE kind='messages' AND id IN (SELECT id FROM matches)
           )
           SELECT m.id FROM matches m
           WHERE EXISTS (SELECT 1 FROM sources WHERE kind='messages')
             AND NOT EXISTS (
                 SELECT 1 FROM sources s LEFT JOIN summary_nodes n ON n.node_id=s.id
                 WHERE s.kind='nodes' AND n.node_id IS NULL
             )
             AND NOT EXISTS (SELECT 1 FROM covered c WHERE c.id=m.id)""",
        (json.dumps(node_ids), json.dumps(list(matches))),
    ).fetchall()
    if stamp is not None and sum(matches[store_id] == 1 for store_id, in uncovered) == 1:
        return True
    return len(uncovered) == 1 and _recorded_carrier_emission(conn, row)


def _recorded_carrier_emission(conn, row):
    """A text-only match needs an emission that predates any binding for this row."""
    from .ingest_ledger import identity_digest

    try:
        salt = conn.execute("SELECT value FROM ledger_meta WHERE key='digest_salt'").fetchone()
        if salt is None:
            return False
        session_id = row["session_id"]
        return conn.execute(
            """SELECT 1 FROM ledger_emissions e
               WHERE e.lineage IN (?, (SELECT value FROM ledger_meta WHERE key=?))
                 AND e.descriptor_digest=? AND e.kind='carrier' AND e.state!='refused'
                 AND NOT EXISTS (SELECT 1 FROM occurrence_bindings b
                                 WHERE b.store_id=? AND b.view_seq<=e.view_seq)
               LIMIT 1""",
            (f"provisional:{session_id}", f"session_lineage:{session_id}", identity_digest(bytes(salt[0]), row),
             row["store_id"]),
        ).fetchone() is not None
    except sqlite3.OperationalError as exc:
        if str(exc) not in {"no such table: ledger_meta", "no such table: ledger_emissions",
                            "no such table: occurrence_bindings"}:
            raise
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
