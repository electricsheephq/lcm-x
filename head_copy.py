"""Recognise stored copies of LCM-X's own summary head.

A marked user row is a known copy when its leading summary parts verify byte for byte
against owned nodes. A carrier remainder must bind to an earlier user occurrence
in the same normalised conversation, beyond the verified summaries' source coverage.
The proof is read-only and leaves the stored row and its relations intact.
"""

import json
import sqlite3
from functools import cache
from types import SimpleNamespace

from .summary_prefix import _LCM_SUMMARY_PART_HEADER_RE, complete_summary_coverage, verify_summary_prefix


def _head_copy_shape_sql(prefix=""):
    return (
        f"{prefix}role='user' AND {prefix}content GLOB '[[]*' "
        f"AND instr({prefix}content, '[Expand for details:') > 0"
    )


def _read_head_copy_candidates(conn, sql, args):
    try:
        return conn.execute(sql, args)
    except sqlite3.OperationalError as exc:
        if str(exc) != "no such table: message_relations":
            raise
        return None  # Older stores may never have created the optional relation table.


def has_head_copy_shape(content) -> bool:
    return isinstance(content, str) and content.startswith("[") and "[Expand for details:" in content


def _could_be_head_copy(role, content) -> bool:
    return role == "user" and has_head_copy_shape(content) and _LCM_SUMMARY_PART_HEADER_RE.match(content) is not None


def _head_copy_candidate(row) -> bool:
    return (
        _could_be_head_copy(row.get("role"), row.get("content"))
        and bool(row.get("session_id") and row.get("store_id"))
        and not any(row.get(key) for key in ("tool_calls", "tool_call_id", "tool_name"))
    )


def _verify_marked_head_copy(conn, row) -> bool:
    content = row["content"]
    session_id = row["session_id"]

    def get_node(node_id):
        node = conn.execute(
            "SELECT session_id, depth, summary, expand_hint FROM summary_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        return SimpleNamespace(node_id=node_id, session_id=node[0], depth=node[1],
                               summary=node[2], expand_hint=node[3] or "") if node else None

    @cache
    def predecessors():
        lineage = conn.execute(
            "WITH RECURSIVE lineage(session, hops) AS (VALUES (?, 0) UNION ALL "
            "SELECT CASE WHEN json_valid(m.value) THEN "
            "CASE WHEN json_type(m.value)='text' THEN json_extract(m.value, '$') END END, hops+1 "
            "FROM lineage JOIN metadata m ON m.key='rotation_predecessor_session:' || session WHERE hops<32) "
            "SELECT session FROM lineage WHERE hops>0 ORDER BY hops", (session_id,),
        )
        found = []
        for (predecessor,) in lineage:
            if not predecessor or predecessor == session_id or predecessor in found:
                break
            found.append(predecessor)
        return found

    def owns_node(owner):
        return owner == session_id or owner in predecessors()

    end, node_ids = verify_summary_prefix(content, get_node, owns_node)
    if end is None or end == len(content):
        return end is not None
    suffix = content[end:]
    for separator in ("\n\n---\n\n", "\n\n"):
        if suffix.startswith(separator):
            remainder = suffix[len(separator):]
            return bool(remainder.strip() and _carrier_remainder_is_replay(
                conn, row, remainder, node_ids, predecessors,
            ))
    return False


def _carrier_remainder_is_replay(conn, row, remainder, node_ids, predecessors):
    if not {"observed_at", "conversation_id", "tool_call_id", "tool_calls", "tool_name"} <= row.keys():
        return False
    stamp = row["observed_at"]
    conversation = (row["conversation_id"] or "").strip()
    sessions = [row["session_id"], *predecessors()]
    matches = [(store_id, session_id, same_stamp) for store_id, session_id, same_stamp, prior_conv in conn.execute(
        "SELECT store_id, session_id, observed_at=?, conversation_id FROM messages "
        f"WHERE session_id IN ({','.join('?' * len(sessions))}) AND store_id<? AND role='user' AND content=?",
        (stamp, *sessions, row["store_id"], remainder),
    ) if (prior_conv or "").strip() in ("", conversation)]
    if not matches:
        return False
    covered_through, covered_matches = complete_summary_coverage(
        conn, node_ids, [store_id for store_id, _, _ in matches],
    )
    if covered_through is None:
        return False
    stamped = [store_id for store_id, _, same_stamp in matches if same_stamp]
    if stamped:
        return len(stamped) == 1 and stamped[0] > covered_through
    uncovered = [(store_id, session_id) for store_id, session_id, _ in matches if store_id not in covered_matches]
    return (len(uncovered) == 1 and uncovered[0][0] > covered_through and uncovered[0][1] == row["session_id"]
            and _recorded_carrier_emission(conn, row))


def _recorded_carrier_emission(conn, row):
    """Require a pre-binding emission and evidence that ingest inserted the carrier before an older row."""
    from .ingest_ledger import identity_digest

    try:
        salt = conn.execute("SELECT value FROM ledger_meta WHERE key='digest_salt'").fetchone()
        if salt is None:
            return False
        session_id = row["session_id"]
        return conn.execute(
            """SELECT 1 FROM ledger_emissions e
               JOIN occurrence_bindings b ON b.store_id=? AND b.lineage=e.lineage
                 AND EXISTS (SELECT 1 FROM occurrence_bindings successor JOIN messages m USING (store_id)
                             WHERE successor.lineage=b.lineage AND successor.view_seq<b.view_seq
                               AND successor.ordinal>b.ordinal AND successor.store_id<b.store_id
                               AND m.session_id=?)
               WHERE e.lineage IN (?, (SELECT value FROM ledger_meta WHERE key=?))
                 AND e.descriptor_digest=? AND e.kind='carrier' AND e.state!='refused'
                 AND NOT EXISTS (SELECT 1 FROM occurrence_bindings prior
                                 WHERE prior.store_id=? AND prior.view_seq<=e.view_seq)
               LIMIT 1""",
            (row["store_id"], session_id, f"provisional:{session_id}", f"session_lineage:{session_id}",
             identity_digest(bytes(salt[0]), row),
             row["store_id"]),
        ).fetchone() is not None
    except sqlite3.OperationalError as exc:
        if str(exc) not in {"no such table: ledger_meta", "no such table: ledger_emissions",
                            "no such table: occurrence_bindings"}:
            raise
        return False


def is_known_head_copy(conn, row) -> bool:
    """Prove this stored user occurrence is fully represented by earlier LCM context."""
    if conn is None or not _head_copy_candidate(row):
        return False
    relation = _read_head_copy_candidates(
        conn, "SELECT 1 FROM message_relations WHERE store_id=? AND kind='generated_unproven'", (row["store_id"],)
    )
    return relation is not None and relation.fetchone() is not None and _verify_marked_head_copy(conn, row)


def known_head_copies_in_rows(conn, rows) -> set[int]:
    """Check a hydrated batch with one relation read, and only for shaped rows."""
    if conn is None:
        return set()
    candidates = {row["store_id"]: row for row in rows if _head_copy_candidate(row)}
    if not candidates:
        return set()
    marked = _read_head_copy_candidates(
        conn, "SELECT DISTINCT store_id FROM message_relations WHERE kind='generated_unproven' "
        "AND store_id IN (SELECT value FROM json_each(?))", (json.dumps(list(candidates)),)
    )
    return ({store_id for (store_id,) in marked if _verify_marked_head_copy(conn, candidates[store_id])}
            if marked is not None else set())


def known_head_copy_ids(conn, session_id=None, *, store_ids=None) -> set[int]:
    if conn is None or store_ids == []:
        return set()
    session_filter = "" if session_id is None else "AND m.session_id=?"
    id_filter = "" if store_ids is None else "AND r.store_id IN (SELECT value FROM json_each(?))"
    args = ([] if session_id is None else [session_id]) + ([] if store_ids is None else [json.dumps(store_ids)])
    candidates = _read_head_copy_candidates(
        conn,
        "SELECT DISTINCT m.store_id, m.session_id, m.conversation_id, m.role, m.content, "
        "m.tool_call_id, m.tool_calls, m.tool_name, m.observed_at "
        "FROM message_relations r JOIN messages m USING (store_id) "
        f"WHERE r.kind='generated_unproven' AND {_head_copy_shape_sql('m.')} {session_filter} {id_filter}", args
    )
    if candidates is None:
        return set()
    columns = [column[0] for column in candidates.description]
    return {row["store_id"] for raw in candidates if _head_copy_candidate(row := dict(zip(columns, raw)))
            and _verify_marked_head_copy(conn, row)}


def exclude_head_copies(where, args, copies, column="store_id"):
    if copies:
        where.append(f"{column} NOT IN (SELECT value FROM json_each(?))")
        args.append(json.dumps(sorted(copies)))
