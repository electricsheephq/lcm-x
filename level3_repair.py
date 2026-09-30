"""Doctor scan for level 3 fragment nodes and the condensed ancestors built on them (#667).

Before the #628 and #652 fixes a refused summary route wrote level 3 truncations into the summary DAG, and
condensations were then built from those fragments.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from .escalation import _L3_TRUNCATION_MARKER
from .tokens import count_tokens

_PROVENANCE_TABLE = "summary_node_provenance"  # #649 sidecar (escalation_level per node), when a store has it


def scan_level3_fragments(engine) -> dict[str, Any]:
    """Return the fragment nodes, their ancestors and per-session counts. Runs SELECTs only.

    A fragment carries the ``_deterministic_truncate`` marker and fits its token bound. Where the #649 sidecar
    records a level for the node, that level decides instead of the bound. A verbatim level 3 (the whole
    source, which already fit) has no marker and is never flagged.
    """
    bound = int(getattr(getattr(engine, "_config", None), "l3_truncate_tokens", 512) or 512)
    conn = engine._store.connection
    has_provenance = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (_PROVENANCE_TABLE,)
    ).fetchone() is not None
    level_sql = (
        f"(SELECT p.escalation_level FROM {_PROVENANCE_TABLE} p WHERE p.node_id = n.node_id)"
        if has_provenance else "NULL"
    )
    rows = conn.execute(
        "SELECT n.node_id, n.session_id, n.depth, n.summary, n.token_count, n.source_ids, n.source_type, "
        f"{level_sql} FROM summary_nodes n WHERE instr(n.summary, ?) > 0 ORDER BY n.node_id",
        (_L3_TRUNCATION_MARKER,),
    ).fetchall()
    flagged: list[dict[str, Any]] = []
    for node_id, session_id, depth, summary, token_count, source_ids, source_type, level in rows:
        if level is not None:
            if int(level) != 3:
                continue  # a recorded model summary that quotes the marker
        else:
            tokens = count_tokens(summary)
            if token_count:
                tokens = min(tokens, int(token_count))  # the count stored at write time
            if tokens > bound:
                continue
        ids = sorted({int(value) for value in json.loads(source_ids or "[]")})
        table, key = ("messages", "store_id") if source_type == "messages" else ("summary_nodes", "node_id")
        stored = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {key} IN (SELECT value FROM json_each(?))", (json.dumps(ids),)
        ).fetchone()[0]
        flagged.append({
            "node_id": int(node_id), "session_id": session_id, "depth": int(depth),
            "leaf": source_type == "messages", "sources": len(ids), "sources_stored": int(stored),
        })
    flagged_ids = {item["node_id"] for item in flagged}
    ancestor_rows = conn.execute(
        """WITH RECURSIVE up(node_id) AS (
               SELECT CAST(value AS INTEGER) FROM json_each(?)
               UNION
               SELECT p.node_id FROM summary_nodes p, json_each(p.source_ids) j, up
               WHERE p.source_type = 'nodes' AND CAST(j.value AS INTEGER) = up.node_id
           ) SELECT n.node_id, n.session_id, n.depth FROM summary_nodes n JOIN up ON n.node_id = up.node_id
           ORDER BY n.depth, n.node_id""",
        (json.dumps(sorted(flagged_ids)),),
    ).fetchall() if flagged_ids else []
    ancestors = [
        {"node_id": int(row[0]), "session_id": row[1], "depth": int(row[2])}
        for row in ancestor_rows if int(row[0]) not in flagged_ids
    ]
    sessions: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"flagged_leaves": 0, "flagged_nodes": 0, "ancestors_by_depth": Counter()}
    )
    for item in flagged:
        sessions[item["session_id"]]["flagged_leaves" if item["leaf"] else "flagged_nodes"] += 1
    for item in ancestors:
        sessions[item["session_id"]]["ancestors_by_depth"][item["depth"]] += 1
    return {
        "bound_tokens": bound, "provenance_table": has_provenance, "flagged": flagged, "ancestors": ancestors,
        "sessions": {sid: {**counts, "ancestors_by_depth": dict(sorted(counts["ancestors_by_depth"].items()))}
                     for sid, counts in sorted(sessions.items())},
    }
