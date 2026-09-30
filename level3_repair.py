"""Doctor scan for level 3 fragment nodes and the condensed ancestors built on them (#667).

Before the #628 and #652 fixes a refused summary route wrote level 3 truncations into the summary DAG, and
condensations were then built from those fragments.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from typing import Any

from .escalation import _L3_TRUNCATION_MARKER, summarize_with_escalation, summary_route_available
from .maintenance import backup_database
from .tokens import count_messages_tokens, count_tokens
from .vector_store import VectorStore

_PROVENANCE_TABLE = "summary_node_provenance"  # #649 sidecar (escalation_level per node), when a store has it


def _stored_sources(conn, source_ids: str, source_type: str) -> tuple[list[int], int]:
    """A node's distinct source ids and how many of them are still stored (message rows or child nodes)."""
    ids = sorted({int(value) for value in json.loads(source_ids or "[]")})
    table, key = ("messages", "store_id") if source_type == "messages" else ("summary_nodes", "node_id")
    stored = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {key} IN (SELECT value FROM json_each(?))", (json.dumps(ids),)
    ).fetchone()[0]
    return ids, int(stored)


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
        ids, stored = _stored_sources(conn, source_ids, source_type)
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


def _groups(conn, scan: dict[str, Any]) -> tuple[dict[int, tuple], list[list[int]]]:
    """Rows of the flagged nodes and their ancestors, split into connected groups, each ordered bottom-up."""
    ids = sorted({item["node_id"] for item in scan["flagged"]} | {item["node_id"] for item in scan["ancestors"]})
    rows = {int(row[0]): row for row in conn.execute(
        "SELECT node_id, session_id, depth, summary, token_count, source_ids, source_type FROM summary_nodes "
        "WHERE node_id IN (SELECT value FROM json_each(?))", (json.dumps(ids),))}
    root = {node_id: node_id for node_id in rows}

    def find(node_id: int) -> int:
        while root[node_id] != node_id:
            node_id = root[node_id]
        return node_id

    for node_id, row in rows.items():
        for child in json.loads(row[5] or "[]") if row[6] == "nodes" else ():
            if int(child) in root:
                root[find(int(child))] = find(node_id)
    members: dict[int, list[int]] = defaultdict(list)
    for node_id in rows:
        members[find(node_id)].append(node_id)
    return rows, [sorted(group, key=lambda i: (rows[i][2], i)) for group in sorted(members.values(), key=min)]


def _summarise_group(engine, rows, group, result) -> tuple[dict[int, tuple[str, int, int, int | None]], str]:
    """New (text, tokens, level, source tokens) per node, leaves first and each ancestor from its repaired children."""
    cfg, new = engine._config, {}
    for node_id in group:
        depth, source_ids, source_type = rows[node_id][2], json.loads(rows[node_id][5] or "[]"), rows[node_id][6]
        if source_type == "messages":
            by_id = engine._store.get_batch(sorted({int(i) for i in source_ids}))
            messages = [by_id[i] for i in sorted(by_id)]
            text, source_tokens = engine._serialize_messages(messages), count_messages_tokens(messages)
            budget, new_source_tokens = engine._leaf_target_tokens(source_tokens), None
        else:
            children = []
            for child_id in (int(i) for i in source_ids):
                if child_id in new:
                    children.append(new[child_id][:2])
                elif (child := engine._dag.get_node(child_id)) is not None:
                    children.append((child.summary, child.token_count))
                else:  # never an ancestor from a subset of its children (#667 review M1)
                    return {}, f"source node {child_id} of node {node_id} is missing"
            text, source_tokens = "\n\n---\n\n".join(c[0] for c in children), sum(c[1] for c in children)
            budget, new_source_tokens = max(1000, int(source_tokens * 0.40)), source_tokens
        result["calls"] += 1
        try:
            summary, level = summarize_with_escalation(
                text=text, source_tokens=source_tokens, token_budget=budget, depth=depth, model=cfg.summary_model,
                fallback_models=cfg.summary_fallback_models, reasoning_effort=cfg.summary_reasoning_effort,
                circuit_breaker=engine._summary_circuit_breaker, spend_guard=engine._summary_spend_guard,
                timeout=cfg.summary_timeout_ms / 1000, l2_budget_ratio=cfg.l2_budget_ratio,
                l3_truncate_tokens=cfg.l3_truncate_tokens, custom_instructions=cfg.custom_instructions,
                route_key_prefix="repair:",  # repair rejections never count against the live route's keys
            )
        except Exception as exc:
            return {}, f"summariser error at node {node_id}: {exc}"
        if level == 3:  # a truncation or a verbatim copy: never written by a repair
            return {}, f"summary route refused at node {node_id} (level 3)"
        new[node_id] = (summary, count_tokens(summary), level, new_source_tokens)
    return new, ""


def _commit_group(engine, rows, group, new) -> str:
    """Replace the group's texts in place in one transaction; return why it was rolled back, or "" once committed."""
    conn, group_json = engine._dag.connection, json.dumps(group)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('nodes_fts', ?)", (_PROVENANCE_TABLE,))}
    with engine._dag._db_lock:
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = dict(conn.execute(
                "SELECT node_id, summary FROM summary_nodes WHERE node_id IN (SELECT value FROM json_each(?))",
                (group_json,)).fetchall())
            new_parent = conn.execute(
                "SELECT 1 FROM summary_nodes p, json_each(p.source_ids) j WHERE p.source_type = 'nodes' "
                "AND CAST(j.value AS INTEGER) IN (SELECT value FROM json_each(?)) "
                "AND p.node_id NOT IN (SELECT value FROM json_each(?)) LIMIT 1", (group_json, group_json)).fetchone()
            with getattr(engine, "_condensation_inflight_lock", threading.Lock()):
                in_flight = sorted(set(group) & set(getattr(engine, "_condensation_inflight_ids", ())))
            if in_flight:
                conn.rollback()
                return (f"a condensation in flight uses node {in_flight[0]}; "
                        "run apply again when the agent is idle")
            if new_parent or any(current.get(i) != rows[i][3] for i in group):
                conn.rollback()
                return "a node changed or a new parent linked in during the repair"
            for node_id in group:
                text, tokens, level, source_tokens = new[node_id]
                conn.execute(
                    "UPDATE summary_nodes SET summary = ?, token_count = ?, expand_hint = ?, "
                    "source_token_count = COALESCE(?, source_token_count) WHERE node_id = ?",
                    (text, tokens, engine._extract_expand_hint(text), source_tokens, node_id))
                if _PROVENANCE_TABLE in tables:  # #649 sidecar: the repaired node's level, other columns kept
                    conn.execute(f"UPDATE {_PROVENANCE_TABLE} SET escalation_level = ? WHERE node_id = ?",
                                 (level, node_id))
                if "nodes_fts" in tables:  # nodes_fts has no update trigger: mirror its delete + insert triggers
                    conn.execute("INSERT INTO nodes_fts(nodes_fts, rowid, summary) VALUES('delete', ?, ?)",
                                 (node_id, rows[node_id][3]))
                    conn.execute("INSERT INTO nodes_fts(rowid, summary) VALUES(?, ?)", (node_id, text))
            for start in range(0, len(group), 256):
                VectorStore.purge_embedding_batch_on_connection(conn, group[start:start + 256])
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return ""


def repair_level3_fragments(engine) -> dict[str, Any]:
    """#667 apply: re-summarise each connected group in place, backup first; refuse while no route is available."""
    scan = scan_level3_fragments(engine)
    flagged = {item["node_id"] for item in scan["flagged"]}
    conn = engine._dag.connection
    rows, groups = _groups(conn, scan)
    result: dict[str, Any] = {"status": "ok", "reason": "", "backup": None, "groups": [], "calls": 0}
    for group in groups:
        missing = []  # any node of the group, fragment or ancestor, with a source that is no longer stored (M1)
        for i in group:
            ids, stored = _stored_sources(conn, rows[i][5], rows[i][6])
            if stored < len(ids):
                kind = "message rows" if rows[i][6] == "messages" else "child nodes"
                missing.append(f"node {i} (session {rows[i][1]}) {stored}/{len(ids)} {kind} stored")
        result["groups"].append({
            "nodes": [(i, rows[i][2], "fragment" if i in flagged else "ancestor") for i in group],
            "top": group[-1], "session_id": rows[group[-1]][1], "levels": {},
            "outcome": "skipped" if missing else "pending",
            "reason": "sources missing: " + ", ".join(missing) if missing else "",
        })
    todo = [(entry, group) for entry, group in zip(result["groups"], groups) if entry["outcome"] == "pending"]
    if todo:
        cfg = engine._config
        if not summary_route_available(cfg.summary_model, cfg.summary_fallback_models, engine._summary_circuit_breaker):
            return {**result, "status": "refused", "groups": [],
                    "reason": "every summary route is refused; nothing was changed"}
        result["backup"] = backup_database(engine)
        if not result["backup"]["ok"]:
            return {**result, "status": "error", "groups": [],
                    "reason": f"backup failed: {result['backup']['error']}; nothing was changed"}
    for entry, group in todo:
        new, refusal = _summarise_group(engine, rows, group, result)
        if refusal:
            entry.update(outcome="skipped", reason=refusal)
        elif rollback := _commit_group(engine, rows, group, new):
            entry.update(outcome="rolled back", reason=rollback)
        else:
            entry.update(outcome="repaired", levels={i: new[i][2] for i in group})
    if any(entry["outcome"] != "repaired" for entry in result["groups"]):
        result["status"] = "partial"
    result["second_scan"] = scan_level3_fragments(engine)
    return result
