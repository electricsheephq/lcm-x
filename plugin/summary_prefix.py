"""Rendered LCM summary parts, prefix verification and source coverage."""

import json
import re


_SUMMARY_DEPTH_LABELS = {0: "Recent", 1: "Session Arc", 2: "Durable"}
_LCM_SUMMARY_PART_HEADER_RE = re.compile(
    r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d(\d+), node (\d+)\)\]\n"
)


def _summary_part_text(node) -> str:
    """One frontier node as context assembly renders it in the summary prefix."""
    label = _SUMMARY_DEPTH_LABELS.get(node.depth, f"Depth-{node.depth}")
    return (
        f"[{label} Summary (d{node.depth}, node {node.node_id})]\n"
        f"{node.summary}\n"
        f"[Expand for details: {node.expand_hint}]"
    )


def verify_summary_prefix(content, get_node, owns_session, header_re=_LCM_SUMMARY_PART_HEADER_RE):
    pos = 0
    node_ids: list[int] = []
    while True:
        header = header_re.match(content, pos)
        if header is None:
            break
        try:
            node = get_node(int(header.group(2)))
        except Exception:
            return None, []
        # A quoted part of an unrelated session's summary is user content (#484 item 11j).
        if node is None or not owns_session(node.session_id) or int(node.depth) != int(header.group(1)):
            return None, []
        part = _summary_part_text(node)
        if not content.startswith(part, pos):
            return None, []
        pos += len(part)
        node_ids.append(int(node.node_id))
        if content.startswith("\n\n---\n\n", pos) and header_re.match(content, pos + 7):
            pos += 7
            continue
        break
    return (pos, node_ids) if node_ids else (None, [])


def summary_coverage_end(conn, node_ids):
    """The greatest store_id under node_ids, walked down to the messages leaves."""
    row = conn.execute(
        f"""WITH RECURSIVE walk(source_type, source_id) AS (
                        SELECT n.source_type, CAST(j.value AS INTEGER) FROM summary_nodes n, json_each(n.source_ids) j
                        WHERE n.node_id IN ({",".join("?" * len(node_ids))})
                        UNION SELECT child.source_type, CAST(j.value AS INTEGER) FROM summary_nodes child
                        JOIN walk ON walk.source_type = 'nodes' AND child.node_id = walk.source_id
                        JOIN json_each(child.source_ids) j
                    ) SELECT MAX(source_id) FROM walk WHERE source_type = 'messages'""",
        [int(node_id) for node_id in node_ids],
    ).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def complete_summary_coverage(conn, node_ids, candidate_ids):
    """Return the coverage boundary and covered candidates if every source resolves."""
    row = conn.execute(
        """WITH RECURSIVE walk(id) AS (
               SELECT value FROM json_each(?)
               UNION
               SELECT CAST(j.value AS INTEGER)
               FROM walk w JOIN summary_nodes n ON n.node_id=w.id
               JOIN json_each(CASE WHEN json_valid(n.source_ids) THEN n.source_ids ELSE '[]' END) j
               WHERE n.source_type='nodes' AND CAST(j.value AS INTEGER) < n.node_id
           ), sources AS (
               SELECT n.node_id, n.source_type, j.type, j.value, CAST(j.value AS INTEGER) AS id
               FROM walk w LEFT JOIN summary_nodes n ON n.node_id=w.id
               LEFT JOIN json_each(CASE
                   WHEN json_valid(n.source_ids) THEN
                       CASE WHEN json_type(n.source_ids)='array' THEN n.source_ids ELSE '[]' END
                   ELSE '[]' END) j
           )
           SELECT MAX(s.id) FILTER (WHERE s.source_type='messages'),
                  json_group_array(s.id) FILTER (WHERE s.source_type='messages'
                      AND s.id IN (SELECT value FROM json_each(?)))
           FROM sources s LEFT JOIN messages m ON s.source_type='messages' AND m.store_id=s.id
           HAVING NOT MAX(CASE
               WHEN s.id IS NULL OR s.id <= 0 OR s.type NOT IN ('integer', 'text')
                   OR CAST(s.id AS TEXT) != CAST(s.value AS TEXT) THEN 1
               WHEN s.source_type='messages' THEN m.store_id IS NULL
               WHEN s.source_type='nodes' THEN s.id >= s.node_id
               ELSE 1 END)""",
        (json.dumps(node_ids), json.dumps(candidate_ids)),
    ).fetchone()
    return (row[0], set(json.loads(row[1]))) if row and row[0] is not None else (None, set())
