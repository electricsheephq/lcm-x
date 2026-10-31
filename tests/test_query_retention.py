"""#1161: output equivalence against the SQL at 76769913."""

import pytest
from typing import List

from hermes_lcm.store import MessageStore
from hermes_lcm.dag import SummaryDAG


def old_scan_session_retention_stats(self, session_id: str) -> List[tuple]:
    """Per-session activity/token stats for one session (messages + summary
    nodes), for ``/lcm doctor retention`` scanning. Callers own the
    staleness/protection policy."""
    return self._conn.execute(
        """
        WITH session_ids AS (
            SELECT session_id FROM messages
            UNION
            SELECT session_id FROM summary_nodes
        ),
        message_stats AS (
            SELECT session_id,
                   COUNT(*) AS message_count,
                   COALESCE(SUM(token_estimate), 0) AS token_total,
                   MIN(timestamp) AS first_message_at,
                   MAX(timestamp) AS last_message_at
            FROM messages
            GROUP BY session_id
        ),
        node_stats AS (
            SELECT session_id,
                   COUNT(*) AS node_count,
                   COALESCE(SUM(token_count), 0) AS node_token_total,
                   MIN(COALESCE(earliest_at, created_at)) AS first_node_at,
                   MAX(COALESCE(latest_at, created_at)) AS last_node_at
            FROM summary_nodes
            GROUP BY session_id
        )
        SELECT s.session_id,
               COALESCE(m.message_count, 0) AS message_count,
               COALESCE(m.token_total, 0) AS token_total,
               COALESCE(n.node_count, 0) AS node_count,
               COALESCE(n.node_token_total, 0) AS node_token_total,
               m.first_message_at,
               m.last_message_at,
               n.first_node_at,
               n.last_node_at
        FROM session_ids s
        LEFT JOIN message_stats m ON m.session_id = s.session_id
        LEFT JOIN node_stats n ON n.session_id = s.session_id
        WHERE s.session_id = ?
        ORDER BY s.session_id
        """,
        (session_id,),
    ).fetchall()


@pytest.mark.parametrize("session", ["messages", "nodes", "both", "unknown"])
def test_retention_matches_whole_store_oracle(tmp_path, session):
    path = tmp_path / "synthetic.db"
    store, dag = MessageStore(path), SummaryDAG(path)
    try:
        for sid in ("messages", "both"):
            for i in range(2):
                store.append(sid, {"role": "user", "content": f"synthetic {i}", "timestamp": 10.0 + i})
        dag._conn.executemany(
            "INSERT INTO summary_nodes(session_id, summary, token_count, created_at, earliest_at, latest_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [("nodes", "first", 3, 20.0, None, None), ("both", "second", 5, 30.0, 10.0, 11.0)],
        )
        dag._conn.commit()
        assert store.scan_session_retention_stats(session) == old_scan_session_retention_stats(store, session)
    finally:
        store.close()
        dag.close()
