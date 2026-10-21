"""#1161: output equivalence against the SQL at 76769913."""

from typing import Any, Dict

import pytest

from hermes_lcm.store import MessageStore, _MESSAGE_SELECT_COLUMNS, _MESSAGE_SELECT_COLUMN_COUNT


def old_scan_evidence_rows(self, *, limit: int = 4096) -> Dict[str, Any]:
    """Return one bounded, read-only whole-corpus evidence snapshot.

    The window metadata and rows come from one SQLite statement, so a
    caller cannot accidentally certify finite coverage from a count and a
    row page taken at different corpus generations.  This API deliberately
    has no query or session filter: a narrower scan is not whole-corpus
    coverage.  Callers must treat ``truncated`` as an honest fallback.
    """
    bounded_limit = min(4096, max(1, int(limit)))
    rows = self._conn.execute(
        f"""
        WITH snapshot AS (
            SELECT {_MESSAGE_SELECT_COLUMNS},
                   COUNT(*) OVER () AS snapshot_total_rows,
                   MAX(store_id) OVER () AS snapshot_max_store_id,
                   SUM(CASE WHEN observed_at IS NULL THEN 1 ELSE 0 END)
                       OVER () AS snapshot_observed_at_missing_rows
            FROM messages
        )
        SELECT * FROM snapshot
        ORDER BY store_id
        LIMIT ?
        """,
        (bounded_limit,),
    ).fetchall()
    if not rows:
        return {
            "rows": [],
            "snapshot_max_store_id": 0,
            "total_rows": 0,
            "returned_rows": 0,
            "truncated": False,
            "observed_at_missing_rows": 0,
        }
    message_column_count = _MESSAGE_SELECT_COLUMN_COUNT
    total_rows = int(rows[0][message_column_count] or 0)
    snapshot_max_store_id = int(rows[0][message_column_count + 1] or 0)
    observed_at_missing_rows = int(rows[0][message_column_count + 2] or 0)
    messages = [self._row_to_dict(row[:message_column_count]) for row in rows]
    return {
        "rows": messages,
        "snapshot_max_store_id": snapshot_max_store_id,
        "total_rows": total_rows,
        "returned_rows": len(messages),
        "truncated": total_rows > len(messages),
        "observed_at_missing_rows": observed_at_missing_rows,
    }


@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("limit", [1, 3, 5])
def test_evidence_scan_matches_window_oracle(tmp_path, count, limit):
    store = MessageStore(tmp_path / "synthetic.db")
    try:
        for i in range(count):
            message = {"role": "user", "content": f"synthetic row {i}"}
            if i % 2:
                message["timestamp"] = 10.0 + i
            store.append(f"s{i % 2}", message)
        expected = old_scan_evidence_rows(store, limit=limit)
        assert store.scan_evidence_rows(limit=limit) == expected
        assert not store._conn.in_transaction
        store._conn.execute("BEGIN")
        assert store.scan_evidence_rows(limit=limit) == expected
        assert store._conn.in_transaction  # Never commits the caller's transaction.
        store._conn.rollback()
    finally:
        store.close()


def test_evidence_scan_keeps_snapshot_across_page_query(tmp_path):
    store = MessageStore(tmp_path / "synthetic.db")
    other = MessageStore(tmp_path / "synthetic.db")
    try:
        store.append("s", {"role": "user", "content": "before"})
        expected = old_scan_evidence_rows(store)
        writes = []

        def trace(sql):
            # A different WAL connection commits after the aggregate read.
            if sql.lstrip().startswith("SELECT store_id,"):
                writes.append((store._conn.in_transaction, store._write_lock._is_owned()))
                other.append("s", {"role": "user", "content": "after"})

        store._conn.set_trace_callback(trace)
        assert store.scan_evidence_rows() == expected
        store._conn.set_trace_callback(None)
        assert store.get_session_count("s") == (2 if writes else 1)
        assert all(transaction and locked for transaction, locked in writes)
    finally:
        store.close()
        other.close()
