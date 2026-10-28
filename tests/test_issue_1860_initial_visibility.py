"""Initial read-only d3 visibility must not depend on a writable parent fence."""
import json

from tests.test_issue_1536_chunk_revision import (
    LONG, add_rows, command, counts, metadata, store as store,
)


def test_d3_initial_readonly_status_keeps_unembedded_low_row_visible(store):
    add_rows(store, [(i, "user", LONG) for i in range(1, 8)], {1, 3, 4, 5, 6, 7})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through",
             json.dumps({"from": 3, "through": 7, "policy": "conversational"}))
    conn, identity = store[1].connection, store[2].identity_hash
    assert 2 not in counts(store)
    assert metadata(store, "chunk_gapfill_scan") is None
    before = list(conn.iterdump()), conn.total_changes

    backlog = command._chunk_backlog_estimate(conn, identity, "conversational")

    assert int(backlog.split()[0].rstrip("+")) >= 1, backlog
    assert backlog == "1 (gap-fill)"
    assert 2 not in counts(store)
    assert (list(conn.iterdump()), conn.total_changes) == before
