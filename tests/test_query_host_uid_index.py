"""#643: the ingest lookup seeks store ids without changing binding coverage."""

from hermes_lcm.store import MessageStore


def test_host_uid_store_index_plan_and_equivalence(tmp_path):
    path = tmp_path / "synthetic.db"
    store = MessageStore(path)
    for lineage in ("a", "b"):
        store.add_host_uid_bindings(lineage, [
            (sid, f"{lineage}-{sid}-{kind}", kind, "synthetic")
            for sid in (1, 2, 3) for kind in ("canonical", "version", "engine", "alias_candidate")
        ])
    store.close()
    store = MessageStore(path)  # Existing stores use the same migrated index.
    try:
        statements = []
        store._conn.set_trace_callback(statements.append)
        result = store.host_uid_uids_of_stores("a", [1, 3, 999])
        store._conn.set_trace_callback(None)
        sql = next(s for s in statements if s.startswith("SELECT store_id, uid"))
        plan = store._conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        assert any("USING INDEX idx_host_uid_store" in row[3] for row in plan)
        old = store._conn.execute(
            "SELECT store_id, uid FROM host_uid_bindings WHERE store_id IN (?, ?, ?) "
            "AND lineage_key = ? AND kind IN ('canonical', 'version')", (1, 3, 999, "a"),
        ).fetchall()
        expected = {}
        for sid, uid in old:
            expected.setdefault(int(sid), set()).add(str(uid))
        assert result == expected
        assert store.host_uid_uids_of_stores("unknown", [1]) == {}
        assert store.host_uid_uids_of_stores("a", []) == {}
    finally:
        store.close()
