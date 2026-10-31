"""#574: one batch per backfill pass, preserving NULL-only updates."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


def test_backfill_many_matches_per_row_loop(tmp_path):
    stores = [MessageStore(tmp_path / f"synthetic-{i}.db") for i in range(2)]
    try:
        for store in stores:
            for i in range(3):
                message = {"role": "user", "content": f"row {i}"}
                if i == 1:
                    message["timestamp"] = 50.0
                store.append("s", message)
        pairs = [(1, 10.0), (2, 20.0), (3, 30.0), (1, 40.0), (999, 99.0)]
        changed = sum(stores[0].backfill_observed_at(*pair) for pair in pairs)
        trace = []
        stores[1]._conn.set_trace_callback(trace.append)
        assert stores[1].backfill_observed_at_many(iter(pairs)) == changed == 2
        assert sum(s.startswith("BEGIN") for s in trace) == 1
        assert trace.count("COMMIT") == 1
        assert stores[1].backfill_observed_at_many([]) == 0
        rows = [store._conn.execute(
            "SELECT store_id, observed_at, observed_at_source FROM messages ORDER BY store_id"
        ).fetchall() for store in stores]
        assert rows[0] == rows[1]
        assert [row[1] for row in rows[1]] == [10.0, 50.0, 30.0]
    finally:
        for store in stores:
            store.close()


def test_backfill_many_empty_batch_leaves_caller_transaction_open(tmp_path):
    store = MessageStore(tmp_path / "empty-batch.db")
    try:
        store.append("s", {"role": "user", "content": "row"})
        store._conn.execute("UPDATE messages SET content = 'pending' WHERE store_id = 1")
        assert store._conn.in_transaction
        assert store.backfill_observed_at_many(iter(())) == 0
        assert store._conn.in_transaction  # an empty pass commits nothing, as the per-row loop did
        store._conn.rollback()
        assert store._conn.execute("SELECT content FROM messages WHERE store_id = 1").fetchone()[0] == "row"
    finally:
        store.close()


@pytest.mark.parametrize("site", ["prefix", "commit"])
def test_backfill_call_sites_batch_once(tmp_path, monkeypatch, site):
    def new_engine():
        engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "synthetic.db"),
                                          temporal_rollups_enabled=False))
        engine.on_session_start("s", platform="cli", context_length=200_000)
        return engine

    engine = new_engine()
    head = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "second"}]
    engine.ingest(head)
    if site == "prefix":
        engine.shutdown()
        engine = new_engine()
    try:
        batches = []
        original = engine._store.backfill_observed_at_many

        def batch(pairs):
            pairs = list(pairs)
            batches.append(pairs)
            return original(pairs)

        def per_row(*args):
            pytest.fail("per-row backfill must not run")

        monkeypatch.setattr(engine._store, "backfill_observed_at_many", batch)
        monkeypatch.setattr(engine._store, "backfill_observed_at", per_row)
        pairs = [(1, 10.0), (2, 20.0)]
        if site == "prefix":
            engine._identity_anchor_commit = lambda *args: None
            engine.ingest([{**message, "timestamp": stamp} for message, (_, stamp) in zip(head, pairs)]
                          + [{"role": "user", "content": "third", "timestamp": 30.0}])
        else:
            engine._identity_anchor_commit({"relations": [], "backfill": iter(pairs), "carry": []})
        assert batches == [pairs]
        rows = engine._store.get_session_messages("s")
        assert [row["observed_at"] for row in rows[:2]] == [10.0, 20.0]
    finally:
        engine.shutdown()
