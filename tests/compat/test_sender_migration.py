"""S0 forward and rollback against the previous GA and oldest supported reader."""
import sqlite3

import pytest

from hermes_lcm.store import MessageStore
from tests.compat.helpers import run_old


OLD = r'''
options.update(fresh_tail_count=4, leaf_chunk_tokens=1000)
engine = LCMEngine(config=LCMConfig(**options), hermes_home=home)
try:
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    before = engine._store.get_session_messages("S")
    view = [{"role": role, "content": f"synthetic {role} {turn} " * 400, "timestamp": 101.0 + 2 * turn + offset}
            for turn in range(16) for offset, role in enumerate(("user", "assistant"))]
    engine.ingest(view)
    engine.compress(view, current_tokens=100_000, force=True)
    assert engine._dag.get_session_nodes("S"), "old-reader compaction must publish a summary"
    assert engine._store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert engine._store.get_session_messages("S")
    print(json.dumps({"read": bool(before), "rows": len(engine._store.get_session_messages("S"))}))
finally:
    engine.shutdown()
'''


@pytest.mark.parametrize("reader", ["v0.28.1", "v0.25.0-rc1"])
def test_t8_sender_forward_and_rollback(tmp_path, old_tree, host_root, reader):
    pins = {"v0.28.1": "77b2b4f795fb6a4f84c55abf6d0441da4469971e",
            "v0.25.0-rc1": "c36b46e3a7b1bff23cfa45fa117565ab8624441d"}
    old = old_tree(reader, pins=pins)
    first = run_old(old, tmp_path, host_root, OLD, mode="write")
    assert first["rows"]
    with sqlite3.connect(tmp_path / "lcm.db") as conn:
        before = list(conn.execute("SELECT store_id FROM messages"))
    store = MessageStore(tmp_path / "lcm.db")
    try:
        assert all(value == (None, None) for value in store.get_message_senders([r[0] for r in before]).values())
        sid = store._append_protected_batch("S", [{"role": "user", "content": "synthetic retained row"}],
                                            conversation_id="conv", senders=[("telegram:synthetic", "gateway_principal")])[0]
    finally:
        store.close()
    second = run_old(old, tmp_path, host_root, OLD, mode="write")
    assert second["read"] and second["rows"] >= first["rows"]
    store = MessageStore(tmp_path / "lcm.db")
    try:
        values = store.get_message_senders([r[0] for r in store.connection.execute("SELECT store_id FROM messages")])
        if values.pop(sid) != ("telegram:synthetic", "gateway_principal"):
            raise AssertionError("old reader changed retained sender")
        assert all(value == (None, None) for value in values.values())
        store.connection.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)")
        assert store.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        store.close()
