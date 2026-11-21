"""The previous GA can read and write a store with the additive tool-call index."""

from contextlib import closing

from hermes_lcm.db_bootstrap import get_schema_version
from hermes_lcm.store import MessageStore
from tests.compat.helpers import run_old


PINS = {"v0.30.0": "6260c31fbce11e4cd542878f1f6a37410d678ae8"}
WRITE_TOOL_ROW = r'''
from hermes_lcm.db_bootstrap import get_schema_version

store = store_mod.MessageStore(db, hermes_home=home)
try:
    index = store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_msg_session_tool_call'"
    ).fetchone()
    assert bool(index) == (mode == "rollback")
    store.append_batch("session", [
        {"role": "tool", "content": mode, "tool_call_id": "call", "tool_name": "shell", "timestamp": 1.0},
    ], token_estimates=[1], conversation_id="conv")
    rows = store.find_session_tool_rows("session", "call", conversation_id="conv", tool_name="shell")
    assert store.connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    print(json.dumps({"version": get_schema_version(store.connection), "rows": rows}))
finally:
    store.close()
'''


def test_v0300_forward_and_rollback_with_tool_call_index(tmp_path, old_tree, host_root):
    old = old_tree("v0.30.0", PINS)
    path = tmp_path / "lcm.db"
    before = run_old(old, tmp_path, host_root, WRITE_TOOL_ROW, mode="seed")
    with closing(MessageStore(path)) as store:
        assert get_schema_version(store.connection) == before["version"]
        assert store.find_session_tool_rows("session", "call", conversation_id="conv", tool_name="shell") == before["rows"]

    after = run_old(old, tmp_path, host_root, WRITE_TOOL_ROW, mode="rollback")
    assert after["version"] == before["version"]
    assert after["rows"][:1] == before["rows"]
    assert [row["content"] for row in after["rows"]] == ["seed", "rollback"]
    with closing(MessageStore(path)) as store:
        assert store.find_session_tool_rows("session", "call", conversation_id="conv", tool_name="shell") == after["rows"]
