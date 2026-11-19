from contextlib import closing
from statistics import median
from time import thread_time
from timeit import Timer

from hermes_lcm.db_bootstrap import get_schema_version
from hermes_lcm.store import MessageStore


INDEX = "idx_msg_session_tool_call"
INSERT = """INSERT INTO messages
    (session_id, conversation_id, role, content, tool_call_id, tool_name, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)"""


def _indexes(conn):
    return {row[1]: (row[2], row[4]) for row in conn.execute("PRAGMA index_list(messages)")}


def test_fresh_store_has_partial_tool_call_index(tmp_path):
    with closing(MessageStore(tmp_path / "lcm.db")) as store:
        conn = store.connection
        assert _indexes(conn)[INDEX] == (0, 1)  # Non-unique and partial.
        assert [row[2] for row in conn.execute(f"PRAGMA index_info({INDEX})")] == ["session_id", "tool_call_id"]
        sql, = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (INDEX,)).fetchone()
        assert " ".join(sql.split()).endswith("WHERE tool_call_id IS NOT NULL")


def test_open_adds_missing_index_to_existing_store(tmp_path):
    path = tmp_path / "lcm.db"
    with closing(MessageStore(path)) as store:
        conn = store.connection
        conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
        conn.execute(INSERT, ("session", "conv", "tool", "stored output", "call", "shell", 1.0))
        conn.commit()
        before = store.find_session_tool_rows("session", "call", conversation_id="conv", tool_name="shell")
        version = get_schema_version(conn)
        assert before and INDEX not in _indexes(conn)

    with closing(MessageStore(path)) as store:
        assert INDEX in _indexes(store.connection)
        assert get_schema_version(store.connection) == version
        assert store.find_session_tool_rows("session", "call", conversation_id="conv", tool_name="shell") == before
        assert store.connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def _lookup_plan(conn, lookup):
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        rows = lookup()
    finally:
        conn.set_trace_callback(None)
    sql, = [statement for statement in statements if "AND role = 'tool' AND tool_call_id =" in statement]
    return rows, [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]


def test_lookup_plan_matches_full_session_scan_in_order_and_is_faster(tmp_path):
    with closing(MessageStore(tmp_path / "lcm.db")) as store:
        conn = store.connection
        siblings = [
            ("session", "", "tool", "blank owner", "call-0", "", 0),
            ("session", " conv ", "tool", "padded owner", "call-0", "shell", 0),
            ("session", None, "tool", "null owner", "call-0", None, 0),
            ("session", "other", "tool", "foreign owner", "call-0", "shell", 0),
            ("session", "conv", "tool", "other tool", "call-0", "other", 0),
            ("session", "conv", "assistant", "wrong role", "call-0", "shell", 0),
            ("foreign-session", "conv", "tool", "wrong session", "call-0", "shell", 0),
        ]
        conn.executemany(INSERT, siblings)
        conn.executemany(INSERT, (
            ("session", "conv", "tool", f"output {i}", f"call-{i // 25 % 200}", "shell", 1)
            if i % 25 == 0 else ("session", "conv", "user", f"input {i}", None, None, 1)
            for i in range(30_000 - sum(row[0] == "session" for row in siblings))
        ))
        conn.commit()

        def lookup():
            return store.find_session_tool_rows("session", "call-0", conversation_id="conv", tool_name="shell")

        indexed, plan = _lookup_plan(conn, lookup)
        assert any(INDEX in step for step in plan), plan
        assert [row["content"] for row in indexed] == [
            "blank owner", "padded owner", "null owner", *(f"output {i}" for i in range(0, 30_000, 5000)),
        ]

        # Compare warmed batches of the identical API call. Thread CPU time keeps
        # scheduling noise out of the ratio, including on busy CI runners.
        timer = Timer(lookup, timer=thread_time)
        indexed_time = median(timer.repeat(repeat=5, number=20))
        conn.execute(f"DROP INDEX {INDEX}")
        assert lookup() == indexed
        scan_time = median(timer.repeat(repeat=5, number=20))
        assert scan_time >= 10 * indexed_time, f"session scan {scan_time:.6f}s, indexed {indexed_time:.6f}s"
