import sqlite3

import pytest

from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.sqlite_util import _run_sqlite_write_with_snapshot_retry
from hermes_lcm.store import MessageStore


def test_dag_write_retries_one_stale_snapshot(tmp_path) -> None:
    database_path = tmp_path / "issue-4-dag-snapshot.db"
    dag = SummaryDAG(database_path)
    writer = sqlite3.connect(database_path)

    try:
        dag.connection.execute("BEGIN")
        dag.connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()
        writer.execute(
            "INSERT OR REPLACE INTO metadata(key, value) "
            "VALUES('issue-4-writer', 'advanced')"
        )
        writer.commit()

        node_id = dag.add_node(
            SummaryNode(
                session_id="issue-4-dag-snapshot",
                depth=0,
                summary="retry from a fresh transaction",
                token_count=5,
                source_ids=[1],
                source_type="messages",
            )
        )
        stored = dag.get_node(node_id)
        in_transaction = dag.connection.in_transaction
    finally:
        writer.close()
        dag.close()

    assert node_id > 0
    assert stored is not None
    assert stored.summary == "retry from a fresh transaction"
    assert in_transaction is False


def test_message_batch_retries_stale_snapshot_with_metadata(tmp_path) -> None:
    database_path = tmp_path / "issue-4-store-snapshot.db"
    store = MessageStore(database_path)
    writer = sqlite3.connect(database_path)

    try:
        store.connection.execute("BEGIN")
        store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()
        writer.execute(
            "INSERT OR REPLACE INTO metadata(key, value) "
            "VALUES('issue-4-writer', 'advanced')"
        )
        writer.commit()

        ids = store._append_protected_batch(
            "issue-4-store-snapshot",
            [{"role": "user", "content": "retry with provenance"}],
            metadata_factory=lambda _message, store_id: [
                (f"issue-4-provenance:{store_id}", '"real-user"')
            ],
        )
        metadata = store.read_metadata_json(
            f"issue-4-provenance:{ids[0]}"
        )
        stored = store.get(ids[0])
        in_transaction = store.connection.in_transaction
    finally:
        writer.close()
        store.close()

    assert stored is not None
    assert stored["content"] == "retry with provenance"
    assert metadata == "real-user"
    assert in_transaction is False


def test_ordinary_busy_uses_one_bounded_attempt(tmp_path) -> None:
    database_path = tmp_path / "issue-4-ordinary-busy.db"
    victim = sqlite3.connect(database_path)
    holder = sqlite3.connect(database_path)
    victim.execute("PRAGMA journal_mode=WAL")
    victim.execute("PRAGMA busy_timeout=1")
    holder.execute("PRAGMA busy_timeout=1")
    victim.execute(
        "CREATE TABLE IF NOT EXISTS issue_4_busy(key TEXT PRIMARY KEY)"
    )
    victim.commit()
    holder.execute("BEGIN IMMEDIATE")
    attempts = 0

    def blocked_write() -> None:
        nonlocal attempts
        attempts += 1
        victim.execute(
            "INSERT INTO issue_4_busy(key) VALUES('blocked')"
        )

    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            _run_sqlite_write_with_snapshot_retry(
                victim,
                blocked_write,
                operation_name="issue-4.ordinary-busy",
            )
        in_transaction = victim.in_transaction
    finally:
        if holder.in_transaction:
            holder.execute("ROLLBACK")
        holder.close()
        victim.close()

    assert attempts == 1
    assert in_transaction is False


def test_stale_snapshot_retry_is_bounded_to_two_attempts() -> None:
    connection = sqlite3.connect(":memory:")
    attempts = 0

    def stale_write() -> None:
        nonlocal attempts
        attempts += 1
        exc = sqlite3.OperationalError("database is locked")
        exc.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
        raise exc

    try:
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            _run_sqlite_write_with_snapshot_retry(
                connection,
                stale_write,
                operation_name="issue-4.stale-snapshot",
            )
        in_transaction = connection.in_transaction
    finally:
        connection.close()

    assert attempts == 2
    assert in_transaction is False
