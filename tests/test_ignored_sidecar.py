from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.db_bootstrap import ensure_ignored_messages_table, run_versioned_migrations
from hermes_lcm.store import MessageStore


def _store(tmp_path: Path, **config_overrides) -> MessageStore:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), **config_overrides)
    return MessageStore(tmp_path / "lcm.db", ingest_protection_config=config)


class TestIgnoredSidecarMigration:
    def test_table_and_index_created_on_open(self, tmp_path):
        store = _store(tmp_path)
        tables = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "ignored_messages" in tables
        indexes = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_ignored_session" in indexes

    def test_migration_is_idempotent(self, tmp_path):
        db_path = tmp_path / "idempotent.db"
        conn = sqlite3.connect(str(db_path))
        try:
            run_versioned_migrations(conn)
            ensure_ignored_messages_table(conn)  # second call must be a no-op
            ensure_ignored_messages_table(conn)
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='ignored_messages'"
            ).fetchone()[0]
            assert count == 1
        finally:
            conn.close()

    def test_no_schema_version_bump(self, tmp_path):
        """The additive sidecar must not advance the schema version fence."""
        db_path = tmp_path / "version.db"
        conn = sqlite3.connect(str(db_path))
        try:
            run_versioned_migrations(conn)
            from hermes_lcm.db_bootstrap import SCHEMA_VERSION, get_schema_version

            assert get_schema_version(conn) == SCHEMA_VERSION
        finally:
            conn.close()


class TestIgnoredSidecarStore:
    def test_append_and_read_back(self, tmp_path):
        store = _store(tmp_path)
        messages = [
            {"role": "user", "content": "noisy heartbeat 1"},
            {"role": "assistant", "content": "noisy heartbeat 2"},
        ]
        ids = store.append_ignored_batch(
            "s1", messages, source="cli", conversation_id="c1",
            matched_patterns=["heartbeat", "heartbeat"],
        )
        assert len(ids) == 2

        rows = store.get_ignored_session_messages("s1")
        assert [r["content"] for r in rows] == ["noisy heartbeat 1", "noisy heartbeat 2"]
        assert rows[0]["matched_pattern"] == "heartbeat"
        assert rows[0]["source"] == "cli"
        assert rows[0]["conversation_id"] == "c1"
        assert store.get_ignored_session_count("s1") == 2

    def test_sidecar_rows_never_reach_messages_table(self, tmp_path):
        store = _store(tmp_path)
        store.append_batch("s1", [{"role": "user", "content": "real turn"}])
        store.append_ignored_batch("s1", [{"role": "user", "content": "dropped turn"}])

        # The replay/FTS/count surface (messages) must be unaffected by sidecar writes.
        assert store.get_session_count("s1") == 1
        active = store.get_session_messages("s1")
        assert [m["content"] for m in active] == ["real turn"]
        assert store.get_ignored_session_count("s1") == 1

    def test_tool_calls_roundtrip(self, tmp_path):
        store = _store(tmp_path)
        store.append_ignored_batch(
            "s1",
            [{"role": "assistant", "content": "call", "tool_calls": [{"id": "c_1", "type": "function"}]}],
        )
        rows = store.get_ignored_session_messages("s1")
        assert rows[0]["tool_calls"] == [{"id": "c_1", "type": "function"}]

    def test_reassign_moves_sidecar_rows(self, tmp_path):
        store = _store(tmp_path)
        store.append_ignored_batch("old", [{"role": "user", "content": "dropped"}])
        moved = store.reassign_session_messages("old", "new")
        assert store.get_ignored_session_count("old") == 0
        assert store.get_ignored_session_count("new") == 1
        # No messages moved (none existed), but sidecar followed the rebind.
        assert moved == 0

    def test_delete_removes_sidecar_rows(self, tmp_path):
        store = _store(tmp_path)
        store.append_ignored_batch("s1", [{"role": "user", "content": "dropped"}])
        store.delete_session_messages("s1")
        assert store.get_ignored_session_count("s1") == 0

    def test_empty_batch_is_noop(self, tmp_path):
        store = _store(tmp_path)
        assert store.append_ignored_batch("s1", []) == []
        assert store.get_ignored_session_count("s1") == 0

    @pytest.mark.parametrize(
        ("metadata", "value"),
        [
            ("token_estimates", [1]),
            ("matched_patterns", ["heartbeat"]),
        ],
    )
    def test_rejects_metadata_length_mismatch_without_partial_write(
        self, tmp_path, metadata, value
    ):
        store = _store(tmp_path)
        messages = [
            {"role": "user", "content": "noisy heartbeat 1"},
            {"role": "assistant", "content": "noisy heartbeat 2"},
        ]

        with pytest.raises(ValueError, match=f"{metadata} length"):
            store.append_ignored_batch("s1", messages, **{metadata: value})

        assert store.get_ignored_session_count("s1") == 0


class TestIgnoredSidecarProtection:
    def test_sensitive_content_is_redacted_before_persist(self, tmp_path):
        store = _store(
            tmp_path,
            large_output_externalization_path=str(tmp_path / "ext"),
        )
        setattr(store._ingest_protection_config, "sensitive_patterns_enabled", True)
        setattr(store._ingest_protection_config, "sensitive_patterns", ["api_key"])
        secret = 'api_key="AKIA1234567890SECRET"'
        store.append_ignored_batch("s1", [{"role": "user", "content": secret}])

        stored = store.get_ignored_session_messages("s1")[0]["content"]
        assert "AKIA1234567890SECRET" not in stored
        assert "[LCM sensitive redaction:" in stored
