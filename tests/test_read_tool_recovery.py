from __future__ import annotations

import json
import os
import sqlite3

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.read_tool_recovery import (
    build_read_tool_call_path_map,
    is_recoverable_read_tool_marker,
    marker_identity_sha,
    plan_read_tool_recovery,
    recover_read_tool_file,
)
from hermes_lcm.store import MessageStore


def _marker(chars: int = 5000) -> str:
    return (
        "preview of the start of the file...\n\n"
        f"[Truncated: tool response was {chars:,} chars. "
        "Full output could not be saved to sandbox.]"
    )


def _read_call(call_id: str, path: str) -> dict:
    return {
        "role": "assistant",
        "content": "reading the file",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
            }
        ],
    }


class TestPureHelpers:
    def test_marker_detection(self):
        assert is_recoverable_read_tool_marker(_marker())
        assert not is_recoverable_read_tool_marker("ordinary tool output")
        assert not is_recoverable_read_tool_marker(None)

    def test_marker_identity_is_stable_and_content_sensitive(self):
        a = marker_identity_sha("tool", "x", "c1")
        assert a == marker_identity_sha("tool", "x", "c1")
        assert a != marker_identity_sha("tool", "y", "c1")
        assert a != marker_identity_sha("tool", "x", "c2")

    def test_path_map_only_absolute_read_calls(self, tmp_path):
        abs_path = str(tmp_path / "f.txt")
        messages = [
            _read_call("c1", abs_path),
            _read_call("c2", "relative/path.txt"),
            {
                "role": "assistant",
                "content": "other tool",
                "tool_calls": [
                    {"id": "c3", "type": "function",
                     "function": {"name": "terminal", "arguments": json.dumps({"path": abs_path})}}
                ],
            },
        ]
        path_map = build_read_tool_call_path_map(messages)
        assert path_map == {"c1": abs_path}

    def test_recover_reads_absolute_regular_file(self, tmp_path):
        f = tmp_path / "src.txt"
        f.write_text("hello lossless world", encoding="utf-8")
        recovered = recover_read_tool_file(str(f))
        assert recovered is not None
        content, _stat = recovered
        assert content == "hello lossless world"

    def test_recover_rejects_symlink(self, tmp_path):
        target = tmp_path / "real.txt"
        target.write_text("secret", encoding="utf-8")
        link = tmp_path / "link.txt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            return  # platform without symlink support
        assert recover_read_tool_file(str(link)) is None

    def test_recover_rejects_oversized(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100, encoding="utf-8")
        assert recover_read_tool_file(str(f), max_bytes=10) is None

    def test_recover_rejects_relative_path(self):
        assert recover_read_tool_file("relative.txt") is None

    def test_plan_pairs_marker_to_path(self, tmp_path):
        f = tmp_path / "src.txt"
        f.write_text("data", encoding="utf-8")
        messages = [
            _read_call("c1", str(f)),
            {"role": "tool", "tool_call_id": "c1", "content": _marker()},
            {"role": "tool", "tool_call_id": "c1", "content": "not a marker"},
        ]
        plan = plan_read_tool_recovery(messages)
        assert len(plan) == 1
        assert plan[0]["path"] == str(f)
        assert plan[0]["tool_call_id"] == "c1"

    def test_plan_empty_without_read_call(self):
        messages = [{"role": "tool", "tool_call_id": "c1", "content": _marker()}]
        assert plan_read_tool_recovery(messages) == []


class TestStoreSidecar:
    def _store(self, tmp_path, **overrides):
        config = LCMConfig(database_path=str(tmp_path / "lcm.db"), **overrides)
        return MessageStore(tmp_path / "lcm.db", ingest_protection_config=config)

    def test_table_created(self, tmp_path):
        store = self._store(tmp_path)
        tables = {
            r[0] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "recovered_tool_content" in tables

    def test_append_and_lookup(self, tmp_path):
        store = self._store(tmp_path)
        sha = marker_identity_sha("tool", _marker(), "c1")
        rid = store.append_recovered_tool_content(
            "s1", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="full recovered content",
        )
        assert rid is not None
        row = store.get_recovered_tool_content("s1", sha)
        assert row["content"] == "full recovered content"
        assert row["source_path"] == "/abs/f.txt"
        assert row["recovered_chars"] == len("full recovered content")
        assert store.get_recovered_tool_content_count("s1") == 1

    def test_append_is_idempotent_per_marker(self, tmp_path):
        store = self._store(tmp_path)
        sha = marker_identity_sha("tool", _marker(), "c1")
        first = store.append_recovered_tool_content(
            "s1", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="content",
        )
        second = store.append_recovered_tool_content(
            "s1", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="content again",
        )
        assert first is not None
        assert second is None
        assert store.get_recovered_tool_content_count("s1") == 1

    def test_idempotency_is_enforced_across_store_instances(self, tmp_path):
        first_store = self._store(tmp_path)
        second_store = MessageStore(
            tmp_path / "lcm.db",
            ingest_protection_config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
        )
        sha = marker_identity_sha("tool", _marker(), "c1")
        first = first_store.append_recovered_tool_content(
            "s1", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="content",
        )
        second = second_store.append_recovered_tool_content(
            "s1", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="content again",
        )
        assert first is not None
        assert second is None
        assert first_store.get_recovered_tool_content_count("s1") == 1

    def test_bootstrap_deduplicates_legacy_rows_before_unique_index(self, tmp_path):
        db_path = tmp_path / "lcm.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE recovered_tool_content (
                recovered_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                tool_call_id TEXT DEFAULT '',
                marker_identity_sha TEXT NOT NULL,
                source_path TEXT DEFAULT '',
                recovered_chars INTEGER DEFAULT 0,
                externalized_ref TEXT DEFAULT '',
                content TEXT,
                timestamp REAL NOT NULL
            );
            CREATE INDEX idx_recovered_session_marker
                ON recovered_tool_content(session_id, marker_identity_sha);
            INSERT INTO recovered_tool_content
                (session_id, marker_identity_sha, content, timestamp)
                VALUES ('s1', 'same', 'older', 1), ('s1', 'same', 'newer', 2);
            """
        )
        conn.commit()
        conn.close()
        store = self._store(tmp_path)
        rows = store._conn.execute(
            "SELECT content FROM recovered_tool_content "
            "WHERE session_id = 's1' AND marker_identity_sha = 'same'"
        ).fetchall()
        index_rows = store._conn.execute(
            "PRAGMA index_list(recovered_tool_content)"
        ).fetchall()
        assert rows == [("newer",)]
        assert any(row[1] == "idx_recovered_session_marker" and row[2] == 1 for row in index_rows)

    def test_sensitive_content_redacted(self, tmp_path):
        store = self._store(tmp_path, large_output_externalization_path=str(tmp_path / "ext"))
        setattr(store._ingest_protection_config, "sensitive_patterns_enabled", True)
        setattr(store._ingest_protection_config, "sensitive_patterns", ["api_key"])
        sha = marker_identity_sha("tool", _marker(), "c1")
        store.append_recovered_tool_content(
            "s1", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content='api_key="AKIA0987654321SECRET"',
        )
        stored = store.get_recovered_tool_content("s1", sha)["content"]
        assert "AKIA0987654321SECRET" not in stored
        assert "[LCM sensitive redaction:" in stored

    def test_reassign_and_delete_follow_parent(self, tmp_path):
        store = self._store(tmp_path)
        sha = marker_identity_sha("tool", _marker(), "c1")
        store.append_recovered_tool_content(
            "old", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="content",
        )
        store.reassign_session_messages("old", "new")
        assert store.get_recovered_tool_content_count("old") == 0
        assert store.get_recovered_tool_content_count("new") == 1
        store.delete_session_messages("new")
        assert store.get_recovered_tool_content_count("new") == 0


class TestEngineRecovery:
    def _engine(self, tmp_path, *, enabled: bool, **overrides):
        config = LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            read_tool_recovery_enabled=enabled,
            **overrides,
        )
        engine = LCMEngine(config=config)
        engine.on_session_start("chat-1", platform="cli", conversation_id="c1", context_length=200000)
        return engine

    def _turn(self, tmp_path):
        f = tmp_path / "source.txt"
        f.write_text("the full 2MB file content that the host truncated", encoding="utf-8")
        return f, [
            {"role": "user", "content": "read the file"},
            _read_call("call_1", str(f)),
            {"role": "tool", "tool_call_id": "call_1", "content": _marker()},
        ]

    def test_disabled_recovers_nothing(self, tmp_path):
        engine = self._engine(tmp_path, enabled=False)
        _f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        assert engine._store.get_recovered_tool_content_count("chat-1") == 0

    def test_enabled_recovers_full_content_without_touching_messages(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        f, turn = self._turn(tmp_path)
        engine.ingest(turn)

        # The stored active row still holds the byte-identical marker.
        rows = engine._store.get_session_messages("chat-1")
        tool_rows = [r for r in rows if r["role"] == "tool"]
        assert len(tool_rows) == 1
        assert is_recoverable_read_tool_marker(tool_rows[0]["content"])

        # ...but the full file content is recovered into the sidecar.
        sha = marker_identity_sha("tool", _marker(), "call_1")
        recovered = engine._store.get_recovered_tool_content("chat-1", sha)
        assert recovered is not None
        assert recovered["content"] == f.read_text(encoding="utf-8")
        assert recovered["source_path"] == str(f)

    def test_ignored_marker_is_not_recovered(self, tmp_path):
        engine = self._engine(
            tmp_path,
            enabled=True,
            ignore_message_patterns=[r"Full output could not be saved to sandbox"],
        )
        _f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        assert not any(
            row["role"] == "tool"
            for row in engine._store.get_session_messages("chat-1")
        )
        assert engine._store.get_recovered_tool_content_count("chat-1") == 0

    def test_recovery_key_uses_protected_stored_marker(self, tmp_path):
        engine = self._engine(
            tmp_path,
            enabled=True,
            sensitive_patterns_enabled=True,
            sensitive_patterns=["api_key"],
        )
        f, turn = self._turn(tmp_path)
        turn[-1]["content"] = 'api_key="supersecretvalue"\n' + _marker()
        engine.ingest(turn)
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-1")
            if row["role"] == "tool"
        )
        stored_sha = marker_identity_sha("tool", tool_row["content"], "call_1")
        raw_sha = marker_identity_sha("tool", turn[-1]["content"], "call_1")
        assert stored_sha != raw_sha
        recovered = engine._store.get_recovered_tool_content("chat-1", stored_sha)
        assert recovered is not None
        assert recovered["content"] == f.read_text(encoding="utf-8")
        assert engine._store.get_recovered_tool_content("chat-1", raw_sha) is None

    def test_recovery_is_attempted_once_per_marker(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        # Mutate the file, then re-ingest the same marker: the in-memory guard
        # must prevent a second re-read, so the first recovery is preserved.
        f.write_text("changed content", encoding="utf-8")
        engine.ingest(turn)
        sha = marker_identity_sha("tool", _marker(), "call_1")
        recovered = engine._store.get_recovered_tool_content("chat-1", sha)
        assert recovered["content"] == "the full 2MB file content that the host truncated"
        assert engine._store.get_recovered_tool_content_count("chat-1") == 1

    def test_status_surfaces_recovery_count_when_enabled(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        _f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        status = engine.get_status()
        assert status["read_tool_recovery_enabled"] is True
        assert status["recovered_read_tool_content_count"] == 1

    def test_no_read_call_means_no_recovery(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        turn = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "orphan", "content": _marker()},
        ]
        engine.ingest(turn)
        assert engine._store.get_recovered_tool_content_count("chat-1") == 0
