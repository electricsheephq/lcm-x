from __future__ import annotations

import json
import multiprocessing
import os
import re
import sqlite3

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm import ingest_protection as ingest_protection_mod
from hermes_lcm import message_patterns as message_patterns_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.db_bootstrap import ensure_recovered_tool_content_table
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.read_tool_recovery import (
    build_read_tool_call_path_map,
    is_recoverable_read_tool_marker,
    marker_identity_sha,
    plan_read_tool_recovery,
    recovered_content_matches_marker,
    recover_read_tool_file,
)
from hermes_lcm.store import MessageStore


def _marker(chars: int = 5000) -> str:
    return (
        "preview of the start of the file...\n\n"
        f"[Truncated: tool response was {chars:,} chars. "
        "Full output could not be saved to sandbox.]"
    )


def _marker_for_content(content: str, preview_chars: int = 16) -> str:
    return (
        content[:preview_chars]
        + "\n\n"
        + f"[Truncated: tool response was {len(content):,} chars. "
        + "Full output could not be saved to sandbox.]"
    )


def _run_recovered_content_migration(db_path: str, start, results) -> None:
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        start.wait(timeout=10)
        ensure_recovered_tool_content_table(conn)
        conn.commit()
        conn.close()
        results.put(None)
    except Exception as exc:
        results.put(repr(exc))


class _TimeoutPattern:
    def __init__(self, pattern: str):
        self._compiled = re.compile(pattern)
        self.pattern = pattern

    def search(self, text: str, *, timeout=None):
        assert timeout is not None
        return self._compiled.search(text)


class _TimeoutRegexEngine:
    @staticmethod
    def compile(pattern: str):
        return _TimeoutPattern(pattern)


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
        content = "hello lossless world"
        f.write_text(content, encoding="utf-8")
        recovered = recover_read_tool_file(
            str(f), marker_content=_marker_for_content(content)
        )
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
        assert recover_read_tool_file(
            str(link), marker_content=_marker_for_content("secret")
        ) is None

    def test_recover_rejects_symlinked_ancestor(self, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        content = "content reached through a symlinked ancestor"
        (real_dir / "source.txt").write_text(content, encoding="utf-8")
        linked_dir = tmp_path / "linked"
        try:
            os.symlink(real_dir, linked_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            return  # platform without symlink support

        assert recover_read_tool_file(
            str(linked_dir / "source.txt"), marker_content=_marker_for_content(content)
        ) is None

    def test_recover_rejects_final_path_replacement_between_stat_and_open(
        self, tmp_path, monkeypatch
    ):
        source = tmp_path / "source.txt"
        original = "shared prefix original-data"
        replacement = "shared prefix attacker-data"
        assert len(original) == len(replacement)
        source.write_text(original, encoding="utf-8")
        replacement_path = tmp_path / "replacement.txt"
        replacement_path.write_text(replacement, encoding="utf-8")
        parked_path = tmp_path / "parked.txt"
        original_open = ingest_protection_mod.os.open
        replaced = False

        def replace_before_open(path, flags, *args, **kwargs):
            nonlocal replaced
            is_final_open = str(path) in {str(source), source.name}
            if is_final_open and not replaced:
                replaced = True
                os.replace(source, parked_path)
                os.replace(replacement_path, source)
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(ingest_protection_mod.os, "open", replace_before_open)

        assert recover_read_tool_file(
            str(source),
            marker_content=_marker_for_content(original, preview_chars=len("shared prefix ")),
        ) is None
        assert replaced is True

    def test_recover_closes_new_directory_fd_when_fstat_fails(self, tmp_path, monkeypatch):
        nested = tmp_path / "unique-fstat-failure-directory"
        nested.mkdir()
        source = nested / "source.txt"
        source.write_text("content", encoding="utf-8")
        original_open = ingest_protection_mod.os.open
        original_fstat = ingest_protection_mod.os.fstat
        opened_directory_fd = None

        def track_directory_open(path, flags, *args, **kwargs):
            nonlocal opened_directory_fd
            fd = original_open(path, flags, *args, **kwargs)
            if path == nested.name:
                opened_directory_fd = fd
            return fd

        def fail_opened_directory_fstat(fd):
            if fd == opened_directory_fd:
                raise OSError("injected directory fstat failure")
            return original_fstat(fd)

        monkeypatch.setattr(ingest_protection_mod.os, "open", track_directory_open)
        monkeypatch.setattr(ingest_protection_mod.os, "fstat", fail_opened_directory_fstat)

        assert ingest_protection_mod._read_regular_file_no_symlink(source) is None
        assert opened_directory_fd is not None
        with pytest.raises(OSError):
            original_fstat(opened_directory_fd)

    def test_recover_rejects_oversized(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100, encoding="utf-8")
        assert recover_read_tool_file(
            str(f), marker_content=_marker_for_content("x" * 100), max_bytes=10
        ) is None

    def test_recover_rejects_relative_path(self):
        assert recover_read_tool_file(
            "relative.txt", marker_content=_marker_for_content("irrelevant")
        ) is None

    def test_recover_rejects_non_regular_file(self, tmp_path):
        assert recover_read_tool_file(
            str(tmp_path), marker_content=_marker_for_content("irrelevant")
        ) is None

    def test_recover_rejects_malformed_marker(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("complete content", encoding="utf-8")
        assert recover_read_tool_file(
            str(source), marker_content="[Truncated: malformed]"
        ) is None

    def test_recover_rejects_changed_file_generation(self, tmp_path, monkeypatch):
        source = tmp_path / "changed-during-read.txt"
        content = "content whose generation changes during recovery"
        source.write_text(content, encoding="utf-8")
        generations = iter([
            {"dev": 1, "ino": 2, "size": len(content), "mtime_ns": 3},
            {"dev": 1, "ino": 2, "size": len(content), "mtime_ns": 4},
        ])
        monkeypatch.setattr(
            ingest_protection_mod,
            "_stat_generation_metadata",
            lambda _stat: next(generations),
        )

        assert recover_read_tool_file(
            str(source), marker_content=_marker_for_content(content)
        ) is None

    def test_recovered_content_matches_original_marker(self):
        content = "the complete original file"
        assert recovered_content_matches_marker(content, _marker_for_content(content))

    def test_recovered_content_rejects_changed_source(self):
        original = "the complete original file"
        changed = "the complete modified file"
        assert len(changed) == len(original)
        assert not recovered_content_matches_marker(changed, _marker_for_content(original))

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

    def test_bootstrap_legacy_index_migration_is_process_safe(self, tmp_path):
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

        context = multiprocessing.get_context("fork")
        start = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_run_recovered_content_migration,
                args=(str(db_path), start, results),
            )
            for _ in range(6)
        ]
        for process in processes:
            process.start()
        start.set()
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
        assert [results.get(timeout=2) for _ in processes] == [None] * len(processes)

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT content FROM recovered_tool_content "
            "WHERE session_id = 's1' AND marker_identity_sha = 'same'"
        ).fetchall()
        indexes = conn.execute("PRAGMA index_list(recovered_tool_content)").fetchall()
        conn.close()
        assert rows == [("newer",)]
        assert any(row[1] == "idx_recovered_session_marker" and row[2] == 1 for row in indexes)

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

    def test_reassign_rolls_back_parent_when_sidecar_update_fails(self, tmp_path):
        store = self._store(tmp_path)
        store.append("old", {"role": "user", "content": "keep me"})
        sha = marker_identity_sha("tool", _marker(), "c1")
        store.append_recovered_tool_content(
            "old", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="content",
        )
        store._conn.execute(
            "CREATE TRIGGER fail_recovered_reassign BEFORE UPDATE "
            "ON recovered_tool_content BEGIN SELECT RAISE(ABORT, 'sidecar update failed'); END"
        )
        store._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="sidecar update failed"):
            store.reassign_session_messages("old", "new")

        assert store.get_session_count("old") == 1
        assert store.get_session_count("new") == 0
        assert store.get_recovered_tool_content_count("old") == 1
        assert store.get_recovered_tool_content_count("new") == 0

    def test_delete_rolls_back_parent_when_sidecar_delete_fails(self, tmp_path):
        store = self._store(tmp_path)
        store.append("session", {"role": "user", "content": "keep me"})
        sha = marker_identity_sha("tool", _marker(), "c1")
        store.append_recovered_tool_content(
            "session", tool_call_id="c1", marker_identity_sha=sha,
            source_path="/abs/f.txt", content="content",
        )
        store._conn.execute(
            "CREATE TRIGGER fail_recovered_delete BEFORE DELETE "
            "ON recovered_tool_content BEGIN SELECT RAISE(ABORT, 'sidecar delete failed'); END"
        )
        store._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="sidecar delete failed"):
            store.delete_session_messages("session")

        assert store.get_session_count("session") == 1
        assert store.get_recovered_tool_content_count("session") == 1


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
        content = "the full 2MB file content that the host truncated"
        f.write_text(content, encoding="utf-8")
        return f, [
            {"role": "user", "content": "read the file"},
            _read_call("call_1", str(f)),
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": _marker_for_content(content),
            },
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
        sha = marker_identity_sha("tool", turn[-1]["content"], "call_1")
        recovered = engine._store.get_recovered_tool_content("chat-1", sha)
        assert recovered is not None
        assert recovered["content"] == f.read_text(encoding="utf-8")
        assert recovered["source_path"] == str(f)

    def test_store_id_expansion_hydrates_recovered_content(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-1")
            if row["role"] == "tool"
        )

        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": tool_row["store_id"], "max_tokens": 1_000_000},
            engine=engine,
        ))

        assert result["content"] == f.read_text(encoding="utf-8")
        assert result["content_source"] == "recovered_read_tool_content"
        assert result["transcript_content"] == tool_row["content"]

    def test_cross_session_store_id_does_not_fetch_plaintext_recovered_sidecar(
        self, tmp_path, monkeypatch
    ):
        engine = self._engine(tmp_path, enabled=True)
        _source, turn = self._turn(tmp_path)
        engine.ingest(turn)
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-1")
            if row["role"] == "tool"
        )

        config = engine._config
        engine.shutdown()
        engine = LCMEngine(config=config)
        engine.on_session_start(
            "chat-2", platform="cli", conversation_id="c2", context_length=200000
        )
        sidecar_fetches = []
        original_fetch = engine._store.get_recovered_tool_content

        def record_fetch(*args, **kwargs):
            sidecar_fetches.append((args, kwargs))
            return original_fetch(*args, **kwargs)

        monkeypatch.setattr(engine._store, "get_recovered_tool_content", record_fetch)
        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": tool_row["store_id"], "max_tokens": 1_000_000},
            engine=engine,
        ))

        assert sidecar_fetches == []
        assert result["from_current_session"] is False
        assert result["content"] == tool_row["content"]
        assert "full 2MB file content" not in json.dumps(result)
        assert "content_source" not in result

    def test_store_id_expansion_hydrates_externalized_recovered_content(self, tmp_path):
        file_content = "LINE ONE\nLINE TWO\n" * 200
        source = tmp_path / "large-source.txt"
        source.write_text(file_content, encoding="utf-8")
        engine = self._engine(
            tmp_path,
            enabled=True,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=500,
            large_output_externalization_path=str(tmp_path / "externalized"),
        )
        engine.ingest([
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [{
                    "id": "call-large",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": str(source)}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-large",
                "content": _marker_for_content(file_content),
            },
        ])
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-1")
            if row["role"] == "tool"
        )

        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": tool_row["store_id"], "max_tokens": 1_000_000},
            engine=engine,
        ))

        assert result["content"] == file_content
        assert result["recovered_chars"] == len(file_content)

    def test_cross_session_expansion_does_not_hydrate_protected_recovered_content(
        self, tmp_path, monkeypatch
    ):
        protected = "PROTECTED_RECOVERED_PAYLOAD_" + ("QUJD" * 1_500)
        source = tmp_path / "protected-source.txt"
        source.write_text(protected, encoding="utf-8")
        engine = self._engine(
            tmp_path,
            enabled=True,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=500,
            large_output_externalization_path=str(tmp_path / "externalized"),
        )
        engine.ingest([
            {"role": "user", "content": "read protected content"},
            _read_call("call-protected", str(source)),
            {
                "role": "tool",
                "tool_call_id": "call-protected",
                "content": _marker_for_content(protected),
            },
        ])
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-1")
            if row["role"] == "tool"
        )
        externalized_ref = next((tmp_path / "externalized").glob("*.json")).name
        ref_store_id = engine._store.append(
            "chat-1",
            {
                "role": "tool",
                "tool_call_id": "call-protected-ref",
                "content": (
                    "[Externalized tool output: tool_call_id=call-protected-ref; "
                    f"chars={len(protected)}; bytes={len(protected.encode())}; "
                    f"ref={externalized_ref}]"
                ),
            },
        )

        config = engine._config
        engine.shutdown()
        engine = LCMEngine(config=config)
        engine.on_session_start(
            "chat-2", platform="cli", conversation_id="c2", context_length=200000
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="chat-2",
            depth=0,
            summary="Foreign protected source",
            token_count=10,
            source_token_count=10,
            source_ids=[tool_row["store_id"], ref_store_id],
            source_type="messages",
        ))

        hydration_calls = []
        original_sidecar_fetch = engine._store.get_recovered_tool_content
        original_recover = lcm_tools._get_recovered_read_tool_content
        original_load = lcm_tools._get_externalized_payload
        original_restore = lcm_tools._restore_ingest_placeholder_for_lookup
        original_find = lcm_tools.find_externalized_payload_for_message

        def record_sidecar_fetch(*args, **kwargs):
            hydration_calls.append("get_recovered_tool_content")
            return original_sidecar_fetch(*args, **kwargs)

        def record_recover(*args, **kwargs):
            hydration_calls.append("_get_recovered_read_tool_content")
            return original_recover(*args, **kwargs)

        def record_load(*args, **kwargs):
            hydration_calls.append("_get_externalized_payload")
            return original_load(*args, **kwargs)

        def record_restore(*args, **kwargs):
            hydration_calls.append("_restore_ingest_placeholder_for_lookup")
            return original_restore(*args, **kwargs)

        def record_find(*args, **kwargs):
            hydration_calls.append("find_externalized_payload_for_message")
            return original_find(*args, **kwargs)

        monkeypatch.setattr(
            engine._store, "get_recovered_tool_content", record_sidecar_fetch
        )
        monkeypatch.setattr(lcm_tools, "_get_recovered_read_tool_content", record_recover)
        monkeypatch.setattr(lcm_tools, "_get_externalized_payload", record_load)
        monkeypatch.setattr(lcm_tools, "_restore_ingest_placeholder_for_lookup", record_restore)
        monkeypatch.setattr(lcm_tools, "find_externalized_payload_for_message", record_find)
        by_store_id = json.loads(lcm_tools.lcm_expand(
            {"store_id": tool_row["store_id"], "max_tokens": 1_000_000},
            engine=engine,
        ))
        by_node_id = json.loads(lcm_tools.lcm_expand(
            {"node_id": node_id, "max_tokens": 1_000_000},
            engine=engine,
        ))

        assert hydration_calls == []
        assert by_store_id["from_current_session"] is False
        assert "PROTECTED_RECOVERED_PAYLOAD_" not in json.dumps(by_store_id)
        assert by_store_id["content"] == tool_row["content"]
        assert "content_source" not in by_store_id
        assert "PROTECTED_RECOVERED_PAYLOAD_" not in json.dumps(by_node_id)
        assert by_node_id["expanded"][0]["content"] == tool_row["content"]
        assert by_node_id["expanded"][0]["content_source"] == "message"
        assert by_node_id["expanded"][1]["content"].startswith("[Externalized tool output:")
        assert by_node_id["expanded"][1]["externalized"] == {
            "ref": externalized_ref,
            "session_id": "chat-1",
            "tool_call_id": "call-protected-ref",
        }

    def test_store_id_expansion_restores_embedded_externalized_payload_in_place(self, tmp_path):
        encoded_one = "QUJD" * 1_500
        encoded_two = "REVG" * 1_500
        file_content = (
            f"prefix remains\n{encoded_one}\nmiddle remains\n"
            f"{encoded_two}\nsuffix remains"
        )
        source = tmp_path / "embedded-payload.txt"
        source.write_text(file_content, encoding="utf-8")
        engine = self._engine(
            tmp_path,
            enabled=True,
            large_output_externalization_enabled=True,
            large_output_externalization_threshold_chars=500,
            large_output_externalization_path=str(tmp_path / "externalized"),
        )
        engine.ingest([
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [{
                    "id": "call-embedded",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": str(source)}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-embedded",
                "content": _marker_for_content(file_content),
            },
        ])
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-1")
            if row["role"] == "tool"
        )

        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": tool_row["store_id"], "max_tokens": 1_000_000},
            engine=engine,
        ))

        assert result["content"] == file_content
        assert result["content"].startswith("prefix remains\n")
        assert result["content"].endswith("\nsuffix remains")

    def test_live_tool_call_recovers_before_same_turn_expand(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path, enabled=True)
        source, turn = self._turn(tmp_path)

        def expand_after_live_ingest(_args, *, engine):
            return json.dumps({
                "recovered": engine._store.get_recovered_tool_content_count("chat-1"),
                "content": source.read_text(encoding="utf-8"),
            })

        monkeypatch.setattr(lcm_tools, "lcm_expand", expand_after_live_ingest)
        result = json.loads(engine.handle_tool_call("lcm_expand", {}, messages=turn))

        assert result["recovered"] == 1
        assert result["content"] == source.read_text(encoding="utf-8")

    def test_session_end_final_flush_recovers_content(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        source, turn = self._turn(tmp_path)

        engine.on_session_end("chat-1", turn)

        identity = marker_identity_sha("tool", turn[-1]["content"], "call_1")
        recovered = engine._store.get_recovered_tool_content("chat-1", identity)
        assert recovered is not None
        assert recovered["content"] == source.read_text(encoding="utf-8")

    def test_node_expansion_hydrates_recovered_content(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-1")
            if row["role"] == "tool"
        )
        node_id = engine._dag.add_node(SummaryNode(
            session_id="chat-1",
            depth=0,
            summary="Recovered read output",
            token_count=10,
            source_token_count=10,
            source_ids=[tool_row["store_id"]],
            source_type="messages",
        ))

        result = json.loads(lcm_tools.lcm_expand(
            {"node_id": node_id, "max_tokens": 1_000_000},
            engine=engine,
        ))
        expanded = result["expanded"][0]

        assert expanded["content"] == f.read_text(encoding="utf-8")
        assert expanded["content_source"] == "recovered_read_tool_content"
        assert expanded["transcript_content"] == tool_row["content"]

    def test_ignored_marker_is_not_recovered(self, tmp_path, monkeypatch):
        monkeypatch.setattr(message_patterns_mod, "_regex_engine", _TimeoutRegexEngine)
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
        content = 'api_key="supersecretvalue"\n' + f.read_text(encoding="utf-8")
        f.write_text(content, encoding="utf-8")
        turn[-1]["content"] = _marker_for_content(content, preview_chars=28)
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
        assert "[LCM sensitive redaction:" in recovered["content"]
        assert "supersecretvalue" not in recovered["content"]
        assert recovered["content"].endswith(
            "the full 2MB file content that the host truncated"
        )
        assert engine._store.get_recovered_tool_content("chat-1", raw_sha) is None

    def test_recovery_is_attempted_once_per_marker(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        # Mutate the file, then re-ingest the same marker: the in-memory guard
        # must prevent a second re-read, so the first recovery is preserved.
        f.write_text("changed content", encoding="utf-8")
        engine.ingest(turn)
        sha = marker_identity_sha("tool", turn[-1]["content"], "call_1")
        recovered = engine._store.get_recovered_tool_content("chat-1", sha)
        assert recovered["content"] == "the full 2MB file content that the host truncated"
        assert engine._store.get_recovered_tool_content_count("chat-1") == 1

    def test_persisted_recovery_skips_file_reread_after_restart(self, tmp_path, monkeypatch):
        """A marker recovered by an earlier process must not trigger the
        expensive file re-read again: the sidecar row is the durable guard,
        checked before recover_read_tool_file, not after via the insert's
        unique constraint."""
        import hermes_lcm.engine as engine_module

        engine = self._engine(tmp_path, enabled=True)
        f, turn = self._turn(tmp_path)
        engine.ingest(turn)
        sha = marker_identity_sha("tool", turn[-1]["content"], "call_1")
        assert engine._store.get_recovered_tool_content("chat-1", sha) is not None

        # Simulate a process restart: the in-memory attempt cache is empty.
        engine._attempted_read_tool_recovery_markers = set()
        reread_calls = []
        original_recover = engine_module.recover_read_tool_file

        def _spy_recover(path, **kwargs):
            reread_calls.append(path)
            return original_recover(path, **kwargs)

        monkeypatch.setattr(
            engine_module, "recover_read_tool_file", _spy_recover
        )
        f.write_text("changed content", encoding="utf-8")
        engine.ingest(turn)

        assert reread_calls == [], "sidecar row must preempt the file re-read"
        recovered = engine._store.get_recovered_tool_content("chat-1", sha)
        assert recovered["content"] == "the full 2MB file content that the host truncated"
        assert engine._store.get_recovered_tool_content_count("chat-1") == 1

    def test_identical_marker_attempted_in_another_session_is_recovered(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        source, turn = self._turn(tmp_path)
        engine.ingest(turn)
        first_session_attempts = set(engine._attempted_read_tool_recovery_markers)

        engine.on_session_start(
            "chat-2", platform="cli", conversation_id="c2", context_length=200000
        )
        # Preserve the process-level attempt cache to model another active
        # session having already encountered this identical marker identity.
        engine._attempted_read_tool_recovery_markers = first_session_attempts
        engine.ingest(turn)

        identity = marker_identity_sha("tool", turn[-1]["content"], "call_1")
        assert engine._store.get_recovered_tool_content("chat-1", identity) is not None
        assert engine._store.get_recovered_tool_content("chat-2", identity) is not None
        tool_row = next(
            row for row in engine._store.get_session_messages("chat-2")
            if row["role"] == "tool"
        )
        result = json.loads(lcm_tools.lcm_expand(
            {"store_id": tool_row["store_id"], "max_tokens": 1_000_000},
            engine=engine,
        ))
        assert result["content"] == source.read_text(encoding="utf-8")

    def test_changed_source_is_not_stored_as_lossless_recovery(self, tmp_path):
        engine = self._engine(tmp_path, enabled=True)
        f, turn = self._turn(tmp_path)
        original = f.read_text(encoding="utf-8")
        changed = original.replace("full", "fake")
        assert len(changed) == len(original)
        f.write_text(changed, encoding="utf-8")
        engine.ingest(turn)
        assert engine._store.get_recovered_tool_content_count("chat-1") == 0

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
