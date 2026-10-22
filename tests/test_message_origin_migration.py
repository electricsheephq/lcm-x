"""Characterize standalone store initialization before removing duplicate origin DDL."""

import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.store import MessageStore

OLD_COMMIT = "6800cf45ad05280e6cf6fa9e1ad96c62b7132646"  # v0.26.2
SHAPES = ("no-origin", "no-source", "column", "nullable", "indexed")
# Hashes cover every sqlite_master SQL definition, including unrelated schema objects.
# Captured on the pre-refactor tree, with the same assertions and fixtures below.
# S0 DECLARE: regenerated for two nullable sender TEXT columns, schema_version still 5.
SCHEMA_HASHES = {
    "no-origin": "7c713ed3172754452fea7db661980ecf2ae50d7ed74f44ec7277ecc3dcf3fa36",
    "no-source": "b404228fe3f62e0041044f5f86a95a5755164859f28fa1760ccc5c6d0d77a038",
    "column": "3f636eca7f43e14dbf1f02448cc1235b8baf1ca02c45fcbdc2697ed120982501",
    "nullable": "201149f63107323c870dbb84e0e548b24a5c7b198896d2ebea7b25d1cce5de28",
    "indexed": "3f636eca7f43e14dbf1f02448cc1235b8baf1ca02c45fcbdc2697ed120982501",
    "fresh": "60f0c61283050a49c4e26b5729dc6618dcaa72e461337a20d73a75f74977d4da",
    "v0.26.2": "8b774fdbd65aa2fcaf3420f2b54c4eb5480d0ff3d38269d329fbb6a8f095c74f",
}


def _seed(path, shape, version):
    with sqlite3.connect(path) as conn:
        source = "" if shape == "no-source" else "source TEXT DEFAULT '',"
        origin = {
            "no-origin": "", "no-source": "",
            "nullable": "conversation_id TEXT,",
        }.get(shape, "conversation_id TEXT DEFAULT '',")
        conn.executescript(f"""
            CREATE TABLE messages (
                store_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, {source} {origin}
                role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT,
                tool_name TEXT, timestamp REAL NOT NULL,
                token_estimate INTEGER DEFAULT 0, pinned INTEGER DEFAULT 0
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
            PRAGMA user_version = 37;
        """)
        conn.execute("INSERT INTO metadata VALUES ('schema_version', ?)", (str(version),))
        rows = [
            (7, "s1", "user", "Keep this context.", None, None, None, 10.0, 5, 1),
            (9, "s1", "assistant", "", None,
             '[{"id":"call","type":"function","function":{"name":"tool","arguments":"{}"}}]',
             None, 11.0, 4, 0),
            (12, "s1", "tool", "Synthetic result", "call", None, "tool", 12.0, 3, 0),
            (20, "s2", "user", "Separate conversation", None, None, None, 13.0, 2, 0),
        ]
        conn.executemany(
            "INSERT INTO messages (store_id, session_id, role, content, tool_call_id, "
            "tool_calls, tool_name, timestamp, token_estimate, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        if source:
            conn.execute("UPDATE messages SET source = 'cli' WHERE store_id = 7")
        if origin:
            conn.execute("UPDATE messages SET conversation_id = 'conv-1' WHERE session_id = 's1'")
        if shape == "indexed":
            conn.execute("CREATE INDEX idx_msg_conversation_session ON messages(conversation_id, session_id, store_id)")


def _snapshot(conn):
    schema = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    columns = [row[1] for row in conn.execute("PRAGMA table_info(messages)")]
    rows = ([dict(zip(columns, row)) for row in conn.execute("SELECT * FROM messages ORDER BY store_id")]
            if columns else [])
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    return schema, rows, version


def _open_and_check(path, monkeypatch):
    with sqlite3.connect(path) as conn:
        before = _snapshot(conn)
    # Derived stem backfill is asynchronous and unrelated to origin migration.
    monkeypatch.setattr(db_bootstrap, "dispatch_background_stem_backfill", lambda *args: None)
    original = db_bootstrap.ensure_message_origin_columns
    calls = []

    def observed(conn):
        original(conn)
        assert "conversation_id" in {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        assert [row[2] for row in conn.execute("PRAGMA index_info(idx_msg_conversation_session)")] == [
            "conversation_id", "session_id", "store_id"]
        calls.append(conn.in_transaction)

    monkeypatch.setattr(db_bootstrap, "ensure_message_origin_columns", observed)
    first = None
    for _ in range(2):
        store = MessageStore(path)
        try:
            conn = store.connection
            after = _snapshot(conn)
            assert after[2] == before[2]
            assert len(after[1]) == len(before[1])
            for old, new in zip(before[1], after[1]):
                assert all(new[key] == value for key, value in old.items())
                assert new["conversation_id"] == old.get("conversation_id", "")
                assert new["source"] == old.get("source", "")
                assert new["ingested_at"] == old.get("ingested_at", old["timestamp"])
                assert new["observed_at"] == old.get("observed_at")
                assert new["observed_at_source"] == old.get("observed_at_source")
            assert conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone() == ("5",)
            assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert not conn.in_transaction
            if first is not None:
                assert after == first
            first = after
        finally:
            store.close()
    assert len(calls) == 2, "bootstrap must migrate origin on each standalone store open"
    return hashlib.sha256(json.dumps(first[0]).encode()).hexdigest()


@pytest.mark.parametrize("version", [0, 5])
@pytest.mark.parametrize("shape", SHAPES)
def test_legacy_origin_schema_and_rows(tmp_path, monkeypatch, shape, version):
    path = tmp_path / "lcm.db"
    _seed(path, shape, version)
    assert _open_and_check(path, monkeypatch) == SCHEMA_HASHES[shape]


def test_fresh_origin_schema(tmp_path, monkeypatch):
    assert _open_and_check(tmp_path / "fresh.db", monkeypatch) == SCHEMA_HASHES["fresh"]


OLD_WRITER = r'''
import importlib.util, sys
from pathlib import Path
from types import SimpleNamespace
old, db = sys.argv[1:]
spec = importlib.util.spec_from_file_location("hermes_lcm", old + "/__init__.py", submodule_search_locations=[old])
sys.modules["hermes_lcm"] = importlib.util.module_from_spec(spec)
from hermes_lcm import db_bootstrap, store, tokens
assert Path(store.__file__).resolve().parent == Path(old).resolve()
db_bootstrap.dispatch_background_stem_backfill = lambda *args: None
tokens._encoder, tokens._encoder_ready = None, True
store.time = SimpleNamespace(time=lambda: 1700000000.0)
writer = store.MessageStore(db)
try:
    for session, conversation in (("s1", "conv-1"), ("s2", "conv-2")):
        writer.append_batch(session, [
            {"role": "user", "content": "Keep archived context.", "timestamp": 10.0},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": "tool", "arguments": "{}"}}]},
            {"role": "tool", "content": "Synthetic result", "tool_call_id": "call", "tool_name": "tool"},
        ], token_estimates=[5, 4, 3], source="cli", conversation_id=conversation)
finally:
    writer.close()
'''


def test_v0262_written_origin_schema_and_rows(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    available = subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{OLD_COMMIT}^{{commit}}"],
                               capture_output=True)
    if available.returncode:
        reason = f"v0.26.2 commit {OLD_COMMIT} must be available for the archived writer"
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)
    archive = subprocess.run(["git", "-C", str(root), "archive", OLD_COMMIT],
                             check=True, capture_output=True).stdout
    old = tmp_path / "old-writer"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter="data")
    path = tmp_path / "lcm.db"
    result = subprocess.run([sys.executable, "-B", "-c", OLD_WRITER, str(old), str(path)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone() == (6,)
    assert _open_and_check(path, monkeypatch) == SCHEMA_HASHES["v0.26.2"]
