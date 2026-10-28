"""#836: host-uid shadow hygiene. Bindings go with their deleted rows on both message-delete paths, and the
persisted lineage key is namespaced by the Hermes home (``<home_tag>:<root>``)."""

from __future__ import annotations

import re
import sqlite3

import pytest

import hermes_lcm.command as command_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 4


@pytest.fixture(autouse=True)
def _shadow(monkeypatch):
    monkeypatch.delenv("LCM_HOST_MESSAGE_UID", raising=False)


def _m(text: str, ts: float, uid: str, role: str = "user") -> dict:
    return {"role": role, "content": text, "timestamp": ts, "message_uid": uid}


def _state_db(home, sessions) -> None:
    home.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(home / "state.db")
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, end_reason TEXT, "
                 "source TEXT, model_config TEXT)")
    conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, NULL, NULL)", sessions)
    conn.commit()
    conn.close()


def _engine(db, home, session: str = "S", conversation: str = "conv") -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=2, leaf_chunk_tokens=1, database_path=str(db)),
                       hermes_home=str(home))
    engine.on_session_start(session, platform="cli", context_length=200_000, conversation_id=conversation)
    return engine


def _bindings(conn) -> list[tuple]:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='host_uid_bindings'").fetchone():
        return []
    return conn.execute("SELECT store_id, uid, lineage_key, kind FROM host_uid_bindings ORDER BY rowid").fetchall()


def _ids(engine, session: str) -> set[int]:
    return {int(row["store_id"]) for row in engine._store.get_session_messages(session)}


# -- 1. purge on delete ---------------------------------------------------------------------------------

def _two_sessions(tmp_path):
    _state_db(tmp_path, [("S", None, None), ("T", None, None)])
    other = _engine(tmp_path / "lcm.db", tmp_path, "T", "conv-t")
    try:
        other.ingest([_m("t one" + PAD, 1.0, "u-t1"), _m("t two" + PAD, 2.0, "u-t2")])
    finally:
        other.shutdown()
    engine = _engine(tmp_path / "lcm.db", tmp_path)
    engine.ingest([_m("s one" + PAD, 3.0, "u-s1"), _m("s two" + PAD, 4.0, "u-s2")])
    return engine


@pytest.mark.parametrize("path", ["delete_session_messages", "doctor_clean"])
def test_bindings_are_deleted_with_their_rows(tmp_path, path):
    engine = _two_sessions(tmp_path)
    conn = engine._store._conn
    try:
        s_ids, t_ids = _ids(engine, "S"), _ids(engine, "T")
        assert {b[0] for b in _bindings(conn)} == s_ids | t_ids
        if path == "delete_session_messages":
            engine._store.delete_session_messages("S")
        else:
            engine.on_session_start("other", platform="cli", context_length=200_000, conversation_id="conv2")
            command_mod._delete_clean_candidates_atomically(engine, {"S"})
        assert not engine._store.get_session_messages("S")
        assert {b[0] for b in _bindings(conn)} == t_ids  # the other session's bindings stay
        assert sorted(b[1] for b in _bindings(conn)) == ["u-t1", "u-t2"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("path", ["delete_session_messages", "doctor_clean"])
def test_delete_without_the_side_table_works_and_does_not_create_it(tmp_path, monkeypatch, path):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "off")
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path / "lcm.db", tmp_path)
    try:
        engine.ingest([_m("s one" + PAD, 3.0, "u-s1")])
        if path == "delete_session_messages":
            assert engine._store.delete_session_messages("S") == 1
        else:
            engine.on_session_start("other", platform="cli", context_length=200_000, conversation_id="conv2")
            assert command_mod._delete_clean_candidates_atomically(engine, {"S"})["messages_deleted"] == 1
        assert not engine._store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='host_uid_bindings'").fetchone()
    finally:
        engine.shutdown()


_LEGACY_MESSAGES = """CREATE TABLE messages (
    store_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, source TEXT DEFAULT '', conversation_id TEXT DEFAULT '',
    role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL,
    token_estimate INTEGER DEFAULT 0, pinned INTEGER DEFAULT 0, ingested_at REAL, observed_at REAL,
    observed_at_source TEXT)"""  # a plain INTEGER PRIMARY KEY, as the legacy rebuild creates: ids can be reused


def test_a_reused_store_id_never_inherits_a_stale_binding(tmp_path):
    conn = sqlite3.connect(tmp_path / "lcm.db")
    conn.execute(_LEGACY_MESSAGES)
    conn.commit()
    conn.close()
    engine = _two_sessions(tmp_path)  # T first, then S: S holds the highest store id
    try:
        s_ids = _ids(engine, "S")
        assert max(s_ids) == max(_ids(engine, "S") | _ids(engine, "T"))
        assert s_ids <= {b[0] for b in _bindings(engine._store._conn)}
        engine._store.delete_session_messages("S")
        reused = engine._store.append("T", {"role": "user", "content": "a new row"})
        assert reused in s_ids  # SQLite reused a deleted id that carried a binding
        assert reused not in {b[0] for b in _bindings(engine._store._conn)}
    finally:
        engine.shutdown()


# -- 2. the persisted lineage key is namespaced by home -------------------------------------------------

def test_two_homes_sharing_one_database_bind_separately(tmp_path):
    db = tmp_path / "shared" / "lcm.db"
    _state_db(tmp_path / "one", [("S", None, None)])
    _state_db(tmp_path / "two", [("S", None, None)])
    results = []
    for home, text in (("one", "first profile" + PAD), ("two", "second profile" + PAD)):
        engine = _engine(db, tmp_path / home)
        try:
            engine.ingest([_m(text, 10.0, "u-shared")])
            results.append(dict(engine._host_uid_counters))
        finally:
            engine.shutdown()
    assert results == [{"unbound.agree_new": 1}, {"unbound.agree_new": 1}]  # no error, no disagreement
    conn = sqlite3.connect(db)
    try:
        rows = _bindings(conn)
    finally:
        conn.close()
    assert [(uid, kind) for _sid, uid, _key, kind in rows] == [("u-shared", "canonical"), ("u-shared", "canonical")]
    assert len({key for _sid, _uid, key, _kind in rows}) == 2
    assert {key.split(":", 1)[1] for _sid, _uid, key, _kind in rows} == {"S"}


def test_the_key_format_and_a_symlinked_home_give_the_same_tag(tmp_path):
    _state_db(tmp_path / "real", [("S", None, None)])
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    keys = []
    for home in ("real", "link"):
        engine = _engine(tmp_path / "lcm.db", tmp_path / home)
        try:
            keys.append(engine._host_uid_lineage_key()[0])
        finally:
            engine.shutdown()
    assert re.fullmatch(r"[0-9a-f]{16}:S", keys[0]), keys
    assert keys[0] == keys[1]
