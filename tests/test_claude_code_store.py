"""Claude Code capture writer and worker: admission (H3), ingest identity, the ledger, the spool
handshake and the store left readable between worker runs. Synthetic transcripts only.

Design: docs/claude-code-adapter.md.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_lcm import claude_code_store as cc
from hermes_lcm.db_bootstrap import REQUIRED_CORE_TABLES
from hermes_lcm.store import MessageStore

PLUGIN = Path(__file__).resolve().parents[1] / "adapters" / "claude-code"


@pytest.fixture
def hermes(tmp_path):
    return tmp_path / "hermes"


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def _entry(kind, uid, content, session="s1", **extra):
    return {"type": kind, "uuid": uid, "sessionId": session, "timestamp": "2026-11-01T00:00:00.000Z",
            "message": {"role": kind, "content": content}, **extra}


def _transcript(tmp_path, entries, session="s1", name=None) -> Path:
    path = tmp_path / "projects" / "-work" / (name or f"{session}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def _open(home, hermes) -> MessageStore:
    return cc.open_store(cc.admit(home, hermes_home=hermes), home)


def _tables(conn):
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}


def _rows(store, session="s1"):
    return store.connection.execute("SELECT store_id, role, content, tool_call_id FROM messages WHERE session_id = ? "
                                    "ORDER BY store_id", (session,)).fetchall()


def _events(store, column="disposition"):
    return [row[0] for row in store.connection.execute(f"SELECT {column} FROM capture_events ORDER BY event_id")]


def _header_mode(database: Path) -> int:
    with open(database, "rb") as handle:
        handle.seek(18)
        return handle.read(1)[0]  # 1 = rollback journal, 2 = WAL


CONVERSATION = [
    _entry("user", "u1", "find the secret word"),
    _entry("assistant", "a1", [{"type": "thinking", "thinking": "hmm", "signature": "x"},
                               {"type": "text", "text": "Reading notes."},
                               {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "notes.txt"}}]),
    _entry("user", "r1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "the secret word is ochre"}]),
    {"type": "attachment", "uuid": "t1", "sessionId": "s1", "attachment": {"type": "plan"}},
    {"type": "api-request-blob", "sessionId": "s1", "hash": "h1", "blob": {"system": "big prompt"}},
    {"type": "api-request-blob", "sessionId": "s1", "hash": "h1", "blob": {"system": "big prompt"}},
    _entry("assistant", "a2", [{"type": "text", "text": "The word is ochre."}]),
]


def test_admission_creates_a_stamped_store_the_reader_can_open(home, hermes):
    database = cc.admit(home, hermes_home=hermes)
    owner = json.loads((home / cc.OWNER_FILE).read_text())
    assert owner["home"] == str(home.resolve()) and owner["host"] == "claude-code"
    assert cc.read_stamp(database)["store_uuid"] == owner["store_uuid"]
    assert _header_mode(database) == 1  # at rest in rollback mode: the reader opens it without sidecars
    with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as conn:
        assert set(REQUIRED_CORE_TABLES) | {"capture_events"} <= _tables(conn)
        assert conn.execute("SELECT 1 FROM lcm_migration_state WHERE step_name = 'capture_events_v1'").fetchone()
    assert cc.admit(home, hermes_home=hermes) == database  # admitted again, unchanged
    assert not list(home.glob(".lcm.db.creating-*")) and not list(home.glob(".capture-owner*"))


def test_admission_refuses_hermes_homes_and_foreign_stores(tmp_path, home, hermes):
    hermes.mkdir()
    for refused in (hermes, hermes / "profiles" / "work"):
        with pytest.raises(cc.AdmissionError, match="hermes_home"):
            cc.admit(refused, hermes_home=hermes)
    marked = tmp_path / "marked"
    marked.mkdir()
    (marked / "config.yaml").write_text("", encoding="utf-8")
    with pytest.raises(cc.AdmissionError, match="hermes_home"):
        cc.admit(marked, hermes_home=hermes)
    plain = tmp_path / "plain"
    MessageStore(plain / "lcm.db").close()  # an unstamped store is never adopted
    with pytest.raises(cc.AdmissionError, match="store_unstamped"):
        cc.admit(plain, hermes_home=hermes)
    assert "capture_events" not in _tables(sqlite3.connect(plain / "lcm.db"))
    cc.admit(home, hermes_home=hermes)
    copy = tmp_path / "copy"
    shutil.copytree(home, copy)
    with pytest.raises(cc.AdmissionError, match="store_moved"):
        cc.admit(copy, hermes_home=hermes)
    other = tmp_path / "other"
    cc.admit(other, hermes_home=hermes)
    shutil.copy(other / "lcm.db", home / "lcm.db")  # a different stamped store under this owner file
    with pytest.raises(cc.AdmissionError, match="stamp_mismatch"):
        cc.admit(home, hermes_home=hermes)


def test_capture_events_is_never_created_in_a_hermes_store(tmp_path):
    store = MessageStore(tmp_path / "lcm.db")
    try:
        with pytest.raises(sqlite3.OperationalError, match="capture:owner"):
            store.ensure_capture_events()
        assert "capture_events" not in _tables(store.connection)
    finally:
        store.close()


def test_ingest_projects_rows_bindings_ledger_and_cursor(tmp_path, home, hermes):
    path = _transcript(tmp_path, CONVERSATION)
    store = _open(home, hermes)
    try:
        assert cc.drain(store, cc.Lineage("s1", None, path)) == "drained"
        rows = _rows(store)
        assert [(role, tool) for _, role, _, tool in rows] == [
            ("user", None), ("assistant", None), ("tool", "toolu_1"), ("assistant", None)]
        assert "hmm" not in " ".join(str(content) for _, _, content, _ in rows)
        bindings = store.connection.execute("SELECT uid, kind, proof_kind FROM host_uid_bindings "
                                            "WHERE lineage_key = 'claude-code:s1' ORDER BY rowid").fetchall()
        assert bindings == [(uid, "canonical", "transcript_uuid") for uid in ("u1", "a1", "r1", "a2")]
        assert _events(store) == ["new", "new", "new", "raw", "raw", "raw", "new"]
        shared = store.connection.execute("SELECT raw IS NULL, raw_from FROM capture_events "
                                          "WHERE entry_class = 'shared_raw' ORDER BY event_id").fetchall()
        assert shared[0] == (0, None) and shared[1][0] == 1 and shared[1][1] is not None
        cursor = cc.read_cursor(store, cc.Lineage("s1", None, path))
        assert cursor["offset"] == path.stat().st_size and cursor["line_index"] == len(CONVERSATION)
    finally:
        store.close()


def test_reruns_and_rewrites_add_no_rows(tmp_path, home, hermes):
    path = _transcript(tmp_path, CONVERSATION)
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        before = len(_rows(store))
        assert cc.drain(store, lineage) == "drained" and len(_rows(store)) == before
        replacement = path.with_suffix(".tmp")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)  # a rewrite: new inode, same entries
        cc.drain(store, lineage)
        assert len(_rows(store)) == before
        assert _events(store)[len(CONVERSATION):].count("duplicate") == 4
        assert cc.read_cursor(store, lineage)["generation"] == 1
    finally:
        store.close()


def test_a_revision_under_a_known_uuid_supersedes_and_never_overwrites(tmp_path, home, hermes):
    path = _transcript(tmp_path, CONVERSATION[:1])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_entry("user", "u1", "find the secret word, please")) + "\n")
        cc.drain(store, lineage)
        (old, _, first, _), (new, _, second, _) = _rows(store)
        assert first == "find the secret word" and second == "find the secret word, please"
        kinds = store.connection.execute("SELECT kind FROM host_uid_bindings WHERE uid = 'u1' ORDER BY rowid").fetchall()
        assert kinds == [("canonical",), ("version",)]
        assert store.connection.execute("SELECT store_id, related_store_id FROM message_relations "
                                        "WHERE kind = 'supersedes'").fetchall() == [(new, old)]
    finally:
        store.close()


def test_fork_copies_are_the_forks_own_rows_linked_to_the_parent(tmp_path, home, hermes):
    parent = _transcript(tmp_path, CONVERSATION[:3])
    copies = [{**entry, "sessionId": "f1", "promptId": "p2"} for entry in CONVERSATION[:3]]
    fork = _transcript(tmp_path, [*copies, _entry("user", "u9", "and in the fork?", session="f1")], session="f1")
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, parent))
        store.update_metadata_json(cc.FORK_PREFIX + "f1", lambda _current: True)
        cc.drain(store, cc.Lineage("f1", None, fork))
        assert len(_rows(store, "f1")) == 4 and len(_rows(store, "s1")) == 3  # the fork reads back complete
        proofs = store.connection.execute("SELECT proof_kind, COUNT(*) FROM host_uid_bindings "
                                          "WHERE lineage_key = 'claude-code:f1' GROUP BY 1 ORDER BY 1").fetchall()
        assert proofs == [("fork_copy", 3), ("transcript_uuid", 1)]
        assert store.connection.execute("SELECT COUNT(*) FROM message_relations WHERE kind = 'fork_copy_of'"
                                        ).fetchone()[0] == 3
        assert store.session_lineage_parents(["f1"]) == {"f1": "s1"}
    finally:
        store.close()


def test_copies_without_a_fork_start_are_ordinary_rows_and_unlinked(tmp_path, home, hermes):
    parent = _transcript(tmp_path, CONVERSATION[:1])
    other = _transcript(tmp_path, [{**CONVERSATION[0], "sessionId": "s2"}], session="s2")
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, parent))
        cc.drain(store, cc.Lineage("s2", None, other))
        assert len(_rows(store, "s2")) == 1 and store.session_lineage_parents(["s2"]) == {}
    finally:
        store.close()


def test_subagent_rows_have_their_own_session_and_link(tmp_path, home, hermes):
    path = _transcript(tmp_path, [_entry("user", "x1", "subtask", isSidechain=True, agentId="a7")],
                       name="s1/subagents/agent-a7.jsonl")
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", "a7", path))
        assert len(_rows(store, "s1/agent:a7")) == 1 and _rows(store, "s1") == []
        assert store.session_lineage_parents(["s1/agent:a7"]) == {"s1/agent:a7": "s1"}
    finally:
        store.close()


def test_saved_tool_outputs_and_tool_results_files_are_kept(tmp_path, home, hermes):
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    saved = folder / "toolu_1.txt"
    saved.write_text("full output\n" * 5000, encoding="utf-8")
    (folder / "shot.png").write_bytes(b"\x89PNG not text")
    marker = (f"<persisted-output>\nOutput too large (60KB). Full output saved to: {saved}\n\n"
              "Preview (first 2KB):\nfull output\n</persisted-output>")
    path = _transcript(tmp_path, [_entry("user", "r1", [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        assert cc.capture_files(store, lineage) == 1
        [(_, _, content, _)] = _rows(store)
        # The full 60,000 characters, not the preview: ingest protection externalizes them.
        assert "chars=60000" in str(content) and "toolu_1" in str(content)
        assert list((home / "lcm-large-outputs").glob("*toolu_1*"))
        files = store.connection.execute("SELECT uid, disposition, raw IS NOT NULL FROM capture_events "
                                         "WHERE source_kind = 'file' ORDER BY uid").fetchall()
        assert files == [("tool-results/shot.png", "kept", 1), ("tool-results/toolu_1.txt", "held", 0)]
        assert cc.capture_files(store, lineage) == 0
    finally:
        store.close()


def test_a_partly_written_last_line_waits(tmp_path, home, hermes):
    path = _transcript(tmp_path, CONVERSATION[:1])
    line = json.dumps(CONVERSATION[6])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line[:20])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        assert len(_rows(store)) == 1
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line[20:] + "\n")
        cc.drain(store, lineage)
        assert len(_rows(store)) == 2
    finally:
        store.close()


def test_a_set_aside_copy_is_drained_before_the_new_generation(tmp_path, home, hermes):
    path = _transcript(tmp_path, CONVERSATION[:1])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_entry("user", "u2", "written after the last drain")) + "\n")
        os.rename(path, path.with_name(path.name + ".superseded-1"))
        path.write_text(json.dumps(_entry("user", "u3", "the new file")) + "\n", encoding="utf-8")
        cc.drain(store, lineage)
        assert [content for _, _, content, _ in _rows(store)] == [
            "find the secret word", "written after the last drain", "the new file"]
    finally:
        store.close()


def test_forgetting_a_session_deletes_its_ledger_rows(tmp_path, home, hermes):
    path = _transcript(tmp_path, CONVERSATION)
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, path))
        store.delete_session_messages("s1")
        assert store.connection.execute("SELECT COUNT(*) FROM capture_events").fetchone()[0] == 0
    finally:
        store.close()


def _hook():
    spec = importlib.util.spec_from_file_location("lcmx_hook_for_store_tests", PLUGIN / "bin" / "lcmx_hook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(hook, home, path, session="s1", event="Stop", **extra):
    record = {"host": "claude-code", "event": event, "session_id": session, "agent_id": None,
              "transcript_path": str(path), "agent_transcript_path": None, **extra}
    hook.spool(home, record)


def test_worker_drains_the_spool_and_leaves_the_store_readable(tmp_path, home, hermes):
    path = _transcript(tmp_path, CONVERSATION)
    _record(_hook(), home, path)
    report = cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["records"] == 1 and not cc.pending(home)
    assert _header_mode(home / "lcm.db") == 1 and not (home / "lcm.db-wal").exists()
    with sqlite3.connect(f"{(home / 'lcm.db').as_uri()}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4


def test_a_record_spooled_during_a_drain_is_not_lost(tmp_path, home, hermes, monkeypatch):
    """The hook sees the lock held at the worker's final scan: the worker's rescan after
    releasing the lock picks the record up."""
    hook = _hook()
    first = _transcript(tmp_path, CONVERSATION[:1])
    late = _transcript(tmp_path, [_entry("user", "v1", "late session", session="s2")], session="s2")
    _record(hook, home, first)
    real = cc._drain_once
    calls = []

    def drain_then_hook_fires(*args, **kwargs):
        status = real(*args, **kwargs)
        if not calls:
            assert hook._lock_held(home / "capture.lock")  # the hook would log worker_running
            _record(hook, home, late, session="s2", event="SessionEnd")
        calls.append(status)
        return status

    monkeypatch.setattr(cc, "_drain_once", drain_then_hook_fires)
    report = cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert calls == ["done", "done"] and report["records"] == 2 and not cc.pending(home)
    store = _open(home, hermes)
    try:
        assert len(_rows(store, "s2")) == 1
    finally:
        store.close()


def test_a_failing_record_is_set_aside_and_never_loops(tmp_path, home, hermes, monkeypatch):
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))

    def broken(*_args, **_kwargs):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(cc, "drain", broken)
    report = cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert report["failed"] >= 1 and not report["complete"] and not cc.pending(home)
    assert len(list((home / "spool").glob("*.failed"))) == 1
    assert json.loads((home / "logs" / "worker.log").read_text().splitlines()[0])["error"] == "RuntimeError"


def test_a_refused_store_stops_the_worker(tmp_path, hermes):
    marked = tmp_path / "marked"
    marked.mkdir()
    (marked / "state.db").write_text("", encoding="utf-8")
    _record(_hook(), marked, _transcript(tmp_path, CONVERSATION[:1]))
    report = cc.run_worker(marked, hermes_home=hermes, settle_s=0)
    assert report["refused"] == "hermes_home" and not (marked / "lcm.db").exists()


def test_the_session_start_source_survives_later_hooks(tmp_path, home):
    hook = _hook()
    path = _transcript(tmp_path, CONVERSATION[:1])
    _record(hook, home, path, event="SessionStart", source="fork")
    _record(hook, home, path, event="UserPromptSubmit")
    [record] = [json.loads(p.read_text()) for p in (home / "spool").glob("*.json")]
    assert record["event"] == "UserPromptSubmit" and record["start_source"] == "fork"


def test_capture_cli_drains_the_spool(tmp_path, home, hermes):
    """The command the hook starts: exit 0 once every record is drained (PreCompact reads it)."""
    pytest.importorskip("hermes_lcm.mcp_reader", reason="the reader's settings loader arrives with #1216")
    import subprocess
    import sys
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION))
    script = Path(__file__).resolve().parents[1] / "scripts" / "lcm_x_capture.py"
    env = {**os.environ, "HERMES_HOME": str(hermes)}
    done = subprocess.run([sys.executable, str(script), "worker", "--home", str(home), "--deadline", "20"],
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    assert not cc.pending(home)
    assert subprocess.run([sys.executable, str(script), "worker", "--home", "relative/home"],
                          env=env, capture_output=True, timeout=60).returncode == 2
