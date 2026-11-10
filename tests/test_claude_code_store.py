"""Claude Code capture writer and worker: admission (H3), ingest identity, the ledger, the spool
handshake and the store left readable between worker runs. Synthetic transcripts only.

Design: docs/claude-code-adapter.md.
"""
from __future__ import annotations

import ast
import collections
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import time
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import claude_code_store as cc
from hermes_lcm.config import LCMConfig
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
            cc.ensure_capture_events(store)
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


def test_pending_session_is_swept_when_all_later_hooks_are_lost(tmp_path, home, hermes):
    from hermes_lcm import claude_code_audit as ca

    path = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(_hook(), home, path, event="SessionStart")
    report = cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert report["missing"] == 1 and not cc.pending(home)
    _transcript(tmp_path, [CONVERSATION[0], CONVERSATION[-1]])
    other = _transcript(tmp_path, [_entry("user", "v1", "other session", session="s2")], session="s2")
    _record(_hook(), home, other, session="s2")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert [(role, content) for _, role, content, _ in _rows(store)] == [
            ("user", "find the secret word"), ("assistant", "The word is ochre.")]
        assert len(_rows(store, "s2")) == 1
    finally:
        cc.to_rollback_journal(store.connection)
        store.close()
    report = ca.audit(home)
    assert {row["lineage"] for row in report["lineages"]} == {"claude-code:s1", "claude-code:s2"}
    assert report["verified"] and report["failures"]["lost"] == report["failures"]["duplicates"] == 0


def test_pending_commit_survives_a_crash_before_claim_unlink(tmp_path, home, hermes, monkeypatch):
    path = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(_hook(), home, path, event="SessionStart")
    real_unlink = Path.unlink
    interrupted = []

    def crash_at_unlink(claim, *args, **kwargs):
        if claim.suffix == ".claimed":
            with sqlite3.connect(home / "lcm.db") as conn:
                row = conn.execute("SELECT value FROM metadata WHERE key = ?",
                                   ("capture:pending:claude-code:s1",)).fetchone()
            interrupted.append((claim, json.loads(row[0]) if row else None))
            raise RuntimeError("crash before unlink")
        return real_unlink(claim, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", crash_at_unlink)
        with pytest.raises(RuntimeError, match="crash before unlink"):
            cc.run_worker(home, hermes_home=hermes, settle_s=0)
    [(claim, pending)] = interrupted
    assert claim.exists() and pending and pending["path"] == str(path)
    # Lose the claim as well: only the committed registry can recover this lineage.
    claim.unlink()
    _transcript(tmp_path, CONVERSATION)
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["records"] == 0
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert [(role, tool) for _, role, _, tool in _rows(store)] == [
            ("user", None), ("assistant", None), ("tool", "toolu_1"), ("assistant", None)]
        assert _events(store) == ["new", "new", "new", "raw", "raw", "raw", "new"]
    finally:
        store.close()


@pytest.mark.parametrize("agent", [None, "worker"])
def test_pending_lineage_retains_first_seen_and_drains_from_start(tmp_path, home, hermes, monkeypatch, agent):
    path = tmp_path / "projects" / "-work" / "s1.jsonl"
    lineage = cc.Lineage("s1", agent, path)
    key = "capture:pending:" + lineage.key
    store = _open(home, hermes)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(cc.time, "time", lambda: 100.0)
            assert cc.drain(store, lineage) == "missing"
        expected = {"session_id": "s1", "agent_id": agent, "path": str(path), "named": [str(path)], "first_seen": 100.0}
        assert store.read_metadata_json(key) == expected
        assert cc.read_cursor(store, lineage) is None
        calls, real_stat = [], os.stat

        def stat(name, *args, **kwargs):
            if name == path:
                calls.append(name)
            return real_stat(name, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(cc.os, "stat", stat)
            assert cc.drain(store, lineage) == "missing"
        assert calls == [path] and store.read_metadata_json(key) == expected
        assert cc.registered(store) == [lineage]
    finally:
        store.close()
    _transcript(tmp_path, CONVERSATION)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert len(_rows(store, lineage.rows_session)) == 4
        assert cc.read_cursor(store, lineage)["line_index"] == len(CONVERSATION)
        assert store.read_metadata_json(key) is None
        assert cc.registered(store) == [lineage]
    finally:
        store.close()


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
    _record(hook, home, path, event="SessionStart", source="fork", observed_at=1234.5)
    _record(hook, home, path, event="UserPromptSubmit", observed_at=2000.0)
    [record] = [json.loads(p.read_text()) for p in (home / "spool").glob("*.json")]
    assert record["event"] == "UserPromptSubmit" and record["start_source"] == "fork"
    assert record["started_at"] == 1234.5  # when the fork started: its copies were written before


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


def _registered(event):
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    [hook] = [hook for group in hooks[event] for hook in group["hooks"]]
    return hook


def _argv(event):
    """What Claude Code spawns for the event's hook in exec form: the command found on PATH, and
    the arguments with ``${CLAUDE_PLUGIN_ROOT}`` substituted, with no shell."""
    hook = _registered(event)
    return [hook["command"], *[arg.replace("${CLAUDE_PLUGIN_ROOT}", str(PLUGIN)) for arg in hook["args"]]]


@pytest.mark.parametrize("killed", ["before_its_record", "after_its_record"])
@pytest.mark.parametrize("case", ["session_goes_on", "print_mode_exit"])
def test_a_killed_async_post_tool_batch_hook_loses_no_row(tmp_path, home, hermes, case, killed):
    """#2141 runs PostToolBatch async: Claude Code kills an async hook still running when a -p
    session exits; #2350 detaches its writer. Wherever that detached child dies, before
    or after it writes its record, the detached Stop and SessionEnd hooks that follow lead the
    worker to every row, and the store matches a run in which no hook died.

    Each hook runs its entrypoint. A worker holds the lock while the detached end hooks
    run, so they leave their records to it, and the worker run after them is that holder's rescan.
    The async hook never takes the lock, so its death cannot leave them waiting on it. The worker
    script it starts is a stub, and a FIFO in place of the hook log stops it just before it exits,
    unless killed first. A launcher records the child PID, stopping it before spooling for that case."""
    import signal
    import subprocess
    assert _registered("PostToolBatch").get("async") is True
    path = _transcript(tmp_path, CONVERSATION[:3])  # the tool batch; the turn's end comes later
    python = tmp_path / "bin" / "python3"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    package = tmp_path / "package"
    (package / "scripts").mkdir(parents=True)
    for name in ("plugin.yaml", "store.py", "scripts/lcm_x_capture.py"):
        (package / name).write_text("", encoding="utf-8")
    env = {"HOME": str(tmp_path), "PATH": str(python.parent), "LCMX_CC_HOME": str(home),
           "CLAUDE_CONFIG_DIR": str(tmp_path), "CLAUDE_PLUGIN_ROOT": str(PLUGIN), "LCMX_PACKAGE_ROOT": str(package)}
    home.mkdir()
    log = home / "logs" / "hook.log"

    def payload(event):
        return json.dumps({"hook_event_name": event, "session_id": "s1", "transcript_path": str(path)})

    def stored(at):
        store = _open(at, hermes)
        try:
            return [row[1:] for row in _rows(store)]
        finally:
            store.close()

    def end_of_turn(*events):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(entry) + "\n" for entry in CONVERSATION[3:]))
        with open(home / "capture.lock", "a") as held:  # the running worker
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for index, event in enumerate(events, 1):
                done = subprocess.run(_argv(event), input=payload(event),
                                      env=env, capture_output=True, text=True, timeout=60)
                assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
                deadline = time.monotonic() + 30
                while not log.exists() or len(log.read_text(encoding="utf-8").splitlines()) < index:
                    assert time.monotonic() < deadline, "the detached end hooks never logged"
                    time.sleep(0.01)
        outcomes = [json.loads(line)["outcome"] for line in log.read_text(encoding="utf-8").splitlines()]
        assert outcomes[-len(events):] == ["worker_running"] * len(events)  # left to the holder
        assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
        return stored(home)

    child_pid = tmp_path / "child.pid"
    launch = ("import os,runpy,signal; os.register_at_fork(after_in_child=lambda: "
              f"(open({str(child_pid)!r},'w').write(str(os.getpid())), "
              f"signal.raise_signal(signal.SIGSTOP) if {killed == 'before_its_record'} else None)); "
              "runpy.run_path(__import__('sys').argv[-1],run_name='__main__')")
    tool_batch = subprocess.Popen([sys.executable, "-I", "-S", "-c", launch, _argv("PostToolBatch")[-1]], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True)  # waits for its payload until it gets one

    def write_its_record():
        log.parent.mkdir(exist_ok=True)
        log.unlink(missing_ok=True)
        os.mkfifo(log)
        tool_batch.stdin.write(payload("PostToolBatch").encode())
        tool_batch.stdin.close()
        deadline = time.monotonic() + 30
        while not any(json.loads(record.read_text(encoding="utf-8"))["event"] == "PostToolBatch"
                      for record in (home / "spool").glob("*.json")):
            assert time.monotonic() < deadline, "the hook never wrote its record"
            time.sleep(0.01)

    def kill():
        if killed == "before_its_record":
            tool_batch.stdin.write(payload("PostToolBatch").encode())
            tool_batch.stdin.close()
        deadline = time.monotonic() + 30
        while not child_pid.exists() or not child_pid.read_text():
            assert time.monotonic() < deadline, "the hook never forked its writer"
            time.sleep(0.01)
        os.kill(int(child_pid.read_text()), signal.SIGKILL)
        assert tool_batch.wait(timeout=60) == 0  # only its detached writer was killed
        tool_batch.stdin.close()
        if log.is_fifo():
            log.unlink()

    if case == "session_goes_on":  # the hook dies, then the turn ends
        if killed == "after_its_record":
            write_its_record()
        kill()
        captured = end_of_turn("Stop")
    else:  # the turn ends, and the exit kills the hook, which may write its record last
        captured = end_of_turn("Stop", "SessionEnd")
        if killed == "after_its_record":
            write_its_record()
        kill()
    (home / "spool").mkdir(exist_ok=True)
    (home / "spool" / ".cut.1.tmp").write_text('{"event": "PostTool', encoding="utf-8")  # a write cut short
    waiting = cc.pending(home)
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]  # any session's next run
    assert not cc.pending(home)
    assert waiting is (case == "print_mode_exit" and killed == "after_its_record")  # only a late record waits

    control = tmp_path / "control"
    _record(_hook(), control, path)
    assert cc.run_worker(control, hermes_home=hermes, settle_s=0)["complete"]
    assert captured == stored(home) == stored(control) and len(captured) == 4


def test_a_worker_started_while_another_holds_the_lock_exits_before_loading_the_store(home):
    """The async PostToolBatch hook starts a worker on every tool batch, so a worker that finds
    another holding the lock exits before any store code loads, with the exit codes it had: 0
    detached, 4 bounded (the holder takes the record)."""
    import subprocess
    script = Path(__file__).resolve().parents[1] / "scripts" / "lcm_x_capture.py"
    home.mkdir()
    with open(home / "capture.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for extra, code in (([], 0), (["--deadline", "10"], 4)):
            done = subprocess.run([sys.executable, "-X", "importtime", str(script), "worker", "--home", str(home),
                                   *extra], capture_output=True, text=True, timeout=60)
            imported = {line.rsplit("|", 1)[-1].strip() for line in done.stderr.splitlines()
                        if line.startswith("import time:")}
            assert done.returncode == code and "argparse" in imported  # the import log is complete
            assert not any(name.startswith("hermes_lcm") for name in imported)


def _embedding_slice(state, outcome="complete", *, vectors=0, again=False, delay=0.0):
    state.update(vectors_written=vectors, delay=delay, **{"continue": again})
    return dict(state, outcome=outcome)


def test_worker_embedding_rotates_past_empty_summaries_and_leaves_rollback_mode(tmp_path, home, hermes, monkeypatch):
    hermes.mkdir()
    for name in ("state.db", "lcm.db"):
        (hermes / name).write_bytes(b"synthetic Hermes bytes: " + name.encode())
    snapshot = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in hermes.iterdir()}
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    config = LCMConfig(embeddings_enabled=True, embedding_provider=" Fast-Embed ",
                       database_path=str(hermes / "lcm.db"))
    calls = []

    def sliced(database, passed_config, *, slice_state):
        assert database == home / "lcm.db" and passed_config is config
        assert Path(f"{database}-wal").exists() and Path(f"{database}-shm").exists()
        stage = slice_state.get("stage", 0)
        calls.append(stage)
        with closing(sqlite3.connect(database)) as conn:
            assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
            conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES('fake_embedding_stage', ?)",
                         (str(stage),))
            conn.commit()
        slice_state["stage"] = stage + 1
        return _embedding_slice(slice_state, "summaries=complete:0" if stage == 0 else "chunks=complete:2 complete",
                                vectors=0 if stage == 0 else 2, again=stage == 0)

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", sliced)
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert calls == [0, 1] and report["vectors"] == 2 and report["complete"]
    assert report["embedding"] == "chunks=complete complete"
    assert _header_mode(home / "lcm.db") == 1 and not cc._wal_without_sidecars(home / "lcm.db")
    assert not (home / "lcm.db-wal").exists() and not (home / "lcm.db-shm").exists()
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in hermes.iterdir()} == snapshot
    assert not (home / "logs" / "worker.log").exists()


def test_the_real_embedding_pass_leaves_the_store_readable(tmp_path, home, hermes, monkeypatch):
    """The catch-up through the real pass, not a stub: with FastEmbed configured but not
    installed, the pass reports its token, downloads nothing, and the store ends in
    rollback-journal mode with no sidecars, for the reader."""
    hermes.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes))
    monkeypatch.setitem(sys.modules, "fastembed", None)  # absent even where a test host installed it
    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed",
                       database_path=str(hermes / "lcm.db"))
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION))
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["drained"] == 1
    assert report["embedding"] == "auto_register_unavailable" and report["vectors"] == 0
    records = [json.loads(line) for line in (home / "logs" / "worker.log").read_text().splitlines()]
    assert records == [{"t": records[0]["t"], "stage": "embed", "error": "auto_register_unavailable"}]
    assert _header_mode(home / "lcm.db") == 1 and not cc._wal_without_sidecars(home / "lcm.db")
    assert not (home / "lcm.db-wal").exists() and not (home / "lcm.db-shm").exists()
    assert list(hermes.iterdir()) == []


def test_embedding_exception_collects_a_failed_constructor_before_switch(tmp_path, home, hermes, monkeypatch):
    class BrokenVectorStore:
        def __init__(self, database):
            self.connection = sqlite3.connect(database)
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.cycle = self
            raise RuntimeError("synthetic /private/model/cache failure")

    def broken(database, config, *, slice_state):
        BrokenVectorStore(database)

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", broken)
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))
    enabled = gc.isenabled()
    gc.disable()
    try:
        report = cc.run_worker(home, config=LCMConfig(), hermes_home=hermes, settle_s=0)
        assert report["complete"] and report["embedding"] == "error" and report["vectors"] == 0
        assert _header_mode(home / "lcm.db") == 1 and not cc._wal_without_sidecars(home / "lcm.db")
        assert not (home / "lcm.db-wal").exists() and not (home / "lcm.db-shm").exists()
        [logged] = [json.loads(line) for line in (home / "logs" / "worker.log").read_text().splitlines()]
        assert {key: value for key, value in logged.items() if key != "t"} == {"stage": "embed", "error": "error"}
    finally:
        gc.collect()
        if enabled:
            gc.enable()


@pytest.mark.parametrize("deadline,enabled,provider,expected", [
    (20, True, "fastembed", "skipped_bounded"),
    (20, False, "voyage", "skipped_bounded"),
    (None, False, "voyage", "skipped_disabled"),
    (None, True, "voyage", "skipped_provider"),
    (None, True, "ollama", "skipped_provider"),
])
def test_embedding_guards_never_call_the_pass(tmp_path, home, hermes, monkeypatch,
                                            deadline, enabled, provider, expected):
    monkeypatch.setattr(cc, "run_incremental_embedding_pass", lambda *a, **k: pytest.fail("embedding called"))
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))
    config = LCMConfig(embeddings_enabled=enabled, embedding_provider=provider)
    report = cc.run_worker(home, deadline_s=deadline, config=config, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["embedding"] == expected and report["vectors"] == 0


def test_idle_worker_does_not_load_embeddings(tmp_path, home, hermes, monkeypatch):
    path = _transcript(tmp_path, CONVERSATION[:1])
    _record(_hook(), home, path)
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    monkeypatch.setattr(cc, "run_incremental_embedding_pass", lambda *a, **k: pytest.fail("idle embedding"))
    _record(_hook(), home, path)
    report = cc.run_worker(home, config=LCMConfig(), hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["embedding"] == "skipped_idle" and report["vectors"] == 0


@pytest.fixture
def fake_fastembed(monkeypatch):
    """The real pass and FastEmbed provider over a fake model in an empty model cache; each load
    is recorded as (start, end), runs ``during`` first, and raises while ``fail`` is set."""
    from hermes_lcm import embedding_provider as provider_mod

    fake = SimpleNamespace(models={}, loads=[], during=[], fail=False)

    class Model:
        def query_embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

        embed = query_embed

    def construct(self, *, allow_download):
        assert not allow_download
        started = time.monotonic()
        for action in fake.during:
            action()
        if fake.fail:
            raise provider_mod.ProviderNotWarmedUp("the fake model is not cached")
        time.sleep(0.05)
        fake.loads.append((started, time.monotonic()))
        return Model()

    monkeypatch.setattr(provider_mod, "_FASTEMBED_MODELS", fake.models)
    monkeypatch.setattr(provider_mod.FastembedProvider, "_construct", construct)
    monkeypatch.setattr(provider_mod, "probe_provider_availability",
                        lambda _config: {"available": True, "detail": "fake model"})
    return fake


def _fresh_worker_after(tmp_path, home, hermes, fake, config, entry):
    """A store with its profiles registered and its chunks embedded; then ``entry`` is appended,
    its record spooled, and the model cache emptied, as in a fresh worker process."""
    hook, path = _hook(), _transcript(tmp_path, CONVERSATION[:2])
    _record(hook, home, path)
    assert cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)["vectors"] > 0
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    _record(hook, home, path)
    fake.models.clear()
    fake.loads.clear()
    return hook


def test_a_fresh_worker_loads_the_chunk_model_before_its_first_slice(tmp_path, home, hermes, monkeypatch,
                                                                    fake_fastembed):
    """The load comes before the first slice and outside the budget, so a slice is embedding only,
    and a record spooled during the load is captured before the first batch."""
    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed")
    hook = _fresh_worker_after(tmp_path, home, hermes, fake_fastembed, config, CONVERSATION[-1])
    late = _transcript(tmp_path, [_entry("user", "v1", "late", session="s2")], session="s2")
    fake_fastembed.during.append(lambda: _record(hook, home, late, session="s2"))
    slices, real_pass = [], cc.run_incremental_embedding_pass

    def observed(database, passed_config, *, slice_state):
        with closing(sqlite3.connect(database)) as conn:
            late_rows = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = 's2'").fetchone()[0]
        slices.append((time.monotonic(), bool(fake_fastembed.models), late_rows,
                       slice_state["capture_budget_end"], slice_state.get("model_load_s")))
        return real_pass(database, passed_config, slice_state=slice_state)

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", observed)
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["records"] == 2 and report["vectors"] == 2
    [(load_start, load_end)] = fake_fastembed.loads
    assert slices and all(loaded and late_rows == 1 for _, loaded, late_rows, _, _ in slices)
    assert load_end <= slices[0][0]
    assert all(load_end - load_start <= load_s < load_end - load_start + 1.0 for *_, load_s in slices)


def test_a_round_after_the_budget_ran_out_loads_no_model(tmp_path, home, hermes, monkeypatch, fake_fastembed):
    """No slice can run once the budget has run out, so a later drain round neither searches for
    a chunk nor loads the model."""
    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed")
    hook = _fresh_worker_after(tmp_path, home, hermes, fake_fastembed, config, CONVERSATION[2])  # a tool result
    late = _transcript(tmp_path, [_entry("user", "v1", "late", session="s2")], session="s2")
    clock, probes = [100.0], []
    real_pass, real_load = cc.run_incremental_embedding_pass, cc._load_chunk_model

    def observed(database, passed_config, *, slice_state):
        if not cc.pending(home):
            _record(hook, home, late, session="s2")  # the next round has a chunk to embed
            clock[0] += cc.EMBED_BUDGET_S + 1.0  # and starts after the budget ran out
        return real_pass(database, passed_config, slice_state=slice_state)

    def load(database, passed_config):
        probes.append(clock[0])
        return real_load(database, passed_config)

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", observed)
    monkeypatch.setattr(cc, "_load_chunk_model", load)
    monkeypatch.setattr(cc, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time,
                                                    sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["records"] == 2 and report["embedding"] == "budget"
    assert probes == [100.0] and fake_fastembed.loads == [] and not fake_fastembed.models
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = 's2'").fetchone()[0] == 1


def test_the_budget_covers_the_discovery_but_not_the_model_load(tmp_path, home, hermes, monkeypatch,
                                                                fake_fastembed):
    from hermes_lcm import command

    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed")
    _fresh_worker_after(tmp_path, home, hermes, fake_fastembed, config, CONVERSATION[-1])
    clock, starts, real_discovery = [100.0], [], command._chunk_pending_rows

    def discovery(*args, **kwargs):
        clock[0] += 1.0  # a slow discovery
        return real_discovery(*args, **kwargs)

    def sliced(database, passed_config, *, slice_state):
        starts.append(clock[0])
        clock[0] += 1.0
        return _embedding_slice(slice_state, "chunks=complete:1", vectors=1, again=True)

    fake_fastembed.during.append(lambda: clock.__setitem__(0, clock[0] + 3.0))  # a slow load
    monkeypatch.setattr(command, "_chunk_pending_rows", discovery)
    monkeypatch.setattr(cc, "run_incremental_embedding_pass", sliced)
    monkeypatch.setattr(cc, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time, sleep=time.sleep))
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert report["embedding"] == "budget" and len(fake_fastembed.loads) == 1
    assert starts == [104.0 + n for n in range(7)]  # 8 s from 100.0 less the 1 s discovery; the 3 s load is extra


def test_a_worker_with_no_chunk_to_embed_loads_no_model(tmp_path, home, hermes, fake_fastembed):
    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed")
    _fresh_worker_after(tmp_path, home, hermes, fake_fastembed, config, CONVERSATION[2])  # a tool result
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["vectors"] == 0 and report["embedding"] == "chunks=complete complete"
    assert fake_fastembed.loads == [] and not fake_fastembed.models


def test_a_model_that_cannot_load_leaves_the_outcome_to_the_pass(tmp_path, home, hermes, monkeypatch,
                                                                fake_fastembed):
    config = LCMConfig(embeddings_enabled=True, embedding_provider="fastembed")
    _fresh_worker_after(tmp_path, home, hermes, fake_fastembed, config, CONVERSATION[-1])
    fake_fastembed.fail = True
    states, real_pass = [], cc.run_incremental_embedding_pass

    def observed(database, passed_config, *, slice_state):
        states.append(slice_state["model_load_s"])
        return real_pass(database, passed_config, slice_state=slice_state)

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", observed)
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["vectors"] == 0 and states and set(states) == {None}
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE content = 'The word is ochre.'").fetchone()[0] == 1


@pytest.mark.parametrize("during_sleep", [False, True])
def test_embedding_yields_to_spool_and_resumes_state_in_the_same_run(tmp_path, home, hermes, monkeypatch, during_sleep):
    hook = _hook()
    late = _transcript(tmp_path, [_entry("user", "v1", "late", session="s2")], session="s2")
    _record(hook, home, _transcript(tmp_path, CONVERSATION[:1]))
    calls, reports, sleeps = [], [], []

    def sliced(database, config, *, slice_state):
        stage = slice_state.get("stage", 0)
        calls.append((id(slice_state), stage, slice_state["capture_budget_end"]))
        slice_state["stage"] = stage + 1
        if stage == 0 and not during_sleep:
            _record(hook, home, late, session="s2")
        return _embedding_slice(slice_state, "chunks=complete:1", vectors=1, again=stage == 0, delay=0.2)

    real_drain = cc._drain_once

    def observed(*args):
        status = real_drain(*args)
        reports.append((args[1]["embedding"], len(calls)))
        return status

    def sleep(delay):
        sleeps.append(delay)
        _record(hook, home, late, session="s2")

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", sliced)
    monkeypatch.setattr(cc, "_drain_once", observed)
    if during_sleep:
        monkeypatch.setattr(cc, "time", SimpleNamespace(monotonic=time.monotonic, time=time.time, sleep=sleep))
    report = cc.run_worker(home, config=LCMConfig(), hermes_home=hermes, settle_s=0)
    assert reports[0] == ("yielded", 1) and [call[1] for call in calls] == [0, 1]
    assert calls[0][0] == calls[1][0] and calls[0][2] == calls[1][2]
    assert report["complete"] and report["records"] == 2 and report["vectors"] == 2 and not cc.pending(home)
    assert not during_sleep or (sleeps and all(0 < delay <= 0.05 for delay in sleeps))
    with closing(sqlite3.connect(home / "lcm.db")) as conn:
        assert conn.execute("SELECT content FROM messages WHERE session_id = 's2'").fetchall() == [("late",)]


@pytest.mark.parametrize("outcome,explicit,expected,logged", [
    ("error", True, "error", True),
    ("summaries=error:0", True, "summaries=error", True),
    ("chunks=failed:2", True, "chunks=failed", True),
    ("chunks=refused", True, "chunks=refused", True),
    ("provider_unavailable", True, "provider_unavailable", True),
    ("auto_register_unavailable", True, "auto_register_unavailable", True),
    ("auto_register_unavailable", False, "auto_register_unavailable", False),
    ("profile_held:voyage", True, "profile_held", False),
    ("lease_held", True, "lease_held", False),
    ("chunks=complete:2 complete", True, "chunks=complete complete", False),
])
def test_embedding_logs_only_error_tokens(tmp_path, home, hermes, monkeypatch, outcome, explicit, expected, logged):
    def sliced(database, config, *, slice_state):
        return dict(_embedding_slice(slice_state, outcome), failure="synthetic /private/model/cache failure")

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", sliced)
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))
    config = LCMConfig(embeddings_enabled_explicit=explicit)
    report = cc.run_worker(home, config=config, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["embedding"] == expected
    log = home / "logs" / "worker.log"
    if logged:
        [entry] = [json.loads(line) for line in log.read_text().splitlines()]
        assert {key: value for key, value in entry.items() if key != "t"} == {"stage": "embed", "error": expected}
    else:
        assert not log.exists()


@pytest.mark.parametrize("delay", [0.1, 20.0])
def test_embedding_budget_counts_slices_and_sleep(tmp_path, home, hermes, monkeypatch, delay):
    clock, calls, sleeps = [100.0], [], []

    def sliced(database, config, *, slice_state):
        calls.append(clock[0])
        clock[0] += 1.0
        return _embedding_slice(slice_state, "chunks=complete:1", vectors=1, again=True, delay=delay)

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(cc, "run_incremental_embedding_pass", sliced)
    monkeypatch.setattr(cc, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time, sleep=sleep))
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))
    report = cc.run_worker(home, config=LCMConfig(), hermes_home=hermes, settle_s=0)
    assert report["embedding"] == "budget" and report["complete"] and report["vectors"] == len(calls)
    assert all(start < 108.0 for start in calls) and clock[0] <= 109.0
    assert (len(calls) > 1 and sleeps) if delay == 0.1 else (len(calls) == 1 and not sleeps)
    assert all(0 < seconds <= 0.05 for seconds in sleeps)


@pytest.mark.parametrize("guard", ["pending", "transaction"])
def test_embedding_skips_pending_records_and_open_transactions(tmp_path, home, hermes, monkeypatch, guard):
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))
    real_registered = cc.registered
    observed = []

    def registered(store):
        found = real_registered(store)
        if not observed:
            observed.append(store)
            if guard == "pending":
                _record(_hook(), home, _transcript(tmp_path, [], session="s2"), session="s2")
            else:
                store.connection.execute("BEGIN")
        return found

    real_drain = cc._drain_once
    tokens = []

    def drain(*args):
        status = real_drain(*args)
        tokens.append(args[1]["embedding"])
        return status

    monkeypatch.setattr(cc, "registered", registered)
    monkeypatch.setattr(cc, "_drain_once", drain)
    monkeypatch.setattr(cc, "run_incremental_embedding_pass", lambda *a, **k: pytest.fail("guarded embedding"))
    report = cc.run_worker(home, config=LCMConfig(), hermes_home=hermes, settle_s=0)
    assert tokens[0] == "skipped_" + guard and report["vectors"] == 0


# -- round 2 review regressions --------------------------------------------------------

FORK_START = datetime(2026, 11, 1, 1, tzinfo=timezone.utc).timestamp()


def _fork(tmp_path, store, fork, entries, own, started=FORK_START):
    """A fork transcript: copies of ``entries`` then its own entries, flagged as a fork."""
    copies = [{**entry, "sessionId": fork, "promptId": "p-" + fork} for entry in entries]
    path = _transcript(tmp_path, [*copies, *own], session=fork)
    store.update_metadata_json(cc.FORK_PREFIX + fork, lambda _current: {"started_at": started})
    return cc.Lineage(fork, None, path)


def _relations(store, kind):
    return store.connection.execute("SELECT store_id, related_store_id FROM message_relations WHERE kind = ? "
                                    "ORDER BY 1, 2", (kind,)).fetchall()


def test_forgetting_a_session_keeps_another_sessions_shared_raw_bytes(tmp_path, home, hermes):
    """REGRESSION: identical request blobs are kept once per session, so forgetting one session
    never takes the bytes another session's ledger rows point to."""
    blob = {"type": "api-request-blob", "hash": "h1", "blob": {"system": "big prompt"}}
    one = _transcript(tmp_path, [blob], session="s1")
    two = _transcript(tmp_path, [blob, blob], session="s2")
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, one))
        cc.drain(store, cc.Lineage("s2", None, two))
        store.delete_session_messages("s1")
        (first, kept, _), (_, second_kept, points_to) = store.connection.execute(
            "SELECT event_id, raw IS NOT NULL, raw_from FROM capture_events WHERE session_id = 's2' "
            "ORDER BY event_id").fetchall()
        assert kept == 1 and second_kept == 0 and points_to == first
    finally:
        store.close()


def test_a_forgotten_session_is_never_captured_again(tmp_path, home, hermes):
    """REGRESSION: after a session is deleted, its transcript (even rewritten) and its
    tool-results files are never read again, so forgotten content cannot come back."""
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    (folder / "shot.png").write_bytes(b"\x89PNG not text")
    path = _transcript(tmp_path, CONVERSATION)
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        assert cc.capture_files(store, lineage) == 1
        store.delete_session_messages("s1")
        replacement = path.with_suffix(".rewrite")
        replacement.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        os.replace(replacement, path)  # a new inode: without the mark this is a new generation
        assert cc.drain(store, lineage) == "forgotten"
        assert cc.capture_files(store, lineage) == 0
        assert store.connection.execute("SELECT COUNT(*) FROM capture_events").fetchone()[0] == 0
        assert _rows(store) == [] and store.read_metadata_json("capture:forgotten:s1")
    finally:
        store.close()


def test_a_store_left_in_wal_without_sidecars_is_switched_after_close(home, hermes):
    """REGRESSION: when a reader blocks the switch and closes first, the writer's close leaves a
    WAL header without sidecars; the stamp still reads (immutable, no file created) and the
    worker switches the store again, still under its lock."""
    database = cc.admit(home, hermes_home=hermes)
    cc.open_store(database, home).close()  # WAL while open; the last close removes the sidecars
    assert _header_mode(database) == 2 and not Path(f"{database}-shm").exists()
    assert cc.read_stamp(database)["host"] == "claude-code"
    assert not Path(f"{database}-shm").exists() and not Path(f"{database}-wal").exists()
    assert cc.leave_readable(database) and _header_mode(database) == 1


def test_a_store_the_switch_fails_on_is_reported_unreadable_and_left_as_it_was(home, hermes, monkeypatch):
    """REGRESSION: when the switch fails, the store is reported as not readable (the run is then
    incomplete) and stays a WAL header without sidecars, for the next run to switch."""
    database = cc.admit(home, hermes_home=hermes)
    cc.open_store(database, home).close()

    def failing(conn, attempts=5):
        conn.execute("SELECT COUNT(*) FROM metadata").fetchone()  # the switching connection reads, then fails
        return False

    monkeypatch.setattr(cc, "to_rollback_journal", failing)
    assert cc.leave_readable(database) is False
    assert _header_mode(database) == 2
    assert not Path(f"{database}-wal").exists() and not Path(f"{database}-shm").exists()


def test_tool_results_files_of_any_size_are_kept_whole_and_read_once(tmp_path, home, hermes, monkeypatch):
    """REGRESSION: a file above the chunk size is kept as rows that rebuild its bytes (no size
    limit drops it); an unchanged file is not read again; a touched one is not stored twice; a
    changed one is kept again; the deadline stops the sweep."""
    monkeypatch.setattr(cc, "FILE_CHUNK_BYTES", 10)
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    data = b"0123456789" * 3 + b"end"  # identical chunks must not collapse
    (folder / "big.bin").write_bytes(data)
    lineage = cc.Lineage("s1", None, _transcript(tmp_path, CONVERSATION[:1]))
    store = _open(home, hermes)

    def chunks():
        return store.connection.execute("SELECT content_digest, position, raw FROM capture_events "
                                        "WHERE uid = 'tool-results/big.bin' ORDER BY event_id").fetchall()
    try:
        assert cc.capture_files(store, lineage) == 1
        assert [row[1] for row in chunks()] == [0, 1, 2, 3]
        assert b"".join(zlib.decompress(row[2]) for row in chunks()) == data
        assert {row[0] for row in chunks()} == {hashlib.sha256(data).hexdigest()}
        keep = cc._keep_file
        monkeypatch.setattr(cc, "_keep_file", lambda *args: pytest.fail("an unchanged file was read again"))
        assert cc.capture_files(store, lineage) == 0
        monkeypatch.setattr(cc, "_keep_file", keep)
        os.utime(folder / "big.bin", ns=(1, 1))
        assert cc.capture_files(store, lineage) == 0 and len(chunks()) == 4
        (folder / "big.bin").write_bytes(data + b"!")
        assert cc.capture_files(store, lineage) == 1 and len(chunks()) == 8
        (folder / "more.bin").write_bytes(b"x")
        assert cc.capture_files(store, lineage, deadline=time.monotonic() - 1) is None
    finally:
        store.close()


def test_a_symlinked_tool_results_folder_is_never_followed(tmp_path, home, hermes):
    """REGRESSION: a tool-results folder that links out of the session folder is neither captured
    nor used to recover a saved output; the tool row keeps its preview."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not this session's\n" * 4000, encoding="utf-8")
    session = tmp_path / "projects" / "-work" / "s1"
    session.mkdir(parents=True)
    (session / "tool-results").symlink_to(outside, target_is_directory=True)
    marker = (f"<persisted-output>\nOutput too large (70KB). Full output saved to: "
              f"{session / 'tool-results' / 'secret.txt'}\n\nPreview (first 2KB):\nnot this\n</persisted-output>")
    path = _transcript(tmp_path, [_entry("user", "r1", [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        assert cc.capture_files(store, lineage) == 0
        [(_, _, content, _)] = _rows(store)
        assert content == marker
        notes = store.connection.execute("SELECT notes FROM capture_events WHERE uid = 'r1'").fetchone()[0]
        assert json.loads(notes) == ["persisted_output_outside_session"]
    finally:
        store.close()


def test_a_revision_that_changes_the_row_count_still_supersedes(tmp_path, home, hermes):
    """REGRESSION: one row becoming two, then two becoming one, links every new row to a previous
    row and every previous row to a new one."""
    result = {"type": "tool_result", "tool_use_id": "t1", "content": "first"}
    versions = [_entry("user", "r1", [result]),
                _entry("user", "r1", [result, {"type": "text", "text": "and a note"}]),
                _entry("user", "r1", [{**result, "content": "third"}])]
    path = _transcript(tmp_path, versions[:1])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        for entry in versions[1:]:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            cc.drain(store, lineage)
        v0, v1a, v1b, v2 = (row[0] for row in _rows(store))
        assert _relations(store, "supersedes") == sorted([(v1a, v0), (v1b, v0), (v2, v1a), (v2, v1b)])
    finally:
        store.close()


def test_sibling_forks_both_link_to_the_session_that_wrote_their_copies(tmp_path, home, hermes):
    """REGRESSION: two forks of one session copy the same entries; every copy points at the
    parent's row and each fork's parent is that session, never its sibling."""
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, _transcript(tmp_path, CONVERSATION[:3])))
        for fork in ("f1", "f2"):
            own = [_entry("user", "u-" + fork, "in the fork", session=fork, timestamp="2026-11-01T02:00:00.000Z")]
            cc.drain(store, _fork(tmp_path, store, fork, CONVERSATION[:3], own))
        assert store.session_lineage_parents(["f1", "f2"]) == {"f1": "s1", "f2": "s1"}
        parent_rows = {row[0] for row in _rows(store, "s1")}
        copies = _relations(store, "fork_copy_of")
        assert len(copies) == 6 and {related for _, related in copies} == parent_rows
    finally:
        store.close()


def test_a_forks_own_entries_are_never_copies_of_a_later_fork(tmp_path, home, hermes):
    """REGRESSION: f2 (forked from f1) is captured before f1's own entry is. That entry stays
    f1's own row (written after f1 started), never a copy of f2's. f2's copy of it has no known
    writer yet, so it gets no ``fork_copy_of`` link rather than a wrong one, and f2's parent is
    the nearest captured writer, s1 (the design's ancestor rule), not f1."""
    first = CONVERSATION[0]
    own_f1 = _entry("user", "u-f1", "f1 writes this", session="f1", timestamp="2026-11-01T01:30:00.000Z")
    own_f2 = _entry("user", "u-f2", "f2 writes this", session="f2", timestamp="2026-11-01T02:30:00.000Z")
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, _transcript(tmp_path, [first])))
        later = datetime(2026, 11, 1, 2, tzinfo=timezone.utc).timestamp()
        cc.drain(store, _fork(tmp_path, store, "f2", [first, {**own_f1, "sessionId": "f2"}], [own_f2], started=later))
        cc.drain(store, _fork(tmp_path, store, "f1", [first], [own_f1]))
        proofs = dict(store.connection.execute("SELECT uid, proof_kind FROM host_uid_bindings "
                                               "WHERE lineage_key = 'claude-code:f1'").fetchall())
        assert proofs == {"u1": "fork_copy", "u-f1": "transcript_uuid"}
        f2_rows = {row[0] for row in _rows(store, "f2")}
        assert not any(related in f2_rows for _, related in _relations(store, "fork_copy_of"))
        assert store.session_lineage_parents(["f1", "f2"]) == {"f1": "s1", "f2": "s1"}
    finally:
        store.close()


def test_a_fork_flag_from_any_claimed_record_applies_before_the_first_drain(tmp_path, home, hermes):
    """REGRESSION: a fork's subagent record can be claimed before its SessionStart(fork) record;
    the fork's copies are still copies."""
    hook = _hook()
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, _transcript(tmp_path, CONVERSATION[:1])))
    finally:
        store.close()
    copies = [{**CONVERSATION[0], "sessionId": "f1", "promptId": "p2"}]
    own = _entry("user", "u9", "own", session="f1", timestamp="2026-11-01T02:00:00.000Z")
    fork = _transcript(tmp_path, [*copies, own], session="f1")

    def key(agent):
        return hashlib.sha256("\0".join(["claude-code", "f1", agent]).encode()).hexdigest()[:32]
    agent = next(f"a{n}" for n in range(1000) if key(f"a{n}") < key(""))  # claimed first
    agent_path = _transcript(tmp_path, [_entry("user", "x1", "subtask", session="f1", isSidechain=True,
                                                timestamp="2026-11-01T02:00:01.000Z")],
                             name=f"f1/subagents/agent-{agent}.jsonl")
    _record(hook, home, fork, session="f1", event="SessionStart", source="fork", observed_at=FORK_START)
    _record(hook, home, fork, session="f1", event="SubagentStop", agent_id=agent,
            agent_transcript_path=str(agent_path))
    report = cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert report["complete"] and report["records"] == 2
    store = _open(home, hermes)
    try:
        proofs = dict(store.connection.execute("SELECT uid, proof_kind FROM host_uid_bindings "
                                               "WHERE lineage_key = 'claude-code:f1'").fetchall())
        assert proofs == {"u1": "fork_copy", "u9": "transcript_uuid"}
    finally:
        store.close()


def test_a_projected_entry_without_uuid_is_kept_raw_and_noted(tmp_path, home, hermes):
    """REGRESSION: a user or assistant entry without a uuid has no identity to deduplicate by; its
    line is kept raw and noted for the audit, never dropped silently."""
    path = _transcript(tmp_path, [{**_entry("user", "u1", "no id"), "uuid": None}])
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, path))
        assert _rows(store) == []
        assert store.connection.execute("SELECT disposition, notes, raw IS NOT NULL FROM capture_events"
                                        ).fetchall() == [("raw", '["projected_entry_without_uuid"]', 1)]
    finally:
        store.close()


def test_local_gc_is_remembered_for_the_session(tmp_path, home, hermes):
    """REGRESSION: a hook that saw Claude Code's local GC on marks the session for the audit,
    and a later hook with it off does not clear the mark."""
    hook = _hook()
    path = _transcript(tmp_path, CONVERSATION[:1])
    for local_gc in (True, False):
        _record(hook, home, path, local_gc=local_gc)
        assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    store = _open(home, hermes)
    try:
        assert store.read_metadata_json(cc.LOCAL_GC_PREFIX + "s1") is True
    finally:
        store.close()


def test_local_gc_survives_a_record_that_replaces_its_pointer_before_a_drain(tmp_path, home, hermes):
    """#2503: a hook that saw local GC on, then a later hook of the same session with it off, both spooled before any
    worker run: the replaced pointer must not drop the flag, or the audit would carry a lossless claim it should not."""
    hook = _hook()
    path = _transcript(tmp_path, CONVERSATION[:1])
    _record(hook, home, path, event="UserPromptSubmit", local_gc=True)
    _record(hook, home, path, event="Stop", local_gc=False)
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    store = _open(home, hermes)
    try:
        assert store.read_metadata_json(cc.LOCAL_GC_PREFIX + "s1") is True
    finally:
        store.close()


@pytest.mark.parametrize("named", ["~nosuchuser-lcmx/x.txt", "/tmp/a\x00b"], ids=["unknown-user", "nul"])
def test_a_saved_output_path_that_cannot_resolve_keeps_the_preview(tmp_path, home, hermes, named):
    """REGRESSION: a marker naming an unknown ~user or holding a NUL keeps its preview, and the
    transcript drains past it (it used to raise on every run)."""
    (tmp_path / "projects" / "-work" / "s1" / "tool-results").mkdir(parents=True)
    marker = f"<persisted-output>\nOutput too large (1KB). Full output saved to: {named}\n\nPreview:\nx\n</persisted-output>"
    path = _transcript(tmp_path, [_entry("user", "r1", [{"type": "tool_result", "tool_use_id": "t1", "content": marker}]),
                                  _entry("user", "u2", "next")])
    store = _open(home, hermes)
    try:
        assert cc.drain(store, cc.Lineage("s1", None, path)) == "drained"
        assert [row[2] for row in _rows(store)] == [marker, "next"]
    finally:
        store.close()


def test_a_line_whose_projection_fails_stays_raw_and_the_rest_drains(tmp_path, home, hermes, monkeypatch):
    """REGRESSION: an unexpected projection error keeps that line raw and noted; later lines drain."""
    project = cc.project_entry

    def failing(entry, **kwargs):
        if entry.get("uuid") == "u1":
            raise TypeError("synthetic")
        return project(entry, **kwargs)
    monkeypatch.setattr(cc, "project_entry", failing)
    path = _transcript(tmp_path, [_entry("user", "u1", "breaks"), _entry("user", "u2", "fine")])
    store = _open(home, hermes)
    try:
        assert cc.drain(store, cc.Lineage("s1", None, path)) == "drained"
        assert [row[2] for row in _rows(store)] == ["fine"]
        assert store.connection.execute("SELECT notes FROM capture_events WHERE uid = 'u1'").fetchone()[0] == (
            '["projection_error:TypeError"]')
    finally:
        store.close()


def test_a_lineage_whose_first_batch_fails_is_retried_by_the_sweep(tmp_path, home, hermes, monkeypatch):
    """REGRESSION: the cursor is registered before the first batch, so when that batch fails the
    sweep retries the transcript (it used to wait for the session's next hook)."""
    path = _transcript(tmp_path, CONVERSATION[:1])
    _record(_hook(), home, path)
    ingest = cc.ingest_lines
    calls = []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("synthetic")
        return ingest(*args, **kwargs)
    monkeypatch.setattr(cc, "ingest_lines", failing_once)
    report = cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert report["failed"] == 1 and len(calls) == 2
    store = _open(home, hermes)
    try:
        assert [row[2] for row in _rows(store)] == ["find the secret word"]
    finally:
        store.close()


def test_a_final_reply_after_a_late_async_record_replaced_the_end_record_is_captured_by_the_next_run(
        tmp_path, home, hermes, monkeypatch):
    """REGRESSION (#2262): A1 (#2350) now keeps an unclaimed end pointer and settles once.
    A lost pointer still defers a later final reply to the next run's registered-lineage sweep:
    every transcript line in order, once."""
    path = _transcript(tmp_path, CONVERSATION[:-1])
    hook = _hook()
    _record(hook, home, path, event="Stop")
    assert hook.spool(home, {"host": "claude-code", "session_id": "s1", "event": "PostToolBatch",
                             "transcript_path": str(path)}) == "kept_end"
    [spooled] = list((home / "spool").glob("*.json"))
    assert json.loads(spooled.read_text(encoding="utf-8"))["event"] == "Stop"
    settled = []
    monkeypatch.setattr(cc, "_settle", lambda *args: settled.append(args))
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    assert len(settled) == 1
    _record(hook, home, path, event="Stop")
    [spooled] = list((home / "spool").glob("*.json"))
    spooled.unlink()  # simulate a lost last pointer; no fresh hook below
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(CONVERSATION[-1]) + "\n")
    final = "The word is ochre."
    store = _open(home, hermes)
    try:  # negative control: without the next run the final reply is absent
        assert final not in [row[2] for row in _rows(store)]
    finally:
        store.close()
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]  # no fresh hook
    store = _open(home, hermes)
    try:
        assert [row[1:3] for row in _rows(store)] == [("user", "find the secret word"),
                                                      ("assistant", "Reading notes."),
                                                      ("tool", "the secret word is ochre"),
                                                      ("assistant", final)]
    finally:
        store.close()
    assert not cc.pending(home)


# -- round 2 re-review regressions -----------------------------------------------------

def test_a_reader_cannot_attach_while_the_store_is_switched_after_close(home, hermes, monkeypatch):
    """REGRESSION: the switch after close holds the store alone from its first read to its close.
    A reader that would block it, then close before the switching connection, cannot attach, so
    that close never leaves a WAL header without sidecars again."""
    database = cc.admit(home, hermes_home=hermes)
    cc.open_store(database, home).close()  # WAL header, sidecars removed by the last close
    switch = cc.to_rollback_journal
    attached = []

    def racing(conn, attempts=5):
        conn.execute("SELECT COUNT(*) FROM metadata").fetchone()  # the switching connection's first read
        reader = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=0)
        try:
            reader.execute("SELECT COUNT(*) FROM metadata").fetchone()
        except sqlite3.OperationalError:  # the store is held: the reader retries after the switch
            reader.close()
            attached.append(False)
            return switch(conn, attempts)
        attached.append(True)
        try:
            conn.execute("PRAGMA busy_timeout = 0")
            return switch(conn, 1)  # the reader blocks the switch ...
        finally:
            reader.close()  # ... and closes first, so the switching connection's close is the last

    monkeypatch.setattr(cc, "to_rollback_journal", racing)
    assert cc.leave_readable(database)
    assert attached == [False] and _header_mode(database) == 1
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as reader:
        assert reader.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] > 0


def test_a_store_still_in_wal_after_the_switch_leaves_the_run_incomplete(tmp_path, home, hermes, monkeypatch):
    """REGRESSION: when the store cannot be left readable the worker says so: the run is
    incomplete (a bounded run's hook then starts a detached worker) and the case is logged."""
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION[:1]))
    monkeypatch.setattr(cc, "to_rollback_journal", lambda conn, attempts=5: False)
    monkeypatch.setattr(cc, "leave_readable", lambda database: False)
    report = cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert report["drained"] == 1 and report["complete"] is False
    lines = (home / "logs" / "worker.log").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["error"] for line in lines] == ["still_wal"]


def test_a_copied_revision_is_never_taken_as_its_writers_row(tmp_path, home, hermes):
    """REGRESSION: f1 copied two versions of one entry before s1's second version was captured.
    Both stay copies (the second supersedes the first), so a later fork holding that second
    version points at s1's row, and its parent is s1, never its sibling f1."""
    v1, v2 = _entry("user", "u1", "first"), _entry("user", "u1", "second")
    parent = _transcript(tmp_path, [v1])
    store = _open(home, hermes)
    try:
        cc.drain(store, cc.Lineage("s1", None, parent))
        own_f1 = [_entry("user", "u-f1", "in f1", session="f1", timestamp="2026-11-01T02:00:00.000Z")]
        cc.drain(store, _fork(tmp_path, store, "f1", [v1, v2], own_f1))
        with open(parent, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(v2) + "\n")
        cc.drain(store, cc.Lineage("s1", None, parent))
        own_f2 = [_entry("user", "u-f2", "in f2", session="f2", timestamp="2026-11-01T02:00:00.000Z")]
        cc.drain(store, _fork(tmp_path, store, "f2", [v2], own_f2))
        assert store.connection.execute(
            "SELECT disposition FROM capture_events WHERE session_id = 'f1' AND uid = 'u1' ORDER BY event_id"
        ).fetchall() == [("fork_copy",), ("fork_copy",)]
        assert store.connection.execute(
            "SELECT kind, proof_kind FROM host_uid_bindings WHERE lineage_key = 'claude-code:f1' AND uid = 'u1' "
            "ORDER BY store_id").fetchall() == [("canonical", "fork_copy"), ("version", "fork_copy")]
        (s1_v1, *_), (s1_v2, *_) = _rows(store, "s1")
        (f1_v1, *_), (f1_v2, *_), _ = _rows(store, "f1")
        (f2_v2, *_), _ = _rows(store, "f2")
        assert (f1_v2, f1_v1) in _relations(store, "supersedes")
        assert {(f1_v1, s1_v1), (f2_v2, s1_v2)} <= set(_relations(store, "fork_copy_of"))
        assert store.session_lineage_parents(["f1", "f2"]) == {"f1": "s1", "f2": "s1"}
    finally:
        store.close()


def test_a_held_file_whose_bytes_change_is_kept(tmp_path, home, hermes):
    """REGRESSION: a tool-results file whose full text a tool row holds is kept as a file once
    its bytes change (it used to be skipped for good); touching it alone stores nothing."""
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    saved = folder / "toolu_1.txt"
    saved.write_text("full output\n" * 5000, encoding="utf-8")
    marker = (f"<persisted-output>\nOutput too large (60KB). Full output saved to: {saved}\n\n"
              "Preview (first 2KB):\nfull output\n</persisted-output>")
    path = _transcript(tmp_path, [_entry("user", "r1", [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)

    def kept():
        return store.connection.execute("SELECT raw FROM capture_events WHERE uid = 'tool-results/toolu_1.txt' "
                                        "AND disposition = 'kept' ORDER BY position").fetchall()
    try:
        cc.drain(store, lineage)
        assert cc.capture_files(store, lineage) == 0 and kept() == []
        os.utime(saved, ns=(1, 1))
        assert cc.capture_files(store, lineage) == 0 and kept() == []
        saved.write_text("changed output\n" * 5000, encoding="utf-8")
        assert cc.capture_files(store, lineage) == 1
        assert b"".join(zlib.decompress(row[0]) for row in kept()) == saved.read_bytes()
        assert cc.capture_files(store, lineage) == 0
    finally:
        store.close()


def test_a_saved_output_the_store_redacts_is_kept_not_held(tmp_path, home, hermes):
    """REGRESSION: with redaction on (``LCM_SENSITIVE_PATTERNS_ENABLED``, which the worker reads
    from its environment), ingest protection redacts secrets in the tool row that holds a saved
    output, and a held file keeps no bytes of its own, so the output's original text was lost;
    now such a file keeps its bytes in the ledger."""
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    saved = folder / "toolu_1.txt"
    saved.write_text("api_key=sk-test-secret-value-123456\n" + "full output\n" * 5000, encoding="utf-8")
    marker = (f"<persisted-output>\nOutput too large (60KB). Full output saved to: {saved}\n\n"
              "Preview (first 2KB):\nfull output\n</persisted-output>")
    path = _transcript(tmp_path, [_entry("user", "r1", [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])])
    lineage = cc.Lineage("s1", None, path)
    store = cc.open_store(cc.admit(home, hermes_home=hermes), home, LCMConfig(sensitive_patterns_enabled=True))
    try:
        cc.drain(store, lineage)
        assert cc.capture_files(store, lineage) == 1
        rows = store.connection.execute("SELECT disposition, raw FROM capture_events WHERE source_kind = 'file' "
                                        "ORDER BY position").fetchall()
        assert [row[0] for row in rows] == ["kept"]
        assert b"".join(zlib.decompress(row[1]) for row in rows) == saved.read_bytes()
        assert cc.capture_files(store, lineage) == 0
    finally:
        store.close()


def test_a_chunk_row_the_ledger_cannot_add_fails_loudly(tmp_path, home, hermes, monkeypatch):
    """REGRESSION: on a store whose file index predates chunk rows, identical chunks used to be
    ignored silently (rows missing, the digest written to the wrong row); now the file's
    transaction fails and rolls back."""
    monkeypatch.setattr(cc, "FILE_CHUNK_BYTES", 10)
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    (folder / "big.bin").write_bytes(b"0123456789" * 2 + b"end")
    lineage = cc.Lineage("s1", None, _transcript(tmp_path, CONVERSATION[:1]))
    store = _open(home, hermes)
    try:
        store.connection.execute("DROP INDEX idx_capture_events_file")
        store.connection.execute("CREATE UNIQUE INDEX idx_capture_events_file "
                                 "ON capture_events(lineage_key, uid, line_digest) WHERE source_kind = 'file'")
        with pytest.raises(sqlite3.IntegrityError):
            cc.capture_files(store, lineage)
        assert store.connection.execute("SELECT COUNT(*) FROM capture_events WHERE source_kind = 'file'"
                                        ).fetchone()[0] == 0
    finally:
        store.close()


@pytest.mark.parametrize("path", ["core", "cli"])
def test_forgetting_through_core_or_the_cli_clears_capture_events(tmp_path, home, hermes, path):
    """Both session delete paths run the purge the adapter registers: core's
    ``delete_session_messages`` and the CLI's doctor clean apply (a scope table, with no arguments
    of its own) delete the session's ledger rows, keep other sessions' rows, and mark it forgotten,
    so the capture writer never reads it again."""
    from hermes_lcm.command import _delete_clean_candidates_atomically

    one = cc.Lineage("s1", None, _transcript(tmp_path, CONVERSATION[:1]))
    two = cc.Lineage("s2", None, _transcript(tmp_path, [{**CONVERSATION[0], "sessionId": "s2"}], session="s2"))
    store = _open(home, hermes)
    try:
        cc.drain(store, one)
        cc.drain(store, two)
        if path == "core":
            store.delete_session_messages("s1")
        else:
            _delete_clean_candidates_atomically(SimpleNamespace(_store=store, _session_id=""), {"s1"})
        assert [row[0] for row in store.connection.execute("SELECT session_id FROM capture_events")] == ["s2"]
        assert store.read_metadata_json("capture:forgotten:s1") and not store.read_metadata_json("capture:forgotten:s2")
        assert cc.drain(store, one) == "forgotten"
    finally:
        store.close()


ADAPTER_MODULES = {"claude_code_audit", "claude_code_capture", "claude_code_store"}


def _imports_the_adapter(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        else:
            continue
        if any(part in ADAPTER_MODULES for name in names for part in name.split(".")):
            return True
    return False


def test_core_never_imports_the_adapter():
    """The Hermes plugin loads core only: no core module imports a Claude Code adapter module, so
    the capture ledger's DDL and purge never load in Hermes. The adapter's own entry points do
    import it (the check finds them). No core module names one either, so a dynamic import
    (``importlib.import_module``) fails the check too."""
    root = Path(__file__).resolve().parents[1]
    core = [path for path in root.glob("*.py") if path.stem not in ADAPTER_MODULES]
    for package in ("access_context", "access_policy", "teams"):
        core += sorted((root / package).rglob("*.py"))
    assert {"__init__.py", "engine.py", "store.py", "command.py"} <= {path.name for path in core}
    assert [path.relative_to(root).as_posix() for path in core if _imports_the_adapter(path)] == []
    assert _imports_the_adapter(root / "scripts" / "lcm_x_capture.py")
    named = [path.relative_to(root).as_posix() for path in core
             if any(module in path.read_text(encoding="utf-8") for module in ADAPTER_MODULES)]
    assert named == []


def test_a_bounded_drain_never_waits_for_the_lock_and_the_next_worker_captures_everything(tmp_path, home, hermes):
    """REGRESSION: a bounded (PreCompact) run waited up to its deadline for a lock another worker
    held; now it returns at once, drains nothing, and the next worker captures every line."""
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION))
    with open(home / "capture.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        report = cc.run_worker(home, deadline_s=10, hermes_home=hermes, settle_s=0)
        assert time.monotonic() - started < 1
    assert report["lock"] == "held" and not report["complete"] and report["records"] == 0 and "deadline" not in report
    assert cc.pending(home)
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    store = _open(home, hermes)
    try:
        assert len(_events(store)) == len(CONVERSATION) and not cc.pending(home)
    finally:
        store.close()


def test_a_bounded_drain_cut_by_its_deadline_keeps_whole_batches_and_its_record(tmp_path, home, hermes, monkeypatch):
    """A PreCompact drain that hits its bound keeps each batch it committed whole, leaves its
    spool record, and the next worker captures the rest exactly once.

    The worker's clock is the real one plus an offset that jumps past the bound once the first
    batch commits, so the bound passes exactly then however long the worker's startup takes
    (#2198: a 0.2 s wall-clock bound could pass before the first batch on a slow runner)."""
    entries = [_entry("user", f"u{index}", f"line {index}") for index in range(9)]
    _record(_hook(), home, _transcript(tmp_path, entries))
    real_read, real_ingest = cc.read_lines, cc.ingest_lines
    monkeypatch.setattr(cc, "read_lines", lambda path, offset, **_kw: real_read(path, offset, max_lines=2))
    offset = [0.0]
    clock = SimpleNamespace(**{name: getattr(time, name) for name in dir(time) if not name.startswith("_")})
    clock.monotonic = lambda: time.monotonic() + offset[0]
    monkeypatch.setattr(cc, "time", clock)

    def bounded(store, lineage, cursor, lines):
        result = real_ingest(store, lineage, cursor, lines)
        offset[0] = 3600.0  # the bound passes after this batch commits
        return result
    monkeypatch.setattr(cc, "ingest_lines", bounded)
    report = cc.run_worker(home, deadline_s=60, hermes_home=hermes, settle_s=0)
    assert not report["complete"] and report["deadline"] and "lock" not in report and cc.pending(home)
    store = _open(home, hermes)
    try:
        assert len(_events(store)) == 2  # one whole batch
    finally:
        store.close()
    monkeypatch.setattr(cc, "ingest_lines", real_ingest)
    monkeypatch.setattr(cc, "time", time)
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    store = _open(home, hermes)
    try:
        positions = [row[0] for row in store.connection.execute(
            "SELECT position FROM capture_events WHERE source_kind = 'line' ORDER BY position")]
        assert positions == list(range(9)) and len(_rows(store)) == 9 and not cc.pending(home)
    finally:
        store.close()


_KILLED_DRIVER = """
import sys, time, types
from pathlib import Path
plugin, home, hermes, stalled = sys.argv[1:5]
package = types.ModuleType("hermes_lcm")
package.__path__, package.__package__ = [plugin], "hermes_lcm"
sys.modules["hermes_lcm"] = package
from hermes_lcm import claude_code_store as cc
real_read, real_write = cc.read_lines, cc._write_events
cc.read_lines = lambda path, offset, **_kw: real_read(path, offset, max_lines=2)
batches = []

def stall_in_the_second_batch(store, lineage, generation, events):
    real_write(store, lineage, generation, events)
    batches.append(len(events))
    if len(batches) == 2:  # its rows and ledger rows are written, not committed
        Path(stalled).touch()
        time.sleep(60)
cc._write_events = stall_in_the_second_batch
cc.run_worker(Path(home), deadline_s=30, hermes_home=Path(hermes), settle_s=0)
"""


def test_a_drain_killed_at_its_timeout_keeps_whole_batches_and_its_record(tmp_path, home, hermes):
    """The hook kills a bounded drain at its timeout (SIGKILL, ``drain_timeout``) while a batch
    is written but not committed: that batch rolls back, the one before it stays whole, the
    spool record stays, and the next worker captures the rest exactly once."""
    import subprocess
    entries = [_entry("user", f"u{index}", f"line {index}") for index in range(9)]
    _record(_hook(), home, _transcript(tmp_path, entries))
    stalled = tmp_path / "stalled"
    worker = subprocess.Popen([sys.executable, "-c", _KILLED_DRIVER, str(PLUGIN.parents[1]), str(home),
                               str(hermes), str(stalled)], stderr=subprocess.PIPE, text=True)
    try:
        end = time.monotonic() + 30
        while not stalled.exists() and worker.poll() is None and time.monotonic() < end:
            time.sleep(0.05)
    finally:
        worker.kill()  # what subprocess.run does at the hook's timeout
        _out, error = worker.communicate(timeout=10)
    assert stalled.exists(), error[-2000:]
    assert cc.pending(home)
    store = _open(home, hermes)
    try:
        assert len(_events(store)) == 2 and len(_rows(store)) == 2  # the first batch, whole
    finally:
        store.close()
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    store = _open(home, hermes)
    try:
        positions = [row[0] for row in store.connection.execute(
            "SELECT position FROM capture_events WHERE source_kind = 'line' ORDER BY position")]
        assert positions == list(range(9)) and len(_rows(store)) == 9 and not cc.pending(home)
    finally:
        store.close()


def test_a_saved_output_ingest_externalizes_in_part_is_kept_not_held(tmp_path, home, hermes):
    """REGRESSION: below the whole-output threshold, ingest protection externalizes a long base64
    run inside a saved output on its own, so the tool row holds a placeholder where the run was;
    the file was held anyway and kept no bytes. Now a file is held only when a row, as stored,
    holds its exact text, and this one keeps its bytes."""
    import base64
    import random
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    saved = folder / "toolu_1.txt"
    blob = base64.b64encode(random.Random(0).randbytes(5000)).decode()
    saved.write_text("image follows\n" + blob + "\nend of output\n", encoding="utf-8")
    marker = (f"<persisted-output>\nOutput too large (7KB). Full output saved to: {saved}\n\n"
              "Preview (first 2KB):\nimage follows\n</persisted-output>")
    path = _transcript(tmp_path, [_entry("user", "r1", [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])])
    lineage = cc.Lineage("s1", None, path)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        [(_, _, content, _)] = _rows(store)
        assert "[Externalized LCM ingest payload" in content and "end of output" in content
        assert cc.capture_files(store, lineage) == 1
        rows = store.connection.execute("SELECT disposition, raw FROM capture_events WHERE source_kind = 'file' "
                                        "ORDER BY position").fetchall()
        assert [row[0] for row in rows] == ["kept"]
        assert b"".join(zlib.decompress(row[1]) for row in rows) == saved.read_bytes()
        assert cc.capture_files(store, lineage) == 0
    finally:
        store.close()


def test_a_sweep_cut_by_its_deadline_keeps_the_claimed_record(tmp_path, home, hermes, monkeypatch):
    """REGRESSION: a drained record was removed before the sweep that captures its saved outputs,
    so a deadline or a kill during the sweep left no record; now the record stays claimed until
    the sweep has run, and the next worker captures the file."""
    folder = tmp_path / "projects" / "-work" / "s1" / "tool-results"
    folder.mkdir(parents=True)
    (folder / "shot.png").write_bytes(b"\x89PNG not text")
    _record(_hook(), home, _transcript(tmp_path, CONVERSATION))
    real = cc.capture_files
    monkeypatch.setattr(cc, "capture_files", lambda store, lineage, deadline=None: None)  # its deadline came first
    report = cc.run_worker(home, deadline_s=30, hermes_home=hermes, settle_s=0)
    assert not report["complete"] and report["deadline"] and report["records"] == 1 and cc.pending(home)
    assert [path.suffix for path in (home / "spool").iterdir() if not path.name.startswith(".")] == [".claimed"]
    monkeypatch.setattr(cc, "capture_files", real)
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"] and not cc.pending(home)
    store = _open(home, hermes)
    try:
        assert store.connection.execute("SELECT uid, disposition FROM capture_events WHERE source_kind = 'file'"
                                        ).fetchall() == [("tool-results/shot.png", "kept")]
        assert store.connection.execute("SELECT COUNT(*) FROM capture_events WHERE source_kind = 'line'"
                                        ).fetchone()[0] == len(CONVERSATION)
    finally:
        store.close()


def test_a_pending_lineage_follows_a_newer_transcript_path(tmp_path, home, hermes):
    """#2533 review: the pending row keeps the path of the latest record, so the sweep looks where the transcript
    will appear, and the first sighting's time is kept."""
    early = tmp_path / "projects" / "-old" / "s1.jsonl"
    lineage = cc.Lineage("s1", None, early)
    store = _open(home, hermes)
    try:
        assert cc.drain(store, lineage) == "missing"
        first_seen = store.read_metadata_json("capture:pending:" + lineage.key)["first_seen"]
        later = tmp_path / "projects" / "-work" / "s1.jsonl"
        assert cc.drain(store, cc.Lineage("s1", None, later)) == "missing"
        row = store.read_metadata_json("capture:pending:" + lineage.key)
        assert (row["path"], row["first_seen"]) == (str(later), first_seen)
    finally:
        store.close()
    _transcript(tmp_path, CONVERSATION)  # appears at the later path; every later record of s1 is lost
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert _rows(store), "the session must be captured at its latest path"
        assert store.read_metadata_json("capture:pending:" + lineage.key) is None
    finally:
        store.close()


def test_a_pending_path_that_now_exists_is_not_replaced(tmp_path, home, hermes):
    """#2533 review: when the recorded transcript has appeared, a newer record naming another missing path keeps
    the recorded one, so the sweep still drains the transcript that holds rows."""
    early = _transcript(tmp_path, [])  # the recorded path, still missing
    early.unlink()
    lineage = cc.Lineage("s1", None, early)
    store = _open(home, hermes)
    try:
        assert cc.drain(store, lineage) == "missing"
    finally:
        store.close()
    _transcript(tmp_path, CONVERSATION)  # the recorded transcript appears with rows, before any sweep reaches it
    later = tmp_path / "projects" / "-elsewhere" / "s1.jsonl"
    store = _open(home, hermes)
    try:
        assert cc.drain(store, cc.Lineage("s1", None, later)) == "missing"
        assert store.read_metadata_json("capture:pending:" + lineage.key)["path"] == str(early)
    finally:
        store.close()
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert _rows(store), "the transcript at the recorded path must be captured"
    finally:
        store.close()


def test_a_resume_from_another_directory_is_captured_from_its_real_path(tmp_path, home, hermes):
    """#2535 reach evidence: a resume from another directory sends SessionStart with a transcript path under the new
    directory's project folder that never exists; every later hook names the real transcript. The session is
    captured in full from the real path, its pending row clears, and the audit verifies it."""
    from hermes_lcm import claude_code_audit as ca

    phantom = tmp_path / "projects" / "-elsewhere" / "s1.jsonl"
    _record(_hook(), home, phantom, event="SessionStart")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    real = _transcript(tmp_path, CONVERSATION)
    _record(_hook(), home, real)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert len(_rows(store)) == len([e for e in CONVERSATION if e.get("type") in ("user", "assistant")])
        assert store.read_metadata_json("capture:pending:claude-code:s1") is None
    finally:
        cc.to_rollback_journal(store.connection)
        store.close()
    report = ca.audit(home)
    assert report["verified"] and report["pending_lineages"] == 0 and report["failures"]["lost"] == 0


def _append(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def test_a_second_named_path_with_uncaptured_lines_is_not_verified(tmp_path, home, hermes):
    """#2535 (ii): the cursor is on A; a record names B while B is missing; B is then written with rows that nothing
    drains. The audit must not report verified, and the resume's phantom path rule keeps (i) verified."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert ca.audit(home)["verified"]  # B never existed so far: the phantom case
    _append(b, CONVERSATION[2:])
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    report = ca.audit(home)
    assert not report["verified"] and "claude-code:s1" in report["unverified"]
    assert report["lineages"][0]["named_other_paths"] == 1
    assert "a second transcript path holds rows; not matched" in report["verdict"]


def test_a_path_named_while_pending_and_later_filled_keeps_the_lineage_unverified(tmp_path, home, hermes):
    """#2535: pending A appears with rows; in the same run a record names a missing B, A is kept and drained, and the
    first cursor inherits B as a named path. When B later fills, the audit does not report verified."""
    from hermes_lcm import claude_code_audit as ca

    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(_hook(), home, a, event="SessionStart")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(a, CONVERSATION[:2])
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert ca.audit(home)["verified"]  # A is captured; B has not appeared
    _append(b, CONVERSATION[2:])
    report = ca.audit(home)
    assert not report["verified"] and "claude-code:s1" in report["unverified"]


def test_cursor_alternation_stays_unverified(tmp_path, home, hermes):
    """#2535 (iii): A drained, B named and drained as a new generation, then A grows: A's new lines are uncaptured."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    b = _append(tmp_path / "projects" / "-other" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(a, CONVERSATION[2:])
    report = ca.audit(home)
    assert not report["verified"] and report["lineages"][0]["named_other_paths"] == 1


def test_a_known_path_adds_no_metadata_write(tmp_path, home, hermes):
    """#2535 (iv): a drain of the path the cursor already names, with no new lines, leaves metadata unchanged."""
    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION)
    lineage = cc.Lineage("s1", None, a)
    store = _open(home, hermes)
    try:
        cc.drain(store, lineage)
        before = store.connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
        assert cc.drain(store, lineage) == "drained"
        assert store.connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall() == before
        assert cc.read_cursor(store, lineage)["named"] == [str(a)]
    finally:
        store.close()


def test_a_cursor_written_before_named_paths_audits_as_before(tmp_path, home, hermes):
    """#2535 (v): a cursor without `named` names only its own path; a complete capture stays verified."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION)
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _tamper_cursor = cc.CURSOR_PREFIX + "claude-code:s1"
    store = _open(home, hermes)
    try:
        cursor = store.read_metadata_json(_tamper_cursor)
        cursor.pop("named", None)
        store.connection.execute("UPDATE metadata SET value = ? WHERE key = ?", (json.dumps(cursor), _tamper_cursor))
        store.connection.commit()
    finally:
        cc.to_rollback_journal(store.connection)
        store.close()
    assert ca.audit(home)["verified"]


def test_an_empty_named_file_never_stands_for_a_generation_with_no_rows(tmp_path, home, hermes):
    """#2548 delta review: A was empty when drained (a generation with no rows), then gained lines that were never
    drained, then a hook named B (drained as a new generation), then A was truncated. A's lines are lost: an empty A
    must not count as the copy of its empty generation."""
    from hermes_lcm import claude_code_audit as ca

    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    a.parent.mkdir(parents=True)
    a.write_bytes(b"")
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(a, CONVERSATION[2:4])
    b = _append(tmp_path / "projects" / "-other" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    a.write_bytes(b"")
    report = ca.audit(home)
    assert not report["verified"] and report["unverified"] == ["claude-code:s1"]


def test_cursor_alternation_without_growth_is_not_matched(tmp_path, home, hermes):
    """#2535 (release-manager ruling): A drained, then B with the same lines drained as a new generation, A unchanged.
    The audit does not match A against the store, so a second path holding rows leaves the lineage unverified."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    b = _append(tmp_path / "projects" / "-other" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    report = ca.audit(home)
    assert not report["verified"] and report["lineages"][0]["named_other_paths"] == 1


def _summarizer_run(tmp_path, home, hermes, *, compact_prompt="c1", agent_type="", stop_first=False,
                    agent_seen=False, extra_event=None):
    """A compaction as real Claude Code sends it (#2535 fixture): PreCompact, then the summarizer's SubagentStop with a
    new agent id, type "" and the same prompt id, whose transcript is never written (no hook finds it on disk)."""
    path = _transcript(tmp_path, CONVERSATION)
    hook = _hook()
    agent = cc.session_dir(path) / "subagents" / "agent-sum1.jsonl"
    stop = dict(event="SubagentStop", agent_id="sum1", agent_transcript_path=str(agent), agent_type=agent_type,
                prompt_id="c1", transcript_seen=True, agent_transcript_seen=agent_seen)
    steps = [lambda: _record(hook, home, path, event="PreCompact", prompt_id=compact_prompt,
                             compact_prompts=[compact_prompt], transcript_seen=True),
             lambda: _record(hook, home, path, **stop)]
    if extra_event:  # another record of the same subagent, claimed in its own run
        steps.append(lambda: _record(hook, home, path, **{**stop, "event": extra_event}))
    for step in reversed(steps) if stop_first else steps:  # each claimed by its own worker run
        step()
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert not agent.exists()


@pytest.mark.parametrize("stop_first", [False, True])
def test_a_compaction_summarizer_that_was_never_written_expires(tmp_path, home, hermes, stop_first):
    from hermes_lcm import claude_code_audit as ca

    _summarizer_run(tmp_path, home, hermes, stop_first=stop_first)
    report = ca.audit(home)
    assert report["verified"] and report["unverified"] == [] and report["pending_lineages"] == 0
    assert report["expired_never_written"] == 1
    assert report["verdict"] == "verified; 1 expired: never written (1 compaction summarizers, 0 sessions with no prompt)"


@pytest.mark.parametrize("change", [{"compact_prompt": "other"}, {"agent_type": "general-purpose"},
                                    {"agent_seen": True}, {"agent_seen": None}, {"extra_event": "PostToolBatch"}])
def test_a_subagent_without_the_summarizer_link_stays_unverified(tmp_path, home, hermes, change):
    from hermes_lcm import claude_code_audit as ca

    _summarizer_run(tmp_path, home, hermes, **change)
    report = ca.audit(home)
    assert not report["verified"] and report["unverified"] == ["claude-code:s1/agent:sum1"]
    assert report["expired_never_written"] == 0 and report["pending_lineages"] == 1


def _quit_session(tmp_path, home, hermes, events, seen=None):
    """Session s2 is captured normally; session s1's transcript is not on disk when the worker drains. ``seen`` maps an
    event to what its hook found on disk (default: absent)."""
    other = _transcript(tmp_path, [_entry("user", "v1", "other session", session="s2")], session="s2")
    _record(_hook(), home, other, session="s2", transcript_seen=True)
    missing = tmp_path / "projects" / "-work" / "s1.jsonl"
    for batch in events:  # each batch is spooled before one worker run claims it
        for event in batch:
            _record(_hook(), home, missing, event=event, transcript_seen=(seen or {}).get(event, False))
        cc.run_worker(home, hermes_home=hermes, settle_s=0)


def test_a_session_that_ended_with_no_prompt_expires(tmp_path, home, hermes):
    from hermes_lcm import claude_code_audit as ca

    _quit_session(tmp_path, home, hermes, [["SessionStart"], ["SessionEnd"]])
    report = ca.audit(home)
    assert report["verified"] and report["expired_never_written"] == 1 and report["pending_lineages"] == 0
    assert report["verdict"].endswith("; 1 expired: never written (0 compaction summarizers, 1 sessions with no prompt)")
    # The session filter judges it the same way: nothing is unverified (and nothing was checked, so not verified).
    assert ca.audit(home, "s1")["unverified"] == []


@pytest.mark.parametrize("seen", [{"SessionEnd": True}, {"SessionStart": None}])
def test_a_session_whose_hooks_saw_its_transcript_or_could_not_tell_stays_unverified(tmp_path, home, hermes, seen):
    """A built-in command alone (/usage) writes a transcript without a UserPromptSubmit, so SessionStart + SessionEnd
    is not proof of an empty session: a transcript a hook found on disk and later removed before any drain is a loss.
    A record from an older hook (unknown) never expires either."""
    from hermes_lcm import claude_code_audit as ca

    _quit_session(tmp_path, home, hermes, [["SessionStart"], ["SessionEnd"]], seen=seen)
    report = ca.audit(home)
    assert not report["verified"] and report["unverified"] == ["claude-code:s1"]
    assert report["expired_never_written"] == 0


@pytest.mark.parametrize("row", ["not a dict", {"agent_id": None, "path": "/x.jsonl"}, {"session_id": "s1"}])
def test_a_malformed_pending_row_is_unverified_and_never_crashes_the_audit(home, hermes, row):
    from hermes_lcm import claude_code_audit as ca

    store = _open(home, hermes)
    try:
        cc._write_metadata(store, "capture:pending:claude-code:s1", row)
    finally:
        store.close()
    for session in (None, "s1"):
        report = ca.audit(home, session)
        assert "claude-code:s1" in report["unverified"] and not report["verified"]
        assert report["expired_never_written"] == 0


@pytest.mark.parametrize("events", [
    [["SessionStart"], ["UserPromptSubmit"], ["SessionEnd"]],  # a turn ran: its transcript was removed, not unwritten
    [["SessionStart"], ["UserPromptSubmit", "Stop", "SessionEnd"]],  # the turn's records replaced before a claim
    [["SessionStart"], ["SessionEnd", "UserPromptSubmit"]],  # a late record kept out by the end pointer
    [["SessionStart"]],  # still open, or its end was lost
])
def test_a_session_with_a_turn_or_no_end_and_no_transcript_stays_unverified(tmp_path, home, hermes, events):
    from hermes_lcm import claude_code_audit as ca

    _quit_session(tmp_path, home, hermes, events)
    report = ca.audit(home)
    assert not report["verified"] and report["unverified"] == ["claude-code:s1"]
    assert report["expired_never_written"] == 0 and report["pending_lineages"] == 1


def test_the_compaction_list_is_capped_and_written_only_when_new(home, hermes):
    store = _open(home, hermes)
    try:
        for index in range(300):
            cc.remember_session(store, {"session_id": "s1", "compact_prompts": [f"p{index}"]})
        saved = store.read_metadata_json("capture:compact:s1")
        assert len(saved) == cc.COMPACT_CAP and saved[0] == "p44" and saved[-1] == "p299"
        before = store.connection.total_changes
        cc.remember_session(store, {"session_id": "s1", "compact_prompts": ["p299"]})
        assert store.connection.total_changes == before
    finally:
        store.close()


def _second_path(tmp_path, home, hermes):
    """The cursor is on A (captured); a record names B while B is missing."""
    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    b.parent.mkdir(parents=True, exist_ok=True)
    return b


def test_a_second_named_path_with_only_saved_outputs_is_not_verified(tmp_path, home, hermes):
    """#2535: B's transcript never appears, but its session folder holds a saved output the worker never sweeps."""
    from hermes_lcm import claude_code_audit as ca

    b = _second_path(tmp_path, home, hermes)
    folder = b.parent / "s1" / "tool-results"
    folder.mkdir(parents=True)
    (folder / "toolu_9.txt").write_text("saved output\n", encoding="utf-8")
    report = ca.audit(home)
    assert not report["verified"] and report["lineages"][0]["named_other_paths"] == 1


def test_an_empty_second_named_path_does_not_make_the_lineage_unverified(tmp_path, home, hermes):
    """#2535: an empty B with no saved outputs holds nothing, like the resume's phantom path."""
    from hermes_lcm import claude_code_audit as ca

    b = _second_path(tmp_path, home, hermes)
    b.write_bytes(b"")
    report = ca.audit(home)
    assert report["verified"] and report["lineages"][0]["named_other_paths"] == 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
@pytest.mark.parametrize("blocked", ["transcript_folder", "session_folder"])
def test_a_second_named_path_that_cannot_be_checked_is_not_verified(tmp_path, home, hermes, blocked):
    """#2535: a named path or its session folder that cannot be checked is not clean, and the audit still reports."""
    from hermes_lcm import claude_code_audit as ca

    b = _second_path(tmp_path, home, hermes)
    b.write_bytes(b"")
    folder = b.parent if blocked == "transcript_folder" else b.parent / "s1"
    if blocked == "session_folder":
        (folder / "tool-results").mkdir(parents=True)
    folder.chmod(0)
    try:
        report = ca.audit(home)
    finally:
        folder.chmod(0o700)
    assert not report["verified"] and report["lineages"][0]["named_other_paths"] == 1


def test_a_subagent_under_another_named_path_is_not_verified(tmp_path, home, hermes):
    """#2548 review: the cursor is on A; B was named and never appeared, but a subagent transcript with lines sits under
    B's session folder (its hook record lost). The worker sweeps only A's folder, so the lineage must not verify."""
    from hermes_lcm import claude_code_audit as ca

    b = _second_path(tmp_path, home, hermes)
    _append(b.parent / "s1" / "subagents" / "agent-x1.jsonl", [dict(e, isSidechain=True) for e in CONVERSATION[2:]])
    report = ca.audit(home)
    assert not report["verified"] and report["lineages"][0]["named_other_paths"] == 1


def test_a_subagent_under_a_pending_rows_earlier_path_is_not_verified(tmp_path, home, hermes):
    """#2548 review: a pending row on A is replaced by B (named [A, B]); a subagent transcript with lines appears under
    A's folder. Every named path's subagents are checked, so it is unverified (another session keeps the audit checked)."""
    from hermes_lcm import claude_code_audit as ca

    other = _append(tmp_path / "projects" / "-work" / "s2.jsonl", [dict(e, sessionId="s2") for e in CONVERSATION[:2]])
    _record(_hook(), home, other, session="s2")
    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(_hook(), home, a, event="SessionStart")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _record(_hook(), home, tmp_path / "projects" / "-other" / "s1.jsonl")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(a.parent / "s1" / "subagents" / "agent-x1.jsonl", [dict(e, isSidechain=True) for e in CONVERSATION[2:]])
    report = ca.audit(home)
    assert not report["verified"] and "claude-code:s1/agent:x1" in report["unverified"]


@pytest.mark.parametrize("copy", ["lines", "empty"])
def test_a_captured_subagents_copy_under_a_pending_rows_other_path_is_not_verified(tmp_path, home, hermes, copy):
    """#2548 review: a pending parent names A and B; the worker sweeps B, so agent x1 under B gets a cursor. Another
    agent-x1.jsonl under A is not a path that cursor names: holding lines it is unverified (no matching, #2551); empty,
    it holds nothing."""
    from hermes_lcm import claude_code_audit as ca

    other = _append(tmp_path / "projects" / "-work" / "s2.jsonl", [dict(e, sessionId="s2") for e in CONVERSATION[:2]])
    _record(_hook(), home, other, session="s2")
    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    _record(_hook(), home, a, event="SessionStart")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(b.parent / "s1" / "subagents" / "agent-x1.jsonl", [dict(e, isSidechain=True) for e in CONVERSATION[2:]])
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert "claude-code:s1/agent:x1" not in ca.audit(home)["unverified"]  # captured under B
    stray = a.parent / "s1" / "subagents" / "agent-x1.jsonl"
    if copy == "lines":
        _append(stray, [dict(e, isSidechain=True) for e in CONVERSATION[:2]])
    else:
        stray.parent.mkdir(parents=True)
        stray.write_bytes(b"")
    report = ca.audit(home)
    if copy == "lines":
        assert not report["verified"] and "claude-code:s1/agent:x1" in report["unverified"]
    else:
        assert "claude-code:s1/agent:x1" not in report["unverified"]


@pytest.mark.parametrize("copy", ["lines", "empty"])
def test_a_subagents_copy_under_a_captured_parents_folder_is_not_verified(tmp_path, home, hermes, copy):
    """#2548 review: x1 is captured under B while the parent was pending; the parent is then captured at A. A copy of
    x1 under A's folder is at a path x1's cursor never named, so x1 itself is unverified when it holds lines. (The
    parent is unverified too, through B's subagents folder: stricter than the rule, #2551.)"""
    from hermes_lcm import claude_code_audit as ca

    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    _record(_hook(), home, a, event="SessionStart")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(b.parent / "s1" / "subagents" / "agent-x1.jsonl", [dict(e, isSidechain=True) for e in CONVERSATION[:2]])
    _record(_hook(), home, b)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(a, CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert "claude-code:s1/agent:x1" not in ca.audit(home)["unverified"]
    stray = a.parent / "s1" / "subagents" / "agent-x1.jsonl"
    _append(stray, [dict(e, isSidechain=True) for e in CONVERSATION[2:4]] if copy == "lines" else [])
    assert ("claude-code:s1/agent:x1" in ca.audit(home)["unverified"]) is (copy == "lines")


@pytest.mark.parametrize("saved", ["file", "empty"])
def test_a_saved_output_under_a_pending_rows_earlier_path_is_not_verified(tmp_path, home, hermes, saved):
    """#2548 review: a pending row on A is replaced by B (named [A, B]); a saved output appears under A's session folder.
    The worker sweeps only B, so it is unverified. An empty saved-outputs folder holds nothing: the session, quit before
    any prompt with neither transcript ever seen, expires (#2535 strict) and is not unverified."""
    from hermes_lcm import claude_code_audit as ca

    other = _append(tmp_path / "projects" / "-work" / "s2.jsonl", [dict(e, sessionId="s2") for e in CONVERSATION[:2]])
    _record(_hook(), home, other, session="s2")
    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(_hook(), home, a, event="SessionStart", transcript_seen=False)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _record(_hook(), home, tmp_path / "projects" / "-other" / "s1.jsonl", event="SessionEnd", transcript_seen=False)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    results = a.parent / "s1" / "tool-results"
    results.mkdir(parents=True)
    if saved == "file":
        (results / "toolu_1.txt").write_text("saved output\n")
    report = ca.audit(home)
    if saved == "file":  # a saved output: never counted as expired, never written (#2560 review)
        assert not report["verified"] and "claude-code:s1" in report["unverified"]
        assert report["expired_never_written"] == 0 and "never written" not in report["verdict"]
    else:
        assert "claude-code:s1" not in report["unverified"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
def test_an_unreadable_subagents_folder_under_a_pending_path_is_not_verified(tmp_path, home, hermes):
    """#2548 review: glob returns nothing for a folder it cannot read, so a pending row's subagents folder that cannot be
    checked must leave the lineage unverified on its own."""
    from hermes_lcm import claude_code_audit as ca

    other = _append(tmp_path / "projects" / "-work" / "s2.jsonl", [dict(e, sessionId="s2") for e in CONVERSATION[:2]])
    _record(_hook(), home, other, session="s2")
    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(_hook(), home, a, event="SessionStart")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _record(_hook(), home, tmp_path / "projects" / "-other" / "s1.jsonl")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _append(a.parent / "s1" / "subagents" / "agent-x1.jsonl", [dict(e, isSidechain=True) for e in CONVERSATION[2:]])
    folder = a.parent / "s1" / "subagents"  # only this folder: the saved-outputs check must not be what flags it
    folder.chmod(0)
    try:
        report = ca.audit(home)
    finally:
        folder.chmod(0o700)
    assert not report["verified"] and "claude-code:s1" in report["unverified"]


# #2535 / #2548: one table for the audit's rule on paths a lineage's records named. The rule: anything holding rows the
# store was not seen to take, or that cannot be read, makes the lineage unverified; nothing else does. "followed" is
# the path capture follows (the cursor's, or the pending row's), "named" another path the lineage's records named, and
# "unnamed" a path no record or cursor names (the audit cannot tie it to the lineage, so it is out of scope).
_SIDE = [dict(e, isSidechain=True) for e in CONVERSATION[:2]]


def _table_state(tmp_path, home, hermes, state):
    """Builds a state whose audit has no s1 key unverified, and returns its locations."""
    other = _append(tmp_path / "projects" / "-work" / "s2.jsonl", [dict(e, sessionId="s2") for e in CONVERSATION[:2]])
    _record(_hook(), home, other, session="s2")
    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    c = tmp_path / "projects" / "-third" / "s1.jsonl"
    if state == "child_elsewhere":  # #2557: the parent names only A; x1's own hook puts it under B's folder. The parent
        # is captured: a never-written pending parent with a child's hook is unknown and never expires (#2535 strict)
        child = _append(b.parent / "s1" / "subagents" / "agent-x1.jsonl", _SIDE)
        _append(a, CONVERSATION[:2])
        _record(_hook(), home, a)
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(child))
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
        return {"followed": child, "folder": b}
    if state == "forgotten_parent":  # #2557: captured at A, a record names B, then the session is forgotten
        _append(a, CONVERSATION[:2])
        _record(_hook(), home, a)
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
        _record(_hook(), home, b)
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
        store = _open(home, hermes)
        try:
            store.delete_session_messages("s1")
        finally:
            store.close()
        return {"named": b}
    if state == "captured":  # cursor on A; a later record names B, which never appeared
        _append(a, CONVERSATION[:2])
        _record(_hook(), home, a)
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
        _record(_hook(), home, b)
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
        return {"followed": a, "named": b, "unnamed": c}
    # pending: records name A, then B; the pending row follows B. Quit before a prompt with neither transcript ever
    # seen, so it expires (#2535 strict) and only the artifact under test can make it unverified.
    _record(_hook(), home, a, event="SessionStart", transcript_seen=False)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    if state == "child":  # agent x1 under B is swept and captured
        _append(b.parent / "s1" / "subagents" / "agent-x1.jsonl", _SIDE)
    _record(_hook(), home, b, event="SessionEnd", transcript_seen=False)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    if state == "child":
        return {"followed": b.parent / "s1" / "subagents" / "agent-x1.jsonl",
                "uncursored": a.parent / "s1" / "subagents" / "agent-x1.jsonl"}
    return {"followed": b, "named": a, "unnamed": c}


def _table_place(where, kind):
    """Puts one artifact of ``kind`` at transcript path ``where`` (or under its session folder); returns a folder to
    unlock after the audit, if any."""
    folder = where.parent / where.stem
    if kind == "lines":
        _append(where, [dict(e, isSidechain=True) for e in CONVERSATION[2:4]] if "subagents" in where.parts
                else CONVERSATION[2:4])
    elif kind == "empty":
        where.parent.mkdir(parents=True, exist_ok=True)
        where.write_bytes(b"")
    elif kind == "saved":
        (folder / "tool-results").mkdir(parents=True, exist_ok=True)
        (folder / "tool-results" / "toolu_9.txt").write_text("saved output\n")
    elif kind == "subfolder_file":  # a tool's own file in a tool-results subfolder (page renders, #2535)
        (folder / "tool-results" / "p1").mkdir(parents=True, exist_ok=True)
        (folder / "tool-results" / "p1" / "page-1.jpg").write_bytes(b"\xff\xd8\xff")
    elif kind in ("saved_symlink", "results_symlink"):  # capture reads neither: a symlinked file, or a symlinked folder
        outside = where.parent.parent / "elsewhere"
        outside.mkdir(parents=True, exist_ok=True)
        (outside / "toolu_9.txt").write_text("saved output\n")
        folder.mkdir(parents=True, exist_ok=True)
        if kind == "results_symlink":
            (folder / "tool-results").symlink_to(outside, target_is_directory=True)
        else:
            (folder / "tool-results").mkdir(exist_ok=True)
            (folder / "tool-results" / "toolu_9.txt").symlink_to(outside / "toolu_9.txt")
    elif kind in ("subagent", "subagent_empty", "unreadable_subagents"):
        child = folder / "subagents" / "agent-y9.jsonl"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_text("" if kind == "subagent_empty" else "".join(json.dumps(e) + "\n" for e in _SIDE))
        if kind == "unreadable_subagents":
            child.parent.chmod(0)
            return child.parent
    return None


_PARENT_KINDS = ["lines", "empty", "saved", "saved_symlink", "results_symlink", "subfolder_file", "subagent",
                 "subagent_empty", "unreadable_subagents"]
_HOLDS = {"lines", "saved", "saved_symlink", "results_symlink", "subfolder_file", "subagent", "unreadable_subagents"}
_TABLE = [(state, where, kind, where != "unnamed" and kind in _HOLDS)
          for state in ("pending", "captured") for where in ("followed", "named", "unnamed") for kind in _PARENT_KINDS
          if not (state == "captured" and where == "followed" and kind == "empty")]  # A exists: no empty file to add
_TABLE += [("child", "followed", "lines", True), ("child", "uncursored", "lines", True),
           ("child", "uncursored", "empty", False)]
# #2557: a child in a folder its parent never named (that folder's artifacts count as the parent's do), and a
# forgotten parent (its own content is excluded on purpose; its subagents are still captured and checked).
_TABLE += [("child_elsewhere", "followed", "lines", True)]
_TABLE += [("child_elsewhere", "folder", kind, kind in _HOLDS) for kind in _PARENT_KINDS
           if kind != "unreadable_subagents"]  # x1 itself is in that folder: the audit raises (#2551), never verified
_TABLE += [("forgotten_parent", "named", kind, kind in {"subagent", "unreadable_subagents"}) for kind in _PARENT_KINDS]
# Where the code is stricter than the rule (it fails closed): an empty file, as a pending row's transcript or as a
# subagent transcript with no cursor, holds no rows but is unverified. Reported to the release manager; strict xfail
# so a change in either direction shows here.
_STRICTER = {("pending", "followed", "empty"), ("pending", "named", "empty"), ("pending", "followed", "subagent_empty"),
             ("pending", "named", "subagent_empty"), ("captured", "followed", "subagent_empty"),
             ("captured", "named", "subagent_empty"), ("child_elsewhere", "folder", "subagent_empty"),
             ("forgotten_parent", "named", "subagent_empty")}
_TABLE = [pytest.param(*row, id="-".join(row[:3]), marks=pytest.mark.xfail(strict=True, reason="stricter than the rule"))
          if row[:3] in _STRICTER else pytest.param(*row, id="-".join(row[:3])) for row in _TABLE]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
@pytest.mark.parametrize("state,where,kind,unverified", _TABLE)
def test_the_named_path_rule_table(tmp_path, home, hermes, state, where, kind, unverified):
    from hermes_lcm import claude_code_audit as ca

    places = _table_state(tmp_path, home, hermes, state)
    assert not [key for key in ca.audit(home)["unverified"] if key.startswith("claude-code:s1")]  # control
    locked = _table_place(places[where], kind)
    try:
        report = ca.audit(home)
    finally:
        if locked is not None:
            locked.chmod(0o700)
    assert bool([key for key in report["unverified"] if key.startswith("claude-code:s1")]) is unverified


def test_files_in_a_tool_results_subfolder_are_counted_and_named(tmp_path, home, hermes):
    """#2535: page renders a tool saved under tool-results/<folder>/ are not stored. The audit counts them on the
    lineage and names them in the verdict; by ruling they leave the lineage unverified."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert ca.audit(home)["verified"]
    pages = a.parent / "s1" / "tool-results" / "p1"
    pages.mkdir(parents=True)
    for n in (1, 2):
        (pages / f"page-{n}.jpg").write_bytes(b"\xff\xd8\xff")
    report = ca.audit(home)
    assert report["lineages"][0]["subfolder_files"] == 2 and report["lineages"][0]["files"]["uncaptured"] == 0
    assert not report["verified"] and "claude-code:s1" in report["unverified"]
    assert report["verdict"].endswith("; 2 files in tool-results subfolders not stored (1 lineages)")


def test_a_deep_tool_results_subfolder_is_counted_never_crashes(tmp_path):
    """#2554 review: the walk does not recurse (os.walk does on Python 3.11), so a tree deeper than the recursion limit
    is counted instead of aborting the audit."""
    import inspect
    import sys

    from hermes_lcm import claude_code_audit as ca

    leaf = tmp_path / "p1"
    for _ in range(60):
        leaf = leaf / "d"
    leaf.mkdir(parents=True)
    (leaf / "page-1.jpg").write_bytes(b"x")
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(len(inspect.stack()) + 30)
    try:
        count = ca._files_under(tmp_path / "p1")
    finally:
        sys.setrecursionlimit(limit)
    assert count == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
@pytest.mark.parametrize("shape,count", [("nested", 2), ("folder_symlink", 1), ("unreadable", 1)])
def test_every_kind_of_entry_under_a_tool_results_subfolder_counts(tmp_path, home, hermes, shape, count):
    """#2554 review: nested folders are walked; a symlink to a folder counts (not followed); a subfolder that cannot be
    listed counts one."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    pages = a.parent / "s1" / "tool-results" / "p1"
    (pages / "inner").mkdir(parents=True)
    if shape == "nested":
        (pages / "page-1.jpg").write_bytes(b"x")
        (pages / "inner" / "page-2.jpg").write_bytes(b"x")
    elif shape == "folder_symlink":
        (pages / "inner").rmdir()
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere" / "page-9.jpg").write_bytes(b"x")  # counted once as the link, never followed
        (pages / "link").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    else:
        (pages / "inner" / "page-1.jpg").write_bytes(b"x")
        (pages / "inner").chmod(0)
    try:
        report = ca.audit(home)
    finally:
        if shape == "unreadable":
            (pages / "inner").chmod(0o700)
    assert report["lineages"][0]["subfolder_files"] == count and not report["verified"]


def test_an_unterminated_last_record_is_reported_and_does_not_change_the_verdict(tmp_path, home, hermes):
    """#2557: capture reads complete lines only (Claude Code may be mid-write). An unterminated record after the cursor
    is counted (partial_tail_bytes) and named in the verdict, and the lineage stays verified."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    tail = json.dumps(CONVERSATION[2]).encode()
    with a.open("ab") as handle:
        handle.write(tail)
    report = ca.audit(home)
    row = report["lineages"][0]
    assert row["partial_tail_bytes"] == len(tail) and row["pending_lines"] == 0 and report["verified"]
    assert report["verdict"].endswith(f"; 1 lineages end in an unterminated record ({len(tail)} bytes, not yet captured)")


@pytest.mark.parametrize("shape", ["forgotten_outputs", "forgotten_subagent", "top_level_child", "another_sessions_folder"])
def test_a_child_outside_its_parents_folders_respects_forget_and_layout(tmp_path, home, hermes, shape):
    """#2557 review: a forgotten parent's own content in the child's folder is excluded (its subagents still count),
    and a child transcript not under subagents/, or under another session's folder, names no parent folder."""
    from hermes_lcm import claude_code_audit as ca

    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    child = {"top_level_child": tmp_path / "projects" / "-other" / "agent-x1.jsonl",
             "another_sessions_folder": b.parent / "s9" / "subagents" / "agent-x1.jsonl"}.get(
        shape, b.parent / "s1" / "subagents" / "agent-x1.jsonl")
    _append(child, _SIDE)
    _append(a, CONVERSATION[:2])
    _record(_hook(), home, a)
    _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(child))
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    if shape == "another_sessions_folder":  # s9's own transcript: not s1's, so it says nothing about s1
        _append(b.parent / "s9.jsonl", [dict(e, sessionId="s9") for e in CONVERSATION[:2]])
    elif shape != "top_level_child":
        store = _open(home, hermes)
        try:
            store.delete_session_messages("s1")
        finally:
            store.close()
        _table_place(b, "saved" if shape == "forgotten_outputs" else "subagent")
    unverified = ca.audit(home)["unverified"]
    assert bool([key for key in unverified if key.startswith("claude-code:s1")]) is (shape == "forgotten_subagent")


@pytest.mark.parametrize("forget", [False, True])
def test_a_pending_parents_uncaptured_subagent_counts_even_when_forgotten(tmp_path, home, hermes, forget):
    """#2557 review: a forgotten session's subagents are still captured, so a pending (never-captured) parent that was
    forgotten still has its subagents checked under every path its records named; with another session captured, the
    audit must not read verified while one waits for the next sweep."""
    from hermes_lcm import claude_code_audit as ca

    other = _append(tmp_path / "projects" / "-work" / "s2.jsonl", [dict(e, sessionId="s2") for e in CONVERSATION[:2]])
    _record(_hook(), home, other, session="s2")
    b = tmp_path / "projects" / "-other" / "s1.jsonl"
    _record(_hook(), home, b, event="SessionStart")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    if forget:
        store = _open(home, hermes)
        try:
            store.delete_session_messages("s1")
        finally:
            store.close()
    _append(b.parent / "s1" / "subagents" / "agent-y9.jsonl", _SIDE)
    report = ca.audit(home)
    assert not report["verified"] and "claude-code:s1/agent:y9" in report["unverified"]
    cc.run_worker(home, hermes_home=hermes, settle_s=0)  # the sweep captures it: then it is verified
    assert "claude-code:s1/agent:y9" not in ca.audit(home)["unverified"]


@pytest.mark.parametrize("shape", ["nested_child", "dotted_session", "two_subagents_segments", "inner_folder_named_s1"])
def test_a_child_elsewhere_finds_its_parents_folder_from_any_layout(tmp_path, home, hermes, shape):
    """#2557 cross-model review: the parent folder of a captured child in a folder its parent never named is the nearest
    ``subagents`` folder's parent, and its transcript is ``<folder name>.jsonl`` even when the session id has a dot."""
    from hermes_lcm import claude_code_audit as ca

    session = "s1.v2" if shape == "dotted_session" else "s1"
    a = tmp_path / "projects" / "-work" / (session + ".jsonl")
    b = tmp_path / "projects" / "-other" / (session + ".jsonl")
    folder = b.parent / session / "subagents"
    child = {"nested_child": folder / "deep" / "agent-x1.jsonl",
             "two_subagents_segments": folder / "deep" / "subagents" / "agent-x1.jsonl",
             "inner_folder_named_s1": folder / session / "subagents" / "agent-x1.jsonl"}.get(shape, folder / "agent-x1.jsonl")
    _append(child, [dict(e, sessionId=session) for e in _SIDE])
    _append(a, [dict(e, sessionId=session) for e in CONVERSATION[:2]])
    _record(_hook(), home, a, session=session)
    _record(_hook(), home, a, session=session, event="SubagentStop", agent_id="x1", agent_transcript_path=str(child))
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert "claude-code:" + session not in ca.audit(home)["unverified"]
    _append(b, [dict(e, sessionId=session) for e in CONVERSATION[:2]])  # rows at B the store never took
    assert "claude-code:" + session in ca.audit(home)["unverified"]


def test_a_child_elsewhere_in_an_inner_session_folder_checks_that_folder(tmp_path, home, hermes):
    """#2557 cross-model review: the folder found above a child is checked itself, not the one session_dir() maps its
    transcript back to (an inner ``s1`` folder directly under ``subagents`` maps to the outer one)."""
    from hermes_lcm import claude_code_audit as ca

    a = tmp_path / "projects" / "-work" / "s1.jsonl"
    inner = tmp_path / "projects" / "-other" / "s1" / "subagents" / "s1"
    child = inner / "subagents" / "agent-x1.jsonl"
    _append(child, _SIDE)
    _append(a, CONVERSATION[:2])
    _record(_hook(), home, a)
    _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(child))
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert not [key for key in ca.audit(home)["unverified"] if key.startswith("claude-code:s1")]
    _append(inner / "subagents" / "agent-y9.jsonl", _SIDE)  # a sibling nothing captured, in the inner folder
    assert "claude-code:s1/agent:y9" in ca.audit(home)["unverified"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any folder")
def test_the_folder_check_is_closed_world_over_generated_layouts(tmp_path, home, hermes):
    """#2557: the session folder check is closed-world. Whatever the folder names around it (nesting, repeated or dotted
    segments, an inner id folder), one transcript-shaped file no cursor names, a symlink, or a folder that cannot be
    listed, placed anywhere below a captured session's folder, makes the audit unverified. Empty folders alone do not
    (the control). Seeded, so a failure names its case."""
    import random
    import shutil

    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    root = a.parent / "s1"
    words = ["subagents", "workflows", "s1", "s1.v2", "deep", "x.y", "agent-x9", "run-1", "subagents.jsonl.d"]
    for seed in range(300):
        rng = random.Random(seed)
        folders = [root]
        for _ in range(rng.randint(1, 8)):  # a random tree of folders, up to 6 deep
            parent = rng.choice([f for f in folders if len(f.relative_to(root).parts) < 6])
            folders.append(parent / rng.choice(words))
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
        assert ca.audit(home)["verified"], f"control, seed {seed}"
        spot = rng.choice(folders)
        kind = rng.choice(["agent", "other", "file_link", "folder_link", "dangling", "unlistable"])
        if kind in ("agent", "other"):
            _append(spot / ("agent-y%d.jsonl" % seed if kind == "agent" else "journal.jsonl"), _SIDE)
        elif kind == "unlistable":
            (spot / "locked").mkdir(exist_ok=True)
            (spot / "locked").chmod(0)
        else:
            target = {"file_link": a, "folder_link": tmp_path, "dangling": tmp_path / "gone"}[kind]
            (spot / "link.jsonl").symlink_to(target)
        try:
            assert not ca.audit(home)["verified"], f"seed {seed}: {kind} at {spot.relative_to(root)}"
        finally:
            if kind == "unlistable":
                (spot / "locked").chmod(0o700)
            shutil.rmtree(root)


@pytest.mark.parametrize("shape", ["forgotten_journal", "linked_direct_agent", "folder_named_like_an_agent"])
def test_the_closed_world_walk_edges(tmp_path, home, hermes, shape):
    """#2557 review: a forgotten session's own files are never read again, so they never keep it unverified; a symlink or
    a folder in a judged agent file's place is still looked at, not skipped as judged."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    agents = a.parent / "s1" / "subagents"
    captured = agents / "run-1" / "agent-x9.jsonl"  # x9's cursor names this nested copy
    _append(captured, _SIDE)
    _record(_hook(), home, a)
    _record(_hook(), home, a, event="SubagentStop", agent_id="x9", agent_transcript_path=str(captured))
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert not [key for key in ca.audit(home)["unverified"] if key.startswith("claude-code:s1")]
    direct = agents / "agent-x9.jsonl"
    if shape == "forgotten_journal":
        store = _open(home, hermes)
        try:
            store.delete_session_messages("s1")
        finally:
            store.close()
        _append(a.parent / "s1" / "workflows-run" / "journal.jsonl", [{"type": "result"}])
    elif shape == "linked_direct_agent":  # a dangling symlink where a direct agent file would be
        direct.symlink_to(tmp_path / "gone.jsonl")
    else:  # a folder with an agent file's name, holding a transcript-shaped file
        _append(direct / "journal.jsonl", [{"type": "result"}])
    unverified = [key for key in ca.audit(home)["unverified"] if key.startswith("claude-code:s1")]
    assert bool(unverified) is (shape != "forgotten_journal")


# #2562: one table for workflow agents and other nested subagent transcripts. The sweep is the safety net for a missed
# hook, so it must not depend on a hook naming the path: it takes every agent-*.jsonl below subagents/, to a bounded
# depth, without following a symlinked folder. Expected: (captured, verified).
_NESTED = {"direct": ("agent-x1.jsonl",), "workflow": ("workflows", "wf_run1", "agent-x1.jsonl"),
           "deep": ("a", "b", "c", "d", "agent-x1.jsonl"), "too_deep": ("a", "b", "c", "d", "e", "agent-x1.jsonl"),
           "linked_folder": ("link", "agent-x1.jsonl"), "linked_file": ("agent-x1.jsonl",)}
_NESTED_TABLE = [(kind, hook, hook or kind in ("direct", "workflow", "deep"), not kind.startswith("linked")
                  and (hook or kind != "too_deep")) for kind in _NESTED for hook in (True, False)]


@pytest.mark.parametrize("kind,hook,captured,verified", _NESTED_TABLE,
                         ids=[f"{k}-{'hook' if h else 'missed'}" for k, h, _, _ in _NESTED_TABLE])
def test_the_nested_subagent_rule_table(tmp_path, home, hermes, kind, hook, captured, verified):
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    agents = a.parent / "s1" / "subagents"
    if kind in ("linked_folder", "linked_file"):  # the transcript lives elsewhere; a symlink in subagents/ reaches it
        elsewhere = tmp_path / "elsewhere"
        _append(elsewhere / "agent-x1.jsonl", _SIDE)
        agents.mkdir(parents=True)
        if kind == "linked_folder":
            (agents / "link").symlink_to(elsewhere)
            path = agents / "link" / "agent-x1.jsonl"
        else:
            path = agents / "agent-x1.jsonl"
            path.symlink_to(elsewhere / "agent-x1.jsonl")
    else:
        path = _append(agents.joinpath(*_NESTED[kind]), _SIDE)
    _record(_hook(), home, a)
    if hook:  # the record Claude Code sends for a workflow agent (probe on 2026-11-10)
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_type="workflow-subagent",
                agent_transcript_path=str(path))
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)  # a second run sweeps what the first registered
    store = _open(home, hermes)
    try:
        assert bool(store.read_metadata_json(cc.CURSOR_PREFIX + "claude-code:s1/agent:x1")) is captured
        assert len(_rows(store, cc.row_session_id("s1", "x1"))) == (len(_SIDE) if captured else 0)
    finally:
        store.close()
    assert ca.audit(home)["verified"] is verified


@pytest.mark.parametrize("place,captured", [(("workflows", "wf_run1"), True), ((), False), (("workflows",), False)],
                         ids=["run_folder", "subagents", "workflows"])
def test_a_workflow_journal_is_captured_only_in_its_run_folder(tmp_path, home, hermes, place, captured):
    """#2562: 31 of 15,262 real journal results were held by no transcript, so a run's journal.jsonl is captured as its
    own child lineage, every line kept raw. A journal anywhere else is not taken, and the closed-world audit flags it."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    run = a.parent / "s1" / "subagents" / "workflows" / "wf_run1"
    _append(run / "agent-x1.jsonl", _SIDE)
    events = [{"type": "started", "agentId": "x1"}, {"type": "result", "agentId": "x1", "result": {"ok": True}}]
    journal = _append(a.parent.joinpath("s1", "subagents", *place, "journal.jsonl"), events)
    _record(_hook(), home, a)
    for _ in range(2):  # the second run sweeps what the first registered
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    key = "claude-code:s1/agent:" + cc.JOURNAL_PREFIX + "wf_run1"
    report = ca.audit(home)
    assert report["verified"] is captured
    if captured:
        _append(journal, [{"type": "launched"}])  # a line written later is taken by the next sweep
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
        report = ca.audit(home)
        row = next(row for row in report["lineages"] if row["lineage"] == key)
        assert row["unknown_types"] == {"started": 1, "result": 1, "launched": 1} and report["verified"]
    else:
        assert key not in {row["lineage"] for row in report["lineages"]}


# #2535 strict: one table for when a pending record expires. The rule: a pending record expires only on facts this
# home observed itself; history that is missing or cannot be read is unknown, and unknown never expires.
_EXPIRY_EVENTS = {"end_only": ["SessionEnd"], "start_end": ["SessionStart", "SessionEnd"],
                  "with_prompt": ["SessionStart", "UserPromptSubmit", "SessionEnd"],
                  "summarizer": ["SubagentStop"], "summarizer_plus": ["SubagentStop", "SubagentStart"]}


def _expiry_case(tmp_path, home, hermes, history, events, seen):
    """Spools ``events`` for a lineage whose transcript never appears and returns its key."""
    hook = _hook()
    agent = events.startswith("summarizer")
    parent = _transcript(tmp_path, CONVERSATION) if agent else tmp_path / "projects" / "-work" / "s1.jsonl"
    path = cc.session_dir(parent) / "subagents" / "agent-sum1.jsonl" if agent else parent
    key = "claude-code:s1/agent:sum1" if agent else "claude-code:s1"
    if agent:  # the compaction this summarizer belongs to, as Claude Code sends it
        _record(hook, home, parent, event="PreCompact", prompt_id="c1", compact_prompts=["c1"], transcript_seen=True)
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    if history == "earlier_build":  # the pending row main writes since #2533: no facts
        store = _open(home, hermes)
        try:
            store.update_metadata_json(cc.PENDING_PREFIX + key, lambda _current: {
                "session_id": "s1", "agent_id": "sum1" if agent else None, "path": str(path),
                "named": [str(path)], "first_seen": 100.0})
        finally:
            store.close()
    names = _EXPIRY_EVENTS[events]
    for n, event in enumerate(names):
        extra = (dict(agent_id="sum1", agent_transcript_path=str(path), agent_type="", prompt_id="c1",
                      transcript_seen=True, agent_transcript_seen=seen) if agent else dict(transcript_seen=seen))
        if history == "pointer_unreadable" and n == len(names) - 1:  # an earlier pointer the hook cannot read
            if n == 0:  # a first pointer to make unreadable
                _record(hook, home, parent, event=event, **extra)
            pointers = sorted((home / "spool").glob("*.json"))
            for pointer in pointers:
                pointer.chmod(0)
            try:
                _record(hook, home, parent, event=event, **extra)
            finally:
                for pointer in pointers:
                    if pointer.exists():
                        pointer.chmod(0o600)
            cc.run_worker(home, hermes_home=hermes, settle_s=0)
            continue
        _record(hook, home, parent, event=event, **extra)
        if history != "pointer_unreadable":
            cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert not path.exists()
    return key


_EXPIRY = [(history, events, seen, history == "known" and seen is False and events in ("start_end", "summarizer"))
           for history in ("known", "earlier_build", "pointer_unreadable") for events in _EXPIRY_EVENTS
           for seen in (True, False, None)]


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
@pytest.mark.parametrize("history,events,seen,expires", _EXPIRY,
                         ids=[f"{h}-{e}-{s}" for h, e, s, _ in _EXPIRY])
def test_the_expiry_rule_table(tmp_path, home, hermes, history, events, seen, expires):
    from hermes_lcm import claude_code_audit as ca

    key = _expiry_case(tmp_path, home, hermes, history, events, seen)
    assert (key not in ca.audit(home)["unverified"]) is expires


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
@pytest.mark.parametrize("damage", ["unreadable", "null", "list", "garbage", "empty"])
def test_a_damaged_pointer_keeps_a_prompted_session_unknown(tmp_path, home, hermes, damage):
    """A prompt's pointer that the next hook cannot use as a pointer object: its history is unknown, so the session
    stays unverified instead of expiring as one quit before any prompt (#2535 strict)."""
    from hermes_lcm import claude_code_audit as ca

    hook = _hook()
    path = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(hook, home, path, event="SessionStart", transcript_seen=False)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _record(hook, home, path, event="UserPromptSubmit", transcript_seen=True)
    (pointer,) = sorted((home / "spool").glob("*.json"))
    if damage == "unreadable":
        pointer.chmod(0)
    else:
        pointer.write_text({"null": "null", "list": "[]", "garbage": "garbage{", "empty": ""}[damage])
    try:
        _record(hook, home, path, event="SessionEnd", transcript_seen=False)
    finally:
        if pointer.exists():
            pointer.chmod(0o600)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert not path.exists() and "claude-code:s1" in ca.audit(home)["unverified"]


@pytest.mark.parametrize("facts", [{}, None, [], {"transcript_seen": False}, {"events": "SessionStart"}],
                         ids=["empty", "null", "list", "no_events", "events_not_a_list"])
def test_a_pending_row_with_damaged_facts_stays_unknown(tmp_path, home, hermes, facts):
    """A pending row whose facts are present but not usable: unknown, never 'not seen' (#2535 strict)."""
    from hermes_lcm import claude_code_audit as ca

    hook = _hook()
    path = tmp_path / "projects" / "-work" / "s1.jsonl"
    store = _open(home, hermes)
    try:
        store.update_metadata_json(cc.PENDING_PREFIX + "claude-code:s1", lambda _current: {
            "session_id": "s1", "agent_id": None, "path": str(path), "named": [str(path)], "first_seen": 100.0,
            "facts": facts})
    finally:
        store.close()
    for event in ("SessionStart", "SessionEnd"):
        _record(hook, home, path, event=event, transcript_seen=False)
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert not path.exists() and "claude-code:s1" in ca.audit(home)["unverified"]


@pytest.mark.parametrize("damage", ["null", "list", "garbage", "empty"])
def test_a_claimed_record_that_cannot_be_read_is_set_aside(tmp_path, home, hermes, damage):
    """The worker claims a record it cannot read: it is set aside as ``.failed`` (reported), never drained as nothing
    and removed, so the events it held cannot be lost and a prompted session cannot expire (#2535 strict)."""
    from hermes_lcm import claude_code_audit as ca

    hook = _hook()
    path = tmp_path / "projects" / "-work" / "s1.jsonl"
    _record(hook, home, path, event="SessionStart", transcript_seen=False)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _record(hook, home, path, event="UserPromptSubmit", transcript_seen=True)
    (pointer,) = sorted((home / "spool").glob("*.json"))
    pointer.write_text({"null": "null", "list": "[]", "garbage": "garbage{", "empty": ""}[damage])
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert len(list((home / "spool").glob("*.failed"))) == 1
    _record(hook, home, path, event="SessionEnd", transcript_seen=False)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    report, filtered = ca.audit(home), ca.audit(home, session="s1")
    assert not path.exists() and len(list((home / "spool").glob("*.failed"))) == 1
    for result in (report, filtered):  # the session's own audit cannot read verified either
        assert not result["verified"] and "spool:unreadable_failed_record" in result["unverified"]


def test_an_unreadable_set_aside_record_is_never_overwritten(tmp_path, home, hermes, monkeypatch):
    """A later record of the same key whose drain raises is set aside beside an unreadable one, never over it, so the
    marker stays until an operator moves the file (#2535 strict, ruling A). A readable ``{}`` is not set aside."""
    from hermes_lcm import claude_code_audit as ca

    hook = _hook()
    path = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(hook, home, path)
    (pointer,) = sorted((home / "spool").glob("*.json"))
    pointer.write_text("")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    _record(hook, home, path)
    monkeypatch.setattr(cc, "drain", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom")))
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    monkeypatch.undo()
    assert len(list((home / "spool").glob("*.failed"))) == 2
    assert "spool:unreadable_failed_record" in ca.audit(home)["unverified"]
    (home / "spool" / "readable.json").write_text("{}")
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    assert len(list((home / "spool").glob("*.failed"))) == 2 and not (home / "spool" / "readable.json").exists()


# #2562 ruling: Claude Code reused one agent id in one real session (a direct and a workflow transcript, different
# content, a few shared lines). Since #2566 the second path has its own lineage; this pins that in four shapes.
_REUSED_SHARED = [{"type": "attachment", "uuid": f"sh{n}", "sessionId": "s1", "isSidechain": True,
                   "attachment": {"type": "plan", "n": n}} for n in range(2)]
_REUSED = {"first": _REUSED_SHARED + [dict(_entry("user", "fa1", "first task"), isSidechain=True),
                                      dict(_entry("assistant", "fa2", [{"type": "text", "text": "one"}]), isSidechain=True)],
           "second": _REUSED_SHARED + [dict(_entry("user", "sb1", "second task"), isSidechain=True),
                                       dict(_entry("assistant", "sb2", [{"type": "text", "text": "two"}]), isSidechain=True),
                                       dict(_entry("assistant", "sb3", [{"type": "text", "text": "three"}]), isSidechain=True)]}


@pytest.mark.parametrize("hooks", ["both_alternating", "first_then_sweep", "sweep_only", "both_then_second_grows"])
def test_a_reused_agent_id_is_pinned(tmp_path, home, hermes, hooks):
    """(file lines, lines missing from the ledger, lines held more than once) per transcript, rows, and the verdict."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    agents = a.parent / "s1" / "subagents"
    paths = {"first": _append(agents / "agent-x1.jsonl", _REUSED["first"]),
             "second": _append(agents / "workflows" / "wf_run1" / "agent-x1.jsonl", _REUSED["second"])}
    _record(_hook(), home, a)
    order = {"both_alternating": ["first", "second", "first", "second"], "first_then_sweep": ["first"],
             "sweep_only": [], "both_then_second_grows": ["first", "second"]}[hooks]
    for name in order:
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(paths[name]))
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    if hooks == "both_then_second_grows":  # before #2566 the sweep moved the shared lineage back and lost this line
        _append(paths["second"], [dict(_entry("assistant", "sb4", [{"type": "text", "text": "late"}]), isSidechain=True)])
    for _ in range(2):
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        held = collections.Counter(row[0] for row in store.connection.execute(
            "SELECT line_digest FROM capture_events WHERE source_kind = 'line'"))
        rows = collections.Counter(row[0] for row in store.connection.execute(
            "SELECT content FROM messages WHERE session_id LIKE 's1/agent:%'"))
    finally:
        store.close()
    seen = {}
    for name, path in paths.items():
        digests = [hashlib.sha256(line).hexdigest() for line in path.read_bytes().splitlines(keepends=True)]
        seen[name] = (len(digests), sum(1 for d in digests if not held[d]), sum(1 for d in digests if held[d] > 1))
    report = ca.audit(home)  # #2566: each transcript is its own lineage, so every shape captures both in full
    assert report["verified"] and not report["unverified"]
    assert [missing for _lines, missing, _held in seen.values()] == [0, 0]
    assert dict(rows) == {"first task": 1, "one": 1, "second task": 1, "two": 1, "three": 1,
                          **({"late": 1} if hooks == "both_then_second_grows" else {})}


# #2566: one table for a reused agent id. Rows: which paths exist x hook order x which file grows afterwards x store
# state (fresh, or already flipped by the old code). ONE rule decides every row: every line of every transcript is stored
# exactly once under a lineage that maps to exactly one path, and the audit is verified exactly when that holds. An
# already-flipped store keeps its rows (its old lineage spans two paths, so it stays unverified) and gains the missing
# lines, with nothing stored twice.
_REUSE_PATHS = {"direct": ("agent-x1.jsonl",), "workflow": ("workflows", "wf_run1", "agent-x1.jsonl"),
                "workflow2": ("workflows", "wf_run2", "agent-x1.jsonl")}  # two workflow runs, one id (#2574 review)
_REUSE_TABLE = [(exist, hooks, grow, state)
                for exist in (("direct",), ("workflow",), ("direct", "workflow"))
                for hooks in ((), ("direct",), ("workflow",), ("direct", "workflow"), ("workflow", "direct"))
                for grow in ((), ("direct",), ("workflow",), ("direct", "workflow"))
                for state in ("fresh", "flipped")
                if set(hooks) <= set(exist) and set(grow) <= set(exist) and (state == "fresh" or len(exist) == 2)] + [
    (("workflow", "workflow2"), hooks, grow, "fresh")
    for hooks in ((), ("workflow",), ("workflow2",), ("workflow", "workflow2"), ("workflow2", "workflow"))
    for grow in ((), ("workflow",), ("workflow2",), ("workflow", "workflow2"))]


@pytest.mark.parametrize("exist,hooks,grow,state", _REUSE_TABLE,
                         ids=["-".join(["+".join(e), "hooks:" + ("+".join(h) or "none"), "grow:" + ("+".join(g) or "none"), s])
                              for e, h, g, s in _REUSE_TABLE])
def test_the_reused_agent_id_rule_table(tmp_path, home, hermes, exist, hooks, grow, state):
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    agents = a.parent / "s1" / "subagents"
    content = {"direct": _REUSED["first"], "workflow": _REUSED["second"], "workflow2": _REUSED["first"]}
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)  # the parent is captured before its agents write
    paths = {name: _append(agents.joinpath(*_REUSE_PATHS[name]), content[name]) for name in exist}
    if state == "flipped":  # what the old code left: one id-keyed lineage moved A -> B -> A
        store = _open(home, hermes)
        try:
            for name in ("direct", "workflow", "direct"):
                cc.drain(store, cc.Lineage("s1", "x1", paths[name]))
        finally:
            store.close()
    for name in hooks:
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(paths[name]))
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    for name in grow:
        _append(paths[name], [dict(_entry("assistant", f"late-{name}", [{"type": "text", "text": f"late {name}"}]),
                                   isSidechain=True)])
    for _ in range(2):
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        conn = store.connection
        held = collections.defaultdict(set)  # line digest -> lineage keys holding it
        for digest, key in conn.execute("SELECT line_digest, lineage_key FROM capture_events WHERE source_kind = 'line'"):
            held[digest].add(key)
        rows = collections.Counter(row[0] for row in conn.execute(
            "SELECT content FROM messages WHERE session_id LIKE 's1/agent:%'"))
        cursors = {key[len(cc.CURSOR_PREFIX):]: json.loads(value) for key, value in conn.execute(
            "SELECT key, value FROM metadata WHERE key LIKE ?", (cc.CURSOR_PREFIX + "claude-code:s1/agent:%",))}
    finally:
        store.close()
    owner = {}  # path -> the one lineage whose cursor names only that path
    for key, cursor in cursors.items():
        named = set(cc.named_paths(cursor))
        if len(named) == 1:
            owner[named.pop()] = key
    lost = twice = 0
    one_to_one = all(str(path) in owner for path in paths.values()) and len(owner) == len(cursors)
    for name, path in paths.items():
        for line in path.read_bytes().splitlines(keepends=True):
            keys = held[hashlib.sha256(line).hexdigest()]
            lost += not keys
            one_to_one = one_to_one and owner.get(str(path)) in keys
    twice = sum(count - 1 for count in rows.values() if count > 1)
    holds = not lost and not twice and one_to_one
    assert (lost, twice) == (0, 0)
    assert holds is (state == "fresh")
    assert ca.audit(home)["verified"] is holds


@pytest.mark.parametrize("shape", ["bare_path_gone", "moved_folder", "slash_never_an_id"])
def test_the_reused_agent_id_edges(tmp_path, home, hermes, shape):
    """#2566 review: a path's own key is kept once assigned (the bare path vanishing must not move it back); an agent
    whose session folder was copied elsewhere (#2535) keeps one id-keyed lineage; a path key's leading "/" cannot be any
    hook or file-name id."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    agents = a.parent / "s1" / "subagents"
    if shape == "slash_never_an_id":
        record = _hook().normalize({"hook_event_name": "SubagentStop", "session_id": "s1", "agent_id": "workflows/x1",
                                    "transcript_path": str(a), "agent_transcript_path": str(agents / "x.jsonl")}, {})
        lineage = cc.path_lineage(cc.Lineage("s1", "x1", agents / "workflows" / "wf_run1" / "agent-x1.jsonl"))
        assert record["problem"] == "bad_agent_id" and lineage.agent_id == "/workflows/wf_run1/agent-x1.jsonl"
        return
    if shape == "bare_path_gone":
        paths = [_append(agents / "agent-x1.jsonl", _REUSED["first"]),
                 _append(agents / "workflows" / "wf_run1" / "agent-x1.jsonl", _REUSED["second"])]
    else:  # the same agent's transcript, also present in a copy of the session folder under another project
        paths = [_append(agents / "agent-x1.jsonl", _REUSED["first"]),
                 _append(tmp_path / "projects" / "-other" / "s1" / "subagents" / "agent-x1.jsonl", _REUSED["first"])]
    for path in paths:
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(path))
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    if shape == "bare_path_gone":
        paths[0].unlink()  # deleted (or set aside) after both were captured; then the workflow agent's hook again
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(paths[1]))
    for _ in range(2):
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        rows = collections.Counter(row[0] for row in store.connection.execute(
            "SELECT content FROM messages WHERE session_id LIKE 's1/agent:%'"))
        keys = sorted(key[len(cc.CURSOR_PREFIX):] for (key,) in store.connection.execute(
            "SELECT key FROM metadata WHERE key LIKE ?", (cc.CURSOR_PREFIX + "claude-code:s1/agent:%",)))
    finally:
        store.close()
    assert max(rows.values()) == 1  # nothing stored twice
    if shape == "bare_path_gone":
        assert keys == ["claude-code:s1/agent:/workflows/wf_run1/agent-x1.jsonl", "claude-code:s1/agent:x1"]
        assert set(rows) == {"first task", "one", "second task", "two", "three"}
    else:  # one id-keyed lineage; the copy is reported, never captured again (#2535)
        assert keys == ["claude-code:s1/agent:x1"] and not ca.audit(home)["verified"]


def test_a_swept_workflow_agent_keeps_its_saved_tool_output(tmp_path, home, hermes):
    """#2574 review: a workflow agent's saved output sits in the session's tool-results/; its message row holds the full
    output, not the preview, when the sweep (no hook) takes the nested transcript."""
    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    saved = a.parent / "s1" / "tool-results" / "toolu_9.txt"
    saved.parent.mkdir(parents=True)
    saved.write_text("full output\n" * 5000, encoding="utf-8")
    marker = (f"<persisted-output>\nOutput too large (60KB). Full output saved to: {saved}\n\n"
              "Preview (first 2KB):\nfull output\n</persisted-output>")
    _append(a.parent / "s1" / "subagents" / "workflows" / "wf_run1" / "agent-x1.jsonl",
            [dict(_entry("user", "w1", [{"type": "tool_result", "tool_use_id": "toolu_9", "content": marker}]),
                  isSidechain=True)])
    assert cc.session_dir(saved.parent.parent / "subagents" / "workflows" / "wf_run1" / "agent-x1.jsonl") == a.parent / "s1"
    _record(_hook(), home, a)
    for _ in range(2):
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        [(_, _, content, _)] = _rows(store, cc.row_session_id("s1", "x1"))
    finally:
        store.close()
    assert "chars=60000" in str(content) and "toolu_9" in str(content)
    from hermes_lcm.externalize import extract_externalized_ref
    [payload] = (home / "lcm-large-outputs").glob(extract_externalized_ref(str(content)))  # the saved bytes, not a preview
    assert json.loads(payload.read_text(encoding="utf-8"))["content"] == "full output\n" * 5000


@pytest.mark.parametrize("where", [(), ("workflows", "wf_run1")], ids=["direct", "nested"])
def test_an_agent_file_with_no_id_is_never_swept_as_its_session(tmp_path, home, hermes, where):
    """#2574 review: `agent-.jsonl` has an empty id, so its lineage key is the session's own. The sweep never takes it
    (the session's key is already seen, and the name check now says so first); the closed-world audit reports the
    session unverified, since the file ties to no cursor."""
    from hermes_lcm import claude_code_audit as ca

    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _append(a.parent.joinpath("s1", "subagents", *where, "agent-.jsonl"), _SIDE)
    _record(_hook(), home, a)
    for _ in range(2):
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert store.read_metadata_json(cc.CURSOR_PREFIX + "claude-code:s1")["path"] == str(a)
        assert len(_rows(store)) == len(CONVERSATION[:2])
    finally:
        store.close()
    assert not ca.audit(home)["verified"]


@pytest.mark.parametrize("then", ["hooks", "sweep", "key_before_delete_then_rewrite", "key_before_delete_then_copied"])
def test_a_forgotten_reused_agent_id_never_comes_back_under_a_path_key(tmp_path, home, hermes, then):
    """#2576 review: a store the old code flipped holds both transcripts under the bare id. Once the user deletes that
    child session, no later hook or sweep may store its lines again under a path key: neither a new one, nor one the
    path got before the delete (its rewritten file is re-read as a new generation)."""
    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    agents = a.parent / "s1" / "subagents"
    paths = [_append(agents / "agent-x1.jsonl", _REUSED["first"]),
             _append(agents / "workflows" / "wf_run1" / "agent-x1.jsonl", _REUSED["second"])]
    store = _open(home, hermes)
    try:
        for path in (paths[0], paths[1], paths[0]):  # what the old code left: one id-keyed lineage moved A -> B -> A
            cc.drain(store, cc.Lineage("s1", "x1", path))
        if then.startswith("key_before_delete"):  # B gets its own path key first: every line a duplicate, no rows
            keyed = cc.resolve(store, cc.Lineage("s1", "x1", paths[1]))
            assert keyed.agent_id.startswith("/") and cc.drain(store, keyed) == "drained"
        assert store.delete_session_messages(cc.row_session_id("s1", "x1"))
    finally:
        store.close()
    if then == "key_before_delete_then_rewrite":
        paths[1].unlink()
        _append(paths[1], _REUSED["second"])  # rewritten: a new generation, read again from the start
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(paths[1]))
    elif then == "key_before_delete_then_copied":  # the session folder copied to another project (#2535's move)
        b = _append(tmp_path / "projects" / "-other" / "s1.jsonl", CONVERSATION[:2])
        copy = _append(b.parent / "s1" / "subagents" / "workflows" / "wf_run1" / "agent-x1.jsonl", _REUSED["second"])
        _record(_hook(), home, b, event="SubagentStop", agent_id="x1", agent_transcript_path=str(copy))
    elif then == "hooks":
        for path in (paths[1], paths[0]):
            _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(path))
            cc.run_worker(home, hermes_home=hermes, settle_s=0)
    else:
        _record(_hook(), home, a)
    for _ in range(2):
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert not store.connection.execute("SELECT COUNT(*) FROM messages WHERE session_id LIKE 's1/agent:%'").fetchone()[0]
    finally:
        store.close()


def test_a_deleted_id_does_not_forget_a_separate_transcript_with_the_same_relative_name(tmp_path, home, hermes):
    """#2576 review: the bare id holds a transcript at subagents/workflows/subagents/agent-x1.jsonl; a separate direct
    transcript gets the path key /agent-x1.jsonl. Deleting the bare session must not forget the direct one."""
    a = _append(tmp_path / "projects" / "-work" / "s1.jsonl", CONVERSATION[:2])
    _record(_hook(), home, a)
    cc.run_worker(home, hermes_home=hermes, settle_s=0)
    agents = a.parent / "s1" / "subagents"
    nested = _append(agents / "workflows" / "subagents" / "agent-x1.jsonl", _REUSED["first"])
    direct = _append(agents / "agent-x1.jsonl", _REUSED["second"])
    for path in (nested, direct):
        _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(path))
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        assert store.delete_session_messages(cc.row_session_id("s1", "x1"))
    finally:
        store.close()
    _append(direct, [dict(_entry("assistant", "late", [{"type": "text", "text": "late"}]), isSidechain=True)])
    _record(_hook(), home, a, event="SubagentStop", agent_id="x1", agent_transcript_path=str(direct))
    for _ in range(2):
        cc.run_worker(home, hermes_home=hermes, settle_s=0)
    store = _open(home, hermes)
    try:
        rows = [row[0] for row in store.connection.execute(
            "SELECT content FROM messages WHERE session_id = ?", (cc.row_session_id("s1", "/agent-x1.jsonl"),))]
    finally:
        store.close()
    assert "late" in rows and len(rows) == len(set(rows))
