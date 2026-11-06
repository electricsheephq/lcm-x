"""Claude Code capture writer and worker: admission (H3), ingest identity, the ledger, the spool
handshake and the store left readable between worker runs. Synthetic transcripts only.

Design: docs/claude-code-adapter.md.
"""
from __future__ import annotations

import ast
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
    session exits, and one can also die in a session that goes on. Wherever the hook dies, before
    or after it writes its record, the synchronous Stop and SessionEnd hooks that follow lead the
    worker to every row, and the store matches a run in which no hook died.

    Each hook runs its registered command. A worker holds the lock while the synchronous hooks
    run, so they leave their records to it, and the worker run after them is that holder's rescan.
    The async hook never takes the lock, so its death cannot leave them waiting on it. The worker
    script it starts is a stub, and a FIFO in place of the hook log stops it just before it exits,
    unless the kill, which waits only for its record, comes first."""
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
            for event in events:
                done = subprocess.run(_argv(event), input=payload(event),
                                      env=env, capture_output=True, text=True, timeout=60)
                assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
        outcomes = [json.loads(line)["outcome"] for line in log.read_text(encoding="utf-8").splitlines()]
        assert outcomes[-len(events):] == ["worker_running"] * len(events)  # left to the holder
        assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
        return stored(home)

    tool_batch = subprocess.Popen(_argv("PostToolBatch"), env=env,
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
        assert tool_batch.poll() is None, "the hook ended before it was killed"
        os.killpg(tool_batch.pid, signal.SIGKILL)
        tool_batch.wait(timeout=60)
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
    """REGRESSION (#2262): an async PostToolBatch record that lands after the session's end record
    replaces it in the spool (one record per session), so the worker skips the settle. A final reply
    written after that drain waits for the next run, whose sweep captures it from the cursor: every
    transcript line in order, once. The transcript stays the source, so nothing is lost."""
    path = _transcript(tmp_path, CONVERSATION[:-1])
    hook = _hook()
    _record(hook, home, path, event="Stop")
    _record(hook, home, path, event="PostToolBatch")  # the late async hook
    [spooled] = list((home / "spool").glob("*.json"))
    assert json.loads(spooled.read_text(encoding="utf-8"))["event"] == "PostToolBatch"
    settled = []
    monkeypatch.setattr(cc, "_settle", lambda *args: settled.append(args))
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    assert settled == []  # no end record was left to settle
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
