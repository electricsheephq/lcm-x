"""Claude Code adapter skeleton: the hook entry's contract and the projection-v1 classes.

Design: docs/claude-code-adapter.md. The hook entry must never fail or print, must only
spool transcripts under the Claude Code projects directory, and must keep Grok Build's
session-file capture off until its own proof.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import io
import json
import os
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm.claude_code_capture import (
    PROJECTED,
    RAW_ONLY,
    SHARED_RAW,
    UNKNOWN,
    classify_entry,
    content_digest,
    lineage_key,
    project_entry,
    row_session_id,
    session_dir,
)

PLUGIN = Path(__file__).resolve().parents[1] / "adapters" / "claude-code"


def _hook_module():
    spec = importlib.util.spec_from_file_location("lcmx_hook_under_test", PLUGIN / "bin" / "lcmx_hook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook():
    return _hook_module()


@pytest.fixture
def env(tmp_path):
    return {"LCMX_CC_HOME": str(tmp_path / "home"), "LCMX_GROK_HOME": str(tmp_path / "grok-home"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "config")}


def _transcript(tmp_path) -> Path:
    path = tmp_path / "config" / "projects" / "-work-project" / "session.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    return path


def _log(env, setting="LCMX_CC_HOME") -> list[dict]:
    lines = (Path(env[setting]) / "logs" / "hook.log").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _spooled(env, setting="LCMX_CC_HOME") -> list[Path]:
    directory = Path(env[setting]) / "spool"
    return sorted(directory.glob("*.json")) if directory.exists() else []


def _wait_hooks(env, count):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        try:
            rows = _log(env)
            if len(rows) >= count:
                return rows
        except (OSError, ValueError):
            pass
        time.sleep(0.02)
    pytest.fail("hook children did not finish")


def _release(pointer):
    """Feed the stalled reader of a FIFO pointer once it has opened it: data written before any reader is lost."""
    import errno
    end = time.monotonic() + 5
    while True:
        try:
            fd = os.open(pointer, os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError as exc:  # ENXIO: no reader has the FIFO open yet
            if exc.errno != errno.ENXIO or time.monotonic() > end:
                raise
            time.sleep(0.01)
    with os.fdopen(fd, "w") as handle:
        handle.write("{}")


@pytest.mark.parametrize("event", ["Stop", "SessionEnd"])
def test_detached_fifo_does_not_block(hook, env, tmp_path, event):
    payload = {"hook_event_name": event, "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    hook.main(io.StringIO(json.dumps(payload)), env)
    [pointer] = _spooled(env)
    pointer.unlink()
    os.mkfifo(pointer)
    started = time.monotonic()
    try:
        done = subprocess.run([sys.executable, "-I", "-S", str(PLUGIN / "bin/lcmx_hook.py")],
                              input=json.dumps(payload), env=env, capture_output=True, text=True, timeout=2,
                              start_new_session=True)
        assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
        assert time.monotonic() - started < 2
    finally:
        _release(pointer)
    row = _wait_hooks(env, 2)[-1]
    assert row["detached"] is True and row["blocking_ms"] < 1000 and row["outcome"]


@pytest.mark.parametrize("event", ["Stop", "SessionEnd"])
def test_a_killed_hook_process_group_still_lands_the_pointer(hook, env, tmp_path, event):
    """A3 (#2350): Claude Code may kill the hook's process group (a -p exit). The detached child has its own
    session, so a SIGKILL to the hook's group while the child is stalled on the pointer does not reach it."""
    import signal
    payload = {"hook_event_name": event, "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    hook.main(io.StringIO(json.dumps(payload)), env)  # a previous pointer to read
    [pointer] = _spooled(env)
    pointer.unlink()
    os.mkfifo(pointer)  # stalls whichever process reads the previous pointer
    proc = subprocess.Popen([sys.executable, "-I", "-S", str(PLUGIN / "bin/lcmx_hook.py")], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        proc.stdin.write(json.dumps(payload).encode())
        proc.stdin.close()
        try:
            proc.wait(timeout=1)  # the parent returns at once; the old synchronous hook stays stalled here
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the hook's whole group, as a host would
        except ProcessLookupError:
            pass  # nothing is left in the group: the child runs in its own session
        proc.wait(timeout=5)
    finally:
        _release(pointer)
    row = _wait_hooks(env, 2)[-1]
    assert row["detached"] is True and row["outcome"]
    [landed] = _spooled(env)
    assert landed.is_file() and json.loads(landed.read_text())["event"] == event


def test_a_synchronous_hook_does_not_wait_for_a_stalled_detached_writer(hook, env, tmp_path):
    """Codex review of #2531: SessionStart and PreCompact take the same pointer lock as the detached children. While a
    stalled child holds it, a synchronous hook waits at most SYNC_LOCK_WAIT_S, skips its own pointer (lock_busy) and
    still starts the worker; the holder writes its pointer and the sweep covers the session."""
    payload = {"session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    hook.main(io.StringIO(json.dumps({**payload, "hook_event_name": "UserPromptSubmit"})), env)
    [lock_path] = list((Path(env["LCMX_CC_HOME"]) / "spool-locks").iterdir())
    with open(lock_path, "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # the stalled detached writer
        started = time.monotonic()
        done = subprocess.run([sys.executable, "-I", "-S", str(PLUGIN / "bin/lcmx_hook.py")],
                              input=json.dumps({**payload, "hook_event_name": "SessionStart", "source": "resume"}),
                              env=env, capture_output=True, text=True, timeout=4)
        waited = time.monotonic() - started
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    assert waited < 3
    assert _log(env)[-1]["outcome"].endswith("+lock_busy")


def test_detached_records_keep_synchronous_start(env, tmp_path):
    payload = {"session_id": "s1", "transcript_path": str(_transcript(tmp_path)), "source": "fork"}
    for index, event in enumerate(("SessionStart", "UserPromptSubmit", "Stop"), 1):
        done = subprocess.run([sys.executable, "-I", "-S", str(PLUGIN / "bin/lcmx_hook.py")],
                              input=json.dumps({**payload, "hook_event_name": event}), env=env,
                              capture_output=True, text=True, timeout=2)
        assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
        if index == 1:
            [pointer] = _spooled(env)
            initial = json.loads(pointer.read_text())
            assert initial["start_source"] == "fork"
        rows = _wait_hooks(env, index)
    final = json.loads(pointer.read_text())
    assert (final["start_source"], final["started_at"]) == ("fork", initial["started_at"])
    assert all(row.get("detached") for row in rows[1:])


@pytest.mark.parametrize("older,newer,expected", [("Stop", "UserPromptSubmit", "Stop"),
                                                  ("UserPromptSubmit", "Stop", "Stop"),
                                                  ("Stop", "SessionEnd", "SessionEnd"),
                                                  ("PostToolUse", "UserPromptSubmit", "UserPromptSubmit")])
def test_spool_end_guard_and_wall_stamp(hook, env, tmp_path, older, newer, expected):
    from hermes_lcm.claude_code_store import claim_records
    home = Path(env["LCMX_CC_HOME"])
    record = hook.normalize({"hook_event_name": newer, "session_id": "s1",
                             "transcript_path": str(_transcript(tmp_path))}, env)
    hook.spool(home, {**record, "hook_started_ns": 20})
    hook.spool(home, {**record, "event": older, "hook_started_ns": 10})
    assert claim_records(home)[0][1]["event"] == expected


def test_delayed_child_and_lost_stop_sweep(hook, env, tmp_path, monkeypatch):
    from hermes_lcm import claude_code_store as cc
    path, parent = _transcript(tmp_path), os.getpid()
    home, children, fork = Path(env["LCMX_CC_HOME"]), [], os.fork
    normal = hook.normalize
    def normalize(*args):
        assert os.getpid() != parent  # no filesystem validation in the parent
        return normal(*args)
    def tracked_fork():
        pid = fork()
        if pid:
            children.append(pid)
        return pid
    monkeypatch.setattr(hook, "normalize", normalize)
    monkeypatch.setattr(os, "fork", tracked_fork)
    def fire(event, delay):
        hook.main(io.StringIO(json.dumps({"hook_event_name": event, "session_id": "s1",
                  "transcript_path": str(path)})), {**env, "LCMX_TEST_HOOK_DELAY_MS": str(delay)}, detach=True)
    fire("UserPromptSubmit", 500)
    fire("Stop", 0)
    for pid in children:
        os.waitpid(pid, 0)
    assert cc.claim_records(home)[0][1]["event"] == "Stop"
    hermes = tmp_path / "hermes"
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    fire("Stop", 500)
    os.kill(children[-1], 9)  # last pointer never lands
    os.waitpid(children[-1], 0)
    path.write_text(json.dumps({"type": "assistant", "uuid": "final", "sessionId": "s1",
                               "message": {"role": "assistant", "content": "last reply"}}) + "\n")
    hook.spool(home, {"host": "claude-code", "session_id": "s2", "event": "UserPromptSubmit"})
    assert cc.run_worker(home, hermes_home=hermes, settle_s=0)["complete"]
    store = cc.open_store(home / "lcm.db", home)
    try:
        assert store.connection.execute("SELECT content FROM messages").fetchall() == [("last reply",)]
    finally:
        store.close()


EXEC_ARGS = ["-I", "-S", "${CLAUDE_PLUGIN_ROOT}/bin/lcmx_hook.py"]
SHELL_FORM = 'python3 -I -S "${CLAUDE_PLUGIN_ROOT}/bin/lcmx_hook.py"'
# Grok Build's manifest search order (crates/codegen/xai-grok-agent/src/plugins/manifest.rs in its
# source at 2bdd1d6a): the first that exists is the plugin's manifest.
GROK_MANIFESTS = ("plugin.json", ".grok-plugin/plugin.json", ".claude-plugin/plugin.json")


def _hooks_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["hooks"]


def _argv(hook: dict) -> list[str]:
    """What the host spawns for a registered command hook. With ``args``, Claude Code's exec form:
    the command found on PATH and the arguments with ``${CLAUDE_PLUGIN_ROOT}`` substituted, with no
    shell. Without, the command through ``sh -c``, as Grok Build runs it."""
    if "args" not in hook:
        return ["sh", "-c", hook["command"]]  # Grok Build finds sh on PATH
    return [hook["command"], *[arg.replace("${CLAUDE_PLUGIN_ROOT}", str(PLUGIN)) for arg in hook["args"]]]


def test_claude_payload_is_spooled_without_output(hook, env, tmp_path, capsys):
    transcript = _transcript(tmp_path)
    payload = {"hook_event_name": "Stop", "session_id": "0b6f2c1e-5f1d-4c1b-9b7e-2a1d3c4b5a69",
               "transcript_path": str(transcript), "cwd": "/work/project", "stop_hook_active": False}

    assert hook.main(io.StringIO(json.dumps(payload)), env) == 0

    assert capsys.readouterr().out == ""
    [record_path] = _spooled(env)
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["host"] == "claude-code"
    assert record["transcript_path"] == str(transcript.resolve())
    # No runtime is set up in the test, so the worker is not started; the hook still succeeds.
    assert _log(env)[-1]["outcome"] == "runtime_missing"


def test_one_record_per_session_and_subagent(hook, env, tmp_path):
    transcript = _transcript(tmp_path)
    base = {"session_id": "s1", "transcript_path": str(transcript)}
    for payload in ({**base, "hook_event_name": "UserPromptSubmit"},
                    {**base, "hook_event_name": "Stop"},
                    {**base, "hook_event_name": "SubagentStop", "agent_id": "a1",
                     "agent_transcript_path": str(transcript)}):
        hook.main(io.StringIO(json.dumps(payload)), env)
    assert len(_spooled(env)) == 2


@pytest.mark.parametrize("raw, outcome", [
    ("not json", "error:JSONDecodeError"),
    ("[1, 2]", "payload_not_object"),
    ('{"hook_event_name": "Teleport", "session_id": "s1"}', "unsupported_event"),
    ('{"hook_event_name": "Stop", "session_id": "../escape"}', "bad_session_id"),
    ('{"hook_event_name": "Stop", "session_id": "s1"}', "no_transcript_path"),
])
def test_bad_input_exits_zero_and_spools_nothing(hook, env, capsys, raw, outcome):
    assert hook.main(io.StringIO(raw), env) == 0
    assert capsys.readouterr().out == ""
    assert _spooled(env) == []
    assert _log(env)[-1]["outcome"] == outcome


def test_oversized_payload_is_refused(hook, env):
    assert hook.main(io.StringIO("x" * (hook.MAX_PAYLOAD + 1)), env) == 0
    assert _log(env)[-1]["outcome"] == "payload_too_large"


def test_transcript_outside_projects_directory_is_refused(hook, env, tmp_path):
    outside = tmp_path / "elsewhere.jsonl"
    outside.write_text("", encoding="utf-8")
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(outside)}
    assert hook.main(io.StringIO(json.dumps(payload)), env) == 0
    assert _spooled(env) == []
    assert _log(env)[-1]["outcome"] == "transcript_path_outside_projects"


def test_capture_off_writes_nothing(hook, env, tmp_path):
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    assert hook.main(io.StringIO(json.dumps(payload)), {**env, "LCMX_CAPTURE": "off"}) == 0
    assert not Path(env["LCMX_CC_HOME"]).exists()


def test_grok_build_payload_is_normalized_and_off_by_default(hook, env):
    grok_env = {**env, "GROK_SESSION_ID": "g-1"}
    payload = {"hookEventName": "Stop", "sessionId": "g-1", "cwd": "/work", "reason": "session_end"}
    record = hook.normalize(payload, grok_env)
    assert record["host"] == "grok-build"
    assert record["session_id"] == "g-1"
    assert record["transcript_path"] is None
    assert record["observe_only"] is True
    assert record["problem"] == "grok_capture_off"
    assert hook.main(io.StringIO(json.dumps(payload)), grok_env) == 0
    assert _spooled(env) == [] and _spooled(env, "LCMX_GROK_HOME") == []


def test_grok_build_events_are_refused_in_their_own_home_whatever_the_old_flag_says(hook, env):
    """REGRESSION (#2189): no Grok session-file reader is in this release. LCMX_GROK_CAPTURE=1 let
    Grok records into the spool, where the worker matched no lineage (they carry no transcript
    path) and deleted each one, having captured nothing. Every Grok event is now refused, and
    logged in Grok's own home (one owning harness per store)."""
    grok_env = {**env, "GROK_SESSION_ID": "g-1", "LCMX_GROK_CAPTURE": "1"}
    for event, reason in (("Stop", "end_turn"), ("PostToolUse", None), ("SessionEnd", None), ("Stop", "session_end")):
        payload = {"hookEventName": event, "sessionId": "g-1", "cwd": "/work", "reason": reason}
        assert hook.main(io.StringIO(json.dumps(payload)), grok_env) == 0
        assert _log(env, "LCMX_GROK_HOME")[-1]["outcome"] == "grok_capture_off"
    assert _spooled(env) == [] and _spooled(env, "LCMX_GROK_HOME") == []
    assert not Path(env["LCMX_CC_HOME"]).exists()


@pytest.mark.parametrize("setting", ["LCMX_CC_HOME", "CLAUDE_CONFIG_DIR"])
def test_relative_paths_are_refused(hook, env, tmp_path, monkeypatch, setting):
    """A relative setting would resolve against whichever project directory the hook runs in."""
    monkeypatch.chdir(tmp_path)
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    assert hook.main(io.StringIO(json.dumps(payload)), {**env, setting: "relative/dir"}) == 0
    assert not (tmp_path / "relative").exists()
    assert _spooled(env) == []
    if setting == "CLAUDE_CONFIG_DIR":
        assert _log(env)[-1]["outcome"] == "config_dir_not_absolute"


def test_plugin_files_are_consistent(hook):
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    hooks = _hooks_file(PLUGIN / "hooks" / "hooks.json")
    servers = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert manifest["name"] == "lcm-x"
    assert set(hooks) == set(hook.EVENTS) - {"PostToolUse"}  # accepted for Grok Build only
    assert hook.ASYNC_EVENTS == {"PostToolBatch"}
    for event, entries in hooks.items():
        for entry in entries:
            # Claude Code's exec form: python3, found on PATH, starts with no shell before it (#2141).
            # -I -S skips site and ignores the user's Python settings.
            assert (entry["hooks"][0]["command"], entry["hooks"][0]["args"]) == ("python3", EXEC_ARGS)
            # Only PostToolBatch runs in the background: the worker settles on the end events, a fork's
            # start source rides on SessionStart's record, and -p kills an async hook still running at exit.
            assert entry["hooks"][0].get("async", False) is (event in hook.ASYNC_EVENTS)
    assert set(servers) == {"lcm-x"}


def test_each_host_has_its_own_hooks_file_with_the_same_hooks():
    """#2141: Claude Code runs the exec form, which starts no shell. Grok Build's hook runner drops
    ``args``, so Grok keeps the shell form in its own file. The two files register the same events,
    matchers, timeouts and async flags; only the command form differs, so they cannot drift."""
    claude = _hooks_file(PLUGIN / "hooks" / "hooks.json")
    grok = _hooks_file(PLUGIN / ".grok-plugin" / "hooks.json")

    def without_command(hooks):
        return {event: [{**group, "hooks": [{key: value for key, value in each.items() if key not in ("command", "args")}
                                            for each in group["hooks"]]} for group in groups]
                for event, groups in hooks.items()}

    def commands(hooks):
        return {(each["command"], tuple(each.get("args", ()))) for groups in hooks.values()
                for group in groups for each in group["hooks"]}

    assert without_command(claude) == without_command(grok)
    assert commands(claude) == {("python3", tuple(EXEC_ARGS))}
    assert commands(grok) == {(SHELL_FORM, ())}


def test_grok_build_reads_its_own_manifest_and_hooks_in_the_shell_form():
    """#2141, a static check; Grok Build's runtime half stays unproven until it is installed (R11).
    Grok takes the first manifest in GROK_MANIFESTS, and a hooks path there replaces hooks/hooks.json
    instead of merging with it. Its runner drops ``args`` and starts a command through ``sh -c``
    only when the command holds a shell character such as a space or ``$``. It runs any other as a
    path in the hooks file's folder, where an exec-form ``python3`` does not exist."""
    assert [name for name in GROK_MANIFESTS if (PLUGIN / name).is_file()][0] == ".grok-plugin/plugin.json"
    grok = json.loads((PLUGIN / ".grok-plugin" / "plugin.json").read_text(encoding="utf-8"))
    claude = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    # Claude Code merges a manifest's hooks into hooks/hooks.json, so this would register each hook twice.
    assert "hooks" not in claude
    assert {**grok, "hooks": None} == {**claude, "hooks": None}  # one name, version and description
    hooks = (PLUGIN / grok["hooks"]).resolve()
    assert hooks == (PLUGIN / ".grok-plugin" / "hooks.json").resolve()
    for groups in _hooks_file(hooks).values():
        for group in groups:
            for each in group["hooks"]:
                assert "args" not in each and any(mark in each["command"] for mark in " |&;<>$")


@pytest.mark.parametrize("entry, expected", [
    ({"type": "user", "message": {"role": "user", "content": "hi"}}, PROJECTED),
    ({"type": "assistant", "message": {"role": "assistant", "content": []}}, PROJECTED),
    ({"type": "user"}, RAW_ONLY),
    ({"type": "attachment", "attachment": {}}, RAW_ONLY),
    ({"type": "system", "subtype": "compact_boundary"}, RAW_ONLY),
    ({"type": "api-request-blob"}, SHARED_RAW),
    ({"type": "tools-used"}, RAW_ONLY),
    ({"type": "bridge-session"}, RAW_ONLY),
    ({"type": "result", "result": "may carry content"}, UNKNOWN),
    ({"type": "a-type-from-a-future-version"}, UNKNOWN),
])
def test_projection_v1_classes(entry, expected):
    assert classify_entry(entry) == expected


def test_lineage_keys_and_subagent_session_ids():
    assert row_session_id("s1") == "s1"
    assert row_session_id("s1", "a7") == "s1/agent:a7"
    assert lineage_key("s1") == "claude-code:s1"
    assert lineage_key("s1", "a7") == "claude-code:s1/agent:a7"
    with pytest.raises(ValueError):
        lineage_key("")


def test_precompact_reports_the_worker_outcome(hook, env, tmp_path, monkeypatch):
    """REGRESSION: a failed or incomplete bounded drain is never reported as drained, and it
    leaves the rest to a detached worker (a bounded worker does not rescan the spool), unless
    the store was refused."""
    package = tmp_path / "package"
    (package / "scripts").mkdir(parents=True)
    (package / "plugin.yaml").write_text("", encoding="utf-8")
    (package / "store.py").write_text("", encoding="utf-8")
    worker = package / "scripts" / "lcm_x_capture.py"
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(package)}
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    continued = home / "continued"
    popen, starts = subprocess.Popen, []

    def counting(*args, **kwargs):  # counts a detached start even if its worker has not run yet
        if kwargs.get("start_new_session"):
            starts.append(args)
        return popen(*args, **kwargs)
    monkeypatch.setattr(subprocess, "Popen", counting)
    for code, outcome, detached in ((0, "drained", False), (3, "drain_incomplete+worker", True),
                                    (1, "drain_failed:1+worker", True), (2, "drain_failed:2", False),
                                    (4, "worker_running", False), (5, "drain_bound+worker", True)):
        continued.unlink(missing_ok=True)
        starts.clear()
        worker.write_text("import pathlib, sys\n"
                          "if '--deadline' not in sys.argv:\n"
                          "    (pathlib.Path(sys.argv[3]) / 'continued').touch()\n"
                          f"raise SystemExit({code})\n", encoding="utf-8")
        assert hook.start_worker(home, lane, deadline_s=5) == outcome
        assert len(starts) == int(detached)
        end = time.monotonic() + 10
        while detached and not continued.exists() and time.monotonic() < end:
            time.sleep(0.05)
        assert continued.exists() == detached


def test_spool_record_carries_seq_and_local_gc(hook, env, tmp_path):
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    hook.main(io.StringIO(json.dumps(payload)), {**env, "CLAUDE_CODE_TRANSCRIPT_LOCAL_GC": "1"})
    [record_path] = _spooled(env)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert isinstance(record["seq"], int) and record["local_gc"] is True


def _entry(kind, content, **extra):
    return {"type": kind, "uuid": "u1", "sessionId": "s1", "timestamp": "2026-11-01T00:00:00.000Z",
            "message": {"role": kind, "content": content}, **extra}


def test_projection_user_and_assistant_rows():
    [(uid, row)] = project_entry(_entry("user", "hello")).rows
    assert uid == "u1" and row == {"role": "user", "content": "hello", "timestamp": "2026-11-01T00:00:00.000Z"}
    parts = [{"type": "thinking", "thinking": "private", "signature": "x"},
             {"type": "text", "text": "Running it."},
             {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}]
    [(uid, row)] = project_entry(_entry("assistant", parts)).rows
    assert uid == "u1" and row["content"] == "Running it."
    assert row["tool_calls"] == [{"id": "toolu_1", "type": "function",
                                  "function": {"name": "Bash", "arguments": '{"command": "ls"}'}}]
    assert "private" not in json.dumps(row)
    assert project_entry(_entry("assistant", parts[:1])).rows == []  # thinking only: ledger only


def test_projection_tool_results_and_system_rows():
    results = [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "one"},
               {"type": "tool_result", "tool_use_id": "toolu_2",
                "content": [{"type": "text", "text": "two"}, {"type": "image", "source": {"type": "base64"}}]}]
    rows = project_entry(_entry("user", results)).rows
    assert [uid for uid, _ in rows] == ["u1#0", "u1#1"]
    assert rows[0][1] == {"role": "tool", "content": "one", "tool_call_id": "toolu_1",
                          "timestamp": "2026-11-01T00:00:00.000Z"}
    assert rows[1][1]["content"][1]["type"] == "image"
    for flag in ("isMeta", "isCompactSummary"):
        assert project_entry(_entry("user", "note", **{flag: True})).rows[0][1]["role"] == "system"
    error = _entry("assistant", [{"type": "text", "text": "API Error"}], isApiErrorMessage=True)
    assert project_entry(error).rows[0][1]["role"] == "system"
    image = _entry("user", [{"type": "text", "text": "see"}, {"type": "image", "source": {}}])
    assert isinstance(project_entry(image).rows[0][1]["content"], list)


def _saved(tmp_path, name, data):
    folder = tmp_path / "proj" / "s1"
    (folder / "tool-results").mkdir(parents=True)
    path = folder / "tool-results" / name
    path.write_bytes(data)
    marker = (f"<persisted-output>\nOutput too large (40.0KB). Full output saved to: {path}\n\n"
              "Preview (first 2KB):\nline 1\n</persisted-output>")
    return folder, path, marker


def test_projection_recovers_the_saved_tool_output(tmp_path):
    folder, path, marker = _saved(tmp_path, "toolu_1.txt", b"line 1\nline 2\n" * 3000)
    entry = _entry("user", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}],
                   toolUseResult={"persistedOutputPath": str(path)})
    projection = project_entry(entry, folder=folder)
    assert projection.rows[0][1]["content"] == path.read_text()
    assert projection.files == ["toolu_1.txt"] and projection.problems == []
    # Without the session folder (or outside it) the preview is kept and the reason recorded.
    assert project_entry(entry).rows[0][1]["content"] == marker
    elsewhere = tmp_path / "other"
    elsewhere.mkdir()
    assert project_entry(entry, folder=elsewhere).problems == ["persisted_output_outside_session"]


@pytest.mark.parametrize("data, problem", [(b"\xff\xfe\x00binary", "persisted_output_binary"),
                                           (None, "persisted_output_missing")])
def test_projection_keeps_the_preview_when_the_saved_output_cannot_be_read(tmp_path, data, problem):
    folder, path, marker = _saved(tmp_path, "toolu_1.bin", data or b"x")
    if data is None:
        path.unlink()
    entry = _entry("user", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": marker}])
    projection = project_entry(entry, folder=folder)
    assert projection.rows[0][1]["content"] == marker and projection.problems == [problem]


def test_projection_ignores_a_persisted_output_marker_quoted_mid_text(tmp_path):
    """REGRESSION (#2187): a tool output that quotes the tag and a real saved-output path in its
    middle keeps its own text; only a result that starts with the tag is a preview."""
    folder, path, marker = _saved(tmp_path, "toolu_9.txt", b"another tool's full output\n")
    text = f"grep found this in a log:\n{marker}\nend of match"
    entry = _entry("user", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": text}])
    projection = project_entry(entry, folder=folder)
    assert projection.rows[0][1]["content"] == text
    assert projection.files == [] and projection.problems == []


def test_content_digest_ignores_only_the_copy_envelope():
    entry = _entry("user", "hello", promptId="p1")
    copy = {**entry, "sessionId": "fork", "promptId": "p2"}
    assert content_digest(entry) == content_digest(copy)
    assert content_digest(entry) != content_digest({**entry, "cwd": "/elsewhere"})
    assert content_digest(entry) != content_digest(_entry("user", "hello!"))


def test_session_folder_of_main_and_subagent_transcripts(tmp_path):
    assert session_dir(tmp_path / "proj" / "s1.jsonl") == tmp_path / "proj" / "s1"
    assert session_dir(tmp_path / "proj" / "s1" / "subagents" / "agent-a1.jsonl") == tmp_path / "proj" / "s1"


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _package(tmp_path) -> Path:
    package = tmp_path / "package"
    (package / "scripts").mkdir(parents=True)
    for name in ("plugin.yaml", "store.py", "scripts/lcm_x_capture.py", "scripts/lcm_x_mcp.py"):
        (package / name).write_text("", encoding="utf-8")
    return package


def test_a_python_on_the_hooks_path_runs_with_the_standard_library_only(hook, env, tmp_path):
    """Without a set-up runtime, a Python 3.11+ on the hook's PATH runs isolated (-I -S): a
    versioned name, or python3 when it is the interpreter running the hook. A configured
    runtime runs as it is, and only the hook's own PATH is searched."""
    runtime, home = hook.lcmx_runtime, Path(env["LCMX_CC_HOME"])
    found = _executable(tmp_path / "bin" / "python3.12")
    assert runtime.runtime_command({**env, "PATH": str(found.parent)}, home) == [str(found), "-I", "-S"]
    configured = {**env, "PATH": str(found.parent), "LCMX_PYTHON": sys.executable}
    assert runtime.runtime_command(configured, home) == [sys.executable]
    assert runtime.runtime_command(env, home) is None
    other = _executable(tmp_path / "other" / "python3")  # not the interpreter running the hook
    assert runtime.runtime_command({**env, "PATH": str(other.parent)}, home) is None
    own = tmp_path / "own" / "python3"
    own.parent.mkdir()
    own.symlink_to(sys.executable)
    assert runtime.runtime_command({**env, "PATH": str(own.parent)}, home) == [str(own), "-I", "-S"]
    _executable(own.parent / "python3.12")  # the hook's own python3 comes before a versioned name
    assert runtime.runtime_command({**env, "PATH": str(own.parent)}, home) == [str(own), "-I", "-S"]


def test_relative_path_entries_are_never_searched_for_a_python(hook, env, tmp_path, monkeypatch):
    """REGRESSION: an empty or relative PATH entry names the folder Claude Code runs the hook in,
    so a python3.12 planted in a project could have run the worker."""
    runtime, home = hook.lcmx_runtime, Path(env["LCMX_CC_HOME"])
    planted = _executable(tmp_path / "project" / "python3.12")
    monkeypatch.chdir(planted.parent)
    for search in (".", os.pathsep + "/nonexistent"):
        assert runtime.runtime_command({**env, "PATH": search}, home) is None
    monkeypatch.chdir(tmp_path)
    assert runtime.runtime_command({**env, "PATH": "project"}, home) is None


def test_the_worker_and_the_reader_start_isolated_on_a_found_python(hook, env, tmp_path, monkeypatch):
    package, found = _package(tmp_path), _executable(tmp_path / "bin" / "python3.11")
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: started.append(command))
    lane = {**env, "PATH": str(found.parent), "LCMX_PACKAGE_ROOT": str(package)}
    assert hook.start_worker(home, lane) == "worker_started"
    assert started == [[str(found), "-I", "-S", str(package / "scripts" / "lcm_x_capture.py"), "worker",
                        "--home", str(home)]]

    spec = importlib.util.spec_from_file_location("lcmx_mcp_under_test", PLUGIN / "bin" / "lcmx_mcp.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    for name in ("LCMX_PYTHON", "CLAUDE_PLUGIN_DATA", "GROK_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**lane, "LCMX_CC_HOME": str(home)}.items():
        monkeypatch.setenv(name, value)
    executed = []
    monkeypatch.setattr(launcher.os, "execv", lambda path, argv: executed.append((path, argv)))
    launcher.main()
    assert executed == [(str(found), [str(found), "-I", "-S", str(package / "scripts" / "lcm_x_mcp.py"),
                                      "--store", str(home / "lcm.db"), "--hermes-home", str(home),
                                      "--log-dir", str(home / "logs")])]


@pytest.mark.parametrize("mode", ["venv", "found"])
def test_a_hook_that_finds_a_worker_running_imports_nothing_it_does_not_need(env, tmp_path, mode):
    """A hook pays for its imports before it runs (#2141): typing never loads, and a synchronous
    hook that finds the worker running loads no subprocess. The hook runs as Claude Code starts
    it. With a set-up runtime it loads no shutil either; a Python found on PATH is looked up with
    shutil before the lock probe, since only a hook that can start a worker may probe."""
    transcript = _transcript(tmp_path)
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "transcript_path": str(transcript)}
    lane = {**env, "PATH": str(Path(sys.executable).parent)}
    if mode == "venv":
        lane["LCMX_PYTHON"] = sys.executable
    with open(home / "capture.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = subprocess.run([sys.executable, "-I", "-S", "-X", "importtime", str(PLUGIN / "bin" / "lcmx_hook.py")],
                              input=json.dumps(payload), env=lane, capture_output=True, text=True, timeout=60)
        _wait_hooks(env, 1)
    assert (done.returncode, done.stdout) == (0, "")
    imported = {line.rsplit("|", 1)[-1].strip() for line in done.stderr.splitlines() if line.startswith("import time:")}
    assert "json" in imported  # the import log is complete
    assert imported.isdisjoint({"subprocess", "typing"} | ({"shutil"} if mode == "venv" else set()))
    assert _log(env)[-1]["outcome"] == "worker_running"


APPLE_PREFIX = "/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9"
APPLE_CACHE = "/Users/u/Library/Caches/com.apple.python"


HOME_ONLY = {b"HOME": b"/Users/u"}


@pytest.mark.parametrize(("platform", "base_prefix", "prefix", "environ", "expected"), [
    ("darwin", APPLE_PREFIX, None, HOME_ONLY, (APPLE_CACHE, False)),
    ("darwin", "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9", None,
     HOME_ONLY, (APPLE_CACHE, False)),
    ("darwin", APPLE_PREFIX, None, {**HOME_ONLY, b"PYTHONDONTWRITEBYTECODE": b"1"}, (APPLE_CACHE, True)),
    ("darwin", APPLE_PREFIX, "/a/cache/already/set", HOME_ONLY, None),
    ("darwin", APPLE_PREFIX, None, {}, None),
    ("darwin", APPLE_PREFIX, None, {b"HOME": b"relative/home"}, None),
    ("darwin", "/Library/Frameworks/Python.framework/Versions/3.12", None, HOME_ONLY, None),
    ("darwin", "/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12", None, HOME_ONLY, None),
    ("linux", "/usr", None, {b"HOME": b"/home/u"}, None),
])
def test_only_apples_python_gets_its_bytecode_cache_back(hook, platform, base_prefix, prefix, environ, expected):
    """REGRESSION (#2154): under -I, Apple's system Python loses its bytecode cache, stops writing
    one and compiles every module the hook imports from source, which doubled the hook's time.
    Only that Python, with no cache set and a home to put one in, gets back the cache and the
    writing it has without -I; any other keeps its own."""
    assert hook._apple_bytecode_cache(platform, base_prefix, prefix, environ) == expected


def _worker_running_lane(env, tmp_path) -> tuple[dict, str]:
    """A hook that reaches the lock probe: a set-up runtime, a worker script and a transcript."""
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(_package(tmp_path))}
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    Path(env["LCMX_CC_HOME"]).mkdir(parents=True)
    return lane, json.dumps(payload)


def test_any_other_python_keeps_its_own_bytecode_cache(env, tmp_path):
    """The cache fix leaves a Python that is not Apple's alone (here the one running the tests):
    run with the registered flags, it reads nothing from and writes nothing to Apple's cache."""
    lane, payload = _worker_running_lane(env, tmp_path)
    user = tmp_path / "user"
    user.mkdir()
    with open(Path(env["LCMX_CC_HOME"]) / "capture.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = subprocess.run([sys.executable, "-I", "-S", "-v", str(PLUGIN / "bin" / "lcmx_hook.py")], input=payload,
                              env={**lane, "HOME": str(user)}, capture_output=True, text=True, timeout=60)
        _wait_hooks(env, 1)
    assert (done.returncode, done.stdout) == (0, "")
    assert "# code object from" in done.stderr  # the import log is complete
    assert "com.apple.python" not in done.stderr and not (user / "Library").exists()
    assert _log(env)[-1]["outcome"] == "worker_running"


def test_the_cache_fix_fails_open(env, tmp_path):
    """Run as if on Apple's Python, a hook whose cache cannot be written (a home it cannot write,
    under a folder it cannot write) or that has no home still spools and prints nothing: a cache
    it cannot use only costs the compile."""
    lane, payload = _worker_running_lane(env, tmp_path)
    as_apple = ("import sys; sys.platform = 'darwin'; sys.base_prefix = %r; sys.pycache_prefix = None; "
                "path = sys.argv[1]; sys.argv = [path]; "
                "exec(compile(open(path).read(), path, 'exec'), {'__name__': '__main__', '__file__': path})"
                % APPLE_PREFIX)
    locked = tmp_path / "locked"
    (locked / "home").mkdir(parents=True)
    homes = {"unwritable": locked / "home", "missing": locked / "absent"}
    try:
        for folder in (locked / "home", locked):
            folder.chmod(0o500)
        with open(Path(env["LCMX_CC_HOME"]) / "capture.lock", "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runs = {name: subprocess.run([sys.executable, "-I", "-S", "-c", as_apple, str(PLUGIN / "bin" / "lcmx_hook.py")],
                                         input=payload, env={**lane, "HOME": str(home)}, capture_output=True, text=True,
                                         timeout=60) for name, home in homes.items()}
            runs["no home"] = subprocess.run([sys.executable, "-I", "-S", "-c", as_apple,
                                              str(PLUGIN / "bin" / "lcmx_hook.py")], input=payload, env=lane,
                                             capture_output=True, text=True, timeout=60)
            _wait_hooks(env, 3)
    finally:
        for folder in (locked, locked / "home"):
            folder.chmod(0o700)
    assert {name: (done.returncode, done.stdout, done.stderr) for name, done in runs.items()} == {
        name: (0, "", "") for name in runs}
    assert [row["outcome"] for row in _log(env)] == ["worker_running"] * 3
    if os.geteuid() != 0:  # root writes through the 0o500 folders
        assert not (locked / "absent").exists() and not (locked / "home" / "Library").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Apple's system Python exists only on macOS")
def test_apples_python_reads_its_bytecode_cache_under_the_registered_flags(env, tmp_path):
    """REGRESSION (#2154): run with the registered -I -S, Apple's system Python compiled json and
    the rest from source on every start. With a home, the second start reads them from the cache
    there; with no home, or one it cannot write, the hook still spools and prints nothing."""
    if subprocess.run(["xcode-select", "-p"], capture_output=True).returncode != 0:
        pytest.skip("no developer tools, so /usr/bin/python3 is only an installer stub")
    probe = subprocess.run(["/usr/bin/python3", "-c", "import sys; print(sys.base_prefix)"], capture_output=True,
                           text=True, timeout=60)
    if "/Python3.framework/" not in probe.stdout:
        pytest.skip("/usr/bin/python3 is not Apple's framework Python")
    lane, payload = _worker_running_lane(env, tmp_path)
    user, unwritable = tmp_path / "user", tmp_path / "unwritable"
    user.mkdir()
    unwritable.mkdir(mode=0o500)
    argv = ["/usr/bin/python3", "-I", "-S", "-v", str(PLUGIN / "bin" / "lcmx_hook.py")]
    try:
        with open(Path(env["LCMX_CC_HOME"]) / "capture.lock", "a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runs = [subprocess.run(argv, input=payload, env={**lane, "HOME": str(home)}, capture_output=True, text=True,
                                   timeout=60) for home in (user, user, unwritable)]
            runs.append(subprocess.run(argv, input=payload, env=lane, capture_output=True, text=True, timeout=60))
            _wait_hooks(env, 4)
    finally:
        unwritable.chmod(0o700)
    assert [(done.returncode, done.stdout) for done in runs] == [(0, "")] * 4
    loaded = [line for line in runs[1].stderr.splitlines() if line.startswith("# code object from")
              and "/json/__init__." in line]
    assert loaded and str(user / "Library" / "Caches" / "com.apple.python") in loaded[0]
    assert [row["outcome"] for row in _log(env)] == ["worker_running"] * 4


def test_only_a_hook_that_can_start_a_worker_probes_the_lock(hook, env, tmp_path, monkeypatch):
    """REGRESSION: the probe holds the lock for an instant, and a hook or a starting worker that
    meets it gives way. A hook with no runtime or worker script that probed could leave another
    session's records with nobody to drain them, so it returns before the probe."""
    home = Path(env["LCMX_CC_HOME"])
    package = tmp_path / "package"  # core, but without the worker script
    package.mkdir()
    for name in ("plugin.yaml", "store.py"):
        (package / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(hook, "_lock_held", lambda lock: pytest.fail("a hook that starts no worker probed the lock"))
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(package)}
    for deadline in (0, hook.PRECOMPACT_DEADLINE_S):
        assert hook.start_worker(home, {**env, "PATH": ""}, deadline_s=deadline) == "runtime_missing"
        assert hook.start_worker(home, lane, deadline_s=deadline) == "worker_missing"


def test_the_async_hook_starts_a_worker_without_probing_the_lock(hook, env, tmp_path, monkeypatch):
    """REGRESSION (review of #2154): Claude Code may kill an async hook at any point, also while
    its probe holds the lock and a Stop or SessionEnd hook that met the probe has left its record
    to it. So the async PostToolBatch hook never probes: it starts a worker, which exits at once
    if another holds the lock. The synchronous hooks still probe."""
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(_package(tmp_path))}
    payload = {"hook_event_name": "PostToolBatch", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    started, probed = [], []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: started.append(command))
    monkeypatch.setattr(hook, "_lock_held", lambda lock: probed.append(lock) or True)
    assert hook.main(io.StringIO(json.dumps(payload)), lane) == 0
    assert (_log(env)[-1]["outcome"], len(started), probed) == ("worker_started", 1, [])
    assert hook.main(io.StringIO(json.dumps({**payload, "hook_event_name": "Stop"})), lane) == 0
    assert (_log(env)[-1]["outcome"], len(started), len(probed)) == ("worker_running", 1, 1)


def test_each_hook_log_line_times_the_steps_that_ran(hook, env, tmp_path, monkeypatch):
    """#2164: a slow hook run must be split by step, so each log line carries the wall-clock start
    of the hook's own code (``t0``, to join Claude Code's debug line) and the milliseconds of each
    step that ran, in order, including the step a run returned or failed in. Measurement only: the
    outcomes and ``ms`` are as before."""
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(_package(tmp_path))}
    payload = {"session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    checks, code = [], [0]
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: None)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, code[0]))
    monkeypatch.setattr(hook, "_lock_held", lambda lock: checks.pop(0))
    spooled = ["imports", "read", "spool", "runtime"]
    drained = spooled + ["probe", "subprocess", "drain"]
    runs = [  # event, environment, what each lock check finds, the drain's exit code, outcome, steps
        ("Stop", lane, [False], 0, "worker_started", spooled + ["probe", "subprocess", "start", "log"]),
        ("Stop", lane, [True], 0, "worker_running", spooled + ["probe", "log"]),
        ("PostToolBatch", lane, [], 0, "worker_started+kept_end", spooled + ["subprocess", "start", "log"]),
        ("PreCompact", lane, [False], 0, "drained", drained + ["log"]),
        ("PreCompact", lane, [False, True], 5, "drain_bound+holder", drained + ["recheck", "log"]),
        ("PreCompact", lane, [False, False], 5, "drain_bound+worker", drained + ["recheck", "start", "log"]),
        ("Stop", env, [], 0, "runtime_missing", spooled + ["log"]),
        ("Notification", lane, [], 0, "unsupported_event", ["imports", "read", "log"]),
    ]
    for event, environ, found, exit_code, outcome, steps in runs:
        checks[:], code[0] = found, exit_code
        assert hook.main(io.StringIO(json.dumps({**payload, "hook_event_name": event})), environ) == 0
        row = _log(env)[-1]
        assert (row["outcome"], list(row["phases"]), checks) == (outcome, steps, [])
        assert all(type(value) is float and value >= 0 for value in [row["ms"], *row["phases"].values()])
        assert row["t0"] == round(hook._T0_WALL, 3)
    assert hook.main(io.StringIO("not json"), lane) == 0
    row = _log(env)[-1]
    assert (row["outcome"], list(row["phases"])) == ("error:JSONDecodeError", ["imports", "read", "log"])


def test_a_slow_step_is_charged_to_the_step_that_took_it(hook, env, tmp_path, monkeypatch):
    """REGRESSION (review of #2165): a PreCompact run's second lock check was charged to starting
    the worker, or to no step at all, and a step that ended in an early return was not logged, so a
    slow run could not be split by step."""
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(_package(tmp_path))}
    payload = {"hook_event_name": "PreCompact", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    checks = []

    def check(lock):
        held, delay = checks.pop(0)
        time.sleep(delay)
        return held

    monkeypatch.setattr(hook, "_lock_held", check)
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: None)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 5))
    for held, outcome in ((True, "drain_bound+holder"), (False, "drain_bound+worker")):
        checks[:] = [(False, 0), (held, 0.06)]  # the probe finds the lock free; the check after the drain is slow
        assert hook.main(io.StringIO(json.dumps(payload)), lane) == 0
        row = _log(env)[-1]
        assert (row["outcome"], row["phases"]["recheck"] >= 50) == (outcome, True)

    def slow_lookup(environ, home):
        time.sleep(0.06)
        return None

    monkeypatch.setattr(hook.lcmx_runtime, "runtime_command", slow_lookup)
    assert hook.main(io.StringIO(json.dumps({**payload, "hook_event_name": "Stop"})), lane) == 0
    row = _log(env)[-1]
    assert (row["outcome"], row["phases"]["runtime"] >= 50) == ("runtime_missing", True)


@pytest.mark.parametrize("setting", ["PYTHONPATH", "PYTHONHOME"])
def test_the_registered_hook_command_ignores_the_users_python_settings(env, tmp_path, setting):
    """REGRESSION (#2141): Claude Code runs the hook command in the user's environment, where a
    PYTHONPATH can shadow the standard library and a PYTHONHOME can stop python3 from starting.
    The registered command runs isolated, so the hook still spools and prints nothing: Claude Code's
    exec form and Grok Build's shell form alike."""
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "json.py").write_text("raise SystemExit('a planted module loaded')\n", encoding="utf-8")
    python = tmp_path / "bin" / "python3"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    lane = {**env, "PATH": f"{python.parent}:/bin", "CLAUDE_PLUGIN_ROOT": str(PLUGIN), setting: str(planted)}
    for runs, hooks_file in enumerate((PLUGIN / "hooks" / "hooks.json", PLUGIN / ".grok-plugin" / "hooks.json"), 1):
        [registered] = _hooks_file(hooks_file)["Stop"][0]["hooks"]
        with open(home / "capture.lock", "a") as handle:  # a worker is running, so the hook starts none
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            done = subprocess.run(_argv(registered), input=json.dumps(payload), env=lane,
                                  capture_output=True, text=True, timeout=60)
            _wait_hooks(env, runs)
        assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
        assert len(_spooled(env)) == 1
        assert [row["outcome"] for row in _log(env)] == ["worker_running"] * runs  # each form logged its own run


def test_the_reader_names_the_one_setup_step_when_no_python_is_found(env, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("lcmx_mcp_under_test", PLUGIN / "bin" / "lcmx_mcp.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    for name in ("LCMX_PYTHON", "CLAUDE_PLUGIN_DATA", "GROK_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LCMX_CC_HOME", env["LCMX_CC_HOME"])
    monkeypatch.setenv("LCMX_PACKAGE_ROOT", str(_package(tmp_path)))
    assert launcher.main() == 1
    assert "uv venv --python 3.11" in capsys.readouterr().err


@pytest.mark.parametrize(("setting", "outcome"), [("LCMX_PYTHON", "runtime_missing"),
                                                  ("LCMX_PACKAGE_ROOT", "package_missing")])
def test_a_relative_runtime_setting_is_refused_and_the_record_waits(hook, env, tmp_path, monkeypatch, capsys,
                                                                    setting, outcome):
    """REGRESSION (#2189): the hook checked a relative LCMX_PYTHON or LCMX_PACKAGE_ROOT from the
    folder it runs in, and the worker then ran in the store home, where the path names nothing.
    Each is refused, as a relative home is: no found Python or other package stands in for it,
    nothing starts, the record stays spooled, and the server names the setting."""
    package = _package(tmp_path)
    python = tmp_path / "bin" / "python3.11"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)  # from here, each relative path below names a real file
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: started.append(command))
    lane = {**env, "PATH": str(python.parent), "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(package)}
    lane[setting] = {"LCMX_PYTHON": "bin/python3.11", "LCMX_PACKAGE_ROOT": "package"}[setting]
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    assert hook.main(io.StringIO(json.dumps(payload)), lane) == 0
    assert (_log(env)[-1]["outcome"], started) == (outcome, [])
    assert len(_spooled(env)) == 1

    spec = importlib.util.spec_from_file_location("lcmx_mcp_under_test", PLUGIN / "bin" / "lcmx_mcp.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    for name in ("CLAUDE_PLUGIN_DATA", "GROK_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in lane.items():
        monkeypatch.setenv(name, value)
    # A valid override, so the guards below catch it starting. Its command is a script the test
    # owns: the interpreter running the tests can be group-writable, as on CI's runners.
    override = str(_executable(tmp_path / "override" / "reader"))
    _write_override(env, {"version": 1, "command": [override]})
    assert launcher.reader_override(Path(env["LCMX_CC_HOME"]) / launcher.OVERRIDE) == ([override], "")
    monkeypatch.setattr(launcher.os, "execv", lambda path, argv: pytest.fail("the reader started"))
    monkeypatch.setattr(launcher.os, "execve", lambda path, argv, environment: pytest.fail("the override started"))
    assert launcher.main() == 1  # the setting is refused before a valid override could start
    assert capsys.readouterr().err == "lcm-x: %s must be an absolute path\n" % setting


class _Started(Exception):
    """Stands in for a successful exec, which never returns."""


@pytest.fixture
def start_reader(env, tmp_path, monkeypatch):
    """Runs the launcher in-process on a set-up runtime. Returns the one exec it made, as
    (which, path, argv, environment): "override" for execve, "plain" for execv."""
    spec = importlib.util.spec_from_file_location("lcmx_mcp_under_test", PLUGIN / "bin" / "lcmx_mcp.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    package = _package(tmp_path)
    Path(env["LCMX_CC_HOME"]).mkdir(parents=True)
    for name in ("CLAUDE_PLUGIN_DATA", "GROK_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(package)}.items():
        monkeypatch.setenv(name, value)
    made = []

    def recorder(which):
        def execute(path, argv, environment=None):
            made.append((which, path, argv, environment))
            raise _Started
        return execute

    monkeypatch.setattr(launcher.os, "execv", recorder("plain"))
    monkeypatch.setattr(launcher.os, "execve", recorder("override"))

    def start():
        made.clear()
        with pytest.raises(_Started):
            launcher.main()
        (only,) = made
        return only
    start.launcher = launcher
    return start


def test_grok_build_offers_no_recall_tools_over_a_store_nothing_writes(start_reader, env, monkeypatch, capsys):
    """REGRESSION (#2267): the hooks capture no Grok Build event, so under Grok Build with no lcm.db
    and no reader override the launcher starts no reader and says why. A store there still starts one."""
    grok = Path(env["LCMX_GROK_HOME"])
    monkeypatch.setenv("GROK_SESSION_ID", "g-1")
    assert start_reader.launcher.main() == 0  # an exec would raise from the fixture's recorder
    err = capsys.readouterr().err
    assert "Grok Build sessions are not captured" in err and str(grok) in err
    assert not grok.exists()
    grok.mkdir()
    (grok / "lcm.db").write_bytes(b"")
    which, _path, argv, _environment = start_reader()
    assert which == "plain" and argv[argv.index("--store") + 1] == str(grok / "lcm.db")


def _plain_reader(env, tmp_path, runtime=(sys.executable,)):
    home = Path(env["LCMX_CC_HOME"])
    return [*runtime, str(tmp_path / "package" / "scripts" / "lcm_x_mcp.py"), "--store", str(home / "lcm.db"),
            "--hermes-home", str(home), "--log-dir", str(home / "logs")]


def _write_override(env, spec):
    path = Path(env["LCMX_CC_HOME"]) / "reader-override.json"
    path.write_text(spec if isinstance(spec, str) else json.dumps(spec), encoding="utf-8")
    path.chmod(0o644)  # whatever the umask: a group-writable file would not be used


@pytest.mark.parametrize("runtime", ["set_up", "found"])
def test_a_valid_reader_override_starts_instead_of_the_plain_reader(start_reader, env, tmp_path, monkeypatch, capsys,
                                                                    runtime):
    """With no file the plain reader starts and nothing is logged. A valid file's command starts
    instead, with the plain reader's exact argv in LCMX_LOCAL_READER_CMD (a found Python's -I -S
    included) and the rest of the environment unchanged; other keys are ignored."""
    python = [sys.executable]
    if runtime == "found":
        found = _executable(tmp_path / "bin" / "python3.11")
        monkeypatch.delenv("LCMX_PYTHON")
        monkeypatch.setenv("PATH", str(found.parent))
        python = [str(found), "-I", "-S"]
    plain = _plain_reader(env, tmp_path, python)
    assert start_reader() == ("plain", plain[0], plain, None)
    assert capsys.readouterr().err == ""
    override = str(_executable(tmp_path / "override" / "reader"))
    _write_override(env, {"version": 1, "command": [override, "--flag"], "other": "ignored"})
    which, path, argv, environment = start_reader()
    assert (which, path, argv) == ("override", override, [override, "--flag"])
    assert environment == {**os.environ, "LCMX_LOCAL_READER_CMD": json.dumps(plain)}
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("case, reason", [
    ("malformed", "not valid JSON"),
    ("not_an_object", "not a JSON object"),
    ("version_text", '"version" is not an integer'),
    ("version_true", '"version" is not an integer'),
    ("version_float", '"version" is not an integer'),
    ("version_2", '"version" is not 1'),
    ("command_text", '"command" is not an array of strings'),
    ("command_number", '"command" is not an array of strings'),
    ("command_empty", '"command" is empty'),
    ("relative", '"command[0]" is not an absolute path'),
    ("missing", '"command[0]" is missing or not executable'),
    ("not_executable", '"command[0]" is missing or not executable'),
    ("directory", "not a regular file"),
    ("device", "not a regular file"),
    pytest.param("no_permission", "could not be read",
                 marks=pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")),
    ("other_owner", "not owned by this user"),
    ("group_writable", "writable by group or others"),
    ("other_writable", "writable by group or others"),
    ("command_other_writable", '"command[0]" is writable by group or others'),
])
def test_a_bad_reader_override_starts_the_plain_reader(start_reader, env, tmp_path, monkeypatch, capsys, case,
                                                       reason):
    """Every fail-open case starts the plain reader and logs one line naming the reason."""
    reader = str(_executable(tmp_path / "override" / "reader"))
    idle = tmp_path / "override" / "idle"
    idle.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")  # no execute bit
    loose = _executable(tmp_path / "override" / "loose")
    loose.chmod(0o757)  # anyone may rewrite it
    monkeypatch.chdir(tmp_path)  # so the relative command names a real executable
    specs = {
        "malformed": '{"version": 1, "command": [',
        "not_an_object": [1],
        "version_text": {"version": "1", "command": [reader]},
        "version_true": {"version": True, "command": [reader]},
        "version_float": {"version": 1.0, "command": [reader]},
        "version_2": {"version": 2, "command": [reader]},
        "command_text": {"version": 1, "command": reader},
        "command_number": {"version": 1, "command": [reader, 1]},
        "command_empty": {"version": 1, "command": []},
        "relative": {"version": 1, "command": ["override/reader"]},
        "missing": {"version": 1, "command": [str(tmp_path / "override" / "missing")]},
        "not_executable": {"version": 1, "command": [str(idle)]},
        "command_other_writable": {"version": 1, "command": [str(loose)]},
    }
    path = Path(env["LCMX_CC_HOME"]) / "reader-override.json"
    if case == "directory":
        path.mkdir()
    elif case == "device":
        path.symlink_to(os.devnull)
    elif case == "no_permission":
        _write_override(env, {"version": 1, "command": [reader]})
        path.chmod(0)
    elif case == "other_owner":  # a test cannot chown, so the launcher runs as another user instead
        _write_override(env, {"version": 1, "command": [reader]})
        uid = os.geteuid()
        monkeypatch.setattr(os, "geteuid", lambda: uid + 1)
    elif case in ("group_writable", "other_writable"):
        _write_override(env, {"version": 1, "command": [reader]})
        path.chmod(0o620 if case == "group_writable" else 0o602)
    else:
        _write_override(env, specs[case])
    plain = _plain_reader(env, tmp_path)
    assert start_reader() == ("plain", plain[0], plain, None)
    (line,) = capsys.readouterr().err.splitlines()
    assert reason in line and line.endswith("; starting the plain reader")


_SH = Path("/bin/sh")


@pytest.mark.skipif(not (_SH.is_file() and _SH.stat().st_uid == 0), reason="needs a root-owned /bin/sh")
def test_a_reader_override_may_name_a_root_owned_interpreter(start_reader, env, capsys):
    """A command[0] that root owns, as system interpreters are, still starts."""
    _write_override(env, {"version": 1, "command": [str(_SH), "-c", "exit 0"]})
    assert start_reader()[:3] == ("override", str(_SH), [str(_SH), "-c", "exit 0"])
    assert capsys.readouterr().err == ""


def test_untrusted_names_the_other_writers_a_file_could_have():
    """The ownership and mode rule both ways, on stat results a test cannot make for real (a file
    another non-root user owns)."""
    spec = importlib.util.spec_from_file_location("lcmx_mcp_under_test", PLUGIN / "bin" / "lcmx_mcp.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)

    def info(mode, uid):
        return os.stat_result((stat.S_IFREG | mode, 0, 0, 1, uid, 0, 0, 0, 0, 0))

    assert launcher.untrusted(info(0o644, 501), (501,)) == ""
    assert launcher.untrusted(info(0o644, 502), (501,)) == "not owned by this user"
    assert launcher.untrusted(info(0o644, 0), (501,)) == "not owned by this user"
    assert launcher.untrusted(info(0o755, 0), (501, 0)) == ""
    assert launcher.untrusted(info(0o755, 502), (501, 0)) == "not owned by this user or root"
    assert launcher.untrusted(info(0o664, 501), (501,)) == "writable by group or others"
    assert launcher.untrusted(info(0o646, 501), (501,)) == "writable by group or others"
    assert launcher.untrusted(info(0o775, 0), (501, 0)) == "writable by group or others"


@pytest.mark.parametrize("case, reason", [
    ("folder", "which is writable by others"),
    ("dot_dot", "which is writable by others"),
    ("symlink_in_folder", "which is writable by others"),
    ("symlink_into_folder", "which is writable by others"),
    ("group", "which is writable by its group crew, whose members include guest"),
    ("unknown_group", "which is writable by its group crew, which the user database does not list"),
    ("unlistable_database", "which is writable by its group crew (user database not provably local)"),
])
def test_a_reader_override_another_user_could_redirect_starts_the_plain_reader(start_reader, env, tmp_path,
                                                                                monkeypatch, capsys, case, reason):
    """REGRESSION (#2261): command[0] passed its own checks while a folder on the way to it, or to a
    symlink on it, let another local user replace or redirect it before the start. Such a folder
    now starts the plain reader, and the line names it."""
    base = Path(os.path.realpath(tmp_path))
    shared = base / "shared"
    shared.mkdir()
    trusted = _executable(base / "override" / "reader")
    command = _executable(shared / "reader") if case != "symlink_in_folder" else shared / "reader"
    if case == "symlink_in_folder":
        command.symlink_to(trusted)
    elif case == "symlink_into_folder":
        command = base / "override" / "link"
        command.symlink_to(shared / "reader")
    elif case == "dot_dot":  # `..` is looked up in the shared folder, so the shared folder is checked
        command = shared / ".." / "override" / "reader"
    shared.chmod(0o775 if case in ("group", "unknown_group", "unlistable_database") else 0o777)
    monkeypatch.setattr(start_reader.launcher, "group_name", lambda gid: "crew")
    monkeypatch.setattr(start_reader.launcher, "user_database_is_local", lambda: case != "unlistable_database")
    if case == "group":
        monkeypatch.setattr(start_reader.launcher, "other_members", lambda gid, owners: ["guest"])
    elif case == "unknown_group":
        def unlisted(gid, owners):
            raise KeyError(gid)

        monkeypatch.setattr(start_reader.launcher, "other_members", unlisted)
    elif case == "unlistable_database":  # SSSD, say: users it does not list may be members
        monkeypatch.setattr(start_reader.launcher, "other_members", lambda gid, owners: [])
    _write_override(env, {"version": 1, "command": [str(command)]})
    plain = _plain_reader(env, tmp_path)
    assert start_reader() == ("plain", plain[0], plain, None)
    (line,) = capsys.readouterr().err.splitlines()
    assert '"command[0]" is under %s, %s' % (shared, reason) in line and line.endswith("; starting the plain reader")


def test_a_command_swapped_between_the_check_and_the_start_never_starts(start_reader, env, tmp_path, monkeypatch):
    """REGRESSION (#2261): the replacement the review reproduced with a second user, made here
    between the launcher's check and its start, as the folder's mode lets another user do. The
    swapped file started; now the folder that allows the swap starts the plain reader."""
    base = Path(os.path.realpath(tmp_path))
    command = _executable(base / "shared" / "reader")
    command.parent.chmod(0o777)
    checked = start_reader.launcher.reader_override

    def check_then_swap(path):
        result = checked(path)
        swapped = command.parent / "swapped"
        swapped.write_text("#!/bin/sh\necho another user's program\n", encoding="utf-8")
        swapped.chmod(0o755)
        os.replace(swapped, command)
        return result

    monkeypatch.setattr(start_reader.launcher, "reader_override", check_then_swap)
    _write_override(env, {"version": 1, "command": [str(command)]})
    plain = _plain_reader(env, tmp_path)
    assert start_reader() == ("plain", plain[0], plain, None)


@pytest.mark.parametrize("case", ["sticky", "group_of_this_user", "symlink"])
def test_a_reader_override_behind_folders_only_this_user_can_write_starts_as_configured(start_reader, env, tmp_path,
                                                                                        monkeypatch, capsys, case):
    """A sticky folder holding this user's own entry passes, as does a folder writable only by a
    group whose members are this user or root. The command starts by its configured path, not the
    one its symlinks resolve to: a venv's Python finds its venv from the path it was started by."""
    base = Path(os.path.realpath(tmp_path))
    reader = _executable(base / "shared" / "reader")
    command = reader
    if case == "sticky":
        reader.parent.chmod(0o1777)
    elif case == "group_of_this_user":
        reader.parent.chmod(0o775)
        monkeypatch.setattr(start_reader.launcher, "user_database_is_local", lambda: True)
        monkeypatch.setattr(start_reader.launcher, "other_members", lambda gid, owners: [])
    else:
        command = base / "venv" / "bin" / "python"
        command.parent.mkdir(parents=True)
        command.symlink_to(reader)
    _write_override(env, {"version": 1, "command": [str(command), "-I"]})
    assert start_reader()[:3] == ("override", str(command), [str(command), "-I"])
    assert capsys.readouterr().err == ""


def _launcher():
    spec = importlib.util.spec_from_file_location("lcmx_mcp_under_test", PLUGIN / "bin" / "lcmx_mcp.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_folder_untrusted_names_who_else_could_replace_an_entry(monkeypatch):
    """The folder rule both ways, on stat results a test cannot make for real: folders and entries
    other users own, and groups with other members."""
    launcher = _launcher()
    members = {80: [], 81: ["guest"]}

    def other_members(gid, owners):
        return members[gid]  # a KeyError, as for a group the user database does not list

    monkeypatch.setattr(launcher, "other_members", other_members)
    monkeypatch.setattr(launcher, "group_name", lambda gid: {80: "crew", 81: "staff"}.get(gid, str(gid)))
    monkeypatch.setattr(launcher, "user_database_is_local", lambda: True)

    def folder(mode, uid=501, gid=80):
        return os.stat_result((stat.S_IFDIR | mode, 0, 0, 1, uid, gid, 0, 0, 0, 0))

    mine, theirs = (os.stat_result((stat.S_IFREG | 0o755, 0, 0, 1, uid, 0, 0, 0, 0, 0)) for uid in (501, 502))
    owners = (501, 0)
    assert launcher.folder_untrusted(folder(0o755), mine, owners) == ""
    assert launcher.folder_untrusted(folder(0o755, uid=0), theirs, owners) == ""
    assert launcher.folder_untrusted(folder(0o755, uid=502), mine, owners) == "not owned by this user or root"
    assert launcher.folder_untrusted(folder(0o775), mine, owners) == ""
    assert launcher.folder_untrusted(folder(0o775, gid=81), mine, owners) == (
        "writable by its group staff, whose members include guest")
    assert launcher.folder_untrusted(folder(0o775, gid=99), mine, owners) == (
        "writable by its group 99, which the user database does not list")
    assert launcher.folder_untrusted(folder(0o757), mine, owners) == "writable by others"
    assert launcher.folder_untrusted(folder(0o1777, uid=0), mine, owners) == ""
    assert launcher.folder_untrusted(folder(0o1777, uid=0), theirs, owners) == (
        "shared, and the entry in it belongs to another user")
    assert launcher.folder_untrusted(folder(0o1775, gid=81), theirs, owners) == (
        "shared, and the entry in it belongs to another user")
    monkeypatch.setattr(launcher, "user_database_is_local", lambda: False)
    assert launcher.folder_untrusted(folder(0o775), mine, owners) == (
        "writable by its group crew (user database not provably local)")


def _group_folder_verdicts(launcher, monkeypatch, database_is_local):
    """What a folder writable by its group comes to under this user database, for a group of only
    this user and root, then for one with guest in it."""
    monkeypatch.setattr(launcher, "user_database_is_local", lambda: database_is_local)
    monkeypatch.setattr(launcher, "group_name", lambda gid: "crew")
    folder = os.stat_result((stat.S_IFDIR | 0o775, 0, 0, 1, 501, 80, 0, 0, 0, 0))
    entry = os.stat_result((stat.S_IFREG | 0o755, 0, 0, 1, 501, 0, 0, 0, 0, 0))
    verdicts = []
    for others in ([], ["guest"]):
        monkeypatch.setattr(launcher, "other_members", lambda gid, owners, others=others: others)
        verdicts.append(launcher.folder_untrusted(folder, entry, (501, 0)))
    return verdicts


_MEMBERS_RULE = ["", "writable by its group crew, whose members include guest"]
_NOT_PROVABLE = ["writable by its group crew (user database not provably local)"] * 2


@pytest.mark.parametrize("conf, libc, local", [
    ("passwd: files\ngroup: files\n", "glibc 2.39", True),
    ("passwd: files systemd\ngroup: files [SUCCESS=merge] systemd\n", "glibc 2.39", True),
    ("passwd: files sss systemd\ngroup: files sss systemd\n", "glibc 2.39", False),
    ("passwd: files ldap\ngroup: files ldap\n", "glibc 2.39", False),
    ("passwd: compat\ngroup: compat\n", "glibc 2.39", False),  # compat can bring in NIS
    ("passwd: files\n", "glibc 2.39", False),  # no group line: glibc's default is not provably local
    ("passwd: sss files\ngroup: files\npasswd: files\n", "glibc 2.39", False),  # every line counts
    ("passwd: files\ngroup: files\ninitgroups: files sss\n", "glibc 2.39", False),
    (None, "glibc 2.39", False),  # no nsswitch.conf
    ("passwd: files\ngroup: files\n", "", False),  # musl does not read nsswitch.conf
])
def test_on_linux_a_group_folder_passes_only_where_nsswitch_conf_lists_local_users(tmp_path, monkeypatch, conf,
                                                                                    libc, local):
    """The members rule needs every user listed, and a directory service such as SSSD or LDAP may
    not list its users: anything about nsswitch.conf that does not prove the users are local refuses
    a folder writable by its group."""
    launcher = _launcher()
    nsswitch = tmp_path / "nsswitch.conf"
    if conf is not None:
        nsswitch.write_text(conf, encoding="utf-8")
    monkeypatch.setattr(launcher, "libc_name", lambda: libc)
    is_local = launcher.linux_database_is_local(nsswitch)
    assert is_local is local
    assert _group_folder_verdicts(launcher, monkeypatch, is_local) == (_MEMBERS_RULE if local else _NOT_PROVABLE)


@pytest.mark.parametrize("script, timeout, local", [
    ("print('SearchPath: /Local/Default')", 30, True),
    ("print('SearchPath: /Local/Default /BSD/local')", 30, True),
    ("print('SearchPath:\\n /Local/Default\\n /Active Directory/CORP/All Domains')", 30, False),
    ("print('SearchPath: /Local/Default /LDAPv3/ldap.example.com')", 30, False),  # as DHCP can add
    ("print('SearchPath: /Local/Default /BSD/corp')", 30, False),  # an NIS domain
    ("print('CSPSearchPath: /Local/Default')", 30, False),  # the custom policy's path, maybe not in use
    ("print('SearchPath: /Local/Default'); raise SystemExit(1)", 30, False),
    ("print('')", 30, False),
    ("import time; time.sleep(30)", 0.5, False),  # dscl times out
])
def test_on_macos_a_group_folder_passes_only_where_the_search_path_has_local_nodes(monkeypatch, script, timeout,
                                                                                    local):
    """The search path macOS uses now must have only local nodes; dscl's output stands in for it."""
    launcher = _launcher()
    is_local = launcher.macos_database_is_local([sys.executable, "-c", script], timeout)
    assert is_local is local
    assert _group_folder_verdicts(launcher, monkeypatch, is_local) == (_MEMBERS_RULE if local else _NOT_PROVABLE)


def test_other_members_counts_listed_members_and_primary_group_users(monkeypatch):
    """A group's members are its listed members and every user whose primary group it is; only this
    user and root are not others, and a listed name the user database does not know is another."""
    launcher = _launcher()
    users = [SimpleNamespace(pw_name=name, pw_uid=uid, pw_gid=gid)
             for name, uid, gid in (("me", 501, 20), ("root", 0, 0), ("guest", 502, 80), ("bob", 503, 20))]
    groups = {20: [], 80: ["root", "me"], 81: ["me", "ghost"], 82: ["root", "me"]}
    monkeypatch.setattr(launcher.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=groups[gid]))
    monkeypatch.setattr(launcher.pwd, "getpwnam", lambda name: {user.pw_name: user for user in users}[name])
    monkeypatch.setattr(launcher.pwd, "getpwall", lambda: users)
    owners = (501, 0)
    assert launcher.other_members(80, owners) == ["guest"]  # listed: root and me; guest by primary group
    assert launcher.other_members(81, owners) == ["ghost"]
    assert launcher.other_members(82, owners) == []
    assert launcher.other_members(20, owners) == ["bob"]
    with pytest.raises(KeyError):
        launcher.other_members(99, owners)


def test_the_walk_looks_entries_up_as_the_kernel_does(tmp_path, monkeypatch):
    """Every entry on the way, in order: from `/` down, the symlink itself, then its target, with
    `..` stepping back through the resolved path. Each entry's folder is checked when it is looked
    up, so `/` and each symlink's own folder are checked too."""
    launcher = _launcher()
    base = Path(os.path.realpath(tmp_path))
    _executable(base / "b" / "reader")
    (base / "a").mkdir()
    (base / "a" / "link").symlink_to("../b")
    looked_up = []
    lstat = os.lstat
    with monkeypatch.context() as patch:  # undone before tmp_path's cleanup, which calls lstat too
        patch.setattr(launcher.os, "lstat", lambda path: looked_up.append(path) or lstat(path))
        assert launcher.path_untrusted(str(base / "a" / "link" / "reader"), (os.geteuid(), 0)) == ""
    prefixes = [os.path.join("/", *base.parts[1:index + 1]) for index in range(1, len(base.parts))]
    # `..` in a/ names base, and is checked in a/ as any entry is
    assert looked_up == [*prefixes, *(str(base / part) for part in ("a", "a/link", "", "b", "b/reader"))]


# Each program the launcher ends up running appends [which, argv] to LCMX_TEST_RECORD.
_RECORDER = ("import json, os, sys\n"
             "with open(os.environ['LCMX_TEST_RECORD'], 'a') as out:\n"
             "    out.write(json.dumps([%r, sys.argv]) + '\\n')\n")


def _launch(env, tmp_path):
    """Runs the launcher as Claude Code does, in its own process, over a recording plain reader."""
    home, package, record = Path(env["LCMX_CC_HOME"]), _package(tmp_path), tmp_path / "started.jsonl"
    (package / "scripts" / "lcm_x_mcp.py").write_text(_RECORDER % "plain", encoding="utf-8")
    lane = {"PATH": os.environ.get("PATH", ""), "LCMX_CC_HOME": str(home), "LCMX_PYTHON": sys.executable,
            "LCMX_PACKAGE_ROOT": str(package), "LCMX_TEST_RECORD": str(record)}
    result = subprocess.run([sys.executable, str(PLUGIN / "bin" / "lcmx_mcp.py")], env=lane,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    started = record.read_text(encoding="utf-8").splitlines() if record.exists() else []
    return result, [json.loads(line) for line in started]


def test_a_reader_override_runs_and_its_reader_command_starts_the_plain_reader(env, tmp_path):
    """The real exec: the override's process starts, and the argv it gets in LCMX_LOCAL_READER_CMD
    starts the plain reader, as the override's own fallback would. The command is a script this
    test owns: the interpreter running the tests can be group-writable, as on CI's runners, and
    the launcher refuses such a command."""
    Path(env["LCMX_CC_HOME"]).mkdir(parents=True)
    override = tmp_path / "override.py"
    override.write_text(_RECORDER % "override" + "command = json.loads(os.environ['LCMX_LOCAL_READER_CMD'])\n"
                        "os.execv(command[0], command)\n", encoding="utf-8")
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.write_text('#!/bin/sh\nexec %s "$@"\n' % shlex.quote(sys.executable), encoding="utf-8")
    python.chmod(0o755)
    _write_override(env, {"version": 1, "command": [str(python), str(override), "--flag"]})
    result, started = _launch(env, tmp_path)
    assert (result.returncode, result.stderr) == (0, "")
    assert started == [["override", [str(override), "--flag"]], ["plain", _plain_reader(env, tmp_path)[1:]]]


@pytest.mark.parametrize("program", ["#!/nonexistent/interpreter\n", "not a program\n"],
                         ids=["missing_interpreter", "no_executable_format"])
def test_a_reader_override_that_does_not_start_falls_back_to_the_plain_reader(env, tmp_path, program):
    """The one start failure an override cannot catch itself: the exec raises OSError, ENOENT for
    a shebang naming a missing interpreter and ENOEXEC for a file with no executable format."""
    Path(env["LCMX_CC_HOME"]).mkdir(parents=True)
    script = tmp_path / "override" / "reader"
    script.parent.mkdir()
    script.write_text(program, encoding="utf-8")
    script.chmod(0o755)
    _write_override(env, {"version": 1, "command": [str(script)]})
    result, started = _launch(env, tmp_path)
    assert result.returncode == 0
    assert started == [["plain", _plain_reader(env, tmp_path)[1:]]]
    (line,) = result.stderr.splitlines()
    assert "the command did not start" in line and line.endswith("; starting the plain reader")


def test_a_fifo_at_the_override_path_starts_the_plain_reader_at_once(env, tmp_path):
    """Opening a FIFO waits for a writer, and a launcher that waited would start no reader. The
    child's timeout bounds a regression."""
    Path(env["LCMX_CC_HOME"]).mkdir(parents=True)
    os.mkfifo(Path(env["LCMX_CC_HOME"]) / "reader-override.json")
    result, started = _launch(env, tmp_path)
    assert result.returncode == 0
    assert started == [["plain", _plain_reader(env, tmp_path)[1:]]]
    (line,) = result.stderr.splitlines()
    assert "not a regular file" in line and line.endswith("; starting the plain reader")


def test_a_precompact_drain_never_waits_on_a_held_lock(hook, env, tmp_path, monkeypatch):
    """REGRESSION: the bounded PreCompact drain waited up to its deadline for a lock another
    worker held; now the hook returns at once and leaves the spooled record to the holder."""
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(_package(tmp_path))}
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("a held lock must not start a drain"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("the holder takes the record"))
    with open(home / "capture.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        assert hook.start_worker(home, lane, deadline_s=10) == "worker_running"
        assert time.monotonic() - started < 1


def test_a_precompact_drain_cut_at_its_timeout_hands_the_rest_to_a_detached_worker(hook, env, tmp_path, monkeypatch):
    """The hook kills a bounded drain at its own timeout (``drain_timeout``), before Claude Code's
    hook timeout: the batch in flight rolls back and the record stays (the store tests), and a
    detached worker takes the rest at once."""
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    assert hook.PRECOMPACT_DEADLINE_S + 2 < hooks["PreCompact"][0]["hooks"][0]["timeout"]
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(_package(tmp_path))}
    bounded, detached = [], []

    def timed_out(command, *, timeout, **kwargs):
        bounded.append((command, timeout))
        raise subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(subprocess, "run", timed_out)
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: detached.append(command))
    assert hook.start_worker(home, lane, deadline_s=hook.PRECOMPACT_DEADLINE_S) == "drain_timeout+worker"
    assert [timeout for _command, timeout in bounded] == [hook.PRECOMPACT_DEADLINE_S + 2]
    assert "--deadline" in bounded[0][0] and len(detached) == 1 and "--deadline" not in detached[0]


def test_a_cut_precompact_drain_leaves_the_rest_to_the_worker_that_took_the_lock(hook, env, tmp_path, monkeypatch):
    """A bounded drain that ends incomplete starts a detached worker only when the lock is free.
    When another worker has taken it since, that worker rescans, and the outcome says so
    (``+holder``, never ``+worker``)."""
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(_package(tmp_path))}
    with open(home / "capture.lock", "a") as handle:
        def incomplete_then_taken(command, *, timeout, **kwargs):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another worker takes the lock meanwhile
            return subprocess.CompletedProcess(command, 3)
        monkeypatch.setattr(subprocess, "run", incomplete_then_taken)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("the holder takes the rest"))
        assert hook.start_worker(home, lane, deadline_s=hook.PRECOMPACT_DEADLINE_S) == "drain_incomplete+holder"


def test_a_refused_grok_home_writes_nothing_into_the_claude_home(hook, tmp_path):
    """#2531 review: a Grok payload whose home setting is refused (relative) logs nowhere, not in Claude Code's home."""
    env = {"LCMX_CC_HOME": str(tmp_path / "home"), "LCMX_GROK_HOME": "relative/grok",
           "CLAUDE_CONFIG_DIR": str(tmp_path / "config")}
    payload = {"hookEventName": "SessionStart", "sessionId": "g-1", "cwd": "/work", "source": "startup"}
    assert hook.main(io.StringIO(json.dumps(payload)), env) == 0
    assert not (tmp_path / "home").exists()


def test_an_unresolvable_home_never_escapes_the_hook(hook, tmp_path):
    """#2531 review: a home that raises while resolving (an unknown ~user) still fails open, on a synchronous hook."""
    env = {"LCMX_CC_HOME": "~no-such-user-2531/home", "CLAUDE_CONFIG_DIR": str(tmp_path / "config")}
    for raw in (json.dumps({"hook_event_name": "SessionStart", "session_id": "s1",
                            "transcript_path": str(_transcript(tmp_path))}), "not json"):
        assert hook.main(io.StringIO(raw), env) == 0


@pytest.mark.parametrize("event", ["Stop", "SessionEnd"])
def test_the_hook_returns_only_after_its_writer_left_the_group(env, tmp_path, event):
    """#2531 review: the parent waits for the child's setsid, so a group kill the moment the hook returns cannot reach
    the writer. setsid is slowed here; without the wait the kill lands first and no pointer is spooled."""
    import signal
    payload = {"hook_event_name": event, "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    driver = ("import importlib.util, os, sys, time\n"
              f"spec = importlib.util.spec_from_file_location('h', {str(PLUGIN / 'bin/lcmx_hook.py')!r})\n"
              "h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
              "real = os.setsid\n"
              "os.setsid = lambda: (time.sleep(0.3), real())[1]\n"
              "sys.exit(h.main(sys.stdin, os.environ, detach=True))\n")
    proc = subprocess.Popen([sys.executable, "-c", driver], env={**os.environ, **env}, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    proc.communicate(json.dumps(payload).encode(), timeout=5)
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # the moment the hook returned, as a host cleaning up its group would
    except ProcessLookupError:
        pass
    row = _wait_hooks(env, 1)[-1]  # a writer the kill reached never logs (the worker may already have drained the pointer)
    assert row["detached"] is True and row["event"] == event and not row["outcome"].startswith("error")
    assert row["blocking_ms"] >= 300  # the parent's wait for the slowed setsid is part of what it blocked


def test_a_detached_writer_logs_a_failure_before_its_record_is_built(env, tmp_path):
    """#2531 review: a detached child that fails before normalize returns still logs its error in the environment's
    home, as main and the earlier head did."""
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    driver = ("import importlib.util, os, sys\n"
              f"spec = importlib.util.spec_from_file_location('h', {str(PLUGIN / 'bin/lcmx_hook.py')!r})\n"
              "h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
              "def boom(*a, **k): raise RuntimeError('no such user')\n"
              "h.normalize = boom\n"
              "sys.exit(h.main(sys.stdin, os.environ, detach=True))\n")
    done = subprocess.run([sys.executable, "-c", driver], env={**os.environ, **env}, input=json.dumps(payload),
                          capture_output=True, text=True, timeout=5)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    row = _wait_hooks(env, 1)[-1]
    assert row["detached"] is True and row["outcome"] == "error:RuntimeError"


def test_a_failed_setsid_still_releases_the_parent_before_logging(env, tmp_path):
    """#2531 review: the handshake's write end closes even if setsid raises, so a slow log cannot hold the parent."""
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    driver = ("import importlib.util, os, sys, time\n"
              f"spec = importlib.util.spec_from_file_location('h', {str(PLUGIN / 'bin/lcmx_hook.py')!r})\n"
              "h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)\n"
              "def boom(): raise OSError('setsid')\n"
              "os.setsid = boom\n"
              "log = h._log\n"
              "h._log = lambda *a, **k: (time.sleep(3), log(*a, **k))\n"
              "sys.exit(h.main(sys.stdin, os.environ, detach=True))\n")
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", driver], env={**os.environ, **env}, input=json.dumps(payload),
                          capture_output=True, text=True, timeout=10, start_new_session=True)
    assert done.returncode == 0 and time.monotonic() - started < 2


def test_a_failed_fork_leaks_no_handshake_descriptor(hook, env, tmp_path, monkeypatch):
    """#2531 review: a fork that fails after its pipe was made closes both ends and runs the hook in place."""
    def boom():
        raise OSError("fork")
    monkeypatch.setattr(hook.os, "fork", boom)
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    before = len(os.listdir("/dev/fd"))
    assert hook.main(io.StringIO(json.dumps(payload)), env, detach=True) == 0
    assert len(os.listdir("/dev/fd")) == before
    assert _log(env)[-1]["outcome"].endswith("+nofork")


def test_records_carry_a_hashed_prompt_id_and_the_raw_agent_type(hook, env, tmp_path):
    """#2535: the audit tells the compaction summarizer (agent_type "") from other subagents, and links it to its
    compaction by prompt id. Only a hash of the id is kept; an absent type stays None, apart from ""."""
    path = str(_transcript(tmp_path))
    summarizer = hook.normalize({"hook_event_name": "SubagentStop", "session_id": "s1", "agent_id": "a1",
                                 "agent_type": "", "prompt_id": "p-1", "transcript_path": path}, env)
    assert summarizer["agent_type"] == "" and summarizer["prompt_id"] == hashlib.sha256(b"p-1").hexdigest()[:16]
    plain = hook.normalize({"hook_event_name": "Stop", "session_id": "s1", "transcript_path": path}, env)
    assert plain["agent_type"] is None and plain["prompt_id"] is None
    assert "p-1" not in json.dumps(summarizer)
    # What the hook found on disk: the session transcript exists, the summarizer's never does; no agent path is None.
    assert summarizer["transcript_seen"] is True and summarizer["agent_transcript_seen"] is None
    absent = hook.normalize({"hook_event_name": "SubagentStop", "session_id": "s1", "agent_id": "a1",
                             "agent_transcript_path": str(Path(path).parent / "s1" / "subagents" / "agent-a1.jsonl"),
                             "transcript_path": path}, env)
    assert absent["agent_transcript_seen"] is False
    assert hook._on_disk(str(tmp_path / "no-such-dir" / "x.jsonl")) is False
    assert hook._on_disk(path + "/below-a-file.jsonl") is False  # NotADirectoryError
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0)
    try:
        assert os.geteuid() == 0 or hook._on_disk(str(blocked / "x.jsonl")) is None  # a failed stat is unknown
    finally:
        blocked.chmod(0o700)


def test_a_replaced_or_kept_pointer_keeps_every_event_seen(hook, env, tmp_path):
    """#2535: the pointer rules replace a record or keep it out, but the claimed pointer still lists its event (and a
    compaction's prompt id), so the audit never mistakes a session that had a turn for one with no prompt."""
    from hermes_lcm.claude_code_store import claim_records
    home = Path(env["LCMX_CC_HOME"])
    base = hook.normalize({"hook_event_name": "Stop", "session_id": "s1",
                           "transcript_path": str(_transcript(tmp_path))}, env)
    for event in ("UserPromptSubmit", "Stop", "SessionEnd"):
        hook.spool(home, {**base, "event": event})
    assert hook.spool(home, {**base, "event": "PostToolUse"}) == "kept_end"
    assert hook.spool(home, {**base, "event": "PreCompact", "prompt_id": "c1"}) == "spooled"
    assert hook.spool(home, {**base, "event": "PostCompact", "prompt_id": "c2"}) == "kept_end"
    record = claim_records(home)[0][1]
    assert record["event"] == "PreCompact" and record["transcript_seen"] is True
    assert record["seen"] == ["PostCompact", "PostToolUse", "PreCompact", "SessionEnd", "Stop", "UserPromptSubmit"]
    assert record["compact_prompts"] == ["c1", "c2"]


@pytest.mark.parametrize("values,merged", [((False, False), False), ((False, True), True), ((True, False), True),
                                           ((None, False), None), ((False, None), None), ((None, True), True)])
def test_the_pointer_keeps_whether_any_record_saw_the_transcript(hook, env, tmp_path, values, merged):
    """#2535: once any hook found the transcript on disk the lineage is never expired; an unknown (an older hook) is
    never turned into "absent"."""
    from hermes_lcm.claude_code_store import claim_records
    home = Path(env["LCMX_CC_HOME"])
    base = hook.normalize({"hook_event_name": "SessionStart", "session_id": "s1",
                           "transcript_path": str(_transcript(tmp_path))}, env)
    for event, seen in zip(("SessionStart", "SessionEnd"), values):
        hook.spool(home, {**base, "event": event, "transcript_seen": seen})
    assert claim_records(home)[0][1]["transcript_seen"] is merged
