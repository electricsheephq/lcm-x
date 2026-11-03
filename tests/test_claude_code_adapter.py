"""Claude Code adapter skeleton: the hook entry's contract and the projection-v1 classes.

Design: docs/claude-code-adapter.md. The hook entry must never fail or print, must only
spool transcripts under the Claude Code projects directory, and must keep Grok Build's
session-file capture off until its own proof.
"""
from __future__ import annotations

import fcntl
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

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


def test_grok_build_uses_its_own_home_and_never_spools_observe_only_events(hook, env):
    """One owning harness per store; a session-end event cannot replace a pending record."""
    grok_env = {**env, "GROK_SESSION_ID": "g-1", "LCMX_GROK_CAPTURE": "1"}
    stop = {"hookEventName": "Stop", "sessionId": "g-1", "cwd": "/work", "reason": "end_turn"}
    assert hook.main(io.StringIO(json.dumps(stop)), grok_env) == 0
    assert _spooled(env) == []
    [record_path] = _spooled(env, "LCMX_GROK_HOME")
    assert json.loads(record_path.read_text(encoding="utf-8"))["observe_only"] is False
    for event, reason in (("SessionEnd", None), ("Stop", "session_end")):
        payload = {"hookEventName": event, "sessionId": "g-1", "cwd": "/work", "reason": reason}
        assert hook.main(io.StringIO(json.dumps(payload)), grok_env) == 0
        assert _log(env, "LCMX_GROK_HOME")[-1]["outcome"] == "observed"
    assert json.loads(record_path.read_text(encoding="utf-8"))["event"] == "Stop"
    assert json.loads(record_path.read_text(encoding="utf-8"))["reason"] == "end_turn"


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
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    servers = json.loads((PLUGIN / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"]
    assert manifest["name"] == "lcm-x"
    assert set(hooks) == set(hook.EVENTS) - {"PostToolUse"}  # accepted for Grok Build only
    assert hook.ASYNC_EVENTS == {"PostToolBatch"}
    for event, entries in hooks.items():
        for entry in entries:
            # The shell form, which every host runs: Grok Build's runner is not yet proven to honour
            # an exec form's args. -I -S skips site and ignores the user's Python settings.
            assert entry["hooks"][0]["command"] == 'python3 -I -S "${CLAUDE_PLUGIN_ROOT}/bin/lcmx_hook.py"'
            # Only PostToolBatch runs in the background: the worker settles on the end events, a fork's
            # start source rides on SessionStart's record, and -p kills an async hook still running at exit.
            assert entry["hooks"][0].get("async", False) is (event in hook.ASYNC_EVENTS)
    assert set(servers) == {"lcm-x"}


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
        ("PostToolBatch", lane, [], 0, "worker_started", spooled + ["subprocess", "start", "log"]),
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
    The registered command runs isolated, so the hook still spools and prints nothing."""
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]
    [command] = {entry["hooks"][0]["command"] for entries in hooks.values() for entry in entries}
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "json.py").write_text("raise SystemExit('a planted module loaded')\n", encoding="utf-8")
    python = tmp_path / "bin" / "python3"
    python.parent.mkdir()
    python.symlink_to(sys.executable)
    home = Path(env["LCMX_CC_HOME"])
    home.mkdir(parents=True)
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    lane = {**env, "PATH": str(python.parent), "CLAUDE_PLUGIN_ROOT": str(PLUGIN), setting: str(planted)}
    with open(home / "capture.lock", "a") as handle:  # a worker is running, so the hook starts none
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = subprocess.run(["/bin/sh", "-c", command], input=json.dumps(payload), env=lane,
                              capture_output=True, text=True, timeout=60)
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")
    assert len(_spooled(env)) == 1
    assert _log(env)[-1]["outcome"] == "worker_running"


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
