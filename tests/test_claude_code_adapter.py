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
    for entries in hooks.values():
        for entry in entries:
            assert entry["hooks"][0]["command"].endswith('bin/lcmx_hook.py"')
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
    popen, starts = hook.subprocess.Popen, []

    def counting(*args, **kwargs):  # counts a detached start even if its worker has not run yet
        if kwargs.get("start_new_session"):
            starts.append(args)
        return popen(*args, **kwargs)
    monkeypatch.setattr(hook.subprocess, "Popen", counting)
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
    monkeypatch.setattr(hook.subprocess, "Popen", lambda command, **kwargs: started.append(command))
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
    ("group", "which is writable by its group, whose members include eve"),
    ("unknown_group", "which is writable by a group the user database does not list"),
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
    shared.chmod(0o775 if case in ("group", "unknown_group") else 0o777)
    if case == "group":
        monkeypatch.setattr(start_reader.launcher, "other_members", lambda gid, owners: ["eve"])
    elif case == "unknown_group":
        def unlisted(gid, owners):
            raise KeyError(gid)

        monkeypatch.setattr(start_reader.launcher, "other_members", unlisted)
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
    members = {80: [], 81: ["eve"]}

    def other_members(gid, owners):
        return members[gid]  # a KeyError, as for a group the user database does not list

    monkeypatch.setattr(launcher, "other_members", other_members)

    def folder(mode, uid=501, gid=80):
        return os.stat_result((stat.S_IFDIR | mode, 0, 0, 1, uid, gid, 0, 0, 0, 0))

    mine, theirs = (os.stat_result((stat.S_IFREG | 0o755, 0, 0, 1, uid, 0, 0, 0, 0, 0)) for uid in (501, 502))
    owners = (501, 0)
    assert launcher.folder_untrusted(folder(0o755), mine, owners) == ""
    assert launcher.folder_untrusted(folder(0o755, uid=0), theirs, owners) == ""
    assert launcher.folder_untrusted(folder(0o755, uid=502), mine, owners) == "not owned by this user or root"
    assert launcher.folder_untrusted(folder(0o775), mine, owners) == ""
    assert launcher.folder_untrusted(folder(0o775, gid=81), mine, owners) == (
        "writable by its group, whose members include eve")
    assert launcher.folder_untrusted(folder(0o775, gid=99), mine, owners) == (
        "writable by a group the user database does not list")
    assert launcher.folder_untrusted(folder(0o757), mine, owners) == "writable by others"
    assert launcher.folder_untrusted(folder(0o1777, uid=0), mine, owners) == ""
    assert launcher.folder_untrusted(folder(0o1777, uid=0), theirs, owners) == (
        "shared, and the entry in it belongs to another user")
    assert launcher.folder_untrusted(folder(0o1775, gid=81), theirs, owners) == (
        "shared, and the entry in it belongs to another user")


def test_other_members_counts_listed_members_and_primary_group_users(monkeypatch):
    """A group's members are its listed members and every user whose primary group it is; only this
    user and root are not others, and a listed name the user database does not know is another."""
    launcher = _launcher()
    users = [SimpleNamespace(pw_name=name, pw_uid=uid, pw_gid=gid)
             for name, uid, gid in (("me", 501, 20), ("root", 0, 0), ("eve", 502, 80), ("bob", 503, 20))]
    groups = {20: [], 80: ["root", "me"], 81: ["me", "ghost"], 82: ["root", "me"]}
    monkeypatch.setattr(launcher.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=groups[gid]))
    monkeypatch.setattr(launcher.pwd, "getpwnam", lambda name: {user.pw_name: user for user in users}[name])
    monkeypatch.setattr(launcher.pwd, "getpwall", lambda: users)
    owners = (501, 0)
    assert launcher.other_members(80, owners) == ["eve"]  # listed: root and me; eve by primary group
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
    monkeypatch.setattr(hook.subprocess, "run", lambda *a, **k: pytest.fail("a held lock must not start a drain"))
    monkeypatch.setattr(hook.subprocess, "Popen", lambda *a, **k: pytest.fail("the holder takes the record"))
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
        raise hook.subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(hook.subprocess, "run", timed_out)
    monkeypatch.setattr(hook.subprocess, "Popen", lambda command, **kwargs: detached.append(command))
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
            return hook.subprocess.CompletedProcess(command, 3)
        monkeypatch.setattr(hook.subprocess, "run", incomplete_then_taken)
        monkeypatch.setattr(hook.subprocess, "Popen", lambda *a, **k: pytest.fail("the holder takes the rest"))
        assert hook.start_worker(home, lane, deadline_s=hook.PRECOMPACT_DEADLINE_S) == "drain_incomplete+holder"
