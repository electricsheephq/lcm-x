"""Claude Code adapter skeleton: the hook entry's contract and the projection-v1 classes.

Design: docs/claude-code-adapter.md. The hook entry must never fail or print, must only
spool transcripts under the Claude Code projects directory, and must keep Grok Build's
session-file capture off until its own proof.
"""
from __future__ import annotations

import importlib.util
import io
import json
import stat
from pathlib import Path

import pytest

from hermes_lcm.claude_code_capture import (
    PROJECTED,
    RAW_ONLY,
    SHARED_RAW,
    UNKNOWN,
    classify_entry,
    lineage_key,
    row_session_id,
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
    """A failed or incomplete drain is never reported as drained."""
    package = tmp_path / "package"
    (package / "scripts").mkdir(parents=True)
    (package / "plugin.yaml").write_text("", encoding="utf-8")
    (package / "store.py").write_text("", encoding="utf-8")
    worker = package / "scripts" / "lcm_x_capture.py"
    import sys
    lane = {**env, "LCMX_PYTHON": sys.executable, "LCMX_PACKAGE_ROOT": str(package)}
    Path(env["LCMX_CC_HOME"]).mkdir(parents=True)
    for code, outcome in ((0, "drained"), (3, "drain_incomplete"), (1, "drain_failed:1")):
        worker.write_text(f"raise SystemExit({code})\n", encoding="utf-8")
        assert hook.start_worker(Path(env["LCMX_CC_HOME"]), lane, deadline_s=5) == outcome


def test_spool_record_carries_seq_and_local_gc(hook, env, tmp_path):
    payload = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(_transcript(tmp_path))}
    hook.main(io.StringIO(json.dumps(payload)), {**env, "CLAUDE_CODE_TRANSCRIPT_LOCAL_GC": "1"})
    [record_path] = _spooled(env)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert isinstance(record["seq"], int) and record["local_gc"] is True
