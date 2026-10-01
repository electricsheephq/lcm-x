"""Inactive peer diagnostics and the registered slot-conflict surfaces (#696)."""

import json
import os
from pathlib import Path
import sys
import types

import pytest

from hermes_lcm import inactive_record as records
from hermes_lcm import tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_622_slow_load import _Ctx, _Manager, _OtherEngine, _install_host, _load_plugin


SLOW = "plugin load took 12.5 s, over plugins.load_timeout_seconds"
SLOT = "context engine slot held by OtherEngine"


def _write_fake(home, monkeypatch, pid, reason=SLOW):
    with monkeypatch.context() as patch:
        patch.setattr(records.os, "getpid", lambda: pid)
        patch.setattr(records, "_process_start", lambda value: f"start-{value}")
        records.write_inactive_record(home, elapsed_s=12.5, reason=reason)


@pytest.mark.parametrize("exits_first", [101, 202])
def test_overlapping_processes_keep_the_survivor(tmp_path, monkeypatch, exits_first):
    _write_fake(tmp_path, monkeypatch, 101)
    _write_fake(tmp_path, monkeypatch, 202, SLOT)
    assert len(list(tmp_path.glob("lcm-x-not-active.*.json"))) == 2
    monkeypatch.setattr(records.os, "kill", lambda pid, signal: None)
    starts = {101: "start-101", 202: "start-202"}
    monkeypatch.setattr(records, "_process_start", starts.get)
    notice = records.inactive_record_notice(tmp_path)
    assert "process 101" in notice and "process 202" in notice
    starts[exits_first] = "reused-pid"
    survivor = 202 if exits_first == 101 else 101
    notice = records.inactive_record_notice(tmp_path)
    assert f"process {survivor}" in notice
    assert f"process {exits_first}" not in notice
    assert not (tmp_path / f"lcm-x-not-active.{exits_first}.json").exists()


@pytest.mark.parametrize("filename", ["lcm-x-not-active.json", "lcm-x-not-active.101.json"])
def test_rewritten_dead_record_is_kept(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    path.write_text(json.dumps({"pid": 101, "process_start": "old"}))
    replacement = json.dumps({"pid": 101, "process_start": "new", "reason": SLOT})

    def dead_after_rewrite(pid, started):
        path.write_text(replacement)
        return False

    monkeypatch.setattr(records, "_process_alive", dead_after_rewrite)
    assert records.inactive_record_notice(tmp_path) is None
    assert path.read_text() == replacement


def test_legacy_record_is_read_and_dead_record_removed(tmp_path, monkeypatch):
    path = tmp_path / "lcm-x-not-active.json"
    path.write_text(json.dumps({"pid": 101, "process_start": "start-101", "reason": SLOW}))
    monkeypatch.setattr(records, "_process_alive", lambda pid, started: True)
    assert "process 101" in records.inactive_record_notice(tmp_path)
    monkeypatch.setattr(records, "_process_alive", lambda pid, started: False)
    assert records.inactive_record_notice(tmp_path) is None
    assert not path.exists()


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(tmp_path))
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.mark.parametrize("reason,action", [(SLOW, "plugins.load_timeout_seconds"), (SLOT, "context.engine: lcm-x")])
def test_doctor_reports_live_peer_and_guidance(engine, monkeypatch, reason, action):
    records.write_inactive_record(engine._hermes_home, elapsed_s=12.5, reason=reason)
    monkeypatch.setattr(records, "_process_alive", lambda pid, started: True)
    notice = records.inactive_record_notice(engine._hermes_home)
    payload = json.loads(tools.lcm_doctor({}, engine=engine))
    check = next(c for c in payload["checks"] if c["check"] == "inactive_process")
    assert check == {"check": "inactive_process", "status": "warn", "detail": notice}
    assert payload["overall"] == "warnings"
    guidance = next(g for g in payload["guidance"] if g["check"] == "inactive_process")
    assert guidance["warning_only"] is True
    assert action in guidance["operator_action"] and "restart" in guidance["operator_action"].lower()
    text = handle_lcm_command("doctor", engine)
    assert "status: issues-found" in text
    assert "inactive_process" in next(line for line in text.splitlines() if line.startswith("issues:"))
    assert guidance["operator_action"] in text.split("recommended_actions:\n", 1)[1].split("triage_guidance:", 1)[0]


def test_doctor_without_record_passes(engine):
    payload = json.loads(tools.lcm_doctor({}, engine=engine))
    assert next(c for c in payload["checks"] if c["check"] == "inactive_process")["status"] == "pass"
    assert payload["overall"] == "healthy"


@pytest.mark.parametrize("fallback", [False, True], ids=["registrar", "hook-table"])
def test_slot_conflict_hook_does_not_write_and_diagnostics_answer(tmp_path, monkeypatch, fallback):
    _install_host(monkeypatch, tmp_path)
    monkeypatch.setenv("LCM_ENABLE_SLASH_COMMAND", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    module = _load_plugin(f"hermes_lcm_696_{fallback}")
    ctx = _Ctx(_Manager(existing=_OtherEngine()))
    ctx.tools = {}
    ctx.commands = {}
    ctx.register_tool = lambda **kw: ctx.tools.update({kw["name"]: kw["handler"]})
    ctx.register_command = lambda name, handler, **kw: ctx.commands.update({name: handler})
    monkeypatch.setattr(module, "_host_forwards_registered_tool_messages", lambda ctx: True)
    if fallback:
        ctx.register_hook = None
        plugin_manager = types.ModuleType("hermes_cli.plugins")
        plugin_manager.get_plugin_manager = lambda: types.SimpleNamespace(_hooks=ctx.hooks)
        monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugin_manager)
    try:
        module.register(ctx)
        conn = ctx.offered._store.connection
        before = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert before == 0
        for hook in ctx.hooks["post_llm_call"]:
            hook(session_id="s-696", conversation_id="c-696", conversation_history=[
                {"role": "user", "content": "a saved turn"},
                {"role": "assistant", "content": "a reply"},
            ])
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == before
        assert "inactive_process" in json.loads(ctx.tools["lcm_status"]({}))
        assert json.loads(ctx.tools["lcm_doctor"]({}))["overall"] == "warnings"
        assert "inactive_process" in ctx.commands["lcm"]("doctor")
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == before
        assert Path(tmp_path / f"lcm-x-not-active.{os.getpid()}.json").exists()
    finally:
        ctx.offered.shutdown()
