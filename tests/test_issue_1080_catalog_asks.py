"""Hermes catalog asks: public hooks, bounded embeddings, plugin-owned markers."""

import json
import logging
import os
from pathlib import Path
import sys
import types

from hermes_lcm import inactive_record as records
from tests.test_issue_622_slow_load import _BareCtx, _install_host, _load_plugin


def test_write_record_under_plugin_data_only(tmp_path):
    records.write_inactive_record(tmp_path, elapsed_s=12.5, reason="catalog test")
    directory = tmp_path / "plugin-data" / "hermes-lcm-x"
    paths = list(directory.glob("lcm-x-not-active.*.json"))
    assert len(paths) == 1
    assert paths[0].name == f"lcm-x-not-active.{os.getpid()}.json"
    assert json.loads(paths[0].read_text())["reason"] == "catalog test"
    assert list(directory.iterdir()) == paths  # no leftover temporary file
    assert not list(tmp_path.glob("lcm-x-not-active*.json"))


def test_notice_reports_live_record_in_plugin_data(tmp_path):
    directory = tmp_path / "plugin-data" / "hermes-lcm-x"
    directory.mkdir(parents=True)
    path = directory / f"lcm-x-not-active.{os.getpid()}.json"
    path.write_text(json.dumps({"pid": os.getpid(), "reason": "new location"}))
    assert records.inactive_record_notice(tmp_path) == (
        f"LCM-X was not active in process {os.getpid()} (new location)"
    )
    assert path.exists()


def test_legacy_per_pid_record_reports_live_and_unlinks_unchanged_dead(tmp_path, monkeypatch):
    path = tmp_path / "lcm-x-not-active.101.json"
    path.write_text(json.dumps({"pid": 101, "reason": "legacy location"}))
    monkeypatch.setattr(records, "_process_alive", lambda pid, started: True)
    assert records.inactive_record_notice(tmp_path) == (
        "LCM-X was not active in process 101 (legacy location)"
    )
    assert path.exists()
    monkeypatch.setattr(records, "_process_alive", lambda pid, started: False)
    assert records.inactive_record_notice(tmp_path) is None
    assert not path.exists()


def test_missing_register_hook_leaves_private_host_hooks_unchanged(tmp_path, monkeypatch, caplog):
    _install_host(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    manager = types.SimpleNamespace(_hooks={"post_llm_call": [object()]})
    before = {name: list(hooks) for name, hooks in manager._hooks.items()}
    plugins = types.ModuleType("hermes_cli.plugins")
    plugins.get_plugin_manager = lambda: manager
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", plugins)
    module = _load_plugin("hermes_lcm_1080_no_hook_api")
    ctx = _BareCtx()
    caplog.set_level(logging.DEBUG)
    try:
        module.register(ctx)
        assert manager._hooks == before
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING
                    and "the per-turn ingest hook is not registered" in r.getMessage()]
        assert len(warnings) == 1  # visible, not debug-only (#1081 review)
    finally:
        if ctx.offered is not None:
            ctx.offered.shutdown()


def test_exactly_one_bounded_fastembed_requirement():
    requirement = Path(__file__).resolve().parents[1] / "requirements-semantic.txt"
    lines = [line.strip() for line in requirement.read_text().splitlines()
             if line.strip().startswith("fastembed")]
    assert lines == ["fastembed>=0.8.0,<1.0"]
