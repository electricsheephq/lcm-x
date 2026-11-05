"""Leased-home opt-in and deadlines, with no auth file operations or live child."""
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from bench.fmp import models


def test_defaults_unchanged_without_touching_auth(tmp_path, monkeypatch):
    symlink = Mock()
    monkeypatch.setattr(Path, "symlink_to", symlink)
    backend = models.Codex(tmp_path, models.Quota(tmp_path / "quota.json"))
    try:
        assert backend.timeouts == (600, 1800)
        assert backend.include_total_tokens is False
        assert backend.home == backend.scratch / "codex-home"
        symlink.assert_called_once_with(Path.home() / ".codex" / "auth.json")
    finally:
        backend.close()


@pytest.mark.parametrize("include_total,reported_total", [(False, 12), (True, 12), (True, None)])
def test_lease_passed_to_child_and_rollout_preserved(tmp_path, monkeypatch, include_total, reported_total):
    # Any accidental return to the default auth setup fails before making a symlink.
    monkeypatch.setattr(Path, "symlink_to", Mock(side_effect=AssertionError("no symlinks")))
    home = tmp_path / "lease"
    home.mkdir()
    rollout = home / "sessions" / "rollout-test-thread.jsonl"
    rollout.parent.mkdir()
    rollout.write_text(json.dumps({"type": "turn_context", "payload": {"model": "gpt-6.1-sol"}}) + "\n")

    def runner(cmd, **kwargs):
        assert kwargs["env"]["CODEX_HOME"] == str(home)
        assert kwargs["timeout"] == 540
        assert kwargs["stdin"].read() == "system\n\nquestion"
        assert "--ephemeral" not in cmd and "model_reasoning_effort=low" in cmd
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, [
            {"type": "thread.started", "thread_id": "thread"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
            {"type": "turn.completed", "usage": {"output_tokens": 2,
                **({"total_tokens": reported_total} if reported_total is not None else {})}},
        ])), "")

    backend = models.Codex(tmp_path / "run", models.Quota(tmp_path / "q.json"),
                           codex_home=home, include_total_tokens=include_total,
                           runner=runner, timeout=(540, 1800))
    try:
        records = []
        assert backend.call("codex:gpt-6.1-sol@low", [
            {"role": "system", "content": "system"}, {"role": "user", "content": "question"}], records)["content"] == "answer"
        assert records[0]["input_tokens"] is None and records[0]["output_tokens"] == 2
        assert records[0]["returned_model"] == "gpt-6.1-sol"
        assert records[0]["transcripts"][1] == str(rollout)
        if include_total and reported_total is not None:
            assert records[0]["total_tokens"] == reported_total
        else:
            assert "total_tokens" not in records[0]
    finally:
        backend.close()
    assert rollout.exists() and not backend.scratch.exists()


def test_timeout_forwarded_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "symlink_to", Mock(side_effect=AssertionError("no symlinks")))

    def runner(cmd, **kwargs):
        assert kwargs["timeout"] == 540
        raise subprocess.TimeoutExpired(cmd, 540, output=b"synthetic partial event")

    backend = models.Codex(tmp_path, models.Quota(tmp_path / "q.json"), codex_home=tmp_path / "lease",
                           timeout=(540, 1800), runner=runner)
    try:
        records = []
        with pytest.raises(RuntimeError, match="codex_call_failed"):
            backend.call("codex:gpt-6.1-sol@low", [{"role": "user", "content": "question"}], records)
        assert records[-1]["error"] == "timeout"
        assert not list((tmp_path / "harness").rglob("prompt.txt"))
    finally:
        backend.close()
