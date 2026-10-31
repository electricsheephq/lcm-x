"""Round 4 synthetic regressions: transport."""
import io
import json
import os
import shlex
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.fmp import delivery, judges, mcp_reader, models, stores


@pytest.mark.parametrize("text,expected", [("Yes.", True), ("No, it misses", False),
    ("**Yes**", True), ("Yes\nThe answer matches.", True)])
def test_valid_lme_verdict(text, expected):
    assert judges.parse(text, "lme-s") is expected
    assert judges.parse("Yes or no? No.", "lme-s") is None



def test_hook_quotes_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(models.sys, "executable", str(tmp_path / "python space"))
    monkeypatch.setattr(models, "DENY_HOOK", tmp_path / "hook space.py")
    backend = models.Claude(tmp_path, models.Quota(tmp_path / "quota"), token=lambda: "fixture")
    try:
        call = tmp_path / "call space;echo nope"
        cmd = backend.command("model", "system", call, {"command": "fixture"})
        hook = json.loads(cmd[cmd.index("--settings") + 1])["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        assert shlex.split(hook) == [sys.executable, "-I", str(models.DENY_HOOK), str(call / "denied.jsonl")]
    finally:
        backend.close()



@pytest.mark.parametrize("metadata", [{"duration_ms": 14291}, {"session_id": "fixture-429"},
                                     {"extra": "quota"}])
def test_glm_metadata_does_not_stop_quota(tmp_path, monkeypatch, metadata):
    monkeypatch.setenv("GLM_API_KEY", "synthetic")
    backend = models.Claude(tmp_path, models.Quota(tmp_path / "quota"), kind="glm")
    events = [{"type": "result", "is_error": True, "result": "server unavailable", **metadata}]
    backend.run = lambda *a, **k: subprocess.CompletedProcess([], 1, "\n".join(map(json.dumps, events)), "HTTP 503")
    try:
        with pytest.raises(RuntimeError, match="claude_call_failed"):
            backend.call("glm:glm-5.3", [], [])
        assert not backend.quota.state["glm"]["stopped"]
    finally:
        backend.close()



def test_single_locomo_speaker_once(monkeypatch):
    monkeypatch.setattr(delivery, "dispatch", lambda *a: '{"hits":[{"store_id":1}]}')
    engine = SimpleNamespace(_store=SimpleNamespace(get_batch=lambda ids: {
        1: {"store_id": 1, "role": "user", "content": "Ada: hello", "observed_at": None}}))
    evidence, _ = delivery.single(SimpleNamespace(question="Q"), engine, {"1": "Ada"})
    assert evidence == "[unknown] Ada: hello"



def test_full_context_reserves_requested_output(monkeypatch):
    monkeypatch.setattr(delivery, "input_bound", lambda *a: 80_000)
    q = SimpleNamespace(qid="0", qtype="1", question="Q", question_date=None, sessions=[])
    client = SimpleNamespace(context_length=lambda _: 100_000, call=Mock(side_effect=AssertionError("must skip")))
    monkeypatch.setattr(delivery, "question_clock", lambda _: nullcontext())
    result = delivery.answer(q, "full_context", client, models.READERS[0])
    assert result["status"] == "skipped"
    client.call.assert_not_called()



def test_voyage_key_loaded_only_in_mcp(tmp_path, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    seen = []
    def fake_stores(*a, **k):
        seen.append(os.environ.get("VOYAGE_API_KEY"))
        return SimpleNamespace(engine=lambda q: nullcontext((None, None)))
    monkeypatch.setattr(mcp_reader, "load_lcm", lambda _: None)
    monkeypatch.setattr(stores, "Stores", fake_stores)
    monkeypatch.setattr(delivery, "schemas", lambda _: [])
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, "synthetic-voyage\n", ""))
    monkeypatch.setattr(subprocess, "run", runner)
    spec = {"repo": "fixture", "store_root": str(tmp_path), "config": "voyage", "dataset_sha": "d",
            "summaries": "none", "log": str(tmp_path / "log"), "max_calls": 1,
            "question": {"qid": "0", "dataset": "lme-s", "question_date": None, "store_id": "0", "session_map": {}}}
    mcp_reader.serve(spec, io.StringIO(""), io.StringIO())
    assert seen == ["synthetic-voyage"]
    assert runner.call_args.args[0] == ["security", "find-generic-password", "-s", "VOYAGE_API_KEY", "-w"]
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
