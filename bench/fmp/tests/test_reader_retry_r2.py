"""Pre-canary stderr acceptance and readiness telemetry; fake calls only."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from bench.fmp import delivery, models, score
from bench.fmp.tests.test_r3 import arm, row
from bench.fmp.tests.test_reader_retry_r1 import fake_call


def test_success_with_stderr_rate_limit_warning(tmp_path, monkeypatch):
    def warning(proc):
        return subprocess.CompletedProcess(proc.args, 0, proc.stdout, "Warning: rate limit telemetry unavailable")
    result, records, receipt, sends, sleeps = fake_call(tmp_path, monkeypatch, warning)
    assert result["content"] == "teal" and records[-1]["status"] == "ok"
    assert len(sends) == 1 and sleeps == [] and receipt == {}


def test_failed_stderr_usage_limit_stops_after_one_send(tmp_path, monkeypatch):
    sends = []
    def limited(proc):
        sends.append(1)
        return subprocess.CompletedProcess(proc.args, 1, "", "You've hit your usage limit")
    with pytest.raises(models.BudgetStopped):
        fake_call(tmp_path, monkeypatch, limited)
    assert len(sends) == 1


def test_second_send_readiness_retry_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    backend = object.__new__(models.Harness)
    sends = []
    def call(spec, messages, records, **kwargs):
        ready = bool(sends)
        sends.append(1)
        log = tmp_path / f"mcp-{len(sends)}.jsonl"
        log.write_text("\n".join(map(json.dumps, [
            dict(event="initialize", startup_status="ok" if ready else "startup_failed"),
            dict(event="tools/list", offered_tools=int(ready)),
            dict(tool="lcm_recall", counted=True)])))
        records.append(dict(status="ok", mcp_log=str(log)))
        return {"content": "teal"}
    backend.call = call
    q = SimpleNamespace(qid="q", qtype="1", question_date=None, question="Q?")
    result = delivery.harness_agentic(q, backend, models.CODEX_DEFAULT_READER, {}, [])
    assert result["hypothesis"] == "teal" and len(sends) == 2
    assert result["calls"][0]["error"] == "mcp_unavailable"
    assert result["calls"][0]["retried"] is True
    assert all(c["retries"] == 0 for c in result["calls"])
    scored = score.score([arm(tmp_path, "local", [row(0, calls=result["calls"])])], tmp_path / "report")
    summary = next(iter(scored["arms"].values()))
    assert summary["mcp"]["mcp_unavailable_retried"] == 1
    assert summary["mcp"]["mcp_unavailable_excluded"] == 0
