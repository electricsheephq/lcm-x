"""One MCP counted-call budget per question, including failed sends; no providers."""
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from bench.fmp import delivery, mcp_reader, models, run
from bench.fmp.tests.test_switches import cli, conversations


def written(tmp_path, monkeypatch, counted, readiness=False):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    monkeypatch.setattr(delivery, "question_clock", lambda _: nullcontext())
    monkeypatch.setattr(delivery, "dispatch", lambda *a: "{}")
    monkeypatch.setattr(delivery, "schemas", lambda _: [])
    monkeypatch.setattr(run, "versions", lambda: {})
    backend = object.__new__(models.Harness)
    sends = []
    failure = RuntimeError("timeout")

    def call(spec, messages, records, **kwargs):
        sends.append(1)
        log = tmp_path / f"mcp-{len(sends)}.jsonl"
        server = mcp_reader.Reader({"max_calls": models.AGENTIC_TOOL_CAP,
            "question": {"question_date": None}}, None, [{"function": {"name": "lcm_recall"}}], log)
        failed = len(sends) == 1
        if failed and readiness:
            server.startup_status, server.offered = "startup_failed", []
        server.event("initialize")
        server.event("tools/list")
        for _ in range(counted if failed else 1):
            server.call("lcm_recall", {})
        records.append(dict(status="error" if failed and not readiness else "ok",
            error="timeout" if failed and not readiness else None, mcp_log=str(log)))
        if failed and not readiness:
            raise failure
        return {"content": "teal"}

    backend.call = call
    args = cli(tmp_path, "answer", "--delivery", "agentic")
    args.reader = models.CODEX_DEFAULT_READER
    args.output = tmp_path
    store = SimpleNamespace(generation="fixture", engine=lambda _: nullcontext((None, None)))
    job = run.run(args, backend, conversations()[:1], "dataset", "ids", {}, store)
    work, source, _ = next(job)
    row = work(source[0])
    job.send(row)
    with pytest.raises(StopIteration):
        job.send(None)
    assert json.loads((tmp_path / "answers.jsonl").read_text()) == row
    return row, sends


@pytest.mark.parametrize("counted", [1, models.AGENTIC_TOOL_CAP, models.AGENTIC_TOOL_CAP + 1])
def test_counted_call_failure_never_retries(tmp_path, monkeypatch, counted):
    row, sends = written(tmp_path, monkeypatch, counted)
    assert row["status"] == "error" and row["error"] == "RuntimeError"
    assert len(sends) == 1
    assert row["tool_calls"] == min(counted, models.AGENTIC_TOOL_CAP)
    assert sum(bool(t.get("counted")) for t in row["trace"]) <= models.AGENTIC_TOOL_CAP
    assert row["calls"][0]["error"] == "timeout" and not row["calls"][0].get("retried")


@pytest.mark.parametrize("readiness", [False, True])
def test_zero_counted_calls_retry_and_record_recovery(tmp_path, monkeypatch, readiness):
    row, sends = written(tmp_path, monkeypatch, 0, readiness)
    assert row["status"] == "ok" and row["hypothesis"] == "teal"
    assert len(sends) == 2 and row["tool_calls"] == 1
    assert row["attempts"] == 2 and row["calls"][0]["retried"] is True
    assert row["attempt_errors"] == ["mcp_unavailable" if readiness else "timeout"]
