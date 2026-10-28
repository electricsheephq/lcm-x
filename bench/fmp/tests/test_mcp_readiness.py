"""Readiness acceptance uses synthetic protocol, CLI and score fixtures only."""
import io
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from bench.fmp import delivery, mcp_reader, models, score, stores
from bench.fmp.common import sha
from bench.fmp.tests.test_r3 import MODEL, arm, row

ENV = ("TMPDIR", "FMP_CACHE_ROOT", "LCM_LONGMEMEVAL_FASTEMBED_CACHE", "FASTEMBED_CACHE_PATH",
       "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
       "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")


def fake_mcp_ready(call):
    if (call / "server.json").exists():
        reader = mcp_reader.Reader({}, None, [{"function": {"name": "lcm_recall"}}], call / "mcp-log.jsonl")
        for method in ("initialize", "tools/list"):
            reader.handle(dict(id=1, method=method))


def test_environment(tmp_path, monkeypatch):
    for name in ENV:
        monkeypatch.setenv(name, "synthetic")
    backend = object.__new__(models.Harness)
    env = backend.server(tmp_path, {})["env"]
    assert all(env.get(name) == "synthetic" for name in ENV)
    assert all(env[name] == "4" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                           "MKL_NUM_THREADS", "ORT_NUM_THREADS"))


@pytest.mark.parametrize("failed", [False, True])
def test_protocol_warmup_barrier(tmp_path, monkeypatch, failed):
    state = []
    @contextmanager
    def engine(q):
        state.append("warm")
        if failed:
            raise RuntimeError("synthetic startup failure")
        yield None, None
    monkeypatch.setattr(mcp_reader, "load_lcm", lambda _: None)
    monkeypatch.setattr(stores, "Stores", lambda *a, **k: SimpleNamespace(engine=engine))
    monkeypatch.setattr(delivery, "schemas", lambda _: [{"function": {"name": "lcm_recall"}}])
    class Output(io.StringIO):
        def write(self, value):
            assert state == ["warm"]
            return super().write(value)
    spec = dict(repo="synthetic", store_root=str(tmp_path), config="local", dataset_sha="d",
                summaries="none", log=str(tmp_path / "log"), max_calls=1,
                question=dict(qid="0", dataset="lme-s", question_date=None, store_id="0", session_map={}))
    requests = [dict(id=i, method=m) for i, m in enumerate(("initialize", "tools/list"), 1)]
    output = Output()
    mcp_reader.serve(spec, io.StringIO("\n".join(map(json.dumps, requests))), output)
    tools = json.loads(output.getvalue().splitlines()[1])["result"]["tools"]
    assert len(tools) == (0 if failed else 1)
    events = [json.loads(line) for line in (tmp_path / "log").read_text().splitlines()]
    assert [e["event"] for e in events][-2:] == ["initialize", "tools/list"]
    assert events[-1]["offered_tools"] == len(tools)
    if failed:
        assert events[0]["event"] == "startup_failed" and events[0]["reason"] == "RuntimeError"


@pytest.mark.parametrize("states", [(False, False), (False, True), (True,)])
def test_records_retry_and_unused(tmp_path, states):
    backend = object.__new__(models.Harness)
    def call(spec, messages, records, **kwargs):
        ready = states[len(records)]
        log = tmp_path / str(len(records))
        events = [dict(event="initialize", startup_status="ok" if ready else "startup_failed"),
                  dict(event="tools/list", offered_tools=int(ready), offered_tools_hash=sha(b'lcm_recall'))]
        log.write_text("\n".join(map(json.dumps, events)))
        records.append(dict(status="ok", mcp_log=str(log)))
        return {"content": "synthetic answer"}
    backend.call = call
    records = []
    if states[-1]:
        assert backend.agentic("spec", [], records, {})["content"] == "synthetic answer"
    else:
        with pytest.raises(RuntimeError, match="mcp_unavailable"):
            backend.agentic("spec", [], records, {})
    assert len(records) == len(states)
    last = records[-1]
    assert last["mcp_ready"] is states[-1] and last["offered_tools"] == int(states[-1])
    assert last["offered_tools_hash"] == sha(b'lcm_recall') and last["lcm_tool_calls"] == 0
    assert last["startup_status"] == ("ok" if states[-1] else "startup_failed")
    assert (last.get("error") == "mcp_unavailable") is (not states[-1])
    if len(states) == 2:
        assert records[0]["error"] == "mcp_unavailable" and last["retries"] == 1


def test_backend_failure_preserves_transport_class(tmp_path):
    backend = object.__new__(models.Harness)
    def call(spec, messages, records, **kwargs):
        log = tmp_path / str(len(records))
        log.write_text(json.dumps(dict(event="initialize", startup_status="startup_failed")))
        records.append(dict(status="error", error="timeout", mcp_log=str(log)))
        raise RuntimeError("timeout")
    backend.call = call
    records = []
    with pytest.raises(RuntimeError, match="timeout"):
        backend.agentic("spec", [], records, {})
    assert [r["error"] for r in records] == ["timeout"]


def test_missing_initialize(tmp_path):
    backend = object.__new__(models.Harness)
    backend.call = lambda s, m, r, **kw: r.append(dict(status="ok", mcp_log=str(tmp_path / "absent")))
    records = []
    with pytest.raises(RuntimeError, match="mcp_unavailable"):
        backend.agentic("spec", [], records, {})
    assert len(records) == 2 and records[-1]["startup_status"] == "no_initialize"


def test_pair_exclusion_and_summary(tmp_path):
    bad = dict(status="error", error="mcp_unavailable", retries=0, mcp_ready=False,
               offered_tools=0, lcm_tool_calls=0)
    good = dict(status="ok", retries=0, mcp_ready=True, offered_tools=9, lcm_tool_calls=0)
    left = [row(0, calls=[bad, {**bad, "retries": 1}], status="error"),
            row(1, calls=[bad, {**good, "retries": 1}])]
    paths = [arm(tmp_path, "local", left), arm(tmp_path, "off", [row(0), row(1)])]
    result = score.score(paths, tmp_path / "report")
    pair = next(iter(result["paired"].values()))
    assert pair["paired_n"] == 1 and pair["dropped_by_reason"] == {"mcp_unavailable": 1}
    summary = next(iter(result["arms"].values()))
    assert summary["valid_n"] == 1 and summary["accuracy"][MODEL] == .5
    assert summary["mcp"]["ready_rate"] == .25 and summary["mcp"]["zero_lcm_calls_ready"] == 1
    assert summary["mcp"]["mcp_unavailable_retried"] == 2 and summary["mcp"]["mcp_unavailable_excluded"] == 1
    text = (tmp_path / "report/report.md").read_text()
    assert all(label in text for label in ("MCP ready rate", "Offered tools distribution", "Zero LCM calls ready",
                                          "mcp_unavailable retried/excluded", "mcp_unavailable=1"))
