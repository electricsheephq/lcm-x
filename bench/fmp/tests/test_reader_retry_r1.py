"""Recovered calls remain measured; fake providers only."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from bench.fmp import datasets, delivery, judges, models, run, score
from bench.fmp.tests.test_995 import locomo, make_answers
from bench.fmp.tests.test_fmp import Scripted, codex_backend
from bench.fmp.tests.test_r3 import MODEL, arm, row


def fake_call(tmp_path, monkeypatch, transform):
    owner = tmp_path / "owner"
    (owner / ".codex").mkdir(parents=True)
    (owner / ".codex/auth.json").write_text("{}")
    monkeypatch.setenv("HOME", str(owner))
    backend, _ = codex_backend(tmp_path)
    fake, sends, records, receipt = backend.runner, [], [], {}
    def transport(*a, **kw):
        proc = fake(*a, **kw)
        sends.append(1)
        return transform(proc)
    backend.runner = transport
    sleeps = []
    monkeypatch.setattr(models.time, "sleep", sleeps.append)
    try:
        result = models.retry_call(lambda: backend.call(models.CODEX_DEFAULT_READER, [], records), records, receipt)
        return result, records, receipt, sends, sleeps
    finally:
        backend.close()


@pytest.mark.parametrize("after", [False, True])
def test_reconnect_stream_accepted_first_attempt(tmp_path, monkeypatch, after):
    def reconnect(proc):
        events = [json.loads(line) for line in proc.stdout.splitlines()]
        events.insert(len(events) if after else 2, {"type": "error", "message": "Reconnecting... 2/5"})
        return subprocess.CompletedProcess(proc.args, 0, "\n".join(map(json.dumps, events)), "")
    if after:
        with pytest.raises(RuntimeError):
            fake_call(tmp_path, monkeypatch, reconnect)
    else:
        result, records, receipt, sends, sleeps = fake_call(tmp_path, monkeypatch, reconnect)
        assert result["content"] == "teal" and records[-1]["status"] == "ok"
        assert len(sends) == 1 and sleeps == [] and receipt == {}


def test_stderr_limit_stops_without_retry(tmp_path, monkeypatch):
    sends = []
    def limited(proc):
        sends.append(1)
        return subprocess.CompletedProcess(proc.args, 1, "", "You've hit your usage limit")
    with pytest.raises(models.BudgetStopped):
        fake_call(tmp_path, monkeypatch, limited)
    assert len(sends) == 1


def recovering(records, failures=1):
    count = 0
    def call():
        nonlocal count
        count += 1
        if count <= failures:
            records.append(dict(status="error", error="timeout", requested_model=MODEL))
            raise RuntimeError("timeout")
        records.append(dict(status="ok", requested_model=MODEL, **dict.fromkeys(score.TOKEN_KEYS, 7)))
        return {"content": "CORRECT"}
    return call


@pytest.mark.parametrize("target", ["calls", "judge_calls"])
def test_recovered_timeout_scores_correct_and_paired(tmp_path, monkeypatch, target):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    records, receipt = [], {}
    models.retry_call(recovering(records), records, receipt)
    recovered = row(0, **{target: records})
    assert not score.invalid(recovered, [MODEL]) and score.correct(recovered, MODEL) is True
    result = score.score([arm(tmp_path, "off", [recovered]), arm(tmp_path, "local", [row(0)])], tmp_path / "score")
    assert next(iter(result["paired"].values()))["n_pairs"] == 1
    assert next(iter(result["arms"].values()))["accuracy"][MODEL] == 1
    assert records[0]["retried"] is True and not records[1].get("retried")


def test_terminal_timeout_remains_invalid(monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    records = []
    with pytest.raises(RuntimeError):
        models.retry_call(recovering(records, 3), records, {})
    assert score.invalid(row(0, status="error", calls=records), [MODEL]) == {"timeout"}
    assert not any(c.get("retried") for c in records)


def test_recovered_tokens_stay_numeric(monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    records = []
    models.retry_call(recovering(records), records, {})
    assert score.usage(records, 1) == dict.fromkeys(score.TOKEN_KEYS, 7)
    assert score.usage([dict(status="error")], 1) == dict.fromkeys(score.TOKEN_KEYS)


def test_judge_reask_attempts_are_per_call(tmp_path, monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    q = datasets.load(locomo(tmp_path), "locomo", None)[0][0]
    class Client(Scripted):
        def call(self, model, messages, records, **kwargs):
            if not records:
                records.append(dict(status="error", error="timeout"))
                raise RuntimeError("timeout")
            return super().call(model, messages, records, **kwargs)
    result, records = judges.judge(q, "teal", Client([{"content": "unclear"}, {"content": "CORRECT"}]), models=[MODEL])
    assert result[MODEL]["correct"] is True
    assert [c["attempts"] for c in records] == [2, 2, 1]
    assert result[MODEL]["attempt_errors"] == ["timeout"] and result[MODEL]["attempts"] == 2


def test_reader_multiple_calls_do_not_inflate_attempts(monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    records, receipt = [], {}
    models.retry_call(recovering(records), records, receipt)
    models.retry_call(recovering(records, 0), records, receipt)
    assert [c["attempts"] for c in records] == [2, 2, 1] and receipt["attempts"] == 2


def test_backoff_excluded_from_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    records, receipt = [], {}
    models.retry_call(recovering(records, 2), records, receipt)
    assert receipt["retry_backoff_s"] == 25
    measured = row(0, calls=records, latency_s=32, **receipt)
    result = score.score([arm(tmp_path, "off", [measured])], tmp_path / "score")
    summary = next(iter(result["arms"].values()))
    assert summary["latency_p50_s"] == summary["latency_p90_s"] == 7


@pytest.mark.parametrize("states,expected", [
    ([(False, None), (True, "timeout"), (True, "timeout"), (True, None)], "timeout"),
    ([(False, "timeout")] * 3, "timeout"),
    ([(False, "codex_failed")] * 3, "codex_failed"),
    ([(False, None)] * 3, "mcp_unavailable"),
    ([(False, "timeout"), (True, None)], None),
])
def test_agentic_send_budget_and_class(tmp_path, monkeypatch, states, expected):
    monkeypatch.setattr(models.time, "sleep", lambda _: None)
    backend = object.__new__(models.Harness)
    sends = []
    def call(spec, messages, records, **kwargs):
        ready, error = states[len(sends)]
        sends.append(1)
        log = tmp_path / f"mcp-{len(sends)}.jsonl"
        log.write_text("\n".join(map(json.dumps, [dict(event="initialize", startup_status="ok" if ready else "startup_failed"),
            dict(event="tools/list", offered_tools=int(ready))] +
            ([dict(tool="lcm_recall", counted=True)] if ready else []))))
        records.append(dict(status="error" if error else "ok", error=error, mcp_log=str(log)))
        if error:
            raise RuntimeError(error)
        return {"content": "teal"}
    backend.call = call
    q = SimpleNamespace(qid="q", qtype="1", question_date=None, question="Q?")
    if expected:
        with pytest.raises(RuntimeError) as raised:
            delivery.harness_agentic(q, backend, models.CODEX_DEFAULT_READER, {}, [])
        result = raised.value.delivery_result
        assert result["calls"][-1]["error"] == expected
        assert len(sends) == (2 if states[0] == (False, None) and states[1][0] else 3)
    else:
        result = delivery.harness_agentic(q, backend, models.CODEX_DEFAULT_READER, {}, [])
        assert result["hypothesis"] == "teal" and len(sends) == 2
    assert result["tool_calls"] == sum(ready for ready, _ in states[:len(sends)])
    assert len({t["attempt"] for t in result["trace"] if "tool" in t}) == result["tool_calls"]


@pytest.mark.parametrize("status", ["tool_violation", "budget_exceeded", "skipped", "error", "not_run"])
def test_check_arms_reports_non_ok_statuses(tmp_path, status):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    path = paths[0] / "answers.jsonl"
    answer = json.loads(path.read_text())
    answer["status"] = status
    path.write_text(json.dumps(answer))
    result = run.check_arms(paths, tmp_path / "check.json")
    assert result["passed"] is (status not in ("error", "not_run"))
    assert result["arms"][0]["non_ok_statuses"] == {status: 1}
    assert "non_ok_statuses" not in result["arms"][1]
