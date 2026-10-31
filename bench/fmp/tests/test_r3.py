"""Round 3 measurement regressions; synthetic fixtures, no live clients."""
import json
import random
import re
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from bench.fmp import delivery, judges, models, run, score

REPO = Path(__file__).resolve().parents[3]
MODEL = models.CODEX_DEFAULT_JUDGE


def arm(root, config, rows):
    path = root / config
    path.mkdir(parents=True)
    (path / "run-manifest.json").write_text(json.dumps({"config": config, "delivery": "agentic",
        "reader": models.CODEX_DEFAULT_READER, "judges": [MODEL], "split_label": "smoke",
        "status": "complete", "question_ids": [r["qid"] for r in rows]}))
    (path / "judged.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return path


def row(i, verdict=True, **extra):
    return {"qid": str(i), "qtype": "temporal-reasoning", "status": "ok", "question": "Synthetic?",
            "gold": "x", "hypothesis": "x", "judges": {MODEL: {"status": "ok", "correct": verdict}}, **extra}


@pytest.mark.parametrize("text,dataset", [("Yes or no? No.", "lme-s"), ("Maybe yes", "lme-s"),
    ("Yes and No", "lme-s"), ("It is right. CORRECT", "locomo"), ('{"label":" CORRECT "}', "locomo"),
    ('{"label":\'CORRECT\'}', "locomo")])
def test_regression_strict_final_verdict(text, dataset):
    assert judges.parse(text, dataset) is None


@pytest.mark.parametrize("text,dataset,expected", [("yes", "lme-s", True), ("NO", "lme-s", False),
    ('{"label":"WRONG"}', "locomo", False),
    ("Explanation\nCORRECT", "locomo", True)])
def test_control_exact_verdict(text, dataset, expected):
    assert judges.parse(text, dataset) is expected


def test_regression_final_no_line():
    assert judges.parse("Explanation\nno", "lme-s") is False


def test_regression_invalid_verdict_recorded():
    q = SimpleNamespace(dataset="lme-s", qtype="multi-session", question="Q?", gold="x", is_abstention=False)
    fake = SimpleNamespace(call=lambda *a, **k: {"content": "Yes or no? No."})
    result, _ = judges.judge(q, "x", fake, models=[MODEL])
    assert result[MODEL]["correct"] is None and result[MODEL]["verdict"] == "invalid"


def test_regression_six_invalid_pairs_excluded(tmp_path):
    left = [row(i, judges={MODEL: {"status": "judge_error", "correct": None}}) for i in range(6)]
    right = [row(i) for i in range(6)]
    result = score.score([arm(tmp_path, "off", left), arm(tmp_path, "local", right)], tmp_path / "report")
    a = next(iter(result["arms"].values()))
    pair = next(iter(result["paired"].values()))
    assert a["accuracy"][MODEL] == 0 and a["scored_n"] == 6
    assert a["invalid_by_reason"] == {"judge_error": 6}
    assert pair["n_pairs"] == 0 and pair["dropped_by_reason"] == {"judge_error": 6}
    assert pair["per_judge"][MODEL]["p"] == 1 and pair["primary_difference_ci95"] is None


def test_regression_scorer_reparses_legacy_raw_verdict(tmp_path):
    judged = {MODEL: {"status": "ok", "correct": True, "raw": ["Yes or no? No."]}}
    result = score.score([arm(tmp_path, "off", [row(0, judges=judged)])], tmp_path / "report")
    a = next(iter(result["arms"].values()))
    assert a["scored_n"] == 1 and a["accuracy"][MODEL] == 0


def test_regression_claude_missing_usage(tmp_path):
    backend = models.Claude(tmp_path, models.Quota(tmp_path / "quota.json"), runner=lambda *a, **k: None,
                            token=lambda: "synthetic")
    events = [{"type": "assistant", "message": {"model": "claude-opus-5-5"}},
              {"type": "result", "subtype": "success", "result": "x", "modelUsage": {"claude-opus-5-5": {}}}]
    backend.run = lambda *a, **k: subprocess.CompletedProcess([], 0, "\n".join(map(json.dumps, events)), "")
    records = []
    try:
        backend.call(models.CLAUDE_AGREEMENT_JUDGE, [], records)
        assert all(records[-1][k] is None for k in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"))
    finally:
        backend.close()


@pytest.mark.parametrize("reason,extra", [("tool_violation", {"status": "tool_violation"}),
    ("timeout", {"calls": [{"status": "error", "error": "timeout"}]}),
    ("model_mismatch", {"judge_calls": [{"requested_model": MODEL, "status": "error", "error": "model_mismatch"}]}),
    ("invalid_verdict", {"judges": {MODEL: {"status": "ok", "correct": None}}}),
    ("reader_error", {"status": "error"})])
def test_regression_shared_validity_mask(tmp_path, reason, extra):
    result = score.score([arm(tmp_path, "off", [row(0), row(1, **extra)]),
                          arm(tmp_path, "local", [row(0), row(1, False)])], tmp_path / "report")
    a = next(iter(result["arms"].values()))
    pair = next(iter(result["paired"].values()))
    assert a["primary"] == .5 and a["scored_n"] == 2
    assert a["invalid_by_reason"] == {reason: 1}
    assert pair["n_pairs"] == 1 and pair["dropped_by_reason"] == {reason: 1}


@pytest.mark.parametrize("reported", ["gpt-response", None])
def test_regression_codex_identity_provenance(tmp_path, reported):
    backend = models.Codex(tmp_path, models.Quota(tmp_path / "quota.json"), runner=lambda *a, **k: None)
    try:
        rollout = tmp_path / "synthetic.jsonl"
        entries = [{"type": "turn_context", "payload": {"model": "gpt-configured"}}]
        if reported:
            entries.append({"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                                                    "model": reported}})
        rollout.write_text("\n".join(map(json.dumps, entries)))
        _, _, info = backend.inspect([], rollout)
        assert info["model"] == (reported or "gpt-configured")
        assert info["identity_source"] == ("response" if reported else "configured")
    finally:
        backend.close()


def test_regression_openrouter_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic")
    meta = {m: {"pricing": {"prompt": "0", "completion": "0"}} for m in models.READERS}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"model": models.READERS[1],
        "provider": "fixture", "usage": {"cost": 0, "prompt_tokens": 1, "completion_tokens": 1},
        "choices": [{"message": {"content": "x"}}]}))
    client = models.OpenRouter(models.Spend(tmp_path / "ledger.json", 1), meta, transport)
    records = []
    try:
        with pytest.raises(RuntimeError):
            client.call(models.READERS[0], [], records)
        assert records[-1]["status"] != "ok" and records[-1]["error"] == "model_mismatch"
    finally:
        client.client.close()


def codex_reply(tmp_path, usage):
    backend = models.Codex(tmp_path, models.Quota(tmp_path / "quota.json"), runner=lambda *a, **k: None)

    def fake(cmd, call, prompt, env, agentic):
        events = [{"type": "item.completed", "item": {"type": "agent_message", "text": "x"}},
                  {"type": "turn.completed", "usage": usage}]
        return subprocess.CompletedProcess(cmd, 0, "\n".join(map(json.dumps, events)), "")
    backend.run = fake
    backend.inspect = lambda *a: ([], 0, {"model": "gpt-6.1-sol", "rate_limits": None,
                                         "requests": 1, "identity_source": "configured"})
    records = []
    try:
        backend.call(models.CODEX_DEFAULT_READER, [{"role": "user", "content": "x"}], records)
        return records[-1]
    finally:
        backend.close()


def test_regression_missing_usage_is_unknown(tmp_path):
    record = codex_reply(tmp_path, {})
    assert all(record[k] is None for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                                          "cached_input_tokens", "wrapper_overhead_tokens"))
    result = score.score([arm(tmp_path / "arms", "off", [row(0, calls=[record])])], tmp_path / "report")
    a = next(iter(result["arms"].values()))
    assert a["reader_tokens_per_question"]["input_tokens"] is None
    assert a["reader_tokens_unknown"]["input_tokens"] == 1
    assert a["quota"]["weekly_unknown"] == 1


def test_regression_missing_weekly_percent(tmp_path):
    quota = models.Quota(tmp_path / "quota.json")
    assert quota.codex({"primary": {"window_minutes": 10080}}) is None
    assert quota.state["codex"]["weekly_used_percent_first"] is None


def test_regression_overhead_label_is_honest(tmp_path):
    record = codex_reply(tmp_path, {"input_tokens": 10, "output_tokens": 1})
    assert record["input_tokens_label"] == "total incl. retrieved evidence"
    assert record["wrapper_overhead_tokens"] is None


@pytest.mark.parametrize("qtype", ["temporal-reasoning", "2", "multi-session"])
def test_regression_date_metadata_on_delivery(tmp_path, qtype):
    q = SimpleNamespace(qid="synthetic", qtype=qtype, question="Q?", question_date="2025-01-01")
    fake = SimpleNamespace(agentic=lambda *a, **kw: {"content": "x"})
    result = delivery.answer(q, "agentic", fake, models.CODEX_DEFAULT_READER, records=[])
    assert result["wall_date_visible"] is True
    assert result["temporal_question"] is (qtype in ("temporal-reasoning", "2"))


def test_regression_runner_interleaves_and_dates_even_failures(tmp_path):
    args = run.parser().parse_args(["answer", "--split-label", "smoke", "--store-root", str(tmp_path / "stores"),
        "--output", str(tmp_path / "out"), "--workers", "1", "--allow-dirty", "--data", str(tmp_path / "lme.json")])
    args.paired_config = "local"
    qs = [SimpleNamespace(qid=str(i), dataset="lme-s", qtype="temporal-reasoning", question="Q?", gold="x", store_id=str(i),
                          question_date="2025-01-01", session_map={}) for i in range(3)]
    calls = []
    store = SimpleNamespace(build=lambda q: None, engine=lambda q: nullcontext((None, None)), generation="s")
    client = SimpleNamespace(close=lambda: None)

    def answer(q, mode, client, reader, **kwargs):
        calls.append((q.qid, kwargs["server"]["config"]))
        raise models.ToolViolation()
    with patch.object(run, "load_lcm"), patch.object(run.datasets, "load", return_value=(qs, "d", "i", {})), \
            patch.object(run, "Stores", return_value=store), patch.object(run, "Backends", return_value=client), \
            patch.object(run, "versions", return_value={}), patch.object(delivery, "schemas", return_value=[]), \
            patch.object(delivery, "answer", side_effect=answer):
        run.execute(args)
    start = random.Random(1189).randrange(2)
    configs = ["off", "local"]
    assert calls == [(str(i), configs[(start + i + j) % 2]) for i in range(3) for j in range(2)]
    for config in configs:
        rows = [json.loads(s) for s in (args.output / f"{config}-agentic/answers.jsonl").read_text().splitlines()]
        assert all(r["wall_date_visible"] and r["temporal_question"] for r in rows)


def test_regression_smoke_uses_paired_runner():
    script = (REPO / "bench/fmp/run_smoke.sh").read_text()
    assert "--paired-config local" in script


def test_regression_no_machine_local_defaults(tmp_path, monkeypatch):
    """REGRESSION: the harness names no local path; a missing dataset is an error, not a default."""
    for path in (REPO / "bench/fmp").rglob("*"):
        if path.is_file() and path.suffix in (".py", ".sh", ".md"):
            assert not re.search(r"/(Users|home)/[A-Za-z0-9_]", path.read_text()), path.name
    monkeypatch.delenv("FMP_LME_DATA", raising=False)
    args = run.parser().parse_args(["answer", "--split-label", "smoke", "--store-root", str(tmp_path / "stores"),
        "--output", str(tmp_path / "out"), "--allow-dirty"])
    with pytest.raises(ValueError, match="FMP_LME_DATA"):
        run.execute(args)
