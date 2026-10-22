"""Offline regressions for #1844; synthetic data and frozen verdicts only."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import datasets, models, run, score
from bench.fmp.tests.test_995 import make_answers
from bench.fmp.tests.test_r3 import MODEL, arm, row
from bench.fmp.tests.test_switches import cli


def test_answer_category5_decoy_is_not_gold(tmp_path, monkeypatch):
    args = cli(tmp_path, "answer", "--delivery", "single", "--hit-label", "role", "--include-adversarial")
    args.output.mkdir()
    questions = [datasets.Question(str(i), "locomo", kind, "Synthetic?", "DECOY", "", kind == "5", [])
                 for i, kind in enumerate(("1", "5"))]
    store = SimpleNamespace(generation="synthetic", engine=lambda q: nullcontext((None, None)))
    monkeypatch.setattr(run.delivery, "schemas", lambda _: [])
    monkeypatch.setattr(run.delivery, "answer", lambda *a, **k: {"status": "ok", "hypothesis": "Unknown"})
    monkeypatch.setattr(run, "versions", lambda: {})
    run.run_jobs([run.run(args, None, questions, "data", "ids", {}, store)], 1)
    normal, adversarial = map(json.loads, (args.output / "answers.jsonl").read_text().splitlines())
    assert normal["gold"] == "DECOY" and "adversarial_answer" not in normal
    assert adversarial["gold"] == "" and adversarial["adversarial_answer"] == "DECOY"


def test_legacy_and_new_scores_identical_with_safe_adjudication(tmp_path):
    other = models.CLAUDE_AGREEMENT_JUDGE
    verdicts = {MODEL: {"status": "ok", "correct": True, "raw": ["yes"]},
                other: {"status": "ok", "correct": False, "raw": ["no"]}}
    reports, adjudications = [], []
    for name in ("legacy", "new"):
        rows = [row(0, qtype="5", gold="DECOY", judges=verdicts),
                row(1, qtype="1", gold="DECOY", judges={
                    MODEL: {"status": "ok", "correct": True, "raw": ['{"label":"CORRECT"}']},
                    other: {"status": "ok", "correct": False, "raw": ['{"label":"WRONG"}']}})]
        if name == "new":
            rows[0].update(gold="", adversarial_answer="DECOY")
        directory = arm(tmp_path / name, "off", rows)
        manifest_path = directory / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(judges=[MODEL, other], include_adversarial=True)
        manifest_path.write_text(json.dumps(manifest))
        output = tmp_path / (name + "-score")
        reports.append(score.score([directory], output))
        adjudications.append([json.loads(line) for line in (output / "adjudicate.jsonl").read_text().splitlines()])
    assert reports[0] == reports[1]
    assert adjudications[0] == adjudications[1]
    by_id = {r["qid"]: r for r in adjudications[0]}
    assert by_id["0"]["gold"] == "" and by_id["0"]["adversarial_answer"] == "DECOY"
    assert by_id["1"]["gold"] == "DECOY" and "adversarial_answer" not in by_id["1"]


CONFIG = {"reader": models.CODEX_DEFAULT_READER, "harness": {"versions": {"codex": "fixture"}},
          "config": "local", "summaries": "none", "store_generation": "local-none-synthetic",
          "delivery_unit": "hits", "delivery_budget": None, "reasoning": {"effort": "high"},
          "max_tokens": {"reader": 32000}, "tool_budget": {"calls": 0, "input_tokens": 200000}}


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("key,value", [
    ("reader", "claude:claude-opus-5-5"),
    ("reader", "codex:gpt-6-astra@high"),
    ("reader", "codex:gpt-6.1-sol@low"),
    ("harness", {"versions": {"codex": "different"}}),
    ("config", "off"), ("summaries", "deterministic"),
    ("store_generation", "local-deterministic-other"),
    ("delivery_unit", "session"), ("delivery_budget", 16000),
    ("reasoning", {"effort": "low"}), ("max_tokens", {"reader": 16000}),
    ("tool_budget", {"calls": 0, "input_tokens": 100000}),
    ("store_generation", None),
])
def test_check_arms_config_mismatch(tmp_path, key, value, reverse):
    paths = [make_answers(tmp_path, "A0", "role"), make_answers(tmp_path, "A1")]
    for path in paths:
        manifest_path = path / "run-manifest.json"
        manifest_path.write_text(json.dumps({**json.loads(manifest_path.read_text()), **CONFIG}))
    if reverse:
        paths.reverse()
    # The one permitted switch, hit_label, stays valid in either direction.
    assert run.check_arms(paths, tmp_path / "control.json")["passed"]
    manifest_path = paths[1] / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if value is None:
        manifest.pop(key)
    else:
        manifest[key] = value
    manifest_path.write_text(json.dumps(manifest))
    before = {p: p.read_bytes() for path in paths for p in path.iterdir()}
    result = run.check_arms(paths, tmp_path / "mismatch.json")
    assert result["passed"] is False
    assert result["configuration_mismatches"] == [key]
    assert key in result["reason"]
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("mode", ["text", "mcp"])
@pytest.mark.parametrize("timeout", [False, True])
def test_canary_stall_reason(tmp_path, monkeypatch, mode, timeout):
    args = cli(tmp_path, "canary")
    args.output.mkdir()
    args.data = tmp_path / "fixture.json"
    monkeypatch.setattr(run, "versions", lambda: {})
    clock = [0.0]
    monkeypatch.setattr(run.time, "monotonic", lambda: clock[0])
    calls = []

    def probe(probe_mode, records):
        calls.append(probe_mode)
        directory = args.output / "harness" / probe_mode
        directory.mkdir(parents=True)
        events = [{"type": "system", "subtype": "init", "message": "PRIVATE_PROMPT"}]
        if not timeout:
            events.append({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}})
        transcript = directory / "events.jsonl"
        transcript.write_text("\n".join(map(json.dumps, events)) + ("\n{partial" if timeout else ""))
        record = {"call_dir": directory.name, "mcp_calls": int(probe_mode == "mcp")}
        if probe_mode == mode:
            clock[0] += 611.0
            record.update(status="error", error="timeout" if timeout else "claude_failed", latency_s=611.0)
            # Timeout records historically omit transcripts; the partial file still exists.
            if not timeout:
                record["transcripts"] = [str(transcript)]
            records.append(record)
            raise RuntimeError("PRIVATE_ERROR")
        records.append({**record, "transcripts": [str(transcript)]})
        return {"content": "Unknown"}

    client = SimpleNamespace(call=lambda spec, messages, records: probe("text", records),
                             agentic=lambda spec, messages, records, server: probe("mcp", records))
    with pytest.raises(RuntimeError, match="canary_failed"):
        run.canary(args, client, "synthetic", SimpleNamespace(qid="fixture", dataset="locomo", store_id="fixture",
                                                             session_map={}, question_date=""))
    output = (args.output / "canary.json").read_text()
    result = next(r for r in json.loads(output)["results"] if r["mode"] == mode)
    assert result["passed"] is False
    assert "611" in result["reason"] and mode in result["reason"]
    assert "system:init" in result["reason"]
    assert ("timeout" if timeout else "rate_limit_event") in result["reason"]
    assert "PRIVATE_" not in output and "REF-" not in output
    assert calls == ["text", "mcp"]  # no retries
    assert not list(tmp_path.rglob("fmp-canary-*.txt"))


def test_readme_receipt_and_per_reader_canary():
    readme = Path(run.__file__).with_name("README.md").read_text()
    assert "Each reader harness a run uses must pass its leakage canary before that run's answer arms." in readme
    assert "`evidence_sha256` is an audit receipt, not evidence verification" in readme
