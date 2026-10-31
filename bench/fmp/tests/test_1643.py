"""Seven #1643 regressions: synthetic calls and a deterministic clock, never live models."""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from bench.fmp import delivery, judges, models, run, score
from bench.fmp.tests.test_r3 import MODEL, arm, row

EXTRA = models.CLAUDE_AGREEMENT_JUDGE


def update_manifest(path, **values):
    target = path / "run-manifest.json"
    target.write_text(json.dumps({**json.loads(target.read_text()), **values}))


def test_1_every_required_judge_stop_excludes_row():
    q = SimpleNamespace(dataset="lme-s", qtype="multi-session", question="Q?", gold="x", is_abstention=False)

    def call(model, messages, records, **kwargs):
        if model == EXTRA:
            raise models.BudgetStopped()
        return {"content": "yes"}

    client = SimpleNamespace(call=call)
    stopped = judges.judge_row(row(0), q, client, [MODEL, EXTRA])
    assert stopped["status"] == "not_run"
    optional = judges.judge_row(row(0), q, client, [MODEL], agreement_model=EXTRA)
    assert optional["status"] == "ok" and optional["judges"][MODEL]["correct"] is True
    assert optional["judges"][EXTRA]["status"] == "not_run"


@pytest.mark.parametrize("failure", [{"error": "timeout"}, {"error": "model_mismatch"},
                                    {"violations": ["tool"]}, {"status": "tool_violation"}])
def test_2_judge_failure_is_scoped_to_requested_model(tmp_path, failure):
    verdicts = {m: {"status": "ok", "correct": True} for m in (MODEL, EXTRA)}
    failed = row(0, judges=verdicts, judge_calls=[{"requested_model": EXTRA, **failure}])
    assert score.correct(failed, MODEL) is True
    assert score.correct(failed, EXTRA) is None
    reader_failed = {**failed, "calls": [failure]}
    assert score.correct(reader_failed, MODEL) is None
    assert score.correct(reader_failed, EXTRA) is None
    paths = [arm(tmp_path, c, [copy.deepcopy(failed)]) for c in ("off", "local")]
    for path in paths:
        update_manifest(path, agreement_judge=EXTRA)
    result = score.score(paths, tmp_path / "report")
    summary = next(iter(result["arms"].values()))
    assert summary["accuracy"][MODEL] == 1
    assert summary["agreement"]["n"] == 0
    assert summary["agreement_sample"]["n"] == 0
    assert next(iter(result["paired"].values()))["paired_n"] == 1


def test_2_openrouter_record_matches_prefixed_judge():
    spec = "openrouter:" + models.JUDGES[0]
    r = row(0, judges={MODEL: {"status": "ok", "correct": True}, spec: {"status": "ok", "correct": True}},
            judge_calls=[{"requested_model": models.JUDGES[0], "error": "timeout"}])
    assert score.correct(r, MODEL) is True
    assert score.correct(r, spec) is None


def test_5_known_reader_contexts_parse_without_model_calls(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(models.httpx, "get", lambda *a, **k: httpx.Response(200,
        json={"data": [{"id": models.READERS[0], "context_length": 1000000}]},
        request=httpx.Request("GET", "https://example.invalid")))
    for reader in (models.CODEX_DEFAULT_READER, EXTRA, "glm:glm-5.3", "openrouter:" + models.READERS[0]):
        args = run.parser().parse_args(["answer", "--reader", reader, "--split-label", "smoke",
                                       "--store-root", str(tmp_path), "--output", str(tmp_path / "out")])
        assert args.reader == reader


def test_3_usage_divisor_matches_attempted_call_population(tmp_path):
    def calls(n):
        return [{k: n for k in score.TOKEN_KEYS}]
    rows = [row(0, calls=calls(10), judge_calls=calls(2)),
            row(1, status="error", calls=calls(30), judge_calls=calls(6)),
            row(2, status="skipped", calls=calls(100), judge_calls=calls(100)),
            row(3, status="not_run", calls=calls(100), judge_calls=calls(100))]
    result = score.score([arm(tmp_path, "off", rows)], tmp_path / "report")
    name, summary = next(iter(result["arms"].items()))
    assert summary["scored_n"] == 2
    assert summary["reader_tokens_per_question"]["input_tokens"] == 20
    assert summary["judge_tokens_per_question"]["input_tokens"] == 4
    assert result["tokens_500_per_arm"][name]["input_tokens"] == 12000
    assert "attempted" in summary["tokens_per_question_population"]


def test_4_pairing_binds_toolset_except_explicit_cross_commit(tmp_path):
    paths = [arm(tmp_path, c, [row(0)]) for c in ("off", "local")]
    for i, path in enumerate(paths):
        update_manifest(path, toolset=["lcm_search"] if i == 0 else [], lcmx_commit=str(i) * 40)
    update_manifest(paths[1], lcmx_commit="0" * 40)
    with pytest.raises(ValueError, match="toolset"):
        score.score(paths, tmp_path / "ordinary")
    update_manifest(paths[1], config="off", lcmx_commit="1" * 40)
    result = score.score(paths, tmp_path / "cross", pair_across_commits=True)
    assert list(result["toolsets"].values()) == [["lcm_search"], []]
    assert len(result["paired"]) == 1


@pytest.mark.parametrize("reader", ["codex:unknown@high", "claude:unknown", "openrouter:unknown"])
def test_5_unsupported_reader_rejected_at_execution(tmp_path, monkeypatch, reader):
    import httpx
    monkeypatch.setattr(models.httpx, "get", lambda *a, **k: httpx.Response(
        200, json={"data": []}, request=httpx.Request("GET", "https://example.invalid")))
    args = run.parser().parse_args(["answer", "--reader", reader, "--split-label", "smoke",
                                   "--store-root", str(tmp_path), "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="reader context length unavailable; use a supported model spec"):
        run.execute(args)


def test_6_paired_wall_time_counts_each_arms_active_work(tmp_path, monkeypatch):
    args = run.parser().parse_args(["answer", "--split-label", "smoke", "--store-root", str(tmp_path / "stores"),
        "--output", str(tmp_path / "out"), "--paired-config", "local", "--workers", "1", "--allow-dirty",
        "--data", str(tmp_path / "synthetic.json")])
    qs = [SimpleNamespace(qid=str(i), dataset="lme-s", qtype="multi-session", question="Q?", gold="x",
                         store_id=str(i), question_date="2025-01-01", session_map={}) for i in range(2)]
    store = SimpleNamespace(build=lambda q: None, engine=lambda q: nullcontext((None, None)), generation="s")
    clock = [0.0]

    def answer(q, mode, client, reader, **kwargs):
        clock[0] += {"off": 2, "local": 5}[kwargs["server"]["config"]]
        return {"status": "ok", "hypothesis": "x"}

    monkeypatch.setattr(run.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run, "load_lcm", lambda *a: None)
    monkeypatch.setattr(run.datasets, "load", lambda *a: (qs, "d", "i", {}))
    monkeypatch.setattr(run, "Stores", lambda *a: store)
    monkeypatch.setattr(run, "Backends", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(run, "versions", lambda: {})
    monkeypatch.setattr(delivery, "schemas", lambda *a: [])
    monkeypatch.setattr(delivery, "answer", answer)
    run.execute(args)
    paths = []
    for config, seconds in (("off", 4), ("local", 10)):
        path = args.output / f"{config}-agentic"
        manifest = json.loads((path / "run-manifest.json").read_text())
        assert manifest["answer_wall_s"] == manifest["wall_s"] == seconds
        rows = [json.loads(line) for line in (path / "answers.jsonl").read_text().splitlines()]
        for r in rows:
            r.update(judges={MODEL: {"status": "ok", "correct": True}}, calls=[{"cost": 1}])
        (path / "judged.jsonl").write_text("\n".join(map(json.dumps, rows)))
        update_manifest(path, status="complete", judges=[MODEL])
        paths.append(path)
    report = score.score(paths, tmp_path / "report")
    assert [s["wall_s"] for s in report["arms"].values()] == [4, 10]
    assert report["full_run_estimate"]["wall_s_at_observed_workers"] == 250 * (4 + 10)


def test_7_projection_names_and_totals_only_supplied_arms(tmp_path):
    path = arm(tmp_path, "off", [row(0, calls=[{"cost": 2}], judge_calls=[{"cost": 1}])])
    update_manifest(path, wall_s=7)
    output = tmp_path / "report"
    result = score.score([path], output)
    estimate = result["full_run_estimate"]
    assert estimate["total_usd"] == 500 * 3
    assert estimate["wall_s_at_observed_workers"] == 500 * 7
    assert list(estimate["arms"]) == list(result["arms"])
    assert next(iter(result["arms"])) in estimate["matrix"]
    assert "6000 answers, 12000 judgments" not in estimate["matrix"]
    assert estimate["matrix"] in (output / "report.md").read_text()


def test_2_unattributed_judge_failure_stays_row_wide():
    from bench.fmp import score
    row = {"status": "ok", "calls": [], "judge_calls": [{"error": "timeout"}],
           "judges": {m: {"status": "ok", "correct": True, "verdict": "valid"} for m in ("codex:a@low", "glm:glm-5.3")}}
    assert score.invalid(row, ["codex:a@low"]) and score.invalid(row, ["glm:glm-5.3"])
