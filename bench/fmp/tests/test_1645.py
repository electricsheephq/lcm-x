"""Four #1645 harness regressions; synthetic fixtures, never live models or HTTP."""
import argparse
from types import SimpleNamespace

import pytest

from bench.fmp import judges, models, run, score
from bench.fmp.tests.test_1643 import update_manifest
from bench.fmp.tests.test_r3 import MODEL, arm, row

EXTRA = models.CLAUDE_AGREEMENT_JUDGE


def test_1_projection_counts_required_and_agreement_judgments(tmp_path):
    for index, (extra, fraction, count) in enumerate(((EXTRA, .101, 102), (EXTRA, .1, 100), (None, .1, 0))):
        root = tmp_path / str(index)
        paths = [arm(root, config, [row(0, calls=[{"cost": 1}])]) for config in ("off", "local")]
        for path in paths:
            update_manifest(path, agreement_judge=extra, agreement_fraction=fraction)
        output = root / "report"
        result = score.score(paths, output)
        matrix = result["full_run_estimate"]["matrix"]
        assert f"1000 required judgments, {count} agreement judgments" in matrix
        assert matrix in (output / "report.md").read_text()


def test_2_reader_admission_only_when_dispatched(tmp_path, monkeypatch):
    def no_http(*args, **kwargs):
        raise AssertionError("reader metadata must not be requested")

    monkeypatch.setattr(models.httpx, "get", no_http)
    common = ["--split-label", "smoke", "--store-root", str(tmp_path), "--output", str(tmp_path / "out")]
    reader = "openrouter:" + models.READERS[0]
    for command in ("judge", "score", "build-stores"):
        args = run.parser().parse_args([command, "--reader", reader, *common])
        assert args.reader == reader
        if command == "score":
            monkeypatch.setattr(run, "score", lambda *args: None)
            run.execute(args)
        else:
            with pytest.raises(ValueError, match="--data FILE"):
                run.execute(args)
    for command in ("answer", "canary"):
        args = run.parser().parse_args([command, "--reader", "codex:unknown@high", *common])
        with pytest.raises(ValueError, match="reader context length unavailable; use a supported model spec"):
            run.execute(args)


def test_3_excluded_row_keeps_completed_judges_without_scoring_them(tmp_path):
    q = SimpleNamespace(dataset="lme-s", qtype="multi-session", question="Q?", gold="x", is_abstention=False)

    def call(model, messages, records, **kwargs):
        records.append({"requested_model": model})
        if model == EXTRA:
            raise models.BudgetStopped()
        return {"content": "yes"}

    answer = row(0)
    del answer["judges"]
    stopped = judges.judge_row(answer, q, SimpleNamespace(call=call), [MODEL, EXTRA])
    assert stopped["status"] == "not_run"
    assert stopped["judges_partial"] == {
        MODEL: {"status": "ok", "correct": True, "verdict": "valid", "raw": ["yes"]}}
    assert "judges" not in stopped
    assert score.invalid(stopped, [MODEL, EXTRA]) == {"not_run"}
    path = arm(tmp_path, "off", [stopped])
    update_manifest(path, judges=[MODEL, EXTRA], status="incomplete")
    result = score.score([path], tmp_path / "report", allow_partial=True)
    summary = next(iter(result["arms"].values()))
    assert summary["scored_n"] == summary["valid_n"] == 0
    assert summary["accuracy"] == {MODEL: None, EXTRA: None}
    assert summary["invalid_by_reason"] == {"not_run": 1}


def test_4_glm_admission_and_delivery_share_known_model_map():
    with pytest.raises(argparse.ArgumentTypeError, match="reader context length unavailable"):
        models.reader_spec("glm:unknown")
    client = object.__new__(models.Backends)
    client.codex = client.claude = client.openrouter = None
    client.glm = object()
    with pytest.raises(KeyError, match="unknown"):
        client.context_length("glm:unknown")
    assert models.reader_spec("glm:glm-5.3") == "glm:glm-5.3"
    assert client.context_length("glm:glm-5.3") == models.GLM_CONTEXT["glm-5.3"] == 128_000
