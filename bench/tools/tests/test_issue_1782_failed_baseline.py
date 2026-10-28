"""A failed baseline cannot certify candidate non-inferiority (#1782)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bench.tools.d8_guard import compare_runs
from bench.tools.tests.test_d8_guard import row, write_run
from bench.tools.tests.test_issue_1740_overlay import overlay


FAILURES = [
    {"status": "failed"}, {"error": "judge failed"}, {"judge_fallback": True},
    {"_failed": True}, {"hypothesis": "fail-close: missing source"},
]


def repro_pair(tmp_path, failure=None, *, candidate_correct=False):
    rows = [row(f"ku{i}") for i in range(78)]
    rows += [row(f"q{i}_abs", kind="single-session-user") for i in range(30)]
    rows += [row(f"q{i}", kind="multi-session") for i in range(392)]
    candidate = write_run(tmp_path / "c", [
        {**r, "score": int(candidate_correct or i >= 3)} for i, r in enumerate(rows)])
    baseline = write_run(tmp_path / "b", [
        {**r, **(failure or {})} if i < 3 else r for i, r in enumerate(rows)])
    return candidate, baseline


@pytest.mark.parametrize("failure", FAILURES)
def test_full_repro_failed_baseline_is_incomplete(tmp_path, failure):
    candidate, baseline = repro_pair(tmp_path, failure)
    result = compare_runs(candidate, baseline)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert result["reason"] == "baseline: failed rows: 3 (rerun the baseline)"
    assert result["categories"]["knowledge-update"]["paired_n"] == 78


def test_same_repro_without_failures_retains_loss(tmp_path):
    candidate, baseline = repro_pair(tmp_path)
    result = compare_runs(candidate, baseline)
    assert not result["INCOMPLETE"]
    assert not result["pass"]
    assert result["categories"]["knowledge-update"]["lower_bound_points"] == pytest.approx(-8.974358974358974)


@pytest.mark.parametrize("source", ["per_question.jsonl", "results/per_question.jsonl", "checkpoint.json"])
def test_failure_after_overlay_outside_categories_is_incomplete(tmp_path, source):
    candidate, baseline = repro_pair(tmp_path, candidate_correct=True)
    if source == "checkpoint.json":
        (baseline / source).write_text(json.dumps({"questions": {
            "q0": {"phases": {p: {"status": "failed" if p == "evaluate" else "completed"}
                              for p in ("search", "answer", "evaluate")}}}}))
    else:
        overlay(baseline, source, [{"question_id": "q0", "_failed": True}])
    result = compare_runs(candidate, baseline)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert result["reason"] == "baseline: failed rows: 1 (rerun the baseline)"
    assert all(category["pass"] for category in result["categories"].values())


@pytest.mark.parametrize("placebo", [False, True])
def test_clean_writer_fixture_output_matches_base(placebo):
    root = Path(__file__).parent / "fixtures" / "d8_writer_pair"
    expected = json.loads((root / "clean_comparison.json").read_text())
    mode = "placebo" if placebo else "non-inferiority"
    assert compare_runs(root / "a", root / "c", subset=True, placebo=placebo) == expected[mode]


def test_full_repro_cli_reports_incomplete_and_rerun_reason(tmp_path):
    candidate, baseline = repro_pair(tmp_path, {"status": "failed"})
    output = subprocess.run([sys.executable, "-m", "bench.tools.d8_guard", str(candidate), str(baseline)],
                            capture_output=True, text=True)
    assert output.returncode == 1, output.stderr
    result = json.loads(output.stdout)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert result["reason"] == "baseline: failed rows: 3 (rerun the baseline)"
