import json
import subprocess
import sys

import pytest

from bench.tools.d8_guard import classify_recall_health, compare_runs, degraded_rate, lower_bound


def health(coverage=None, reason=None, degraded=False, timeout=False):
    return {"coverage": coverage, "degradedReason": reason, "degraded": degraded, "timeout": timeout}


@pytest.mark.parametrize("value,config,expected", [
    (None, "OFF", "missing"),
    (health({"fts": "ok"}), None, "missing"),
    (health({"fts": "ok"}), "UNKNOWN", "missing"),
    (health({"fts": "ok"}), "VOYAGE-TRACKER", "missing"),
    (health(), "OFF", "missing"),  # quota has no coverage
    (health({"fts": "ok"}), "OFF", "healthy"),
    (health({"fts": "ok"}, "semantic retrieval is disabled", True), "OFF", "healthy"),
    (health({"fts": "ok"}, "semantic retrieval is disabled; unavailable", True), "OFF", "unhealthy"),
    (health({"fts": "ok"}, "unavailable", True), "OFF", "unhealthy"),
    (health({"fts": "ok"}, None, True), "OFF", "unhealthy"),
    (health({}), "OFF", "unhealthy"),
    (health({"fts": "none"}), "OFF", "unhealthy"),
    (health({"fts": "ok"}, timeout=True), "OFF", "unhealthy"),
    (health({"summary": "full"}, timeout=True), "LOCAL", "unhealthy"),
    (health({"summary": "full"}), "LOCAL", "healthy"),  # one arm is enough
    (health({"chunk": "full", "fts": "ok"}), "LOCAL", "healthy"),
    (health({"fts": "ok"}), "LOCAL", "unhealthy"),
    (health({"summary": "full", "fts": "none"}), "LOCAL", "unhealthy"),
    (health({"summary": "bounded"}), "LOCAL", "unhealthy"),
    (health({"summary": "none"}), "LOCAL", "unhealthy"),
    (health({"chunk": "disabled"}), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx"}), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx"}, "chunk arm coverage full_approx: x", True), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx"}, "summary arm coverage full_approx: x", True), "LOCAL", "healthy"),
    (health({"summary": "full_approx"}, "summary arm coverage full_approx: x"), "LOCAL", "unhealthy"),
    (health({"summary": "full", "chunk": "full_approx"}, "chunk arm coverage full_approx: x", True), "LOCAL", "healthy"),
    (health({"summary": "full_approx", "chunk": "full_approx"},
            "summary arm coverage full_approx: x; chunk arm coverage full_approx: y", True), "LOCAL", "healthy"),
    (health({"summary": "full_approx", "chunk": "full_approx"},
            "summary arm coverage full_approx: x", True), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx"},
            "summary arm coverage full_approx: x; query embedding failed", True), "LOCAL", "unhealthy"),
    (health({"summary": "full"}, "provider preparing", True), "LOCAL", "unhealthy"),
    (health({"summary": "full"}, "stale identity", True), "LOCAL", "unhealthy"),
    (health({"summary": "full"}, None, True), "LOCAL", "unhealthy"),
])
def test_classifier(value, config, expected):
    assert classify_recall_health(value, config) == expected


def row(qid, score=1, kind="knowledge-update", **extra):
    return {"questionId": qid, "score": score, "questionType": kind,
            "recallHealth": health({"fts": "ok"}), **extra}


def write_run(path, rows, benchmark="longmemeval", config="OFF"):
    path.mkdir()
    (path / "report.json").write_text(json.dumps({
        "benchmark": benchmark, "recallConfig": config, "evaluations": rows}))
    return path


def test_missing_health_and_failed_rows_count_in_all_question_denominator():
    result = degraded_rate([row("a"), row("b", recallHealth=None),
                            row("c", status="failed"), row("d", recallHealth=health({"fts": "ok"}, timeout=True))], "OFF")
    assert result == {"total": 4, "degraded": 3, "rate_points": 75, "missing": 1, "INCOMPLETE": True}


def test_missing_candidate_is_incorrect_and_extra_question_excluded(tmp_path):
    baseline = write_run(tmp_path / "b", [row("a_abs"), row("b")])
    candidate = write_run(tmp_path / "c", [row("b"), row("extra_abs")])
    result = compare_runs(candidate, baseline)
    assert result["categories"]["abstention"]["lower_bound_points"] == -100
    assert result["categories"]["abstention"]["paired_n"] == 1
    assert result["excluded_absent_from_baseline"] == 1
    assert result["runs"]["candidate"]["INCOMPLETE"]
    assert not result["pass"]


def test_failed_overlay_overrides_correct_score(tmp_path):
    baseline = write_run(tmp_path / "b", [row("a_abs"), row("b")])
    candidate = write_run(tmp_path / "c", [row("a_abs"), row("b")])
    (candidate / "per_question.jsonl").write_text(json.dumps({"question_id": "a_abs", "status": "failed"}) + "\n")
    result = compare_runs(candidate, baseline)["categories"]["abstention"]
    assert result["lower_bound_points"] == -100
    assert result["candidate"]["degraded"] == 1


def test_checkpoint_retains_unevaluated_questions(tmp_path):
    baseline = write_run(tmp_path / "b", [row("a_abs"), row("b")])
    candidate = write_run(tmp_path / "c", [])
    (candidate / "checkpoint.json").write_text(json.dumps({"questions": {
        "a_abs": {"questionType": "single-session-user", "phases": {"search": {"status": "failed"}}},
        "b": {"questionType": "knowledge-update", "phases": {}},
    }}))
    result = compare_runs(candidate, baseline)
    assert result["runs"]["candidate"]["total"] == 2
    assert result["runs"]["candidate"]["degraded"] == 2
    assert result["runs"]["candidate"]["INCOMPLETE"]


def test_seeded_nearest_rank_bound():
    # Unequal cluster sizes pin question weighting as well as seed/percentile.
    assert lower_bound([[1, 1], [-1], [0, 1, 0], [0]]) == -50.0


def test_exact_minus_two_fails(tmp_path):
    rows = [row(f"conversation-q{i}", kind="adversarial") for i in range(50)]
    baseline = write_run(tmp_path / "b", rows, "locomo")
    candidate = write_run(tmp_path / "c", [{**r, "score": 0 if i == 0 else 1} for i, r in enumerate(rows)], "locomo")
    result = compare_runs(candidate, baseline)["categories"]["adversarial"]
    assert result["clusters"] == 1
    assert result["lower_bound_points"] == -2.0
    assert not result["interval_pass"]


def test_rate_boundary_and_cli_exit(tmp_path):
    rows = [row("a_abs", kind="single-session-user")] + [row(f"q{i}") for i in range(100)]
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", rows)
    command = [sys.executable, "-m", "bench.tools.d8_guard", str(candidate), str(baseline)]
    passed = subprocess.run(command, capture_output=True, text=True)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["pass"]
    rows[1]["recallHealth"] = health({"fts": "none"})
    (candidate / "report.json").write_text(json.dumps({"benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    assert compare_runs(candidate, baseline)["categories"]["knowledge-update"]["rate_pass"] is True
    rows[2]["recallHealth"] = health({"fts": "none"})
    (candidate / "report.json").write_text(json.dumps({"benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    assert compare_runs(candidate, baseline)["categories"]["knowledge-update"]["rate_pass"] is False
    rows[0]["recallHealth"] = None
    (candidate / "report.json").write_text(json.dumps({"benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["runs"]["candidate"]["INCOMPLETE"]
