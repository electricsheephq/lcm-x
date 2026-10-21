import json
import subprocess
import sys

import pytest

from bench.tools import d8_guard
from bench.tools.d8_guard import classify_recall_health, compare_runs, degraded_rate


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
    (health({"summary": "full", "chunk": "full", "fts": "ok"}), "LOCAL", "healthy"),
    (health({"fts": "ok"}), "LOCAL", "unhealthy"),
    (health({"summary": "full", "chunk": "full", "fts": "none"}), "LOCAL", "unhealthy"),
    (health({"summary": "bounded", "chunk": "full", "fts": "ok"}), "LOCAL", "unhealthy"),
    (health({"summary": "none", "chunk": "full", "fts": "ok"}), "LOCAL", "unhealthy"),
    (health({"summary": "full", "chunk": "disabled", "fts": "ok"}), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx", "chunk": "full", "fts": "ok"}), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx", "chunk": "full", "fts": "ok"},
            "chunk arm coverage full_approx: x", True), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx", "chunk": "full", "fts": "ok"},
            "summary arm coverage full_approx: x", True), "LOCAL", "healthy"),
    (health({"summary": "full_approx", "chunk": "full", "fts": "ok"},
            "summary arm coverage full_approx: x"), "LOCAL", "unhealthy"),
    (health({"summary": "full", "chunk": "full_approx", "fts": "ok"},
            "chunk arm coverage full_approx: x", True), "LOCAL", "healthy"),
    (health({"summary": "full_approx", "chunk": "full_approx", "fts": "ok"},
            "summary arm coverage full_approx: x; chunk arm coverage full_approx: y", True), "LOCAL", "healthy"),
    (health({"summary": "full_approx", "chunk": "full_approx", "fts": "ok"},
            "summary arm coverage full_approx: x", True), "LOCAL", "unhealthy"),
    (health({"summary": "full_approx", "chunk": "full", "fts": "ok"},
            "summary arm coverage full_approx: x; query embedding failed", True), "LOCAL", "unhealthy"),
    (health({"summary": "full", "chunk": "full", "fts": "ok"}, "provider preparing", True), "LOCAL", "unhealthy"),
    (health({"summary": "full", "chunk": "full", "fts": "ok"}, "stale identity", True), "LOCAL", "unhealthy"),
    (health({"summary": "full", "chunk": "full", "fts": "ok"}, None, True), "LOCAL", "unhealthy"),
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
    result = compare_runs(candidate, baseline, subset=True)
    assert result["categories"]["abstention"]["lower_bound_points"] == -100
    assert result["categories"]["abstention"]["paired_n"] == 1
    assert result["excluded_absent_from_baseline"] == 1
    assert result["runs"]["candidate"]["INCOMPLETE"]
    assert not result["pass"]


def test_failed_overlay_overrides_correct_score(tmp_path):
    baseline = write_run(tmp_path / "b", [row("a_abs"), row("b")])
    candidate = write_run(tmp_path / "c", [row("a_abs"), row("b")])
    (candidate / "per_question.jsonl").write_text(json.dumps({"question_id": "a_abs", "status": "failed"}) + "\n")
    result = compare_runs(candidate, baseline, subset=True)["categories"]["abstention"]
    assert result["lower_bound_points"] == -100
    assert result["candidate"]["degraded"] == 1


def test_checkpoint_retains_unevaluated_questions(tmp_path):
    baseline = write_run(tmp_path / "b", [row("a_abs"), row("b")])
    candidate = write_run(tmp_path / "c", [])
    (candidate / "checkpoint.json").write_text(json.dumps({"questions": {
        "a_abs": {"questionType": "single-session-user", "phases": {"search": {"status": "failed"}}},
        "b": {"questionType": "knowledge-update", "phases": {}},
    }}))
    result = compare_runs(candidate, baseline, subset=True)
    assert result["runs"]["candidate"]["total"] == 2
    assert result["runs"]["candidate"]["degraded"] == 2
    assert result["runs"]["candidate"]["INCOMPLETE"]


def test_seeded_nearest_rank_bound():
    # Unequal cluster sizes pin question weighting as well as seed/percentile.
    assert d8_guard.bootstrap_bounds([[1, 1], [-1], [0, 1, 0], [0]])[0] == -50.0


def test_exact_minus_two_fails(tmp_path):
    rows = [row(f"conversation-q{i}", kind="adversarial") for i in range(50)]
    baseline = write_run(tmp_path / "b", rows, "locomo")
    candidate = write_run(tmp_path / "c", [{**r, "score": 0 if i == 0 else 1} for i, r in enumerate(rows)], "locomo")
    result = compare_runs(candidate, baseline, subset=True)["categories"]["adversarial"]
    assert result["clusters"] == 1
    assert result["lower_bound_points"] == -2.0
    assert not result["interval_pass"]


def test_rate_boundary_and_cli_exit(tmp_path):
    rows = [row("a_abs", kind="single-session-user")] + [row(f"q{i}") for i in range(100)]
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", rows)
    command = [sys.executable, "-m", "bench.tools.d8_guard", "--subset", str(candidate), str(baseline)]
    passed = subprocess.run(command, capture_output=True, text=True)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["pass"]
    rows[1]["recallHealth"] = health({"fts": "none"})
    (candidate / "report.json").write_text(json.dumps({"benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    assert compare_runs(candidate, baseline, subset=True)["rate_pass"] is True
    rows[2]["recallHealth"] = health({"fts": "none"})
    (candidate / "report.json").write_text(json.dumps({"benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    assert compare_runs(candidate, baseline, subset=True)["rate_pass"] is False
    rows[0]["recallHealth"] = None
    (candidate / "report.json").write_text(json.dumps({"benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["runs"]["candidate"]["INCOMPLETE"]


def full_rows(benchmark="longmemeval", count=None):
    if benchmark == "locomo":
        return [row(f"conversation-q{i}", kind="adversarial")
                for i in range(1986 if count is None else count)]
    rows = [row("a_abs", kind="single-session-user"), row("b")]
    return rows + [row(f"q{i}", kind="single-session-user")
                   for i in range((500 if count is None else count) - 2)]


def test_local_requires_fulltext_coverage():
    assert classify_recall_health(health({"summary": "full", "chunk": "full"}), "LOCAL") == "unhealthy"


@pytest.mark.parametrize("missing_arm", ["summary", "chunk"])
def test_local_requires_both_semantic_keys(missing_arm):
    coverage = {"fts": "ok", "summary": "full", "chunk": "full"}
    del coverage[missing_arm]
    assert classify_recall_health(health(coverage), "LOCAL") == "unhealthy"


def test_whole_run_rate_includes_questions_outside_categories(tmp_path):
    rows = full_rows()
    baseline = write_run(tmp_path / "b", rows)
    candidate_rows = [{**r, "recallHealth": health({"fts": "none"})} if 2 <= i < 8 else r
                      for i, r in enumerate(rows)]
    candidate = write_run(tmp_path / "c", candidate_rows)
    result = compare_runs(candidate, baseline)
    assert not result["pass"]  # Category-only rates were both zero.
    assert result["rate_pass"] is False
    assert result["runs"]["candidate"]["rate_points"] == 1.2
    assert all(c["interval_pass"] for c in result["categories"].values())
    assert not result["INCOMPLETE"]


def test_category_rate_spike_does_not_gate_whole_run(tmp_path):
    rows = full_rows()
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [{**rows[0], "recallHealth": health({"fts": "none"})}, *rows[1:]])
    result = compare_runs(candidate, baseline)
    assert result["pass"]  # One unhealthy abstention is only 0.2 points overall.
    assert result["rate_pass"] is True
    assert result["categories"]["abstention"]["candidate"]["rate_points"] == 100
    assert "rate_pass" not in result["categories"]["abstention"]


def test_overall_rate_exactly_one_point_passes(tmp_path):
    rows = full_rows()
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [
        {**r, "recallHealth": health({"fts": "none"})} if 2 <= i < 7 else r
        for i, r in enumerate(rows)])
    result = compare_runs(candidate, baseline)
    assert result["rate_pass"] is True
    assert result["runs"]["candidate"]["rate_points"] == 1
    assert result["pass"]


def test_candidate_only_ids_are_incomplete_and_cli_fails(tmp_path):
    rows = full_rows()
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [*rows, row("extra_abs")])
    result = compare_runs(candidate, baseline)
    assert not result["pass"]
    assert result["INCOMPLETE"]
    assert result["excluded_absent_from_baseline"] == 1
    assert result["categories"]["abstention"]["paired_n"] == 1
    output = subprocess.run([sys.executable, "-m", "bench.tools.d8_guard", str(candidate), str(baseline)],
                            capture_output=True, text=True)
    assert output.returncode == 1
    assert json.loads(output.stdout)["INCOMPLETE"]
    assert not compare_runs(candidate, baseline, subset=True)["pass"]


@pytest.mark.parametrize("benchmark,count", [
    ("longmemeval", 499), ("longmemeval", 501), ("locomo", 1985), ("locomo", 1987),
])
def test_default_requires_exact_d8_baseline_count(tmp_path, benchmark, count):
    rows = full_rows(benchmark, count)
    baseline = write_run(tmp_path / "b", rows, benchmark)
    candidate = write_run(tmp_path / "c", rows, benchmark)
    result = compare_runs(candidate, baseline)
    assert not result["pass"]
    assert result["INCOMPLETE"]
    assert result["d8_row"] is True
    assert result["runs"]["baseline"]["total"] == count


@pytest.mark.parametrize("benchmark", ["longmemeval", "locomo"])
def test_full_d8_count_is_labelled_and_passes(tmp_path, benchmark):
    rows = full_rows(benchmark)
    baseline = write_run(tmp_path / "b", rows, benchmark)
    candidate = write_run(tmp_path / "c", rows, benchmark)
    result = compare_runs(candidate, baseline)
    assert result["d8_row"] is True
    assert not result["INCOMPLETE"]
    assert result["pass"]


def test_subset_cli_skips_count_and_is_not_a_d8_row(tmp_path):
    rows = [row("a_abs"), row("b")]
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", rows)
    command = [sys.executable, "-m", "bench.tools.d8_guard", str(candidate), str(baseline)]
    output = subprocess.run([*command, "--subset"], capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    result = json.loads(output.stdout)
    assert result["d8_row"] is False
    assert not result["INCOMPLETE"]
    assert result["pass"]
    full = subprocess.run(command, capture_output=True, text=True)
    assert full.returncode == 1
    assert json.loads(full.stdout)["INCOMPLETE"]


def test_bootstrap_returns_both_seeded_nearest_rank_bounds():
    assert d8_guard.bootstrap_bounds([[1, 1], [-1], [0, 1, 0], [0]]) == (-50.0, 77.77777777777777)
    assert d8_guard.bootstrap_bounds([]) == (None, None)


def test_bootstrap_upper_is_9750th_value(monkeypatch):
    class Resamples:
        def __init__(self, seed):
            assert seed == 1607
            self.calls = 0

        def choices(self, totals, k):
            assert k == 1
            self.calls += 1
            return [(self.calls, 100)]

    monkeypatch.setattr(d8_guard.random, "Random", Resamples)
    assert d8_guard.bootstrap_bounds([[0]]) == (250.0, 9750.0)


@pytest.mark.parametrize("difference,expected", [(-3, False), (-2, False), (-1, True),
                                                (0, True), (1, True), (2, False), (3, False)])
def test_placebo_requires_strict_two_sided_interval(tmp_path, difference, expected):
    # One LoCoMo conversation makes every resample's paired difference exact.
    baseline_rows = [row(f"conversation-q{i}", score=int(i >= 50), kind="adversarial") for i in range(100)]
    candidate_rows = [dict(r) for r in baseline_rows]
    indices = range(abs(difference)) if difference > 0 else range(50, 50 + abs(difference))
    for i in indices:
        candidate_rows[i]["score"] = int(difference > 0)
    baseline = write_run(tmp_path / "b", baseline_rows, "locomo")
    candidate = write_run(tmp_path / "c", candidate_rows, "locomo")
    result = compare_runs(candidate, baseline, subset=True, placebo=True)
    category = result["categories"]["adversarial"]
    assert result["mode"] == "placebo"
    assert category["clusters"] == 1
    assert category["paired_n"] == 100
    assert category["lower_bound_points"] == difference
    assert category["upper_bound_points"] == difference
    assert category["interval_pass"] is expected
    assert result["pass"] is expected
    if difference == 3:
        assert compare_runs(candidate, baseline, subset=True)["pass"]


def test_placebo_cli_does_not_apply_rate_gate(tmp_path):
    rows = full_rows()
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [
        {**r, "recallHealth": health({"fts": "none"})} if 2 <= i < 8 else r
        for i, r in enumerate(rows)])
    output = subprocess.run([sys.executable, "-m", "bench.tools.d8_guard", str(candidate), str(baseline), "--placebo"],
                            capture_output=True, text=True)
    assert output.returncode == 0, output.stderr
    result = json.loads(output.stdout)
    assert result["mode"] == "placebo"
    assert result["rate_pass"] is None
    assert result["runs"]["candidate"]["rate_points"] == 1.2
    assert result["pass"]
    # Placebo never bypasses completeness, even though its rate rule is disabled.
    rows[0]["recallHealth"] = None
    (candidate / "report.json").write_text(json.dumps({
        "benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    result = compare_runs(candidate, baseline, placebo=True)
    assert result["INCOMPLETE"]
    assert not result["pass"]


@pytest.mark.parametrize("status", ["error", "pending", "in_progress", "timed_out", "unknown", "", None])
def test_non_success_status_overrides_stale_score_and_healthy_recall(status):
    failed = row("a_abs", status=status)
    assert not d8_guard._is_correct(failed)
    assert degraded_rate([failed], "OFF")["degraded"] == 1
    for valid in [row("a_abs"), row("a_abs", status="completed")]:
        assert d8_guard._is_correct(valid)
        assert degraded_rate([valid], "OFF")["degraded"] == 0


def test_error_overlay_overrides_stale_evaluation(tmp_path):
    rows = full_rows()
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", rows)
    (candidate / "per_question.jsonl").write_text(json.dumps({"question_id": "a_abs", "status": "error"}) + "\n")
    result = compare_runs(candidate, baseline)
    assert result["categories"]["abstention"]["lower_bound_points"] == -100
    assert result["runs"]["candidate"]["degraded"] == 1
    assert not result["pass"]
