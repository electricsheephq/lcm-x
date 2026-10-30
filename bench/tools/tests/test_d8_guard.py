from fractions import Fraction
import json
import math
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


def write_run(path, rows, benchmark="longmemeval", config="OFF", **fields):
    path.mkdir()
    (path / "report.json").write_text(json.dumps({
        "benchmark": benchmark, "recallConfig": config, "evaluations": rows, **fields}))
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
    result = compare_runs(candidate, baseline, subset=True)
    assert not result["INCOMPLETE"]
    assert not result["pass"]
    assert result["categories"]["abstention"]["lower_bound_points"] == -100
    assert result["categories"]["abstention"]["candidate"]["degraded"] == 1
    assert not result["categories"]["abstention"]["candidate"]["INCOMPLETE"]


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
    assert not result["interval_inside_band"]


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
    """Pinned D8 category sizes (d8-count-margin-v1); `count` sets the run total."""
    if benchmark == "locomo":
        # 446 adversarial over ten conversations, then other categories.
        return [row(f"conv-{i % 10}-q{i}", kind="adversarial" if i < 446 else "single-hop")
                for i in range(1986 if count is None else count)]
    # rows[0] is abstention and rows[1] knowledge-update; rows[2:] up to the tail are neither.
    rows = [row("a_abs", kind="single-session-user"), row("b")]
    rows += [row(f"q{i}", kind="single-session-user") for i in range((500 if count is None else count) - 108)]
    rows += [row(f"x{i}_abs", kind="single-session-user") for i in range(29)]
    return rows + [row(f"k{i}") for i in range(77)]


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
    assert all(c["count_pass"] and c["gating"] for c in result["categories"].values())
    assert not result["INCOMPLETE"]


def test_category_rate_spike_does_not_gate_whole_run(tmp_path):
    rows = full_rows()
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [{**rows[0], "recallHealth": health({"fts": "none"})}, *rows[1:]])
    result = compare_runs(candidate, baseline)
    assert result["pass"]  # One unhealthy abstention is only 0.2 points overall.
    assert result["rate_pass"] is True
    assert result["categories"]["abstention"]["candidate"]["rate_points"] == pytest.approx(100 / 30)
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
    assert result["categories"]["abstention"]["paired_n"] == 30
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
    rows = full_rows(count=110)  # Pinned categories, 110 questions in all.
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


# d8-count-margin-v1: +/-2 now pass on the count margin (T(100) = 3); the interval only discloses.
@pytest.mark.parametrize("difference,expected,inside", [
    (-3, False, False), (-2, True, False), (-1, True, True), (0, True, True),
    (1, True, True), (2, True, False), (3, False, False)])
def test_placebo_count_margin_and_disclosed_interval(tmp_path, difference, expected, inside):
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
    assert category["interval_inside_band"] is inside
    assert category["interval_gates"] is False
    assert category["margin_questions"] == 3
    assert category["count_pass"] is expected
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
    abstention = result["categories"]["abstention"]
    assert (abstention["c"], abstention["c_failed"]) == (1, 1)  # The overlay is a loss.
    assert result["runs"]["candidate"]["degraded"] == 1
    assert result["pass"]  # d8-count-margin-v1: one abstention loss is inside T(30) = 3.
    assert not result["INCOMPLETE"]
    assert not result["runs"]["candidate"]["INCOMPLETE"]


# d8-count-margin-v1: two-way fixtures for every rule branch.
CATEGORY_ROWS = {
    "abstention": lambda r: r["questionId"].endswith("_abs"),
    "knowledge-update": lambda r: r["questionType"] == "knowledge-update",
    "adversarial": lambda r: r["questionType"] == "adversarial",
    "conv-0": lambda r: r["questionType"] == "adversarial" and r["questionId"].startswith("conv-0-"),
    "conv-0-4": lambda r: r["questionType"] == "adversarial" and int(r["questionId"].split("-")[1]) < 5,
    "conv-5-9": lambda r: r["questionType"] == "adversarial" and int(r["questionId"].split("-")[1]) >= 5,
}


def counted_pair(tmp_path, counts, benchmark="longmemeval", rows=None, failed=None, **fields):
    """Full-row pair with b candidate gains and c candidate losses per selected category.

    `failed` marks the losing side's discordant rows as failed ("candidate" or "baseline").
    """
    rows = full_rows(benchmark) if rows is None else rows
    base, cand = [dict(r) for r in rows], [dict(r) for r in rows]
    for name, (b, c) in counts.items():
        picked = [i for i, r in enumerate(rows) if CATEGORY_ROWS[name](r)]
        for i in picked[:b]:
            base[i]["score"] = 0
            if failed == "baseline":
                base[i]["status"] = "failed"
        for i in picked[b:b + c]:
            cand[i]["score"] = 0
            if failed == "candidate":
                cand[i]["status"] = "failed"
    return (write_run(tmp_path / "c", cand, benchmark, **fields),
            write_run(tmp_path / "b", base, benchmark, **fields))


@pytest.mark.parametrize("benchmark,counts", [
    ("longmemeval", {"abstention": (0, 0), "knowledge-update": (0, 0)}),
    ("longmemeval", {"abstention": (1, 0)}),
    ("longmemeval", {"knowledge-update": (1, 0)}),
    ("locomo", {"adversarial": (15, 12)}),
    ("locomo", {"adversarial": (12, 15)}),
    ("locomo", {"adversarial": (19, 11)}),  # boundary: net 8
    ("longmemeval", {"abstention": (2, 0)}),  # boundary
    ("longmemeval", {"knowledge-update": (0, 2)}),  # boundary
])
def test_placebo_count_margin_passes(tmp_path, benchmark, counts):
    result = compare_runs(*counted_pair(tmp_path, counts, benchmark), placebo=True)
    assert result["pass"]
    assert not result["INCOMPLETE"]
    for name, (b, c) in counts.items():
        category = result["categories"][name]
        assert (category["b"], category["c"], category["net"]) == (b, c, b - c)
        assert category["count_pass"] and category["gating"] and category["pass"]
        assert category["interval_gates"] is False


def test_placebo_old_interval_no_longer_gates(tmp_path):
    # 12/15 with gains and losses in different conversations: a wide interval.
    counts = {"conv-0-4": (12, 0), "conv-5-9": (0, 15)}
    result = compare_runs(*counted_pair(tmp_path, counts, "locomo"), placebo=True)
    category = result["categories"]["adversarial"]
    assert (category["b"], category["c"]) == (12, 15)
    assert category["lower_bound_points"] < -2.0
    assert not category["interval_inside_band"]
    assert result["pass"]


@pytest.mark.parametrize("benchmark,counts", [
    ("longmemeval", {"abstention": (3, 0)}),
    ("locomo", {"adversarial": (12, 21)}),
    ("locomo", {"adversarial": (21, 12)}),
    ("locomo", {"adversarial": (13, 24)}),
])
def test_placebo_count_margin_fails(tmp_path, benchmark, counts):
    result = compare_runs(*counted_pair(tmp_path, counts, benchmark), placebo=True)
    assert not result["pass"]
    assert not result["INCOMPLETE"]
    assert not result["categories"][next(iter(counts))]["count_pass"]


@pytest.mark.parametrize("defect", ["failed baseline", "missing health"])
def test_placebo_count_fail_with_defect_is_incomplete(tmp_path, defect):
    rows = full_rows("locomo")
    base_rows, cand_rows = [dict(r) for r in rows], [dict(r) for r in rows]
    if defect == "failed baseline":
        base_rows[-1]["status"] = "failed"  # Outside the category; b/c unchanged.
    else:
        cand_rows[-1]["recallHealth"] = None
    picked = [i for i, r in enumerate(rows) if r["questionType"] == "adversarial"]
    for i in picked[:13]:
        base_rows[i]["score"] = 0
    for i in picked[13:37]:
        cand_rows[i]["score"] = 0
    result = compare_runs(write_run(tmp_path / "c", cand_rows, "locomo"),
                          write_run(tmp_path / "b", base_rows, "locomo"), placebo=True)
    category = result["categories"]["adversarial"]
    assert (category["b"], category["c"]) == (13, 24)
    assert not category["count_pass"]
    assert result["INCOMPLETE"]  # INCOMPLETE takes precedence over the count verdict.
    assert not result["pass"]
    if defect == "failed baseline":
        assert result["reason"] == "baseline: failed rows: 1 (rerun the baseline)"


@pytest.mark.parametrize("benchmark,counts,expected", [
    ("longmemeval", {"abstention": (0, 3)}, False),
    ("longmemeval", {"abstention": (3, 0)}, True),
    ("longmemeval", {"knowledge-update": (0, 3)}, False),
    ("locomo", {"adversarial": (12, 21)}, False),
    ("locomo", {"adversarial": (21, 12)}, True),
    ("locomo", {"adversarial": (11, 19)}, True),  # c - b = 8
    # All of a net -8 in one conversation still passes: the disclosed loss of cluster sensitivity.
    ("locomo", {"conv-0": (0, 8)}, True),
])
def test_net_loss_margin(tmp_path, benchmark, counts, expected):
    result = compare_runs(*counted_pair(tmp_path, counts, benchmark))
    assert result["mode"] == "non-inferiority"
    assert result["claim"] == "count-margin"
    assert result["rate_pass"] is True
    assert result["pass"] is expected
    name = "adversarial" if benchmark == "locomo" else next(iter(counts))
    assert result["categories"][name]["count_pass"] is expected
    if "conv-0" in counts:
        category = result["categories"]["adversarial"]
        assert category["per_conversation"]["conv-0"] == {"n": 45, "b": 0, "c": 8, "net": -8}
        assert sum(v["net"] for v in category["per_conversation"].values()) == -8
        assert category["sign_flip_p"] == 1.0


def test_net_loss_clean_counts_fail_on_degraded_rate(tmp_path):
    rows = full_rows()
    candidate = write_run(tmp_path / "c", [{**r, "recallHealth": health({"fts": "none"})} if 2 <= i < 8 else r
                                           for i, r in enumerate(rows)])
    result = compare_runs(candidate, write_run(tmp_path / "b", rows))
    assert all(c["count_pass"] for c in result["categories"].values())
    assert result["rate_pass"] is False
    assert not result["pass"]


@pytest.mark.parametrize("benchmark", ["longmemeval", "locomo"])
def test_full_row_empty_category_is_incomplete(tmp_path, benchmark):
    if benchmark == "locomo":
        rows = [{**r, "questionType": "single-hop"} for r in full_rows("locomo")]
        name = "adversarial"
    else:
        rows = [{**r, "questionId": r["questionId"].replace("_abs", "_x")} for r in full_rows()]
        name = "abstention"
    result = compare_runs(write_run(tmp_path / "c", rows, benchmark), write_run(tmp_path / "b", rows, benchmark))
    category = result["categories"][name]
    assert category["paired_n"] == 0 and category["INCOMPLETE"] and not category["pass"]
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert f"category {name}: n 0" in result["reason"]


@pytest.mark.parametrize("placebo", [False, True])
def test_full_row_category_size_must_match_pinned(tmp_path, placebo):
    rows = full_rows()
    rows[-1] = row("k-moved", kind="multi-session")  # knowledge-update 77, total still 500.
    result = compare_runs(write_run(tmp_path / "c", rows), write_run(tmp_path / "b", rows), placebo=placebo)
    assert result["categories"]["knowledge-update"]["paired_n"] == 77
    assert result["categories"]["knowledge-update"]["INCOMPLETE"]
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert "category knowledge-update: n 77 (pinned 78)" in result["reason"]
    assert compare_runs(tmp_path / "c", tmp_path / "b", subset=True, placebo=placebo)["pass"]


def test_subset_small_category_is_not_gating_and_never_passes(tmp_path):
    rows = [row("a_abs", kind="single-session-user"), row("b_abs", kind="single-session-user"),
            row("k0"), row("k1")]
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", rows)
    for placebo in (False, True):
        result = compare_runs(candidate, baseline, subset=True, placebo=placebo)
        for category in result["categories"].values():
            assert category["margin_questions"] == 3
            assert category["can_fail"] is False and category["gating"] is False
            assert category["count_pass"] and not category["pass"]
        assert not result["INCOMPLETE"]
        assert not result["pass"]  # No gating category.
    # One gating category carries the row; the small one stays non-gating.
    rows += [row(f"ku{i}") for i in range(76)]
    result = compare_runs(write_run(tmp_path / "c2", rows), write_run(tmp_path / "b2", rows), subset=True)
    assert result["categories"]["knowledge-update"]["gating"]
    assert not result["categories"]["abstention"]["gating"]
    assert not result["categories"]["abstention"]["pass"]
    assert result["pass"]


@pytest.mark.parametrize("losses,expected", [(2, True), (3, False)])
def test_reader_treatment_uses_count_rule(tmp_path, losses, expected):
    candidate, baseline = counted_pair(tmp_path, {"knowledge-update": (0, losses)}, readerPromptVersion="v1")
    report = json.loads((candidate / "report.json").read_text())
    (candidate / "report.json").write_text(json.dumps({**report, "readerPromptVersion": "v2"}))
    result = compare_runs(candidate, baseline, reader_treatment=True)
    assert result["identity_differences"] == {"readerPromptVersion": {"candidate": "v2", "baseline": "v1"}}
    assert not result["INCOMPLETE"]
    assert result["categories"]["knowledge-update"]["count_pass"] is expected
    assert result["pass"] is expected


@pytest.mark.parametrize("n,expected", [(30, 3), (78, 3), (446, 9), (450, 9), (500, 10)])
def test_count_margin_values(n, expected):
    assert d8_guard.count_margin(n) == expected


def test_count_margin_integer_form_matches_ceiling():
    for n in range(1, 5001):
        assert d8_guard.count_margin(n) == max(3, math.ceil(Fraction(2 * n, 100)))


@pytest.mark.parametrize("placebo,failed,field", [
    (True, "candidate", "c_failed"), (False, "candidate", "c_failed"), (False, "baseline", "b_failed")])
def test_failed_discordances_are_split_out(tmp_path, placebo, failed, field):
    counts = {"knowledge-update": (2, 2)}
    result = compare_runs(*counted_pair(tmp_path, counts, failed=failed), placebo=placebo)
    category = result["categories"]["knowledge-update"]
    assert (category["b"], category["c"]) == (2, 2)
    assert category[field] == 2
    assert category["b_failed" if field == "c_failed" else "c_failed"] == 0
    assert result["INCOMPLETE"] is (failed == "baseline")


def test_receipt_fields(tmp_path):
    result = compare_runs(*counted_pair(tmp_path, {"adversarial": (15, 12)}, "locomo"), placebo=True)
    assert result["rule_id"] == "d8-count-margin-v1"
    category = result["categories"]["adversarial"]
    assert category["interval_gates"] is False
    assert {"lower_bound_points", "upper_bound_points", "interval_inside_band"} <= category.keys()
    assert "interval_pass" not in category
    assert (category["discordant"], category["net"], category["margin_questions"]) == (27, 3, 9)
    assert category["margin_points"] == pytest.approx(100 * 9 / 446)
    assert category["discordance_rate"] == pytest.approx(27 / 446)
    assert category["mcnemar_p"] == pytest.approx(
        min(1, 2 * sum(math.comb(27, k) for k in range(13)) / 2 ** 27))
    assert len(category["per_conversation"]) == 10
    assert sum(v["n"] for v in category["per_conversation"].values()) == 446
    assert 0 < category["sign_flip_p"] <= 1
    assert category["cluster_dispersion"] is not None
    (tmp_path / "lme").mkdir()
    lme = compare_runs(*counted_pair(tmp_path / "lme", {}), placebo=True)
    assert "per_conversation" not in lme["categories"]["abstention"]
    assert lme["categories"]["abstention"]["mcnemar_p"] == 1.0


def test_disclosure_statistics():
    assert d8_guard.mcnemar_p(0, 3) == 0.25
    assert d8_guard.mcnemar_p(0, 0) == 1.0
    assert d8_guard.sign_flip_p([3, 1]) == 0.5
    assert d8_guard.sign_flip_p([2, -2]) == 1.0
    assert d8_guard.sign_flip_p([1] * 10) == 2 / 2 ** 10
    conversations = {"x": {"b": 2, "c": 0, "net": 2}, "y": {"b": 0, "c": 2, "net": -2}}
    assert d8_guard.cluster_dispersion(conversations, 0, 4) == 2.0
    assert d8_guard.cluster_dispersion({"x": {"b": 0, "c": 0, "net": 0}}, 0, 0) is None
