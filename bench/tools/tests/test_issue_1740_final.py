"""Final-round accounting and baseline-conflict regressions for #1740."""
import json

import pytest

from bench.tools import d8_guard, failclose
from bench.tools.tests.test_d8_guard import row, write_run
from bench.tools.tests.test_issue_1740_overlay import FAILURES, SOURCES, overlay


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("verdict", [{"correct": "malformed"}, {"score": "malformed"},
                                     {"label": "malformed"}])
@pytest.mark.parametrize("side", ["candidate", "baseline"])
def test_accounting_rejects_selected_malformed_verdict(tmp_path, capsys, source, verdict, side):
    candidate = write_run(tmp_path / "c", [row("a_abs", score=0), row("b")])
    baseline = write_run(tmp_path / "b", [row("a_abs", score=0), row("b")])
    run = candidate if side == "candidate" else baseline
    overlay(run, source, [{"question_id": "a_abs", **verdict}])
    with pytest.raises(ValueError):
        failclose.account_run(run)
    assert failclose.main([str(candidate), "--compare", str(baseline)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(run) in captured.err
    assert len(captured.err.splitlines()) == 1
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("verdict", [{"correct": None, "score": 1},
                                     {"correct": 1}, {"score": True},
                                     {"score": float("inf")}, {"score": -float("inf")},
                                     {"score": float("nan")}])
@pytest.mark.parametrize("dropped", [False, True])
def test_typed_verdict_rejected_even_in_union_drop(verdict, dropped):
    bad = {"questionId": "a", **verdict}
    if dropped:
        bad["fail_closed"] = True
    with pytest.raises(ValueError):
        failclose.account_rows([bad])
    # The invalid right verdict must be checked even when the left is incorrect.
    with pytest.raises(ValueError):
        failclose.compare_rows([row("a", score=0)], [bad])
    if not dropped:
        assert d8_guard._unscorable(bad)


@pytest.mark.parametrize("verdict,correct", [({"correct": False, "score": 1}, False),
                                           ({"score": 1, "label": "incorrect"}, True),
                                           ({"score": 0.5}, False),
                                           ({"label": True}, False),
                                           ({"label": " Passed "}, True)])
def test_valid_verdict_precedence(verdict, correct):
    assert failclose.account_rows([{"questionId": "a", **verdict}])["score_raw"]["correct"] == int(correct)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("failure,success", FAILURES)
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_failure_contradiction_is_baseline_only(tmp_path, source, failure, success, checkpoint, side):
    clean = [row("a_abs"), row("b")]
    candidate = write_run(tmp_path / "c", clean)
    baseline = write_run(tmp_path / "b", clean)
    run = baseline if side == "baseline" else candidate
    (run / "report.json").write_text(json.dumps({"benchmark": "longmemeval", "recallConfig": "OFF",
                                               "evaluations": [{**clean[0], **failure}, clean[1]]}))
    overlay(run, source, [{**clean[0], **success}])
    if checkpoint:
        (run / "checkpoint.json").write_text(json.dumps({"questions": {
            "a_abs": {"phases": {p: {"status": "completed"} for p in ("search", "answer", "evaluate")}}}}))
    result = d8_guard.compare_runs(candidate, baseline, subset=True)
    assert not result["pass"]
    assert result["INCOMPLETE"] is (side == "baseline")
    if side == "baseline":
        assert "baseline failure contradicted by detail: a_abs" in result["reason"]
    else:
        assert result["categories"]["abstention"]["lower_bound_points"] == -100


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("failure,success", [
    ({"status": "failed"}, {"status": "completed"}),
    ({"status": "failed"}, {"status": "completed", "score": 1, "label": "correct"}),
    ({"error": "boom"}, {"error": None}),
    ({"judge_fallback": True}, {"judge_fallback": False}),
])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_probe3x_partial_baseline_rejudge(tmp_path, source, failure, success, checkpoint):
    baseline = write_run(tmp_path / "b", [row("a_abs", **failure), row("b")])
    candidate = write_run(tmp_path / "c", [row("a_abs", score=0), row("b")])
    overlay(baseline, source, [{"question_id": "a_abs", **success}])
    if checkpoint:
        (baseline / "checkpoint.json").write_text(json.dumps({"questions": {
            "a_abs": {"phases": {p: {"status": "completed"} for p in ("search", "answer", "evaluate")}}}}))
    result = d8_guard.compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert "baseline failure contradicted by detail: a_abs" in result["reason"]


@pytest.mark.parametrize("detail", [None, {"score": "malformed"}, {"score": 1, "status": "failed"}])
def test_baseline_failure_without_usable_success_keeps_verdict(tmp_path, detail):
    candidate = write_run(tmp_path / "c", [row("a_abs", score=0), row("b")])
    baseline = write_run(tmp_path / "b", [row("a_abs", status="failed"), row("b")])
    if detail:
        overlay(baseline, SOURCES[0], [{"question_id": "a_abs", **detail}])
    result = d8_guard.compare_runs(candidate, baseline, subset=True)
    assert result["pass"]
    assert not result["INCOMPLETE"]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("kind_key", ["questionType", "category"])
def test_null_category_keeps_full_s4_interval(tmp_path, source, kind_key):
    rows = [row(f"ku{i:03d}") for i in range(78)]
    rows += [row(f"q{i}_abs", kind="single-session-user") for i in range(30)]
    rows += [row(f"q{i}", kind="multi-session") for i in range(392)]
    if kind_key == "category":
        for r in rows:
            r["category"] = r.pop("questionType")
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [{**r, "score": 0 if i in (5, 6, 7) else 1}
                                         for i, r in enumerate(rows)])
    overlay(baseline, source, [{"question_id": f"ku{i:03d}", "score": 1, kind_key: None,
                                "recallHealth": None, "annotation": None} for i in (5, 6, 7)])
    result = d8_guard.compare_runs(candidate, baseline)
    assert not result["pass"]
    assert not result["INCOMPLETE"]
    assert result["categories"]["knowledge-update"]["paired_n"] == 78
    assert result["categories"]["knowledge-update"]["lower_bound_points"] < -2
    loaded = {r["questionId"]: r for r in failclose.load_rows(baseline)}["ku005"]
    assert loaded[kind_key] == "knowledge-update"
    assert loaded["recallHealth"] == rows[5]["recallHealth"]


def test_accounting_main_names_missing_compare_run(tmp_path, capsys):
    candidate = write_run(tmp_path / "c", [row("a")])
    missing = tmp_path / "missing"
    assert failclose.main([str(candidate), "--compare", str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(missing) in captured.err
    assert len(captured.err.splitlines()) == 1
    assert "Traceback" not in captured.err
