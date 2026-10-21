"""Report metadata and failure retention with atomic detail verdicts (#1740)."""
import json

import pytest

from bench.tools import d8_guard, failclose
from bench.tools.tests.test_d8_guard import row, write_run


SOURCES = ["per_question.jsonl", "results/per_question.jsonl"]


def overlay(root, source, rows):
    path = root / source
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("kind_key", ["questionType", "category"])
def test_full_category_interval_keeps_report_metadata(tmp_path, source, kind_key):
    rows = [row(f"ku{i}") for i in range(78)]
    rows += [row(f"q{i}_abs", kind="single-session-user") for i in range(30)]
    rows += [row(f"q{i}", kind="multi-session") for i in range(392)]
    if kind_key == "category":
        for r in rows:
            r["category"] = r.pop("questionType")
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [
        {**r, "score": 0 if i < 3 else 1} for i, r in enumerate(rows)])
    overlay(baseline, source, [{"question_id": f"ku{i}", "score": 1} for i in range(3)])
    result = d8_guard.compare_runs(candidate, baseline)
    assert not result["pass"]
    assert not result["INCOMPLETE"]
    assert result["categories"]["knowledge-update"]["paired_n"] == 78
    assert result["categories"]["knowledge-update"]["lower_bound_points"] < -2
    assert result["categories"]["knowledge-update"]["candidate"]["total"] == 78


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("annotation,failed", [
    ({"status": "completed"}, False), ({"error": "ordinary error"}, True),
    ({"error": "fail-close: missing source"}, True),
])
def test_partial_annotations_keep_report_verdict_and_health(tmp_path, source, annotation, failed):
    rows = [row("a_abs"), row("b")]
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", rows)
    overlay(candidate, source, [{"question_id": "a_abs", **annotation}])
    result = d8_guard.compare_runs(candidate, baseline, subset=True)
    assert not result["INCOMPLETE"]
    assert result["pass"] is (not failed)
    assert result["runs"]["candidate"]["missing"] == 0
    assert result["runs"]["candidate"]["degraded"] == int(failed)
    loaded = failclose.load_rows(candidate)[0]
    assert loaded["score"] == 1
    assert loaded["questionType"] == "knowledge-update"


FAILURES = [
    ({"judge_fallback": True}, {"judge_fallback": False}),
    ({"error": "judge failed"}, {"error": ""}),
    ({"status": "error"}, {"status": "completed"}),
    ({"hypothesis": "fail-close: no source"}, {"hypothesis": "answer"}),
    ({"explanation": "no validated exact source reference"}, {"explanation": "ok"}),
    ({"fail_closed": True}, {"fail_closed": False}),
    ({"_failed": True}, {"_failed": False}),
    ({"label": "fail-close: no source"}, {"label": "correct"}),
]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("failure,success", FAILURES)
@pytest.mark.parametrize("checkpoint", [False, True])
def test_report_failure_survives_detail_success(tmp_path, source, failure, success, checkpoint):
    rows = [row("a_abs"), row("b")]
    baseline = write_run(tmp_path / "b", rows)
    candidate = write_run(tmp_path / "c", [{**rows[0], **failure}, rows[1]])
    overlay(candidate, source, [{**rows[0], **success}])
    if checkpoint:
        (candidate / "checkpoint.json").write_text(json.dumps({"questions": {
            "a_abs": {"phases": {p: {"status": "completed"}
                                  for p in ("search", "answer", "evaluate")}}}}))
    result = d8_guard.compare_runs(candidate, baseline, subset=True)
    assert not result["pass"]
    assert not result["INCOMPLETE"]
    assert result["categories"]["abstention"]["lower_bound_points"] == -100
    loaded = failclose.load_rows(candidate)[0]
    assert d8_guard._failed(loaded)
    if "label" not in failure:
        for key, value in failure.items():
            assert loaded[key] == value
    if failclose.fail_close_signature(failure):
        assert failclose.fail_close_signature(loaded)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("report_verdict,detail_verdict,expected", [
    ({"score": 1}, {"label": "malformed"}, None),
    ({"correct": True}, {"label": "malformed"}, None),
    ({"label": "correct"}, {"score": "malformed"}, None),
    ({"correct": "malformed"}, {"label": "correct"}, True),
    ({"correct": "malformed"}, {"score": 1}, True),
    ({"score": 1}, {"label": "incorrect"}, False),
])
def test_verdict_unit_in_both_consumers(tmp_path, source, report_verdict, detail_verdict, expected):
    original = row("a_abs")
    del original["score"]
    original.update(report_verdict)
    candidate = write_run(tmp_path / "c", [original, row("b")])
    baseline = write_run(tmp_path / "b", [row("a_abs"), row("b")])
    overlay(candidate, source, [{"question_id": "a_abs", **detail_verdict}])
    loaded = failclose.load_rows(candidate)[0]
    assert {k: loaded[k] for k in ("correct", "score", "label") if k in loaded} == detail_verdict
    assert d8_guard._unscorable(loaded) is (expected is None)
    result = d8_guard.compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"] is (expected is None)
    assert result["pass"] is (expected is True)
    if expected is None:
        with pytest.raises(ValueError, match="unrecognised label"):
            failclose.account_run(candidate)
    elif expected is not None:
        assert failclose.account_run(candidate)["score_raw"]["correct"] == 1 + int(expected)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("preferred,later", [
    ("questionId", "question_id"), ("questionId", "qid"), ("questionId", "id"),
    ("question_id", "qid"), ("question_id", "id"), ("qid", "id"),
])
def test_null_alias_in_both_overlay_paths(tmp_path, capsys, source, side, preferred, later):
    candidate = write_run(tmp_path / "c", [row("a_abs"), row("b")])
    baseline = write_run(tmp_path / "b", [row("a_abs"), row("b")])
    run = candidate if side == "candidate" else baseline
    overlay(run, source, [{preferred: None, later: "a_abs", "score": 1}])
    with pytest.raises(ValueError, match="invalid question id"):
        failclose.load_rows(run)
    with pytest.raises(ValueError, match="invalid question id"):
        d8_guard.compare_runs(candidate, baseline, subset=True)
    assert d8_guard.main([str(candidate), str(baseline), "--subset"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["INCOMPLETE"] and not result["pass"]
