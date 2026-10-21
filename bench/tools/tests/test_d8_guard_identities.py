"""Synthetic regressions for #1729; no real run outputs or model calls."""
import json

import pytest

from bench.tools.d8_guard import compare_runs, main


def write_run(root, **metadata):
    root.mkdir()
    rows = [{"questionId": qid, "questionType": kind, "score": 1,
             "recallHealth": {"coverage": {"fts": "ok"}, "degraded": False,
                              "degradedReason": None, "timeout": False}}
            for qid, kind in [("a_abs", "single-session-user"),
                              ("b", "knowledge-update"), ("c", "single-session-user")]]
    report = {"benchmark": "longmemeval", "recallConfig": "OFF",
              "evaluations": rows, **metadata}
    (root / "report.json").write_text(json.dumps(report))
    return root


def update_report(root, change):
    path = root / "report.json"
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))


@pytest.mark.parametrize("field,left,right", [
    ("datasetPath", "questions-v1.json", "questions-v2.json"),
    ("datasetSha256", "a" * 64, "b" * 64),
    ("answeringModel", "reader-a", "reader-b"),
    ("answererProvenance", {"model": "reader", "reasoningEffort": "low"},
     {"model": "reader", "reasoningEffort": "high"}),
    ("judge", "judge-a", "judge-b"),
    ("judgeProvenance", {"model": "judge", "reasoningEffort": "low"},
     {"model": "judge", "reasoningEffort": "high"}),
    ("scoringPath", "score-v1", "score-v2"),
    ("promptVersion", "v1", "v2"),
    ("recallConfig", "OFF", "LOCAL"),
])
@pytest.mark.parametrize("placebo", [False, True])
def test_identity_mismatch_is_incomplete(tmp_path, field, left, right, placebo):
    baseline = write_run(tmp_path / "b", **{field: left})
    candidate = write_run(tmp_path / "c", **{field: right})
    result = compare_runs(candidate, baseline, subset=True, placebo=placebo)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert field in result["reason"]


@pytest.mark.parametrize("side", ["candidate", "baseline"])
def test_identity_missing_on_one_side_is_incomplete(tmp_path, side):
    runs = {name: write_run(tmp_path / name) for name in ("candidate", "baseline")}
    update_report(runs[side], lambda report: report.update(answeringModel="reader"))
    result = compare_runs(runs["candidate"], runs["baseline"], subset=True)
    assert result["INCOMPLETE"]
    assert "identity missing: answeringModel" in result["reason"]


def test_missing_both_is_disclosed_and_matching_identities_pass(tmp_path):
    baseline = write_run(tmp_path / "b", answeringModel="reader")
    candidate = write_run(tmp_path / "c", answeringModel="reader")
    result = compare_runs(candidate, baseline, subset=True)
    assert result["pass"]
    assert "datasetSha256" in result["identities_missing_both"]
    assert "answeringModel" not in result["identities_missing_both"]


@pytest.mark.parametrize("source", ["checkpoint.json", "per_question.jsonl", "results/per_question.jsonl"])
def test_identity_from_checkpoint_or_detail_is_checked(tmp_path, source):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c")
    for root, effort in [(baseline, "low"), (candidate, "high")]:
        path = root / source
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"questionId": "c", "answererProvenance": {
            "model": "reader", "reasoningEffort": effort}}) + "\n")
    result = compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"]
    assert "answererProvenance.reasoningEffort" in result["reason"]


def test_conflicting_identity_within_run_is_incomplete(tmp_path):
    baseline = write_run(tmp_path / "b", answeringModel="reader")
    candidate = write_run(tmp_path / "c", answeringModel="reader")
    (candidate / "checkpoint.json").write_text(json.dumps({"answeringModel": "other"}))
    result = compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"]
    assert "answeringModel" in result["reason"]


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("source", ["report.json", "per_question.jsonl", "results/per_question.jsonl"])
def test_conflicting_duplicate_ids_are_incomplete(tmp_path, side, source):
    runs = {name: write_run(tmp_path / name) for name in ("candidate", "baseline")}
    failed = {"questionId": "c", "score": 0, "status": "failed"}
    passed = {"questionId": "c", "score": 1, "status": "completed"}
    if source == "report.json":
        update_report(runs[side], lambda report: report["evaluations"].extend([failed, passed]))
    else:
        path = runs[side] / source
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(map(json.dumps, [failed, passed])) + "\n")
    result = compare_runs(runs["candidate"], runs["baseline"], subset=True)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert f"{side}: duplicate question ids: {2 if source == 'report.json' else 1}" in result["reason"]
    assert "first id: c" in result["reason"]


@pytest.mark.parametrize("qid", ["a_abs", "b", "c"])
@pytest.mark.parametrize("label", [None, "", "   "])
def test_unscorable_healthy_rows_are_incomplete_in_every_category(tmp_path, qid, label):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c")

    def remove_score(report):
        target = next(row for row in report["evaluations"] if row["questionId"] == qid)
        del target["score"]
        target["label"] = label

    update_report(candidate, remove_score)
    result = compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert result["runs"]["candidate"]["degraded"] == 1
    assert result["runs"]["candidate"]["unscorable"] == 1
    assert "candidate: unscorable rows: 1" in result["reason"]


@pytest.mark.parametrize("evaluation", [{"correct": False}, {"score": 0}, {"label": "incorrect"},
                                       {"status": "failed"}])
def test_scored_incorrect_and_explicit_failures_are_not_unscorable(tmp_path, evaluation):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c")

    def replace_score(report):
        target = report["evaluations"][2]
        del target["score"]
        target.update(evaluation)

    update_report(candidate, replace_score)
    result = compare_runs(candidate, baseline, subset=True)
    assert not result["INCOMPLETE"]
    assert result["runs"]["candidate"].get("unscorable", 0) == 0


def test_cli_identity_failure_keeps_json_shape(tmp_path, capsys):
    baseline = write_run(tmp_path / "b", judge="judge-a")
    candidate = write_run(tmp_path / "c", judge="judge-b")
    assert main([str(candidate), str(baseline), "--subset", "--placebo"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["INCOMPLETE"]
    assert result["rate_pass"] is None
    assert result["d8_row"] is False
    assert "judge" in result["reason"]
