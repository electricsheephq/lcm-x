"""Synthetic regressions for #1729; no real run outputs or model calls."""
import json
from pathlib import Path

import pytest

from bench.tools.d8_guard import _unscorable, compare_runs, main


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


def count_clean(result):
    """Complete, with every category inside its count margin (d8-count-margin-v1).

    One-question subset categories never gate, so `pass` is false here by rule.
    """
    return not result["INCOMPLETE"] and all(c["count_pass"] for c in result["categories"].values())


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
    assert count_clean(result)  # d8-count-margin-v1: was `pass`
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


def test_recall_configs_compare_only_for_placebo_and_stay_consistent_within_run(tmp_path, capsys):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c", recallConfig="LOCAL")

    def local_health(report):
        for row in report["evaluations"]:
            row["recallHealth"]["coverage"].update(summary="full", chunk="full")

    update_report(candidate, local_health)
    result = compare_runs(candidate, baseline, subset=True)
    assert count_clean(result)  # d8-count-margin-v1: was `pass`
    assert not result["INCOMPLETE"]
    assert result["recall_configs"] == {"candidate": "LOCAL", "baseline": "OFF"}
    assert all(run["degraded"] == 0 for run in result["runs"].values())

    assert main([str(candidate), str(baseline), "--subset", "--placebo"]) == 1
    placebo = json.loads(capsys.readouterr().out)
    assert placebo["INCOMPLETE"]
    assert "identity mismatch: recallConfig" in placebo["reason"]

    (candidate / "checkpoint.json").write_text(json.dumps({"recallConfig": "OFF"}))
    for placebo_mode in (False, True):
        mixed = compare_runs(candidate, baseline, subset=True, placebo=placebo_mode)
        assert mixed["INCOMPLETE"]
        assert "candidate: identity mixed within run: recallConfig" in mixed["reason"]


@pytest.mark.parametrize("source", ["report.json", "checkpoint.json", "per_question.jsonl",
                                   "results/per_question.jsonl"])
@pytest.mark.parametrize("role", ["answererProvenance", "judgeProvenance"])
@pytest.mark.parametrize("field,value", [
    ("model", "mixed stored CLI call identities"), ("reasoningEffort", "mixed"),
    ("mixedExecutionIdentity", True), ("executionIdentityCount", 2),
])
@pytest.mark.parametrize("placebo", [False, True])
def test_matching_writer_mixed_identity_is_incomplete(tmp_path, source, role, field, value, placebo):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c")
    for root in (candidate, baseline):
        document = {role: {field: value}}
        if source == "report.json":
            update_report(root, lambda report: report.update(document))
        else:
            path = root / source
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({"questionId": "c", **document}) + "\n")
    result = compare_runs(candidate, baseline, subset=True, placebo=placebo)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    for side in ("candidate", "baseline"):
        assert f"{side}: identity mixed within run: {role}.{field}" in result["reason"]


@pytest.mark.parametrize("placebo", [False, True])
def test_matching_ordinary_writer_identity_passes(tmp_path, placebo):
    metadata = {role: {"model": "reader", "reasoningEffort": "low",
                       "mixedExecutionIdentity": False, "executionIdentityCount": 1}
                for role in ("answererProvenance", "judgeProvenance")}
    baseline = write_run(tmp_path / "b", **metadata)
    candidate = write_run(tmp_path / "c", **metadata)
    # d8-count-margin-v1: was `pass`
    assert count_clean(compare_runs(candidate, baseline, subset=True, placebo=placebo))


@pytest.mark.parametrize("field", ["score", "correct"])
@pytest.mark.parametrize("label", [None, "", "   "])
def test_null_verdict_without_label_is_unscorable(tmp_path, field, label):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c")

    def null_verdict(report):
        row = report["evaluations"][2]
        del row["score"]
        row.update({field: None, "label": label})

    update_report(candidate, null_verdict)
    result = compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert result["runs"]["candidate"]["unscorable"] == 1
    assert "candidate: unscorable rows: 1" in result["reason"]


@pytest.mark.parametrize("field", ["answeringModel", "answererProvenance.reasoningEffort"])
@pytest.mark.parametrize("left,right", [("", " \t"), (" \n", ""), ("", "reader"), ("reader", " ")])
def test_blank_identity_values_are_missing(tmp_path, field, left, right):
    def metadata(value):
        parts = field.split(".")
        return {parts[0]: value if len(parts) == 1 else {parts[1]: value}}

    baseline = write_run(tmp_path / "b", **metadata(left))
    candidate = write_run(tmp_path / "c", **metadata(right))
    result = compare_runs(candidate, baseline, subset=True)
    if not left.strip() and not right.strip():
        assert count_clean(result)  # d8-count-margin-v1: was `pass`
        assert field in result["identities_missing_both"]
    else:
        assert result["INCOMPLETE"]
        assert f"identity missing: {field}" in result["reason"]


@pytest.mark.parametrize("targets", [["c", "c"], [1, "1"]])
def test_checkpoint_target_ids_reject_normalized_duplicates(tmp_path, targets):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c")
    (candidate / "checkpoint.json").write_text(json.dumps({"targetQuestionIds": targets}))
    result = compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"]
    assert (f"candidate: duplicate question ids: 1; first id: {targets[0]} "
            "(checkpoint.json targetQuestionIds)") in result["reason"]


def test_presentation_mode_is_placebo_only_but_consistent_within_run(tmp_path):
    baseline = write_run(tmp_path / "b", answerPresentationMode="raw")
    candidate = write_run(tmp_path / "c", answerPresentationMode="default")
    result = compare_runs(candidate, baseline, subset=True)
    assert count_clean(result)  # d8-count-margin-v1: was `pass`
    assert result["answer_presentation_modes"] == {"candidate": "default", "baseline": "raw"}
    placebo = compare_runs(candidate, baseline, subset=True, placebo=True)
    assert placebo["INCOMPLETE"]
    assert "identity mismatch: answerPresentationMode" in placebo["reason"]

    (candidate / "checkpoint.json").write_text(json.dumps({"answerPresentationMode": "raw"}))
    for placebo_mode in (False, True):
        mixed = compare_runs(candidate, baseline, subset=True, placebo=placebo_mode)
        assert mixed["INCOMPLETE"]
        assert "candidate: identity mixed within run: answerPresentationMode" in mixed["reason"]


@pytest.mark.parametrize("evaluation,scorable", [
    ({"correct": True}, True), ({"correct": False}, True),
    ({"correct": "yes"}, False), ({"correct": 1}, False), ({"correct": None}, False),
    ({"correct": {}, "score": 1}, False), ({"correct": False, "score": "x"}, True),
    ({"score": 0}, True), ({"score": 0.5}, True), ({"score": float("inf")}, False),
    ({"score": 10**400}, True), ({"score": True}, False),
    ({"score": float("nan")}, False), ({"score": "x"}, False),
    ({"score": None, "label": "pass"}, False),
    ({"label": "correct"}, True), ({"label": "incorrect"}, True),
    ({"label": "unknown"}, False), ({"label": True}, True), ({"label": False}, True),
    ({"label": {}}, False), ({"label": []}, False), ({"label": 1}, False),
    ({"label": None}, False), ({"label": " \t"}, False), ({}, False),
])
def test_outcome_types_use_scorer_precedence(tmp_path, evaluation, scorable):
    baseline = write_run(tmp_path / "b")
    candidate = write_run(tmp_path / "c")

    def replace_score(report):
        row = report["evaluations"][2]
        del row["score"]
        row.update(evaluation)

    update_report(candidate, replace_score)
    result = compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"] is (not scorable)
    assert result["runs"]["candidate"].get("unscorable", 0) == int(not scorable)
    assert _unscorable(evaluation) is (not scorable)
    assert not _unscorable({**evaluation, "status": "failed"})


WRITER_FIXTURES = Path(__file__).parent / "fixtures" / "d8_writer_pair"


def test_writer_generated_matching_identities_are_present():
    result = compare_runs(WRITER_FIXTURES / "a", WRITER_FIXTURES / "c", subset=True)
    assert count_clean(result)  # d8-count-margin-v1: was `pass`
    assert not result["INCOMPLETE"]
    for field in ("datasetSha256", "datasetPath", "promptVersion", "scoringPath"):
        assert field not in result["identities_missing_both"]
    for filename in ("checkpoint.json", "report.json"):
        documents = {arm: json.loads((WRITER_FIXTURES / arm / filename).read_text())
                     for arm in ("a", "b", "c")}
        for field in ("questionSetSha256", "readerPromptVersion", "answerPresentationMode"):
            assert documents["a"][field]
            assert documents["a"][field] == documents["c"][field]
        assert documents["a"]["datasetSha256"] != documents["b"]["datasetSha256"]
        varying = {"datasetSha256", "runId", "dataSourceRunId", "createdAt", "updatedAt",
                   "timestamp"}
        assert {key: value for key, value in documents["a"].items() if key not in varying} == {
            key: value for key, value in documents["b"].items() if key not in varying}


def test_writer_generated_dataset_byte_mismatch_is_incomplete():
    result = compare_runs(WRITER_FIXTURES / "a", WRITER_FIXTURES / "b", subset=True)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert "identity mismatch: datasetSha256" in result["reason"]


def copy_writer_run(tmp_path, arm, name):
    root = tmp_path / name
    root.mkdir()
    for filename in ("checkpoint.json", "report.json"):
        (root / filename).write_text((WRITER_FIXTURES / arm / filename).read_text())
    return root


def edit_writer_run(root, field, value):
    for filename in ("checkpoint.json", "report.json"):
        path = root / filename
        document = json.loads(path.read_text())
        if value is None:
            del document[field]
        else:
            document[field] = value
        path.write_text(json.dumps(document))


@pytest.mark.parametrize("field", ["questionSetSha256", "readerPromptVersion"])
def test_writer_identity_mismatch_is_incomplete(tmp_path, field):
    baseline = copy_writer_run(tmp_path, "a", "baseline")
    candidate = copy_writer_run(tmp_path, "c", "candidate")
    assert count_clean(compare_runs(candidate, baseline, subset=True))  # d8-count-margin-v1: was `pass`
    edit_writer_run(candidate, field, "changed-" + field)
    result = compare_runs(candidate, baseline, subset=True)
    if field == "questionSetSha256":
        assert count_clean(result)  # d8-count-margin-v1: was `pass`
        assert result["identity_differences"] == {field: {
            "candidate": "changed-" + field,
            "baseline": json.loads((baseline / "report.json").read_text())[field]}}
        return
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert f"identity mismatch: {field}" in result["reason"]


@pytest.mark.parametrize("field", ["questionSetSha256", "readerPromptVersion"])
def test_writer_identity_missing_on_one_side_is_incomplete(tmp_path, field):
    baseline = copy_writer_run(tmp_path, "a", "baseline")
    candidate = copy_writer_run(tmp_path, "c", "candidate")
    edit_writer_run(baseline, field, None)
    result = compare_runs(candidate, baseline, subset=True)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert f"identity missing: {field}" in result["reason"]
