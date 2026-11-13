"""Malformed-evidence regressions for #1740; synthetic runs only."""
import json

import pytest

from bench.tools.d8_guard import main
from bench.tools.failclose import _correct, _qid


def write_run(root):
    root.mkdir()
    rows = [{"questionId": qid, "questionType": kind, "score": 1,
             "recallHealth": {"coverage": {"fts": "ok"}, "degraded": False,
                              "degradedReason": None, "timeout": False}}
            for qid, kind in [("a_abs", "single-session-user"),
                              ("b", "knowledge-update"), ("c", "single-session-user")]]
    (root / "report.json").write_text(json.dumps({
        "benchmark": "longmemeval", "recallConfig": "OFF", "evaluations": rows}))
    return root


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("source", ["report.json", "per_question.jsonl",
                                   "results/per_question.jsonl", "checkpoint.json"])
@pytest.mark.parametrize("shape,value", [
    ("missing", None), ("empty", ""), ("whitespace", " \t"),
    ("dict", {"id": "c"}), ("list", ["c"]), ("null", None),
    ("bool", True), ("float", 1.5),
])
def test_invalid_question_id_is_incomplete(tmp_path, capsys, side, source, shape, value):
    runs = {name: write_run(tmp_path / name) for name in ("candidate", "baseline")}
    path = runs[side] / source
    if source == "checkpoint.json":
        # A null target represents an id missing from a target slot.
        path.write_text(json.dumps({"targetQuestionIds": [value]}))
    else:
        report_path = runs[side] / "report.json"
        report = json.loads(report_path.read_text())
        row = report["evaluations"][2]
        if shape == "missing":
            del row["questionId"]
        else:
            row["questionId"] = value
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report if source == "report.json" else row) + "\n")
    assert main([str(runs["candidate"]), str(runs["baseline"]), "--subset"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert "invalid question id" in result["error"]
    assert source.split("/")[-1] in result["error"]


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("source", ["report.json", "per_question.jsonl",
                                   "results/per_question.jsonl"])
@pytest.mark.parametrize("label", ["pending", "error", "unknown"])
def test_unrecognised_label_is_unscorable(tmp_path, capsys, side, source, label):
    runs = {name: write_run(tmp_path / name) for name in ("candidate", "baseline")}
    report_path = runs[side] / "report.json"
    report = json.loads(report_path.read_text())
    row = dict(report["evaluations"][2])
    del row["score"]
    row["label"] = label
    # Detail overlays must not inherit a report score.
    if source == "report.json":
        report["evaluations"][2] = row
        report_path.write_text(json.dumps(report))
    else:
        path = runs[side] / source
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(row) + "\n")
    assert main([str(runs["candidate"]), str(runs["baseline"]), "--subset"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["INCOMPLETE"]
    assert result["runs"][side]["unscorable"] == 1
    assert f"{side}: unscorable rows: 1" in result["reason"]
    with pytest.raises(ValueError, match="unrecognised label"):
        _correct(row)


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("source", ["per_question.jsonl", "results/per_question.jsonl"])
def test_valid_overlay_replaces_malformed_report_verdict(tmp_path, capsys, side, source):
    runs = {name: write_run(tmp_path / name) for name in ("candidate", "baseline")}
    report_path = runs[side] / "report.json"
    report = json.loads(report_path.read_text())
    row = report["evaluations"][2]
    overlay = dict(row)
    del overlay["score"]
    overlay["label"] = "correct"
    row["correct"] = "malformed"
    report_path.write_text(json.dumps(report))
    path = runs[side] / source
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(overlay) + "\n")
    # The selected detail row replaces the report verdict, including invalid fields.
    # d8-count-margin-v1: exit 1 and `pass` false, as one-question subset categories never gate.
    assert main([str(runs["candidate"]), str(runs["baseline"]), "--subset"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert not result["INCOMPLETE"]
    assert all(category["count_pass"] for category in result["categories"].values())
    assert result["runs"][side].get("unscorable", 0) == 0


@pytest.mark.parametrize("preferred,later", [
    ("questionId", "question_id"), ("questionId", "qid"), ("questionId", "id"),
    ("question_id", "qid"), ("question_id", "id"), ("qid", "id"),
])
@pytest.mark.parametrize("value", [None, ""])
@pytest.mark.parametrize("side", ["candidate", "baseline"])
def test_invalid_preferred_id_does_not_fall_back(tmp_path, capsys, preferred, later, value, side):
    aliases = {preferred: value, later: "c"}
    with pytest.raises(ValueError, match="invalid question id"):
        _qid(aliases)
    runs = {name: write_run(tmp_path / name) for name in ("candidate", "baseline")}
    report_path = runs[side] / "report.json"
    report = json.loads(report_path.read_text())
    row = report["evaluations"][2]
    del row["questionId"]
    row.update(aliases)
    report_path.write_text(json.dumps(report))
    assert main([str(runs["candidate"]), str(runs["baseline"]), "--subset"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["INCOMPLETE"]
    assert not result["pass"]
    assert "invalid question id" in result["error"]


@pytest.mark.parametrize("value", [0, 1, -1, "c", "  c  "])
@pytest.mark.parametrize("key", ["questionId", "question_id", "qid", "id"])
def test_usable_scalar_ids_keep_their_canonical_value(key, value):
    assert _qid({key: value}) == str(value)


@pytest.mark.parametrize("label,expected", [
    ("correct", True), ("pass", True), ("passed", True), ("incorrect", False),
    (" CORRECT ", True), (" Incorrect ", False),
])
def test_recognised_labels(label, expected):
    assert _correct({"questionId": "c", "label": label}) is expected
