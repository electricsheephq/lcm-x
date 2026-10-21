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
    row = report["evaluations"][2]
    del row["score"]
    row["label"] = label
    # Detail overlays must not inherit a report score.
    report_path.write_text(json.dumps(report))
    if source != "report.json":
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
