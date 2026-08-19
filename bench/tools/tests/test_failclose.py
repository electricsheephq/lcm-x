from __future__ import annotations

import json

from bench.tools import failclose


def _write_run(path, evaluations, details=()):
    path.mkdir()
    (path / "report.json").write_text(
        json.dumps({"evaluations": evaluations}), encoding="utf-8"
    )
    if details:
        (path / "per_question.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in details),
            encoding="utf-8",
        )


def test_account_run_reports_raw_adjusted_and_signature(tmp_path, capsys):
    run = tmp_path / "run"
    _write_run(
        run,
        [
            {"questionId": "q1", "score": 1},
            {"questionId": "q2", "score": 0},
            {"questionId": "q3", "score": 0},
        ],
        [
            {
                "question_id": "q3",
                "error": "evidence-card item has no validated exact source reference",
            }
        ],
    )

    result = failclose.account_run(run)

    assert result["score_raw"] == {"correct": 1, "total": 3, "rate": 1 / 3}
    assert result["score_adjusted"] == {"correct": 1, "total": 2, "rate": 0.5}
    assert result["fail_closed_n"] == 1
    assert result["fail_closed_qids"] == ["q3"]
    assert result["per_row_signature"] == {
        "q1": None,
        "q2": None,
        "q3": "evidence-card item has no validated exact source reference",
    }
    assert failclose.main([str(run)]) == 0
    assert json.loads(capsys.readouterr().out) == result


def test_paired_comparison_union_drops_either_arms_fail_close():
    left = [
        {"qid": "q1", "correct": True},
        {"qid": "q2", "correct": False, "fail_closed": True},
        {"qid": "q3", "correct": False},
    ]
    right = [
        {"qid": "q1", "correct": False},
        {"qid": "q2", "correct": True},
        {"qid": "q3", "correct": False, "status": "fail-close"},
    ]

    result = failclose.compare_rows(left, right)

    assert result == {
        "drop_convention": "union-drop",
        "dropped_qids": ["q2", "q3"],
        "paired_n": 1,
        "left_score": {"correct": 1, "total": 1, "rate": 1.0},
        "right_score": {"correct": 0, "total": 1, "rate": 0.0},
        "left_only_correct": 1,
        "right_only_correct": 0,
    }
