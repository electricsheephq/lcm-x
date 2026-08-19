from __future__ import annotations

import json

import pytest

from bench.instruments.scale389.metrics import (
    answer_turn_delivery_metrics,
    answer_turns_from_question,
    emit_probe_question_pin,
    select_questions,
    session_gold_metrics,
)


def _question() -> dict:
    return {
        "answer_turns": [
            {
                "content": "The exact locker passcode is ALBATROSS-441.",
                "date": "2026-01-01",
                "session_id": "gold-1",
            },
            {
                "content": "The backup locker passcode is WREN-992.",
                "date": "2026-01-02",
                "session_id": "gold-2",
            },
        ],
        "gold": ["gold-1", "gold-2"],
        "question_id": "q1",
    }


def test_answer_turn_join_is_content_based_and_non_degenerate():
    question = _question()
    sidecar = {
        "distractor": "2026-01-03",
        "gold-2": "2026-01-02",
        "gold-1": "2026-01-01",
    }
    complete_hits = [
        {
            "content": "context: The exact locker passcode is ALBATROSS-441.",
            "session_id": "gold-1",
        },
        {
            "content": "context: The backup locker passcode is WREN-992.",
            "session_id": "gold-2",
        },
    ]

    complete = answer_turn_delivery_metrics(question, complete_hits, sidecar)
    incomplete = answer_turn_delivery_metrics(question, complete_hits[:1], sidecar)

    assert complete["answer_turn_delivered_complete"] == 1
    assert complete["answer_turn_delivered_found"] == 2
    assert incomplete["answer_turn_delivered_complete"] == 0
    assert incomplete["answer_turn_delivered_found"] == 1


def test_answer_turn_join_refuses_positional_identity():
    with pytest.raises(ValueError, match=r"\.dates\.json sidecar"):
        answer_turn_delivery_metrics(
            _question(),
            [
                {
                    "content": "The exact locker passcode is ALBATROSS-441.",
                    "session_id": "session-at-position-zero",
                }
            ],
            {"gold-1": "2026-01-01"},
        )


def test_answer_turn_fallback_uses_corpus_union_and_fails_on_missing_gold(tmp_path):
    union_path = tmp_path / "corpus" / "union.jsonl"
    union_path.parent.mkdir()
    union_path.write_text(
        json.dumps(
            {
                "date": "2023/02/22 (Wed) 20:54",
                "messages": [
                    {
                        "content": "The corpus-assigned answer is ALBATROSS-441.",
                        "has_answer": True,
                        "role": "assistant",
                    }
                ],
                "sid": "gold-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    question = {
        "gold": ["gold-1"],
        "haystack_dates": ["2023/05/25 (Thu) 15:36"],
        "question_id": "q-f39-shape",
    }
    sidecar = {"gold-1": "2023/02/22 (Wed) 20:54"}
    hits = [
        {
            "content": "The corpus-assigned answer is ALBATROSS-441.",
            "session_id": "gold-1",
        }
    ]

    assert answer_turns_from_question(question, union_path) == [
        {
            "content": "The corpus-assigned answer is ALBATROSS-441.",
            "date": "2023/02/22 (Wed) 20:54",
            "session_id": "gold-1",
        }
    ]
    result = answer_turn_delivery_metrics(question, hits, sidecar, union_path)
    assert result == {
        "answer_turn_delivered_complete": 1,
        "answer_turn_delivered_found": 1,
        "answer_turn_delivered_total": 1,
    }

    union_path.write_text(
        union_path.read_text(encoding="utf-8")
        .replace('"sid": "gold-1"', '"sid": "other-session"'),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match=r"corpus union .*union\.jsonl.*gold-1.*q-f39-shape",
    ):
        answer_turn_delivery_metrics(question, hits, sidecar, union_path)


def test_session_metric_keeps_legacy_alias_with_explicit_label():
    metrics = session_gold_metrics(
        _question(),
        [
            {"content": "x", "session_id": "gold-1"},
            {"content": "y", "session_id": "gold-2"},
        ],
    )

    assert metrics["session_gold_all"] == 1
    assert metrics["all_gold"] == 1


def test_uncensored_population_defaults_to_full_primary_and_is_configurable():
    qeval = {
        "questions": [
            {"question_id": f"q{index:03d}"} for index in range(100)
        ]
    }
    primary = select_questions(qeval, "A3")
    widened = select_questions(qeval, "A3u")
    legacy_size = select_questions(qeval, "A3u", uncensored_n=20)

    assert len(primary) == 50
    assert widened == primary
    assert len(legacy_size) == 20
    assert set(row["question_id"] for row in legacy_size) <= set(
        row["question_id"] for row in primary
    )


def test_probe_question_list_is_emitted_and_pin_verified(tmp_path):
    questions = [{"question_id": "q2"}, {"question_id": "q1"}]

    result = emit_probe_question_pin("A3u", questions, tmp_path)

    assert (tmp_path / "probe-questions-A3u.txt").read_text() == "q2\nq1\n"
    pins = json.loads((tmp_path / "probe-questions-A3u.pins.yaml").read_text())
    assert pins["files"]["probe_questions"]["sha256"] == result["sha256"]
    assert "status: PASS" in (
        tmp_path / "PINS-PROBE-A3u-PRERUN.txt"
    ).read_text()

    with pytest.raises(ValueError, match="drifted probe list"):
        emit_probe_question_pin(
            "A3u", [{"question_id": "different"}], tmp_path
        )
