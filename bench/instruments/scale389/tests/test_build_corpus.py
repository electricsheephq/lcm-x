from __future__ import annotations

import json

from bench.instruments.scale389.build_corpus import build_corpus


def test_build_corpus_persists_has_answer_turns(tmp_path):
    dataset = [
        {
            "answer": "BLUE",
            "answer_session_ids": ["s1"],
            "haystack_dates": ["2026-01-01", "2026-01-02"],
            "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [
                [
                    {
                        "content": "The answer-bearing phrase is BLUE and complete.",
                        "has_answer": True,
                        "role": "assistant",
                    }
                ],
                [{"content": "distractor", "role": "user"}],
            ],
            "question": "What was the phrase?",
            "question_date": "2026-01-03",
            "question_id": "q1",
            "question_type": "single-session-assistant",
        }
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    output = tmp_path / "corpus"

    build_corpus(
        dataset_path,
        output,
        n_eval_questions=1,
        scales=(1, 2),
    )

    qeval = json.loads((output / "qeval.json").read_text())
    assert qeval["questions"][0]["answer_turns"] == [
        {
            "content": "The answer-bearing phrase is BLUE and complete.",
            "date": "2026-01-01",
            "session_id": "s1",
        }
    ]
