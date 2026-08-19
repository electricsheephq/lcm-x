from __future__ import annotations

import json

from bench.tools import deliveryprofile


def _write_results(path, rows):
    path.mkdir()
    (path / "query-A3-500.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_capture_profile_counts_arms_and_distributions(tmp_path):
    run = tmp_path / "run"
    _write_results(
        run,
        [
            {
                "arm": "A3",
                "question_id": "q1",
                "delivered_hits": [
                    {"source": "fts", "content": "alpha", "tokens": 10},
                    {"metadata": {"arm": "semantic"}, "content": "beta", "tokens": 20},
                ],
            },
            {
                "arm": "A3",
                "question_id": "q2",
                "delivered_hits": [
                    {"source": "fts", "content": "gamma", "delivered_chars": 20, "delivered_tokens": 30}
                ],
            },
        ],
    )

    profile = deliveryprofile.capture_profile(run)

    assert profile["total_hits"] == 3
    assert profile["questions"] == 2
    assert profile["arm_hit_counts"] == {"fts": 2, "semantic": 1}
    assert profile["arm_contribution_shares"] == {"fts": 2 / 3, "semantic": 1 / 3}
    assert profile["delivered_chars"]["median"] == 5
    assert profile["delivered_tokens"]["median"] == 20
    assert profile["hits_per_question"]["median"] == 1.5


def test_compare_pass_and_fail_on_arm_death_and_share_drift(tmp_path):
    baseline_run = tmp_path / "baseline-run"
    candidate_run = tmp_path / "candidate-run"
    _write_results(
        baseline_run,
        [
            {"question_id": "q1", "delivered_hits": [{"source": "fts", "content": "a"}, {"source": "semantic", "content": "b"}]},
            {"question_id": "q2", "delivered_hits": [{"source": "fts", "content": "c"}, {"source": "semantic", "content": "d"}]},
        ],
    )
    _write_results(
        candidate_run,
        [
            {"question_id": "q1", "delivered_hits": [{"source": "fts", "content": "a"}, {"source": "semantic", "content": "b"}]},
            {"question_id": "q2", "delivered_hits": [{"source": "fts", "content": "c"}, {"source": "semantic", "content": "d"}]},
        ],
    )
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    deliveryprofile.write_profile(baseline_run, baseline)
    deliveryprofile.write_profile(candidate_run, candidate)

    assert deliveryprofile.compare_profiles(baseline, candidate)["ok"] is True

    candidate_run.joinpath("query-A3-500.jsonl").write_text(
        json.dumps({"question_id": "q1", "delivered_hits": [{"source": "fts", "content": "a"}]}) + "\n"
        + json.dumps({"question_id": "q2", "delivered_hits": [{"source": "fts", "content": "c"}]}) + "\n",
        encoding="utf-8",
    )
    deliveryprofile.write_profile(candidate_run, candidate)
    result = deliveryprofile.compare_profiles(baseline, candidate)

    assert result["ok"] is False
    assert "arm-death:semantic" in result["failures"]
    assert "arm-share:semantic" in result["failures"]


def test_compare_names_share_drift_without_arm_death(tmp_path):
    baseline_run = tmp_path / "baseline-run"
    candidate_run = tmp_path / "candidate-run"
    _write_results(
        baseline_run,
        [{"question_id": "q1", "delivered_hits": [{"source": "fts", "content": "a"}, {"source": "semantic", "content": "b"}]}],
    )
    _write_results(
        candidate_run,
        [{"question_id": "q1", "delivered_hits": [{"source": "fts", "content": "a"}, {"source": "fts", "content": "b"}, {"source": "semantic", "content": "c"}]}],
    )
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    deliveryprofile.write_profile(baseline_run, baseline)
    deliveryprofile.write_profile(candidate_run, candidate)

    result = deliveryprofile.compare_profiles(baseline, candidate)

    assert result["ok"] is False
    assert "arm-share:fts" in result["failures"]
