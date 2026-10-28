"""Round 4 synthetic regressions: score."""
import json

import pytest

from bench.fmp import score
from bench.fmp.tests.test_r3 import MODEL, arm, row


def test_score_rejects_truncated_judging(tmp_path):
    path = arm(tmp_path, "off", [row(0)])
    manifest = json.loads((path / "run-manifest.json").read_text())
    for change in ({"status": "judging", "question_ids": ["0"]},
                   {"status": "complete", "question_ids": ["0", "1"]}):
        (path / "run-manifest.json").write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError, match="judged|question ids"):
            score.score([path], tmp_path / change["status"])



def test_attempted_failures_count_in_accuracy(tmp_path):
    paths = [arm(tmp_path, "off", [row(0), row(1, status="error"),
                row(2, judges={MODEL: {"status": "judge_error", "correct": None}}),
                row(3, status="skipped"), row(4, status="not_run")])]
    result = score.score(paths, tmp_path / "report")
    summary = next(iter(result["arms"].values()))
    assert summary["scored_n"] == 3 and summary["valid_n"] == 1
    assert summary["accuracy"][MODEL] == pytest.approx(1 / 3)



def test_cost_projection_not_doubled(tmp_path):
    path = arm(tmp_path, "off", [row(0, calls=[{"cost": 2}], judge_calls=[{"cost": 1}])])
    result = score.score([path], tmp_path / "report")
    assert result["full_run_estimate"]["total_usd"] == 500 * 3



@pytest.mark.parametrize("factor,value", [("summaries", "deterministic"), ("agreement_sample", ["0"]),
    ("agreement_seed", 2), ("agreement_fraction", .2)])
def test_pairing_binds_experimental_factors(tmp_path, factor, value):
    paths = [arm(tmp_path, c, [row(0)]) for c in ("off", "local")]
    path = paths[1] / "run-manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), factor: value}))
    with pytest.raises(ValueError, match=factor):
        score.score(paths, tmp_path / "report")



def test_score_emits_card_and_preserves_rescore_inputs(tmp_path):
    path = arm(tmp_path, "off", [row(0)])
    output = tmp_path / "report"
    score.score([path], output)
    card = json.loads((output / "RESULT-MANIFEST.json").read_text())
    assert card["schema"] == "result-manifest/1" and card["code"]["sha"]
    assert card["headline"][0]["value"] == 1 and card["headline"][0]["n"] == 1
    assert "Not proven:" in (output / "RESULT.md").read_text()
    assert card["rescore"]["inputs"]
    for entry in card["rescore"]["inputs"]:
        assert (output / entry).exists() and entry in card["keep"]
    assert next((output / "score-inputs").glob("*/0/judged.jsonl")).read_bytes() == (path / "judged.jsonl").read_bytes()
