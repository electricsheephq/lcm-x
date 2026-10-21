"""Cross-commit scoring regressions using synthetic judged rows only."""
import json

import pytest

from bench.fmp import models, run, score

COMMITS = ("a" * 40, "b" * 40)
MODEL = models.CODEX_DEFAULT_JUDGE


def arm(root, name, commit, verdicts, **overrides):
    path = root / name
    path.mkdir()
    manifest = {"config": "off", "delivery": "single", "reader": models.CODEX_DEFAULT_READER,
                "lcmx_commit": commit, "harness_commit": "synthetic-harness", "split_label": "smoke",
                "dataset_sha256": "synthetic-data", "id_list_sha256": "synthetic-ids",
                "judges": [MODEL], "agreement_judge": None, "status": "complete",
                "question_ids": [str(i) for i in range(len(verdicts))], **overrides}
    (path / "run-manifest.json").write_text(json.dumps(manifest))
    rows = [{"qid": str(i), "qtype": "synthetic", "status": "ok", "question": "Synthetic?",
             "gold": "x", "hypothesis": "x", "judges": {MODEL: {"status": "ok", "correct": value}}}
            for i, value in enumerate(verdicts)]
    (path / "judged.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return path


def test_pair_across_commits_statistics(tmp_path):
    runs = [arm(tmp_path, "a", COMMITS[0], [True, False, True, False]),
            arm(tmp_path, "b", COMMITS[1], [False, True, True, True])]
    # The base checks duplicate arm names before commit identity; preserve that order.
    with pytest.raises(ValueError, match="duplicate arm"):
        score.score(runs, tmp_path / "default-report")
    result = score.score(runs, tmp_path / "report", pair_across_commits=True)
    names = [f"off/single/{models.CODEX_DEFAULT_READER}@{commit[:12]}" for commit in COMMITS]
    assert list(result["arms"]) == names
    assert list(result["paired"]) == [" vs ".join(names)]
    pair = next(iter(result["paired"].values()))
    assert pair["per_judge"][MODEL] == {"n01": 2, "n10": 1, "p": 1.0, "n": 4}
    assert pair["paired_n"] == 4 and pair["dropped_for_errors"] == 0
    assert result["paired_across"] == "lcmx_commit"
    assert result["lcmx_commits"] == sorted(COMMITS)
    assert json.loads((tmp_path / "report/report.json").read_text()) == result


def test_pair_across_commits_requires_same_harness(tmp_path):
    runs = [arm(tmp_path, "a", COMMITS[0], [True]),
            arm(tmp_path, "b", COMMITS[1], [True], harness_commit="other-harness")]
    with pytest.raises(ValueError, match="paired runs require identical harness_commit"):
        score.score(runs, tmp_path / "report", pair_across_commits=True)


def test_pair_across_commits_requires_distinct_commits(tmp_path):
    runs = [arm(tmp_path, "a", COMMITS[0], [True]),
            arm(tmp_path, "b", COMMITS[0], [True], delivery="agentic")]
    with pytest.raises(ValueError, match="at least 2 distinct lcmx_commit"):
        score.score(runs, tmp_path / "report", pair_across_commits=True)


def test_pair_across_commits_compares_each_delivery_separately(tmp_path):
    runs = [arm(tmp_path, f"{i}-{delivery}", commit, [True], delivery=delivery)
            for i, commit in enumerate(COMMITS) for delivery in ("single", "agentic")]
    result = score.score(runs, tmp_path / "report", pair_across_commits=True)
    expected = {" vs ".join(f"off/{delivery}/{models.CODEX_DEFAULT_READER}@{commit[:12]}" for commit in COMMITS)
                for delivery in ("single", "agentic")}
    assert len(result["arms"]) == 4 and set(result["paired"]) == expected


def test_pair_across_commits_still_rejects_duplicate_arms(tmp_path):
    runs = [arm(tmp_path, "a", COMMITS[0], [True]), arm(tmp_path, "b", COMMITS[1], [True]),
            arm(tmp_path, "duplicate", COMMITS[0], [True])]
    with pytest.raises(ValueError, match="duplicate arm"):
        score.score(runs, tmp_path / "report", pair_across_commits=True)


def test_pair_across_commits_cli(tmp_path):
    runs = [arm(tmp_path, "a", COMMITS[0], [True]), arm(tmp_path, "b", COMMITS[1], [False])]
    argv = ["score", "--split-label", "smoke", "--store-root", str(tmp_path / "stores"),
            "--output", str(tmp_path / "report"), "--runs", *map(str, runs)]
    assert run.parser().parse_args(argv).pair_across_commits is False
    run.execute(run.parser().parse_args([*argv, "--pair-across-commits"]))
    result = json.loads((tmp_path / "report/report.json").read_text())
    assert len(result["arms"]) == 2 and len(result["paired"]) == 1
