"""Round 4 synthetic regressions: identity."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.fmp import common, run, score, stores


def test_dirty_identity_and_store_generation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE).decode().strip()
    git("init", "-q")
    (repo / "code.py").write_text("one")
    git("add", "code.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    clean = common.commit(repo)
    (repo / "code.py").write_text("two")
    first = stores.Stores(tmp_path / "stores", "off", repo, "data", None)
    (repo / "code.py").write_text("three")
    second = stores.Stores(tmp_path / "stores", "off", repo, "data", None)
    assert first.identity["lcmx_commit"] != second.identity["lcmx_commit"]
    assert first.generation != second.generation
    tracked_dirty = common.commit(repo)
    (repo / "new.py").write_text("new")
    untracked_dirty = common.commit(repo)
    (repo / "new.py").write_text("changed")
    assert len({clean, tracked_dirty, untracked_dirty, common.commit(repo)}) == 4
    assert tracked_dirty.endswith("-dirty")
    assert common.commit(repo) == common.commit(repo)



def test_dirty_cross_commit_arm_labels(tmp_path):
    from bench.fmp.tests.test_score_commits import arm as commit_arm
    sha = "a" * 40
    paths = [commit_arm(tmp_path, str(i), c, [True])
             for i, c in enumerate((sha, sha + "-111-dirty", sha + "-222-dirty"))]
    result = score.score(paths, tmp_path / "report", pair_across_commits=True)
    assert len(result["arms"]) == 3



def test_smoke_refuses_in_checkout_output_before_writing(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    script = tmp_path / "repo/bench/fmp/run_smoke.sh"
    script.parent.mkdir(parents=True)
    script.write_bytes((repo / "bench/fmp/run_smoke.sh").read_bytes())
    target = tmp_path / "repo/artifacts/r4-test-must-not-be-created"
    env = {**os.environ, "FMP_NICED": "1", "FMP_PYTHON": "/definitely/missing/python",
           "FMP_LME_DATA": "fixture", "FMP_LOCOMO": "fixture"}
    proc = subprocess.run(["bash", str(script), str(target)],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 1 and "outside the checkout" in proc.stderr
    assert not target.exists()



@pytest.mark.parametrize("factor", ["harness_commit", "harness_tree_sha256", "lcmx_commit"])
def test_judge_checks_current_code(tmp_path, monkeypatch, factor):
    q = SimpleNamespace(qid="0")
    args = run.parser().parse_args(["judge", "--split-label", "smoke", "--store-root", str(tmp_path),
                                   "--output", str(tmp_path)])
    manifest = {"dataset_sha256": "d", "id_list_sha256": "i", "split_label": "smoke", "question_ids": ["0"],
                "prompts_sha256": run.prompt_hashes(), **run.switches(args), "harness_commit": common.commit(Path(run.__file__).parents[2]),
                "lcmx_commit": common.commit(args.lcmx_repo),
                "harness_tree_sha256": common.sha(b"".join(p.read_bytes() for p in sorted(Path(run.__file__).parent.glob("*.py"))))}
    manifest[factor] = "different"
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch"):
        next(run.run(args, None, [q], "d", "i", {}, None))
