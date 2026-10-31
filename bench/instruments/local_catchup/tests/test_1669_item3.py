"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from argparse import Namespace as NS
from pathlib import Path

import pytest

from bench.instruments.local_catchup import common, measurement

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


@pytest.mark.parametrize("relative", ["sibling-home/state.db", "baseline.db"])
@pytest.mark.parametrize("runner_home", [".", "runner-home"])
def test_item3_sibling_and_baseline_handles_fail_safety(tmp_path, monkeypatch, relative, runner_home):
    import psutil
    scratch = tmp_path/"scratch"
    foreign = scratch/relative
    monkeypatch.setattr(psutil.Process, "open_files", lambda _: [NS(path=str(foreign))])
    a = NS(root=tmp_path/"evidence", home=scratch/"own-home", db=scratch/"cell.db",
           model_cache=tmp_path/"cache", user_home=tmp_path/runner_home, lcmx_repo=tmp_path/"repo",
           hermes_source=tmp_path/"host", scratch=scratch, _next_progress=float("inf"))
    a.root.mkdir()
    with pytest.raises(RuntimeError, match="cross-profile access"):
        measurement.safety(a)
    assert common.load(a.root/"file-access.jsonl")["unexpected_path_hashes"]
