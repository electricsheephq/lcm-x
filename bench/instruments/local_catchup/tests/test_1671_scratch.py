"""Real open-file evidence through the measured worker's safety path."""
from argparse import Namespace as NS
from contextlib import ExitStack
import hashlib

import pytest

from bench.instruments.local_catchup import common, measurement


@pytest.mark.parametrize("runner_home", [".", "runner-home"])
def test_safety_reports_real_sibling_and_baseline_handles(tmp_path, runner_home):
    scratch = tmp_path/"scratch"
    a = NS(root=tmp_path/"evidence", home=scratch/"0-OFF-home", db=scratch/"cell.db",
           model_cache=tmp_path/"cache", user_home=tmp_path/runner_home,
           lcmx_repo=tmp_path/"repo", hermes_source=tmp_path/"host", scratch=scratch,
           _next_progress=float("inf"))
    own = [a.home/"state.db", a.db, a.root/"output.json", a.model_cache/"model.bin"]
    foreign = [scratch/"0-ON-catching-up-home"/"state.db", scratch/"baseline.db"]
    for path in own + foreign:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    with ExitStack() as handles:
        for path in own:
            handles.enter_context(path.open("rb"))
        assert measurement.safety(a)["violations"] == 0
        for path in foreign:
            handles.enter_context(path.open("rb"))
        with pytest.raises(RuntimeError, match="cross-profile access"):
            measurement.safety(a)
    evidence = common.load(a.root/"file-access.jsonl")
    assert set(evidence["unexpected_path_hashes"]) == {
        hashlib.sha256(str(path.resolve()).encode()).hexdigest() for path in foreign}
