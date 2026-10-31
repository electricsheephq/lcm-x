"""#1819: trace sandboxes must canonicalize paths before observation."""

import importlib
import json
from pathlib import Path
import tempfile

import pytest


@pytest.mark.parametrize("harness,scenario", [
    ("assembly", "system-note"),
    ("backfill", None),
    ("compress", "externalized-payloads"),
    ("ingest", "sequence-rewrite"),
    ("reconcile", "persisted-full"),
])
def test_trace_regeneration_with_symlinked_tmpdir(tmp_path, harness, scenario):
    module = importlib.import_module(f"tests.behaviour.{harness}_trace_harness")
    case = next(case for case in module.SCENARIOS if scenario is None or case.name == scenario)
    real = (tmp_path / "real").resolve()
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    traces = []
    for parent in (link, real):
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            trace = json.dumps(module.observe(case, Path(directory) / case.name), sort_keys=True)
            assert str(real) not in trace
            assert "/private<sandbox>" not in trace
            traces.append(trace)
    assert traces[0] == traces[1]
