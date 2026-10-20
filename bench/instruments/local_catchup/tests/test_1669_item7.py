"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from pathlib import Path



pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


def test_item7_workflow_records_and_enforces_runner_temp_headroom():
    import yaml
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["floor"]["steps"]
    measured = next(i for i, s in enumerate(steps) if s.get("name", "").startswith("Enforce 4 GiB"))
    setup = "\n".join(s.get("run", "") for s in steps[:measured])
    assert 'df -' in setup and '"$RUNNER_TEMP"' in setup
    assert "26" in setup and "::error::" in setup and "rm -rf" in setup
    assert '--scratch "$RUNNER_TEMP/floor-scratch"' in steps[measured]["run"]
