"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from pathlib import Path


from bench.instruments.local_catchup import floor

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


def test_item4_stale_incomplete_with_complete_final_audit_is_invalid():
    assert floor.rate(dict(state="INCOMPLETE", rows_s=10, audit=dict(complete=True)),
                      7200, 250000)["state"] == "INVALID"
