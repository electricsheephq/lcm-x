"""Issue #1669: round-five instrument claims, reproduced before implementation."""
from argparse import Namespace as NS
from pathlib import Path


from bench.instruments.local_catchup import audit, measurement

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]
WORKFLOW = Path(__file__).resolve().parents[4]/".github/workflows/lcmx-1586-floor.yml"


def test_item1_full_audit_only_at_covered_setup_boundaries(tmp_path, monkeypatch):
    batches, audits = [], []
    manifest = dict(expected=dict(chunks={str(i): 1 for i in range(96)}, leaves={}))
    def checked(*args):
        audits.append(len(batches))
        return dict(complete=len(batches) >= 6)
    monkeypatch.setattr(measurement, "audit", checked)
    monkeypatch.setattr(measurement, "progress", lambda *args: dict(complete=len(batches) >= 6))
    monkeypatch.setattr(audit, "vector_counts", lambda *args: dict(chunks=min(96, len(batches)*16), leaves=0))
    monkeypatch.setattr(measurement, "safety", lambda *args: None)
    monkeypatch.setattr(measurement.time, "sleep", lambda *_: None)
    command = NS(_embedding_backfill_summary_run=lambda *a, **k: batches.append(1),
                 _chunk_backfill_run=lambda *a, **k: batches.append(1))
    measurement.wait_covered(NS(db=tmp_path/"db", root=tmp_path, max_seconds=3),
                             lambda _: command, NS(_config=None), manifest)
    assert audits == [0, 6]
