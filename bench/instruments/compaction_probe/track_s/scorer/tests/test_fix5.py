"""S4 summary binds the manifest without altering its transcript digest."""
import hashlib

import score_s as sc
from test_fix4 import checkpoint_run, material  # noqa: F401


def test_g1_s4_records_manifest_digest_in_all_summaries(checkpoint_run, material):  # noqa: F811
    _, _, root = checkpoint_run
    manifest_digest = hashlib.sha256((material / "material.manifest.json").read_bytes()).hexdigest()
    transcript_digest = hashlib.sha256((material / "transcript.jsonl").read_bytes()).hexdigest()
    summaries = list(root.glob("cp-*/summary.json")) + [root / "summary.json"]
    assert len(summaries) == 5
    for path in summaries:
        summary = sc.jload(path)
        assert summary.get("material_sha256") == manifest_digest
        assert summary["material_sha"] == transcript_digest
