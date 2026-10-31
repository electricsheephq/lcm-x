"""Synthetic split-rescore provenance and preservation regressions for #1697."""
import json
import sys

import pytest
from .bot1_support import DECISION, load

sys.path.insert(0, str(DECISION))
rescore = load(DECISION / "rescore_split.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    write(root / "INPUT-MANIFEST.json", dict(entries=[]))
    fact = dict(id="F", value="current", fixture="alpha", **{"class": "name"})
    specs = [dict(id="R", kind="current_request", answer="Audit", row_index=0, row_id="r0")]
    for seed in (1, 2):
        mat = root / f"c1/material/seed-{seed}"
        write(mat / "deep_metadata.json", dict(continuity_probes=specs, checkpoint_answers={"0": [fact]}))
        write(mat / "transcript.jsonl", dict(id="r0", role="user", content="Audit"))
        write(mat / "material.manifest.json", dict(shas={n: rescore.digest(mat / n)
              for n in ("deep_metadata.json", "transcript.jsonl")}))
        run = root / f"c1/runs/LCMX-fleet-rc3/seed-{seed}/d{seed}-r1"
        summary = dict(material_sha256=rescore.digest(mat / "material.manifest.json"), events=[])
        write(run / ("summary.json" if seed == 1 else "partial-summary.json"), summary)
        if seed == 1:
            write(run / "cp-0/summary.json", summary)
            write(run / "cp-0/results.jsonl", dict(probe_id="F", answer="current"))
            write(run / "cp-0/assembled_context.json", dict(system="current", view=[]))
            write(root / "c1/decision/scores/cp-0/LCMX-fleet-rc3.seed-1.d1-r1.json",
                  dict(probes=dict(F=dict(success=True)), metrics=dict(facts_kept=dict(correct=1))))
    write(root / "rca/per-fact.jsonl", dict(checkpoint=0, id="F", reader_success=True, present_exact=True))
    (root / "rca/events.jsonl").write_text("")
    return root


@pytest.mark.parametrize("receipt", ["seed-1/d1-r1/summary.json", "seed-1/d1-r1/cp-0/summary.json",
                                     "seed-2/d2-r1/partial-summary.json"])
@pytest.mark.parametrize("pin", [None, "other-material"])
def test_round2_rescore_binds_all_material_receipts(inputs, tmp_path, receipt, pin):
    assert rescore.rescore(inputs, tmp_path / "valid")["checkpoints"]["0"]["semantic"]["pass_count"] == 1
    path = inputs / "c1/runs/LCMX-fleet-rc3" / receipt
    summary = rescore.load(path)
    summary["material_sha256"] = pin
    write(path, summary)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="material.*mismatch"):
        rescore.rescore(inputs, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["facts-split.jsonl", "events-split.jsonl", "request-ground-truth.json",
                                  "rescore-summary.json", "RESCORE.md"])
def test_round2_rescore_never_overwrites_results(inputs, tmp_path, name):
    output = tmp_path / "out"
    output.mkdir()
    (output / name).write_text("prior-result")
    with pytest.raises(ValueError, match="existing result"):
        rescore.rescore(inputs, output, rule="c1b-terms-v1")
    assert (output / name).read_text() == "prior-result"
    assert list(output.iterdir()) == [output / name]
