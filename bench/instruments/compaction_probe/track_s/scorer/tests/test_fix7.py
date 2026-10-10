"""Fix 7 continuity sources: synthetic transcript and mocked Codex writer only."""
from types import SimpleNamespace

import pytest
import fixtures as fx
import score_s as sc
import test_eval2_harness as harness
import test_fix4 as fix4

checkpoint_run, material = fix4.checkpoint_run, fix4.material


@pytest.mark.parametrize("kind", ["current_request", "active_constraint", "host_instruction"])
@pytest.mark.parametrize("source", ["explicit", "earliest", "fallback"])
def test_k1_beyond_declared_uses_first_continuity_presentation(monkeypatch, tmp_path, kind, source):
    m = harness.module(harness.TRACK / "s4/run_s_codex.py", monkeypatch, tmp_path)
    rows = [dict(content=text) for text in ("earlier item", "plain", "new item", "earlier item repeated")]
    item = dict(id=kind, value="new item", row_index=0)
    if source == "explicit":
        item["first_presentation_row_index"] = 2
        item["value"] = "earlier item"  # explicit source overrides transcript search
    elif source == "fallback":
        item.update(value="not present", row_index=2)
    repeated = dict(id="repeated", value="earlier item", row_index=3)
    result = m.beyond_declared(rows, [], 1, continuity=[item, repeated])
    assert result == dict(count=2, row_indices=[2, 3], facts=[], corrections=[], compactions=[], continuity=[kind])


def test_k1_writer_records_manifest_continuity_in_summary_and_readmit(checkpoint_run, material):
    m, _, root = checkpoint_run
    man = sc.jload(material / "material.manifest.json")
    item = dict(id="X-CONT-current_request", value="continuity-only fixture", row_index=9)
    man["continuity"] = [item]
    fx.wj(material / "material.manifest.json", man)
    rows = sc.jlines(material / "transcript.jsonl")
    rows[5]["content"] = item["value"]
    rows[9]["content"] = item["value"]
    fx.wl(material / "transcript.jsonl", rows)
    args = SimpleNamespace(checkpoints="lifecycle", stop_row=None, seed="1", run="continuity", auth_file=root / "fake-auth.json",
                           slice=0, dictation="user", dry_run=False, readmit=False, force_event=False, batches=0)
    assert m.main(args) == 0
    output = root.parent / "continuity"
    def check():
        early = sc.jload(output / "cp-S1-CP20000/summary.json")
        repeated = sc.jload(output / "cp-S1-CP40000/summary.json")
        assert early["beyond_declared"]["continuity"] == [item["id"]]
        assert repeated["beyond_declared"]["continuity"] == []
        assert sc.score(material, output / "cp-S1-CP20000", "codex-native")["beyond_declared"] == early["beyond_declared"]
    check()
    args.readmit = True
    for cp in harness.cp.select(material, "lifecycle"):
        args.stop_row = cp["id"]
        assert m.main(args, dict(row=-1)) == 0
    check()
