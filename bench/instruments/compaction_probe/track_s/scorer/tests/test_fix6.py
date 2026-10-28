"""Round-3 source horizons and continuity; fake CLI and synthetic material only."""
import pytest
import score_s as sc
import fixtures as fx
import test_eval2_harness as harness
import test_fix4 as fix4

checkpoint_run, material = fix4.checkpoint_run, fix4.material


@pytest.mark.parametrize("kind", ["current_request", "active_constraint"])
@pytest.mark.parametrize("explicit", [False, True])
def test_item6_first_presentation_covers_event_and_checkpoint(material, tmp_path, kind, explicit):
    man = sc.jload(material / "material.manifest.json")
    item = dict(id=f"X-CONT-{kind}", value="synthetic continuity", row_index=8)
    if explicit:
        item["first_presentation_row_index"] = 2
    man["continuity"] = [item]
    fx.wj(material / "material.manifest.json", man)
    rows = sc.jlines(material / "transcript.jsonl")
    if not explicit:
        rows[2]["content"] = "first synthetic continuity"
    rows[8]["content"] = "again synthetic continuity"
    fx.wl(material / "transcript.jsonl", rows)
    event = fx.s2_event(1, row_index=3)
    event["continuity"] = [dict(id=item["id"], present=False)]
    run = fx.s2_run(tmp_path, events=[event])
    summ = sc.jload(run / "summary.json")
    summ.update(stop_row_index=4, checkpoint_id="S1-CP20000", final_continuity=[dict(id=item["id"], present=True)])
    fx.wj(run / "summary.json", summ)
    scored = sc.score(material, run, "LCMX-a")
    cont = scored["metrics"]["continuity"]
    assert [(g["item"], g["strict"]) for g in cont["grid"]] == [(item["id"], False)]
    assert cont["checkpoint_grid"] == [dict(item=item["id"], strict=True)]
    assert cont["status"].startswith("FAIL(")
    # The scorer must resolve sources without editing the manifest/material.
    assert sc.jload(material / "material.manifest.json")["continuity"] == [item]


def test_item1_runner_records_extra_source_rows_and_scorer_keeps_declared_fact_cutoff(checkpoint_run, material):
    _, _, root = checkpoint_run
    run = root / "cp-S1-CP20000"
    summ = sc.jload(run / "summary.json")
    beyond = summ["beyond_declared"]
    assert beyond == dict(count=1, row_indices=[5], facts=["X-F5"], corrections=[], compactions=[], continuity=[])
    scored = sc.score(material, run, "codex-native")
    assert scored["beyond_declared"] == beyond
    assert scored["checkpoint_row"] == 4 and "X-F5" not in scored["probes"]
    final = sc.jload(root / "cp-S1-CP340000/summary.json")
    assert final["beyond_declared"]["count"] == 0


def test_item1_beyond_declared_classifies_facts_corrections_compactions(monkeypatch, tmp_path):
    m = harness.module(harness.TRACK / "s4/run_s_codex.py", monkeypatch, tmp_path)
    rows = [dict(id=f"R{i}", turn=1, content=f"fixture-{i}", role="user") for i in range(4)]
    facts = [dict(id="corrected", row_index=2, correction_source=dict(row_index=1))]
    survival = [dict(window_number=1, turn_last_row_index=0), dict(window_number=2, turn_last_row_index=3)]
    result = m.beyond_declared(rows, facts, 0, survival)
    assert result == dict(count=3, row_indices=[1, 2, 3], facts=["corrected"],
                          corrections=["corrected"], compactions=[2], continuity=[])


def test_item4_s4_reader_calls_are_checkpoint_local(checkpoint_run):
    _, _, root = checkpoint_run
    calls = []
    for cp in root.glob("cp-*"):
        summ = sc.jload(cp / "summary.json")
        assert len(summ["reader_calls"]) == len(summ["forks"])
        assert [c["batch"] for c in summ["reader_calls"]] == [f["batch"] for f in summ["forks"]]
        calls.append(len(summ["reader_calls"]))
    assert len(calls) == 4 and all(calls)
