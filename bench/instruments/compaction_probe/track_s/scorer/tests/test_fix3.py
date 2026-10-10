"""Amendment 3: late re-asks, status formatting and tolerant JSON, offline only."""
from types import SimpleNamespace

import pytest
import score_s as sc
import fixtures as fx
import test_eval2_harness as harness

TRACK, cp, material, module = harness.TRACK, harness.cp, harness.material, harness.module

ROBUST = ["**STATUS: NOT LIVE**\nx", "`STATUS: NOT LIVE`\nx", "status: not live\nx",
          "Status: Not Live\nx", "STATUS: NOT LIVE.\nx", "STATUS: NOT LIVE. It was cancelled.",
          "STATUS: NOT LIVE - replaced.", "STATUS:NOT LIVE\nx", "STATUS:  NOT LIVE\nx",
          '"STATUS: NOT LIVE"\nx', "# STATUS: NOT LIVE\nx", "**STATUS:** NOT LIVE\nx", "STATUS: NOT  LIVE\nx"]
CONTRACT = [("STATUS: NOT LIVE\nx", True), ("\n\n   STATUS: NOT LIVE  \nx", True),
            ("STATUS: NOT LIVE\r\nx", True), ("STATUS: LIVE\nx", False), ("No; do new-task.", False),
            ("I don't know.", False), ("Explanation first.\nSTATUS: NOT LIVE", False),
            ("STATUS: LIVE\nSTATUS: NOT LIVE", False), ("STATUS: NOT LIVE\nSTATUS: LIVE", True),
            ("", False), ("STATUS: UNKNOWN", False), ("STATUS: NOT LIVE\nStill pending.", True)]


@pytest.mark.parametrize("answer", ROBUST)
def test_thirteen_status_formatting_variants(material, answer):
    p = sc.jlines(material / "lifecycle_probes.jsonl")[0]
    assert sc.stale_status(answer) == "STATUS: NOT LIVE"
    assert sc.lifecycle_correct(p, answer, {}, [])
    assert sc.stale_status(answer.replace("NOT LIVE", "LIVE").replace("not live", "live").replace("Not Live", "Live").replace("NOT  LIVE", "LIVE")) == "STATUS: LIVE"


@pytest.mark.parametrize("answer,expected", CONTRACT)
def test_twelve_status_contract_cases(material, answer, expected):
    p = sc.jlines(material / "lifecycle_probes.jsonl")[0]
    assert sc.lifecycle_correct(p, answer, {}, []) is expected


@pytest.mark.parametrize("runner", ["s2/run_s_lcmx.py", "s4/run_s_codex.py"])
def test_raw_newline_json_reply(monkeypatch, tmp_path, runner):
    m = module(TRACK / runner, monkeypatch, tmp_path)
    raw = '{"P1": "STATUS: NOT LIVE\nCancelled."}'
    got = m.R.parse_json_obj(raw) if runner.startswith("s2") else m.parse_answers([raw])
    assert got == {"P1": "STATUS: NOT LIVE\nCancelled."}


@pytest.mark.parametrize("arm", ["L0", "L1", "L1-H", "L1-noptr", "H", "codex-native"])
def test_every_arm_generates_every_late_corrected_probe(material, monkeypatch, tmp_path, arm):
    m = module(TRACK / ("s4/run_s_codex.py" if arm == "codex-native" else "s2/run_s_lcmx.py"), monkeypatch, tmp_path)
    original = [p for p in sc.jlines(material / "lifecycle_probes.jsonl") if p["kind"] == "corrected_value"]
    chosen = next(c for c in sc.jload(material / "material.manifest.json")["checkpoints"] if c["id"].endswith("-CP340000"))
    batches = m.batches_for(material, chosen)
    late = {p["id"]: p for b in batches for p in b["probes"] if p["id"].endswith("@late")}
    assert set(late) == {p["id"] + "@late" for p in original}
    for p in original:
        assert all(late[p["id"] + "@late"][k] == p[k] for k in ("text", "answer", "expect", "row_index"))
    if arm == "codex-native":
        return  # S4's full synthetic multi-checkpoint fork is exercised in test_eval2_harness.
    from s2lib.fakes import Reader
    run = SimpleNamespace(sdir=material, arm=dict(name=arm, kind="fixture", open=False), receipts_out=[],
        m=SimpleNamespace(tokens=SimpleNamespace(count_tokens=len)), sysmsg=dict(role="system", content="fixture"),
        ntok=len, args=SimpleNamespace(reader="fake", batches=0, run="fake", lane="fake"), system="fixture",
        events=[dict(row_index=3, is_compaction=True)], rows=[], seed="seed-1", run_id="fixture", timing_label="fixture", reader_calls=[])
    rows = m.probe(run, [], Reader(), False, dict(db=None, home=None, dir=tmp_path, row=chosen["row_index"], checkpoint=chosen))
    assert {r["probe_id"] for r in rows if r["probe_id"].endswith("@late")} == set(late)
    assert all(r["probe_kind"] == "corrected_value" and r["actual_compactions"] == 1 for r in rows if r["probe_id"].endswith("@late"))


def test_late_scorer_identity_and_actual_horizon(material, tmp_path):
    run = fx.s2_run(tmp_path)
    summary = sc.jload(run / "summary.json")
    chosen = next(c for c in sc.jload(material / "material.manifest.json")["checkpoints"] if c["id"].endswith("-CP340000"))
    summary.update(checkpoint_id=chosen["id"], checkpoint=11, stop_row_index=11,
                   events=[dict(event=i, row_index=row, is_compaction=comp) for i, (row, comp) in enumerate([(0, True), (1, True), (7, False), (11, True), (12, True)])])
    fx.wj(run / "summary.json", summary)
    original = next(p for p in sc.jlines(material / "lifecycle_probes.jsonl") if p["kind"] == "corrected_value")
    rows = sc.jlines(run / "results.jsonl") + [dict(schema="s2-result-v1", arm="LCMX-a", kind="probe",
        probe_id=original["id"] + "@late", probe_kind="corrected_value", answer=original["answer"], status="OK", batch_id="X-BLIFE")]
    fx.wl(run / "results.jsonl", rows)
    scored = sc.score(material, run, "LCMX-a")
    p = scored["probes"][original["id"] + "@late"]
    assert p["kind"] == "corrected_value" and p["class"] == "CORRECT" and p["actual_compactions"] == 2
    assert not sc.lifecycle_correct(dict(original, id=original["id"] + "@late"), "old-alpha", {}, sc.jload(material / "facts.json"))


def test_late_checkpoint_fallback_without_changing_material(material, tmp_path):
    # Mutate only this synthetic fixture, never the frozen v4 material.
    man = sc.jload(material / "material.manifest.json")
    man["checkpoints"] = [c for c in man["checkpoints"] if not c["id"].endswith("-CP340000")]
    man["checkpoints"] += [dict(id="S1-CP320000", row_index=140, tokens=320000),
                           dict(id="S1-CP360000", row_index=142, tokens=360000)]
    fx.wj(material / "material.manifest.json", man)
    before = (material / "lifecycle_probes.jsonl").read_bytes()
    late = [p for p in cp.due(material).values() if p["id"].endswith("@late")]
    assert late and all(p["checkpoint_row"] == 142 for p in late)
    assert (material / "lifecycle_probes.jsonl").read_bytes() == before
