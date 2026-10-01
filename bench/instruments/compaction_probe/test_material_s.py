"""Offline Track S generator/verifier acceptance, using the kit's script loader style."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_script("gen_material")
verifier = load_script("verify_material")
scorer = load_script("score_probes")


def read(directory, name):
    return json.loads((directory / name).read_text())


@pytest.fixture
def material(tmp_path):
    gen.generate(1, tmp_path)
    return tmp_path


def test_deterministic_full_material_and_smoke(tmp_path):
    first, again, other, smoke = [tmp_path / n for n in ("first", "again", "other", "smoke")]
    gen.generate(1, first)
    gen.generate(1, again)
    gen.generate(2, other)
    gen.generate(1, smoke, smoke=True)
    assert all(p.read_bytes() == (again / p.name).read_bytes() for p in first.iterdir())
    assert (first / "facts.json").read_bytes() != (other / "facts.json").read_bytes()
    full, wiring = verifier.verify(first), verifier.verify(smoke)
    assert full["status"] == wiring["status"] == "PASS"
    assert full["tokens"] >= 2 * 244800 + 20000
    assert 0.45 < full["tool_result_token_share"] < 0.55
    assert full["planned_post_supersession_trigger_spans"] >= 2
    assert 264800 <= full["decision_checkpoint_tokens"] < 300000
    assert read(first, "facts.json") == read(smoke, "facts.json")
    full_rows = (first / "transcript.jsonl").read_text().splitlines()
    smoke_rows = (smoke / "transcript.jsonl").read_text().splitlines()
    end = read(smoke, "continuation.json")["row_index"] + 1
    assert full_rows[:end] == smoke_rows[:end]
    assert len({json.loads(r)["turn"] for r in smoke_rows}) == 10
    assert full["runtime_receipts"].startswith("UNTESTED")


def test_probe_format_scores_without_scorer_changes(material):
    facts, traps = read(material, "facts.json"), read(material, "traps.json")
    results = material / "answers.jsonl"
    results.write_text("".join(json.dumps(dict(probe_id=f["id"], raw_answer=f["answer"] if f["answer"] != "ABSTAIN" else "I don\'t know.")) + "\n" for f in facts + traps))
    scores = scorer.score(results, material / "canaries.json", material / "probes.jsonl")
    assert scores["totals"]["retention"] == 1
    assert scores["three_way_totals"] == {"CORRECT": 60, "ABSTAIN": 5, "HALLUCINATE": 0}
    batches = [json.loads(r) for r in (material / "probe_batches.jsonl").read_text().splitlines()]
    assert [len(b["probes"]) for b in batches] == [10] * 6 + [5]
    assert {p["id"] for b in batches for p in b["probes"]} == {f["id"] for f in facts + traps}


@pytest.mark.parametrize("defect,expected", [("value", "missing value"), ("clip", "clip zone"),
    ("stale", "stale ordering"), ("admission", "admission coverage"), ("system", "system row"),
    ("continuity", "continuity text"), ("target", "unmatchable id"), ("external", "externalization thresholds")])
def test_verifier_rejects_bad_material(material, defect, expected):
    rows = [json.loads(r) for r in (material / "transcript.jsonl").read_text().splitlines()]
    facts, manifest = read(material, "facts.json"), read(material, "material.manifest.json")
    middle = next(f for f in facts if f["placement"] == "middle")
    if defect == "value":
        facts[0]["value"] = "missing-value"
    elif defect == "clip":
        text = rows[middle["row_index"]]["content"]
        rows[middle["row_index"]]["content"] = text[:2000] + text[2000:].replace(middle["value"], "")
        rows[middle["row_index"]]["content"] = middle["value"] + rows[middle["row_index"]]["content"]
        middle["char_offset"] = 0
    elif defect == "stale":
        f = next(f for f in facts if f["stale"])
        f["stale"] = "unrecorded-old-choice"
    elif defect == "admission":
        (material / "admission.manifest.json").write_text("[]")
    elif defect == "system":
        rows[0]["role"] = "user"
    elif defect == "continuity":
        manifest["continuity"][0]["value"] = "absent instruction"
    elif defect == "target":
        manifest["receipt_targets"][0]["row_index"] += 1
        manifest["receipt_targets"][0]["row_id"] = rows[manifest["receipt_targets"][0]["row_index"]]["id"]
    else:
        t = next(t for t in manifest["receipt_targets"] if t["kind"] == "externalization")
        rows[t["row_index"]]["content"] = rows[t["row_index"]]["content"][:12000]
        # Reconcile derived checkpoints after truncation; isolate the threshold gate.
        from itertools import accumulate
        prefix = list(accumulate(gen.token_counter()(r["content"]) for r in rows))
        crossed = next(n for n in prefix if n >= manifest["params"]["min_tokens"])
        manifest["decision_checkpoint"]["row_index"] = next(c["row_index"] for c in manifest["checkpoints"] if prefix[c["row_index"]] >= crossed + 20000)
        manifest["presented_tokens"] = sum(gen.token_counter()(r["content"]) for r in rows[:read(material, "continuation.json")["row_index"] + 1])
    (material / "facts.json").write_text(json.dumps(facts))
    (material / "material.manifest.json").write_text(json.dumps(manifest))
    (material / "transcript.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match=expected):
        verifier.verify(material)


def test_trigger_budget_and_cli_defaults(material, capsys):
    args = gen.build_parser().parse_args(["--seed", "1", "--out-dir", str(material)])
    assert args.placements and args.classes12 and args.min_tokens == 244800 and args.min_events == 2
    with pytest.raises(ValueError, match="trigger spans"):
        verifier.verify(material, min_events=20)
    assert verifier.main([str(material)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
