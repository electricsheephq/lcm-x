"""Offline Track S generator/verifier acceptance, using the kit's script loader style."""
from __future__ import annotations

import importlib.util
import json
import sys
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
    gen.generate(1, tmp_path, placements=True, classes12=True)
    return tmp_path


def test_deterministic_full_material_and_smoke(tmp_path):
    first, again, other, smoke = [tmp_path / n for n in ("first", "again", "other", "smoke")]
    gen.generate(1, first, placements=True, classes12=True)
    gen.generate(1, again, placements=True, classes12=True)
    gen.generate(2, other, placements=True, classes12=True)
    gen.generate(1, smoke, placements=True, classes12=True, smoke=True)
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
    assert not args.placements and not args.classes12
    assert args.min_tokens == 244800 and args.min_events == 2
    with pytest.raises(ValueError, match="trigger spans"):
        verifier.verify(material, min_events=20)
    assert verifier.main([str(material)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_legacy_api_and_cli_are_default(tmp_path):
    api, cli, explicit = [tmp_path / n for n in ("api", "cli", "explicit")]
    gen.generate(1, api, 35, 100)
    gen.main(["--seed", "1", "--out-dir", str(cli), "--tokens-per-turn", "100"])
    gen._generate_legacy(1, explicit, 35, 100)
    for name in ("turns.jsonl", "canaries.json", "probes.jsonl", "material.manifest.json"):
        assert (api / name).read_bytes() == (cli / name).read_bytes() == (explicit / name).read_bytes()


def test_prefix_identifier_echo_never_scores_correct(material):
    facts = [f for f in read(material, "facts.json") if f["class"] == "prefix"]
    assert len(facts) == 5
    echoes = [f["probe"].split("fixture ", 1)[1].rstrip("?") for f in facts]
    assert all(scorer.normalize(f["answer"]) not in scorer.normalize(f["probe"] + f["id"])
               for f in facts)
    results = material / "echoes.jsonl"
    results.write_text("".join(json.dumps(dict(probe_id=f["id"], raw_answer=echo)) + "\n"
                               for f, echo in zip(facts, echoes)))
    scores = scorer.score(results, material / "canaries.json", material / "probes.jsonl")
    assert scores["three_way_totals"]["CORRECT"] == 0


def test_material_version_identifies_new_generator(material):
    assert read(material, "material.manifest.json")["material_version"] == "track-s-v2"


def test_ancestry_required_events_follow_plan(tmp_path):
    manifest = gen.generate(1, tmp_path, placements=True, classes12=True, min_events=3)
    targets = [t for t in manifest["receipt_targets"] if t["kind"] == "ancestry"]
    assert len(targets) == 5 and all(t["required_events"] == 3 for t in targets)
    assert verifier.verify(tmp_path)["status"] == "PASS"


def test_verifier_rejects_ancestry_event_mismatch(material):
    manifest = read(material, "material.manifest.json")
    manifest["receipt_targets"][0]["required_events"] = 3
    gen._json_write(material / "material.manifest.json", manifest)
    with pytest.raises(ValueError, match="ancestry event count mismatch"):
        verifier.verify(material)


def test_hashes_include_only_generated_files(tmp_path):
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("unrelated fixture")
    manifest = gen.generate(1, tmp_path, placements=True, classes12=True, smoke=True)
    assert set(manifest["shas"]) == {
        "facts.json", "canaries.json", "traps.json", "continuation.json", "admission.manifest.json",
        "transcript.jsonl", "turns.jsonl", "probes.jsonl", "probe_batches.jsonl",
    }
    unrelated.write_text("changed unrelated fixture")
    assert verifier.verify(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize("name", ["canaries.json", "probes.jsonl", "probe_batches.jsonl"])
def test_verifier_rejects_divergent_scoring_artifact_with_updated_digest(material, name):
    if name == "canaries.json":
        payload = read(material, name)
        payload[0]["value"] = payload[0]["answer"] = "divergent-answer"
        gen._json_write(material / name, payload)
        expected = "canary answer key mismatch"
    else:
        payload = [json.loads(line) for line in (material / name).read_text().splitlines()]
        probe = payload[0] if name == "probes.jsonl" else payload[0]["probes"][0]
        probe["text"] = "divergent question"
        (material / name).write_text("".join(json.dumps(row) + "\n" for row in payload))
        expected = "probe facts/traps mismatch" if name == "probes.jsonl" else "probe batch mismatch"
    manifest = read(material, "material.manifest.json")
    manifest["shas"][name] = gen._sha256(material / name)
    gen._json_write(material / "material.manifest.json", manifest)
    with pytest.raises(ValueError, match=expected):
        verifier.verify(material)


def test_turn_projection_preserves_replay_metadata(material):
    rows = [json.loads(line) for line in (material / "transcript.jsonl").read_text().splitlines()]
    turns = [json.loads(line) for line in (material / "turns.jsonl").read_text().splitlines()]
    assert turns == [dict(turn=r["turn"], text=r["content"], id=r["id"], role=r["role"],
                          tool_call_id=r["tool_call_id"], tool_calls=r.get("tool_calls", [])) for r in rows]


@pytest.mark.parametrize("driver_name", ["drive_codex", "drive_hermes", "drive_hermes_acp", "drive_hermes_oneshot"])
@pytest.mark.parametrize("role", ["system", "assistant", "tool"])
def test_prompt_drivers_reject_non_user_rows_before_session(tmp_path, capsys, monkeypatch, driver_name, role):
    driver = load_script(driver_name)
    def forbidden_session(*args, **kwargs):
        pytest.fail("non-user material reached session startup")
    if driver_name == "drive_hermes":
        monkeypatch.setattr(driver, "PtySession", forbidden_session)
    elif driver_name == "drive_hermes_acp":
        monkeypatch.setattr(driver, "StdioJsonRpc", forbidden_session)
    else:
        monkeypatch.setattr(driver.subprocess, "run", forbidden_session)
    material, probes = tmp_path / "turns.jsonl", tmp_path / "probes.jsonl"
    material.write_text(json.dumps(dict(turn=1, text="fixture", role=role, id="R1")) + "\n")
    probes.write_text(json.dumps(dict(id="P1", text="question")) + "\n")
    argv = ["--material", str(material), "--probes", str(probes)]
    if driver_name == "drive_codex":
        argv += ["--codex-bin", sys.executable, "--model", "fixture-model", "--out-dir", str(tmp_path / "run")]
        args = driver.build_parser().parse_args(argv)
        with pytest.raises(ValueError, match="non-user row requires role-faithful replay"):
            driver.drive(args)
    else:
        home = tmp_path / "home"
        home.mkdir()
        (home / "config.yaml").write_text("context:\n  engine: lcm\nmodel:\n  default: fixture-model\n")
        argv += ["--hermes-home", str(home), "--log", str(tmp_path / "raw.log"),
                 "--expect-engine", "lcm", "--expect-model", "fixture-model"]
        with pytest.raises(SystemExit) as exc:
            driver.main(argv)
        assert exc.value.code == 2
        assert "non-user row requires role-faithful replay" in capsys.readouterr().err
    assert not (tmp_path / "run.manifest.json").exists()
    assert not (tmp_path / "run").exists()
