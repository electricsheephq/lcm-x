"""Offline Track S generator/verifier acceptance, using the kit's script loader style."""
from __future__ import annotations

import importlib.util
import json
import re
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
    assert read(material, "material.manifest.json")["material_version"] == "track-s-v3"


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


def rewrite_material(directory, name, payload):
    """Reconcile the digest so a negative test exercises the semantic gate."""
    if name.endswith(".jsonl"):
        (directory / name).write_text("".join(json.dumps(row) + "\n" for row in payload))
    else:
        gen._json_write(directory / name, payload)
    manifest = read(directory, "material.manifest.json")
    manifest["shas"] = {n: gen._sha256(directory / n) for n in manifest["shas"]}
    gen._json_write(directory / "material.manifest.json", manifest)


def transcript_rows(directory):
    return [json.loads(line) for line in (directory / "transcript.jsonl").read_text().splitlines()]


def rewrite_transcript(directory, rows):
    rewrite_material(directory, "transcript.jsonl", rows)
    rewrite_material(directory, "turns.jsonl", [dict(
        turn=r["turn"], text=r["content"], role=r["role"], id=r["id"],
        tool_call_id=r["tool_call_id"], tool_calls=r.get("tool_calls", [])) for r in rows])


@pytest.mark.parametrize("seed", [1, 2, 3, 17])
def test_mb1_traps_match_canary_wording_without_transcript_leaks(tmp_path, seed):
    gen.generate(seed, tmp_path, placements=True, classes12=True, smoke=True)
    facts, traps = read(tmp_path, "facts.json"), read(tmp_path, "traps.json")
    classes, names = set(), set()
    rows = transcript_rows(tmp_path)
    for trap in traps:
        match = re.fullmatch(r"What is the current (.+) for fixture ([a-z]+)-(-?\d+)-(\d{2})-([5-9])\?", trap["probe"])
        assert match is not None
        cls, word, fixture_seed, class_index, _ = match.groups()
        assert cls.replace(" ", "_") in gen.CLASSES
        assert word in gen.VALUE_WORDS and int(fixture_seed) == seed
        assert 0 <= int(class_index) < len(gen.CLASSES)
        name = trap["probe"].split("for fixture ", 1)[1][:-1]
        classes.add(cls)
        names.add(name)
        assert trap["answer"] == "ABSTAIN"
        assert all(name not in f["probe"] for f in facts)
        assert all(name not in r["content"] and trap["probe"] not in r["content"] for r in rows)
    assert len(classes) == len(names) == 5


def test_mb1r_trap_ids_are_fact_shaped(material):
    for trap in read(material, "traps.json"):
        assert re.fullmatch(r"S1-F\d{2}-[5-9]", trap["id"])
        assert "TRAP" not in trap["id"] + trap["probe"]


def test_mb1r_trap_fixture_digits_match_class_and_id(material):
    for trap in read(material, "traps.json"):
        cls, name = trap["probe"].removeprefix("What is the current ").split(" for fixture ")
        c, k = name.removesuffix("?").rsplit("-", 2)[1:]
        assert int(c) == gen.CLASSES.index(cls.replace(" ", "_"))
        assert 5 <= int(k) <= 9
        assert trap["id"] == f"S1-F{c}-{k}"


def test_mb1r_fact_shaped_traps_keep_scoring_metadata(material):
    traps = {t["id"]: t for t in read(material, "traps.json")}
    probes = [json.loads(line) for line in (material / "probes.jsonl").read_text().splitlines()]
    for probe in probes:
        if probe["id"] in traps:
            assert re.fullmatch(r"S1-F\d{2}-[5-9]", probe["id"])
            assert probe["kind"] == "trap" and probe["expect"] == "ABSTAIN"
            assert traps[probe["id"]]["answer"] == "ABSTAIN"
            assert scorer._is_trap(probe)
            assert not scorer._is_trap(dict(probe, kind="canary"))
        else:
            assert probe["kind"] == "canary" and probe["expect"] == "value"
            assert not scorer._is_trap(probe)
    test_probe_format_scores_without_scorer_changes(material)


def test_n9_stale_row_links_fixture_before_unique_current_value(material):
    rows = transcript_rows(material)
    for fact in read(material, "facts.json"):
        if not fact["stale"]:
            continue
        old = fact["stale_source"]["row_index"]
        assert rows[old]["content"] == (
            f"[{fact['id']}-OLD; fixture {fact['fixture']}] Initial choice: {fact['stale']}.")
        assert old < fact["row_index"]
        assert sum(r["content"].count(fact["value"]) for r in rows) == 1
        assert fact["value"] in rows[fact["row_index"]]["content"]


@pytest.mark.parametrize("seed", [1, 2, 3, 17])
def test_mb2_answers_have_independent_payload_tokens(tmp_path, seed):
    gen.generate(seed, tmp_path, placements=True, classes12=True, smoke=True)
    facts, rows, seen = read(tmp_path, "facts.json"), transcript_rows(tmp_path), set()
    for fact in facts:
        nonce = fact["probe"].split("for fixture ", 1)[1][:-1]
        for answer in (fact["answer"], fact["stale"]):
            if answer is None:
                continue
            assert nonce not in answer
            # Fixed class wording is not an answer payload (e.g. MiB, because).
            tokens = set(re.findall(r"(?<![a-f0-9])[a-f0-9]{12,16}(?![a-f0-9])|\b\d{7,8}\b", answer))
            assert tokens and tokens.isdisjoint(seen)
            seen.update(tokens)
        assert fact["stale"] != fact["answer"]
        row = rows[fact["row_index"]]
        assert row["content"].count(fact["value"]) == 1
        assert row["content"][fact["char_offset"]:fact["char_offset"] + len(fact["value"])] == fact["value"]
        assert nonce in row["content"].split("]", 1)[0]
        assert sum(r["content"].count(nonce) for r in rows) == (2 if fact["stale"] else 1)
        assert not any(fact["value"] in r["content"] for r in rows[fact["row_index"] + 1:])
    assert verifier.verify(tmp_path)["status"] == "PASS"


@pytest.mark.parametrize("version", [None, "track-s-v1", "track-s-v2"])
def test_v1_verifier_rejects_old_material_version(material, version):
    manifest = read(material, "material.manifest.json")
    if version is None:
        del manifest["material_version"]
    else:
        manifest["material_version"] = version
    rewrite_material(material, "material.manifest.json", manifest)
    with pytest.raises(ValueError, match="material version mismatch"):
        verifier.verify(material)


@pytest.mark.parametrize("defect", ["missing", "extra", "id", "type", "name", "turn", "text", "role", "row_id", "tool_call_id", "tool_calls"])
def test_v2_verifier_rejects_call_metadata_and_projection(material, defect):
    rows = transcript_rows(material)
    index = next(i - 1 for i, r in enumerate(rows) if r["role"] == "tool")
    calls = rows[index]["tool_calls"]
    if defect in ("missing", "extra", "id", "type", "name"):
        if defect == "missing":
            rows[index]["tool_calls"] = []
        elif defect == "extra":
            calls.append(dict(calls[0]))
        elif defect == "name":
            calls[0]["function"]["name"] = "other_function"
        else:
            calls[0][defect] = "other"
        rewrite_transcript(material, rows)
        expected = "tool call metadata mismatch"
    else:
        turns = [json.loads(line) for line in (material / "turns.jsonl").read_text().splitlines()]
        field = "id" if defect == "row_id" else defect
        turns[index][field] = [] if field == "tool_calls" else "divergent"
        rewrite_material(material, "turns.jsonl", turns)
        expected = "turn projection mismatch"
    with pytest.raises(ValueError, match=expected):
        verifier.verify(material)


def test_v3_verifier_rejects_incomplete_class_placement_coverage(material):
    rows, facts = transcript_rows(material), read(material, "facts.json")
    fact = facts[0]
    row = rows[fact["row_index"]]
    line, body = row["content"].split("\n", 1)
    row["content"] = body + line + "\n"
    fact["placement"] = "tail"
    fact["char_offset"] = row["content"].index(fact["value"])
    rewrite_material(material, "facts.json", facts)
    rewrite_material(material, "canaries.json", facts)
    rewrite_transcript(material, rows)
    with pytest.raises(ValueError, match="placement coverage mismatch"):
        verifier.verify(material)


@pytest.mark.parametrize("leak", ["name", "probe"])
def test_v4_verifier_rejects_traps_in_transcript(material, leak):
    rows, trap = transcript_rows(material), read(material, "traps.json")[0]
    text = trap["probe"] if leak == "probe" else trap["probe"].split("for fixture ", 1)[1][:-1]
    # Replace equal-length filler after all scored items; token checkpoints stay valid.
    rows[-1]["content"] = text + rows[-1]["content"][len(text):]
    rewrite_transcript(material, rows)
    with pytest.raises(ValueError, match="trap leaked into transcript"):
        verifier.verify(material)


@pytest.mark.parametrize("defect", ["id", "text", "boundaries"])
def test_v5_verifier_rejects_divergent_batch_metadata(material, defect):
    batches = [json.loads(line) for line in (material / "probe_batches.jsonl").read_text().splitlines()]
    if defect == "boundaries":
        batches[1]["probes"].insert(0, batches[0]["probes"].pop())
    else:
        batches[0][defect] = "divergent"
    rewrite_material(material, "probe_batches.jsonl", batches)
    with pytest.raises(ValueError, match="probe batch mismatch"):
        verifier.verify(material)


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
