"""Model-free v4 requirements and exact base-sha compatibility receipts."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen, verifier, scorer = (load(n) for n in ("gen_material", "verify_material", "score_probes"))
# Unmodified origin/main ce186095, README Python 3.12 environment, default seed 1.
GOLDEN = {'v3': {'canaries.json': 'ae5400fae533c32ee1ce4014c836f7d87d63d931b4e959ad070c7e8b710ffff9', 'traps.json': '16e1dbb78e23f5274a70e9653cd52768741f356467fc3b16c144d6d733132d56', 'turns.jsonl': '154a74fff7410537d27659a7be7295334f4d0140975c782a233604dc07704ae6', 'transcript.jsonl': '7f995d13ffeefdab67f3b93c4c24b6a352ca5e8527dfc8b7c08a08cb7633ce46', 'material.manifest.json': '73dd4e557e7f44c7daea15eca414cd617bab3d12b03236a2a0088113486c9e11', 'facts.json': 'ae5400fae533c32ee1ce4014c836f7d87d63d931b4e959ad070c7e8b710ffff9', 'admission.manifest.json': 'de0e79315c49dcc0397fb4586e7ad00262799a8d4e55f7306178c5e5dbf52b4e', 'continuation.json': '47e9a4c88ac16da6a63c395d94bc09356e6f8905d3b99f40218a7700f9823f92', 'probes.jsonl': '60a68e1091afeae5b0f8b7902bc5b0918aea72e17dbf3dcc7f6388f319bf7e60', 'probe_batches.jsonl': '987176466d63f735bece6904c11d47b3beac1061d9285c73d92646f6ebfbf72e'}, 'legacy': {'canaries.json': 'f32e942f3c5e8b814bcb4774719f251caff0c688afe4bba12d7432a35ecc4478', 'turns.jsonl': 'e20fe63e69a1e5ed3a85130602e8f79f5b79ba4bb5592b590e663ccad7399490', 'material.manifest.json': '478d73ce2502d6994fe4c4c76b44b96a022ff09538d389796e9daf948c78a58f', 'probes.jsonl': 'b80c2fb6ff43875a0473dee5eea6cd121a3f8c9289122fc9406860102cbcb1db'}}


def read(directory, name):
    text = (directory / name).read_text()
    return [json.loads(line) for line in text.splitlines()] if name.endswith(".jsonl") else json.loads(text)


@pytest.fixture
def material(tmp_path):
    gen.generate(1, tmp_path, v4=True)
    return tmp_path


def bias(directory, *, facts=None, traps=None, manifest=None, probes=None):
    rows = read(directory, "transcript.jsonl")
    prefix, total = [], 0
    count = gen.token_counter()
    for row in rows:
        total += count(row["content"])
        prefix.append(total)
    def check(ok, message):
        if not ok:
            raise ValueError(message)
    return verifier._verify_v4(rows, facts or read(directory, "facts.json"), traps or read(directory, "traps.json"),
                              manifest or read(directory, "material.manifest.json"),
                              probes or read(directory, "lifecycle_probes.jsonl"), prefix, count, check)


def test_requirement_1_independent_values_and_template_baseline(material):
    receipt = verifier.verify(material)["bias_receipts"]
    assert all(v["exact"] == 0 and v["rate"] <= .02 for v in receipt["guessability"].values())
    assert all(v <= .2 for v in receipt["max_shared_sixgram_fraction"].values())
    facts = read(material, "facts.json")
    facts[1]["value"] = facts[0]["value"]
    with pytest.raises(ValueError, match="shared value substring"):
        bias(material, facts=facts)


def test_requirement_2_balanced_user_thirds_and_long_pastes(material):
    receipt = verifier.verify(material)["bias_receipts"]
    assert all(.25 <= v / receipt["user_facts"] <= .4 for v in receipt["placement_histogram"].values())
    assert all(.25 <= v / receipt["user_facts"] <= .4 for v in receipt["token_placement_histogram"].values())
    assert receipt["long_user_middle"] / receipt["user_facts"] >= .2
    facts = read(material, "facts.json")
    for f in facts:
        if f["row_role"] == "user":
            f["placement"] = "head"
    with pytest.raises(ValueError, match="long user middle coverage"):
        bias(material, facts=facts)


def test_requirement_3_request_lifecycle_corrections_and_horizons(material):
    receipt = verifier.verify(material)["bias_receipts"]
    assert receipt["lifecycle_counts"] == dict(completed=2, cancelled=2, superseded=1, pending=1)
    assert receipt["lifecycle_probe_counts"] == dict(corrected_value=3, stale_task=3, current_request=1)
    probes = read(material, "lifecycle_probes.jsonl")
    probes[0]["probe_token_position"] += 1
    with pytest.raises(ValueError, match="lifecycle token horizon"):
        bias(material, probes=probes)


def test_requirement_4_traps_share_real_template_and_sibling(material):
    assert verifier.verify(material)["bias_receipts"]["trap_template_check"] == "PASS"
    traps = read(material, "traps.json")
    traps[0]["probe"] = "If unstated, abstain. " + traps[0]["probe"]
    with pytest.raises(ValueError, match="trap template mismatch"):
        bias(material, traps=traps)


@pytest.mark.parametrize("seed,smoke", [(i, False) for i in range(1, 9)] + [(9, True)])
def test_requirement_5_eight_independent_seeds_and_full_smoke(tmp_path, seed, smoke):
    gen.generate(seed, tmp_path, v4=True, smoke=smoke)
    result = verifier.verify(tmp_path)
    assert result["status"] == "PASS" and result["tokens"] >= 244800
    assert result["bias_receipts"]["user_facts"] == 22
    assert result["planned_post_supersession_trigger_spans"] >= 2
    assert read(tmp_path, "material.manifest.json")["material_version"] == "track-s-v4"


@pytest.mark.parametrize("path", ["v3", "legacy"])
def test_base_sha_byte_identity(tmp_path, path):
    gen.generate(1, tmp_path, placements=path == "v3", classes12=path == "v3")
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()} == GOLDEN[path]


def test_determinism_and_scorer_compatibility(material, tmp_path):
    again, other = tmp_path / "again", tmp_path / "other"
    gen.generate(1, again, v4=True)
    gen.generate(2, other, v4=True)
    assert all(p.read_bytes() == (again / p.name).read_bytes() for p in material.iterdir() if p.is_file())
    assert read(material, "facts.json") != read(other, "facts.json")
    facts, traps = read(material, "facts.json"), read(material, "traps.json")
    answers = tmp_path / "answers.jsonl"
    answers.write_text("".join(json.dumps(dict(probe_id=f["id"], raw_answer=f["answer"])) + "\n" for f in facts) +
                       "".join(json.dumps(dict(probe_id=t["id"], raw_answer="I don't know.")) + "\n" for t in traps))
    result = scorer.score(answers, material / "canaries.json", material / "probes.jsonl")
    assert result["three_way_totals"] == {"CORRECT": 60, "ABSTAIN": 5, "HALLUCINATE": 0}
    assert len(read(material, "probes.jsonl")) == 65
    assert {p["kind"] for p in read(material, "probes.jsonl")} == {"canary", "trap"}
