"""Regression for PR 1689 history."""
import json
from .bot1_support import HERE, load


def test_deep_generator_preserves_original_stale_chain(tmp_path):
    gen = load(HERE / "gen_material.py")
    base, fixed = tmp_path / "base", tmp_path / "deep"
    gen.generate(1, base, turns=130, placements=True, classes12=True, min_tokens=244800, min_events=2)
    original = json.loads((base / "facts.json").read_text())
    gen.generate(1, fixed, deep=True)
    facts = {f["id"]: f for f in json.loads((fixed / "facts.json").read_text())}
    rows = [json.loads(r) for r in (fixed / "transcript.jsonl").read_text().splitlines()]
    affected = [f for f in original if f["stale"] and len(facts[f["id"]]["versions"]) > 1]
    assert affected
    for f in affected:
        new = facts[f["id"]]
        assert bool(f["stale"] in [v["value"] for v in new.get("stale_history", [])]), "original stale version absent"
        for stale in new["stale_history"]:
            row = rows[stale["row_index"]]
            assert row["id"] == stale["row_id"]
            assert bool(row["content"][stale["char_offset"]:stale["char_offset"] + len(stale["value"])] == stale["value"]), "stale source offset mismatch"
    assert gen.DEEP_VERSION == "track-s-deep-v2"
