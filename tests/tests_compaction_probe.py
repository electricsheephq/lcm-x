"""Self-contained acceptance tests for the compaction probe scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "bench" / "instruments" / "compaction_probe"


def _load_script(name: str):
    path = PROBE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"compaction_probe_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_gen_material_shape_determinism_and_seed_salt(tmp_path):
    generator = _load_script("gen_material")
    first = tmp_path / "first"
    second = tmp_path / "second"
    other = tmp_path / "other"
    generator.generate(42, first)
    generator.generate(42, second)
    generator.generate(43, other)

    names = ("turns.jsonl", "canaries.json", "probes.jsonl", "material.manifest.json")
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in names)

    manifest = json.loads((first / "material.manifest.json").read_text())
    assert 500_000 <= manifest["estimated_total_tokens"] <= 700_000
    assert len(manifest["turn_estimates"]) == 35
    assert all("estimated_tokens" in row for row in manifest["turn_estimates"])

    turns = _jsonl(first / "turns.jsonl")
    canaries = json.loads((first / "canaries.json").read_text())
    probes = _jsonl(first / "probes.jsonl")
    assert len(turns) == 35
    assert len(canaries) == 30
    assert len(probes) == 35
    assert len({row["value"] for row in canaries}) == 30
    assert {row["class"] for row in canaries} == {"C1", "C2", "C3", "C4", "C5"}
    assert {row["epoch"] for row in canaries} == {"E0", "E1", "E2"}
    assert all("char_offset" in row for row in canaries)
    assert all(
        (1 <= row["turn"] <= 5)
        if row["epoch"] == "E0"
        else (15 <= row["turn"] <= 19)
        if row["epoch"] == "E1"
        else 32 <= row["turn"] <= 35
        for row in canaries
    )
    all_text = "\n".join(row["text"] for row in turns)
    assert all(all_text.count(row["value"]) == 1 for row in canaries)
    assert {row["kind"] for row in probes} == {"canary", "trap"}
    assert sum(row["kind"] == "canary" for row in probes) == 30
    assert sum(row["kind"] == "trap" for row in probes) == 5
    assert json.loads((other / "canaries.json").read_text()) != canaries


def test_drive_codex_dry_run_echoes_exact_command_shapes(tmp_path, capsys):
    driver = _load_script("drive_codex")
    material = tmp_path / "turns.jsonl"
    probes = tmp_path / "probes.jsonl"
    material.write_text(json.dumps({"turn": 1, "text": "material text"}) + "\n")
    probes.write_text(json.dumps({"id": "P1", "kind": "trap", "text": "probe text", "expect": "ABSTAIN"}) + "\n")
    out_dir = tmp_path / "run"

    assert driver.main(
        [
            "--codex-bin",
            sys.executable,
            "--model",
            "gpt-5.6-sol",
            "--material",
            str(material),
            "--probes",
            str(probes),
            "--out-dir",
            str(out_dir),
            "--dry-run",
        ]
    ) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("exec --json -m gpt-5.6-sol --skip-git-repo-check 'material text'")
    assert lines[1].endswith("exec resume '<THREAD_ID>' --json --skip-git-repo-check 'probe text'")
    assert all("--ephemeral" not in line for line in lines)
    manifest = json.loads((out_dir / "run.manifest.json").read_text())
    assert manifest["sid"] is None
    assert manifest["material_shas"]


@pytest.mark.parametrize(
    ("session_id", "expected_compactions"),
    [
        ("01a01e86-9c74-71f2-8fe8-5adf53b60747", 0),
        ("01a01dd1-d81c-7e00-a5ba-d2f0a05b2faa", 3),
    ],
)
def test_parse_named_real_rollouts(tmp_path, session_id, expected_compactions):
    parser = _load_script("parse_rollout")
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        pytest.skip("Codex session root is not present on this machine")
    report = parser.parse_rollout(session_id, sessions_root)
    assert report["model_context_window"] == 258400
    assert report["token_series"]
    assert len(report["compactions"]) == expected_compactions
    if expected_compactions:
        assert [row["window_number"] for row in report["compactions"]] == [1, 2, 3]
        # This rollout's schema carries no ContextCompaction completion items:
        # stall must be reported as UNMEASURED (null + source), never as the
        # marker-spacing milliseconds (that would fabricate a metric).
        for row in report["compactions"]:
            if row["stall_source"] == "item_completed":
                assert row["stall_ms"] > 0
            else:
                assert row["stall_ms"] is None
                assert row["stall_source"].startswith("unmeasured")
    else:
        assert report["compactions"] == []

