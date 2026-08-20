from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


def _load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drive_hermes = _load_module("compaction_probe_drive_hermes", "drive_hermes.py")
report_pilot = _load_module("compaction_probe_report_pilot", "report_pilot.py")
score_probes = _load_module("compaction_probe_score_probes", "score_probes.py")


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_score_probes_covers_three_way_unparseable_and_trap_negative(tmp_path):
    canaries = tmp_path / "canaries.json"
    canaries.write_text(json.dumps({"c-blue": "Blue Key", "c-green": "Green Key"}), encoding="utf-8")
    probes = tmp_path / "probes.jsonl"
    _jsonl(
        probes,
        [
            {"probe_id": "p-correct", "canary_id": "c-blue", "epoch": "E0", "class": "decisions"},
            {"probe_id": "p-abstain", "canary_id": "c-green", "epoch": "E0", "class": "decisions"},
            {"probe_id": "p-hallucinate", "canary_id": "c-green", "epoch": "E1", "class": "paths"},
            {"probe_id": "p-empty", "trap": True, "epoch": "E1", "class": "traps"},
            {"probe_id": "p-trap-abstain", "trap": True, "epoch": "E2", "class": "traps"},
        ],
    )
    results = tmp_path / "results.jsonl"
    _jsonl(
        results,
        [
            {"turn_index": 1, "kind": "probe", "probe_id": "p-correct", "raw_answer": "blue-key"},
            {"turn_index": 2, "kind": "probe", "probe_id": "p-abstain", "raw_answer": "I don't know."},
            {"turn_index": 3, "kind": "probe", "probe_id": "p-hallucinate", "raw_answer": "Red key."},
            {"turn_index": 4, "kind": "probe", "probe_id": "p-empty", "raw_answer": ""},
            {"turn_index": 5, "kind": "probe", "probe_id": "p-trap-abstain", "raw_answer": "not sure"},
        ],
    )

    payload = score_probes.score(results, canaries, probes)

    assert len(payload["probes"]) == 5
    assert [row["classification"] for row in payload["probes"]] == [
        "CORRECT",
        "ABSTAIN",
        "HALLUCINATE",
        "HALLUCINATE",
        "ABSTAIN",
    ]
    assert payload["probes"][3]["unparseable"] is True
    assert payload["probes"][4]["correct_negative"] is True
    assert payload["three_way_totals"] == {"CORRECT": 1, "ABSTAIN": 2, "HALLUCINATE": 2}
    assert payload["totals"]["retention"] == 1 / 3
    assert payload["per_epoch"]["E0"]["total"] == 2


def _write_arm(directory: Path, name: str, classifications: list[str]) -> None:
    directory.mkdir()
    probes = [
        {"probe_id": f"p{index}", "classification": value}
        for index, value in enumerate(classifications, 1)
    ]
    scores = {
        "probes": probes,
        "totals": {
            "total": len(probes),
            "correct": classifications.count("CORRECT"),
            "abstain": classifications.count("ABSTAIN"),
            "hallucinate": classifications.count("HALLUCINATE"),
            "canary_total": len(probes),
            "retention": classifications.count("CORRECT") / len(probes),
        },
        "per_epoch": {"E0": {"correct": classifications.count("CORRECT"), "canary_total": len(probes)}},
        "per_class": {"decisions": {"correct": classifications.count("CORRECT"), "canary_total": len(probes)}},
    }
    (directory / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps({"compaction_count": 2, "stall_total_ms": 99, "total_tokens": 1234}),
        encoding="utf-8",
    )
    (directory / "run.manifest.json").write_text(
        json.dumps({"config_sha256": f"sha-{name}"}), encoding="utf-8"
    )


def test_report_pilot_emits_aa_prime_discordance_section(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_arm(left, "R2-A", ["CORRECT", "ABSTAIN"])
    _write_arm(right, "R2-A′", ["CORRECT", "HALLUCINATE"])

    report = report_pilot.render([f"R2-A={left}", f"R2-A′={right}"])

    assert "## A/A′" in report
    assert "discordance_count" in report
    assert "| p2 | ABSTAIN | HALLUCINATE | 1 |" in report
    assert "0.5" in report
    assert "sha-R2-A" in report
    assert "2 | 99 | 1234" in report


def test_drive_hermes_dry_run_probes_only_asserts_config_and_does_not_spawn(tmp_path, capsys):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config = hermes_home / "config.yaml"
    config.write_text(
        "context:\n  engine: lcm\nmodel:\n  default: gpt-5.6-sol\n",
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"probe_id": "p1", "prompt": "What?"}])
    raw_log = tmp_path / "raw.pty.log"

    rc = drive_hermes.main(
        [
            "--probes",
            str(probes),
            "--probes-only",
            "--dry-run",
            "--log",
            str(raw_log),
            "--hermes-home",
            str(hermes_home),
            "--expect-engine",
            "lcm",
            "--expect-model",
            "gpt-5.6-sol",
        ]
    )

    output = capsys.readouterr().out
    assert rc == 0
    assert "turn_count: 1" in output
    assert "material_count: 0" in output
    assert "probe_count: 1" in output
    assert "turn_timeout_s: 600" in output
    assert "expected_engine: lcm" in output
    assert "expected_model: gpt-5.6-sol" in output
    assert hashlib.sha256(config.read_bytes()).hexdigest() in output
    assert not raw_log.exists()
