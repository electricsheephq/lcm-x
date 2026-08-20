"""Mocked-subprocess acceptance tests for the pty-free Hermes driver."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "bench" / "instruments" / "compaction_probe" / "drive_hermes_oneshot.py"
SPEC = importlib.util.spec_from_file_location("compaction_probe_drive_hermes_oneshot", DRIVER_PATH)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _base_args(
    tmp_path: Path,
    *,
    material: Path | None = None,
    probes_only: bool = False,
) -> list[str]:
    home = tmp_path / "hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "context:\n  engine: lcm\nmodel:\n  default: gpt-5.6-sol\n",
        encoding="utf-8",
    )
    probes = tmp_path / "probes.jsonl"
    _jsonl(probes, [{"probe_id": "P1", "prompt": "probe"}])
    args = [
        "--probes",
        str(probes),
        "--hermes-home",
        str(home),
        "--expect-engine",
        "lcm",
        "--expect-model",
        "gpt-5.6-sol",
        "--log",
        str(tmp_path / "raw.log"),
        "--results",
        str(tmp_path / "results.jsonl"),
        "--manifest",
        str(tmp_path / "run.manifest.json"),
        "--turn-timeout",
        "2",
    ]
    if material is not None:
        args.extend(["--material", str(material)])
    if probes_only:
        args.append("--probes-only")
    return args


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_rows_commands_environment_and_manifest_are_compatible(tmp_path, monkeypatch):
    material = tmp_path / "material.jsonl"
    _jsonl(material, [{"prompt": "material"}, {"status": True}])
    args = _base_args(tmp_path, material=material)
    calls: list[list[str]] = []
    envs: list[dict[str, str]] = []

    def fake_run(command, *, env, capture_output, text, timeout, check):
        assert capture_output is True
        assert text is True
        assert timeout == 2.0
        assert check is False
        calls.append(command)
        envs.append(env)
        prompt = command[2]
        if prompt == "probe":
            return subprocess.CompletedProcess(
                command,
                0,
                "Tool: lcm_recall({})\nanswer",
                "warning",
            )
        return subprocess.CompletedProcess(command, 0, f"answer for {prompt}", "")

    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    assert driver.main(args) == 0
    assert calls == [
        ["hermes", "-z", "material"],
        ["hermes", "-z", "/lcm status", "--continue"],
        ["hermes", "-z", "probe", "--continue"],
    ]
    assert all(env["HERMES_HOME"] == str((tmp_path / "hermes").resolve()) for env in envs)
    assert all(env["LCM_ENABLE_SLASH_COMMAND"] == "true" for env in envs)

    rows = _rows(tmp_path / "results.jsonl")
    assert rows[0]["kind"] == "material"
    assert rows[1]["kind"] == "status"
    assert rows[1]["status_via"] == "slash"
    assert rows[2] == {
        "turn_index": 3,
        "kind": "probe",
        "probe_id": "P1",
        "raw_answer": "Tool: lcm_recall({})\nanswer",
        "wall_ms": rows[2]["wall_ms"],
        "ts": rows[2]["ts"],
        "lcm_tool_fired": True,
    }
    assert "warning" in (tmp_path / "raw.log").read_text(encoding="utf-8")

    manifest = json.loads((tmp_path / "run.manifest.json").read_text(encoding="utf-8"))
    assert manifest["material_sha256"]
    assert manifest["probes_sha256"]
    assert manifest["argv"] == ["hermes", "-z", "<TEXT>"]
    assert manifest["continue_argv"][-1] == "--continue"
    assert manifest["env_names"] == ["HERMES_HOME", "LCM_ENABLE_SLASH_COMMAND"]
    assert '"prompt": "material"' not in json.dumps(manifest)


def test_timeout_writes_partial_row_and_aborts_without_sending_next_turn(tmp_path, monkeypatch):
    material = tmp_path / "material.jsonl"
    _jsonl(material, [{"prompt": "first"}, {"prompt": "must-not-run"}])
    args = _base_args(tmp_path, material=material)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 2, output="partial", stderr="late error")

    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    assert driver.main(args) == driver.BOOT_FAILURE_EXIT_CODE
    assert calls == [["hermes", "-z", "first"]]
    rows = _rows(tmp_path / "results.jsonl")
    assert rows == [
        {
            "turn_index": 1,
            "kind": "material",
            "raw_answer": "partial",
            "wall_ms": rows[0]["wall_ms"],
            "ts": rows[0]["ts"],
            "timed_out": True,
        }
    ]
    assert "late error" in (tmp_path / "raw.log").read_text(encoding="utf-8")


def test_config_assertion_failure_is_before_subprocess(tmp_path, monkeypatch, capsys):
    args = _base_args(tmp_path, probes_only=True)
    args[args.index("--expect-model") + 1] = "wrong-model"
    monkeypatch.setattr(driver.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("spawned"))

    with pytest.raises(SystemExit) as exc:
        driver.main(args)
    assert exc.value.code == 2
    assert "config assertion failed" in capsys.readouterr().err


def test_contamination_guard_blocks_material_but_probes_only_skips_it(tmp_path, monkeypatch):
    material = tmp_path / "material.jsonl"
    _jsonl(material, [{"prompt": "material"}])
    canaries = tmp_path / "canaries.json"
    canaries.write_text(json.dumps({"c1": "secret-canary"}), encoding="utf-8")
    args = _base_args(tmp_path, material=material)
    db = tmp_path / "hermes" / "lcm.db"
    db.write_bytes(b"secret-canary")
    args.extend(["--canaries", str(canaries)])
    monkeypatch.setattr(driver.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("spawned"))
    assert driver.main(args) == 71

    probe_args = _base_args(tmp_path, probes_only=True)
    probe_args.extend(["--canaries", str(canaries)])
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "probe answer", ""),
    )
    assert driver.main(probe_args) == 0
    assert _rows(tmp_path / "results.jsonl")[-1]["kind"] == "probe"


def test_status_unknown_command_uses_tool_fallback_and_records_path(tmp_path, monkeypatch):
    material = tmp_path / "material.jsonl"
    _jsonl(material, [{"status": True}])
    args = _base_args(tmp_path, material=material)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        if command[2] == "/lcm status":
            return subprocess.CompletedProcess(command, 1, "Unknown command: /lcm status", "")
        return subprocess.CompletedProcess(command, 0, "effective status", "")

    monkeypatch.setattr(driver.subprocess, "run", fake_run)

    assert driver.main(args) == 0
    assert calls == [
        ["hermes", "-z", "/lcm status", "--continue"],
        [
            "hermes",
            "-z",
            "Call the lcm_status tool and paste its full output verbatim.",
            "--continue",
        ],
        ["hermes", "-z", "probe", "--continue"],
    ]
    row = _rows(tmp_path / "results.jsonl")[0]
    assert row["raw_answer"] == "effective status"
    assert row["status_via"] == "tool"


def test_dry_run_prints_commands_without_spawning(tmp_path, monkeypatch, capsys):
    material = tmp_path / "material.jsonl"
    _jsonl(material, [{"prompt": "material"}])
    args = _base_args(tmp_path, material=material)
    args.append("--dry-run")
    monkeypatch.setattr(driver.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("spawned"))

    assert driver.main(args) == 0
    output = capsys.readouterr().out.splitlines()
    assert output == [
        "hermes -z material",
        "hermes -z probe --continue",
    ]
    assert not (tmp_path / "raw.log").exists()
