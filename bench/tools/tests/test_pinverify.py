from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bench.tools import pinverify


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_pre_and_post_verify_all_pin_classes(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-qm", "fixture")
    head = _git(repo, "rev-parse", "HEAD")

    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("BENCH_PIN_TEST", "fixed")
    pins = {
        "version": 1,
        "worktrees": {
            "fixture": {"path": str(repo), "git_sha": head[:12], "clean": True}
        },
        "binaries": {
            "python": {
                "name": sys.executable,
                "sha256": pinverify.sha256_file(Path(sys.executable)),
            }
        },
        "files": {
            "dataset": {
                "path": str(dataset),
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        },
        "env": {"BENCH_PIN_TEST": "fixed"},
    }
    pins_path = tmp_path / "pins.yaml"
    pins_path.write_text(json.dumps(pins), encoding="utf-8")

    passed, report_path, report = pinverify.verify(pins_path, "pre-run")

    assert passed
    assert report_path.name == "PINS-PRERUN.txt"
    assert "status: PASS" in report
    assert "signed_off: yes" in report

    dataset.write_text("[1]\n", encoding="utf-8")
    passed, report_path, report = pinverify.verify(pins_path, "post-run")

    assert not passed
    assert report_path.name == "PINS-POSTRUN.txt"
    assert "MISMATCH files.dataset.sha256" in report
    assert "signed_off: no" in report
    assert pinverify.main(["post-run", str(pins_path)]) == 1
    assert "status: FAIL" in capsys.readouterr().out


def test_binary_missing_is_a_want_got_mismatch():
    checks = pinverify.run_checks(
        {
            "version": 1,
            "binaries": {
                "missing": {
                    "name": "definitely-not-a-real-benchmark-binary",
                    "sha256": "wanted",
                }
            },
        }
    )

    assert checks == [
        pinverify.Check(
            "binaries.missing.sha256", "wanted", "<not found on PATH>", False
        )
    ]
