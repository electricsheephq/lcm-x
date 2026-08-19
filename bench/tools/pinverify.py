#!/usr/bin/env python3
"""Verify benchmark worktree, binary, file, and environment pins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Check:
    name: str
    want: str
    got: str
    ok: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pins(path: str | Path) -> dict[str, Any]:
    """Load the documented JSON-compatible YAML schema."""
    pins = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(pins, dict) or pins.get("version") != 1:
        raise ValueError("pins.yaml must be an object with version: 1")
    return pins


def _run_git(path: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_sha_check(label: str, config: dict[str, Any]) -> Check:
    path = Path(config["path"])
    want = str(config["git_sha"])
    try:
        got = _run_git(path, "rev-parse", "HEAD")
        resolved_want = _run_git(path, "rev-parse", f"{want}^{{commit}}")
        ok = got == resolved_want
    except (OSError, subprocess.CalledProcessError) as exc:
        got = f"<error: {exc}>"
        ok = False
    return Check(f"worktrees.{label}.git_sha", want, got, ok)


def _git_clean_check(label: str, config: dict[str, Any]) -> Check:
    path = Path(config["path"])
    want_clean = bool(config.get("clean", True))
    want = "clean" if want_clean else "dirty"
    try:
        got = "clean" if not _run_git(path, "status", "--porcelain") else "dirty"
        ok = got == want
    except (OSError, subprocess.CalledProcessError) as exc:
        got = f"<error: {exc}>"
        ok = False
    return Check(f"worktrees.{label}.clean", want, got, ok)


def run_checks(pins: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []

    for label, config in sorted(pins.get("worktrees", {}).items()):
        checks.append(_git_sha_check(label, config))
        checks.append(_git_clean_check(label, config))

    for label, config in sorted(pins.get("binaries", {}).items()):
        want = str(config["sha256"])
        resolved = shutil.which(str(config["name"]))
        got = sha256_file(Path(resolved)) if resolved else "<not found on PATH>"
        checks.append(Check(f"binaries.{label}.sha256", want, got, got == want))

    for label, config in sorted(pins.get("files", {}).items()):
        want = str(config["sha256"])
        path = Path(config["path"])
        got = sha256_file(path) if path.is_file() else "<missing>"
        checks.append(Check(f"files.{label}.sha256", want, got, got == want))

    for name, expected in sorted(pins.get("env", {}).items()):
        want = str(expected)
        got = os.environ.get(name, "<unset>")
        checks.append(Check(f"env.{name}", want, got, got == want))

    return checks


def render_report(phase: str, checks: list[Check]) -> str:
    passed = all(check.ok for check in checks)
    lines = [
        f"PINS-{phase.upper()}",
        f"generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"status: {'PASS' if passed else 'FAIL'}",
    ]
    for check in checks:
        state = "OK" if check.ok else "MISMATCH"
        lines.append(
            f"{state} {check.name} want={json.dumps(check.want)} "
            f"got={json.dumps(check.got)}"
        )
    unsigned = "\n".join(lines) + "\n"
    report_sha = hashlib.sha256(unsigned.encode()).hexdigest()
    lines.append(f"report_sha256: {report_sha}")
    lines.append(f"signed_off: {'yes' if passed else 'no'}")
    return "\n".join(lines) + "\n"


def verify(
    pins_path: str | Path,
    phase: str,
    output: str | Path | None = None,
) -> tuple[bool, Path, str]:
    pins_path = Path(pins_path)
    checks = run_checks(load_pins(pins_path))
    phase_name = "PRERUN" if phase == "pre-run" else "POSTRUN"
    report = render_report(phase_name, checks)
    output_path = (
        Path(output) if output else pins_path.parent / f"PINS-{phase_name}.txt"
    )
    output_path.write_text(report, encoding="utf-8")
    return all(check.ok for check in checks), output_path, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a stdlib-only, JSON-compatible pins.yaml."
    )
    parser.add_argument("phase", choices=("pre-run", "post-run"))
    parser.add_argument("pins", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    passed, output_path, report = verify(args.pins, args.phase, args.output)
    print(report, end="")
    print(f"report: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
