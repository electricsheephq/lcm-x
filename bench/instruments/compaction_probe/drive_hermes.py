#!/usr/bin/env python3
"""Drive a Hermes CLI session from JSONL material and probe scripts.

This is intentionally a small fork of ``drive_hermes_cli.py``.  It keeps the
pty/prompt protocol and ``SHELL::`` escape hatch, but never accepts turns on
argv: long turns belong in JSONL files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pty
import re
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROMPT_RE = re.compile(r"\x1b\[[0-9;]*m?\s*❯")
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
DEFAULT_TURN_TIMEOUT = 600.0
DEFAULT_BOOT_TIMEOUT = 90.0
DEFAULT_QUIET_SECONDS = 6.0
DEFAULT_STATUS_COMMAND = "/lcm status"
# Compatibility names retained from the base script and the run-sheet wording.
TURN_TIMEOUT = DEFAULT_TURN_TIMEOUT
BOOT_TIMEOUT = DEFAULT_BOOT_TIMEOUT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_inline_comment(value: str) -> str:
    """Remove a YAML comment without treating a # inside quotes as one."""
    quoted: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted == '"':
            escaped = True
            continue
        if char in "'\"":
            if quoted == char:
                quoted = None
            elif quoted is None:
                quoted = char
            continue
        if char == "#" and quoted is None and (
            index == 0 or value[index - 1].isspace()
        ):
            return value[:index].rstrip()
    return value.strip()


def _yaml_scalar(value: str) -> str | None:
    value = _strip_inline_comment(value).strip()
    if not value or value in {"null", "Null", "NULL", "~"}:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        if value[0] == '"':
            try:
                return str(json.loads(value))
            except json.JSONDecodeError:
                pass
        return value[1:-1]
    return value


def read_config_values(config_path: Path) -> dict[str, str | None]:
    """Read the two scalar paths needed for the arm assertion.

    PyYAML is deliberately not a dependency of the instrument.  The Hermes
    config paths are ordinary nested scalar mappings, so this parser handles
    those mappings, quoted values, comments, and dotted keys.
    """
    text = config_path.read_text(encoding="utf-8")
    stack: list[tuple[int, str]] = []
    values: dict[str, str | None] = {}
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.startswith("-") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip().strip('"\'')
        if not key:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if raw_value.strip():
            path = ".".join(part for _, part in stack + [(indent, key)])
            values[path] = _yaml_scalar(raw_value)
        else:
            stack.append((indent, key))
    return {
        "context.engine": values.get("context.engine"),
        "model.default": values.get("model.default"),
    }


def load_jsonl(path: Path, label: str) -> list[Any]:
    rows: list[Any] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open {label} JSONL {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid {label} JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def row_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if not isinstance(row, dict):
        raise ValueError(f"turn row must be an object or string, got {type(row).__name__}")
    for key in ("turn", "prompt", "content", "message", "text", "question"):
        if key in row:
            value = row[key]
            if isinstance(value, str):
                return value
            if value is not None:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise ValueError(f"turn row has no text field: {row!r}")


def is_status_row(row: Any) -> bool:
    return isinstance(row, dict) and row.get("status") is True


def probe_id(row: Any, index: int) -> str:
    if isinstance(row, dict):
        for key in ("probe_id", "id", "name"):
            if row.get(key) is not None:
                return str(row[key])
    return f"probe-{index}"


def _prompt_seen(data: bytes) -> bool:
    text = data.decode("utf-8", errors="replace")
    return bool(PROMPT_RE.search(text) or "❯" in text)


def clean_output(data: bytes) -> str:
    return ANSI_RE.sub("", data.decode("utf-8", errors="replace")).strip()


class PtySession:
    """Minimal pty pump copied from the base driver, with per-turn capture."""

    def __init__(self, log_path: Path):
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execvp("hermes", ["hermes", "--cli"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("ab")
        self.current = bytearray()
        self.last_output = time.monotonic()
        self.prompt_seen = False
        self.closed = False

    def pump(self, timeout: float = 0.5) -> bool:
        if self.closed:
            return False
        try:
            ready, _, _ = select.select([self.fd], [], [], timeout)
        except (OSError, ValueError):
            return False
        if self.fd not in ready:
            return True
        try:
            chunk = os.read(self.fd, 65536)
        except OSError:
            return False
        if not chunk:
            return False
        self.current.extend(chunk)
        self.log.write(chunk)
        self.log.flush()
        self.last_output = time.monotonic()
        self.prompt_seen = self.prompt_seen or _prompt_seen(bytes(self.current))
        return True

    def wait_idle(self, max_wait: float, quiet_seconds: float) -> bool:
        started = time.monotonic()
        while time.monotonic() - started < max_wait:
            if not self.pump(0.5):
                return False
            if self.prompt_seen and time.monotonic() - self.last_output >= quiet_seconds:
                return True
        return False

    def send(self, text: str, timeout: float, quiet_seconds: float) -> tuple[str, bool]:
        self.current = bytearray()
        self.prompt_seen = False
        os.write(self.fd, text.encode("utf-8") + b"\r")
        completed = self.wait_idle(timeout, quiet_seconds)
        return clean_output(bytes(self.current)), completed

    def drain_and_close(self, seconds: float = 30.0) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and self.pump(0.5):
            pass
        try:
            os.waitpid(self.pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass
        self.log.close()
        self.closed = True


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def build_manifest(
    *,
    config_path: Path,
    config_sha256: str,
    values: dict[str, str | None],
    args: argparse.Namespace,
    raw_log: Path,
    results: Path,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "config_sha": config_sha256,
        "context": {"engine": values.get("context.engine")},
        "model": {"default": values.get("model.default")},
        "context.engine": values.get("context.engine"),
        "model.default": values.get("model.default"),
        "context_engine": values.get("context.engine"),
        "model_default": values.get("model.default"),
        "expected_engine": args.expect_engine,
        "expected_model": args.expect_model,
        "turn_timeout_s": args.turn_timeout,
        "boot_timeout_s": args.boot_timeout,
        "quiet_seconds": args.quiet_seconds,
        "status_command": args.status_command,
        "probes_only": bool(args.probes_only),
        "raw_log": str(raw_log),
        "results": str(results),
    }


def assert_config(values: dict[str, str | None], args: argparse.Namespace) -> None:
    actual_engine = values.get("context.engine")
    actual_model = values.get("model.default")
    mismatches = []
    if actual_engine != args.expect_engine:
        mismatches.append(f"context.engine={actual_engine!r} expected {args.expect_engine!r}")
    if actual_model != args.expect_model:
        mismatches.append(f"model.default={actual_model!r} expected {args.expect_model!r}")
    if mismatches:
        raise ValueError("config assertion failed: " + "; ".join(mismatches))


def _shell_turn(text: str) -> str:
    command = text[len("SHELL::") :]
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return f"rc={result.returncode}\n{result.stdout}{result.stderr}".strip()


def _dispatch_turn(
    session: PtySession,
    text: str,
    *,
    timeout: float,
    quiet_seconds: float,
) -> tuple[str, bool]:
    if text.startswith("SHELL::"):
        try:
            return _shell_turn(text), True
        except subprocess.TimeoutExpired as exc:
            return f"shell timeout: {exc}", False
    return session.send(text, timeout, quiet_seconds)


def run(args: argparse.Namespace) -> int:
    hermes_home = Path(args.hermes_home).expanduser()
    config_path = hermes_home if hermes_home.suffix in {".yaml", ".yml"} else hermes_home / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Hermes config not found: {config_path}")
    values = read_config_values(config_path)
    config_sha = sha256_file(config_path)

    raw_log = Path(args.raw_log)
    results_path = Path(args.results) if args.results else raw_log.with_name("results.jsonl")
    manifest_path = Path(args.manifest) if args.manifest else raw_log.with_name("run.manifest.json")
    manifest = build_manifest(
        config_path=config_path,
        config_sha256=config_sha,
        values=values,
        args=args,
        raw_log=raw_log,
        results=results_path,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert_config(values, args)

    material_rows = [] if args.probes_only else load_jsonl(Path(args.material), "material")
    probe_rows = load_jsonl(Path(args.probes), "probe")
    plan_rows: list[tuple[str, Any]] = []
    if not args.probes_only:
        plan_rows.extend(("status" if is_status_row(row) else "material", row) for row in material_rows)
    plan_rows.extend(("probe", row) for row in probe_rows)

    if args.dry_run:
        print("pty plan")
        print(f"turn_count: {len(plan_rows)}")
        print(f"material_count: {sum(kind == 'material' for kind, _ in plan_rows)}")
        print(f"probe_count: {sum(kind == 'probe' for kind, _ in plan_rows)}")
        print(f"status_count: {sum(kind == 'status' for kind, _ in plan_rows)}")
        print(f"probes_only: {str(bool(args.probes_only)).lower()}")
        print(f"turn_timeout_s: {args.turn_timeout:g}")
        print(f"boot_timeout_s: {args.boot_timeout:g}")
        print(f"expected_engine: {args.expect_engine}")
        print(f"expected_model: {args.expect_model}")
        print(f"config_sha256: {config_sha}")
        return 0

    # Contamination guard (run-sheet §3): before any material turn, the fresh
    # store must contain ZERO canary values. Fail closed on any hit.
    if args.canaries and not args.probes_only:
        canary_rows = load_jsonl(Path(args.canaries), "canaries")
        values = [str(r.get("value")) for r in canary_rows if isinstance(r, dict) and r.get("value")]
        lcm_db_hits = 0
        home = Path(args.hermes_home).expanduser() if args.hermes_home else Path.home() / ".hermes"
        for db in home.rglob("lcm.db"):
            blob = db.read_bytes()
            lcm_db_hits += sum(1 for v in values if v.encode() in blob)
        if lcm_db_hits:
            sys.stderr.write(f"[driver] CONTAMINATION: {lcm_db_hits} canary-value hits in pre-run store(s) under {home}\n")
            return 71

    session = PtySession(raw_log)
    try:
        session.wait_idle(args.boot_timeout, args.quiet_seconds)
        for turn_index, (kind, row) in enumerate(plan_rows, 1):
            if kind == "status":
                if isinstance(row, dict):
                    text = str(row.get("command") or args.status_command)
                else:
                    text = args.status_command
            else:
                text = row_text(row)
            started = time.monotonic()
            answer, completed = _dispatch_turn(
                session,
                text,
                timeout=args.turn_timeout,
                quiet_seconds=args.quiet_seconds,
            )
            entry: dict[str, Any] = {
                "turn_index": turn_index,
                "kind": kind,
                "raw_answer": answer,
                "wall_ms": round((time.monotonic() - started) * 1000, 3),
                "ts": utc_now(),
            }
            if kind == "probe":
                entry["probe_id"] = probe_id(row, turn_index)
                # R3 diagnostic (run-sheet §1): did any lcm_* retrieval tool
                # fire during this probe turn? Detected from the pty transcript
                # of the turn -- distinguishes "recall policy never triggered"
                # from "retrieval failed". Diagnostic only, never a score.
                entry["lcm_tool_fired"] = bool(
                    re.search(r"lcm_(recall|grep|retrieve|recent|expand|query_state|load_session|compile_evidence|evidence_pack)", answer)
                )
            if not completed:
                entry["timed_out"] = True
            write_jsonl(results_path, entry)
            sys.stderr.write(
                f"[driver] sent {kind} turn {turn_index}/{len(plan_rows)}: {text[:60]!r}"
                + (" (timeout)" if not completed else "")
                + "\n"
            )
        try:
            session.send("/exit", min(args.turn_timeout, 30.0), args.quiet_seconds)
        except (OSError, ValueError):
            pass
    finally:
        session.drain_and_close()
    sys.stderr.write("[driver] done\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", help="material turns JSONL")
    parser.add_argument("--probes", required=True, help="probe turns JSONL")
    parser.add_argument("--log", "--raw-log", dest="raw_log", default="raw.pty.log")
    parser.add_argument("--results")
    parser.add_argument("--manifest")
    parser.add_argument("--hermes-home", default="~/.hermes")
    parser.add_argument("--expect-engine", required=True)
    parser.add_argument("--expect-model", required=True)
    parser.add_argument("--turn-timeout", type=float, default=DEFAULT_TURN_TIMEOUT)
    parser.add_argument("--boot-timeout", type=float, default=DEFAULT_BOOT_TIMEOUT)
    parser.add_argument("--quiet-seconds", type=float, default=DEFAULT_QUIET_SECONDS)
    parser.add_argument("--status-command", default=DEFAULT_STATUS_COMMAND)
    parser.add_argument("--probes-only", action="store_true")
    parser.add_argument("--canaries", help="canaries.json for the pre-material contamination guard")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.probes_only and not args.material:
        parser.error("--material is required unless --probes-only is set")
    try:
        return run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"drive_hermes.py: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
