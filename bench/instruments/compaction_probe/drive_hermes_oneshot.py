#!/usr/bin/env python3
"""Drive Hermes with one subprocess per material, status, or probe turn.

This is the pty-free pilot driver.  Each invocation receives one complete
turn through ``hermes -z``; later invocations use ``--continue`` so Hermes can
reuse the session stored under the pinned ``HERMES_HOME``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TURN_TIMEOUT = 600.0
DEFAULT_BOOT_TIMEOUT = 90.0
DEFAULT_QUIET_SECONDS = 6.0
DEFAULT_STATUS_COMMAND = "/lcm status"
BOOT_FAILURE_EXIT_CODE = 72
MAX_TURN_CHARS = 200_000
LCM_TOOL_NAMES = (
    "recall",
    "grep",
    "retrieve",
    "recent",
    "expand",
    "query_state",
    "load_session",
    "compile_evidence",
    "evidence_pack",
)
# Hermes' CLI invocation markers are diagnostic-only.  A prose mention of an
# lcm_* tool must not be counted as a fired tool.
LCM_TOOL_CALL_RE = re.compile(
    rf"(?m)^\s*(?:📞\s*)?Tool(?:\s+\d+)?\s*:\s*lcm_(?:{'|'.join(LCM_TOOL_NAMES)})\s*\("
    rf"|^\s*⚡\s*Concurrent:.*\blcm_(?:{'|'.join(LCM_TOOL_NAMES)})\b"
)
TURN_TIMEOUT = DEFAULT_TURN_TIMEOUT
BOOT_TIMEOUT = DEFAULT_BOOT_TIMEOUT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


RUNTIME_CONFIG_PATH: Path | None = None
RUNTIME_CONFIG_SHA: str | None = None


class ConfigDriftError(RuntimeError):
    """config.yaml changed while a multi-hour run was in flight."""


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
    """Read the two scalar config paths needed for arm assertions."""
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


def _canary_value(row: Any) -> Any:
    if isinstance(row, dict):
        for key in ("value", "answer", "expected", "expected_value", "canary"):
            if row.get(key) is not None:
                return row[key]
        return None
    return row


def load_canary_values(path: Path) -> list[str]:
    """Load canonical canaries JSON (list, flat object, or wrapped object)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read canaries JSON {path}: {exc}") from exc
    if isinstance(payload, dict) and "canaries" in payload:
        payload = payload["canaries"]
    if isinstance(payload, list):
        raw_values = [_canary_value(row) for row in payload]
    elif isinstance(payload, dict):
        raw_values = [_canary_value(value) for value in payload.values()]
    else:
        raise ValueError("canaries JSON must be a list or object")
    values = [str(value) for value in raw_values if value is not None and str(value)]
    if not values:
        raise ValueError(f"canaries JSON {path} contains no extractable values")
    return values


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


def assert_config(
    values: dict[str, str | None],
    args: argparse.Namespace,
    config_sha256: str | None = None,
) -> None:
    actual_engine = values.get("context.engine")
    actual_model = values.get("model.default")
    mismatches = []
    if actual_engine != args.expect_engine:
        mismatches.append(f"context.engine={actual_engine!r} expected {args.expect_engine!r}")
    if actual_model != args.expect_model:
        mismatches.append(f"model.default={actual_model!r} expected {args.expect_model!r}")
    expected_sha = getattr(args, "expect_config_sha", None)
    if expected_sha is not None and config_sha256 != expected_sha:
        mismatches.append(f"config_sha256={config_sha256!r} expected {expected_sha!r}")
    if mismatches:
        raise ValueError("config assertion failed: " + "; ".join(mismatches))


def build_command(text: str, *, continue_session: bool) -> list[str]:
    """Build one hermes argv without invoking a shell."""
    if len(text) >= MAX_TURN_CHARS:
        raise ValueError(
            f"turn text is {len(text)} characters; must be shorter than {MAX_TURN_CHARS}"
        )
    command = ["hermes", "-z", text]
    if continue_session:
        command.append("--continue")
    return command


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _invoke(
    command: list[str],
    child_env: dict[str, str],
    timeout: float,
) -> tuple[str, str, bool]:
    """Run one turn and return stdout, stderr, and whether it timed out.

    ``subprocess.run`` kills and waits for the child before raising
    ``TimeoutExpired``.  Keeping that operation inside this helper makes the
    timeout boundary easy to mock while preserving fail-closed behavior.
    """
    try:
        current_sha = sha256_file(RUNTIME_CONFIG_PATH) if RUNTIME_CONFIG_PATH else None
        if RUNTIME_CONFIG_SHA is not None and current_sha != RUNTIME_CONFIG_SHA:
            raise ConfigDriftError(
                f"config.yaml changed mid-run: {current_sha} != {RUNTIME_CONFIG_SHA}"
            )
        completed = subprocess.run(
            command,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _text(exc.stdout), _text(exc.stderr), True
    return _text(completed.stdout), _text(completed.stderr), False


def _append_raw_log(path: Path, stdout: str, stderr: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if stdout:
            handle.write(stdout)
            if not stdout.endswith("\n"):
                handle.write("\n")
        if stderr:
            handle.write("[stderr]\n")
            handle.write(stderr)
            if not stderr.endswith("\n"):
                handle.write("\n")
        handle.flush()


def _input_identity(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def build_manifest(
    *,
    config_path: Path,
    config_sha256: str,
    values: dict[str, str | None],
    args: argparse.Namespace,
    raw_log: Path,
    results: Path,
    material: Path | None = None,
    probes: Path | None = None,
    canaries: Path | None = None,
) -> dict[str, Any]:
    input_files = {
        "material": _input_identity(material),
        "probes": _input_identity(probes),
        "canaries": _input_identity(canaries),
    }
    # The manifest records only safe command templates and environment names.
    # Turn text and environment values stay out of durable artifacts.
    return {
        "created_at": utc_now(),
        "driver": "drive_hermes_oneshot.py",
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
        "expected_config_sha256": getattr(args, "expect_config_sha", None),
        "turn_timeout_s": args.turn_timeout,
        "boot_timeout_s": args.boot_timeout,
        "quiet_seconds": args.quiet_seconds,
        "status_command": args.status_command,
        "probes_only": bool(args.probes_only),
        "raw_log": str(raw_log),
        "results": str(results),
        "argv": ["hermes", "-z", "<TEXT>"],
        "continue_argv": ["hermes", "-z", "<TEXT>", "--continue"],
        "env_names": ["HERMES_HOME", "LCM_ENABLE_SLASH_COMMAND"],
        "env": ["HERMES_HOME", "LCM_ENABLE_SLASH_COMMAND"],
        "input_files": input_files,
        "material_path": input_files["material"]["path"] if input_files["material"] else None,
        "material_sha256": input_files["material"]["sha256"] if input_files["material"] else None,
        "probes_path": input_files["probes"]["path"] if input_files["probes"] else None,
        "probes_sha256": input_files["probes"]["sha256"] if input_files["probes"] else None,
        "canaries_path": input_files["canaries"]["path"] if input_files["canaries"] else None,
        "canaries_sha256": input_files["canaries"]["sha256"] if input_files["canaries"] else None,
    }


def _contamination_hits(home: Path, values: list[str]) -> int:
    hits = 0
    for db in home.rglob("lcm.db"):
        blob = db.read_bytes()
        hits += sum(1 for value in values if value.encode() in blob)
    return hits


def _plan_rows(
    material_rows: list[Any],
    probe_rows: list[Any],
    *,
    probes_only: bool,
) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if not probes_only:
        rows.extend(("status" if is_status_row(row) else "material", row) for row in material_rows)
    rows.extend(("probe", row) for row in probe_rows)
    return rows


def _turn_text(kind: str, row: Any, status_command: str) -> str:
    if kind == "status":
        if isinstance(row, dict):
            return str(row.get("command") or status_command)
        return status_command
    return row_text(row)


def _display_command(command: list[str]) -> str:
    # Keep one command per line even when a turn contains newlines.
    return shlex.join(command).replace("\n", "\\n")


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def run(args: argparse.Namespace) -> int:
    hermes_home_arg = Path(args.hermes_home).expanduser()
    if hermes_home_arg.suffix in {".yaml", ".yml"}:
        # Hermes loads <HERMES_HOME>/config.yaml; accepting a bare config file
        # would validate one file while launching against whatever config.yaml
        # its parent happens to hold. Require the home DIRECTORY.
        sys.stderr.write(
            "[driver] --hermes-home must be a HERMES_HOME directory, not a "
            "config file path\n"
        )
        return 64
    config_path = (
        hermes_home_arg
        if hermes_home_arg.suffix in {".yaml", ".yml"}
        else hermes_home_arg / "config.yaml"
    )
    hermes_home = config_path.parent
    global RUNTIME_CONFIG_PATH, RUNTIME_CONFIG_SHA
    RUNTIME_CONFIG_PATH = config_path
    RUNTIME_CONFIG_SHA = sha256_file(config_path)
    if not config_path.exists():
        raise ValueError(f"Hermes config not found: {config_path}")
    values = read_config_values(config_path)
    config_sha = sha256_file(config_path)

    raw_log = Path(args.raw_log)
    results_path = Path(args.results) if args.results else raw_log.with_name("results.jsonl")
    manifest_path = Path(args.manifest) if args.manifest else raw_log.with_name("run.manifest.json")
    material_path = None if args.probes_only else Path(args.material)
    probes_path = Path(args.probes)
    canaries_path = Path(args.canaries) if args.canaries else None
    material_rows = [] if args.probes_only else load_jsonl(material_path, "material")
    probe_rows = load_jsonl(probes_path, "probe")
    if canaries_path is not None:
        load_canary_values(canaries_path)

    assert_config(values, args, config_sha)
    plan_rows = _plan_rows(material_rows, probe_rows, probes_only=args.probes_only)
    plan_commands: list[list[str]] = []
    for turn_index, (kind, row) in enumerate(plan_rows, 1):
        text = _turn_text(kind, row, args.status_command)
        # Status is explicitly a continuation command, even if it is the first
        # declared row.  Normal turns continue after the first invocation.
        plan_commands.append(
            build_command(text, continue_session=(turn_index > 1 or kind == "status"))
        )

    manifest = build_manifest(
        config_path=config_path,
        config_sha256=config_sha,
        values=values,
        args=args,
        raw_log=raw_log,
        results=results_path,
        material=material_path,
        probes=probes_path,
        canaries=canaries_path,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        for command in plan_commands:
            print(_display_command(command))
        return 0

    # Fail closed before the first material turn.  Probe-only runs do not
    # write material and therefore intentionally skip this guard.
    if args.canaries and not args.probes_only:
        canary_values = load_canary_values(canaries_path)
        hits = _contamination_hits(hermes_home, canary_values)
        if hits:
            sys.stderr.write(
                f"[driver] CONTAMINATION: {hits} canary-value hits in pre-run "
                f"store(s) under {hermes_home}\n"
            )
            return 71

    child_env = os.environ.copy()
    child_env["HERMES_HOME"] = str(hermes_home.resolve())
    child_env["LCM_ENABLE_SLASH_COMMAND"] = "true"

    probe_ordinal = 0
    for turn_index, (kind, row) in enumerate(plan_rows, 1):
        text = _turn_text(kind, row, args.status_command)
        command = plan_commands[turn_index - 1]
        started = time.monotonic()
        stdout, stderr, timed_out = _invoke(command, child_env, args.turn_timeout)
        _append_raw_log(raw_log, stdout, stderr)
        status_via: str | None = None

        if kind == "status":
            status_via = "slash"
            if not timed_out and "Unknown command" in (stdout + stderr):
                fallback_text = "Call the lcm_status tool and paste its full output verbatim."
                fallback_command = build_command(fallback_text, continue_session=True)
                fallback_stdout, fallback_stderr, fallback_timed_out = _invoke(
                    fallback_command,
                    child_env,
                    args.turn_timeout,
                )
                _append_raw_log(raw_log, fallback_stdout, fallback_stderr)
                stdout = fallback_stdout
                stderr = fallback_stderr
                timed_out = fallback_timed_out
                status_via = "tool"

        entry: dict[str, Any] = {
            "turn_index": turn_index,
            "kind": kind,
            "raw_answer": stdout,
            "wall_ms": round((time.monotonic() - started) * 1000, 3),
            "ts": utc_now(),
        }
        if kind == "probe":
            probe_ordinal += 1
            entry["probe_id"] = probe_id(row, probe_ordinal)
            entry["lcm_tool_fired"] = bool(LCM_TOOL_CALL_RE.search(stdout + stderr))
        if status_via is not None:
            entry["status_via"] = status_via
        if timed_out:
            entry["timed_out"] = True
        write_jsonl(results_path, entry)

        sys.stderr.write(
            f"[driver] sent {kind} turn {turn_index}/{len(plan_rows)}: {text[:60]!r}"
            + (" (timeout)" if timed_out else "")
            + "\n"
        )
        if timed_out:
            sys.stderr.write(
                f"[driver] ABORT: turn {turn_index} timed out; refusing to send "
                "further turns\n"
            )
            return BOOT_FAILURE_EXIT_CODE

    sys.stderr.write("[driver] done\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", help="material turns JSONL")
    parser.add_argument("--probes", required=True, help="probe turns JSONL")
    # Keep the base driver's option names/defaults for score/report tooling.
    parser.add_argument("--log", "--raw-log", dest="raw_log", default="raw.pty.log")
    parser.add_argument("--results")
    parser.add_argument("--manifest")
    parser.add_argument("--hermes-home", default="~/.hermes")
    parser.add_argument("--expect-engine", required=True)
    parser.add_argument("--expect-model", required=True)
    parser.add_argument("--expect-config-sha", help="expected config.yaml SHA-256")
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
        parser.exit(2, f"drive_hermes_oneshot.py: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
