#!/usr/bin/env python3
"""Drive one threaded Hermes session through the ACP stdio transport.

R2s keeps material and probe turns in one ACP process/session.  R3 closes the
material process, starts a fresh ``hermes acp`` process, and creates a new ACP
session for probes while retaining the same HERMES_HOME and LCM database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TURN_TIMEOUT = 600.0
TIMEOUT_EXIT_CODE = 72
CONFIG_DRIFT_EXIT_CODE = 73
ACP_SHUTDOWN_SECONDS = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DriverError(RuntimeError):
    """An ACP protocol, process, or artifact failure."""


class ConfigDriftError(DriverError):
    """config.yaml changed (or vanished) while a run was in flight."""


class TurnTimeout(DriverError):
    """A per-turn ACP response timeout with the required abort status."""

    def __init__(self, message: str, partial_answer: str = "") -> None:
        self.partial_answer = partial_answer
        super().__init__(message)


class JsonRpcError(DriverError):
    def __init__(self, method: str, error: Any) -> None:
        self.method = method
        self.error = error
        super().__init__(
            f"{method} returned JSON-RPC error: {json.dumps(error, sort_keys=True)}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_inline_comment(value: str) -> str:
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
        key = key.strip().strip("'\"")
        if not key:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if raw_value.strip():
            key_path = ".".join(part for _, part in stack + [(indent, key)])
            values[key_path] = _yaml_scalar(raw_value)
        else:
            stack.append((indent, key))
    return {
        "context.engine": values.get("context.engine"),
        "model.default": values.get("model.default"),
    }


def assert_config(
    config_path: Path,
    initial_sha: str,
    values: dict[str, str | None],
    args: argparse.Namespace,
) -> None:
    try:
        current_sha = sha256_file(config_path)
    except OSError as exc:
        raise ConfigDriftError(f"config.yaml unreadable: {exc}") from exc
    if current_sha != initial_sha:
        raise ConfigDriftError(
            f"config.yaml changed mid-run: {current_sha} != {initial_sha}"
        )
    mismatches: list[str] = []
    if values.get("context.engine") != args.expect_engine:
        mismatches.append(
            f"context.engine={values.get('context.engine')!r} expected {args.expect_engine!r}"
        )
    if values.get("model.default") != args.expect_model:
        mismatches.append(
            f"model.default={values.get('model.default')!r} expected {args.expect_model!r}"
        )
    if args.expect_config_sha is not None and current_sha != args.expect_config_sha:
        mismatches.append(
            f"config_sha256={current_sha!r} expected {args.expect_config_sha!r}"
        )
    if mismatches:
        raise ValueError("config assertion failed: " + "; ".join(mismatches))


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
    for key in ("text", "prompt", "content", "message", "question"):
        if key in row:
            value = row[key]
            if isinstance(value, str):
                return value
            if value is not None:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise ValueError(f"turn row has no text field: {row!r}")


def probe_id(row: Any, index: int) -> str:
    if isinstance(row, dict):
        for key in ("probe_id", "id", "name"):
            if row.get(key) is not None:
                return str(row[key])
    return f"probe-{index}"


def build_plan_phases(
    material_rows: list[Any],
    probe_rows: list[Any],
    *,
    restart_before_probes: bool,
) -> list[list[tuple[str, Any]]]:
    material = [("material", row) for row in material_rows]
    probes = [("probe", row) for row in probe_rows]
    if restart_before_probes and material and probes:
        return [material, probes]
    combined = material + probes
    return [combined] if combined else []


def jsonrpc_request(request_id: int, method: str, params: dict[str, Any]) -> bytes:
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def agent_message_chunk(message: dict[str, Any]) -> str | None:
    if message.get("method") != "session/update":
        return None
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    update = params.get("update")
    if not isinstance(update, dict):
        return None
    kind = update.get("sessionUpdate", update.get("session_update"))
    if kind != "agent_message_chunk":
        return None
    content = update.get("content")
    if not isinstance(content, dict) or not isinstance(content.get("text"), str):
        return None
    return content["text"]


class StdioJsonRpc:
    """Synchronous newline-delimited JSON-RPC client for ``hermes acp``."""

    def __init__(self, home: Path) -> None:
        child_env = os.environ.copy()
        child_env["HERMES_HOME"] = str(home)
        child_env["LCM_ENABLE_SLASH_COMMAND"] = "true"
        child_env["HERMES_ACP_SKIP_CONFIGURED_MCP"] = "1"
        child_env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            ["hermes", "acp"],
            cwd=str(home),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self._kill_group(signal.SIGKILL)
            raise DriverError("failed to open Hermes ACP stdio pipes")
        self._selector = selectors.DefaultSelector()
        self._selector.register(self.process.stdout, selectors.EVENT_READ)
        self._next_id = 1
        self._stdin_closed = False
        self.stderr_bytes = 0
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self.process.stderr,),
            name="acp-driver-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, stream: Any) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            self.stderr_bytes += len(chunk)

    def _kill_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def send(self, method: str, params: dict[str, Any]) -> int:
        if self._stdin_closed or self.process.stdin is None:
            raise DriverError(f"cannot send {method}: ACP stdin is closed")
        request_id = self._next_id
        self._next_id += 1
        try:
            self.process.stdin.write(jsonrpc_request(request_id, method, params))
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DriverError(f"failed to send {method}: {exc}") from exc
        return request_id

    def _send_error(self, request_id: Any) -> None:
        if self._stdin_closed or self.process.stdin is None:
            return
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": "ACP driver has no client handler"},
        }
        try:
            self.process.stdin.write(
                (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def _read_message(self, timeout_seconds: float) -> dict[str, Any]:
        if not self._selector.select(max(timeout_seconds, 0.0)):
            raise TurnTimeout(f"ACP response timeout after {timeout_seconds:.0f}s")
        if self.process.stdout is None:
            raise DriverError("ACP stdout is unavailable")
        line = self.process.stdout.readline()
        if not line:
            raise DriverError(f"ACP stdout closed (returncode={self.process.poll()})")
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DriverError(f"invalid JSON-RPC frame: {line[:200]!r}") from exc
        if not isinstance(message, dict):
            raise DriverError(f"invalid JSON-RPC object: {message!r}")
        return message

    def wait_for_response(
        self,
        method: str,
        request_id: int,
        timeout_seconds: float,
        answer_chunks: list[str] | None = None,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_seconds
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TurnTimeout(f"{method} response timeout after {timeout_seconds:.0f}s")
                message = self._read_message(remaining)
                if message.get("method"):
                    if "id" in message:
                        self._send_error(message.get("id"))
                    else:
                        chunk = agent_message_chunk(message)
                        if chunk is not None and answer_chunks is not None:
                            answer_chunks.append(chunk)
                    continue
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise JsonRpcError(method, message.get("error"))
                result = message.get("result")
                return result if isinstance(result, dict) else None
        except TurnTimeout as exc:
            partial = "" if answer_chunks is None else "".join(answer_chunks).strip()
            raise TurnTimeout(str(exc), partial) from exc

    def initialize(self, home: Path, timeout_seconds: float) -> str:
        request_id = self.send(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "drive_hermes_acp", "version": "0.1.0"},
            },
        )
        self.wait_for_response("initialize", request_id, timeout_seconds)
        request_id = self.send("session/new", {"cwd": str(home), "mcpServers": []})
        result = self.wait_for_response("session/new", request_id, timeout_seconds)
        if not result or not isinstance(result.get("sessionId"), str):
            raise DriverError(f"session/new returned no sessionId: {result!r}")
        return result["sessionId"]

    def prompt(self, session_id: str, text: str, timeout_seconds: float) -> str:
        chunks: list[str] = []
        request_id = self.send(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": text}],
            },
        )
        response = self.wait_for_response(
            "session/prompt", request_id, timeout_seconds, chunks
        )
        if response and response.get("stopReason") not in (None, "end_turn"):
            raise DriverError(f"session/prompt stopReason={response.get('stopReason')!r}")
        return "".join(chunks).strip()

    def close(self) -> None:
        if not self._stdin_closed and self.process.stdin is not None:
            self._stdin_closed = True
            try:
                self.process.stdin.close()
            except OSError:
                pass
        self._kill_group(signal.SIGTERM)
        try:
            self.process.wait(timeout=ACP_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            self._kill_group(signal.SIGKILL)
            self.process.wait(timeout=ACP_SHUTDOWN_SECONDS)
        finally:
            # The process leader may have exited before a bridge child.  A
            # final group kill closes that leak class without touching peers.
            self._kill_group(signal.SIGKILL)
            self._selector.close()
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


def _input_identity(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": sha256_file(path)}


def build_manifest(
    *,
    config_path: Path,
    config_sha: str,
    values: dict[str, str | None],
    args: argparse.Namespace,
    raw_log: Path,
    results: Path,
    material: Path,
    probes: Path,
    canaries: Path | None,
) -> dict[str, Any]:
    input_files = {
        "material": _input_identity(material),
        "probes": _input_identity(probes),
        "canaries": _input_identity(canaries),
    }
    return {
        "created_at": utc_now(),
        "driver": "drive_hermes_acp.py",
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_sha": config_sha,
        "context": {"engine": values.get("context.engine")},
        "model": {"default": values.get("model.default")},
        "context.engine": values.get("context.engine"),
        "model.default": values.get("model.default"),
        "context_engine": values.get("context.engine"),
        "model_default": values.get("model.default"),
        "expected_engine": args.expect_engine,
        "expected_model": args.expect_model,
        "expected_config_sha256": args.expect_config_sha,
        "turn_timeout_s": args.turn_timeout,
        "boot_timeout_s": None,
        "quiet_seconds": None,
        "status_command": None,
        "probes_only": False,
        "raw_log": str(raw_log),
        "results": str(results),
        "argv": ["hermes", "acp"],
        "continue_argv": ["hermes", "acp"],
        "env_names": [
            "HERMES_HOME",
            "LCM_ENABLE_SLASH_COMMAND",
            "HERMES_ACP_SKIP_CONFIGURED_MCP",
        ],
        "env": [
            "HERMES_HOME",
            "LCM_ENABLE_SLASH_COMMAND",
            "HERMES_ACP_SKIP_CONFIGURED_MCP",
        ],
        "input_files": input_files,
        "material_path": input_files["material"]["path"],
        "material_sha256": input_files["material"]["sha256"],
        "probes_path": input_files["probes"]["path"],
        "probes_sha256": input_files["probes"]["sha256"],
        "canaries_path": input_files["canaries"]["path"] if input_files["canaries"] else None,
        "canaries_sha256": input_files["canaries"]["sha256"] if input_files["canaries"] else None,
        "transport": "acp",
        "restart_before_probes": bool(args.restart_before_probes),
        "acp_sessions": [],
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def append_raw_answer(path: Path, answer: str) -> None:
    if not answer:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(answer)
        if not answer.endswith("\n"):
            handle.write("\n")
        handle.flush()


def contamination_hits(home: Path, values: list[str]) -> int:
    hits = 0
    for db in home.rglob("lcm.db"):
        blob = db.read_bytes()
        hits += sum(1 for value in values if value.encode() in blob)
    return hits


def _read_count(db_path: Path, table: str) -> int:
    if not db_path.is_file():
        raise DriverError(f"database was not created: {db_path}")
    uri = f"file:{db_path.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=5) as connection:
            row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    except sqlite3.Error as exc:
        raise DriverError(f"cannot count {table} in {db_path}: {exc}") from exc
    if row is None:
        raise DriverError(f"count query returned no row for {table} in {db_path}")
    return int(row[0])


def diagnostics(home: Path) -> dict[str, Any]:
    return {
        "ts": utc_now(),
        "summary_nodes": _read_count(home / "lcm.db", "summary_nodes"),
        "messages": _read_count(home / "lcm.db", "messages"),
        "sessions": _read_count(home / "state.db", "sessions"),
    }


def run(args: argparse.Namespace) -> int:
    home = Path(args.hermes_home).expanduser()
    if home.suffix in {".yaml", ".yml"} or not home.is_dir():
        sys.stderr.write("[driver] --hermes-home must be an existing HERMES_HOME directory\n")
        return 64
    home = home.resolve()
    config_path = home / "config.yaml"
    if not config_path.is_file():
        sys.stderr.write(f"[driver] config not found: {config_path}\n")
        return 64

    material_path = Path(args.material)
    probes_path = Path(args.probes)
    raw_log = Path(args.raw_log)
    results_path = Path(args.results) if args.results else raw_log.with_name("results.jsonl")
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else raw_log.with_name("run.manifest.json")
    )
    canaries_path = Path(args.canaries) if args.canaries else None
    material_rows = load_jsonl(material_path, "material")
    probe_rows = load_jsonl(probes_path, "probe")
    canary_values = load_canary_values(canaries_path) if canaries_path else []
    phases = build_plan_phases(
        material_rows,
        probe_rows,
        restart_before_probes=args.restart_before_probes,
    )
    if not phases:
        raise ValueError("material and probes contain no turns")

    values = read_config_values(config_path)
    config_sha = sha256_file(config_path)
    assert_config(config_path, config_sha, values, args)
    manifest = build_manifest(
        config_path=config_path,
        config_sha=config_sha,
        values=values,
        args=args,
        raw_log=raw_log,
        results=results_path,
        material=material_path,
        probes=probes_path,
        canaries=canaries_path,
    )
    write_manifest(manifest_path, manifest)

    if canary_values:
        hits = contamination_hits(home, canary_values)
        if hits:
            sys.stderr.write(
                f"[driver] CONTAMINATION: {hits} canary-value hits in pre-run stores under {home}\n"
            )
            return 71

    raw_log.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log.write_text("", encoding="utf-8")
    results_path.write_text("", encoding="utf-8")

    turn_index = 0
    probe_ordinal = 0
    config_checked_before_probes = False
    for phase in phases:
        client: StdioJsonRpc | None = None
        try:
            if phase[0][0] == "probe" and not config_checked_before_probes:
                assert_config(config_path, config_sha, values, args)
                config_checked_before_probes = True
            client = StdioJsonRpc(home)
            session_id = client.initialize(home, args.turn_timeout)
            manifest["acp_sessions"].append(session_id)
            write_manifest(manifest_path, manifest)
            for kind, row in phase:
                if kind == "probe" and not config_checked_before_probes:
                    assert_config(config_path, config_sha, values, args)
                    config_checked_before_probes = True
                turn_index += 1
                text = row_text(row)
                started = time.monotonic()
                timed_out = False
                try:
                    answer = client.prompt(session_id, text, args.turn_timeout)
                except TurnTimeout as exc:
                    answer = exc.partial_answer
                    timed_out = True
                append_raw_answer(raw_log, answer)
                entry: dict[str, Any] = {
                    "turn_index": turn_index,
                    "kind": kind,
                    "ts": utc_now(),
                    "wall_ms": round((time.monotonic() - started) * 1000, 3),
                    "raw_answer": answer,
                    "timed_out": timed_out,
                }
                if kind == "probe":
                    probe_ordinal += 1
                    entry["probe_id"] = probe_id(row, probe_ordinal)
                write_jsonl(results_path, entry)
                sys.stderr.write(
                    f"[driver] sent {kind} turn {turn_index}: {text[:60]!r}"
                    + (" (timeout)" if timed_out else "")
                    + "\n"
                )
                if timed_out:
                    sys.stderr.write(
                        f"[driver] ABORT: turn {turn_index} timed out; refusing further turns\n"
                    )
                    return TIMEOUT_EXIT_CODE
        finally:
            if client is not None:
                client.close()

    assert_config(config_path, config_sha, values, args)
    manifest["diagnostics"] = diagnostics(home)
    write_manifest(manifest_path, manifest)
    sys.stderr.write("[driver] done\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", default="~/.hermes")
    parser.add_argument("--material", required=True, help="material turns JSONL")
    parser.add_argument("--probes", required=True, help="probe turns JSONL")
    parser.add_argument("--log", "--raw-log", dest="raw_log", default="raw.pty.log")
    parser.add_argument("--results")
    parser.add_argument("--manifest")
    parser.add_argument("--expect-engine", required=True)
    parser.add_argument("--expect-model", required=True)
    parser.add_argument("--expect-config-sha", help="expected config.yaml SHA-256")
    parser.add_argument("--canaries")
    parser.add_argument("--turn-timeout", type=float, default=DEFAULT_TURN_TIMEOUT)
    parser.add_argument("--restart-before-probes", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return run(args)
    except ConfigDriftError as exc:
        sys.stderr.write(f"drive_hermes_acp.py: {exc}\n")
        return CONFIG_DRIFT_EXIT_CODE
    except (DriverError, OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"drive_hermes_acp.py: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
