#!/usr/bin/env python3
"""Drive a Codex CLI session through material and probe turns.

The normal mode executes the exact commands registered by SPEC-COMPACTION-PROBE-A.
Use ``--dry-run`` to validate command construction without making a model call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Per-turn wall bound, matching the hermes drivers' --turn-timeout default.
TURN_TIMEOUT_S = 600
# Abort exit code shared across the pilot drivers (timeout/abort semantics).
ABORT_EXIT_CODE = 72


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            rows.append(row)
    return rows


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _json_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Codex may emit a human-readable informational line even with
            # --json. It is not an execution failure; stderr/returncode is.
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def _event_type(event: dict[str, Any]) -> str | None:
    direct = event.get("type")
    if isinstance(direct, str):
        return direct
    payload = event.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get("type"), str):
        return payload["type"]
    return None


def _thread_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        for node in _walk(event):
            if node.get("type") in {"thread.started", "thread_started"}:
                value = node.get("thread_id") or node.get("id")
                if isinstance(value, str) and value:
                    return value
    return None


def _agent_messages(events: list[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    for event in events:
        for node in _walk(event):
            node_type = node.get("type")
            if node_type in {"agent_message", "AgentMessage"}:
                text = node.get("text") or node.get("message")
                if isinstance(text, str):
                    messages.append(text)
    return messages


def _usage(events: list[dict[str, Any]]) -> dict[str, int]:
    latest = {"input": 0, "cached": 0, "output": 0}
    for event in events:
        for node in _walk(event):
            if node.get("type") != "token_count":
                continue
            info = node.get("info")
            if not isinstance(info, dict):
                continue
            last = info.get("last_token_usage")
            if not isinstance(last, dict):
                continue
            latest = {
                "input": int(last.get("input_tokens", 0) or 0),
                "cached": int(last.get("cached_input_tokens", 0) or 0),
                "output": int(last.get("output_tokens", 0) or 0),
            }
    return latest


def _compaction_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for event in events:
        for node in _walk(event):
            if node.get("type") == "context_compacted":
                found.append(node)
    return found


def _turn_model(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        for node in _walk(event):
            if node.get("type") == "turn_context":
                value = node.get("model")
                if isinstance(value, str) and value:
                    return value
    return None


def _command(codex_bin: Path, model: str, text: str, sid: str | None) -> list[str]:
    if sid:
        # -m is REQUIRED on resume too: without it, resumed turns silently
        # fall back to the account/config default model (measured: an R4-ref
        # run requested gpt-5.4 and got 69/70 turns of the gpt-5.6-sol
        # default — invisible whenever requested == default).
        return [
            str(codex_bin),
            "exec",
            "resume",
            sid,
            "--json",
            "-m",
            model,
            "--skip-git-repo-check",
            text,
        ]
    return [
        str(codex_bin),
        "exec",
        "--json",
        "-m",
        model,
        "--skip-git-repo-check",
        text,
    ]


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def drive(args: argparse.Namespace) -> int:
    codex_bin = args.codex_bin.expanduser().resolve()
    if not codex_bin.is_file():
        raise FileNotFoundError(f"codex binary not found: {codex_bin}")
    turns = _read_jsonl(args.material)
    probes = _read_jsonl(args.probes)
    for index, row in enumerate(turns, 1):
        if not isinstance(row.get("text"), str):
            raise ValueError(f"material turn {index} has no text")
    for index, row in enumerate(probes, 1):
        if not isinstance(row.get("id"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"probe {index} requires id and text")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    start_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "codex_bin": str(codex_bin),
        "codex_bin_sha256": _sha256(codex_bin),
        "model": args.model,
        "material_shas": {
            "turns.jsonl": _sha256(args.material),
            "probes.jsonl": _sha256(args.probes),
        },
        "sid": args.resume_sid,
        "start_ts": start_ts,
    }
    manifest_path = args.out_dir / "run.manifest.json"
    _write_manifest(manifest_path, manifest)

    commands = []
    sid = args.resume_sid
    for row in turns + probes:
        commands.append(_command(codex_bin, args.model, row["text"], sid))
        # A fresh session gets its SID from the first stdout event.  Dry-run
        # keeps the symbolic placeholder so every later command is visible.
        if sid is None:
            sid = "<THREAD_ID>"

    if args.dry_run:
        for command in commands:
            # Keep one planned argv per output line even when a material turn
            # itself contains newlines.  The escaped ``\\n`` is display-only;
            # normal execution passes the original text argument unchanged.
            print(shlex.join(command).replace("\n", "\\n"))
        return 0

    # The placeholder above is only for command planning.  Execute first turn
    # separately so the real thread id can be extracted before resuming.
    events_path = args.out_dir / "events.jsonl"
    results_path = args.out_dir / "results.jsonl"
    actual_sid = args.resume_sid
    with tempfile.TemporaryDirectory(prefix="compaction-probe-") as throwaway:
        for turn_index, row in enumerate(turns + probes, 1):
            command = _command(codex_bin, args.model, row["text"], actual_sid)
            try:
                completed = subprocess.run(
                    command,
                    cwd=throwaway,
                    text=True,
                    capture_output=True,
                    check=False,
                    # codex exec blocks at startup waiting for stdin EOF when it
                    # inherits an open pipe (measured: 17-min zero-CPU stall on
                    # the first live R1 turn, 2026-08-20; same class as the
                    # documented `< /dev/null` dispatch rule).
                    stdin=subprocess.DEVNULL,
                    # Per-turn wall bound (parity with the hermes drivers'
                    # --turn-timeout): a hung turn must abort the arm, not
                    # stall it unbounded.
                    timeout=TURN_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as exc:
                print(
                    f"Codex turn {turn_index} exceeded {TURN_TIMEOUT_S}s; aborting arm "
                    f"(partial stdout {len(exc.stdout or '')} chars)",
                    file=sys.stderr,
                )
                raise SystemExit(ABORT_EXIT_CODE) from exc
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"Codex turn {turn_index} failed with exit {completed.returncode}: {detail}"
                )
            events = _json_events(completed.stdout)
            # Fail-loud model identity per turn: the arm is void if any turn
            # served a different model than requested (the resume-fallback
            # class above).
            served = _turn_model(events)
            if served is not None and served != args.model:
                raise RuntimeError(
                    f"Codex turn {turn_index} served model {served!r}, "
                    f"requested {args.model!r} — arm invalid (model drift)"
                )
            if actual_sid is None:
                actual_sid = _thread_id(events)
                if not actual_sid:
                    raise RuntimeError(f"Codex turn {turn_index} did not emit thread.started")
                manifest["sid"] = actual_sid
                _write_manifest(manifest_path, manifest)

            usage = _usage(events)
            telemetry = {
                "turn_index": turn_index,
                "kind": "material" if turn_index <= len(turns) else "probe",
                "usage": usage,
                "context_compacted": _compaction_events(events),
            }
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(telemetry, sort_keys=True) + "\n")
                handle.flush()

            if turn_index > len(turns):
                answer = _agent_messages(events)[-1] if _agent_messages(events) else ""
                result = {
                    "probe_id": row["id"],
                    "raw_answer": answer,
                    "turn_index": turn_index,
                    "usage": usage,
                }
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex-bin", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume-sid")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return drive(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"drive_codex.py: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
