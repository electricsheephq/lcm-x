"""Transcript readers for the LCM-X dev-agent hub: Claude Code and Codex session files, as the nodes mirror them.

A transcript is an append-only JSONL file; one file is one session (a Claude Code subagent has its own file and its
own session). The hub reads complete lines after a byte cursor and projects each entry to LCM-X rows (OpenAI-style
dicts). Every line, projected or not, is kept raw in the hub ledger, so nothing is lost in the projection.

Identity of an entry: (session, entry id). Claude Code: the entry's ``uuid`` plus its content digest (upstream
projection v1, ``claude_code_capture``). Codex: the line's ``ordinal`` (its position in the rollout).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

CLAUDE = "claude-code"
CODEX = "codex"


@dataclass
class Entry:
    """One transcript line: its identity, its raw bytes and the rows it gives (may be none)."""

    key: str
    raw: bytes
    rows: list[dict] = field(default_factory=list)
    kind: str = ""


@dataclass
class Transcript:
    path: Path
    harness: str
    node: str
    session: str  # LCM-X session id: "cc:<node>:<session>[/agent:<id>]" or "codex:<node>:<session>"
    folder: Optional[Path] = None  # Claude Code session folder (tool-results) for saved outputs


def find(mirror: Path) -> Iterator[Transcript]:
    """Every transcript under ``<mirror>/<node>/{claude,codex}``."""
    if not mirror.is_dir():
        return
    for node_dir in sorted(p for p in mirror.iterdir() if p.is_dir()):
        node = node_dir.name
        claude = node_dir / "claude" / "projects"
        if claude.is_dir():
            for path in sorted(claude.rglob("*.jsonl")):
                if path.parent.name == "subagents":  # <session>/subagents/agent-<id>.jsonl
                    session_dir = path.parent.parent
                    session = f"cc:{node}:{session_dir.name}/agent:{path.stem.removeprefix('agent-')}"
                    yield Transcript(path, CLAUDE, node, session, session_dir)
                elif path.parent.parent == claude:  # <project>/<session>.jsonl
                    yield Transcript(path, CLAUDE, node, f"cc:{node}:{path.stem}", path.with_suffix(""))
        codex = node_dir / "codex" / "sessions"
        if codex.is_dir():
            for path in sorted(codex.rglob("rollout-*.jsonl")):
                yield Transcript(path, CODEX, node, f"codex:{node}:{codex_session_id(path)}")


def codex_session_id(path: Path) -> str:
    """The session uuid at the end of ``rollout-<time>-<uuid>.jsonl``, else the file stem."""
    parts = path.stem.split("-")
    return "-".join(parts[-5:]) if len(parts) >= 6 else path.stem


def read_lines(path: Path, offset: int, limit_bytes: int = 32 * 1024 * 1024) -> tuple[list[bytes], int]:
    """Complete lines after ``offset`` (a line still being written stays for the next read), and the new offset."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read(limit_bytes)
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    chunk = data[: end + 1]
    return [line for line in chunk.split(b"\n")[:-1]], offset + len(chunk)


def first_line_digest(path: Path) -> str:
    """Digest of the first line: a different first line means the file was replaced, not appended to."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.readline()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# -- Claude Code ---------------------------------------------------------------------------------------

def claude_entry(transcript: Transcript, raw: bytes, index: int) -> Entry:
    """Upstream's projection v1 (``claude_code_capture.project_entry``): user, assistant, tool and system rows, saved
    tool outputs read back from ``tool-results``. The key is the entry uuid plus its content digest, so a revised entry
    is a new entry and a copied one is the same."""
    from hermes_lcm.claude_code_capture import PROJECTED, classify_entry, content_digest, project_entry

    try:
        entry = json.loads(raw)
    except ValueError:
        return Entry(f"line:{index}:{hashlib.sha256(raw).hexdigest()[:16]}", raw, [], "unparsed")
    if not isinstance(entry, dict):
        return Entry(f"line:{index}:{hashlib.sha256(raw).hexdigest()[:16]}", raw, [], "unparsed")
    kind = classify_entry(entry)
    uid = entry.get("uuid") if isinstance(entry.get("uuid"), str) else None
    key = f"{uid}:{content_digest(entry)[:16]}" if uid else f"line:{index}:{hashlib.sha256(raw).hexdigest()[:16]}"
    rows: list[dict] = []
    if kind == PROJECTED and uid:
        rows = [row for _uid, row in project_entry(entry, folder=transcript.folder).rows]
    return Entry(key, raw, rows, str(entry.get("type") or kind))


# -- Codex ---------------------------------------------------------------------------------------------

def _codex_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    pieces = []
    for part in content or []:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                pieces.append(text)
            elif part.get("type"):
                pieces.append(f"[{part['type']}]")
    return "\n\n".join(pieces)


def _codex_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict) and isinstance(output.get("output"), str):
        return output["output"]
    if isinstance(output, list):
        return _codex_text(output)
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def codex_entry(raw: bytes, index: int) -> Entry:
    """Codex rollout projection v1. ``response_item`` messages become user, assistant or system rows (developer
    messages are system); every ``*_call`` becomes an assistant row with one tool call, every ``*_call_output`` a tool
    row. Reasoning, tool lists, events and session metadata stay raw in the ledger. The key is the line ordinal."""
    try:
        line = json.loads(raw)
    except ValueError:
        return Entry(f"line:{index}", raw, [], "unparsed")
    if not isinstance(line, dict):
        return Entry(f"line:{index}", raw, [], "unparsed")
    ordinal = line.get("ordinal")
    key = f"ord:{ordinal}" if isinstance(ordinal, int) else f"line:{index}:{_digest(line)[:16]}"
    stamp = {"timestamp": line["timestamp"]} if isinstance(line.get("timestamp"), str) else {}
    payload = line.get("payload") if isinstance(line.get("payload"), dict) else {}
    kind = str(line.get("type") or "")
    if kind != "response_item":
        return Entry(key, raw, [], kind)
    item = str(payload.get("type") or "")
    rows: list[dict] = []
    if item == "message":
        role = payload.get("role")
        text = _codex_text(payload.get("content"))
        if text:
            if role == "assistant":
                rows = [{"role": "assistant", "content": text, **stamp}]
            elif role == "user":
                rows = [{"role": "user", "content": text, **stamp}]
            else:  # developer and system: instructions the harness injects
                rows = [{"role": "system", "content": text, **stamp}]
    elif item.endswith("_call_output"):
        rows = [{"role": "tool", "content": _codex_output(payload.get("output")), "tool_call_id": payload.get("call_id"),
                 **stamp}]
    elif item.endswith("_call"):
        arguments = payload.get("arguments", payload.get("input", payload.get("action")))
        rows = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": payload.get("call_id"), "type": "function", "function": {
                "name": payload.get("name") or item.removesuffix("_call"),
                "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)}}],
            **stamp}]
    return Entry(key, raw, rows, f"response_item:{item}")


def entry_of(transcript: Transcript, raw: bytes, index: int) -> Entry:
    return claude_entry(transcript, raw, index) if transcript.harness == CLAUDE else codex_entry(raw, index)
