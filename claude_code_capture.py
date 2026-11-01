"""Claude Code transcript capture into an LCM-X store that Claude Code owns.

Design: docs/claude-code-adapter.md. The transcript file is the capture source; hooks
only say which transcript changed. This module holds projection v1: entry classes, the
rows of an entry and its content digest. The writer and worker are claude_code_store.py.
Nothing here compacts, blocks or injects context.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, NamedTuple

PROJECTION_VERSION = "claude-code-v1"
SOURCE = "claude-code"

# Entry classes. The audit counts every transcript entry by class, so nothing is
# dropped silently: an unknown type keeps its raw line and is counted.
PROJECTED = "projected"      # becomes message rows; raw line kept in the ledger
RAW_ONLY = "raw_only"        # no rows; protected raw line kept in the ledger
SHARED_RAW = "shared_raw"    # no rows; protected raw line kept once per content digest
UNKNOWN = "unknown_type"     # no rows; raw line kept and counted

PROJECTED_TYPES = frozenset({"user", "assistant"})
# Request snapshots repeat content-addressed blobs (system prompt, tools, request bodies) many
# times per session; their unique content is kept once, never reduced to a digest.
SHARED_RAW_TYPES = frozenset({"api-request", "api-request-blob", "api-request-shape"})
RAW_ONLY_TYPES = frozenset({
    "system", "attachment", "summary", "queue-operation", "last-prompt", "mode", "custom-title",
    "agent-name", "cost-state", "pr-link", "file-history-snapshot", "file-history-delta", "tools-used",
    # Desktop bookkeeping seen in real transcripts. Types that may carry content (``result``,
    # ``started``) stay unknown, so the audit shows them until a projection version handles them.
    "atis-latch", "bridge-session", "frame-link", "history-suppression",
    "artifact-autoreact-ledger", "artifact-comment-monitor",
})
# Envelope fields that differ between an entry and its copy in a forked session.
COPY_ENVELOPE = frozenset({"sessionId", "promptId"})
# User entries Claude Code writes itself (command output, compaction summaries) become system rows.
SYSTEM_FLAGS = ("isMeta", "isCompactSummary", "isVisibleInTranscriptOnly")
# Claude Code saves a large tool output to <session>/tool-results/ and keeps a preview:
# "<persisted-output>\nOutput too large (30.7KB). Full output saved to: <path>\n\nPreview ...".
PERSISTED_TAG = "<persisted-output>"
SAVED_TO = re.compile(r"Full output saved to:\s*(?P<path>[^\r\n]+)")
MAX_RECOVERED_OUTPUT = 64 * 1024 * 1024  # as Hermes's persisted-output recovery


def classify_entry(entry: Mapping[str, Any]) -> str:
    """The projection-v1 class of one decoded transcript entry."""
    kind = entry.get("type")
    if kind in PROJECTED_TYPES and isinstance(entry.get("message"), Mapping):
        return PROJECTED
    if kind in SHARED_RAW_TYPES:
        return SHARED_RAW
    if kind in RAW_ONLY_TYPES or kind in PROJECTED_TYPES:
        return RAW_ONLY
    return UNKNOWN


def row_session_id(session_id: str, agent_id: str | None = None) -> str:
    """The rows' ``session_id``. A subagent's entries carry the parent's ``sessionId``; its
    rows get their own id, so the parent's chronology stays clean and the child has its own
    ``session_lineage`` link."""
    if not session_id:
        raise ValueError("session_id is required")
    return f"{session_id}/agent:{agent_id}" if agent_id else session_id


def lineage_key(session_id: str, agent_id: str | None = None) -> str:
    """``host_uid_bindings`` lineage: the session, or a subagent inside it."""
    return f"{SOURCE}:{row_session_id(session_id, agent_id)}"


class Projection(NamedTuple):
    rows: list[tuple[str, dict[str, Any]]]  # (uid, message for MessageStore.append_batch)
    files: list[str]  # tool-results file names whose full text a row now holds
    problems: list[str]  # persisted outputs kept as their preview, by reason


def session_dir(transcript: Path) -> Path:
    """``<session>/`` beside ``<session>.jsonl``; a subagent transcript lives inside it."""
    return transcript.parent.parent if transcript.parent.name == "subagents" else transcript.with_suffix("")


def project_entry(entry: Mapping[str, Any], *, folder: Path | None = None) -> Projection:
    """Message rows for one projected entry; uid ``uuid``, or ``uuid#n`` when it gives several.

    User text and images give one ``user`` row; each ``tool_result`` part gives one ``tool``
    row with ``tool_call_id``, holding the full text of the ``tool-results`` file under
    ``folder`` (the session folder) when Claude Code saved the output there. An assistant
    entry gives one row with its text and ``tool_use`` parts, none if it holds only thinking.
    Meta entries, compact summaries and API-error entries become ``system`` rows. Thinking
    and other parts stay in the ledger's raw line only.
    """
    uid = entry.get("uuid")
    message = entry.get("message")
    if classify_entry(entry) != PROJECTED or not isinstance(uid, str) or not uid:
        return Projection([], [], [])
    stamp = {"timestamp": entry["timestamp"]} if isinstance(entry.get("timestamp"), str) else {}
    parts = message.get("content")
    parts = [{"type": "text", "text": parts}] if isinstance(parts, str) else [
        part for part in parts or [] if isinstance(part, Mapping)]
    rows: list[dict[str, Any]] = []
    files: list[str] = []
    problems: list[str] = []
    if entry.get("type") == "assistant":
        text = _joined_text(parts)
        if entry.get("isApiErrorMessage"):
            rows.append({"role": "system", "content": text, **stamp})
        else:
            calls = [{"id": part.get("id"), "type": "function", "function": {
                "name": part.get("name"), "arguments": json.dumps(part.get("input", {}), ensure_ascii=False)}}
                for part in parts if part.get("type") == "tool_use"]
            if text or calls:
                rows.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {}),
                             **stamp})
    else:
        saved = _saved_output_path(entry)
        for part in parts:
            if part.get("type") != "tool_result":
                continue
            content = _result_content(part.get("content"))
            if folder is not None and isinstance(content, str) and PERSISTED_TAG in content:
                content, file_name, problem = _recover_saved_output(content, saved, folder)
                files += [file_name] if file_name else []
                problems += [problem] if problem else []
            rows.append({"role": "tool", "content": content, "tool_call_id": part.get("tool_use_id"), **stamp})
        rest = [part for part in parts if part.get("type") in ("text", "image")]
        if rest:
            role = "system" if any(entry.get(flag) for flag in SYSTEM_FLAGS) else "user"
            content = _joined_text(rest) if all(part.get("type") == "text" for part in rest) else rest
            rows.append({"role": role, "content": content, **stamp})
    if len(rows) == 1:
        return Projection([(uid, rows[0])], files, problems)
    return Projection([(f"{uid}#{index}", row) for index, row in enumerate(rows)], files, problems)


def content_digest(entry: Mapping[str, Any]) -> str:
    """Digest of an entry without the envelope fields that change when Claude Code copies
    it into another session (``sessionId`` and ``promptId``, observed on ``--fork-session``)."""
    kept = {key: value for key, value in entry.items() if key not in COPY_ENVELOPE}
    canonical = json.dumps(kept, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()


def _joined_text(parts: list[Mapping[str, Any]]) -> str:
    return "\n\n".join(part["text"] for part in parts
                       if part.get("type") == "text" and isinstance(part.get("text"), str))


def _result_content(content: Any) -> Any:
    """A tool result's text, or its parts when it holds an image."""
    if isinstance(content, list):
        parts = [part for part in content if isinstance(part, Mapping)]
        if all(part.get("type") == "text" for part in parts):
            return _joined_text(parts)
        return parts
    return content if isinstance(content, str) else ""


def _saved_output_path(entry: Mapping[str, Any]) -> str | None:
    result = entry.get("toolUseResult")
    path = result.get("persistedOutputPath") if isinstance(result, Mapping) else None
    return path if isinstance(path, str) and path else None


def _recover_saved_output(preview: str, saved: str | None, folder: Path) -> tuple[str, str | None, str | None]:
    """The saved file's full text when it lies in ``<folder>/tool-results``, else the preview."""
    match = SAVED_TO.search(preview)  # the marker names its own file; toolUseResult is per entry
    named = (match.group("path").strip() if match else None) or saved
    if not named:
        return preview, None, "persisted_output_unnamed"
    results = (folder / "tool-results").resolve()
    try:
        path = Path(named).expanduser()
        if path.is_symlink() or not path.resolve().is_relative_to(results):
            return preview, None, "persisted_output_outside_session"
        with open(path, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return preview, None, "persisted_output_not_a_file"
            data = handle.read(MAX_RECOVERED_OUTPUT + 1)
    except OSError:
        return preview, None, "persisted_output_missing"
    if len(data) > MAX_RECOVERED_OUTPUT:
        return preview, None, "persisted_output_too_large"
    try:
        return data.decode("utf-8"), path.name, None
    except UnicodeDecodeError:
        return preview, None, "persisted_output_binary"
