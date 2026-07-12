"""Lossless recovery of read-tool results the host truncated unrecoverably.

When an oversized read-tool result overflows the per-turn budget and the host
cannot persist it to a sandbox file, it is replaced by an unrecoverable
``[Truncated: ... could not be saved to sandbox.]`` marker and the original
content is lost from LCM's durable store. For the ``read_file`` tool the source
path is still known (the originating tool call's ``path`` argument), so on the
local host LCM can re-read the file and preserve the full content.

This module holds only the pure, side-effect-light helpers (pairing, marker
identity, safe recovery). Persistence into the ``recovered_tool_content``
sidecar and the ingest wiring live in the store/engine so this stays testable
in isolation and free of reconciliation coupling: the parent ``messages`` row
always keeps the byte-identical marker, so replay/FTS/reconcile never change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .ingest_protection import (
    _is_unrecoverable_tool_truncation_marker,
    _read_regular_file_no_symlink,
)
from .message_content import normalize_content_value

READ_TOOL_NAMES = frozenset({"read_file"})
DEFAULT_READ_TOOL_RECOVERY_MAX_BYTES = 8 * 1024 * 1024


def is_recoverable_read_tool_marker(text: str | None) -> bool:
    """True when a tool message carries an unrecoverable truncation marker."""
    return _is_unrecoverable_tool_truncation_marker(text)


def marker_identity_sha(role: str, content: str, tool_call_id: str) -> str:
    """Stable digest of the stored marker row, used as the sidecar lookup key.

    Keyed on the exact bytes the parent ``messages`` row holds so an expand-time
    lookup finds the recovered content deterministically, with no dependence on
    the reconciliation identity machinery.
    """
    h = hashlib.sha256()
    h.update((role or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((content or "").encode("utf-8"))
    h.update(b"\x00")
    h.update((tool_call_id or "").encode("utf-8"))
    return h.hexdigest()


def _tool_call_function(tool_call: Any) -> tuple[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    if not isinstance(name, str):
        return None
    return name, function.get("arguments")


def _extract_path_argument(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return ""
    if not isinstance(arguments, dict):
        return ""
    path = arguments.get("path")
    return path if isinstance(path, str) else ""


def build_read_tool_call_path_map(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map each read-tool ``tool_call_id`` to its absolute ``path`` argument.

    Only calls to a known read tool with an absolute path are included; anything
    else is skipped so recovery never touches an arbitrary or relative path.
    """
    path_map: Dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            parsed = _tool_call_function(tool_call)
            if parsed is None:
                continue
            name, arguments = parsed
            if name not in READ_TOOL_NAMES:
                continue
            call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            if not isinstance(call_id, str) or not call_id:
                continue
            path = _extract_path_argument(arguments)
            if path and Path(path).is_absolute():
                path_map[call_id] = path
    return path_map


def recover_read_tool_file(
    path: str,
    *,
    max_bytes: int = DEFAULT_READ_TOOL_RECOVERY_MAX_BYTES,
) -> tuple[str, dict[str, int]] | None:
    """Re-read an absolute source path with the hardened no-symlink primitive.

    Returns ``(content, file_stat)`` or ``None`` when the path is unsafe,
    non-regular, oversized, a symlink, or changed mid-read (TOCTOU).
    """
    if not path or not Path(path).is_absolute():
        return None
    return _read_regular_file_no_symlink(Path(path), max_bytes=max_bytes)


def plan_read_tool_recovery(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return recovery candidates: tool messages with a marker and a paired path.

    Each candidate is ``{"index", "tool_call_id", "content", "path"}``. Pairing
    uses the read-tool call map built from the same message list, so recovery is
    scoped to the turn that produced the marker.
    """
    path_map = build_read_tool_call_path_map(messages)
    if not path_map:
        return []
    candidates: List[Dict[str, Any]] = []
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict) or str(msg.get("role") or "") != "tool":
            continue
        content = normalize_content_value(msg.get("content")) or ""
        if not is_recoverable_read_tool_marker(content):
            continue
        tool_call_id = str(msg.get("tool_call_id") or "")
        path = path_map.get(tool_call_id)
        if not path:
            continue
        candidates.append(
            {
                "index": index,
                "tool_call_id": tool_call_id,
                "content": content,
                "path": path,
            }
        )
    return candidates
