"""Stable identity helpers for assistant tool calls."""

from __future__ import annotations

import json
from typing import Any


def _json_has_duplicate_object_keys(value: str) -> bool:
    duplicate = False

    def detect_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in seen:
                duplicate = True
            seen.add(key)
            result[key] = item
        return result

    try:
        json.loads(value, object_pairs_hook=detect_pairs)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return duplicate


def canonicalize_tool_call_identity_value(value: Any) -> Any:
    """Canonicalize mappings recursively while preserving list order and values."""
    if isinstance(value, dict):
        return {
            key: canonicalize_tool_call_identity_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_tool_call_identity_value(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{" and not _json_has_duplicate_object_keys(value):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return value
            if isinstance(parsed, (dict, list)):
                canonical = canonicalize_tool_call_identity_value(parsed)
                return json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
        return value
    return value


def stable_tool_calls_identity(tool_calls: Any) -> str:
    """Serialize tool calls for durable storage and replay identity comparisons."""
    if not tool_calls:
        return ""
    try:
        canonical = canonicalize_tool_call_identity_value(tool_calls)
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return str(tool_calls)
