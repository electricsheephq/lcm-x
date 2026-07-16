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


def _canonicalize_plain_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonicalize_plain_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonicalize_plain_value(item) for item in value]
    return value


def _canonicalize_function_arguments(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{" and not _json_has_duplicate_object_keys(value):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return value
            if isinstance(parsed, (dict, list)):
                canonical = _canonicalize_plain_value(parsed)
                return json.dumps(
                    canonical,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
        return value
    return _canonicalize_plain_value(value)


def _canonicalize_tool_call(value: Any) -> Any:
    if not isinstance(value, dict):
        return _canonicalize_plain_value(value)
    canonical = _canonicalize_plain_value(value)
    function = value.get("function")
    if isinstance(function, dict) and "arguments" in function:
        canonical["function"]["arguments"] = _canonicalize_function_arguments(
            function["arguments"]
        )
    return canonical


def canonicalize_tool_call_identity_value(value: Any) -> Any:
    """Canonicalize tool-call structure and direct function arguments only."""
    if isinstance(value, list):
        return [_canonicalize_tool_call(item) for item in value]
    return _canonicalize_tool_call(value)


def stable_tool_calls_identity(tool_calls: Any) -> str:
    """Serialize tool calls for replay identity comparisons."""
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


def stable_serialized_tool_calls_identity(tool_calls: str | None) -> str:
    """Parse a durable legacy value before deriving its comparison identity."""
    if not tool_calls:
        return ""
    try:
        parsed = json.loads(tool_calls)
    except (TypeError, ValueError, json.JSONDecodeError):
        return tool_calls
    return stable_tool_calls_identity(parsed)
