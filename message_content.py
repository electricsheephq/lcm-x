"""Message content normalization helpers.

Hermes/OpenAI-format messages may carry ``content`` as plain text or as
structured content parts (for example text + image blocks). LCM persists and
accounts for message content as text, so all write/matching/token paths should
use deliberate normalization.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}
_MULTIMODAL_IMAGE_PART_TYPES = {"image", "image_url"}
_SURROGATES = re.compile(r"[\ud800-\udfff]")


def canonical_text(text: str) -> str:
    """Match Hermes' one-for-one replacement of surrogate code points."""
    if not text.isascii():
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return _SURROGATES.sub("\ufffd", text)
    return text


def canonical_value(value: Any) -> Any:
    """Canonicalize JSON-shaped values, copying only containers that change."""
    if isinstance(value, str):
        return canonical_text(value)
    if isinstance(value, list):
        result = None
        for index, item in enumerate(value):
            clean = canonical_value(item)
            if clean is not item:
                result = value.copy() if result is None else result
                result[index] = clean
        return value if result is None else result
    if isinstance(value, dict):
        result = None
        renamed = None
        for key, item in value.items():
            clean_key = canonical_text(key) if isinstance(key, str) else key
            clean = canonical_value(item)
            if clean_key is not key or clean is not item:
                result = value.copy() if result is None else result
                result[key] = clean
                if clean_key is not key:
                    renamed = {} if renamed is None else renamed
                    renamed[key] = clean_key
        if renamed:
            for key, base in sorted(renamed.items()):
                clean_key, suffix = base, 2
                while clean_key in result:
                    clean_key = f"{base}~{suffix}"
                    suffix += 1
                result[clean_key] = result.pop(key)
        return value if result is None else result
    return value


def canonical_message(message: dict) -> dict:
    return canonical_value(message)


def count_surrogates(value: Any) -> int:
    if isinstance(value, str):
        return len(_SURROGATES.findall(value))
    if isinstance(value, dict):
        return sum(count_surrogates(key) + count_surrogates(item) for key, item in value.items())
    if isinstance(value, list):
        return sum(map(count_surrogates, value))
    return 0


def _extract_text_part_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, str):
            return nested
        nested = value.get("content")
        if isinstance(nested, str):
            return nested
    return None


def normalize_content_value(content: Any) -> str | None:
    """Return a stable text representation for message content.

    ``None`` remains ``None`` so callers that distinguish SQL NULL from an empty
    string can preserve that behavior. Strings use the host's surrogate policy. Structured
    content is serialized deterministically so storage, source-id matching, and
    token accounting all see the same value.
    """
    if content is None:
        return None
    if isinstance(content, str):
        return canonical_text(content)
    try:
        return canonical_text(json.dumps(content, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return canonical_text(str(content))


def _part_type(part: Any) -> str | None:
    """The ``type`` of a content part when it is a dict with a string type; else None."""
    kind = part.get("type") if isinstance(part, dict) else None
    return kind if isinstance(kind, str) else None


def split_image_parts(content: Any) -> tuple[Any, int]:
    """Return ``content`` with structured image parts stripped, and their number.

    Mirrors the host's token estimate: a list part of an image type becomes
    ``{"type": <type>, "image": "[stripped]"}``, and a ``_multimodal`` tool
    result with image parts becomes its ``text_summary``. Every other value,
    strings included, is returned unchanged with a count of 0.
    """
    if isinstance(content, list):
        count = sum(
            1 for part in content if _part_type(part) in _IMAGE_PART_TYPES
        )
        if count:
            return [
                {"type": part.get("type"), "image": "[stripped]"}
                if _part_type(part) in _IMAGE_PART_TYPES
                else part
                for part in content
            ], count
    elif isinstance(content, dict) and content.get("_multimodal"):
        parts = content.get("content")
        if isinstance(parts, list):
            count = sum(
                1
                for part in parts
                if _part_type(part) in _MULTIMODAL_IMAGE_PART_TYPES
            )
            if count:
                return content.get("text_summary") or "", count
    return content, 0


def text_content_for_pattern_matching(content: Any) -> str | None:
    """Return the operator-visible text string used by message filters.

    Structured multimodal payloads often arrive as lists of content parts. For
    ignore-pattern matching, prefer concatenated text parts so anchored patterns
    bind to the text an operator sees. If no text parts are present, fall back to
    the stable normalized representation used for storage.
    """
    if content is None or isinstance(content, str):
        return normalize_content_value(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = part.get("type")
                if part_type in _TEXT_PART_TYPES:
                    text = _extract_text_part_value(part.get("text"))
                    if text is None:
                        text = _extract_text_part_value(part.get("content"))
                    if text:
                        parts.append(text)
        if parts:
            return "\n".join(parts)
    return normalize_content_value(content)


def stored_text_content_for_pattern_matching(content: Any) -> str | None:
    """Return message-filter text for content read back from storage.

    Structured content is persisted as canonical JSON. Decode that legacy stored
    representation when it round-trips to the same normalized string so restart
    reconciliation applies the same text-first ignore policy to durable rows as
    it applies to live structured messages.
    """
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return text_content_for_pattern_matching(content)
        if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
            return text_content_for_pattern_matching(decoded)
    return text_content_for_pattern_matching(content)
