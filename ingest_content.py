"""Content helpers for storage-boundary ingest protection."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

from .externalize import extract_externalized_refs, find_externalized_payload_for_message

_MEDIA_TYPE_HINTS = ("image", "audio", "video")
_MEDIA_VALUE_KEYS = (
    "image_url",
    "input_image",
    "output_image",
    "audio_url",
    "video_url",
    "image",
    "audio",
    "video",
)


def _contains_media_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(_DATA_URI_BASE64_RE.search(value))
    if isinstance(value, list):
        return any(_contains_media_payload(item) for item in value)
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").lower()
        if any(hint in block_type for hint in _MEDIA_TYPE_HINTS):
            return True
        for key, nested in value.items():
            key_text = str(key).lower()
            if key_text in _MEDIA_VALUE_KEYS:
                return True
            if _contains_media_payload(nested):
                return True
    return False


def _externalization_kind_for_message(message: Dict[str, Any]) -> str:
    role = str(message.get("role") or "unknown")
    if role == "tool":
        return "tool_result"
    if _contains_media_payload(message.get("content")):
        return "media_payload"
    return "raw_payload"


# Any data URI base64 payload, not just image/audio/video. Keep the trailing
# payload alphabet conservative so we do not slurp surrounding JSON/markdown.
# Raw scans can see JSON-escaped slashes before decoding, including both `\/`
# and unicode escapes such as `\u002f` in duplicate-key argument strings.
_JSON_ESCAPED_SLASH_RE = r"(?:/|\\/|\\u002[fF])"
_DATA_URI_BASE64_RE = re.compile(
    rf"data:(?:[A-Za-z0-9.+-]|{_JSON_ESCAPED_SLASH_RE})*"
    rf"(?:;[A-Za-z0-9_.+%-]+=(?:[-A-Za-z0-9_.+%]|{_JSON_ESCAPED_SLASH_RE})*)*"
    rf";base64,(?:[A-Za-z0-9+=]|{_JSON_ESCAPED_SLASH_RE}){{256,}}(?=$|[^A-Za-z0-9+/=])",
    re.IGNORECASE,
)

_BASE64_RUN_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/=_-]{4096,})(?![A-Za-z0-9+/=_-])")
# Line-wrapped base64 (MIME 76 / PEM 64 chars per line) never forms a single
# 4096-char contiguous run, so _BASE64_RUN_RE misses it entirely. Match a block
# of consecutive base64-alphabet lines; looks_like_long_base64 makes the final
# call on the whitespace-compacted block.
_WRAPPED_BASE64_MIN_LINE_CHARS = 40
_WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS = 16
_BASE64_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/=_\s-]+$")
_BASE64_LINE_ALPHABET_RE = re.compile(r"^[A-Za-z0-9+/=_-]+$")
_EXTERNALIZED_PLACEHOLDER_PREFIX = "[Externalized LCM ingest payload:"
_QUARANTINED_ASSISTANT_KIND = "quarantined_assistant_output"
_QUARANTINED_ASSISTANT_REASON = "high_repetition"
_QUARANTINED_ASSISTANT_MIN_CHARS = 65_536
_QUARANTINED_ASSISTANT_MIN_TOKENS = 1_000
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_REPETITION_SEGMENT_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
_HEARTBEAT_NOISE_RE = re.compile(
    r"^(?:still\s+working|working\s+on\s+it|processing|checking|one\s+moment|ping|heartbeat|no\s+update)(?:[.!…\s-]*)$",
    re.IGNORECASE,
)
_HEARTBEAT_NOISE_MAX_CHARS = 256
_GENERIC_BASE64_MIN_CHARS = 4096
_INGEST_PLACEHOLDER_RE = re.compile(r"\[Externalized LCM ingest payload:.*?;\s*ref=([^;\]\s]+)\]")


def _is_wrapped_base64_line(line: str) -> bool:
    stripped = line.strip("\r\n")
    return (
        len(stripped) >= _WRAPPED_BASE64_MIN_LINE_CHARS
        and _BASE64_LINE_ALPHABET_RE.fullmatch(stripped) is not None
    )


def _is_wrapped_base64_terminal_line(line: str) -> bool:
    stripped = line.strip("\r\n")
    return (
        _WRAPPED_BASE64_MIN_TERMINAL_LINE_CHARS
        <= len(stripped)
        < _WRAPPED_BASE64_MIN_LINE_CHARS
        and len(stripped) % 4 == 0
        and _BASE64_LINE_ALPHABET_RE.fullmatch(stripped) is not None
    )


def _looks_like_hex_hash_inventory(payload: str) -> bool:
    """Return True for newline inventories of hex digests, not base64 payloads."""
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    digest_lengths = {40, 56, 64, 96, 128}
    return all(
        len(line) in digest_lengths and re.fullmatch(r"[0-9a-fA-F]+", line) is not None
        for line in lines
    )


def _iter_wrapped_base64_blocks(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield (start, end, payload) for line-wrapped base64 blocks.

    Implemented as a line scanner instead of a wide regex so long
    base64-alphabet single lines that are not actually wrapped do not trigger
    repeated failed block matches.
    """
    offset = 0
    block_start: int | None = None
    block_parts: list[str] = []
    block_end = 0

    def finish_block() -> tuple[int, int, str] | None:
        nonlocal block_start, block_parts, block_end
        if block_start is not None and block_parts:
            payload = "".join(block_parts)
            start, end = block_start, block_end
            block_start = None
            block_parts = []
            block_end = 0
            if not _looks_like_hex_hash_inventory(payload) and looks_like_long_base64(payload):
                return (start, end, payload)
        block_start = None
        block_parts = []
        block_end = 0
        return None

    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        if _is_wrapped_base64_line(line) or (
            block_start is not None
            and block_parts
            and _is_wrapped_base64_terminal_line(line)
        ):
            if block_start is None:
                block_start = line_start
            block_parts.append(line)
            block_end = offset
            continue
        block = finish_block()
        if block is not None:
            yield block
    block = finish_block()
    if block is not None:
        yield block


def _replace_wrapped_base64_blocks(text: str, replace: Callable[[str], str]) -> str:
    chunks: list[str] = []
    cursor = 0
    changed = False
    for start, end, payload in _iter_wrapped_base64_blocks(text):
        chunks.append(text[cursor:start])
        chunks.append(replace(payload))
        cursor = end
        changed = True
    if not changed:
        return text
    chunks.append(text[cursor:])
    return "".join(chunks)


def is_externalized_ingest_placeholder(text: str) -> bool:
    return isinstance(text, str) and bool(_INGEST_PLACEHOLDER_RE.fullmatch(text.strip()))


def contains_data_uri_base64(text: str) -> bool:
    return isinstance(text, str) and bool(_DATA_URI_BASE64_RE.search(text))


def contains_long_base64_run(text: str, *, min_chars: int = _GENERIC_BASE64_MIN_CHARS) -> bool:
    if not isinstance(text, str) or len(text) < min_chars:
        return False
    if any(looks_like_long_base64(match.group(1), min_chars=min_chars) for match in _BASE64_RUN_RE.finditer(text)):
        return True
    # Also catch line-wrapped base64 blocks (MIME/PEM), which never form a
    # single contiguous run.
    return any(
        looks_like_long_base64(payload, min_chars=min_chars)
        for _start, _end, payload in _iter_wrapped_base64_blocks(text)
    )


def extract_ingest_externalized_refs(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for match in _INGEST_PLACEHOLDER_RE.finditer(text):
        ref = match.group(1).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _is_basename_ref(ref: str) -> bool:
    return bool(ref) and ref.endswith(".json") and "/" not in ref and "\\" not in ref and Path(ref).name == ref


def extract_all_externalized_payload_refs(text: str) -> list[str]:
    """Return deduplicated refs from recognized externalized payload placeholders."""
    if not isinstance(text, str) or not text:
        return []
    refs: list[str] = []
    for ref in extract_ingest_externalized_refs(text) + extract_externalized_refs(text):
        if _is_basename_ref(ref) and ref not in refs:
            refs.append(ref)
    return refs


def _safe_placeholder_metadata(value: Any) -> str:
    text = str(value or "?")
    safe = re.sub(r"[^A-Za-z0-9_.:/-]+", "-", text).strip("-")
    return (safe or "?")[:120]


def _normalized_repetition_segments(text: str) -> list[str]:
    segments = []
    for segment in _REPETITION_SEGMENT_SPLIT_RE.split(text):
        normalized = re.sub(r"\s+", " ", segment.strip().lower())
        if len(normalized) >= 32:
            segments.append(normalized)
    return segments


def heartbeat_noise_reason(role: str, text: str) -> str | None:
    """Return a read-only doctor category for short heartbeat/progress noise.

    This intentionally does not drive ingest protection or cleanup. It only
    surfaces metadata-only candidates for operator review.
    """
    role = str(role or "")
    if role not in {"assistant", "tool", "system"}:
        return None
    if not isinstance(text, str):
        return None
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized or len(normalized) > _HEARTBEAT_NOISE_MAX_CHARS:
        return None
    if _HEARTBEAT_NOISE_RE.match(normalized):
        return "heartbeat_progress"
    return None


def assistant_output_quarantine_reason(text: str) -> str | None:
    """Return a quarantine reason for obviously broken assistant output.

    The gate is intentionally conservative: content must be very large and show
    both low token novelty and repeated sentence/line segments. Long diverse
    reports and code with varied identifiers should stay inline.
    """
    if not isinstance(text, str) or len(text) < _QUARANTINED_ASSISTANT_MIN_CHARS:
        return None

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    tokens = _WORD_TOKEN_RE.findall(normalized)
    if len(tokens) < _QUARANTINED_ASSISTANT_MIN_TOKENS:
        if len(normalized) >= _QUARANTINED_ASSISTANT_MIN_CHARS and len(set(normalized)) <= 12:
            return _QUARANTINED_ASSISTANT_REASON
        return None

    token_counts = Counter(tokens)
    unique_token_ratio = len(token_counts) / max(1, len(tokens))
    top_token_ratio = token_counts.most_common(1)[0][1] / max(1, len(tokens))

    segments = _normalized_repetition_segments(text)
    top_segment_ratio = 0.0
    duplicate_segment_ratio = 0.0
    if len(segments) >= 20:
        segment_counts = Counter(segments)
        top_segment_ratio = segment_counts.most_common(1)[0][1] / len(segments)
        duplicate_segment_ratio = 1.0 - (len(segment_counts) / len(segments))

    if unique_token_ratio <= 0.03 and (
        top_segment_ratio >= 0.10
        or duplicate_segment_ratio >= 0.50
        or top_token_ratio >= 0.08
    ):
        return _QUARANTINED_ASSISTANT_REASON

    # Covers degenerate long loops with little punctuation/newline structure.
    if unique_token_ratio <= 0.015 and len(set(normalized)) <= 64:
        return _QUARANTINED_ASSISTANT_REASON

    return None


def _quarantined_assistant_placeholder(summary: Dict[str, Any], *, reason: str) -> str:
    return (
        "[Externalized LCM ingest payload: assistant output quarantined; "
        f"kind={_safe_placeholder_metadata(summary.get('kind') or _QUARANTINED_ASSISTANT_KIND)}; "
        f"reason={_safe_placeholder_metadata(reason)}; "
        f"field={_safe_placeholder_metadata(summary.get('field_path') or 'content')}; "
        f"chars={summary.get('content_chars', 0)}; bytes={summary.get('content_bytes', 0)}; "
        f"ref={summary.get('ref', '')}]"
    )


def _volatile_quarantined_assistant_placeholder(content: str, *, reason: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return (
        "[LCM active replay placeholder: assistant output quarantined; "
        f"kind={_QUARANTINED_ASSISTANT_KIND}; "
        f"reason={_safe_placeholder_metadata(reason)}; "
        "scope=ignored_message_pattern; field=content; "
        f"chars={len(content)}; bytes={len(content.encode('utf-8'))}; "
        f"sha256={digest}]"
    )


def _existing_quarantined_assistant_placeholder(
    content: str,
    *,
    role: str,
    session_id: str,
    config: object,
    hermes_home: str,
    reason: str,
) -> str | None:
    existing = find_externalized_payload_for_message(
        content,
        session_id=session_id,
        kind=_QUARANTINED_ASSISTANT_KIND,
        role=role,
        config=config,
        hermes_home=hermes_home,
    )
    if existing is None:
        return None
    return _quarantined_assistant_placeholder(existing, reason=reason)


def looks_like_long_base64(text: str, *, min_chars: int = _GENERIC_BASE64_MIN_CHARS) -> bool:
    """Conservative long-base64 heuristic.

    Avoids short hashes/IDs/JWT-ish snippets by requiring a very long run and a
    high base64 alphabet ratio. PEM blocks and ordinary logs contain delimiters
    or whitespace/headers that keep them from matching as one clean run.
    """
    if not isinstance(text, str) or len(text) < min_chars:
        return False
    compact = "".join(text.split())
    if len(compact) < min_chars:
        return False
    if len(compact) % 4 == 1:
        return False
    if not _BASE64_ALPHABET_RE.match(text):
        return False
    # Compute the base64 density over the whitespace-stripped content, not the
    # raw text: otherwise line-ending overhead sinks the ratio and canonical
    # CRLF-wrapped MIME (76/78 = 0.974) and PEM (64/66 = 0.970) blocks fall
    # below 0.98 and are wrongly left inline.
    base64_chars = sum(1 for ch in compact if ch.isalnum() or ch in "+/=_-")
    ratio = base64_chars / max(1, len(compact))
    if ratio < 0.98:
        return False
    # Require at least a bit of mixed alphabet so a long log line of one
    # repeated character is not treated as a binary payload.
    return len(set(compact.rstrip("="))) >= 8


def _maybe_parse_json_string(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    candidates = [text]
    if '\\"' in stripped:
        candidates.append(stripped.replace('\\"', '"'))
    for candidate in candidates:
        if _json_has_duplicate_object_keys(candidate):
            return None
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def _json_has_duplicate_object_keys(text: str) -> bool:
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped or stripped[0] not in "[{":
        return False
    duplicate = False

    def detect_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        seen: set[str] = set()
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                duplicate = True
            seen.add(key)
            result[key] = value
        return result

    try:
        json.loads(text, object_pairs_hook=detect_pairs)
    except Exception:
        return False
    return duplicate


def _dict_field_path(parent: str, key: Any) -> str:
    component = str(key)
    return f"{parent}.{component}" if parent else component


def _payload_key_field_path(parent: str) -> str:
    return f"{parent}.<key>" if parent else "<key>"
