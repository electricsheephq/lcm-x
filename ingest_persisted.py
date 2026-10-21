"""Persisted-output recovery for storage-boundary ingest protection."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from .ingest_policy import (
    _SENSITIVE_PLACEHOLDER_PREFIX,
)

_PERSISTED_OUTPUT_TAG = "<persisted-output>"
_PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
_PERSISTED_OUTPUT_SAVED_TO_RE = re.compile(r"^Full output saved to:\s*(?P<path>.+?)\s*$", re.MULTILINE)
_PERSISTED_OUTPUT_PREVIEW_RE = re.compile(
    r"^Preview \(first \d+ chars\):\s*\r?\n(?P<preview>.*?)\r?\n</persisted-output>\s*$",
    re.MULTILINE | re.DOTALL,
)
_PERSISTED_OUTPUT_CHAR_COUNT_RE = re.compile(r"too large\s*\((?P<count>[\d,]+)\s+characters\b", re.IGNORECASE)
_PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE = re.compile(
    r"\r?\n\[LCM persisted-output marker identity: preview_sha256=(?P<sha256>[0-9a-f]{64})\]"
    r"(?:\r?\n\[LCM persisted-output file generation: size=\d+; mtime_ns=\d+; ctime_ns=\d+\])?"
    r"\r?\n</persisted-output>\s*$"
)
_PERSISTED_OUTPUT_INLINE_GENERATION_RE = re.compile(
    r"\r?\n\[LCM persisted-output file generation: size=(?P<size>\d+); mtime_ns=(?P<mtime_ns>\d+); ctime_ns=(?P<ctime_ns>\d+)\]\r?\n</persisted-output>\s*$"
)
_PERSISTED_OUTPUT_INLINE_METADATA_RE = re.compile(
    r"\r?\n\[LCM persisted-output (?:file generation|marker identity):[^\r\n]*\]\s*$"
)
_UNRECOVERABLE_TRUNCATION_RE = re.compile(
    r"\[Truncated:\s*tool response was [\d,]+ chars\.\s*Full output could not be saved to sandbox\.\]",
    re.IGNORECASE,
)
_HERMES_RESULTS_DIRNAME = "hermes-results"
_MAX_RECOVERED_PERSISTED_OUTPUT_BYTES = 64 * 1024 * 1024


def _is_unrecoverable_tool_truncation_marker(text: str | None) -> bool:
    return isinstance(text, str) and bool(_UNRECOVERABLE_TRUNCATION_RE.search(text))


def _expected_persisted_output_chars(text: str | None) -> int | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_CHAR_COUNT_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group("count").replace(",", ""))
    except ValueError:
        return None


def _persisted_output_preview_prefix(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_PREVIEW_RE.search(text.strip())
    if not match:
        return None
    preview = match.group("preview")
    if preview.endswith("\r\n..."):
        preview = preview[: -len("\r\n...")]
    elif preview.endswith("\n..."):
        preview = preview[: -len("\n...")]
    return preview


def _persisted_output_preview_prefix_digest(text: str | None) -> str | None:
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix:
        return None
    return hashlib.sha256(
        preview_prefix.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _persisted_output_inline_preview_sha256(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_PREVIEW_SHA256_RE.search(text)
    if not match:
        return None
    return match.group("sha256")


def _inline_persisted_output_generation_metadata(text: str | None) -> dict[str, int] | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_INLINE_GENERATION_RE.search(text)
    if not match:
        return None
    try:
        return {
            "size": int(match.group("size")),
            "mtime_ns": int(match.group("mtime_ns")),
            "ctime_ns": int(match.group("ctime_ns")),
        }
    except (TypeError, ValueError):
        return None


def _has_inline_persisted_output_generation_metadata(text: str | None) -> bool:
    return _inline_persisted_output_generation_metadata(text) is not None


def _persisted_output_marker_identity_digest(text: str | None) -> str | None:
    return _persisted_output_inline_preview_sha256(text) or _persisted_output_preview_prefix_digest(text)


def _has_lossy_sensitive_redaction(text: str | None) -> bool:
    if not isinstance(text, str) or _SENSITIVE_PLACEHOLDER_PREFIX not in text:
        return False
    for match in re.finditer(r"\[LCM sensitive redaction: (?P<body>[^\]]+)\]", text):
        body = match.group("body")
        fields = {
            key: value
            for key, value in re.findall(r"([A-Za-z0-9_]+)=([^;]+)", body)
        }
        if fields.get("name") == "password_assignment" and "sha256" not in fields:
            return True
    return False


def _persisted_output_saved_path(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = _PERSISTED_OUTPUT_SAVED_TO_RE.search(text.strip())
    if not match:
        return None
    raw_path = match.group("path").strip()
    if not raw_path or "\x00" in raw_path:
        return None
    return raw_path


def _safe_temp_hermes_results_file(path: Path) -> Path | None:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        return None
    parent = path.parent
    if parent.name != _HERMES_RESULTS_DIRNAME:
        return None
    try:
        expected_parent = (Path(tempfile.gettempdir()) / _HERMES_RESULTS_DIRNAME).resolve()
        parent_is_valid_dir = parent.exists() and parent.is_dir() and not parent.is_symlink()
        if not parent_is_valid_dir or parent.resolve() != expected_parent:
            return None
        return expected_parent / path.name
    except OSError:
        return None


def _is_hermes_persisted_output_marker(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    marker = text.strip()
    return (
        marker.startswith(_PERSISTED_OUTPUT_TAG)
        and marker.endswith(_PERSISTED_OUTPUT_CLOSING_TAG)
        and _expected_persisted_output_chars(marker) is not None
        and _PERSISTED_OUTPUT_SAVED_TO_RE.search(marker) is not None
    )


def _stat_generation_metadata(stats: os.stat_result) -> dict[str, int]:
    return {
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
        "ctime_ns": int(stats.st_ctime_ns),
    }


def _read_regular_file_no_symlink(path: Path) -> tuple[str, dict[str, int]] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd: int | None = None
    try:
        lstat_result = os.lstat(str(path))
        if not stat.S_ISREG(lstat_result.st_mode):
            return None
        if lstat_result.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        fd = os.open(str(path), flags)
        stats_before = os.fstat(fd)
        if not stat.S_ISREG(stats_before.st_mode):
            return None
        if stats_before.st_size > _MAX_RECOVERED_PERSISTED_OUTPUT_BYTES:
            return None
        with os.fdopen(fd, "rb") as handle:
            fd = None
            raw = handle.read()
            stats_after = os.fstat(handle.fileno())
        if _stat_generation_metadata(stats_before) != _stat_generation_metadata(stats_after):
            return None
        return raw.decode("utf-8"), _stat_generation_metadata(stats_after)
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def recover_hermes_persisted_output_with_file_stat(text: str | None) -> tuple[str, dict[str, int]] | None:
    """Recover Hermes host `<persisted-output>` content when the backing file is safe.

    Recovery is intentionally conservative: the marker must include Hermes'
    character count, the file path must be an absolute basename under a
    `hermes-results` temp directory, the target must be a regular non-symlink
    file, and the recovered character count must match the marker. If any check
    fails, callers should keep the marker/preview instead of claiming lossless
    recovery from an unsafe or stale file.
    """
    if not isinstance(text, str) or not _is_hermes_persisted_output_marker(text):
        return None
    expected_chars = _expected_persisted_output_chars(text)
    if expected_chars is None:
        return None
    raw_path = _persisted_output_saved_path(text)
    if raw_path is None:
        return None
    path = Path(raw_path)
    safe_path = _safe_temp_hermes_results_file(path)
    if safe_path is None:
        return None
    recovered_with_stat = _read_regular_file_no_symlink(safe_path)
    if recovered_with_stat is None:
        return None
    recovered, file_stat = recovered_with_stat
    if len(recovered) != expected_chars:
        return None
    preview_prefix = _persisted_output_preview_prefix(text)
    if not preview_prefix or not recovered.startswith(preview_prefix):
        return None
    return recovered, file_stat


def recover_hermes_persisted_output(text: str | None) -> str | None:
    recovered_with_stat = recover_hermes_persisted_output_with_file_stat(text)
    if recovered_with_stat is None:
        return None
    recovered, _file_stat = recovered_with_stat
    return recovered


def _add_inline_persisted_output_generation_metadata(text: str, file_stat: dict[str, int] | None) -> str:
    if not file_stat or not isinstance(text, str) or "</persisted-output>" not in text:
        return text
    generation = (
        "[LCM persisted-output file generation: "
        f"size={file_stat['size']}; "
        f"mtime_ns={file_stat['mtime_ns']}; "
        f"ctime_ns={file_stat['ctime_ns']}]"
    )
    if generation in text:
        return text
    return text.replace("</persisted-output>", f"{generation}\n</persisted-output>", 1)


def _add_inline_persisted_output_identity_metadata(text: str, preview_sha256: str | None) -> str:
    if (
        not isinstance(text, str)
        or "</persisted-output>" not in text
        or not isinstance(preview_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", preview_sha256)
    ):
        return text
    if _has_lossy_sensitive_redaction(text) or _persisted_output_inline_preview_sha256(text):
        return text
    identity = f"[LCM persisted-output marker identity: preview_sha256={preview_sha256}]"
    return text.replace("</persisted-output>", f"{identity}\n</persisted-output>", 1)
