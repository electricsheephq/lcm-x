"""Externalized-payload integrity reporting for storage-boundary ingest protection."""

from __future__ import annotations

import json
import re
import stat as stat
from pathlib import Path
from typing import Any, Iterator as Iterator
from .externalize import extract_externalized_ref, get_large_output_storage_dir, is_externalized_placeholder, is_generated_payload_ref_name, load_externalized_payload
from .ingest_content import (
    _EXTERNALIZED_PLACEHOLDER_PREFIX,
    _GENERIC_BASE64_MIN_CHARS,
    _HEARTBEAT_NOISE_MAX_CHARS,
    _INGEST_PLACEHOLDER_RE,
    _QUARANTINED_ASSISTANT_KIND,
    _QUARANTINED_ASSISTANT_MIN_CHARS,
    _is_basename_ref,
    _maybe_parse_json_string,
    assistant_output_quarantine_reason,
    contains_data_uri_base64,
    contains_long_base64_run,
    extract_all_externalized_payload_refs,
    extract_ingest_externalized_refs,
    heartbeat_noise_reason,
    is_externalized_ingest_placeholder,
)

_EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE = re.compile(
    r"\[(?:Externalized|GC'd externalized) (?:tool output|payload):.*?;\s*ref=([^;\]\s]+)\]"
)
_SOURCE_LITERAL_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:(?:\d+[|:])|(?:[+-](?![+-])))?\s*"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\[[^\r\n]+\])?\s*=\s*(?:[rubf]{0,2})?[\"']\s*$",
    re.IGNORECASE,
)
_SOURCE_LITERAL_LINE_RE = re.compile(
    r"^\s*(?:(?:\d+[|:])|(?:[+-](?![+-])))\s*(?:[rubf]{0,2})?[\"']\s*$",
    re.IGNORECASE,
)
_SOURCE_DIFF_LITERAL_RE = re.compile(
    r"^\s*[+-](?![+-])\s*(?:[rubf]{0,2})?[\"'][^\r\n]*$",
    re.IGNORECASE,
)


def restore_ingest_payload_placeholders(
    text: str,
    *,
    config: Any,
    hermes_home: str = "",
    session_id: str = "",
) -> str:
    """Restore ingest placeholders in a stored identity string for matching only.

    Missing or mismatched payload files leave the placeholder untouched so callers
    never fabricate content or hide a recovery problem.
    """
    if not isinstance(text, str) or _EXTERNALIZED_PLACEHOLDER_PREFIX not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        payload = load_externalized_payload(ref, config=config, hermes_home=hermes_home)
        if payload is None or payload.get("kind") != "ingest_payload":
            return match.group(0)
        payload_session_id = payload.get("session_id") or ""
        if session_id and payload_session_id and payload_session_id != session_id:
            return match.group(0)
        content = payload.get("content")
        return content if isinstance(content, str) else match.group(0)

    return _INGEST_PLACEHOLDER_RE.sub(replace, text)


def _append_unique_refs(target: list[str], refs: list[str]) -> None:
    for ref in refs:
        if ref not in target:
            target.append(ref)


def _walk_string_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        parsed = _maybe_parse_json_string(value)
        if parsed is None:
            yield value
        else:
            yield from _walk_string_values(parsed)
    elif isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_string_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_string_values(item)


def _walk_tool_call_argument_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "arguments":
                yield nested
            yield from _walk_tool_call_argument_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_tool_call_argument_values(item)


def _is_inside_token_quote_span(text: str, start: int, token: str) -> bool:
    in_span = False
    i = 0
    while i < start:
        if text.startswith(token, i):
            in_span = not in_span
            i += len(token)
        else:
            i += 1
    return in_span


def _looks_like_example_quote_context(context: str) -> bool:
    return re.search(r"(?:pytest\s+output|log|example|traceback|failure)\s*:\s*$", context.lower()) is not None


def _has_local_escaped_quote_before(text: str, start: int) -> bool:
    boundary = max(text.rfind(delimiter, 0, start) for delimiter in (",", "{", "["))
    segment = text[boundary + 1:start]
    matches = list(re.finditer(r"\\+[\"']", segment))
    if not matches:
        return False
    quote = matches[-1]
    context = segment[max(0, quote.start() - 80):quote.start()]
    return _looks_like_example_quote_context(context)


def _is_escaped_placeholder_example(text: str, start: int) -> bool:
    prefix = text[max(0, start - 8):start]
    return prefix.endswith("\\") or _has_local_escaped_quote_before(text, start)


def _is_quoted_placeholder_example(text: str, start: int) -> bool:
    for quote_token in ('"', "'"):
        if not _is_inside_token_quote_span(text, start, quote_token):
            continue
        quote = text.rfind(quote_token, 0, start)
        if quote < 0:
            continue
        context = text[max(0, quote - 80):quote]
        if _looks_like_example_quote_context(context):
            return True
    return False


def _looks_like_json_container_string(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _looks_like_example_payload_ref(ref: str) -> bool:
    name = Path(ref).name.lower()
    return name.startswith(("example-", "example_", "fake-", "fake_", "dummy-", "dummy_", "placeholder-", "placeholder_"))


def _placeholder_line_prefix(text: str, start: int) -> str:
    separators = (("\n", 1), ("\r", 1), ("\\n", 2), ("\\r", 2))
    separator_start, separator_length = max(
        ((text.rfind(token, 0, start), length) for token, length in separators),
        key=lambda item: item[0],
    )
    prefix = text[separator_start + separator_length:start]
    return re.sub(r"\\+([\"'])", r"\1", prefix)


def _is_incidental_source_placeholder(text: str, start: int) -> bool:
    """Return true when a placeholder match *looks like* quoted source/docs.

    This is a text-shape heuristic only. It cannot distinguish an ingest-generated
    placeholder that happens to sit inside a code fence, a backticked span, or a
    line-numbered/diff excerpt from an incidental quotation of one, because both
    look identical in isolation. Callers must treat a true result as "suspect",
    not "drop"; see ``_refs_for_externalized_integrity_scan`` and
    ``scan_externalized_payload_integrity``, which settle the question with the
    robust discriminators (does the referenced payload file exist, or does the ref
    name carry externalize.py's minting shape?).
    """
    prefix = _placeholder_line_prefix(text, start)
    if (
        _SOURCE_LITERAL_ASSIGNMENT_RE.search(prefix)
        or _SOURCE_LITERAL_LINE_RE.fullmatch(prefix)
        or _SOURCE_DIFF_LITERAL_RE.fullmatch(prefix)
    ):
        return True
    before = text[:start]
    if before.count("```") % 2:
        return True
    return prefix.count("`") % 2 == 1


def _record_source_context_ref(target: list[str] | None, ref: str) -> None:
    if target is not None and ref not in target:
        target.append(ref)


def _extract_unescaped_externalized_payload_refs(
    text: str,
    *,
    ignore_quoted_spans: bool = False,
    source_context_refs: list[str] | None = None,
) -> list[str]:
    """Return confident refs; park source-context-shaped ones in ``source_context_refs``.

    A ref whose match sits in quoted-source context is not dropped here: it is
    appended to ``source_context_refs`` (when the caller supplies a list) so the
    layer that owns ``hermes_home`` can promote it back when a payload file with
    that name exists or the name carries externalize.py's minting shape. With no
    list supplied the parked refs are simply not returned, preserving the
    conservative text-only behavior.
    """
    refs: list[str] = []
    for pattern in (_INGEST_PLACEHOLDER_RE, _EXTERNALIZED_PAYLOAD_PLACEHOLDER_RE):
        for match in pattern.finditer(text):
            ref = match.group(1).strip()
            if not _is_basename_ref(ref):
                continue
            if _looks_like_example_payload_ref(ref) and _is_escaped_placeholder_example(text, match.start()):
                continue
            if (
                ignore_quoted_spans
                and _looks_like_example_payload_ref(ref)
                and _is_quoted_placeholder_example(text, match.start())
            ):
                continue
            if _is_incidental_source_placeholder(text, match.start()):
                _record_source_context_ref(source_context_refs, ref)
                continue
            if ref not in refs:
                refs.append(ref)
    return refs


def _refs_for_externalized_integrity_scan(
    value: str,
    *,
    role: str,
    field: str,
    source_context_refs: list[str] | None = None,
) -> list[str]:
    """Return refs that plausibly came from LCM storage-boundary placeholders.

    Tool outputs and tool-call arguments often contain escaped code snippets,
    pytest failures, or docs that mention placeholder examples. Counting those
    as live payload references turns doctor into a false-positive machine. Exact
    placeholders are still counted everywhere; embedded unescaped placeholders
    are counted for message content, raw JSON-container tool-call argument
    strings, and raw free-form tool-call argument strings so ingestion-produced
    refs do not disappear while quoted examples stay ignored.

    Refs suppressed only by the source-context text heuristic are appended to
    ``source_context_refs`` instead of being discarded, so the caller can settle
    them on payload-file existence rather than on text shape alone.
    """
    if not isinstance(value, str) or not value:
        return []
    stripped = value.strip()
    if is_externalized_ingest_placeholder(stripped) or is_externalized_placeholder(stripped):
        return extract_all_externalized_payload_refs(stripped)
    if field == "tool_calls":
        parsed = _maybe_parse_json_string(value)
        if parsed is None:
            return _extract_unescaped_externalized_payload_refs(
                value,
                ignore_quoted_spans=True,
                source_context_refs=source_context_refs,
            )
        # Raw JSON text erases whether a match came from quoted source. Once the
        # envelope parses, scan decoded values only and keep raw scanning as the
        # malformed-envelope fallback above.
        refs: list[str] = []
        for argument in _walk_tool_call_argument_values(parsed):
            if isinstance(argument, str):
                parsed_argument = _maybe_parse_json_string(argument)
                if parsed_argument is None:
                    _append_unique_refs(
                        refs,
                        _extract_unescaped_externalized_payload_refs(
                            argument,
                            ignore_quoted_spans=True,
                            source_context_refs=source_context_refs,
                        ),
                    )
                else:
                    for nested in _walk_string_values(parsed_argument):
                        nested_stripped = nested.strip()
                        if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                            _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                        else:
                            _append_unique_refs(
                                refs,
                                _extract_unescaped_externalized_payload_refs(
                                    nested,
                                    ignore_quoted_spans=True,
                                    source_context_refs=source_context_refs,
                                ),
                            )
            else:
                for nested in _walk_string_values(argument):
                    nested_stripped = nested.strip()
                    if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                        _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                    else:
                        _append_unique_refs(
                            refs,
                            _extract_unescaped_externalized_payload_refs(
                                nested,
                                ignore_quoted_spans=True,
                                source_context_refs=source_context_refs,
                            ),
                        )
        for nested in _walk_string_values(parsed):
            nested_stripped = nested.strip()
            if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
            else:
                _append_unique_refs(
                    refs,
                    _extract_unescaped_externalized_payload_refs(
                        nested,
                        ignore_quoted_spans=True,
                        source_context_refs=source_context_refs,
                    ),
                )
        return refs
    if role == "tool":
        refs = _extract_unescaped_externalized_payload_refs(
            value,
            source_context_refs=source_context_refs,
        )
        parsed = _maybe_parse_json_string(value)
        if parsed is not None:
            for nested in _walk_string_values(parsed):
                nested_stripped = nested.strip()
                if is_externalized_ingest_placeholder(nested_stripped) or is_externalized_placeholder(nested_stripped):
                    _append_unique_refs(refs, extract_all_externalized_payload_refs(nested_stripped))
                else:
                    _append_unique_refs(
                        refs,
                        _extract_unescaped_externalized_payload_refs(
                            nested,
                            ignore_quoted_spans=True,
                            source_context_refs=source_context_refs,
                        ),
                    )
        return refs
    return _extract_unescaped_externalized_payload_refs(
        value,
        source_context_refs=source_context_refs,
    )


def scan_externalized_payload_integrity(conn: Any, config: Any, *, hermes_home: str = "", limit: int = 5) -> dict[str, Any]:
    """Compare externalized payload refs stored in messages with JSON files.

    This is intentionally read-only and metadata-only. It does not open payload
    files except through directory metadata, and row samples never include raw
    message content or tool-call arguments.
    """

    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    existing_files: set[str] = set()
    if storage_dir.exists() and storage_dir.is_dir():
        existing_files = {path.name for path in storage_dir.glob("*.json") if path.is_file()}

    referenced_refs: set[str] = set()
    first_location_by_ref: dict[str, dict[str, Any]] = {}
    for store_id, session_id, source, role, content, tool_calls in conn.execute(
        """
        SELECT store_id, session_id, source, role, content, tool_calls
        FROM messages
        WHERE COALESCE(content, '') LIKE '%ref=%]%'
           OR COALESCE(tool_calls, '') LIKE '%ref=%]%'
        ORDER BY store_id ASC
        """
    ).fetchall():
        for field, value in (("content", content), ("tool_calls", tool_calls)):
            if not isinstance(value, str):
                continue
            source_context_refs: list[str] = []
            scanned_refs = _refs_for_externalized_integrity_scan(
                value,
                role=str(role or ""),
                field=field,
                source_context_refs=source_context_refs,
            )
            # The text heuristics cannot tell an ingest-GENERATED placeholder that
            # happens to sit inside a code fence, a backticked span, or a
            # line-numbered excerpt from an incidental quotation of one; assistant
            # turns routinely wrap real tool output in fences. Two independent
            # signals settle a parked ref, and either one is enough:
            #   1. A backing payload file exists, so the placeholder is live.
            #   2. The ref name carries externalize.py's minting shape, so LCM
            #      generated it. This one does not consult the filesystem, which is
            #      what keeps a generated ref diagnosable after its payload is
            #      deleted or GC'd -- existence alone would drop the ref precisely
            #      when recovery is broken and doctor most needs to report it.
            # A ref with neither signal is the quoted-source case the heuristics
            # target: hand-written docs examples have no file and no minted shape.
            #
            # Accepted residual, deliberate -- do NOT "fix" it by re-gating signal 2
            # on local state: a stored excerpt quoting a real placeholder minted on
            # a DIFFERENT installation carries the minted shape with no local file,
            # so it is reported missing. That is a visible, self-correcting false
            # alarm (the operator sees a ref name nothing local ever minted), and it
            # is exactly what main did before any of this filtering existed, so it
            # is not a behavior this filter introduced. The conservative direction
            # for an integrity diagnostic is to over-report a broken ref rather than
            # go silent on one; re-gating on existence would restore the silence
            # this promotion was added to remove. Separating a foreign minted ref
            # from a local GC'd one needs durable local provenance -- a persisted
            # minted-ref ledger or a per-install salt in the filename -- which
            # changes the payload storage contract and is future work.
            promoted_refs = [
                ref
                for ref in source_context_refs
                if ref not in scanned_refs
                and (ref in existing_files or is_generated_payload_ref_name(ref))
            ]
            for ref in (*scanned_refs, *promoted_refs):
                referenced_refs.add(ref)
                first_location_by_ref.setdefault(
                    ref,
                    {
                        "store_id": int(store_id),
                        "session_id": session_id,
                        "source": source,
                        "role": role,
                        "field": field,
                        "externalized_ref": ref,
                    },
                )

    missing_refs = sorted(ref for ref in referenced_refs if ref not in existing_files)
    existing_ref_count = sum(1 for ref in referenced_refs if ref in existing_files)
    unreferenced_files = sorted(ref for ref in existing_files if ref not in referenced_refs)

    return {
        "externalized_payload_refs_total": len(referenced_refs),
        "externalized_payload_refs_existing": existing_ref_count,
        "externalized_payload_refs_missing": len(missing_refs),
        "externalized_payload_files_unreferenced": len(unreferenced_files),
        "missing_externalized_payload_refs": [
            first_location_by_ref[ref] for ref in missing_refs[:limit] if ref in first_location_by_ref
        ],
        "unreferenced_externalized_payload_files": [
            {"externalized_ref": ref} for ref in unreferenced_files[:limit]
        ],
    }


def scan_sqlite_payload_risks(conn: Any, *, limit: int = 5) -> dict[str, Any]:
    """Return bounded diagnostics for suspicious inline payload storage.

    Diagnostics intentionally omit previews/raw payload text. Rows include only
    metadata needed for triage and a recoverability ref when a compact
    externalized placeholder is present.
    """

    def make_row(row: Any, *, field: str, length_key: str, category: str) -> dict[str, Any]:
        store_id, session_id, source, role, length, value = row
        value = value or ""
        result = {
            "store_id": int(store_id),
            "session_id": session_id,
            "source": source,
            "role": role,
            "field": field,
            "length": int(length or 0),
            length_key: int(length or 0),
            "suspicious_category": category,
        }
        refs = extract_ingest_externalized_refs(value) if isinstance(value, str) else []
        ref = refs[0] if refs else (extract_externalized_ref(value) if isinstance(value, str) else None)
        if ref:
            result["externalized_ref"] = ref
        return result

    largest_content = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    largest_tool_calls = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(tool_calls), 0) AS tool_calls_len, tool_calls
        FROM messages
        ORDER BY tool_calls_len DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    candidate_cap = max(limit * 20, limit)
    # Pre-filter broadly in SQL, then apply the same conservative Python regex
    # used by ingest externalization. This avoids false positives from code or
    # doctor text that quotes scaffolds such as "data:%;base64,%" or
    # `DATA_URI = "data:image/png;base64," + DATA_PAYLOAD`.
    data_uri_content_candidates = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE lower(content) GLOB '*data:*;base64,*'
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (candidate_cap,),
    ).fetchall()
    data_uri_tool_call_candidates = conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(tool_calls), 0) AS tool_calls_len, tool_calls
        FROM messages
        WHERE lower(tool_calls) GLOB '*data:*;base64,*'
        ORDER BY tool_calls_len DESC
        LIMIT ?
        """,
        (candidate_cap,),
    ).fetchall()
    data_uri_content = [
        row for row in data_uri_content_candidates if isinstance(row[-1], str) and contains_data_uri_base64(row[-1])
    ][:limit]
    data_uri_tool_calls = [
        row for row in data_uri_tool_call_candidates if isinstance(row[-1], str) and contains_data_uri_base64(row[-1])
    ][:limit]

    generic_rows = []
    for store_id, session_id, source, role, content, tool_calls in conn.execute(
        """
        SELECT store_id, session_id, source, role, content, tool_calls
        FROM messages
        WHERE COALESCE(length(content), 0) >= ? OR COALESCE(length(tool_calls), 0) >= ?
        ORDER BY MAX(COALESCE(length(content), 0), COALESCE(length(tool_calls), 0)) DESC
        LIMIT ?
        """,
        (_GENERIC_BASE64_MIN_CHARS, _GENERIC_BASE64_MIN_CHARS, candidate_cap),
    ).fetchall():
        for field, value in (("content", content), ("tool_calls", tool_calls)):
            if isinstance(value, str) and contains_long_base64_run(value):
                result = {
                    "store_id": int(store_id),
                    "session_id": session_id,
                    "source": source,
                    "role": role,
                    "field": field,
                    "length": len(value),
                    "suspicious_category": "base64_like",
                }
                refs = extract_ingest_externalized_refs(value)
                ref = refs[0] if refs else extract_externalized_ref(value)
                if ref:
                    result["externalized_ref"] = ref
                generic_rows.append(result)
                break
        if len(generic_rows) >= limit:
            break

    quarantined_assistant_rows = [
        make_row(row, field="content", length_key="content_len", category=_QUARANTINED_ASSISTANT_KIND)
        for row in conn.execute(
            """
            SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
            FROM messages
            WHERE role = 'assistant'
              AND content LIKE '%Externalized LCM ingest payload:%quarantined_assistant_output%'
            ORDER BY store_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    ]

    suspicious_repetitive_assistant_rows = []
    for row in conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE role = 'assistant'
          AND COALESCE(length(content), 0) >= ?
          AND content NOT LIKE '%Externalized LCM ingest payload:%quarantined_assistant_output%'
        ORDER BY content_len DESC
        LIMIT ?
        """,
        (_QUARANTINED_ASSISTANT_MIN_CHARS, candidate_cap),
    ).fetchall():
        value = row[-1]
        if isinstance(value, str):
            reason = assistant_output_quarantine_reason(value)
            if reason:
                suspicious_repetitive_assistant_rows.append(
                    make_row(row, field="content", length_key="content_len", category=reason)
                )
        if len(suspicious_repetitive_assistant_rows) >= limit:
            break

    heartbeat_noise_rows = []
    for row in conn.execute(
        """
        SELECT store_id, session_id, source, role, COALESCE(length(content), 0) AS content_len, content
        FROM messages
        WHERE role IN ('assistant', 'tool', 'system')
          AND COALESCE(length(content), 0) BETWEEN 1 AND ?
          AND (
            lower(trim(content)) GLOB 'still working*'
            OR lower(trim(content)) GLOB 'working on it*'
            OR lower(trim(content)) GLOB 'processing*'
            OR lower(trim(content)) GLOB 'checking*'
            OR lower(trim(content)) GLOB 'one moment*'
            OR lower(trim(content)) GLOB 'ping*'
            OR lower(trim(content)) GLOB 'heartbeat*'
            OR lower(trim(content)) GLOB 'no update*'
          )
        ORDER BY store_id ASC
        LIMIT ?
        """,
        (_HEARTBEAT_NOISE_MAX_CHARS, candidate_cap),
    ).fetchall():
        _store_id, _session_id, _source, role, _length, value = row
        reason = heartbeat_noise_reason(str(role or ""), value if isinstance(value, str) else "")
        if reason:
            heartbeat_noise_rows.append(
                make_row(row, field="content", length_key="content_len", category=reason)
            )
        if len(heartbeat_noise_rows) >= limit:
            break

    return {
        "largest_content_rows": [
            make_row(row, field="content", length_key="content_len", category="largest_content")
            for row in largest_content
        ],
        "largest_tool_calls_rows": [
            make_row(row, field="tool_calls", length_key="tool_calls_len", category="largest_tool_calls")
            for row in largest_tool_calls
        ],
        "suspicious_data_uri_content_rows": [
            make_row(row, field="content", length_key="content_len", category="data_uri_base64")
            for row in data_uri_content
        ],
        "suspicious_data_uri_tool_calls_rows": [
            make_row(row, field="tool_calls", length_key="tool_calls_len", category="data_uri_base64")
            for row in data_uri_tool_calls
        ],
        "suspicious_base64_like_rows": generic_rows,
        "quarantined_assistant_rows": quarantined_assistant_rows,
        "suspicious_repetitive_assistant_rows": suspicious_repetitive_assistant_rows,
        "heartbeat_noise_rows": heartbeat_noise_rows,
    }

def externalized_payload_stats(config: Any, hermes_home: str = "") -> dict[str, Any]:
    from .externalize import get_large_output_storage_dir

    storage_dir = get_large_output_storage_dir(config, hermes_home=hermes_home, create=False)
    count = 0
    total_bytes = 0
    total_chars = 0
    latest_path = ""
    latest_mtime = 0.0
    if storage_dir.exists() and storage_dir.is_dir():
        for path in storage_dir.glob("*.json"):
            if not path.is_file():
                continue
            count += 1
            try:
                stat = path.stat()
                total_bytes += int(stat.st_size)
                if stat.st_mtime > latest_mtime:
                    latest_mtime = stat.st_mtime
                    latest_path = str(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                total_chars += int(payload.get("content_chars") or len(payload.get("content", "") or ""))
            except Exception:
                continue
    return {
        "externalized_payload_dir": str(storage_dir),
        "externalized_payload_count": count,
        "externalized_payload_bytes": total_bytes,
        "externalized_payload_chars": total_chars,
        "latest_externalized_payload_path": latest_path,
        "latest_externalized_payload_mtime": latest_mtime,
    }
