"""Tool handlers for LCM — the code that runs when the LLM calls each tool."""

from __future__ import annotations

import copy
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from . import recall_delivery
from .externalize import (
    _inspect_top_level_json_string_fields_before_content as _externalized_top_level_fields_before_content,
    extract_externalized_ref,
    find_externalized_payload_for_message,
    get_large_output_storage_dir,
    load_externalized_payload,
    read_externalized_payload_metadata_prefix,
    read_externalized_payload_search_prefix,
)
from .embedding_provider import (
    VoyageError,
    default_chunk_model,
    probe_provider_availability,
    resolve_provider,
)
from .diagnostics import (
    _has_lifecycle_fragmentation,
    _state_db_path_for_engine,
    doctor_guidance_for_checks,
    survival_fit_count,
)
from .dag import build_nodes_fts_spec
from .db_bootstrap import (
    check_external_content_fts_integrity,
    inspect_lcm_schema_health,
    load_integrity_failed,
)
from .extraction import sanitize_pre_compaction_content
from .inactive_record import inactive_record_notice
from .level3_repair import scan_level3_fragments
from .ingest_protection import (
    EmbeddingIdentityStaleError,
    EmbeddingPrivacyPolicyError,
    embedding_privacy_revision,
    embedding_provider_requires_privacy,
    externalized_payload_stats,
    extract_ingest_externalized_refs,
    protect_embedding_text,
    restore_ingest_payload_placeholders,
    scan_externalized_payload_integrity,
    scan_sqlite_payload_risks,
    sensitive_pattern_status,
    validate_embedding_privacy_dispatch,
)
from . import evidence_tools as _evidence_tools
from . import inspect_tools as _inspect_tools
from . import expansion_evidence
from . import expand_query_tools as _expand_query_tools
from . import expand_tools as _expand_tools
from .presets import preset_status_payload
from . import recent as _recent
from .retrieval_context import RetrievalContext
from .retrieval_core import (
    _hit_identity,
    _lcm_recall_excerpt_expand_hint,
    _lcm_recall_recency_boost as _lcm_recall_recency_boost,  # re-export for direct imports; the prior runs it in retrieval_core
    apply_prior,
    fuse_candidates,
    place_fts_anchor,
    _lcm_grep_confidence,
    _lcm_grep_deadline_error,
    _resolve_semantic_conversation_scope,
    _shape_message_hit,
    _shape_summary_hit,
    hydrate_chunk_hits,
    hydrate_semantic_nodes,
    rrf_fuse,
    run_chunk_knn,
    run_knn,
)
from .search_query import (
    AGE_DECAY_RATE, build_recall_or_query, contains_emoji, normalize_search_sort, recall_content_terms,
    requires_like_fallback, sanitize_fts5_query,
)
from .store import _normalize_observed_at, build_message_fts_spec
from .vector_store import VectorStore
from .config import LCMConfig

if TYPE_CHECKING:
    from .engine import LCMEngine


logger = logging.getLogger(__name__)


def _combined_result_sort_key(result: dict[str, Any], sort: str) -> tuple:
    sort_timestamp = float(result.get("_sort_ts") or 0.0)
    rank = result.get("_sort_rank")
    rank_value = float(rank) if rank is not None else float("inf")
    directness = float(result.get("_sort_directness") or 0.0)
    type_bias = 0 if result.get("type") == "message" else 1
    # BM25 ranks and payload byte offsets are incomparable. Keep the existing
    # history ordering in tier 0; sidecars follow in tier 1 and use byte offset
    # only as their native in-tier tie-break.
    rank_tier = 1 if result.get("type") == "externalized" else 0
    role = result.get("role")
    if role == "user":
        role_bias = 0
    elif role == "assistant":
        role_bias = 1
    elif role == "tool":
        role_bias = 2
    else:
        role_bias = 1

    effective_directness = directness if result.get("type") == "message" else (directness * 0.8)

    if sort == "relevance":
        return (rank_tier, rank_value, -effective_directness, role_bias, -sort_timestamp, type_bias)

    if sort == "hybrid":
        age_hours = max(0.0, (time.time() - sort_timestamp) / 3600.0)
        blended = rank_value / (1 + (age_hours * AGE_DECAY_RATE)) if rank is not None else float("inf")
        summary_override = int(result.get("_hybrid_summary_override") or 0)
        return (
            -summary_override,
            rank_tier,
            blended,
            -effective_directness,
            role_bias,
            -sort_timestamp,
            type_bias,
        )

    if result.get("type") == "message":
        return (rank_tier, -sort_timestamp, type_bias, role_bias, rank_value, 0.0, float("inf"))
    return (rank_tier, -sort_timestamp, type_bias, 0, rank_value, 0.0, role_bias)

def _require_engine(kwargs: Dict[str, Any]) -> "LCMEngine | None":
    engine = kwargs.get("engine")
    return engine if engine is not None else None


def _get_session_node(
    engine: "LCMEngine",
    node_id: int,
    *,
    session_id: str | None = None,
):
    session_id = engine.current_session_id if session_id is None else session_id
    node = engine._dag.get_node(node_id)
    if node is None or node.session_id != session_id:
        return None
    return node


def _session_expand_hint(node_id: int, session_id: str) -> str:
    return f"lcm_expand(node_id={node_id}, session_id={session_id!r})"


def _get_externalized_payload(
    engine: "LCMEngine",
    ref: str,
    *,
    allowed_session_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    payload = load_externalized_payload(ref, config=engine._config, hermes_home=engine._hermes_home)
    if payload is None:
        return None
    payload_session_id = payload.get("session_id") or ""
    allowed = allowed_session_ids
    if not allowed:
        # #680: a ref written before a compression-boundary rotation stays readable.
        lineage = getattr(engine, "_rotation_predecessor_session_ids", None)
        allowed = {engine.current_session_id, *(lineage(engine.current_session_id) if callable(lineage) else [])}
    if payload_session_id and payload_session_id not in allowed:
        return None
    return payload


def _truncate_text_to_token_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    from .tokens import count_tokens

    if max_tokens <= 0 or not text:
        return "", bool(text)

    if count_tokens(text) <= max_tokens:
        return text, False

    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best, True


def _coerce_query_arg(value: Any) -> str:
    """Coerce a tool ``query`` argument to a stripped string.

    A model can emit ``{"query": null}`` or a bare number. Only ``None`` means
    "no query" -- a falsy number such as ``0`` is a legitimate search term, so
    this cannot collapse to ``value or ""``.
    """
    return "" if value is None else str(value).strip()


def _parse_int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_non_negative_int(value: Any, default: int) -> int:
    return max(0, _parse_int_value(value, default))


def _parse_positive_int(value: Any, default: int) -> int:
    return max(1, _parse_int_value(value, default))


def _parse_optional_float(value: Any, name: str) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be a number"


def _parse_optional_timestamp(value: Any, name: str) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    if isinstance(value, (int, float)):
        try:
            return float(value), None
        except (TypeError, ValueError, OverflowError):
            return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    text = str(value).strip()
    if not text:
        return None, f"{name} must not be empty"
    try:
        return float(text), None
    except (TypeError, ValueError, OverflowError):
        pass
    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        return None, f"{name} must be a Unix timestamp or timezone-aware ISO 8601 string"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None, f"{name} ISO timestamp must include a timezone offset or Z"
    return parsed.timestamp(), None


def _parse_grep_role(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    role = str(value or "").strip()
    valid_roles = {"system", "user", "assistant", "tool", "unknown"}
    if role not in valid_roles:
        return None, "role must be one of: system, user, assistant, tool, unknown"
    return role, None


def _parse_retrieval_excluded_session_ids(
    args: Dict[str, Any], current_session_id: str
) -> tuple[set[str], str | None]:
    exclude_current_session = args.get("exclude_current_session", False)
    if not isinstance(exclude_current_session, bool):
        return set(), "exclude_current_session must be a boolean"
    raw_session_ids = args.get("exclude_session_ids", [])
    if not isinstance(raw_session_ids, list):
        return set(), "exclude_session_ids must be an array of strings"
    excluded: set[str] = set()
    for value in raw_session_ids:
        if not isinstance(value, str) or not value.strip():
            return set(), "exclude_session_ids must contain only non-empty strings"
        excluded.add(value.strip())
    if exclude_current_session and current_session_id:
        excluded.add(current_session_id)
    return excluded, None


# Tables whose distinct ``session_id`` values define the eligible retrieval
# scope. Both carry a ``session_id``-LEADING index (``idx_msg_session_ts`` and
# ``idx_nodes_session_latest``), which is what makes the skip-scan below an
# indexed seek rather than a corpus scan. Fixed literals -- never interpolate
# caller input into the statement.
_RETRIEVAL_SESSION_SCOPE_TABLES = ("messages", "summary_nodes")

# Loose index scan ("skip scan"): seek once per DISTINCT session_id instead of
# reading every row and sorting a temp B-tree. Cost is O(sessions * log rows),
# not O(rows). Measured on 300k messages + 9k summary nodes over 300 sessions:
# 66.2 ms (SCAN + USE TEMP B-TREE FOR DISTINCT) -> 0.5 ms (SEARCH ... session_id>?).
_RETRIEVAL_DISTINCT_SESSIONS_SQL = """
        WITH RECURSIVE distinct_sessions(session_id) AS (
            SELECT MIN(session_id) FROM {table}
            UNION ALL
            SELECT (
                SELECT MIN(session_id) FROM {table}
                WHERE session_id > distinct_sessions.session_id
            )
            FROM distinct_sessions
            WHERE distinct_sessions.session_id IS NOT NULL
        )
        SELECT session_id FROM distinct_sessions WHERE session_id IS NOT NULL
"""


def _retrieval_candidate_session_ids(
    engine: "LCMEngine",
    excluded_session_ids: set[str],
    *,
    deadline: float | None = None,
) -> tuple[list[str] | None, bool]:
    """Resolve the eligible-session whitelist for a retrieval exclusion set.

    Returns ``(candidate_session_ids, enumeration_timed_out)``. ``None`` means
    "no exclusions, no whitelist needed"; ``[]`` means "exclusions left no
    eligible session". The second element is True only when ``deadline`` lapsed
    mid-enumeration -- callers must fail CLOSED on it (drop the arms that depend
    on the whitelist), never fall back to ``None``, which would silently
    discard the caller's hard exclusion filter.
    """
    if not excluded_session_ids:
        return None, False
    # Liveness gate only. The enumeration itself runs on its OWN connection
    # (below); a closed store still means "no eligible session", fail closed.
    if engine._store.connection is None:
        return [], False
    # An already-lapsed budget fails closed before any DB work: the progress
    # handler only fires every N opcodes, so a short query could otherwise slip
    # past an expired deadline purely because it was cheap.
    if deadline is not None and time.monotonic() >= deadline:
        return [], True
    sessions: set[str] = set()
    # A progress handler is CONNECTION-GLOBAL, and ``MessageStore._conn`` is
    # explicitly shared across threads (opened ``check_same_thread=False``).
    # Installing this request's deadline there would let it interrupt a
    # concurrent read/ingest on the same connection, and the `finally` teardown
    # would clear a concurrent request's handler -- silently disabling ITS
    # timeout. So the enumeration gets a private read-only connection and owns
    # its handler alone: the same idiom every other deadline-guarded read path
    # here uses (``_lcm_recall_summary_sources``, the chunk/summary hydrators in
    # ``retrieval_core``). Python's sqlite3 exposes no getter for the installed
    # handler, so save/restore on the shared connection is not expressible.
    connection: sqlite3.Connection | None = None
    try:
        db_path = Path(engine._store.db_path).resolve()
        connection = sqlite3.connect(
            f"{db_path.as_uri()}?mode=ro",
            uri=True,
            timeout=(
                5.0
                if deadline is None
                else max(0.001, deadline - time.monotonic())
            ),
        )
        connection.execute("PRAGMA query_only=ON")
        if deadline is not None:
            connection.set_progress_handler(
                lambda: 1 if time.monotonic() >= deadline else 0, 1000
            )
        for table in _RETRIEVAL_SESSION_SCOPE_TABLES:
            rows = connection.execute(
                _RETRIEVAL_DISTINCT_SESSIONS_SQL.format(table=table)
            ).fetchall()
            sessions.update(str(row[0]) for row in rows if row[0] is not None)
    except sqlite3.OperationalError:
        if deadline is not None and time.monotonic() >= deadline:
            return [], True
        raise
    finally:
        if connection is not None:
            connection.close()
    # The exclusion difference moves to Python: both ``session_id`` columns are
    # declared TEXT NOT NULL, so set difference on the stringified values is
    # exactly the SQL ``NOT IN`` it replaces, without a second pass over rows.
    return sorted(sessions - excluded_session_ids), False


def _parse_strict_int(value: Any, name: str) -> tuple[int | None, str | None]:
    try:
        if isinstance(value, bool):
            raise ValueError
        return int(value), None
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be an integer"


_LCM_GREP_VALID_SCOPES = frozenset({"current", "all", "session"})
_LCM_GREP_VALID_CONTENT_SCOPES = frozenset({"history", "externalized", "both"})
_LCM_GREP_HARD_LIMIT_CAP = 200
_LCM_GREP_EXTERNALIZED_FILE_CAP = 256
_LCM_GREP_EXTERNALIZED_DISCOVERY_CAP = 4096
_LCM_GREP_EXTERNALIZED_METADATA_READ_BYTES = 64 * 1024
_LCM_GREP_EXTERNALIZED_CONTENT_BYTES = 512_000
_LCM_GREP_EXTERNALIZED_DOCUMENT_TAIL_BYTES = 64 * 1024
_LCM_GREP_RESPONSE_CHAR_CAP = 64_000
_LCM_RECENT_DEFAULT_LIMIT = 10
_LCM_RECENT_HARD_LIMIT_CAP = 200
_LCM_RECENT_FRONTIER_WORK_LIMIT = 4096
_LCM_GREP_HYBRID_CANDIDATE_CAP = 500
_LCM_GREP_SEMANTIC_SNIPPET_CHARS = 300
_LCM_GREP_RRF_K = 60
# -- lcm_recall (cross-conversation forever-memory recall) --
_LCM_RECALL_DEFAULT_LIMIT = 8
_LCM_RECALL_LIMIT_CAP = 25
_LCM_RECALL_DEFAULT_SCOPE_BIAS = 0.5
_LCM_RECALL_SNIPPET_CHARS = 300
_LCM_RECALL_RESPONSE_CHAR_CAP = 64_000
_LCM_QUERY_STATE_DEFAULT_LIMIT = 25
_LCM_QUERY_STATE_LIMIT_CAP = 50
_LCM_QUERY_STATE_RESPONSE_CHAR_CAP = 64_000
_LCM_COMPUTE_RESPONSE_CHAR_CAP = 64_000
_LCM_RECALL_VALID_INCLUDE = frozenset({"all", "summaries", "verbatim"})
# A slow first FTS arm must not consume the whole shared recall deadline when a
# real semantic route is configured. FTS-only/degraded installs keep the full
# budget; hybrid recall reserves the remainder for provider resolution, vector
# arms, hydration, fusion, and optional reranking.
_LCM_RECALL_FTS_MAX_BUDGET_FRACTION = 0.25
# The vector-corpus preflight guards the budget split above, so its own scan
# must never be able to spend the request budget it exists to protect: a
# no-match probe over a large fully-orphaned corpus has to exhaust candidates
# despite LIMIT 1, and aborting that scan only at the REQUEST deadline would
# hand FTS an already-expired budget — no arm runs at all. The preflight
# therefore gets a small independent scan budget; expiry fails open to False,
# which keeps the full remaining budget with the FTS arm.
_LCM_RECALL_PREFLIGHT_MAX_BUDGET_S = 0.25
_LCM_RECALL_PREFLIGHT_BUDGET_FRACTION = 0.1
_LCM_RECALL_VALID_DETAIL = frozenset({"snippets", "answer_ready"})
_LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT = 5
_LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT = 8
# Bounded per-node fan-out when reference-strict delivery carries summary-KNN
# relevance onto the source messages beneath a ranked node. Small on purpose:
# the point is to make the SESSION reachable with citable evidence, and the FTS
# and chunk arms are what rank individual messages inside it.
_LCM_RECALL_SUMMARY_SOURCE_PER_NODE = 4
# How many candidate rows reference-strict selection reads per batch while it
# walks the ranking. Verification needs the row, so the walk reads AHEAD of the
# cursor in waves: a bounded number of batched reads per request rather than one
# read per candidate it has to skip.
_LCM_RECALL_STRICT_READ_WAVE = 32
_LCM_RECALL_ANSWER_READY_CONTENT_CHARS = 2_400
_LCM_LOAD_SESSION_DEFAULT_LIMIT = 100
_LCM_LOAD_SESSION_HARD_LIMIT_CAP = 200
_LCM_LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS = 4000
_LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS = 20_000
_LCM_RECENT_MAX_RESPONSE_CHARS = _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS
_LCM_INSPECT_DEFAULT_LIMIT = 20
_LCM_INSPECT_HARD_LIMIT_CAP = 200
_LCM_EXPAND_QUERY_SESSION_ID_CAP = 20
_LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT = 10_000
_LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES = 16_384
_LCM_INSPECT_MAX_RESPONSE_CHARS = 20_000
_OPERATOR_TEXT_FIELD_MAX_CHARS = 1_000


def lcm_query_state(args: Dict[str, Any], **kwargs) -> str:
    """Return bounded typed assertion state with exact source provenance."""
    return _evidence_tools.lcm_query_state(args, **kwargs)


def lcm_compute(args: Dict[str, Any], **kwargs) -> str:
    """Run a pure operation over exact, selector-supplied evidence refs."""
    return _evidence_tools.lcm_compute(args, **kwargs)


def lcm_evidence_pack(args: Dict[str, Any], **kwargs) -> str:
    """Build a bounded exact-evidence packet and optional canonical trace."""
    return _evidence_tools.lcm_evidence_pack(args, **kwargs)


def lcm_compile_evidence(args: Dict[str, Any], **kwargs) -> str:
    """Compile evidence through legacy proposal or deterministic auto mode."""
    return _evidence_tools.lcm_compile_evidence(args, **kwargs)


_LCM_RETRIEVE_RESPONSE_CHAR_CAP = 64_000
_LCM_RETRIEVE_ARGUMENTS = frozenset({
    "action",
    "retrieval_id",
    "question",
    "question_date",
    "identity",
    "requirements",
    "missing_slot",
    "tool",
    "tool_args",
    "resolved_slots",
    "selected_refs",
    "computation",
})


def lcm_retrieve(args: Dict[str, Any], **kwargs) -> str:
    """Drive one bounded retrieval episode inside the answerer's tool turn."""
    return _evidence_tools.lcm_retrieve(args, **kwargs)


def _slice_content_for_response(content: str, max_tokens: int, content_offset: int = 0) -> dict[str, Any]:
    content = content or ""
    content_offset = min(max(0, content_offset), len(content))
    sliced, _ = _truncate_text_to_token_budget(content[content_offset:], max_tokens)
    if not sliced and content_offset < len(content):
        # A tiny token budget can fail to fit even the next character. Return one
        # character anyway so callers make deterministic, lossless cursor progress
        # instead of receiving has_more=true with the same content_offset forever.
        sliced = content[content_offset:content_offset + 1]
    next_content_offset = content_offset + len(sliced)
    has_more = next_content_offset < len(content)
    return {
        "content": sliced,
        "content_chars": len(content),
        "content_offset": content_offset,
        "content_returned_chars": len(sliced),
        "content_truncated": has_more,
        "next_content_offset": next_content_offset if has_more else 0,
        "has_more": has_more,
    }


def _query_terms_for_match_window(query: str | None) -> list[str]:
    if not query:
        return []
    terms: list[str] = []
    normalized_query = " ".join(re.findall(r"\w+", query))
    if normalized_query:
        terms.append(normalized_query)

    def add_term(term: str) -> None:
        term = term.strip()
        if not term:
            return
        terms.append(term)
        parts = [part for part in re.split(r"[^\w]+", term) if part]
        if len(parts) > 1:
            terms.append(" ".join(parts))
        terms.extend(part for part in parts if len(part) >= 2)

    for quoted in re.findall(r'"([^"]+)"', query):
        add_term(quoted)
    for token in re.findall(r"[\w][\w:-]*\*?", query):
        token = token.rstrip("*").strip()
        if not token or token.upper() in {"AND", "OR", "NOT", "NEAR"}:
            continue
        if ":" in token:
            token = token.rsplit(":", 1)[-1]
        if len(token) >= 2:
            add_term(token)
    seen: set[str] = set()
    unique: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


# Between the tokens of a quoted phrase: an FTS5 phrase matches adjacent tokens across punctuation and line breaks.
# Only characters that are certainly unicode61 separators count (whitespace, ASCII punctuation); marks and
# private-use characters stay inside a token, and any other gap falls back to the raw-query offset.
_RECALL_PHRASE_GAP = r"[\s!-/:-@\[-`{-~]+"


def _content_offset_for_query_match(content: str, query: str | None) -> int:
    folded = content.casefold()
    for term in _query_terms_for_match_window(query):
        index = folded.find(term.casefold())
        if index >= 0:
            return index
    return 0


def _full_content_slice(content: str, content_offset: int = 0) -> dict[str, Any]:
    content = content or ""
    content_offset = min(max(0, content_offset), len(content))
    sliced = content[content_offset:]
    return {
        "content": sliced,
        "content_chars": len(content),
        "content_offset": content_offset,
        "content_returned_chars": len(sliced),
        "content_truncated": False,
        "next_content_offset": 0,
        "has_more": False,
    }


def _restore_ingest_placeholder_for_lookup(
    content: str,
    ref: str | None,
    payload: dict[str, Any] | None,
    *,
    config,
    hermes_home: str,
    session_id: str,
) -> str | None:
    if not content or not ref or not payload or payload.get("kind") != "ingest_payload":
        return None
    restored = restore_ingest_payload_placeholders(
        content,
        config=config,
        hermes_home=hermes_home,
        session_id=session_id,
    )
    return restored if restored != content else None


def _is_compact_externalized_marker(content: str, ref: str | None) -> bool:
    if not ref or not content:
        return False
    if len(content) > 512:
        return False
    return (
        content.startswith("[Externalized tool output:")
        or content.startswith("[GC'd externalized tool output:")
        or content.startswith("[Externalized payload:")
        or content.startswith("[GC'd externalized payload:")
        or "[Externalized LCM ingest payload:" in content
    )


def _expand_message_sources(
    engine: "LCMEngine",
    node,
    max_tokens: int,
    *,
    source_offset: int = 0,
    source_limit: int | None = None,
    content_offset: int = 0,
    hydrate_externalized_content: bool = False,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .tokens import count_tokens

    # ``session_id`` selects the lookup scope: which session owns these rows and
    # may hydrate their externalized payloads. ``from_current_session`` answers a
    # separate question — whether a row belongs to the ACTIVE session — so a
    # cross-session expansion must not advertise archived rows as current.
    current_session_id = engine.current_session_id
    session_id = current_session_id if session_id is None else session_id
    total_sources = len(node.source_ids)
    source_offset = min(max(0, source_offset), total_sources)
    remaining_source_count = max(0, total_sources - source_offset)
    if source_limit is None:
        source_limit = remaining_source_count
    else:
        source_limit = min(max(0, source_limit), remaining_source_count)
    content_offset = max(0, content_offset)
    source_ids = node.source_ids[source_offset:source_offset + source_limit]
    stored_by_id = engine._store.get_batch(source_ids)

    messages: list[dict[str, Any]] = []
    budget_used = 0
    next_source_offset: int | None = source_offset
    next_content_offset = content_offset
    has_more = source_offset < total_sources

    for relative_index, store_id in enumerate(source_ids):
        source_index = source_offset + relative_index
        remaining_tokens = max_tokens - budget_used
        if remaining_tokens <= 0:
            next_source_offset = source_index
            next_content_offset = 0
            has_more = True
            break
        stored = stored_by_id.get(store_id)
        if not stored:
            next_source_offset = source_index + 1
            next_content_offset = 0
            has_more = next_source_offset < total_sources
            continue
        transcript_content = stored.get("content", "")
        content = transcript_content
        content_source = "message"
        externalized = None
        ref_payload = None
        ingest_refs = extract_ingest_externalized_refs(transcript_content)
        ref = ingest_refs[0] if ingest_refs else extract_externalized_ref(transcript_content)
        if ref:
            ref_payload = _get_externalized_payload(
                engine,
                ref,
                allowed_session_ids={session_id, stored.get("session_id", "")},
            )
            if ref_payload is not None and ref_payload.get("kind") != "ingest_payload":
                externalized = ref_payload
        if hydrate_externalized_content and externalized is not None:
            content = externalized.get("content", "")
            content_source = "externalized_payload"
        effective_content_offset = content_offset if source_index == source_offset else 0
        if not hydrate_externalized_content and _is_compact_externalized_marker(content, ref):
            sliced = _full_content_slice(content, effective_content_offset)
        else:
            sliced = _slice_content_for_response(content, remaining_tokens, effective_content_offset)
        expanded = {
            "store_id": stored["store_id"],
            "source_index": source_index,
            "session_id": stored.get("session_id", ""),
            "source": stored.get("source") or "",
            "from_current_session": bool(current_session_id)
            and stored.get("session_id", "") == current_session_id,
            "role": stored["role"],
            "content": sliced["content"],
            "content_chars": sliced["content_chars"],
            "content_offset": sliced["content_offset"],
            "content_returned_chars": sliced["content_returned_chars"],
            "content_truncated": sliced["content_truncated"],
            "next_content_offset": sliced["next_content_offset"],
            "content_source": content_source,
        }
        if content_source == "externalized_payload":
            expanded["transcript_content"] = transcript_content
        if stored.get("role") == "tool":
            if externalized is not None:
                externalized_summary = dict(externalized)
                externalized_summary.pop("content", None)
                expanded["externalized"] = externalized_summary
            if "externalized" not in expanded:
                lookup_candidates = [transcript_content]
                restored_ingest_content = _restore_ingest_placeholder_for_lookup(
                    transcript_content,
                    ref,
                    ref_payload,
                    config=engine._config,
                    hermes_home=engine._hermes_home,
                    session_id=stored.get("session_id", ""),
                )
                if restored_ingest_content is not None:
                    lookup_candidates.insert(0, restored_ingest_content)
                    sanitized_restored = sanitize_pre_compaction_content(restored_ingest_content)
                    if sanitized_restored != restored_ingest_content:
                        lookup_candidates.insert(0, sanitized_restored)
                sanitized_content = sanitize_pre_compaction_content(transcript_content)
                if sanitized_content != transcript_content:
                    lookup_candidates.insert(0, sanitized_content)
                for candidate in lookup_candidates:
                    externalized = find_externalized_payload_for_message(
                        candidate,
                        tool_call_id=stored.get("tool_call_id", ""),
                        session_id=stored.get("session_id", ""),
                        config=engine._config,
                        hermes_home=engine._hermes_home,
                    )
                    if externalized is not None:
                        expanded["externalized"] = externalized
                        break
        messages.append(expanded)
        budget_used += count_tokens(sliced["content"])
        if sliced["has_more"]:
            next_source_offset = source_index
            next_content_offset = sliced["next_content_offset"]
            has_more = True
            break
        next_source_offset = source_index + 1
        next_content_offset = 0
        has_more = next_source_offset < total_sources
    else:
        has_more = (source_offset + source_limit) < total_sources
        next_source_offset = source_offset + source_limit if has_more else None
        next_content_offset = 0

    pagination = expansion_evidence._pagination_payload(
        total_sources=total_sources,
        source_offset=source_offset,
        content_offset=content_offset,
        source_limit=source_limit,
        returned_sources=len(messages),
        next_source_offset=next_source_offset,
        next_content_offset=next_content_offset,
        has_more=has_more,
    )
    return messages, pagination


def _expand_child_nodes(
    engine: "LCMEngine",
    node,
    max_tokens: int | None = None,
    *,
    source_offset: int = 0,
    source_limit: int | None = None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .tokens import count_tokens

    session_id = engine.current_session_id if session_id is None else session_id
    total_sources = len(node.source_ids)
    source_offset = min(max(0, source_offset), total_sources)
    remaining_source_count = max(0, total_sources - source_offset)
    if source_limit is None:
        source_limit = remaining_source_count
    else:
        source_limit = min(max(0, source_limit), remaining_source_count)
    selected_source_ids = node.source_ids[source_offset : source_offset + source_limit]
    children: list[tuple[int, Any]] = []
    for relative_index, child_id in enumerate(selected_source_ids):
        child = engine._dag.get_node(child_id)
        if child is None or child.session_id != session_id:
            continue
        children.append((source_offset + relative_index, child))

    expanded: list[dict[str, Any]] = []
    budget_used = 0
    next_source_offset: int | None = None
    has_more = (source_offset + source_limit) < total_sources
    for source_index, child in children:
        summary = child.summary
        summary_truncated = False
        if max_tokens is not None:
            remaining_tokens = max_tokens - budget_used
            if remaining_tokens <= 0:
                next_source_offset = source_index
                has_more = True
                break
            summary, summary_truncated = _truncate_text_to_token_budget(
                summary, remaining_tokens
            )
        expanded.append(
            {
                "node_id": child.node_id,
                "source_index": source_index,
                "depth": child.depth,
                "summary": summary[:1000] if max_tokens is None else summary,
                "summary_truncated": summary_truncated
                or (max_tokens is None and len(child.summary) > 1000),
                "token_count": child.token_count,
                "source_token_count": child.source_token_count,
                "expand_hint": child.expand_hint,
            }
        )
        budget_used += count_tokens(summary)
        if summary_truncated:
            next_source_offset = source_index + 1
            has_more = next_source_offset < total_sources
            break
        next_source_offset = source_index + 1

    if has_more and next_source_offset is None:
        next_source_offset = source_offset + source_limit

    return expanded, expansion_evidence._pagination_payload(
        total_sources=total_sources,
        source_offset=source_offset,
        content_offset=0,
        source_limit=source_limit,
        returned_sources=len(expanded),
        next_source_offset=next_source_offset,
        next_content_offset=0,
        has_more=has_more,
    )


def _parse_load_session_roles(value: Any) -> tuple[list[str], str | None]:
    if value is None:
        return [], None
    if not isinstance(value, list):
        return [], "roles must be an array of strings"
    roles: list[str] = []
    seen: set[str] = set()
    for item in value:
        role = str(item or "").strip()
        if not role:
            return [], "roles must contain only non-empty strings"
        if role not in seen:
            roles.append(role)
            seen.add(role)
    return roles, None


def _slice_loaded_content(content: Any, max_content_chars: int) -> dict[str, Any]:
    text = content or ""
    sliced = text[:max_content_chars]
    has_more = len(sliced) < len(text)
    return {
        "content": sliced,
        "content_chars": len(text),
        "content_returned_chars": len(sliced),
        "content_truncated": has_more,
        "next_content_offset": len(sliced) if has_more else 0,
    }


def _serialize_loaded_message(
    engine: "LCMEngine",
    row: dict[str, Any],
    max_content_chars: int,
    *,
    include_exact_ref: bool = False,
) -> dict[str, Any]:
    stored_session_id = row.get("session_id", "")
    content_slice = _slice_loaded_content(row.get("content", "") or "", max_content_chars)
    item: dict[str, Any] = {
        "store_id": row.get("store_id"),
        "session_id": stored_session_id,
        "source": row.get("source") or "",
        "role": row.get("role"),
        "timestamp": row.get("timestamp", 0),
        "content": content_slice["content"],
        "content_chars": content_slice["content_chars"],
        "content_returned_chars": content_slice["content_returned_chars"],
        "content_truncated": content_slice["content_truncated"],
        "next_content_offset": content_slice["next_content_offset"],
        "from_current_session": bool(engine.current_session_id) and stored_session_id == engine.current_session_id,
    }
    if row.get("tool_call_id"):
        item["tool_call_id"] = row.get("tool_call_id")
    if row.get("tool_calls"):
        item["tool_calls"] = row.get("tool_calls")
    if row.get("tool_name"):
        item["tool_name"] = row.get("tool_name")
    store_id = row.get("store_id")
    if include_exact_ref and isinstance(store_id, int) and content_slice["content_returned_chars"] > 0:
        item["exact_ref"] = f"lcm:{store_id}:0-{content_slice['content_returned_chars']}"
    return item


def lcm_load_session(args: Dict[str, Any], **kwargs) -> str:
    """Load an ordered, bounded raw-message page for one explicit session_id."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    session_id = str(args.get("session_id") or "").strip()
    if not session_id:
        return json.dumps({"error": "session_id is required"})

    raw_limit_arg = args.get("limit", _LCM_LOAD_SESSION_DEFAULT_LIMIT)
    parsed_limit, limit_error = _parse_strict_int(raw_limit_arg, "limit")
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_LOAD_SESSION_HARD_LIMIT_CAP)

    raw_max_content_chars = args.get("max_content_chars", _LCM_LOAD_SESSION_DEFAULT_MAX_CONTENT_CHARS)
    max_content_chars, max_content_error = _parse_strict_int(raw_max_content_chars, "max_content_chars")
    if max_content_error:
        return json.dumps({"error": max_content_error})
    if max_content_chars is None or max_content_chars <= 0:
        return json.dumps({"error": "max_content_chars must be a positive integer"})
    requested_max_content_chars = max_content_chars
    max_content_chars = min(max_content_chars, _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS)

    after_store_id, cursor_error = _parse_strict_int(args.get("after_store_id", 0), "after_store_id")
    if cursor_error:
        return json.dumps({"error": cursor_error})
    if after_store_id is None or after_store_id < 0:
        return json.dumps({"error": "after_store_id must be a non-negative integer"})

    roles, roles_error = _parse_load_session_roles(args.get("roles"))
    if roles_error:
        return json.dumps({"error": roles_error})

    time_from, time_from_error = _parse_optional_float(args.get("time_from"), "time_from")
    if time_from_error:
        return json.dumps({"error": time_from_error})
    time_to, time_to_error = _parse_optional_float(args.get("time_to"), "time_to")
    if time_to_error:
        return json.dumps({"error": time_to_error})
    if time_from is not None and time_to is not None and time_to < time_from:
        return json.dumps({"error": "time_to must be greater than or equal to time_from"})
    raw_include_exact_ref = args.get("include_exact_ref", False)
    if not isinstance(raw_include_exact_ref, bool):
        return json.dumps({"error": "include_exact_ref must be a boolean"})
    include_exact_ref = raw_include_exact_ref

    total_messages = engine._store.count_session_load_messages(
        session_id,
        roles=roles or None,
        time_from=time_from,
        time_to=time_to,
    )
    rows = engine._store.load_session_page(
        session_id,
        after_store_id=after_store_id,
        limit=limit + 1,
        roles=roles or None,
        time_from=time_from,
        time_to=time_to,
    )
    page_rows = rows[:limit]
    has_more = len(rows) > limit
    next_cursor = page_rows[-1]["store_id"] if has_more and page_rows else None

    response: dict[str, Any] = {
        "session_id": session_id,
        "limit": limit,
        "max_content_chars": max_content_chars,
        "after_store_id": after_store_id,
        "total_messages": total_messages,
        "returned_messages": len(page_rows),
        "messages": [
            _serialize_loaded_message(
                engine,
                row,
                max_content_chars,
                include_exact_ref=include_exact_ref,
            )
            for row in page_rows
        ],
        "next_cursor": next_cursor,
        "has_more": has_more,
    }
    if roles:
        response["roles"] = roles
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
    if requested_limit > _LCM_LOAD_SESSION_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    if requested_max_content_chars > _LCM_LOAD_SESSION_HARD_MAX_CONTENT_CHARS:
        response["max_content_chars_clamped_from"] = requested_max_content_chars
    return json.dumps(response)


def lcm_recent(args: Dict[str, Any], **kwargs) -> str:
    """Serve conversation rollups or fall back; cross-session rollups are future work."""
    return _recent.lcm_recent(args, **kwargs)


def _lcm_grep_full_text(args: Dict[str, Any], **kwargs) -> str:
    """Search raw messages + summaries with optional cross-session scoping.

    Default scope is the current session, preserving historical behavior and returning
    both raw-message and summary-node hits. Callers may explicitly request
    ``session_scope='all'`` (every session in the local LCM database) or
    ``session_scope='session'`` (a single ``session_id``); broader scopes return
    raw-message hits only and exist for bounded archive recovery over rows already
    present in ``lcm.db``. ``limit`` is clamped to ``_LCM_GREP_HARD_LIMIT_CAP``
    regardless of input.
    """
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = _coerce_query_arg(args.get("query"))
    if not query:
        return json.dumps({"error": "No query provided"})

    raw_limit_arg = args.get("limit", 10)
    parsed_limit = _parse_int_value(raw_limit_arg, 10)
    if parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit_cap = int(kwargs.get("_limit_cap", _LCM_GREP_HARD_LIMIT_CAP))
    limit = min(requested_limit, limit_cap)
    sort = normalize_search_sort(args.get("sort"))
    source_limit = max(limit * 4, limit, 20)

    content_scope = str(args.get("content_scope") or "history").strip().lower()
    if content_scope not in _LCM_GREP_VALID_CONTENT_SCOPES:
        return json.dumps({"error": "content_scope must be one of: history, externalized, both"})
    raw_externalized_refs = args.get("externalized_refs")
    if raw_externalized_refs == []:
        # An empty optional array means "not provided" (#1194).
        raw_externalized_refs = None
    if raw_externalized_refs is not None and content_scope == "history":
        return json.dumps({"error": "externalized_refs requires content_scope=externalized or both"})
    externalized_refs: list[str] | None = None
    if raw_externalized_refs is not None:
        if not isinstance(raw_externalized_refs, list):
            return json.dumps({"error": "externalized_refs must be an array of ref filenames"})
        if len(raw_externalized_refs) > _LCM_GREP_EXTERNALIZED_FILE_CAP:
            return json.dumps({"error": f"externalized_refs is limited to {_LCM_GREP_EXTERNALIZED_FILE_CAP} refs"})
        externalized_refs = []
        for value in raw_externalized_refs:
            ref = str(value or "").strip()
            if (
                not ref
                or Path(ref).name != ref
                or "/" in ref
                or "\\" in ref
                or not ref.endswith(".json")
            ):
                return json.dumps({"error": f"Invalid externalized ref: {ref or '<empty>'}"})
            if ref not in externalized_refs:
                externalized_refs.append(ref)

    requested_session_scope = str(args.get("session_scope", "current")).lower()
    raw_session_id_arg = args.get("session_id")
    explicit_session_id = (
        str(raw_session_id_arg).strip() if raw_session_id_arg is not None else ""
    )
    source = str(args.get("source") or "").strip() or None
    conversation_id = str(args.get("conversation_id") or "").strip() or None
    role, role_error = _parse_grep_role(args.get("role"))
    if role_error:
        return json.dumps({"error": role_error})
    time_from, time_from_error = _parse_optional_timestamp(args.get("time_from"), "time_from")
    if time_from_error:
        return json.dumps({"error": time_from_error})
    time_to, time_to_error = _parse_optional_timestamp(args.get("time_to"), "time_to")
    if time_to_error:
        return json.dumps({"error": time_to_error})
    if time_from is not None and time_to is not None and time_to < time_from:
        return json.dumps({"error": "time_to must be greater than or equal to time_from"})
    raw_message_filter_active = (
        role is not None
        or time_from is not None
        or time_to is not None
        or conversation_id is not None
    )
    externalized_filter_active = raw_message_filter_active or source is not None

    if requested_session_scope == "current":
        if explicit_session_id:
            return json.dumps({
                "error": "session_id is only valid with session_scope=session",
            })
        # MessageStore.search and SummaryDAG.search treat session_id="" as a
        # literal scoped filter, so an unbound engine searching scope=current
        # returns zero results rather than leaking cross-session matches.
        # Read current_session_id (the foreground view) so a cron-style side
        # channel that briefly owns engine._session_id does not redirect the
        # default search scope away from the operator's real conversation.
        search_session_id: str | None = engine.current_session_id
        session_scope = "current"
    elif requested_session_scope == "all":
        if explicit_session_id:
            return json.dumps({
                "error": "session_id is not used with session_scope=all",
            })
        search_session_id = None
        session_scope = "all"
    elif requested_session_scope == "session":
        if not explicit_session_id:
            return json.dumps({
                "error": "session_scope=session requires session_id",
            })
        search_session_id = explicit_session_id
        session_scope = "session"
    else:
        # Preserve historical behavior for unknown scopes: route through the
        # current-session path and report. The data-layer empty-string scoping
        # contract keeps an unbound engine from leaking cross-session matches
        # here too.
        search_session_id = engine.current_session_id
        session_scope = "current"
        logger.warning(
            "Ignoring unsupported session_scope=%s for lcm_grep",
            requested_session_scope,
        )

    searches_externalized = content_scope in {"externalized", "both"}
    if searches_externalized and session_scope != "current":
        return json.dumps({
            "error": "Externalized payload search supports session_scope=current only",
        })
    if searches_externalized and not engine.current_session_id:
        return json.dumps({"error": "Externalized payload search requires an active session"})

    current_session_id = engine.current_session_id
    has_current_session = bool(current_session_id)
    excluded_session_ids = set(args.get("_excluded_session_ids") or ())
    results: list[Dict[str, Any]] = []
    message_search_error: str | None = None

    if (
        content_scope in {"history", "both"}
        and search_session_id not in excluded_session_ids
    ):
        try:
            msg_hits = engine._store.search(
                query,
                session_id=search_session_id,
                limit=source_limit,
                sort=sort,
                source=source,
                conversation_id=conversation_id,
                role=role,
                time_from=time_from,
                time_to=time_to,
                exclude_session_ids=excluded_session_ids,
                allow_operators=bool(args.get("_allow_operators", False)),
                **({"stemmed": True} if args.get("_stemmed") else {}),
                **({"agent_operators": True} if args.get("_agent_operators") else {}),
            )
            for hit in msg_hits:
                if hit.get("session_id") in excluded_session_ids:
                    continue
                results.append(
                    _shape_message_hit(
                        hit,
                        current_session_id=current_session_id,
                        has_current_session=has_current_session,
                    )
                )
        except Exception as exc:
            # Grep degrades to whatever the other stages produce, but the
            # failure must reach the PAYLOAD, not just the log: lcm_recall's
            # FTS arm consumes only these message rows, so a swallowed crash
            # here previously reported coverage.fts='ok' with zero hits --
            # on an FTS-only install that made a crashed search arm
            # indistinguishable from an empty corpus (#256).
            message_search_error = str(exc) or type(exc).__name__
            logger.warning("Message search failed: %s", exc)

    # Summary-node search is intentionally current-session only. Cross-session
    # DAG expansion is deferred; returning summary hits without an expansion
    # contract would push this tool toward a memory-system shape rather than
    # a plugin-local archive search. Raw-message hits remain expandable across
    # sessions via lcm_expand(store_id=...).
    if (
        content_scope in {"history", "both"}
        and session_scope == "current"
        and not raw_message_filter_active
        and search_session_id not in excluded_session_ids
    ):
        try:
            node_hits = engine._dag.search(
                query,
                session_id=search_session_id,
                limit=source_limit,
                sort=sort,
                source=source,
                **({"agent_operators": True} if args.get("_agent_operators") else {}),
                **({"allow_operators": True} if args.get("_grep_any_term") else {}),
            )
            for node in node_hits:
                if node.session_id in excluded_session_ids:
                    continue
                results.append(_shape_summary_hit(node))
        except Exception as exc:
            logger.warning("Node search failed: %s", exc)

    externalized_scan: dict[str, Any] | None = None
    externalized_results_omitted = False
    if searches_externalized and externalized_filter_active:
        # Externalized sidecars are tool/ingest payloads, not raw messages, and
        # carry no role/timestamp/source/conversation lane comparable to history
        # rows. Suppress sidecar search whenever one of those filters is active
        # rather than leak unscoped payloads. Source remains valid for summary
        # search above, so it intentionally does not affect summary omission.
        externalized_results_omitted = True
    elif searches_externalized and current_session_id not in excluded_session_ids:
        try:
            storage_dir = get_large_output_storage_dir(
                engine._config,
                hermes_home=engine._hermes_home,
                create=False,
            )
        except (OSError, ValueError) as exc:
            return json.dumps({
                "error": f"Externalized payload storage is unavailable: {exc}",
            })
        scan_counts = {
            "candidate_files": 0,
            "discovery_files": 0,
            "discovery_truncated": False,
            "scanned_files": 0,
            "matched_files": 0,
            "rejected_symlink": 0,
            "rejected_invalid_or_unreadable": 0,
            "rejected_session_mismatch": 0,
        }
        if externalized_refs is None:
            discovered_refs = []
            try:
                for entries_seen, path in enumerate(storage_dir.iterdir(), start=1):
                    if entries_seen > _LCM_GREP_EXTERNALIZED_DISCOVERY_CAP:
                        scan_counts["discovery_truncated"] = True
                        break
                    if not path.name.endswith(".json"):
                        continue
                    discovered_refs.append(path.name)
            except OSError:
                discovered_refs = []
            # Bound directory consumption before sorting so one busy payload
            # directory cannot make a search materialize every sidecar name.
            discovered_refs.sort(reverse=True)
            candidate_refs = []
            for ref in discovered_refs:
                scan_counts["discovery_files"] += 1
                path = storage_dir / ref
                if path.is_symlink():
                    scan_counts["rejected_symlink"] += 1
                    continue
                metadata = _inspect_externalized_payload_metadata(
                    engine,
                    ref,
                    engine.current_session_id,
                    max_read_bytes=_LCM_GREP_EXTERNALIZED_METADATA_READ_BYTES,
                )
                if metadata.get("readable"):
                    try:
                        payload = read_externalized_payload_search_prefix(
                            ref,
                            config=engine._config,
                            hermes_home=engine._hermes_home,
                            # Re-open with no-follow/inode checks after strict
                            # metadata validation; the candidate scan reads content.
                            max_content_bytes=1,
                        )
                    except (OSError, ValueError) as exc:
                        return json.dumps({
                            "error": f"Externalized payload storage is unavailable: {exc}",
                        })
                    status = payload.get("status")
                    if status == "symlink":
                        scan_counts["rejected_symlink"] += 1
                    elif status != "ok":
                        scan_counts["rejected_invalid_or_unreadable"] += 1
                    elif payload.get("session_id") != engine.current_session_id:
                        scan_counts["rejected_session_mismatch"] += 1
                    else:
                        candidate_refs.append(ref)
                        if len(candidate_refs) >= _LCM_GREP_EXTERNALIZED_FILE_CAP:
                            break
                elif metadata.get("error") == "session_mismatch":
                    scan_counts["rejected_session_mismatch"] += 1
                else:
                    scan_counts["rejected_invalid_or_unreadable"] += 1
        else:
            candidate_refs = externalized_refs
        scan_counts["candidate_files"] = len(candidate_refs)

        externalized_matches: list[dict[str, Any]] = []
        for ref in candidate_refs:
            if externalized_refs is not None:
                path = storage_dir / ref
                if path.is_symlink():
                    scan_counts["rejected_symlink"] += 1
                    return json.dumps({"error": f"Externalized ref is a symlink: {ref}"})
                metadata = _inspect_externalized_payload_metadata(
                    engine,
                    ref,
                    engine.current_session_id,
                    max_read_bytes=_LCM_GREP_EXTERNALIZED_METADATA_READ_BYTES,
                    require_valid_document_tail=True,
                )
                if not metadata.get("readable"):
                    if metadata.get("error") == "session_mismatch":
                        scan_counts["rejected_session_mismatch"] += 1
                        return json.dumps({"error": f"Externalized ref is not owned by the active session: {ref}"})
                    scan_counts["rejected_invalid_or_unreadable"] += 1
                    return json.dumps({"error": f"Externalized ref is not readable: {ref}"})
            try:
                payload = read_externalized_payload_search_prefix(
                    ref,
                    config=engine._config,
                    hermes_home=engine._hermes_home,
                    max_content_bytes=_LCM_GREP_EXTERNALIZED_CONTENT_BYTES,
                )
            except (OSError, ValueError) as exc:
                return json.dumps({
                    "error": f"Externalized payload storage is unavailable: {exc}",
                })
            status = payload.get("status")
            if status == "symlink":
                scan_counts["rejected_symlink"] += 1
                if externalized_refs is not None:
                    return json.dumps({"error": f"Externalized ref is a symlink: {ref}"})
                continue
            if status != "ok":
                scan_counts["rejected_invalid_or_unreadable"] += 1
                if externalized_refs is not None:
                    return json.dumps({"error": f"Externalized ref is not readable: {ref}"})
                continue
            scan_counts["scanned_files"] += 1
            if payload.get("session_id") != engine.current_session_id:
                scan_counts["rejected_session_mismatch"] += 1
                if externalized_refs is not None:
                    return json.dumps({"error": f"Externalized ref is not owned by the active session: {ref}"})
                continue
            content = str(payload.get("content") or "")
            match = re.search(re.escape(query), content, flags=re.IGNORECASE)
            if match is None:
                continue
            byte_position = len(content[: match.start()].encode("utf-8"))
            line = content.count("\n", 0, match.start()) + 1
            snippet_start = max(0, match.start() - 120)
            snippet_end = min(len(content), match.end() + 180)
            item = {
                "type": "externalized",
                "depth": "payload",
                "ref": ref,
                "tool_call_id": payload.get("tool_call_id") or "",
                "snippet": content[snippet_start:snippet_end],
                "line": line,
                "byte_position": byte_position,
                "original_content_bytes": payload.get("original_content_bytes"),
                "original_content_chars": payload.get("original_content_chars"),
                "scan_truncated": bool(payload.get("scan_truncated")),
                "content_scanned_bytes": payload.get("content_scanned_bytes", 0),
                "from_current_session": True,
                "_sort_ts": payload.get("created_at") or 0,
                "_sort_rank": byte_position,
                "_sort_directness": 10.0,
            }
            externalized_matches.append(item)
            scan_counts["matched_files"] += 1

        # Search every file in the bounded candidate set before truncating.
        # Discovery order is not a ranking signal: relevance and hybrid use the
        # payload's native byte position, while recency uses its timestamp.
        externalized_matches.sort(key=lambda result: _combined_result_sort_key(result, sort))
        response_chars = 0
        for item in externalized_matches:
            item_chars = len(json.dumps(item, ensure_ascii=False))
            if response_chars + item_chars > _LCM_GREP_RESPONSE_CHAR_CAP:
                break
            response_chars += item_chars
            results.append(item)
        externalized_scan = {
            **scan_counts,
            "file_limit": _LCM_GREP_EXTERNALIZED_FILE_CAP,
            "discovery_limit": _LCM_GREP_EXTERNALIZED_DISCOVERY_CAP,
            "content_bytes_per_file": _LCM_GREP_EXTERNALIZED_CONTENT_BYTES,
            "response_char_limit": _LCM_GREP_RESPONSE_CHAR_CAP,
            "active_session_only": True,
        }

    if sort == "hybrid":
        max_message_directness = max(
            (float(result.get("_sort_directness") or 0.0) for result in results if result.get("type") == "message"),
            default=0.0,
        )
        for result in results:
            if result.get("type") == "summary":
                result["_hybrid_summary_override"] = 1 if float(result.get("_sort_directness") or 0.0) >= (max_message_directness + 8.0) else 0

    results.sort(key=lambda result: _combined_result_sort_key(result, sort))
    for result in results:
        result.pop("_sort_ts", None)
        result.pop("_sort_rank", None)
        result.pop("_sort_directness", None)
        result.pop("_hybrid_summary_override", None)

    response: Dict[str, Any] = {
        "query": query,
        "sort": sort,
        "session_scope": session_scope,
        "content_scope": content_scope,
        "source": source,
        "conversation_id": conversation_id,
        "limit": limit,
        "total_results": len(results),
        "results": results[:limit],
    }
    if args.get("_stemmed"):
        response["fts_index"] = getattr(engine._store, "_last_fts_index", "plain")
    if role is not None:
        response["role"] = role
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
    if raw_message_filter_active:
        response["summary_results_omitted"] = True
    if externalized_results_omitted:
        response["externalized_results_omitted"] = True
    if session_scope == "session":
        response["session_id"] = explicit_session_id
    if requested_limit > limit_cap:
        response["limit_clamped_from"] = requested_limit
    if requested_session_scope not in _LCM_GREP_VALID_SCOPES:
        response["ignored_session_scope"] = requested_session_scope
        response["scope_note"] = (
            "Unsupported session_scope; stayed on current. "
            "Valid values: current, all, session."
        )
    if externalized_refs is not None:
        response["externalized_refs"] = externalized_refs
    if externalized_scan is not None:
        response["externalized_scan"] = externalized_scan
    if message_search_error is not None:
        response["message_search_error"] = message_search_error
    return json.dumps(response)


# A hard-timed semantic operation cannot kill the worker thread it abandons
# (Python has no thread cancellation), so bound how many can be live at once.
# A worker releases its slot when it eventually finishes; once every slot is
# held by a stuck worker, further requests degrade to full-text immediately
# instead of spawning unbounded threads under repeated timeouts.
_LCM_SEMANTIC_MAX_WORKERS = 4
_lcm_semantic_worker_slots = threading.BoundedSemaphore(_LCM_SEMANTIC_MAX_WORKERS)
# FTS fallback has its own bounded lane so abandoned semantic calls cannot
# consume the capacity required to degrade safely.
_LCM_FULL_TEXT_MAX_WORKERS = 4
_lcm_full_text_worker_slots = threading.BoundedSemaphore(_LCM_FULL_TEXT_MAX_WORKERS)


class _WorkerCapacityError(RuntimeError):
    """Raised when no bounded worker slot is available."""


def _run_within_deadline(
    fn,
    *,
    remaining_s: float,
    name: str,
    worker_slots: threading.BoundedSemaphore | None = None,
):
    """Run ``fn`` in a bounded daemon worker, raising if the deadline lapses.

    ``remaining_s`` is the time left in the operation's single absolute
    deadline. On timeout the worker is abandoned but keeps its slot until it
    finishes, so stuck workers cannot accumulate without bound.
    """
    remaining_s = float(remaining_s)
    if remaining_s <= 0:
        raise TimeoutError("semantic latency budget exhausted")
    slots = _lcm_semantic_worker_slots if worker_slots is None else worker_slots
    if not slots.acquire(blocking=False):
        raise _WorkerCapacityError(f"{name} worker capacity is exhausted")
    outcome: list[tuple[bool, Any]] = []

    def invoke() -> None:
        try:
            outcome.append((True, fn()))
        except BaseException as exc:  # noqa: BLE001 - forwarded to caller
            outcome.append((False, exc))
        finally:
            slots.release()

    worker = threading.Thread(target=invoke, name=name, daemon=True)
    try:
        worker.start()
    except BaseException:
        slots.release()
        raise
    worker.join(remaining_s)
    if worker.is_alive():
        raise TimeoutError(f"{name} exceeded the semantic latency budget")
    succeeded, value = outcome[0]
    if not succeeded:
        raise value
    return value


def _lcm_grep_full_text_with_deadline(
    args: Dict[str, Any],
    *,
    engine: "LCMEngine",
    deadline: float,
    limit_cap: int = _LCM_GREP_HARD_LIMIT_CAP,
) -> dict[str, Any]:
    """Run FTS on independent read connections under the request deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return _lcm_grep_deadline_error(
            str(args.get("mode") or "semantic").lower(), "full_text"
        )

    def invoke() -> dict[str, Any]:
        message_conn: sqlite3.Connection | None = None
        dag_conn: sqlite3.Connection | None = None
        expired = [False]

        def require_remaining(stage: str) -> float:
            stage_remaining = deadline - time.monotonic()
            if stage_remaining <= 0:
                raise TimeoutError(f"lcm full-text deadline exhausted before {stage}")
            return stage_remaining

        def interrupt_if_expired() -> int:
            if time.monotonic() >= deadline:
                expired[0] = True
                return 1
            return 0

        try:
            require_remaining("database path resolution")
            db_path = Path(engine._store.db_path).resolve()
            require_remaining("database path resolution")
            uri = f"{db_path.as_uri()}?mode=ro"
            require_remaining("database URI construction")
            message_conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=max(0.001, require_remaining("message connection")),
            )
            require_remaining("message connection")
            require_remaining("DAG connection")
            dag_conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=max(0.001, require_remaining("DAG connection")),
            )
            require_remaining("DAG connection")
            for conn in (message_conn, dag_conn):
                require_remaining("connection setup")
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only=ON")
                require_remaining("connection setup")
                conn.set_progress_handler(interrupt_if_expired, 1000)
                require_remaining("connection setup")
            read_engine = copy.copy(engine)
            read_io = RetrievalContext(engine._store, engine._dag).bind(
                message_conn, dag_conn,
            )
            read_engine._store = read_io.store
            read_engine._dag = read_io.dag
            require_remaining("full-text search")
            payload = json.loads(
                _lcm_grep_full_text(
                    args,
                    engine=read_engine,
                    _limit_cap=limit_cap,
                )
            )
            if expired[0] or time.monotonic() >= deadline:
                return _lcm_grep_deadline_error(
                    str(args.get("mode") or "semantic").lower(), "full_text"
                )
            return payload
        finally:
            if message_conn is not None:
                message_conn.close()
            if dag_conn is not None:
                dag_conn.close()

    try:
        return _run_within_deadline(
            invoke,
            remaining_s=remaining,
            name="lcm-full-text",
            worker_slots=_lcm_full_text_worker_slots,
        )
    except (_WorkerCapacityError, TimeoutError):
        return _lcm_grep_deadline_error(
            str(args.get("mode") or "semantic").lower(), "full_text"
        )
    except Exception as exc:
        return {
            "error": f"full-text fallback failed: {exc}",
            "mode": str(args.get("mode") or "semantic").lower(),
        }


_EMBEDDING_IDENTITY_STALE_REASON = (
    "embedding_identity_stale: stored vectors use an older embedding privacy "
    "revision; run /lcm embed warmup, then /lcm embed backfill --apply"
)


def _lcm_embedding_profile_revision(
    conn: sqlite3.Connection, *, provider: str, model: str, task: str
) -> str | None:
    try:
        row = conn.execute(
            "SELECT revision FROM lcm_embedding_profile "
            "WHERE active=1 AND archived_at IS NULL AND task=? "
            "AND provider=? AND model_name=? "
            "ORDER BY registered_at DESC, identity_hash DESC LIMIT 1",
            (task, provider, model),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table: lcm_embedding_profile" not in str(exc):
            raise
        return None
    return str(row[0] or "") if row is not None else None


def _lcm_active_embedding_revision(
    engine: "LCMEngine", provider: Any, *, task: str
) -> str:
    db_path = Path(engine._store.db_path).resolve()
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        revision = _lcm_embedding_profile_revision(
            conn,
            task=str(task),
            provider=str(getattr(provider, "provider_id", "") or "").strip().lower(),
            model=str(getattr(provider, "model_id", "") or "").strip(),
        )
    finally:
        conn.close()
    if revision is None:
        raise EmbeddingIdentityStaleError(
            f"no active {task} vector identity matches the configured cloud provider; "
            "run `/lcm embed warmup` before semantic retrieval"
        )
    return revision


def _lcm_grep_embed_query(
    provider: Any,
    query: str,
    *,
    engine: "LCMEngine | None" = None,
    task: str = "summary",
    remaining_s: float,
    reuse_vector: list[float] | None = None,
) -> list[float]:
    """Embed one query within the operation's remaining absolute budget."""
    outbound_query = str(query)
    privacy_revision: str | None = None
    if embedding_provider_requires_privacy(
        str(getattr(provider, "provider_id", "") or "")
    ):
        if engine is None:
            raise EmbeddingPrivacyPolicyError(
                "cloud semantic query requires an engine-bound privacy identity"
            )
        outbound_query, current_revision, _changed = protect_embedding_text(
            outbound_query,
            engine._config,
        )
        privacy_revision = _lcm_active_embedding_revision(
            engine, provider, task=task
        )
        if current_revision != privacy_revision:
            raise EmbeddingIdentityStaleError(
                "cloud embedding privacy policy differs from registered vector "
                "identity; run `/lcm embed warmup` before semantic retrieval"
            )

    if reuse_vector is not None:
        return reuse_vector  # Identity must match even when sharing a query embed.

    def invoke() -> list[float]:
        if privacy_revision is not None:
            validate_embedding_privacy_dispatch(
                [outbound_query],
                engine._config,
                expected_revision=privacy_revision,
            )
        interactive = getattr(provider, "embed_query_interactive", None)
        if callable(interactive):
            return interactive(outbound_query, timeout=max(0.001, remaining_s))
        return provider.embed_query(outbound_query)

    vector = _run_within_deadline(
        invoke, remaining_s=remaining_s, name="lcm-query-embed"
    )
    return [float(value) for value in vector]


def _lcm_embedding_query_metric(provider: Any) -> dict[str, Any]:
    """Return non-secret provider accounting for one completed query embed."""
    raw_tokens = getattr(provider, "last_usage_tokens", None)
    try:
        usage_tokens = max(0, int(raw_tokens)) if raw_tokens is not None else None
    except (TypeError, ValueError, OverflowError):
        usage_tokens = None
    return {
        "provider": str(getattr(provider, "provider_id", "") or "unknown"),
        "model": str(getattr(provider, "model_id", "") or "unknown"),
        "usage_tokens": usage_tokens,
    }


def _lcm_grep_resolve_provider(
    engine: "LCMEngine", *, deadline: float | None = None
) -> Any:
    config = engine._config
    cache_key = (
        str(getattr(config, "embedding_provider", "") or "").strip().lower(),
        str(getattr(config, "embedding_model", "") or "").strip(),
    )
    cached = getattr(engine, "_lcm_embedding_provider_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    provider = resolve_provider(config)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    engine._lcm_embedding_provider_cache = (cache_key, provider)
    return provider


def _resolve_recall_provider(
    engine: "LCMEngine",
    *,
    deadline: float | None = None,
    provider_override: str | None = None,
) -> Any:
    """Resolve the embedding provider for a recall query.

    ``provider_override`` (SPEC F proactive injection) lets the injection path
    embed its query with a different provider than interactive search — e.g. a
    local fastembed provider for the offline path even when search uses voyage.
    It caches under its own engine slot so it never evicts the main
    interactive-search provider cache (no per-turn thrash between the two).
    Empty override falls straight through to the shared resolver, so normal
    tool calls are byte-identical.
    """
    override = str(provider_override or "").strip()
    if not override:
        return _lcm_grep_resolve_provider(engine, deadline=deadline)
    config = engine._config
    cache_key = (
        override.lower(),
        str(getattr(config, "embedding_model", "") or "").strip(),
    )
    cached = getattr(engine, "_lcm_proactive_provider_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    override_config = copy.copy(config)
    override_config.embedding_provider = override
    provider = resolve_provider(override_config)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("provider resolution deadline exhausted")
    engine._lcm_proactive_provider_cache = (cache_key, provider)
    return provider


def _resolve_recall_chunk_provider(
    engine: "LCMEngine", summary_provider: Any, *, deadline: float | None = None
) -> Any:
    """Resolve the provider/model identity registered for the chunk corpus."""
    chunk_model = default_chunk_model(
        summary_provider.provider_id, summary_provider.model_id
    )
    if chunk_model == summary_provider.model_id:
        return summary_provider
    cache_key = (str(summary_provider.provider_id).lower(), chunk_model)
    cached = getattr(engine, "_lcm_chunk_provider_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("chunk provider resolution deadline exhausted")
    chunk_config = copy.copy(engine._config)
    chunk_config.embedding_provider = summary_provider.provider_id
    chunk_config.embedding_model = chunk_model
    provider = resolve_provider(chunk_config)
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("chunk provider resolution deadline exhausted")
    engine._lcm_chunk_provider_cache = (cache_key, provider)
    return provider


def _lcm_grep_semantic(
    args: Dict[str, Any],
    *,
    engine: "LCMEngine",
    deadline: float,
    candidate_limit: int | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    mode = str(args.get("mode") or "semantic").lower()
    excluded_session_ids = set(args.get("_excluded_session_ids") or ())
    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error(mode, "semantic_entry")
    query = _coerce_query_arg(args.get("query"))
    if not query:
        return {"error": "No query provided"}

    raw_limit_arg = args.get("limit", 10)
    parsed_limit = _parse_int_value(raw_limit_arg, 10)
    if parsed_limit <= 0:
        return {"error": "limit must be a positive integer"}
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_GREP_HARD_LIMIT_CAP)
    knn_limit = candidate_limit if candidate_limit is not None else limit

    requested_session_scope = str(args.get("session_scope", "current")).lower()
    raw_session_id_arg = args.get("session_id")
    explicit_session_id = (
        str(raw_session_id_arg).strip() if raw_session_id_arg is not None else ""
    )
    if requested_session_scope == "current":
        if explicit_session_id:
            return {"error": "session_id is only valid with session_scope=session"}
        session_scope = "current"
        search_session_id: str | None = engine.current_session_id
    elif requested_session_scope == "all":
        if explicit_session_id:
            return {"error": "session_id is not used with session_scope=all"}
        session_scope = "all"
        search_session_id = None
    elif requested_session_scope == "session":
        if not explicit_session_id:
            return {"error": "session_scope=session requires session_id"}
        session_scope = "session"
        search_session_id = explicit_session_id
    else:
        session_scope = "current"
        search_session_id = engine.current_session_id
        logger.warning(
            "Ignoring unsupported session_scope=%s for semantic lcm_grep",
            requested_session_scope,
        )

    source = str(args.get("source") or "").strip() or None
    conversation_id = str(args.get("conversation_id") or "").strip() or None
    role, role_error = _parse_grep_role(args.get("role"))
    if role_error:
        return {"error": role_error}
    time_from, time_from_error = _parse_optional_timestamp(args.get("time_from"), "time_from")
    if time_from_error:
        return {"error": time_from_error}
    time_to, time_to_error = _parse_optional_timestamp(args.get("time_to"), "time_to")
    if time_to_error:
        return {"error": time_to_error}
    if time_from is not None and time_to is not None and time_to < time_from:
        return {"error": "time_to must be greater than or equal to time_from"}

    def degraded(reason: str) -> dict[str, Any]:
        if not allow_fallback:
            return {
                "mode": mode,
                "degraded_to_fts": True,
                "degraded_reason": reason,
                "coverage": "none",
                "results": [],
            }
        fts_args = dict(args)
        fts_args.pop("mode", None)
        payload = _lcm_grep_full_text_with_deadline(
            fts_args,
            engine=engine,
            deadline=deadline,
        )
        if "error" not in payload:
            payload["mode"] = mode
            payload["degraded_to_fts"] = True
            payload["degraded_reason"] = reason
            payload["coverage"] = "none"
        return payload

    if not bool(getattr(engine._config, "embeddings_enabled", False)):
        return degraded("semantic retrieval is disabled")

    # role is a raw-message dimension; a summary vector has no single role, so
    # it cannot be enforced over embedded summaries. Rather than silently
    # ignoring it, degrade to full_text — which does enforce role — so a
    # role=user query never returns assistant/tool summaries.
    if role is not None:
        return degraded("role filtering is only supported by full_text retrieval")

    # The advertised lcm_grep contract (schemas.LCM_GREP) returns RAW message
    # hits only for broader scopes and for time/conversation filters; full_text
    # honors this by omitting summary hits in exactly these cases. Embedded
    # summaries have no single lane and are cross-session/unexpandable, so the
    # semantic arm degrades to the raw full_text path rather than emit summary
    # hits that violate the contract (and, for conversation_id, leak
    # wrong-lane summaries from a session that carries multiple conversations).
    if session_scope != "current":
        return degraded("broader scopes return raw-message hits only")
    if time_from is not None or time_to is not None:
        return degraded("time-scoped queries return raw-message hits only")
    if conversation_id is not None:
        return degraded("conversation-scoped queries return raw-message hits only")

    # content_scope is a payload-search dimension owned by the full-text arm.
    # Externalized payloads are never embedded (embedded_kind='summary'), so a
    # semantic request scoped to them has no vector corpus to search. Degrade
    # to full_text — which implements bounded externalized payload scanning —
    # rather than silently returning history-only semantic hits that ignore
    # the requested scope. In hybrid mode this surfaces as the full-text-arm
    # result (which honors content_scope) plus the explicit degraded marker.
    content_scope = str(args.get("content_scope") or "history").strip().lower()
    if content_scope != "history":
        return degraded(
            "content_scope beyond history is served by full_text retrieval"
        )

    # Scope the (current-session) summaries to the active session id. With the
    # raw-only degradations above, conversation_id is always None here, so this
    # resolves to the current session set.
    knn_conversation_ids = _resolve_semantic_conversation_scope(
        engine, search_session_id=search_session_id, conversation_id=conversation_id
    )
    if excluded_session_ids:
        knn_conversation_ids = [
            session_id
            for session_id in knn_conversation_ids or []
            if session_id not in excluded_session_ids
        ]
    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error(mode, "scope_resolution")
    if knn_conversation_ids == []:
        response: dict[str, Any] = {
            "query": query,
            "mode": "semantic",
            "sort": normalize_search_sort(args.get("sort")),
            "session_scope": session_scope,
            "source": source,
            "conversation_id": conversation_id,
            "limit": limit,
            "total_results": 0,
            "results": [],
            "coverage": "full",
            "degraded_to_fts": False,
        }
        if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
            response["limit_clamped_from"] = requested_limit
        if requested_session_scope not in _LCM_GREP_VALID_SCOPES:
            response["ignored_session_scope"] = requested_session_scope
            response["scope_note"] = (
                "Unsupported session_scope; stayed on current. "
                "Valid values: current, all, session."
            )
        return response

    try:
        provider = _run_within_deadline(
            lambda: _lcm_grep_resolve_provider(engine, deadline=deadline),
            remaining_s=deadline - time.monotonic(),
            name="lcm-provider-resolution",
        )
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except TimeoutError:
        return _lcm_grep_deadline_error(mode, "provider_resolution")
    except Exception as exc:
        return degraded(f"embedding provider unavailable: {exc}")
    if provider is None:
        return degraded("embedding provider is not configured")

    try:
        query_vector = _lcm_grep_embed_query(
            provider,
            query,
            engine=engine,
            task="summary",
            remaining_s=deadline - time.monotonic(),
        )
    except EmbeddingIdentityStaleError:
        return degraded(_EMBEDDING_IDENTITY_STALE_REASON)
    except VoyageError as exc:
        if exc.kind == "auth":
            return {
                "error": f"Embedding provider authentication failed; {exc}",
                "mode": mode,
            }
        return degraded(f"query embedding failed: {exc}")
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except Exception as exc:
        return degraded(f"query embedding failed: {exc}")

    def _run_knn() -> Any:
        # VectorStore is resolved through this module's namespace so tests that
        # monkeypatch ``tools.VectorStore`` continue to govern the KNN backend.
        return run_knn(
            engine,
            query_vector=query_vector,
            provider=provider,
            knn_limit=knn_limit,
            deadline=deadline,
            since=time_from,
            until=time_to,
            conversation_ids=knn_conversation_ids,
            source=source,
            vector_store_cls=VectorStore,
        )

    try:
        knn_results = _run_within_deadline(
            _run_knn,
            remaining_s=deadline - time.monotonic(),
            name="lcm-knn",
        )
        coverage = knn_results.coverage
        ranked_rows = list(knn_results)
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except TimeoutError as exc:
        return degraded(f"semantic vector search exceeded the latency budget: {exc}")
    except Exception as exc:
        return degraded(f"semantic vector search failed: {exc}")

    if coverage == "none":
        return degraded("semantic vectors are unavailable (coverage=none)")
    if not ranked_rows:
        return degraded("semantic retrieval returned no vector candidates")

    try:
        hydrated_nodes = _run_within_deadline(
            lambda: hydrate_semantic_nodes(
                engine,
                ranked_rows=ranked_rows,
                knn_limit=knn_limit,
                deadline=deadline,
            ),
            remaining_s=deadline - time.monotonic(),
            name="lcm-result-hydration",
        )
    except _WorkerCapacityError as exc:
        return degraded(f"semantic capacity exhausted: {exc}")
    except TimeoutError:
        return _lcm_grep_deadline_error(mode, "result_resolution")
    except Exception as exc:
        return degraded(f"semantic result hydration failed: {exc}")

    current_session_id = engine.current_session_id
    has_current_session = bool(current_session_id)
    results: list[dict[str, Any]] = []
    excluded_candidates = False
    for node, score in hydrated_nodes:
        if time.monotonic() >= deadline:
            return _lcm_grep_deadline_error(mode, "result_resolution")
        if node.session_id in excluded_session_ids:
            excluded_candidates = True
            continue
        # conversation/role/source/time filters are enforced inside knn() before
        # the top-k cap, so no eligible lower-ranked vector was dropped for an
        # ineligible top hit; nothing further to post-filter here.
        confidence = _lcm_grep_confidence(score)
        result = {
            "type": "summary",
            "depth": f"d{node.depth}",
            "node_id": node.node_id,
            "session_id": node.session_id,
            "snippet": node.summary[:_LCM_GREP_SEMANTIC_SNIPPET_CHARS],
            "token_count": node.token_count,
            "expand_hint": node.expand_hint,
            "earliest_at": node.earliest_at,
            "latest_at": node.latest_at,
            "from_current_session": has_current_session and node.session_id == current_session_id,
            "score": score,
            "cosine_score": score,
            "confidence": confidence,
            "confidence_band": confidence,
        }
        results.append(result)
        if len(results) >= knn_limit:
            break

    if not results and not excluded_candidates:
        return degraded("semantic vector candidates could not be resolved")

    response: dict[str, Any] = {
        "query": query,
        "mode": "semantic",
        "sort": normalize_search_sort(args.get("sort")),
        "session_scope": session_scope,
        "source": source,
        "conversation_id": conversation_id,
        "limit": limit,
        "total_results": len(results),
        "results": results[:limit] if candidate_limit is None else results,
        "coverage": coverage,
        "degraded_to_fts": False,
    }
    if role is not None:
        response["role"] = role
    if time_from is not None:
        response["time_from"] = time_from
    if time_to is not None:
        response["time_to"] = time_to
    if session_scope == "session":
        response["session_id"] = explicit_session_id
    if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    if requested_session_scope not in _LCM_GREP_VALID_SCOPES:
        response["ignored_session_scope"] = requested_session_scope
        response["scope_note"] = (
            "Unsupported session_scope; stayed on current. "
            "Valid values: current, all, session."
        )
    return response


def _lcm_grep_hybrid(
    args: Dict[str, Any], *, engine: "LCMEngine", deadline: float
) -> dict[str, Any]:
    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error("hybrid", "hybrid_entry")
    requested_limit = _parse_int_value(args.get("limit", 10), 10)
    if requested_limit <= 0:
        return {"error": "limit must be a positive integer"}
    limit = min(requested_limit, _LCM_GREP_HARD_LIMIT_CAP)
    candidate_limit = min(
        _LCM_GREP_HYBRID_CANDIDATE_CAP,
        max(50, limit * 3),
    )

    fts_args = dict(args)
    fts_args["mode"] = "hybrid"
    fts_args["limit"] = candidate_limit
    fts = _lcm_grep_full_text_with_deadline(
        fts_args,
        engine=engine,
        deadline=deadline,
        limit_cap=_LCM_GREP_HYBRID_CANDIDATE_CAP,
    )
    if "error" in fts:
        return fts

    def degraded_to_fts(reason: str, *, coverage: str = "none") -> dict[str, Any]:
        response = dict(fts)
        response["mode"] = "hybrid"
        response["limit"] = limit
        response["total_results"] = len(fts.get("results", []))
        response["results"] = list(fts.get("results", []))[:limit]
        response["degraded_to_fts"] = True
        response["degraded_reason"] = reason
        response["coverage"] = coverage
        if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
            response["limit_clamped_from"] = requested_limit
        else:
            response.pop("limit_clamped_from", None)
        return response

    if time.monotonic() >= deadline:
        return degraded_to_fts(
            "semantic arm skipped because the request deadline was exhausted"
        )

    semantic_args = dict(args)
    semantic_args["mode"] = "hybrid"
    semantic_args["limit"] = candidate_limit
    semantic = _lcm_grep_semantic(
        semantic_args,
        engine=engine,
        deadline=deadline,
        candidate_limit=candidate_limit,
        allow_fallback=False,
    )
    if "error" in semantic:
        if semantic.get("timeout") is True:
            return degraded_to_fts(
                "semantic arm exceeded the request deadline",
                coverage=str(semantic.get("coverage", "none")),
            )
        return semantic
    if semantic.get("degraded_to_fts"):
        return degraded_to_fts(
            str(semantic.get("degraded_reason", "semantic arm unavailable")),
            coverage=str(semantic.get("coverage", "none")),
        )

    if time.monotonic() >= deadline:
        return _lcm_grep_deadline_error("hybrid", "fusion")
    # FTS is arm 0, semantic is arm 1; rrf_fuse merges by hit identity and
    # accumulates 1/(k+rank). Arm-specific metadata (which arm, confidence,
    # snippet provenance) is grep-presentation and stays here.
    ordered = rrf_fuse(
        [fts.get("results", []), semantic.get("results", [])],
        k=_LCM_GREP_RRF_K,
    )
    semantic_by_identity = {
        _hit_identity(hit): hit for hit in semantic.get("results", [])
    }
    for entry in ordered:
        ranks = entry["ranks"]
        if 0 in ranks and 1 in ranks:
            # The semantic form carries score/confidence while the FTS form carries
            # its exact house snippet/provenance. Preserve the latter as the base.
            sem_hit = semantic_by_identity[_hit_identity(entry["hit"])]
            entry["hit"].setdefault("semantic_snippet", sem_hit.get("snippet", ""))

    results: list[dict[str, Any]] = []
    for entry in ordered[:limit]:
        if time.monotonic() >= deadline:
            return _lcm_grep_deadline_error("hybrid", "fusion")
        ranks = entry["ranks"]
        hit = dict(entry["hit"])
        hit["score"] = float(entry["rrf_score"])
        hit["rrf_score"] = float(entry["rrf_score"])
        if 0 in ranks:
            hit["fts_rank"] = ranks[0]
        if 1 in ranks:
            sem_hit = semantic_by_identity[_hit_identity(entry["hit"])]
            hit["semantic_rank"] = ranks[1]
            hit["semantic_score"] = sem_hit.get("score")
            hit["confidence"] = sem_hit.get("confidence")
            hit["confidence_band"] = sem_hit.get("confidence_band")
        results.append(hit)

    response = dict(fts)
    response["mode"] = "hybrid"
    response["limit"] = limit
    response["total_results"] = len(ordered)
    response["results"] = results
    response["coverage"] = semantic.get("coverage", "none")
    response["degraded_to_fts"] = False
    response["fusion"] = "rrf"
    response["rrf_k"] = _LCM_GREP_RRF_K
    if requested_limit > _LCM_GREP_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    else:
        response.pop("limit_clamped_from", None)
    return response


def lcm_grep(args: Dict[str, Any], **kwargs) -> str:
    """Search LCM history using full-text, semantic, or RRF hybrid retrieval."""
    request_started = time.monotonic()
    mode = str(args.get("mode") or "full_text").strip().lower()
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    excluded_session_ids, exclusions_error = _parse_retrieval_excluded_session_ids(
        args, engine.current_session_id
    )
    if exclusions_error:
        return json.dumps({"error": exclusions_error})
    parsed_args = dict(args)
    parsed_args["_excluded_session_ids"] = excluded_session_ids
    parsed_args.pop("_allow_operators", None)
    parsed_args.pop("_agent_operators", None)
    parsed_args.pop("_grep_any_term", None)
    parsed_args.pop("_stemmed", None)
    parsed_args["_agent_operators"] = True
    if mode == "full_text":
        payload = json.loads(_lcm_grep_full_text(parsed_args, **kwargs))
        query = _coerce_query_arg(args.get("query"))
        safe = sanitize_fts5_query(query, agent_operators=True)
        tokens = re.findall(r'"[^"]*"|[^\s"]+', safe)
        operands = list(dict.fromkeys(
            token for token in tokens
            if token.startswith('"') or token.upper() not in {"NOT", "NEAR", "OR", "AND"}
        ))
        or_query = " OR ".join(operands)
        if (
            payload.get("total_results") == 0 and not payload.get("message_search_error")
            and payload.get("content_scope") in {"history", "both"}
            and not requires_like_fallback(query, safe)
            and not {"OR", "NOT"}.intersection(tokens)
            and not re.search(r"[()]", re.sub(r'"[^"]*"', "", query))
            and len(operands) >= 2
        ):
            retry_args = {**parsed_args, "query": or_query,
                          "_agent_operators": True, "_allow_operators": True, "_grep_any_term": True}
            payload = json.loads(_lcm_grep_full_text(retry_args, **kwargs))
            payload["query"] = query
            payload["retried"] = "any_term"
        return json.dumps(payload)

    timeout_s = max(
        0.001,
        float(getattr(engine._config, "embedding_query_timeout_s", 3.0)),
    )
    deadline = request_started + timeout_s
    if time.monotonic() >= deadline:
        return json.dumps(_lcm_grep_deadline_error(mode, "tool_entry"))
    if mode == "semantic":
        return json.dumps(_lcm_grep_semantic(parsed_args, engine=engine, deadline=deadline))
    if mode == "hybrid":
        return json.dumps(_lcm_grep_hybrid(parsed_args, engine=engine, deadline=deadline))
    return json.dumps({
        "error": "mode must be one of: full_text, semantic, hybrid",
    })


def _lcm_recall_summary_expand_hint(hit: dict[str, Any]) -> str:
    if hit.get("from_current_session"):
        return f"lcm_expand(node_id={hit.get('node_id')})"
    # _session_expand_hint applies !r quoting: imported session IDs may carry
    # apostrophes/backslashes, and an unescaped interpolation would advertise
    # a handle lcm_expand cannot parse back.
    return _session_expand_hint(hit.get("node_id"), hit.get("session_id") or "")


def _lcm_recall_reference_shape(hit: dict[str, Any], *, hydratable: bool) -> str | None:
    """Name the delivery shape that gives this hit a truthful source reference.

    Reference-strict delivery (FINDING-F35 §2). This is only the CHEAP,
    rank-ordered ADMISSION test -- it says a candidate could plausibly resolve to
    a ``(store_id, char_start, char_end)`` span, not that it does. Truth is
    established later by :func:`_lcm_recall_verified_span`, which reads the row
    and checks that the delivered text really sits at the claimed offset.

    A summary is rejected here and cannot be given a reference. Its text is
    model-generated prose, not a verbatim span of any row, so
    ``lcm:<store_id>:<start>-<end>`` would assert bytes that are not at that
    offset. ``SummaryNode.source_ids`` is the list of *every* message a leaf node
    summarizes, so even a message-sourced node's first source is lineage, not a
    citation (#164a). The summary arm keeps its ranking influence through
    :func:`_lcm_recall_summary_source_hits`, which lets the nodes' SOURCE
    MESSAGES compete as ordinary citable candidates.
    """
    if hit.get("kind") == "summary":
        return None
    if hit.get("store_id") is None:
        return None
    if hydratable:
        return "content_offset"
    if hit.get("chunk_span"):
        return "chunk_span"
    # An arm that already knows where its excerpt sits (the chunk arm's
    # char_start, a summary-source hit's row prefix) can be cited without
    # hydration -- subject to the verification below.
    if hit.get("content_offset") is not None:
        return "content_offset"
    return None


def _lcm_recall_verified_span(
    item: dict[str, Any], row: dict[str, Any] | None
) -> tuple[int, int] | None:
    """Return the ``(offset, chars)`` the delivered text ACTUALLY occupies.

    The one honest definition of a validated source reference: the bytes handed
    to the caller must be present at the claimed offset of the CURRENT row. That
    single check subsumes three failure modes a structural test misses --

    * a stale ``chunk_span`` whose row was deleted or rewritten between chunk
      hydration and response shaping (no coherent snapshot spans those reads);
    * an FTS snippet, which is a match window with markers rather than a
      verbatim prefix, so its ``content_offset`` of 0 is not a real location;
    * a hydration miss, where the promised window never arrived.

    Returning the span rather than a bool is deliberate: the caller PUBLISHES it
    (``content_offset``/``content_returned_chars``) so consumers do not have to
    re-derive it. ``__init__.py``'s ``_answer_ready_baseline`` substitutes offset
    0 when the field is absent, which silently fabricates a reference for every
    hit whose excerpt does not start at the beginning of its row.
    """
    if item.get("kind") == "summary" or item.get("store_id") is None:
        return None
    text = item.get("content") if item.get("content") is not None else item.get("snippet")
    text = str(text or "")
    if not text:
        return None
    raw_offset = item.get("content_offset")
    if raw_offset is None:
        return None
    try:
        offset = int(raw_offset)
    except (TypeError, ValueError, OverflowError):
        return None
    if offset < 0 or row is None:
        return None
    content = str(row.get("content") or "")
    if content[offset:offset + len(text)] != text:
        return None
    return offset, len(text)


def _lcm_recall_citable_entries(
    ordered: list[dict[str, Any]],
    *,
    limit: int,
    per_session_limit: int,
    expanded_limit: int,
    engine: "LCMEngine" | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Reference-strict selection helper (kept for direct unit use)."""
    selector = _LcmRecallStrictSelector(
        ordered,
        engine=engine,
        per_session_limit=per_session_limit,
        expanded_limit=expanded_limit,
    )
    selected = selector.take(limit)
    return selected, selector.diversity_dropped, selector.unreferenced_dropped


class _LcmRecallSelectionLedger:
    """Per-entry lifecycle, and every resource an entry holds while it is live.

    Session density, the hydration budget and row retention used to be three
    counters maintained by hand at four call sites, and each review round found a
    different seam where one of them leaked: a refund that never happened, a
    refund applied twice, a store that stayed pinned after its entry was let go.
    All three are consequences of ONE fact -- whether an entry is still live --
    so they are DERIVED from that fact here instead of tracked alongside it.

    ``PENDING -> ADMITTED -> (DELIVERED | RELEASED)``. Transitions are one-way.
    Resources are charged on ADMITTED and refunded exactly once on RELEASED;
    DELIVERED is terminal and keeps them. A second release is a counted no-op,
    not a second refund -- the double-refund that let three hits through a cap
    of two.
    """

    ADMITTED = "admitted"
    DELIVERED = "delivered"
    RELEASED = "released"

    def __init__(self) -> None:
        # Keyed by id(); the entry itself is held so the id cannot be recycled.
        self._records: dict[int, dict[str, Any]] = {}
        self._session_counts: dict[str, int] = {}
        self._live_stores: dict[int, int] = {}
        self._live = 0
        self.double_releases = 0

    def state(self, entry: dict[str, Any]) -> str | None:
        record = self._records.get(id(entry))
        return record["state"] if record else None

    @property
    def live_count(self) -> int:
        """ADMITTED or DELIVERED -- the entries currently holding resources."""
        return self._live

    def delivered_entries(self) -> list[dict[str, Any]]:
        return [
            record["entry"]
            for record in self._records.values()
            if record["state"] == self.DELIVERED
        ]

    def session_count(self, session_key: str) -> int:
        return self._session_counts.get(session_key, 0)

    def holds_store(self, store_id: Any) -> bool:
        return store_id is not None and int(store_id) in self._live_stores

    def admit(
        self, entry: dict[str, Any], *, session_key: str, store_id: Any
    ) -> None:
        """PENDING -> ADMITTED, charging every resource in this one place."""
        self._records[id(entry)] = {
            "entry": entry,
            "state": self.ADMITTED,
            "session_key": session_key,
            "store_id": None if store_id is None else int(store_id),
        }
        self._session_counts[session_key] = self._session_counts.get(session_key, 0) + 1
        if store_id is not None:
            key = int(store_id)
            self._live_stores[key] = self._live_stores.get(key, 0) + 1
        self._live += 1

    def deliver(self, entry: dict[str, Any]) -> bool:
        """ADMITTED -> DELIVERED. Terminal; the entry keeps what it holds."""
        record = self._records.get(id(entry))
        if record is None or record["state"] != self.ADMITTED:
            return False
        record["state"] = self.DELIVERED
        return True

    def release(self, entry: dict[str, Any]) -> bool:
        """ADMITTED -> RELEASED, refunding once. Idempotent by construction."""
        record = self._records.get(id(entry))
        if record is None or record["state"] != self.ADMITTED:
            self.double_releases += 1
            return False
        record["state"] = self.RELEASED
        session_key = record["session_key"]
        if self._session_counts.get(session_key):
            self._session_counts[session_key] -= 1
        store_id = record["store_id"]
        if store_id is not None and self._live_stores.get(store_id):
            self._live_stores[store_id] -= 1
            if not self._live_stores[store_id]:
                del self._live_stores[store_id]
        self._live -= 1
        return True


class _LcmRecallStrictSelector:
    """Rank-ordered admission that VERIFIES a candidate before it is admitted.

    Reference-strict replacement for :func:`recall_delivery._diverse_entries`. Same
    stable rank-preserving walk with bounded session density, plus the rule that
    makes the mode meaningful: a candidate is admitted only once its delivered
    text has been found at its claimed offset in the CURRENT row.

    Verifying BEFORE admission is what keeps the invariant whole. A candidate
    that fails never becomes a result, so it cannot spend a session slot a valid
    lower-ranked candidate needs; the walk simply continues, which IS the
    backfill -- there is no separate replacement pass for another stage to
    bypass. Reads stay batched: the walk prefetches a wave of rows ahead of the
    cursor, and remembers what it ATTEMPTED, so a row the store does not have
    settles the candidate instead of dragging in the rest of the corpus.

    Two rejections, two different lifetimes. Failing the shape test or
    verification is a PERMANENT property of the candidate, so it is discarded and
    its row released. Being over the session cap is not -- density is a
    refundable resource, and a later stage handing back a slot (delta dropping a
    reference the caller already holds) can make a blocked candidate admissible.
    Those are DEFERRED in rank order and reconsidered on the next wave, which is
    what lets a refund actually reach the ranking instead of arriving after the
    walk has consumed it. The reserve is bounded, because no more slots can ever
    be refunded than were admitted.

    Two candidate shapes are proved differently. Inside the hydration budget the
    delivered text is a window cut from the row, so the row's existence IS the
    proof and the same snapshot is handed to hydration. Past that budget the hit
    ships its own excerpt, so the excerpt must be found at its offset. When a row
    surfaced through several arms, each arm's representation is tried in turn: an
    FTS match window is not a verbatim prefix and will not verify, but the chunk
    or summary-source representation of the same row does, and delivering that is
    not a swap -- it is the same row, quoted somewhere it really says.
    """

    def __init__(
        self,
        ordered: list[dict[str, Any]],
        *,
        engine: "LCMEngine" | None,
        per_session_limit: int,
        expanded_limit: int,
        wave_size: int = _LCM_RECALL_STRICT_READ_WAVE,
    ) -> None:
        self._ordered = ordered
        self._engine = engine
        self._cursor = 0
        self._per_session_limit = per_session_limit
        self._expanded_limit = expanded_limit
        self._wave_size = max(1, wave_size)
        # How far the ranking is known to be SETTLED. Monotone: an entry the
        # ledger knows, or one rejected in-budget, can never become admissible
        # again, so this only ever advances. It is a scan hint, never a source
        # of truth -- the resume point itself is derived on demand.
        self._settled_prefix = 0
        self._blocked: set[int] = set()
        # id(entry) -> the budget the rejection was made under.
        self._rejected: dict[int, bool] = {}
        self._examining = 0
        self._prefetched_to = 0
        self._missing: set[int] = set()
        self.ledger = _LcmRecallSelectionLedger()
        self.rows: dict[int, dict[str, Any]] = {}
        self.batched_reads = 0

    @property
    def unreferenced_dropped(self) -> int:
        """Candidates currently held out for want of a validated reference."""
        return len(self._rejected)

    @property
    def diversity_dropped(self) -> int:
        """Candidates the density cap is still keeping out of the response."""
        return len(self._blocked)

    def exhausted(self) -> bool:
        return self._cursor >= len(self._ordered)

    def deliver(self, entry: dict[str, Any]) -> bool:
        return self.ledger.deliver(entry)

    def release(self, entry: dict[str, Any]) -> bool:
        """Hand back the slot an entry a later stage discarded never used."""
        released = self.ledger.release(entry)
        if released:
            self._forget(entry["hit"].get("store_id"))
            # Resume from the earliest candidate that could still be admitted.
            # DERIVED, never stored: a remembered resume position is state that
            # outlives its precondition -- cleared by one rewind, not re-armed by
            # the next skip, and a candidate silently becomes unreachable. Asking
            # the ranking costs a scan and cannot go stale.
            resume = self._earliest_revisitable()
            if resume < self._cursor:
                self._cursor = resume
                self._prefetched_to = min(self._prefetched_to, self._cursor)
        return released

    def _prefetch(self) -> bool:
        """Read the next WINDOW of candidate rows in ONE batch.

        The window is a span of POSITIONS, not a quota of new ids. Counting new
        ids instead lets a window that mostly hits rows already held scan far
        past itself to fill its quota, so every rewind pulls in another wave and
        retention grows without bound. Advancing by position keeps read-ahead --
        and therefore retention -- inside a single window.

        Starts at the candidate being examined, NOT after it, so the row the walk
        needs right now is always inside the window it triggers.
        """
        if self._engine is None:
            return False
        start = max(self._examining, self._prefetched_to)
        if start >= len(self._ordered):
            return False
        window: list[int] = []
        index = start
        while index < len(self._ordered) and index - start < self._wave_size:
            hit = self._ordered[index]["hit"]
            index += 1
            store_id = hit.get("store_id")
            if hit.get("kind") == "summary" or store_id is None:
                continue
            store_id = int(store_id)
            if store_id not in window:
                window.append(store_id)
        self._prefetched_to = index
        wanted = [
            store_id
            for store_id in window
            if store_id not in self.rows and store_id not in self._missing
        ]
        if wanted:
            self.batched_reads += 1
            fetched = self._engine._store.get_batch(wanted)
            # A row the store does not have is remembered as MISSING, or the walk
            # cannot tell "not fetched yet" from "fetched and absent" and keeps
            # calling for windows that can never contain it. Rows merely evicted
            # stay re-fetchable, which is what lets the walk revisit a candidate
            # it skipped earlier without holding its row all along.
            self._missing.update(sid for sid in wanted if sid not in fetched)
            self.rows.update(fetched)
        self._trim_rows(window)
        return True

    def _trim_rows(self, window: list[int]) -> None:
        """Hold only the current window and whatever is live."""
        keep = set(window)
        for store_id in list(self.rows):
            if store_id not in keep and not self.ledger.holds_store(store_id):
                del self.rows[store_id]

    def _row_for(self, store_id: Any) -> dict[str, Any] | None:
        if store_id is None:
            return None
        store_id = int(store_id)
        while (
            store_id not in self.rows
            and store_id not in self._missing
            and self._prefetch()
        ):
            pass
        return self.rows.get(store_id)

    def _forget(self, store_id: Any) -> None:
        """Drop a row nothing live or still-in-play depends on."""
        if store_id is None:
            return
        store_id = int(store_id)
        if self.ledger.holds_store(store_id):
            return
        self.rows.pop(store_id, None)

    def _verify(self, entry: dict[str, Any], *, hydratable: bool) -> bool:
        """Prove this candidate can be cited, adopting a representation if needed."""
        hit = entry["hit"]
        row = self._row_for(hit.get("store_id"))
        if row is None or not str(row.get("content") or ""):
            return False
        if hydratable:
            # Hydration cuts the delivered window out of this very row.
            return True
        for candidate in [hit, *entry.get("_alternates", [])]:
            probe = {
                "kind": hit.get("kind"),
                "store_id": hit.get("store_id"),
                "snippet": candidate.get("snippet"),
                "content_offset": candidate.get("content_offset"),
            }
            span = _lcm_recall_verified_span(probe, row)
            if span is None:
                continue
            if candidate is not hit:
                # Same row, quoted where it really says it.
                hit["snippet"] = candidate.get("snippet")
                hit["content_offset"] = candidate.get("content_offset")
                if candidate.get("chunk_span"):
                    hit["chunk_span"] = candidate["chunk_span"]
                hit["expand_hint"] = _lcm_recall_excerpt_expand_hint(hit)
            entry["_strict_span"] = span
            return True
        return False

    @staticmethod
    def _session_key(hit: dict[str, Any]) -> str:
        raw_session_id = hit.get("session_id")
        # Missing session identities must not collapse into one synthetic session.
        return (
            str(raw_session_id)
            if raw_session_id not in {None, ""}
            else f"missing:{_hit_identity(hit)!r}"
        )

    def _admit(self, entry: dict[str, Any]) -> None:
        hit = entry["hit"]
        self.ledger.admit(
            entry,
            session_key=self._session_key(hit),
            store_id=hit.get("store_id"),
        )

    def _is_settled(self, entry: dict[str, Any]) -> bool:
        """True when this candidate can never be admitted, whatever happens next.

        Terminal for exactly two reasons: the ledger already knows it (admitted,
        delivered or released -- none of which return to the pool), or it was
        rejected IN-BUDGET, which means the row itself is unusable. Everything
        else -- density-blocked, post-budget rejected -- is revisitable by
        definition, because the only thing standing in its way is a budget a
        refund can give back.
        """
        return (
            self.ledger.state(entry) is not None
            or self._rejected.get(id(entry)) is True
        )

    def _earliest_revisitable(self) -> int:
        """Position of the first candidate a refund could still make admissible."""
        index = self._settled_prefix
        while index < len(self._ordered) and self._is_settled(self._ordered[index]):
            index += 1
        self._settled_prefix = index
        return index

    def _next_admissible(self) -> dict[str, Any] | None:
        """Walk forward to the next candidate that can be admitted right now.

        Entries the ledger already knows (live or released) are skipped, as are
        entries whose rejection still applies under the current budget.
        """
        while self._cursor < len(self._ordered):
            entry = self._ordered[self._cursor]
            self._examining = self._cursor
            self._cursor += 1
            hit = entry["hit"]
            if self.ledger.state(entry) is not None:
                continue
            hydratable = self.ledger.live_count < self._expanded_limit
            # A rejection is only as durable as the rule that produced it. An
            # IN-BUDGET rejection means the row itself is unusable -- missing,
            # empty, or not a message at all -- which no later budget can undo.
            # A POST-BUDGET rejection only means the candidate's own excerpt did
            # not check out, and that same candidate is still admissible
            # in-budget, where the delivered text is a window cut from the row
            # rather than an excerpt it carried. Skipping it there would underfill
            # against a rule that no longer applies.
            rejected_under = self._rejected.get(id(entry))
            if rejected_under is True or (rejected_under is False and not hydratable):
                continue
            # Verification is deliberately NOT carried across examinations. It is
            # only valid while the row it was proved against is still held, and a
            # blocked candidate releases its row -- which may then be rewritten or
            # deleted before a refund brings the walk back. Re-proving against the
            # row as REFETCHED is what makes the rewind safe.
            if _lcm_recall_reference_shape(hit, hydratable=hydratable) is None or (
                not self._verify(entry, hydratable=hydratable)
            ):
                self._rejected[id(entry)] = hydratable
                self._blocked.discard(id(entry))
                self._forget(hit.get("store_id"))
                continue
            session_key = self._session_key(hit)
            if self.ledger.session_count(session_key) >= self._per_session_limit:
                # Refundable, unlike a shape or verification failure, so the
                # candidate is not consumed -- only its POSITION is remembered,
                # and the row it is not using is released. It stays in the
                # ranking, so no bound on a buffer can discard it.
                self._blocked.add(id(entry))
                self._forget(hit.get("store_id"))
                continue
            # No blind refetch here: verification above just proved this
            # candidate against the row as it stands and left that row held, so
            # hydration reads the same bytes the admission was granted on.
            self._blocked.discard(id(entry))
            self._rejected.pop(id(entry), None)
            self._admit(entry)
            return entry
        return None

    def take(self, target: int) -> list[dict[str, Any]]:
        """Top the live set up to ``target`` verified candidates.

        Expressed as a target rather than a count so a slot handed back between
        waves is immediately reusable: the next wave asks for whatever the
        response is still missing.
        """
        taken: list[dict[str, Any]] = []
        while self.ledger.live_count < target:
            entry = self._next_admissible()
            if entry is None:
                break
            taken.append(entry)
        return taken



def _lcm_recall_content_window(
    content: Any,
    *,
    match_start: int,
    match_end: int,
    char_cap: int,
) -> dict[str, Any]:
    """Return a bounded window centered on the selected evidence span."""
    text = str(content or "")
    content_chars = len(text)
    start = min(max(0, int(match_start)), content_chars)
    end = min(max(start, int(match_end)), content_chars)
    if content_chars <= char_cap:
        offset = 0
    else:
        midpoint = (start + end) // 2
        offset = min(max(0, midpoint - char_cap // 2), content_chars - char_cap)
    bounded = text[offset:offset + char_cap]
    return {
        "content": bounded,
        "content_chars": content_chars,
        "content_offset": offset,
        "content_returned_chars": len(bounded),
        "content_truncated": len(bounded) < content_chars,
        "evidence_span": {"char_start": start, "char_end": end},
    }


def _lcm_recall_event_time(stored: dict[str, Any] | None) -> dict[str, str]:
    """Return a message hit's event-time fields from its stored host observation.

    ``observed_at`` is the host/platform event time (stamped at insert from the
    host message timestamp, or backfilled later). ``messages.timestamp`` is LCM's
    own write time and is never shown as event time, so a row without
    ``observed_at`` yields no fields at all (#995).
    """
    if not stored:
        return {}
    observed_at = _normalize_observed_at(stored.get("observed_at"))
    if observed_at is None:
        return {}
    fields = {
        "event_time": datetime.fromtimestamp(observed_at, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    source = stored.get("observed_at_source")
    if source:
        fields["event_time_source"] = str(source)
    return fields


def _lcm_recall_event_time_rows(
    engine: "LCMEngine", store_ids: list[int], *, deadline: float
) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Read the rows that carry message hits' event time, inside the deadline.

    Returns ``(rows_by_id, unavailable_reason)``. The reason is ``"deadline"``
    when the request budget was already spent (no read is attempted) or ran out
    during or by the end of the read, and ``"read_error"`` for any other SQLite
    failure or a filesystem error opening the database; it is ``None`` after a
    successful read, including one that finds ``observed_at`` NULL. The field is
    optional, so the read gets a private read-only connection whose busy wait
    and statement run are bounded by the remaining budget, the idiom of
    ``_lcm_recall_summary_source_hits``; the shared store connection's 30 s busy
    timeout is never used and never changed. It selects only the three columns
    the field needs, never ``content`` or ``tool_calls``, and a read that ends
    past the deadline is dropped rather than returned late (#995).
    """
    if time.monotonic() >= deadline:
        return {}, "deadline"
    expired = [False]

    def interrupt_if_expired() -> int:
        if time.monotonic() >= deadline:
            expired[0] = True
            return 1
        return 0

    conn: sqlite3.Connection | None = None
    try:
        db_path = Path(engine._store.db_path).resolve()
        conn = sqlite3.connect(
            f"{db_path.as_uri()}?mode=ro",
            uri=True,
            timeout=max(0.001, deadline - time.monotonic()),
        )
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(interrupt_if_expired, 1000)
        placeholders = ",".join("?" for _ in store_ids)
        rows = conn.execute(
            "SELECT store_id, observed_at, observed_at_source FROM messages "
            f"WHERE store_id IN ({placeholders})",
            store_ids,
        ).fetchall()
    except (sqlite3.Error, OSError) as exc:
        logger.debug("lcm_recall event_time read failed: %s", exc)
        if expired[0] or time.monotonic() >= deadline:
            return {}, "deadline"
        return {}, "read_error"
    finally:
        if conn is not None:
            conn.close()
    if time.monotonic() >= deadline:
        return {}, "deadline"
    return {
        int(store_id): {"observed_at": observed_at, "observed_at_source": source}
        for store_id, observed_at, source in rows
    }, None


def _lcm_recall_answer_ready_content(
    engine: "LCMEngine",
    entries: list[dict[str, Any]],
    *,
    query: str,
    expanded_limit: int = _LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT,
    rows_by_id: dict[int, dict[str, Any]] | None = None,
    fts_arm_index: int | None = None,
) -> dict[tuple, dict[str, Any]]:
    """Hydrate selected exact refs with bounded reads and no retrieval search.

    ``rows_by_id`` lets a caller that has ALREADY read these rows hand them over
    instead of paying for a second read. Reference-strict selection verifies each
    candidate against its row before admitting it, so reusing that same snapshot
    also removes the window in which a row could vanish between the two reads and
    leave an admitted candidate unhydrated.
    """
    selected = entries[:expanded_limit]
    store_ids = [
        int(entry["hit"]["store_id"])
        for entry in selected
        if entry["hit"].get("kind") == "message_excerpt"
        and entry["hit"].get("store_id") is not None
    ]
    stored_by_id = (
        rows_by_id if rows_by_id is not None else engine._store.get_batch(store_ids)
    )
    hydrated: dict[tuple, dict[str, Any]] = {}

    for entry in selected:
        hit = entry["hit"]
        identity = _hit_identity(hit)
        if hit.get("kind") == "summary":
            raw_node_id = hit.get("node_id")
            if raw_node_id is None:
                continue
            node = engine._dag.get_node(int(raw_node_id))
            if node is None:
                continue
            content = node.summary or ""
            window = _lcm_recall_content_window(
                content,
                match_start=0,
                match_end=min(len(content), _LCM_RECALL_SNIPPET_CHARS),
                char_cap=_LCM_RECALL_ANSWER_READY_CONTENT_CHARS,
            )
            hydrated[identity] = {
                **window,
                "content_source": "summary",
            }
            continue

        raw_store_id = hit.get("store_id")
        if raw_store_id is None:
            continue
        stored = stored_by_id.get(int(raw_store_id))
        if stored is None:
            continue
        content = str(stored.get("content") or "")
        span = hit.get("chunk_span") or {}
        try:
            match_start = int(span["char_start"])
            match_end = int(span["char_end"])
        except (KeyError, TypeError, ValueError):
            match_start = _content_offset_for_query_match(content, query)
            match_end = match_start + min(
                max(1, len(query)), _LCM_RECALL_SNIPPET_CHARS
            )
            if fts_arm_index is not None and fts_arm_index in entry.get("ranks", {}):
                matches = [
                    match for term in recall_content_terms(query)
                    if (match := re.search(
                        r"(?<!\w)" + _RECALL_PHRASE_GAP.join(
                            "(?ai:" + re.escape(token) + ")" for token in term.strip('"').split()
                        ) + r"(?!\w)", content
                    )) is not None
                ]
                if matches:
                    match = min(matches, key=lambda match: match.start())
                    match_start, match_end = match.span()
        window = _lcm_recall_content_window(
            content,
            match_start=match_start,
            match_end=match_end,
            char_cap=_LCM_RECALL_ANSWER_READY_CONTENT_CHARS,
        )
        hydrated[identity] = {
            **window,
            "content_source": "message",
            "role": stored.get("role"),
            "source": stored.get("source") or "",
            **_lcm_recall_event_time(stored),
        }
    return hydrated


def _lcm_recall_bounded_reason(
    arm: str, scanned: int | None, total: int | None
) -> str:
    """Degraded-reasons text for a ``coverage='bounded'`` arm (SCAN-1).

    Names the arm and the scanned/total ratio so a caller can see that the arm
    scored only the most-recent slice of the corpus (older archived vectors were
    excluded by the recency-bounded candidate scan), rather than the truncation
    being silent.
    """
    if scanned is not None and total is not None:
        return (
            f"{arm} arm coverage bounded: scored the {scanned} most-recent of "
            f"{total} vectors (older vectors excluded)"
        )
    return (
        f"{arm} arm coverage bounded: scored only the most-recent slice of the "
        "corpus (older vectors excluded)"
    )


def _lcm_recall_approx_reason(arm: str) -> str:
    """Disclosure text for a ``coverage='full_approx'`` arm (FIX 2).

    The two-stage path reaches the WHOLE corpus but stage-1 Hamming keeps only
    the M=mult*k lowest-distance survivors before the exact rescore, so its top-k
    is an approximate (recall@M) result rather than the exact top-k the
    exact-scan 'full' coverage gives. Surfaced like 'bounded' so a caller can see
    the ranking is approximate rather than assuming exhaustive exactness.
    """
    return (
        f"{arm} arm coverage full_approx: whole corpus reached via the binary "
        "prescreen, but top-k is approximate (stage-1 keeps only the closest "
        "survivors before exact rescore)"
    )


class _RecallFtsHits(list):
    """Keep the served index with this arm's rows, including an empty result."""
    fts_index = "plain"


def _lcm_recall_fts_arm(
    engine: "LCMEngine",
    query: str,
    *,
    candidate_limit: int,
    deadline: float,
    excluded_session_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """FTS arm: raw messages across all eligible sessions."""
    # Emoji cannot reach MATCH: keep the raw query on the existing LIKE route.
    or_query = "" if contains_emoji(query) else build_recall_or_query(query)
    payload = _lcm_grep_full_text_with_deadline(
        {
            "query": or_query or query,
            "_allow_operators": bool(or_query),
            "_stemmed": True,
            "sort": "relevance" if or_query or contains_emoji(query) else "recency",
            "mode": "recall",
            "session_scope": "all",
            "limit": candidate_limit,
            "_excluded_session_ids": excluded_session_ids,
        },
        engine=engine,
        deadline=deadline,
        limit_cap=_LCM_GREP_HYBRID_CANDIDATE_CAP,
    )
    if "error" in payload:
        return [], payload
    # Grep survives a message-search crash on its other stages, but THIS arm
    # consumes only the message rows -- a disclosed crash means the arm
    # produced nothing, which must surface as coverage 'none', never as an
    # 'ok' empty result indistinguishable from an empty corpus (#256).
    message_search_error = payload.get("message_search_error")
    if message_search_error:
        return [], {"error": f"full-text message search failed: {message_search_error}"}
    hits = _RecallFtsHits()
    hits.fts_index = payload.get("fts_index", "plain")
    for row in payload.get("results", []):
        store_id = row.get("store_id")
        if store_id is None:
            continue
        hit = {
            "kind": "message_excerpt",
            "store_id": store_id,
            "session_id": row.get("session_id"),
            "source": row.get("source") or "",
            "role": row.get("role"),
            "timestamp": row.get("timestamp") or 0,
            "content_offset": 0,
            "snippet": (row.get("snippet") or "")[:_LCM_RECALL_SNIPPET_CHARS],
            "from_current_session": bool(row.get("from_current_session")),
        }
        hit["expand_hint"] = _lcm_recall_excerpt_expand_hint(hit)
        hits.append(hit)
    return hits, None


def _lcm_recall_scan_bounds(engine: "LCMEngine") -> dict[str, Any]:
    """Full-corpus scan settings shared by both vector arms.

    lcm_recall promises "all conversations, all time", so both arms scan the
    WHOLE corpus in ``recall_scan_rows``-sized batches. The optional cap and
    latency budget default to 0 (no early stop); either one firing degrades the
    arm to ``coverage='bounded'``, which the caller already discloses.
    """
    return {
        "full_scan": True,
        "scan_max_rows": max(0, int(getattr(engine._config, "recall_scan_max_rows", 0))),
        "scan_budget_s": max(
            0.0, float(getattr(engine._config, "recall_scan_budget_s", 0.0))
        ),
    }


def _lcm_recall_summary_source_hits(
    engine: "LCMEngine",
    nodes: list[tuple[Any, float]],
    *,
    current: str | None,
    candidate_limit: int,
    lead_limit: int,
    deadline: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Carry summary-KNN relevance onto the nodes' SOURCE MESSAGES.

    Reference-strict delivery cannot hand out a summary, and simply dropping the
    hits would silently amputate the arm: RRF keys a summary by node and a
    message by store_id, so with rerank off (the default) removing summary
    entries after fusion leaves every message score and the whole order exactly
    as if the arm had never run -- a session reachable ONLY by summary KNN would
    contribute no evidence at all.

    So the arm emits ordinary ``message_excerpt`` candidates for the rows beneath
    each ranked node, in node-rank order. They are NOT the summary wearing a
    message's content: each is a real row with its own identity, its own verbatim
    excerpt and its own truthful ``content_offset``, fused by RRF against the FTS
    and chunk arms like any other candidate and competing on merit rather than
    inheriting the node's slot. The node handles come back separately as
    non-evidence leads.

    Bounded fan-out: within a node there is no per-message relevance signal (that
    is what the FTS and chunk arms are for), so a deterministic ``store_id``-
    ordered slice per node is taken and the arm stays inside its candidate
    budget. The purpose is to make the SESSION reachable with citable evidence,
    not to rank inside it.
    """
    def require_remaining(stage: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"summary source expansion deadline exhausted before {stage}")
        return remaining

    require_remaining("database connection")
    db_path = Path(engine._store.db_path).resolve()
    uri = f"{db_path.as_uri()}?mode=ro"
    conn: sqlite3.Connection | None = None
    expired = [False]

    def interrupt_if_expired() -> int:
        if time.monotonic() >= deadline:
            expired[0] = True
            return 1
        return 0

    try:
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=max(0.001, require_remaining("database connection")),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(interrupt_if_expired, 1000)
        read_io = RetrievalContext(engine._store, engine._dag).bind(conn, conn)
        read_store, read_dag = read_io.store, read_io.dag

        leads: list[dict[str, Any]] = []
        ordered_ids: list[int] = []
        seen: set[int] = set()
        for node, _score in nodes:
            require_remaining("lineage traversal")
            if len(leads) < max(0, lead_limit):
                lead: dict[str, Any] = {
                    "node_id": node.node_id,
                    "session_id": node.session_id,
                    "from_current_session": bool(current)
                    and node.session_id == current,
                }
                hint = _lcm_recall_summary_expand_hint(lead)
                if hint:
                    lead["expand_hint"] = hint
                leads.append(lead)
            if len(ordered_ids) >= candidate_limit:
                continue
            for store_id in read_dag.source_message_ids(
                node.node_id, limit=_LCM_RECALL_SUMMARY_SOURCE_PER_NODE
            ):
                if store_id in seen:
                    continue
                seen.add(store_id)
                ordered_ids.append(store_id)
        require_remaining("message hydration")

        hits: list[dict[str, Any]] = []
        rows = (
            read_store.get_batch(ordered_ids[:candidate_limit]) if ordered_ids else {}
        )
        for store_id in ordered_ids[:candidate_limit]:
            require_remaining("message shaping")
            row = rows.get(store_id)
            if row is None:
                continue
            content = str(row.get("content") or "")
            if not content:
                continue
            session_id = row.get("session_id")
            hit = {
                "kind": "message_excerpt",
                "store_id": store_id,
                "session_id": session_id,
                "source": row.get("source") or "",
                "role": row.get("role"),
                "timestamp": row.get("timestamp") or 0,
                # The excerpt is the row's own prefix, so offset 0 is the truth
                # here rather than the fabricated default a consumer would guess.
                "content_offset": 0,
                "snippet": content[:_LCM_RECALL_SNIPPET_CHARS],
                "from_current_session": bool(current) and session_id == current,
            }
            hit["expand_hint"] = _lcm_recall_excerpt_expand_hint(hit)
            hits.append(hit)
        return hits, leads
    except sqlite3.OperationalError as exc:
        if expired[0] or time.monotonic() >= deadline:
            raise TimeoutError("summary source expansion deadline exhausted") from exc
        raise
    finally:
        if conn is not None:
            conn.close()


def _lcm_recall_summary_arm(
    engine: "LCMEngine",
    *,
    query_vector: list[float],
    provider: Any,
    candidate_limit: int,
    lead_limit: int,
    deadline: float,
    candidate_session_ids: list[str] | None,
    reference_strict: bool = False,
) -> tuple[list[dict[str, Any]], str, int | None, int | None, list[dict[str, Any]]]:
    """Summary KNN arm: embedded summaries across all eligible sessions."""
    knn_results = _run_within_deadline(
        lambda: run_knn(
            engine,
            query_vector=query_vector,
            provider=provider,
            knn_limit=candidate_limit,
            deadline=deadline,
            since=None,
            until=None,
            conversation_ids=candidate_session_ids,
            source=None,
            vector_store_cls=VectorStore,
            scan_rows=max(1, int(getattr(engine._config, "recall_scan_rows", 25_000))),
            **_lcm_recall_scan_bounds(engine),
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-summary-knn",
    )
    coverage = knn_results.coverage
    ranked_rows = list(knn_results)
    if coverage == "none" or not ranked_rows:
        return [], coverage, knn_results.scanned, knn_results.total, []
    nodes = _run_within_deadline(
        lambda: hydrate_semantic_nodes(
            engine,
            ranked_rows=ranked_rows,
            knn_limit=candidate_limit,
            deadline=deadline,
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-summary-hydrate",
    )
    current = engine.current_session_id
    if reference_strict:
        source_hits, leads = _lcm_recall_summary_source_hits(
            engine,
            nodes,
            current=current,
            candidate_limit=candidate_limit,
            lead_limit=lead_limit,
            deadline=deadline,
        )
        return source_hits, coverage, knn_results.scanned, knn_results.total, leads
    hits: list[dict[str, Any]] = []
    for node, _score in nodes:
        source_store_id = (
            int(node.source_ids[0])
            if node.source_type == "messages" and node.source_ids
            else None
        )
        hit = {
            "kind": "summary",
            "node_id": node.node_id,
            "store_id": source_store_id,
            "session_id": node.session_id,
            "timestamp": node.latest_at or node.created_at or 0,
            "snippet": (node.summary or "")[:_LCM_RECALL_SNIPPET_CHARS],
            "from_current_session": bool(current) and node.session_id == current,
        }
        hit["expand_hint"] = _lcm_recall_summary_expand_hint(hit)
        hits.append(hit)
    return hits, coverage, knn_results.scanned, knn_results.total, []


def _lcm_recall_chunk_arm(
    engine: "LCMEngine",
    *,
    query_vector: list[float],
    provider: Any,
    candidate_limit: int,
    deadline: float,
    candidate_session_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], str, int | None, int | None]:
    """Chunk KNN arm: verbatim chunk vectors across all eligible sessions."""
    knn_results = _run_within_deadline(
        lambda: run_chunk_knn(
            engine,
            query_vector=query_vector,
            provider=provider,
            knn_limit=candidate_limit,
            deadline=deadline,
            since=None,
            until=None,
            conversation_ids=candidate_session_ids,
            source=None,
            vector_store_cls=VectorStore,
            scan_rows=max(1, int(getattr(engine._config, "recall_scan_rows", 25_000))),
            **_lcm_recall_scan_bounds(engine),
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-chunk-knn",
    )
    coverage = knn_results.coverage
    ranked_rows = list(knn_results)
    if coverage == "none" or not ranked_rows:
        return [], coverage, knn_results.scanned, knn_results.total
    raw_hits = _run_within_deadline(
        lambda: hydrate_chunk_hits(
            engine,
            ranked_rows=ranked_rows,
            knn_limit=candidate_limit,
            deadline=deadline,
            snippet_chars=_LCM_RECALL_SNIPPET_CHARS,
        ),
        remaining_s=deadline - time.monotonic(),
        name="lcm-recall-chunk-hydrate",
    )
    current = engine.current_session_id
    hits: list[dict[str, Any]] = []
    for hit, _score in raw_hits:
        hit["from_current_session"] = bool(current) and hit.get("session_id") == current
        hit["expand_hint"] = _lcm_recall_excerpt_expand_hint(hit)
        hits.append(hit)
    return hits, coverage, knn_results.scanned, knn_results.total


def _lcm_recall_rerank(
    provider: Any,
    query: str,
    ordered: list[dict[str, Any]],
    *,
    window: int,
    deadline: float,
    config: Any,
) -> tuple[list[dict[str, Any]], str, list[list[Any]]]:
    """Optionally REORDER the top ``window`` fused candidates in ONE API call.

    Default OFF. This is a pure rank-reorder WITHIN the window: the reranker's
    relevance scores decide the intra-window order but are never spliced onto the
    ``rrf_score``/``_final_score`` scale (the two live on incompatible scales, so a
    ~0-1 voyage score would otherwise dominate the ~0.05-max RRF score). Entries
    outside the window keep their incoming (post scope/recency prior) order.

    ``ordered`` MUST already carry the scope/recency prior so the window reflects
    the true top-N rather than the raw RRF order (RERANK-1). Any failure (no
    provider, non-voyage, network, deadline) skips silently back to the incoming
    order with a ``skipped: <reason>`` status. Privacy-policy failures remain
    loud, matching the cloud query-embedding path.
    """
    if not bool(getattr(config, "rerank_enabled", False)):
        return ordered, "disabled", []
    if (
        provider is None
        or getattr(provider, "provider_id", "") != "voyage"
        or not hasattr(provider, "rerank")
    ):
        return ordered, "skipped: rerank requires the voyage provider", []
    head = ordered[:window]
    if not head:
        return ordered, "skipped: no candidates to rerank", []
    outbound_query = str(query)
    documents = [str(entry["hit"].get("snippet") or "") for entry in head]
    privacy_revision: str | None = None
    if embedding_provider_requires_privacy(
        str(getattr(provider, "provider_id", "") or "")
    ):
        outbound_query, privacy_revision, _changed = protect_embedding_text(
            outbound_query, config
        )
        documents = [
            protect_embedding_text(
                document, config, expected_revision=privacy_revision
            )[0]
            for document in documents
        ]

    def invoke() -> list[tuple[int, float]]:
        if privacy_revision is not None:
            validate_embedding_privacy_dispatch(
                [outbound_query, *documents],
                config,
                expected_revision=privacy_revision,
            )
        return provider.rerank(
            outbound_query,
            documents,
            top_k=len(documents),
            timeout=max(0.001, deadline - time.monotonic()),
            model=(
                str(getattr(config, "rerank_model", "") or "").strip()
                or "rerank-2.5-lite"
            ),
        )

    try:
        ranked = _run_within_deadline(
            invoke,
            remaining_s=deadline - time.monotonic(),
            name="lcm-recall-rerank",
        )
    except EmbeddingPrivacyPolicyError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure => skip rerank
        return ordered, f"skipped: {exc}", []
    if not ranked:
        return ordered, "skipped: empty rerank result", []

    rerank_margin = float(getattr(config, "rerank_margin", 0.0) or 0.0)
    if rerank_margin <= 0:
        # This is the historical rerank path. Keep it structurally unchanged for
        # rerank_margin == 0.0 so the default output ordering remains byte-identical.
        rerank_scores = [
            [int(index), head[index]["hit"].get("session_id"), float(relevance)]
            for index, relevance in ranked
            if 0 <= index < len(head)
        ]
        reordered: list[dict[str, Any]] = []
        seen: set[int] = set()
        for index, _relevance in ranked:
            if 0 <= index < len(head) and index not in seen:
                reordered.append(head[index])
                seen.add(index)
        for index, entry in enumerate(head):
            if index not in seen:
                reordered.append(entry)
        reordered.extend(ordered[window:])
        return reordered, "applied", rerank_scores

    # Keep score telemetry in provider order, before the positive-margin gate
    # changes the resulting candidate order. VoyageProvider already filters
    # invalid indices; the bounds check also keeps test doubles safe.
    valid_ranked = [
        (int(index), float(relevance))
        for index, relevance in ranked
        if 0 <= int(index) < len(head)
    ]
    rerank_scores = [
        [index, head[index]["hit"].get("session_id"), relevance]
        for index, relevance in valid_ranked
    ]

    # If the provider did not score the incoming incumbent, fail open rather
    # than guessing its relevance.
    incumbent_relevance = next(
        (relevance for index, relevance in valid_ranked if index == 0),
        None,
    )
    if incumbent_relevance is None:
        return ordered, "skipped: partial rerank result", rerank_scores
    challenger_index, challenger_relevance = max(
        valid_ranked,
        key=lambda item: (item[1], -item[0]),
    )
    if (
        challenger_index == 0
        or challenger_relevance - incumbent_relevance < rerank_margin
    ):
        reordered = [head[0]]
        seen = {0}
        for index, _relevance in valid_ranked:
            if index != 0 and index not in seen:
                reordered.append(head[index])
                seen.add(index)
        for index, entry in enumerate(head):
            if index not in seen:
                reordered.append(entry)
        reordered.extend(ordered[window:])
        return reordered, "applied: rank1-held", rerank_scores

    # The positive-margin override takes exactly the historical rerank order.
    reordered: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, _relevance in valid_ranked:
        if index not in seen:
            reordered.append(head[index])
            seen.add(index)
    for index, entry in enumerate(head):
        if index not in seen:
            reordered.append(entry)
    reordered.extend(ordered[window:])
    return reordered, "applied", rerank_scores


# Re-export for existing direct callers only. lcm_recall runs retrieval_core.place_fts_anchor, so patch that, not this name.
_lcm_recall_fts_anchor = place_fts_anchor


def _lcm_recall_preflight_scan_deadline(now: float, deadline: float) -> float:
    """Derive the corpus preflight's independent scan deadline.

    Bounded by a small absolute cap and a fraction of the remaining request
    budget, and never past the request deadline itself. A preflight that hits
    this bound returns False (fail-open), leaving the full remaining budget
    with the FTS arm instead of starving every arm at once.
    """
    remaining = max(0.0, deadline - now)
    return min(
        deadline,
        now
        + min(
            _LCM_RECALL_PREFLIGHT_MAX_BUDGET_S,
            max(0.001, remaining * _LCM_RECALL_PREFLIGHT_BUDGET_FRACTION),
        ),
    )


def _lcm_recall_has_usable_vector_corpus(
    engine: "LCMEngine",
    *,
    run_summary: bool,
    run_chunk: bool,
    provider_name: str,
    model_name: str,
    excluded_session_ids: set[str],
    deadline: float,
) -> bool:
    """Return whether a requested semantic arm has vectors for its selected profile.

    This read-only preflight mirrors ``VectorStore`` profile ordering and requires
    a live metadata+vector pair whose SOURCE ROW is still present -- the same
    liveness predicate the real scans apply (``summary_nodes`` for the summary
    arm, ``messages`` for the chunk arm). Merely configuring embeddings,
    registering an empty profile, or leaving orphaned vectors behind a
    best-effort purge therefore cannot shorten the only usable FTS fallback.
    Missing or partial optional schema fails closed without materializing
    feature tables.

    Session eligibility is part of the real scans' predicate too: the vector
    arms consume ``candidate_session_ids`` as a hard scope filter, so a corpus
    whose only live pairs sit in EXCLUDED sessions is not a semantic route for
    this request. Both source joins therefore carry the same exclusion
    predicate (``NOT IN`` over the request's exclusion set is equivalent to the
    whitelist for an existence probe: every source row's session is enumerated
    by the scope tables, so eligible == not excluded).
    """
    provider_name = str(provider_name or "").strip().lower()
    # Profiles are registered under ``provider.provider_id`` (command.py warmup),
    # never under the operator's config spelling, so EVERY alias
    # ``resolve_provider`` accepts must collapse to the same canonical ID here.
    # An alias left unmapped makes the preflight miss a real corpus, the cap
    # never engages, and a slow FTS scan can still starve the valid vector arm --
    # the exact starvation this preflight exists to prevent. The authority for
    # these sets is embedding_provider.py:1965-1985 (voyage/voyageai,
    # fastembed/fast-embed, openai-compatible/openai/siliconflow); ollama has no
    # alias. Adding an alias there requires adding it here.
    provider_name = {
        "voyageai": "voyage",
        "fast-embed": "fastembed",
        "openai": "openai-compatible",
        "siliconflow": "openai-compatible",
    }.get(provider_name, provider_name)
    model_name = str(model_name or "").strip()
    if not provider_name or not model_name or time.monotonic() >= deadline:
        return False
    privacy_revision = None
    if embedding_provider_requires_privacy(provider_name):
        try:
            privacy_revision = embedding_privacy_revision(engine._config)
        except EmbeddingPrivacyPolicyError:
            pass  # Preserve the preflight; query dispatch still raises (#367).
    scan_deadline = _lcm_recall_preflight_scan_deadline(time.monotonic(), deadline)

    conn: sqlite3.Connection | None = None
    try:
        db_path = Path(engine._store.db_path).resolve()
        remaining_s = max(0.001, scan_deadline - time.monotonic())
        conn = sqlite3.connect(
            f"{db_path.as_uri()}?mode=ro",
            uri=True,
            timeout=min(0.05, remaining_s),
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.set_progress_handler(
            lambda: 1 if time.monotonic() >= scan_deadline else 0,
            1000,
        )

        def table_columns(table: str) -> set[str]:
            return {
                str(row[1])
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }

        exclusion_where = ""
        exclusion_params: tuple[str, ...] = ()
        if excluded_session_ids:
            exclusion_params = tuple(sorted(excluded_session_ids))
            placeholders = ",".join("?" for _ in exclusion_params)
            exclusion_where = f" AND src.session_id NOT IN ({placeholders})"

        def corpus_has_vectors(
            *,
            task: str,
            selected_model: str,
            meta_table: str,
            vector_table: str,
            id_column: str,
            source_join: str,
            source_where: str = "",
        ) -> bool:
            # The progress handler only fires every 1000 VM ops, so a probe
            # that starts after scan expiry could still slip through whole;
            # this check makes the scan budget binding regardless of handler
            # granularity.
            if time.monotonic() >= scan_deadline:
                return False
            if privacy_revision is not None and _lcm_embedding_profile_revision(
                conn, provider=provider_name, model=selected_model, task=task
            ) != privacy_revision:
                return False
            profile = conn.execute(
                """
                SELECT identity_hash
                FROM lcm_embedding_profile
                WHERE model_name = ? AND provider = ? AND task = ?
                ORDER BY (active = 1 AND archived_at IS NULL) DESC,
                         registered_at DESC, identity_hash DESC
                LIMIT 1
                """,
                (selected_model, provider_name, task),
            ).fetchone()
            if profile is None:
                return False
            identity_hash = str(profile["identity_hash"])
            row = conn.execute(
                f"""
                SELECT 1
                FROM {meta_table} m
                JOIN {vector_table} v
                  ON v.{id_column} = m.{id_column}
                 AND v.identity_hash = m.identity_hash
                {source_join}
                WHERE m.identity_hash = ? AND m.archived = 0{source_where}{exclusion_where}
                LIMIT 1
                """,
                (identity_hash, *exclusion_params),
            ).fetchone()
            return row is not None

        # A metadata+vector pair is NOT a usable corpus on its own: both real
        # scans INNER-join the source row, so a vector whose source is gone (or,
        # for summaries, suppressed) scores nothing. That state is reachable and
        # supported on both arms -- ``_purge_embeddings_for_nodes``
        # (engine.py:4116-4147) and ``_archive_chunks_for_messages``
        # (engine.py:4149-4183) both swallow failures on purpose, the former
        # precisely because "the summary_nodes join still keeps orphaned vectors
        # out of ranking". Without these joins such an orphan reports a semantic
        # route, cuts FTS to a quarter of the budget, and leaves the request with
        # no arm that can return anything. Authority for the predicates:
        # vector_store.py:1414 + 1892 with suppressed_at at 1395/1713/1872
        # (summary), vector_store.py:1839 + 2647 (chunk).
        if run_summary:
            # Legacy databases predate ``suppressed_at``; the real scan gates the
            # clause on the column existing, so this one does too.
            summary_where = (
                " AND src.suppressed_at IS NULL"
                if "suppressed_at" in table_columns("summary_nodes")
                else ""
            )
            if corpus_has_vectors(
                task="summary",
                selected_model=model_name,
                meta_table="lcm_embedding_meta",
                vector_table="lcm_embedding_vectors",
                id_column="embedded_id",
                source_join=(
                    "JOIN summary_nodes src "
                    "ON src.node_id = CAST(m.embedded_id AS INTEGER)"
                ),
                source_where=summary_where,
            ):
                return True
        if run_chunk and corpus_has_vectors(
            task="chunk",
            selected_model=default_chunk_model(provider_name, model_name),
            meta_table="lcm_chunk_meta",
            vector_table="lcm_chunk_vectors",
            id_column="chunk_id",
            source_join="JOIN messages src ON src.store_id = m.store_id",
        ):
            return True
        return False
    except (OSError, ValueError, sqlite3.Error):
        return False
    finally:
        if conn is not None:
            conn.close()


def lcm_recall(args: Dict[str, Any], **kwargs) -> str:
    """Search the agent's entire memory (all conversations, all time) by meaning.

    Fuses three arms over the whole local database — FTS raw messages, embedded
    summary KNN, and verbatim chunk KNN — with RRF, then applies a soft
    scope/recency prior and (optionally) a cross-encoder rerank. The current
    conversation is a ranking boost by default; explicit session exclusions are
    hard filters applied before fusion.
    """
    request_started = time.monotonic()
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    query = _coerce_query_arg(args.get("query"))
    if not query:
        return json.dumps({"error": "No query provided"})

    excluded_session_ids, exclusions_error = _parse_retrieval_excluded_session_ids(
        args, engine.current_session_id
    )
    if exclusions_error:
        return json.dumps({"error": exclusions_error})

    parsed_limit = _parse_int_value(args.get("limit", _LCM_RECALL_DEFAULT_LIMIT), _LCM_RECALL_DEFAULT_LIMIT)
    if parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _LCM_RECALL_LIMIT_CAP)

    scope_bias, scope_bias_error = _parse_optional_float(args.get("scope_bias"), "scope_bias")
    if scope_bias_error:
        return json.dumps({"error": scope_bias_error})
    if scope_bias is None:
        scope_bias = _LCM_RECALL_DEFAULT_SCOPE_BIAS
    scope_bias = max(0.0, min(1.0, float(scope_bias)))

    include = str(args.get("include") or "all").strip().lower()
    if include not in _LCM_RECALL_VALID_INCLUDE:
        return json.dumps({"error": "include must be one of: all, summaries, verbatim"})

    detail = str(args.get("detail") or "snippets").strip().lower()
    if detail not in _LCM_RECALL_VALID_DETAIL:
        return json.dumps({"error": "detail must be one of: snippets, answer_ready"})

    # Reference-strict delivery applies to the citation-bearing mode only: the
    # default 'snippets' response makes no source-span claim and carries no
    # hydration, so strictness there would drop evidence for no reference gain.
    reference_strict = detail == "answer_ready" and bool(
        getattr(engine._config, "recall_reference_strict", True)
    )

    raw_seen_refs = args.get("seen_refs")
    # An empty cursor starts an answer_ready delta; with any other detail it
    # means "not provided" (#1194).
    delta_requested = "seen_refs" in args and not (
        raw_seen_refs == [] and detail != "answer_ready"
    )
    if delta_requested and detail != "answer_ready":
        return json.dumps({"error": "seen_refs requires detail='answer_ready'"})
    if delta_requested and not isinstance(raw_seen_refs, list):
        return json.dumps({"error": "seen_refs must be an array"})
    if delta_requested and len(raw_seen_refs) > 100:
        return json.dumps({"error": "seen_refs accepts at most 100 refs"})
    seen_refs = {str(value) for value in (raw_seen_refs or [])}
    include_occurrence_time = bool(args.get("include_occurrence_time", False))
    if include_occurrence_time and detail != "answer_ready":
        return json.dumps({"error": "include_occurrence_time requires detail='answer_ready'"})

    # lcm_recall fans out three arms + fusion/hydration/rerank, so it uses its own
    # (larger) budget rather than lcm_grep's single-arm query deadline (sprint-opt-2).
    timeout_s = max(0.001, float(getattr(engine._config, "recall_query_timeout_s", 8.0)))
    deadline = request_started + timeout_s

    # Resolved under the deadline, not before it: this is DB work, and the
    # vector arms below consume its result as a hard scope filter.
    candidate_session_ids, session_scope_timed_out = _retrieval_candidate_session_ids(
        engine, excluded_session_ids, deadline=deadline
    )

    candidate_limit = min(_LCM_GREP_HYBRID_CANDIDATE_CAP, max(50, limit * 4))
    rerank_window = min(50, max(1, limit * 4))
    rerank_window_limit = int(getattr(engine._config, "rerank_window_limit", 0))
    if rerank_window_limit > 0:
        rerank_window = min(rerank_window, rerank_window_limit)

    embeddings_enabled = bool(getattr(engine._config, "embeddings_enabled", False))
    # FTS runs for verbatim/all normally, but ALSO whenever embeddings are
    # disabled -- including for include='summaries', whose only vector arm is dead
    # in that state. Without this, include='summaries' + embeddings-off returns
    # zero hits instead of degrading to the full-text arm (F4-degrade-to-fts).
    run_fts = include in {"all", "verbatim"} or not embeddings_enabled
    run_summary = include in {"all", "summaries"}
    run_chunk = include in {"all", "verbatim"}

    arm_hits: dict[str, list[dict[str, Any]]] = {}
    # Node handles from the summary arm. Reference-strict delivery routes summary
    # relevance through the source messages, so the nodes themselves are surfaced
    # here as non-evidence drill-down leads instead of as hits.
    summary_leads: list[dict[str, Any]] = []
    coverage: dict[str, str] = {}
    degraded_reasons: list[str] = []
    embedding_query_metrics: list[dict[str, Any]] = []
    timed_out = False
    provider: Any = None
    provider_override = str(kwargs.get("provider_override") or "").strip()

    served_fts_index = "plain"
    # -- FTS arm (the default-on value: works with embeddings disabled) --
    def _run_fts_arm(fallback: bool = False) -> None:
        nonlocal timed_out, served_fts_index
        fts_deadline = deadline
        fts_sub_budget_applied = False
        configured_provider = provider_override or str(
            getattr(engine._config, "embedding_provider", "") or ""
        ).strip()
        configured_model = str(
            getattr(engine._config, "embedding_model", "") or ""
        ).strip()
        # A fallback runs because the semantic arm already failed, so it gets the
        # whole remaining budget rather than the cap reserved for that arm (#887).
        semantic_available = (
            not fallback
            and embeddings_enabled
            and (run_summary or run_chunk)
            # A timed-out scope resolution drops the vector arms (fail closed)
            # and an empty whitelist leaves them nothing eligible to return;
            # in both states FTS is the only arm that can produce hits, so the
            # sub-budget cap must not engage.
            and not session_scope_timed_out
            and candidate_session_ids != []
            and _lcm_recall_has_usable_vector_corpus(
                engine,
                run_summary=run_summary,
                run_chunk=run_chunk,
                provider_name=configured_provider,
                model_name=configured_model,
                excluded_session_ids=excluded_session_ids,
                deadline=deadline,
            )
        )
        if semantic_available:
            now = time.monotonic()
            remaining_s = max(0.0, deadline - now)
            fts_budget_s = min(
                remaining_s,
                max(0.001, remaining_s * _LCM_RECALL_FTS_MAX_BUDGET_FRACTION),
            )
            fts_deadline = now + fts_budget_s
            fts_sub_budget_applied = fts_deadline < deadline
        try:
            hits, fts_error = _lcm_recall_fts_arm(
                engine,
                query,
                candidate_limit=candidate_limit,
                deadline=fts_deadline,
                excluded_session_ids=excluded_session_ids,
            )
        except (_WorkerCapacityError, TimeoutError) as exc:
            hits, fts_error = [], {"error": str(exc)}
        if fts_error is not None:
            coverage["fts"] = "none"
            # The sub-budget above is a DELIBERATE cap, so its expiry is not
            # budget exhaustion: the request still has time left and the vector
            # arms are about to spend it. Reporting it as the request-level
            # ``timeout`` would tell the agent its whole recall was cut short
            # when in fact only this arm was fenced -- an assertion the run
            # itself refutes. It degrades as missing FTS coverage instead
            # (coverage stays in the none/ok vocabulary, with its own reason),
            # and a genuine request-deadline overrun still falls through to the
            # exhaustion branch below.
            # ``timeout``-shaped errors are not all deadline expiries: worker
            # saturation returns the same shape (``_run_within_deadline``
            # raises ``_WorkerCapacityError``; ``_lcm_grep_full_text_with_
            # deadline`` converts both causes to the deadline payload). The
            # cap may only take credit when the sub-deadline actually elapsed
            # -- otherwise an arm that never ran would be reported as
            # deliberately fenced.
            capped_expiry = (
                fts_sub_budget_applied
                and bool(fts_error.get("timeout"))
                and time.monotonic() >= fts_deadline
                and time.monotonic() < deadline
            )
            if capped_expiry:
                degraded_reasons.append(
                    "full-text arm capped to preserve semantic recall budget"
                )
            else:
                degraded_reasons.append("full-text arm unavailable")
                timed_out = timed_out or bool(fts_error.get("timeout"))
        else:
            served_fts_index = getattr(hits, "fts_index", "plain")
            arm_hits["fts"] = hits
            coverage["fts"] = "ok"

    if run_fts:
        _run_fts_arm()
    # Summaries-only recall falls back to full text when its semantic arm is
    # unavailable, provided the request still has time left.
    summary_fts_fallback = False

    # -- Vector arms. Local/same-model corpora share one query embedding;
    # Voyage's context chunk corpus resolves and embeds with its own model. --
    if run_summary or run_chunk:
        # Ordered before the empty-scope branch on purpose: a timed-out
        # enumeration also yields [], and must NOT be reported as the
        # 'full' coverage of a genuinely empty eligible scope.
        if session_scope_timed_out:
            timed_out = True
            degraded_reasons.append("session exclusion scope resolution timed out")
            if run_summary:
                coverage["summary"] = "none"
            if run_chunk:
                coverage["chunk"] = "none"
        elif candidate_session_ids == []:
            if run_summary:
                arm_hits["summary"] = []
                coverage["summary"] = "full"
            if run_chunk:
                arm_hits["chunk"] = []
                coverage["chunk"] = "full"
        elif not embeddings_enabled:
            degraded_reasons.append("semantic retrieval is disabled")
            if run_summary:
                coverage["summary"] = "disabled"
            if run_chunk:
                coverage["chunk"] = "disabled"
        elif time.monotonic() >= deadline:
            timed_out = True
        else:
            query_vector: list[float] | None = None
            chunk_provider: Any = None
            chunk_query_vector: list[float] | None = None
            try:
                provider = _run_within_deadline(
                    lambda: _resolve_recall_provider(
                        engine,
                        deadline=deadline,
                        provider_override=provider_override,
                    ),
                    remaining_s=deadline - time.monotonic(),
                    name="lcm-provider-resolution",
                )
                if provider is None:
                    degraded_reasons.append("embedding provider is not configured")
                    summary_fts_fallback = not run_fts and run_summary
                elif run_summary:
                    query_vector = _lcm_grep_embed_query(
                        provider,
                        query,
                        engine=engine,
                        task="summary",
                        remaining_s=deadline - time.monotonic(),
                    )
                    embedding_query_metrics.append(
                        _lcm_embedding_query_metric(provider)
                    )
            except EmbeddingIdentityStaleError:
                coverage["summary"] = "none"
                degraded_reasons.append(_EMBEDDING_IDENTITY_STALE_REASON)
                summary_fts_fallback = not run_fts and run_summary
            except EmbeddingPrivacyPolicyError:
                # Deterministic configuration error — never degrade (#367).
                raise
            except VoyageError as exc:
                provider = None
                degraded_reasons.append(f"query embedding failed: {exc}")
                summary_fts_fallback = not run_fts and run_summary
            except TimeoutError:
                provider = None
                timed_out = True
                summary_fts_fallback = not run_fts and run_summary
            except Exception as exc:  # noqa: BLE001 - degrade, never bare-error the whole tool
                provider = None
                degraded_reasons.append(f"embedding provider unavailable: {exc}")
                summary_fts_fallback = not run_fts and run_summary
            if summary_fts_fallback and "fts" not in coverage and time.monotonic() < deadline:
                _run_fts_arm(fallback=True)

            if run_chunk and provider is not None:
                try:
                    chunk_provider = _run_within_deadline(
                        lambda: _resolve_recall_chunk_provider(
                            engine, provider, deadline=deadline
                        ),
                        remaining_s=deadline - time.monotonic(),
                        name="lcm-chunk-provider-resolution",
                    )
                    if chunk_provider is None:
                        degraded_reasons.append(
                            "chunk embedding provider is not configured"
                        )
                    elif (
                        query_vector is not None
                        and chunk_provider.provider_id == provider.provider_id
                        and chunk_provider.model_id == provider.model_id
                    ):
                        chunk_query_vector = _lcm_grep_embed_query(
                            chunk_provider, query, engine=engine, task="chunk",
                            remaining_s=deadline - time.monotonic(),
                            reuse_vector=query_vector,
                        )
                    else:
                        chunk_query_vector = _lcm_grep_embed_query(
                            chunk_provider,
                            query,
                            engine=engine,
                            task="chunk",
                            remaining_s=deadline - time.monotonic(),
                        )
                        embedding_query_metrics.append(
                            _lcm_embedding_query_metric(chunk_provider)
                        )
                except EmbeddingIdentityStaleError:
                    coverage["chunk"] = "none"
                    degraded_reasons.append(_EMBEDDING_IDENTITY_STALE_REASON)
                except EmbeddingPrivacyPolicyError:
                    # Deterministic configuration error — never degrade (#367).
                    raise
                except VoyageError as exc:
                    degraded_reasons.append(f"chunk query embedding failed: {exc}")
                except TimeoutError:
                    timed_out = True
                except Exception as exc:  # noqa: BLE001
                    degraded_reasons.append(
                        f"chunk embedding provider unavailable: {exc}"
                    )

            if query_vector is not None:
                if run_summary:
                    try:
                        hits, cov, scanned, total, summary_leads = _lcm_recall_summary_arm(
                            engine,
                            query_vector=query_vector,
                            provider=provider,
                            candidate_limit=candidate_limit,
                            lead_limit=limit,
                            deadline=deadline,
                            candidate_session_ids=candidate_session_ids,
                            reference_strict=reference_strict,
                        )
                        arm_hits["summary"] = hits
                        coverage["summary"] = cov
                        if cov == "none":
                            degraded_reasons.append("summary vectors are unavailable")
                            summary_fts_fallback = not run_fts and run_summary
                        elif cov == "bounded":
                            degraded_reasons.append(
                                _lcm_recall_bounded_reason("summary", scanned, total)
                            )
                        elif cov == "full_approx":
                            degraded_reasons.append(
                                _lcm_recall_approx_reason("summary")
                            )
                    except TimeoutError:
                        timed_out = True
                        coverage["summary"] = "none"
                        summary_fts_fallback = not run_fts and run_summary
                    except EmbeddingIdentityStaleError:
                        coverage["summary"] = "none"
                        degraded_reasons.append(_EMBEDDING_IDENTITY_STALE_REASON)
                        summary_fts_fallback = not run_fts and run_summary
                    except EmbeddingPrivacyPolicyError:
                        # Deterministic configuration error — never degrade (#367).
                        raise
                    except Exception as exc:  # noqa: BLE001
                        coverage["summary"] = "none"
                        degraded_reasons.append(f"summary arm failed: {exc}")
                        summary_fts_fallback = not run_fts and run_summary
            # Scan failures use the same fallback, without repeating an earlier FTS arm.
            if summary_fts_fallback and "fts" not in coverage and time.monotonic() < deadline:
                _run_fts_arm(fallback=True)
            if chunk_query_vector is not None:
                if run_chunk:
                    try:
                        hits, cov, scanned, total = _lcm_recall_chunk_arm(
                            engine,
                            query_vector=chunk_query_vector,
                            provider=chunk_provider,
                            candidate_limit=candidate_limit,
                            deadline=deadline,
                            candidate_session_ids=candidate_session_ids,
                        )
                        arm_hits["chunk"] = hits
                        coverage["chunk"] = cov
                        if cov == "none":
                            degraded_reasons.append("chunk vectors are unavailable")
                        elif cov == "bounded":
                            degraded_reasons.append(
                                _lcm_recall_bounded_reason("chunk", scanned, total)
                            )
                        elif cov == "full_approx":
                            degraded_reasons.append(
                                _lcm_recall_approx_reason("chunk")
                            )
                    except TimeoutError:
                        timed_out = True
                        coverage["chunk"] = "none"
                    except EmbeddingIdentityStaleError:
                        coverage["chunk"] = "none"
                        degraded_reasons.append(_EMBEDDING_IDENTITY_STALE_REASON)
                    except EmbeddingPrivacyPolicyError:
                        # Deterministic configuration error — never degrade (#367).
                        raise
                    except Exception as exc:  # noqa: BLE001
                        coverage["chunk"] = "none"
                        degraded_reasons.append(f"chunk arm failed: {exc}")

    ordered, arm_order, arm_weights = fuse_candidates(
        arm_hits, getattr(engine._config, "recall_arm_weights", None)
    )
    fts_arm_index = arm_order.index("fts") if "fts" in arm_order else None
    now = time.time()
    scope_session_ids = set(_recent._recent_conversation_scope_session_ids(engine))
    ordered = apply_prior(ordered, scope_session_ids, now, scope_bias)

    # -- Optional rerank stage (default OFF): a pure rank-REORDER within the top
    #    window of the post-prior order (no score splicing onto the RRF scale). --
    ordered, rerank_status, rerank_scores = _lcm_recall_rerank(
        provider, query, ordered, window=rerank_window, deadline=deadline, config=engine._config
    )

    # -- FTS anchor slot (#950), after rerank so rerank cannot undo it. Inert
    #    with a single arm, so FTS-only (embeddings-off) output is unchanged.
    #    Proactive recall passes ``fts_anchor=False`` (an internal keyword, not
    #    a tool argument): it filters hits by a score floor after the call, so
    #    an FTS-only anchor below that floor would displace an eligible hit. --
    fts_anchor_info: dict[str, Any] | None = None
    if bool(kwargs.get("fts_anchor", True)) and bool(
        getattr(engine._config, "recall_fts_anchor", True)
    ):
        ordered, fts_anchor_info = place_fts_anchor(
            ordered, arm_hits, arm_order, limit, arm_weights=arm_weights
        )

    return recall_delivery.deliver_recall(
        ordered,
        query=query,
        limit=limit,
        requested_limit=requested_limit,
        scope_bias=scope_bias,
        include=include,
        detail=detail,
        reference_strict=reference_strict,
        delta_requested=delta_requested,
        seen_refs=seen_refs,
        include_occurrence_time=include_occurrence_time,
        deadline=deadline,
        fts_arm_index=fts_arm_index,
        arm_order=arm_order,
        arm_weights=arm_weights,
        arm_hits=arm_hits,
        coverage=coverage,
        served_fts_index=served_fts_index,
        rerank_status=rerank_status,
        rerank_scores=rerank_scores,
        embedding_query_metrics=embedding_query_metrics,
        degraded_reasons=degraded_reasons,
        timed_out=timed_out,
        fts_anchor_info=fts_anchor_info,
        summary_leads=summary_leads,
        limit_cap=_LCM_RECALL_LIMIT_CAP,
        snippet_chars=_LCM_RECALL_SNIPPET_CHARS,
        response_char_cap=_LCM_RECALL_RESPONSE_CHAR_CAP,
        per_session_limit=_LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
        expanded_hit_limit=_LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT,
        content_chars=_LCM_RECALL_ANSWER_READY_CONTENT_CHARS,
        strict_selector_factory=partial(_LcmRecallStrictSelector, engine=engine),
        hydrate=partial(_lcm_recall_answer_ready_content, engine),
        read_event_time=partial(_lcm_recall_event_time_rows, engine),
        event_time_fields=_lcm_recall_event_time,
        read_occurrence_source=lambda store_id: (
            getattr(engine, "_session_occurrence_dates", {}) or {},
            engine._store.get(store_id),
        ),
        rerank_enabled=lambda: bool(getattr(engine._config, "rerank_enabled", False)),
    )


def lcm_describe(args: Dict[str, Any], **kwargs) -> str:
    """Inspect a summary node's subtree or get session DAG overview."""
    return _expand_tools.lcm_describe(args, **kwargs)


def lcm_expand(args: Dict[str, Any], **kwargs) -> str:
    """Expand a summary node, externalized payload, or raw message to its content.

    Mode selection (exactly one is required):
    - ``externalized_ref``: open a stored externalized payload by ref filename (current session only)
    - ``store_id``: fetch a single raw message by store_id; works across sessions
    - ``node_id``: expand a summary node to its source content (explicit ``session_id`` required cross-session)

    Omitting ``session_id`` preserves current-session node lookup. Carried-over
    current-session nodes may reference raw source rows from the previous session.
    """
    return _expand_tools.lcm_expand(args, **kwargs)


def lcm_expand_query(args: Dict[str, Any], **kwargs) -> str:
    """Answer a question by expanding matching summaries or explicit node ids."""
    return _expand_query_tools.lcm_expand_query(args, **kwargs)


def _active_embedding_profile_exists(engine: "LCMEngine") -> bool:
    """Report whether the store holds an active summary embedding profile.

    Read-only and defensive: a missing table, an older schema, or any SQLite
    read problem answers ``False`` rather than raising, so a doctor run never
    fails because embeddings were never initialized on this database.
    """
    conn = getattr(getattr(engine, "_dag", None), "connection", None)
    if conn is None:
        return False
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM lcm_embedding_profile
            WHERE active = 1 AND archived_at IS NULL AND task = 'summary'
            LIMIT 1
            """
        ).fetchone()
    except Exception:
        return False
    return row is not None


def _lcm_store_has_live_chunk_vectors(conn: sqlite3.Connection) -> bool:
    # Same joins as the recall preflight: a chunk counts only with its vector
    # and its source message.
    try:
        return conn.execute(
            """
            SELECT 1
            FROM lcm_chunk_meta m
            JOIN lcm_chunk_vectors v
              ON v.chunk_id = m.chunk_id AND v.identity_hash = m.identity_hash
            JOIN messages src ON src.store_id = m.store_id
            WHERE m.archived = 0
            LIMIT 1
            """
        ).fetchone() is not None
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):
            raise
        return False


def _embedding_provider_health_check(engine: "LCMEngine") -> dict[str, Any]:
    """Return the ``embedding_provider_health`` lcm_doctor check (#672).

    Answers one operator question: *is the semantic retrieval I configured
    actually able to run?* `lcm_recall` already discloses ``degraded`` when the
    provider is missing, but nothing surfaced that state in `lcm_doctor`, so a
    provider that disappeared (e.g. the optional ``fastembed`` dependency
    dropped by a Hermes virtualenv rebuild) left semantic search silently
    degraded to full-text behind an all-`pass` doctor report.

    The probe is inert: configuration, import specs, and environment variables
    only. No model download, no network call, no provider warmup.
    """
    config = engine._config
    enabled = bool(getattr(config, "embeddings_enabled", False))
    profile_active = _active_embedding_profile_exists(engine)

    if not enabled:
        if profile_active:
            return {
                "check": "embedding_provider_health",
                "status": "warn",
                "detail": {
                    "embeddings_enabled": False,
                    "active_embedding_profile": True,
                    "available": False,
                    "reason": (
                        "embeddings are disabled (LCM_EMBEDDINGS_ENABLED=false) but this "
                        "database still has an active summary embedding profile; stored "
                        "vectors are no longer being maintained and will go stale"
                    ),
                },
            }
        return {
            "check": "embedding_provider_health",
            "status": "pass",
            "detail": "semantic embeddings are disabled (LCM_EMBEDDINGS_ENABLED=false)",
        }

    probe = probe_provider_availability(config)
    detail = {
        "embeddings_enabled": True,
        "active_embedding_profile": profile_active,
        "provider": probe.get("provider"),
        "model": probe.get("model"),
        "available": bool(probe.get("available")),
        "reachability_probed": bool(probe.get("probed")),
        "reason": probe.get("detail"),
    }
    # An unavailable provider is the prerequisite blocker: warmup cannot succeed until it is fixed.
    if not probe.get("available"):
        detail["impact"] = (
            "semantic retrieval is degraded: lcm_recall reports degraded=true and "
            "lcm_grep mode=semantic falls back to full-text"
        )
        return {
            "check": "embedding_provider_health",
            "status": "warn",
            "detail": detail,
        }
    provider_id = str(probe.get("provider") or "")
    if embedding_provider_requires_privacy(provider_id):
        provider_id = {
            "voyageai": "voyage",
            "openai": "openai-compatible",
            "siliconflow": "openai-compatible",
        }.get(provider_id, provider_id)
        try:
            current_revision = embedding_privacy_revision(config)
        except EmbeddingPrivacyPolicyError as exc:
            detail["reason"] = str(exc)
            return {
                "check": "embedding_provider_health",
                "status": "warn",
                "detail": detail,
            }
        model = str(probe.get("model") or "")
        db_path = Path(engine._store.db_path).resolve()
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        try:
            stale_tasks = []
            if _lcm_embedding_profile_revision(
                conn, provider=provider_id, model=model, task="summary",
            ) != current_revision:
                stale_tasks.append("summary")
            # The chunk corpus is opt-in, so its identity matters only once the
            # store holds live chunk vectors (#888).
            if _lcm_store_has_live_chunk_vectors(conn) and _lcm_embedding_profile_revision(
                conn, provider=provider_id, model=default_chunk_model(provider_id, model),
                task="chunk",
            ) != current_revision:
                stale_tasks.append("chunk")
        finally:
            conn.close()
        if stale_tasks:
            corpus_option = ""
            consent_option = ""
            consent_note = ""
            if "chunk" in stale_tasks:
                from .command import _is_local_embedding_provider

                corpus_option = " --corpus both" if "summary" in stale_tasks else " --corpus chunks"
                if not _is_local_embedding_provider(provider_id):
                    consent_option = " --confirm-raw-text"
                    consent_note = (
                        "; --confirm-raw-text acknowledges sending raw message text to the "
                        f"'{provider_id}' cloud embedding provider"
                    )
            remedy = f"/lcm embed warmup, then /lcm embed backfill{corpus_option} --apply{consent_option}"
            detail.update({
                "embedding_identity_stale": True,
                "stale_tasks": stale_tasks,
                "reason": (
                    "embedding_identity_stale: stored vectors use an older embedding privacy "
                    f"revision; run {remedy}{consent_note}"
                    "; repeat the backfill apply only while remaining is above 0 and each run lowers it, "
                    "but stop and inspect its warnings and details on status: partial, status: error, "
                    "status: failed, or if remaining stops falling"
                ),
                "impact": "lcm_recall skips the stale semantic arm until the profile is re-registered",
                "remedy": remedy,
            })
            return {
                "check": "embedding_provider_health",
                "status": "warn",
                "detail": detail,
            }
    return {
        "check": "embedding_provider_health",
        "status": "pass",
        "detail": detail,
    }


def _summary_quality_stats(engine: "LCMEngine", session_id: str) -> dict[str, Any]:
    """Return read-only summary compression quality diagnostics for one session."""
    conn = engine._dag.connection
    if conn is None:
        raise RuntimeError("LCM DAG connection is not initialized")
    rows = conn.execute(
        """
        SELECT node_id, session_id, depth, token_count, source_token_count
        FROM summary_nodes
        WHERE session_id = ? AND source_token_count > 0
        ORDER BY
            CASE WHEN token_count <= 0 THEN 1 ELSE 0 END DESC,
            CASE WHEN token_count > 0
                 THEN CAST(source_token_count AS REAL) / token_count
                 ELSE source_token_count
            END DESC
        LIMIT 5
        """,
        (session_id,),
    ).fetchall()
    totals = conn.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(source_token_count), 0),
            COALESCE(SUM(token_count), 0),
            SUM(CASE WHEN source_token_count >= 100000
                      AND token_count < 500 THEN 1 ELSE 0 END),
            SUM(CASE WHEN token_count > 0
                      AND CAST(source_token_count AS REAL) / token_count >= 400
                     THEN 1 ELSE 0 END)
        FROM summary_nodes
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    total_nodes = int(totals[0] or 0)
    total_source_tokens = int(totals[1] or 0)
    total_summary_tokens = int(totals[2] or 0)
    tiny_large_source_nodes = int(totals[3] or 0)
    extreme_ratio_nodes = int(totals[4] or 0)
    overall_ratio = (
        round(total_source_tokens / total_summary_tokens, 1)
        if total_summary_tokens > 0
        else 0.0
    )
    worst_nodes = []
    for node_id, session_id, depth, token_count, source_token_count in rows:
        ratio = (
            round(float(source_token_count) / float(token_count), 1)
            if token_count and token_count > 0
            else None
        )
        worst_nodes.append({
            "node_id": int(node_id),
            "session_id": session_id,
            "depth": int(depth),
            "source_token_count": int(source_token_count or 0),
            "token_count": int(token_count or 0),
            "compression_ratio": ratio,
        })
    return {
        "total_nodes": total_nodes,
        "session_id": session_id,
        "total_source_tokens": total_source_tokens,
        "total_summary_tokens": total_summary_tokens,
        "overall_compression_ratio": overall_ratio,
        "extreme_ratio_threshold": 400,
        "tiny_large_source_threshold": {
            "source_token_count_min": 100000,
            "token_count_max": 500,
        },
        "extreme_ratio_nodes": extreme_ratio_nodes,
        "tiny_large_source_nodes": tiny_large_source_nodes,
        "worst_nodes": worst_nodes,
        "recommendation": (
            "Inspect worst_nodes with lcm_expand; tiny summaries for very large sources often indicate degraded fallback summarization."
            if extreme_ratio_nodes or tiny_large_source_nodes
            else "summary compression ratios are within the diagnostic thresholds"
        ),
    }


def _inspect_top_level_json_string_fields_before_content(text: str) -> tuple[dict[str, str], bool]:
    return _externalized_top_level_fields_before_content(text)


def _read_externalized_payload_metadata_prefix(
    path: Path,
    *,
    max_read_bytes: int = _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES,
) -> tuple[str, bool, bool]:
    """Read bounded JSON metadata before the externalized payload body.

    Returns ``(prefix_text, content_string_seen, prefix_truncated)``. The content
    string body is intentionally not consumed; ``lcm_inspect`` reports bounded
    metadata only and leaves full JSON/body validation to explicit expansion.
    """
    return read_externalized_payload_metadata_prefix(
        path,
        max_read_bytes=max_read_bytes,
    )


def _validate_externalized_payload_json_tail(
    path: Path,
    metadata_prefix_text: str,
    *,
    max_tail_bytes: int = _LCM_GREP_EXTERNALIZED_DOCUMENT_TAIL_BYTES,
) -> dict[str, Any] | None:
    """Validate the closing JSON structure using only a bounded tail window."""
    metadata_prefix = metadata_prefix_text.encode("utf-8")
    try:
        file_size = path.stat().st_size
        tail_offset = max(len(metadata_prefix), file_size - max_tail_bytes)
        with path.open("rb") as handle:
            handle.seek(tail_offset)
            tail = handle.read(max_tail_bytes + 1)
    except OSError:
        return None
    if len(tail) > max_tail_bytes:
        return None

    for index, byte in enumerate(tail):
        if byte != ord('"'):
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and tail[cursor] == ord("\\"):
            backslashes += 1
            cursor -= 1
        if backslashes % 2:
            continue
        try:
            payload = json.loads((metadata_prefix + b'"' + tail[index + 1 :]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _inspect_externalized_payload_metadata(
    engine: "LCMEngine",
    ref: str,
    session_id: str,
    *,
    max_read_bytes: int = _LCM_INSPECT_PAYLOAD_METADATA_READ_BYTES,
    require_valid_document_tail: bool = False,
) -> dict[str, Any]:
    if not ref or Path(ref).name != ref:
        return {"readable": False, "error": "invalid_ref"}
    try:
        storage_dir = get_large_output_storage_dir(
            engine._config,
            hermes_home=engine._hermes_home,
            create=False,
        )
        path = storage_dir / ref
        if not path.exists():
            return {"readable": False, "error": "missing"}
        if not path.is_file():
            return {"readable": False, "error": "not_a_file"}
        metadata_prefix_text, content_key_seen, prefix_truncated = _read_externalized_payload_metadata_prefix(
            path,
            max_read_bytes=max_read_bytes,
        )
    except FileNotFoundError:
        return {"readable": False, "error": "missing"}
    except (OSError, ValueError) as exc:
        return {"readable": False, "error": str(exc)}

    metadata_fields, _content_key_seen = _inspect_top_level_json_string_fields_before_content(metadata_prefix_text)
    payload_session_id = metadata_fields.get("session_id", "")
    if payload_session_id and payload_session_id != session_id:
        return {"readable": False, "error": "session_mismatch"}
    if not payload_session_id:
        return {"readable": False, "error": "session_metadata_unavailable"}
    if not content_key_seen:
        error = "metadata_prefix_truncated" if prefix_truncated else "invalid_payload"
        return {"readable": False, "error": error}

    if require_valid_document_tail:
        full_payload = _validate_externalized_payload_json_tail(path, metadata_prefix_text)
        if full_payload is None:
            return {"readable": False, "error": "invalid_payload"}
        if (full_payload.get("session_id") or "") != session_id:
            return {"readable": False, "error": "session_mismatch"}

    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"readable": False, "error": "missing"}
    except OSError as exc:
        return {"readable": False, "error": str(exc)}

    metadata: dict[str, Any] = {
        "readable": True,
        "file_size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
        "payload_validation": "document_tail" if require_valid_document_tail else "metadata_prefix",
    }
    if payload_session_id:
        metadata["payload_session_id"] = payload_session_id
    return metadata


# Dispatch resolves this name at call time, so patching it changes behavior (rule 6).
lcm_inspect = _inspect_tools.lcm_inspect


def lcm_status(args: Dict[str, Any], **kwargs) -> str:
    """Quick health overview of the LCM engine for the current session."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    # Read the foreground view so a side-channel session that briefly owns
    # engine._session_id (cron tick inside the gateway process, debug probe,
    # etc.) does not divert lcm_status away from the operator's real
    # conversation. Falls back to the bound id when no foreground has ever
    # been bound, so cron-only or stateless-only deployments still report
    # something usable.
    session_id = engine.current_session_id
    # #622: another process of this profile where Hermes did not take LCM-X.
    inactive_process = inactive_record_notice(getattr(engine, "_hermes_home", ""))
    inactive_payload = {"inactive_process": inactive_process} if inactive_process else {}
    if not session_id:
        return json.dumps({
            "error": "No active session",
            "runtime_identity": engine.get_runtime_identity(),
            "identity_migration": getattr(engine, "identity_migration", None),
            **inactive_payload,
        })

    # Store stats
    store_messages = engine._store.get_session_count(session_id)
    store_tokens = engine._store.get_session_token_total(session_id)

    # DAG stats by depth
    depths = engine._dag.get_session_depth_stats(session_id)

    total_dag_tokens = sum(d["tokens"] for d in depths.values())
    total_source_tokens = sum(d["source_tokens"] for d in depths.values())
    total_dag_nodes = sum(d["count"] for d in depths.values())
    compression_ratio = round(total_source_tokens / total_dag_tokens, 1) if total_dag_tokens > 0 else 0
    full_status = engine.get_status()
    lifecycle = full_status.get("lifecycle")
    lifecycle_fragmentation = full_status.get("lifecycle_fragmentation")
    source_lineage = full_status.get("source_lineage")
    runtime_identity = full_status.get("runtime_identity")
    ingest_reconciliation = full_status.get("ingest_reconciliation")
    config_sources = full_status.get("config_sources") or {}
    config_source_warnings = full_status.get("config_source_warnings") or []
    ignored_config_yaml_lcm_keys = full_status.get("ignored_config_yaml_lcm_keys") or []

    # Filter classification for the session lcm_status is reporting on.
    # The engine encapsulates the foreground vs bound divergence; this tool
    # just reads the property contract.
    side_channel_active = engine.side_channel_active

    return json.dumps({
        "session_id": session_id,
        "compression_count": engine.compression_count,
        "total_compactions": full_status.get("total_compactions", 0),
        "total_compactions_scope": full_status.get(
            "total_compactions_scope", "current_conversation"
        ),
        "last_compression_status": full_status.get("last_compression_status", "idle"),
        "last_compression_noop_reason": full_status.get("last_compression_noop_reason", ""),
        "last_survival_fit": full_status.get("last_survival_fit"),
        "threshold_full_sweep": full_status.get("threshold_full_sweep"),
        "last_stub_first_exit": full_status.get("last_stub_first_exit"),
        "no_progress_hold": full_status.get("no_progress_hold"),
        **({"background_leaf_prep": full_status["background_leaf_prep"]}  # #787; absent while off
           if "background_leaf_prep" in full_status else {}),
        "zero_leaf_streak": full_status.get("zero_leaf_streak"),  # #1162
        "model": full_status.get("model", ""),
        "provider": full_status.get("provider", ""),
        "raw_context_length": full_status.get("raw_context_length", engine.context_length),
        "context_length": engine.context_length,
        "effective_context_length_cap": full_status.get("effective_context_length_cap"),
        "effective_context_length_reason": full_status.get("effective_context_length_reason", ""),
        "context_length_source": full_status.get("context_length_source", ""),
        "configured_context_threshold": full_status.get("configured_context_threshold", engine._config.context_threshold),
        "context_threshold": full_status.get("context_threshold", engine._config.context_threshold),
        "context_threshold_source": full_status.get("context_threshold_source", ""),
        "context_threshold_autoraised": full_status.get("context_threshold_autoraised"),
        "threshold_tokens": engine.threshold_tokens,
        "last_prompt_tokens": engine.last_prompt_tokens,
        "last_input_tokens": engine.last_input_tokens,
        "last_output_tokens": engine.last_output_tokens,
        "last_cache_read_tokens": engine.last_cache_read_tokens,
        "last_cache_write_tokens": engine.last_cache_write_tokens,
        "last_reasoning_tokens": engine.last_reasoning_tokens,
        "cache_metrics_available": engine.cache_metrics_available,
        "cache_read_ratio": round(engine.cache_read_ratio, 4),
        "store": {
            "messages": store_messages,
            "estimated_tokens": store_tokens,
        },
        "dag": {
            "total_nodes": total_dag_nodes,
            "nodes_by_escalation_level": engine._dag.count_nodes_by_escalation_level(session_id),
            "total_tokens": total_dag_tokens,
            "compression_ratio": f"{compression_ratio}:1",
            "depths": {
                f"d{depth}": info for depth, info in sorted(depths.items())
            },
        },
        "config": {
            "fresh_tail_count": engine._config.fresh_tail_count,
            "fresh_tail_max_tokens": engine._config.fresh_tail_max_tokens,
            "leaf_chunk_tokens": engine._config.leaf_chunk_tokens,
            "leaf_target_ratio": getattr(
                engine._config, "leaf_target_ratio", LCMConfig.leaf_target_ratio
            ),
            "leaf_target_min_tokens": getattr(
                engine._config, "leaf_target_min_tokens", LCMConfig.leaf_target_min_tokens
            ),
            "leaf_target_max_tokens": getattr(
                engine._config, "leaf_target_max_tokens", LCMConfig.leaf_target_max_tokens
            ),
            "stub_first_exit_target_ratio": getattr(
                engine._config, "stub_first_exit_target_ratio", LCMConfig.stub_first_exit_target_ratio
            ),
            "dynamic_leaf_chunk_enabled": engine._config.dynamic_leaf_chunk_enabled,
            "dynamic_leaf_chunk_max": engine._config.dynamic_leaf_chunk_max,
            "cache_friendly_condensation_enabled": engine._config.cache_friendly_condensation_enabled,
            "cache_friendly_min_debt_groups": engine._config.cache_friendly_min_debt_groups,
            "deferred_maintenance_enabled": engine._config.deferred_maintenance_enabled,
            "deferred_maintenance_max_passes": engine._config.deferred_maintenance_max_passes,
            "critical_budget_pressure_ratio": engine._config.critical_budget_pressure_ratio,
            "threshold_full_sweep_enabled": engine._config.threshold_full_sweep_enabled,
            "summary_prefix_target_tokens": engine._config.summary_prefix_target_tokens,
            "threshold_full_sweep_max_passes": 12,
            "threshold_full_sweep_max_seconds": engine._foreground_budget_seconds()[1],
            "foreground_soft_seconds": engine._foreground_budget_seconds()[0],
            "context_threshold": engine._config.context_threshold,
            "max_depth": engine._config.incremental_max_depth,
            "condensation_fanin": engine._config.condensation_fanin,
            "summary_model": engine._config.summary_model or "(auxiliary)",
            "summary_reasoning_effort": engine._config.summary_reasoning_effort or "(task default)",
            "summary_timeout_ms": engine._config.summary_timeout_ms,
            "summary_prompt_version": getattr(engine._config, "summary_prompt_version", 1),
            "summary_spend_max_calls": engine._config.summary_spend_max_calls,
            "summary_spend_window_seconds": engine._config.summary_spend_window_seconds,
            "summary_spend_backoff_seconds": engine._config.summary_spend_backoff_seconds,
            "expansion_model": engine._config.expansion_model or "(summary model)",
            "expansion_reasoning_effort": engine._config.expansion_reasoning_effort or "(task default)",
        },
        "proactive_recall": {
            "enabled": bool(getattr(engine._config, "proactive_recall_enabled", False)),
            "min_score": getattr(engine._config, "proactive_recall_min_score", 0.0),
            "budget_tokens": getattr(engine._config, "proactive_recall_budget_tokens", 0),
            "provider_override": getattr(engine._config, "proactive_recall_provider", "") or "",
            "injected": int(getattr(engine, "_proactive_recall_injected_count", 0) or 0),
            "skipped": int(getattr(engine, "_proactive_recall_skipped_count", 0) or 0),
            "timeout": int(getattr(engine, "_proactive_recall_timeout_count", 0) or 0),
            "privacy_policy_errors": int(
                getattr(engine, "_proactive_recall_privacy_error_count", 0) or 0
            ),
        },
        "config_sources": config_sources,
        "config_source_warnings": config_source_warnings,
        "ignored_config_yaml_lcm_keys": ignored_config_yaml_lcm_keys,
        "session_filters": {
            "ignored": engine.current_session_ignored,
            "stateless": engine.current_session_stateless,
            "ignore_session_patterns": full_status.get("ignore_session_patterns", []),
            "ignore_session_patterns_source": full_status.get("ignore_session_patterns_source", "default"),
            "stateless_session_patterns": full_status.get("stateless_session_patterns", []),
            "stateless_session_patterns_source": full_status.get("stateless_session_patterns_source", "default"),
            "ignore_message_patterns": full_status.get("ignore_message_patterns", []),
            "ignore_message_patterns_source": full_status.get("ignore_message_patterns_source", "default"),
            "ignored_message_count": full_status.get("ignored_message_count", 0),
            "side_channel_active": side_channel_active,
            **(
                {"side_channel_session_id": engine._session_id}
                if side_channel_active
                else {}
            ),
        },
        "source_lineage": source_lineage,
        "ingest_protection": full_status.get("ingest_protection", sensitive_pattern_status(engine._config)),
        "preset_suggestion": preset_status_payload(engine),
        "ingest_reconciliation": ingest_reconciliation,
        "runtime_identity": runtime_identity,
        "identity_migration": full_status.get("identity_migration"),
        "lifecycle": lifecycle,
        "lifecycle_fragmentation": lifecycle_fragmentation,
        "summary_route": engine._summary_route_status(),
        **inactive_payload,
    })


def lcm_doctor(args: Dict[str, Any], **kwargs) -> str:
    """Run diagnostics on the LCM database and configuration."""
    engine = _require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    action = str(args.get("action") or "").strip().lower()
    if action:
        if action != "repair_level3":
            return json.dumps({"error": f"unknown lcm_doctor action: {action}"})
        if args.get("apply"):
            return json.dumps({"error": "apply is operator-only: use `/lcm doctor repair level3 apply`"})
        cursor = args.get("cursor", 0)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            return json.dumps({"error": "cursor must be a non-negative integer"})
        scan = scan_level3_fragments(engine, session_id=engine.current_session_id or "")
        total_flagged, total_ancestors = len(scan["flagged"]), len(scan["ancestors"])
        return json.dumps({
            "action": action, "status": "repair-needed" if scan["flagged"] else "ok", "read_only": True, **scan,
            "flagged": scan["flagged"][cursor:cursor + 50], "ancestors": scan["ancestors"][cursor:cursor + 50],
            "total_flagged": total_flagged, "total_ancestors": total_ancestors,
            **({"next_cursor": cursor + 50} if cursor + 50 < max(total_flagged, total_ancestors) else {}),
        })

    checks: list[dict] = []
    inactive_process = inactive_record_notice(getattr(engine, "_hermes_home", ""))
    checks.append({
        "check": "inactive_process",
        "status": "warn" if inactive_process else "pass",
        "detail": inactive_process or "no live inactive process",
    })
    # Diagnose the foreground session, not whatever side-channel session
    # currently owns engine._session_id. Falls back to the bound id when no
    # foreground has ever been bound.
    session_id = engine.current_session_id

    # 0. Plugin identity: legacy hermes-lcm / context.engine: lcm config (#471).
    identity_migration = getattr(engine, "identity_migration", None)
    checks.append({
        "check": "identity_migration",
        "status": "warn" if identity_migration else "pass",
        "detail": identity_migration or "config uses hermes-lcm-x / lcm-x",
    })

    # 1. Database integrity
    try:
        result = engine._store.connection.execute("PRAGMA integrity_check").fetchone()
        ok = result and result[0] == "ok"
        checks.append({
            "check": "database_integrity",
            "status": "pass" if ok else "fail",
            "detail": result[0] if result else "no response",
        })
    except Exception as e:
        checks.append({
            "check": "database_integrity",
            "status": "fail",
            "detail": str(e),
        })

    # Ingest health: a swallowed persistence error means turns were not
    # durably stored, silently breaking the lossless guarantee. Surface it.
    ingest_failures = int(getattr(engine, "_ingest_failure_count", 0) or 0)
    consecutive_failures = int(getattr(engine, "_consecutive_ingest_failures", 0) or 0)
    if consecutive_failures > 0:
        ingest_status = "fail"
    elif ingest_failures > 0:
        ingest_status = "warn"
    else:
        ingest_status = "pass"
    checks.append({
        "check": "ingest_health",
        "status": ingest_status,
        "detail": {
            "total_failures": ingest_failures,
            "consecutive_failures": consecutive_failures,
            "last_error": getattr(engine, "_last_ingest_error", "") or "",
            "last_error_time": getattr(engine, "_last_ingest_error_time", 0) or 0,
        } if ingest_failures else "no ingest failures recorded",
    })

    # ignore_message_patterns drops discard raw content that is never persisted.
    # A non-zero count is worth surfacing so an over-broad pattern is noticed.
    dropped = int(getattr(engine, "_ignore_pattern_dropped_count", 0) or 0)
    checks.append({
        "check": "ignore_pattern_drops",
        "status": "warn" if dropped else "pass",
        "detail": (
            f"{dropped} message(s) dropped by ignore_message_patterns and not "
            "persisted; verify the pattern is not matching substantive turns"
            if dropped
            else "no messages dropped by ignore_message_patterns"
        ),
    })

    try:
        conn = engine._store.connection
        if conn is None:
            raise RuntimeError("LCM store connection is not initialized")
        schema_health = inspect_lcm_schema_health(
            conn,
            database_path=str(engine._store.db_path),
        )
        missing_tables = schema_health.get("missing_tables")
        has_missing = isinstance(missing_tables, list) and bool(missing_tables)
        checks.append({
            "check": "schema_core_tables",
            "status": "fail" if has_missing or schema_health.get("error") else "pass",
            "detail": schema_health,
        })
    except Exception as e:
        checks.append({
            "check": "schema_core_tables",
            "status": "fail",
            "detail": str(e),
        })

    # 1b. FTS5 integrity, separated from generic SQLite integrity so malformed
    # inverted indexes point at the exact table and repair path.
    for check_name, conn, spec in (
        ("messages_fts_integrity", engine._store.connection, build_message_fts_spec()),
        ("nodes_fts_integrity", engine._dag.connection, build_nodes_fts_spec()),
    ):
        try:
            fts_integrity = check_external_content_fts_integrity(conn, spec)
            status = fts_integrity["status"]
            checks.append({
                "check": check_name,
                "status": "warn" if status == "unchecked" else status,
                "detail": fts_integrity if status == "unchecked" else fts_integrity["detail"],
            })
        except Exception as e:
            checks.append({
                "check": check_name,
                "status": "fail",
                "detail": str(e),
            })
        # A prior non-blocking background integrity scan records a persisted
        # ``fts_integrity_failed:<table>`` flag when it finds corruption
        # without rebuilding. Surface it even when this run's live check is
        # throttled/unchecked, mirroring the /lcm doctor text path.
        try:
            failed_flag = load_integrity_failed(conn, spec)
        except Exception:  # pragma: no cover - defensive
            failed_flag = None
        if failed_flag:
            checks.append({
                "check": f"{check_name}_background_flag",
                "status": "fail",
                "detail": {
                    "flagged_at": failed_flag.get("at"),
                    "detail": failed_flag.get("detail"),
                    "guidance": "background integrity scan flagged this index; run `/lcm doctor repair apply`",
                },
            })

    # 2. SQLite storage posture and payload diagnostics
    try:
        journal_mode_row = engine._store.connection.execute("PRAGMA journal_mode").fetchone()
        quick_check_row = engine._store.connection.execute("PRAGMA quick_check").fetchone()
        db_path = Path(engine._store.db_path)
        wal_path = Path(str(db_path) + "-wal")
        checks.append({
            "check": "sqlite_storage",
            "status": "pass" if quick_check_row and quick_check_row[0] == "ok" else "fail",
            "detail": {
                "database_path": str(db_path),
                "database_exists": db_path.exists(),
                "journal_mode": journal_mode_row[0] if journal_mode_row else "unknown",
                "quick_check": quick_check_row[0] if quick_check_row else "unknown",
                "database_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
                "wal_size_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
            },
        })
        payload_risks = scan_sqlite_payload_risks(engine._store.connection)
        externalized_stats = externalized_payload_stats(engine._config, hermes_home=engine._hermes_home)
        externalized_integrity = scan_externalized_payload_integrity(
            engine._store.connection,
            engine._config,
            hermes_home=engine._hermes_home,
        )
        suspicious_count = (
            len(payload_risks["suspicious_data_uri_content_rows"])
            + len(payload_risks["suspicious_data_uri_tool_calls_rows"])
            + len(payload_risks["suspicious_base64_like_rows"])
            + len(payload_risks["suspicious_repetitive_assistant_rows"])
            + len(payload_risks["heartbeat_noise_rows"])
        )
        missing_externalized_refs = int(externalized_integrity.get("externalized_payload_refs_missing", 0) or 0)
        checks.append({
            "check": "payload_storage",
            "status": "warn" if suspicious_count or missing_externalized_refs else "pass",
            "detail": {
                **payload_risks,
                **externalized_stats,
                **externalized_integrity,
            },
        })
    except Exception as e:
        checks.append({
            "check": "payload_storage",
            "status": "fail",
            "detail": str(e),
        })

    try:
        protection = sensitive_pattern_status(engine._config)
        protection_status = "pass"
        if protection["enabled"] and not protection["active_patterns"]:
            protection_status = "warn"
        elif protection["unknown_patterns"]:
            protection_status = "warn"
        checks.append({
            "check": "sensitive_pattern_handling",
            "status": protection_status,
            "detail": protection,
        })
    except Exception as e:
        checks.append({
            "check": "sensitive_pattern_handling",
            "status": "fail",
            "detail": str(e),
        })

    # 3. FTS index sync
    try:
        msg_count = engine._store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        fts_count = engine._store.connection.execute(
            """
            SELECT COUNT(*)
            FROM messages_fts
            JOIN messages ON messages_fts.rowid = messages.store_id
            WHERE messages.session_id = ?
            """,
            (session_id,),
        ).fetchone()[0]
        checks.append({
            "check": "fts_index_sync",
            "status": "pass" if fts_count >= msg_count else "warn",
            "detail": f"{fts_count} session FTS rows, {msg_count} session messages",
        })
    except Exception as e:
        checks.append({
            "check": "fts_index_sync",
            "status": "fail",
            "detail": str(e),
        })

    # 3. Orphaned DAG nodes (nodes referencing store_ids that don't exist)
    try:
        all_nodes = engine._dag.get_session_nodes(session_id)
        orphaned = 0
        for node in all_nodes:
            if node.source_type == "messages":
                for sid in node.source_ids:
                    stored = engine._store.get(sid)
                    if stored is None:
                        orphaned += 1
                        break
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "pass" if orphaned == 0 else "warn",
            "detail": f"{orphaned} nodes reference missing store messages" if orphaned else "all nodes have valid sources",
        })
    except Exception as e:
        checks.append({
            "check": "orphaned_dag_nodes",
            "status": "fail",
            "detail": str(e),
        })

    try:
        summary_quality = _summary_quality_stats(engine, session_id)
        degraded_count = (
            summary_quality.get("extreme_ratio_nodes", 0)
            + summary_quality.get("tiny_large_source_nodes", 0)
        )
        checks.append({
            "check": "summary_quality",
            "status": "warn" if degraded_count else "pass",
            "detail": summary_quality,
        })
    except Exception as e:
        checks.append({
            "check": "summary_quality",
            "status": "fail",
            "detail": str(e),
        })

    # 4. Configuration validation
    config_warnings = []
    c = engine._config
    if c.fresh_tail_count < 2:
        config_warnings.append("fresh_tail_count < 2 may cause aggressive compaction")
    runtime_context_threshold = float(getattr(engine, "context_threshold", c.context_threshold))
    if runtime_context_threshold > 0.95:
        config_warnings.append("runtime context_threshold > 0.95 leaves very little headroom")
    if runtime_context_threshold < 0.3:
        config_warnings.append("runtime context_threshold < 0.3 triggers compaction very early")
    if c.condensation_fanin < 2:
        config_warnings.append("condensation_fanin < 2 creates excessive depth growth")
    if c.incremental_max_depth == 0:
        config_warnings.append("incremental_max_depth=0 disables condensation entirely")
    for warning in getattr(c, "config_source_warnings", []) or []:
        config_warnings.append(warning)
    for key in getattr(c, "ignored_config_yaml_lcm_keys", []) or []:
        config_warnings.append(
            f"config.yaml lcm.{key} is not a supported LCM config.yaml key and was ignored; use the matching LCM_* env var if this setting is intentional"
        )

    checks.append({
        "check": "config_validation",
        "status": "pass" if not config_warnings else "warn",
        "detail": config_warnings if config_warnings else "all settings within normal ranges",
    })

    # 4b. Embedding provider health (#672)
    #
    # Without this check lcm_doctor stays silent while a configured embedding
    # provider is dead: lcm_recall reports degraded=true and lcm_grep
    # mode=semantic falls back to full-text, but every doctor check passes, so
    # the operator has no signal that the semantic retrieval they enabled has
    # stopped running. The probe is inert and offline-safe by contract -- no
    # model download, no socket, no provider API call.
    try:
        checks.append(_embedding_provider_health_check(engine))
    except Exception as e:
        checks.append({
            "check": "embedding_provider_health",
            "status": "fail",
            "detail": str(e),
        })

    # 5. Source-lineage hygiene
    try:
        source_stats = engine._store.get_source_stats()
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "pass",
            "detail": {
                **source_stats,
                "normalization_mode": "backcompat-normalization",
            },
        })
    except Exception as e:
        checks.append({
            "check": "source_lineage_hygiene",
            "status": "fail",
            "detail": str(e),
        })

    # 6. Lifecycle/session fragmentation
    try:
        lifecycle_fragmentation = engine._lifecycle.get_fragmentation_stats(
            state_db_path=_state_db_path_for_engine(engine)
        )
        checks.append({
            "check": "lifecycle_fragmentation",
            "status": "warn" if _has_lifecycle_fragmentation(lifecycle_fragmentation) else "pass",
            "detail": lifecycle_fragmentation,
        })
    except Exception as e:
        checks.append({
            "check": "lifecycle_fragmentation",
            "status": "fail",
            "detail": str(e),
        })

    # 7. Context pressure
    if engine.context_length > 0:
        used_tokens = max(int(engine.last_prompt_tokens or 0), int(getattr(engine, "_last_gate_tokens", 0) or 0))
        usage_pct = round(used_tokens / engine.context_length * 100, 1) if engine.context_length else 0
        runtime_threshold = float(getattr(engine, "context_threshold", c.context_threshold))
        threshold_pct = round(runtime_threshold * 100, 1)
        checks.append({
            "check": "context_pressure",
            "status": "pass" if usage_pct < threshold_pct else "warn",
            "detail": f"{usage_pct}% used, compaction triggers at {threshold_pct}%",
        })

    # #1162: expose the same survival-fit counter (including last_cause) as /lcm doctor.
    try:
        survival_fit = engine._store.read_metadata_json("survival_fit:counter")
    except Exception:
        survival_fit = None

    # 8. Compaction progress (#1162): consecutive over-threshold passes that stored no leaf
    streak = engine._zero_leaf_streak_status() if hasattr(engine, "_zero_leaf_streak_status") else None
    checks.append({
        "check": "compaction_progress",
        "status": "warn" if streak and streak.get("passes", 0) >= 3 else "pass",
        "detail": streak or {"passes": 0},
    })

    overall = "healthy"
    if any(ch["status"] == "fail" for ch in checks):
        overall = "unhealthy"
    elif any(ch["status"] == "warn" for ch in checks):
        overall = "warnings"

    return json.dumps({
        "overall": overall,
        "survival_fit": survival_fit if survival_fit_count(survival_fit) != 0 else None,
        "runtime_identity": engine.get_runtime_identity(),
        "checks": checks,
        "guidance": doctor_guidance_for_checks(checks),
    })
