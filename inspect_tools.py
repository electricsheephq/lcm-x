"""Read-only inspect metadata behind the public tools facade."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, TYPE_CHECKING

# Resolve shared helpers/constants at call time: tools also imports this family.
from . import tools as _tools
from .externalize import extract_externalized_refs
from .session_patterns import build_session_match_keys, compile_session_pattern

if TYPE_CHECKING:
    from .engine import LCMEngine


# Preserve the tools logger under the package name used by the host.
logger = logging.getLogger(f"{__package__}.tools" if __package__ else "tools")


_TEMPORAL_ROLLUP_PERIOD_KINDS = ("day", "week", "month")
_TEMPORAL_ROLLUP_STATUSES = ("ready", "stale", "building", "failed")



def _bounded_operator_field(value: object) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= _tools._OPERATOR_TEXT_FIELD_MAX_CHARS:
        return text, False
    suffix = "..."
    return text[: _tools._OPERATOR_TEXT_FIELD_MAX_CHARS - len(suffix)] + suffix, True


def _bound_operator_strings(value: Any) -> tuple[Any, int]:
    """Return a JSON-compatible copy with every free-text field bounded."""
    if isinstance(value, str):
        bounded, truncated = _bounded_operator_field(value)
        return bounded, int(truncated)
    if isinstance(value, list):
        result: list[Any] = []
        truncated_fields = 0
        for item in value:
            bounded, count = _bound_operator_strings(item)
            result.append(bounded)
            truncated_fields += count
        return result, truncated_fields
    if isinstance(value, dict):
        result: dict[Any, Any] = {}  # type: ignore[no-redef]
        truncated_fields = 0
        for key, item in value.items():
            bounded, count = _bound_operator_strings(item)
            result[key] = bounded
            truncated_fields += count
        return result, truncated_fields
    return value, 0


def _bounded_inspect_json(response: dict[str, Any]) -> str:
    """Serialize ``lcm_inspect`` under one final response-size invariant."""
    payload, truncated_fields = _bound_operator_strings(response)
    rollup_truncated_fields = (
        (payload.get("temporal_rollups") or {}).get("truncated_fields") or []
    )
    total_truncated_fields = truncated_fields + len(rollup_truncated_fields)
    payload["char_limit"] = _tools._LCM_INSPECT_MAX_RESPONSE_CHARS
    payload["truncated"] = bool(total_truncated_fields)
    if total_truncated_fields:
        payload["truncated_field_count"] = total_truncated_fields
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded) <= _tools._LCM_INSPECT_MAX_RESPONSE_CHARS:
        return encoded

    # If cardinality rather than one text field exceeds the cap, keep whole
    # top-level sections in a deterministic priority order.  Never cut encoded
    # JSON mid-token; omitted sections are reported explicitly.
    priority = [
        "read_only",
        "session_id",
        "conversation_id",
        "limit",
        "temporal_rollups",
        "runtime_identity",
        "lineage",
        "messages",
        "compaction",
        "dag",
        "externalized_refs",
        "ingest_protection",
        "filters",
        "limit_clamped_from",
    ]
    compact: dict[str, Any] = {
        "char_limit": _tools._LCM_INSPECT_MAX_RESPONSE_CHARS,
        "truncated": True,
        "truncation": {
            "reason": "response_char_limit",
            "omitted_top_level_sections": [],
        },
    }
    if total_truncated_fields:
        compact["truncated_field_count"] = total_truncated_fields
    retained: list[str] = []
    omitted: list[str] = []
    ordered_keys = priority + [key for key in payload if key not in priority]
    for key in dict.fromkeys(ordered_keys):
        if key in {"char_limit", "truncated", "truncated_field_count"}:
            continue
        if key not in payload:
            continue
        compact[key] = payload[key]
        if len(json.dumps(compact, ensure_ascii=False)) <= _tools._LCM_INSPECT_MAX_RESPONSE_CHARS - 1_000:
            retained.append(key)
        else:
            compact.pop(key)
            omitted.append(key)
    compact["truncation"]["omitted_top_level_sections"] = omitted
    encoded = json.dumps(compact, ensure_ascii=False)
    while len(encoded) > _tools._LCM_INSPECT_MAX_RESPONSE_CHARS and retained:
        key = retained.pop()
        compact.pop(key, None)
        omitted.append(key)
        compact["truncation"]["omitted_top_level_sections"] = omitted
        encoded = json.dumps(compact, ensure_ascii=False)
    return encoded


def _matched_session_patterns(session_keys: list[str], patterns: list[str]) -> list[str]:
    """Return configured session glob patterns that match the supplied keys."""
    matched: list[str] = []
    for pattern in patterns:
        try:
            compiled = compile_session_pattern(pattern)
        except re.error:
            continue
        if any(compiled.match(key) for key in session_keys if key):
            matched.append(pattern)
    return matched


def _inspect_externalized_refs_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        sources = [value]
    else:
        try:
            sources = [json.dumps(value, ensure_ascii=False)]
        except (TypeError, ValueError):
            sources = [str(value)]

    refs: list[str] = []
    for source in sources:
        for ref in _tools.extract_ingest_externalized_refs(source) + extract_externalized_refs(source):  # type: ignore[attr-defined]
            if ref not in refs:
                refs.append(ref)
    return refs


def _inspect_message_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Return row metadata only; never include raw message content."""
    content = row.get("content") or ""
    item: dict[str, Any] = {
        "store_id": row.get("store_id"),
        "session_id": row.get("session_id") or "",
        "source": row.get("source") or "",
        "conversation_id": row.get("conversation_id") or "",
        "role": row.get("role") or "unknown",
        "timestamp": row.get("timestamp", 0),
        "token_estimate": row.get("token_estimate", 0),
        "content_chars": len(content),
    }
    if row.get("tool_call_id"):
        item["tool_call_id"] = row.get("tool_call_id")
    if row.get("tool_name"):
        item["tool_name"] = row.get("tool_name")

    refs: list[str] = []
    for value in (row.get("content"), row.get("tool_calls")):
        for ref in _inspect_externalized_refs_from_value(value):
            if ref not in refs:
                refs.append(ref)
    if refs:
        item["externalized_refs"] = refs
    return item


def _inspect_lifecycle_state(engine: "LCMEngine", session_id: str, conversation_id: str) -> dict[str, Any] | None:
    state = None
    if conversation_id:
        state = engine._lifecycle.get_by_conversation(conversation_id)
    if state is None and session_id:
        state = engine._lifecycle.get_by_session(session_id)
    if state is None:
        return None
    return {
        "conversation_id": state.conversation_id,
        "current_session_id": state.current_session_id,
        "last_finalized_session_id": state.last_finalized_session_id,
        "current_frontier_store_id": state.current_frontier_store_id,
        "last_finalized_frontier_store_id": state.last_finalized_frontier_store_id,
        "debt_kind": state.debt_kind,
        "debt_size_estimate": state.debt_size_estimate,
        "current_bound_at": state.current_bound_at,
        "last_finalized_at": state.last_finalized_at,
        "debt_updated_at": state.debt_updated_at,
        "last_maintenance_attempt_at": state.last_maintenance_attempt_at,
        "last_rollover_at": state.last_rollover_at,
        "last_reset_at": state.last_reset_at,
        "updated_at": state.updated_at,
    }


def _inspect_highest_compacted_source_store_id(engine: "LCMEngine", session_id: str) -> int:
    highest = 0
    rows = engine._dag.connection.execute(  # type: ignore[union-attr]
        """
        SELECT source_ids
        FROM summary_nodes
        WHERE session_id = ? AND source_type = 'messages'
        """,
        (session_id,),
    ).fetchall()
    for (raw_source_ids,) in rows:
        try:
            source_ids = json.loads(raw_source_ids or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for source_id in source_ids:
            try:
                highest = max(highest, int(source_id))
            except (TypeError, ValueError, OverflowError):
                continue
    return highest


def _inspect_externalized_refs(engine: "LCMEngine", session_id: str, limit: int) -> dict[str, Any]:
    message_total = engine._store.get_session_count(session_id)
    rows = engine._store.load_session_page(session_id, limit=_tools._LCM_INSPECT_REF_SCAN_MESSAGE_LIMIT)
    scan_truncated = message_total > len(rows)
    items: list[dict[str, Any]] = []
    total_known = 0
    seen: set[tuple[int, str]] = set()
    for row in rows:
        refs: list[str] = []
        for value in (row.get("content"), row.get("tool_calls")):
            for ref in _inspect_externalized_refs_from_value(value):
                if ref not in refs:
                    refs.append(ref)
        for ref in refs:
            key = (int(row.get("store_id") or 0), ref)
            if key in seen:
                continue
            seen.add(key)
            total_known += 1
            if len(items) >= limit:
                continue
            metadata = _tools._inspect_externalized_payload_metadata(engine, ref, session_id)
            item: dict[str, Any] = {
                "externalized_ref": ref,
                "store_id": row.get("store_id"),
                "session_id": row.get("session_id") or "",
                "source": row.get("source") or "",
                "conversation_id": row.get("conversation_id") or "",
                "role": row.get("role") or "unknown",
                "timestamp": row.get("timestamp", 0),
                "readable": metadata.get("readable") is True,
            }
            if row.get("tool_call_id"):
                item["tool_call_id"] = row.get("tool_call_id")
            item.update(metadata)
            items.append(item)

    return {
        "total_known": total_known,
        "total_known_exact": not scan_truncated,
        "scanned_messages": len(rows),
        "scan_truncated": scan_truncated,
        "returned": len(items),
        "has_more": total_known > len(items) or scan_truncated,
        "items": items,
    }


def _temporal_rollups_status(engine: "LCMEngine") -> dict[str, Any]:
    """Return the cheap, read-only temporal-rollup operator status payload."""
    enabled = bool(engine._config.temporal_rollups_enabled)
    scope = engine.current_session_id or ""
    payload: dict[str, Any] = {
        "enabled": enabled,
        "scope": scope,
        "counts": {
            kind: {status: 0 for status in _TEMPORAL_ROLLUP_STATUSES}
            for kind in _TEMPORAL_ROLLUP_PERIOD_KINDS
        },
        "oldest_stale_age_seconds": None,
        "last_build_cursors": {kind: None for kind in _TEMPORAL_ROLLUP_PERIOD_KINDS},
        "last_built_at": {kind: None for kind in _TEMPORAL_ROLLUP_PERIOD_KINDS},
        "last_error": None,
    }
    # Disabled deployments deliberately avoid even the metadata reads. This
    # keeps the optional feature inert while preserving a stable status shape.
    if not enabled or not scope:
        return payload

    conn = engine._dag.connection
    if conn is None:
        return payload
    try:
        rows = conn.execute(
            """
            SELECT period_kind, status, COUNT(*)
            FROM lcm_rollups INDEXED BY sqlite_autoindex_lcm_rollups_1
            WHERE period_kind IN ('day', 'week', 'month') AND scope = ?
            GROUP BY period_kind, status
            """,
            (scope,),
        ).fetchall()
        for period_kind, status, count in rows:
            kind_key = str(period_kind)
            status_key = str(status)
            if kind_key in payload["counts"] and status_key in payload["counts"][kind_key]:
                payload["counts"][kind_key][status_key] = int(count or 0)

        oldest_stale = conn.execute(
            """
            SELECT period_start
            FROM lcm_rollups INDEXED BY sqlite_autoindex_lcm_rollups_1
            WHERE period_kind IN ('day', 'week', 'month')
              AND scope = ? AND status = 'stale'
            ORDER BY period_start
            LIMIT 1
            """,
            (scope,),
        ).fetchone()
        if oldest_stale and oldest_stale[0]:
            stale_start = datetime.combine(
                datetime.fromisoformat(str(oldest_stale[0])).date(),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            payload["oldest_stale_age_seconds"] = max(
                0, int((datetime.now(timezone.utc) - stale_start).total_seconds())
            )

        cursor_rows = conn.execute(
            """
            SELECT period_kind, last_build_cursor, last_built_at
            FROM lcm_rollup_state
            WHERE period_kind IN ('day', 'week', 'month') AND scope = ?
            """,
            (scope,),
        ).fetchall()
        for period_kind, cursor, built_at in cursor_rows:
            kind_key = str(period_kind)
            if kind_key in payload["last_build_cursors"]:
                payload["last_build_cursors"][kind_key] = cursor
                payload["last_built_at"][kind_key] = built_at

        error_row = conn.execute(
            """
            SELECT substr(error, 1, ?)
            FROM lcm_rollups
            WHERE scope = ? AND error IS NOT NULL AND error != ''
            ORDER BY rollup_id DESC
            LIMIT 1
            """,
            (_tools._OPERATOR_TEXT_FIELD_MAX_CHARS + 1, scope),
        ).fetchone()
        if error_row:
            last_error, was_truncated = _bounded_operator_field(error_row[0])
            payload["last_error"] = last_error
            if was_truncated:
                payload.setdefault("truncated_fields", []).append("last_error")
    except Exception as exc:  # pragma: no cover - defensive legacy-schema degradation
        logger.debug("LCM temporal rollup status query failed", exc_info=True)
        query_error, was_truncated = _bounded_operator_field(
            f"{type(exc).__name__}: {exc}"
        )
        payload["query_error"] = query_error
        if was_truncated:
            payload.setdefault("truncated_fields", []).append("query_error")
    return payload


def lcm_inspect(args: Dict[str, Any], **kwargs: Any) -> str:
    """Return a read-only metadata inventory of the current LCM session."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    raw_limit_arg = args.get("limit", _tools._LCM_INSPECT_DEFAULT_LIMIT)
    parsed_limit, limit_error = _tools._parse_strict_int(raw_limit_arg, "limit")
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _tools._LCM_INSPECT_HARD_LIMIT_CAP)

    session_id = engine.current_session_id
    conversation_id = engine.current_conversation_id
    if not session_id:
        full_status = engine.get_status()
        return _bounded_inspect_json({
            "error": "No active session",
            "read_only": True,
            "runtime_identity": full_status.get("runtime_identity") or engine.get_runtime_identity(),
            "ingest_protection": full_status.get("ingest_protection"),
            "temporal_rollups": _temporal_rollups_status(engine),
        })

    full_status = engine.get_status()
    runtime_identity = full_status.get("runtime_identity") or engine.get_runtime_identity()
    lifecycle = _inspect_lifecycle_state(engine, session_id, conversation_id)

    store_totals_row = engine._store.connection.execute(  # type: ignore[union-attr]
        """
        SELECT COUNT(*), MIN(store_id), MAX(store_id), COALESCE(SUM(token_estimate), 0)
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    message_total = int(store_totals_row[0] or 0) if store_totals_row else 0
    min_store_id = store_totals_row[1] if store_totals_row else None
    max_store_id = store_totals_row[2] if store_totals_row else None
    estimated_tokens = int(store_totals_row[3] or 0) if store_totals_row else 0
    fresh_tail_count = max(0, int(engine._config.fresh_tail_count or 0))
    fresh_tail_rows, fresh_tail_boundary = engine._get_session_fresh_tail(session_id)
    fresh_tail_display_rows = fresh_tail_rows[-limit:]
    fresh_tail_items = [
        _inspect_message_metadata(row)
        for row in fresh_tail_display_rows
    ]

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    total_dag_nodes = sum(info["count"] for info in depth_stats.values())
    total_dag_tokens = sum(info["tokens"] for info in depth_stats.values())
    total_dag_source_tokens = sum(info["source_tokens"] for info in depth_stats.values())
    latest_node_rows = engine._dag.connection.execute(  # type: ignore[union-attr]
        """
        SELECT node_id, session_id, depth, token_count, source_token_count,
               source_type, created_at, earliest_at, latest_at, expand_hint
        FROM summary_nodes
        WHERE session_id = ?
        ORDER BY created_at DESC, node_id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    latest_nodes = [
        {
            "node_id": int(row[0]),
            "session_id": row[1],
            "depth": int(row[2]),
            "token_count": int(row[3] or 0),
            "source_token_count": int(row[4] or 0),
            "source_type": row[5],
            "created_at": row[6],
            "earliest_at": row[7],
            "latest_at": row[8],
            "expand_hint_available": bool(row[9]),
            "expand_hint_chars": len(row[9] or ""),
        }
        for row in latest_node_rows
    ]

    highest_compacted_source_store_id = _inspect_highest_compacted_source_store_id(engine, session_id)
    lifecycle_current_frontier = int((lifecycle or {}).get("current_frontier_store_id") or 0)
    lifecycle_finalized_frontier = int((lifecycle or {}).get("last_finalized_frontier_store_id") or 0)
    runtime_last_compacted = int(getattr(engine, "_last_compacted_store_id", 0) or 0)

    platform = engine.current_session_platform
    session_keys = build_session_match_keys(session_id, platform=platform)
    ignore_patterns = list(engine._config.ignore_session_patterns or [])
    stateless_patterns = list(engine._config.stateless_session_patterns or [])

    response: dict[str, Any] = {
        "read_only": True,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "limit": limit,
        "runtime_identity": runtime_identity,
        "lineage": {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "session_platform": platform,
            "side_channel_active": engine.side_channel_active,
            "bound_session_id": getattr(engine, "_session_id", ""),
            "bound_conversation_id": getattr(engine, "_conversation_id", ""),
            "lifecycle": lifecycle,
            "source_lineage": full_status.get("source_lineage"),
        },
        "messages": {
            "total": message_total,
            "estimated_tokens": estimated_tokens,
            "min_store_id": min_store_id,
            "max_store_id": max_store_id,
            "fresh_tail_count": fresh_tail_count,
            "fresh_tail_max_tokens": engine._config.fresh_tail_max_tokens,
            "effective_fresh_tail_count": len(fresh_tail_rows),
            "effective_fresh_tail_tokens": fresh_tail_boundary.tokens,
            "pre_tail_message_count": max(0, message_total - len(fresh_tail_rows)),
            "fresh_tail": {
                "returned": len(fresh_tail_items),
                "token_limited": fresh_tail_boundary.token_limited,
                "tool_group_extended": fresh_tail_boundary.tool_group_extended,
                "items": fresh_tail_items,
            },
        },
        "compaction": {
            "last": {
                "status": full_status.get("last_compression_status", "idle"),
                "noop_reason": full_status.get("last_compression_noop_reason", ""),
                "condensation_suppressed_reason": full_status.get("condensation_suppressed_reason", ""),
                "compression_count": engine.compression_count,
                "last_prompt_tokens": engine.last_prompt_tokens,
                "threshold_tokens": engine.threshold_tokens,
            },
            "frontier": {
                "runtime_last_compacted_store_id": runtime_last_compacted,
                "highest_compacted_source_store_id": highest_compacted_source_store_id,
                "lifecycle_current_frontier_store_id": lifecycle_current_frontier,
                "lifecycle_last_finalized_frontier_store_id": lifecycle_finalized_frontier,
            },
        },
        "dag": {
            "total_nodes": total_dag_nodes,
            "total_tokens": total_dag_tokens,
            "total_source_tokens": total_dag_source_tokens,
            "depths": {f"d{depth}": info for depth, info in sorted(depth_stats.items())},
            "latest_nodes": latest_nodes,
        },
        "temporal_rollups": _temporal_rollups_status(engine),
        "externalized_refs": _inspect_externalized_refs(engine, session_id, limit),
        "ingest_protection": full_status.get("ingest_protection"),
        "filters": {
            "session_keys": session_keys,
            "ignored": engine.current_session_ignored,
            "stateless": engine.current_session_stateless,
            "ignore_session_patterns": ignore_patterns,
            "stateless_session_patterns": stateless_patterns,
            "matched_ignore_session_patterns": _matched_session_patterns(session_keys, ignore_patterns),
            "matched_stateless_session_patterns": _matched_session_patterns(session_keys, stateless_patterns),
            "ignore_message_patterns": list(engine._config.ignore_message_patterns or []),
            "ignored_message_count": full_status.get("ignored_message_count", 0),
        },
    }
    if requested_limit > _tools._LCM_INSPECT_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    return _bounded_inspect_json(response)
