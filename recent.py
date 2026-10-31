"""Recent temporal retrieval behind the public tools facade."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, TYPE_CHECKING

# Resolve shared helpers/constants at call time: tools also imports this family.
from . import tools as _tools
from .rollup_periods import (
    CoverageNode,
    RecentPeriodWindow,
    canonical_frontier,
    load_source_lineage,
    parse_recent_period,
)
from .rollup_store import RollupStore
from .sqlite_util import _sqlite_savepoint

if TYPE_CHECKING:
    from .engine import LCMEngine


# The tools logger, under whatever package name the host loads this plugin as (``hermes_lcm`` in tests,
# ``hermes_plugins.<name>`` in Hermes), so the moved recent family keeps logging where tools.py logs.
logger = logging.getLogger(f"{__package__}.tools" if __package__ else "tools")


def _recent_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _recent_rollup_bounds(window: RecentPeriodWindow) -> tuple[str, str]:
    start = window.start.date().isoformat()
    if window.rollup_kind == "day":
        end = (window.end - timedelta(microseconds=1)).date().isoformat()
        return start, end
    return start, start


def _recent_has_unready_rollups(
    store: RollupStore,
    window: RecentPeriodWindow,
    scope: str,
) -> bool:
    """True when the window is not fully covered by ``ready`` rollups.

    Detects MISSING days (no row at all), not only existing non-ready rows: every
    period the window requires must have a ``ready`` rollup, otherwise the whole
    window falls back (maintainer #389 blocker 1).
    """
    connection = store.connection
    if connection is None:
        return True
    if window.rollup_kind == "day":
        first = window.start.date()
        last = (window.end - timedelta(microseconds=1)).date()
        expected_count = (last - first).days + 1
    else:
        expected_count = 1
    if expected_count <= 0 or expected_count > _tools._LCM_RECENT_FRONTIER_WORK_LIMIT:
        return True
    start, end = _recent_rollup_bounds(window)
    ready_row = connection.execute(
        """
        SELECT COUNT(*)
        FROM lcm_rollups
        WHERE period_kind = ?
          AND period_start >= ?
          AND period_start <= ?
          AND scope = ?
          AND status = 'ready'
        """,
        (window.rollup_kind, start, end, scope),
    ).fetchone()
    return int(ready_row[0] or 0) != expected_count


def _session_has_window_content(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    session_id: str,
) -> bool:
    """True when ``session_id`` has a summary node whose covered span overlaps
    the window (earliest < end AND latest >= start), matching the leaf-fallback's
    overlap semantics rather than newest-timestamp-only.
    """
    connection = engine._dag.connection
    if connection is None or not session_id:
        return False
    try:
        with engine._dag._db_lock:
            row = connection.execute(
                """
                SELECT 1
                FROM summary_nodes
                WHERE session_id = ?
                  AND COALESCE(earliest_at, created_at) < ?
                  AND COALESCE(latest_at, created_at) >= ?
                LIMIT 1
                """,
                (session_id, window.end.timestamp(), window.start.timestamp()),
            ).fetchone()
    except Exception:  # pragma: no cover - defensive read-only degradation
        return False
    return row is not None


def _recent_ready_rollups(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    scope: str,
) -> tuple[list[dict[str, object]], str | None]:
    if window.subday:
        return [], "subday_window"
    if not engine._config.temporal_rollups_enabled:
        return [], "temporal_rollups_disabled"

    # Rollups are session-scoped, but the leaf fallback spans the whole
    # conversation family (current + last-finalized session). Only serve rollups
    # when the rollup scope solely covers the window: if another session in that
    # span holds overlapping content (post-rotation retained lineage), fall back
    # to leaf sections, which span the same sessions — so rollup mode never drops
    # a finalized session's content (maintainer #389: match the fallback span).
    for other in _recent_conversation_scope_session_ids(engine):
        if other != scope and _session_has_window_content(engine, window, other):
            return [], "rollups_span_multiple_sessions"

    store: RollupStore | None = None
    try:
        store = RollupStore(engine._dag.db_path)
        # A summary mutation and its invalidation event commit atomically.  Do
        # not serve a previously-ready rollup while that durable event is still
        # waiting for bounded maintenance to reconcile the affected periods.
        if store.has_pending_invalidations(scope):
            return [], "rollups_invalidation_pending"
        start, end = _recent_rollup_bounds(window)
        if _recent_has_unready_rollups(store, window, scope):
            return [], "rollups_unavailable"
        rollups = store.ready_rollups_for_window(
            window.rollup_kind,
            start,
            end,
            scope,
        )
        if not rollups:
            return [], "rollups_unavailable"
        return rollups, None
    except Exception:
        logger.debug("LCM recent rollup read failed; using leaf summaries", exc_info=True)
        return [], "rollups_unavailable"
    finally:
        if store is not None:
            store.close()


def _recent_conversation_scope_session_ids(engine: "LCMEngine") -> list[str]:
    """Session ids that make up the current conversation family for fallback.

    Rotation reassigns retained higher-depth/carry-forward summaries into the
    new session, but a just-finalized session may still hold retained lineage, so
    the fallback spans the current session plus the conversation's finalized
    session (maintainer #389 blocker 2). Scope identity is otherwise session-based
    (see the operator guide) because summary nodes carry no conversation key.
    """
    ids: list[str] = []
    current = str(engine.current_session_id or "")
    if current:
        ids.append(current)
    lifecycle = getattr(engine, "_lifecycle", None)
    conversation_id = getattr(engine, "current_conversation_id", None)
    if lifecycle is not None and conversation_id:
        try:
            state = lifecycle.get_by_conversation(conversation_id)
        except Exception:  # pragma: no cover - defensive read-only degradation
            state = None
        if state is not None:
            for sid in (state.current_session_id, state.last_finalized_session_id):
                if sid and str(sid) not in ids:
                    ids.append(str(sid))
    return ids or [current]


def _recent_leaf_sections(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    requested_scope: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Load fallback nodes without retaining their TEMP-staging snapshot."""
    with engine._dag._db_lock:
        connection = engine._dag.connection
        if connection is None:
            return []
        with _sqlite_savepoint(connection):
            return _recent_leaf_sections_staged(
                engine,
                window,
                requested_scope,
                limit,
            )


def _recent_leaf_sections_staged(
    engine: "LCMEngine",
    window: RecentPeriodWindow,
    requested_scope: str,
    limit: int,
) -> list[dict[str, Any]]:
    connection = engine._dag.connection
    if connection is None:
        return []
    # Include retained higher-depth/carry-forward summaries, not just depth-0
    # current-session leaves, mirroring how lcm_grep/describe select across
    # depths and retained lineage (maintainer #389 blocker 2).
    # Include any summary whose covered span INTERSECTS the window, not only
    # those whose newest timestamp lands inside it: a summary spanning several
    # days (earliest before the window, latest inside, or vice versa) still holds
    # window content and must be returned (maintainer #389: overlap, not
    # latest_at). Overlap = earliest < window.end AND latest >= window.start.
    where = [
        "COALESCE(earliest_at, created_at) < ?",
        "COALESCE(latest_at, created_at) >= ?",
    ]
    params: list[object] = [window.end.timestamp(), window.start.timestamp()]
    if requested_scope == "conversation":
        session_ids = _recent_conversation_scope_session_ids(engine)
        placeholders = ",".join("?" for _ in session_ids)
        where.append(f"session_id IN ({placeholders})")
        params.extend(session_ids)
    # Probe one sentinel row beyond the work cap without an expression ORDER BY.
    # The session index can stop at the sentinel instead of scanning/sorting the
    # full matching corpus; ordering happens only after this set is proven small.
    probe_params = [*params, _tools._LCM_RECENT_FRONTIER_WORK_LIMIT + 1]
    try:
        with engine._dag._db_lock:
            id_rows = connection.execute(
                f"""
                SELECT node_id
                FROM summary_nodes
                WHERE {' AND '.join(where)}
                LIMIT ?
                """,
                probe_params,
            ).fetchall()
            if len(id_rows) > _tools._LCM_RECENT_FRONTIER_WORK_LIMIT:
                logger.warning(
                    "LCM recent fallback exceeded the %d-node canonical frontier "
                    "bound; returning no partial frontier",
                    _tools._LCM_RECENT_FRONTIER_WORK_LIMIT,
                )
                return []
            if not id_rows:
                return []

            connection.execute(
                "CREATE TEMP TABLE IF NOT EXISTS lcm_recent_frontier_ids "
                "(node_id INTEGER PRIMARY KEY) WITHOUT ROWID"
            )
            connection.execute("DELETE FROM temp.lcm_recent_frontier_ids")
            try:
                connection.executemany(
                    "INSERT INTO temp.lcm_recent_frontier_ids(node_id) VALUES(?)",
                    ((int(row[0]),) for row in id_rows),
                )
                rows = connection.execute(
                    """
                    SELECT node.node_id, node.session_id, node.summary,
                           node.token_count, node.depth, node.source_ids,
                           node.source_type,
                           COALESCE(node.earliest_at, node.created_at) AS earliest_at,
                           COALESCE(node.latest_at, node.created_at) AS latest_at
                    FROM temp.lcm_recent_frontier_ids wanted
                    JOIN summary_nodes node ON node.node_id = wanted.node_id
                    ORDER BY COALESCE(node.latest_at, node.created_at) DESC,
                             node.node_id DESC
                    """
                ).fetchall()
            finally:
                connection.execute("DELETE FROM temp.lcm_recent_frontier_ids")

            source_lineage = load_source_lineage(
                connection,
                [int(row[0]) for row in rows],
                limit=_tools._LCM_RECENT_FRONTIER_WORK_LIMIT,
            )
    except Exception:
        logger.debug(
            "LCM recent fallback or transitive lineage read failed closed",
            exc_info=True,
        )
        return []

    candidates: list[CoverageNode] = []
    by_id: dict[int, sqlite3.Row] = {}
    for row in rows:
        node_id = int(row[0])
        source_type = str(row[6] or "")
        source_node_ids: tuple[int, ...] = ()
        if source_type == "nodes" and row[5]:
            try:
                source_node_ids = tuple(int(value) for value in json.loads(row[5]))
            except (TypeError, ValueError):
                source_node_ids = ()
        candidates.append(
            CoverageNode(
                node_id=node_id,
                depth=int(row[4] or 0),
                source_node_ids=source_node_ids,
                earliest_at=row[7],
                latest_at=row[8],
            )
        )
        by_id[node_id] = row

    # Suppress any child whose coverage is contained by an overlapping selected
    # parent, THEN apply the limit — so duplicated lineage cannot consume the
    # public budget twice (maintainer #389 C1). ``rows`` is already ordered
    # newest-first, and canonical_frontier preserves that order.
    try:
        frontier_rows = [
            by_id[node.node_id]
            for node in canonical_frontier(
                candidates, source_lineage=source_lineage
            )
        ][:limit]
    except Exception:
        logger.debug("LCM recent canonical frontier failed closed", exc_info=True)
        return []
    return [
        {
            "kind": "leaf_summary",
            "node_id": int(row[0]),
            "session_id": str(row[1]),
            "token_count": int(row[3] or 0),
            "earliest_at": row[7],
            "latest_at": row[8],
            "content": str(row[2] or ""),
            "content_truncated": False,
        }
        for row in frontier_rows
    ]


def _recent_rollup_sections(rollups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for rollup in sorted(rollups, key=lambda row: str(row["period_start"]), reverse=True):
        token_count = int(rollup.get("token_count") or 0)
        sections.append(
            {
                "kind": "rollup",
                "rollup_id": int(rollup["rollup_id"]),
                "period_kind": str(rollup["period_kind"]),
                "period_start": str(rollup["period_start"]),
                "status": str(rollup["status"]),
                "token_count": token_count,
                "content": f"Tokens: {token_count}\n{rollup.get('summary') or ''}",
                "content_truncated": False,
            }
        )
    return sections


def _bounded_recent_json(response: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    response["sections"] = []
    response["total_sections"] = len(sections)
    response["returned_sections"] = 0
    response["truncated"] = False
    provenance = response.setdefault("provenance", {})

    def encode() -> str:
        # Provenance is bound to the sections actually RETURNED, not to every
        # candidate rollup: a large ready-rollup set must not serialize thousands
        # of provenance rows outside the char budget (maintainer #389 C2). Because
        # this runs inside every fit check, the per-section provenance entry is
        # counted against the cap in lockstep with its section.
        provenance["rollups"] = [
            {"rollup_id": int(section["rollup_id"]), "status": str(section["status"])}
            for section in response["sections"]
            if section.get("kind") == "rollup"
        ]
        return json.dumps(response, ensure_ascii=False)

    for section in sections:
        response["sections"].append(section)
        response["returned_sections"] = len(response["sections"])
        if len(encode()) <= _tools._LCM_RECENT_MAX_RESPONSE_CHARS:
            continue

        response["sections"].pop()
        response["returned_sections"] = len(response["sections"])
        content = str(section.get("content") or "")
        low, high = 0, len(content)
        best: dict[str, Any] | None = None
        while low <= high:
            midpoint = (low + high) // 2
            candidate = dict(section)
            candidate["content"] = content[:midpoint]
            candidate["content_truncated"] = midpoint < len(content)
            response["sections"].append(candidate)
            response["returned_sections"] = len(response["sections"])
            fits = len(encode()) <= _tools._LCM_RECENT_MAX_RESPONSE_CHARS
            response["sections"].pop()
            response["returned_sections"] = len(response["sections"])
            if fits:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best is not None:
            response["sections"].append(best)
            response["returned_sections"] = len(response["sections"])
        response["truncated"] = True
        break

    if response["returned_sections"] < response["total_sections"]:
        response["truncated"] = True
    return encode()


def lcm_recent(args: Dict[str, Any], **kwargs: Any) -> str:
    """Serve conversation rollups or fall back; cross-session rollups are future work."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})

    try:
        # The parser intentionally validates missing/untyped tool input.
        window = parse_recent_period(args.get("period"))  # type: ignore[arg-type]
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    requested_scope = str(args.get("scope", "conversation")).strip().lower()
    if requested_scope != "conversation":
        return json.dumps({"error": "scope must be one of: conversation"})

    parsed_limit, limit_error = _tools._parse_strict_int(
        args.get("limit", _tools._LCM_RECENT_DEFAULT_LIMIT),
        "limit",
    )
    if limit_error:
        return json.dumps({"error": limit_error})
    if parsed_limit is None or parsed_limit <= 0:
        return json.dumps({"error": "limit must be a positive integer"})
    requested_limit = parsed_limit
    limit = min(requested_limit, _tools._LCM_RECENT_HARD_LIMIT_CAP)

    rollup_scope = engine.current_session_id
    rollups, fallback_reason = _recent_ready_rollups(engine, window, rollup_scope)
    fallback = not rollups
    if fallback:
        sections = _recent_leaf_sections(engine, window, requested_scope, limit)
    else:
        sections = _recent_rollup_sections(rollups)[:limit]

    response: dict[str, Any] = {
        "period": window.period,
        "scope": requested_scope,
        "window": {
            "start": _recent_iso(window.start),
            "end": _recent_iso(window.end),
        },
        "limit": limit,
        "char_limit": _tools._LCM_RECENT_MAX_RESPONSE_CHARS,
        "mode": "leaf_summary_fallback" if fallback else "rollup",
        # ``provenance.rollups`` is filled by _bounded_recent_json from the
        # sections actually returned (bounded by limit + char cap);
        # ``rollups_covered`` is the O(1) aggregate count of ready rollups the
        # window matched, so operators still see "N covered, showing M"
        # (maintainer #389 C2).
        "provenance": {"fallback": fallback},
        "rollups_covered": len(rollups),
    }
    if fallback_reason is not None:
        response["fallback_reason"] = fallback_reason
    if requested_limit > _tools._LCM_RECENT_HARD_LIMIT_CAP:
        response["limit_clamped_from"] = requested_limit
    return _bounded_recent_json(response, sections)
