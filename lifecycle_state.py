"""Durable lifecycle/checkpoint state for hermes-lcm.

This is the smallest viable substrate for cross-turn/session lifecycle state:
- which logical conversation a session belongs to
- which session is currently bound
- which session was last finalized
- the active session frontier/checkpoint marker
- the last finalized frontier marker
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .db_bootstrap import configure_connection, refuse_schema_version_too_new, run_versioned_migrations


class LifecycleBindingChangedError(RuntimeError):
    """Raised when publication loses its active session binding."""


class LifecyclePublicationConflictError(RuntimeError):
    """Raised when compaction publication cannot prove its source frontier."""


_LEGACY_ASCII_WHITESPACE = "\t\n\v\f\r "


def legacy_blank_clause(column: str) -> str:
    """Use the existing legacy common-ASCII whitespace convention in SQL."""
    chars = "char(9) || char(10) || char(11) || char(12) || char(13) || char(32)"
    return f"({column} IS NULL OR TRIM({column}, {chars}) = '')"


def _legacy_attribution(conn, kind, identity):
    row = conn.execute("SELECT value FROM metadata WHERE key = ?",
        ("lcm_legacy_attribution:" + kind + ":" + json.dumps(identity),)).fetchone()
    if row is None:
        return None
    try:
        values = json.loads(row[0])
        return set(values) if isinstance(values, list) and all(isinstance(v, str) for v in values) else set()
    except (ValueError, TypeError):
        return set()


def _write_legacy_attribution(conn, kind, identity, values):
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("lcm_legacy_attribution:" + kind + ":" + json.dumps(identity), json.dumps(sorted(values))))


def _record_legacy_bindings(conn, conversation_id, new_session_id):
    row = conn.execute("SELECT current_session_id, last_finalized_session_id "
        "FROM lcm_lifecycle_state WHERE conversation_id = ?", (conversation_id,)).fetchone()
    bound = {value for value in (row or ()) if value}
    proven = unambiguous_legacy_session_ids(conn, conversation_id, bound)
    history = _legacy_attribution(conn, "conversation", conversation_id) or set()
    for producer in bound | {new_session_id}:
        owners = _legacy_attribution(conn, "producer", producer)
        if owners is None:
            has_blank = conn.execute("SELECT 1 FROM messages WHERE session_id = ? AND "
                + legacy_blank_clause("conversation_id") + " LIMIT 1", (producer,)).fetchone()
            owners = {conversation_id} if producer in proven or not has_blank else set()
        elif owners:
            owners.add(conversation_id)
        # Empty is a durable refusal: a newly bound SID cannot adopt old blanks.
        _write_legacy_attribution(conn, "producer", producer, owners)
        if conversation_id in owners:
            history.add(producer)
    _write_legacy_attribution(conn, "conversation", conversation_id, history)


def unambiguous_legacy_session_ids(
    conn: sqlite3.Connection,
    conversation_id: str,
    session_ids: set[str],
) -> set[str]:
    """Return blank-row session owners with exactly one lifecycle binding.

    Explicit ``messages.conversation_id`` values remain authoritative.  A
    blank legacy row may use a session binding only when neither lifecycle
    index nor explicitly owned raw rows associate it with another conversation.
    """
    candidates = {str(value) for value in session_ids if str(value)}
    candidates |= _legacy_attribution(conn, "conversation", conversation_id) or set()
    if not candidates:
        return set()
    return {
        candidate
        for candidate in candidates
        if (_legacy_attribution(conn, "producer", candidate) in (None, {conversation_id}))
        and conn.execute(
            f"""
            SELECT (? OR EXISTS(
                SELECT 1 FROM lcm_lifecycle_state
                WHERE conversation_id = ?
                  AND (current_session_id = ? OR last_finalized_session_id = ?)
            )) AND NOT EXISTS(
                SELECT 1
                FROM lcm_lifecycle_state
                     INDEXED BY idx_lcm_lifecycle_current_session
                WHERE current_session_id = ? AND conversation_id != ?
            ) AND NOT EXISTS(
                SELECT 1
                FROM lcm_lifecycle_state
                     INDEXED BY idx_lcm_lifecycle_last_finalized_session
                WHERE last_finalized_session_id = ? AND conversation_id != ?
            ) AND NOT EXISTS(
                SELECT 1 FROM messages INDEXED BY idx_msg_session
                WHERE session_id = ? AND NOT {legacy_blank_clause('conversation_id')}
                  AND conversation_id != ?
            )
            """,
            (
                _legacy_attribution(conn, "producer", candidate) == {conversation_id},
                conversation_id, candidate, candidate,
                candidate, conversation_id, candidate, conversation_id,
                candidate, conversation_id,
            ),
        ).fetchone()[0]
    }


def conversation_producers(conn, conversation_id, session_id):
    """Bound candidate queries by explicit producers and durable bindings."""
    row = conn.execute(
        "SELECT current_session_id, last_finalized_session_id FROM lcm_lifecycle_state "
        "WHERE conversation_id = ?", (conversation_id,),
    ).fetchone()
    bound = {str(value) for value in (row or ()) if value}
    producers = {str(row[0]) for row in conn.execute(
        "SELECT DISTINCT session_id FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    )}
    legacy = unambiguous_legacy_session_ids(conn, conversation_id, bound)
    return producers | bound | legacy | {session_id}, legacy


def admitted_summary_roots(conn, conversation_id, session_id, *, before_node_id=None,
                           include_descendants=False):
    """Return roots with complete same-owner lineage, ignoring foreign roots.

    Queries use the existing producer index; raw ownership, never producer
    equality, admits a node. Invalid parents cannot hide valid child roots.
    ``include_descendants`` applies the same validation to explicit expansion.
    """
    producers, legacy = conversation_producers(conn, conversation_id, session_id)
    rows = conn.execute(
        "SELECT node_id, depth, source_ids, source_type, LENGTH(summary), token_count "
        "FROM summary_nodes INDEXED BY idx_nodes_session_latest "
        "WHERE session_id IN (SELECT value FROM json_each(?)) "
        "AND (? IS NULL OR node_id < ?)",
        (json.dumps(sorted(producers)), before_node_id, before_node_id),
    ).fetchall()
    nodes = {int(row[0]): row for row in rows}
    validated, visiting = {}, set()

    def validate(node_id):
        if node_id in validated:
            return validated[node_id]
        row = nodes.get(node_id)
        if node_id in visiting:
            return None
        if row is None or not row[4] or not row[5]:
            validated[node_id] = None
            return None
        visiting.add(node_id)
        represented = set()
        try:
            sources = [int(value) for value in json.loads(row[2])]
            if not sources or len(sources) != len(set(sources)):
                raise ValueError("invalid source list")
            if row[3] == "messages":
                raw = conn.execute(
                    "SELECT store_id, conversation_id, session_id FROM messages "
                    "WHERE store_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(sources),),
                ).fetchall()
                if len(raw) != len(sources) or any(
                    str(item[1] or "").strip(_LEGACY_ASCII_WHITESPACE) != conversation_id and not
                    (not str(item[1] or "").strip(_LEGACY_ASCII_WHITESPACE) and item[2] in legacy)
                    for item in raw
                ):
                    raise ValueError("foreign or missing source")
                represented.update(sources)
            elif row[3] == "nodes":
                for child in sources:
                    if child not in nodes or nodes[child][1] >= row[1]:
                        raise ValueError("invalid child")
                    child_sources = validate(child)
                    if child_sources is None or represented & child_sources:
                        raise ValueError("invalid or repeated lineage")
                    represented.update(child_sources)
            else:
                raise ValueError("invalid source type")
        except (ValueError, TypeError):
            represented = None
        visiting.remove(node_id)
        validated[node_id] = represented
        return represented

    for node_id in nodes:
        validate(node_id)
    children = {int(child) for node_id, row in nodes.items()
                if validated[node_id] is not None and row[3] == "nodes"
                for child in json.loads(row[2])}
    return {node_id: sources for node_id, sources in validated.items()
            if sources is not None and (include_descendants or node_id not in children)}


def _synchronized(method):
    """Serialize a read-modify-write method on the store's reentrant lock.

    The lifecycle connection is shared across threads (check_same_thread=False,
    autocommit). Without serialization, two callers reading state and then
    writing can interleave and clobber each other's update.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


@dataclass
class LifecycleState:
    conversation_id: str
    current_session_id: str | None
    last_finalized_session_id: str | None
    current_frontier_store_id: int
    last_finalized_frontier_store_id: int
    debt_kind: str | None
    debt_size_estimate: int
    current_bound_at: float | None
    last_finalized_at: float | None
    debt_updated_at: float | None
    last_maintenance_attempt_at: float | None
    last_rollover_at: float | None
    last_reset_at: float | None
    updated_at: float


def _binding_transaction(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            result = method(self, *args, **kwargs)
            self._conn.commit()
            return result
        except BaseException:
            self._conn.rollback()
            raise
    return wrapped


class LifecycleStateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        # The connection is opened check_same_thread=False in autocommit mode
        # and is shared across the gateway thread, dispatcher, and sub-agents.
        # Serialize read-modify-write flows so concurrent binds/frontier
        # advances cannot interleave and regress the checkpoint.
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        refuse_schema_version_too_new(self._conn)
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        run_versioned_migrations(self._conn)
        self._conn.commit()

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error:
                pass
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The live SQLite connection, or ``None`` once :meth:`close` has run.

        Exposed for read-oriented diagnostics -- for example the doctor's
        maintenance-debt scan -- that need ad-hoc queries the store does not wrap
        in a purpose-built method. Callers must treat it as read-only; writes go
        through the store's own methods.
        """
        return getattr(self, "_conn", None)

    def row_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM lcm_lifecycle_state").fetchone()
        return int(row["count"] if row else 0)

    def _row_to_state(self, row: sqlite3.Row | None) -> LifecycleState | None:
        if row is None:
            return None
        return LifecycleState(
            conversation_id=row["conversation_id"],
            current_session_id=row["current_session_id"],
            last_finalized_session_id=row["last_finalized_session_id"],
            current_frontier_store_id=int(row["current_frontier_store_id"] or 0),
            last_finalized_frontier_store_id=int(row["last_finalized_frontier_store_id"] or 0),
            debt_kind=row["debt_kind"],
            debt_size_estimate=int(row["debt_size_estimate"] or 0),
            current_bound_at=row["current_bound_at"],
            last_finalized_at=row["last_finalized_at"],
            debt_updated_at=row["debt_updated_at"],
            last_maintenance_attempt_at=row["last_maintenance_attempt_at"],
            last_rollover_at=row["last_rollover_at"],
            last_reset_at=row["last_reset_at"],
            updated_at=float(row["updated_at"] or 0.0),
        )

    def get_by_conversation(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        row = self._conn.execute(
            "SELECT * FROM lcm_lifecycle_state WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return self._row_to_state(row)

    def get_by_session(self, session_id: str | None) -> LifecycleState | None:
        if not session_id:
            return None
        row = self._conn.execute(
            """
            SELECT *
            FROM lcm_lifecycle_state
            WHERE current_session_id = ? OR last_finalized_session_id = ?
            ORDER BY CASE WHEN current_session_id = ? THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1
            """,
            (session_id, session_id, session_id),
        ).fetchone()
        return self._row_to_state(row)

    @_synchronized
    @_binding_transaction
    def bind_session(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> LifecycleState:
        existing = self.get_by_conversation(conversation_id) if conversation_id else self.get_by_session(session_id)
        conversation_id = conversation_id or (existing.conversation_id if existing else session_id)
        _record_legacy_bindings(self._conn, conversation_id, session_id)
        now = time.time()
        current_frontier = 0
        current_bound_at = now
        last_finalized_session_id = None
        last_finalized_frontier = 0
        debt_kind = None
        debt_size_estimate = 0
        last_finalized_at = None
        debt_updated_at = None
        last_maintenance_attempt_at = None
        last_rollover_at = None
        last_reset_at = None

        if existing is not None:
            if existing.current_session_id == session_id:
                return existing
            current_frontier = (
                existing.current_frontier_store_id if existing.current_session_id == session_id else 0
            )
            current_bound_at = (
                existing.current_bound_at if existing.current_session_id == session_id else now
            )
            last_finalized_session_id = existing.last_finalized_session_id
            last_finalized_frontier = existing.last_finalized_frontier_store_id
            debt_kind = existing.debt_kind
            debt_size_estimate = existing.debt_size_estimate
            last_finalized_at = existing.last_finalized_at
            debt_updated_at = existing.debt_updated_at
            last_maintenance_attempt_at = existing.last_maintenance_attempt_at
            last_rollover_at = (
                now
                if (
                    (existing.current_session_id and existing.current_session_id != session_id)
                    or (
                        existing.current_session_id is None
                        and existing.last_finalized_session_id
                        and existing.last_finalized_session_id != session_id
                    )
                )
                else existing.last_rollover_at
            )
            last_reset_at = existing.last_reset_at

        self._conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id,
                current_session_id,
                last_finalized_session_id,
                current_frontier_store_id,
                last_finalized_frontier_store_id,
                debt_kind,
                debt_size_estimate,
                current_bound_at,
                last_finalized_at,
                debt_updated_at,
                last_maintenance_attempt_at,
                last_rollover_at,
                last_reset_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = excluded.current_frontier_store_id,
                last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id,
                debt_kind = excluded.debt_kind,
                debt_size_estimate = excluded.debt_size_estimate,
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                debt_updated_at = excluded.debt_updated_at,
                last_maintenance_attempt_at = excluded.last_maintenance_attempt_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                session_id,
                last_finalized_session_id,
                current_frontier,
                last_finalized_frontier,
                debt_kind,
                debt_size_estimate,
                current_bound_at,
                last_finalized_at,
                debt_updated_at,
                last_maintenance_attempt_at,
                last_rollover_at,
                last_reset_at,
                now,
            ),
        )
        self._conn.commit()
        state = self.get_by_conversation(conversation_id)
        assert state is not None
        return state

    @_synchronized
    @_binding_transaction
    def finalize_session(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int = 0,
    ) -> LifecycleState | None:
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        _record_legacy_bindings(self._conn, conversation_id, session_id)
        now = time.time()
        current_session_id = state.current_session_id
        current_frontier = state.current_frontier_store_id
        if current_session_id == session_id:
            current_session_id = None
            current_frontier = 0
        finalized_frontier = max(
            int(frontier_store_id or 0),
            state.last_finalized_frontier_store_id,
        )
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_session_id = ?,
                last_finalized_session_id = ?,
                current_frontier_store_id = ?,
                last_finalized_frontier_store_id = ?,
                debt_kind = debt_kind,
                debt_size_estimate = debt_size_estimate,
                last_finalized_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (
                current_session_id,
                session_id,
                current_frontier,
                finalized_frontier,
                now,
                now,
                state.conversation_id,
            ),
        )
        self._conn.commit()
        return self.get_by_conversation(state.conversation_id)

    @_synchronized
    @_binding_transaction
    def record_rollover(
        self,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        finalized_frontier_store_id: int = 0,
    ) -> LifecycleState:
        state = self.get_by_conversation(conversation_id)
        _record_legacy_bindings(self._conn, conversation_id, new_session_id)
        if (
            state is not None
            and state.current_session_id == new_session_id
            and state.last_finalized_session_id == old_session_id
        ):
            return state

        now = time.time()
        last_finalized_frontier = max(
            int(finalized_frontier_store_id or 0),
            state.last_finalized_frontier_store_id if state else 0,
        )
        self._conn.execute(
            """
            INSERT INTO lcm_lifecycle_state(
                conversation_id,
                current_session_id,
                last_finalized_session_id,
                current_frontier_store_id,
                last_finalized_frontier_store_id,
                current_bound_at,
                last_finalized_at,
                last_rollover_at,
                last_reset_at,
                updated_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                current_session_id = excluded.current_session_id,
                last_finalized_session_id = excluded.last_finalized_session_id,
                current_frontier_store_id = 0,
                last_finalized_frontier_store_id = excluded.last_finalized_frontier_store_id,
                current_bound_at = excluded.current_bound_at,
                last_finalized_at = excluded.last_finalized_at,
                last_rollover_at = excluded.last_rollover_at,
                last_reset_at = excluded.last_reset_at,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                new_session_id,
                old_session_id,
                last_finalized_frontier,
                now,
                now,
                now,
                now,
                now,
            ),
        )
        self._conn.commit()
        updated = self.get_by_conversation(conversation_id)
        assert updated is not None
        return updated

    def get_fragmentation_stats(self, state_db_path: str | Path | None = None) -> dict[str, Any]:
        """Return read-only lifecycle/session fragmentation diagnostics.

        This intentionally reports mismatches only. It does not infer that every
        mismatch is corrupt, and it never rewrites lifecycle, message, DAG, or
        Hermes host state. Repair/cleanup flows must stay explicit and separate.
        """
        conn = self._conn
        assert conn is not None

        def _count(query: str, params: tuple[Any, ...] = ()) -> int:
            row = conn.execute(query, params).fetchone()
            return int(row[0] if row else 0)

        def _session_ids(query: str) -> set[str]:
            return {
                str(row[0])
                for row in conn.execute(query).fetchall()
                if row[0]
            }

        message_sessions = _session_ids("SELECT DISTINCT session_id FROM messages WHERE session_id IS NOT NULL")
        node_sessions = _session_ids("SELECT DISTINCT session_id FROM summary_nodes WHERE session_id IS NOT NULL")
        lcm_any_sessions = message_sessions | node_sessions
        state_sessions: set[str] = set()
        state_db_read_success = False
        lifecycle_current_sessions = _session_ids(
            "SELECT DISTINCT current_session_id FROM lcm_lifecycle_state WHERE current_session_id IS NOT NULL"
        )
        lifecycle_last_finalized_sessions = _session_ids(
            "SELECT DISTINCT last_finalized_session_id FROM lcm_lifecycle_state WHERE last_finalized_session_id IS NOT NULL"
        )
        lifecycle_referenced_sessions = lifecycle_current_sessions | lifecycle_last_finalized_sessions

        empty_lifecycle_rows = 0
        for row in conn.execute(
            """
            SELECT current_session_id, last_finalized_session_id
            FROM lcm_lifecycle_state
            """
        ).fetchall():
            refs = {
                str(value)
                for value in (row["current_session_id"], row["last_finalized_session_id"])
                if value
            }
            if not refs or refs.isdisjoint(lcm_any_sessions):
                empty_lifecycle_rows += 1

        stats: dict[str, Any] = {
            "read_only": True,
            "lifecycle_rows": _count("SELECT COUNT(*) FROM lcm_lifecycle_state"),
            "empty_lifecycle_rows": empty_lifecycle_rows,
            "messages_total": _count("SELECT COUNT(*) FROM messages"),
            "summary_nodes_total": _count("SELECT COUNT(*) FROM summary_nodes"),
            "distinct_message_sessions": len(message_sessions),
            "distinct_node_sessions": len(node_sessions),
            "distinct_lcm_any_sessions": len(lcm_any_sessions),
            "lifecycle_current_sessions": len(lifecycle_current_sessions),
            "lifecycle_last_finalized_sessions": len(lifecycle_last_finalized_sessions),
            "lifecycle_current_missing_in_messages": len(lifecycle_current_sessions - message_sessions),
            "lifecycle_current_missing_in_nodes": len(lifecycle_current_sessions - node_sessions),
            "lifecycle_current_missing_in_lcm_any": len(lifecycle_current_sessions - lcm_any_sessions),
            "lifecycle_last_finalized_missing_in_messages": len(lifecycle_last_finalized_sessions - message_sessions),
            "lifecycle_last_finalized_missing_in_nodes": len(lifecycle_last_finalized_sessions - node_sessions),
            "lifecycle_last_finalized_missing_in_lcm_any": len(lifecycle_last_finalized_sessions - lcm_any_sessions),
            "message_sessions_without_lifecycle_current": len(message_sessions - lifecycle_current_sessions),
            "message_sessions_without_lifecycle_reference": len(message_sessions - lifecycle_referenced_sessions),
            "node_sessions_without_lifecycle_reference": len(node_sessions - lifecycle_referenced_sessions),
            "state_db_checked": False,
            "state_db_error": "",
            "state_sessions_total": 0,
            "lifecycle_current_missing_in_state": 0,
            "lifecycle_last_finalized_missing_in_state": 0,
            "lcm_message_sessions_missing_in_state": 0,
            "lcm_node_sessions_missing_in_state": 0,
            "state_sessions_missing_in_lcm_messages": 0,
            "state_sessions_missing_in_lcm_any": 0,
        }

        if state_db_path:
            path = Path(state_db_path).expanduser()
            if path.exists():
                stats["state_db_checked"] = True
                try:
                    state_uri = path.resolve().as_uri() + "?mode=ro"
                    state_conn = sqlite3.connect(state_uri, uri=True)
                    try:
                        state_rows = state_conn.execute("SELECT id FROM sessions WHERE id IS NOT NULL").fetchall()
                    finally:
                        state_conn.close()
                    state_sessions = {str(row[0]) for row in state_rows if row[0]}
                    state_db_read_success = True
                    stats.update({
                        "state_sessions_total": len(state_sessions),
                        "lifecycle_current_missing_in_state": len(lifecycle_current_sessions - state_sessions),
                        "lifecycle_last_finalized_missing_in_state": len(
                            lifecycle_last_finalized_sessions - state_sessions
                        ),
                        "lcm_message_sessions_missing_in_state": len(message_sessions - state_sessions),
                        "lcm_node_sessions_missing_in_state": len(node_sessions - state_sessions),
                        "state_sessions_missing_in_lcm_messages": len(state_sessions - message_sessions),
                        "state_sessions_missing_in_lcm_any": len(state_sessions - lcm_any_sessions),
                    })
                except Exception as exc:  # pragma: no cover - defensive
                    stats["state_db_error"] = str(exc)
            else:
                stats["state_db_error"] = f"state database not found: {path}"

        stats["classification"] = self._classify_fragmentation(
            lifecycle_rows=stats["lifecycle_rows"],
            lifecycle_current_sessions=lifecycle_current_sessions,
            lifecycle_last_finalized_sessions=lifecycle_last_finalized_sessions,
            message_sessions=message_sessions,
            node_sessions=node_sessions,
            lcm_any_sessions=lcm_any_sessions,
            lifecycle_referenced_sessions=lifecycle_referenced_sessions,
            state_sessions=state_sessions,
            state_db_read_success=state_db_read_success,
        )

        return stats

    @staticmethod
    def _classify_fragmentation(
        *,
        lifecycle_rows: int,
        lifecycle_current_sessions: set[str],
        lifecycle_last_finalized_sessions: set[str],
        message_sessions: set[str],
        node_sessions: set[str],
        lcm_any_sessions: set[str],
        lifecycle_referenced_sessions: set[str],
        state_sessions: set[str],
        state_db_read_success: bool,
    ) -> dict[str, Any]:
        """Bucket lifecycle mismatches into operator-readable read-only categories."""

        def sample(session_ids: set[str], limit: int = 5) -> list[str]:
            return sorted(session_ids)[:limit]

        categories: list[dict[str, Any]] = []

        def add_category(
            name: str,
            session_ids: set[str],
            *,
            severity: str,
            description: str,
            recommended_action: str,
        ) -> None:
            if not session_ids:
                return
            categories.append({
                "name": name,
                "severity": severity,
                "count": len(session_ids),
                "sample_session_ids": sample(session_ids),
                "description": description,
                "recommended_action": recommended_action,
            })

        add_category(
            "stale_lifecycle_current",
            lifecycle_current_sessions - lcm_any_sessions,
            severity="warn",
            description="Lifecycle current-session references that no longer have raw messages or summary nodes in LCM.",
            recommended_action="Inspect samples before cleanup; these are often old or ephemeral lifecycle rows, not automatic corruption.",
        )
        add_category(
            "stale_lifecycle_finalized",
            lifecycle_last_finalized_sessions - lcm_any_sessions,
            severity="warn",
            description="Lifecycle finalized-session references that no longer have raw messages or summary nodes in LCM.",
            recommended_action="Inspect samples before cleanup; only remove with an explicit backup-first lifecycle cleanup flow.",
        )
        if lifecycle_rows > 0:
            add_category(
                "lcm_message_sessions_without_lifecycle_reference",
                message_sessions - lifecycle_referenced_sessions,
                severity="notice",
                description="Raw-message sessions exist in LCM but are not referenced by current or finalized lifecycle state.",
                recommended_action="Usually safe as historical retained context; investigate only if the sessions should belong to an active conversation.",
            )
            add_category(
                "lcm_node_sessions_without_lifecycle_reference",
                node_sessions - lifecycle_referenced_sessions,
                severity="notice",
                description="Summary-node sessions exist in LCM but are not referenced by current or finalized lifecycle state.",
                recommended_action="Usually safe as historical retained context; verify expand/search still work before considering cleanup.",
            )

        if state_db_read_success:
            add_category(
                "lcm_message_sessions_missing_in_state",
                message_sessions - state_sessions,
                severity="notice",
                description="LCM raw-message sessions are absent from the Hermes session database.",
                recommended_action="Treat as retained or imported context unless the session should still be browsable in host session history.",
            )
            add_category(
                "lcm_node_sessions_missing_in_state",
                node_sessions - state_sessions,
                severity="notice",
                description="LCM summary-node sessions are absent from the Hermes session database.",
                recommended_action="Keep read-only; this can happen after host session pruning while LCM retained summaries remain useful.",
            )
            add_category(
                "state_only_sessions",
                state_sessions - lcm_any_sessions,
                severity="notice",
                description="Hermes host sessions exist without raw messages or summary nodes in LCM.",
                recommended_action="Usually benign for sessions outside LCM scope, ignored sessions, or sessions that never reached durable LCM ingest.",
            )

        warn_count = sum(1 for item in categories if item["severity"] == "warn")
        status = "warn" if warn_count else ("notice" if categories else "pass")
        summary = (
            "no lifecycle fragmentation categories detected"
            if not categories
            else f"{len(categories)} lifecycle fragmentation categories need review"
        )
        return {
            "read_only": True,
            "status": status,
            "summary": summary,
            "categories": categories,
        }

    @_synchronized
    def record_debt(
        self,
        conversation_id: str | None,
        *,
        kind: str,
        size_estimate: int,
    ) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET debt_kind = ?,
                debt_size_estimate = ?,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (kind, max(0, int(size_estimate or 0)), now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    def clear_debt(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET debt_kind = NULL,
                debt_size_estimate = 0,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def record_maintenance_attempt(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET last_maintenance_attempt_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def record_reset(self, conversation_id: str | None) -> LifecycleState | None:
        if not conversation_id:
            return None
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
        now = time.time()
        self._conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET last_reset_at = ?,
                debt_kind = NULL,
                debt_size_estimate = 0,
                debt_updated_at = ?,
                updated_at = ?
            WHERE conversation_id = ?
            """,
            (now, now, now, conversation_id),
        )
        self._conn.commit()
        return self.get_by_conversation(conversation_id)

    @_synchronized
    def prune_empty_sessions(
        self,
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        max_age_hours: float | None = None,
    ) -> int:
        """Delete lifecycle rows for sessions with no stored data.

        A row is eligible when BOTH referenced session IDs
        (``current_session_id`` and ``last_finalized_session_id``)
        have zero messages AND zero summary_nodes in the main store.

        Only the lifecycle table is modified — messages, nodes, and FTS
        indexes are untouched (they already contain no data for these sessions).

        Args:
            protected_session_ids: Sessions that must never be deleted
                (typically the actively-bound engine session).
            max_age_hours: Only delete rows older than this many hours.
                ``None`` means delete all eligible rows regardless of age.

        Returns:
            Number of rows deleted.
        """
        conn = self._conn
        assert conn is not None
        protected = {str(s) for s in (protected_session_ids or ()) if s}

        conn.execute("BEGIN IMMEDIATE")
        try:
            sessions_with_data: set[str] = set()
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            def _session_has_data(session_id: str) -> bool:
                if not session_id:
                    return False
                if "messages" in tables and conn.execute(
                    "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone():
                    return True
                if "summary_nodes" in tables and conn.execute(
                    "SELECT 1 FROM summary_nodes WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone():
                    return True
                return False

            if "messages" in tables:
                for row in conn.execute(
                    "SELECT DISTINCT session_id FROM messages"
                ).fetchall():
                    sessions_with_data.add(str(row[0]))
            if "summary_nodes" in tables:
                for row in conn.execute(
                    "SELECT DISTINCT session_id FROM summary_nodes"
                ).fetchall():
                    sessions_with_data.add(str(row[0]))

            now = time.time()
            max_age_seconds = (
                float(max_age_hours) * 3600.0
                if max_age_hours is not None
                else None
            )
            deleted = 0

            rows = conn.execute(
                "SELECT * FROM lcm_lifecycle_state"
            ).fetchall()
            for row in rows:
                cur = str(row["current_session_id"] or "")
                fin = str(row["last_finalized_session_id"] or "")

                if ((cur and cur in sessions_with_data)
                        or (fin and fin in sessions_with_data)):
                    continue

                refs = {r for r in (cur, fin) if r}
                if refs & protected:
                    continue

                if max_age_seconds is not None:
                    row_age = (
                        row["current_bound_at"]
                        or row["last_finalized_at"]
                        or row["updated_at"]
                    )
                    if row_age is not None and (now - float(row_age)) < max_age_seconds:
                        continue

                # Recheck against the tables right before deletion. BEGIN
                # IMMEDIATE blocks concurrent writers while this transaction is
                # open; this fresh query also keeps the safety check honest if
                # the broad snapshot logic above changes later.
                if _session_has_data(cur) or _session_has_data(fin):
                    continue

                conn.execute(
                    "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
                    (row["conversation_id"],),
                )
                deleted += 1

            if deleted:
                conn.commit()
            else:
                conn.rollback()
            return deleted
        except Exception:
            conn.rollback()
            raise

    def delete_safe_rows_for_sessions(
        self,
        session_ids: set[str] | list[str] | tuple[str, ...],
        *,
        protected_session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> tuple[int, int]:
        candidates = {str(s) for s in session_ids if s}
        if not candidates:
            return 0, 0
        protected = {str(s) for s in (protected_session_ids or ()) if s}
        deleted = 0
        skipped = 0
        rows = self._conn.execute("SELECT * FROM lcm_lifecycle_state").fetchall()
        for row in rows:
            refs = {
                str(value)
                for value in (row["current_session_id"], row["last_finalized_session_id"])
                if value
            }
            if not refs or not (refs & candidates):
                continue
            if refs & protected:
                skipped += 1
                continue
            if refs <= candidates:
                self._conn.execute(
                    "DELETE FROM lcm_lifecycle_state WHERE conversation_id = ?",
                    (row["conversation_id"],),
                )
                deleted += 1
                continue
            skipped += 1
        if deleted:
            self._conn.commit()
        return deleted, skipped

    def advance_frontier(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int,
    ) -> LifecycleState | None:
        if not conversation_id:
            return None
        with self._lock:
            state = self.get_by_conversation(conversation_id)
            if state is None or state.current_session_id != session_id:
                return state
            now = time.time()
            conn = self._conn
            assert conn is not None
            # MAX() in SQL keeps the advance monotonic even if a concurrent
            # writer bumped the frontier between the read above and this write.
            # A Python-side max() over the stale read could otherwise regress
            # the checkpoint and force the same range to be compacted twice.
            cursor = conn.execute(
                """
                UPDATE lcm_lifecycle_state
                SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                    updated_at = ?
                WHERE conversation_id = ? AND current_session_id = ?
                """,
                (int(frontier_store_id or 0), now, conversation_id, session_id),
            )
            if cursor.rowcount == 0:
                return self.get_by_conversation(conversation_id)
            conn.commit()
            return self.get_by_conversation(conversation_id)

    @staticmethod
    def stage_frontier_advance(
        conn: sqlite3.Connection,
        conversation_id: str,
        session_id: str,
        frontier_store_id: int,
    ) -> None:
        """Stage a monotonic frontier advance in a caller-owned transaction.

        This intentionally uses the publishing DAG connection rather than the
        lifecycle connection or ``self._lock``. SQLite serializes the whole
        node/frontier write transaction; the conditional UPDATE linearizes it
        against bind/rollover writers, while MAX keeps same-session advances
        monotonic.
        """
        cursor = conn.execute(
            """
            UPDATE lcm_lifecycle_state
            SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                updated_at = ?
            WHERE conversation_id = ? AND current_session_id = ?
            """,
            (
                int(frontier_store_id or 0),
                time.time(),
                conversation_id,
                session_id,
            ),
        )
        if cursor.rowcount == 1:
            return
        row = conn.execute(
            "SELECT current_session_id FROM lcm_lifecycle_state "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                "Cannot publish summary without lifecycle state"
            )
        raise LifecycleBindingChangedError(
            "Lifecycle session binding changed during summary publication"
        )

    def stage_compaction_publication(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        session_id: str,
        publication_node_id: int,
        expected_frontier_store_id: int,
        covered_store_ids: list[int],
        excluded_store_ids: list[int] | None = None,
        filter_exclusion_proofs: dict[int, Any] | None = None,
        selected_retained_root_node_ids: list[int] | None = None,
    ) -> int:
        """Validate and stage one contiguous compaction-frontier advance.

        The caller owns the node/frontier transaction. The expected frontier is
        its optimistic publication generation.
        """
        expected_frontier = max(0, int(expected_frontier_store_id or 0))
        all_covered_ids = sorted(
            {int(store_id) for store_id in covered_store_ids if int(store_id) > 0}
        )
        if not all_covered_ids:
            raise LifecyclePublicationConflictError(
                "Compaction publication has no durable source coverage"
            )
        covered_end = max(expected_frontier, all_covered_ids[-1])
        covered_ids = [
            store_id
            for store_id in all_covered_ids
            if store_id > expected_frontier
        ]
        excluded_ids = sorted(
            {
                int(store_id)
                for store_id in (excluded_store_ids or [])
                if expected_frontier < int(store_id) <= covered_end
            }
        )
        if set(covered_ids) & set(excluded_ids):
            raise LifecyclePublicationConflictError(
                "Compaction publication coverage overlaps an explicit exclusion"
            )
        exclusion_proofs = filter_exclusion_proofs or {}
        snapshot_ids = sorted(
            set(all_covered_ids + excluded_ids + list(exclusion_proofs))
        )
        state_row = conn.execute(
            """
            SELECT current_session_id, last_finalized_session_id,
                   current_frontier_store_id
            FROM lcm_lifecycle_state
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if state_row is None:
            raise RuntimeError("Cannot publish summary without lifecycle state")
        if str(state_row[0] or "") != session_id:
            raise LifecycleBindingChangedError(
                "Lifecycle session binding changed during summary publication"
            )
        actual_frontier = int(state_row[2] or 0)
        if actual_frontier != expected_frontier:
            raise LifecyclePublicationConflictError(
                "Compaction publication frontier generation changed "
                f"(expected={expected_frontier}, actual={actual_frontier})"
            )
        candidate_legacy_session_ids = {
            str(value or "")
            for value in (session_id, state_row[0], state_row[1])
            if value
        }
        legacy_session_ids = unambiguous_legacy_session_ids(
            conn,
            conversation_id,
            candidate_legacy_session_ids,
        )

        def belongs_to_conversation(row: sqlite3.Row | tuple[Any, ...]) -> bool:
            row_session_id = str(row[1] or "")
            row_conversation_id = str(row[2] or "").strip(_LEGACY_ASCII_WHITESPACE)
            return row_conversation_id == conversation_id or (
                not row_conversation_id and row_session_id in legacy_session_ids
            )

        snapshot_rows = conn.execute(
            """
            SELECT m.store_id, m.session_id, m.conversation_id, m.content
            FROM json_each(?) AS source
            CROSS JOIN messages AS m
            WHERE m.store_id = source.value
            ORDER BY m.store_id
            """,
            (json.dumps(snapshot_ids),),
        ).fetchall()
        if [int(row[0]) for row in snapshot_rows] != snapshot_ids:
            raise LifecyclePublicationConflictError(
                "Compaction publication source ownership changed"
            )
        if any(not belongs_to_conversation(row) for row in snapshot_rows):
            raise LifecyclePublicationConflictError(
                "Compaction publication source ownership changed"
            )
        snapshot_by_id = {int(row[0]): row for row in snapshot_rows}
        if any(
            snapshot_by_id[store_id][3] != proof
            for store_id, proof in exclusion_proofs.items()
        ):
            raise LifecyclePublicationConflictError(
                "Compaction publication filter exclusion changed"
            )
        rows = conn.execute(
            f"""
            SELECT store_id, session_id, conversation_id, content
            FROM messages
            WHERE store_id > ? AND store_id <= ?
              AND (conversation_id = ? OR
                   ({legacy_blank_clause('conversation_id')} AND session_id IN
                    (SELECT value FROM json_each(?))))
            ORDER BY store_id
            """,
            (expected_frontier, covered_end, conversation_id,
             json.dumps(sorted(legacy_session_ids))),
        ).fetchall()
        authoritative_ids = [
            int(row[0]) for row in rows if belongs_to_conversation(row)
        ]
        # Exclusions may precede or interrupt the retained prefix; they do not
        # move the boundary between retained and newly covered source rows.
        excluded = set(excluded_ids)
        authoritative_ids = [value for value in authoritative_ids if value not in excluded]
        proven_ids = covered_ids
        first_new_index = (
            authoritative_ids.index(proven_ids[0])
            if proven_ids and proven_ids[0] in authoritative_ids
            else -1
        )
        # Claiming a previously unclaimed source below the frontier does not
        # cross any new interval. Full-source ownership and duplicate checks
        # below still apply, and the frontier remains unchanged.
        if not authoritative_ids and not proven_ids:
            first_new_index = 0
        retained_prefix_ids = (
            authoritative_ids[:first_new_index]
            if first_new_index >= 0
            else []
        )
        if first_new_index < 0 or authoritative_ids[first_new_index:] != proven_ids:
            raise LifecyclePublicationConflictError(
                "Compaction publication source coverage is not contiguous "
                f"(expected_frontier={expected_frontier}, "
                f"authoritative={authoritative_ids}, covered={covered_ids}, "
                f"excluded={excluded_ids})"
            )
        duplicate_node_session_ids = sorted(
            legacy_session_ids
            | {session_id}
            | {str(row[1] or "") for row in snapshot_rows if str(row[1] or "")}
        )
        if conn.execute(
            """
            SELECT 1
            FROM summary_nodes AS node INDEXED BY idx_nodes_session_latest,
                 json_each(node.source_ids) AS source
            WHERE node.source_type = 'messages' AND node.node_id != ?
              AND node.session_id IN (SELECT value FROM json_each(?))
              AND source.value IN (SELECT value FROM json_each(?))
            LIMIT 1
            """,
            (
                publication_node_id,
                json.dumps(duplicate_node_session_ids),
                json.dumps(all_covered_ids),
            ),
        ).fetchone():
            raise LifecyclePublicationConflictError(
                "Compaction publication source lineage is already claimed"
            )
        admitted = admitted_summary_roots(conn, conversation_id, session_id,
                                          before_node_id=publication_node_id)
        selected = set(selected_retained_root_node_ids or [])
        represented = {source for node_id in selected for source in admitted.get(node_id, set())}
        if not selected <= admitted.keys() or not set(retained_prefix_ids) <= represented:
            raise LifecyclePublicationConflictError(
                "Compaction publication retained summary lineage is not admitted"
            )
        self.stage_frontier_advance(
            conn,
            conversation_id,
            session_id,
            covered_end,
        )
        return covered_end
