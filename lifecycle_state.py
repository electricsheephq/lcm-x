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
    def bind_session(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> LifecycleState:
        existing = self.get_by_conversation(conversation_id) if conversation_id else self.get_by_session(session_id)
        conversation_id = conversation_id or (existing.conversation_id if existing else session_id)
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
            # A gateway restart can re-admit the exact session that was last
            # finalized after the active binding was cleared.  That session
            # already owns the persisted finalized checkpoint, so restoring
            # it is safe and monotonic.  A different session must start at
            # zero until a proven rollover edge advances it.
            same_finalized_session = (
                existing.current_session_id is None
                and existing.last_finalized_session_id == session_id
            )
            current_frontier = (
                max(
                    existing.current_frontier_store_id,
                    existing.last_finalized_frontier_store_id,
                )
                if same_finalized_session
                else 0
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
    def finalize_session(
        self,
        conversation_id: str | None,
        session_id: str,
        frontier_store_id: int = 0,
    ) -> LifecycleState | None:
        state = self.get_by_conversation(conversation_id)
        if state is None:
            return None
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
    def record_rollover(
        self,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        finalized_frontier_store_id: int = 0,
    ) -> LifecycleState:
        state = self.get_by_conversation(conversation_id)
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
                last_reset_at = lcm_lifecycle_state.last_reset_at,
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

    @_synchronized
    def stage_rollover(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        *,
        old_session_id: str,
        new_session_id: str,
        finalized_frontier_store_id: int = 0,
    ) -> None:
        """Stage finalization and rebinding on a caller-owned SQLite transaction.

        The DAG connection owns the transaction when rollover also reassigns
        retained nodes.  This method deliberately performs no commit, allowing
        lifecycle state and node ownership to roll back together on any fault.
        """
        row = conn.execute(
            """
            SELECT current_session_id, last_finalized_session_id,
                   last_finalized_frontier_store_id
            FROM lcm_lifecycle_state
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is not None and str(row[0] or "") == new_session_id and str(row[1] or "") == old_session_id:
            return

        if row is not None:
            current_session_id = str(row[0] or "")
            last_finalized_session_id = str(row[1] or "")
            # A rollover callback may be replayed after a gateway restart, but
            # it must never replace a newer active binding.  The only safe
            # states are the expected old session or its already-finalized
            # form; an absent row remains insertable below.
            if not (
                current_session_id == old_session_id
                or (
                    not current_session_id
                    and last_finalized_session_id == old_session_id
                )
            ):
                raise LifecyclePublicationConflictError(
                    "Lifecycle rollover binding changed before staging "
                    f"(expected current={old_session_id!r}, "
                    f"actual current={current_session_id!r}, "
                    f"finalized={last_finalized_session_id!r})"
                )

        last_finalized_frontier = max(
            int(finalized_frontier_store_id or 0),
            int(row[2] or 0) if row is not None else 0,
        )
        now = time.time()
        conn.execute(
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
                last_reset_at = lcm_lifecycle_state.last_reset_at,
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

    @staticmethod
    def _conversation_owner_session_ids(
        conn: sqlite3.Connection,
        conversation_id: str,
        session_id: str,
    ) -> set[str]:
        """Resolve proven producing sessions for one logical conversation.

        The lifecycle binding contributes the current and last-finalized
        sessions. Explicit conversation ids on durable message rows extend
        that set for older producing sessions; blank legacy conversation ids
        are admitted only for the already-proven lifecycle sessions.
        """
        owner_sessions = {str(session_id or "")}
        state = conn.execute(
            """
            SELECT current_session_id, last_finalized_session_id
            FROM lcm_lifecycle_state
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if state is not None:
            owner_sessions.update(
                str(value or "")
                for value in (state[0], state[1])
                if value
            )
        rows = conn.execute(
            """
            SELECT DISTINCT session_id
            FROM messages
            WHERE conversation_id = ? AND session_id IS NOT NULL AND session_id != ''
            """,
            (conversation_id,),
        ).fetchall()
        owner_sessions.update(str(row[0]) for row in rows if row[0])
        owner_sessions.discard("")
        return owner_sessions

    @staticmethod
    def audit_conversation_ownership(
        conn: sqlite3.Connection,
        conversation_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Audit message ownership before a conversation is admitted.

        Durable rows with an explicit conversation id are authoritative.  A
        blank legacy id is admissible only when its producing session is the
        current or last-finalized lifecycle binding for this conversation.
        This read-only audit deliberately does not inspect ``summary_nodes``;
        publication validates only the bounded owner-session scope below.
        """
        conversation_id = str(conversation_id or "").strip()
        session_id = str(session_id or "").strip()
        if not conversation_id:
            return {"ok": bool(session_id), "reason": "missing conversation id"}

        target_state = conn.execute(
            """
            SELECT conversation_id, current_session_id, last_finalized_session_id,
                   current_frontier_store_id, last_finalized_frontier_store_id,
                   debt_kind, debt_size_estimate
            FROM lcm_lifecycle_state
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        explicit_rows = conn.execute(
            """
            SELECT DISTINCT session_id
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchall()
        explicit_sessions = {
            str(row[0] or "").strip() for row in explicit_rows if str(row[0] or "").strip()
        }
        referenced_sessions = set(explicit_sessions)
        if target_state is not None:
            referenced_sessions.update(
                str(value or "").strip()
                for value in (target_state[1], target_state[2])
                if str(value or "").strip()
            )
        if session_id:
            referenced_sessions.add(session_id)
        if referenced_sessions:
            placeholders = ",".join("?" for _ in referenced_sessions)
            lifecycle_rows = conn.execute(
                f"""
                SELECT conversation_id, current_session_id, last_finalized_session_id,
                       current_frontier_store_id, last_finalized_frontier_store_id,
                       debt_kind, debt_size_estimate
                FROM lcm_lifecycle_state
                WHERE conversation_id = ?
                   OR current_session_id IN ({placeholders})
                   OR last_finalized_session_id IN ({placeholders})
                """,
                (
                    conversation_id,
                    *sorted(referenced_sessions),
                    *sorted(referenced_sessions),
                ),
            ).fetchall()
        else:
            lifecycle_rows = conn.execute(
                """
                SELECT conversation_id, current_session_id, last_finalized_session_id,
                       current_frontier_store_id, last_finalized_frontier_store_id,
                       debt_kind, debt_size_estimate
                FROM lcm_lifecycle_state
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchall()

        def has_ownership_evidence(row: sqlite3.Row | tuple[Any, ...]) -> bool:
            row_conversation = str(row[0] or "").strip()
            if row_conversation == conversation_id:
                return True
            if (
                int(row[3] or 0) > 0
                or int(row[4] or 0) > 0
                or str(row[5] or "").strip()
                or int(row[6] or 0) > 0
            ):
                return True
            if conn.execute(
                "SELECT 1 FROM messages WHERE conversation_id = ? LIMIT 1",
                (row_conversation,),
            ).fetchone():
                return True
            current_session = str(row[1] or "").strip()
            if not current_session:
                # A row with no current binding cannot prove that its shared
                # last-finalized reference is an empty alias.
                return True
            if conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                (current_session,),
            ).fetchone():
                return True
            if conn.execute(
                "SELECT 1 FROM summary_nodes WHERE session_id = ? LIMIT 1",
                (current_session,),
            ).fetchone():
                return True
            return False

        by_session: dict[str, set[str]] = {}
        non_owning_aliases: list[str] = []
        for row in lifecycle_rows:
            if not has_ownership_evidence(row):
                non_owning_aliases.append(str(row[0] or "").strip())
                continue
            row_conversation = str(row[0] or "").strip()
            for value in (row[1], row[2]):
                value = str(value or "").strip()
                if value:
                    by_session.setdefault(value, set()).add(row_conversation)

        state = target_state
        proven_sessions: set[str] = set()
        if state is not None:
            proven_sessions = {
                str(value or "").strip()
                for value in (state[1], state[2])
                if str(value or "").strip()
            }
        candidate_sessions = set(proven_sessions) | explicit_sessions
        if session_id:
            candidate_sessions.add(session_id)
        orphan_legacy_sessions: list[str] = []
        if candidate_sessions:
            placeholders = ",".join("?" for _ in candidate_sessions)
            legacy_rows = conn.execute(
                f"""
                SELECT DISTINCT session_id
                FROM messages
                WHERE (conversation_id IS NULL OR conversation_id = '')
                  AND session_id IN ({placeholders})
                """,
                tuple(sorted(candidate_sessions)),
            ).fetchall()
            orphan_legacy_sessions = sorted(
                {
                    str(row[0] or "").strip()
                    for row in legacy_rows
                    if str(row[0] or "").strip()
                    and str(row[0] or "").strip() not in proven_sessions
                }
            )
        ambiguous_sessions = sorted(
            session for session, conversations in by_session.items() if len(conversations) > 1
        )
        ok = not ambiguous_sessions and not orphan_legacy_sessions
        return {
            "ok": ok,
            "conversation_id": conversation_id,
            "session_id": session_id,
            "proven_sessions": sorted(proven_sessions),
            "explicit_sessions": sorted(explicit_sessions),
            "ambiguous_sessions": ambiguous_sessions,
            "orphan_legacy_sessions": orphan_legacy_sessions,
            "non_owning_aliases": sorted(non_owning_aliases),
            "reason": "" if ok else "ambiguous or orphan legacy message ownership",
        }

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
    ) -> int:
        """Validate and stage one contiguous compaction-frontier advance.

        The caller owns the node/frontier transaction. The expected frontier is
        its optimistic publication generation.
        """
        ownership = self.audit_conversation_ownership(
            conn,
            conversation_id,
            session_id,
        )
        if not ownership.get("ok"):
            raise LifecyclePublicationConflictError(
                "Compaction publication source ownership is ambiguous"
            )
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
        owner_session_ids = self._conversation_owner_session_ids(
            conn,
            conversation_id,
            session_id,
        )
        snapshot_ids = sorted(
            set(all_covered_ids + excluded_ids + list(exclusion_proofs))
        )
        snapshot_rows = conn.execute(
            """
            SELECT m.store_id, m.session_id, m.conversation_id, m.content
            FROM json_each(?) AS source
            CROSS JOIN messages AS m
            WHERE m.store_id = source.value
            ORDER BY m.store_id
            """,
            (str(snapshot_ids),),
        ).fetchall()
        if [int(row[0]) for row in snapshot_rows] != snapshot_ids:
            raise LifecyclePublicationConflictError(
                "Compaction publication source ownership changed"
            )

        def belongs_to_conversation(row: sqlite3.Row | tuple[Any, ...]) -> bool:
            row_session_id = str(row[1] or "")
            row_conversation_id = str(row[2] or "").strip()
            return row_conversation_id == conversation_id or (
                not row_conversation_id and row_session_id in owner_session_ids
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
            """
            SELECT store_id, session_id, conversation_id, content
            FROM messages
            WHERE store_id > ? AND store_id <= ?
            ORDER BY store_id
            """,
            (
                expected_frontier,
                covered_end,
            ),
        ).fetchall()
        if any(
            str(row[1] or "") in owner_session_ids
            and str(row[2] or "").strip() not in {"", conversation_id}
            for row in rows
        ):
            raise LifecyclePublicationConflictError(
                "Compaction publication source ownership changed"
            )
        authoritative_ids = [
            int(row[0]) for row in rows if belongs_to_conversation(row)
        ]
        proven_ids = sorted(covered_ids + excluded_ids)
        if authoritative_ids != proven_ids:
            raise LifecyclePublicationConflictError(
                "Compaction publication source coverage is not contiguous "
                f"(expected_frontier={expected_frontier}, "
                f"authoritative={authoritative_ids}, covered={covered_ids}, "
                f"excluded={excluded_ids})"
            )
        if owner_session_ids:
            owner_placeholders = ",".join("?" for _ in owner_session_ids)
            if conn.execute(
                f"""
                SELECT 1
                FROM summary_nodes AS node, json_each(node.source_ids) AS source
                WHERE node.session_id IN ({owner_placeholders})
                  AND node.source_type = 'messages'
                  AND node.node_id != ?
                  AND source.value IN (SELECT value FROM json_each(?))
                LIMIT 1
                """,
                (*sorted(owner_session_ids), publication_node_id, json.dumps(all_covered_ids)),
            ).fetchone():
                raise LifecyclePublicationConflictError(
                    "Compaction publication source lineage is already claimed"
                )
        row = conn.execute(
            """
            SELECT current_session_id, current_frontier_store_id
            FROM lcm_lifecycle_state
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Cannot publish summary without lifecycle state")
        if str(row[0] or "") != session_id:
            raise LifecycleBindingChangedError(
                "Lifecycle session binding changed during summary publication"
            )
        actual_frontier = int(row[1] or 0)
        if actual_frontier != expected_frontier:
            raise LifecyclePublicationConflictError(
                "Compaction publication frontier generation changed "
                f"(expected={expected_frontier}, actual={actual_frontier})"
            )
        self.stage_frontier_advance(
            conn,
            conversation_id,
            session_id,
            covered_end,
        )
        return covered_end
