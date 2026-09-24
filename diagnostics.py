"""Shared read-only diagnostic helpers for LCM tools and commands."""

from __future__ import annotations

import hashlib
import os
from collections import deque
from pathlib import Path
from typing import Any


DOCTOR_ACTION_SAFE_IGNORE = "safe/ignore"
DOCTOR_ACTION_INSPECT = "inspect"
DOCTOR_ACTION_BACKUP_FIRST_CLEANUP = "backup-first cleanup"


def _enforce_state_db_containment(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    env_base = os.environ.get("LCM_HERMES_BASE_DIR")
    if env_base:
        allowed_base = Path(env_base).expanduser().resolve()
        try:
            resolved.relative_to(allowed_base)
        except ValueError:
            raise ValueError(
                f"{description} resolves to {resolved} which is not within allowed base {allowed_base}"
            )
    return resolved


def state_db_path_for_engine(engine: Any) -> Path:
    """Return the Hermes state database path for an LCM engine.

    The path is read-only diagnostic input. When ``LCM_HERMES_BASE_DIR`` is
    configured, enforce the same containment guard for all diagnostic surfaces.
    """
    hermes_home = getattr(engine, "_hermes_home", "") or ""
    if hermes_home:
        return _enforce_state_db_containment(
            Path(hermes_home) / "state.db",
            description=f"hermes_home {hermes_home}",
        )
    db_path = Path(getattr(engine._store, "db_path", Path.home() / ".hermes" / "lcm.db"))
    return _enforce_state_db_containment(
        db_path.parent / "state.db",
        description=f"state database fallback from LCM database {db_path}",
    )


def has_lifecycle_fragmentation(stats: dict[str, Any]) -> bool:
    """Return whether lifecycle diagnostics should be treated as warning evidence.

    Retained-history drift is intentionally read-only diagnostic context. Keep the
    doctor warning for concrete operator action (empty lifecycle rows that the
    explicit backup-first cleanup path can prune) or diagnostic unreadability, but
    do not make overall health unhealthy solely because historical LCM/state
    indexes no longer agree.
    """
    empty_lifecycle_rows = int(stats.get("empty_lifecycle_rows", 0) or 0)
    return empty_lifecycle_rows > 0 or (
        bool(stats.get("state_db_checked")) and bool(stats.get("state_db_error"))
    )


COMPACTION_REPLAY_MIN_RUN = 3
# A #483 replay re-stores rows that were already in the host's active list when
# it compacted, so the original lies at most one active window (fresh tail plus
# the turns since the last compaction) behind the replay. 2,000 rows is far
# above any active window a model context holds, and bounds the scan's memory.
COMPACTION_REPLAY_WINDOW = 2000
COMPACTION_REPLAY_MAX_ROWS = 1_000_000
COMPACTION_REPLAY_DUPLICATES_ACTION = (
    "candidate replay runs (repeated rows, likely from a pre-fix compaction replay, #483) are present; "
    "recall may show repeats; a repair tool is tracked in a follow-up; do not delete rows by hand"
)
COMPACTION_REPLAY_SCAN_INCOMPLETE = "scan incomplete"
_COMPACTION_REPLAY_CANDIDATES_PER_KEY = 8


def scan_compaction_replay_duplicates(
    conn: Any,
    *,
    sample_limit: int = 10,
    window: int = COMPACTION_REPLAY_WINDOW,
    max_rows: int = COMPACTION_REPLAY_MAX_ROWS,
) -> dict[str, Any]:
    """Count candidate replay runs (#483 class) per session. Read-only; never mutates.

    A stale compaction-commit ingest re-stores an already durable run of rows:
    a contiguous run of at least ``COMPACTION_REPLAY_MIN_RUN`` rows that repeats,
    in order, an earlier run of the same session within ``window`` rows. Lone
    organic repeats ("ok", "continue") are not counted. Rows are streamed and at
    most ``window`` of them are tracked per session; the scan stops after
    ``max_rows`` rows and reports ``scan_truncated``. ``window_limited_sessions``
    counts sessions longer than the window; with either bound hit, a zero count
    covers only the scanned rows (``scan_complete`` is False).
    """
    sessions: dict[str, dict[str, int]] = {}
    counter: _ReplayRunCounter | None = None
    current_session: str | None = None
    rows_scanned = 0
    peak_tracked_rows = 0
    truncated = False
    window_limited_sessions = 0

    def flush() -> None:
        nonlocal window_limited_sessions
        if counter is not None and current_session is not None:
            replayed, runs = counter.finish()
            if replayed:
                sessions[current_session] = {"replayed_rows": replayed, "runs": runs}
            window_limited_sessions += int(counter.base > 0)

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(messages)")}
    # Assistant tool-call rows carry empty content; their calls are the row.
    tool_calls_column = "tool_calls" if "tool_calls" in columns else "''"
    for session_id, role, content, tool_call_id, tool_calls in conn.execute(
        "SELECT session_id, role, content, tool_call_id, "
        f"{tool_calls_column} FROM messages ORDER BY session_id, store_id"
    ):
        if rows_scanned >= max_rows:
            truncated = True
            break
        rows_scanned += 1
        if session_id != current_session:
            flush()
            current_session = session_id
            counter = _ReplayRunCounter(window)
        assert counter is not None
        counter.push(
            hashlib.sha1(
                f"{role}\0{content or ''}\0{tool_call_id or ''}\0{tool_calls or ''}".encode(
                    "utf-8", "surrogatepass"
                )
            ).digest()
        )
        peak_tracked_rows = max(peak_tracked_rows, counter.tracked_rows)
    flush()
    ranked = sorted(sessions.items(), key=lambda item: item[1]["replayed_rows"], reverse=True)
    return {
        "sessions_with_replayed_runs": len(sessions),
        "replayed_rows_total": sum(item["replayed_rows"] for item in sessions.values()),
        "min_run": COMPACTION_REPLAY_MIN_RUN,
        "window": window,
        "rows_scanned": rows_scanned,
        "scan_truncated": truncated,
        "window_limited_sessions": window_limited_sessions,
        "scan_complete": not truncated and not window_limited_sessions,
        "peak_tracked_rows": peak_tracked_rows,
        "sessions": [{"session_id": sid, **counts} for sid, counts in ranked[:sample_limit]],
    }


class _ReplayRunCounter:
    """Streaming count of replayed runs in one session, over a bounded window.

    Greedy, left to right: a replay starts at a row whose key appeared earlier in
    the window, and extends while the rows keep matching the earlier run in order
    (the earlier run must end before the replay starts). A run of at least
    ``COMPACTION_REPLAY_MIN_RUN`` rows counts; a shorter one is re-scanned from its
    second row.
    """

    def __init__(self, window: int) -> None:
        self.window = max(1, int(window))
        self.keys: deque[bytes] = deque()
        self.base = 0
        self.positions: dict[bytes, deque[int]] = {}
        self.registered = -1
        self.candidates: set[int] = set()
        self.run = 0
        self.start = 0
        self.replayed = 0
        self.runs = 0

    @property
    def tracked_rows(self) -> int:
        return len(self.keys)

    def _key_at(self, pos: int) -> bytes | None:
        if pos < self.base or pos >= self.base + len(self.keys):
            return None
        return self.keys[pos - self.base]

    def push(self, key: bytes) -> None:
        self.keys.append(key)
        pos = self.base + len(self.keys) - 1
        self._step(pos)
        while len(self.keys) > self.window:
            old = self.keys.popleft()
            recent = self.positions[old]
            recent.popleft()
            if not recent:
                del self.positions[old]
            self.base += 1

    def _register(self, pos: int) -> None:
        if pos <= self.registered:
            return
        self.registered = pos
        # Every tracked row sits in exactly one deque, so these hold <= window rows.
        self.positions.setdefault(self.keys[pos - self.base], deque()).append(pos)

    def _step(self, pos: int) -> None:
        key = self._key_at(pos)
        if self.candidates:
            continued = {
                offset
                for offset in self.candidates
                if pos - offset < self.start and self._key_at(pos - offset) == key
            }
            if continued:
                self.candidates = continued
                self.run += 1
                self._register(pos)
                return
            if self._close(pos):
                # Rows before pos were re-scanned; pos itself is still pending.
                self._step(pos)
                return
        self._begin(pos)

    def _begin(self, pos: int) -> None:
        key = self._key_at(pos)
        earlier: list[int] = []
        for origin in reversed(self.positions.get(key, ()) if key is not None else ()):
            if origin < pos:
                earlier.append(origin)
                if len(earlier) == _COMPACTION_REPLAY_CANDIDATES_PER_KEY:
                    break
        self.candidates = {pos - origin for origin in earlier}
        self.run = 1 if self.candidates else 0
        self.start = pos
        self._register(pos)

    def _close(self, end: int) -> bool:
        """End the current run before ``end``; True when rows were re-scanned."""
        run, start = self.run, self.start
        self.candidates = set()
        self.run = 0
        if run >= COMPACTION_REPLAY_MIN_RUN:
            self.replayed += run
            self.runs += 1
            return False
        for rescan in range(start + 1, end):
            self._step(rescan)
        return True

    def finish(self) -> tuple[int, int]:
        end = self.base + len(self.keys)
        while self.candidates:
            self._close(end)
        return self.replayed, self.runs


def doctor_guidance_for_check(check: dict[str, Any]) -> dict[str, Any] | None:
    """Return operator triage guidance for one lcm_doctor check.

    Guidance is deliberately conservative: most warning classes are inspect-only
    evidence, and any mutation path is framed as preview/backup/apply rather than
    implied automatic cleanup.
    """
    status = str(check.get("status") or "")
    if status not in {"warn", "fail"}:
        return None

    name = str(check.get("check") or "unknown")
    detail = check.get("detail")
    action = DOCTOR_ACTION_INSPECT
    command = "inspect the reported detail and confirm the active HERMES_HOME/LCM_DATABASE_PATH"
    warning_only = False
    rationale = "operator review required before changing persisted LCM state"

    if name == "database_integrity":
        command = "stop and inspect the SQLite database path; restore from backup if integrity_check is not ok"
    elif name == "schema_core_tables":
        command = "verify HERMES_HOME/LCM_DATABASE_PATH points at the intended LCM database before repair or restore"
    elif name in {"messages_fts_integrity", "nodes_fts_integrity", "fts_index_sync"}:
        if status == "warn" and isinstance(detail, dict) and detail.get("status") == "unchecked":
            action = DOCTOR_ACTION_INSPECT
            command = "rerun `/lcm doctor` with read-write SQLite access if a deep FTS integrity result is needed"
            warning_only = True
            rationale = "the deep FTS check could not run, but this is not evidence that the index is corrupt"
        else:
            action = DOCTOR_ACTION_BACKUP_FIRST_CLEANUP
            command = "run `/lcm doctor repair` first; if it still recommends repair, run `/lcm backup` before `/lcm doctor repair apply`"
            rationale = "FTS repair is rebuildable, but it still mutates SQLite indexes"
    elif name == "sqlite_storage":
        command = "inspect journal/quick_check output and database/WAL size; restore from backup if SQLite reports corruption"
    elif name == "payload_storage":
        missing_refs = 0
        heartbeat_rows = 0
        suspicious_rows = 0
        if isinstance(detail, dict):
            missing_refs = int(detail.get("externalized_payload_refs_missing", 0) or 0)
            heartbeat_rows = len(detail.get("heartbeat_noise_rows") or [])
            suspicious_rows = sum(
                len(detail.get(key) or [])
                for key in (
                    "suspicious_data_uri_content_rows",
                    "suspicious_data_uri_tool_calls_rows",
                    "suspicious_base64_like_rows",
                    "suspicious_repetitive_assistant_rows",
                )
            )
        if status == "warn" and heartbeat_rows and not missing_refs and not suspicious_rows:
            action = DOCTOR_ACTION_SAFE_IGNORE
            command = "safe to ignore unless heartbeat/progress noise is crowding useful recall; consider message/session filters for future rows"
            rationale = "heartbeat rows are read-only noise diagnostics, not corruption"
        else:
            command = "inspect payload rows/refs; restore missing externalized payload files from backup before deleting or rewriting anything"
            if status == "warn":
                warning_only = True
                rationale = "payload warnings may represent preserved user/tool data"
            else:
                rationale = "payload diagnostic failures mean doctor could not read storage risk state reliably"
    elif name == "sensitive_pattern_handling":
        command = "inspect LCM_SENSITIVE_PATTERNS settings; remove unknown names or configure supported catalog entries"
    elif name == "orphaned_dag_nodes":
        command = "inspect affected DAG/source IDs; do not auto-delete summaries without confirming recall impact"
        if status == "warn":
            warning_only = True
        else:
            rationale = "DAG diagnostic failures mean doctor could not read summary/source state reliably"
    elif name == "summary_quality":
        command = "inspect worst_nodes and retrieval behavior; treat as summary quality evidence, not cleanup input"
        if status == "warn":
            warning_only = True
        else:
            rationale = "summary-quality diagnostic failures mean doctor could not read DAG quality state reliably"
    elif name == "config_validation":
        command = "inspect LCM_* environment/config values and adjust only intentional operator overrides"
    elif name == "source_lineage_hygiene" and status == "warn":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore legacy blank-source observations; use `/lcm doctor source` only when you intentionally want backup-first normalization"
        rationale = "legacy blank sources are normalized to unknown for compatibility"
    elif name == "source_lineage_hygiene":
        command = "inspect source-lineage diagnostics and SQLite read errors before running any source normalization workflow"
        rationale = "source-lineage failures indicate the doctor could not read attribution state reliably"
    elif name == "lifecycle_fragmentation":
        command = "inspect lifecycle categories; only use explicit backup-first lifecycle cleanup for empty lifecycle rows"
        if status == "warn":
            warning_only = True
            rationale = "not every lifecycle/state mismatch is harmful or safe to mutate"
        else:
            rationale = "lifecycle diagnostic failures mean doctor could not read session lifecycle state reliably"
    elif name == "compaction_replay_duplicates":
        command = (
            f"{COMPACTION_REPLAY_DUPLICATES_ACTION}; if a session's summaries stop publishing, "
            "continue the conversation in a new session"
        )
        if status == "warn" and isinstance(detail, dict) and not detail.get("replayed_rows_total"):
            command = (
                "the bounded scan found no candidate replay runs within its coverage "
                f"({detail.get('rows_scanned')} rows; window {detail.get('window')}); "
                "runs outside that coverage are not reported"
            )
            warning_only = True
            rationale = "an incomplete scan is not evidence that the store is free of replayed rows"
        elif status == "warn":
            warning_only = True
            rationale = "duplicate rows are preserved history; no automatic cleanup exists for them"
        else:
            rationale = "the duplicate scan could not read the message store"
    elif name == "context_pressure":
        action = DOCTOR_ACTION_SAFE_IGNORE
        command = "safe to ignore if compaction proceeds normally; inspect lcm_status only if pressure stays high or compaction loops"
        warning_only = True
        rationale = "context pressure is an operating state, not persisted-state corruption"
    elif name == "cleanup_candidates":
        action = DOCTOR_ACTION_BACKUP_FIRST_CLEANUP
        command = "run `/lcm doctor clean` first; if candidates are expected junk/noise, run `/lcm backup` before `/lcm doctor clean apply`"
        rationale = "candidate cleanup deletes rows and must stay preview-and-backup gated"

    return {
        "check": name,
        "status": status,
        "action": action,
        "operator_action": command,
        "warning_only": warning_only,
        "rationale": rationale,
    }


def doctor_guidance_for_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return actionable guidance for warning/failing lcm_doctor checks."""
    guidance = []
    for check in checks:
        item = doctor_guidance_for_check(check)
        if item is not None:
            guidance.append(item)
    return guidance


# Backward-compatible private aliases for existing command/tool internals and tests.
_state_db_path_for_engine = state_db_path_for_engine
_has_lifecycle_fragmentation = has_lifecycle_fragmentation
