"""SQLite lock-contention helpers shared by the LCM engine.

Isolated from ``engine.py`` (WS5 seam): lock-contention detection, bounded
``busy_timeout`` changes, and transaction-preserving savepoints are pure SQLite
concerns with no engine state. Callers keep their own policy constants (for
example the session-end timeout budget).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, TypeVar

from .sqlite_session import TransactionOwner


_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def _is_sqlite_locked_error(exc: BaseException) -> bool:
    """Return True when an exception chain represents SQLite lock contention."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, sqlite3.Error) and "locked" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_ingest_busy_error(exc: BaseException, *, require_code=False) -> bool:
    """The ingest owner also retries code-based BUSY, including BUSY_SNAPSHOT."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, sqlite3.Error) and (
            not require_code and ("locked" in str(current).lower() or "busy" in str(current).lower())
            or (getattr(current, "sqlite_errorcode", 0) or 0) & 255 in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_sqlite_busy_snapshot_error(exc: BaseException) -> bool:
    """Return True only for SQLite's stale read-snapshot write-upgrade error."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            isinstance(current, sqlite3.Error)
            and getattr(current, "sqlite_errorcode", None)
            == sqlite3.SQLITE_BUSY_SNAPSHOT
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _run_sqlite_write_with_snapshot_retry(
    conn: sqlite3.Connection,
    operation: Callable[[], _T],
    *,
    operation_name: str,
    owner_active: bool = False,
) -> _T:
    """Run one write transaction, retrying one stale-snapshot upgrade.

    ``SQLITE_BUSY_SNAPSHOT`` cannot be fixed by waiting: the connection must
    roll back its old read snapshot and begin the write from current state.
    Ordinary ``SQLITE_BUSY`` already receives SQLite's configured busy timeout
    and is not multiplied here. An explicit owner runs the operation directly
    and retains responsibility for rollback and retry of the complete decision.
    """
    if owner_active:
        # The owner must roll back and repeat classification, not just the INSERT.
        return operation()
    for attempt in range(2):
        try:
            with conn:
                return operation()
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.rollback()
            if attempt == 0 and _is_sqlite_busy_snapshot_error(exc):
                continue
            if _is_sqlite_locked_error(exc):
                logger.warning(
                    "SQLite write lock recovery exhausted "
                    "(operation=%s, code=%s, name=%s)",
                    operation_name,
                    getattr(exc, "sqlite_errorcode", None),
                    getattr(exc, "sqlite_errorname", None),
                )
            raise
    raise AssertionError("unreachable SQLite write retry state")


def _sqlite_busy_timeout_ms(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


class _SqliteSavepointError(RuntimeError):
    """The failed writer could not be isolated; its enclosing owner must abort."""


@contextmanager
def _sqlite_savepoint(conn: sqlite3.Connection, *, abort_on_cleanup_failure: bool = False,
                      owner_active: bool = False) -> Iterator[None]:
    """Isolate helper writes without taking ownership of a caller transaction."""
    # Owned scopes nest with SQLite's innermost-name semantics. Stable names
    # keep diagnostic observers from changing the ordered write trace.
    owner_active = owner_active and TransactionOwner.owns_connection(conn)
    name = "lcm_ingest" if owner_active else f"lcm_{uuid.uuid4().hex}"
    if owner_active and not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")  # RELEASE must never commit an empty owner's transaction.
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
        if abort_on_cleanup_failure:
            conn.execute(f"RELEASE SAVEPOINT {name}")
    except BaseException:
        if abort_on_cleanup_failure and not conn.in_transaction:
            raise  # SQLite already aborted the owner; preserve the original error.
        try:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
            finally:
                conn.execute(f"RELEASE SAVEPOINT {name}")
        except Exception as exc:
            if abort_on_cleanup_failure:
                raise _SqliteSavepointError("Could not roll back writer savepoint") from exc
            raise
        raise
    else:
        if not abort_on_cleanup_failure:
            conn.execute(f"RELEASE SAVEPOINT {name}")


@contextmanager
def _temporary_sqlite_busy_timeout(
    connections: List[sqlite3.Connection | None],
    timeout_ms: int,
) -> Iterator[None]:
    """Temporarily bound SQLite lock waits for gateway-critical paths."""
    bounded_timeout = max(0, int(timeout_ms))
    originals: list[tuple[sqlite3.Connection, int]] = []
    for conn in connections:
        if conn is None:
            continue
        original = _sqlite_busy_timeout_ms(conn)
        conn.execute(f"PRAGMA busy_timeout={bounded_timeout}")
        originals.append((conn, original))
    try:
        yield
    finally:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")


_SLOW_STATEMENT_SECONDS = 1.0
_get_ident, _monotonic = threading.get_ident, time.monotonic  # bound once: the proxies' per-statement path


class PhaseSqlTrace:
    """#1127: one thread's slow SQL statements and lock waits on some store objects while a phase runs.

    ``install`` swaps each named connection or lock for a timing proxy; ``uninstall`` puts back each one that
    is still the proxy. Only the installing thread's statements and waits are counted. A slow statement keeps
    the first 40 characters of its SQL template, never its parameters."""

    def __init__(self) -> None:
        self.owner = threading.get_ident()
        self.lock_wait = self.slow_seconds = self.slowest = 0.0
        self.slow_count, self.slowest_sql = 0, ""
        self.spans: Dict[str, float] = {}
        self.steps: Dict[str, float] = {}
        self._swapped: list[tuple[Any, str, Any, Any]] = []

    def install(self, targets) -> None:
        for part, attr in targets:
            real = getattr(part, attr, None)
            if real is None or isinstance(real, (_TimedConnection, _TimedLock)):
                continue
            proxy = (_TimedConnection if isinstance(real, sqlite3.Connection) else _TimedLock)(real, self)
            setattr(part, attr, proxy)
            self._swapped.append((part, attr, real, proxy))

    def uninstall(self) -> None:
        self.owner = None  # a proxy still held by a caller passes straight through
        swapped, self._swapped = self._swapped, []
        for part, attr, real, proxy in reversed(swapped):
            if getattr(part, attr, None) is proxy:
                setattr(part, attr, real)

    def timed(self, run: Callable[..., _T], sql: Any, *args: Any) -> _T:
        started = _monotonic()
        try:
            return run(sql, *args)
        finally:
            if _monotonic() - started >= _SLOW_STATEMENT_SECONDS:
                self.slow(started, sql)

    def slow(self, started: float, sql: Any) -> None:
        """The rare path: a statement of at least a second. Another thread's is not counted."""
        if self.owner != _get_ident():
            return
        if getattr(self, "ingest_admission", False) and str(sql) == "BEGIN IMMEDIATE":
            sql = "INSERT INTO messages (ingest admission)"  # Charge reservation to its logical write.
        took = _monotonic() - started
        self.slow_count, self.slow_seconds = self.slow_count + 1, self.slow_seconds + took
        if took > self.slowest:
            self.slowest, self.slowest_sql = took, " ".join(str(sql).split())[:40].replace('"', "'")


class _TimedConnection:
    """A connection whose statements a PhaseSqlTrace times; everything else passes through."""

    __slots__ = ("_lcm_real", "_lcm_trace")

    def __init__(self, real: sqlite3.Connection, trace: PhaseSqlTrace) -> None:
        object.__setattr__(self, "_lcm_real", real)
        object.__setattr__(self, "_lcm_trace", trace)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lcm_real, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._lcm_real, name, value)

    def __enter__(self):
        self._lcm_real.__enter__()
        return self

    def __exit__(self, *exc):  # a ``with conn:`` block ends in an implicit COMMIT or ROLLBACK, which can wait too
        return self._lcm_trace.timed(lambda _sql, *args: self._lcm_real.__exit__(*args),
                                     "ROLLBACK" if exc and exc[0] is not None else "COMMIT", *exc)

    def execute(self, sql, parameters=()):  # the hot path: two clock reads per statement
        started = _monotonic()
        try:
            return self._lcm_real.execute(sql, parameters)
        finally:
            if _monotonic() - started >= _SLOW_STATEMENT_SECONDS:
                self._lcm_trace.slow(started, sql)

    def executemany(self, sql, *args):
        return self._lcm_trace.timed(self._lcm_real.executemany, sql, *args)

    def executescript(self, sql):  # a lazy schema ensure writes, so it can wait on the write lock too
        return self._lcm_trace.timed(self._lcm_real.executescript, sql)

    def commit(self):
        return self._lcm_trace.timed(lambda _sql: self._lcm_real.commit(), "COMMIT")


class _TimedLock:
    """A lock whose ``with`` acquisitions a PhaseSqlTrace adds to its lock wait."""

    __slots__ = ("_lcm_real", "_lcm_trace")

    def __init__(self, real: Any, trace: PhaseSqlTrace) -> None:
        self._lcm_real, self._lcm_trace = real, trace

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lcm_real, name)

    def __enter__(self):
        if self._lcm_trace.owner != _get_ident():
            return self._lcm_real.__enter__()
        started = _monotonic()
        entered = self._lcm_real.__enter__()
        self._lcm_trace.lock_wait += _monotonic() - started
        return entered

    def __exit__(self, *exc):
        return self._lcm_real.__exit__(*exc)
