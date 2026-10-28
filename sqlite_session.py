"""Explicit SQLite open recipes; schema and connection lifetime stay with stores."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
import sqlite3


@dataclass(frozen=True)
class ConnectionPolicy:
    """An exact connect call and the ordered operations before its row factory.

    Keyword pairs deliberately distinguish an omitted SQLite default from an
    explicitly supplied value (especially isolation_level=None and uri=True).
    A None row_factory leaves SQLite's default tuple rows untouched.
    Callbacks retain the caller's live bootstrap bindings and ownership timing.
    No schema, retries, transaction or close policy is added here.
    """

    database: str
    connect_kwargs: tuple[tuple[str, object], ...]
    before_row_factory: tuple[str | Callable[[sqlite3.Connection], None], ...]
    row_factory: type[sqlite3.Row] | None
    on_open: Callable[[sqlite3.Connection], None] | None
    on_setup_error: Callable[[Exception], None] | None


def open_connection(policy: ConnectionPolicy) -> sqlite3.Connection:
    """Run exactly the supplied recipe, preserving caller-owned error cleanup."""
    conn = sqlite3.connect(policy.database, **dict(policy.connect_kwargs))
    # Some stores publish the handle (and create their lock) before setup.
    # Those operations were outside their setup exception handler as well.
    if policy.on_open is not None:
        policy.on_open(conn)
    try:
        for operation in policy.before_row_factory:
            if isinstance(operation, str):
                conn.execute(operation)
            else:
                operation(conn)
        if policy.row_factory is not None:
            conn.row_factory = policy.row_factory
    except Exception as exc:
        if policy.on_setup_error is not None:
            policy.on_setup_error(exc)
        raise
    return conn


class TransactionOwner:
    """Join transactions on one store connection under its existing lock.

    The default retains implicit first-write opening. Explicit owners request
    BEGIN IMMEDIATE at entry. Savepoints, retry policies and schema scripts
    retain their existing owners; this primitive does not absorb those paths.
    """

    def __init__(self, lock: AbstractContextManager[object]) -> None:
        self._lock = lock
        self._active_connection: sqlite3.Connection | None = None

    def commit(self, conn: sqlite3.Connection) -> None:
        """Flush standalone helper writes; an enclosing atomic block flushes itself."""
        if self._active_connection is None or self._active_connection is not conn:
            conn.commit()

    @contextmanager
    def atomic(self, conn: sqlite3.Connection, *, immediate: bool = False) -> Iterator[None]:
        with self._lock:
            if self._active_connection is conn:
                if immediate and not conn.in_transaction:
                    # An empty enclosing scope has not reserved the database yet; reserve it here so a nested
                    # read-before-write keeps BEGIN IMMEDIATE semantics. The enclosing owner still commits.
                    conn.execute("BEGIN IMMEDIATE")
                yield
                return
            if immediate:
                conn.execute("BEGIN IMMEDIATE")
            self._active_connection = conn
            try:
                yield
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                self._active_connection = None
