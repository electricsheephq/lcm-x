"""Bind retrieval readers to caller-owned connections and independent locks.

Connection setup, deadlines and closing remain with the tool that owns the read.
Shallow copies preserve store/DAG configuration and instance-level overrides.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import sqlite3
import threading
from typing import Generic, Protocol, TypeVar


class _MessageBinding(Protocol):
    _conn: sqlite3.Connection | None
    _write_lock: threading.RLock


class _DAGBinding(Protocol):
    _conn: sqlite3.Connection | None
    _db_lock: threading.RLock


_StoreT = TypeVar("_StoreT", bound=_MessageBinding)
_DAGT = TypeVar("_DAGT", bound=_DAGBinding)


@dataclass(frozen=True)
class RetrievalIO(Generic[_StoreT, _DAGT]):
    """Read clones; their connections are borrowed, never opened or closed here."""

    store: _StoreT
    dag: _DAGT


@dataclass(frozen=True)
class RetrievalContext(Generic[_StoreT, _DAGT]):
    """Live bindings from which a request derives its isolated read clones."""

    store: _StoreT
    dag: _DAGT

    def bind(
        self, message_conn: sqlite3.Connection, dag_conn: sqlite3.Connection,
    ) -> RetrievalIO[_StoreT, _DAGT]:
        """Keep each clone's actual lock independent of the live connection owner."""
        read_store = copy.copy(self.store)
        read_dag = copy.copy(self.dag)
        read_store._conn = message_conn
        read_store._write_lock = threading.RLock()
        read_dag._conn = dag_conn
        read_dag._db_lock = threading.RLock()
        return RetrievalIO(read_store, read_dag)
