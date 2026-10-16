"""Read-only views of one live LCM-X store for out-of-process readers (#1192 ``lcm-x mcp``).

The stock ``MessageStore``/``SummaryDAG``/``VectorStore``/``LifecycleStateStore``
constructors run the full write chain on open (mkdir, WAL conversion, DDL, FTS
ensure/repair, integrity scans, stem backfill, migrations, marker commits), and
their ``close()``/``__del__`` checkpoint the WAL. The subclasses here are built
WITHOUT those constructors over ``mode=ro`` connections:

- no ``query_only``: the vector arms stage candidate ids in TEMP tables, which a
  ``mode=ro`` connection allows while SQLite refuses every main-database write;
- autocommit (``isolation_level=None``) so no read transaction outlives a statement
  and pins the writer's WAL;
- ``close()``/``__del__`` never checkpoint and never clear caches shared with copies.

Open-time checks become capability flags (``probe_capabilities``), never repairs.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import LCMConfig
from .dag import SummaryDAG, build_nodes_fts_spec
from .db_bootstrap import (
    SCHEMA_VERSION,
    _fts_missing_triggers,
    external_content_fts_needs_repair,
    get_existing_table_names,
    inspect_lcm_schema_health,
    read_existing_schema_version,
    read_store_uuid,
    refuse_schema_version_too_new,
    verify_chunk_schema,
    verify_embedding_schema,
)
from .lifecycle_state import LifecycleStateStore
from .store import MessageStore, build_message_fts_spec, build_message_stem_fts_spec
from .vector_store import VectorStore

# Oldest stamp the read path understands (the v5 core: conversation_id/origin columns).
MIN_SCHEMA_VERSION = 5
BUSY_TIMEOUT_MS = 5_000
# Missing tables the reader serves around (flags) instead of refusing the store.
_OPTIONAL_CORE_TABLES = frozenset({"lcm_lifecycle_state", "messages_fts", "nodes_fts"})
_WAL_HEADER_OFFSET = 18  # SQLite header bytes 18/19 (write/read format) == 2 in WAL mode

# Structural FTS verdicts are cached per process: the probe COUNT(*)s the content
# table and the docsize shadow table, which is O(rows) on a large store.
_FTS_STRUCTURAL_CACHE: dict[tuple[str, str], bool] = {}
_FTS_CACHE_LOCK = threading.Lock()


class SchemaRefusedError(RuntimeError):
    """The store's schema is outside what this reader supports; nothing was opened for write."""


class StoreOfflineError(RuntimeError):
    """The WAL store cannot be opened read-only right now (its -wal/-shm are absent)."""


def wal_sidecars_missing(db_path: str | Path) -> bool:
    """True for a WAL-mode database (header bytes 18/19 == 2) missing its -wal or -shm file.

    Read without SQLite: a ``mode=ro`` open of such a store would RECREATE the missing
    files when the directory is writable (a filesystem write, possibly as the wrong user),
    or fail when it is not. Stock SQLite deletes both when the last connection closes.
    """
    path = Path(db_path)
    try:
        with path.open("rb") as handle:
            header = handle.read(_WAL_HEADER_OFFSET + 2)
    except OSError:
        return False
    # SQLite takes the WAL path when EITHER format byte (write 18, read 19) is 2 or more,
    # so a mixed (1, 2) header must count as WAL too.
    if len(header) < _WAL_HEADER_OFFSET + 2 or max(header[_WAL_HEADER_OFFSET:_WAL_HEADER_OFFSET + 2]) < 2:
        return False  # rollback-journal (or unreadable) store: opens as before
    return not (Path(f"{path}-wal").exists() and Path(f"{path}-shm").exists())


_OFFLINE_MESSAGE = ("store_offline: the store's -wal/-shm files are absent (Hermes is stopped); "
                    "start Hermes and retry")


def open_readonly_connection(
    db_path: str | Path, *, row_factory: Any = None, timeout: float = BUSY_TIMEOUT_MS / 1000
) -> sqlite3.Connection:
    """Open ``file:<abs>?mode=ro`` in autocommit; raise StoreOfflineError for a stopped WAL store.

    The sidecar check runs BEFORE the open, so the reader never creates -wal/-shm. A writer
    that stops between the check and the open is a residual race (see the operator guide).
    The open is forced with one schema read so a failure surfaces here, not inside a tool.
    """
    if wal_sidecars_missing(db_path):
        raise StoreOfflineError(_OFFLINE_MESSAGE)
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True, timeout=timeout,
                           isolation_level=None, check_same_thread=False)
    try:
        conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.OperationalError as exc:
        conn.close()
        if wal_sidecars_missing(db_path):
            raise StoreOfflineError(_OFFLINE_MESSAGE) from exc
        raise
    conn.row_factory = row_factory
    return conn


def store_uuid_for(conn: sqlite3.Connection, db_path: str | Path) -> tuple[str, str]:
    """Return ``(store_uuid, source)``: the stamped id, else one derived from the file identity."""
    stamped = read_store_uuid(conn)
    return (stamped, "metadata") if stamped else (derived_store_uuid(db_path), "derived")


def derived_store_uuid(db_path: str | Path) -> str:
    """``sha256(realpath, st_dev, st_ino)[:32]`` for a store without the stamp; raises OSError if absent."""
    real = os.path.realpath(db_path)
    stat = os.stat(real)
    return hashlib.sha256(f"{real}\x1f{stat.st_dev}\x1f{stat.st_ino}".encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class StoreCapabilities:
    schema_version: int
    fts_messages: bool
    fts_nodes: bool
    stem: bool
    embeddings: bool
    chunks: bool
    summary_node_provenance: bool
    lifecycle_state: bool
    host_uid_bindings: bool

    def as_flags(self) -> dict[str, Any]:
        return asdict(self)


def check_schema(conn: sqlite3.Connection) -> int:
    """Refuse a store stamped above this build or below the v5 minimum; return the stamp.

    Uses ``read_existing_schema_version`` (never ``get_schema_version``, which CREATEs).
    """
    version = read_existing_schema_version(conn)
    if version > SCHEMA_VERSION:
        try:
            refuse_schema_version_too_new(conn)
        except Exception as exc:  # SchemaVersionTooNewError, with its operator guidance
            raise SchemaRefusedError(f"schema_refused: {exc}") from exc
    if version < MIN_SCHEMA_VERSION:
        raise SchemaRefusedError(
            f"schema_refused: LCM database schema version {version} is older than the "
            f"supported minimum v{MIN_SCHEMA_VERSION}; let Hermes open (migrate) it first"
        )
    health = inspect_lcm_schema_health(conn)
    if health.get("error"):
        raise SchemaRefusedError(f"schema_refused: {health['error']}")
    missing = [name for name in health.get("missing_tables", []) if name not in _OPTIONAL_CORE_TABLES]
    if missing:
        raise SchemaRefusedError("schema_refused: required LCM tables are missing: " + ", ".join(missing))
    return version


def _fts_usable(conn: sqlite3.Connection, db_path: str | Path, spec: Any) -> bool:
    if not get_existing_table_names(conn, [spec.table_name]):
        return False
    key = (os.path.realpath(db_path), spec.table_name)
    with _FTS_CACHE_LOCK:
        cached = _FTS_STRUCTURAL_CACHE.get(key)
    if cached is None:
        cached = not external_content_fts_needs_repair(conn, spec)
        with _FTS_CACHE_LOCK:
            _FTS_STRUCTURAL_CACHE[key] = cached
    return cached


def stem_usable(conn: sqlite3.Connection) -> bool:
    """The stem index answers only when ready, present and fully triggered (computed, never defaulted)."""
    spec = build_message_stem_fts_spec()
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'fts_stem_state'").fetchone()
    except sqlite3.Error:
        return False
    return (
        bool(row) and str(row[0]) == "ready"
        and bool(get_existing_table_names(conn, [spec.table_name]))
        and not _fts_missing_triggers(conn, spec)
    )


def probe_capabilities(conn: sqlite3.Connection, db_path: str | Path) -> StoreCapabilities:
    """Schema checks plus optional-capability flags, inside ONE read snapshot.

    Raises SchemaRefusedError; a missing optional capability is a False flag.
    """
    conn.execute("BEGIN")
    try:
        version = check_schema(conn)
        tables = get_existing_table_names(conn, ["summary_node_provenance", "lcm_lifecycle_state",
                                                 "host_uid_bindings"])
        embeddings = not verify_embedding_schema(conn) and conn.execute(
            "SELECT 1 FROM lcm_embedding_profile WHERE active = 1 AND archived_at IS NULL "
            "AND task = 'summary' LIMIT 1"
        ).fetchone() is not None
        return StoreCapabilities(
            schema_version=version,
            fts_messages=_fts_usable(conn, db_path, build_message_fts_spec()),
            fts_nodes=_fts_usable(conn, db_path, build_nodes_fts_spec()),
            stem=stem_usable(conn),
            embeddings=embeddings,
            chunks=not verify_chunk_schema(conn),
            summary_node_provenance="summary_node_provenance" in tables,
            lifecycle_state="lcm_lifecycle_state" in tables,
            host_uid_bindings="host_uid_bindings" in tables,
        )
    finally:
        conn.execute("COMMIT")


def active_profile_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """The store's active embedding identity per task; {} when the tables are absent."""
    if verify_embedding_schema(conn):
        return {}
    rows = conn.execute(
        "SELECT task, identity_hash FROM lcm_embedding_profile "
        "WHERE active = 1 AND archived_at IS NULL ORDER BY registered_at DESC, identity_hash DESC"
    ).fetchall()
    found: dict[str, str] = {}
    for row in rows:
        found.setdefault(str(row[0]), str(row[1]))
    return found


class _CloseWithoutCheckpoint:
    """``close``/``__del__`` for the views: release the connection, never checkpoint or clear caches.

    Listed first in each view's bases so it overrides the stock checkpointing pair, which
    otherwise still fires on constructor-free instances and on every ``copy.copy``.
    """

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None:
            conn.close()
            self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass


class ReadOnlyMessageStore(_CloseWithoutCheckpoint, MessageStore):
    """MessageStore fields over a caller-owned ``mode=ro`` connection; no constructor writes."""

    def __init__(self, db_path: str | Path, conn: sqlite3.Connection, *, config: LCMConfig,
                 hermes_home: str, stem_usable: bool, fts_usable: bool = True) -> None:
        self.db_path = Path(db_path)
        self._ingest_protection_config = config
        self._hermes_home = hermes_home
        self._context_window_tokens = {}
        self._conn = conn
        self._write_lock = threading.RLock()
        self._stem_usable = bool(stem_usable)
        self._fts_usable = bool(fts_usable)
        self._last_fts_index = "plain"

    def _search_fts(self, query: str, session_id: str | None = None, limit: int = 20,
                    sort: str | None = None, source: str | None = None,
                    conversation_id: str | None = None, role: str | None = None,
                    time_from: float | None = None, time_to: float | None = None,
                    exclude_session_ids: Any = None, allow_operators: bool = False,
                    agent_operators: bool = False, stemmed: bool = False) -> list[dict[str, Any]]:
        if self._fts_usable:
            return super()._search_fts(
                query, session_id, limit, sort, source, conversation_id, role, time_from, time_to,
                exclude_session_ids, allow_operators=allow_operators,
                agent_operators=agent_operators, stemmed=stemmed,
            )
        # A structurally stale index cannot be repaired from here; LIKE answers instead.
        self._last_fts_index = "plain"
        return self._search_like(
            query, session_id=session_id, limit=limit, sort=sort, source=source,
            conversation_id=conversation_id, role=role, time_from=time_from, time_to=time_to,
            exclude_session_ids=exclude_session_ids,
        )


class ReadOnlySummaryDAG(_CloseWithoutCheckpoint, SummaryDAG):
    """SummaryDAG fields over a caller-owned ``mode=ro`` connection; no constructor writes."""

    def __init__(self, db_path: str | Path, conn: sqlite3.Connection, *, fts_usable: bool = True,
                 provenance_table: bool = True) -> None:
        self.db_path = Path(db_path)
        self._conn = conn
        self._db_lock = threading.RLock()
        self._fts_usable = bool(fts_usable)
        self._provenance_table = bool(provenance_table)

    def search(self, query: str, session_id: str | None = None, limit: int = 20,
               sort: str | None = None, source: str | None = None, *,
               agent_operators: bool = False, allow_operators: bool = False) -> list[Any]:
        if self._fts_usable:
            return super().search(
                query, session_id, limit, sort, source,
                agent_operators=agent_operators, allow_operators=allow_operators,
            )
        return self._search_like(query, session_id=session_id, limit=limit, sort=sort, source=source)

    def get_node_provenance(self, node_id: int) -> dict[str, Any] | None:
        # The sidecar arrived without a schema bump (#441); an older v5 store lacks it.
        if not self._provenance_table:
            return None
        return super().get_node_provenance(node_id)


class ReadOnlyLifecycleStateStore(_CloseWithoutCheckpoint, LifecycleStateStore):
    """Lifecycle reads over a caller-owned ``mode=ro`` connection (row factory as production)."""

    def __init__(self, db_path: str | Path, conn: sqlite3.Connection) -> None:
        self.db_path = Path(db_path)
        self._conn = conn
        self._lock = threading.RLock()


class ReadOnlyVectorStore(_CloseWithoutCheckpoint, VectorStore):
    """A poolable VectorStore whose construction and close never write the store.

    Field for field the stock ``__init__`` minus the parent ``mkdir`` and the write-path
    ``_init_db`` (migrations, embedding-table ensure, marker commit). ``close`` keeps the
    matrix caches: the deadline reader's shallow copies share them with the pooled
    original, and clearing them on every copy's collection would empty the cache.
    """

    _supports_pooling = True

    def __init__(self, db_path: str | Path, *, config: LCMConfig | None = None,
                 bounded_scan_rows: int | None = None) -> None:
        self.db_path = Path(db_path)
        resolved_config = config or LCMConfig.from_env()
        self.bounded_scan_rows = (resolved_config.embedding_bounded_scan_rows if bounded_scan_rows is None
                                  else int(bounded_scan_rows))
        self.knn_prescreen_multiplier = max(1, int(getattr(resolved_config, "knn_prescreen_multiplier", 4) or 4))
        self._write_binary_prescreen = bool(getattr(resolved_config, "embedding_binary_prescreen", False))
        self._conn = None
        self._write_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._txn_depth = 0
        self._matrix_cache = OrderedDict()
        self._chunk_matrix_cache = OrderedDict()
        self._binary_matrix_cache = OrderedDict()
        self._chunk_binary_matrix_cache = OrderedDict()
        self._chunk_schema_ready = False
        self._init_db()

    def _init_db(self) -> None:
        self._conn = open_readonly_connection(self.db_path, row_factory=sqlite3.Row)
