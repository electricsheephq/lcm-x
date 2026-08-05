"""Per-item scope storage, Teams setup backfill, and verification.

Stage one deliberately stores a nullable owner scope.  A NULL value is the
legacy/un-stamped state; the Teams setup hook is the only path that turns that
state into an owner attribution for historical rows.
"""

from __future__ import annotations

import ast
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .db_bootstrap import add_column_if_missing

ACCESS_SCOPE_COLUMN = "access_scope"
# Kept as a compatibility alias for callers of the first staging build.  All
# SQL and schema work uses the unambiguous name above.
SCOPE_COLUMN = ACCESS_SCOPE_COLUMN

# Rollup ``scope`` columns pre-date this staging and remain the durable
# partition key.  Each rollup table receives a separate nullable
# ``access_scope`` column.
SCOPE_BEARING_TABLES = (
    "messages",
    "summary_nodes",
    "lcm_rollups",
    "lcm_rollup_invalidations",
    "lcm_rollup_state",
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
    "lcm_embedding_binary",
    "lcm_chunk_meta",
    "lcm_chunk_vectors",
    "lcm_chunk_binary",
)

_ACCESS_SCOPE_TABLES = (
    "messages",
    "summary_nodes",
    "lcm_rollups",
    "lcm_rollup_invalidations",
    "lcm_rollup_state",
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
    "lcm_embedding_binary",
    "lcm_chunk_meta",
    "lcm_chunk_vectors",
    "lcm_chunk_binary",
)

_SESSION_SCOPE_TABLES = ("messages", "summary_nodes")
_OPTIONAL_SCOPE_TABLES = {
    "lcm_embedding_meta",
    "lcm_embedding_vectors",
    "lcm_embedding_binary",
    "lcm_chunk_meta",
    "lcm_chunk_vectors",
    "lcm_chunk_binary",
}

ScopeResolver = Callable[[str], str | None]
_TEAMS_ENABLED_ATTRIBUTE = "lcm_" + "teams_enabled"


def teams_enabled(engine: object) -> bool:
    """Read the host's explicit Teams flag for storage-only decisions."""

    return bool(getattr(engine, _TEAMS_ENABLED_ATTRIBUTE, False))


def mark_teams_enabled(engine: object) -> None:
    """Publish the Teams setup completion flag after backfill succeeds."""

    setattr(engine, _TEAMS_ENABLED_ATTRIBUTE, True)


@dataclass(frozen=True)
class ScopeWriter:
    """One source-level INSERT into an access-scope-bearing table."""

    table: str
    path: str
    line: int
    function: str
    populates_access_scope: bool

    @property
    def populates_scope(self) -> bool:
        """Compatibility view for the first staging API."""

        return self.populates_access_scope

    @property
    def name(self) -> str:
        location = f"{self.path}:{self.line}"
        return f"{location}:{self.function}" if self.function else location


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def ensure_scope_columns(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Add nullable ``access_scope`` to every materialized item table.

    The migration intentionally does not create optional feature tables.  A
    disabled store therefore keeps exactly the same table set; a later lazy
    feature initializer calls this function again after creating its tables.
    ``add_column_if_missing`` is the repository's race-tolerant ALTER idiom.
    """

    added: list[str] = []
    existing: list[str] = []
    absent: list[str] = []
    rollup_tables = {
        "lcm_rollups", "lcm_rollup_invalidations", "lcm_rollup_state"
    }
    for table in _ACCESS_SCOPE_TABLES:
        if not _table_exists(conn, table):
            absent.append(table)
            continue
        columns = _table_columns(conn, table)
        # Databases created by the first, defective staging build used
        # ``scope`` on the non-rollup tables.  Rename that column in place so
        # its values survive the corrective migration.  Rollup ``scope`` is a
        # different, NOT NULL partition key and must never be renamed.
        if table not in rollup_tables and "scope" in columns and ACCESS_SCOPE_COLUMN not in columns:
            conn.execute(f'ALTER TABLE "{table}" RENAME COLUMN scope TO {ACCESS_SCOPE_COLUMN}')
            columns.remove("scope")
            columns.add(ACCESS_SCOPE_COLUMN)
            existing.append(table)
            continue
        if ACCESS_SCOPE_COLUMN in columns:
            existing.append(table)
            continue
        add_column_if_missing(
            conn,
            columns,
            ACCESS_SCOPE_COLUMN,
            f'ALTER TABLE "{table}" ADD COLUMN {ACCESS_SCOPE_COLUMN} TEXT',
        )
        added.append(table)
    return {"added": added, "existing": existing, "absent": absent}


def _resolve_scope(resolver: ScopeResolver, session_id: object) -> str:
    normalized_session = str(session_id or "")
    if not normalized_session:
        raise ValueError("cannot stamp a scope-bearing row without session_id")
    value = resolver(normalized_session)
    if value is None or not str(value).strip():
        raise ValueError(f"pre-Teams owner is unresolved for session_id={normalized_session}")
    return str(value)


def _backfill_session_table(
    conn: sqlite3.Connection,
    table: str,
    resolver: ScopeResolver,
    batch_size: int,
) -> int:
    updated = 0
    while True:
        rows = conn.execute(
            f'SELECT rowid, session_id FROM "{table}" '
            f"WHERE {ACCESS_SCOPE_COLUMN} IS NULL ORDER BY rowid LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            return updated
        values = [(_resolve_scope(resolver, row[1]), int(row[0])) for row in rows]
        conn.executemany(
            f'UPDATE "{table}" SET {ACCESS_SCOPE_COLUMN}=? WHERE rowid=? '
            f'AND {ACCESS_SCOPE_COLUMN} IS NULL', values
        )
        conn.commit()
        updated += len(values)


def _backfill_joined_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    source_table: str,
    join_sql: str,
    batch_size: int,
) -> int:
    """Copy a source row's already-stamped scope to a derived item table."""

    updated = 0
    while True:
        rows = conn.execute(
            f"""
            SELECT target.rowid, source.access_scope
            FROM {table} AS target
            JOIN {source_table} AS source ON {join_sql}
            WHERE target.access_scope IS NULL AND source.access_scope IS NOT NULL
            ORDER BY target.rowid LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
        if not rows:
            return updated
        conn.executemany(
            f"UPDATE {table} SET {ACCESS_SCOPE_COLUMN}=? WHERE rowid=? "
            f"AND {ACCESS_SCOPE_COLUMN} IS NULL",
            [(str(row[1]), int(row[0])) for row in rows],
        )
        conn.commit()
        updated += len(rows)


def _backfill_rollup_table(
    conn: sqlite3.Connection,
    table: str,
    resolver: ScopeResolver,
    batch_size: int,
) -> int:
    """Attribute rollup bookkeeping from its unchanged partition key."""

    updated = 0
    while True:
        rows = conn.execute(
            f'SELECT rowid, scope FROM "{table}" '
            f"WHERE {ACCESS_SCOPE_COLUMN} IS NULL ORDER BY rowid LIMIT ?",
            (batch_size,),
        ).fetchall()
        if not rows:
            return updated
        values = [(_resolve_scope(resolver, row[1]), int(row[0])) for row in rows]
        conn.executemany(
            f'UPDATE "{table}" SET {ACCESS_SCOPE_COLUMN}=? WHERE rowid=? '
            f'AND {ACCESS_SCOPE_COLUMN} IS NULL',
            values,
        )
        conn.commit()
        updated += len(values)


def backfill_scopes(
    conn: sqlite3.Connection,
    owner_for_session: ScopeResolver | None = None,
    *,
    batch_size: int = 256,
) -> dict[str, object]:
    """Stamp all pre-existing rows, leaving already-stamped rows untouched.

    The NULL predicate makes repeated runs idempotent.  Each batch commits on
    its own, so a crash after any batch leaves a deterministic NULL remainder
    that a later run resumes.  Derived rows copy their leaf's scope only after
    messages/summary nodes have been processed.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ensure_scope_columns(conn)
    if owner_for_session is None:
        raise ValueError(
            "owner_for_session is required to attribute pre-Teams rows to their owner"
        )
    resolver = owner_for_session
    updated: dict[str, int] = {}
    for table in _SESSION_SCOPE_TABLES:
        if _table_exists(conn, table):
            updated[table] = _backfill_session_table(conn, table, resolver, batch_size)

    # Chunk metadata points at a raw message; vectors/binary point at metadata.
    if _table_exists(conn, "lcm_chunk_meta") and _table_exists(conn, "messages"):
        updated["lcm_chunk_meta"] = _backfill_joined_table(
            conn,
            table="lcm_chunk_meta",
            source_table="messages",
            join_sql="source.store_id = target.store_id",
            batch_size=batch_size,
        )
    for table in ("lcm_chunk_vectors", "lcm_chunk_binary"):
        if _table_exists(conn, table) and _table_exists(conn, "lcm_chunk_meta"):
            updated[table] = _backfill_joined_table(
                conn,
                table=table,
                source_table="lcm_chunk_meta",
                join_sql="source.chunk_id = target.chunk_id AND source.identity_hash = target.identity_hash",
                batch_size=batch_size,
            )

    # Rollups retain their session partition key in ``scope``.  Their access
    # scope is attributed from that key only during the explicit Teams setup.
    for table in ("lcm_rollups", "lcm_rollup_invalidations", "lcm_rollup_state"):
        if _table_exists(conn, table):
            updated[table] = _backfill_rollup_table(
                conn, table, owner_for_session, batch_size
            )

    # Summary embedding metadata points at a summary node; vector/binary rows
    # point at that metadata.  ``embedded_id`` is the persisted node id.
    if _table_exists(conn, "lcm_embedding_meta") and _table_exists(conn, "summary_nodes"):
        updated["lcm_embedding_meta"] = _backfill_joined_table(
            conn,
            table="lcm_embedding_meta",
            source_table="summary_nodes",
            join_sql="source.node_id = CAST(target.embedded_id AS INTEGER)",
            batch_size=batch_size,
        )
    for table in ("lcm_embedding_vectors", "lcm_embedding_binary"):
        if _table_exists(conn, table) and _table_exists(conn, "lcm_embedding_meta"):
            updated[table] = _backfill_joined_table(
                conn,
                table=table,
                source_table="lcm_embedding_meta",
                join_sql="source.embedded_id = target.embedded_id AND source.identity_hash = target.identity_hash",
                batch_size=batch_size,
            )

    conn.commit()
    return {"updated": updated, "total_updated": sum(updated.values())}


def setup_teams_scope(
    conn: sqlite3.Connection,
    owner_for_session: ScopeResolver | None = None,
    *,
    batch_size: int = 256,
) -> dict[str, object]:
    """Run the additive schema stage and Teams-only historical backfill."""

    columns = ensure_scope_columns(conn)
    result = backfill_scopes(conn, owner_for_session, batch_size=batch_size)
    result["columns"] = columns
    return result


def _constant_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else "{expr}"
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_text(node.left)
        right = _constant_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


_INSERT_RE = re.compile(
    r"\bINSERT(?:\s+OR\s+[A-Z]+)?\s+INTO\s+([\"`]?[A-Za-z_][A-Za-z0-9_]*[\"`]?)"
    r"\s*(?:\(([^)]*)\))?",
    re.IGNORECASE | re.DOTALL,
)


class _WriterVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, targets: set[str]) -> None:
        self.path = path
        self.targets = targets
        self.stack: list[str] = []
        self.writers: list[ScopeWriter] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Constant(self, node: ast.Constant) -> None:
        text = node.value if isinstance(node.value, str) else None
        if text:
            self._record(str(text), node.lineno)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        text = _constant_text(node)
        if text:
            self._record(text, node.lineno)
        self.generic_visit(node)

    def _record(self, sql: str, line: int) -> None:
        for match in _INSERT_RE.finditer(sql):
            table = match.group(1).strip('"`').lower()
            if table not in self.targets:
                continue
            columns = match.group(2) or ""
            self.writers.append(
                ScopeWriter(
                    table=table,
                    path=self.path.as_posix(),
                    line=line,
                    function=".".join(self.stack),
                    populates_access_scope=bool(
                        re.search(r"\baccess_scope\b", columns, re.IGNORECASE)
                    ),
                )
            )


def _source_files(source_root: Path) -> Iterable[Path]:
    excluded = {"tests", "bench", "benchmarks", "__pycache__"}
    for path in sorted(source_root.rglob("*.py")):
        if any(part.startswith(".venv") or part in excluded for part in path.parts):
            continue
        yield path


def enumerate_scope_writers(source_root: str | Path | None = None) -> list[ScopeWriter]:
    """Discover scope-bearing INSERT writers from current source text."""

    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    writers: list[ScopeWriter] = []
    for path in _source_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        visitor = _WriterVisitor(path, set(SCOPE_BEARING_TABLES))
        visitor.visit(tree)
        writers.extend(visitor.writers)
    return writers


def verify_scope_storage(
    conn: sqlite3.Connection | None,
    *,
    source_root: str | Path | None = None,
    teams_enabled: bool = True,
) -> dict[str, object]:
    """Doctor-style storage verification with a non-vacuity sentinel."""

    tables: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    observed_rows = 0
    if conn is None:
        return {
            "status": "fail",
            "message": "scope storage connection is not initialized",
            "tables": tables,
            "writer_guard": {"status": "fail", "writers": [], "violations": []},
            "observed_rows": 0,
        }

    for table in SCOPE_BEARING_TABLES:
        if not _table_exists(conn, table):
            tables[table] = {
                "exists": False,
                "stamped": 0,
                "unstamped": 0,
                "total": 0,
            }
            continue
        columns = _table_columns(conn, table)
        if ACCESS_SCOPE_COLUMN not in columns:
            errors.append(f"{table}: missing access_scope column")
            tables[table] = {
                "exists": True,
                "stamped": 0,
                "unstamped": 0,
                "total": 0,
                "error": "missing access_scope column",
            }
            continue
        total = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        unstamped = int(
            conn.execute(
                f'SELECT COUNT(*) FROM "{table}" '
                f"WHERE {ACCESS_SCOPE_COLUMN} IS NULL"
            ).fetchone()[0]
        )
        stamped = total - unstamped
        observed_rows += total
        tables[table] = {
            "exists": True,
            "stamped": stamped,
            "unstamped": unstamped,
            "total": total,
        }

    writers = enumerate_scope_writers(source_root)
    violations = [
        writer.name for writer in writers if not writer.populates_access_scope
    ]
    writer_guard = {
        "status": "fail" if violations else "pass",
        "writers": [writer.name for writer in writers],
        "violations": violations,
        "discovered": len(writers),
    }
    if violations:
        errors.append("writer guard: " + ", ".join(violations))
    unstamped_total = sum(int(item.get("unstamped", 0)) for item in tables.values())
    if teams_enabled and unstamped_total:
        errors.append(f"{unstamped_total} unstamped access-scope row(s)")

    if errors:
        status = "fail"
        message = "; ".join(errors)
    elif not teams_enabled:
        status = "not-enabled"
        message = "Teams not enabled: NULL access_scope values remain legacy-compatible"
    elif observed_rows == 0:
        status = "nothing-to-verify"
        message = "nothing to verify: no scope-bearing rows were observed"
    else:
        status = "verified"
        message = "all observed scope-bearing rows are stamped"
    return {
        "status": status,
        "message": message,
        "tables": tables,
        "writer_guard": writer_guard,
        "observed_rows": observed_rows,
        "unstamped_rows": unstamped_total,
    }


# Names used by callers that describe the operation as an explicit migration
# rather than a backfill.  Keep one implementation so idempotence/resumability
# cannot drift between entry points.
backfill_scope = backfill_scopes
verify_scope = verify_scope_storage
