"""Backup and rotate maintenance operations for the LCM store.

These are the data-layer maintenance primitives behind ``/lcm backup`` and
``/lcm rotate``: they flush the engine's SQLite connections and snapshot the
store to a timestamped or rolling backup file. They are pure functions that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the store/dag/lifecycle connection handling lives in one place.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any


def flush_engine_connections(engine) -> None:
    """Quiesce every SQLite connection before taking a backup.

    Shared by ``backup_database`` (timestamped backup) and
    ``rotate_backup_database`` (rolling backup) so the connection-flush
    contract stays in one place. MessageStore writes are operation-owned and
    must already be committed or rolled back; the other helpers retain their
    existing explicit flush APIs.
    """
    # Clones share every helper connection and the path-wide operation lock.
    # Keep that lock from the transaction-idle check through the final helper
    # commit; otherwise a clone can open a transaction after the check and have
    # this maintenance path commit its partial work behind its back.
    with engine._store.locked_connection():
        engine._store.assert_transaction_idle()
        engine._dag._conn.commit()
        lifecycle_conn = getattr(getattr(engine, "_lifecycle", None), "_conn", None)
        if lifecycle_conn is not None:
            lifecycle_conn.commit()
        assertion_store = getattr(engine, "_assertions", None)
        if assertion_store is not None:
            # AssertionStore owns a multi-statement publication transaction. Its
            # lock-taking API must serialize this flush with publish_source() so a
            # backup cannot commit a half-written receipt behind the publisher.
            assertion_store.commit()
        query_views = getattr(engine, "_query_views", None)
        if query_views is not None:
            query_views.commit()


def backup_database(engine) -> dict[str, Any]:
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_dir = engine.backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}-{timestamp}.sqlite3"

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        flush_engine_connections(engine)

        dest = sqlite3.connect(str(backup_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
    except (OSError, sqlite3.Error) as exc:
        return {
            "ok": False,
            "db_path": db_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }


def rotate_backup_database(engine) -> dict[str, Any]:
    """Write a rolling rotate-latest SQLite snapshot of the LCM store.

    Atomic via tmp-then-rename so the slot is never half-written. Unlike
    ``backup_database`` which produces timestamped files, this overwrites a
    single rolling slot so disk usage stays bounded across repeated rotates.
    """
    db_path = Path(engine._store.db_path)
    if not db_path.exists():
        return {
            "ok": False,
            "db_path": db_path,
            "error": "database file does not exist",
        }

    backup_path = engine.rotate_backup_path()
    backup_dir = backup_path.parent
    tmp_path = backup_path.with_name(backup_path.name + ".tmp")

    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        flush_engine_connections(engine)

        if tmp_path.exists():
            tmp_path.unlink()
        dest = sqlite3.connect(str(tmp_path))
        try:
            engine._store.backup(dest)
        finally:
            dest.close()
        # Atomic replace so the rolling slot is never half-written.
        tmp_path.replace(backup_path)
    except (OSError, sqlite3.Error) as exc:
        # Best-effort cleanup of the tmp file if something failed midway.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return {
            "ok": False,
            "db_path": db_path,
            "backup_path": backup_path,
            "error": str(exc),
        }

    backup_size = backup_path.stat().st_size if backup_path.exists() else 0
    return {
        "ok": True,
        "db_path": db_path,
        "backup_path": backup_path,
        "backup_size": backup_size,
    }
