"""Backup and rotate maintenance operations for the LCM store.

These are the data-layer maintenance primitives behind ``/lcm backup`` and
``/lcm rotate``: they flush the engine's SQLite connections and snapshot the
store to a timestamped or rolling backup file. They are pure functions that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the store/dag/lifecycle connection handling lives in one place.
"""

from __future__ import annotations

from contextlib import closing, nullcontext, suppress
from datetime import datetime
from pathlib import Path
import os
import sqlite3
import secrets
from typing import Any


from .externalize import get_large_output_storage_dir


def externalized_payload_inventory(engine) -> dict:
    """Describe the separate payload directory without creating it or raising."""
    def reason(exc):
        return str(exc).splitlines()[0][:160] if str(exc) else type(exc).__name__
    try:
        path = get_large_output_storage_dir(engine._config, engine._hermes_home, create=False)
    except Exception as exc:
        return {"path": None, "error": reason(exc)}
    files = size = unreadable = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    if entry.is_file(follow_symlinks=False):
                        size += entry.stat(follow_symlinks=False).st_size
                        files += 1
                except OSError:
                    unreadable += 1  # reported, so a partial count never reads as complete
    except FileNotFoundError:
        return {"path": path, "exists": False, "files": 0, "bytes": 0, "unreadable": 0}
    except Exception as exc:
        # The path is known; only the scan failed, so keep naming it (#1077 review).
        return {"path": path, "exists": True, "files": 0, "bytes": 0, "unreadable": 0, "scan_error": reason(exc)}
    return {"path": path, "exists": True, "files": files, "bytes": size, "unreadable": unreadable}


def flush_engine_connections(engine) -> None:
    """Commit pending writes on every SQLite connection the engine owns.

    Shared by ``backup_database`` (timestamped backup) and
    ``rotate_backup_database`` (rolling backup) so the connection-flush
    contract stays in one place.
    """
    engine._store.commit()
    # #667: never commit a DAG transaction another thread holds open (the level 3 repair)
    with getattr(engine._dag, "_db_lock", nullcontext()):
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

        _backup_pair(engine._store, backup_path)
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

    Both SQLite copies finish before replacing the previous pair. Unlike
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
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        flush_engine_connections(engine)
        _backup_pair(engine._store, backup_path)
    except (OSError, sqlite3.Error) as exc:
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


def _backup_pair(store, destination):
    """Stage both SQLite copies before publication; restore the old pair on failure."""
    staging = destination.with_name(destination.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with closing(sqlite3.connect(str(staging))) as dest:
            store.backup(dest)
    except BaseException:
        for path in (staging, Path(str(staging) + ".records")):
            with suppress(OSError):
                path.unlink(missing_ok=True)
        raise
    staged = [Path(str(staging) + ".records"), staging]
    targets = [Path(str(destination) + ".records"), destination]
    previous, published = [], []
    try:
        for source, target in zip(staged, targets):
            saved = target.with_name(target.name + "." + secrets.token_hex(8) + ".previous")
            if target.exists():
                target.replace(saved)
                previous.append((saved, target))
            if source.exists():
                source.replace(target)
                published.append(target)
    except BaseException:
        for target in published:
            target.unlink()
        for saved, target in previous:
            saved.replace(target)
        raise
    for saved, _ in previous:
        with suppress(OSError):
            saved.unlink()
