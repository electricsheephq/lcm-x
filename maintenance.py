"""Backup and rotate maintenance operations for the LCM store.

These are the data-layer maintenance primitives behind ``/lcm backup`` and
``/lcm rotate``: they flush the engine's SQLite connections and snapshot the
store to a timestamped or rolling backup file. They are pure functions that
take the engine so the command layer (``command.py``) keeps only the text
formatting, and the store/dag/lifecycle connection handling lives in one place.
"""

from __future__ import annotations

from contextlib import closing, contextmanager, nullcontext, suppress
from datetime import datetime
from pathlib import Path
import errno
import logging
import os
import sqlite3
from typing import Any


from .externalize import get_large_output_storage_dir
from .store import RecordsFormatError


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
    except (OSError, sqlite3.Error, RecordsFormatError) as exc:
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
    except (OSError, sqlite3.Error, RecordsFormatError) as exc:
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


@contextmanager
def _backup_folder_lock(folder):
    with os.fdopen(os.open(folder / ".lcm-backup.lock", os.O_CREAT | os.O_RDWR, 0o600), "r+b") as lock:
        try:
            import fcntl
        except ImportError:
            logging.getLogger(__name__).warning("Backup locks unavailable; continuing unlocked")
        else:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise OSError("another LCM-X backup in this folder is running") from exc
                logging.getLogger(__name__).warning("Backup lock failed; continuing unlocked: %s", exc)
        yield


def _backup_pair(store, destination):
    with _backup_folder_lock(destination.parent):
        _backup_pair_locked(store, destination)


def _backup_pair_locked(store, destination):
    """Publish records first, then atomically replace the complete canonical slot."""
    staging = destination.with_name(destination.name + ".tmp")
    staged_records = Path(str(staging) + ".records")
    target_records = Path(str(destination) + ".records")
    previous_records = Path(str(target_records) + ".previous")
    previous_tmp = Path(str(previous_records) + ".tmp")
    # Crash states: during staging or previous.tmp copying, the old pair remains.
    # After previous is linked/copied, the old pair also has a recovery records file.
    # After records publish (or unlink for a records-free copy), old lcm.db remains;
    # staging + previous identify its recoverable records until canonical replace.
    # After canonical replace, staging is absent: the new pair is complete and
    # previous is stale. Crashes during cleanup leave that same complete pair.
    # A crash during restart restoration leaves either the recovery file or the
    # restored old pair; clearing other leftovers can then safely be retried.
    # A failed exception-path restore keeps staging + previous for restart recovery.
    if staging.exists() and previous_records.exists():
        previous_records.replace(target_records)
    leftovers = (staging, staged_records, Path(str(staged_records) + ".tmp"),
                 previous_records, previous_tmp)
    for path in leftovers:
        path.unlink(missing_ok=True)
    published_records = False
    keep_recovery = False
    try:
        with closing(sqlite3.connect(str(staging))) as dest:
            store.backup(dest)
        if target_records.exists():
            try:
                previous_records.hardlink_to(target_records)
            except OSError:
                import shutil
                shutil.copyfile(target_records, previous_tmp)
                os.replace(previous_tmp, previous_records)
        if staged_records.exists():
            staged_records.replace(target_records)
            published_records = True
        else:
            target_records.unlink(missing_ok=True)
        staging.replace(destination)
    except BaseException as original:
        try:
            if previous_records.exists():
                previous_records.replace(target_records)
            elif published_records:
                target_records.unlink(missing_ok=True)
        except OSError as restore_error:
            keep_recovery = True
            original.add_note(f"Backup restore failed: {restore_error}; recovery files kept at {staging} and {previous_records}")
        raise
    finally:
        for path in leftovers:
            if keep_recovery and path in (staging, previous_records):
                continue
            with suppress(OSError):
                path.unlink(missing_ok=True)
