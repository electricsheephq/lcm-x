"""#2232: header probes must preserve the process's SQLite POSIX locks."""

import os
import sqlite3
import subprocess
import sys

import pytest

from hermes_lcm import readonly_store
from hermes_lcm.readonly_store import _wal_header_state, open_readonly_connection

pytestmark = pytest.mark.skipif(os.name != "posix", reason="SQLite POSIX record locks")


def test_header_check_preserves_shared_lock(tmp_path):
    db = tmp_path / "lcm.db"
    writer = sqlite3.connect(db)
    try:
        assert writer.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
        writer.execute("CREATE TABLE messages (content TEXT)")
        writer.commit()
    finally:
        writer.close()

    first = open_readonly_connection(db)
    second = None
    try:
        first.execute("BEGIN")
        first.execute("SELECT count(*) FROM sqlite_master").fetchone()
        second = open_readonly_connection(db)
        # Keep both SQLite connections alive until the other process tries its lock.
        child = subprocess.run(
            [sys.executable, "-c", """
import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1], timeout=0)
try:
    conn.execute("BEGIN EXCLUSIVE")
except sqlite3.OperationalError as exc:
    if str(exc) != "database is locked":
        raise
    print(exc)
else:
    sys.exit("exclusive lock acquired while the reader holds SHARED")
finally:
    conn.close()
""", str(db)],
            capture_output=True, text=True, timeout=10,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert child.stdout.strip() == "database is locked"
    finally:
        first.rollback()
        if second is not None:
            second.close()
        first.close()


@pytest.mark.parametrize("old_format,new_format", [(1, 2), (2, 1)])
def test_header_check_follows_replaced_inode(tmp_path, old_format, new_format):
    db = tmp_path / "lcm.db"
    db.write_bytes(b"SQLite format 3\x00\x10\x00" + bytes((old_format, old_format)))
    inode = db.stat().st_ino
    assert _wal_header_state(db) is (old_format == 2)
    replacement = tmp_path / "replacement.db"
    replacement.write_bytes(b"SQLite format 3\x00\x10\x00" + bytes((new_format, new_format)))
    replacement.replace(db)
    assert db.stat().st_ino != inode
    assert _wal_header_state(db) is (new_format == 2)
    assert _wal_header_state(db) is (new_format == 2)


@pytest.mark.parametrize("old_format,new_format", [(1, 2), (2, 1)])
@pytest.mark.parametrize("replacement_cached", [False, True])
def test_header_check_replacement_between_stat_and_open(
    tmp_path, monkeypatch, old_format, new_format, replacement_cached,
):
    db = tmp_path / "lcm.db"
    replacement = tmp_path / "replacement.db"
    db.write_bytes(b"SQLite format 3\x00\x10\x00" + bytes((old_format, old_format)))
    replacement.write_bytes(b"SQLite format 3\x00\x10\x00" + bytes((new_format, new_format)))
    if replacement_cached:
        assert _wal_header_state(replacement) is (new_format == 2)
    real_open = os.open
    opened = []

    def open_after_replacement(path, flags, *args, **kwargs):
        if path == db and replacement.exists():
            os.replace(replacement, db)
            fd = real_open(path, flags, *args, **kwargs)
            opened.append(fd)
            return fd
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(readonly_store.os, "open", open_after_replacement)
    assert _wal_header_state(db) is (new_format == 2)
    assert opened
    for key, fd in readonly_store._HEADER_FDS.items():
        stat = os.fstat(fd)
        assert key == (stat.st_dev, stat.st_ino)
    if replacement_cached:
        assert opened[0] in readonly_store._HEADER_RETAINED_FDS
    assert _wal_header_state(db) is (new_format == 2)


def test_header_check_without_pread(tmp_path, monkeypatch):
    db = tmp_path / "lcm.db"
    db.write_bytes(b"SQLite format 3\x00\x10\x00\x01\x01")
    monkeypatch.delattr(os, "pread", raising=False)
    assert _wal_header_state(db) is False
