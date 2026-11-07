"""Capture the SQLite WAL and files without modifying input bytes."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime

import pytest

from bench.instruments.d5_clone import capture as cap


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    store, host = tmp_path / "store ?#.db", tmp_path / "host.db"
    connections = [sqlite3.connect(store), sqlite3.connect(host)]
    for db, column in zip(connections, ["store_id", "id"]):
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_autocheckpoint=0")
        db.execute(f"CREATE TABLE messages ({column} INTEGER PRIMARY KEY, session_id TEXT)")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.executemany("INSERT INTO messages VALUES (?, 'synthetic')", [(1,), (7,)])
        db.commit()
    yield store, host
    for db in connections:
        db.close()


def test_backup_wal_readonly_and_exclusive(inputs, tmp_path):
    source = inputs[0]
    wal = source.with_name(source.name + "-wal")
    before = digest(source), digest(wal)
    assert wal.stat().st_size > 0
    dest = tmp_path / "backup.db"
    cap.backup(source, dest)
    with sqlite3.connect(dest) as db:
        assert db.execute("SELECT store_id FROM messages ORDER BY store_id").fetchall() == [(1,), (7,)]
    assert (digest(source), digest(wal)) == before
    before_dest = dest.read_bytes()
    with pytest.raises(FileExistsError):
        cap.backup(source, dest)
    assert dest.read_bytes() == before_dest
    with cap._readonly(source) as db:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM messages")


@pytest.mark.parametrize("kind", ["T0", "H8"])
def test_capture_layout_hashes_and_extras(inputs, tmp_path, kind):
    payload = tmp_path / "payload"
    (payload / "nested").mkdir(parents=True)
    (payload / "nested" / "opaque").write_bytes(b"synthetic payload")
    log = tmp_path / "doctor-source"
    log.write_bytes(b"synthetic diagnostic\r\n")
    filters = tmp_path / "filters-source"
    filters.write_bytes(b'{"count": 0}\n')
    out = tmp_path / kind
    result = cap.capture(kind, out, store_db=inputs[0], host_db=inputs[1], payload_dir=payload,
                         extras={"doctor.log": log, "filters.json": filters, "RC3/store.db": inputs[0]})
    hashes = out / "capture-hashes.json"
    manifest = json.loads(hashes.read_text())
    assert manifest["schema"] == "d5-clone-capture/1"
    assert manifest["kind"] == kind
    assert datetime.fromisoformat(manifest["captured_at"]).utcoffset().total_seconds() == 0
    assert len(manifest["captured_at"].split(".")[1].split("+")[0]) == 6
    assert set(manifest["files"]) == {"store.db", "host-0.db", "doctor.log", "filters.json", "RC3/store.db"}
    for name, sha in manifest["files"].items():
        assert digest(out / name) == sha
    assert (out / "doctor.log").read_bytes() == log.read_bytes()
    assert (out / "filters.json").read_bytes() == filters.read_bytes()
    with sqlite3.connect(out / "RC3/store.db") as db:
        assert db.execute("SELECT max(store_id) FROM messages").fetchone() == (7,)
    payload_sha = digest(payload / "nested" / "opaque")
    expected = hashlib.sha256(f"lcm-large-outputs/nested/opaque {payload_sha}\n".encode()).hexdigest()
    assert manifest["payload_sha256"] == expected
    assert manifest["payload_files"] == 1
    assert "opaque" not in hashes.read_text()
    assert (out / "lcm-large-outputs/nested/opaque").read_bytes() == b"synthetic payload"
    assert result == dict(kind=kind, captured_at=manifest["captured_at"],
                          capture_hashes_sha256=digest(hashes), files=5, payload_files=1,
                          store_max_store_id=7, host_max_id=7)
    with pytest.raises(FileExistsError):
        cap.capture(kind, out, store_db=inputs[0], host_db=inputs[1], payload_dir=payload)
    with pytest.raises(FileExistsError):
        cap.clone_tree(payload, out)


def test_empty_payload_and_removal(inputs, tmp_path):
    out = tmp_path / "empty"
    cap.capture("T0", out, store_db=inputs[0], host_db=inputs[1], payload_dir=None)
    assert list((out / "lcm-large-outputs").iterdir()) == []
    assert json.loads((out / "capture-hashes.json").read_text())["payload_files"] == 0
    with pytest.raises(cap.CaptureError):
        cap.remove_capture(out, receipt_written=False)
    assert out.is_dir()
    cap.remove_capture(out, receipt_written=True)
    assert not out.exists()
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    with pytest.raises(cap.CaptureError):
        cap.remove_capture(unmarked, receipt_written=True)
    assert unmarked.is_dir()


@pytest.mark.parametrize("result", ["failed", "ok\nfailed"])
def test_quick_check_failure(inputs, tmp_path, monkeypatch, result):
    monkeypatch.setattr(cap, "_quick_check", lambda path: result)
    with pytest.raises(cap.CaptureError, match="quick_check"):
        cap.backup(inputs[0], tmp_path / "bad.db")


def test_free_space(monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")(100, 40, 60)
    monkeypatch.setattr(cap.shutil, "disk_usage", lambda path: usage)
    assert cap.free_space_ok(".", 60)
    assert not cap.free_space_ok(".", 61)


def test_missing_or_symlinked_payload_dir_is_refused(inputs, tmp_path):
    with pytest.raises(cap.CaptureError):  # A mistyped path must never read as "no payload".
        cap.capture("T0", tmp_path / "typo", store_db=inputs[0], host_db=inputs[1], payload_dir=tmp_path / "nope")
    real = tmp_path / "real-payload"
    real.mkdir()
    (real / "a.json").write_text("{}")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(cap.CaptureError):
        cap.capture("T0", tmp_path / "out1", store_db=inputs[0], host_db=inputs[1], payload_dir=tmp_path / "linked")
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (real / "inner-link.json").symlink_to(outside)
    with pytest.raises(cap.CaptureError):
        cap.capture("T0", tmp_path / "out2", store_db=inputs[0], host_db=inputs[1], payload_dir=real)


def test_dotted_extra_name_cannot_overwrite_the_manifest(inputs, tmp_path):
    extra = tmp_path / "extra.txt"
    extra.write_text("x")
    with pytest.raises(cap.CaptureError):
        cap.capture("T0", tmp_path / "out", store_db=inputs[0], host_db=inputs[1], payload_dir=None,
                    extras={"./capture-hashes.json": extra})


def test_inner_symlink_is_refused_on_the_portable_copy_path(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(cap.sys, "platform", "linux")  # shutil.copytree would copy the link's target as a file.
    real = tmp_path / "payload"
    real.mkdir()
    (tmp_path / "outside.json").write_text("{}")
    (real / "inner-link.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(cap.CaptureError):
        cap.capture("T0", tmp_path / "out", store_db=inputs[0], host_db=inputs[1], payload_dir=real)
    assert not (tmp_path / "out" / "lcm-large-outputs" / "inner-link.json").exists()


def test_reading_a_stopped_wal_input_writes_no_sidecar(tmp_path):
    db = tmp_path / "stopped.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
    assert not Path(str(db) + "-wal").exists()  # Closed cleanly: checkpointed, no sidecars.
    with closing(cap._readonly(db)) as conn:
        assert conn.execute("SELECT count(*) FROM t").fetchone() == (1,)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stopped.db"]


def _hot_journal_copy(tmp_path, name="hot.db"):
    """A rollback-journal store copied mid-write: its main file holds uncommitted pages, its -journal is hot."""
    live, copy = tmp_path / "live", tmp_path / "copy"
    live.mkdir()
    copy.mkdir()
    with closing(sqlite3.connect(live / name, isolation_level=None)) as conn:
        conn.execute("CREATE TABLE t (x TEXT)")
        conn.executemany("INSERT INTO t VALUES (?)", [("a" * 900,)] * 400)
        conn.execute("PRAGMA cache_size=1")
        conn.execute("BEGIN")
        conn.execute("UPDATE t SET x = ?", ("b" * 900,))  # The tiny cache spills pages into the main file.
        assert Path(str(live / name) + "-journal").exists()
        for suffix in ("", "-journal"):
            (copy / (name + suffix)).write_bytes((live / (name + suffix)).read_bytes())
        conn.execute("ROLLBACK")
    assert "b" * 900 in (copy / name).read_bytes().decode("latin-1")  # Uncommitted content reached the copy.
    return copy / name


def test_a_hot_rollback_journal_fails_loudly_instead_of_being_read_past(tmp_path):
    source = _hot_journal_copy(tmp_path)
    with pytest.raises(sqlite3.DatabaseError):
        cap.backup(source, tmp_path / "out.db")


def test_a_locked_rollback_journal_input_is_not_read_without_its_lock(tmp_path):
    db = tmp_path / "locked.db"
    with closing(sqlite3.connect(db, isolation_level=None)) as writer:
        writer.execute("CREATE TABLE t (x)")
        writer.execute("BEGIN EXCLUSIVE")
        writer.execute("INSERT INTO t VALUES (1)")
        with closing(cap._readonly(db)) as reader, pytest.raises(sqlite3.OperationalError, match="locked"):
            reader.execute("SELECT count(*) FROM t").fetchone()
        writer.execute("ROLLBACK")


def test_a_symlinked_database_input_is_refused_before_any_read(tmp_path):
    real = tmp_path / "real.db"
    with closing(sqlite3.connect(real)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("CREATE TABLE t (x)")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        assert Path(str(real) + "-wal").stat().st_size > 0  # A committed row lives only in the target's WAL.
        link = tmp_path / "link.db"
        link.symlink_to(real)
        with pytest.raises(cap.CaptureError, match="symlink"):
            cap._readonly(link)
        with pytest.raises(cap.CaptureError, match="symlink"):
            cap.backup(link, tmp_path / "out.db")
