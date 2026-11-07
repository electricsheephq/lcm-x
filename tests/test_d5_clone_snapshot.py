"""Snapshot preflight hashes inputs and opens only disposable SQLite copies."""

import hashlib
import json
import sqlite3

import pytest

from bench.instruments.d5_clone import snapshot
from bench.instruments.d5_clone.capture import _readonly


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path):
    root, scratch = tmp_path / "snapshot", tmp_path / "scratch"
    root.mkdir()
    scratch.mkdir()
    db = sqlite3.connect(root / "store.db")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA wal_autocheckpoint=0")
    db.execute("CREATE TABLE messages(store_id INTEGER PRIMARY KEY)")
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.execute("INSERT INTO messages VALUES (7)")
    db.commit()
    (root / "settings.json").write_bytes(b"{}")
    payload = root / "lcm-large-outputs" / "opaque-payload-name"
    payload.parent.mkdir()
    payload.write_bytes(b"synthetic payload")
    pin = tmp_path / "pin.json"
    entries = [{"name": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(root.rglob("*")) if p.is_file()]
    pin.write_text(json.dumps({"schema": "d5-clone-snapshot-pin/1", "files": entries}))
    yield pin, root, scratch
    db.close()


def test_pass_unchanged_and_check_clone(fixture, monkeypatch):
    pin, root, scratch = fixture
    before = {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
              for p in root.rglob("*") if p.is_file()}
    checked = []
    original = snapshot._quick_check

    def check(path):
        assert path.is_relative_to(scratch)
        assert not path.is_relative_to(root)
        assert path != root / "store.db"
        assert path.with_name(path.name + "-wal").exists()
        assert path.with_name(path.name + "-shm").exists()
        with _readonly(path) as db:
            assert db.execute("SELECT store_id FROM messages").fetchall() == [(7,)]
        checked.append(path)
        return original(path)

    monkeypatch.setattr(snapshot, "_quick_check", check)
    result = snapshot.verify(pin, root, scratch)
    assert result["verdict"] == "PASS"
    assert result["entries"] == result["ok"] == len(before)
    assert result["pin_sha256"] == sha(pin)
    assert result["quick_check"] == {"store.db": "ok"}
    assert len(checked) == 1
    assert not checked[0].exists()
    assert list(scratch.iterdir()) == []
    assert before == {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("mode", ["size", "sha", "missing", "extra"])
def test_failures_and_payload_redaction(fixture, mode):
    pin, root, scratch = fixture
    entries = json.loads(pin.read_text())
    payload_name = "lcm-large-outputs/opaque-payload-name"
    if mode in ("size", "sha"):
        for entry in entries["files"]:
            if entry["name"] in ("settings.json", payload_name):
                if mode == "size":
                    entry["size"] += 1
                else:
                    entry["sha256"] = "0" * 64
        pin.write_text(json.dumps(entries))
    elif mode == "missing":
        (root / "settings.json").unlink()
        (root / payload_name).unlink()
    else:
        (root / "extra.json").write_bytes(b"{}")
        (root / "lcm-large-outputs" / "extra-opaque-name").write_bytes(b"extra")
    result = snapshot.verify(pin, root, scratch)
    assert result["verdict"] == "FAIL"
    category = {"size": "size_mismatch", "sha": "sha_mismatch",
                "missing": "missing", "extra": "extra_on_disk"}[mode]
    assert result[category] == {"names": ["extra.json" if mode == "extra" else "settings.json"],
                                "payload": 1}
    assert "opaque" not in json.dumps(result)
    assert list(scratch.iterdir()) == []


def test_quick_check_failure_and_cleanup(fixture, monkeypatch):
    monkeypatch.setattr(snapshot, "_quick_check", lambda path: "failed")
    result = snapshot.verify(*fixture)
    assert result["verdict"] == "FAIL"
    assert result["quick_check"] == {"store.db": "failed"}
    assert list(fixture[2].iterdir()) == []


def test_corrupt_database_is_fail(fixture):
    pin, root, scratch = fixture
    corrupt = root / "broken.db"
    corrupt.write_bytes(b"not a database")
    result = snapshot.verify(pin, root, scratch)
    assert result["verdict"] == "FAIL"
    assert result["quick_check"]["broken.db"] == "failed"
    assert result["extra_on_disk"] == {"names": ["broken.db"], "payload": 0}
    assert list(scratch.iterdir()) == []


def test_empty_pin_fails(tmp_path):
    root, scratch, pin = tmp_path / "snapshot", tmp_path / "scratch", tmp_path / "pin.json"
    root.mkdir()
    scratch.mkdir()
    pin.write_text(json.dumps({"schema": "d5-clone-snapshot-pin/1", "files": []}))
    assert snapshot.verify(pin, root, scratch)["verdict"] == "FAIL"


def _pin_of(root, pin):
    entries = [{"name": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha(p)}
               for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()]
    pin.write_text(json.dumps({"schema": "d5-clone-snapshot-pin/1", "files": entries}))


@pytest.mark.parametrize("kind", ["file", "db", "payload-dir"])
def test_symlinks_fail_and_are_never_followed(fixture, tmp_path, kind):
    pin, root, scratch = fixture
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.db").write_bytes((root / "store.db").read_bytes())
    if kind == "file":
        (root / "linked.bin").symlink_to(outside / "x.db")
    elif kind == "db":
        (root / "other.db").symlink_to(outside / "x.db")
    else:
        (root / "lcm-large-outputs" / "nested").symlink_to(outside, target_is_directory=True)
    _pin_of(root, pin)
    result = snapshot.verify(pin, root, scratch)
    assert result["verdict"] == "FAIL"
    assert result["symlinks"]["names"] or result["symlinks"]["payload"]


def test_duplicate_entries_and_dot_prefixed_payload_names(fixture):
    pin, root, scratch = fixture
    data = json.loads(pin.read_text())
    data["files"].append(dict(data["files"][0]))
    data["files"].append({"name": "./lcm-large-outputs/leakname.txt", "size": 1, "sha256": "0" * 64})
    pin.write_text(json.dumps(data))
    result = snapshot.verify(pin, root, scratch)
    assert result["verdict"] == "FAIL"
    assert result["duplicate_entries"]["names"] or result["duplicate_entries"]["payload"]
    assert "leakname" not in json.dumps(result)


def test_scratch_inside_snapshot_is_refused(fixture):
    pin, root, _ = fixture
    (root / "scratch").mkdir()
    with pytest.raises(ValueError):
        snapshot.verify(pin, root, root / "scratch")
