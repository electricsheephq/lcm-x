"""Read-only SQLite inputs and explicitly retained D5 captures."""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


class CaptureError(ValueError):
    """A capture failed its safety or integrity precondition."""


def _readonly(path):
    """A read-only connection that never writes beside the input. immutable=1 (no locks, no sidecars) is used only for
    a WAL-mode file with neither -wal nor -journal beside it: no connection has it open and nothing is pending. A
    rollback-journal file keeps plain mode=ro, so a held lock reads as "database is locked" and a hot journal fails
    loudly instead of being read past."""
    path = Path(path).absolute()
    with path.open("rb") as stream:
        header = stream.read(20)
    wal_mode = len(header) == 20 and header[18] == 2 and header[19] == 2
    quiet = wal_mode and not any(Path(str(path) + s).exists() for s in ("-wal", "-journal"))
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro" + ("&immutable=1" if quiet else ""), uri=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quick_check(path):
    with closing(_readonly(path)) as db:
        return "ok" if db.execute("PRAGMA quick_check").fetchall() == [("ok",)] else "failed"


def _relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("expected a relative file name")
    return root / path


def backup(src, dest):
    dest = Path(dest)
    with dest.open("xb"):
        pass
    with closing(_readonly(src)) as source, closing(sqlite3.connect(dest)) as target:
        source.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")
    try:
        result = _quick_check(dest)
    except sqlite3.DatabaseError:
        raise CaptureError("backup quick_check failed") from None
    if result != "ok":
        raise CaptureError("backup quick_check failed")


def _holds_symlink(root):
    """True when any entry under root is a symlink; never follows one."""
    return any((Path(directory) / name).is_symlink()
               for directory, dirs, names in os.walk(root, followlinks=False) for name in dirs + names)


def clone_tree(src, dest):
    if Path(dest).exists():
        raise FileExistsError("clone destination exists")
    if sys.platform == "darwin":
        subprocess.run(["cp", "-cR", str(src), str(dest)], check=True)
    else:
        shutil.copytree(src, dest, symlinks=True)  # Copy a link as a link, so the check below sees it.


def capture(kind, out_dir, *, store_db, host_db, payload_dir, extras=None):
    if kind not in ("T0", "H8"):
        raise CaptureError("unknown capture kind")
    out = Path(out_dir)
    out.mkdir()
    backup(store_db, out / "store.db")
    backup(host_db, out / "host-0.db")
    payload = out / "lcm-large-outputs"
    if payload_dir is None:
        payload.mkdir()  # Only an explicit None means "no payload"; a wrong path must never read as empty.
    elif Path(payload_dir).is_symlink() or not Path(payload_dir).is_dir():
        raise CaptureError("payload dir must be an existing real directory")
    elif _holds_symlink(payload_dir):
        raise CaptureError("payload dir holds a symlink")  # Checked before any copy, on every platform.
    else:
        clone_tree(payload_dir, payload)
        if any(path.is_symlink() for path in payload.rglob("*")):
            raise CaptureError("payload dir holds a symlink")  # A link would point the capture at live data.
    for name, source in (extras or {}).items():
        dest = _relative(out, name)
        normalized = dest.relative_to(out).as_posix()
        if dest.exists() or normalized == "capture-hashes.json" or dest.is_relative_to(payload):
            raise CaptureError("extra collides with capture layout")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".db":
            backup(source, dest)
        else:
            with Path(source).open("rb") as reader, dest.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
    files, payload_lines = {}, []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            name, digest = path.relative_to(out).as_posix(), _sha256(path)
            if path.is_relative_to(payload):
                payload_lines.append(f"{name} {digest}\n")
            else:
                files[name] = digest
    captured_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    manifest = dict(schema="d5-clone-capture/1", kind=kind, captured_at=captured_at, files=files,
                    payload_files=len(payload_lines),
                    payload_sha256=hashlib.sha256("".join(payload_lines).encode()).hexdigest())
    hashes = out / "capture-hashes.json"
    hashes.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with closing(_readonly(out / "store.db")) as store, closing(_readonly(out / "host-0.db")) as host:
        return dict(kind=kind, captured_at=captured_at, capture_hashes_sha256=_sha256(hashes),
                    files=len(files), payload_files=len(payload_lines),
                    store_max_store_id=store.execute("SELECT max(store_id) FROM messages").fetchone()[0],
                    host_max_id=host.execute("SELECT max(id) FROM messages").fetchone()[0])


def free_space_ok(path, need_bytes):
    return shutil.disk_usage(path).free >= need_bytes


def remove_capture(out_dir, receipt_written: bool):
    out = Path(out_dir)
    if not receipt_written or not (out / "capture-hashes.json").is_file():
        raise CaptureError("capture removal requires a receipt and capture hashes")
    shutil.rmtree(out)
