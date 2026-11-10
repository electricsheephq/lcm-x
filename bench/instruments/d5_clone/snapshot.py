"""Hash the snapshot as files; SQLite only sees disposable scratch copies.

Symlinks anywhere under the snapshot fail the check: a snapshot made of links to a live store must never pass.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from .capture import _quick_check, _relative, _sha256

PAYLOAD = "lcm-large-outputs/"
# The file names a capture writes. Any other name is reported by its sha256 12: receipts carry counts, ids and hashes
# only (release manager, 2026-11-10), and a file added to a snapshot folder can name anything.
KNOWN_NAMES = frozenset(db + s for db in ("store.db", "host-0.db") for s in ("", "-wal", "-shm", "-journal"))


def _raise(error):
    raise error


def _walk(root):
    """(regular files, symlinks) under root, relative POSIX names; never follows a link. A directory that cannot be
    listed raises, so an unread subtree can never pass as empty."""
    files, links = set(), set()
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=_raise):
        for name in dirs + names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                links.add(relative)
            elif name in names and path.is_file():
                files.add(relative)
    return files, links


def _shown(name):
    """A non-payload file name as a receipt may show it: itself when a capture writes it, else its sha256 12."""
    return name if name in KNOWN_NAMES else "sha256-12:" + hashlib.sha256(os.fsencode(name)).hexdigest()[:12]  # fsencode: any name the OS returns


def verify(pin_path, snapshot_dir, scratch_dir):
    root = Path(snapshot_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("snapshot dir must be a real directory")
    if Path(scratch_dir).resolve().is_relative_to(root.resolve()):
        raise ValueError("scratch dir must be outside the snapshot")
    pin = json.loads(Path(pin_path).read_text(encoding="utf-8"))
    if pin["schema"] != "d5-clone-snapshot-pin/1":
        raise ValueError("unknown snapshot pin schema")
    categories = ("size_mismatch", "sha_mismatch", "missing", "extra_on_disk", "symlinks", "duplicate_entries")
    result = dict(verdict="PASS", unreadable=False, pin_sha256=_sha256(pin_path), entries=len(pin["files"]), ok=0,
                  **{c: {"names": [], "payload": 0} for c in categories}, quick_check={})

    def report(category, name):
        if name.startswith(PAYLOAD):
            result[category]["payload"] += 1  # Payload names are never echoed.
        else:
            result[category]["names"].append(_shown(name))
        result["verdict"] = "FAIL"

    if not pin["files"]:
        result["verdict"] = "FAIL"  # A pin that names nothing proves nothing.
    try:
        disk, links = _walk(root)
    except OSError:
        result.update(verdict="FAIL", unreadable=True)  # No message: it could carry a payload path.
        return result
    for name in sorted(links):
        report("symlinks", name)
    pinned = set()
    for entry in pin["files"]:
        name = _relative(root, entry["name"]).relative_to(root).as_posix()
        if name in pinned:
            report("duplicate_entries", name)
            continue
        pinned.add(name)
        if name not in disk:
            report("missing", name)
            continue
        path = root / name
        try:
            size_ok = path.stat().st_size == entry["size"]
            sha_ok = _sha256(path) == entry["sha256"]
        except OSError:
            result.update(verdict="FAIL", unreadable=True)  # No message: it could carry a payload path.
            return result
        if not size_ok:
            report("size_mismatch", name)
        if not sha_ok:
            report("sha_mismatch", name)
        result["ok"] += size_ok and sha_ok
    for name in sorted(disk - pinned):
        report("extra_on_disk", name)
    with tempfile.TemporaryDirectory(prefix="d5-snapshot-", dir=scratch_dir) as scratch:
        for name in sorted(n for n in disk if n.endswith(".db")):
            clone = _relative(Path(scratch), name)
            clone.parent.mkdir(parents=True, exist_ok=True)
            try:
                for suffix in ("", "-wal", "-shm", "-journal"):  # A hot rollback journal must come along.
                    if name + suffix in disk:
                        target = Path(str(clone) + suffix)
                        if sys.platform == "darwin":  # An APFS clone: no full copy of a multi-GB store.
                            subprocess.run(["cp", "-c", str(root / (name + suffix)), str(target)], check=True,
                                           capture_output=True)
                        else:
                            shutil.copy2(root / (name + suffix), target)
                check = _quick_check(clone)
            except (sqlite3.DatabaseError, OSError, subprocess.CalledProcessError):
                check = "failed"  # No message: it could carry a payload path.
            if name.startswith(PAYLOAD):
                result["quick_check"].setdefault("payload", []).append(check)
            else:
                result["quick_check"][_shown(name)] = check
            if check != "ok":
                result["verdict"] = "FAIL"
    return result
