"""Hash the snapshot as files; SQLite only sees disposable scratch copies."""

import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from .capture import _quick_check, _relative, _sha256


def verify(pin_path, snapshot_dir, scratch_dir):
    root = Path(snapshot_dir)
    pin = json.loads(Path(pin_path).read_text(encoding="utf-8"))
    if pin["schema"] != "d5-clone-snapshot-pin/1":
        raise ValueError("unknown snapshot pin schema")
    result = dict(verdict="PASS", pin_sha256=_sha256(pin_path), entries=len(pin["files"]), ok=0,
                  size_mismatch={"names": [], "payload": 0}, sha_mismatch={"names": [], "payload": 0},
                  missing={"names": [], "payload": 0}, extra_on_disk={"names": [], "payload": 0},
                  quick_check={})

    def report(category, name):
        if name.startswith("lcm-large-outputs/"):
            result[category]["payload"] += 1
        else:
            result[category]["names"].append(name)
        result["verdict"] = "FAIL"

    if not pin["files"]:
        result["verdict"] = "FAIL"  # A pin that names nothing proves nothing.
    pinned = set()
    for entry in pin["files"]:
        name = entry["name"]
        path = _relative(root, name)
        pinned.add(path.relative_to(root).as_posix())
        if not path.is_file():
            report("missing", name)
            continue
        size_ok = path.stat().st_size == entry["size"]
        sha_ok = _sha256(path) == entry["sha256"]
        if not size_ok:
            report("size_mismatch", name)
        if not sha_ok:
            report("sha_mismatch", name)
        result["ok"] += size_ok and sha_ok
    disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    for name in sorted(disk - pinned):
        report("extra_on_disk", name)
    with tempfile.TemporaryDirectory(prefix="d5-snapshot-", dir=scratch_dir) as scratch:
        for name in sorted(n for n in disk if n.endswith(".db")):
            clone = _relative(Path(scratch), name)
            clone.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("", "-wal", "-shm"):
                source = _relative(root, name + suffix)
                if source.is_file():
                    target = Path(str(clone) + suffix)
                    if sys.platform == "darwin":  # An APFS clone: no full copy of a multi-GB store.
                        subprocess.run(["cp", "-c", str(source), str(target)], check=True)
                    else:
                        shutil.copy2(source, target)
            try:
                check = _quick_check(clone)
            except sqlite3.DatabaseError:
                check = "failed"
            if name.startswith("lcm-large-outputs/"):
                result["quick_check"].setdefault("payload", []).append(check)
            else:
                result["quick_check"][name] = check
            if check != "ok":
                result["verdict"] = "FAIL"
    return result
