"""Record of a Hermes process in which LCM-X did not become active (#622).

``register()`` leaves a per-process JSON file under Hermes home
``plugin-data/hermes-lcm-x/`` (not in lcm.db: the store open may be slow).
Diagnostic surfaces report it while the process lives; legacy root files are
also read and cleaned for one release. Standalone: no package imports.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

RECORD_DIR = ("plugin-data", "hermes-lcm-x")
RECORD_NAME = "lcm-x-not-active.{pid}.json"
LEGACY_RECORD_NAME = "lcm-x-not-active.json"


def _process_start(pid: int) -> str | None:
    """A start-time token for *pid*, so a reused pid does not match the record."""
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5, env={**os.environ, "LC_ALL": "C"}).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip() or None


def write_inactive_record(hermes_home, *, elapsed_s: float, reason: str) -> None:
    """Atomically write this process's record; one file per process."""
    home = Path(hermes_home).joinpath(*RECORD_DIR)
    home.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    name = RECORD_NAME.format(pid=pid)
    record = {"pid": pid, "process_start": _process_start(pid),
              "elapsed_s": round(float(elapsed_s), 2), "reason": reason, "written_at": time.time()}
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", dir=str(home))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(tmp, home / name)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _process_alive(pid: int, started: str | None) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        pass
    except OSError:
        return False
    current = _process_start(pid)
    return not (started and current and started != current)


def inactive_record_notice(hermes_home) -> str | None:
    """Report live inactive processes; remove unchanged dead records only."""
    if not hermes_home:
        return None
    home = Path(hermes_home)
    try:
        paths = sorted(home.joinpath(*RECORD_DIR).glob(RECORD_NAME.format(pid="*")))
        # legacy root location; remove in v0.28.0 (#1080)
        paths.extend(sorted(home.glob(RECORD_NAME.format(pid="*"))))
        paths.append(home / LEGACY_RECORD_NAME)
    except OSError:
        return None
    notices = []
    for path in paths:
        try:
            content = path.read_bytes()
        except OSError:
            continue
        try:
            record = json.loads(content)
            pid = int(record["pid"])
        except (ValueError, KeyError, TypeError):
            record, pid = {}, 0
        if pid > 0 and _process_alive(pid, record.get("process_start")):
            notice = f"LCM-X was not active in process {pid} ({record.get('reason') or 'unknown reason'})"
            if notice not in notices:
                notices.append(notice)
            continue
        with contextlib.suppress(OSError):
            if path.read_bytes() == content:
                path.unlink()
    return "\n".join(notices) or None
