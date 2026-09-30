"""Record of a Hermes process in which LCM-X did not become active (#622).

That process has no ``/lcm`` command or LCM hooks, so ``register()`` leaves this
JSON file under the Hermes home (not in lcm.db: the store open may be the slow
part). ``lcm_status`` and ``/lcm doctor`` in other processes report it while the
recorded process lives. Standalone: no package imports.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

RECORD_NAME = "lcm-x-not-active.json"


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
    """Atomically write this process's record; one file per profile."""
    home = Path(hermes_home)
    home.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "process_start": _process_start(os.getpid()),
              "elapsed_s": round(float(elapsed_s), 2), "reason": reason, "written_at": time.time()}
    fd, tmp = tempfile.mkstemp(prefix=f".{RECORD_NAME}.", dir=str(home))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(tmp, home / RECORD_NAME)
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
    """``LCM-X was not active in process <pid> (<reason>)`` while that pid lives; else drop the record."""
    if not hermes_home:
        return None
    path = Path(hermes_home) / RECORD_NAME
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        pid = int(record["pid"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        record, pid = {}, 0
    if pid > 0 and _process_alive(pid, record.get("process_start")):
        return f"LCM-X was not active in process {pid} ({record.get('reason') or 'unknown reason'})"
    with contextlib.suppress(OSError):
        path.unlink()
    return None
