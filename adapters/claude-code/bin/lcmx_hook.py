"""LCM-X capture hook for Claude Code (and Grok Build, which runs Claude Code plugins).

Every hook event lands here. The contract: read the payload, record which transcript
changed, start the capture worker if none is running, and exit 0 without printing.
The transcript file, not this payload, is the capture source, so a hook that fires
early, late, twice or never cannot lose or double a stored row. Nothing here blocks
the session: PreCompact waits for a bounded drain, every other event returns at once.

Stdlib only and Python 3.9 compatible: hooks run on the system ``python3``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcmx_runtime  # noqa: E402

MAX_PAYLOAD = 1024 * 1024
PRECOMPACT_DEADLINE_S = 10
LOG_ROTATE_BYTES = 5 * 1024 * 1024
# PostToolBatch fires once per batch of tool calls; StopFailure replaces Stop when a turn
# ends on an API error. Grok Build has no PostToolBatch, so its PostToolUse is accepted too.
EVENTS = frozenset({
    "SessionStart", "UserPromptSubmit", "PostToolBatch", "PostToolUse", "Stop", "StopFailure",
    "SubagentStop", "PreCompact", "PostCompact", "SessionEnd",
})
SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
OFF_VALUES = frozenset({"0", "off", "false", "no"})


def _pick(payload: Mapping[str, Any], snake: str, camel: str) -> Any:
    value = payload.get(snake)
    return payload.get(camel) if value is None else value


def _within(value: Any, root: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        return None
    resolved = path.resolve()
    return resolved if resolved.is_relative_to(root.resolve()) else None


def normalize(payload: Mapping[str, Any], environ: Mapping[str, str]) -> Dict[str, Any]:
    """One record shape for Claude Code's snake_case and Grok Build's camelCase payloads.

    ``problem`` is set when the event must not be spooled; ``observe_only`` marks
    events that are logged only, never spooled.
    """
    grok = bool(environ.get("GROK_SESSION_ID")) or "hookEventName" in payload or "sessionId" in payload
    event = _pick(payload, "hook_event_name", "hookEventName")
    session = _pick(payload, "session_id", "sessionId")
    if session is None and grok:
        session = environ.get("GROK_SESSION_ID")
    record: Dict[str, Any] = {
        "host": "grok-build" if grok else "claude-code",
        "event": event if isinstance(event, str) else None,
        "session_id": session,
        "agent_id": _pick(payload, "agent_id", "agentId"),
        "source": payload.get("source"),
        "reason": payload.get("reason"),
        "transcript_path": None,
        "agent_transcript_path": None,
        "observe_only": False,
        "observed_at": time.time(),
        # The worker acknowledges each record's seq; a record newer than its ack is pending.
        "seq": time.time_ns(),
        # Claude Code's opt-in local GC deletes pre-compaction history from the transcript, so
        # capture cannot be lossless under it; the audit reports such sessions.
        "local_gc": environ.get("CLAUDE_CODE_TRANSCRIPT_LOCAL_GC") == "1",
    }
    if record["event"] not in EVENTS:
        record["problem"] = "unsupported_event"
        return record
    if not isinstance(session, str) or not SESSION_ID.match(session):
        record["problem"] = "bad_session_id"
        return record
    agent = record["agent_id"]
    if agent is not None and (not isinstance(agent, str) or not SESSION_ID.match(agent)):
        record["problem"] = "bad_agent_id"
        return record
    if grok:
        # Grok Build has no transcript path. Its end-of-session Stop and the 1.5 s
        # SessionEnd are logged only. Its session-file reader stays off until
        # its own real-session proof passes.
        event_name = record["event"]
        record["observe_only"] = event_name == "SessionEnd" or (
            event_name == "Stop" and record["reason"] not in (None, "end_turn"))
        if environ.get("LCMX_GROK_CAPTURE") != "1":
            record["problem"] = "grok_capture_off"
        return record
    root = lcmx_runtime.claude_projects_root(environ)
    if root is None:
        record["problem"] = "config_dir_not_absolute"
        return record
    for key in ("transcript_path", "agent_transcript_path"):
        value = payload.get(key)
        if value is None:
            continue
        path = _within(value, root)
        if path is None:
            record["problem"] = key + "_outside_projects"
            return record
        record[key] = str(path)
    if record["transcript_path"] is None:
        record["problem"] = "no_transcript_path"
    return record


def spool(home: Path, record: Mapping[str, Any]) -> str:
    """Atomically replace this session's (or subagent's) pointer record."""
    directory = home / "spool"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = "\0".join([record["host"], record["session_id"], record.get("agent_id") or ""])
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    temporary = directory / (".%s.%d.tmp" % (key, os.getpid()))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(record), handle, sort_keys=True)
    os.replace(temporary, directory / (key + ".json"))
    return "spooled"


def _lock_held(lock: Path) -> bool:
    import fcntl  # POSIX; on other platforms the ImportError is logged and the hook exits 0
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(lock, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def start_worker(home: Path, environ: Mapping[str, str], *, deadline_s: int = 0) -> str:
    """Start one detached drain, or with ``deadline_s`` run a bounded drain and wait for it."""
    python = lcmx_runtime.runtime_python(environ, home)
    if python is None:
        return "runtime_missing"
    package = lcmx_runtime.package_root(environ)
    if package is None:
        return "package_missing"
    script = package / "scripts" / "lcm_x_capture.py"
    if not script.is_file():
        return "worker_missing"
    command = [str(python), str(script), "worker", "--home", str(home)]
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
             "close_fds": True, "cwd": str(home)}
    if deadline_s:
        try:
            done = subprocess.run(command + ["--deadline", str(deadline_s)], timeout=deadline_s + 2, **quiet)
        except subprocess.TimeoutExpired:
            return "drain_timeout"  # the batch in flight rolls back; the next drain resumes from the cursor
        # The worker exits 0 only after a completed drain; 3 means its deadline left work pending.
        return {0: "drained", 3: "drain_incomplete"}.get(done.returncode, "drain_failed:%d" % done.returncode)
    # The record is already spooled. A held lock means a worker that will see it: the worker
    # releases the lock first and rescans the spool after, so no record can fall between.
    if _lock_held(home / "capture.lock"):
        return "worker_running"
    subprocess.Popen(command, start_new_session=True, **quiet)
    return "worker_started"


def _log(home: Optional[Path], record: Mapping[str, Any], outcome: str, started: float) -> None:
    """One line per hook, without content or paths."""
    if home is None:
        return
    try:
        directory = home / "logs"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        log = directory / "hook.log"
        if log.exists() and log.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(log, directory / "hook.log.1")
        line = {"t": round(time.time(), 3), "event": record.get("event"), "host": record.get("host"),
                "outcome": outcome, "ms": round((time.monotonic() - started) * 1000, 1)}
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
    except OSError:
        pass


def main(stdin: TextIO = sys.stdin, environ: Mapping[str, str] = os.environ) -> int:
    started = time.monotonic()
    if environ.get("LCMX_CAPTURE", "").strip().lower() in OFF_VALUES:
        return 0
    record: Dict[str, Any] = {}
    outcome = "error"
    home: Optional[Path] = None
    try:
        # A relative home setting gives None: nothing is written anywhere, not even the log.
        home = lcmx_runtime.store_home(environ, "grok-build" if environ.get("GROK_SESSION_ID") else "claude-code")
        raw = stdin.read(MAX_PAYLOAD + 1)
        if len(raw) > MAX_PAYLOAD:
            outcome = "payload_too_large"
            return 0
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            outcome = "payload_not_object"
            return 0
        record = normalize(payload, environ)
        home = lcmx_runtime.store_home(environ, record["host"])  # camelCase payloads are Grok's too
        if home is None:
            return 0
        if record.get("problem"):
            outcome = record["problem"]
            return 0
        if record["observe_only"]:
            outcome = "observed"  # never spooled, so it cannot replace a pending record
            return 0
        spool(home, record)
        deadline = PRECOMPACT_DEADLINE_S if record["event"] == "PreCompact" else 0
        outcome = start_worker(home, environ, deadline_s=deadline)
    except Exception as exc:  # noqa: BLE001 - a hook never fails the session
        outcome = "error:" + type(exc).__name__
    finally:
        _log(home, record, outcome, started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
