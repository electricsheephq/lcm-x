"""LCM-X capture hook for Claude Code (and Grok Build, which runs Claude Code plugins).

Every hook event lands here. The contract: read the payload, record which transcript
changed, start the capture worker if none is running, and exit 0 without printing.
The transcript file, not this payload, is the capture source, so a hook that fires
early, late, twice or never cannot lose or double a stored row. Nothing here blocks
the session: PreCompact waits for a bounded drain when no worker holds the store, every
other event returns at once, and no hook waits for the store lock.

Stdlib only and Python 3.9 compatible: hooks run on the system ``python3``.
"""
from __future__ import annotations

import sys
import time  # built in and loaded at start-up, like sys: nothing to compile before the bytecode cache is set

_T0 = time.perf_counter()  # the hook's own clock starts here; its log row times each phase from it (#2164)
_T0_WALL = time.time()


def _apple_bytecode_cache(platform: str, base_prefix: str, prefix: Optional[str],
                          environ: Mapping[bytes, bytes]) -> Optional[Tuple[str, bool]]:
    """Apple's system Python (Xcode or the Command Line Tools) keeps its bytecode cache in
    ~/Library/Caches/com.apple.python only while it reads its environment. The hook's -I turns
    that off and stops writing bytecode, so every start compiled json, re, pathlib and the rest
    from source, which doubled the hook's time (#2141). This gives back the cache and the
    writing that Apple's Python has without -I (none if PYTHONDONTWRITEBYTECODE is set), or
    None: any other Python keeps its own. A relative HOME gets none, so no cache lands in the
    project the hook runs in. A cache it cannot write only costs the compile."""
    home = environ.get(b"HOME")
    if platform == "darwin" and prefix is None and "/Python3.framework/" in base_prefix and home and home.startswith(b"/"):
        cache = home.decode(sys.getfilesystemencoding(), "surrogateescape") + "/Library/Caches/com.apple.python"
        return cache, bool(environ.get(b"PYTHONDONTWRITEBYTECODE"))
    return None


if sys.platform == "darwin":  # before the hook's own imports; __future__ and start-up modules come first
    import posix  # loaded at start-up; os is not

    # pycache_prefix exists from Python 3.8; an older one keeps its own cache.
    _cache = _apple_bytecode_cache(sys.platform, sys.base_prefix, getattr(sys, "pycache_prefix", ""), posix.environ)
    if _cache:
        sys.pycache_prefix, sys.dont_write_bytecode = _cache

import hashlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # before stdin, so A7 allows it
if __name__ != "__main__":
    import lcmx_runtime  # noqa: E402

_IMPORTS_MS = round((time.perf_counter() - _T0) * 1000, 1)

# Every hook pays for its imports before it runs: typing serves the annotations only, and
# subprocess loads only once the hook starts a worker or a bounded drain.
TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Any, Dict, Mapping, Optional, TextIO, Tuple

DETACHED_DEADLINE_S = 30
SYNC_LOCK_WAIT_S = 1.0  # a synchronous hook never waits longer than this for a detached writer's pointer lock
MAX_PAYLOAD = 1024 * 1024
PRECOMPACT_DEADLINE_S = 10
LOG_ROTATE_BYTES = 5 * 1024 * 1024
# PostToolBatch fires once per batch of tool calls; StopFailure replaces Stop when a turn
# ends on an API error. Grok Build has no PostToolBatch, so its PostToolUse is accepted too.
EVENTS = frozenset({
    "SessionStart", "UserPromptSubmit", "PostToolBatch", "PostToolUse", "Stop", "StopFailure",
    "SubagentStop", "PreCompact", "PostCompact", "SessionEnd",
})
DETACHED_EVENTS = EVENTS - {"SessionStart", "PreCompact"}
END_EVENTS = frozenset({"Stop", "StopFailure", "SubagentStop", "SessionEnd", "PreCompact"})
# Registered "async" in hooks.json: Claude Code does not wait for these and may kill one at any
# point, so they never probe the store lock (start_worker).
ASYNC_EVENTS = frozenset({"PostToolBatch"})
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
        # For diagnostics only: nothing orders or acknowledges records by it, so a late async
        # PostToolBatch record that replaces a newer one is still drained.
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
        # Grok Build has no transcript path, and its session-file reader is not in this release:
        # a spooled Grok record would match no lineage, and the worker would delete it having
        # captured nothing. So every Grok event is refused until that reader and its real-session
        # proof land. Its end-of-session Stop and the 1.5 s SessionEnd are observe-only for it.
        event_name = record["event"]
        record["observe_only"] = event_name == "SessionEnd" or (
            event_name == "Stop" and record["reason"] not in (None, "end_turn"))
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


def spool(home: Path, record: Mapping[str, Any], *, wait_s: Optional[float] = None) -> str:
    """Atomically replace this session's (or subagent's) pointer record.

    ``wait_s`` bounds the wait for the pointer's lock (synchronous hooks): if a stalled detached writer of the same
    session holds it that long, nothing is written and ``lock_busy`` is returned. That holder writes its own pointer,
    and the worker this hook still starts sweeps every registered lineage, so the skip delays nothing."""
    directory = home / "spool"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = "\0".join([record["host"], record["session_id"], record.get("agent_id") or ""])
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    import fcntl
    locks = home / "spool-locks"
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(locks / key, "a") as lock:
        if wait_s is None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        else:
            give_up = time.monotonic() + wait_s
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= give_up:
                        return "lock_busy"
                    time.sleep(0.01)
        return _spool_locked(directory, key, record)


def _spool_locked(directory: Path, key: str, record: Mapping[str, Any]) -> str:
    record = dict(record)
    if record.get("event") == "SessionStart" and record.get("source"):
        record["start_source"] = record["source"]
        record["started_at"] = record.get("observed_at")
    else:  # a later hook of the session keeps how and when it started (a fork's copies predate it)
        try:
            with open(directory / (key + ".json"), encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            previous = None
        # SessionStart is synchronous; children retain its fork provenance. Only the transcript
        # names content; local_gc is session environment. The guard preserves end-event ordering.
        if isinstance(previous, dict):
            if previous.get("event") in END_EVENTS and record["event"] not in END_EVENTS:
                return "kept_end"
            if ((previous.get("event") in END_EVENTS) == (record["event"] in END_EVENTS)
                    and previous.get("hook_started_ns", -1) > record.get("hook_started_ns", 0)):
                return "stale"
        if isinstance(previous, dict) and previous.get("start_source"):
            record["start_source"] = previous["start_source"]
            record["started_at"] = previous.get("started_at")
    temporary = directory / (".%s.%d.tmp" % (key, os.getpid()))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
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


class _Steps:
    """The milliseconds each step of one hook run took (#2164). Beginning a step ends the one in
    progress, so no time falls between two steps, and a run that returns early or fails still
    logs the step it was in."""

    def __init__(self, imports_ms: float) -> None:
        self.ms = {"imports": imports_ms}
        self.name = None
        self.since = time.perf_counter()

    def begin(self, name: Optional[str]) -> None:
        now = time.perf_counter()
        if self.name is not None:
            self.ms[self.name] = round((now - self.since) * 1000, 1)
        self.name, self.since = name, now

    def end(self) -> Dict[str, float]:
        self.begin(None)
        return self.ms


def start_worker(home: Path, environ: Mapping[str, str], *, deadline_s: int = 0, probe: bool = True,
                 steps: Optional[_Steps] = None) -> str:
    """Start one detached drain, or with ``deadline_s`` run a bounded drain and wait for it.
    Without ``probe`` a worker starts even when one is running; it then exits at once."""
    steps = _Steps(0.0) if steps is None else steps
    steps.begin("runtime")
    runtime = lcmx_runtime.runtime_command(environ, home)
    if runtime is None:
        return "runtime_missing"
    package = lcmx_runtime.package_root(environ)
    if package is None:
        return "package_missing"
    script = package / "scripts" / "lcm_x_capture.py"
    if not script.is_file():
        return "worker_missing"
    # The record is already spooled. A held lock means a worker that will see it: a detached
    # worker releases the lock first and rescans the spool after, and a bounded one ends with
    # the step below, so no record can fall between. A hook never waits on another worker. The
    # probe itself holds the lock for an instant, and a hook or worker that meets it gives way,
    # so only a hook that will start a worker probes: one that can, and is not async, since a
    # killed async hook could hold the probe and start nothing.
    if probe:
        steps.begin("probe")
        if _lock_held(home / "capture.lock"):
            return "worker_running"
    steps.begin("subprocess")  # its import, apart from starting the worker
    import subprocess

    command = [*runtime, str(script), "worker", "--home", str(home)]
    quiet = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
             "close_fds": True, "cwd": str(home)}
    if deadline_s:
        steps.begin("drain")
        try:
            done = subprocess.run(command + ["--deadline", str(deadline_s)], timeout=deadline_s + 2, **quiet)
            # The worker exits 0 only after a completed drain; 5 means its deadline left work pending,
            # 3 that work is pending for another reason; 4 that another worker took the lock first
            # and holds the record.
            outcome = {0: "drained", 3: "drain_incomplete", 4: "worker_running", 5: "drain_bound"}.get(
                done.returncode, "drain_failed:%d" % done.returncode)
        except subprocess.TimeoutExpired:
            outcome = "drain_timeout"  # the batch in flight rolls back; the next drain resumes from the cursor
        if outcome in ("drained", "drain_failed:2", "worker_running"):  # 2: the store was refused
            return outcome
        # A bounded worker does not rescan: records left behind, including another session's
        # spooled while it held the lock, go to a detached worker now, or to the worker that has
        # taken the lock since, which rescans after it releases it.
        steps.begin("recheck")
        if _lock_held(home / "capture.lock"):
            return outcome + "+holder"
        steps.begin("start")
        subprocess.Popen(command, start_new_session=True, **quiet)
        return outcome + "+worker"
    steps.begin("start")
    subprocess.Popen(command, start_new_session=True, **quiet)
    return "worker_started"


def _log(home: Optional[Path], record: Mapping[str, Any], outcome: str, started: float,
         steps: Optional[_Steps] = None) -> None:
    """One line per hook, without content or paths. ``t0`` is when the hook's own code began, so
    the gap from Claude Code's start of the hook is the start-up of the shell and the interpreter;
    ``phases`` holds the milliseconds of each step that ran, the last being the log's own set-up
    (#2164)."""
    if home is None:
        return
    try:
        if steps is not None:
            steps.begin("log")
        directory = home / "logs"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        log = directory / "hook.log"
        if log.exists() and log.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(log, directory / "hook.log.1")
        line = {"t": round(time.time(), 3), "event": record.get("event"), "host": record.get("host"),
                "session": record.get("session_id"), "outcome": outcome,
                "ms": round((time.monotonic() - started) * 1000, 1)}
        if "blocking_ms" in record:
            line.update(detached=True, blocking_ms=record["blocking_ms"])
        if steps is not None:
            line["t0"] = round(_T0_WALL, 3)
            line["phases"] = steps.end()
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
    except OSError:
        pass


def main(stdin: TextIO = sys.stdin, environ: Mapping[str, str] = os.environ, *, detach: bool = False) -> int:
    global lcmx_runtime
    started, hook_started_ns = time.monotonic(), time.time_ns()
    child, nofork, blocking_ms = False, "", None
    if environ.get("LCMX_CAPTURE", "").strip().lower() in OFF_VALUES:
        return 0
    steps = _Steps(_IMPORTS_MS)
    steps.begin("read")  # the payload: read, parsed and checked
    record: Dict[str, Any] = {}
    outcome = "error"
    home: Optional[Path] = None
    try:
        # A relative home setting gives None: nothing is written anywhere, not even the log.
        raw = stdin.read(MAX_PAYLOAD + 1)
        if len(raw) > MAX_PAYLOAD:
            outcome = "payload_too_large"
            return 0
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            outcome = "payload_not_object"
            return 0
        if detach and _pick(payload, "hook_event_name", "hookEventName") in DETACHED_EVENTS and hasattr(os, "fork"):
            blocking_ms = round((time.monotonic() - started) * 1000, 1)
            try:
                pid = os.fork()
            except OSError:
                nofork = "+nofork"
            else:
                if pid:
                    return 0
                child = True  # set first: the child must always end in os._exit, never return into the caller
                # Leave the hook's group and release Claude Code's pipes before anything that can touch disk.
                os.setsid()
                null = os.open(os.devnull, os.O_RDWR)
                for fd in (0, 1, 2):
                    os.dup2(null, fd)
                os.close(null)
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
                signal.alarm(DETACHED_DEADLINE_S)
                try:  # test-only delay before spooling; a malformed value never costs the record
                    delay_ms = float(environ.get("LCMX_TEST_HOOK_DELAY_MS", "0"))
                except ValueError:
                    delay_ms = 0
                time.sleep(min(500, max(0, delay_ms)) / 1000)
        import lcmx_runtime
        record = normalize(payload, environ)
        record["hook_started_ns"] = hook_started_ns
        home = lcmx_runtime.store_home(environ, record["host"])  # camelCase payloads are Grok's too
        if home is None:
            return 0
        if record.get("problem"):
            outcome = record["problem"]
            return 0
        if record["observe_only"]:
            outcome = "observed"  # never spooled, so it cannot replace a pending record
            return 0
        steps.begin("spool")
        spooled = spool(home, record, wait_s=None if child else SYNC_LOCK_WAIT_S)
        deadline = PRECOMPACT_DEADLINE_S if record["event"] == "PreCompact" else 0
        outcome = start_worker(home, environ, deadline_s=deadline, probe=record["event"] not in ASYNC_EVENTS,
                               steps=steps)
        if spooled != "spooled":
            outcome += "+" + spooled
    except Exception as exc:  # noqa: BLE001 - a hook never fails the session
        outcome = "error:" + type(exc).__name__
    finally:
        try:
            if not (detach and not child and blocking_ms is not None and not nofork):
                if child:
                    record["blocking_ms"] = blocking_ms
                import lcmx_runtime
                home = home or lcmx_runtime.store_home(environ, "grok-build" if environ.get("GROK_SESSION_ID") else "claude-code")
                _log(home, record, outcome + nofork, started, steps)
        finally:
            if child:
                os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(detach=True))
