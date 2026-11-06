"""Opt-in event recorder (#2243): the host calls into LCM, what LCM returns, and the host state LCM reads.

``register()`` imports this module only when ``LCM_RECORDER_ENABLED`` is true. Recording uses
``sys.monitoring`` (Python 3.12+) on the code objects of the host-facing roots, so no callable is wrapped and
no frame is added. Callbacks never raise into, return a value for, or disable the code they observe. Files are
local: ``<dir>/proc/<pid>-<start>-<db8>/`` (events, snapshot, checkpoints, STATUS.json) and ``<dir>/blobs/``.
See https://github.com/electricsheephq/lcm-x/blob/main/docs/recorder.md; ``recorder_read.read`` rebuilds a recording.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib
import inspect
import itertools
import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import threading
import time
import weakref
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
TOOL_NAME = "lcm-recorder"
_SLOT = "_lcm_recorder_process"  # one recorder per process, shared by every loaded copy of the plugin
_MIN_PYTHON = (3, 12)
# Public callables of Hermes agent.context_engine.ContextEngine at the bench pin (71d34bad), used when the
# host class cannot be imported; ``name`` is a property and is never a root.
HOST_ENGINE_METHODS = (
    "clone_for_agent", "compress", "get_automatic_compaction_status_message", "get_status", "get_tool_schemas",
    "handle_tool_call", "has_content_to_compress", "on_session_end", "on_session_reset", "on_session_start",
    "on_turn_complete", "prune_tool_results_only", "select_context", "should_compress", "should_compress_info",
    "should_compress_preflight", "should_defer_preflight_to_real_usage", "update_from_response", "update_model",
)
# LCM names Hermes calls beyond ContextEngine (bench pin: run_agent.py:375, :422; conversation_compression.py:1563,
# :1569, :3248, :3515; turn_context_compaction.py:271); "where present" on the engine class.
EXTRA_ENGINE_ROOTS = (
    "bind_session_state", "clone_for_agent", "shutdown", "record_rejected_compaction",
    "_automatic_compression_blocked", "_compression_block_reason", "record_completed_compaction",
    "maybe_seed_preflight_display_tokens", "carry_over_new_session_context",
)
CTX_REGISTRARS = (("register_hook", "hook"), ("register_tool", "tool"), ("register_command", "command"),
                  ("register_system_prompt_section", "prompt_section"))
ATEXIT_ROOTS = ("_flush_pending_tails_at_exit", "_quiesce_embedding_maintenance")
# Engine attributes Hermes assigns (bench pin: conversation_compression.py:176-189, :414, :1970-1982; codex_runtime.py
# :192, :233-243; agent_init.py:1758, :1981; turn_usage.py:171-172; turn_finalizer.py:314). Read from __dict__ only.
HOST_ATTRS = (
    "threshold_tokens", "threshold_percent", "tail_token_budget", "_tail_token_budget", "_aux_context_ceiling",
    "last_prompt_tokens", "last_completion_tokens", "last_compression_rough_tokens",
    "awaiting_real_usage_after_compression", "_last_compress_refused_would_grow", "_compression_cancelled_check",
    "_compression_cancelled_check_owner", "_anthropic_rejected_thinking", "compression_count", "summary_model",
    "context_length", "model_thresholds", "_config_context_length", "_context_probed", "_context_probe_persistable",
    "_verify_compaction_cleared_threshold", "_proactive_prune_rearm_tokens", "_flush_scan_cursor_invalidated",
    "_compression_attempt_generation", "_compression_working_attempt_generation", "_compression_attempt_serial_lock",
    "_previous_summary", "_last_summary_error", "_last_summary_fallback_used", "_summary_failure_cooldown_until",
    "_last_compression_telemetry", "_last_aux_model_failure_error", "_last_aux_model_failure_model",
    "_fallback_compression_streak", "_last_compression_status", "_last_compression_noop_reason",
)
# Engine-generated values a replay cannot reproduce, recorded once per engine (engine.py:921).
SEED_ATTRS = ("_compaction_telemetry_counter_epoch",)
SET_ONLY_NAMES = frozenset({"api_key", "base_url", "session_db"})
_SECRET_MARKS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
AUX_FIELDS = ("session_id", "_parent_session_id", "_delegate_role", "_subagent_id", "_parent_subagent_id",
              "_delegate_depth", "_memory_write_origin", "_memory_write_context", "log_prefix", "enabled_toolsets",
              "ephemeral_system_prompt")
AUX_DEPTH = 48
SESSION_COLUMNS = ("id", "source", "parent_session_id", "started_at", "ended_at", "end_reason", "model_config")
COMMAND_CONTEXT = ("HERMES_SESSION_ID", "HERMES_SESSION_KEY", "HERMES_SESSION_PROFILE")
MAX_QUEUE_EVENTS, MAX_QUEUE_BYTES = 10_000, 512 << 20
DEFAULT_MAX_BYTES, DISK_FLOOR = 20 << 30, 20 << 30
DIAG_RESERVE = 64 << 10
CHECKPOINT_HOURS = (24, 72, 144)
_SYNC_ROOTS = frozenset({"on_session_end", "shutdown"})
_CO_SKIP = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR | inspect.CO_ITERABLE_COROUTINE
_LOCK_TYPES = tuple({type(threading.Lock()), type(threading.RLock()), threading.Condition, threading.Event,
                     threading.Semaphore})
_MISSING = object()
clock = time.monotonic  # tests inject checkpoint time through this


def is_secret_name(name: str) -> bool:
    """Credential-valued by name: a whole ``_`` segment is a mark (``*_API_KEY`` is, ``*_MAX_TOKENS`` is not)."""
    return name in SET_ONLY_NAMES or any(part in _SECRET_MARKS for part in str(name).upper().split("_"))


def _hermes_home(engine) -> Path:
    from .config import lcm_env_get

    home = engine.__dict__.get("_hermes_home") or lcm_env_get("HERMES_HOME") or "~/.hermes"
    return Path(home).expanduser().resolve()


class Tag:
    """A value recorded by kind: engine token, set/unset, unsupported, uncapturable or a set's items."""
    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    def __eq__(self, other):
        return isinstance(other, Tag) and (self.kind, self.value) == (other.kind, other.value)

    __hash__ = None


class Blob:
    """A structural copy the writer stores once by content hash."""
    __slots__ = ("copy",)

    def __init__(self, copy):
        self.copy = copy


class View:
    """Per-position row ops: ("s", t) same, ("e", t, blob) same+edited, ("c", t, of) clone, ("n", t, blob) new."""
    __slots__ = ("ops",)

    def __init__(self, ops):
        self.ops = ops


def plain(value, blob=None):
    """JSON form of a recorded value; ``blob`` stores a Blob and returns its reference."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"$f": repr(value)}
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in value):
            return {k: plain(v, blob) for k, v in value.items()}
        return {"$d": [[plain(k, blob), plain(v, blob)] for k, v in value.items()]}
    if isinstance(value, list):
        return [plain(v, blob) for v in value]
    if isinstance(value, tuple):
        return {"$t": [plain(v, blob) for v in value]}
    if isinstance(value, Tag):
        return {"$" + value.kind: plain(value.value, blob)}
    if isinstance(value, Blob):
        return blob(value) if blob is not None else {"$row": plain(value.copy)}
    if isinstance(value, View):
        return {"$v": [[op[0], op[1], plain(op[2], blob)] if len(op) > 2 else list(op) for op in value.ops]}
    raise TypeError(f"unrecordable {type(value).__name__}")


def canonical(value) -> bytes:
    """Compact JSON in the value's own key order: LCM stores rows' nested JSON in that order."""
    return json.dumps(plain(value), separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _row_copy(value, marks):
    """Structural copy that shares strings and scalars; foreign types become ``unsupported`` tags."""
    if isinstance(value, dict):
        return {k: _row_copy(v, marks) for k, v in value.items()}
    if isinstance(value, list):
        return [_row_copy(v, marks) for v in value]
    if isinstance(value, tuple):
        return tuple(_row_copy(v, marks) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    marks.append("unsupported")
    return Tag("unsupported", type(value).__name__)


def _git_sha(root: Path) -> str:
    try:
        git = root / ".git"
        if git.is_file():
            git = Path(git.read_text().split("gitdir:", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head[4:].strip()
        common = Path((git / "commondir").read_text().strip()) if (git / "commondir").exists() else git
        common = common if common.is_absolute() else git / common
        for base in (git, common):
            if (base / ref).exists():
                return (base / ref).read_text().strip()
        for line in (common / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except Exception:
        pass
    return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mkdir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _pin(db_path: Path):
    """Open a separate read connection and hold a read snapshot: ``BEGIN`` plus one row (None: no database)."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False, timeout=5)
    try:
        conn.execute("BEGIN")
        conn.execute("SELECT * FROM sqlite_master LIMIT 1").fetchall()
    except BaseException:
        conn.close()
        raise
    return conn


def _disk_room(target: Path, need: int) -> bool:
    return shutil.disk_usage(target).free - need >= DISK_FLOOR


def _backup(conn, dest: Path) -> dict:
    target = None
    try:
        _mkdir(dest.parent)
        if conn is None:
            return {"path": None, "state": "no_database"}
        target = sqlite3.connect(dest)
        target.execute("PRAGMA journal_mode=OFF")  # disposable copy: no unbudgeted rollback journal
        conn.backup(target, pages=-1)
        os.chmod(dest, 0o600)
        return {"path": str(dest), "state": "ok", "bytes": dest.stat().st_size}
    finally:
        try:
            if target is not None:
                target.close()
        finally:
            _close_pin(conn)


def _close_pin(conn):
    if conn is not None:
        try:
            conn.rollback()
        except sqlite3.ProgrammingError:  # already closed by _backup
            pass
        finally:
            conn.close()


class _Stopped(Exception):
    """A planned recorder write was refused; never escapes into the host."""


_SESSION_SELECT = f"SELECT {', '.join(SESSION_COLUMNS)} FROM sessions"
_SESSION_SHAPE = "SELECT count(*), max(rowid), (SELECT id FROM sessions ORDER BY rowid DESC LIMIT 1) FROM sessions"
_PARENT = SESSION_COLUMNS.index("parent_session_id")
_CHAIN_ROWS = 258  # host_uid.py:122 reads a session and up to _MAX_LINEAGE_HOPS + 1 (257) parents


class _StateDb:
    """The Hermes ``sessions`` projection through one persistent read-only connection.

    LCM reads ``sessions`` by id along a session's parent chain (host_uid.py:118, identity_anchor.py:95-99,
    aux_session.py:713-727, :783 and :1006, engine.py:2602-2610) and reads every id only for lifecycle stats
    (lifecycle_state.py:547). So a root re-reads the parent chains of its sessions when the file changed since
    they were read, and a change also records the rows inserted since the last look. A deleted or reused rowid
    falls back to the full diff. Each look is one read transaction, so its rows and shape agree.
    """

    def __init__(self, path: Path):
        self.path, self.conn, self.version, self.last = path, None, None, {}
        self.shape, self.read_at, self.ident = None, {}, None

    def _connect(self):
        if self.conn is None:
            if not self.path.exists():
                return False
            st = os.stat(self.path)
            self.conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True,
                                        check_same_thread=False, timeout=1.0)
            self.ident = (st.st_dev, st.st_ino)
        return True

    def _same_file(self):
        """Whether the open connection still reads the file at ``path`` (not one replaced, deleted or made unreadable)."""
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        return (st.st_dev, st.st_ino) == self.ident and os.access(self.path, os.R_OK)

    def _read(self, look):
        self.conn.execute("BEGIN")
        try:
            return look(self.conn.execute("PRAGMA data_version").fetchone()[0])
        finally:
            self.conn.execute("COMMIT")

    def _all(self, version):
        rows = self.conn.execute(_SESSION_SELECT).fetchall()
        self.shape, self.version = self.conn.execute(_SESSION_SHAPE).fetchone(), version
        self.read_at = dict.fromkeys((row[0] for row in rows), version)
        return rows

    def full(self):
        try:
            if not self._connect():
                return "no_database"
            rows = self._read(self._all)
        except Exception:
            self.conn = None
            return "read_error"
        self.last = {row[0]: row for row in rows}
        return [list(row) for row in rows]

    def changes(self, session_ids):
        """Rows that differ from their last recorded outcome: new rows, and the parent chains of ``session_ids``."""
        had = self.conn is not None
        try:
            if had and not self._same_file():
                self.conn, had = None, False
            if not self._connect():
                changed = {sid: "no_database" for sid in session_ids if self.last.get(sid) != "no_database"}
                self.last.update(changed)
                return changed or None
            if not had:
                return {"$full": self.full()}
            version = self.conn.execute("PRAGMA data_version").fetchone()[0]
            if version == self.version and all(self._fresh(sid, version) for sid in session_ids):
                return None
            return self._read(lambda now: self._look(now, session_ids)) or None
        except Exception:
            self.conn = None  # a failed look may have marked rows recorded: the next call reads every row
            changed = {sid: "read_error" for sid in session_ids if sid and self.last.get(sid) != "read_error"}
            self.last.update(changed)
            return changed or None

    def _fresh(self, sid, version):
        """Whether every row on ``sid``'s parent chain was read at ``version`` (an inserted row is fresh before its parent)."""
        hops = set()
        while sid and sid not in hops and len(hops) < _CHAIN_ROWS:
            if self.read_at.get(sid) != version:
                return False
            hops.add(sid)
            value = self.last.get(sid)
            sid = value[_PARENT] if isinstance(value, tuple) else None
        return True

    def _look(self, version, session_ids):
        out = {}
        if version != self.version:
            shape = self.conn.execute(_SESSION_SHAPE).fetchone()
            if shape != self.shape:
                count, top, top_id = self.shape or (0, None, None)
                grown = self.conn.execute(_SESSION_SELECT + " WHERE rowid > ?", (top or 0,)).fetchall()
                kept = top is None or self.conn.execute(
                    "SELECT id FROM sessions WHERE rowid = ?", (top,)).fetchone() == (top_id,)
                if kept and shape[0] == count + len(grown):
                    self.shape = shape
                    self.read_at.update((row[0], version) for row in grown)
                else:  # a row was deleted, or a rowid reused
                    grown = self._all(version)
                    out.update((sid, "no_row") for sid in self.last.keys() - {row[0] for row in grown})
                    self.last = {sid: value for sid, value in self.last.items() if sid in self.read_at}
                for row in grown:
                    if row != self.last.get(row[0]):
                        self.last[row[0]], out[row[0]] = row, list(row)
            self.version = version
        for sid in session_ids:
            hops = set()
            while sid and sid not in hops and len(hops) < _CHAIN_ROWS:
                hops.add(sid)
                if self.read_at.get(sid) == version:
                    value = self.last.get(sid)
                else:
                    row = self.conn.execute(_SESSION_SELECT + " WHERE id = ?", (sid,)).fetchone()
                    value, self.read_at[sid] = (row if row is not None else "no_row"), version
                    if value != self.last.get(sid):
                        self.last[sid], out[sid] = value, list(row) if row is not None else value
                sid = value[_PARENT] if isinstance(value, tuple) else None
        return out


class _Entry:
    __slots__ = ("frame", "code", "rec", "engine", "root", "name", "t0", "views", "touched")

    def __init__(self, frame, code, rec, engine, root, name):
        self.frame, self.code, self.rec, self.engine, self.root, self.name = frame, code, rec, engine, root, name
        self.t0, self.views, self.touched = clock(), 0, set()


class _Writer(threading.Thread):
    """Serializes, hashes and appends on its own thread; admission never blocks the caller."""

    def __init__(self, rec):
        super().__init__(name="lcm-recorder-writer", daemon=True)
        self.rec, self.items, self.cv = rec, deque(), threading.Condition()
        self.events = self.bytes = 0
        self.io = threading.RLock()
        self.sync = False
        self.segment = self.segment_name = None
        self.segment_bytes, self.total_bytes = 0, 0
        self.last_fsync = self.last_status = 0.0
        self.known_blobs = set()
        self.dead = None
        self.closed = False
        self.stall = None  # tests hold the writer on an Event
        self.backups = 0
        self.budget = threading.RLock()

    def offer(self, item, nbytes=256, force=False) -> bool:
        if self.dead is not None or self.closed:
            return False
        if self.sync:
            if item[0] == "backup":
                with self.cv:
                    self.backups += 1
            with self.io:
                self._process(item)
            return True
        with self.cv:
            if item[0] == "backup":
                self.backups += 1
            if not force and (self.events >= MAX_QUEUE_EVENTS or (self.events and self.bytes + nbytes > MAX_QUEUE_BYTES)):
                return False
            self.items.append((item, nbytes))
            self.events += 1
            self.bytes += nbytes
            self.cv.notify()
        return True

    def drain(self, timeout: float, skip_backup=False) -> bool:
        if self.dead is not None or self.closed or (skip_backup and self.backups):
            return False
        self.rec._flush_gone()
        done = threading.Event()
        if not self.offer(("sync", done), 0, force=True):
            return False
        deadline = time.monotonic() + timeout
        while not done.is_set():
            if skip_backup and self.backups:
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            done.wait(min(remaining, 0.01) if skip_backup else remaining)
        return True

    def run(self):
        try:
            self._run()
        except BaseException as exc:
            self.rec.writer_failed(("run", None), exc)
            self.closed = True
            try:
                self.rec.write_status()
            except BaseException:
                pass
        finally:
            with self.cv:
                for item, _size in self.items:
                    if item[0] == "sync":
                        item[1].set()
                    elif item[0] == "event":
                        self.rec.counters["dropped_after_stop"] += 1
                self.items.clear()
                self.events = self.bytes = self.backups = 0
            for cleanup in (self.rec._close_pins, self._close_state,
                            lambda: self.segment.close() if self.segment is not None else None,
                            self.rec.write_status, self.rec._log_pending):
                try:
                    cleanup()
                except BaseException as exc:
                    try:
                        self.rec.writer_failed(("cleanup", None), exc)
                    except BaseException:
                        pass

    def _close_state(self):
        with self.rec.lock:
            if self.rec.state_db is not None and self.rec.state_db.conn is not None:
                conn, self.rec.state_db.conn = self.rec.state_db.conn, None
                conn.close()

    def _run(self):
        while not self.closed:
            with self.cv:
                if not self.items:
                    self.cv.wait(1.0)
                item = self.items.popleft() if self.items else None
            if self.stall is not None:
                self.stall.wait()
            if item is not None:
                with self.io:
                    self._process(item[0])
                with self.cv:
                    self.events -= 1
                    self.bytes -= item[1]
            self.rec._log_pending()
            with self.io:
                try:
                    self._tick()
                except Exception as exc:
                    self.rec.writer_failed(("tick", None), exc)

    def _tick(self, force=False):
        now = time.monotonic()
        # Diagnostics may use the reserve; only an actual data write can stop recording.
        self.rec.cap()
        if self.segment is not None and (force or now - self.last_fsync >= 1.0):
            self.segment.flush()
            os.fsync(self.segment.fileno())
            self.last_fsync = now
        if force or now - self.last_status >= 1.0:
            self.last_status = now
            self.rec.write_status()

    def _process(self, item):
        kind = item[0]
        if kind == "event" and (self.dead is not None or self.rec.stopped is not None):
            self.rec.counters["dropped_after_stop"] += 1
            return
        try:
            if kind in ("event", "diagnostic"):
                self._append(item[1], diagnostic=kind == "diagnostic")
            elif kind in ("call", "backup"):
                item[1]()
            elif kind == "sync":
                self._tick(force=True)
                self.rec._log_pending()
                item[1].set()
        except _Stopped:
            if kind == "event":
                self.rec.counters["dropped_after_stop"] += 1
        except Exception as exc:
            self.rec.writer_failed(item, exc)
            if kind == "sync":
                item[1].set()
        finally:
            if kind == "backup":
                with self.cv:
                    self.backups -= 1

    def _blob(self, blob):
        data = canonical(blob.copy)
        sha = hashlib.sha256(data).hexdigest()
        if sha not in self.known_blobs:
            path = self.rec.root / "blobs" / sha[:2] / sha
            if not path.exists():
                self.rec._reserve(len(data))
                _mkdir(path.parent)
                tmp = path.with_name(f"{sha}.{os.getpid()}.tmp")
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
                for directory in (path.parent, path.parent.parent):
                    fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            self.known_blobs.add(sha)
        return {"$b": sha}

    def _append(self, event, *, diagnostic=False):
        line = json.dumps(plain(event, self._blob), separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
        data = line.encode()
        if not self.rec._admit(len(data), diagnostic=diagnostic):
            raise _Stopped()
        name = time.strftime("%Y%m%d", time.gmtime())
        if self.segment is None or not self.segment_name.startswith(name) or self.segment_bytes > 1 << 30:
            self._open_segment(name)
        self.segment.write(data)
        self.segment_bytes += len(data)
        if self.sync:
            self._tick(force=True)

    def _open_segment(self, day):
        if self.segment is not None:
            self.segment.flush()
            os.fsync(self.segment.fileno())
            self.segment.close()
        events = _mkdir(self.rec.proc / "events")
        for n in itertools.count(1):
            candidate = events / f"{day}-{n:02d}.jsonl"
            if not candidate.exists():
                break
        fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_EXCL, 0o600)
        self.segment, self.segment_name, self.segment_bytes = os.fdopen(fd, "ab"), candidate.name, 0


class Recorder:
    def __init__(self, tool: int, root: Path, engine, ctx):
        self.tool, self.root = tool, root
        self.lock = threading.RLock()
        self.opened = False
        self.seq = itertools.count(1)
        self.last_seq = 0
        self.lost = 0
        self.counters = Counter()
        self.marks: dict = {}
        self.first_cause = None
        self.stopped = None
        self.end_state = "running"
        self.engines = weakref.WeakKeyDictionary()
        self.home_changed = weakref.WeakSet()
        self.engine_ids = itertools.count(1)
        self.engine_classes: tuple = ()
        self.threads = weakref.WeakKeyDictionary()
        self.thread_ids = itertools.count(1)
        self.callers = weakref.WeakKeyDictionary()
        self.caller_ids = itertools.count(1)
        self.local = threading.local()
        # Keyed by id(): code objects compare by value, so two loaded copies of one source would share a key.
        self.roots: dict = {}  # id(code) -> [kind, name, shared, code]
        self.callbacks: dict = {}  # id(code) -> [(kind, name, closure values)]
        self.skipped: list = []
        self.rows: dict = {}  # id(row) -> [token, row, copy]
        self.by_content: dict = {}
        self.row_ids = itertools.count(1)
        self.views: dict = {}
        self.view_owners: dict = {}
        self.gone = deque()
        self.warnings = deque()
        self.pre_llm = ()
        self.pre_llm_session = ""
        self.attr_base: dict = {}
        self.busy = Counter()
        self.inflight = 0
        self.env_digest = None
        self.config_stats: dict = {}
        self.packages: dict = {}
        self.ctx_capture = "ok"
        self.started = clock()
        self.checkpoints = list(CHECKPOINT_HOURS)
        self.last_checkpoint = None
        self.wall = [None, None]
        self.engine = engine
        self.db_path = Path(engine.__dict__["_db_path"]).resolve()
        db8 = hashlib.sha256(str(self.db_path).encode()).hexdigest()[:8]
        self.proc = _mkdir(root / "proc" / f"{os.getpid()}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{db8}")
        _mkdir(root / "blobs")
        self.max_bytes = DEFAULT_MAX_BYTES
        self.writer = _Writer(self)
        self.state_db = None
        self.session = ""
        self.cur = None
        self.monitoring = False
        self.pins = set()
        self.status_lock, self.status_reserved = threading.Lock(), 0
        self.pending_cap = None

    # -- install ---------------------------------------------------------------------------------------
    def start(self, ctx, engine):
        from .config import lcm_env_get

        raw = lcm_env_get("LCM_RECORDER_MAX_BYTES")
        try:
            self.max_bytes = int(raw) if raw not in (None, "") else DEFAULT_MAX_BYTES
        except ValueError:
            logger.warning("LCM recorder: LCM_RECORDER_MAX_BYTES=%r is not an integer; using 20 GiB", raw)
        self.writer.total_bytes = self._existing_bytes()
        self.writer.start()
        payload_dir = self._payload_dir(engine)
        need = self.db_path.stat().st_size if self.db_path.exists() else 0
        need += sum(p.stat().st_size for p in payload_dir.rglob("*") if p.is_file())
        if not self._admit(need, reserve=False):
            self._snapshot_refused(self.stopped)
            return
        pinned = self._pin_db()
        self._token(engine)
        self._resolve(type(engine))
        self.home = _hermes_home(engine)
        home = str(engine.__dict__.get("_hermes_home") or "")
        state_path = Path(engine._state_db_path())
        self.state_db = _StateDb(state_path)
        facts = self._process_facts(engine, home, state_path)
        self.env_digest = hash(json.dumps(plain(facts["env"]), sort_keys=True))
        self._capture_payloads(payload_dir)
        if not self.writer.offer(("backup", lambda: self._snapshot(pinned, payload_dir)), force=True):
            _close_pin(pinned)
            self.pins.discard(pinned)
        self._emit({"kind": "process_start", **facts}, single=True)
        self.opened = True
        self._seed(self.engines[engine], engine)
        self._monitor_on()
        self.capture_ctx(ctx)
        atexit.register(self._atexit)
        os.register_at_fork(after_in_child=self._fork_child)

    def add(self, ctx, engine):
        if self.stopped is not None or self._foreign(engine):
            return
        with self.lock:
            self._check_home(engine)
            token = self._token(engine)
            self._resolve(type(engine))
            changed = self._config_files(engine)
            if changed:
                self._emit({"kind": "config_files", "files": changed}, single=True)
            env = self._env_values(engine)
            self._emit({"kind": "register", "engine": token,
                        "hermes_home": str(engine.__dict__.get("_hermes_home") or ""),
                        "env": env}, single=True)
            # replay applies the registration's environment, so the next root is compared against it
            self.env_digest = hash(json.dumps(plain(env), sort_keys=True))
        self.capture_ctx(ctx)

    def _existing_bytes(self) -> int:
        total = 0
        for directory, _, files in os.walk(self.root):
            for name in files:
                try:
                    total += (Path(directory) / name).stat().st_size
                except OSError:
                    pass  # another process can replace or remove its STATUS temp file
        return total

    def _admit(self, need, *, reserve=True, diagnostic=False, checkpoint=None):
        with self.writer.budget:
            if self.stopped is not None and not diagnostic:
                return False
            headroom = 0 if diagnostic else DIAG_RESERVE
            cause = "cap" if self.writer.total_bytes + need > self.max_bytes - headroom else (
                "disk_floor" if not _disk_room(self.root, need - (DIAG_RESERVE if diagnostic else 0)) else None)
            if cause is not None:
                if checkpoint is not None:
                    self._skip_checkpoint(checkpoint, cause)
                elif not diagnostic and self.stopped is None:
                    self.stopped = self.end_state = cause
                    self.pending_cap = (cause, need)
                    self.lose(cause)
                return False
            if reserve:
                self.writer.total_bytes += need
            return True

    def _reserve(self, need):
        if not self._admit(need):
            raise _Stopped()

    def _pin_db(self):
        pinned = _pin(self.db_path)
        if pinned is not None:
            self.pins.add(pinned)
        return pinned

    def _close_pins(self):
        while True:
            try:
                pin = self.pins.pop()
            except KeyError:
                return
            try:
                _close_pin(pin)
            except BaseException:
                pass

    def _process_facts(self, engine, home, state_path):
        import dataclasses

        config = engine.__dict__.get("_config")
        fields = {}
        for field in dataclasses.fields(config) if dataclasses.is_dataclass(config) else ():
            value = getattr(config, field.name, None)
            fields[field.name] = Tag("set", bool(value)) if is_secret_name(field.name) else self._copy(value, [])
        plugin_root = Path(sys.modules[type(engine).__module__].__file__).resolve().parent
        hermes = ""
        try:
            hermes_root = Path(sys.modules["agent.context_engine"].__file__).resolve().parents[1]
            hermes = _git_sha(hermes_root) or ""
        except Exception:
            pass
        try:
            resolved_state = str(state_path.resolve())
        except OSError:
            resolved_state = str(state_path)
        version = next((line.split(":", 1)[1].strip() for line in (plugin_root / "plugin.yaml").read_text()
                        .splitlines() if line.startswith("version:")), "") if (plugin_root / "plugin.yaml").exists() else ""
        return {
            "python": sys.version.split()[0], "tool_id": self.tool, "lcm_version": version,
            "lcm_sha": _git_sha(plugin_root), "hermes_sha": hermes, "hermes_home": home,
            "db_path": str(self.db_path), "state_db_path": str(state_path),
            "lineage_tag": hashlib.sha256(resolved_state.encode()).hexdigest()[:16],
            "config": fields, "config_files": self._config_files(engine, initial=True),
            "env": self._env_values(engine), "sessions": self.state_db.full(),
            "roots": sorted({f"{info[0]}:{info[1]}" for info in self.roots.values()}),
            "skipped": self.skipped, "ctx_capture": self.ctx_capture, "max_bytes": self.max_bytes,
        }

    def _payload_dir(self, engine) -> Path:
        config = engine.__dict__.get("_config")
        configured = getattr(config, "large_output_externalization_path", "") or ""
        if configured:
            return Path(configured).expanduser().resolve()
        home = engine.__dict__.get("_hermes_home") or ""
        base = Path(home).expanduser().resolve() if home else Path("~/.hermes").expanduser().resolve()
        return base / "lcm-large-outputs"

    def _capture_payloads(self, payload_dir):
        snap = _mkdir(self.proc / "snapshot")
        result = {}
        size = 0
        if payload_dir.is_dir():
            files = sorted(p for p in payload_dir.rglob("*") if p.is_file())
            size = sum(p.stat().st_size for p in files)
            room = _disk_room(self.root, size)
            if not room or shutil.disk_usage(self.root).free < 2 * size:
                self.counters["recorder.payload_copy_skipped"] += 1
                result["payloads"] = "payloads_missing"
                result["cause"] = "disk_floor" if not room else "space"
            else:
                out = _mkdir(snap / "large-outputs")
                for path in files:
                    rel = path.relative_to(payload_dir)
                    _mkdir((out / rel).parent)
                    self._reserve(path.stat().st_size)
                    try:
                        os.link(path, out / rel)
                    except OSError:  # another filesystem, or links refused or unsupported there
                        self._payload_copy(path, out / rel)
                result["payloads"] = "ok"
        else:
            result["payloads"] = "absent"
        result["payload_dir"] = str(payload_dir)
        self.payload_snapshot = result

    @staticmethod
    def _payload_copy(source, target):
        st = source.stat()
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with open(source, "rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))

    def _snapshot_refused(self, cause="disk_floor"):
        self.snapshot = {"state": "refused", "cause": cause}
        self.stopped = self.end_state = cause
        self.lose(cause)
        self.write_status()

    def _snapshot(self, pinned, payload_dir: Path):
        try:
            self._snapshot_body(pinned, payload_dir)
        except _Stopped:
            self._snapshot_refused(self.stopped)
        except Exception as exc:
            self.snapshot = {"state": "failed", "cause": type(exc).__name__}
            raise
        finally:
            _close_pin(pinned)
            self.pins.discard(pinned)

    def _snapshot_body(self, pinned, payload_dir):
        snap = self.proc / "snapshot"
        files = [p for p in (snap / "large-outputs").rglob("*") if p.is_file()]
        db_size = (pinned.execute("PRAGMA page_count").fetchone()[0] *
                   pinned.execute("PRAGMA page_size").fetchone()[0]) if pinned is not None else 0
        need = db_size + sum(p.stat().st_size for p in files)
        if not self._admit(need, reserve=False):
            raise _Stopped()
        manifest = []
        for path in sorted((snap / "large-outputs").rglob("*")):
            if not path.is_file():
                continue
            tmp = path.with_name(path.name + ".snapshot-tmp")
            self._reserve(path.stat().st_size)
            self._payload_copy(path, tmp)
            os.replace(tmp, path)
            st = path.stat()
            manifest.append({"name": str(path.relative_to(snap / "large-outputs")), "size": st.st_size,
                             "mode": st.st_mode, "uid": st.st_uid, "gid": st.st_gid,
                             "mtime_ns": st.st_mtime_ns, "sha256": _sha256_file(path)})
        data = json.dumps(manifest, sort_keys=True).encode()
        self._reserve(len(data))
        fd = os.open(snap / "large-outputs.manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        self._reserve(db_size)
        self.snapshot = {**_backup(pinned, snap / "lcm.db"), **self.payload_snapshot}
        self.write_status()

    def _monitor_on(self):
        mon = sys.monitoring
        mon.register_callback(self.tool, mon.events.PY_START, self._on_start)
        mon.register_callback(self.tool, mon.events.PY_RETURN, self._on_return)
        mon.register_callback(self.tool, mon.events.PY_UNWIND, self._on_unwind)
        mon.set_events(self.tool, mon.events.PY_UNWIND)
        for info in self.roots.values():
            mon.set_local_events(self.tool, info[3], mon.events.PY_START | mon.events.PY_RETURN)
        self.monitoring = True

    def _watch(self, code, kind, name):
        if code.co_flags & _CO_SKIP:
            self.skipped.append([name, "generator"])
            return
        known = self.roots.get(id(code))
        if known is not None:
            known[2] = known[2] or known[1] != name
            return
        self.roots[id(code)] = [kind, name, False, code]
        if self.monitoring:
            sys.monitoring.set_local_events(self.tool, code, sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN)

    def _resolve(self, cls):
        if cls in self.engine_classes:
            return
        self.engine_classes += (cls,)
        names = set(HOST_ENGINE_METHODS) | set(EXTRA_ENGINE_ROOTS)
        try:
            from agent.context_engine import ContextEngine
            names |= {n for n in dir(ContextEngine) if not n.startswith("_")
                      and callable(inspect.getattr_static(ContextEngine, n, None))}
        except Exception:
            pass
        for name in sorted(names):
            try:
                attr = inspect.getattr_static(cls, name)
            except AttributeError:
                self.skipped.append([name, "absent"])
                continue
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            code = getattr(attr, "__code__", None) if not isinstance(attr, property) else None
            if code is None:
                self.skipped.append([name, "property" if isinstance(attr, property) else "not a function"])
                continue
            self._watch(code, "engine", name)
        package = cls.__module__.rpartition(".")[0]
        escalation = importlib.import_module(package + ".escalation")
        self._watch(escalation._invoke_summary_attempt.__code__, "summary", "_invoke_summary_attempt")
        for module_name in ("maintenance_scheduler", "engine"):
            module = sys.modules.get(f"{package}.{module_name}")
            for name in ATEXIT_ROOTS:
                if module is not None and callable(getattr(module, name, None)):
                    self._watch(getattr(module, name).__code__, "atexit", name)
        persisted = importlib.import_module(package + ".ingest_persisted")
        self._watch(persisted.recover_hermes_persisted_output_with_file_stat.__code__, "input", "persisted_output")
        self.packages[package] = sys.modules.get(package + ".sender_capture")

    # -- ctx capture -----------------------------------------------------------------------------------
    def capture_ctx(self, ctx):
        for attr, kind in CTX_REGISTRARS:
            original = getattr(ctx, attr, None)
            if not callable(original):
                continue
            try:
                setattr(ctx, attr, self._capturer(original, kind))
            except Exception:
                self.ctx_capture = "unavailable"
                self.lose("ctx_capture_unavailable")

    def _capturer(self, original, kind):
        recorder = self

        def capture(*args, **kwargs):
            try:
                recorder._register(kind, args, kwargs)
            except KeyboardInterrupt:
                recorder.failed("capture", interrupted=True)
                raise
            except BaseException as exc:
                recorder.failed("capture", exc)
            return original(*args, **kwargs)
        capture.__wrapped__ = original
        return capture

    def _register(self, kind, args, kwargs):
        if self.stopped is not None:
            return
        name = kwargs.get("name", args[0] if args else "")
        callback = next((kwargs[k] for k in ("handler", "callback", "fn", "func") if k in kwargs), None)
        if callback is None:
            callback = next((a for a in args[1:] if callable(a)), None)
        target, code = callback, None
        while code is None and target is not None:
            code = getattr(target, "__code__", None) or getattr(getattr(target, "__func__", None), "__code__", None)
            target = getattr(target, "func", None) if code is None else None
        with self.lock:
            if code is None:
                self.mark("", "uncapturable")
                self.lose("uncapturable", f"{kind}:{name}")
                return
            cells = tuple(cell.cell_contents for cell in (getattr(callback, "__closure__", None) or ()))
            self.callbacks.setdefault(id(code), []).append((kind, str(name), cells))
            self._watch(code, kind, str(name))

    # -- identity --------------------------------------------------------------------------------------
    def _token(self, engine) -> int:
        token = self.engines.get(engine)
        if token is None:
            token = self.engines[engine] = next(self.engine_ids)
            finalizer = weakref.finalize(engine, self._gone, token)
            finalizer.atexit = False
            self.attr_base[token] = self._attrs(engine)
            if self.opened:
                self._seed(token, engine)
        return token

    def _seed(self, token, engine):
        values = {n: self._attr_value(engine.__dict__[n]) for n in SEED_ATTRS if n in engine.__dict__}
        if values:
            self._emit({"kind": "seed", "engine": token, "values": values}, single=True)

    def _gone(self, token):
        self.gone.append(token)

    def _thread_token(self) -> int:
        thread = threading.current_thread()
        token = self.threads.get(thread)
        if token is None:
            token = self.threads[thread] = next(self.thread_ids)
        return token

    def _caller_token(self, caller) -> int:
        try:
            token = self.callers.get(caller)
            if token is None:
                token = self.callers[caller] = next(self.caller_ids)
            return token
        except TypeError:
            return -id(caller)

    def _is_engine(self, value) -> bool:
        return isinstance(value, self.engine_classes)

    def _stack(self) -> list:
        stack = getattr(self.local, "stack", None)
        if stack is None:
            stack = self.local.stack = []
        return stack

    # -- monitoring callbacks: never raise (except KeyboardInterrupt), never return DISABLE -------------
    def _on_start(self, code, offset):
        try:
            if self.stopped is None:
                self._enter(code, sys._getframe(1))
        except KeyboardInterrupt:
            self.failed("PY_START", interrupted=True)
            raise
        except BaseException as exc:
            self.failed("PY_START", exc)
        return None

    def _on_return(self, code, offset, value):
        try:
            self._exit(code, sys._getframe(1), "return", value)
        except KeyboardInterrupt:
            self.failed("PY_RETURN", interrupted=True)
            raise
        except BaseException as exc:
            self.failed("PY_RETURN", exc)
        finally:
            self._discard(code, sys._getframe(1))
        return None

    def _on_unwind(self, code, offset, exc):
        if id(code) not in self.roots:
            return None
        try:
            self._exit(code, sys._getframe(1), "raise", exc)
        except KeyboardInterrupt:
            self.failed("PY_UNWIND", interrupted=True)
            raise
        except BaseException as error:
            self.failed("PY_UNWIND", error)
        finally:
            self._discard(code, sys._getframe(1))
        return None

    def _enter(self, code, frame):
        info = self.roots.get(id(code))
        if info is None:
            return
        kind, name, shared, _code = info
        loc = frame.f_locals
        if kind == "input" and not frame.f_globals["_is_hermes_persisted_output_marker"](loc.get("text")):
            return
        stack = self._stack()
        parent = next((e for e in reversed(stack) if e.rec is not None), None)
        if kind == "input" and parent is None:
            return
        if any(e.rec is None for e in stack):
            return  # inside a call that is not recorded (a foreign path)
        engine = None
        if kind == "engine":
            engine = loc.get("self")
            if not self._is_engine(engine):
                return
            if shared:
                name = next((v.__name__ for n in code.co_freevars if callable(v := loc.get(n))
                             and hasattr(v, "__name__")), name)
        elif kind in ("summary", "atexit", "input"):
            engine = parent.engine if parent is not None else None
        else:
            name, engine = self._callback(code, loc, name)
        with self.lock:
            self._flush_gone()
            if engine is not None and self._foreign(engine):
                self.counters["recorder.foreign_path"] += 1
                stack.append(_Entry(frame, code, None, None, parent is None, name))
                return
            root = parent is None
            if root and kind == "summary":
                kind = "bg"
            etoken = self._token(engine) if engine is not None else None
            self.session = self._session_of(code, loc, engine)
            if root:
                if engine is None:
                    self.counters["recorder.engine_unknown"] += 1
                self._before_root(engine, kind, code, loc, frame)
            seq = self._next()
            rec = {"kind": kind, "name": name, "seq_enter": seq, "parent_seq": parent.rec["seq_enter"] if parent else None,
                   "thread": threading.current_thread().name, "thread_token": self._thread_token(),
                   "engine": etoken if etoken is not None else "unknown", "t_enter": [time.time(), time.monotonic()],
                   **self._identity(engine), "session": self.session}
            if root and engine is not None:
                store = engine.__dict__.get("_store")
                rec["_ledger_ready"] = bool(store and store.__dict__.get("_ingest_ledger_schema_ready"))
            marks = []
            entry = _Entry(frame, code, rec, engine, root, name)
            (stack[0] if stack else entry).touched.add(etoken)
            if kind == "input":
                rec["path"] = frame.f_globals["_persisted_output_saved_path"](loc.get("text"))
            self.cur = entry
            if kind in ("summary", "bg"):
                rec["args"] = self._summary_args(loc)
            else:
                rec["args"], rec["sig"] = self._args(code, loc, marks), self.signature(code)
            if root:
                if kind == "command":
                    rec["context"] = self._command_context(engine)
                rec["aux"] = self._aux(frame, engine)
                if any("aux" in found["rules"] for found in rec["aux"]):
                    self.mark(self.session, "aux_context")
            if marks:
                self._unsupported(rec, marks)
            if kind == "hook" and name == "pre_llm_call":
                self.pre_llm_session = self.session
                self.pre_llm = tuple(id(m) for m in (loc.get("payload") or {}).get("conversation_history") or ())
            stack.append(entry)
            self.cur = None
            if root:
                self.inflight += 1
            self.busy[rec["engine"]] += 1

    def _pop(self, code, frame):
        stack = self._stack()
        index = next((i for i in range(len(stack) - 1, -1, -1)
                      if stack[i].frame is frame and stack[i].code is code), None)
        if index is None:
            return []
        entries = stack[index:]
        del stack[index:]
        return entries

    def _discard(self, code, frame):
        entries = self._pop(code, frame)
        if entries and self.stopped != "fork":
            with self.lock:
                for entry in entries:
                    if entry.rec is not None:
                        self._close(entry)

    def _exit(self, code, frame, status, value):
        entries = self._pop(code, frame)
        if not entries or self.stopped == "fork":
            return
        entry = entries[0]
        salt = None
        if self.stopped is None and entry.root and entry.engine is not None and entry.rec is not None:
            store = entry.engine.__dict__.get("_store")
            if not entry.rec.get("_ledger_ready", False) and store is not None and \
                    store.__dict__.get("_ingest_ledger_schema_ready"):
                try:   # read outside self.lock (F5); a raising read must not skip this root's cleanup below
                    salt = store.ingest_ledger_salt()
                except Exception as exc:
                    salt = None
                    self.failed("ledger_salt", exc)
        with self.lock:
            try:
                if self.stopped is not None:
                    return
                for orphan in entries[1:]:
                    if orphan.rec is not None:
                        orphan.rec.update(status="stack_mismatch", seq_exit=self._next())
                        self._emit(orphan.rec)
                        self.lose("stack_mismatch", orphan.rec["seq_enter"])
                if entry.rec is None:
                    return
                rec = entry.rec
                self.cur = entry
                self.session = rec.get("session") or ""
                marks = []
                if status == "return":
                    rec["status"] = "return"
                    if rec["kind"] == "input":
                        rec["result"] = {"outcome": "ok" if value is not None else "unavailable",
                                         "stat": value[1] if value else None,
                                         "sha256": hashlib.sha256(value[0].encode()).hexdigest() if value else None,
                                         "content": Blob(value[0]) if value else None}
                    else:
                        rec["result"] = self._summary_result(value, entry.t0) if rec["kind"] in ("summary", "bg") \
                            else self._copy(value, marks)
                else:
                    rec["status"] = "raise"
                    rec["exception"] = _exception(value)
                if entry.name == "compress" and entry.engine is not None:
                    d = entry.engine.__dict__
                    rec["compression"] = [d.get("_last_compression_status"), d.get("_last_compression_noop_reason")]
                if marks:
                    self._unsupported(rec, marks)
                rec.pop("_ledger_ready", None)
                if salt is not None:
                    rec["ledger_salt"] = salt.hex()
                rec["seq_exit"] = self._next()
                rec["t_exit"] = [time.time(), time.monotonic()]
                self._emit(rec)
            finally:
                self.cur = None
                for completed in entries:
                    if completed.rec is not None:
                        self._close(completed)
            if entry.root and entry.rec is not None:
                self._after_root(entry)
        if entry.root and entry.name in _SYNC_ROOTS and self.stopped is None:
            self.writer.drain(2.0, skip_backup=True)

    def _close(self, entry):
        if entry.root:
            self.inflight -= 1
        self.busy[entry.rec.get("engine")] -= 1

    def _callback(self, code, loc, name):
        regs = self.callbacks.get(id(code)) or []
        cells = tuple(loc.get(n, _MISSING) for n in code.co_freevars)
        match = next((r for r in regs if len(r[2]) == len(cells) and all(a is b for a, b in zip(r[2], cells))),
                     regs[0] if len(regs) == 1 else None)
        if match is None:
            return name, None
        engine = next((v for v in match[2] if self._is_engine(v)), None)
        return match[1], engine

    def _foreign(self, engine) -> bool:
        raw = engine.__dict__.get("_db_path")
        try:
            return raw is not None and Path(raw).resolve() != self.db_path
        except OSError:
            return True

    def _check_home(self, engine) -> bool:
        if engine is None or _hermes_home(engine) == self.home:
            return True
        if engine not in self.home_changed:
            self.home_changed.add(engine)
            self.counters["recorder.hermes_home_changed"] += 1
            self.lose("hermes_home_changed")
        return False

    def _session_of(self, code, loc, engine) -> str:
        for holder in (loc, loc.get("kwargs"), loc.get("payload"), loc.get("info")):
            if isinstance(holder, Mapping) and isinstance(holder.get("session_id"), str) and holder["session_id"]:
                return holder["session_id"]
        return str(engine.__dict__.get("_session_id") or "") if engine is not None else ""

    def _identity(self, engine) -> dict:
        d = engine.__dict__ if engine is not None else {}
        return {"session": d.get("_session_id"), "conversation": d.get("_conversation_id"),
                "platform": d.get("_session_platform")}

    def _next(self) -> int:
        if self.gone:
            self._flush_gone()
        seq = next(self.seq)
        self.last_seq = max(self.last_seq, seq)
        return seq

    # -- values ----------------------------------------------------------------------------------------
    @staticmethod
    def signature(code) -> list:
        """[positional, keyword-only, *args name, **kwargs name]: how a replay passes the recorded arguments."""
        count = code.co_argcount + code.co_kwonlyargcount
        star = code.co_varnames[count] if code.co_flags & inspect.CO_VARARGS else None
        count += star is not None
        starstar = code.co_varnames[count] if code.co_flags & inspect.CO_VARKEYWORDS else None
        return [list(code.co_varnames[:code.co_argcount]), list(code.co_varnames[code.co_argcount:count - (star is not None)]),
                star, starstar]

    def _args(self, code, loc, marks) -> dict:
        positional, kwonly, star, starstar = self.signature(code)
        names = positional + kwonly + [n for n in (star, starstar) if n]
        return {n: self._value(n, loc[n], marks) for n in names if n in loc}

    def _value(self, name, value, marks):
        if name in SET_ONLY_NAMES:
            return Tag("set", value is not None and value != "")
        return self._copy(value, marks)

    def _copy(self, value, marks):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {k: self._value(k, v, marks) for k, v in value.items()}
        if isinstance(value, list):
            if value and all(isinstance(m, dict) and "role" in m for m in value):
                return self._view(value, marks)
            return [self._copy(v, marks) for v in value]
        if isinstance(value, tuple):
            return tuple(self._copy(v, marks) for v in value)
        if isinstance(value, (set, frozenset)):
            return Tag("set_items", sorted((self._copy(v, marks) for v in value), key=repr))
        if self._is_engine(value):
            return Tag("engine", self._token(value))
        if callable(value) or isinstance(value, _LOCK_TYPES):
            return Tag("set", True)
        marks.append("unsupported")
        return Tag("unsupported", type(value).__name__)

    def _view(self, rows, marks) -> View:
        session = self.session
        previous = self.views.get(session)
        before = previous[-1] if previous else ()
        ops, ids = [], []
        for position, obj in enumerate(rows):
            oid = id(obj)
            row = self.rows.get(oid)
            if row is not None and row[1] is obj:
                if obj == row[2]:
                    ops.append(("s", row[0]))
                else:
                    row[2] = _row_copy(obj, marks)
                    ops.append(("e", row[0], Blob(row[2])))
                    self._index(row)
            else:
                source = None
                content = obj.get("content")
                if type(content) is str:
                    other = self.rows.get(self.by_content.get(id(content)))
                    if other is not None and other[2].get("content") is content and obj == other[2]:
                        source = other
                if source is None and position < len(before):
                    other = self.rows.get(before[position])
                    if other is not None and obj == other[2]:
                        source = other
                token = next(self.row_ids)
                if source is not None:
                    row = [token, obj, source[2]]
                    ops.append(("c", token, source[0]))
                else:
                    row = [token, obj, _row_copy(obj, marks)]
                    ops.append(("n", token, Blob(row[2])))
                self.rows[oid] = row
                self._index(row)
            ids.append(oid)
        self.views.setdefault(session, deque(maxlen=2)).append(ids)
        if self.cur is not None and self.cur.engine is not None:
            self.view_owners.setdefault(session, set()).add(self._token(self.cur.engine))
        while len(self.views) > 256:
            self._drop_view(next(iter(self.views)))
        if self.cur is not None:
            self.cur.views += 1
        return View(ops)

    def _index(self, row):
        content = row[2].get("content")
        if type(content) is str:
            self.by_content[id(content)] = id(row[1])

    def _drop_view(self, session):
        self.views.pop(session, None)
        self.view_owners.pop(session, None)
        if session == self.pre_llm_session:
            self.pre_llm = ()
            self.pre_llm_session = ""

    def _retain(self):
        keep = set(self.pre_llm)
        for history in self.views.values():
            for ids in history:
                keep.update(ids)
        for engine in list(self.engines):
            for holder in (engine.__dict__, getattr(engine.__dict__.get("_store"), "__dict__", {})):
                state = holder.get("_host_rewrite_watch")
                detached = holder.get("_detached_host_rewrite_state")
                for watch in (state, detached[0] if isinstance(detached, tuple) else None):
                    for entry in (watch or {}).values():
                        keep.update(id(o) for o in entry[2])
        for module in self.packages.values():
            for pending in list(getattr(module, "_PENDING", {}).values()):
                keep.add(id(getattr(pending, "obj", None)))
        for oid in [oid for oid in self.rows if oid not in keep]:
            row = self.rows.pop(oid)
            content = row[2].get("content")
            if type(content) is str and self.by_content.get(id(content)) == oid:
                del self.by_content[id(content)]

    def _summary_args(self, loc) -> dict:
        prompt = loc.get("prompt")
        text = prompt if isinstance(prompt, str) else ""
        return {"prompt_sha256": hashlib.sha256(text.encode()).hexdigest(), "prompt_len": len(text),
                "model": loc.get("model"), "max_tokens": loc.get("max_tokens"), "timeout": loc.get("timeout"),
                "reasoning_effort": loc.get("reasoning_effort"), "budget": Tag("set", loc.get("budget") is not None)}

    def _summary_result(self, attempt, t0) -> dict:
        timing = getattr(attempt, "timing", None)
        started = getattr(timing, "started", None)
        ended = getattr(timing, "ended", None)
        text = getattr(attempt, "text", None)
        return {"text": Blob(text) if text is not None else None, "route": self._copy(getattr(attempt, "route", {}), []),
                "contract": getattr(attempt, "contract", ""), "error": _stand_in(getattr(attempt, "error", None)),
                "timing": [None if started is None else started - t0, None if ended is None else ended - t0,
                           bool(getattr(timing, "deadline_installed", False))]}

    def _unsupported(self, rec, marks):
        for mark in marks:
            self.mark(self.session, mark)
        rec["lost_observation"] = marks[0]
        self.lose(marks[0], rec["seq_enter"])

    # -- state captures --------------------------------------------------------------------------------
    def _attrs(self, engine) -> dict:
        d = engine.__dict__
        return {n: _base(d[n]) for n in HOST_ATTRS + SEED_ATTRS if n in d}

    def _diff_attrs(self, touched=()):
        for other, token in list(self.engines.items()):
            if self.busy[token]:
                continue
            now, base = self._attrs(other), self.attr_base.get(token, {})
            names = set(SEED_ATTRS) if token in touched else set(now) | set(base)
            changed = {n: self._attr_value(other.__dict__.get(n, _MISSING)) for n in names
                       if not _same(now.get(n, _MISSING), base.get(n, _MISSING))}
            if changed:
                self._emit({"kind": "attrs", "engine": token, "values": changed}, single=True)
            self.attr_base[token] = now

    def _before_root(self, engine, kind, code, loc, frame):
        if not self._check_home(engine):
            return  # the pinned state database belongs to the recording's home
        self._diff_attrs()
        env = self._env_values(self.engine)
        digest = hash(json.dumps(plain(env), sort_keys=True))
        if digest != self.env_digest:
            if self.env_digest is not None:
                self._emit({"kind": "env", "values": env}, single=True)
            self.env_digest = digest
        changed = self._config_files(engine or self.engine)
        if changed:
            self._emit({"kind": "config_files", "files": changed}, single=True)
        ids = {self.session, engine.__dict__.get("_session_id") if engine is not None else None}
        if engine is not None:
            ids |= {engine.__dict__.get(n) for n in ("_foreground_rebind_session_id",
                                                    "_foreground_rebind_parent_session_id")}
        ids |= {v for k, v in _flat_items(loc) if k.endswith("session_id") and isinstance(v, str)}
        rows = self.state_db.changes(sorted(i for i in ids if i))
        if rows:
            self._emit({"kind": "sessions", "rows": rows}, single=True)

    def _attr_value(self, value):
        if value is _MISSING:
            return Tag("absent", True)
        if callable(value) or isinstance(value, _LOCK_TYPES):
            return Tag("set", True)
        return self._copy(value, [])

    def _env_values(self, engine) -> dict:
        package = type(engine).__module__.rpartition(".")[0]
        env = sys.modules[package + ".config"].lcm_env()
        return {k: (Tag("set", bool(v)) if is_secret_name(k) else v) for k, v in env.items()
                if k.startswith("LCM_") or k == "HERMES_HOME"}

    def _config_files(self, engine, initial=False):
        homes = {str(e.__dict__.get("_hermes_home") or "") for e in list(self.engines)} | {
            str(engine.__dict__.get("_hermes_home") or "")}
        changed = {}
        for home in sorted(h for h in homes if h):
            path = Path(home) / "config.yaml"
            try:
                st = path.stat()
                stamp = (st.st_mtime_ns, st.st_size)
            except OSError:
                stamp = None
            if self.config_stats.get(str(path), _MISSING) == stamp:
                continue
            self.config_stats[str(path)] = stamp
            entry = {"state": "absent"} if stamp is None else {"sha256": _sha256_file(path), "mtime_ns": stamp[0]}
            if stamp is not None:
                try:
                    package = type(engine).__module__.rpartition(".")[0]
                    config_module = sys.modules[package + ".config"]
                    values = config_module._load_hermes_config_yaml(path)
                    values = {k: values[k] for k in ("lcm", "compression", "auxiliary", "context", "plugins") if k in values}
                    if "auxiliary" in values:
                        values["auxiliary"] = {"compression": (values["auxiliary"] or {}).get("compression", {})}
                    for key, keep in (("context", "engine"), ("plugins", "enabled")):  # plugin_identity.py:49-54
                        if key in values:
                            section = values[key]
                            values[key] = {keep: section[keep]} if isinstance(section, dict) and keep in section else {}
                    entry["values"] = self._config_values(values)
                    reader = config_module.hermes_auxiliary_compression_max_concurrency
                    entry["auxiliary.compression.max_concurrency"] = reader(home)
                except Exception:
                    pass
            changed[str(path)] = entry
        return changed if changed or not initial else {}

    def _config_values(self, value):
        if isinstance(value, dict):
            return {k: Tag("set", bool(v)) if is_secret_name(k) else self._config_values(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._config_values(v) for v in value]
        return self._copy(value, [])

    def _command_context(self, engine) -> dict:
        values = {}
        for name in COMMAND_CONTEXT:
            try:
                from gateway.session_context import get_session_env
                values[name] = str(get_session_env(name, "") or "")
            except ImportError:
                from .config import lcm_env_get
                values[name] = str(lcm_env_get(name, "") or "")
            except Exception:
                values[name] = Tag("read_error", True)
        try:
            from hermes_constants import profile_name_for_home
            home = engine.__dict__.get("_hermes_home", "") if engine is not None else ""
            values["engine_profile"] = str(profile_name_for_home(home) or "") if home else ""
        except Exception:
            values["engine_profile"] = ""
        return values

    def _aux(self, frame, engine) -> list:
        engine = engine if engine is not None else self.engine
        predicate = getattr(type(engine), "_caller_is_auxiliary_agent_frame", None)
        if predicate is None:
            return []
        found, current = [], frame.f_back
        for distance in range(1, AUX_DEPTH + 1):
            if current is None:
                break
            if "self" in current.f_code.co_varnames:
                caller = current.f_locals.get("self")
                rules = [] if caller is None or self._is_engine(caller) else [
                    rule for rule, ok in (("aux", lambda: predicate(engine, caller)), ("session_id", lambda: bool(
                        self.session) and str(getattr(caller, "session_id", "") or "") == self.session)) if ok()]
                if rules:
                    fields = {}
                    for name in AUX_FIELDS:
                        value = getattr(caller, name, _MISSING)
                        if value is _MISSING:
                            continue
                        fields[name] = Tag("set", bool(value)) if name == "ephemeral_system_prompt" \
                            else self._copy(value, [])
                    found.append({"distance": distance, "caller": self._caller_token(caller), "rules": rules,
                                  "fields": fields})
            current = current.f_back
        return found

    def _after_root(self, entry):
        self._diff_attrs(entry.touched)
        if entry.name == "on_session_end":
            self._drop_view(entry.rec.get("session") or "")
        if entry.views or entry.name == "on_session_end":
            self._retain()
        if self.inflight == 0 and self.checkpoints and clock() - self.started >= self.checkpoints[0] * 3600:
            self._checkpoint(self.checkpoints.pop(0))

    def _skip_checkpoint(self, hours, cause):
        self.counters["recorder.checkpoint_skipped"] += 1
        self.warnings.append(("LCM recorder: checkpoint +%dh skipped: %s", hours, cause))

    def _checkpoint(self, hours):
        try:
            size = self.db_path.stat().st_size if self.db_path.exists() else 0
            if not self._admit(size, reserve=False, checkpoint=hours):
                return
            if shutil.disk_usage(self.root).free < 2 * size:
                self._skip_checkpoint(hours, "space")
                return
            pinned = self._pin_db()
        except Exception as exc:
            self._skip_checkpoint(hours, f"{type(exc).__name__}: {exc}")
            return
        watermark = self.last_seq

        def seal():
            try:
                size = (pinned.execute("PRAGMA page_count").fetchone()[0] *
                        pinned.execute("PRAGMA page_size").fetchone()[0]) if pinned is not None else 0
                if not self._admit(size, checkpoint=hours):
                    return
                _backup(pinned, self.proc / "checkpoint" / str(hours) / "lcm.db")
                self.writer._tick(force=True)
                self.last_checkpoint = {"hours": hours, "seq": watermark}
                self._emit({"kind": "checkpoint_sealed", "hours": hours, "through": watermark}, single=True)
                self.writer._tick(force=True)
            finally:
                _close_pin(pinned)
                self.pins.discard(pinned)
        if not self.writer.offer(("backup", seal), force=True):
            _close_pin(pinned)
            self.pins.discard(pinned)

    # -- events, coverage, failures --------------------------------------------------------------------
    def _flush_gone(self):
        with self.lock:
            tokens = []
            while self.gone:
                token = self.gone.popleft()
                tokens.append(token)
                self.attr_base.pop(token, None)
                self.busy.pop(token, None)
                for session, owners in list(self.view_owners.items()):
                    owners.discard(token)
                    if not owners:
                        self._drop_view(session)
            if tokens:
                self._retain()
            for token in tokens:
                self._emit({"kind": "engine_gone", "engine": token}, single=True)

    def _emit(self, rec, single=False):
        with self.lock:
            if self.gone:
                self._flush_gone()
            if self.stopped is not None:
                return
            if single:
                rec["seq"] = self._next()
            rec.update(v=FORMAT_VERSION, pid=os.getpid(), lost=self.lost)
            rec.setdefault("t", time.time())
            self.wall = [self.wall[0] or rec["t"], rec["t"]]
            if not self.writer.offer(("event", rec), _cost(rec)):
                self.stopped = "gap"
                self.end_state = "gap"
                self.counters["recorder.gap"] += 1
                self.lose("gap")
                self.writer.offer(("diagnostic", {"kind": "gap", "seq": self._next(), "v": FORMAT_VERSION,
                                             "pid": os.getpid(), "lost": self.lost}), force=True)

    def mark(self, session, name):
        self.marks.setdefault(session or "", Counter())[name] += 1

    def lose(self, cause, seq=None):
        if cause in ("ctx_capture_unavailable", "uncapturable", "hermes_home_changed"):
            self.lost += 1
            self._emit({"kind": "loss", "lost_observation": cause}, single=True)
        self.counters["recorder.coverage_lost"] += 1
        if self.first_cause is None:
            self.first_cause = {"cause": cause, "seq": seq if seq is not None else self.last_seq}
            self.warnings.append(("LCM recorder: replayable coverage ended (%s at seq %s); recording continues for "
                                  "diagnosis. Restart the gateway to start a new epoch.", cause, self.first_cause["seq"]))

    def _log_pending(self):
        if not self.lock._is_owned():
            while True:
                try:
                    warning = self.warnings.popleft()
                except IndexError:
                    break
                logger.warning(*warning)

    def failed(self, where, exc=None, interrupted=False):
        try:
            with self.lock:
                self.counters["recorder.failed"] += 1
                self.lost += 1
                self.lose("interrupted" if interrupted else f"recorder_failed:{where}")
        except BaseException:
            pass

    def writer_failed(self, item, exc):
        self.counters["recorder.failed"] += 1
        seq = item[1].get("seq", item[1].get("seq_enter")) if item[0] == "event" else None
        self.lose(f"writer:{type(exc).__name__}", seq)
        if isinstance(exc, OSError) or not isinstance(exc, Exception):
            self.writer.dead = type(exc).__name__
            self.writer.closed = True
            self.stopped = self.stopped or "writer_failed"
            self.end_state = "writer_failed"

    def cap(self):
        if self.pending_cap is None or not self.lock.acquire(blocking=False):
            return  # only the writer tick appends; retry when the host releases its lock
        try:
            cause, need = self.pending_cap
            self.pending_cap = None
            try:
                self.writer._append({"kind": "cap", "reason": cause, "need": need, "seq": self._next(),
                                     "v": FORMAT_VERSION, "pid": os.getpid(), "lost": self.lost}, diagnostic=True)
            except _Stopped:
                pass
        finally:
            self.lock.release()

    def write_status(self):
        with self.status_lock:
            self._write_status()

    def _write_status(self):
        status = {"v": FORMAT_VERSION, "pid": os.getpid(), "last_seq": self.last_seq, "lost": self.lost,
                  "bytes": self.writer.total_bytes, "counters": dict(self.counters),
                  "marks": {k: dict(v) for k, v in list(self.marks.items())}, "first_cause": self.first_cause,
                  "last_checkpoint": self.last_checkpoint, "end_state": self.end_state,
                  "ended": "ended without exit" if self.end_state in ("running", "gap", "cap") else self.end_state,
                  "ctx_capture": self.ctx_capture, "replayable": self.ctx_capture == "ok"
                  and getattr(self, "snapshot", {}).get("state") in ("ok", "no_database")
                  and getattr(self, "snapshot", {}).get("payloads") != "payloads_missing",
                  "snapshot": getattr(self, "snapshot", None), "wall": self.wall}
        data = json.dumps(plain(status), sort_keys=True).encode()
        growth = max(0, 2 * len(data) - self.status_reserved)
        if not self._admit(growth, diagnostic=True):
            return
        self.status_reserved += growth
        tmp = self.proc / f"STATUS.json.{os.getpid()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, self.proc / "STATUS.json")

    def _atexit(self):
        if self.stopped == "fork":
            return
        try:
            self._emit({"kind": "exit_begin"}, single=True)
            self.end_state = "exit_begin"
            if self.writer.drain(2.0):
                self.writer.sync = True
            else:
                self.stopped = self.end_state = "exit_timeout"
                self.lose("exit_timeout")
                self.write_status()
        except BaseException:
            self.failed("atexit")

    def _fork_child(self):
        self.stopped = self.end_state = "fork"

    def shutdown_for_tests(self):
        mon = sys.monitoring
        for info in self.roots.values():
            mon.set_local_events(self.tool, info[3], 0)
        mon.set_events(self.tool, 0)
        for event in (mon.events.PY_START, mon.events.PY_RETURN, mon.events.PY_UNWIND):
            mon.register_callback(self.tool, event, None)
        mon.free_tool_id(self.tool)
        self.writer.drain(5.0)
        self.stopped = self.stopped or "reset"
        with self.writer.cv:
            self.writer.closed = True
            self.writer.cv.notify()
        self.writer.join(5.0)
        atexit.unregister(self._atexit)


def _base(value):
    if isinstance(value, (dict, list, tuple)):
        try:
            return ("copy", _row_copy(value, []))
        except RecursionError:
            return ("id", id(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    return ("id", id(value))


def _same(a, b) -> bool:
    if a is b:
        return True
    try:
        return a == b
    except Exception:
        return False


def _flat_items(loc):
    for key, value in loc.items():
        yield key, value
        if isinstance(value, dict):
            yield from ((k, v) for k, v in value.items() if isinstance(k, str))


def _cost(rec) -> int:
    def walk(value):
        size = sys.getsizeof(value)
        if isinstance(value, Blob):
            return size + walk(value.copy)
        if isinstance(value, View):
            return size + sys.getsizeof(value.ops) + 160 * len(value.ops) + sum(
                walk(op[2]) for op in value.ops if op[0] in ("e", "n"))
        if isinstance(value, Tag):
            return size + walk(value.value)
        if isinstance(value, dict):
            return size + sum(walk(k) + walk(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return size + sum(walk(v) for v in value)
        return size
    return 256 + walk(rec)


def _exception(exc) -> dict:
    try:
        text = hashlib.sha256(str(exc).encode()).hexdigest()
    except Exception:
        text = None
    out = {"type": type(exc).__qualname__, "mro": [c.__name__ for c in type(exc).__mro__], "sha256": text}
    if hasattr(exc, "status_code"):
        out["status_code"] = _scalar(getattr(exc, "status_code", None))
    return out


def _stand_in(exc):
    """The attributes the summary classifiers read (summary_attempt.py:44-95), each with its absence."""
    if exc is None:
        return None
    response = getattr(exc, "response", _MISSING)
    attrs = {}
    for key, owner, attr in (("status", exc, "status"), ("status_code", exc, "status_code"), ("body", exc, "body"),
                             ("response.status_code", response, "status_code"), ("response.text", response, "text")):
        if owner is not _MISSING and hasattr(owner, attr):
            attrs[key] = _scalar(getattr(owner, attr))
    return {"str": str(exc), "type": type(exc).__qualname__, "mro": [c.__name__ for c in type(exc).__mro__],
            "response": response is not _MISSING and response is not None, "attrs": attrs}


def _scalar(value):
    try:
        return _row_copy(value, [])
    except Exception:
        return Tag("unsupported", type(value).__name__)


def _slot():
    holder = sys.modules.get(_SLOT)
    if holder is None:
        import types
        holder = sys.modules[_SLOT] = types.ModuleType(_SLOT)
        holder.recorder = None
    return holder


def current() -> Recorder | None:
    return _slot().recorder


def install(ctx, engine) -> None:
    """Start recording this process, or add ``engine`` and ``ctx`` to the running recorder. Never raises."""
    try:
        if sys.version_info < _MIN_PYTHON:
            logger.warning("LCM recorder needs Python 3.12 or newer; nothing recorded")
            return
        slot = _slot()
        if slot.recorder is not None:
            slot.recorder.add(ctx, engine)
            return
        mon = sys.monitoring
        tool = next((t for t in (3, 4) if mon.get_tool(t) is None), None)
        if tool is None:
            logger.warning("LCM recorder: sys.monitoring tool ids 3 and 4 are both in use (%s, %s); nothing recorded",
                           mon.get_tool(3), mon.get_tool(4))
            return
        from .config import lcm_env_get

        home = _hermes_home(engine)
        root = Path(lcm_env_get("LCM_RECORDER_DIR") or Path(home) / "lcm-recorder").expanduser()
        mon.use_tool_id(tool, TOOL_NAME)
        recorder = None
        try:
            recorder = Recorder(tool, _mkdir(root), engine, ctx)
            slot.recorder = recorder
            recorder.start(ctx, engine)
        except BaseException:
            slot.recorder = None
            if recorder is not None:
                recorder.writer.closed = True
                with recorder.writer.cv:
                    recorder.writer.cv.notify()
            for info in recorder.roots.values() if recorder is not None else ():
                mon.set_local_events(tool, info[3], 0)
            mon.set_events(tool, 0)
            mon.free_tool_id(tool)
            raise
        if getattr(recorder, "snapshot", {}).get("state") == "refused":
            logger.warning("LCM recorder snapshot refused: %s (%s)", recorder.stopped, recorder.proc)
        else:
            logger.info("LCM recorder on: %s (tool id %d)", recorder.proc, tool)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        logger.warning("LCM recorder could not start; nothing recorded: %s", exc)


def reset_for_tests() -> None:
    slot = _slot()
    if slot.recorder is not None:
        slot.recorder.shutdown_for_tests()
    slot.recorder = None
