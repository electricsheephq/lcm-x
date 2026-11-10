"""Claude Code capture writer and worker: admission, transcript ingest and the spool handshake.

Design: docs/claude-code-adapter.md. This module owns the capture ledger (``capture_events``):
its DDL (``ensure_capture_events``, only in stores it stamps) and its purge, which it registers
with core, so both session delete paths clear it; core never imports this module, so Hermes never
loads either. It writes through the store's own APIs plus plain DML on its ledger and cursor
rows, inside one ``ingest_atomic()`` transaction per batch. One worker per store holds
``capture.lock``; it never opens a store it did not create and stamp.
"""
from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import sqlite3
import time
import uuid
import zlib
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from .claude_code_capture import (
    PROJECTED,
    PROJECTION_VERSION,
    SHARED_RAW,
    SOURCE,
    UNKNOWN,
    classify_entry,
    content_digest,
    lineage_key,
    project_entry,
    results_dir,
    row_session_id,
    session_dir,
)
from .dag import SummaryDAG
from .db_bootstrap import mark_migration_step_complete
from .config import embedding_provider_id
from .embedding_maintenance import run_incremental_embedding_pass
from .embedding_provider import FastembedProvider
from .externalize import extract_externalized_refs, load_externalized_payload
from .store import MessageStore, register_session_purge

OWNER_FILE = "capture-owner.json"
OWNER_KEY = "capture:owner"  # the stamp this writer puts on every store it creates; Hermes never does
HERMES_MARKERS = ("config.yaml", "state.db")
FORGOTTEN_PREFIX = "capture:forgotten:"  # a forgotten session's transcript and files are never read again
CURSOR_PREFIX = "capture:cursor:"
PENDING_PREFIX = "capture:pending:"
COMPACT_PREFIX = "capture:compact:"  # hashed prompt ids of a session's compactions (#2535)
COMPACT_CAP = 256
FORK_PREFIX = "capture:fork:"  # sessions whose SessionStart source was ``fork``, and when they started
LOCAL_GC_PREFIX = "capture:local_gc:"  # sessions seen with Claude Code's local GC on (sticky)
BATCH_LINES = 1000
BATCH_BYTES = 8 * 1024 * 1024
FILE_CHUNK_BYTES = 64 * 1024 * 1024  # a larger tool-results file is kept as several rows
SETTLE_S = 2.0
QUIET_S = 0.5
EMBED_BUDGET_S = 8.0
END_EVENTS = frozenset({"Stop", "StopFailure", "SubagentStop", "SessionEnd", "PreCompact"})
SET_ASIDE_TAGS = (".orphaned-", ".superseded-")
CAPTURE_EVENTS_DDL = (
    "CREATE TABLE IF NOT EXISTS capture_events (event_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, "
    "lineage_key TEXT NOT NULL, source_kind TEXT NOT NULL, generation INTEGER NOT NULL, position INTEGER NOT NULL, "
    "uid TEXT, entry_type TEXT, entry_class TEXT NOT NULL, line_digest TEXT NOT NULL, content_digest TEXT, "
    "disposition TEXT NOT NULL, store_ids TEXT, notes TEXT, raw BLOB, raw_from INTEGER, "
    "projection_version TEXT NOT NULL, captured_at REAL NOT NULL)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_capture_events_line ON capture_events(lineage_key, generation, position) "
    "WHERE source_kind = 'line'",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_capture_events_file "
    "ON capture_events(lineage_key, uid, content_digest, position) WHERE source_kind = 'file'",
    "CREATE INDEX IF NOT EXISTS idx_capture_events_uid ON capture_events(uid) WHERE uid IS NOT NULL",
    # Shared raw lines are kept once per session, so forgetting one session never takes another's bytes.
    "CREATE INDEX IF NOT EXISTS idx_capture_events_shared ON capture_events(session_id, line_digest) "
    "WHERE entry_class = 'shared_raw'",
)


def ensure_capture_events(store: MessageStore) -> None:
    """The capture ledger: the NAMED step ``capture_events_v1``, created only in a store stamped
    ``capture:owner`` by this writer, never in a Hermes profile's store. No SCHEMA_VERSION bump, no
    existing table touched; older readers ignore the table."""
    conn = store.connection
    if conn.in_transaction:
        raise sqlite3.OperationalError("capture ledger setup skipped: the connection is in a transaction")
    if not store.read_metadata_json(OWNER_KEY):
        raise sqlite3.OperationalError("capture ledger refused: the store has no capture:owner stamp")
    with store.atomic(immediate=True):
        for sql in CAPTURE_EVENTS_DDL:
            conn.execute(sql)
        mark_migration_step_complete(conn, "capture_events_v1")


def delete_capture_events(conn: sqlite3.Connection, sessions_sql: str, args: tuple = ()) -> None:
    """A deleted session's capture ledger rows and raw lines go with its messages, on the caller's
    transaction, and the session is marked forgotten: its capture writer never reads its transcript
    or files again, so forgotten content cannot come back. Never creates the table."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'capture_events'").fetchone():
        conn.execute(f"DELETE FROM capture_events WHERE session_id IN ({sessions_sql})", args)
        conn.execute(f"WITH scoped(session_id) AS ({sessions_sql}) INSERT OR REPLACE INTO metadata(key, value) "
                     "SELECT ? || session_id, ? FROM scoped",
                     (*args, FORGOTTEN_PREFIX, json.dumps({"at": round(time.time(), 3)})))


register_session_purge("capture_events", delete_capture_events)


class AdmissionError(RuntimeError):
    """The home or store is not one this writer may write; ``code`` names why."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class Lineage(NamedTuple):
    session_id: str
    agent_id: Optional[str]
    path: Path

    @property
    def key(self) -> str:
        return lineage_key(self.session_id, self.agent_id)

    @property
    def rows_session(self) -> str:
        return row_session_id(self.session_id, self.agent_id)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


# -- admission (H3: one owning harness per store) --

def admit(home: Path, *, host: str = SOURCE, hermes_home: Optional[Path] = None) -> Path:
    """The store path once ``home`` is admitted, creating the store when the home has none.

    Refuses a Hermes home (``hermes_home``, which the worker command reads from HERMES_HOME
    before the store's own settings load; default ``~/.hermes``), an existing unstamped
    database (never adopted), a copied or moved store, and a store whose stamp does not match
    its owner file. Nothing here opens a store for writing before the stamp is checked read-only."""
    home = home.resolve()
    hermes = (hermes_home or Path.home() / ".hermes").expanduser().resolve()
    if home == hermes or home.is_relative_to(hermes) or any((home / name).exists() for name in HERMES_MARKERS):
        raise AdmissionError("hermes_home")
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    database = home / "lcm.db"
    owner = _read_json(home / OWNER_FILE)
    if owner is None:
        if database.exists():
            raise AdmissionError("store_unstamped")
        owner = _publish_owner(home, host)
    if owner.get("home") != str(home):
        raise AdmissionError("store_moved")
    if owner.get("host") != host:
        raise AdmissionError("other_harness")
    if not database.exists():
        _create_store(home, owner)
    stamp = read_stamp(database)
    if not stamp or stamp.get("store_uuid") != owner.get("store_uuid") or stamp.get("host") != host:
        raise AdmissionError("stamp_mismatch")
    return database


def _publish_owner(home: Path, host: str) -> dict:
    """Write the owner file under a temporary name, then link it into place exclusively."""
    owner = {"format": 1, "host": host, "store_uuid": str(uuid.uuid4()), "home": str(home),
             "created_at": round(time.time(), 3)}
    temporary = home / (".%s.%d.tmp" % (OWNER_FILE, os.getpid()))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(owner, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, home / OWNER_FILE)
    except FileExistsError:
        owner = _read_json(home / OWNER_FILE) or {}
    finally:
        temporary.unlink()
    return owner


def _create_store(home: Path, owner: dict) -> None:
    """Build the full core schema under a temporary name, stamp it, and link it into place."""
    temporary = home / (".lcm.db.creating-%d" % os.getpid())
    for leftover in (temporary, *(Path(f"{temporary}{suffix}") for suffix in ("-wal", "-shm", "-journal"))):
        leftover.unlink(missing_ok=True)  # this process's own earlier attempt only
    stamp = {"host": owner["host"], "format": owner["format"], "store_uuid": owner["store_uuid"]}
    store = MessageStore(temporary, hermes_home=str(home))
    try:
        SummaryDAG(temporary).close()  # the reader requires summary_nodes; nothing compacts
        store.update_metadata_json(OWNER_KEY, lambda _current: stamp)
        ensure_capture_events(store)
        to_rollback_journal(store.connection)
    finally:
        store.close()
    try:
        os.link(temporary, home / "lcm.db")
    except FileExistsError:
        raise AdmissionError("store_appeared") from None
    finally:
        temporary.unlink()


def read_stamp(database: Path) -> Optional[dict]:
    """The ``capture:owner`` stamp, read without a write-capable open and without migrating.

    A WAL-mode file whose sidecars are gone is read ``immutable`` only, since a ``mode=ro``
    open could create them; the stamp is written and checkpointed before the store is linked
    into place and never rewritten, so the main file holds it."""
    for query in ("immutable=1",) if _wal_without_sidecars(database) else ("mode=ro", "immutable=1"):
        try:
            with closing(sqlite3.connect(f"{database.resolve().as_uri()}?{query}", uri=True)) as conn:
                row = conn.execute("SELECT value FROM metadata WHERE key = ?", (OWNER_KEY,)).fetchone()
            value = json.loads(row[0]) if row and row[0] else None
            return value if isinstance(value, dict) else None
        except (sqlite3.Error, ValueError):
            continue
    return None


def _wal_without_sidecars(database: Path) -> bool:
    """A WAL-mode header (either format byte 2) whose -wal or -shm file is missing."""
    try:
        with open(database, "rb") as handle:
            handle.seek(18)
            header = handle.read(2)
    except OSError:
        return False
    return len(header) == 2 and max(header) >= 2 and not (
        Path(f"{database}-wal").exists() and Path(f"{database}-shm").exists())


def to_rollback_journal(conn: Optional[sqlite3.Connection], attempts: int = 5) -> bool:
    """Checkpoint and leave WAL mode, so the read-only reader opens the store between worker runs
    (it refuses a WAL store whose sidecars SQLite deleted at the last close). A reader mid-call
    can block the switch; its sidecars then stay and recall keeps working."""
    if conn is None:
        return False
    for attempt in range(attempts):
        if attempt:
            time.sleep(0.1)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if str(conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower() == "delete":
                return True
        except sqlite3.OperationalError:
            pass
    return False


def leave_readable(database: Path) -> bool:
    """After the writer's close, still under the lock: a store left in WAL mode without its
    sidecars (a reader blocked the switch, then closed before the writer, whose close removed
    them) is switched now, so the reader can open it. The switching connection uses exclusive
    locking: from its first read to its close no reader can attach and no shared-memory file
    exists, so a reader can neither block the switch nor leave this close to remove sidecars
    it needs. Sidecars that remain belong to a reader still open; it keeps reading, and the
    next run switches. True when the store, as this close left it, opens for the reader."""
    if not _wal_without_sidecars(database):
        return True
    try:
        with closing(sqlite3.connect(database, timeout=0.2)) as conn:
            conn.execute("PRAGMA locking_mode = EXCLUSIVE")  # before the first read, or it has no effect
            to_rollback_journal(conn)
    except sqlite3.Error:
        pass
    return not _wal_without_sidecars(database)


def open_store(database: Path, home: Path, config: Any = None) -> MessageStore:
    store = MessageStore(database, ingest_protection_config=config, hermes_home=str(home))
    ensure_capture_events(store)
    return store


# -- cursor --

def read_cursor(store: MessageStore, lineage: Lineage) -> Optional[dict]:
    return store.read_metadata_json(CURSOR_PREFIX + lineage.key)


def cursor_state(path: Path, cursor: Optional[dict]) -> str:
    """``ok`` to read on from the cursor; ``missing``; or why a new generation starts.

    Constant time: device and inode, the first line, the last consumed line at its offset,
    and a size not below the offset. The audit re-reads every line in full."""
    try:
        status = os.stat(path)
    except FileNotFoundError:
        return "missing"
    if not cursor:
        return "new"
    if (status.st_dev, status.st_ino) != (cursor["dev"], cursor["ino"]):
        return "replaced"
    if status.st_size < cursor["offset"]:
        return "truncated"
    if cursor["offset"]:
        with open(path, "rb") as handle:
            if _digest(handle.readline()) != cursor["first"]:
                return "rewritten"
            handle.seek(cursor["last_offset"])
            if _digest(handle.readline()) != cursor["last"] or handle.tell() != cursor["offset"]:
                return "rewritten"
    return "ok"


def _new_generation(path: Path, lineage: Lineage, cursor: Optional[dict]) -> dict:
    status = os.stat(path)
    kept = {key: cursor[key] for key in ("fork_parent", "fork_linked") if cursor and key in cursor}
    return {"path": str(path), "session_id": lineage.session_id, "agent_id": lineage.agent_id,
            "dev": status.st_dev, "ino": status.st_ino, "generation": cursor["generation"] + 1 if cursor else 0,
            "offset": 0, "line_index": 0, "first": None, "last_offset": None, "last": None, **kept,
            "named": named_paths(cursor, path)}


def named_paths(saved: Optional[dict], path: Optional[Path] = None) -> list[str]:
    """Every transcript path a lineage's records named (#2535): its saved ``named`` list, its own ``path`` and
    ``path``. A cursor or pending row written before #2535 has no list and names only its own path."""
    names = set()
    if isinstance(saved, dict):
        listed = saved.get("named")
        if isinstance(listed, list):
            names.update(name for name in listed if isinstance(name, str) and name)
        if isinstance(saved.get("path"), str) and saved["path"]:
            names.add(saved["path"])
    if path is not None:
        names.add(str(path))
    return sorted(names)


def _write_metadata(store: MessageStore, key: str, value: Any) -> None:
    with store.atomic():
        store.connection.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, sort_keys=True)))


def read_lines(path: Path, offset: int, *, max_lines: int = BATCH_LINES,
               max_bytes: int = BATCH_BYTES) -> list[tuple[int, bytes]]:
    """Complete, newline-terminated lines from ``offset``; a partly written last line waits.
    No length limit: a long line is read whole, never skipped."""
    lines: list[tuple[int, bytes]] = []
    total = 0
    with open(path, "rb") as handle:
        handle.seek(offset)
        while len(lines) < max_lines and total < max_bytes:
            start = handle.tell()
            line = handle.readline()
            if not line.endswith(b"\n"):
                break
            lines.append((start, line))
            total += len(line)
    return lines


# -- ingest --

def forgotten(store: MessageStore, lineage: Lineage) -> bool:
    """The session was deleted from the store (this module's purge marks it): never read its transcript or files again."""
    return bool(store.read_metadata_json(FORGOTTEN_PREFIX + lineage.rows_session))


def record_facts(record: dict, lineage: Lineage) -> dict:
    """What a pending row keeps from the records that named it (#2535): the events they saw and, for the record's own
    subagent, its type and hashed prompt id. The audit expires only lineages these facts show were never written."""
    seen = record.get("seen")
    events = [name for name in seen if isinstance(name, str)] if isinstance(seen, list) else []
    if isinstance(record.get("event"), str):
        events.append(record["event"])
    own = lineage.agent_id is not None and lineage.agent_id == record.get("agent_id")
    agent_type = record.get("agent_type") if own and isinstance(record.get("agent_type"), str) else None
    prompt = record.get("prompt_id") if own and isinstance(record.get("prompt_id"), str) else None
    seen = record.get("agent_transcript_seen" if lineage.agent_id is not None else "transcript_seen")
    return {"events": sorted(set(events)), "agent_type": agent_type, "prompt": prompt,
            "transcript_seen": seen if isinstance(seen, bool) else None}


def _merge_facts(saved: Any, facts: Optional[dict]) -> Optional[dict]:
    if facts is None:
        return saved if isinstance(saved, dict) else None
    old = saved if isinstance(saved, dict) else {}
    events = set(facts["events"]) | {name for name in old.get("events") or () if isinstance(name, str)}
    # True once any record saw the transcript on disk; an unknown (an older hook, or a row from before this field)
    # stays unknown, so the audit never expires on it.
    values = [facts.get("transcript_seen"), old.get("transcript_seen") if old else False]
    seen = True if True in values else (False if all(value is False for value in values) else None)
    return {"events": sorted(events),
            "agent_type": facts["agent_type"] if facts["agent_type"] is not None else old.get("agent_type"),
            "prompt": facts["prompt"] if facts["prompt"] is not None else old.get("prompt"),
            "transcript_seen": seen}


def drain(store: MessageStore, lineage: Lineage, *, deadline: Optional[float] = None,
          facts: Optional[dict] = None) -> str:
    """Ingest a transcript to its current end: ``drained``, ``missing``, ``forgotten`` or ``deadline``.
    ``facts`` (from :func:`record_facts`) are kept on a pending row for the audit."""
    if forgotten(store, lineage):
        return "forgotten"
    cursor = read_cursor(store, lineage)
    if cursor and str(lineage.path) not in named_paths(cursor):
        # Another path named for a captured lineage is remembered once, so the audit checks it if it ever holds lines
        # (#2535). The cursor still follows one path; capture is unchanged.
        cursor = {**cursor, "named": named_paths(cursor, lineage.path)}
        _write_metadata(store, CURSOR_PREFIX + lineage.key, cursor)
    state = cursor_state(lineage.path, cursor)
    if state == "missing":
        key = PENDING_PREFIX + lineage.key
        pending = store.read_metadata_json(key) if not cursor else None
        recorded = pending.get("path") if isinstance(pending, dict) else None
        replace = recorded != str(lineage.path) and not (isinstance(recorded, str) and recorded and Path(recorded).exists())
        named = named_paths(pending, lineage.path)
        merged = _merge_facts(pending.get("facts") if isinstance(pending, dict) else None, facts)
        if not cursor and (replace or not isinstance(pending, dict) or named != pending.get("named")
                           or merged != pending.get("facts")):
            # A newer record's path replaces the recorded one while that one is still absent (the sweep must look
            # where the transcript will be); a recorded transcript that has appeared keeps its reference, so the
            # sweep drains it. Every named path is kept for the audit (#2535). first_seen keeps the first sighting.
            first_seen = pending.get("first_seen") if isinstance(pending, dict) else None
            _write_metadata(store, key, {
                "session_id": lineage.session_id, "agent_id": lineage.agent_id,
                "path": str(lineage.path) if replace else recorded, "named": named,
                "first_seen": first_seen if isinstance(first_seen, (int, float)) else time.time(),
                **({"facts": merged} if merged is not None else {})})
            # Durable before the worker can mark the claimed record done and unlink it.
        return "missing"
    if state == "replaced":
        old = _set_aside_copy(lineage.path, cursor)
        if old is not None and cursor_state(old, cursor) == "ok":
            cursor = _drain_file(store, lineage, old, cursor, deadline)
            if cursor is None:
                return "deadline"
    if state != "ok":
        cursor = _new_generation(lineage.path, lineage, cursor)
        if state == "new":  # the pending row's named paths carry into the first cursor
            cursor["named"] = sorted(set(cursor["named"])
                                     | set(named_paths(store.read_metadata_json(PENDING_PREFIX + lineage.key))))
        if state == "new":  # registered before its first batch, so the sweep retries a lineage that fails
            with store.atomic():
                store.connection.execute(
                    "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (CURSOR_PREFIX + lineage.key, json.dumps(cursor, sort_keys=True)))
                store.connection.execute("DELETE FROM metadata WHERE key = ?", (PENDING_PREFIX + lineage.key,))
    return "deadline" if _drain_file(store, lineage, lineage.path, cursor, deadline) is None else "drained"


def _drain_file(store: MessageStore, lineage: Lineage, path: Path, cursor: dict,
                deadline: Optional[float]) -> Optional[dict]:
    while True:
        if deadline is not None and time.monotonic() > deadline:
            return None
        lines = read_lines(path, cursor["offset"])
        if not lines:
            return cursor
        cursor = ingest_lines(store, lineage, cursor, lines)


def _set_aside_copy(path: Path, cursor: dict) -> Optional[Path]:
    """The old file of a replaced transcript, if Claude Code set it aside beside the new one."""
    try:
        siblings = list(path.parent.iterdir())
    except OSError:
        return None
    for sibling in siblings:
        if sibling == path or not any(tag in sibling.name for tag in SET_ASIDE_TAGS):
            continue
        try:
            status = sibling.stat()
        except OSError:
            continue
        if (status.st_dev, status.st_ino) == (cursor["dev"], cursor["ino"]):
            return sibling
    return None


class _Line(NamedTuple):
    position: int
    raw: bytes
    entry: Optional[dict]
    entry_class: str
    uid: Optional[str]
    digest: Optional[str]


def ingest_lines(store: MessageStore, lineage: Lineage, cursor: dict, lines: list[tuple[int, bytes]]) -> dict:
    """One batch: rows, bindings, relations, ledger rows and the cursor in one transaction.

    Identity is (lineage, uid, content digest): the same again is a duplicate (no rows); a known
    uid with other content is a revision (new rows, ``version`` bindings, ``supersedes``). In a
    session that started as a fork, an entry written before the fork started is a copy: the
    fork's own rows, marked ``fork_copy`` and ``fork_copy_of`` the row that first held that
    content in the session that wrote it. A copied revision is both: it supersedes the fork's
    earlier version and stays marked as a copy, so it is never taken as its writer's row. A
    line whose projection fails stays raw, noted."""
    folder = session_dir(lineage.path)
    parsed: list[_Line] = []
    for index, (_offset, raw) in enumerate(lines):
        try:
            entry = json.loads(raw)
        except ValueError:
            entry = None
        entry = entry if isinstance(entry, dict) else None
        uid = entry.get("uuid") if entry and isinstance(entry.get("uuid"), str) and entry["uuid"] else None
        parsed.append(_Line(cursor["line_index"] + index, raw, entry,
                            classify_entry(entry) if entry else UNKNOWN, uid,
                            content_digest(entry) if entry else None))
    uids = {line.uid for line in parsed if line.uid and line.entry_class == PROJECTED}
    known = _versions(store, lineage.key, uids)
    fork, started = _fork_start(store, lineage)
    origins = _origins(store, lineage.key, uids) if fork else {}
    copied_from = cursor.get("fork_parent")  # the session that wrote the latest copy so far
    parent: Optional[str] = None

    messages: list[dict] = []
    pending: list[tuple[str, str, list[tuple[str, Any]], dict]] = []
    events: list[dict] = []
    for line in parsed:
        event: dict[str, Any] = {"line": line, "disposition": "raw", "version": None, "files": [], "notes": []}
        events.append(event)
        if line.entry_class != PROJECTED:
            continue
        if not line.uid:
            event["notes"] = ["projected_entry_without_uuid"]  # kept raw; the audit counts it
            continue
        versions = known.setdefault(line.uid, [])
        same = next((version for version in versions if version["digest"] == line.digest), None)
        if same is not None:
            event.update(disposition="duplicate", version=same)
            continue
        try:
            projection = project_entry(line.entry, folder=folder)
        except Exception as error:  # noqa: BLE001 - the line stays raw and noted; capture goes on
            event["notes"] = ["projection_error:" + type(error).__name__]
            continue
        origin = origins.get((line.uid, line.digest))
        links: list[tuple[str, dict[str, int]]] = []
        if versions:  # a revision supersedes this lineage's previous version, copied or not
            links.append(("supersedes", versions[-1]["row_ids"]))
        if fork and _copied(line.entry, started, origin):
            disposition, proof = "fork_copy", "fork_copy"
            if origin:
                links.append(("fork_copy_of", origin["row_ids"]))
                copied_from = origin["session"]
        else:
            disposition, proof = ("version" if versions else "new"), "transcript_uuid"
            if fork and not cursor.get("fork_linked") and parent is None:
                parent = copied_from or ""  # the fork's own first entry ends its copied prefix
        kind = "version" if versions else "canonical"
        # Filled in place inside the transaction, so later lines of this batch see the store ids.
        version = {"digest": line.digest, "store_ids": [], "row_ids": {}}
        versions.append(version)
        event.update(disposition=disposition, version=version, files=projection.files, notes=projection.problems)
        pending.append((kind, proof, projection.rows, {"event": event, "links": links}))
        messages.extend(row for _uid, row in projection.rows)
    _link_lineage(store, lineage, cursor, parent)

    with store.ingest_atomic():
        store_ids = iter(store.append_batch(lineage.rows_session, messages, source=SOURCE) if messages else [])
        bindings: list[tuple[int, str, str, str]] = []
        relations: list[list[tuple]] = []
        for kind, proof, rows, state in pending:
            version = state["event"]["version"]
            for row_uid, _row in rows:
                store_id = next(store_ids)
                version["store_ids"].append(store_id)
                version["row_ids"][row_uid] = store_id
                bindings.append((store_id, row_uid, kind, proof))
            for relation, previous in state["links"]:
                relations.extend([(new, relation, old, 0, None)] for new, old in _pairs(version["store_ids"], previous))
        store.add_host_uid_bindings(lineage.key, bindings)
        if relations:
            store.add_message_relations(relations)
        _write_events(store, lineage, cursor["generation"], events)
        cursor = _advance(cursor, lines)
        if fork:
            cursor.update(fork_parent=copied_from, fork_linked=bool(cursor.get("fork_linked")) or parent is not None)
        store.connection.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (CURSOR_PREFIX + lineage.key, json.dumps(cursor, sort_keys=True)))
    return cursor


def _fork_start(store: MessageStore, lineage: Lineage) -> tuple[bool, Optional[float]]:
    """Whether this is a fork's main thread, and when the fork started (None if unknown)."""
    if lineage.agent_id is not None:
        return False, None
    flag = store.read_metadata_json(FORK_PREFIX + lineage.session_id)
    started = flag.get("started_at") if isinstance(flag, dict) else None
    return bool(flag), float(started) if isinstance(started, (int, float)) else None


def _copied(entry: dict, started: Optional[float], origin: Optional[dict]) -> bool:
    """A fork's entry is a copy when it was written before the fork started (Claude Code keeps
    each copied entry's timestamp), else, with no start time, when its writer's row is known."""
    try:
        written = datetime.fromisoformat(str(entry["timestamp"]).replace("Z", "+00:00")).timestamp()
    except (KeyError, TypeError, ValueError):
        written = None
    if started is None or written is None:
        return origin is not None
    return written < started


def _component(uid: str) -> int:
    head, mark, tail = uid.rpartition("#")
    return int(tail) if mark and head and tail.isdigit() else 0


def _pairs(new_ids: list[int], previous: dict[str, int]) -> list[tuple[int, int]]:
    """(new row, previous row) links, row for row in projection order. When a revision changed
    the number of rows, extra new rows link to the last previous row and extra previous rows to
    the last new row, so every previous row has a successor and every new row a predecessor."""
    old_ids = [store_id for _uid, store_id in sorted(previous.items(), key=lambda item: _component(item[0]))]
    if not old_ids or not new_ids:
        return []
    pairs = [(new_id, old_ids[min(index, len(old_ids) - 1)]) for index, new_id in enumerate(new_ids)]
    return pairs + [(new_ids[-1], old_id) for old_id in old_ids[len(new_ids):]]


def _advance(cursor: dict, lines: list[tuple[int, bytes]]) -> dict:
    last_offset, last = lines[-1]
    first = cursor["first"] or (_digest(lines[0][1]) if cursor["offset"] == 0 else cursor["first"])
    return {**cursor, "first": first, "offset": last_offset + len(last), "line_index": cursor["line_index"] + len(lines),
            "last_offset": last_offset, "last": _digest(last)}


def _versions(store: MessageStore, key: str, uids: Iterable[str]) -> dict[str, list[dict]]:
    """Per uid, the content versions this lineage already holds, oldest first. The uid index is
    named: without statistics SQLite would scan the whole lineage on every batch."""
    found: dict[str, list[dict]] = {}
    uids = sorted(uids)
    for start in range(0, len(uids), 500):
        chunk = uids[start:start + 500]
        for uid, digest, store_ids in store.connection.execute(
                f"SELECT uid, content_digest, store_ids FROM capture_events INDEXED BY idx_capture_events_uid "
                f"WHERE uid IN ({','.join('?' * len(chunk))}) AND lineage_key = ? AND source_kind = 'line' "
                "AND disposition IN ('new', 'version', 'fork_copy') ORDER BY event_id", [*chunk, key]):
            found.setdefault(uid, []).append({"digest": digest, "store_ids": json.loads(store_ids or "[]")})
    _fill_row_ids(store, key, [version for versions in found.values() for version in versions])
    return found


def _fill_row_ids(store: MessageStore, key: str, versions: list[dict]) -> None:
    """Each version's ``row_ids`` (row uid -> store id), from one binding lookup for all of them."""
    owners = store.host_uid_uids_of_stores(key, [store_id for version in versions for store_id in version["store_ids"]])
    for version in versions:
        version["row_ids"] = {uid: store_id for store_id in version["store_ids"] for uid in owners.get(store_id, ())}


def _origins(store: MessageStore, key: str, uids: Iterable[str]) -> dict[tuple[str, str], dict]:
    """For a fork's entries, per (uid, content digest): the row that first held that content in
    a session that wrote it (``new`` or ``version``), never another fork's copy, so sibling forks
    point at the same writer."""
    found: dict[tuple[str, str], dict] = {}
    uids = sorted(uids)
    for start in range(0, len(uids), 500):
        chunk = uids[start:start + 500]
        for uid, other, session, digest, store_ids in store.connection.execute(
                f"SELECT uid, lineage_key, session_id, content_digest, store_ids FROM capture_events "
                f"INDEXED BY idx_capture_events_uid WHERE uid IN ({','.join('?' * len(chunk))}) AND lineage_key != ? "
                "AND source_kind = 'line' AND disposition IN ('new', 'version') ORDER BY event_id", [*chunk, key]):
            found.setdefault((uid, digest), {"lineage": other, "session": session,
                                             "store_ids": json.loads(store_ids or "[]")})
    by_lineage: dict[str, list[dict]] = {}
    for origin in found.values():
        by_lineage.setdefault(origin["lineage"], []).append(origin)
    for other, origins in by_lineage.items():
        _fill_row_ids(store, other, origins)
    return found


def _link_lineage(store: MessageStore, lineage: Lineage, cursor: dict, parent: Optional[str]) -> None:
    """Parent links go in before the batch's transaction (the lineage write runs its own and is
    idempotent), so a crash after the batch commits cannot lose them. A fork's parent is the
    session that wrote its last copied entry, known once the fork writes its own first entry."""
    if lineage.agent_id and cursor["offset"] == 0:
        store.record_session_lineage(lineage.rows_session, lineage.session_id, source=SOURCE,
                                     child_subagent_id=lineage.agent_id, child_role="subagent")
    if parent and parent != lineage.rows_session:
        store.record_session_lineage(lineage.rows_session, parent, source=SOURCE, child_role="fork")


INSERT_EVENT = (
    "INSERT OR IGNORE INTO capture_events(session_id, lineage_key, source_kind, generation, position, uid, "
    "entry_type, entry_class, line_digest, content_digest, disposition, store_ids, notes, raw, raw_from, "
    "projection_version, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
# A file's chunk rows are always new; a conflict raises rather than leave ``lastrowid`` stale.
INSERT_CHUNK = INSERT_EVENT.replace("INSERT OR IGNORE", "INSERT", 1)


def _write_events(store: MessageStore, lineage: Lineage, generation: int, events: list[dict]) -> None:
    conn = store.connection
    now = time.time()
    shared: dict[str, int] = {}
    results = results_dir(session_dir(lineage.path)) if any(event["files"] for event in events) else None
    for event in events:
        line: _Line = event["line"]
        digest = _digest(line.raw)
        raw, raw_from = zlib.compress(line.raw), None
        if line.entry_class == SHARED_RAW:  # kept once per session: a forget never takes another session's bytes
            if digest not in shared:
                held = conn.execute("SELECT event_id FROM capture_events WHERE session_id = ? AND line_digest = ? "
                                    "AND entry_class = 'shared_raw' AND raw IS NOT NULL LIMIT 1",
                                    (lineage.rows_session, digest)).fetchone()
                if held:
                    shared[digest] = int(held[0])
            if digest in shared:
                raw, raw_from = None, shared[digest]
        store_ids = event["version"]["store_ids"] if event["version"] else None
        cursor = conn.execute(INSERT_EVENT, (
            lineage.rows_session, lineage.key, "line", generation, line.position, line.uid,
            line.entry.get("type") if line.entry else None, line.entry_class, digest, line.digest,
            event["disposition"], json.dumps(store_ids) if store_ids is not None else None,
            json.dumps(event["notes"]) if event["notes"] else None, raw, raw_from, PROJECTION_VERSION, now))
        if line.entry_class == SHARED_RAW and raw is not None and cursor.lastrowid:
            shared[digest] = int(cursor.lastrowid)
        for name in event["files"]:
            row = _held_file_row(store, lineage, results, name, store_ids, now)
            if row is not None:
                conn.execute(INSERT_EVENT, row)


def _fingerprint(path: Path) -> str:
    status = path.stat()
    return json.dumps({"size": status.st_size, "mtime_ns": status.st_mtime_ns}, sort_keys=True)


def _held_file_row(store: MessageStore, lineage: Lineage, results: Optional[Path], name: str,
                   store_ids: Optional[list[int]], now: float) -> Optional[tuple]:
    """A ``tool-results`` file whose exact text a tool row holds, as stored: digest and row ids,
    no bytes. None when no row holds it whole (ingest protection redacted it or externalized only
    parts of it, or the file changed after its line was read): the sweep keeps the file's bytes."""
    if results is None:
        return None
    try:
        fingerprint = _fingerprint(results / name)
        data = (results / name).read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not _rows_hold(store, lineage.rows_session, store_ids or [], text):
        return None
    digest = _digest(data)
    return (lineage.rows_session, lineage.key, "file", 0, 0, "tool-results/" + name, None, "tool_results_file",
            digest, digest, "held", json.dumps(store_ids or []), fingerprint, None, None, PROJECTION_VERSION, now)


def _rows_hold(store: MessageStore, session: str, store_ids: list[int], text: str) -> bool:
    """Whether one of these rows, as stored, holds ``text`` exactly: inline, or as the whole
    content of an externalized payload it references."""
    for store_id in store_ids:
        row = store.connection.execute("SELECT content FROM messages WHERE session_id = ? AND store_id = ?",
                                       (session, store_id)).fetchone()
        content = row[0] if row else None
        if content == text:
            return True
        for ref in extract_externalized_refs(content) if isinstance(content, str) else ():
            try:
                payload = load_externalized_payload(ref, config=store._ingest_protection_config,
                                                    hermes_home=store._hermes_home)
            except (OSError, ValueError):
                payload = None
            if payload and payload.get("content") == text:
                return True
    return False


class _KnownBytes(Exception):
    """The ledger already holds this file's bytes: the file was only touched."""


def capture_files(store: MessageStore, lineage: Lineage, *, deadline: Optional[float] = None) -> Optional[int]:
    """Every ``tool-results`` file of a session gets ledger rows holding its bytes, in rows of at
    most ``FILE_CHUNK_BYTES``, unless a tool row holds its full text. A file is read only when
    it is new or its size or modification time changed, a held one too: its new bytes are kept.
    ``None`` when the deadline came first."""
    results = results_dir(session_dir(lineage.path))
    if results is None or forgotten(store, lineage):
        return 0
    try:
        paths = sorted(path for path in results.iterdir() if path.is_file() and not path.is_symlink())
    except OSError:
        return 0
    conn = store.connection
    added = 0
    for path in paths:
        if deadline is not None and time.monotonic() > deadline:
            return None
        uid = "tool-results/" + path.name
        try:
            fingerprint = _fingerprint(path)
        except OSError:
            continue
        if conn.execute("SELECT 1 FROM capture_events WHERE source_kind = 'file' AND lineage_key = ? AND uid = ? "
                        "AND notes = ? LIMIT 1", (lineage.key, uid, fingerprint)).fetchone():
            continue
        try:
            _keep_file(store, lineage, path, uid, fingerprint)
        except _KnownBytes:
            with store.ingest_atomic():  # remember the new fingerprint, so the file is not read again
                conn.execute("UPDATE capture_events SET notes = ? WHERE source_kind = 'file' AND lineage_key = ? "
                             "AND uid = ? AND position = 0", (fingerprint, lineage.key, uid))
            continue
        except OSError:
            continue
        added += 1
    return added


def _keep_file(store: MessageStore, lineage: Lineage, path: Path, uid: str, fingerprint: str) -> None:
    """The file's bytes as ledger rows, one per chunk, in one transaction (rolled back when the
    ledger already holds the same bytes)."""
    conn = store.connection
    whole = hashlib.sha256()
    now = time.time()
    with store.ingest_atomic(), open(path, "rb") as handle:
        event_ids = []
        while True:
            data = handle.read(FILE_CHUNK_BYTES)
            if event_ids and not data:
                break
            whole.update(data)
            cursor = conn.execute(INSERT_CHUNK, (
                lineage.rows_session, lineage.key, "file", 0, len(event_ids), uid, None, "tool_results_file",
                _digest(data), None, "kept", None, fingerprint if not event_ids else None, zlib.compress(data),
                None, PROJECTION_VERSION, now))
            event_ids.append(cursor.lastrowid)
            if len(data) < FILE_CHUNK_BYTES:
                break
        digest = whole.hexdigest()
        if conn.execute("SELECT 1 FROM capture_events WHERE source_kind = 'file' AND lineage_key = ? AND uid = ? "
                        "AND content_digest = ? LIMIT 1", (lineage.key, uid, digest)).fetchone():
            raise _KnownBytes()
        conn.executemany("UPDATE capture_events SET content_digest = ? WHERE event_id = ?",
                         [(digest, event_id) for event_id in event_ids])


# -- worker --

def registered(store: MessageStore) -> list[Lineage]:
    """Every cursor or pending transcript, with cursors winning: the sweep's list."""
    lineages = {}
    for prefix in (CURSOR_PREFIX, PENDING_PREFIX):
        for _key, value in store.connection.execute(
                "SELECT key, value FROM metadata WHERE key >= ? AND key < ?", (prefix, prefix + "￿")):
            try:
                saved = json.loads(value)
                lineage = Lineage(saved["session_id"], saved.get("agent_id"), Path(saved["path"]))
                lineages.setdefault(lineage.key, lineage)
            except (TypeError, ValueError, KeyError):
                continue
    return list(lineages.values())


def lineages_of(record: dict) -> list[Lineage]:
    """The transcripts a spool record names: the session's, and a subagent's if any."""
    session = record.get("session_id")
    found = []
    if record.get("transcript_path"):
        found.append(Lineage(session, None, Path(record["transcript_path"])))
    if record.get("agent_id") and record.get("agent_transcript_path"):
        found.append(Lineage(session, record["agent_id"], Path(record["agent_transcript_path"])))
    return found


def _subagents(lineage: Lineage) -> list[Lineage]:
    if lineage.agent_id:
        return []
    folder = session_dir(lineage.path) / "subagents"
    try:
        names = sorted(path.name for path in folder.glob("agent-*.jsonl"))
    except OSError:
        return []
    return [Lineage(lineage.session_id, name[len("agent-"):-len(".jsonl")], folder / name) for name in names]


def _settle(store: MessageStore, lineage: Lineage, deadline: Optional[float], settle_s: float) -> None:
    """The transcript lags the hooks (the final reply was missing at Stop in 2 of 4 R0 runs):
    keep draining until the file has been quiet for a moment, bounded."""
    end = time.monotonic() + settle_s
    quiet_since = time.monotonic()
    size = _size(lineage.path)
    while time.monotonic() < end and (deadline is None or time.monotonic() < deadline):
        time.sleep(0.1)
        now_size = _size(lineage.path)
        if now_size != size:
            size, quiet_since = now_size, time.monotonic()
            drain(store, lineage, deadline=deadline)
        elif time.monotonic() - quiet_since >= QUIET_S:
            return


def _size(path: Path) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return -1


def claim_records(home: Path) -> list[tuple[Path, dict]]:
    """Claim every spool record by renaming it, so a hook firing meanwhile writes a new record
    instead of having its record deleted unread. Claimed files left by a crashed worker are
    processed again; ingest is idempotent."""
    directory = home / "spool"
    claimed: list[tuple[Path, dict]] = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return claimed
    for name in names:
        path = directory / name
        if name.endswith(".json") and not name.startswith("."):
            target = directory / (name[:-len(".json")] + ".claimed")
            if target.exists():
                continue  # an older claim of the same key is processed first; this one waits
            try:
                os.rename(path, target)
            except OSError:
                continue
            path = target
        elif not name.endswith(".claimed"):
            continue
        record = _read_json(path)
        claimed.append((path, record or {}))
    return claimed


def pending(home: Path) -> bool:
    """Records waiting: new or claimed. ``.failed`` records wait for their session's next hook."""
    try:
        return any(name.endswith((".json", ".claimed")) and not name.startswith(".")
                   for name in os.listdir(home / "spool"))
    except OSError:
        return False


class _Lock:
    def __init__(self, home: Path):
        self.handle = open(home / "capture.lock", "a")

    def acquire(self, wait_s: float) -> bool:
        end = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() >= end:
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        fcntl.flock(self.handle, fcntl.LOCK_UN)

    def close(self) -> None:
        self.handle.close()


def run_worker(home: Path, *, deadline_s: Optional[float] = None, config: Any = None,
               settle_s: float = SETTLE_S, hermes_home: Optional[Path] = None) -> dict:
    """Drain the spool under the store lock, then release the lock and rescan: a record that
    arrived while the lock was held is processed by this worker (it retakes the lock) or by the
    worker now holding it, which took the lock after the record was written and scans first.

    ``deadline_s`` (the PreCompact drain) bounds the run; ``complete`` is true only when every
    record was drained. No run waits for the lock, so no hook waits on another worker: a held lock
    means a holder that sees every spooled record when it rescans. A bounded run that finds the
    lock held at once reports ``lock: held``; one that ends incomplete leaves the rest to a
    detached worker, which the hook starts."""
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    report: dict[str, Any] = {"records": 0, "drained": 0, "missing": 0, "forgotten": 0, "failed": 0, "files": 0,
                              "complete": True, "embedding": "skipped_bounded" if deadline is not None else
                              "skipped_idle", "vectors": 0}
    embedding_state: dict[str, Any] = {}
    lock = _Lock(home)
    try:
        first = True
        while True:
            if not lock.acquire(0.0):
                if deadline_s is not None:
                    report["complete"] = False
                    if first:
                        report["lock"] = "held"  # this bounded run drained nothing: the holder will
                return report
            first = False
            try:
                status = _drain_once(home, report, deadline, config, settle_s, hermes_home, embedding_state)
            finally:
                lock.release()
            if status == "deadline":
                report["deadline"] = True  # the bound, not a failure or the lock, ended this run
            if status != "done" or not pending(home):
                return report
    finally:
        lock.close()


def _drain_once(home: Path, report: dict, deadline: Optional[float], config: Any, settle_s: float,
                hermes_home: Optional[Path], embedding_state: dict) -> str:
    """``done``, ``refused`` or ``deadline``. A record whose drain raises is set aside as
    ``.failed`` (the next hook of that session spools a fresh one), so one bad transcript
    cannot stall the others or loop the worker. A drained record is removed only after the
    sweep, which captures its saved outputs: a deadline or a kill before then leaves it claimed."""
    try:
        database = admit(home, hermes_home=hermes_home)
    except AdmissionError as refused:
        report.update(refused=refused.code, complete=False)
        return "refused"
    store = open_store(database, home, config)
    try:
        before = store.connection.execute("SELECT MAX(store_id) FROM messages").fetchone()[0]
        claimed = claim_records(home)
        done: list[Path] = []  # removed only once the sweep below has run: a cut sweep keeps them
        for _path, record in claimed:  # before any drain, so a fork's first rows already know it is one
            remember_session(store, record)
        for path, record in claimed:
            report["records"] += 1
            try:
                for lineage in lineages_of(record):
                    outcome = drain(store, lineage, deadline=deadline, facts=record_facts(record, lineage))
                    if outcome == "deadline":
                        report["complete"] = False
                        return "deadline"  # the claimed record stays for the next run
                    report[outcome] += 1
                    if outcome == "drained" and record.get("event") in END_EVENTS:
                        _settle(store, lineage, deadline, settle_s)
            except Exception as error:  # noqa: BLE001 - logged without content; capture goes on
                report["failed"] += 1
                report["complete"] = False
                _log_failure(home, "record", error)
                os.replace(path, path.with_suffix(".failed"))
                continue
            done.append(path)
        seen: set[str] = set()
        for lineage in registered(store):
            for each in (lineage, *_subagents(lineage)):
                if each.key in seen:
                    continue
                seen.add(each.key)
                try:
                    if drain(store, each, deadline=deadline) == "deadline":
                        report["complete"] = False
                        return "deadline"
                    if each.agent_id is None:
                        added = capture_files(store, each, deadline=deadline)
                        if added is None:
                            report["complete"] = False
                            return "deadline"
                        report["files"] += added
                except Exception as error:  # noqa: BLE001
                    report["failed"] += 1
                    report["complete"] = False
                    _log_failure(home, "sweep", error)
        for path in done:
            path.unlink(missing_ok=True)
        token, vectors = _embed_after_drain(home, database, store, config, deadline, before, embedding_state)
        report["embedding"] = token
        report["vectors"] += vectors
        return "done"
    finally:
        switched = to_rollback_journal(store.connection)
        store.close()
        if not switched and not leave_readable(database):
            report["complete"] = False  # a bounded run's hook starts a worker; else the next hook's does
            _log_failure(home, "journal", "still_wal")


def _embed_after_drain(home: Path, database: Path, store: MessageStore, config: Any,
                       deadline: Optional[float], before: Optional[int], state: dict) -> tuple[str, int]:
    """Catch up local vectors in slices, yielding to capture and the run's budget. The chunk model
    loads before the first slice and outside the budget, so a slice is embedding only and a record
    spooled during the load is captured before the first batch; once the budget has run out, no
    round searches for a chunk or loads the model."""
    if deadline is not None:
        return "skipped_bounded", 0
    if config is None or not config.embeddings_enabled:
        return "skipped_disabled", 0
    provider = str(getattr(config, "embedding_provider", "") or "").strip().lower()
    if embedding_provider_id(provider) != "fastembed":
        return "skipped_provider", 0
    if pending(home):
        return "skipped_pending", 0
    if store.connection.in_transaction:
        return "skipped_transaction", 0
    if store.connection.execute("SELECT MAX(store_id) FROM messages").fetchone()[0] == before:
        return "skipped_idle", 0
    end = state.setdefault("capture_budget_end", time.monotonic() + EMBED_BUDGET_S)
    if state.get("model_load_s") is None and time.monotonic() < end:
        state["model_load_s"] = _load_chunk_model(database, config)
        end = state["capture_budget_end"] = end + (state["model_load_s"] or 0.0)  # the load is timed apart
    token, vectors = "budget", 0
    try:
        while time.monotonic() < end:
            if pending(home):
                token = "yielded"
                break
            try:
                result = run_incremental_embedding_pass(database, config, slice_state=state)
            except Exception:  # noqa: BLE001 - embedding never changes capture's result
                token = "error"
                break
            token = " ".join(part.split(":", 1)[0] for part in result["outcome"].split())
            vectors += state["vectors_written"]
            if not state["continue"]:
                break
            if pending(home):
                token = "yielded"
                break
            remaining = end - time.monotonic()
            if remaining <= 0 or state["delay"] > remaining:
                token = "budget"
                break
            wake = time.monotonic() + state["delay"]
            while True:
                if pending(home):
                    token = "yielded"
                    break
                delay = wake - time.monotonic()
                if delay <= 0:
                    break
                time.sleep(min(0.05, delay))
            if token == "yielded":
                break
        else:
            token = "budget"
    finally:
        gc.collect()
    if (token == "error" or any(mark in token for mark in ("=error", "=failed", "=refused"))
            or token == "provider_unavailable"
            or (token == "auto_register_unavailable" and config.embeddings_enabled_explicit)):
        _log_failure(home, "embed", token)
    return token, vectors


def _load_chunk_model(database: Path, config: Any) -> Optional[float]:
    """Load the chunk model when the pass's own read-only discovery finds a chunk to embed, and
    return the load's seconds. None, with nothing loaded, when no chunk waits, when no chunk
    profile exists yet (registering one loads the model in a slice of its own), or when the load
    fails (the pass then reports its own outcome)."""
    from . import command  # lazy: command imports most of the plugin
    from .chunking import normalize_content_policy

    try:
        conn = command._embedding_read_connection(database)
        try:
            profile = command._chunk_current_profile(conn)
            if profile is None or embedding_provider_id(str(profile["provider"])) != "fastembed":
                return None
            policy = normalize_content_policy(getattr(config, "embedding_content_policy", "conversational"))
            _, waiting, _ = command._chunk_pending_rows(conn, str(profile["identity_hash"]), policy, 1,
                                                        incremental=True)
        finally:
            conn.close()
        if not waiting:
            return None
        started = time.monotonic()
        FastembedProvider(str(profile["model_name"]))._ensure_local()  # cached files only, never a download
        return time.monotonic() - started
    except Exception:  # noqa: BLE001 - embedding never changes capture's result
        return None


def remember_session(store: MessageStore, record: dict) -> None:
    """What the fork rules and the audit need from a hook record, kept before the record is
    acknowledged: that the session started as a fork, and when; and that Claude Code's local
    GC was on for it, which stays set once seen."""
    session = record.get("session_id")
    if not isinstance(session, str) or not session:
        return
    if record.get("start_source") == "fork" and not isinstance(store.read_metadata_json(FORK_PREFIX + session), dict):
        started = record.get("started_at")
        flag = {"started_at": started} if isinstance(started, (int, float)) else True
        store.update_metadata_json(FORK_PREFIX + session, lambda current: current if isinstance(current, dict) else flag)
    if record.get("local_gc") is True and not store.read_metadata_json(LOCAL_GC_PREFIX + session):
        store.update_metadata_json(LOCAL_GC_PREFIX + session, lambda _current: True)
    prompts = [value for value in record.get("compact_prompts") or () if isinstance(value, str)]
    if prompts:  # kept even when the session has a cursor: the summarizer subagent is matched against it (#2535)
        known = store.read_metadata_json(COMPACT_PREFIX + session)
        known = [value for value in known if isinstance(value, str)] if isinstance(known, list) else []
        new = [value for value in prompts if value not in known]
        if new:
            _write_metadata(store, COMPACT_PREFIX + session, (known + new)[-COMPACT_CAP:])


def _log_failure(home: Path, stage: str, error: BaseException | str) -> None:
    """One line per failure, with the error type only: never content or paths."""
    try:
        directory = home / "logs"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(directory / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        kind = error if isinstance(error, str) else type(error).__name__
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": round(time.time(), 3), "stage": stage, "error": kind}) + "\n")
    except OSError:
        pass
