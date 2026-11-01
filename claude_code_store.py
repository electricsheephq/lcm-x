"""Claude Code capture writer and worker: admission, transcript ingest and the spool handshake.

Design: docs/claude-code-adapter.md. The DDL lives in store.py (``ensure_capture_events``);
this module writes through the store's own APIs plus plain DML on its ledger and cursor
rows, inside one ``ingest_atomic()`` transaction per batch. One worker per store holds
``capture.lock``; it never opens a store it did not create and stamp.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
import uuid
import zlib
from contextlib import closing
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
    row_session_id,
    session_dir,
)
from .dag import SummaryDAG
from .store import CAPTURE_OWNER_KEY, MessageStore

OWNER_FILE = "capture-owner.json"
HERMES_MARKERS = ("config.yaml", "state.db")
CURSOR_PREFIX = "capture:cursor:"
FORK_PREFIX = "capture:fork:"  # sessions whose SessionStart source was ``fork``
BATCH_LINES = 1000
BATCH_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 256 * 1024 * 1024
SETTLE_S = 2.0
QUIET_S = 0.5
END_EVENTS = frozenset({"Stop", "StopFailure", "SubagentStop", "SessionEnd", "PreCompact"})
SET_ASIDE_TAGS = (".orphaned-", ".superseded-")


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

    Refuses a Hermes home, an existing unstamped database (never adopted), a copied or moved
    store, and a store whose stamp does not match its owner file. Nothing here opens a store
    for writing before the stamp is checked read-only."""
    home = home.resolve()
    hermes = (hermes_home or Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")).expanduser().resolve()
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
        store.update_metadata_json(CAPTURE_OWNER_KEY, lambda _current: stamp)
        store.ensure_capture_events()
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
    """The ``capture:owner`` stamp, read without a write-capable open: no migration, no sidecars.

    ``immutable`` serves a WAL-mode file whose sidecars are gone, when everything is in the
    main file (the stamp is written and checkpointed before the store is linked into place)."""
    for query in ("mode=ro", "immutable=1"):
        try:
            with closing(sqlite3.connect(f"{database.resolve().as_uri()}?{query}", uri=True)) as conn:
                row = conn.execute("SELECT value FROM metadata WHERE key = ?", (CAPTURE_OWNER_KEY,)).fetchone()
            value = json.loads(row[0]) if row and row[0] else None
            return value if isinstance(value, dict) else None
        except (sqlite3.Error, ValueError):
            continue
    return None


def to_rollback_journal(conn: Optional[sqlite3.Connection], attempts: int = 5) -> bool:
    """Checkpoint and leave WAL mode, so the read-only reader opens the store between worker runs
    (it refuses a WAL store whose sidecars SQLite deleted at the last close). A reader mid-call
    can block the switch; its sidecars then stay and recall keeps working."""
    if conn is None:
        return False
    for _ in range(attempts):
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if str(conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]).lower() == "delete":
                return True
        except sqlite3.OperationalError:
            pass
        time.sleep(0.1)
    return False


def open_store(database: Path, home: Path, config: Any = None) -> MessageStore:
    store = MessageStore(database, ingest_protection_config=config, hermes_home=str(home))
    store.ensure_capture_events()
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
    return {"path": str(path), "session_id": lineage.session_id, "agent_id": lineage.agent_id,
            "dev": status.st_dev, "ino": status.st_ino, "generation": cursor["generation"] + 1 if cursor else 0,
            "offset": 0, "line_index": 0, "first": None, "last_offset": None, "last": None}


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

def drain(store: MessageStore, lineage: Lineage, *, deadline: Optional[float] = None) -> str:
    """Ingest a transcript to its current end: ``drained``, ``missing`` or ``deadline``."""
    cursor = read_cursor(store, lineage)
    state = cursor_state(lineage.path, cursor)
    if state == "missing":
        return "missing"
    if state == "replaced":
        old = _set_aside_copy(lineage.path, cursor)
        if old is not None and cursor_state(old, cursor) == "ok":
            cursor = _drain_file(store, lineage, old, cursor, deadline)
            if cursor is None:
                return "deadline"
    if state != "ok":
        cursor = _new_generation(lineage.path, lineage, cursor)
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
    session that started as a fork, an entry first captured in another session with the same
    content is a fork copy: the fork's own rows, marked ``fork_copy`` and ``fork_copy_of``."""
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
    fork = lineage.agent_id is None and bool(store.read_metadata_json(FORK_PREFIX + lineage.session_id))
    copies = _copies_elsewhere(store, lineage.key, uids - set(known)) if fork else {}
    _link_lineage(store, lineage, cursor, copies)

    messages: list[dict] = []
    pending: list[tuple[str, str, str, list[tuple[str, Any]], dict]] = []
    events: list[dict] = []
    for line in parsed:
        event: dict[str, Any] = {"line": line, "disposition": "raw", "version": None, "files": [], "notes": []}
        events.append(event)
        if line.entry_class != PROJECTED or not line.uid:
            continue
        versions = known.setdefault(line.uid, [])
        same = next((version for version in versions if version["digest"] == line.digest), None)
        if same is not None:
            event.update(disposition="duplicate", version=same)
            continue
        projection = project_entry(line.entry, folder=folder)
        if versions:
            disposition, kind, proof, previous = "version", "version", "transcript_uuid", versions[-1]["row_ids"]
        elif line.uid in copies and copies[line.uid]["digest"] == line.digest:
            disposition, kind, proof, previous = "fork_copy", "canonical", "fork_copy", copies[line.uid]["row_ids"]
        else:
            disposition, kind, proof, previous = "new", "canonical", "transcript_uuid", {}
        # Filled in place inside the transaction, so later lines of this batch see the store ids.
        version = {"digest": line.digest, "store_ids": [], "row_ids": {}}
        versions.append(version)
        event.update(disposition=disposition, version=version, files=projection.files, notes=projection.problems)
        pending.append((disposition, kind, proof, projection.rows, {"event": event, "previous": previous}))
        messages.extend(row for _uid, row in projection.rows)

    with store.ingest_atomic():
        store_ids = iter(store.append_batch(lineage.rows_session, messages, source=SOURCE) if messages else [])
        bindings: list[tuple[int, str, str, str]] = []
        relations: list[list[tuple]] = []
        for disposition, kind, proof, rows, state in pending:
            version = state["event"]["version"]
            for row_uid, _row in rows:
                store_id = next(store_ids)
                version["store_ids"].append(store_id)
                version["row_ids"][row_uid] = store_id
                bindings.append((store_id, row_uid, kind, proof))
                related = state["previous"].get(row_uid)
                if related is not None:
                    kind_of = "supersedes" if disposition == "version" else "fork_copy_of"
                    relations.append([(store_id, kind_of, related, 0, None)])
        store.add_host_uid_bindings(lineage.key, bindings)
        if relations:
            store.add_message_relations(relations)
        _write_events(store, lineage, cursor["generation"], events)
        cursor = _advance(cursor, lines)
        store.connection.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (CURSOR_PREFIX + lineage.key, json.dumps(cursor, sort_keys=True)))
    return cursor


def _advance(cursor: dict, lines: list[tuple[int, bytes]]) -> dict:
    last_offset, last = lines[-1]
    first = cursor["first"] or (_digest(lines[0][1]) if cursor["offset"] == 0 else cursor["first"])
    return {**cursor, "first": first, "offset": last_offset + len(last), "line_index": cursor["line_index"] + len(lines),
            "last_offset": last_offset, "last": _digest(last)}


def _versions(store: MessageStore, key: str, uids: Iterable[str]) -> dict[str, list[dict]]:
    """Per uid, the content versions this lineage already holds, oldest first."""
    found: dict[str, list[dict]] = {}
    uids = sorted(uids)
    for start in range(0, len(uids), 500):
        chunk = uids[start:start + 500]
        for uid, digest, store_ids in store.connection.execute(
                f"SELECT uid, content_digest, store_ids FROM capture_events WHERE uid IN ({','.join('?' * len(chunk))}) "
                "AND lineage_key = ? AND source_kind = 'line' AND disposition IN ('new', 'version', 'fork_copy') "
                "ORDER BY event_id", [*chunk, key]):
            ids = json.loads(store_ids or "[]")
            found.setdefault(uid, []).append({"digest": digest, "store_ids": ids, "row_ids": _row_ids(store, key, ids)})
    return found


def _row_ids(store: MessageStore, key: str, ids: list[int]) -> dict[str, int]:
    return {uid: store_id for store_id, uids in store.host_uid_uids_of_stores(key, ids).items() for uid in uids}


def _copies_elsewhere(store: MessageStore, key: str, uids: Iterable[str]) -> dict[str, dict]:
    """The latest capture of each uid in another lineage: a fork's copied entries, so a fork of a
    fork links to its direct parent."""
    found: dict[str, dict] = {}
    uids = sorted(uids)
    for start in range(0, len(uids), 500):
        chunk = uids[start:start + 500]
        for uid, other, session, digest, store_ids in store.connection.execute(
                f"SELECT uid, lineage_key, session_id, content_digest, store_ids FROM capture_events "
                f"WHERE uid IN ({','.join('?' * len(chunk))}) AND lineage_key != ? AND source_kind = 'line' "
                "AND disposition IN ('new', 'fork_copy') ORDER BY event_id DESC", [*chunk, key]):
            if uid not in found:
                ids = json.loads(store_ids or "[]")
                found[uid] = {"lineage": other, "session": session, "digest": digest,
                              "row_ids": _row_ids(store, other, ids)}
    return found


def _link_lineage(store: MessageStore, lineage: Lineage, cursor: dict, copies: dict[str, dict]) -> None:
    """Parent links go in before the batch's transaction (the lineage write runs its own and is
    idempotent), so a crash after the batch commits cannot lose them."""
    if lineage.agent_id and cursor["offset"] == 0:
        store.record_session_lineage(lineage.rows_session, lineage.session_id, source=SOURCE,
                                     child_subagent_id=lineage.agent_id, child_role="subagent")
    parents = {copy["session"] for copy in copies.values()} - {lineage.rows_session}
    if len(parents) == 1:
        store.record_session_lineage(lineage.rows_session, parents.pop(), source=SOURCE, child_role="fork")


INSERT_EVENT = (
    "INSERT OR IGNORE INTO capture_events(session_id, lineage_key, source_kind, generation, position, uid, "
    "entry_type, entry_class, line_digest, content_digest, disposition, store_ids, notes, raw, raw_from, "
    "projection_version, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")


def _write_events(store: MessageStore, lineage: Lineage, generation: int, events: list[dict]) -> None:
    conn = store.connection
    now = time.time()
    shared: dict[str, int] = {}
    for event in events:
        line: _Line = event["line"]
        digest = _digest(line.raw)
        raw, raw_from = zlib.compress(line.raw), None
        if line.entry_class == SHARED_RAW:
            if digest not in shared:
                held = conn.execute("SELECT event_id FROM capture_events WHERE line_digest = ? AND entry_class = ? "
                                    "AND raw IS NOT NULL LIMIT 1", (digest, SHARED_RAW)).fetchone()
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
            row = _held_file_row(lineage, session_dir(lineage.path) / "tool-results" / name, store_ids, now)
            if row is not None:
                conn.execute(INSERT_EVENT, row)


def _held_file_row(lineage: Lineage, path: Path, store_ids: Optional[list[int]], now: float) -> Optional[tuple]:
    """A ``tool-results`` file whose full text a tool row holds: digest and row ids, no bytes."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return (lineage.rows_session, lineage.key, "file", 0, 0, "tool-results/" + path.name, None, "tool_results_file",
            _digest(data), None, "held", json.dumps(store_ids or []), None, None, None, PROJECTION_VERSION, now)


def capture_files(store: MessageStore, lineage: Lineage) -> int:
    """Every ``tool-results`` file of a session gets a ledger row; bytes are kept unless a row holds them."""
    folder = session_dir(lineage.path) / "tool-results"
    try:
        paths = sorted(path for path in folder.iterdir() if path.is_file() and not path.is_symlink())
    except OSError:
        return 0
    conn = store.connection
    added = 0
    for path in paths:
        uid = "tool-results/" + path.name
        try:
            size = path.stat().st_size
            data = path.read_bytes() if size <= MAX_FILE_BYTES else None
        except OSError:
            continue
        digest = _digest(data) if data is not None else "size:%d" % size
        if conn.execute("SELECT 1 FROM capture_events WHERE source_kind = 'file' AND lineage_key = ? AND uid = ? "
                        "AND (line_digest = ? OR disposition = 'held')", (lineage.key, uid, digest)).fetchone():
            continue
        with store.ingest_atomic():
            conn.execute(
                INSERT_EVENT,
                (lineage.rows_session, lineage.key, "file", 0, 0, uid, None, "tool_results_file", digest, None,
                 "kept" if data is not None else "too_large", None, None,
                 zlib.compress(data) if data is not None else None, None, PROJECTION_VERSION, time.time()))
        added += 1
    return added


# -- worker --

def registered(store: MessageStore) -> list[Lineage]:
    """Every transcript capture has a cursor for: the sweep's list."""
    lineages = []
    for _key, value in store.connection.execute(
            "SELECT key, value FROM metadata WHERE key >= ? AND key < ?", (CURSOR_PREFIX, CURSOR_PREFIX + "￿")):
        try:
            cursor = json.loads(value)
            lineages.append(Lineage(cursor["session_id"], cursor.get("agent_id"), Path(cursor["path"])))
        except (TypeError, ValueError, KeyError):
            continue
    return lineages


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

    ``deadline_s`` (the PreCompact drain) waits for the lock until the deadline; ``complete`` is
    true only when every record was drained."""
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    deadline = None if deadline_s is None else time.monotonic() + deadline_s
    report: dict[str, Any] = {"records": 0, "drained": 0, "missing": 0, "failed": 0, "files": 0, "complete": True}
    lock = _Lock(home)
    try:
        wait_s = deadline_s or 0.0
        while True:
            if not lock.acquire(wait_s):
                if deadline_s is not None:
                    report["complete"] = False  # another worker holds the store past this deadline
                return report
            try:
                status = _drain_once(home, report, deadline, config, settle_s, hermes_home)
            finally:
                lock.release()
            if status != "done" or not pending(home):
                return report
            wait_s = 0.0
    finally:
        lock.close()


def _drain_once(home: Path, report: dict, deadline: Optional[float], config: Any, settle_s: float,
                hermes_home: Optional[Path]) -> str:
    """``done``, ``refused`` or ``deadline``. A record whose drain raises is set aside as
    ``.failed`` (the next hook of that session spools a fresh one), so one bad transcript
    cannot stall the others or loop the worker."""
    try:
        database = admit(home, hermes_home=hermes_home)
    except AdmissionError as refused:
        report.update(refused=refused.code, complete=False)
        return "refused"
    store = open_store(database, home, config)
    try:
        for path, record in claim_records(home):
            report["records"] += 1
            try:
                if record.get("start_source") == "fork" and record.get("session_id"):
                    store.update_metadata_json(FORK_PREFIX + record["session_id"], lambda _current: True)
                for lineage in lineages_of(record):
                    outcome = drain(store, lineage, deadline=deadline)
                    if outcome == "deadline":
                        report["complete"] = False
                        return "deadline"  # the claimed record stays for the next run
                    report[outcome] += 1
                    if record.get("event") in END_EVENTS:
                        _settle(store, lineage, deadline, settle_s)
            except Exception as error:  # noqa: BLE001 - logged without content; capture goes on
                report["failed"] += 1
                report["complete"] = False
                _log_failure(home, "record", error)
                os.replace(path, path.with_suffix(".failed"))
                continue
            path.unlink()
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
                        report["files"] += capture_files(store, each)
                except Exception as error:  # noqa: BLE001
                    report["failed"] += 1
                    report["complete"] = False
                    _log_failure(home, "sweep", error)
        return "done"
    finally:
        to_rollback_journal(store.connection)
        store.close()


def _log_failure(home: Path, stage: str, error: BaseException) -> None:
    """One line per failure, with the error type only: never content or paths."""
    try:
        directory = home / "logs"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(directory / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": round(time.time(), 3), "stage": stage, "error": type(error).__name__}) + "\n")
    except OSError:
        pass
