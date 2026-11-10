"""Read-only Claude Code capture audit, the gate in docs/claude-code-adapter.md.

Each lineage's cursor, ledger, messages and relations are read in one transaction.
Transcript and saved-output reads happen afterwards; the audit never opens a writer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import time
import zlib
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .claude_code_capture import (
    PROJECTED,
    PROJECTION_VERSION,
    classify_entry,
    content_digest,
    lineage_key,
    project_entry,
    results_dir,
    session_dir,
)
from .claude_code_store import (
    CURSOR_PREFIX,
    FORGOTTEN_PREFIX,
    FORK_PREFIX,
    LOCAL_GC_PREFIX,
    PENDING_PREFIX,
    SET_ASIDE_TAGS,
    Lineage,
    _digest,
    _wal_without_sidecars,
    cursor_state,
    named_paths,
    read_stamp,
)
from .config import LCMConfig
from .externalize import DEFAULT_LARGE_OUTPUT_DIRNAME, extract_externalized_refs, load_externalized_payload

FAILURES = ("lost_lines", "digest_mismatch", "surplus_lines", "corrupt", "lost", "duplicates",
            "missing_rows", "orphans", "doubles", "stray_rows", "row_count", "held_mismatch")
PROJECTED_DISPOSITIONS = frozenset({"new", "version", "fork_copy"})
EVENT_FIELDS = ("source_kind", "generation", "position", "uid", "entry_type", "entry_class", "line_digest",
                "content_digest", "disposition", "store_ids", "notes", "captured_at")


class AuditError(ValueError):
    """An invalid audit home, reported without paths or contents."""


def _metadata(conn: sqlite3.Connection) -> dict:
    return {row[0]: json.loads(row[1]) for row in conn.execute(
        "SELECT key, value FROM metadata WHERE key LIKE 'capture:%'")}


def _line_checks(row: sqlite3.Row, checks: dict) -> None:
    data = None
    if row["raw"] is None:
        checks["corrupt"] += not (row["raw_held"] and row["raw_session"] == row["session_id"]
                                   and row["raw_digest"] == row["line_digest"])
    else:
        try:
            data = zlib.decompress(row["raw"])
            checks["corrupt"] += _digest(data) != row["line_digest"]
        except (zlib.error, TypeError):
            checks["corrupt"] += 1
    if row["disposition"] not in PROJECTED_DISPOSITIONS:
        return
    if row["projection_version"] != PROJECTION_VERSION:
        checks["other_projection"] += 1
        return
    try:
        count = len(project_entry(json.loads(data), folder=None).rows)
        checks["row_count"] += len(json.loads(row["store_ids"] or "[]")) != count
    except Exception:  # noqa: BLE001 - a failed projection is a mismatch, with no content reported
        checks["row_count"] += 1


def _kept_corrupt(conn: sqlite3.Connection, key: str) -> int:
    """Verify one file version at a time, retaining at most one decompressed chunk."""
    version, whole, position, valid, count = None, hashlib.sha256(), 0, True, 0
    rows = conn.execute(
        "SELECT uid, content_digest, position, line_digest, raw FROM capture_events "
        "INDEXED BY idx_capture_events_file WHERE lineage_key = ? AND source_kind = 'file' "
        "AND disposition = 'kept' ORDER BY uid, content_digest, position", (key,))
    for row in rows:
        identity = (row["uid"], row["content_digest"])
        if identity != version:
            if version is not None:
                count += not (valid and whole.hexdigest() == version[1])
            version, whole, position, valid = identity, hashlib.sha256(), 0, True
        valid = valid and row["position"] == position
        position += 1
        try:
            data = zlib.decompress(row["raw"])
            valid = valid and _digest(data) == row["line_digest"]
            whole.update(data)
            del data
        except (zlib.error, TypeError):
            valid = False
    if version is not None:
        count += not (valid and whole.hexdigest() == version[1])
    return count


def _ledger(conn: sqlite3.Connection, key: str) -> tuple[list[dict], dict]:
    """Verify raw rows in the snapshot, retaining only compact ledger metadata."""
    fields = ", ".join("e." + name for name in EVENT_FIELDS)
    events = []
    checks = dict.fromkeys(("corrupt", "row_count", "held_mismatch", "other_projection"), 0)
    rows = conn.execute(
        f"SELECT {fields}, e.raw, e.session_id, e.projection_version, r.session_id AS raw_session, "
        "r.line_digest AS raw_digest, r.raw IS NOT NULL AS raw_held FROM capture_events e "
        "LEFT JOIN capture_events r ON r.event_id = e.raw_from "
        "WHERE e.lineage_key = ? AND e.source_kind = 'line'", (key,))
    for row in rows:
        _line_checks(row, checks)
        event = {name: row[name] for name in EVENT_FIELDS}
        event["store_ids"] = tuple(json.loads(event["store_ids"] or "[]"))
        event["notes"] = tuple(json.loads(event["notes"] or "[]"))
        events.append(event)
    events.extend(dict(row) for row in conn.execute(
        "SELECT source_kind, uid, content_digest, disposition, store_ids FROM capture_events "
        "WHERE lineage_key = ? AND source_kind = 'file' GROUP BY uid, content_digest, disposition, store_ids", (key,)))
    for event in events:
        if event["source_kind"] == "file":
            event["store_ids"] = tuple(json.loads(event["store_ids"] or "[]"))
    # A projected entry's rows are its own: an id it lists twice, or that another projected entry
    # also lists, hides a lost row from the count and from missing_rows.
    claimed: set = set()
    for event in events:
        if event["source_kind"] == "line" and event["disposition"] in PROJECTED_DISPOSITIONS:
            for store_id in event["store_ids"]:
                checks["row_count"] += store_id in claimed
                claimed.add(store_id)
    del claimed
    checks["corrupt"] += _kept_corrupt(conn, key)
    return events, checks


def _messages(conn: sqlite3.Connection, session: str) -> list[dict]:
    messages = []
    for store_id, role, tool_call_id, tool_calls in conn.execute(
            "SELECT store_id, role, tool_call_id, tool_calls FROM messages WHERE session_id = ?", (session,)):
        try:
            calls = json.loads(tool_calls or "[]")
        except (ValueError, TypeError):
            calls = []
        ids = [call.get("id") for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []
        del calls, tool_calls
        messages.append({"store_id": store_id, "role": role, "tool_call_id": tool_call_id, "call_ids": ids})
    return messages


def _payload_matches(ref: str, digest: str, home: Path, config: LCMConfig) -> bool:
    try:
        payload = load_externalized_payload(ref, config=config, hermes_home=str(home))
        text = payload.get("content") if payload else None
        return isinstance(text, str) and _digest(text.encode("utf-8", "surrogatepass")) == digest
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _held_matches(conn: sqlite3.Connection, session: str, event: dict, home: Path, config: LCMConfig) -> bool:
    for store_id in event["store_ids"]:
        row = conn.execute("SELECT content FROM messages WHERE session_id = ? AND store_id = ?",
                           (session, store_id)).fetchone()
        text = row[0] if row else None
        if not isinstance(text, str):
            continue
        if _digest(text.encode("utf-8", "surrogatepass")) == event["content_digest"]:
            return True
        for ref in extract_externalized_refs(text):
            if _payload_matches(ref, event["content_digest"], home, config):
                return True
    return False


def _snapshot(conn: sqlite3.Connection, key: str, home: Path | None = None) -> tuple:
    """Read one consistent database view, including shared raw references."""
    conn.execute("BEGIN")
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        cursor = json.loads(conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()[0])
        lineage = Lineage(cursor["session_id"], cursor.get("agent_id"), Path(cursor["path"]))
        forgotten = conn.execute("SELECT 1 FROM metadata WHERE key = ?",
                                 (FORGOTTEN_PREFIX + lineage.rows_session,)).fetchone() is not None
        if forgotten:
            return lineage, cursor, [], [], [], {}, True
        events, checks = _ledger(conn, lineage.key)
        messages = _messages(conn, lineage.rows_session) if "messages" in tables else []
        relations = [dict(row) for row in conn.execute(
            "SELECT store_id, kind, related_store_id FROM message_relations "
            "WHERE store_id IN (SELECT store_id FROM messages WHERE session_id = ?)",
            (lineage.rows_session,))] if {"message_relations", "messages"} <= tables else []
        held = [event for event in events if event["source_kind"] == "file" and event["disposition"] == "held"]
        if held:
            home = home or Path(conn.execute("PRAGMA database_list").fetchone()[2]).parent
            config = LCMConfig.from_env()
            checks["held_mismatch"] = sum(not _held_matches(conn, lineage.rows_session, event, home, config)
                                           for event in held) if "messages" in tables else len(held)
    finally:
        conn.execute("COMMIT")
    return lineage, cursor, events, messages, relations, checks, False


def _set_aside_siblings(path: Path) -> list[Path]:
    return [sibling for sibling in sorted(path.parent.iterdir())
            if sibling != path and any(tag in sibling.name for tag in SET_ASIDE_TAGS)
            and sibling.is_file()] if path.parent.is_dir() else []


def _matches_generation(path: Path, rows: dict) -> bool:
    # A generation with no ledger rows proves nothing: it may have lost them all, so it never
    # matches, and stays gone (unverified), even if it was empty.
    if not rows or 0 not in rows:
        return False
    count = 0
    with path.open("rb") as handle:
        for raw in handle:
            row = rows.get(count)
            if not raw.endswith(b"\n") or row is None or _digest(raw) != row["line_digest"]:
                return False
            count += 1
    return count == len(rows)


def _named_others(lineage: Lineage, cursor: dict, report: dict) -> None:
    """#2535: another transcript path this lineage's records named. Capture follows one path, and the audit does not
    match another one against the store: one that holds lines, or saved outputs under its session folder, leaves the
    lineage unverified. One that never appeared (a resume's SessionStart names a path under the new directory), or an
    empty one with no saved outputs, holds nothing. One that cannot be checked is not clean."""
    try:
        own = lineage.path.stat()
        own = (own.st_dev, own.st_ino)
    except OSError:
        own = None
    own_folder = results_dir(session_dir(lineage.path))
    for name in named_paths(cursor):
        other = Path(name)
        if other == lineage.path:
            continue
        try:
            try:
                status = other.stat()
            except (FileNotFoundError, NotADirectoryError):  # never appeared, or cannot exist
                status = None
            holds = (status is not None and (status.st_dev, status.st_ino) != own
                     and (not stat.S_ISREG(status.st_mode) or status.st_size > 0))
            results = session_dir(other) / "tool-results"
            try:
                results.lstat()
            except (FileNotFoundError, NotADirectoryError):  # no saved outputs there
                results = None
            if not holds and results is not None:
                folder = results_dir(session_dir(other))  # None when a symlink or unreadable: not a clean folder
                holds = folder is None or (folder != own_folder and any(True for _ in folder.iterdir()))
            agents = session_dir(other) / "subagents"  # subagent transcripts the worker sweeps only under the cursor's
            if not holds and session_dir(other) != session_dir(lineage.path):
                try:
                    agents.lstat()
                except (FileNotFoundError, NotADirectoryError):
                    agents = None
                holds = agents is not None and any(True for _ in agents.iterdir())
        except OSError:
            holds = True
        report["named_other_paths"] += holds


def _check_folder(folder: Path, session_id: str, key: str, metadata: dict, unknown: set, outputs: bool) -> None:
    """A session folder: a subagents folder that cannot be listed, subagent transcripts nothing captured, and
    (``outputs``: no cursor of the session follows its transcript) saved outputs, which nothing captured either."""
    if _unlistable_subagents(folder):
        unknown.add(key)
    if outputs:
        try:  # a folder that cannot be checked is not clean
            (folder / "tool-results").lstat()
            results = results_dir(folder)
            if results is None or any(True for _ in results.iterdir()):
                unknown.add(key)
        except (FileNotFoundError, NotADirectoryError):
            pass
        except OSError:
            unknown.add(key)
    try:  # this folder's own subagents (session_dir() would map an inner <id>/ under subagents/ to the outer folder)
        names = sorted(path.name for path in (folder / "subagents").glob("agent-*.jsonl"))
    except OSError:
        names = []
    for name in names:
        child = Lineage(session_id, name[len("agent-"):-len(".jsonl")], folder / "subagents" / name)
        if _uncaptured_child(child, metadata):
            unknown.add(child.key)


def _uncaptured_child(child: Lineage, metadata: dict) -> bool:
    """A subagent transcript with no cursor, or a copy at a path its cursor never named that holds lines (#2535: no
    matching). A forgotten one is never read again."""
    if FORGOTTEN_PREFIX + child.rows_session in metadata:
        return False
    if CURSOR_PREFIX + child.key not in metadata:
        return True
    return str(child.path) not in named_paths(metadata[CURSOR_PREFIX + child.key]) and _holds_lines(child.path)


def _unlistable_subagents(session: Path) -> bool:
    """The session folder's subagents folder exists but cannot be listed: glob would just return nothing."""
    folder = session / "subagents"
    try:
        folder.lstat()
        next(iter(folder.iterdir()), None)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return False


def _holds_lines(path: Path) -> bool:
    """Not empty (a regular file with bytes, or anything else that exists), or cannot be checked (fail closed)."""
    try:
        status = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return not stat.S_ISREG(status.st_mode) or status.st_size > 0


def _present(path: Path) -> bool:
    """Exists, or cannot be checked (fail closed): only a clean not-found is absent."""
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return True


def _older_copy(siblings: list[Path], rows: dict) -> Path | None:
    for sibling in siblings:
        if _matches_generation(sibling, rows):
            return sibling
    return None


def _compare(path: Path, rows: dict, report: dict, expected: set, lag: list[float],
             limit: int | None = None) -> int:
    """Compare and project one complete line at a time, never retaining its raw bytes."""
    position = 0
    with path.open("rb") as handle:
        while limit is None or handle.tell() < limit:
            raw = handle.readline() if limit is None else handle.readline(limit - handle.tell())
            if not raw.endswith(b"\n"):
                break
            row = rows.get(position)
            if row is None:
                report["failures"]["lost_lines"] += 1
            elif row["line_digest"] != _digest(raw):
                report["failures"]["digest_mismatch"] += 1
            _expected(raw, row, expected, lag)
            position += 1
    report["lines"] += position
    return position


def _transcripts(lineage: Lineage, cursor: dict, events: list[dict], report: dict, lag: list[float]) -> set:
    """Stream readable generations and collect their projected identities and lag."""
    generations = defaultdict(dict)
    for row in events:
        if row["source_kind"] == "line":
            generations[row["generation"]][row["position"]] = row
    expected = set()
    state = cursor_state(lineage.path, cursor)
    report["transcript_state"] = state
    if state == "ok":
        rows = generations[cursor["generation"]]
        _compare(lineage.path, rows, report, expected, lag, cursor["offset"])
        with lineage.path.open("rb") as handle:
            handle.seek(cursor["offset"])
            for line in handle:
                if line.endswith(b"\n"):
                    report["pending_lines"] += 1
                else:  # an unterminated last record may still be mid-write: reported, not judged (#2557)
                    report["partial_tail_bytes"] = len(line)
        report["failures"]["surplus_lines"] += sum(i >= cursor["line_index"] for i in rows)
    siblings = _set_aside_siblings(lineage.path) if cursor["generation"] else []
    for generation in range(cursor["generation"]):
        rows = generations[generation]
        copy = _older_copy(siblings, rows)
        if copy is None:
            report["generations_gone"] += 1
            continue
        length = _compare(copy, rows, report, expected, lag)
        report["failures"]["surplus_lines"] += sum(i >= length for i in rows)
    _named_others(lineage, cursor, report)
    return expected


def _expected(raw: bytes, row: dict | None, expected: set, lag: list[float]) -> None:
    try:
        entry = json.loads(raw)
    except ValueError:
        return
    if not isinstance(entry, dict) or classify_entry(entry) != PROJECTED:
        return
    uid = entry.get("uuid")
    if isinstance(uid, str) and uid:
        expected.add((uid, content_digest(entry)))
    if row is not None and isinstance(entry.get("timestamp"), str):
        try:
            stamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")).timestamp()
            lag.append(row["captured_at"] - stamp)
        except (ValueError, OverflowError, OSError):
            pass


def _identities(events: list[dict], messages: list[dict], expected: set, report: dict) -> None:
    lines = [row for row in events if row["source_kind"] == "line"]
    projected = [row for row in lines if row["disposition"] in PROJECTED_DISPOSITIONS]
    present = {(row["uid"], row["content_digest"]) for row in projected}
    errors = defaultdict(list)
    for row in lines:
        errors[row["uid"], row["content_digest"]].append(row)
    for pair in expected - present:
        rows = errors[pair]
        if rows and all(row["disposition"] == "raw" and any(
                note.startswith("projection_error:") for note in row["notes"]) for row in rows):
            report["projection_errors"] += 1
        else:
            report["failures"]["lost"] += 1
    identities = Counter((row["uid"], row["content_digest"]) for row in projected)
    report["failures"]["duplicates"] = sum(n > 1 for n in identities.values())
    column = "surplus" if report["transcript_state"] == "ok" and not report["generations_gone"] else "unverifiable"
    report[column] = len(identities.keys() - expected)
    report["fork_copies"] = sum(row["disposition"] == "fork_copy" for row in projected)
    ids = {row["store_id"] for row in messages}
    referenced = [*projected, *(row for row in events if row["source_kind"] == "file"
                              and row["disposition"] == "held")]
    report["failures"]["missing_rows"] = sum(
        store_id not in ids for row in referenced for store_id in row["store_ids"])
    referenced_ids = {store_id for row in referenced for store_id in row["store_ids"]}
    report["failures"]["stray_rows"] = len(ids - referenced_ids)


def _tools(messages: list[dict], relations: list[dict], report: dict) -> None:
    superseded = {row["related_store_id"] for row in relations if row["kind"] == "supersedes"}
    calls = defaultdict(set)
    results = Counter()
    for row in messages:
        if row["store_id"] in superseded:
            continue
        if row["role"] == "assistant":
            for call_id in row["call_ids"]:
                calls[call_id].add(row["store_id"])
        elif row["role"] == "tool":
            results[row["tool_call_id"]] += 1
    report["failures"]["orphans"] = sum(n for uid, n in results.items() if not uid or uid not in calls)
    report["failures"]["doubles"] = sum(len(rows) > 1 or results[uid] > 1 for uid, rows in calls.items())
    report["unanswered"] = sum(not results[uid] for uid in calls)


def _files(lineage: Lineage, events: list[dict], report: dict) -> None:
    if lineage.agent_id is not None:
        return
    folder = results_dir(session_dir(lineage.path))
    if folder is None:  # none, or one capture will not read (a symlink, or not a real folder): that holds rows
        try:
            (session_dir(lineage.path) / "tool-results").lstat()
            report["files"]["uncaptured"] += 1
        except (FileNotFoundError, NotADirectoryError):
            pass
        except OSError:
            report["files"]["uncaptured"] += 1
        return
    versions = {(row["uid"], row["content_digest"]): row["disposition"]
                for row in events if row["source_kind"] == "file"}
    for path in sorted(folder.iterdir()):
        if path.is_dir() and not path.is_symlink():  # a tool's own files (page renders): capture does not store them
            report["subfolder_files"] += _files_under(path)
            continue
        if path.is_symlink() or not (path.is_file() or path.is_dir()):  # capture never reads it
            report["files"]["uncaptured"] += 1
        if path.is_symlink() or not path.is_file():
            continue
        disposition = versions.get(("tool-results/" + path.name, _file_digest(path)))
        if disposition in ("held", "kept"):
            report["files"][disposition] += 1
        else:
            report["files"]["uncaptured"] += 1


def _files_under(folder: Path) -> int:
    """Every entry under ``folder`` that is not a real folder (a symlink to a folder counts; it is not followed); a
    folder that cannot be listed counts one."""
    count, stack = 0, [folder]
    while stack:  # no recursion: os.walk recurses on Python 3.11, and a deep tree must be counted, never crash
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    else:
                        count += 1
        except OSError:
            count += 1
    return count


def _subfolder_files_unverified(row: dict) -> bool:
    """#2535: files a tool wrote into a tool-results subfolder are not stored and match nothing the model was shown
    inline in 30 of 42 real cases. Release-manager ruling: a lineage with any is unverified until #2554 lands."""
    return row["subfolder_files"] > 0


def _file_digest(path: Path) -> str:
    whole = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            whole.update(chunk)
    return whole.hexdigest()


def _lineage_report(lineage: Lineage, cursor: dict, events: list[dict], messages: list[dict],
                    relations: list[dict], checks: dict, forgotten: bool, lag: list[float]) -> dict:
    report = {"lineage": lineage.key, "session": lineage.rows_session, "generation": cursor["generation"],
              "lines": 0, "pending_lines": 0, "generations_gone": 0, "projection_errors": 0,
              "surplus": 0, "unverifiable": 0, "fork_copies": 0, "unanswered": 0,
              "forgotten": forgotten, "transcript_state": "forgotten" if forgotten else "ok",
              "other_projection": 0, "projected_without_uuid": 0,
              "named_other_paths": 0, "subfolder_files": 0, "partial_tail_bytes": 0,
              "files": dict.fromkeys(("held", "kept", "uncaptured"), 0), "failures": dict.fromkeys(FAILURES, 0)}
    report.update(classes={}, unknown_types={})
    if forgotten:
        return report
    lines = [row for row in events if row["source_kind"] == "line"]
    report["classes"] = dict(Counter(row["entry_class"] for row in lines))
    report["unknown_types"] = dict(Counter(row["entry_type"] or "null" for row in lines
                                           if row["entry_class"] == "unknown_type"))
    report["projected_without_uuid"] = sum("projected_entry_without_uuid" in row["notes"] for row in lines)
    expected = _transcripts(lineage, cursor, events, report, lag)
    report["other_projection"] = checks["other_projection"]
    report["failures"].update({key: value for key, value in checks.items() if key in FAILURES})
    _identities(events, messages, expected, report)
    _tools(messages, relations, report)
    _files(lineage, events, report)
    return report


def _compact_record(snapshot: tuple) -> dict:
    lineage, cursor, events, _messages, relations, _checks, forgotten = snapshot
    origins = {row["store_id"] for row in relations if row["kind"] == "fork_copy_of"}
    unlinked = sum(row["disposition"] == "fork_copy" and not origins.intersection(row["store_ids"])
                   for row in events)
    return {"lineage": lineage, "forgotten": forgotten,
            "parent_unknown": bool(cursor.get("fork_linked") and not cursor.get("fork_parent")),
            "unlinked": unlinked}


def _audit_lineage(conn: sqlite3.Connection, key: str, home: Path, lag: list[float]) -> tuple[dict, dict]:
    snapshot = _snapshot(conn, key, home)
    report = _lineage_report(*snapshot, lag)
    return report, _compact_record(snapshot)


def _session_report(metadata: dict, records: list[dict], session: str | None) -> dict:
    def selected(value):
        return session is None or value == session or value.startswith(session + "/agent:")

    local_gc = sorted(key[len(LOCAL_GC_PREFIX):] for key in metadata
                      if key.startswith(LOCAL_GC_PREFIX) and selected(key[len(LOCAL_GC_PREFIX):]))
    forks = {key[len(FORK_PREFIX):] for key in metadata
             if key.startswith(FORK_PREFIX) and selected(key[len(FORK_PREFIX):])}
    unknown = set()
    unlinked = 0
    for record in records:
        lineage = record["lineage"]
        if record["forgotten"] or lineage.session_id not in forks:
            continue
        if record["parent_unknown"]:
            unknown.add(lineage.rows_session)
        unlinked += record["unlinked"]
    forgotten = sum(key.startswith(FORGOTTEN_PREFIX) and selected(key[len(FORGOTTEN_PREFIX):]) for key in metadata)
    return {"local_gc": local_gc, "forgotten": forgotten, "forks": len(forks),
            "fork_parent_unknown": sorted(unknown), "fork_copy_without_origin": unlinked}


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    values = sorted(values)
    return {"p50": values[math.ceil(len(values) * 0.50) - 1],
            "p95": values[math.ceil(len(values) * 0.95) - 1], "max": values[-1]}


def _log_records(path: Path):
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    yield record


def _home_report(home: Path) -> dict:
    """Hook timing per event. Each session's first hook (a cold interpreter start) is reported
    apart, as ``first``, and left out of its event's percentiles; ``precompact`` counts which
    path each PreCompact run took (a drain, a poke of the worker holding the lock, or neither:
    no runtime, package or worker, a record the hook would not spool, or a hook error) and how
    many drains reached their bound."""
    blocking = defaultdict(list)
    hooks, first, seen = defaultdict(list), [], set()
    precompact = {"drain": 0, "poke": 0, "bound": 0, "other": 0}
    for name in ("hook.log.1", "hook.log"):
        for row in _log_records(home / "logs" / name):
            ms, session, outcome = row.get("ms"), row.get("session"), row.get("outcome")
            if not (isinstance(row.get("event"), str) and type(ms) in (int, float) and math.isfinite(ms)):
                continue
            if row["event"] == "PreCompact" and isinstance(outcome, str):
                path = "poke" if outcome == "worker_running" else "drain" if outcome.startswith("drain") else "other"
                precompact[path] += 1
                # The worker says when its deadline ended the run (drain_bound); the hook's kill at
                # its own timeout (drain_timeout) also means the drain ran out its time.
                precompact["bound"] += outcome.startswith(("drain_bound", "drain_timeout"))
            value = row.get("blocking_ms")
            if type(value) in (int, float) and math.isfinite(value):
                # Per event only: a session's first hook is the synchronous SessionStart, which never forks.
                blocking[row["event"]].append(value)
            if isinstance(session, str) and session not in seen:
                seen.add(session)
                first.append(ms)
                continue
            hooks[row["event"]].append(ms)
    latency = {event: {"n": len(values), **_percentiles(values)} for event, values in hooks.items()}
    latency["first"] = {"n": len(first), "max": max(first, default=None)}
    worker = Counter(f"{row['stage']}:{row['error']}" for row in _log_records(home / "logs" / "worker.log")
                     if isinstance(row.get("stage"), str) and isinstance(row.get("error"), str))
    spool = list((home / "spool").iterdir()) if (home / "spool").is_dir() else []
    return {"spool": {"failed": sum(path.is_file() and path.suffix == ".failed" for path in spool),
                      "waiting": sum(path.is_file() and path.suffix in (".json", ".claimed") for path in spool)},
            "hooks_in_process_ms": latency, "precompact": precompact,
            "hooks_blocking_ms": {event: {"n": len(values), **_percentiles(values)} for event, values in blocking.items()},
            "worker_log": dict(worker)}


def _growth(home: Path, records: list[dict]) -> dict:
    files = [home / "lcm.db", home / "lcm.db-wal"]
    payloads = home / DEFAULT_LARGE_OUTPUT_DIRNAME  # where the worker's ingest externalizes, in its home
    if payloads.is_dir():
        files.extend(path for path in payloads.rglob("*") if path.is_file())
    store_bytes = sum(path.stat().st_size for path in files if path.is_file())
    transcript_bytes = sum(record["lineage"].path.stat().st_size for record in records
                           if not record["forgotten"] and record["lineage"].path.is_file())
    return {"store_bytes": store_bytes, "transcript_bytes": transcript_bytes,
            "ratio": store_bytes / transcript_bytes if transcript_bytes else None}


def _failed_records(home: Path) -> list[Lineage | None]:
    """The lineages each set-aside (``.failed``) spool record names: its session's, and its
    subagent's if any. None for a record that cannot be read."""
    directory = home / "spool"
    found: list[Lineage | None] = []
    for path in sorted(directory.glob("*.failed")) if directory.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            session, agent = record["session_id"], record.get("agent_id")
            if not isinstance(session, str) or not isinstance(agent, (str, type(None))):
                raise TypeError("lineage ids")
            lineage_key(session, agent)  # the check the writer's lineage ids passed
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            found.append(None)
            continue
        found.append(Lineage(session, None, path))
        # A subagent counts even when its transcript is not on disk: one never written (the
        # compaction summarizer's) looks the same as one removed before it was captured.
        if agent:
            found.append(Lineage(session, agent, path))
    return found


def _unverified(metadata: dict, records: list[dict], lineages: list[dict], session: str | None,
                failed: list[Lineage | None]) -> list[str]:
    unknown = {row["lineage"] for row in lineages if not row["forgotten"] and (
        row["transcript_state"] != "ok" or row["generations_gone"] or row["pending_lines"]
        or row["files"]["uncaptured"] or row["named_other_paths"] or _subfolder_files_unverified(row))}
    if session is not None and CURSOR_PREFIX + lineage_key(session) not in metadata:
        unknown.add(lineage_key(session))
    # A pending lineage (#2533) whose transcript now exists but has no cursor is uncaptured. One whose transcript never
    # appeared holds nothing to lose: it is counted in pending_lineages, not left unverified forever.
    # A pending row without a usable path cannot be checked, so it is unverified too. A pending main session's
    # subagents are checked like a captured one's: the worker sweeps them.
    for key, pending in metadata.items():
        if not key.startswith(PENDING_PREFIX) or (session is not None and pending.get("session_id") != session):
            continue
        if not isinstance(pending, dict) or not isinstance(pending.get("path"), str) or not pending["path"]:
            unknown.add(key[len(PENDING_PREFIX):])
            continue
        lineage = Lineage(pending.get("session_id"), pending.get("agent_id"), Path(pending["path"]))
        if CURSOR_PREFIX + lineage.key in metadata or FORGOTTEN_PREFIX + lineage.rows_session in metadata:
            if lineage.agent_id is None and CURSOR_PREFIX + lineage.key not in metadata:  # #2557: forgotten, but its
                for name in named_paths(pending):  # subagents are still captured: check them under every named path
                    _check_folder(session_dir(Path(name)), lineage.session_id, lineage.key, metadata, unknown, outputs=False)
            continue
        if any(_present(Path(name)) for name in named_paths(pending)):  # every path its records named (#2535)
            unknown.add(lineage.key)
        if lineage.agent_id is None:  # under every path its records named (#2535)
            for name in named_paths(pending):
                _check_folder(session_dir(Path(name)), lineage.session_id, lineage.key, metadata, unknown, outputs=True)
    # A record set aside before its transcript had a cursor names a lineage nothing else shows.
    for lineage in failed:
        if lineage is None:
            unknown.add("spool:unreadable_failed_record")
        elif ((session is None or lineage.session_id == session) and CURSOR_PREFIX + lineage.key not in metadata
              and FORGOTTEN_PREFIX + lineage.rows_session not in metadata):
            unknown.add(lineage.key)
    for record in records:  # a forgotten session's subagents are still captured (#2557: under every named path)
        lineage = record["lineage"]
        if lineage.agent_id is None:
            for name in named_paths(metadata.get(CURSOR_PREFIX + lineage.key), lineage.path):
                _check_folder(session_dir(Path(name)), lineage.session_id, lineage.key, metadata, unknown, outputs=False)
            continue
        parent = lineage_key(lineage.session_id)  # #2557: a child in a session folder its parent never named
        named = named_paths(metadata.get(CURSOR_PREFIX + parent)) or named_paths(metadata.get(PENDING_PREFIX + parent))
        folders = {up.parent for up in lineage.path.parents  # every <session>/subagents above the child
                   if up.name == "subagents" and up.parent.name == lineage.session_id}
        for folder in sorted(folders - {session_dir(Path(name)) for name in named}):
            transcript = folder.parent / (folder.name + ".jsonl")  # not with_suffix: a session id may hold a dot
            gone = FORGOTTEN_PREFIX + lineage.session_id in metadata  # a forgotten parent's own content is never read
            if not gone and _holds_lines(transcript):
                unknown.add(parent)
            _check_folder(folder, lineage.session_id, parent, metadata, unknown, outputs=not gone)
    return sorted(unknown)


def audit(home: Path, session: str | None = None) -> dict:
    """Audit a stamped capture store without changing its home or taking its lock."""
    if not home.is_absolute():
        raise AuditError("home_not_absolute")
    database = home / "lcm.db"
    if not database.is_file():
        raise AuditError("store_missing")
    if not read_stamp(database):
        raise AuditError("store_unstamped")
    immutable = _wal_without_sidecars(database)
    query = "immutable=1" if immutable else "mode=ro"
    lag, lineages, records = [], [], []
    with closing(sqlite3.connect(f"{database.as_uri()}?{query}", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        metadata = _metadata(conn)
        keys = sorted(key for key, cursor in metadata.items() if key.startswith(CURSOR_PREFIX)
                      and (session is None or cursor["session_id"] == session))
        for key in keys:
            report, record = _audit_lineage(conn, key, home, lag)
            lineages.append(report)
            records.append(record)
    failures = {key: sum(row["failures"][key] for row in lineages) for key in FAILURES}
    unverified = _unverified(metadata, records, lineages, session, _failed_records(home))
    home_report = _home_report(home)
    checked = any(not row["forgotten"] for row in lineages)
    # A pending row next to a cursor (a store downgraded and upgraded) or a forgotten session's is not pending.
    pending = [value for key, value in metadata.items() if key.startswith(PENDING_PREFIX) and isinstance(value, dict)
               and (session is None or value.get("session_id") == session)
               and CURSOR_PREFIX + key[len(PENDING_PREFIX):] not in metadata
               and FORGOTTEN_PREFIX + Lineage(value.get("session_id"), value.get("agent_id"), Path(".")).rows_session not in metadata]
    seen = [row["first_seen"] for row in pending if isinstance(row.get("first_seen"), (int, float))]
    oldest_pending_age = max(0.0, time.time() - min(seen)) if seen else None
    verified = checked and not unverified and not home_report["spool"]["waiting"]
    # The one-line verdict says in words when sessions are pending (#2533): they are counted, not checked, so a
    # verified verdict never reads as "everything accounted for".
    verdict = "verified" if verified else "not verified"
    if pending:
        # Lineages, not sessions (a subagent is its own lineage); an undated row can be older than the oldest dated one.
        absent = sum(1 for row in pending if not (isinstance(row.get("path"), str) and row["path"] and Path(row["path"]).exists()))
        age = "unknown" if oldest_pending_age is None else f"{oldest_pending_age / 3600:.1f} h"
        verdict += (f"; {len(pending)} lineages awaiting capture ({absent} transcripts not yet seen), "
                    f"oldest{' known' if seen and len(seen) < len(pending) else ''} {age}")
    others = sum(1 for row in lineages if not row["forgotten"] and row["named_other_paths"])
    if others:  # #2535: capture follows one path, and another that holds rows is not matched against the store
        verdict += f"; {others} lineages: a second transcript path holds rows; not matched"
    tails = [row["partial_tail_bytes"] for row in lineages if not row["forgotten"] and row["partial_tail_bytes"]]
    if tails:  # #2557: counted, never judged (a live session is often mid-write)
        verdict += f"; {len(tails)} lineages end in an unterminated record ({sum(tails)} bytes, not yet captured)"
    subfolder = [row["subfolder_files"] for row in lineages if not row["forgotten"] and row["subfolder_files"]]
    if subfolder:  # #2535: e.g. page renders a tool saved beside the transcript
        verdict += f"; {sum(subfolder)} files in tool-results subfolders not stored ({len(subfolder)} lineages)"
    return {"verdict": verdict, "ok": not any(failures.values()), "opened_immutable": immutable, "failures": failures,
            "pending_lineages": len(pending), "oldest_pending_age_s": oldest_pending_age,
            "unverified": unverified, "verified": verified,
            "lineages": lineages, "sessions": _session_report(metadata, records, session),
            **home_report, "lag_s": _percentiles(lag), "growth": _growth(home, records)}
