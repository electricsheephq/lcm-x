"""Read-only Claude Code capture audit, the gate in docs/claude-code-adapter.md.

Each lineage's cursor, ledger, messages and relations are read in one transaction.
Transcript and saved-output reads happen afterwards; the audit never opens a writer.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zlib
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .claude_code_capture import PROJECTED, classify_entry, content_digest, results_dir, session_dir
from .claude_code_store import (
    CURSOR_PREFIX,
    FORGOTTEN_PREFIX,
    FORK_PREFIX,
    LOCAL_GC_PREFIX,
    SET_ASIDE_TAGS,
    Lineage,
    _digest,
    _wal_without_sidecars,
    cursor_state,
    read_stamp,
)

FAILURES = ("lost_lines", "digest_mismatch", "surplus_lines", "corrupt", "lost", "duplicates",
            "missing_rows", "orphans", "doubles", "files_lost")
PROJECTED_DISPOSITIONS = frozenset({"new", "version", "fork_copy"})
EVENT_FIELDS = ("source_kind", "generation", "position", "uid", "entry_type", "entry_class", "line_digest",
                "content_digest", "disposition", "store_ids", "notes", "captured_at")


class AuditError(ValueError):
    """An invalid audit home, reported without paths or contents."""


def _metadata(conn: sqlite3.Connection) -> dict:
    return {row[0]: json.loads(row[1]) for row in conn.execute(
        "SELECT key, value FROM metadata WHERE key LIKE 'capture:%'")}


def _line_corrupt(row: sqlite3.Row) -> bool:
    if row["raw"] is None:
        return not (row["raw_held"] and row["raw_session"] == row["session_id"]
                    and row["raw_digest"] == row["line_digest"])
    try:
        return _digest(zlib.decompress(row["raw"])) != row["line_digest"]
    except (zlib.error, TypeError):
        return True


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


def _ledger(conn: sqlite3.Connection, key: str) -> tuple[list[dict], int]:
    """Verify raw rows in the snapshot, retaining only compact ledger metadata."""
    fields = ", ".join("e." + name for name in EVENT_FIELDS)
    events, corrupt = [], 0
    rows = conn.execute(
        f"SELECT {fields}, e.raw, e.session_id, r.session_id AS raw_session, "
        "r.line_digest AS raw_digest, r.raw IS NOT NULL AS raw_held FROM capture_events e "
        "LEFT JOIN capture_events r ON r.event_id = e.raw_from "
        "WHERE e.lineage_key = ? AND e.source_kind = 'line'", (key,))
    for row in rows:
        corrupt += _line_corrupt(row)
        events.append({name: row[name] for name in EVENT_FIELDS})
    events.extend(dict(row) for row in conn.execute(
        f"SELECT {', '.join(EVENT_FIELDS)} FROM capture_events "
        "WHERE lineage_key = ? AND source_kind = 'file'", (key,)))
    return events, corrupt + _kept_corrupt(conn, key)


def _snapshot(conn: sqlite3.Connection, key: str) -> tuple:
    """Read one consistent database view, including shared raw references."""
    conn.execute("BEGIN")
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        cursor = json.loads(conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()[0])
        lineage = Lineage(cursor["session_id"], cursor.get("agent_id"), Path(cursor["path"]))
        events, corrupt = _ledger(conn, lineage.key)
        messages = [dict(row) for row in conn.execute(
            "SELECT store_id, role, tool_call_id, tool_calls FROM messages WHERE session_id = ?",
            (lineage.rows_session,))] if "messages" in tables else []
        relations = [dict(row) for row in conn.execute(
            "SELECT store_id, kind, related_store_id FROM message_relations "
            "WHERE store_id IN (SELECT store_id FROM messages WHERE session_id = ?)",
            (lineage.rows_session,))] if {"message_relations", "messages"} <= tables else []
    finally:
        conn.execute("COMMIT")
    return lineage, cursor, events, messages, relations, corrupt


def _older_copy(path: Path, first: dict | None) -> Path | None:
    if first is None:
        return None
    for sibling in sorted(path.parent.iterdir()) if path.parent.is_dir() else []:
        if sibling == path or not any(tag in sibling.name for tag in SET_ASIDE_TAGS) or not sibling.is_file():
            continue
        with sibling.open("rb") as handle:
            if _digest(handle.readline()) == first["line_digest"]:
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
            report["pending_lines"] = sum(line.endswith(b"\n") for line in handle)
        report["failures"]["surplus_lines"] += sum(i >= cursor["line_index"] for i in rows)
    for generation in range(cursor["generation"]):
        rows = generations[generation]
        copy = _older_copy(lineage.path, rows.get(0))
        if copy is None:
            report["generations_gone"] += 1
            continue
        length = _compare(copy, rows, report, expected, lag)
        report["failures"]["surplus_lines"] += sum(i >= length for i in rows)
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
                note.startswith("projection_error:") for note in json.loads(row["notes"] or "[]")) for row in rows):
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
        store_id not in ids for row in referenced for store_id in json.loads(row["store_ids"] or "[]"))


def _tools(messages: list[dict], relations: list[dict], report: dict) -> None:
    superseded = {row["related_store_id"] for row in relations if row["kind"] == "supersedes"}
    calls = defaultdict(set)
    results = Counter()
    for row in messages:
        if row["store_id"] in superseded:
            continue
        if row["role"] == "assistant":
            for call in json.loads(row["tool_calls"] or "[]"):
                calls[call.get("id")].add(row["store_id"])
        elif row["role"] == "tool":
            results[row["tool_call_id"]] += 1
    report["failures"]["orphans"] = sum(n for uid, n in results.items() if not uid or uid not in calls)
    report["failures"]["doubles"] = sum(len(rows) > 1 or results[uid] > 1 for uid, rows in calls.items())
    report["unanswered"] = sum(not results[uid] for uid in calls)


def _files(lineage: Lineage, events: list[dict], forgotten: bool, report: dict) -> None:
    if lineage.agent_id is not None or forgotten:
        return
    folder = results_dir(session_dir(lineage.path))
    if folder is None:
        return
    newest = max((row["captured_at"] for row in events), default=0)
    versions = {(row["uid"], row["content_digest"]): row["disposition"]
                for row in events if row["source_kind"] == "file"}
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        disposition = versions.get(("tool-results/" + path.name, _file_digest(path)))
        if disposition in ("held", "kept"):
            report["files"][disposition] += 1
        elif path.stat().st_mtime > newest:
            report["files"]["pending"] += 1
        else:
            report["failures"]["files_lost"] += 1


def _file_digest(path: Path) -> str:
    whole = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            whole.update(chunk)
    return whole.hexdigest()


def _lineage_report(lineage: Lineage, cursor: dict, events: list[dict], messages: list[dict],
                    relations: list[dict], corrupt: int, metadata: dict, lag: list[float]) -> dict:
    report = {"lineage": lineage.key, "session": lineage.rows_session, "generation": cursor["generation"],
              "lines": 0, "pending_lines": 0, "generations_gone": 0, "projection_errors": 0,
              "surplus": 0, "unverifiable": 0, "fork_copies": 0, "unanswered": 0,
              "files": dict.fromkeys(("held", "kept", "pending"), 0), "failures": dict.fromkeys(FAILURES, 0)}
    lines = [row for row in events if row["source_kind"] == "line"]
    report["classes"] = dict(Counter(row["entry_class"] for row in lines))
    report["unknown_types"] = dict(Counter(row["entry_type"] or "null" for row in lines
                                           if row["entry_class"] == "unknown_type"))
    expected = _transcripts(lineage, cursor, events, report, lag)
    report["failures"]["corrupt"] = corrupt
    _identities(events, messages, expected, report)
    _tools(messages, relations, report)
    _files(lineage, events, bool(metadata.get(FORGOTTEN_PREFIX + lineage.rows_session)), report)
    return report


def _session_report(metadata: dict, snapshots: list[tuple], session: str | None) -> dict:
    def selected(value):
        return session is None or value == session or value.startswith(session + "/agent:")

    local_gc = sorted(key[len(LOCAL_GC_PREFIX):] for key in metadata
                      if key.startswith(LOCAL_GC_PREFIX) and selected(key[len(LOCAL_GC_PREFIX):]))
    forks = {key[len(FORK_PREFIX):] for key in metadata
             if key.startswith(FORK_PREFIX) and selected(key[len(FORK_PREFIX):])}
    unknown = set()
    unlinked = 0
    for lineage, cursor, events, _messages, relations, _corrupt_count in snapshots:
        if lineage.session_id not in forks:
            continue
        if cursor.get("fork_linked") and not cursor.get("fork_parent"):
            unknown.add(lineage.rows_session)
        origins = {row["store_id"] for row in relations if row["kind"] == "fork_copy_of"}
        unlinked += sum(row["disposition"] == "fork_copy" and not origins.intersection(
            json.loads(row["store_ids"] or "[]")) for row in events)
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
    hooks = defaultdict(list)
    for name in ("hook.log.1", "hook.log"):
        for row in _log_records(home / "logs" / name):
            ms = row.get("ms")
            if isinstance(row.get("event"), str) and type(ms) in (int, float) and math.isfinite(ms):
                hooks[row["event"]].append(ms)
    latency = {event: {"n": len(values), **_percentiles(values)} for event, values in hooks.items()}
    worker = Counter(f"{row['stage']}:{row['error']}" for row in _log_records(home / "logs" / "worker.log")
                     if isinstance(row.get("stage"), str) and isinstance(row.get("error"), str))
    spool = list((home / "spool").iterdir()) if (home / "spool").is_dir() else []
    return {"spool": {"failed": sum(path.is_file() and path.suffix == ".failed" for path in spool),
                      "waiting": sum(path.is_file() and path.suffix in (".json", ".claimed") for path in spool)},
            "hooks": latency, "hook_budget_ok": latency.get("PostToolBatch", {}).get("p95", 0) <= 50,
            "worker_log": dict(worker)}


def _growth(home: Path, snapshots: list[tuple]) -> dict:
    files = [home / "lcm.db", home / "lcm.db-wal"]
    if (home / "payloads").is_dir():
        files.extend(path for path in (home / "payloads").rglob("*") if path.is_file())
    store_bytes = sum(path.stat().st_size for path in files if path.is_file())
    transcript_bytes = sum(lineage.path.stat().st_size for lineage, *_rest in snapshots if lineage.path.is_file())
    return {"store_bytes": store_bytes, "transcript_bytes": transcript_bytes,
            "ratio": store_bytes / transcript_bytes if transcript_bytes else None}


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
    with closing(sqlite3.connect(f"{database.as_uri()}?{query}", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        metadata = _metadata(conn)
        keys = sorted(key for key, cursor in metadata.items() if key.startswith(CURSOR_PREFIX)
                      and (session is None or cursor["session_id"] == session))
        snapshots = [_snapshot(conn, key) for key in keys]
    lag = []
    lineages = [_lineage_report(*snapshot, metadata, lag) for snapshot in snapshots]
    failures = {key: sum(row["failures"][key] for row in lineages) for key in FAILURES}
    unverified = sorted(row["lineage"] for row in lineages
                        if row["transcript_state"] != "ok" or row["generations_gone"])
    return {"ok": not any(failures.values()), "opened_immutable": immutable, "failures": failures,
            "unverified": unverified, "verified": not unverified,
            "lineages": lineages, "sessions": _session_report(metadata, snapshots, session),
            **_home_report(home), "lag_s": _percentiles(lag), "growth": _growth(home, snapshots)}
