#!/usr/bin/env python3
"""LCM-X capture writer for Eve agents.

The Eve memory provider (``adapters/eve/src``) writes one spool record per capture into
``<home>/spool/``. Each record holds the Eve session history at that moment, either in full or as a
delta on a prefix the provider sent before. This writer drains the spool into the home's own LCM-X
store: one writer per store (``capture.lock``), one ``ingest_atomic()`` transaction per record, the
upstream Claude Code unit's admission, stamp and capture ledger (``capture_events``).

Identity of a message is (session, generation, position). Inside one generation the Eve history only
grows at the end, so a position never changes its message. When the history is rewritten (Eve
compaction, ``clear()``, a step that replaced earlier messages), a new generation starts: a message
of the new history that the previous history held unchanged is ``carried`` (no new row), one whose
tool call ids match a previous message is ``carried_modified`` (no new row, its raw bytes kept), and
anything else is ``new``. The compaction checkpoint (Eve's marker plus the summary) is stored as
``system`` rows, so recall never takes the summary as the assistant's own words.

Stdlib only; Python 3.11+ (the LCM-X core requirement).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import types
import zlib
from pathlib import Path
from typing import Any, Optional

API = "lcmx.eve.capture/v1"
HOST = "eve"
PROJECTION_VERSION = "eve-v1"
VIEW_PREFIX = "eve:view:"
SPOOL = "spool"
PACKAGE_NAME = "hermes_lcm"
CHECKPOINT_KIND = "context.compaction"
CHECKPOINT_TAG = "[eve compaction checkpoint]"


# -- package bootstrap ------------------------------------------------------------------------------

def bootstrap(package_root: Optional[str]) -> Path:
    """Register the LCM-X checkout as the bare ``hermes_lcm`` package (never runs its __init__)."""
    root = Path(package_root).expanduser().resolve() if package_root else Path(__file__).resolve().parents[3]
    if not (root / "store.py").is_file() or not (root / "plugin.yaml").is_file():
        raise SystemExit(f"lcmx-eve: no LCM-X package at {root}")
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(root)]
        package.__package__ = PACKAGE_NAME
        sys.modules[PACKAGE_NAME] = package
    return root


def _cc():
    from hermes_lcm import claude_code_store

    return claude_code_store


# -- pure helpers (no store) ------------------------------------------------------------------------

def chain(digests: list[str]) -> str:
    """The provider's prefix digest: sha256 over the message digests joined by newlines."""
    return hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()


def session_rows_id(session_id: str) -> str:
    return f"{HOST}:{session_id}"


def call_key(message: dict) -> Optional[str]:
    """Tool call ids of a message: how a tool result that Eve capped during compaction is matched
    to the full result an earlier generation stored."""
    content = message.get("content")
    if not isinstance(content, list):
        return None
    ids = sorted({str(part.get("toolCallId")) for part in content if isinstance(part, dict)
                  and part.get("type") in ("tool-call", "tool-result") and part.get("toolCallId")})
    return f"{message.get('role')}:{','.join(ids)}" if ids else None


def message_text(message: dict) -> str:
    """The text of a user or assistant message (string content, or its text parts joined)."""
    content = message.get("content")
    return content if isinstance(content, str) else _text_parts(content if isinstance(content, list) else [])


def text_key(message: dict) -> Optional[str]:
    """Role and text of a user or assistant message: how an assistant message that Eve reduced to
    its text during compaction, or a user message it replayed, is matched to the original."""
    role = message.get("role")
    if role not in ("user", "assistant"):
        return None
    content = message.get("content")
    text = content if isinstance(content, str) else _text_parts(content if isinstance(content, list) else [])
    return f"{role}:" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest() if text else None


def is_framework(message: Optional[dict]) -> bool:
    """Eve's own user-role messages (checkpoint marker, memory load, resumption), not the user's."""
    return bool(message) and message.get("role") == "user" and message.get("kind") not in (None, "user")


def message_meta(message: dict, *, checkpoint: bool = False) -> list:
    """[role, call key, text key, framework]: what alignment needs of a position, kept in the view."""
    return [message.get("role"), call_key(message), text_key(message), is_framework(message) or checkpoint]


def _text_parts(parts: list[Any]) -> str:
    return "\n\n".join(part["text"] for part in parts
                       if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str))


def _placeholder(part: dict) -> str:
    kind = part.get("type", "part")
    media = part.get("mediaType") or part.get("mimeType") or ""
    name = part.get("filename") or ""
    label = " ".join(bit for bit in (kind, media, name) if bit)
    return f"[{label}]"


def _content_text(content: Any) -> str:
    """User or system content as text; non-text parts become a short placeholder (the raw message
    keeps the bytes in the ledger)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content, ensure_ascii=False, sort_keys=True)
    pieces = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            pieces.append(part["text"])
        else:
            pieces.append(_placeholder(part))
    return "\n\n".join(pieces)


def output_text(output: Any) -> str:
    """An AI SDK ``ToolResultOutput`` as the text a tool row holds."""
    if not isinstance(output, dict):
        return "" if output is None else json.dumps(output, ensure_ascii=False, sort_keys=True)
    kind = output.get("type")
    value = output.get("value")
    if kind == "text":
        return str(value or "")
    if kind == "error-text":
        return f"[error] {value or ''}"
    if kind == "json":
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if kind == "error-json":
        return "[error] " + json.dumps(value, ensure_ascii=False, sort_keys=True)
    if kind == "execution-denied":
        reason = output.get("reason")
        return "[execution denied]" + (f" {reason}" if reason else "")
    if kind == "content" and isinstance(value, list):
        return _content_text(value)
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def project(message: dict, *, checkpoint: bool = False) -> list[dict]:
    """LCM-X rows (OpenAI-style dicts) for one Eve history message. Reasoning, approval parts and
    binary data stay in the ledger's raw message only."""
    role = message.get("role")
    content = message.get("content")
    if role == "system":
        return [{"role": "system", "content": _content_text(content)}]
    if role == "user":
        kind = message.get("kind")
        text = _content_text(content)
        if kind and kind != "user":  # Eve's own framework message (checkpoint marker, memory load, ...)
            return [{"role": "system", "content": f"[eve {kind}]\n{text}"}]
        return [{"role": "user", "content": text}]
    if role == "assistant":
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else [
            part for part in content or [] if isinstance(part, dict)]
        text = _text_parts(parts)
        if checkpoint:
            return [{"role": "system", "content": f"{CHECKPOINT_TAG}\n{text}"}]
        calls = [{"id": part.get("toolCallId"), "type": "function", "function": {
            "name": part.get("toolName"), "arguments": json.dumps(part.get("input"), ensure_ascii=False, sort_keys=True)}}
            for part in parts if part.get("type") == "tool-call"]
        rows: list[dict] = []
        if text or calls:
            rows.append({"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})})
        rows += [{"role": "tool", "content": output_text(part.get("output")), "tool_call_id": part.get("toolCallId"),
                  "tool_name": part.get("toolName")} for part in parts if part.get("type") == "tool-result"]
        return rows
    if role == "tool":
        return [{"role": "tool", "content": output_text(part.get("output")), "tool_call_id": part.get("toolCallId"),
                 "tool_name": part.get("toolName")}
                for part in content or [] if isinstance(part, dict) and part.get("type") == "tool-result"]
    return []


def checkpoint_positions(messages: list[Optional[dict]]) -> set[int]:
    """Positions of Eve's compaction checkpoint summary: the assistant message right after the
    ``context.compaction`` marker."""
    found = set()
    for index, message in enumerate(messages[:-1]):
        following = messages[index + 1]
        if (message and message.get("role") == "user" and message.get("kind") == CHECKPOINT_KIND
                and following and following.get("role") == "assistant"):
            found.add(index + 1)
    return found


def align(view: dict, digests: list[str], meta: list[list], *, same_turn: bool = False, later_turn: bool = False,
          fresh: frozenset[str] = frozenset(), force_new: bool = False,
          texts: Optional[list[Optional[str]]] = None) -> tuple[int, list[tuple[str, Optional[int]]]]:
    """(generation, per-position disposition) of a history against the stored view.

    Same generation when the stored view is a prefix of the history (new positions are ``new``,
    earlier ones ``held``) or the history is a prefix of the view (a replayed or stale record: all
    ``held``; not for a later turn, whose shorter history is a rewrite). Otherwise the history was rewritten (Eve compaction, ``clear()``) and a new generation
    starts. Eve's compaction gives ``[marker, summary, tail..., replay?]``: the tail is the end of the
    old history, with tool results capped, or tool messages dropped and assistant messages reduced to
    their text; the replay is the turn's last user message, re-sent when the compaction ran inside
    that turn. So, in order:

    1. leading framework messages (marker, summary) are ``new``;
    2. the longest tail run that ends exactly at the old view's end is ``carried`` (same digest) or
       ``carried_modified`` (same tool call ids, or same role and text); old tool and framework
       messages may be missing from it;
    3. when ``same_turn``, one user message equal to the old view's last user message is ``carried``;
    4. everything else is ``new``, so a new message never merges with an older identical one.

    A message in ``fresh`` (the input of a turn the old view did not reach) never matches: after
    ``clear()`` the same question asked again is a new row.

    Carried positions add no row; the int is the old position they came from."""
    old = view.get("digests", [])
    old_meta = view.get("meta", [])
    generation = int(view.get("generation", 0))
    stale = old[:len(digests)] == digests and not later_turn  # a later turn's history never shrinks
    if not force_new and (digests[:len(old)] == old or stale):
        return generation, [("held", index) if index < len(old) else ("new", None) for index in range(len(digests))]

    def old_info(index: int) -> list:
        return old_meta[index] if index < len(old_meta) else [None, None, None, False]

    def match(new: int, previous: int) -> Optional[str]:
        if digests[new] in fresh:
            return None
        if digests[new] == old[previous]:
            return "carried"
        mine, theirs = meta[new], old_info(previous)
        if mine[1] and mine[1] == theirs[1]:
            return "carried_modified"
        if mine[2] and mine[2] == theirs[2]:
            return "carried_modified"
        return None

    def droppable(previous: int) -> bool:
        role, _call, _text, framework = old_info(previous)
        return role == "tool" or bool(framework)

    result: list[tuple[str, Optional[int]]] = [("new", None)] * len(digests)
    position = 0
    while position < len(digests) and meta[position][3]:
        position += 1

    def walk(start: int) -> Optional[tuple[list[tuple[int, str, int]], int]]:
        found: list[tuple[int, str, int]] = []
        new, previous = position, start
        while new < len(digests) and previous < len(old):
            kind = match(new, previous)
            if kind:
                found.append((new, kind, previous))
                new, previous = new + 1, previous + 1
            elif droppable(previous):
                previous += 1
            else:
                return None
        while previous < len(old) and droppable(previous):
            previous += 1
        return (found, new) if found and previous == len(old) else None

    for start in range(len(old)):
        tail = walk(start)
        if tail is not None:
            for new, kind, previous in tail[0]:
                result[new] = (kind, previous)
            position = tail[1]
            break
    if same_turn and position < len(digests) and meta[position][0] == "user" and not meta[position][3]:
        last_user = next((index for index in range(len(old) - 1, -1, -1)
                          if old_info(index)[0] == "user" and not old_info(index)[3]), None)
        if last_user is not None and meta[position][2] and meta[position][2] == old_info(last_user)[2]:
            result[position] = ("carried", last_user)
        elif last_user is not None and texts and position < len(texts) and is_cut_copy(
                texts[position], view.get("last_user_text")):
            result[position] = ("carried_modified", last_user)  # Eve re-sent it with its middle cut out
    return generation + 1, result


EVE_CUT_NOTE = re.compile(r"\n\n\[eve: \d+ characters from the middle of this message were cut during context compaction "
                          r"to fit the context window\.\]\n\n")


def is_cut_copy(candidate: Optional[str], original: Optional[str]) -> bool:
    """Eve's compaction re-sends a turn's user message that no longer fits with its middle cut out and a note
    ("[eve: N characters from the middle of this message were cut ...]"). Such a copy is the original when the
    text before the note starts it and the text after the note ends it."""
    if not candidate or not original:
        return False
    parts = EVE_CUT_NOTE.split(candidate)
    if len(parts) != 2:
        return False
    head, tail = parts
    return bool(head or tail) and original.startswith(head) and original.endswith(tail) and (
        len(head) + len(tail) < len(original))


# -- spool ------------------------------------------------------------------------------------------

def spool_dir(home: Path) -> Path:
    return home / SPOOL


def pending_records(home: Path) -> list[Path]:
    folder = spool_dir(home)
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix == ".json" and not path.name.startswith("."))


def read_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("api") != API:
        raise ValueError("not an lcmx.eve.capture/v1 record")
    session = record.get("session")
    if not isinstance(session, dict) or not isinstance(session.get("id"), str) or not session["id"]:
        raise ValueError("record has no session id")
    base = record.get("base") or {"count": 0, "chain": chain([])}
    messages = record.get("messages")
    if not isinstance(messages, list) or not all(
            isinstance(item, dict) and isinstance(item.get("d"), str) and isinstance(item.get("m"), dict)
            for item in messages):
        raise ValueError("record messages are malformed")
    if not isinstance(base.get("count"), int) or base["count"] < 0:
        raise ValueError("record base is malformed")
    record["base"] = base
    return record


# -- ingest -----------------------------------------------------------------------------------------

INSERT_EVENT = (
    "INSERT OR IGNORE INTO capture_events(session_id, lineage_key, source_kind, generation, position, uid, "
    "entry_type, entry_class, line_digest, content_digest, disposition, store_ids, notes, raw, raw_from, "
    "projection_version, captured_at) VALUES (?, ?, 'line', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)")


def plan_record(view: Optional[dict], record: dict) -> dict:
    """What one spool record adds, against the stored view of its session (no I/O).

    Returns ``session``, ``generation``, ``events`` (one per position that is not ``held``: position,
    digest, message, disposition, rows, notes, ``framework``), ``rows`` (the LCM-X rows of the ``new``
    events, in order), ``notes`` and ``view`` (the view to store when ``update_view``)."""
    session = session_rows_id(record["session"]["id"])
    view = view or {"generation": 0, "digests": [], "meta": []}
    old = list(view.get("digests", []))
    old_meta = list(view.get("meta", []))
    base = record["base"]
    incoming = record["messages"]
    record_notes: list[str] = []
    mismatch = bool(base["count"]) and (base["count"] > len(old) or chain(old[:base["count"]]) != base["chain"])
    if mismatch:
        # The provider's prefix is not what this store holds (a lost or failed record): keep this
        # record's messages as a history of their own, so nothing in it is lost; the audit counts it.
        record_notes.append("base_mismatch")
        prefix = 0
    else:
        prefix = base["count"]
    digests = old[:prefix] + [item["d"] for item in incoming]
    messages: list[Optional[dict]] = [None] * prefix + [item["m"] for item in incoming]
    checkpoints = checkpoint_positions(messages)
    meta = [(old_meta[index] if index < len(old_meta) else [None, None, None, False]) if message is None
            else message_meta(message, checkpoint=index in checkpoints) for index, message in enumerate(messages)]
    turn = record.get("turn") if isinstance(record.get("turn"), dict) else {}
    same_turn = bool(turn.get("id")) and turn.get("id") == view.get("turn")
    later_turn = isinstance(turn.get("sequence"), int) and isinstance(view.get("sequence"), int) and (
        turn["sequence"] > view["sequence"])
    fresh = frozenset() if same_turn else frozenset(
        digest for digest in turn.get("input", []) if isinstance(digest, str))
    texts = [message_text(message) if message is not None and message.get("role") == "user" else None
             for message in messages]
    generation, plan = align(view, digests, meta, same_turn=same_turn, later_turn=later_turn, fresh=fresh,
                             force_new=mismatch, texts=texts)
    stamp = record.get("at") if isinstance(record.get("at"), str) else None

    batch: list[dict] = []
    events: list[dict] = []
    for position, ((disposition, origin), message, digest) in enumerate(zip(plan, messages, digests)):
        if disposition == "held":
            continue
        if message is None:  # cannot happen: a prefix position is always held
            record_notes.append(f"missing_message:{position}")
            continue
        rows = project(message, checkpoint=position in checkpoints) if disposition == "new" else []
        if stamp:
            rows = [{**row, "timestamp": stamp} for row in rows]
        notes = list(record_notes)
        if origin is not None:
            notes.append(f"from:g{view.get('generation', 0)}:{origin}")
        if position in checkpoints:
            notes.append("checkpoint")
        events.append({"position": position, "digest": digest, "message": message, "rows": rows, "store_ids": [],
                       "disposition": "raw_only" if disposition == "new" and not rows else disposition,
                       "notes": notes, "framework": bool(meta[position][3])})
        batch.extend(rows)
    update_view = generation != int(view.get("generation", 0)) or len(digests) >= len(old)
    last_user_text = view.get("last_user_text")
    for index in range(len(messages) - 1, -1, -1):  # the newest real user message this record shows
        if messages[index] is not None and meta[index][0] == "user" and not meta[index][3]:
            if not (plan[index][0] == "carried_modified" and is_cut_copy(texts[index], last_user_text)):
                last_user_text = texts[index]
            break
    new_view = {"generation": generation, "digests": digests, "meta": meta, "turn": turn.get("id"),
                "last_user_text": last_user_text,
                "sequence": turn.get("sequence") if isinstance(turn.get("sequence"), int) else view.get("sequence"),
                "op": record.get("op"), "phase": record.get("phase"), "at": stamp}
    return {"session": session, "generation": generation, "events": events, "rows": batch, "notes": record_notes,
            "view": new_view, "update_view": update_view}


def ingest_record(store: Any, record: dict) -> dict:
    """Write one spool record in one transaction: rows, bindings, ledger rows and the new view."""
    plan = plan_record(store.read_metadata_json(VIEW_PREFIX + session_rows_id(record["session"]["id"])), record)
    session, generation, events, batch = plan["session"], plan["generation"], plan["events"], plan["rows"]
    record_notes = plan["notes"]

    with store.ingest_atomic():
        store_ids = iter(store.append_batch(session, batch, source=HOST) if batch else [])
        bindings = []
        for event in events:
            for index, _row in enumerate(event["rows"]):
                store_id = next(store_ids)
                event["store_ids"].append(store_id)
                uid = f"g{generation}:{event['position']}" + (f"#{index}" if len(event["rows"]) > 1 else "")
                bindings.append((store_id, uid, "canonical", "eve_position"))
        store.add_host_uid_bindings(session, bindings)
        now = time.time()
        conn = store.connection
        for event in events:
            raw = json.dumps(event["message"], ensure_ascii=False, sort_keys=True).encode("utf-8", "surrogatepass")
            conn.execute(INSERT_EVENT, (
                session, session, generation, event["position"], f"g{generation}:{event['position']}",
                event["message"].get("role"), "projected" if event["rows"] else "raw_only", event["digest"],
                event["digest"], event["disposition"],
                json.dumps(event["store_ids"]) if event["rows"] else None,
                json.dumps(event["notes"]) if event["notes"] else None,
                None if event["disposition"] == "carried" else zlib.compress(raw), PROJECTION_VERSION, now))
        if plan["update_view"]:
            conn.execute("INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET "
                         "value = excluded.value", (VIEW_PREFIX + session, json.dumps(plan["view"])))
    return {"session": session, "generation": generation, "rows": len(batch),
            "new": sum(1 for event in events if event["disposition"] in ("new", "raw_only")),
            "carried": sum(1 for event in events if event["disposition"].startswith("carried")),
            "notes": record_notes}


# -- worker -----------------------------------------------------------------------------------------

def _log(home: Path, entry: dict) -> None:
    try:
        logs = home / "logs"
        logs.mkdir(mode=0o700, parents=True, exist_ok=True)
        with open(logs / "writer.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": round(time.time(), 3), **entry}, sort_keys=True) + "\n")
    except OSError:
        pass


def drain(home: Path, *, embed: bool = True) -> dict:
    """Drain the spool of one home under its store lock; rescan after release, as the upstream
    worker does, so a record spooled while the lock was held is never left behind."""
    cc = _cc()
    home = home.resolve()
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    report: dict[str, Any] = {"records": 0, "drained": 0, "failed": 0, "rows": 0, "notes": []}
    lock = cc._Lock(home)
    try:
        while True:
            if not lock.acquire(0.0):
                report["lock"] = "held"
                return report
            try:
                if not _drain_once(home, report, embed=embed):
                    return report
            finally:
                lock.release()
            if not pending_records(home):
                return report
    finally:
        lock.close()


def _drain_once(home: Path, report: dict, *, embed: bool) -> bool:
    cc = _cc()
    try:
        database = cc.admit(home, host=HOST)
    except cc.AdmissionError as refused:
        report["refused"] = refused.code
        _log(home, {"event": "refused", "code": refused.code})
        return False
    store = cc.open_store(database, home)
    try:
        before = store.connection.execute("SELECT MAX(store_id) FROM messages").fetchone()[0]
        for path in pending_records(home):
            report["records"] += 1
            try:
                record = read_record(path)
                result = ingest_record(store, record)
            except Exception as error:  # noqa: BLE001 - set aside, logged without content; capture goes on
                report["failed"] += 1
                _log(home, {"event": "record_failed", "file": path.name, "error": type(error).__name__,
                            "detail": str(error)[:200]})
                os.replace(path, path.with_suffix(".failed"))
                continue
            report["drained"] += 1
            report["rows"] += result["rows"]
            report["notes"] += result["notes"]
            _log(home, {"event": "record", "file": path.name, **{k: v for k, v in result.items() if k != "session"}})
            path.unlink(missing_ok=True)
        if embed:
            try:
                report["embedding"], report["vectors"] = embed_after_drain(home, database, store, before)
            except Exception as error:  # noqa: BLE001 - vectors are optional; full-text recall still works
                report["embedding"] = "failed:" + type(error).__name__
                _log(home, {"event": "embed_failed", "error": type(error).__name__, "detail": str(error)[:200]})
        return True
    finally:
        switched = cc.to_rollback_journal(store.connection)
        store.close()
        if not switched and not cc.leave_readable(database):
            _log(home, {"event": "still_wal"})


def embed_after_drain(home: Path, database: Path, store: Any, before: Optional[int]) -> tuple[str, int]:
    """Catch up vectors after a drain, with the ``LCM_EMBEDDING_*`` settings in the environment (the
    Eve provider passes its process environment on). Only FastEmbed, through the upstream Claude Code
    unit's local path. An API provider (``openai-compatible``) is skipped: the upstream read-only reader
    never loads an API key and serves full-text recall only, so API vectors need a reader of our own
    first (follow-up). Full-text recall works without any embedding."""
    from hermes_lcm.config import LCMConfig

    config = LCMConfig.from_env()
    if not config.embeddings_enabled:
        return "skipped_disabled", 0
    return _cc()._embed_after_drain(home, database, store, config, None, before, {})


# -- audit ------------------------------------------------------------------------------------------

def audit(home: Path) -> dict:
    """Lossless check of one home: every position of every session's current view has a ledger row
    in that generation; every ``new`` ledger row's rows exist; every stored eve row has exactly one
    ledger owner (0 lost, 0 duplicated)."""
    import sqlite3
    from contextlib import closing

    database = home.resolve() / "lcm.db"
    report: dict[str, Any] = {"sessions": 0, "positions": 0, "missing_positions": 0, "rows": 0,
                              "ledger_rows": 0, "dangling_ids": 0, "unowned_rows": 0, "multi_owned_rows": 0,
                              "base_mismatch": 0, "pending": len(pending_records(home)),
                              "failed_records": len(list(spool_dir(home).glob("*.failed"))) if spool_dir(home).is_dir() else 0}
    if not database.exists():
        report["store"] = "missing"
        return report
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as conn:
        views = conn.execute("SELECT key, value FROM metadata WHERE key LIKE ?", (VIEW_PREFIX + "%",)).fetchall()
        for key, value in views:
            view = json.loads(value)
            session = key[len(VIEW_PREFIX):]
            report["sessions"] += 1
            positions = {row[0] for row in conn.execute(
                "SELECT position FROM capture_events WHERE lineage_key = ? AND generation = ?",
                (session, view["generation"]))}
            report["positions"] += len(view["digests"])
            report["missing_positions"] += sum(1 for index in range(len(view["digests"])) if index not in positions)
        owners: dict[int, int] = {}
        for store_ids, in conn.execute(
                "SELECT store_ids FROM capture_events WHERE projection_version = ? AND store_ids IS NOT NULL",
                (PROJECTION_VERSION,)):
            report["ledger_rows"] += 1
            for store_id in json.loads(store_ids or "[]"):
                owners[store_id] = owners.get(store_id, 0) + 1
        stored = {row[0] for row in conn.execute("SELECT store_id FROM messages WHERE source = ?", (HOST,))}
        report["rows"] = len(stored)
        report["dangling_ids"] = sum(1 for store_id in owners if store_id not in stored)
        report["unowned_rows"] = sum(1 for store_id in stored if store_id not in owners)
        report["multi_owned_rows"] = sum(1 for count in owners.values() if count > 1)
        report["base_mismatch"] = conn.execute(
            "SELECT COUNT(*) FROM capture_events WHERE projection_version = ? AND notes LIKE '%base_mismatch%'",
            (PROJECTION_VERSION,)).fetchone()[0]
    report["ok"] = not any(report[name] for name in (
        "missing_positions", "dangling_ids", "unowned_rows", "multi_owned_rows", "pending", "failed_records",
        "base_mismatch"))
    return report


# -- CLI --------------------------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    if sys.version_info < (3, 11):
        sys.stderr.write("lcmx-eve writer needs Python 3.11 or newer\n")
        return 2
    parser = argparse.ArgumentParser(prog="lcmx_eve_writer.py", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("drain", "audit"))
    parser.add_argument("--home", required=True, help="absolute store home (one Eve memory scope)")
    parser.add_argument("--package-root", default=os.environ.get("LCMX_PACKAGE_ROOT", ""),
                        help="LCM-X checkout (default: the checkout holding this file)")
    parser.add_argument("--no-embed", action="store_true", help="skip the local vector catch-up")
    options = parser.parse_args(argv)
    home = Path(options.home).expanduser()
    if not home.is_absolute():
        sys.stderr.write("lcmx-eve: --home must be absolute\n")
        return 2
    bootstrap(options.package_root or None)
    result = drain(home, embed=not options.no_embed) if options.command == "drain" else audit(home)
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0 if options.command == "drain" or result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
