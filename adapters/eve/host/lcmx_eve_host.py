#!/usr/bin/env python3
"""LCM-X engine host for Eve agents.

This process is to an Eve agent what Hermes is to LCM-X: it runs the real ``LCMEngine`` (upstream, unchanged)
for one store home and drives it the way the Hermes plugin hooks do. For each Eve session it keeps the
engine's own view of the conversation (the "shadow", as a Hermes host keeps its message list):

- Eve's capture handlers spool the session history (``adapters/eve/src``). The host aligns each record
  with what it saw before (``lcmx_eve_writer.plan_record``: generation, position, Eve's compaction tail),
  appends only the new messages to the shadow, then calls ``engine.ingest(shadow)`` and
  ``note_turn_complete()`` (Hermes ``post_llm_call``) and, past the engine's threshold,
  ``engine.compress(shadow)`` (Hermes before a model call). So the engine stores every message once,
  builds its summary DAG and runs its maintenance exactly as under Hermes. Eve's own compaction output
  (marker and summary) stays in the raw ledger and never enters the shadow.
- The model's ``lcmx__*`` tools call ``engine.handle_tool_call`` with the engine bound to the caller's session.

One host per store home holds ``host.lock`` for its lifetime (one writer per store, upstream #1192). The raw
ledger (``eve-ledger.db``: every message Eve showed, zlib, with its disposition), the views and the shadows live
beside the engine's ``lcm.db``, which only the engine writes.

    lcmx_eve_host.py serve --home H [--core C] [--context-length N]   stdio JSON-RPC, until stdin closes
    lcmx_eve_host.py drain --home H [--core C]                         drain the spool once, then exit
    lcmx_eve_host.py audit --home H
    lcmx_eve_host.py check [--core C]                                  can this runtime run the engine?

LLM for summaries and expand_query: ``shim/agent/auxiliary_client.py`` (OpenAI-compatible, key from the env).
Python 3.11+, stdlib (FastEmbed optional for vectors).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
import zlib
from contextlib import closing
from pathlib import Path
from typing import Any, Optional, TextIO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "shim"))  # agent.context_engine, agent.auxiliary_client (the Hermes host modules)
sys.path.insert(0, str(HERE.parent / "writer"))
import lcmx_eve_writer as planner  # noqa: E402

logger = logging.getLogger("lcmx-eve-host")
LEDGER = "eve-ledger.db"
LOCK = "host.lock"
PLATFORM = "eve"
IDLE_SCAN_S = 2.0
LOCK_WAIT_S = 60.0    # a request waits this long for an earlier Eve process's host to release the store
TURN_MAX_S = 600.0      # a turn that never completed stops holding back idle summaries after this
BACKLOG_QUIET_S = 30.0  # idle summary check for sessions left over their threshold
EMBED_IDLE_S = 5.0     # more vectors may wait: try again soon
EMBED_QUIET_S = 60.0   # nothing was left: look again later
LEDGER_DDL = (
    "CREATE TABLE IF NOT EXISTS views (session TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS shadows (session TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS events (event_id INTEGER PRIMARY KEY, session TEXT NOT NULL, generation INTEGER "
    "NOT NULL, position INTEGER NOT NULL, digest TEXT NOT NULL, role TEXT, disposition TEXT NOT NULL, fed_rows "
    "INTEGER NOT NULL, notes TEXT, raw BLOB, op TEXT, captured_at REAL NOT NULL, "
    "UNIQUE(session, generation, position))",
)


# -- runtime --------------------------------------------------------------------------------------

def _cut_query_vectors_to_store_dim() -> None:
    """Core defect (still on main when this adapter was written): a stored vector is cut to
    ``LCM_EMBEDDING_STORE_DIM`` (Matryoshka) on write, but the query vector is not cut before KNN ("embedding
    dimension mismatch"). Cut it the same way here, so the adapter edits no core file."""
    from hermes_lcm.vector_store import VectorStore

    normalized = VectorStore._normalized
    if getattr(normalized, "_lcmx_eve_cut", False):
        return

    def cut_then_normalize(values, *, expected_dim):
        values = list(values)
        return normalized(values[:expected_dim] if len(values) > expected_dim else values, expected_dim=expected_dim)

    cut_then_normalize._lcmx_eve_cut = True  # type: ignore[attr-defined]
    VectorStore._normalized = staticmethod(cut_then_normalize)


class Bridge:
    """``call_llm`` over stdio to the Eve process: a request ``llm/complete`` with an id of its own (``h<n>``); the
    stdin reader hands the reply back. One call at a time (low priority, never parallel to itself)."""

    def __init__(self, write: "Any"):
        self.write = write
        self.lock = threading.Lock()
        self.waiting: dict[str, tuple[threading.Event, dict]] = {}
        self.counter = 0
        self.closed = False

    def close(self) -> None:
        """Eve's process ended: no model answers any more. Fail every waiting call now, and later ones at once,
        so the host never holds the store lock while it waits for a model that is gone."""
        self.closed = True
        for event, box in list(self.waiting.values()):
            box["error"] = "the Eve process ended"
            event.set()

    def complete(self, params: dict, timeout: float) -> dict:
        if self.closed:
            raise RuntimeError("Eve model: the Eve process ended")
        with self.lock:
            self.counter += 1
            ident = f"h{self.counter}"
            event, box = threading.Event(), {}
            self.waiting[ident] = (event, box)
            try:
                self.write({"jsonrpc": "2.0", "id": ident, "method": "llm/complete", "params": params})
                if not event.wait(timeout):
                    raise TimeoutError(f"the Eve model did not answer within {timeout:.0f} s")
            finally:
                self.waiting.pop(ident, None)
        if "error" in box:
            raise RuntimeError(f"Eve model: {box['error']}")
        return box.get("result") or {}

    def deliver(self, message: dict) -> bool:
        """True when ``message`` is a reply to one of our requests."""
        ident = message.get("id")
        if not isinstance(ident, str) or "method" in message or ident not in self.waiting:
            return False
        event, box = self.waiting[ident]
        if message.get("error"):
            box["error"] = (message["error"] or {}).get("message", "error")
        else:
            box["result"] = message.get("result")
        event.set()
        return True

class Lock:
    def __init__(self, home: Path):
        self.handle = open(home / LOCK, "a")

    def try_acquire(self) -> bool:
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def close(self) -> None:
        self.handle.close()


def open_ledger(home: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(home / LEDGER, isolation_level=None, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    for sql in LEDGER_DDL:
        conn.execute(sql)
    return conn


def load_json(conn: sqlite3.Connection, table: str, session: str) -> Any:
    row = conn.execute(f"SELECT value FROM {table} WHERE session = ?", (session,)).fetchone()
    return json.loads(row[0]) if row else None


class Host:
    """One engine, one store home, the shadows of its sessions."""

    def __init__(self, home: Path, *, context_length: int, model: str, summarize: bool = False):
        """``summarize``: an LLM is attached (the Eve bridge), so the engine may compact its view and write LLM
        summaries. Without it (a timer drain) the host only ingests; summaries wait for the next live turn."""
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.engine import LCMEngine

        self.home = home
        self.context_length = context_length
        self.model = model
        self.summarize = summarize
        config = LCMConfig.from_env()
        config.database_path = str(home / "lcm.db")
        self.config = config
        _cut_query_vectors_to_store_dim()
        self.engine = LCMEngine(config=config, hermes_home=str(home))
        self.ledger = open_ledger(home)
        self.bound: Optional[str] = None
        self.profiles_registered = False
        self.backlog_tried: dict[str, int] = {}  # session -> shadow tokens at the last idle summary attempt
        self.turn_started_at: Optional[float] = None  # an Eve turn runs (from turn/started until turn.completed)

    def bind(self, session: str) -> None:
        """Hermes ``on_session_start`` when the turn's session changes (as a gateway rebinds)."""
        if self.bound == session:
            return
        self.engine.on_session_start(session, platform=PLATFORM, conversation_id=session, model=self.model,
                                     provider=PLATFORM, context_length=self.context_length)
        self.engine.update_model(model=self.model, context_length=self.context_length, provider=PLATFORM)
        self.bound = session

    # -- capture --

    def drain(self) -> dict:
        report: dict[str, Any] = {"records": 0, "drained": 0, "failed": 0, "fed": 0, "compressed": 0, "notes": []}
        for path in planner.pending_records(self.home):
            report["records"] += 1
            try:
                result = self.ingest_record(planner.read_record(path))
            except Exception as error:  # noqa: BLE001 - set aside, logged without content; capture goes on
                report["failed"] += 1
                logger.exception("record %s failed", path.name)
                planner._log(self.home, {"event": "record_failed", "file": path.name, "error": type(error).__name__,
                                         "detail": str(error)[:200]})
                os.replace(path, path.with_suffix(".failed"))
                continue
            report["drained"] += 1
            report["fed"] += result["fed"]
            report["compressed"] += int(result["compressed"])
            report["notes"] += result["notes"]
            planner._log(self.home, {"event": "record", "file": path.name, **{k: v for k, v in result.items()
                                                                              if k != "session"}})
            path.unlink(missing_ok=True)
        return report

    def ingest_record(self, record: dict) -> dict:
        """Hermes per turn, on the shadow: ingest + turn complete, then compress past the threshold. The ledger
        commits after the engine: a crash in between replays the record, and the engine's exactly-once ingest
        takes the same shadow again without a second row."""
        from hermes_lcm import tokens

        session = planner.session_rows_id(record["session"]["id"])
        if record.get("phase") == "turn.completed":
            self.turn_started_at = None
        plan = planner.plan_record(load_json(self.ledger, "views", session), record)
        shadow: list[dict] = load_json(self.ledger, "shadows", session) or []
        fed: list[dict] = []
        for event in plan["events"]:
            # Eve's own scaffolding (its compaction marker and summary, memory loads) stays raw in the ledger:
            # the engine keeps its own summaries, as under Hermes.
            event["fed"] = [] if event["framework"] or "checkpoint" in event["notes"] else event["rows"]
            fed += event["fed"]
        compressed = False
        if fed:
            shadow = shadow + fed
            self.bind(session)
            self.engine.ingest(shadow)
            self.engine.note_turn_complete()
            current = tokens.count_messages_tokens(shadow)
            # Summaries only between turns (after turn.completed), never inside one (compaction.requested runs
            # inside Eve's turn), and only with an LLM attached: they never compete with the agent's own calls.
            if self.summarize and record.get("phase") == "turn.completed" and self.engine.should_compress(current):
                shadow = self.engine.compress(shadow, current_tokens=current)
                compressed = True
        now = time.time()
        conn = self.ledger
        conn.execute("BEGIN IMMEDIATE")
        try:
            for event in plan["events"]:
                raw = json.dumps(event["message"], ensure_ascii=False, sort_keys=True).encode("utf-8", "surrogatepass")
                conn.execute(
                    "INSERT OR IGNORE INTO events(session, generation, position, digest, role, disposition, fed_rows, "
                    "notes, raw, op, captured_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (session, plan["generation"], event["position"], event["digest"], event["message"].get("role"),
                     event["disposition"], len(event["fed"]), json.dumps(event["notes"]) if event["notes"] else None,
                     None if event["disposition"] == "carried" else zlib.compress(raw), record.get("op"), now))
            if plan["update_view"]:
                conn.execute("INSERT INTO views(session, value) VALUES (?, ?) ON CONFLICT(session) DO UPDATE SET "
                             "value = excluded.value", (session, json.dumps(plan["view"])))
            conn.execute("INSERT INTO shadows(session, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(session) DO "
                         "UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                         (session, json.dumps(shadow, ensure_ascii=False), now))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return {"session": session, "generation": plan["generation"], "fed": len(fed), "compressed": compressed,
                "shadow": len(shadow), "notes": plan["notes"]}

    def summarize_backlog(self) -> Optional[dict]:
        """Idle time, LLM attached: summarize one session whose engine view is still over the threshold, for
        example when Eve ended during that session's last summary call. Never inside a turn (the host is idle),
        never twice for an unchanged view."""
        from hermes_lcm import tokens

        if not self.summarize or self.in_turn():
            return None
        for session, value in self.ledger.execute("SELECT session, value FROM shadows ORDER BY updated_at DESC"):
            shadow = json.loads(value)
            current = tokens.count_messages_tokens(shadow)
            if self.backlog_tried.get(session) == current:
                continue
            self.bind(session)
            if not self.engine.should_compress(current):
                self.backlog_tried[session] = current
                continue
            compressed = self.engine.compress(shadow, current_tokens=current)
            after = tokens.count_messages_tokens(compressed)
            self.backlog_tried[session] = after
            self.save_shadow(session, compressed)
            return {"session": session, "tokens": [current, after]}
        return None

    def in_turn(self) -> bool:
        """An Eve turn runs: it started (``turn/started``) and no ``turn.completed`` came yet, for at most 10 min."""
        return self.turn_started_at is not None and time.monotonic() - self.turn_started_at < TURN_MAX_S

    def save_shadow(self, session: str, shadow: list[dict]) -> None:
        self.ledger.execute("INSERT INTO shadows(session, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(session) DO "
                            "UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                            (session, json.dumps(shadow, ensure_ascii=False), time.time()))

    # -- vectors --

    def embed_catchup(self, budget_s: float = 20.0) -> dict:
        """Vectors for an API provider (openai-compatible), between turns, bounded: register the profiles once,
        then upstream's summary and chunk backfills until nothing is left, the budget ends or a record waits.
        ``confirm_raw_text`` is the operator's consent to send raw text to the provider: choosing API embeddings
        in ``defineLcmxMemory`` is that consent. FastEmbed embeds by itself in the engine's maintenance."""
        from hermes_lcm import command
        from hermes_lcm.chunking import normalize_content_policy
        from hermes_lcm.config import embedding_provider_id

        config = self.config
        provider = embedding_provider_id(str(getattr(config, "embedding_provider", "") or "").strip().lower())
        if not config.embeddings_enabled or provider in ("", "fastembed"):
            return {"embedding": "local_or_off"}
        db = config.database_path
        if not self.profiles_registered:
            registered = command._embedding_register_profiles(config, db, allow_download=False)
            if isinstance(registered, str):
                return {"embedding": "register_refused", "detail": registered.splitlines()[-1][:200]}
            self.profiles_registered = True
        limit = command._embedding_backfill_batch_size(getattr(config, "embedding_max_batch_items", None))
        policy = normalize_content_policy(getattr(config, "embedding_content_policy", "conversational"))
        end = time.monotonic() + budget_s
        report = {"embedding": provider, "summaries": 0, "chunks": 0}
        for kind in ("summaries", "chunks"):
            while time.monotonic() < end and not planner.pending_records(self.home):
                if kind == "summaries":
                    result = command._embedding_backfill_summary_run(config, db, apply=True, limit=limit,
                                                                      retry_uncertain=False)
                else:
                    result = command._chunk_backfill_run(config, db, apply=True, limit=limit, retry_uncertain=False,
                                                         policy=policy, confirm_raw_text=True, incremental=True)
                if isinstance(result, str):
                    report[kind + "_refused"] = result.splitlines()[-1][:200]
                    break
                got = int(result.get("embedded", 0) or 0)
                report[kind] += got
                if got == 0:
                    break
        return report

    # -- tools --

    def tool_schemas(self) -> list[dict]:
        return self.engine.get_tool_schemas()

    def call_tool(self, session: Optional[str], name: str, arguments: dict) -> str:
        if session:
            self.bind(planner.session_rows_id(session))
        return self.engine.handle_tool_call(name, arguments or {})

    def close(self) -> None:
        try:
            self.engine.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("engine shutdown")
        self.ledger.close()
        planner._cc().leave_readable(self.home / "lcm.db")


# -- audit ----------------------------------------------------------------------------------------

def audit(home: Path) -> dict:
    """Lossless check: every position of every session's view has a ledger event in that generation; the engine
    holds exactly the rows the host fed it (no row lost, none twice); the spool is empty."""
    home = home.resolve()
    report: dict[str, Any] = {"sessions": 0, "positions": 0, "missing_positions": 0, "fed_rows": 0, "store_rows": 0,
                              "row_mismatch_sessions": 0, "pending": len(planner.pending_records(home)),
                              "failed_records": len(list((home / planner.SPOOL).glob("*.failed")))
                              if (home / planner.SPOOL).is_dir() else 0, "base_mismatch": 0}
    if not (home / LEDGER).exists():
        report["ledger"] = "missing"
        report["ok"] = report["pending"] == 0 and report["failed_records"] == 0
        return report
    with closing(sqlite3.connect(f"{(home / LEDGER).as_uri()}?mode=ro", uri=True)) as ledger, \
            closing(sqlite3.connect(f"{(home / 'lcm.db').as_uri()}?mode=ro", uri=True)) as store:
        for session, value in ledger.execute("SELECT session, value FROM views"):
            view = json.loads(value)
            report["sessions"] += 1
            positions = {row[0] for row in ledger.execute(
                "SELECT position FROM events WHERE session = ? AND generation = ?", (session, view["generation"]))}
            report["positions"] += len(view["digests"])
            report["missing_positions"] += sum(1 for index in range(len(view["digests"])) if index not in positions)
            fed = ledger.execute("SELECT COALESCE(SUM(fed_rows), 0) FROM events WHERE session = ?",
                                 (session,)).fetchone()[0]
            stored = store.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session,)).fetchone()[0]
            report["fed_rows"] += fed
            report["store_rows"] += stored
            if fed != stored:
                report["row_mismatch_sessions"] += 1
                report.setdefault("mismatch", []).append({"session": session, "fed": fed, "stored": stored})
        report["base_mismatch"] = ledger.execute(
            "SELECT COUNT(*) FROM events WHERE notes LIKE '%base_mismatch%'").fetchone()[0]
    report["ok"] = not any(report[name] for name in (
        "missing_positions", "row_mismatch_sessions", "pending", "failed_records", "base_mismatch"))
    return report


# -- stdio JSON-RPC ---------------------------------------------------------------------------------

def _isolate_stdio(log_path: Path) -> TextIO:
    """Keep the real stdout for protocol lines; send fds 1 and 2 (engine logs, warnings) to the log file."""
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    sys.stdout = sys.stderr = open(log_path, "a", encoding="utf-8", buffering=1)
    return protocol


def serve(home: Path, *, context_length: int, model: str, llm_bridge: bool) -> int:
    logs = home / "logs"
    logs.mkdir(mode=0o700, parents=True, exist_ok=True)
    protocol = _isolate_stdio(logs / "host.log")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    requests: "queue.Queue[Optional[dict]]" = queue.Queue()
    write_lock = threading.Lock()
    gone = threading.Event()  # Eve's process ended (stdin closed or the pipe broke)

    def ended() -> None:
        if not gone.is_set():
            gone.set()
            if bridge is not None:
                bridge.close()  # a call_llm waiting on the main thread fails now, not after its timeout
            requests.put(None)

    def write(message: dict) -> None:
        if gone.is_set():
            return
        with write_lock:
            try:
                protocol.write(json.dumps(message, ensure_ascii=False) + "\n")
                protocol.flush()
            except (BrokenPipeError, OSError, ValueError):
                # Eve is gone: no reply can reach it. The loop ends after the final drain (store only).
                threading.Thread(target=ended, daemon=True).start()

    bridge = Bridge(write) if llm_bridge else None
    if bridge is not None:
        import agent.auxiliary_client as auxiliary  # the shim LCM-X imports as Hermes's auxiliary client

        auxiliary.BRIDGE = bridge

    def read() -> None:
        for line in open(sys.stdin.fileno(), "r", encoding="utf-8", errors="replace", closefd=False):
            if line.strip():
                try:
                    message = json.loads(line)
                except ValueError:
                    requests.put({"jsonrpc": "2.0", "id": None, "method": "__parse_error__"})
                    continue
                if bridge is not None and isinstance(message, dict) and bridge.deliver(message):
                    continue  # a model answer for call_llm, which waits on the main thread
                requests.put(message)
        ended()  # stdin closed: Eve's process ended

    threading.Thread(target=read, name="stdin", daemon=True).start()

    def reply(ident: Any, result: Any = None, error: Optional[str] = None) -> None:
        if ident is None:
            return
        write({"jsonrpc": "2.0", "id": ident, **({"error": {"code": -32000, "message": error}} if error
                                                  else {"result": result})})

    lock = Lock(home)
    host: Optional[Host] = None
    next_embed = 0.0
    next_backlog = 0.0

    def start(wait_s: float = 0.0) -> Optional[Host]:
        """The engine, once this process holds the store lock. ``wait_s``: how long a request waits for a host of an
        earlier Eve process to finish and release the lock."""
        end = time.monotonic() + wait_s
        while not lock.try_acquire():
            if time.monotonic() >= end:
                return None
            time.sleep(0.5)
        started = Host(home, context_length=context_length, model=model, summarize=bridge is not None)
        logger.info("engine host serving %s (LLM bridge: %s)", home, bridge is not None)
        started.drain()  # records an earlier process left
        return started

    try:
        while True:
            if host is None:
                host = start()
            try:
                request = requests.get(timeout=IDLE_SCAN_S)
            except queue.Empty:
                if host is not None and planner.pending_records(home):
                    host.drain()  # a record another Eve process spooled
                elif host is not None and time.monotonic() >= next_backlog and host.summarize_backlog() is None:
                    next_backlog = time.monotonic() + BACKLOG_QUIET_S  # nothing over the threshold: look later
                elif host is not None and time.monotonic() >= next_embed:
                    report = host.embed_catchup()  # idle between turns: API vectors, bounded
                    next_embed = time.monotonic() + (EMBED_IDLE_S if report.get("chunks") or report.get("summaries")
                                                     else EMBED_QUIET_S)
                continue
            if request is None:
                break
            ident, method = request.get("id"), request.get("method")
            params = request.get("params") if isinstance(request.get("params"), dict) else {}
            try:
                if method == "initialize":
                    reply(ident, {"serverInfo": {"name": "lcmx-eve-host", "version": "0.1.0"},
                                  "engine": host is not None})
                elif host is None and (host := start(LOCK_WAIT_S)) is None:
                    # Another process's host still holds this store: its drain takes our spooled records.
                    reply(ident, error="the LCM-X store is held by another Eve process's host")
                elif method == "drain":
                    reply(ident, host.drain())
                elif method == "turn/started":
                    host.turn_started_at = time.monotonic()  # no idle summaries until this turn completes
                    reply(ident, {})
                elif method == "tools/list":
                    reply(ident, {"tools": host.tool_schemas()})
                elif method == "tools/call":
                    text = host.call_tool(params.get("session"), str(params.get("name")), params.get("arguments") or {})
                    reply(ident, {"content": [{"type": "text", "text": text}]})
                elif method == "ping":
                    reply(ident, {})
                else:
                    reply(ident, error=f"method not found: {method}")
            except Exception as error:  # noqa: BLE001 - one bad request never ends the host
                logger.exception("request %s failed", method)
                reply(ident, error=f"{type(error).__name__}: {error}")
        if host is not None:
            host.summarize = False  # no model after Eve ended: store only; the next live turn summarizes
            host.drain()
        return 0
    finally:
        if host is not None:
            host.close()
        lock.close()


def drain_once(home: Path, *, context_length: int, model: str) -> dict:
    lock = Lock(home)
    try:
        if not lock.try_acquire():
            return {"lock": "held"}
        host = Host(home, context_length=context_length, model=model)
        try:
            return host.drain()
        finally:
            host.close()
    finally:
        lock.close()


MIN_SQLITE = (3, 51, 3)


def check() -> dict:
    """What this runtime can do: the engine loads and lists its tools; SQLite, FastEmbed, the embedding key. LLM calls
    go through the Eve process (the agent's model), so no LLM key is checked here."""
    import importlib.util
    import tempfile

    report: dict[str, Any] = {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version,
                              "sqlite_ok": tuple(int(part) for part in sqlite3.sqlite_version.split(".")) >= MIN_SQLITE,
                              "fastembed": importlib.util.find_spec("fastembed") is not None}
    key_env = os.environ.get("LCM_EMBEDDING_API_KEY_ENV") or "OPENAI_API_KEY"
    report["embedding_key"] = bool(os.environ.get(key_env, "").strip())  # presence only, never the value
    with tempfile.TemporaryDirectory() as home:
        os.environ.setdefault("LCM_EMBEDDINGS_ENABLED", "false")
        host = Host(Path(home), context_length=200_000, model="check")
        try:
            report["tools"] = len(host.tool_schemas())
        finally:
            host.close()
    report["ok"] = report["sqlite_ok"] and report["tools"] > 0
    return report


def retime(home: Path, sessions: dict) -> dict:
    """Give the rows of imported sessions (``import-<eve id>``) their original times, one time per message in order.

    Only for sessions that ``lcmx-eve import`` made: each of their messages is one text row. A session whose row count
    differs from its list of times is left as it is and reported. Only ``messages.timestamp`` changes; content, ids,
    order and the ledger stay as they are."""
    from datetime import datetime
    report: dict[str, Any] = {"sessions": 0, "rows": 0, "skipped": {}}
    with closing(sqlite3.connect(home / "lcm.db", timeout=30)) as store:
        for session, times in sessions.items():
            if not str(session).startswith("import-") or not isinstance(times, list):
                report["skipped"][session] = "not an imported session"
                continue
            ids = [row[0] for row in store.execute(
                "SELECT store_id FROM messages WHERE session_id = ? ORDER BY store_id", (planner.session_rows_id(session),))]
            if len(ids) != len(times):
                report["skipped"][session] = f"rows {len(ids)} != times {len(times)}"
                continue
            stamps = [datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() for value in times]
            with store:
                store.executemany("UPDATE messages SET timestamp = ? WHERE store_id = ?", list(zip(stamps, ids)))
            report["sessions"] += 1
            report["rows"] += len(ids)
    report["ok"] = True
    return report


def main(argv: Optional[list[str]] = None) -> int:
    if sys.version_info < (3, 11):
        sys.stderr.write("lcmx-eve host needs Python 3.11 or newer\n")
        return 2
    parser = argparse.ArgumentParser(prog="lcmx_eve_host.py", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("serve", "drain", "audit", "check", "retime"))
    parser.add_argument("--home", default="", help="absolute store home (one Eve memory scope)")
    parser.add_argument("--core", default=os.environ.get("LCMX_PACKAGE_ROOT", ""), help="LCM-X core folder")
    parser.add_argument("--context-length", type=int, default=int(os.environ.get("LCMX_CONTEXT_LENGTH", "200000")),
                        help="the context window the engine compacts its shadow for (Hermes model context length)")
    parser.add_argument("--model", default=os.environ.get("LCMX_MODEL_NAME", "eve-agent"))
    parser.add_argument("--llm-bridge", action="store_true",
                        help="serve: LCM-X's LLM calls go to the Eve process (the agent's model) over stdio")
    options = parser.parse_args(argv)
    if options.command == "check":
        planner.bootstrap(options.core or None)
        result = check()
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
        return 0 if result["ok"] else 1
    home = Path(options.home).expanduser()
    if not home.is_absolute():
        sys.stderr.write("lcmx-eve host: --home must be absolute\n")
        return 2
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    planner.bootstrap(options.core or None)
    if options.command == "serve":
        return serve(home, context_length=options.context_length, model=options.model,
                     llm_bridge=options.llm_bridge)
    if options.command == "retime":  # {"sessions": {"import-<id>": [ISO time per message]}} on stdin
        result = retime(home, json.loads(sys.stdin.read() or "{}").get("sessions") or {})
        sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
        return 0
    result = (drain_once(home, context_length=options.context_length, model=options.model)
              if options.command == "drain" else audit(home))
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0 if options.command == "drain" or result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
