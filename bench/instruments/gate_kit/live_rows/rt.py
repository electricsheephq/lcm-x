"""Process setup shared by the live-rows runners: isolation, the host stand-in, one tree, one engine.

Call `isolate` before anything imports a tree. One tree per process (busy-bar's load-by-path registration).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench.instruments.busy_bar import driver  # noqa: E402
from bench.instruments.gate_kit.live_rows import host  # noqa: E402
from bench.instruments.gate_kit.live_rows.host import auxiliary_client as aux  # noqa: E402

KIT = Path(__file__).resolve().parent / "kit.json"
PROFILE: dict[str, str] = {}  # run.py's diagnostic profile path (shared by `run` and `cells`)
CONTROL: dict[str, str] = {}  # run.py's instrument self-test mode (shared by `run` and `cells`)


class CellError(Exception):
    """A cell that could not reach its pass; run.main writes a receipt naming the step and the class. Defined here
    (one module) because under `python -m ...run` the runner is `__main__` and `cells` imports a second copy."""

    def __init__(self, step: str, error: BaseException) -> None:
        super().__init__(f"{step}: {type(error).__name__}")
        self.step, self.error_class = step, type(error).__name__


def load_kit(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or KIT).read_text())


def ro_uri(db: Path) -> str:
    """A read-only SQLite URI; `as_uri()` escapes '?' and '#' so they cannot cut off `mode=ro`."""
    return Path(db).resolve().as_uri() + "?mode=ro"


def plain_dir(path: Path) -> Path:
    """Refuse a working path that is, passes through, or holds (current/, home/) a symlink: every write stays in the
    synthetic workspace."""
    path = Path(path).absolute()
    for part in [path, *path.parents, path / "current", path / "home"]:
        if part.is_symlink():
            raise SystemExit(f"refusing a symlinked working path: {part.name}")
    return path


def isolate(home: Path, env: dict[str, str] | None = None) -> None:
    """Drop every LCM_ and HERMES_ variable, point HOME and HERMES_HOME at an empty folder, apply the arm (LCM_
    variables only), and refuse the network before anything imports a tree."""
    env = dict(env or {})
    bad = [key for key in env if not key.startswith("LCM_")]
    if bad:
        raise SystemExit(f"an arm or budget may set LCM_ variables only, not {bad}")
    for key in list(os.environ):
        if key.startswith(("LCM_", "HERMES_")):
            del os.environ[key]
    home = plain_dir(home)
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = os.environ["HERMES_HOME"] = str(home)
    os.environ.update(env)
    deny_network()


def deny_network() -> None:
    """Refuse name resolution and every connect or send, so nothing leaves the host and no resolver delay is timed."""
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError("live-rows is offline: network access forbidden")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.socket.sendto = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]
    socket.gethostbyname = refuse  # type: ignore[assignment]
    socket.gethostbyname_ex = refuse  # type: ignore[assignment]
    socket.gethostbyaddr = refuse  # type: ignore[assignment]
    socket.getnameinfo = refuse  # type: ignore[assignment]


class Lines(logging.Handler):
    """Fixed-text LCM-X log lines with their thread and monotonic time (messages are product text, no content)."""

    PREFIXES = ("LCM compaction stop", "LCM automatic compaction held", "LCM compression no-op", "LCM threshold sweep",
                "LCM summary route", "LLM summarization failed", "LCM compaction #", "LCM store-complete leaf",
                "LCM leaf summary input", "LCM survival fit", "LCM ledger ordinals respace", "LCM compress failed")

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if message.startswith(self.PREFIXES):
            self.lines.append({"t": time.monotonic(), "thread": record.thread, "level": record.levelname,
                               "message": message[:300]})


def load(tree: Path, kit: dict[str, Any]) -> tuple[Any, Any, Lines]:
    host.install()
    aux.configure(**{k: v for k, v in kit["summary_route"].items() if k in aux.SETTINGS})
    deny_network()  # again, in case a caller skipped `isolate`
    config_module, engine_module = driver.load_tree(tree.resolve())
    lines = Lines()
    logger = logging.getLogger("hermes_lcm")
    logger.addHandler(lines)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return config_module, engine_module, lines


def config_for(config_module: Any, db: Path) -> Any:
    config = config_module.LCMConfig.from_env()
    config.database_path = str(db)
    config.embeddings_enabled = False
    config.large_output_externalization_path = str(db.parent / "externalized")
    return config


def effective(config: Any) -> dict[str, Any]:
    names = ("foreground_soft_seconds", "foreground_hard_seconds", "summary_timeout_ms", "leaf_chunk_tokens",
             "fresh_tail_count", "context_threshold", "threshold_full_sweep_enabled")
    found = {name: getattr(config, name) for name in names if hasattr(config, name)}
    found["env"] = {key: value for key, value in sorted(os.environ.items()) if key.startswith("LCM_")}
    return found


def engine(engine_module: Any, config: Any, kit: dict[str, Any], home: Path, session: str | None = None,
           conversation: str | None = None) -> Any:
    target = kit["target"]
    built = engine_module.LCMEngine(config=config, hermes_home=str(home))
    built.on_session_start(session or target["session_id"], platform=target["platform"],
                           context_length=target["context_length"],
                           conversation_id=conversation or target["conversation_id"])
    return built


def store_census(db: Path) -> dict[str, Any]:
    """Counts, ids and a digest of every stored row's identity and order (read-only)."""
    with closing(sqlite3.connect(ro_uri(db), uri=True)) as conn:
        rows = conn.execute("SELECT store_id, conversation_id, session_id, role, length(content) "
                            "FROM messages ORDER BY store_id").fetchall()
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return {"rows": len(rows), "max_store_id": rows[-1][0] if rows else 0,
            "conversations": len({row[1] for row in rows}), "rows_digest": digest}


IDENTITY_COLUMNS = ("conversation_id", "session_id", "source", "role", "content", "tool_call_id", "tool_calls",
                    "tool_name", "timestamp", "sender_id", "sender_source")  # owner, attribution and value


def stored_identity(db: Path) -> list[tuple[int, bytes]]:
    """Every stored row of the whole store, in store order: its id and a digest of its owner and value."""
    found = []
    with closing(sqlite3.connect(ro_uri(db), uri=True)) as conn:
        names = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        columns = [c for c in IDENTITY_COLUMNS if c in names]
        for row in conn.execute(f"SELECT store_id, {', '.join(columns)} FROM messages ORDER BY store_id"):
            found.append((row[0], hashlib.blake2b(repr(row[1:]).encode(), digest_size=16).digest()))
    payloads = Path(db).parent / "externalized"  # an externalized row's content is a reference: its file counts too
    if payloads.exists():
        for path in sorted(p for p in payloads.rglob("*") if p.is_file()):
            found.append((f"payload:{path.relative_to(payloads)}",
                          hashlib.blake2b(path.read_bytes(), digest_size=16).digest()))
    return found


def stored_rows(db: Path, conversation: str) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(ro_uri(db), uri=True)) as conn:
        return conn.execute("SELECT store_id, session_id, role, content, tool_call_id FROM messages WHERE conversation_id = ? "
                            "ORDER BY store_id", (conversation,)).fetchall()


def open_reader(db: Path) -> sqlite3.Connection:
    """An idle read-only connection: it makes SQLite create the WAL shared-memory file and holds no write lock."""
    conn = sqlite3.connect(ro_uri(db), uri=True)
    conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    return conn


def host_state_db(home: Path, sessions: list[str]) -> Path:
    """The host's session store with one root row per session (no parent, not ended), so the product can resolve
    each session's lineage and write host-uid bindings, as on a real host. Its path is part of the lineage key."""
    path = home / "state.db"
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, source TEXT, "
                         "parent_session_id TEXT, end_reason TEXT, started_at REAL)")
            conn.executemany("INSERT OR IGNORE INTO sessions (id, source, started_at) VALUES (?, 'live-rows', 0)",
                             [(sid,) for sid in sessions])
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
