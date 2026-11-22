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


def load_kit(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or KIT).read_text())


def isolate(home: Path, env: dict[str, str] | None = None) -> None:
    """Drop every LCM_ and HERMES_ variable, point HOME and HERMES_HOME at an empty folder, apply the arm."""
    for key in list(os.environ):
        if key.startswith(("LCM_", "HERMES_")):
            del os.environ[key]
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = os.environ["HERMES_HOME"] = str(home)
    os.environ.update(env or {})


def deny_network() -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("live-rows is offline: network access forbidden")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]


class Lines(logging.Handler):
    """Fixed-text LCM-X log lines with their thread and monotonic time (messages are product text, no content)."""

    PREFIXES = ("LCM compaction stop", "LCM automatic compaction held", "LCM compression no-op", "LCM threshold sweep",
                "LCM summary route", "LLM summarization failed", "LCM compaction #", "LCM store-complete leaf",
                "LCM leaf summary input", "LCM survival fit", "LCM ledger ordinals respace")

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
    config_module, engine_module = driver.load_tree(tree.resolve())
    deny_network()
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
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        rows = conn.execute("SELECT store_id, conversation_id, session_id, role, length(content) "
                            "FROM messages ORDER BY store_id").fetchall()
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return {"rows": len(rows), "max_store_id": rows[-1][0] if rows else 0,
            "conversations": len({row[1] for row in rows}), "rows_digest": digest}


def stored_rows(db: Path, conversation: str) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        return conn.execute("SELECT store_id, session_id, role, content FROM messages WHERE conversation_id = ? "
                            "ORDER BY store_id", (conversation,)).fetchall()


def open_reader(db: Path) -> sqlite3.Connection:
    """An idle read-only connection: it makes SQLite create the WAL shared-memory file and holds no write lock."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
    return conn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
