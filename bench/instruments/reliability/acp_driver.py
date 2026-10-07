"""A stdlib JSON-RPC client for ``hermes acp`` over stdio (newline-delimited frames, read with selectors).

Ported from the WS3 gauntlet ``drive_hermes_acp_rc4.py`` (StdioJsonRpc: framing, the server-request refusal,
process-group lifecycle) plus ``--kill-after-rotation`` from ``drive_hermes_acp_519.py`` (``RotationKiller``). All
live-model and reference-agent coupling is gone: no auth.json, no contamination guard, no session model, no canaries. The
caller supplies argv/env/cwd (an isolated HERMES_HOME/HOME per cell) and owns the process: ``kill()`` SIGKILLs
the process group this client spawned (``start_new_session``), never anything else.
"""
from __future__ import annotations

import contextlib
import json
import os
import selectors
import signal
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

SHUTDOWN_SECONDS = 5.0


class DriverError(RuntimeError):
    """An ACP protocol or process failure."""


class ProcessGone(DriverError):
    """The host process closed stdout (it exited or was killed)."""


class JsonRpcError(DriverError):
    def __init__(self, method: str, error: Any) -> None:
        self.method, self.error = method, error
        super().__init__(f"{method} returned JSON-RPC error: {json.dumps(error, sort_keys=True)[:300]}")


def frame(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def parse_frames(buffer: bytearray) -> tuple[list[dict], bytearray]:
    """Split complete newline-terminated frames off ``buffer``; blank lines are skipped, a bad frame raises."""
    out = []
    while b"\n" in buffer:
        line, _, rest = buffer.partition(b"\n")
        buffer = bytearray(rest)
        if not line.strip():
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DriverError(f"invalid JSON-RPC frame: {bytes(line[:200])!r}") from exc
        if not isinstance(msg, dict):
            raise DriverError(f"invalid JSON-RPC object: {msg!r}"[:300])
        out.append(msg)
    return out, buffer


def agent_text(msg: dict) -> str | None:
    """The text of a ``session/update`` agent_message_chunk notification, else None."""
    update = (msg.get("params") or {}).get("update") if msg.get("method") == "session/update" else None
    if not isinstance(update, dict) or update.get("sessionUpdate", update.get("session_update")) != "agent_message_chunk":
        return None
    content = update.get("content")
    return content.get("text") if isinstance(content, dict) and isinstance(content.get("text"), str) else None


class AcpProcess:
    """One ``hermes acp`` process and a synchronous client for it. ``notify`` may be called from another thread."""

    def __init__(self, argv: list[str], env: dict, cwd: Path, stderr_path: Path) -> None:
        self.stderr_fh = open(stderr_path, "ab")
        self.process = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr_fh, bufsize=0, start_new_session=True)
        self.pid = self.process.pid
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer, self.pending = bytearray(), []
        self.next_id, self.write_lock, self.killed = 1, threading.Lock(), False
        self.sent_term = False  # close() sent SIGTERM to a still-running host (a -SIGTERM exit is then ours)

    def _write(self, payload: dict) -> None:
        data = frame(payload)
        with self.write_lock:
            try:
                while data:
                    data = data[self.process.stdin.write(data):]
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise ProcessGone(f"ACP stdin write failed: {exc}") from exc

    def notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _read(self, deadline: float) -> dict:
        while not self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise DriverError("ACP response timeout")
            chunk = os.read(self.process.stdout.fileno(), 64 * 1024)
            if not chunk:
                raise ProcessGone(f"ACP stdout closed (returncode={self.process.poll()})")
            self.buffer.extend(chunk)
            msgs, self.buffer = parse_frames(self.buffer)
            self.pending.extend(msgs)
        return self.pending.pop(0)

    def request(self, method: str, params: dict, timeout: float, chunks: list | None = None) -> dict | None:
        rid, self.next_id = self.next_id, self.next_id + 1
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            msg = self._read(deadline)
            if msg.get("method"):
                if "id" in msg:  # a server->client request (permission, fs): this client has no handler
                    self._write({"jsonrpc": "2.0", "id": msg["id"],
                                 "error": {"code": -32601, "message": "reliability driver has no client handler"}})
                elif chunks is not None and (text := agent_text(msg)) is not None:
                    chunks.append(text)
                continue
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                raise JsonRpcError(method, msg["error"])
            return msg.get("result") if isinstance(msg.get("result"), dict) else None

    def initialize(self, timeout: float) -> dict | None:
        return self.request("initialize", {"protocolVersion": 1, "clientCapabilities": {},
                                           "clientInfo": {"name": "lcmx-reliability-r2", "version": "0.1.0"}}, timeout)

    def new_session(self, cwd: Path, timeout: float) -> str:
        result = self.request("session/new", {"cwd": str(cwd), "mcpServers": []}, timeout)
        if not result or not isinstance(result.get("sessionId"), str):
            raise DriverError(f"session/new returned no sessionId: {result!r}"[:300])
        return result["sessionId"]

    def load_session(self, session_id: str, cwd: Path, timeout: float) -> None:
        self.request("session/load", {"cwd": str(cwd), "sessionId": session_id, "mcpServers": []}, timeout)

    def prompt(self, session_id: str, text: str, timeout: float) -> tuple[str, str | None]:
        chunks: list[str] = []
        result = self.request("session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
                              timeout, chunks)
        return "".join(chunks), (result or {}).get("stopReason")

    def kill(self) -> None:
        """SIGKILL this client's own process group (its leader is the PID this client spawned)."""
        self.killed = True
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.pid, signal.SIGKILL)

    def close(self) -> int | None:
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.pid, signal.SIGTERM)
                self.sent_term = True
            try:
                self.process.wait(timeout=SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                self.kill()
                self.process.wait(timeout=SHUTDOWN_SECONDS)
        with contextlib.suppress(ProcessLookupError, PermissionError):  # a bridge child that outlived its leader
            os.killpg(self.pid, signal.SIGKILL)
        self.selector.close()
        with contextlib.suppress(OSError):
            self.process.stdout.close()
        self.stderr_fh.close()
        return self.process.returncode


def rotation_probe(db_path: Path, conversation_id: str) -> tuple[str | None, int]:
    """Read-only: the conversation's lifecycle current_session_id and its lcm message row count."""
    if not db_path.is_file():
        return None, 0
    try:
        with contextlib.closing(sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True, timeout=1)) as conn:
            row = conn.execute("SELECT current_session_id FROM lcm_lifecycle_state WHERE conversation_id = ?",
                               (conversation_id,)).fetchone()
            if row is None or not row[0]:
                return None, 0
            return str(row[0]), int(conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (row[0],)).fetchone()[0])
    except sqlite3.Error:
        return None, 0


class RotationKiller(threading.Thread):
    """``--kill-after-rotation``: poll (20 ms) for a rotation; SIGKILL the ACP group while the child has 0 lcm rows."""

    def __init__(self, client: AcpProcess, db_path: Path, conversation_id: str, on_kill=None) -> None:
        super().__init__(name="rotation-killer", daemon=True)
        self.client, self.db_path, self.conversation_id, self.on_kill = client, db_path, conversation_id, on_kill
        self.baseline, _ = rotation_probe(db_path, conversation_id)
        self.stop = threading.Event()
        self.event: dict | None = None
        self.missed = 0

    def check(self, kill: bool) -> None:
        current, rows = rotation_probe(self.db_path, self.conversation_id)
        if self.event is None and self.baseline and current and current != self.baseline:
            if kill and rows > 0:  # the poll lost the race for this rotation: wait for the next one
                self.baseline, self.missed = current, self.missed + 1
                return
            self.event = {"parent_session_id": self.baseline, "child_session_id": current, "child_rows_at_detect": rows,
                          "missed_rotations": self.missed, "killed": kill}
            if kill:
                if self.on_kill:
                    self.on_kill(self.event)
                self.client.kill()

    def run(self) -> None:
        while not self.stop.is_set() and self.event is None:
            self.check(kill=True)
            self.stop.wait(0.02)
