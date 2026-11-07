"""Verbatim inbound turns and content-free delivery receipts on loopback."""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger(__name__)


class D5LocalAdapter(BasePlatformAdapter):
    def __init__(self, config):
        super().__init__(config=config, platform=Platform("d5-local"))
        # turns: chat -> [event, done, outcome]; replies: chat -> every reply sha256 of the run, in send order.
        self.server, self.counter, self.turns, self.replies = None, 0, {}, {}

    async def connect(self, *, is_reconnect=False):
        self.server = await asyncio.start_server(self._http, "127.0.0.1", 0)
        port_file = (self.config.extra or {}).get("port_file") or os.environ["D5_LOCAL_PORT_FILE"]
        Path(port_file).write_text(str(self.server.sockets[0].getsockname()[1]))
        self._mark_connected()
        return True

    async def disconnect(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self._mark_disconnected()

    def _host_restoring(self):
        """The host's startup-restore gate: True while it queues every inbound turn (and later reports it complete
        without running the agent), False once it opens, None when the gate cannot be found. The pinned host keeps
        it on GatewayRunner (``_startup_restore_in_progress``), reached through the message handler it gave us."""
        seen, todo = set(), [getattr(self, "_message_handler", None)]
        while todo and len(seen) < 64:
            obj = todo.pop()
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(type(obj), "_startup_restore_in_progress"):
                return bool(obj._startup_restore_in_progress)
            todo += [getattr(obj, "__self__", None), getattr(obj, "__wrapped__", None)]
            for cell in getattr(obj, "__closure__", None) or ():
                try:
                    todo.append(cell.cell_contents)
                except ValueError:  # An empty cell.
                    pass
        return None

    async def on_processing_complete(self, event, outcome):
        turn = self.turns.get(event.source.chat_id)
        if turn and turn[0] is event:
            turn[2] = getattr(outcome, "value", outcome)
            turn[1].set()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.counter += 1
        digest = hashlib.sha256(content.encode()).hexdigest()
        self.replies.setdefault(chat_id, []).append(digest)
        logger.info("reply count=%d sha256=%s", self.counter, digest)
        return SendResult(success=True, message_id=str(self.counter))

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}

    async def _turn(self, body):
        chat, step, text = body["chat_id"], body["step"], body["text"]
        if not all(isinstance(v, str) for v in (chat, step, text)):
            raise ValueError("invalid turn")
        if chat in self.turns and not self.turns[chat][1].is_set():
            return 409, {"completed": False}
        source = self.build_source(chat_id=chat, chat_name=chat, chat_type="dm", user_id="d5-user",
                                   user_name="d5-user", message_id=step)
        # Replayed text is user text, never gateway control: no slash command, no plain-text restart, and no
        # approval or clarify answer, whatever the text holds after the host strips its media prefixes.
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=step,
                             allow_gateway_control=False)
        done = asyncio.Event()
        self.turns[chat] = turn = [event, done, None]
        try:
            async with asyncio.timeout(float(body.get("timeout", 600))):
                await self.handle_message(event)
                await done.wait()
        except TimeoutError:
            pass
        return 200, {"step": step, "completed": done.is_set(), "outcome": turn[2]}

    async def _http(self, reader, writer):
        status, result = 400, {"error": "invalid request"}
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode("ascii").split("\r\n")
            method, path, _ = lines[0].split()
            headers = dict(line.split(":", 1) for line in lines[1:] if line)
            headers = {k.lower(): v.strip() for k, v in headers.items()}
            url = urlsplit(path)
            if method == "GET" and path == "/health":
                status, result = 200, {"connected": True, "restoring": self._host_restoring()}
            elif method == "GET" and url.path == "/replies":
                chat = parse_qs(url.query, strict_parsing=True)["chat_id"][0]
                hashes = list(self.replies.get(chat, ()))
                status, result = 200, {"chat_id": chat, "count": len(hashes), "reply_sha256": hashes}
            elif method == "POST" and path == "/turn":
                body = await reader.readexactly(int(headers["content-length"]))
                status, result = await self._turn(json.loads(body))
            else:
                status, result = 404, {"error": "not found"}
        except (ValueError, KeyError, TypeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            pass
        finally:
            payload = json.dumps(result).encode()
            writer.write(f"HTTP/1.1 {status} Response\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode() + payload)
            await writer.drain()
            writer.close()
            await writer.wait_closed()


def register(ctx):
    ctx.register_platform(name="d5-local", label="d5-local", adapter_factory=D5LocalAdapter,
                          check_fn=lambda: True, validate_config=lambda cfg: True,
                          is_connected=lambda cfg: True, required_env=[], trusted_inbound=True,
                          pii_safe=True, platform_hint="")
