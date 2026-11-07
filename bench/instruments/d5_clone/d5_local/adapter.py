"""Verbatim inbound turns and content-free delivery receipts on loopback."""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger(__name__)


class D5LocalAdapter(BasePlatformAdapter):
    def __init__(self, config):
        super().__init__(config=config, platform=Platform("d5-local"))
        self.server, self.counter, self.turns = None, 0, {}

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

    async def on_processing_complete(self, event, outcome):
        turn = self.turns.get(event.source.chat_id)
        if turn and turn[0] is event:
            turn[1].set()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.counter += 1
        if chat_id in self.turns:
            self.turns[chat_id][2].put_nowait(content)
        logger.info("reply count=%d sha256=%s", self.counter, hashlib.sha256(content.encode()).hexdigest())
        return SendResult(success=True, message_id=str(self.counter))

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}

    async def _turn(self, body):
        chat, step, text = body["chat_id"], body["step"], body["text"]
        if not all(isinstance(v, str) for v in (chat, text)) or not isinstance(step, (str, int)):
            raise ValueError("invalid turn")
        if chat in self.turns and not self.turns[chat][1].is_set():
            return 409, {"completed": False}
        source = self.build_source(chat_id=chat, chat_name=chat, chat_type="dm", user_id="d5-user",
                                   user_name="d5-user", message_id=step)
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=step)
        done, replies = asyncio.Event(), asyncio.Queue()
        self.turns[chat] = event, done, replies
        try:
            async with asyncio.timeout(float(body.get("timeout", 600))):
                await self.handle_message(event)
                await done.wait()
        except TimeoutError:
            pass
        hashes = [hashlib.sha256(replies.get_nowait().encode()).hexdigest() for _ in range(replies.qsize())]
        return 200, {"step": step, "completed": done.is_set(), "replies": len(hashes), "reply_sha256": hashes}

    async def _http(self, reader, writer):
        status, result = 400, {"error": "invalid request"}
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            lines = header.decode("ascii").split("\r\n")
            method, path, _ = lines[0].split()
            headers = dict(line.split(":", 1) for line in lines[1:] if line)
            headers = {k.lower(): v.strip() for k, v in headers.items()}
            if method == "GET" and path == "/health":
                status, result = 200, {"connected": True}
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
