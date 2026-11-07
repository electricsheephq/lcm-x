"""Host-free transport proofs and optional pinned-host contract checks."""

import asyncio
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from bench.instruments.d5_clone.gateway import session_key

ADAPTER = Path(__file__).resolve().parents[1] / "bench/instruments/d5_clone/d5_local/adapter.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("d5_test_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter_module(monkeypatch):
    class Base:
        def __init__(self, config, platform):
            self.config, self.platform, self.seen = config, platform, []

        def build_source(self, **kwargs):
            return SimpleNamespace(platform=self.platform, **kwargs)

        async def handle_message(self, event):
            self.seen.append(event)

            async def finish():
                await asyncio.sleep(0)
                await self.send(event.source.chat_id, "synthetic reply")
                await self.on_processing_complete(event, "success")

            asyncio.create_task(finish())

        def _mark_connected(self):
            pass

        def _mark_disconnected(self):
            pass

        def end_session(self, *args, **kwargs):
            raise AssertionError("transport must never end a session")

    @dataclass
    class Event:
        text: str
        message_type: str
        source: object
        message_id: str
        allow_gateway_control: bool = True

    for name, members in {
        "gateway": {}, "gateway.platforms": {},
        "gateway.config": {"Platform": str},
        "gateway.platforms.base": {"BasePlatformAdapter": Base, "SendResult": SimpleNamespace},
        "gateway.platforms.event": {"MessageEvent": Event, "MessageType": SimpleNamespace(TEXT="text")},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    return load_adapter()


def test_verbatim_completion_and_chat_identity(adapter_module, caplog):
    async def exercise():
        adapter = adapter_module.D5LocalAdapter(SimpleNamespace(extra={}))
        text = "  synthetic marker\r\nsecond line   "
        for step in ("one", "2"):
            status, result = await adapter._turn({"chat_id": "chat-a", "step": step, "text": text, "timeout": 1})
            assert status == 200
            assert result == {"step": step, "completed": True, "outcome": "success"}
        assert adapter.replies == {"chat-a": [hashlib.sha256(b"synthetic reply").hexdigest()] * 2}
        assert [e.text for e in adapter.seen] == [text, text]
        assert all(e.message_type == "text" for e in adapter.seen)
        first, second = [vars(e.source).copy() for e in adapter.seen]
        assert first.pop("message_id") == "one"
        assert second.pop("message_id") == "2"
        assert first == second == {"platform": "d5-local", "chat_id": "chat-a", "chat_name": "chat-a",
                                   "chat_type": "dm", "user_id": "d5-user", "user_name": "d5-user"}
        assert "synthetic reply" not in json.dumps(result)
        assert text not in caplog.text
        assert "synthetic reply" not in caplog.text

    with caplog.at_level("INFO"):
        asyncio.run(exercise())
    assert "sha256=" in caplog.text


def test_turn_reports_the_hook_outcome_and_refuses_bad_input(adapter_module):
    async def exercise():
        adapter = adapter_module.D5LocalAdapter(SimpleNamespace(extra={}))

        async def fail(event):
            await adapter.on_processing_complete(event, SimpleNamespace(value="failure"))

        adapter.handle_message = fail
        body = {"chat_id": "chat-a", "step": "one", "text": "synthetic", "timeout": 1}
        assert (await adapter._turn(body))[1] == {"step": "one", "completed": True, "outcome": "failure"}
        for bad in ({"step": 1}, {"text": None}, {"chat_id": 2}):
            with pytest.raises(ValueError):
                await adapter._turn({**body, **bad})

    asyncio.run(exercise())


@pytest.mark.parametrize("text", ["/new", " \n/reset", "@file:x\n/new", "restart gateway"])
def test_replayed_text_is_never_gateway_control(adapter_module, text):
    async def exercise():
        adapter = adapter_module.D5LocalAdapter(SimpleNamespace(extra={}))
        status, result = await adapter._turn({"chat_id": "chat-a", "step": "one", "text": text, "timeout": 1})
        assert status == 200 and result["completed"]
        (event,) = adapter.seen
        assert event.text == text and event.allow_gateway_control is False

    asyncio.run(exercise())


def test_timeout_retains_message_ownership(adapter_module):
    async def exercise():
        adapter = adapter_module.D5LocalAdapter(SimpleNamespace(extra={}))

        async def hold(event):
            pass

        adapter.handle_message = hold
        body = {"chat_id": "chat-a", "step": "one", "text": "synthetic", "timeout": 0.01}
        assert (await adapter._turn(body))[1]["completed"] is False
        assert (await adapter._turn({**body, "step": "two"}))[0] == 409
        event = adapter.turns["chat-a"][0]
        await adapter.on_processing_complete(SimpleNamespace(source=event.source), "success")
        assert not adapter.turns["chat-a"][1].is_set()
        await adapter.on_processing_complete(event, "success")
        assert adapter.turns["chat-a"][1].is_set()

    asyncio.run(exercise())


def test_http_framing_without_socket(adapter_module):
    async def exercise():
        adapter = adapter_module.D5LocalAdapter(SimpleNamespace(extra={}))

        class Writer:
            def write(self, data):
                self.data = data

            async def drain(self):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                pass

        for method, path, body, expected in (
            ("GET", "/health", b"", {"connected": True}),
            ("POST", "/turn", json.dumps({"chat_id": "chat-a", "step": "1", "text": "verbatim\r\n  "}).encode(), None),
        ):
            reader, writer = asyncio.StreamReader(), Writer()
            reader.feed_data(f"{method} {path} HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
            reader.feed_eof()
            await adapter._http(reader, writer)
            header, payload = writer.data.split(b"\r\n\r\n", 1)
            assert header.startswith(b"HTTP/1.1 200 ")
            assert f"Content-Length: {len(payload)}".encode() in header
            if expected:
                assert json.loads(payload) == expected
            else:
                assert json.loads(payload)["completed"] is True
                assert adapter.seen[0].text == "verbatim\r\n  "

    asyncio.run(exercise())


@pytest.mark.parametrize("from_env", [False, True])
def test_http_and_port_file(adapter_module, tmp_path, monkeypatch, from_env):
    port_file = tmp_path / "port"
    monkeypatch.setenv("D5_LOCAL_PORT_FILE", str(port_file))

    async def exercise():
        adapter = adapter_module.D5LocalAdapter(SimpleNamespace(extra={} if from_env else {"port_file": str(port_file)}))
        assert await adapter.connect()
        try:
            port = int(port_file.read_text())
            assert adapter.server.sockets[0].getsockname() == ("127.0.0.1", port)

            async def request(method, path, body=b""):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
                await writer.drain()
                response = await reader.read()
                writer.close()
                await writer.wait_closed()
                return json.loads(response.split(b"\r\n\r\n", 1)[1])

            assert await request("GET", "/health") == {"connected": True}
            await adapter.send("chat-b", "outside any turn")
            body = {"chat_id": "chat-a", "step": "one", "text": "verbatim\r\n  ", "timeout": 1}
            result = await request("POST", "/turn", json.dumps(body).encode())
            assert result["completed"] is True and result["outcome"] == "success"
            for chat, text in (("chat-a", "synthetic reply"), ("chat-b", "outside any turn")):
                assert await request("GET", f"/replies?chat_id={chat}") == {
                    "chat_id": chat, "count": 1, "reply_sha256": [hashlib.sha256(text.encode()).hexdigest()]}
            assert adapter.seen[0].text == body["text"]
        finally:
            await adapter.disconnect()

    asyncio.run(exercise())


@pytest.fixture
def real_host(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(os.environ["D5_HOST_SRC"])
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    return load_adapter()


@pytest.mark.skipif(not os.environ.get("D5_HOST_SRC"), reason="D5_HOST_SRC not set")
def test_real_host_register_platform(real_host):
    from gateway.platform_registry import PlatformEntry
    from hermes_cli.plugins import PluginContext

    captured = {}

    class Context:
        def register_platform(self, **kwargs):
            inspect.signature(PluginContext.register_platform).bind(self, **kwargs)
            inspect.signature(PlatformEntry).bind(**kwargs)
            captured.update(kwargs)

    real_host.register(Context())
    assert captured["name"] == "d5-local"
    assert captured["trusted_inbound"] and captured["pii_safe"]
    assert captured["platform_hint"] == "" and captured["required_env"] == []


@pytest.mark.skipif(not os.environ.get("D5_HOST_SRC"), reason="D5_HOST_SRC not set")
def test_real_host_session_key(real_host):
    from gateway.config import Platform
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.session import SessionSource, build_session_key

    scope = platform_registry.current_scope_key()
    previous = platform_registry.snapshot_registration("d5-local", scope=scope)
    platform_registry.register(PlatformEntry(name="d5-local", label="d5-local",
                                            adapter_factory=real_host.D5LocalAdapter, check_fn=lambda: True), scope=scope)
    current = platform_registry.snapshot_registration("d5-local", scope=scope)
    try:
        source = SessionSource(platform=Platform("d5-local"), chat_id="chat-a", chat_type="dm",
                               user_id="d5-user", user_name="d5-user")
        assert build_session_key(source) == session_key("chat-a")
    finally:
        platform_registry.restore_registration("d5-local", current, previous, scope=scope)


@pytest.mark.skipif(not os.environ.get("D5_HOST_SRC"), reason="D5_HOST_SRC not set")
@pytest.mark.parametrize("text", ["/new", "@file:x\n/new", "restart gateway"])
def test_real_host_never_reads_replayed_text_as_a_command(real_host, text):
    from gateway.platforms.base import coerce_plaintext_gateway_command

    async def exercise():
        adapter = real_host.D5LocalAdapter(SimpleNamespace(extra={}))
        seen = []

        async def handle(event):
            seen.append(event)
            await adapter.on_processing_complete(event, "success")

        adapter.handle_message = handle
        await adapter._turn({"chat_id": "chat-a", "step": "one", "text": text, "timeout": 1})
        (event,) = seen
        assert not event.is_command() and event.get_command() is None
        if event.allow_gateway_control:  # The host's own gate in handle_message.
            coerce_plaintext_gateway_command(event)
        assert event.text == text

    asyncio.run(exercise())
