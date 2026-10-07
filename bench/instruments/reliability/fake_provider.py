"""A deterministic localhost model provider for the R2 process cells (stdlib ``ThreadingHTTPServer`` on 127.0.0.1:0).

Endpoints: OpenAI ``POST /v1/chat/completions`` (JSON, or SSE when ``stream`` is true: one role/content delta chunk,
tool-call deltas, a finish chunk, a usage chunk when ``stream_options.include_usage``, then ``[DONE]``, the
openai-python chunk shape the host's ``chat.completions.create(stream=True)`` path reads), ``GET /v1/models``, and
Anthropic ``POST /v1/messages`` (JSON, or the SSE event sequence message_start .. message_stop).

The role is the requested model name (``rel/<role>`` or ``<role>``): ``main`` -> the cell scenario callback,
``lcm-summary`` -> a tag-preserving summary in the LCM integrity envelope, ``aux`` / ``aux-title`` -> the host
compression summary / a title. Usage = ((characters of every received message) // 4 + 800) * ``usage_scale``.

A reply is a dict: ``content`` or ``tool_calls`` ([{id, name, arguments}]), and optionally one fault:
``status`` (an HTTP error such as 429/500), ``hold`` (seconds: ``slow``) or ``hold_until_killed`` (the request is
held until its client socket closes, i.e. the host process died, or ``hold_cap`` seconds pass). Every request is
logged to ``log_path`` (jsonl: api, role, model, stream, messages sha256, count, token estimate, fault fired): the
ground truth of what the host actually sent.
"""
from __future__ import annotations

import hashlib
import json
import re
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bench.instruments.reliability.scorers.continuity import markers

TAG = re.compile(r"\[([A-Z]\d{2,3})\] user")
NONCE = re.compile(r'<lcm-summary nonce="([0-9a-f]+)">')


ROLES = ("main", "lcm-summary", "aux", "aux-title")


# The exact endpoints the fake serves; any other method/path is unrouted (404, and the cell ERRORs). The /api/v1 and
# /anthropic/api/v1 catalog paths are what the hosts actually request (provider-requests.jsonl, R2a.1 runs).
MODEL_PATHS = frozenset({"/v1/models", "/api/v1/models", "/anthropic/api/v1/models"})
RETRIEVE_PATHS = frozenset(f"/v1/models/rel/{r}" for r in ROLES)
COMPLETION_PATHS = {"/v1/chat/completions": "openai", "/v1/messages": "anthropic", "/anthropic/v1/messages": "anthropic"}


def route_of(method: str, path: str) -> str:
    """Exact-match classification, shared by the provider and process_cell.accounting()."""
    if method == "GET" and (path in MODEL_PATHS or path in RETRIEVE_PATHS):
        return "models"
    if method == "POST" and path in COMPLETION_PATHS:
        return "completion"
    return "unrouted"


def role_of(model: str) -> str:
    return (model or "").rsplit("/", 1)[-1]


def text_of(content) -> str:
    """Flatten OpenAI/Anthropic content (a string or a list of blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(text_of(b.get("text") if b.get("type") == "text" else b.get("content"))
                       if isinstance(b, dict) else str(b) for b in content)
    return "" if content is None else json.dumps(content)


def normalize(body: dict, api: str) -> list[dict]:
    """The request as OpenAI-shaped messages: role, content (text), tool_call_id for tool results."""
    if api == "openai":
        return [{"role": m.get("role"), "content": text_of(m.get("content")) if m.get("role") != "tool" else
                 (m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content"))),
                 "tool_call_id": m.get("tool_call_id")} for m in body.get("messages") or []]
    out = [{"role": "system", "content": text_of(body.get("system"))}] if body.get("system") else []
    for m in body.get("messages") or []:
        blocks = m.get("content") if isinstance(m.get("content"), list) else [{"type": "text", "text": m.get("content")}]
        for b in blocks:
            if b.get("type") == "tool_result":
                c = b.get("content")
                out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                            "content": c if isinstance(c, str) else text_of(c)})
        text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
        if text or m.get("role") == "assistant":
            out.append({"role": m.get("role"), "content": text, "tool_call_id": None})
    return out


def usage_for(messages: list[dict], scale: float) -> int:
    return int((sum(len(m.get("content") or "") for m in messages) // 4 + 800) * scale)


def lcm_summary(messages: list[dict], n: int) -> str:
    """Tag-preserving: ``U<tag>`` never collides with a ``[Tnn]`` user tag; inside the nonce envelope if one was asked."""
    joined = "\n".join(m.get("content") or "" for m in messages)
    tags = sorted(set(TAG.findall(joined)))
    body = (f"Stub summary #{n} covers " + " ".join("U" + t for t in tags) + ". The scripted reliability run "
            "exchanged numbered user turns and short acknowledgements.\nExpand for details about: stub")
    nonce = NONCE.search(joined)
    return f'<lcm-summary nonce="{nonce.group(1)}">\n{body}\n</lcm-summary>' if nonce else body


class FakeProvider:
    def __init__(self, log_path, main=None, usage_scale: float = 1.0, hold_cap: float = 120.0, continuity_context=None):
        self.log_path, self.main, self.usage_scale, self.hold_cap = log_path, main, usage_scale, hold_cap
        self.continuity_context = continuity_context
        self.lock, self.n_summary, self.n_requests, self.in_flight = threading.Lock(), 0, 0, threading.Event()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def do_GET(self):
                route = route_of("GET", self.path)
                if route == "models" and self.path in MODEL_PATHS:
                    provider.log(rid=provider.next_rid(), ts=time.time(), method="GET", path=self.path, route="models")
                    # context_length: a /models key the host reads (agent/model_metadata.py); 1M >= every cell window,
                    # so the host never auto-lowers the main threshold to an unknown aux model's default window.
                    data = [{"id": f"rel/{r}", "object": "model", "owned_by": "rel", "context_length": 1_000_000}
                            for r in ROLES]
                    return self.send_json(200, {"object": "list", "data": data})
                if route == "models":  # a served role's retrieve-model read: logged, answered 404 as it always was
                    provider.log(rid=provider.next_rid(), ts=time.time(), method="GET", path=self.path, route="models",
                                 status=404)
                    return self.send_json(404, {"error": {"message": f"no model detail at {self.path}"}})
                self.unrouted()

            def do_POST(self):
                if route_of("POST", self.path) != "completion":
                    return self.unrouted()
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                return provider.serve(self, body, COMPLETION_PATHS[self.path])

            def unrouted(self):
                """Every other method/path is logged (the accounting makes it an ERROR) and refused."""
                provider.log(rid=provider.next_rid(), ts=time.time(), method=self.command, path=self.path, route="unrouted")
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_json(404, {"error": {"message": f"no route {self.command} {self.path}"}})
            do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = unrouted

            def send_json(self, status, payload):
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, name="fake-provider", daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self) -> "FakeProvider":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def next_rid(self) -> int:
        with self.lock:
            self.n_requests += 1
            return self.n_requests

    def log(self, **rec) -> None:
        with self.lock, open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def reply_for(self, role: str, messages: list[dict]) -> dict:
        if role == "main":
            return self.main(messages) if self.main else {"content": "ok"}
        if role == "lcm-summary":
            with self.lock:
                self.n_summary += 1
                n = self.n_summary
            return {"content": lcm_summary(messages, n)}
        if role == "aux":
            return {"content": "## Goal\nstub\n## Progress\nstub"}
        if role == "aux-title":
            return {"content": "Title"}
        return {"status": 400, "error": f"unknown model role {role!r}"}

    def serve(self, h, body: dict, api: str) -> None:
        messages = normalize(body, api)
        role, stream = role_of(body.get("model", "")), bool(body.get("stream"))
        blob = json.dumps(messages, sort_keys=True).encode()
        rec = {"rid": self.next_rid(), "ts": time.time(), "method": "POST", "path": h.path, "route": "completion",
               "api": api, "role": role, "model": body.get("model"), "stream": stream,
               "messages_sha256": hashlib.sha256(blob).hexdigest(), "messages": len(messages),
               "token_estimate": usage_for(messages, 1.0)}
        context = self.continuity_context() if role == "main" and self.continuity_context else {}
        rec["continuity"] = markers(messages, **context)
        rec["current_user_tag"] = context.get("current")
        if role == "main":
            rec["tools"] = sorted((t.get("function") or t).get("name", "?") for t in body.get("tools") or [])
        try:
            reply = self.reply_for(role, messages)
        except Exception as exc:  # a scenario bug is a 500 the host sees, and it is in the log
            reply = {"status": 500, "error": f"scenario raised {exc!r}"[:300]}
        rec.update(reply.get("log") or {})
        if rec.get("current_user_missing"):
            rec["continuity"]["F4"] = False
        if reply.get("hold_until_killed") or reply.get("hold"):
            rec["fault"] = "hold_until_killed" if reply.get("hold_until_killed") else "slow"
            self.log(**rec, phase="held")
            self.in_flight.set()
            if reply.get("on_hold"):
                reply["on_hold"]()
            closed = self.hold(h, self.hold_cap if reply.get("hold_until_killed") else float(reply["hold"]))
            if closed:
                return self.log(**rec, phase="client_closed")
        if reply.get("status"):
            rec["fault"] = f"http_{reply['status']}"
            self.log(**rec)
            return h.send_json(reply["status"], {"error": {"message": reply.get("error", "injected"), "type": "rel_fault"}})
        usage = usage_for(messages, self.usage_scale)
        self.log(**rec, usage=usage, reply={k: reply[k] for k in ("content", "tool_calls") if k in reply})
        try:
            (self.sse if stream else self.json_reply)(h, api, body.get("model"), reply, usage,
                                                      bool((body.get("stream_options") or {}).get("include_usage")))
        except (BrokenPipeError, ConnectionResetError):
            self.log(**rec, phase="client_closed_during_reply")

    @staticmethod
    def hold(h, seconds: float) -> bool:
        """Block until the client closes its socket (True) or ``seconds`` pass (False)."""
        end = time.monotonic() + seconds
        sock = h.connection
        while time.monotonic() < end:
            ready, _, _ = select.select([sock], [], [], min(0.2, max(0.0, end - time.monotonic())))
            if ready:
                try:
                    if not sock.recv(1, socket.MSG_PEEK):  # EOF: the host end is gone
                        return True
                except OSError:
                    return True
        return False

    @staticmethod
    def json_reply(h, api, model, reply, usage, _include_usage) -> None:
        calls = reply.get("tool_calls") or []
        if api == "openai":
            msg = {"role": "assistant", "content": None if calls else reply.get("content", "")}
            if calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
            return h.send_json(200, {"id": "rel-1", "object": "chat.completion", "created": int(time.time()), "model": model,
                                     "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                                     "usage": {"prompt_tokens": usage, "completion_tokens": 20, "total_tokens": usage + 20}})
        content = [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": json.loads(c["arguments"])} for c in calls] \
            or [{"type": "text", "text": reply.get("content", "")}]
        h.send_json(200, {"id": "msg_rel", "type": "message", "role": "assistant", "model": model, "content": content,
                          "stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None,
                          "usage": {"input_tokens": usage, "output_tokens": 20}})

    @staticmethod
    def sse(h, api, model, reply, usage, include_usage) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True
        for event in (openai_events if api == "openai" else anthropic_events)(model, reply, usage, include_usage):
            h.wfile.write(event.encode())
            h.wfile.flush()


def openai_events(model, reply, usage, include_usage):
    calls = reply.get("tool_calls") or []

    def chunk(delta, finish=None, **extra):
        return "data: " + json.dumps({"id": "rel-1", "object": "chat.completion.chunk", "created": int(time.time()),
                                      "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                                      **extra}) + "\n\n"
    if calls:
        yield chunk({"role": "assistant", "content": None, "tool_calls": [
            {"index": k, "id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for k, c in enumerate(calls)]})
    else:
        yield chunk({"role": "assistant", "content": reply.get("content", "")})
    yield chunk({}, "tool_calls" if calls else "stop")
    if include_usage:
        yield "data: " + json.dumps({"id": "rel-1", "object": "chat.completion.chunk", "created": int(time.time()),
                                     "model": model, "choices": [], "usage": {"prompt_tokens": usage, "completion_tokens": 20,
                                                                              "total_tokens": usage + 20}}) + "\n\n"
    yield "data: [DONE]\n\n"


def anthropic_events(model, reply, usage, _include_usage):
    calls = reply.get("tool_calls") or []

    def ev(name, payload):
        return f"event: {name}\ndata: {json.dumps({'type': name, **payload})}\n\n"
    yield ev("message_start", {"message": {"id": "msg_rel", "type": "message", "role": "assistant", "model": model,
                                           "content": [], "stop_reason": None, "stop_sequence": None,
                                           "usage": {"input_tokens": usage, "output_tokens": 0}}})
    blocks = [("tool_use", c) for c in calls] or [("text", reply.get("content", ""))]
    for i, (kind, item) in enumerate(blocks):
        if kind == "text":
            yield ev("content_block_start", {"index": i, "content_block": {"type": "text", "text": ""}})
            yield ev("content_block_delta", {"index": i, "delta": {"type": "text_delta", "text": item}})
        else:
            yield ev("content_block_start", {"index": i, "content_block": {"type": "tool_use", "id": item["id"],
                                                                           "name": item["name"], "input": {}}})
            yield ev("content_block_delta", {"index": i, "delta": {"type": "input_json_delta", "partial_json": item["arguments"]}})
        yield ev("content_block_stop", {"index": i})
    yield ev("message_delta", {"delta": {"stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": 20}})
    yield ev("message_stop", {})


class ProxySink:
    """The host's HTTP(S)_PROXY / ALL_PROXY target: records every request a host process tried to send through a
    proxy (the CONNECT target or absolute URL) and refuses it with 403. Nothing is forwarded anywhere."""

    def __init__(self, log_path):
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def refuse(self):
                with sink.lock, open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": time.time(), "method": self.command, "target": self.path,
                                         "host": self.headers.get("Host")}) + "\n")
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
            do_GET = do_POST = do_PUT = do_HEAD = do_CONNECT = do_DELETE = do_PATCH = do_OPTIONS = refuse

        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, name="proxy-sink", daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
