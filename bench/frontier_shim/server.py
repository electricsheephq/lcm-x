"""Minimal non-streaming OpenAI seam over FMP; no credential handling."""
import argparse
import copy
import hashlib
import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bench.fmp.models import Backends, BudgetStopped, Codex, Quota

from .models import MODELS


def encode(body):
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def error(code):
    return {"error": {"message": code, "type": code}}


def usage_from(record):
    return {public: record[source] for public, source in (
        ("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"))
        if type(record.get(source)) is int and record[source] >= 0}


class ShimCodex(Codex):
    def run(self, *args, **kwargs):
        proc = super().run(*args, **kwargs)  # Preserve the original trace before classification.
        events = [json.loads(line) for line in proc.stdout.split("\n") if line.startswith("{")]  # As Codex.call.
        done = any(e.get("type") == "turn.completed" for e in events)
        fatal = next((e for e in reversed(events) if e.get("type") == "turn.failed"), None)
        fatal = fatal or next((e for e in reversed(events) if e.get("type") == "error"), {})
        # derived_cache.backend_call's exact filter, without mutating shared run().
        proc = copy.copy(proc)
        proc.stdout = "\n".join(json.dumps(e) for e in events if e.get("type") not in
                                ("error", "turn.failed") or (not done and e is fatal))
        if done:
            proc.returncode = 0
        return proc


class Shim:
    def __init__(self, backends, log, concurrency=4):
        if type(concurrency) is not int or not 1 <= concurrency <= 4:
            raise ValueError("Codex concurrency must be 1..4")
        self.backends, self.log = backends, Path(log)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.slots = threading.BoundedSemaphore(concurrency)
        self.stopped, self.draining, self.log_lock = threading.Event(), threading.Event(), threading.Lock()

    def check_quota(self):
        if self.stopped.is_set():
            raise BudgetStopped()
        self.backends.quota.check("codex")

    def complete(self, request, records):
        if self.draining.is_set():
            return 503, error("shutting_down")
        self.check_quota()
        if not self.slots.acquire(timeout=30):
            return 503, error("lane_busy")
        try:
            if self.draining.is_set():  # Queued before the signal; calls already running still finish.
                return 503, error("shutting_down")
            self.check_quota()
            message = self.backends.call(MODELS[request["model"]], request["messages"], records)
            if not records:
                return 503, error("lane_unavailable")
            record = records[-1]
            if not record.get("transcripts") or len(record["transcripts"]) < 2 or not record["transcripts"][1]:
                return 503, error("missing_rollout")
            served = record.get("returned_model")
            if served != "gpt-6.1-sol" or not isinstance(message.get("content"), str):
                return 422, error("invalid_lane_response")
            return 200, {"object": "chat.completion", "model": served, "choices": [
                {"index": 0, "message": {"role": "assistant", "content": message["content"]},
                 "finish_reason": "stop"}], "usage": usage_from(record)}
        finally:
            # Never release on an HTTP client timeout; only the returning backend releases this.
            self.slots.release()

    def handle(self, raw):
        started, at = time.monotonic(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        request, records = {}, []
        try:
            request = json.loads(raw)
        except (json.JSONDecodeError, UnicodeError):
            request = {}
        try:
            if not isinstance(request, dict):
                request = {}
            name = request.get("model")
            messages = request.get("messages")
            temperature = request.get("temperature")
            if not isinstance(name, str) or name not in MODELS:
                status, body = 400, error("unknown_model")
            elif (set(request) - {"model", "messages", "temperature", "stream"}
                  or request.get("stream", False) is not False
                  or not isinstance(messages, list) or not messages
                  or any(not isinstance(m, dict) or m.get("role") not in ("system", "user")
                         or not isinstance(m.get("content"), str) for m in messages)
                  or (temperature is not None and (type(temperature) not in (int, float)
                                                   or not math.isfinite(temperature)))):
                status, body = 400, error("invalid_request")
            else:
                status, body = self.complete(request, records)
        except BudgetStopped:
            self.stopped.set()
            status, body = 429, error("codex_stopped_restart_required")
        except Exception as exc:
            # Backend exceptions can contain prompts, subprocess output or environment values.
            transient = (not records or isinstance(exc, (json.JSONDecodeError, UnicodeError))
                         or records[-1].get("error") in ("timeout", "codex_failed"))
            status, body = (503 if transient else 422), error("lane_unavailable")
        payload = encode(body)
        name = request.get("model")
        name = name if isinstance(name, str) and name in MODELS else None
        record = records[-1] if records else {}
        used = record.get("weekly_used_percent")
        receipt = {"time": at, "model": name, "spec": MODELS.get(name),
                   "served_model": "gpt-6.1-sol" if record.get("returned_model") == "gpt-6.1-sol" else None,
                   "latency_s": time.monotonic() - started,
                   "usage": usage_from(record), "status": status, "retries": 0,
                   "temperature_applied": False,
                   "identity_source": record.get("identity_source") if record.get("identity_source") in ("response", "configured") else None,
                   "weekly_used_percent": used if type(used) in (int, float) and math.isfinite(used) else None,
                   "request_sha256": hashlib.sha256(raw).hexdigest(),
                   "response_sha256": hashlib.sha256(payload).hexdigest()}
        with self.log_lock, self.log.open("ab") as log:
            log.write(encode(receipt) + b"\n")
        return status, {"Retry-After": "60"} if status == 429 else {}, payload


class Handler(BaseHTTPRequestHandler):
    timeout = 60

    def log_message(self, *_):
        pass  # Suppress request/header/error text in the standard HTTP access log.

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        status, headers, payload = self.answer()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def answer(self):
        # Local clients only: a page rebinding its hostname to 127.0.0.1 sends its own Host, and browsers send Origin.
        port = self.server.server_address[1]
        if "Origin" in self.headers or self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            return 403, {}, encode(error("forbidden"))
        # A cross-origin page cannot send this type without a preflight, and OPTIONS is never answered.
        if self.headers.get_content_type() != "application/json":
            return 415, {}, encode(error("unsupported_media_type"))
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or self.headers.get("Transfer-Encoding"):
                raise ValueError()
            raw = self.rfile.read(length)
        except (ValueError, OSError):
            return 400, {}, encode(error("invalid_request"))
        try:
            return self.server.shim.handle(raw)
        except Exception:  # For example a failed receipt write after the backend returned.
            return 503, {}, encode(error("lane_unavailable"))


class Server(ThreadingHTTPServer):
    # Graceful shutdown waits for active calls rather than abandoning subprocess owners.
    daemon_threads = False

    def __init__(self, shim, port=8765):
        self.shim = shim
        super().__init__(("127.0.0.1", port), Handler)

    def handle_error(self, *_):
        pass  # Never print a traceback containing backend or request data.


def build(root, log, concurrency=4):
    home = os.environ.get("LCMX_SHIM_CODEX_HOME")
    if not home or not Path(home).is_absolute() or not Path(home).is_dir():
        raise ValueError("LCMX_SHIM_CODEX_HOME must name an existing absolute leased directory")
    root = Path(root).resolve()  # FMP passes it as the child cwd and as --cd; relative would nest.
    root.mkdir(parents=True, exist_ok=True)
    # A new quota on every restart; previous receipts are retained, never pruned.
    quota_root = root / ("instance-" + str(time.time_ns()))
    quota_root.mkdir()
    quota = Quota(quota_root / "quota.json")
    codex = ShimCodex(quota_root, quota, codex_home=home, include_total_tokens=True, timeout=(540, 1800))
    backends = Backends(quota_root, quota, MODELS.values(), codex=codex)
    return Shim(backends, log, concurrency)


def serve(shim, port):
    previous, stopping = {}, threading.Event()
    try:
        with Server(shim, port) as server:
            def stop(*_):
                if not stopping.is_set():
                    stopping.set()
                    shim.draining.set()
                    threading.Thread(target=server.shutdown).start()
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, stop)
            server.serve_forever()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        shim.backends.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    shim = build(args.root, args.log, args.concurrency)
    serve(shim, args.port)


if __name__ == "__main__":
    main()
