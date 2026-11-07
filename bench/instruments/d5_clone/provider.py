"""Loopback OpenAI-compatible replay and digest provider, with content-free stats."""

from collections import Counter
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import socket
import threading


MAX_BODY = 64 * 1024 * 1024
OUTSIDE = "(outside a step)"  # The stats key for agent calls with no active step; never a valid step id.
NONCE = re.compile(r'<lcm-summary nonce="([0-9a-f]+)">')  # LCM-X's summary integrity envelope opener.


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()



def _envelope_nonce(messages):
    """The output nonce of LCM-X's summary contract, or None. The contract's opener is in the trusted system policy;
    only system messages are read, because the transcript is untrusted data and may quote an older envelope."""
    found = {n for m in messages if isinstance(m, dict) and m.get("role") == "system"
             and isinstance(m.get("content"), str) for n in NONCE.findall(m["content"])}
    return found.pop() if len(found) == 1 else None


BYTES_PER_SUMMARY_TOKEN = 160  # A token is at least one UTF-8 byte, so source/summary <= 160 even for the densest text.
BUDGET = re.compile(r"(?:about |Target ~|Maximum )(\d+) tokens")  # The token budget LCM-X's summary prompts state.
HIDDEN = re.compile(r"\[(?:GC'd )?[Ee]xternalized [^\]]*?\bchars=(\d+)")  # The size a tool-output placeholder hides.


def _padding(messages):
    """Deterministic filler that brings the digest to the length a real summarizer writes.

    The stored leaf can be far larger than the prompt shows (clipped chunks, externalized tool output), and LCM-X's
    summary_quality doctor check reports a summary as degraded at >= 400:1, or under 500 tokens for a 100k-token
    source. A fixed 40-token digest would warn on every large leaf in both arms and hide a real regression. So the
    length is the prompt's stated budget, at most 40% of the visible transcript (an unclipped source is never
    summarized longer than itself), and at least one token per BYTES_PER_SUMMARY_TOKEN bytes of the request plus the
    characters its placeholders hide."""
    system = [m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "system"
              and isinstance(m.get("content"), str)]
    shown = [_json(m.get("content")) for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    budgets = [int(b) for text in system for b in BUDGET.findall(text)]
    hidden = sum(int(n) for text in shown for n in HIDDEN.findall(text))
    transcript = sum(len(text.encode("utf-8")) for text in shown)
    target = max(-(-(len(_json(messages).encode("utf-8")) + hidden) // BYTES_PER_SUMMARY_TOKEN),
                 min(min(budgets), transcript // 10) if budgets else 0)
    # One token per word under both counters (cl100k encodes " pad" as one token; the estimate counts 4 characters).
    return f"\nd5 pad {_sha(messages)[:8]}" + "".join("\n" + " ".join(["pad"] * 16) for _ in range(-(-target // 16)))


def _tool_call(call):
    return (isinstance(call, dict) and isinstance(call.get("id"), str) and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), (str, dict, list)))


def _message(entry):
    """The assistant message for a reply entry (None is a miss). It is encoded once here, so an entry the provider
    cannot serve fails before any reply is sent."""
    message = {"role": "assistant", "content": (entry or {}).get("text", "d5 replay miss")}
    if entry is not None and entry.get("tool_calls"):
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {
                "name": c["name"], "arguments": c["arguments"] if isinstance(c["arguments"], str)
                else _json(c["arguments"])}} for c in entry["tool_calls"]]}
    _json(message).encode("utf-8")
    return message


def _scripted(script, index):
    """The scripted reply at index, or None (a replay miss) when there is none or it is not a reply this provider can
    serve: a dict whose "tool_calls" is a non-empty list of calls with a str id, a str name and arguments, or, when
    "tool_calls" is absent, null or empty, whose "text" is a str; and whose message encodes. Every other entry is a
    miss, never a silent fallback or a failed request that the counter rollback would erase."""
    entry = script[index] if isinstance(script, list) and index < len(script) else None
    if not isinstance(entry, dict):
        return None
    calls = entry.get("tool_calls")
    if not (isinstance(calls, list) and all(map(_tool_call, calls)) if calls else isinstance(entry.get("text"), str)):
        return None
    try:
        _message(entry)
    except (ValueError, TypeError, RecursionError):  # A lone surrogate, an unencodable argument value, deep nesting.
        return None
    return entry


class Provider:
    """Agent step indices are zero-based; log hashes cover canonical request/completion JSON."""

    def __init__(self, plan, host="127.0.0.1", port=0):
        host = "127.0.0.1" if host == "localhost" else host
        if host not in ("127.0.0.1", "::1"):  # The sandbox profile allows exactly these loopback addresses.
            raise ValueError("provider requires a loopback host")
        self.plan = deepcopy(plan)
        self.agent = plan.get("agent_model", "d5-agent")
        self.cron = plan.get("cron_model", "d5-cron")
        self.models = sorted(set(plan.get("models", [])) | {
            self.agent, self.cron, plan.get("summary_model", "d5-summary")})
        self.step, self.cron_index = None, 0
        self.lock = threading.Lock()
        self.stats = {k: Counter() for k in
                      ("calls_by_model", "calls_per_step", "replay_misses", "unknown_models")}
        self.stats.update(agent_calls_outside_step=0, request_log=[])
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send_json(self, value, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_json(value).encode("utf-8"))

            def do_GET(self):
                if self.path == "/v1/models":
                    self.send_json({"object": "list", "data": [
                        {"id": m, "object": "model", "owned_by": "d5"} for m in provider.models]})
                elif self.path == "/_d5/stats":
                    with provider.lock:
                        self.send_json(provider.stats)
                else:
                    self.send_json({"error": "not found"}, 404)

            def do_POST(self):
                if self.path == "/v1/embeddings":
                    self.send_json({"error": "embeddings are not served"}, 404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if not 0 <= length <= MAX_BODY:
                        raise ValueError("request body size")
                    request = json.loads(self.rfile.read(length) or b"{}")
                    if self.path in ("/_d5/step", "/_d5/end"):
                        step = request["step"] if self.path.endswith("step") else None
                        if self.path.endswith("step") and (not isinstance(step, str) or step == OUTSIDE):
                            raise TypeError("step ids are strings")
                        with provider.lock:
                            provider.step = step
                        self.send_json({"ok": True})
                    elif self.path == "/v1/chat/completions":
                        reply = provider.complete(request)
                        if not request.get("stream"):
                            self.send_json(reply)
                            return
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        choice = reply["choices"][0]
                        delta = deepcopy(choice["message"])
                        for i, call in enumerate(delta.get("tool_calls", [])):
                            call["index"] = i
                        chunk = {**reply, "object": "chat.completion.chunk", "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}]}
                        final = {**chunk, "choices": [
                            {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]}
                        self.wfile.write((f"data: {_json(chunk)}\n\ndata: {_json(final)}\n\n"
                                          "data: [DONE]\n\n").encode("utf-8"))
                    else:
                        self.send_json({"error": "not found"}, 404)
                except (ValueError, KeyError, TypeError, RecursionError, OSError):
                    self.send_json({"error": "invalid request"}, 400)

        class Server(ThreadingHTTPServer):
            request_queue_size = 128
            address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

        self.server = Server((host, port), Handler)
        address = self.server.server_address
        self.url = f"http://{'[' + address[0] + ']' if ':' in address[0] else address[0]}:{address[1]}/v1"
        self.thread = None

    def complete(self, request):
        """One reply. A request that fails part way changes no counter, so it cannot shift the replay's indices."""
        with self.lock:
            saved = ({k: v.copy() if hasattr(v, "copy") else v for k, v in self.stats.items() if k != "request_log"},
                     self.cron_index)
            try:
                return self._complete(request)
            except Exception:
                self.stats.update(saved[0])
                self.cron_index = saved[1]
                raise

    def _complete(self, request):
        if True:
            model, messages = request["model"], request.get("messages", [])
            self.stats["calls_by_model"][model] += 1
            index = self.stats["calls_by_model"][model] - 1
            entry = None
            if model == self.agent:
                if self.step is None:  # Outside any step: always a miss, and never counted under a step id.
                    index = self.stats["agent_calls_outside_step"]
                    self.stats["agent_calls_outside_step"] += 1
                    self.stats["replay_misses"][OUTSIDE] += 1
                else:
                    index = self.stats["calls_per_step"][self.step]
                    self.stats["calls_per_step"][self.step] += 1
                    script = self.plan.get("steps", {}).get(self.step, [])
                    entry = _scripted(script, index)
                    if entry is None:
                        self.stats["replay_misses"][self.step] += 1
            elif model == self.cron:
                index = self.cron_index
                self.cron_index += 1
                script = self.plan.get("cron", [])
                entry = _scripted(script, index)
                if entry is None:
                    self.stats["replay_misses"]["cron"] += 1
            else:
                if model not in self.models:
                    self.stats["unknown_models"][model] += 1
                text = f"D5 digest v1\nmessages={len(messages)}\n"
                text += "\n".join(f"{m['role']} {_sha(m.get('content'))[:12]}" for m in messages)
                text += f"\nrequest={_sha(messages)}"
                nonce = _envelope_nonce(messages)
                if nonce:  # The fixed digest inside the envelope LCM-X's summary route requires, with its closing hint.
                    text += _padding(messages)
                    text = f'<lcm-summary nonce="{nonce}">\n{text}\nExpand for details about: d5 digest\n</lcm-summary>'
                entry = {"text": text}
            message = _message(entry)
            # Token counts are a fixed estimate (UTF-8 bytes / 4, rounded up), so host pre-checks see realistic sizes.
            prompt, completion = (-(-len(_json(value).encode("utf-8")) // 4) for value in (messages, message))
            reply = {"id": "chatcmpl-" + _sha(message), "object": "chat.completion", "created": 0,
                    "model": model, "choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if "tool_calls" in message else "stop"}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                              "total_tokens": prompt + completion}}
            self.stats["request_log"].append((model, self.step, index, _sha(request), _sha(reply)))
            return reply

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()
