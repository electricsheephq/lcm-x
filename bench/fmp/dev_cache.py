"""Private full-request cache. Failed/uncertain sends never silently replay."""
import fcntl
import json
import subprocess
import threading
import time
from pathlib import Path

from .common import digest_file, sha, write_json
from .dev_panel import encoded
from .models import Codex, JUDGE_MAX_TOKENS, parse_spec, render

_CODEX_VERSION = None
_VERSION_LOCK = threading.Lock()


def endpoint():
    global _CODEX_VERSION
    with _VERSION_LOCK:
        if _CODEX_VERSION is None:
            _CODEX_VERSION = subprocess.check_output(["codex", "--version"], text=True).strip()
    return dict(transport="codex-exec/chatgpt-plan", codex_version=_CODEX_VERSION)


class Incomplete(Exception):
    pass


def request_identity(spec, messages, *, endpoint, effort, temperature, max_tokens, tools,
                     tool_choice, rubric, renderer, sources):
    return dict(request_bytes_sha256=sha(encoded(messages)), wire_prompt_sha256=sha(render(messages).encode()),
                endpoint=endpoint, model=spec, effort=effort, temperature=temperature,
                output_limit=max_tokens, tools=tools, tool_choice=tool_choice,
                rubric=rubric, renderer=renderer, retrieved_sources=sources)


def source_identity(repo):
    names = subprocess.check_output(["git", "-C", str(repo), "ls-files", "--cached", "--others",
                                     "--exclude-standard", "-z"], text=False).split(b"\0")
    files = {n.decode(): digest_file(path) if (path := Path(repo) / n.decode()).exists() else sha(b"deleted") for n in names
             if n.endswith(b".py") and not n.startswith((b"bench/", b"tests/", b"scripts/"))}
    return sha(encoded(files))


class MemoryCodex(Codex):
    """Keep harness event bodies in memory; inspect native traces, then sanitize them."""
    def run(self, cmd, call, prompt, env, agentic=False):
        timeout = min(self.timeouts[0], self.deadline - time.monotonic())
        if timeout <= 0:
            raise Incomplete("deadline")
        return self.runner(cmd, input=prompt, capture_output=True, text=True, env=env,
                           cwd=self.workdir(call), timeout=timeout)

    def inspect(self, events, rollout):
        result = super().inspect(events, rollout)
        if rollout:
            # This is a newly-created private transport trace, never a result input.
            rollout.write_text(json.dumps({"status": "inspected", "model": result[2]["model"]}) + "\n")
        return result


class Responses:
    def __init__(self, root, backend, deadline, renderer):
        self.root, self.backend, self.deadline, self.renderer = Path(root), backend, deadline, renderer
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.endpoint = endpoint()
        self.counts = {r: {"fresh": 0, "cached": 0} for r in ("reader", "judge")}

    def call(self, spec, messages, records, tools=None, max_tokens=JUDGE_MAX_TOKENS,
             tool_choice="auto", temperature=None, *, role, rubric, sources, validate=None):
        kind, _, effort = parse_spec(spec)
        if kind != "codex":
            raise ValueError("development loop currently admits Codex subscription only")
        identity = request_identity(spec, messages, endpoint=self.endpoint, effort=effort,
            temperature=temperature, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice,
            rubric=rubric, renderer=self.renderer, sources=sources)
        key = sha(encoded(identity))
        path = self.root / (key + ".json")
        with (self.root / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                cached = json.loads(path.read_bytes())
                if cached["identity"] != identity or cached["status"] != "ok":
                    raise Incomplete("cache attempt incomplete")
                if sha(encoded(cached["response"])) != cached["response_sha256"]:
                    raise Incomplete("response digest mismatch")
                if validate is not None and not validate(cached["response"]):
                    cached.update(status="error", error="invalid judge verdict")
                    write_json(path, cached)
                    raise Incomplete("invalid judge verdict")
                with self.lock:
                    self.counts[role]["cached"] += 1
                records.extend({**r, "cache": "cached", "cache_key": key} for r in cached["attempts"])
                return cached["response"]
            if time.monotonic() >= self.deadline:
                raise Incomplete("deadline")
            state = dict(identity=identity, status="pending", attempts=[], response_sha256=None)
            write_json(path, state)  # durable uncertain-send marker before dispatch
            calls = []
            with self.lock:
                self.counts[role]["fresh"] += 1
            try:
                response = self.backend.call(spec, messages, calls, tools=tools, max_tokens=max_tokens,
                    tool_choice=tool_choice, temperature=temperature)
                if not response.get("content") or any(r.get("status") != "ok" for r in calls):
                    raise Incomplete("invalid model response")
                if validate is not None and not validate(response):
                    raise Incomplete("invalid judge verdict")
                state.update(status="ok", response=response, response_sha256=sha(encoded(response)))
            except Exception as exc:
                state.update(status="error", error=type(exc).__name__)
                raise
            finally:
                # No prompt, answer, native transcript path or thread identifier in attempt metadata.
                state["attempts"] = [{k: r.get(k) for k in ("status", "error", "requested_model", "returned_model",
                    "input_tokens", "output_tokens", "latency_s", "retries", "identity_source")} for r in calls]
                write_json(path, state)
            records.extend({**r, "cache": "fresh", "cache_key": key} for r in state["attempts"])
            return response
