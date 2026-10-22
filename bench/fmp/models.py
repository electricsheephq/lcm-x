"""Subscription-plan harness backends (codex exec, claude -p) and the opt-in OpenRouter path
with its process-safe reservation/spend ledger."""
import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx

from .common import sha, utc, write_json

READERS = ("openai/gpt-6.1-sol", "anthropic/claude-opus-5.5", "anthropic/claude-fable-5.1")
JUDGES = READERS[:2]
REASONING = {"effort": "high"}
READER_MAX_TOKENS, JUDGE_MAX_TOKENS = 32_000, 4096  # D11: room for reasoning plus the answer


class PayloadError(Exception):
    """An HTTP-200 body carrying an error; retried like a 5xx."""


class ModelMismatch(Exception):
    pass


def input_bound(messages, tools=None):
    # UTF-8 bytes are a conservative tokenizer-independent bound, plus framing.
    return len(json.dumps([messages, tools], ensure_ascii=False).encode()) + 256


class BudgetStopped(Exception):
    pass


class Spend:
    def __init__(self, path, cap):
        self.path, self.cap = Path(path), float(cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        if not math.isfinite(self.cap) or self.cap <= 0:
            raise ValueError("positive spend cap required")

    def change(self, reserve=0, release=0, cost=0, stop=False):
        with self.lock, self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                "cap": self.cap, "spent": 0, "reserved": 0, "stopped": False}
            if state["cap"] != self.cap:
                raise ValueError("spend cap changed for existing ledger")
            if reserve and (state["stopped"] or state["spent"] + state["reserved"] + reserve > self.cap):
                raise BudgetStopped()
            state["reserved"] = max(0, state["reserved"] + reserve - release)
            state["spent"] += cost
            state["stopped"] |= stop or state["spent"] >= self.cap
            write_json(self.path, state)
            return state


class OpenRouter:
    def __init__(self, spend, metadata=None, transport=None, sleeper=time.sleep):
        self.spend, self.sleep = spend, sleeper
        self.client = httpx.Client(base_url="https://openrouter.ai/api/v1/", timeout=180,
                                   transport=transport)
        self.metadata = metadata or {m["id"]: m for m in self.client.get("models").json()["data"]}
        if not all(m in self.metadata for m in READERS):
            raise RuntimeError("pinned model unavailable")

    def call(self, model, messages, records, tools=None, max_tokens=JUDGE_MAX_TOKENS, tool_choice="auto",
             temperature=None):
        if model not in READERS:
            raise ValueError("model is not pinned")
        meta = self.metadata[model]
        price = meta["pricing"]
        prices = [price, *price.get("overrides", [])]
        reserve = input_bound(messages, tools) * max(float(p["prompt"]) for p in prices)
        reserve += max_tokens * max(float(p["completion"]) for p in prices)
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "reasoning": REASONING, "usage": {"include": True},
                "provider": {"require_parameters": True, "allow_fallbacks": False}}
        if tools:
            body.update(tools=tools, tool_choice=tool_choice)
        # Only where the pinned endpoint supports it; require_parameters would otherwise refuse the call.
        if temperature is not None and "temperature" in meta.get("supported_parameters", ["temperature"]):
            body["temperature"] = temperature
        for attempt in range(4):
            self.spend.change(reserve=reserve)
            started = time.monotonic()
            record = {"requested_model": model, "returned_model": None, "provider": None,
                      "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                      "cost": 0, "latency_s": 0, "retries": attempt, "at": utc()}
            response = None
            try:
                response = self.client.post("chat/completions", json=body,
                    headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
                record["http_status"] = response.status_code
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                if response.status_code == 402:  # account out of credits: a run state, like the spend guard
                    raise BudgetStopped()
                if response.is_error:
                    raise RuntimeError("nonretryable_api_error")
                data = response.json()
                errors = [data.get("error"), *[c.get("error") for c in data.get("choices") or []]]
                if any(isinstance(e, dict) and str(e.get("code")) == "402" for e in errors):
                    raise BudgetStopped()  # out of credits inside an HTTP-200 body
                if data.get("error") or not data.get("choices") or data["choices"][0].get("error"):
                    raise PayloadError()
                usage = data.get("usage", {})
                cost = float(usage["cost"])
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError("invalid_cost")
                record.update(returned_model=data.get("model"), provider=data.get("provider"), cost=cost,
                    temperature=body.get("temperature"), finish_reason=data["choices"][0].get("finish_reason"),
                    native_finish_reason=data["choices"][0].get("native_finish_reason"),
                    input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                    reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"), identity_source="response")
                if record["returned_model"] != model:
                    raise ModelMismatch()
                self.spend.change(release=reserve, cost=cost)
                record["status"] = "ok"
                records.append({**record, "latency_s": time.monotonic() - started})
                message = data["choices"][0]["message"]
                if not record["returned_model"] or not record["provider"]:
                    raise RuntimeError("missing_provider_identity")
                return message
            except Exception as exc:
                if record.get("status") == "ok":
                    raise
                retryable = isinstance(exc, (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError,
                                             PayloadError))
                # Unknown billing (timeouts, malformed accounting) consumes its
                # reservation conservatively. No unaccounted retries at the cap.
                # Any 402 is unbilled and stops the ledger, so no further dispatch.
                stopped = isinstance(exc, BudgetStopped)
                uncertain = not stopped and (response is None or not response.is_error)
                self.spend.change(release=reserve, cost=reserve if uncertain else 0, stop=stopped)
                record.update(status="error", error="model_mismatch" if isinstance(exc, ModelMismatch) else type(exc).__name__, cost=reserve if uncertain else 0,
                              cost_estimated=uncertain, latency_s=time.monotonic() - started)
                records.append(record)
                if isinstance(exc, BudgetStopped):
                    raise
                if not retryable or attempt == 3:
                    raise RuntimeError("openrouter_call_failed") from None
                self.sleep(min(2 ** attempt, 8))
        raise RuntimeError("unreachable")


# ---- Subscription-plan harness backends: codex exec and claude -p, no metered API ----

CODEX_DEFAULT_READER, CODEX_DEFAULT_JUDGE = "codex:gpt-6.1-sol@high", "codex:gpt-6.1-sol@medium"
CLAUDE_AGREEMENT_JUDGE = "claude:claude-opus-5-5"
# Context windows the harnesses report (codex task_started model_context_window; claude modelUsage).
CONTEXT = {"gpt-6.1-sol": 475_000, "claude-opus-5-5": 1_000_000, "claude-fable-5-1": 1_000_000}
GLM_CONTEXT = {"glm-5.3": 128_000}  # conservative harness admission cap, not a provider capacity claim
# Every Codex built-in that is not the code-mode wrapper is off. Code mode stays on: in this Codex
# it is the only route to MCP tools, and its JS runtime has no file system or imports (canary-proven).
CODEX_DISABLED = ("shell_tool", "unified_exec", "view_image", "image_generation", "multi_agent", "multi_agent_v2",
                  "memories", "plugins", "apps", "browser_use", "browser_use_external", "computer_use",
                  "in_app_browser", "goals", "sleep_tool", "skill_search", "tool_suggest", "hooks", "chronicle",
                  "workspace_dependencies", "shell_snapshot")
CODEX_WRAPPERS = {"exec", "wait"}  # model-visible code-mode calls; nested tool calls arrive as items
# MCP protocol operations Codex runs against the configured servers (only lcmx is configured).
CODEX_MCP_PROTOCOL = {"list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource"}
SERVER = "lcmx"
AGENTIC_TOOL_CAP = 30
USAGE_LIMIT = ("usage limit", "rate limit", "quota", "limit reached", "out of credits")
GLM_BASE_URL = "https://api.z.ai/api/anthropic"
GLM_HOST = "api.z.ai"
GLM_ENV = ("HOME", "USER", "PATH", "LANG", "TMPDIR")
BASE_ENV = ("HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "SHELL")  # harness children get only these


class ToolViolation(Exception):
    """The harness used a tool other than the lcmx MCP read tools."""


def parse_spec(spec):
    """`codex:<model>@<effort>`, `claude:<model>`, `glm:<model>`, or the opt-in `openrouter:<id>` (legacy: a bare id)."""
    kind, sep, rest = spec.partition(":")
    if not sep or kind not in ("codex", "claude", "glm", "openrouter"):
        return "openrouter", spec, None
    model, _, effort = rest.partition("@")
    if not model or (kind == "codex") != bool(effort):
        raise ValueError("invalid model spec")
    return kind, model, effort or None


def reader_spec(spec):
    """CLI admission: every accepted reader must have a known context length."""
    try:
        kind, model, _ = parse_spec(spec)
        if kind == "openrouter":
            response = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
            response.raise_for_status()
            context = next((m.get("context_length") for m in response.json()["data"] if m["id"] == model), None)
        else:
            context = (GLM_CONTEXT if kind == "glm" else CONTEXT).get(model)
        if not isinstance(context, int) or context <= 0:
            raise ValueError("unknown context length")
    except (ValueError, KeyError, httpx.HTTPError) as exc:
        raise argparse.ArgumentTypeError("reader context length unavailable; use a supported model spec") from exc
    return spec


def harness(spec):
    return parse_spec(spec)[0] in ("codex", "claude", "glm")


def render(messages):
    return "\n\n".join(m["content"] for m in messages if m.get("content"))


class Quota:
    """In-run brakes: Codex stops at a share of the weekly window; Claude stops on a usage-limit error."""
    def __init__(self, path, codex_stop=95.0):
        self.path, self.codex_stop, self.lock = Path(path), float(codex_stop), threading.Lock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "codex": {"weekly_used_percent_first": None, "weekly_used_percent_last": None, "stopped": False},
            "claude": {"usage_limit_errors": 0, "stopped": False}}
        self.state.setdefault("glm", {"usage_limit_errors": 0, "stopped": False})

    def check(self, kind):
        with self.lock:
            if self.state.get(kind, {}).get("stopped"):
                raise BudgetStopped()

    def codex(self, rate_limits):
        windows = [w for w in ((rate_limits or {}).get(k) for k in ("primary", "secondary")) if w]
        weekly = [w for w in windows if (w.get("window_minutes") or 0) >= 10080]
        if not weekly:
            return None
        if any(w.get("used_percent") is None for w in weekly):
            return None
        used = max(float(w["used_percent"]) for w in weekly)
        with self.lock:
            state = self.state["codex"]
            state["weekly_used_percent_first"] = state["weekly_used_percent_first"] if \
                state["weekly_used_percent_first"] is not None else used
            state["weekly_used_percent_last"] = used
            state["stopped"] |= used >= self.codex_stop
            write_json(self.path, self.state)
        return used

    def stop(self, kind):
        with self.lock:
            self.state[kind]["stopped"] = True
            if kind in ("claude", "glm"):
                self.state[kind]["usage_limit_errors"] += 1
            write_json(self.path, self.state)


def limited(text):
    return any(k in (text or "").lower() for k in USAGE_LIMIT)


def retry_call(call, records, receipt):
    """Retry transport failures only; retain the caller's frozen prompt and receipts."""
    first = len(records)
    for attempt in range(1, 4):
        before = len(records)
        try:
            result = call()
        except RuntimeError:
            error = records[-1].get("error") if len(records) > before else None
            if error not in ("codex_failed", "claude_failed", "timeout", "mcp_unavailable"):
                raise
            receipt["attempts"] = attempt
            receipt.setdefault("attempt_errors", []).append(error)
            if attempt == 3:
                raise
            delay = (5, 20)[attempt - 1]
            receipt["retry_backoff_s"] = receipt.get("retry_backoff_s", 0) + delay
            time.sleep(delay)
        else:
            for record in records[first:before]:
                record["retried"] = True
            return result
        finally:
            if attempt > 1 or "attempts" in receipt:
                for record in records[first:]:
                    record["attempts"] = attempt
            if attempt > 1:
                receipt["attempts"] = attempt


class Harness:
    def __init__(self, root, quota, runner=None, binary=None, timeout=(600, 1800)):
        self.root = Path(root) / "harness"
        self.root.mkdir(parents=True, exist_ok=True)
        self.quota, self.timeouts = quota, timeout  # (text call, agentic call) seconds
        self.runner, self.binary = runner or subprocess.run, binary or self.kind
        self.scratch = Path(tempfile.mkdtemp(prefix=f"fmp-{self.kind}-"))
        os.chmod(self.scratch, 0o700)

    def close(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def directory(self):
        call = self.root / uuid.uuid4().hex
        (call / "cwd").mkdir(parents=True)  # a fresh empty working root per call
        return call

    def server(self, call, server):
        # The MCP reader's spec: one frozen store, the question's clock, a per-call metadata log.
        spec = {**server, "log": str(call / "mcp-log.jsonl"), "max_calls": server.get("max_calls", AGENTIC_TOOL_CAP)}
        write_json(call / "server.json", spec)
        env = {"HOME": str(Path.home()), "PATH": os.environ.get("PATH", ""), "HERMES_DISABLE_LAZY_INSTALLS": "1",
               "PYTHONDONTWRITEBYTECODE": "1", **{k: "4" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                     "MKL_NUM_THREADS", "ORT_NUM_THREADS")}}
        env.update({k: os.environ[k] for k in ("TMPDIR", "FMP_CACHE_ROOT", "LCM_LONGMEMEVAL_FASTEMBED_CACHE",
            "FASTEMBED_CACHE_PATH", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY") if k in os.environ})
        return {"command": sys.executable, "args": [str(Path(__file__).with_name("mcp_reader.py")), str(call / "server.json")],
                "env": env}

    def workdir(self, call):
        return call / "cwd"

    def run(self, cmd, call, prompt, env, agentic=False):
        prompt_path = call / "prompt.txt"
        prompt_path.write_text(prompt)
        try:
            with prompt_path.open() as stdin:
                proc = self.runner(cmd, stdin=stdin, capture_output=True, text=True, env=env, cwd=self.workdir(call),
                                   timeout=self.timeouts[agentic])
        except subprocess.TimeoutExpired as exc:  # keep the partial transcript for the audit
            out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
            if env.get("ANTHROPIC_AUTH_TOKEN"):
                out = out.replace(env["ANTHROPIC_AUTH_TOKEN"], "[REDACTED]")
            (call / "events.jsonl").write_text(out)
            raise
        finally:
            prompt_path.unlink(missing_ok=True)  # corpus stays out of the run tree; the trace keeps metadata
        if env.get("ANTHROPIC_AUTH_TOKEN"):
            proc.stdout = (proc.stdout or "").replace(env["ANTHROPIC_AUTH_TOKEN"], "[REDACTED]")
            proc.stderr = (proc.stderr or "").replace(env["ANTHROPIC_AUTH_TOKEN"], "[REDACTED]")
        (call / "events.jsonl").write_text(proc.stdout or "")
        return proc

    def agentic(self, spec, messages, records, server, readiness_attempts=2):
        for attempt in range(readiness_attempts):
            failure = None
            start = len(records)
            try:
                result = self.call(spec, messages, records, server=server)
            except (RuntimeError, ToolViolation, BudgetStopped) as exc:
                if len(records) == start:
                    raise
                failure = exc
            record = records[-1]
            log = Path(record["mcp_log"])
            events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            initialized = next((e for e in events if e.get("event") == "initialize"), {})
            offered = next((e for e in reversed(events) if e.get("event") == "tools/list"), {})
            status = initialized.get("startup_status", "no_initialize")
            if any(e.get("event") == "startup_failed" for e in events):
                status = "startup_failed"
            count = offered.get("offered_tools", 0)
            record.update(mcp_ready=status == "ok" and count > 0, offered_tools=count,
                offered_tools_hash=offered.get("offered_tools_hash", sha(b"")), startup_status=status,
                lcm_tool_calls=sum("tool" in e for e in events), retries=attempt)
            if failure:
                raise failure
            if not record["mcp_ready"] or not count:
                record.update(status="error", error="mcp_unavailable", backend_error=record.get("error"))
                if attempt + 1 < readiness_attempts:
                    continue
                raise RuntimeError("mcp_unavailable") from failure
            return result


class Codex(Harness):
    kind = "codex"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Isolated CODEX_HOME (no user config, AGENTS.md, rules or MCP servers) and HOME (no host skills).
        # Auth is a symlink to the plan login: no token is copied or written.
        self.home, self.fake_home = self.scratch / "codex-home", self.scratch / "home"
        self.home.mkdir()
        self.fake_home.mkdir()
        (self.home / "auth.json").symlink_to(Path.home() / ".codex" / "auth.json")

    def command(self, model, effort, call, mcp):
        cmd = [self.binary, "exec", "--json", "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
               "-m", model, "-c", f"model_reasoning_effort={effort}", "--cd", str(call / "cwd"), "-s", "read-only",
               "-c", "web_search=disabled"]
        for feature in CODEX_DISABLED:
            cmd += ["--disable", feature]
        if mcp:
            prefix = f"mcp_servers.{SERVER}."
            cmd += ["-c", prefix + "command=" + json.dumps(mcp["command"]),
                    "-c", prefix + "args=" + json.dumps(mcp["args"]),
                    "-c", prefix + "env={" + ",".join(f"{k}={json.dumps(v)}" for k, v in mcp["env"].items()) + "}",
                    "-c", prefix + "startup_timeout_sec=180", "-c", prefix + "tool_timeout_sec=300",
                    "-c", prefix + 'default_tools_approval_mode="approve"']
        return cmd + ["-"]

    def rollout(self, thread):
        found = list((self.home / "sessions").rglob(f"rollout-*{thread}.jsonl")) if thread else []
        return found[0] if found else None

    def inspect(self, events, rollout):
        # Violations: any model-visible call other than the code-mode wrapper, any executed item
        # that is not an lcmx MCP call (or an MCP protocol listing), from both event streams.
        violations, mcp_calls, info = [], 0, {"model": None, "rate_limits": None, "requests": 0}
        reported = [e.get("model") or (e.get("response") or {}).get("model") or (e.get("item") or {}).get("model")
                    for e in events if e.get("type") in ("turn.completed", "response.completed", "item.completed")]
        for item in (e.get("item") or {} for e in events if e.get("type", "").startswith("item.")):
            kind = item.get("type")
            if kind == "mcp_tool_call":
                continue  # counted from the rollout below
            if kind not in ("agent_message", "reasoning", "error", None):
                violations.append(kind)
        lines = rollout.read_text().split("\n") if rollout else []
        while lines and not lines[-1]:
            lines.pop()
        for line in lines:
            entry = json.loads(line)
            payload = entry.get("payload") or {}
            kind = payload.get("type")
            if entry.get("type") == "response_item" and payload.get("role") == "assistant":
                reported.append(payload.get("model"))
            if entry.get("type") == "turn_context":
                info["model"] = payload.get("model")
            elif kind == "token_count":
                info["rate_limits"] = payload.get("rate_limits") or info["rate_limits"]
                info["requests"] += bool(payload.get("info"))
            elif entry.get("type") == "response_item" and kind and kind.endswith("_call") and \
                    payload.get("name") not in CODEX_WRAPPERS:
                violations.append(payload.get("name") or kind)
            elif kind == "item_completed":
                item = payload.get("item") or {}
                if item.get("type") == "McpToolCall":
                    if item.get("server") == SERVER:  # the reader serves only the offered set
                        mcp_calls += 1
                    elif not (item.get("server") == "codex" and item.get("tool") in CODEX_MCP_PROTOCOL):
                        violations.append(f"{item.get('server')}:{item.get('tool')}")
                elif item.get("type") not in ("UserMessage", "AgentMessage", "Reasoning", "Error", None):
                    violations.append(item.get("type"))
        reported = [m for m in reported if m]
        info["identity_source"] = "response" if reported else "configured"
        info["model"] = reported[-1] if reported else info["model"]
        return violations, mcp_calls, info

    def call(self, spec, messages, records, tools=None, server=None, **_):
        if tools:
            raise ValueError("harness readers get tools over MCP only")
        self.quota.check("codex")
        _, model, effort = parse_spec(spec)
        call = self.directory()
        mcp = self.server(call, server) if server else None
        prompt = render(messages)
        env = {"HOME": str(self.fake_home), "CODEX_HOME": str(self.home), "PATH": os.environ.get("PATH", ""),
               "TMPDIR": os.environ.get("TMPDIR", "/tmp"), "LANG": "en_US.UTF-8", "USER": os.environ.get("USER", "")}
        started = time.monotonic()
        record = {"backend": "codex", "requested_model": spec, "mcp_log": str(call / "mcp-log.jsonl") if mcp else None, "returned_model": None, "input_tokens": None,
                  "cached_input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "cost": 0, "retries": 0,
                  "at": utc(), "call_dir": call.name, "prompt_tokens_estimate": -(-len(prompt) // 4)}
        try:
            proc = self.run(self.command(model, effort, call, mcp), call, prompt, env, bool(mcp))
        except subprocess.TimeoutExpired:
            records.append({**record, "status": "error", "error": "timeout", "latency_s": time.monotonic() - started})
            raise RuntimeError("codex_call_failed") from None
        events = [json.loads(line) for line in proc.stdout.split("\n") if line.startswith("{")]
        thread = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None)
        rollout = self.rollout(thread)
        violations, mcp_calls, info = self.inspect(events, rollout)
        used = self.quota.codex(info["rate_limits"])
        done = next((e for e in reversed(events) if e.get("type") == "turn.completed"), None)
        failure = " ".join(str((e.get("error") or {}).get("message") or e.get("message") or "")
                           for e in events if e.get("type") in ("turn.failed", "error"))
        texts = [(e.get("item") or {}).get("text") for e in events
                 if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message"]
        usage = (done or {}).get("usage") or {}
        record.update(returned_model=info["model"] or model, latency_s=time.monotonic() - started, thread_id=thread,
            model_requests=info["requests"], weekly_used_percent=used, mcp_calls=mcp_calls, violations=violations,
            input_tokens=int(usage.get("input_tokens") or 0), cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0), reasoning_tokens=int(usage.get("reasoning_output_tokens") or 0),
            transcripts=[str(call / "events.jsonl"), str(rollout) if rollout else None], exit_code=proc.returncode)
        record.update({k: usage.get(v) for k, v in zip(("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens"),
                                                     ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"))})
        record.update(identity_source=info["identity_source"], wrapper_overhead_tokens=None,
                      input_tokens_label="total incl. retrieved evidence")
        if limited(failure) or limited(proc.stderr):
            self.quota.stop("codex")
            records.append({**record, "status": "error", "error": "usage_limit"})
            raise BudgetStopped()
        if proc.returncode != 0 or done is None or not texts or any(
                e.get("type") == "turn.failed" or (e.get("type") == "error" and not any(
                    later.get("type") == "turn.completed" for later in events[i + 1:])) for i, e in enumerate(events)):
            records.append({**record, "status": "error", "error": "codex_failed"})
            raise RuntimeError("codex_call_failed")
        if info["model"] and info["model"] != model:
            records.append({**record, "status": "error", "error": "model_mismatch"})
            raise RuntimeError("model_mismatch")
        records.append({**record, "status": "ok" if not violations else "tool_violation", "finish_reason": "stop"})
        if violations:
            raise ToolViolation()
        return {"role": "assistant", "content": texts[-1]}


DENY_HOOK = Path(__file__).with_name("deny_hook.py")


class Claude(Harness):
    kind = "claude"

    def __init__(self, *args, token=None, kind="claude", **kwargs):
        self.kind = kind
        kwargs.setdefault("binary", "claude")
        super().__init__(*args, **kwargs)
        # Isolated config: an empty settings.json, so no user routing, hooks or MCP servers apply.
        self.config = self.scratch / "config"
        self.config.mkdir()
        (self.config / "settings.json").write_text("{}\n")
        self.token = token or keychain_token

    def workdir(self, call):
        # Outside the home tree: Claude Code loads CLAUDE.md memory from every ancestor directory, and
        # a run output under /Users/<user> pulls in ~/.claude/CLAUDE.md (measured: +19k input tokens).
        path = self.scratch / "cwd" / call.name
        path.mkdir(parents=True)
        return path

    def command(self, model, system, call, mcp):
        cmd = [self.binary, "-p", "--model", model, "--output-format", "stream-json", "--verbose",
               "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands", "--tools", "",
               "--system-prompt", system]
        if mcp:
            hook = shlex.join(map(str, (sys.executable, "-I", DENY_HOOK, call / "denied.jsonl")))
            cmd += ["--mcp-config", json.dumps({"mcpServers": {SERVER: mcp}}),
                    "--allowedTools", f"mcp__{SERVER}", "--max-turns", str(AGENTIC_TOOL_CAP + 10),
                    "--settings", json.dumps({"hooks": {"PreToolUse": [
                        {"matcher": "*", "hooks": [{"type": "command", "command": hook, "timeout": 20}]}]}})]
        return cmd

    def call(self, spec, messages, records, tools=None, server=None, **_):
        if tools:
            raise ValueError("harness readers get tools over MCP only")
        auth = glm_key() if self.kind == "glm" else None
        self.quota.check(self.kind)
        _, model, _ = parse_spec(spec)
        call = self.directory()
        mcp = self.server(call, server) if server else None
        system = render([m for m in messages if m["role"] == "system"])
        prompt = render([m for m in messages if m["role"] != "system"])
        # An allowlisted environment: host-session markers (CLAUDECODE, CLAUDE_CODE_*, SDK and OAuth
        # routing variables) would make the child wait for host auth or route elsewhere.
        env = {k: os.environ[k] for k in (GLM_ENV if auth else BASE_ENV) if k in os.environ}
        env.update(LANG=env.get("LANG", "en_US.UTF-8"), CLAUDE_CONFIG_DIR=str(self.config))
        if auth:
            env.update(ANTHROPIC_BASE_URL=GLM_BASE_URL, ANTHROPIC_AUTH_TOKEN=auth)
        else:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.token()
        started = time.monotonic()
        record = {"backend": self.kind, "requested_model": spec, "mcp_log": str(call / "mcp-log.jsonl") if mcp else None, "returned_model": None, "input_tokens": None,
                  "output_tokens": None, "reasoning_tokens": None, "cost": 0, "retries": 0, "at": utc(),
                  "call_dir": call.name, "prompt_tokens_estimate": -(-len(system + prompt) // 4)}
        if auth:
            record["endpoint_host"] = GLM_HOST
        try:
            cmd = self.command(model, system, call, mcp) + (["--effort", "low"] if auth else [])
            proc = self.run(cmd, call, prompt, env, bool(mcp))
        except subprocess.TimeoutExpired:
            records.append({**record, "status": "error", "error": "timeout", "latency_s": time.monotonic() - started})
            raise RuntimeError("claude_call_failed") from None
        finally:
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
            env.pop("ANTHROPIC_AUTH_TOKEN", None)
        events = [json.loads(line) for line in proc.stdout.split("\n") if line.strip()]
        result = next((e for e in reversed(events) if e.get("type") == "result"), None) or {}
        violations, mcp_calls, models = [], 0, set()
        for event in events:
            if event.get("type") == "assistant":
                message = event.get("message") or {}
                models.add(message.get("model"))
                for block in message.get("content") or []:
                    if block.get("type") == "tool_use":
                        if block.get("name", "").startswith(f"mcp__{SERVER}__"):
                            mcp_calls += 1
                        else:
                            violations.append(block.get("name"))
        denied = call / "denied.jsonl"
        violations += [json.loads(line)["tool"] for line in denied.read_text().split("\n") if line.strip()] if denied.exists() else []
        violations += [d.get("tool_name") for d in result.get("permission_denials") or []]
        usage = result.get("usage") or {}
        # Every reply must come from the requested model. Claude Code's own side calls (Haiku, e.g. on
        # MCP sessions) are recorded as auxiliary, with their tokens, never as the reader or judge.
        auxiliary = {k: v for k, v in (result.get("modelUsage") or {}).items() if k != model}
        models.discard(None)
        rejected = any(e.get("type") == "rate_limit_event" and (e.get("rate_limit_info") or {}).get("status") == "rejected"
                       for e in events)
        record.update(returned_model=sorted(models), latency_s=time.monotonic() - started, mcp_calls=mcp_calls,
            auxiliary_models={k: {t: v.get(t) for t in ("inputTokens", "outputTokens", "cacheReadInputTokens")}
                              for k, v in auxiliary.items()},
            violations=violations, model_requests=result.get("num_turns"), exit_code=proc.returncode,
            terminal_reason=result.get("terminal_reason"),
            input_tokens=sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_read_input_tokens",
                                                              "cache_creation_input_tokens")),
            cached_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            reasoning_tokens=int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0),
            transcripts=[str(call / "events.jsonl")])
        record.update(input_tokens=record["input_tokens"] if all(k in usage for k in
                      ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")) else None,
                      cached_input_tokens=usage.get("cache_read_input_tokens"), output_tokens=usage.get("output_tokens"),
                      reasoning_tokens=(usage.get("output_tokens_details") or {}).get("thinking_tokens"),
                      identity_source="response", wrapper_overhead_tokens=None, input_tokens_label="total incl. retrieved evidence")
        glm_limited = auth and (proc.returncode != 0 or result.get("is_error")) and (
            limited(result.get("result")) or limited(proc.stderr)
            or re.search(r"\b(?:http|status|error)[ :=]*429\b|rate[-_ ]limit", proc.stderr, re.IGNORECASE))
        if rejected or glm_limited or (result.get("is_error") and (result.get("api_error_status") == 429 or limited(result.get("result")))):
            self.quota.stop(self.kind)
            records.append({**record, "status": "error", "error": "usage_limit"})
            raise BudgetStopped()
        if proc.returncode != 0 or result.get("subtype") != "success" or result.get("is_error") or not result.get("result"):
            records.append({**record, "status": "error", "error": "claude_failed"})
            raise RuntimeError("claude_call_failed")
        if models != {model} or model not in (result.get("modelUsage") or {}):
            records.append({**record, "status": "error", "error": "model_mismatch"})
            raise RuntimeError("model_mismatch")
        records.append({**record, "status": "ok" if not violations else "tool_violation", "finish_reason":
                        result.get("stop_reason")})
        if violations:
            raise ToolViolation()
        return {"role": "assistant", "content": result["result"]}


def glm_key():
    key = os.environ.get("GLM_API_KEY")
    if not key:
        raise ValueError("GLM_API_KEY is not set")
    return key


def keychain_token():
    # Read per call into the child environment only; never printed or written.
    raw = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials",
                          "-a", os.environ.get("USER", ""), "-w"], capture_output=True, text=True, check=True).stdout
    return json.loads(raw)["claudeAiOauth"]["accessToken"]


class Backends:
    """Routes a model spec to its backend; OpenRouter exists only when a spec names it explicitly."""
    def __init__(self, root, quota, specs, openrouter=None, codex=None, claude=None):
        kinds = {parse_spec(s)[0] for s in specs}
        if "glm" in kinds:
            glm_key()  # fail before constructing any backend or running any subprocess
        self.quota = quota
        self.codex = codex or (Codex(root, quota) if "codex" in kinds else None)
        self.claude = claude or (Claude(root, quota) if "claude" in kinds else None)
        self.glm = Claude(root, quota, kind="glm") if "glm" in kinds else None
        self.openrouter = openrouter if "openrouter" in kinds else None
        if "openrouter" in kinds and openrouter is None:
            raise ValueError("an OpenRouter spec needs an explicit spend ledger (opt-in only)")

    def backend(self, spec):
        kind, model, _ = parse_spec(spec)
        return {"codex": self.codex, "claude": self.claude, "glm": self.glm, "openrouter": self.openrouter}[kind], model

    def call(self, spec, messages, records, tools=None, **kwargs):
        backend, model = self.backend(spec)
        return backend.call(spec if backend is not self.openrouter else model, messages, records, tools=tools, **kwargs)

    def agentic(self, spec, messages, records, server, **kwargs):
        backend, _ = self.backend(spec)
        return backend.agentic(spec, messages, records, server, **kwargs)

    def context_length(self, spec):
        backend, model = self.backend(spec)
        if parse_spec(spec)[0] == "glm":
            return GLM_CONTEXT[model]
        return backend.metadata[model]["context_length"] if backend is self.openrouter else CONTEXT[model]

    def close(self):
        for backend in (self.codex, self.claude, self.glm):
            if backend:
                backend.close()
        if self.openrouter:
            self.openrouter.client.close()
