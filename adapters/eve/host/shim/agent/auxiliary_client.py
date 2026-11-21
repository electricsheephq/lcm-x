"""Hermes ``agent.auxiliary_client`` for the Eve host. LCM-X calls ``call_llm`` for summaries (the summary DAG),
``lcm_expand_query``, extraction and evidence compiling, exactly where Hermes gives it its auxiliary model.

Two routes, in this order:

1. The Eve bridge (``BRIDGE``, set by ``lcmx_eve_host.py serve``): the request goes over stdio to the Eve process,
   which runs it on the agent's own model (or its ``summaryModel``) with the agent's credentials, such as a ChatGPT
   subscription that only the Eve process can use. It is a separate, stateless call with only LCM-X's prompt: no
   agent history, no tools, nothing in the agent's session. Hermes does the same with its auxiliary model.
2. A coding CLI on this machine with its own subscription login (``LCMX_LLM_CLI=codex`` or ``claude``): one
   non-interactive, ephemeral run per call (``codex exec`` / ``claude -p``) in an empty folder, read-only, without the
   user's config or rules. Used by the dev-agent hub, where no agent process lends its model.
3. Opt-in only: an OpenAI-compatible ``/chat/completions`` endpoint (stdlib only), when no bridge is attached and
   ``LCMX_LLM_BASE_URL`` is set explicitly (for example a local model server). An API key in the environment for
   embeddings is never used for chat unless this is set:
  LCMX_LLM_BASE_URL     no default: unset means no HTTP route
  LCMX_LLM_API_KEY_ENV  name of the variable that holds the key; default OPENAI_API_KEY
  LCMX_LLM_MODEL        default gpt-5.4-mini

With neither, ``call_llm`` raises and LCM-X writes no LLM summary (the raw rows stay), as on a Hermes host without an
auxiliary model.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_TIMEOUT_S = 120.0
# Reasoning-model families take max_completion_tokens and no temperature.
REASONING_PREFIXES = ("gpt-5", "gpt-6", "o1", "o3", "o4")
# The Eve bridge: an object with ``complete(params: dict, timeout: float) -> dict`` (set by the engine host).
BRIDGE: Any = None


class AuxiliaryLLMError(RuntimeError):
    pass


def _settings() -> tuple[str, str, str]:
    base = (os.environ.get("LCMX_LLM_BASE_URL") or "").strip().rstrip("/")
    key = os.environ.get(os.environ.get("LCMX_LLM_API_KEY_ENV") or "OPENAI_API_KEY", "").strip()
    model = (os.environ.get("LCMX_LLM_MODEL") or DEFAULT_MODEL).strip()
    return base, key, model


def call_llm(*, messages: list[dict[str, Any]], task: str = "", temperature: float | None = None,
             max_tokens: int | None = None, model: str | None = None, provider: str | None = None,
             timeout: float | None = None, reasoning_effort: str | None = None,
             route_info: dict | None = None, **_ignored: Any) -> Any:
    """One chat completion; returns an object with ``choices[0].message.content`` and ``finish_reason``."""
    if BRIDGE is not None:
        result = BRIDGE.complete({"task": task, "messages": messages, "max_tokens": max_tokens,
                                  "temperature": temperature, "reasoning_effort": reasoning_effort, "model": model},
                                 float(timeout or DEFAULT_TIMEOUT_S))
        if isinstance(route_info, dict):
            route_info.update(provider="eve", model=result.get("model") or "eve-agent-model")
        return _response(result.get("content") or "", result.get("finish_reason") or "stop",
                         result.get("model") or "eve-agent-model", result.get("usage") or {})
    cli = (os.environ.get("LCMX_LLM_CLI") or "").strip()
    if cli:
        content = _call_cli(cli, messages, model or os.environ.get("LCMX_LLM_MODEL") or "", reasoning_effort,
                            float(timeout or DEFAULT_TIMEOUT_S))
        if isinstance(route_info, dict):
            route_info.update(provider=cli, model=model or os.environ.get("LCMX_LLM_MODEL") or "cli-default")
        return _response(content, "stop", model or cli, {})
    base, key, default_model = _settings()
    if not base:
        raise AuxiliaryLLMError("no LLM route: no Eve bridge attached and LCMX_LLM_BASE_URL not set")
    if not key:
        raise AuxiliaryLLMError("no API key for the LCM-X auxiliary model (LCMX_LLM_API_KEY_ENV)")
    chosen = (model or default_model).strip()
    reasoning = chosen.startswith(REASONING_PREFIXES)
    body: dict[str, Any] = {"model": chosen, "messages": messages}
    if max_tokens:
        body["max_completion_tokens" if reasoning else "max_tokens"] = int(max_tokens)
    if temperature is not None and not reasoning:
        body["temperature"] = temperature
    if reasoning_effort and reasoning:
        body["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(
        f"{base}/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=float(timeout or DEFAULT_TIMEOUT_S)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise AuxiliaryLLMError(f"auxiliary model HTTP {error.code}: {detail}") from None
    if isinstance(route_info, dict):
        route_info.update(provider=provider or "openai-compatible", model=payload.get("model", chosen))
    choices = payload.get("choices") or []
    if not choices:
        raise AuxiliaryLLMError("auxiliary model returned no choices")
    first = choices[0]
    return _response((first.get("message") or {}).get("content") or "", first.get("finish_reason"),
                     payload.get("model", chosen), payload.get("usage") or {})


def _prompt(messages: list[dict[str, Any]]) -> str:
    """One prompt for a CLI: the system text first, then each message under its role. The task's own envelope (for
    example LCM-X's summary nonce) stays word for word."""
    parts = ["Answer the request below directly in your reply. Do not run commands, read files or call tools."]
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        parts.append(f"=== {str(message.get('role') or 'user').upper()} ===\n{text}")
    return "\n\n".join(parts)


def _call_cli(cli: str, messages: list[dict[str, Any]], model: str, effort: str | None, timeout: float) -> str:
    """One run of a subscription-backed coding CLI; returns its final message."""
    with tempfile.TemporaryDirectory(prefix="lcmx-llm-") as folder:
        out = os.path.join(folder, "answer.txt")
        if cli == "codex":
            command = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--ephemeral",
                       "--ignore-user-config", "--ignore-rules", "-C", folder, "-o", out]
            if model:
                command += ["-m", model]
            if effort:
                command += ["-c", f'model_reasoning_effort="{effort}"']
            command.append("-")
        elif cli == "claude":
            command = ["claude", "-p", "--output-format", "text", "--tools", ""]
            if model:
                command += ["--model", model]
        else:
            raise AuxiliaryLLMError(f"unknown LCMX_LLM_CLI: {cli}")
        try:
            run = subprocess.run(command, input=_prompt(messages), capture_output=True, text=True, timeout=timeout,
                                 cwd=folder)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AuxiliaryLLMError(f"{cli}: {type(error).__name__}") from None
        if run.returncode != 0:
            raise AuxiliaryLLMError(f"{cli} exited {run.returncode}: {run.stderr.strip()[-300:]}")
        if cli == "codex":
            try:
                with open(out, encoding="utf-8") as handle:
                    return handle.read()
            except OSError:
                raise AuxiliaryLLMError("codex wrote no final message") from None
        return run.stdout


def _response(content: str, finish_reason: Any, model: str, usage: dict) -> Any:
    """The OpenAI-shaped response object LCM-X reads."""
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content, role="assistant"))],
        model=model, usage=SimpleNamespace(prompt_tokens=usage.get("prompt_tokens", 0),
                                           completion_tokens=usage.get("completion_tokens", 0),
                                           total_tokens=usage.get("total_tokens", 0)))
