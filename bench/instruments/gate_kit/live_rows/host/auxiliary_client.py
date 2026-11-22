"""Stand-in for the host's auxiliary model client (live-rows producer, #2868 requirement 1).

The product calls ``call_llm(task=..., messages=..., max_tokens=..., timeout=...)`` and, where the host exports
it, runs the call inside ``aux_stream_deadline(deadline)``. This stand-in keeps that boundary and nothing else:

- latency = max(floor_s, input_tokens / rate_tps), from the kit (``configure``);
- the call is given min(timeout, stream deadline left); when the latency exceeds it, the stand-in sleeps exactly
  that long and raises ``TimeoutError("Request timed out.")`` (a cut), as a real route would;
- otherwise it answers inside the product's nonce envelope with content-free filler shorter than its source.

Every call is appended to ``CALLS`` (times, tokens, granted seconds, outcome; never content). ``HOOKS`` lets a
runner observe or hold a call (the L1 session-switch check holds the call open on an event).
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

SETTINGS: dict[str, Any] = {"floor_s": 44.4, "rate_tps": 2970.0, "ratio": 0.1, "min_tokens": 64,
                            "max_tokens_out": 3200, "chars_per_token": 4.0, "force_cut": False}
CALLS: list[dict[str, Any]] = []
HOOKS: dict[str, Callable[[dict[str, Any]], None]] = {}
_LOCK = threading.Lock()
_DEADLINE = threading.local()
_NONCE = re.compile(r'<lcm-summary nonce="([0-9a-f]+)">')
_WORDS = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
          "et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris").split()


def configure(**values: Any) -> None:
    unknown = set(values) - set(SETTINGS)
    if unknown:
        raise ValueError(f"unknown stand-in settings: {sorted(unknown)}")
    SETTINGS.update(values)


def latency_s(input_tokens: int) -> float:
    return max(float(SETTINGS["floor_s"]), input_tokens / float(SETTINGS["rate_tps"]))


def _tokens(text: str) -> int:
    return int(len(text) / float(SETTINGS["chars_per_token"]))


@contextlib.contextmanager
def aux_stream_deadline(deadline: float | None):
    previous = getattr(_DEADLINE, "value", None)
    _DEADLINE.value = deadline
    try:
        yield
    finally:
        _DEADLINE.value = previous


def _current_aux_stream_deadline() -> float | None:
    return getattr(_DEADLINE, "value", None)


def filler(tokens: int, serial: int) -> str:
    words, i = [], 0
    target = max(1, int(tokens * float(SETTINGS["chars_per_token"])))
    while sum(len(w) + 1 for w in words) < target:
        words.append(_WORDS[(i * 7 + serial) % len(_WORDS)])
        i += 1
    return " ".join(words)


def call_llm(task: str = "", messages: list[dict[str, Any]] | None = None, max_tokens: int = 0,
             timeout: float | None = None, route_info: dict | None = None, **_: Any) -> Any:
    messages = messages or []
    text = "\n".join(str(m.get("content") or "") for m in messages)
    source = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
    started = time.monotonic()
    input_tokens = _tokens(text)
    want = latency_s(input_tokens)
    deadline = _current_aux_stream_deadline()
    granted = [x for x in (timeout, None if deadline is None else deadline - started) if x is not None]
    limit = max(0.0, min(granted)) if granted else None
    with _LOCK:
        serial = len(CALLS) + 1
        record = {"serial": serial, "task": task, "t_start": started, "input_tokens": input_tokens,
                  "latency_s": round(want, 3), "timeout": timeout,
                  "deadline_left": None if deadline is None else round(deadline - started, 3),
                  "granted_s": None if limit is None else round(limit, 3), "thread": threading.get_ident()}
        CALLS.append(record)
    hook = HOOKS.get("enter")
    if hook:
        hook(record)
    cut = bool(SETTINGS["force_cut"]) or (limit is not None and want > limit)
    time.sleep(limit if cut and limit is not None else (0.0 if cut else want))
    record["t_end"] = time.monotonic()
    record["slept_s"] = round(record["t_end"] - started, 3)
    record["outcome"] = "cut" if cut else "ok"
    hook = HOOKS.get("exit")
    if hook:
        hook(record)
    if route_info is not None:
        route_info.update({"provider": "live-rows-stand-in", "model": "stand-in"})
    if cut:
        raise TimeoutError("Request timed out.")
    match = _NONCE.search(text)
    out_tokens = max(int(SETTINGS["min_tokens"]),
                     min(int(SETTINGS["max_tokens_out"]), int(_tokens(source) * float(SETTINGS["ratio"]))))
    if max_tokens:
        out_tokens = min(out_tokens, max(16, int(max_tokens) // 2))
    body = (f"Stand-in summary {serial} of earlier turns: " + filler(out_tokens, serial)
            + "\nExpand for details about: earlier turns of this conversation")
    content = f'<lcm-summary nonce="{match.group(1)}">\n{body}\n</lcm-summary>' if match else body
    record["output_tokens"] = _tokens(content)
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
