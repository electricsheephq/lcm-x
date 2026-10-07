"""GLM lane (spec Track P lane 3): z.ai coding chat completions; key read in-process, never emitted."""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request

from . import CALL_TIMEOUT_S, Lane, LaneError, Reply

URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
MODEL = "glm-5.3"


def _key() -> str:
    value = os.environ.get("GLM_API_KEY")
    if not value:
        raise LaneError("GLM_API_KEY is required in the environment")
    return value


class GLMLane(Lane):
    name = "glm"

    def __init__(self):
        self._k = _key()

    def _call(self, system, user_parts, max_tokens) -> Reply:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": u} for u in user_parts]
        body = {"model": MODEL, "messages": msgs}
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self._k})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=CALL_TIMEOUT_S) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise LaneError(f"HTTP {e.code}: {detail}", round(time.monotonic() - t0, 2))
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise LaneError(f"{type(e).__name__}: {e}"[:300], round(time.monotonic() - t0, 2))
        dt = round(time.monotonic() - t0, 2)
        try:
            choice, msg = data["choices"][0], data["choices"][0].get("message") or {}
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LaneError(f"malformed response: {json.dumps(data)[:300]}", dt)
        return Reply(msg.get("content") or "", {
            "lane_model": data.get("model", MODEL), "latency_s": dt, "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"), "reasoning_chars": len(msg.get("reasoning_content") or ""),
            "max_tokens_sent": body.get("max_tokens"), "max_tokens_enforced": bool(max_tokens)})
