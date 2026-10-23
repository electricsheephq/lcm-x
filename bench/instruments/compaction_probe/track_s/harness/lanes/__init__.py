"""Lanes: `lane.call(system|None, user_parts, max_tokens|None) -> Reply`; a timeout or transport error raises
`LaneError` (never retried). `max_tokens` None = the arm sets no ceiling (omitted where a lane accepts one)."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

CALL_TIMEOUT_S = 300
D = Path(__file__).resolve().parents[2]  # <D>/eval
HEADER = "Reply with the summary only; use no tools."


@dataclass
class Reply:
    text: str
    meta: dict = field(default_factory=dict)


class LaneError(RuntimeError):
    def __init__(self, msg: str, latency_s: float | None = None):
        super().__init__(msg)
        self.latency_s = latency_s


def flat_prompt(system, user_parts) -> str:
    """Codex / claude CLI lanes: header line, then system + blank line + user (spec Track P lane 1)."""
    body = "\n\n".join(user_parts)
    return f"{HEADER}\n{system}\n\n{body}" if system else f"{HEADER}\n{body}"


def scratch_dir(lane: str) -> Path:
    p = Path(os.environ["TRACK_S_OUT"]).resolve() / "scratch" / lane / f"{os.getpid()}-{threading.get_ident()}"
    p.mkdir(parents=True, exist_ok=True)
    return p


LIMITS = {"codex-sol": 6, "codex-astra": 6, "glm": 4, "claude": 2}


def get_lane(name: str):
    from . import claude, codex_exec, glm
    lane = {"codex-sol": lambda: codex_exec.CodexLane("codex-sol", "gpt-6.1-sol"),
            "codex-astra": lambda: codex_exec.CodexLane("codex-astra", "gpt-6-astra"),
            "glm": glm.GLMLane, "claude": claude.ClaudeLane}[name]()
    lane.sem = threading.BoundedSemaphore(LIMITS[name])
    return lane


class Lane:
    name = "?"
    sem: threading.BoundedSemaphore

    def call(self, system, user_parts, max_tokens) -> Reply:
        with self.sem:
            return self._call(system, list(user_parts), max_tokens)  # subclasses define _call
