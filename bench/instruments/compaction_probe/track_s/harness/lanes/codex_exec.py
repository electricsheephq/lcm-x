"""codex exec lane (spec Track P lanes 1-2): exact command shape, run in an EMPTY scratch dir."""
from __future__ import annotations

import shutil
import subprocess
import time
import uuid

from . import CALL_TIMEOUT_S, Lane, LaneError, Reply, flat_prompt, scratch_dir


class CodexLane(Lane):
    def __init__(self, name: str, model: str):
        self.name, self.model = name, model

    def _call(self, system, user_parts, max_tokens) -> Reply:
        base = scratch_dir(self.name) / uuid.uuid4().hex[:12]
        (work := base / "work").mkdir(parents=True)  # stays empty: the model's cwd
        prompt, out = base / "prompt.txt", base / "last-message.txt"
        prompt.write_text(flat_prompt(system, user_parts))
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-m", self.model,
               "-c", "model_reasoning_effort=medium", "--output-last-message", str(out), "-"]
        t0 = time.monotonic()
        try:
            with prompt.open() as stdin:
                p = subprocess.run(cmd, cwd=work, stdin=stdin, capture_output=True, text=True,
                                   timeout=CALL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise LaneError(f"timeout after {CALL_TIMEOUT_S}s", CALL_TIMEOUT_S)
        dt = round(time.monotonic() - t0, 2)
        text = out.read_text() if out.exists() else ""
        meta = {"lane_model": self.model, "latency_s": dt, "returncode": p.returncode,
                "max_tokens_enforced": False}
        if p.returncode != 0:
            tail = (p.stderr or "").strip().splitlines()[-3:]
            raise LaneError(f"codex exit {p.returncode}: {' | '.join(tail)[:400]}", dt)
        shutil.rmtree(base, ignore_errors=True)
        return Reply(text, meta)
