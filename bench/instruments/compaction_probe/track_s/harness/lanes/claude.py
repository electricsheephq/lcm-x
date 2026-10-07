"""claude -p lane (optional): isolated CLAUDE_CONFIG_DIR, scrubbed CLAUDE*/ANTHROPIC* env, keychain OAuth token
passed only in the child's env (never printed/logged/written) — the `reference_claude_p_automation_auth` pattern."""
from __future__ import annotations

import getpass
import json
import os
import subprocess
import time
from pathlib import Path

from . import CALL_TIMEOUT_S, Lane, LaneError, Reply, flat_prompt, scratch_dir

MODEL = "claude-opus-5-5"
def _binary() -> str:
    """Use the caller's CLI selection or its PATH executable."""
    return os.environ.get("CLAUDE_BIN") or "claude"


def _oauth_token() -> str:
    raw = subprocess.run(["security", "find-generic-password", "-s", "Claude Code-credentials",
                          "-a", getpass.getuser(), "-w"], capture_output=True, text=True, timeout=20)
    if raw.returncode != 0:
        raise LaneError("keychain item 'Claude Code-credentials' not readable")
    try:
        return json.loads(raw.stdout)["claudeAiOauth"]["accessToken"]
    except (ValueError, KeyError, TypeError):
        raise LaneError("keychain item has no claudeAiOauth.accessToken")


class ClaudeLane(Lane):
    name = "claude"

    def __init__(self):
        self.cfg = Path(os.environ["TRACK_S_OUT"]).resolve() / "scratch" / "claude-config"
        self.cfg.mkdir(parents=True, exist_ok=True)
        (self.cfg / "settings.json").write_text("{}\n")
        self.bin = _binary()

    def _call(self, system, user_parts, max_tokens) -> Reply:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC"))}
        env["CLAUDE_CONFIG_DIR"] = str(self.cfg)
        env["CLAUDE_CODE_OAUTH_TOKEN"] = _oauth_token()
        cmd = [self.bin, "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config",
               "--no-session-persistence", "--output-format", "json"]
        t0 = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=scratch_dir(self.name), input=flat_prompt(system, user_parts),
                               capture_output=True, text=True, env=env, timeout=CALL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise LaneError(f"timeout after {CALL_TIMEOUT_S}s", CALL_TIMEOUT_S)
        dt = round(time.monotonic() - t0, 2)
        try:
            out = json.loads(p.stdout)
        except ValueError:
            raise LaneError(f"claude exit {p.returncode}: {(p.stderr or p.stdout)[-300:]}", dt)
        if p.returncode != 0 or out.get("is_error"):
            raise LaneError(f"claude exit {p.returncode}: {str(out.get('result'))[:300]}", dt)
        return Reply(out.get("result") or "", {
            "lane_model": MODEL, "latency_s": dt, "cli": self.bin, "max_tokens_enforced": False,
            "usage": out.get("usage"), "models_used": list((out.get("modelUsage") or {}).keys())})
