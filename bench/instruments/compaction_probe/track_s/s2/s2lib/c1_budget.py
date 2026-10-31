"""Opt-in C1 account throttle shared by summary and reader subprocesses.

An exclusive advisory lock deliberately serializes this lane's calls (<=2).
Persist only counters, timestamps and classifications, never provider output.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import selectors
import shutil
import subprocess
import time
from pathlib import Path

LIMIT = re.compile(r"429|usage.?limit|rate.?limit|too many requests", re.I)


def rate_limits():
    """Read account/rateLimits/read; expose window counters only."""
    proc = subprocess.Popen(["codex", "app-server", "--listen", "stdio://"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def send(value):
        proc.stdin.write(json.dumps(value) + "\n")
        proc.stdin.flush()
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    send(dict(id=1, method="initialize", params=dict(clientInfo=dict(name="c1-counter-reader", version="1"))))
    deadline = time.monotonic() + 40
    try:
        while time.monotonic() < deadline:
            if not selector.select(1):
                continue
            line = proc.stdout.readline()
            if not line:
                break
            value = json.loads(line)
            if value.get("id") == 1:
                if "error" in value:
                    break
                send(dict(method="initialized"))
                send(dict(id=2, method="account/rateLimits/read"))
            if value.get("id") == 2:
                if "error" in value:
                    break
                limits = value["result"].get("rateLimits", {})
                windows = [limits.get(k) for k in ("primary", "secondary")]
                counters = [{k: w.get(k) for k in ("usedPercent", "windowDurationMins", "resetsAt")}
                            for w in windows if w]
                weekly = [w["usedPercent"] for w in counters if w["windowDurationMins"] == 10080]
                if not weekly:
                    raise SystemExit("C1 STOP: weekly rate counter unavailable")
                return dict(windows=counters, weekly_used=max(weekly), read_at=time.time())
        raise SystemExit("C1 STOP: account/rateLimits/read unavailable")
    finally:
        selector.close()
        proc.terminate()
        proc.wait(timeout=5)


def _save(path, state):
    path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")


def _pause(state, path):
    while state.get("pause_until", 0) > time.time():
        left = state["pause_until"] - time.time()
        print(f"C1 throttle: paused ({int(left)}s remain)", flush=True)
        time.sleep(min(60, left))
    state.pop("pause_until", None)
    _save(path, state)


def run(cmd, **kwargs):
    root = os.environ.get("C1_BUDGET_ROOT")
    if not root:
        return subprocess.run(cmd, **kwargs)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "counters.json"
    with (root / "call.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else dict(calls=0, consecutive_blocks=0)
        if state.get("terminal") or state["consecutive_blocks"] >= 3:
            raise SystemExit("C1 STOP: three consecutive provider blocks")
        if shutil.disk_usage(root).free < 60 * 1024**3:
            raise SystemExit("C1 STOP: free disk below 60 GB")
        _pause(state, path)
        if state["calls"] % 50 == 0 and state.get("rate_checked_at_call") != state["calls"]:
            receipt = rate_limits()
            state.setdefault("rate_reads", []).append(dict(call=state["calls"], **receipt))
            state["rate_checked_at_call"] = state["calls"]
            if receipt["weekly_used"] >= 85:
                state["pause_until"] = time.time() + 600
                _save(path, state)
                _pause(state, path)
        state["calls"] += 1
        state["in_flight"] = True
        _save(path, state)
        blocked, limited = False, False
        try:
            result = subprocess.run(cmd, **kwargs)
            blocked = result.returncode != 0
            limited = bool(LIMIT.search((result.stderr or "") + (result.stdout or ""))) if blocked else False
            return result
        except subprocess.TimeoutExpired:
            blocked = True
            raise
        finally:
            state["in_flight"] = False
            state["consecutive_blocks"] = state["consecutive_blocks"] + 1 if blocked else 0
            state.setdefault("outcomes", []).append(dict(call=state["calls"], blocked=blocked, limited=limited))
            if limited:
                state["pause_until"] = time.time() + 600
            if state["consecutive_blocks"] >= 3:
                state["terminal"] = True
            _save(path, state)
            if state.get("terminal"):
                raise SystemExit("C1 STOP: three consecutive provider blocks")


@contextlib.contextmanager
def budget_env(root):
    """Small test helper; real runs supply the environment at launch."""
    saved = os.environ.get("C1_BUDGET_ROOT")
    os.environ["C1_BUDGET_ROOT"] = str(root)
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("C1_BUDGET_ROOT", None)
        else:
            os.environ["C1_BUDGET_ROOT"] = saved
