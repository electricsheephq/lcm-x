#!/usr/bin/env python3
"""Smoke the packaged Claude Code unit (docs/claude-code-adapter.md, "Release mechanics").

Build one ref, capture a synthetic transcript through the installed hook and worker,
then recall its marker through the installed MCP entry. No Hermes or pip is needed.
Only a fresh temporary directory is written; --keep preserves it for diagnosis.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import selectors
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {"lcm_recall", "lcm_grep", "lcm_expand", "lcm_load_session", "lcm_describe"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def timings(runs: list[tuple[str, float]]) -> dict[str, dict]:
    """Nearest-rank percentiles of the unrounded samples per event, in milliseconds. The
    session's first hook run (a cold interpreter start) is reported apart, as ``first``, and
    left out of its event's percentiles."""
    (first_event, first_ms), rest = runs[0], runs[1:]
    samples: dict[str, list[float]] = {}
    for event, value in rest:
        samples.setdefault(event, []).append(value)
    result = {"first": {"event": first_event, "n": 1, "max": round(first_ms, 1)}}
    for event, values in samples.items():
        ordered = sorted(values)
        result[event] = {"n": len(ordered), "p50": round(ordered[math.ceil(len(ordered) * 0.50) - 1], 1),
                         "p95": round(ordered[math.ceil(len(ordered) * 0.95) - 1], 1),
                         "max": round(ordered[-1], 1)}
    return result


def environment(home: Path, runtime: str) -> dict[str, str]:
    """An allowlist, including bootstrap children: never inherit host configuration."""
    env = {
        "PATH": os.pathsep.join((str(home / "bin") if runtime == "found"
                                 else str(Path(sys.executable).parent), os.defpath)),
        "HOME": str(home / "user"),
        "HERMES_HOME": str(home / "user/.hermes"),
        "CLAUDE_CONFIG_DIR": str(home / "user/.claude"),
        "LCMX_CC_HOME": str(home / "store"),
        "CLAUDE_PLUGIN_ROOT": str(home / "unpacked"),
        "TMPDIR": str(home / "tmp"),
    }
    if runtime == "venv":
        env["LCMX_PYTHON"] = str(home / "venv/bin/python")
    return env


class Smoke:
    def __init__(self, home: Path, ref: str, runtime: str) -> None:
        self.home, self.ref = home, ref
        self.runtime_mode = runtime
        self.env = environment(home, runtime)
        self.runtime_command: list[str] = []
        self.found_name = f"python3.{sys.version_info.minor}"
        self.hook_events: list[str] = []
        self.deadline = time.monotonic() + 50
        self.summary = {"pass": False, "ref": ref, "runtime": runtime,
                        "core_commit": None, "unit": None, "steps": [],
                        "hook_wall_ms": {}, "hook_in_process_ms": {}}
        self.current = "setup"
        self.stderr = ""

    def remaining(self, limit: float) -> float:
        left = min(limit, self.deadline - time.monotonic())
        if left <= 0:
            raise TimeoutError("smoke deadline exceeded")
        return left

    @contextmanager
    def step(self, name: str):
        self.current = name
        started = time.monotonic()
        result = {"step": name, "outcome": "fail"}
        self.summary["steps"].append(result)
        try:
            yield
            result["outcome"] = "pass"
        finally:
            result["ms"] = round((time.monotonic() - started) * 1000, 1)

    def run(self, command: list[str], *, timeout: float = 10, payload: str | None = None,
            accepted: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
        done = subprocess.run(command, input=payload, capture_output=True, text=True,
                              env=self.env, cwd=self.home, timeout=self.remaining(timeout))
        self.stderr = done.stderr
        require(done.returncode in accepted, f"child exited {done.returncode}")
        return done

    def build(self) -> Path:
        commit = self.run(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{self.ref}^{{commit}}"])
        self.summary["core_commit"] = commit.stdout.strip()
        # Keep the builder's git children inside the same clean environment and timeout.
        code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                "from build_claude_code_unit import build; print(build(sys.argv[2], Path(sys.argv[3])))")
        built = self.run([sys.executable, "-B", "-c", code, str(ROOT / "scripts"),
                          self.ref, str(self.home / "asset")])
        asset = Path(built.stdout.strip())
        digest, name = asset.with_name(asset.name + ".sha256").read_text().split()
        require(name == asset.name and digest == hashlib.sha256(asset.read_bytes()).hexdigest(),
                "asset checksum mismatch")
        unpacked = self.home / "unpacked"
        unpacked.mkdir()
        root = unpacked.resolve()  # a temporary directory can sit behind a symlink (/var on macOS)
        with tarfile.open(asset) as archive:
            # The builder emits regular files only; refuse paths escaping this fresh root. The
            # data filter refuses them again during extraction (Python 3.11.4 and newer).
            require(hasattr(tarfile, "data_filter"), "this smoke needs Python 3.11.4 or newer")
            for member in archive.getmembers():
                require(member.isfile() and (root / member.name).resolve().is_relative_to(root),
                        "unsafe archive member")
            archive.extractall(unpacked, filter="data")
        plugin = unpacked / asset.name.removesuffix(".tar.gz")
        bundle = json.loads((plugin / "bundle.json").read_text())
        require(bundle["core_commit"] == self.summary["core_commit"] and bundle["ref"] == self.ref,
                "bundle does not name the requested commit")
        manifest = json.loads((plugin / ".claude-plugin/plugin.json").read_text())
        require(bundle["unit"] == manifest["version"], "bundle unit differs from plugin version")
        core_version = next(line.split(":", 1)[1].strip()
                            for line in (plugin / "lcm-x/plugin.yaml").read_text().splitlines()
                            if line.startswith("version:"))
        require(bundle["core_version"] == core_version, "bundle core version differs from plugin.yaml")
        self.summary["unit"] = bundle["unit"]
        self.env["CLAUDE_PLUGIN_ROOT"] = str(plugin)
        return plugin

    def runtime(self) -> None:
        if self.runtime_mode == "venv":
            self.run([sys.executable, "-m", "venv", "--without-pip", str(self.home / "venv")])
            self.runtime_command = [self.env["LCMX_PYTHON"]]
        else:
            require(not (self.home / "venv").exists() and "LCMX_PYTHON" not in self.env,
                    "found runtime must have no venv or LCMX_PYTHON")
            self.runtime_command = [str(self.home / "bin" / self.found_name), "-I", "-S"]
            self.run([*self.runtime_command, "-c",
                      f"import sys; assert sys.version_info[:2] == {sys.version_info[:2]!r}; "
                      "assert sys.version_info >= (3, 11) and sys.flags.isolated and sys.flags.no_site"])
        self.run([*self.runtime_command, "-c",
                  "import importlib.util; "
                  "assert all(importlib.util.find_spec(n) is None for n in ('agent', 'fastembed', 'pip'))"])
        require(not Path(self.env["HERMES_HOME"]).exists(), "Hermes home must remain absent")

    def transcript(self, marker: str) -> dict:
        session = str(uuid.uuid4())
        path = Path(self.env["CLAUDE_CONFIG_DIR"]) / "projects/-smoke" / f"{session}.jsonl"
        path.parent.mkdir(parents=True)
        contents = [
            ("user", marker),
            ("assistant", [{"type": "tool_use", "id": "toolu_smoke", "name": "Read", "input": {}}]),
            ("user", [{"type": "tool_result", "tool_use_id": "toolu_smoke", "content": marker}]),
            ("assistant", [{"type": "text", "text": marker}]),
        ]
        entries = [{"type": kind, "uuid": str(uuid.uuid4()), "sessionId": session,
                    "timestamp": "2026-11-01T00:00:00.000Z", "message": {"role": kind, "content": content}}
                   for kind, content in contents]
        path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
        return {"session_id": session, "transcript_path": str(path), "cwd": str(self.home)}

    def hooks(self, plugin: Path, payload: dict) -> None:
        wall: list[tuple[str, float]] = []
        for event, count in (("SessionStart", 1), ("UserPromptSubmit", 1), ("PostToolBatch", 20), ("Stop", 1)):
            with self.step(event):
                registered = json.loads((plugin / "hooks/hooks.json").read_text())["hooks"].get(event, [])
                hooks = [hook for group in registered for hook in group["hooks"]]
                require(len(hooks) == 1 and hooks[0].get("type") == "command",
                        f"{event} must register exactly one command hook")
                hook = hooks[0]
                require(isinstance(hook.get("timeout"), (int, float)) and hook["timeout"] > 0,
                        f"{event} must register a positive timeout")
                body = {**payload, "hook_event_name": event}
                if event == "SessionStart":
                    body["source"] = "startup"
                raw = json.dumps(body)
                for _ in range(count):
                    started = time.monotonic()
                    done = self.run(["/bin/sh", "-c", hook["command"]], payload=raw, timeout=hook["timeout"])
                    wall.append((event, (time.monotonic() - started) * 1000))
                    self.hook_events.append(event)
                    require(not done.stdout and not done.stderr, "hook printed output")
        self.summary["hook_wall_ms"] = timings(wall)

    def hook_records(self) -> None:
        records = [json.loads(line) for line in (self.home / "store/logs/hook.log").read_text().splitlines()]
        require(len(records) == len(self.hook_events), "hook log must contain one record per hook run")
        measured: list[tuple[str, float]] = []
        for event, record in zip(self.hook_events, records):
            require(record.get("event") == event and record.get("host") == "claude-code",
                    "hook log event or host differs from the fired hook")
            outcome = record.get("outcome")
            require(outcome in {"worker_started", "worker_running"}, f"{event}: {outcome}")
            require(isinstance(record.get("ms"), (int, float)) and math.isfinite(record["ms"])
                    and record["ms"] >= 0, f"{event}: invalid in-process timing")
            measured.append((event, record["ms"]))
        self.summary["hook_in_process_ms"] = timings(measured)

    def wait_worker(self, timeout: float = 5) -> None:
        lock = self.home / "store/capture.lock"
        if not lock.exists():
            return
        end = time.monotonic() + timeout
        with lock.open("a") as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    if time.monotonic() >= end:
                        raise TimeoutError("capture worker still holds the lock")
                    time.sleep(0.05)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    return

    def wait_capture(self) -> None:
        # Fail with the hook's own token before waiting on records it could not drain.
        self.hook_records()
        spool = self.home / "store/spool"
        end = time.monotonic() + self.remaining(30)
        while True:
            require(not any(spool.glob("*.failed")), "hook-started worker left a failed spool record")
            if not any(spool.glob("*.json")) and not any(spool.glob("*.claimed")):
                try:
                    self.wait_worker(0)
                except TimeoutError:
                    pass
                else:
                    # A worker may release its lock and rescan; check pending work again.
                    if not any(spool.glob("*.json")) and not any(spool.glob("*.claimed")):
                        require(not any(spool.glob("*.failed")), "hook-started worker left a failed spool record")
                        self.hook_records()
                        return
            require(time.monotonic() < end, "hook-started worker did not drain before the capture deadline")
            time.sleep(min(0.05, self.remaining(0.05)))

    def recall(self, plugin: Path, marker: str) -> None:
        server = json.loads((plugin / ".mcp.json").read_text())["mcpServers"]["lcm-x"]
        command = [server["command"], *[arg.replace("${CLAUDE_PLUGIN_ROOT}", str(plugin))
                                          for arg in server["args"]]]
        with (self.home / "mcp-stderr.log").open("w+") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=stderr, env=self.env, cwd=self.home, bufsize=0)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    buffer = bytearray()

                    def send(method: str, params: dict, ident: int | None = None):
                        request = {"jsonrpc": "2.0", "method": method, "params": params}
                        if ident is not None:
                            request["id"] = ident
                        process.stdin.write((json.dumps(request) + "\n").encode())

                    def call(method: str, params: dict, ident: int) -> dict:
                        send(method, params, ident)
                        end = time.monotonic() + self.remaining(15)
                        while b"\n" not in buffer:
                            left = end - time.monotonic()
                            require(left > 0 and bool(selector.select(max(0, left))), "MCP reply timed out")
                            chunk = os.read(process.stdout.fileno(), 65536)
                            require(bool(chunk), "MCP closed stdout before reply")
                            buffer.extend(chunk)
                            require(len(buffer) <= 1024 * 1024, "MCP reply exceeded smoke limit")
                        line, _, rest = buffer.partition(b"\n")
                        buffer[:] = rest
                        reply = json.loads(line)
                        require(reply.get("jsonrpc") == "2.0" and reply.get("id") == ident
                                and "error" not in reply, "MCP returned an invalid or error reply")
                        return reply["result"]

                    def tool(name: str, arguments: dict, ident: int) -> dict:
                        result = call("tools/call", {"name": name, "arguments": arguments}, ident)
                        require(not result.get("isError"), f"{name} returned an error")
                        return json.loads(result["content"][0]["text"])

                    with self.step("initialize"):
                        initialized = call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                                           "clientInfo": {"name": "lcmx-smoke", "version": "1"}}, 1)
                        require(initialized["serverInfo"]["name"] == "lcm-x", "unexpected MCP server")
                        send("notifications/initialized", {})
                    with self.step("tools/list"):
                        listed = call("tools/list", {}, 2)["tools"]
                        require(len(listed) == 5 and {item["name"] for item in listed} == TOOLS,
                                "MCP did not list the five recall tools")
                    with self.step("lcm_grep"):
                        found = tool("lcm_grep", {"query": marker, "mode": "full_text"}, 3)
                        # FTS wraps individual query terms in >>>...<<<, including the
                        # hyphen-separated parts of our marker. Compare its plain snippet.
                        hits = [hit for hit in found["results"] if hit.get("store_id") is not None
                                and hit.get("role") == "user"
                                and marker in hit.get("snippet", "").replace(">>>", "").replace("<<<", "")]
                        require(bool(hits), "grep did not find the marker")
                    with self.step("lcm_expand"):
                        expanded = tool("lcm_expand", {"store_id": hits[0]["store_id"]}, 4)
                        require(expanded["content"] == marker and not expanded["content_truncated"]
                                and not expanded["has_more"], "expand did not return the full stored marker")
                    with self.step("MCP exit"):
                        process.stdin.close()
                        require(process.wait(timeout=self.remaining(5)) == 0, "MCP exited unsuccessfully")
                        require(not buffer and not process.stdout.read(1), "MCP printed unsolicited output")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
                process.stdin.close()
                process.stdout.close()
                stderr.seek(0)
                self.stderr = stderr.read()[-2000:]

    def failure(self, exc: Exception) -> None:
        tails = [self.stderr]
        if isinstance(exc, subprocess.TimeoutExpired) and exc.stderr:
            tails.append(exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr)
        for log in (self.home / "store/logs").glob("*.log"):
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 1000))
                tails.append(handle.read().decode(errors="replace"))
        self.summary["failure"] = {"step": self.current, "error": f"{type(exc).__name__}: {exc}",
                                   "stderr_tail": "\n".join(tails)[-3000:].replace(str(self.home), "<tmp>")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--runtime", choices=("venv", "found"), default="venv")
    parser.add_argument("--keep", action="store_true", help="keep the fresh temporary directory")
    args = parser.parse_args(argv)
    home = Path(tempfile.mkdtemp(prefix="smoke-"))
    smoke = Smoke(home, args.ref, args.runtime)
    try:
        with smoke.step("setup"):
            (home / "user").mkdir()
            (home / "tmp").mkdir()
            if args.runtime == "found":
                (home / "bin").mkdir()
                (home / "bin" / smoke.found_name).symlink_to(sys.executable)
        with smoke.step("build and unpack"):
            plugin = smoke.build()
        with smoke.step("fresh runtime"):
            smoke.runtime()
        with smoke.step("synthetic session"):
            marker = "lcmx-smoke-" + uuid.uuid4().hex
            payload = smoke.transcript(marker)
        smoke.hooks(plugin, payload)
        with smoke.step("hook worker capture"):
            smoke.wait_capture()
        smoke.recall(plugin, marker)
        require(not Path(smoke.env["HERMES_HOME"]).exists(), "smoke created a Hermes home")
        smoke.summary["pass"] = True
    except Exception as exc:  # noqa: BLE001 - the CLI always returns one JSON result
        smoke.failure(exc)
    finally:
        try:
            smoke.wait_worker()
        except Exception as exc:  # noqa: BLE001 - still attempt cleanup after a failed wait
            smoke.summary["pass"] = False
            smoke.current = "worker exit"
            smoke.failure(exc)
        try:
            if args.keep:
                smoke.summary["temp_dir"] = str(home)
            else:
                shutil.rmtree(home)
        except Exception as exc:  # noqa: BLE001 - cleanup failure also fails the gate
            smoke.summary["pass"] = False
            smoke.current = "cleanup"
            smoke.failure(exc)
    print(json.dumps(smoke.summary, sort_keys=True))
    return 0 if smoke.summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
