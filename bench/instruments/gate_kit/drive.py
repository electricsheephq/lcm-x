"""Gate-kit driver: the v0.30.0 driven run (docs/exactly-once-ingest.md section 5, r3.8, "The driven run") against a
prepared Hermes home, writing the events log that ``audit.py --events`` reads.

    python -m bench.instruments.gate_kit.drive --hermes-python <python> --hermes-src <host src> --home <HERMES_HOME>
        --workdir <dir> --events <events.jsonl> [--turns N] [--children N] [--compactions N] [--age-turns N]
        [--seed 2055] [--max-hours 6] [--dry-run]

Two host entry points, both real processes over newline-delimited JSON-RPC on stdio:
- ``hermes acp`` (acp_driver.AcpProcess): turns, repeated tool calls, ``/compress``, delegate prompts,
  cancel-and-resend (``session/cancel`` while the prompt is in flight) and cold restarts (SIGKILL the process group,
  start a new process, ``session/load``). ACP has no session end and no undo, so it cannot do the rest.
- the TUI gateway, ``python -m tui_gateway.entry`` (the Desktop/TUI backend): warm restarts (``session.close``, which
  ends the session through ``commit_memory_session``, then ``session.resume`` in the same process), undo-and-resend
  (``session.undo``, then the same prompt again) and the second host source (its sessions are ``tui``, not ``acp``).

The model is whatever the home is configured for: the driver never reads, prints or edits keys or model config. It
reports the context window the host says it uses (ACP ``/context``). The events log holds kinds, session ids, prompt
bank indices and stop reasons only, never prompt or reply text. A failed operation is an ``error`` event and the run
goes on; nothing is retried silently. Claim class: instrument; the gate verdict comes from audit.py on the run.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bench.instruments.gate_kit import audit as AUDIT  # noqa: E402
from bench.instruments.gate_kit.audit import DELEGATE_SOURCE, FLOORS  # noqa: E402
from bench.instruments.reliability import acp_driver as AD  # noqa: E402
from bench.instruments.reliability.hosts import under_real_hermes  # noqa: E402

ACP_SLOTS, TUI_SLOTS = 6, 3
EXIT_GRACE = 60.0  # seconds a host gets to exit on stdin EOF at the end of the run
# LCM-X is not the context engine in a host process: the plugin's own notice when its load overran
# plugins.load_timeout_seconds (LCM-X __init__.py), and the host's fallback warning (agent/agent_init.py:1901). Such a
# host stores nothing in lcm.db, so its rows would read as a loss: the run stops (exit 3) before it reaches the audit.
INACTIVE = re.compile(r"LCM-X is NOT active in this process|Context engine 'lcm(?:-x)?' not found")
STOP_REASON = re.compile(r"compaction stop: reason=(\w+)")
PRODUCT_FRESH_TAIL = 24  # config.py LCMConfig.fresh_tail_count


def fresh_tail(env: dict, home: Path | None = None) -> tuple[int, str]:
    """LCM-X's effective fresh tail for the hosts and where it came from. The hosts load ``<home>/.env`` over their
    process environment at startup, so that file's LCM_FRESH_TAIL_COUNT wins. Only that one key is read; the file is
    never printed."""
    value, source = env.get("LCM_FRESH_TAIL_COUNT") or None, "env"
    dotenv = home / ".env" if home is not None else None
    if dotenv is not None and dotenv.is_file():
        for line in dotenv.read_text(errors="replace").splitlines():
            key, sep, raw = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == "LCM_FRESH_TAIL_COUNT":
                value, source = raw.strip().strip("'\"") or None, ".env"
    try:
        return int(value), source
    except (TypeError, ValueError):
        return PRODUCT_FRESH_TAIL, "product default"


def default_age_turns(env: dict, home: Path | None = None) -> int:
    """Ageing turns that put the paste outside LCM-X's fresh tail. After the paste come its reply, two rows per ageing
    turn and the triggering turn's prompt and reply, so ceil(tail / 2) turns leave at least the tail behind it."""
    return max(1, -(-fresh_tail(env, home)[0] // 2))
EXIT_LCM_INACTIVE = 3
EXIT_EVENT_LOG = 4
PLANNED_FLOORS = ("turns", "delegated_children", "compactions", "compactions_then_restart", "restarts_cold",
                  "restarts_warm", "undo_resend", "tool_calls", "tool_call_repeats")

# -- the prompt bank (fixed; a prompt is named by its index, never by its text) ------------------------------------
TOPICS = ("binary search", "a hash map", "TCP slow start", "the CAP theorem", "garbage collection", "a B-tree",
          "public-key encryption", "a mutex", "DNS resolution", "a bloom filter", "copy-on-write", "a write-ahead log",
          "consistent hashing", "an LRU cache", "a vector clock", "tail recursion", "a race condition", "backpressure",
          "idempotency", "a Merkle tree")
WORDS = ("amber", "basalt", "cobalt", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
         "lumen", "meadow", "nectar", "onyx", "prairie", "quartz", "ridge", "saffron", "tundra")
ALWAYS = " Call the tool even if you did it before in this conversation."


def _bank() -> list[tuple[str, str]]:
    bank = [("plain", f"In two sentences, what is {t}?") for t in TOPICS]
    bank += [("plain", f"Give one practical tip about {t}.") for t in TOPICS]
    bank += [("tool", f"Use the read_file tool to read {{workdir}}/notes/note-{k:02d}.txt and reply with its first "
                      f"line only." + ALWAYS) for k in range(30)]
    bank += [("tool", f"Use the todo tool to add the item 'check {w}' to the todo list, then confirm." + ALWAYS)
             for w in WORDS[:10]]
    bank += [("tool", f"Use the lcm_grep tool to search this conversation for the word '{w}' and say how many hits "
                      f"it found." + ALWAYS) for w in WORDS[10:]]
    bank += [("long", "Keep these pasted notes in mind; reply only 'noted'.\n\n{long:%d}" % k) for k in range(30)]
    bank += [("delegate", f"Use the delegate_task tool to have a subagent compute {a} + {a * 3 + 7} and reply with "
                          f"the number only. Then tell me the subagent's answer.") for a in range(11, 31)]
    bank += [("undo", f"Write a detailed 600-word explanation of {t}, with an example.") for t in TOPICS]
    return bank


BANK = _bank()
CATS = {c: [i for i, (cat, _t) in enumerate(BANK) if cat == c] for c in ("plain", "tool", "long", "delegate", "undo")}
LONG_CHARS = 9000


def long_text(k: int) -> str:
    rng = random.Random(f"long-{k}")
    return " ".join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(LONG_CHARS // 10))[:LONG_CHARS]


def prompt_text(index: int, workdir: Path) -> str:
    text = BANK[index][1].replace("{workdir}", str(workdir))
    return re.sub(r"\{long:(\d+)\}", lambda m: long_text(int(m.group(1))), text)


def write_notes(workdir: Path) -> None:
    """The files the read_file prompts name (fixed, non-sensitive text)."""
    (workdir / "notes").mkdir(parents=True, exist_ok=True)
    for k in range(30):
        (workdir / "notes" / f"note-{k:02d}.txt").write_text(f"note {k:02d}: {WORDS[k % 20]} {k * 7}\n")


# -- the planner (a pure function of the seed and the targets) -----------------------------------------------------
def margin(n: int) -> int:
    return (n * 6 + 4) // 5  # ceil(n * 1.2) in integers (200 * 1.2 is 240.00000000000003 in floats)


def targets(turns: int | None = None, children: int | None = None, compactions: int | None = None) -> dict:
    """Every section 5 floor x1.2, rounded up; optional targets may raise those floors, never lower them."""
    t = {k: margin(FLOORS[k]) for k in PLANNED_FLOORS}
    t["turns"], t["delegated_children"] = max(t["turns"], turns or 0), max(t["delegated_children"], children or 0)
    t["compactions"] = max(t["compactions"], compactions or 0)
    return t


def spread(n: int, slots: list[str]) -> list[str]:
    return [slots[i % len(slots)] for i in range(n)]


def plan(seed: int, tg: dict) -> list[dict]:
    """Ops in execution order: ``{"op", "slot", "prompt"?}``. Slots ``A*`` are ACP sessions, ``T*`` TUI sessions.
    Every restart has a compaction before it in its slot and at least one op after it."""
    rng = random.Random(seed)
    acp, tui = [f"A{i}" for i in range(1, ACP_SLOTS + 1)], [f"T{i}" for i in range(1, TUI_SLOTS + 1)]
    body = {s: [] for s in acp + tui}
    restarts = {s: [] for s in acp + tui}
    for s in spread(tg["restarts_cold"], acp):
        restarts[s].append("restart_cold")
    for s in spread(tg["restarts_warm"], tui):
        restarts[s].append("restart_warm")
    pinned = sum(len(r) for r in restarts.values())  # one compaction just before each restart
    long_ix = iter(CATS["long"] * (tg["compactions"] // len(CATS["long"]) + 1))
    for s in spread(max(0, tg["compactions"] - pinned), acp + tui):
        body[s].append({"op": "compress", "slot": s})
    for s in spread(tg["delegated_children"], acp):
        body[s].append({"op": "delegate", "slot": s, "prompt": CATS["delegate"][len(body[s]) % len(CATS["delegate"])]})
    undo, cancels = tg["undo_resend"], tg["undo_resend"] // 2  # ACP has no undo: its cancel-and-resend is separate
    for s in spread(undo, tui):
        body[s].append({"op": "undo_resend", "slot": s, "prompt": rng.choice(CATS["undo"])})
    for s in spread(cancels, acp):
        body[s].append({"op": "cancel_resend", "slot": s, "prompt": rng.choice(CATS["undo"])})
    repeats = tg["tool_call_repeats"]
    originals = max(tg["tool_calls"] - repeats, 1)
    tools = rng.sample(CATS["tool"], len(CATS["tool"]))
    tools = (tools * (originals // len(tools) + 1))[:originals]
    for s, ix in zip(spread(originals, acp + tui), tools):
        body[s].append({"op": "tool", "slot": s, "prompt": ix})
    bearing = tg["tool_calls"] + tg["delegated_children"] + tg["compactions"] + undo + cancels  # besides plain turns
    for s in spread(max(0, tg["turns"] - bearing), acp + tui):
        body[s].append({"op": "turn", "slot": s, "prompt": rng.choice(CATS["plain"])})
    for s in body:
        rng.shuffle(body[s])
    holders = [s for s in body if any(o["op"] == "tool" for o in body[s])]
    for k in range(repeats):  # a word-for-word repeat of an earlier tool prompt in the same session
        s = holders[k % len(holders)]
        first = next(i for i, o in enumerate(body[s]) if o["op"] == "tool")
        at = rng.randrange(first + 1, len(body[s]) + 1)
        body[s].insert(at, {"op": "repeat", "slot": s, "prompt": body[s][first]["prompt"]})
    seqs = {}
    for s in body:
        ops, cuts = body[s], len(restarts[s])
        seq, start = [], 0
        for r, kind in enumerate(restarts[s], 1):
            end = max(start + 1, len(ops) * r // (cuts + 1))
            seq += ops[start:end] + [{"op": "compress", "slot": s}, {"op": kind, "slot": s}]
            start = end
        seq += ops[start:] or [{"op": "turn", "slot": s, "prompt": rng.choice(CATS["plain"])}]
        seqs[s] = seq
    for s in seqs:
        for o in seqs[s]:
            if o["op"] == "compress":
                o["prompt"] = next(long_ix)
    out = []
    while any(seqs.values()):
        live = [s for s in seqs if seqs[s]]
        s = rng.choices(live, weights=[len(seqs[x]) for x in live])[0]
        out.append(seqs[s].pop(0))
    return out


def planned_counts(ops: list[dict]) -> dict:
    """What the plan should produce, in the audit's coverage names (children are the delegate prompts)."""
    kinds = Counter(o["op"] for o in ops)
    restarted, then_restart, seen = set(), 0, Counter()
    for o in reversed(ops):
        if o["op"].startswith("restart"):
            restarted.add(o["slot"])
        elif o["op"] == "compress" and o["slot"] in restarted:
            then_restart += 1
    tool_prompts = [o["prompt"] for o in ops if o["op"] in ("tool", "repeat")]
    for ix in tool_prompts:
        seen[ix] += 1
    bearing = ("turn", "tool", "repeat", "delegate", "compress", "undo_resend", "cancel_resend")
    return {"turns": sum(kinds[k] for k in bearing), "delegated_children": kinds["delegate"],
            "compactions": kinds["compress"], "compactions_then_restart": then_restart,
            "restarts_cold": kinds["restart_cold"], "restarts_warm": kinds["restart_warm"],
            "undo_resend": kinds["undo_resend"], "cancel_resend": kinds["cancel_resend"], "tool_calls": len(tool_prompts),
            "tool_call_repeats": sum(n - 1 for n in seen.values()),
            "host_sources": len({o["slot"][0] for o in ops}), "ops": dict(kinds)}


# -- events --------------------------------------------------------------------------------------------------------
class EventLogError(RuntimeError):
    """The events file cannot be written: the run stops (the operations' own error handling never catches this)."""


class Events:
    """One JSON object per line, appended and flushed per event: kinds, session ids and short non-content details."""

    def __init__(self, path: Path, append: bool = False):
        self.path, self.errors = Path(path), Counter()
        if not append and self.path.exists() and self.path.stat().st_size:
            raise ValueError("--events exists and is non-empty; use a new path or --append-events to retain it")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, session: str | None = None, detail: str = "", **extra) -> None:
        rec = {"t": time.time(), "kind": kind, "session": session, "detail": detail, **extra}
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError as exc:
            raise EventLogError(f"cannot write {self.path}: {exc}") from exc
        if kind == "error":
            self.errors[detail.split(" ", 1)[0]] += 1


# -- the TUI gateway client ----------------------------------------------------------------------------------------
class Gateway(AD.AcpProcess):
    """``python -m tui_gateway.entry``: the same newline JSON-RPC framing and process-group lifecycle as AcpProcess,
    plus ``event`` notifications (``params.type``, ``params.session_id``), of which the turn-ending ones are kept: a
    turn is over once ``message.complete`` and then a settled ``session.info`` (``running`` false) have arrived."""

    KEEP = frozenset({"gateway.ready", "message.complete", "error", "session.info"})

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.events: list[dict] = []

    def _pump(self, deadline: float) -> dict | None:
        msg = self._read(deadline)
        if msg.get("method"):
            if "id" in msg:  # a server->client request (approval, clarify): this client has no handler
                self._write({"jsonrpc": "2.0", "id": msg["id"],
                             "error": {"code": -32601, "message": "gate-kit driver has no client handler"}})
            elif msg["method"] == "event" and (p := msg.get("params") or {}).get("type") in self.KEEP and not (
                    p["type"] == "session.info" and (p.get("payload") or {}).get("running") is not False):
                self.events.append(p)
            return None
        return msg

    def call(self, method: str, params: dict, timeout: float) -> dict:
        rid, self.next_id = self.next_id, self.next_id + 1
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            msg = self._pump(deadline)
            if msg is None or msg.get("id") != rid:
                continue
            if "error" in msg:
                raise AD.JsonRpcError(method, msg["error"])
            return msg.get("result") if isinstance(msg.get("result"), dict) else {}

    def wait(self, kinds: tuple, sid: str | None, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            for i, e in enumerate(self.events):
                if e.get("type") in kinds and e.get("session_id") == sid:
                    return self.events.pop(i)
            self._pump(deadline)

    def ready(self, timeout: float) -> None:
        self.wait(("gateway.ready",), None, timeout)


# -- the executor --------------------------------------------------------------------------------------------------
def host_env(base: dict, home: Path, workdir: Path, src: Path) -> dict:
    """``base`` plus an isolated HOME/TMPDIR under the workdir; provider settings stay the home's own."""
    scratch = workdir / "host-home"
    scratch.mkdir(parents=True, exist_ok=True)
    return {**base, "HERMES_HOME": str(home), "HOME": str(scratch), "TMPDIR": str(scratch),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "HERMES_ACP_SKIP_CONFIGURED_MCP": "1",
            "HERMES_PYTHON_SRC_ROOT": str(src)}


class Driver:
    def __init__(self, *, python: str, src: Path, home: Path, workdir: Path, events: Events, env: dict,
                 prompt_timeout: float = 600.0, max_hours: float = 6.0, cancel_after: float = 2.0, wrap=(),
                 acp_factory=None, tui_factory=None, age_turns: int = 4):
        self.src, self.home, self.workdir, self.ev = Path(src), Path(home), Path(workdir), events
        self.timeout, self.cancel_after = prompt_timeout, cancel_after
        self.age_turns = age_turns
        self.stop_at = time.monotonic() + max_hours * 3600
        # Children of an earlier run in a retained home are outside this run, as the audit's --since excludes them.
        self.run_start = time.time()
        logs = self.logs = workdir / "host-logs"
        logs.mkdir(parents=True, exist_ok=True)
        self.scanned: dict[Path, int] = {}
        for path in (logs / "acp.stderr", logs / "tui.stderr", self.home / "logs" / "agent.log"):
            with contextlib.suppress(OSError):
                self.scanned[path] = path.stat().st_size
        acp_argv = [*wrap, str(Path(python).parent / "hermes"), "acp"]
        tui_env = {**env, "PYTHONPATH": str(src)}
        self.make_acp = acp_factory or (lambda: AD.AcpProcess(acp_argv, env, workdir, logs / "acp.stderr"))
        self.make_tui = tui_factory or (lambda: Gateway([*wrap, python, "-m", "tui_gateway.entry"], tui_env, src,
                                                        logs / "tui.stderr"))
        self.acp = self.tui = None
        self.loaded: set[str] = set()
        self.tui_loaded: set[str] = set()
        self.cold_pending: set[str] = set()
        self.killed_at: dict[str, float] = {}
        self.slots: dict[str, dict] = {}
        self.done, self.prompts, self.attempts = Counter(), Counter(), Counter()
        self.compactions = Counter()
        self.context = None

    # processes and sessions
    def acp_sid(self, slot: str, planned_restart: bool = False) -> str:
        if self.acp is None:
            self.acp, self.loaded = self.make_acp(), set()
            self.acp.initialize(self.timeout)
        if slot not in self.slots:
            self.slots[slot] = {"session": self.acp.new_session(self.workdir, self.timeout)}
            self.loaded.add(slot)
        elif slot not in self.loaded:  # a new process binds the session (ACP _restore from state.db)
            self.acp.load_session(self.slots[slot]["session"], self.workdir, self.timeout)
            self.loaded.add(slot)
            if slot in self.cold_pending:
                self.ev.write("restart_cold" if planned_restart else "restart_cold_shared",
                              self.slots[slot]["session"], f"slot={slot}", killed_at=self.killed_at.pop(slot, None))
                self.cold_pending.remove(slot)
        return self.slots[slot]["session"]

    def tui_rt(self, slot: str) -> str:
        if self.tui is None:
            self.tui = self.make_tui()
            self.tui_loaded = set()
            self.tui.ready(self.timeout)
        if slot not in self.slots:
            res = self.tui.call("session.create", {"cwd": str(self.workdir)}, self.timeout)
            self.slots[slot] = {"rt": res["session_id"], "session": res.get("stored_session_id") or res["session_id"]}
            self.tui_loaded.add(slot)
            self.ev.write("source2_session", self.slots[slot]["session"], f"slot={slot} entry=tui_gateway")
        elif slot not in self.tui_loaded:
            res = self.tui.call("session.resume", {"session_id": self.slots[slot]["session"], "omit_messages": True},
                                self.timeout)
            self.slots[slot].update(rt=res["session_id"],
                                    session=res.get("stored_session_id") or self.slots[slot]["session"])
            self.tui_loaded.add(slot)
            if slot in self.cold_pending:
                self.ev.write("restart_cold_shared", self.slots[slot]["session"], f"slot={slot}",
                              killed_at=self.killed_at.pop(slot, None))
                self.cold_pending.remove(slot)
        return self.slots[slot]["rt"]

    def drop(self, which: str, graceful: bool = False) -> None:
        """SIGKILL the entry point's process group (a cold restart, a dead host), or end it gracefully: stdin EOF, so
        the host ends its sessions and exits by itself (AcpProcess.close escalates after EXIT_GRACE)."""
        proc = getattr(self, which)
        if proc is not None:
            if not graceful:
                proc.kill()
                loaded = self.loaded if which == "acp" else self.tui_loaded
                self.cold_pending.update(loaded)
                killed = time.time()
                for slot in loaded:  # a sibling's restart_cold_shared waits for its reload, with the kill time
                    self.killed_at[slot] = killed
            elif getattr(proc, "process", None) is not None:
                with contextlib.suppress(OSError):
                    proc.process.stdin.close()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.process.wait(timeout=EXIT_GRACE)
            rc = proc.close()
            setattr(self, which, None)
            if graceful:
                self.ev.write("host_exit", None, f"entry={which} returncode={rc}")
        setattr(self, which, None)

    # one prompt
    def send(self, slot: str, ix: int, cancel: bool = False) -> str:
        text = prompt_text(ix, self.workdir)
        self.prompts[BANK[ix][0]] += 1
        if slot.startswith("A"):
            sid = self.acp_sid(slot)
            timer = threading.Timer(self.cancel_after, self.acp.notify, ("session/cancel", {"sessionId": sid})) \
                if cancel else None
            if timer:
                timer.start()
            try:
                _reply, stop = self.acp.prompt(sid, text, self.timeout)
            finally:
                if timer:
                    timer.cancel()
            stop = stop or "none"
        else:
            rt = self.tui_rt(slot)
            self.tui.events = [e for e in self.tui.events if e.get("session_id") != rt]  # nothing stale
            sub = self.tui.call("prompt.submit", {"session_id": rt, "text": text}, self.timeout)
            if sub.get("status") != "streaming":
                stop = f"submit_{sub.get('status')}"
            else:
                end = self.tui.wait(("message.complete", "error"), rt, self.timeout)
                stop = (end.get("payload") or {}).get("status") or "complete" if end["type"] == "message.complete" \
                    else "session_error"
                self.tui.wait(("session.info",), rt, self.timeout)  # settled: the session takes the next request
        self.ev.write("turn", self.slots[slot]["session"], f"slot={slot} prompt={ix} cat={BANK[ix][0]} stop={stop}")
        if stop not in ("end_turn", "complete", "cancelled"):
            self.ev.write("error", self.slots[slot]["session"], f"turn_{stop} slot={slot} prompt={ix}")
        return stop

    def probe_context(self, slot: str) -> None:
        """The context window the host uses for this session (ACP ``/context``); only the numbers are kept."""
        reply, _stop = self.acp.prompt(self.acp_sid(slot), "/context", self.timeout)
        m = re.search(r"/ ([\d,]+) tokens", reply)
        self.context = int(m.group(1).replace(",", "")) if m else None
        self.ev.write("context_window", self.slots[slot]["session"], f"context_length={self.context}")

    def compress_log_result(self, offsets):
        """Only new bytes of this entry's stderr and the home's agent.log; never classify quoted host replies.
        A published compaction wins over a manual no-op after automatic compaction in the triggering turn.
        """
        compactions, boundary, noop, reasons = set(), False, False, []
        for path, (inode, offset) in offsets.items():
            try:
                with path.open("rb") as fh:
                    stat = os.fstat(fh.fileno())
                    fh.seek(offset if stat.st_ino == inode and stat.st_size >= offset else 0)
                    lines = fh.read().decode("utf-8", "replace").splitlines()
            except OSError:
                continue
            for line in lines:
                hit = AUDIT.LOG_LINE.match(line)
                if not hit or not AUDIT.LCM_LOGGER.search(hit["logger"] or hit["logger2"] or hit["logger3"]):
                    continue
                msg = hit["msg"]
                if AUDIT.COMPACTION.search(msg):
                    compactions.add((hit["ts"], hit["frac"], hit["sid"] or hit["sid2"], msg))
                boundary |= "in-place compression boundary kept" in msg
                noop |= "LCM compression no-op" in msg or "compaction stop: reason=noop" in msg
                stop = STOP_REASON.search(msg)
                if stop:
                    reasons.append(stop.group(1))
        n = len(compactions) or int(boundary)
        return ("compressed" if n else "noop" if noop else "none"), n, sorted(reasons)

    def compress(self, o):
        slot = o["slot"]
        self.acp_sid(slot) if slot.startswith("A") else self.tui_rt(slot)
        for retry in range(3):  # the planned attempt and at most two retries without a real compaction
            if time.monotonic() >= self.stop_at:
                return True if retry else None  # an attempt already ran and was counted
            paths = (self.logs / ("acp.stderr" if slot.startswith("A") else "tui.stderr"),
                     self.home / "logs" / "agent.log")
            offsets = {}
            for path in paths:
                try:
                    stat = path.stat()
                    offsets[path] = (stat.st_ino, stat.st_size)
                except FileNotFoundError:
                    offsets[path] = (None, 0)
            # The paste or an ageing turn may compact automatically: that publication belongs to this attempt (the
            # offsets above), and the audit ties a sessionless compaction line only to a compress_request at most
            # 2 s after it, so a phase=paste request opens an attribution window before the paste. It is a window
            # only: requests counts the phase=trigger request below.
            self.ev.write("compress_request", self.slots[slot]["session"], f"slot={slot} phase=paste", retry=retry)
            # Age the paste outside LCM-X's eight-message fresh tail before the measured triggering turn.
            for ix in (o["prompt"], *(CATS["plain"][(o["prompt"] + j) % len(CATS["plain"])] for j in range(self.age_turns))):
                try:
                    stop = self.send(slot, ix)
                except (AD.DriverError, OSError, KeyError, ValueError):
                    self.ev.write("compress_request_failed", self.slots[slot]["session"], f"slot={slot} phase=paste")
                    raise
                if stop not in ("end_turn", "complete"):
                    self.ev.write("compress_request_failed", self.slots[slot]["session"], f"slot={slot} phase=paste")
                    return False
            if not self.compress_log_result(offsets)[1]:
                # Nothing compacted during the paste or ageing turns: close the paste window, so it cannot claim
                # another session's sessionless line (the audit drops a session's latest request on this event).
                self.ev.write("compress_request_failed", self.slots[slot]["session"], f"slot={slot} phase=paste unused")
            self.ev.write("compress_request", self.slots[slot]["session"], f"slot={slot} phase=trigger", retry=retry)
            self.compactions["requests"] += 1
            try:
                stop = self.send(slot, CATS["plain"][(o["prompt"] + self.age_turns) % len(CATS["plain"])])
            except (AD.DriverError, OSError, KeyError, ValueError):
                self.ev.write("compress_request_failed", self.slots[slot]["session"], f"slot={slot}")
                raise
            if stop not in ("end_turn", "complete"):
                self.ev.write("compress_request_failed", self.slots[slot]["session"], f"slot={slot}")
                return False
            if slot.startswith("A"):
                reply, _stop = self.acp.prompt(self.acp_sid(slot), "/compress", self.timeout)
                host = "compressed" if reply.startswith("Context compressed") else "not_compressed"
            else:
                res = self.tui.call("session.compress", {"session_id": self.tui_rt(slot)}, self.timeout)
                host = "compressed" if res.get("status") == "compressed" else "not_compressed"
            logged, n, reasons = self.compress_log_result(offsets)
            result = host if logged == "none" else logged
            self.compactions["real"] += n
            self.compactions["noop"] += int(logged == "noop")
            self.compactions["log_none"] += int(logged == "none")
            for reason in reasons:  # LCM-X's own stop reasons (sanitized, time_budget_exhausted, error, ...)
                self.compactions[f"stop_{reason}"] += 1
            self.ev.write("compress_result", self.slots[slot]["session"], f"slot={slot} manual={result} log={logged}",
                          log=logged, host_answer=host, compactions=n, retry=retry, stop_reasons=reasons)
            if n:
                break
        return True

    # operations
    def op(self, o: dict) -> None:
        kind, slot = o["op"], o["slot"]
        if kind in ("turn", "tool", "repeat", "delegate"):
            stop = self.send(slot, o["prompt"])
            if kind == "delegate":
                self.ev.write("delegate_prompt", self.slots[slot]["session"], f"slot={slot} prompt={o['prompt']}")
            if self.context is None and slot.startswith("A") and stop == "end_turn":
                self.probe_context(slot)
        elif kind == "compress":
            if not self.compress(o):
                return
        elif kind == "cancel_resend":  # ACP: cancel mid-turn, then the same prompt; never counted as an undo
            if self.send(slot, o["prompt"], cancel=True) != "cancelled":
                self.ev.write("error", self.slots[slot]["session"], f"cancel_late slot={slot}")
                return
            if self.send(slot, o["prompt"]) != "end_turn":
                return  # a failed resend is not executed: no event, no count
            self.ev.write("cancel_resend", self.slots[slot]["session"], f"slot={slot}")
        elif kind == "undo_resend":
            if slot.startswith("A"):
                raise ValueError("undo_resend needs a TUI slot: ACP has no undo (its cancel-and-resend is cancel_resend)")
            if self.send(slot, o["prompt"]) != "complete":
                return  # send() already records the failed turn; never undo a previous turn
            removed = self.tui.call("session.undo", {"session_id": self.tui_rt(slot)}, self.timeout).get("removed")
            if not removed:
                self.ev.write("error", self.slots[slot]["session"], f"undo_removed_nothing slot={slot}")
                return
            if self.send(slot, o["prompt"]) != "complete":
                return  # a failed resend is not executed: no event, no count
            self.ev.write("undo_resend", self.slots[slot]["session"], f"slot={slot} via=tui_undo removed={removed}")
        elif kind == "restart_cold":
            self.acp_sid(slot)
            self.drop("acp")
            self.acp_sid(slot, planned_restart=True)  # count only after session/load succeeds
        elif kind == "restart_warm":
            rt = self.tui_rt(slot)
            closed = self.tui.call("session.close", {"session_id": rt}, self.timeout).get("closed")
            if not closed:
                self.ev.write("error", self.slots[slot]["session"], f"warm_not_closed slot={slot}")
                return
            self.tui_loaded.discard(slot)
            self.tui_rt(slot)  # rebind only after resume succeeds; a later operation can retry a failed resume
            self.ev.write("restart_warm", self.slots[slot]["session"], f"slot={slot} closed={bool(closed)}")
        self.done[kind] += 1

    def children(self) -> int | None:
        """This run's delegated children so far, read-only from state.db (audit.py counts them the same way,
        with --since set to the run's start). None without a state.db (no host has run); an unreadable one is an error."""
        if not (self.home / "state.db").exists():
            return None
        try:
            con = sqlite3.connect(f"file:{self.home / 'state.db'}?mode=ro", uri=True, timeout=5)
            try:
                sessions = AUDIT.read_sessions(con)
                paths = (self.logs / "acp.stderr", self.logs / "tui.stderr", self.home / "logs" / "agent.log")
                aux, _unproven = AUDIT.auxiliary_sessions(AUDIT.read_logs(p for p in paths if p.is_file())["auxiliary"])
                return sum(s not in aux and v["source"] == DELEGATE_SOURCE and AUDIT.eligible_child(s, sessions)
                           and (v["started"] or 0) >= self.run_start for s, v in sessions.items())
            finally:
                con.close()
        except (sqlite3.Error, OSError, ValueError) as exc:
            self.ev.write("error", None, f"child_count_failed {type(exc).__name__}")
            return None

    def inactive(self, entry: str) -> list[tuple[str, str]]:
        """LCM-X-inactive lines written since the last check: each host's stderr (the ACP host logs only there,
        acp_adapter/entry.py:65-76), and the home's agent.log, whose new lines belong to the host that just ran."""
        found = []
        for who, path in (("acp", self.logs / "acp.stderr"), ("tui", self.logs / "tui.stderr"),
                          (entry, self.home / "logs" / "agent.log")):
            try:
                with open(path, "rb") as fh:
                    offset = self.scanned.get(path, 0)
                    fh.seek(offset if os.fstat(fh.fileno()).st_size >= offset else 0)
                    new = fh.read()
                    self.scanned[path] = fh.tell()
            except OSError:
                continue
            if INACTIVE.search(new.decode("utf-8", "replace")):
                found.append((who, path.name))
        return found

    def stop_if_inactive(self, entry: str) -> bool:
        bad = self.inactive(entry)
        for who, source in bad:
            self.ev.write("error", None, f"lcm_inactive entry={who} source={source}")
        return bool(bad)

    def guarded(self, o: dict) -> None:
        self.attempts[o["op"]] += 1
        try:
            self.op(o)
        except AD.ProcessGone as exc:
            which = "acp" if o["slot"].startswith("A") else "tui"
            self.ev.write("error", (self.slots.get(o["slot"]) or {}).get("session"),
                          f"process_gone op={o['op']} entry={which} {str(exc)[:80]}")
            self.drop(which)  # the next op on this entry point starts a new process
        except (AD.DriverError, OSError, KeyError, ValueError) as exc:
            code = (getattr(exc, "error", None) or {}).get("code") if isinstance(exc, AD.JsonRpcError) else None
            self.ev.write("error", (self.slots.get(o["slot"]) or {}).get("session"),
                          f"{type(exc).__name__} op={o['op']} slot={o['slot']} code={code}")

    def run(self, ops: list[dict], tg: dict) -> dict:
        stopped = None
        try:
            for o in ops:
                if time.monotonic() >= self.stop_at:
                    stopped = "max_hours"
                    break
                self.guarded(o)
                if self.stop_if_inactive("acp" if o["slot"].startswith("A") else "tui"):
                    stopped = "lcm_inactive"
                    break
            # Children are counted from state.db: keep prompting until the target or the attempt budget (2x).
            extra, delegates = 0, CATS["delegate"]
            while stopped is None and self.attempts["delegate"] < 2 * tg["delegated_children"]:
                n = self.children()
                if n is None or n >= tg["delegated_children"]:
                    break
                if time.monotonic() >= self.stop_at:
                    stopped = "max_hours"
                    break
                self.guarded({"op": "delegate", "slot": "A1", "prompt": delegates[extra % len(delegates)]})
                extra += 1
                if self.stop_if_inactive("acp"):
                    stopped = "lcm_inactive"
        finally:
            failure = None
            for which in ("acp", "tui"):  # both hosts, even when the event log or one host's shutdown fails
                try:
                    self.drop(which, graceful=True)
                except Exception as exc:  # noqa: BLE001 - re-raised below once both hosts were attempted
                    failure = failure or exc
            if failure is not None:
                raise failure
        if stopped != "lcm_inactive" and self.stop_if_inactive("tui"):  # lines written while the hosts exited
            stopped = "lcm_inactive"
        summary = {"planned": planned_counts(ops), "executed": dict(self.done), "errors": dict(self.ev.errors),
                   "compactions": {"planned": sum(o["op"] == "compress" for o in ops),
                                   **{k: self.compactions[k] for k in ("requests", "real", "noop", "log_none")},
                                   "stop_reasons": {k[5:]: v for k, v in sorted(self.compactions.items())
                                                    if k.startswith("stop_")}},
                   "prompts_by_category": dict(self.prompts),
                   "delegate": {"prompts": self.done["delegate"], "extra_prompts": extra,
                                "attempts": self.attempts["delegate"],
                                "children_in_state_db": self.children()},
                   "context_length": self.context, "stopped": stopped, "sessions": len(self.slots),
                   "age_turns": self.age_turns, "fresh_tail": getattr(self, "fresh_tail", None)}
        self.ev.write("summary", None, "end of run", **summary)
        summary["exit"] = EXIT_LCM_INACTIVE if stopped == "lcm_inactive" else 0
        return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hermes-python", required=True)
    ap.add_argument("--hermes-src", type=Path, required=True)
    ap.add_argument("--home", type=Path, required=True, help="the prepared HERMES_HOME (never the live ~/.hermes)")
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--append-events", action="store_true", help="explicitly retain and append to a non-empty events file")
    ap.add_argument("--turns", type=int, help="at least this many prompts (never below the floor x1.2)")
    ap.add_argument("--children", type=int, help="at least this many delegated children (never below the floor x1.2)")
    ap.add_argument("--compactions", type=int, help="at least this many compaction ops (never below the floor x1.2)")
    ap.add_argument("--age-turns", type=int, help="plain turns to age each paste before the triggering turn "
                    "(default: ceil(LCM_FRESH_TAIL_COUNT / 2) from the hosts' environment, product tail 24 when unset)")
    ap.add_argument("--seed", type=int, default=2055)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--prompt-timeout", type=float, default=600.0)
    ap.add_argument("--cancel-after", type=float, default=2.0, help="seconds into a prompt before session/cancel")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.age_turns is not None and a.age_turns < 0:
        ap.error("--age-turns must be non-negative")
    for name in ("home", "workdir", "events"):
        if under_real_hermes(getattr(a, name)):
            ap.error(f"--{name} must not be at or under the live ~/.hermes")
    for name in ("home", "workdir", "events", "hermes_src"):
        setattr(a, name, getattr(a, name).expanduser().resolve())
    a.hermes_python = str(Path(a.hermes_python).expanduser().absolute())  # retain the venv executable's symlink
    tg = targets(a.turns, a.children, a.compactions)
    ops = plan(a.seed, tg)
    if a.dry_run:
        print(json.dumps({"targets": tg, "planned": planned_counts(ops)}, indent=1, sort_keys=True))
        return 0
    if not (a.home / "config.yaml").is_file():
        ap.error(f"{a.home} has no config.yaml: prepare the home first (see the gate_kit README)")
    try:
        events = Events(a.events, append=a.append_events)
    except ValueError as exc:
        ap.error(str(exc))
    a.workdir.mkdir(parents=True, exist_ok=True)
    write_notes(a.workdir)
    env = host_env(dict(os.environ), a.home, a.workdir, a.hermes_src)
    tail, tail_source = fresh_tail(env, a.home)
    if a.age_turns is None:
        a.age_turns = default_age_turns(env, a.home)
    d = Driver(python=a.hermes_python, src=a.hermes_src, home=a.home, workdir=a.workdir, events=events,
               env=env, prompt_timeout=a.prompt_timeout, max_hours=a.max_hours, cancel_after=a.cancel_after,
               age_turns=a.age_turns)
    d.fresh_tail = {"count": tail, "source": tail_source}
    try:
        summary = d.run(ops, tg)
    except EventLogError as exc:
        print(f"the run stopped: {exc}; both hosts were shut down; do not audit this run", file=sys.stderr)
        return EXIT_EVENT_LOG
    print(json.dumps({k: summary[k] for k in ("executed", "errors", "delegate", "context_length", "stopped")},
                     sort_keys=True))
    if summary["exit"]:
        print("LCM-X was not active in a host process (an lcm_inactive error event): do not audit this run; raise "
              "plugins.load_timeout_seconds in the home and run again", file=sys.stderr)
    return summary["exit"]


if __name__ == "__main__":
    sys.exit(main())
