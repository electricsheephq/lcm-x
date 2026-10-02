"""R2: run one R1 cell through a REAL host process (``run_matrix.py --transport acp-process``).

The host is ``<host venv>/bin/hermes acp`` over stdio (acp_driver.py), with an isolated HERMES_HOME/HOME in a
$TMPDIR scratch dir (deleted once the cell is scored), and every model route (main, LCM summariser, host aux) pointed at the localhost fake provider
(fake_provider.py), so the summariser and aux LLM are real HTTP code paths. The cell scenario is R1's (prompts,
replies, tool plans, faults) and the scorers are R1's: the host observer (observer/) writes the same transcript
events R1's probe writes in-process, the scenario writes what the provider emitted, and the runner writes what
the client sent. Faults are real: a crash is a SIGKILL of the host process group while the provider holds the
turn's request; a cancel is an ACP ``session/cancel`` while the request is in flight; the final forced compaction
is the ACP ``/compress`` command. Containment: localhost-only base URLs, no real API-key env, every proxy variable
pointed at a localhost sink that records and refuses (fake_provider.ProxySink), the host model catalog off, macOS ``sandbox-exec`` (localhost-only network) when present, and an observer socket guard
that refuses and records any non-localhost attempt (any attempt makes the cell ERROR: stop and report).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import time
from pathlib import Path

from bench.instruments.reliability import acp_driver as AD, cells as C, fake_provider as FP, probe as P1, run_matrix as RM

OBSERVER = Path(__file__).with_name("observer")
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
SANDBOX = ('(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'
           '(allow network-outbound (remote unix-socket))')
FAKE_KEY = "rel-fake-key-not-a-secret"
PROCESS_FAULTS = {"acp-process": {"crash_after_compaction_before_reply", "cancel_then_retry",
                                   "crash_after_rotation_before_child_row", "clean_exit_before_turn"}}
# R2-only: the main route over the Anthropic Messages API (a ``/anthropic`` base path selects the anthropic_messages
# transport, hermes_cli/runtime_provider.py _detect_api_mode_for_url); the #550 class.
R2_CELLS = [{**C.cell("anthropic-route/acp-process", [], in_place=True,
                      doc="baseline/in-place/acp with the main model on the Anthropic Messages API (fake provider)"),
             "api": "anthropic"}]
# gateway-process: why no local platform can drive an R1 gateway cell, cited per host at run time.
GATEWAY_ANCHORS = {
    "turn_runner": ("gateway/run_turn.py", "TurnRunner(self, turn_ctx)"),
    "webhook_session": ("gateway/platforms/webhook.py", 'session_chat_id = f"webhook:{route_name}:{delivery_id}"'),
    "webhook_close": ("gateway/platforms/webhook.py", "async def on_processing_complete"),
    "api_server_bypass": ("gateway/platforms/api_server.py", "never passes through ``TurnRunner``"),
}


def graceful_exit(returncode: int | None, killed: bool, sent_term: bool) -> bool:
    """A clean host stop: it exited on stdin EOF (0), or on the SIGTERM close() itself sent to the still-running host
    within the shutdown grace (-SIGTERM, how a service manager stops a gateway). A SIGKILL escalation, a SIGTERM from
    anywhere else (the host had already ended before close()) or any other code is not clean."""
    return not killed and (returncode == 0 or (returncode == -signal.SIGTERM and sent_term))

def gateway_unsupported(src: str) -> str:
    c = cite_all(src, GATEWAY_ANCHORS)
    return (f"no local TurnRunner-backed platform holds one chat across turns at this host sha: the webhook platform "
            f"(TurnRunner at {c['turn_runner']}) keys every POST to its own one-shot session ({c['webhook_session']}), "
            f"closed when the run ends ({c['webhook_close']}); api_server never passes through TurnRunner "
            f"({c['api_server_bypass']}); the remaining platforms bridge to external messaging services")


def append(path: Path, rec: dict) -> None:
    """One O_APPEND write per record, so runner and host records in one file never interleave mid-line."""
    data = (json.dumps(rec, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        while data:
            data = data[os.write(fd, data):]
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def unsupported(cell: dict, transport: str) -> str | None:
    if transport == "gateway-process":
        return None if cell["transport"] == "gateway" else "an R1 acp-transport cell; it runs under acp-process"
    if transport == "api-server":
        return "api-server transport cells are not implemented in R2"
    if cell["transport"] != "acp":
        return "an R1 gateway-transport cell; it runs under gateway-process"
    if cell.get("cron_every"):
        return "cron fires in-process in R1 (cron/scheduler.py); the ACP process has no cron driver"
    extra = sorted({f["kind"] for f in cell["faults"]} - PROCESS_FAULTS[transport])
    return f"fault(s) {extra} are injected in-process by R1 and have no process-level driver in R2" if extra else None


def config_yaml(cell: dict, plugin: dict, base_url: str) -> str:
    route = f'provider: custom\n    base_url: "{base_url}"\n    api_key: "{FAKE_KEY}"\n'
    main_url = base_url[:-len("/v1")] + "/anthropic" if cell.get("api") == "anthropic" else base_url
    return (RM.config_yaml(cell, plugin)
            + f'model:\n  default: rel/main\n  provider: custom\n  base_url: "{main_url}"\n  api_key: "{FAKE_KEY}"\n'
            f"  context_length: {cell['window']}\n"
            f"auxiliary:\n  compression:\n    {route}    model: rel/aux\n  title_generation:\n    {route}    model: rel/aux-title\n"
            "  background_review:\n    enabled: false\n"
            "memory:\n  memory_enabled: false\n  user_profile_enabled: false\n  nudge_interval: 0\n"
            "skills:\n  creation_nudge_interval: 0\n"
            # enabled: the model catalog fetch (hermes_cli/model_catalog.py); excluded_providers (hermes_cli/
            # inventory.py:54 at customer-0.21.2): the keyless opencode-free provider, whose live /models fetch
            # ACP session/new runs via model_catalog.build_model_state -> _fetch_opencode_free_models.
            "model_catalog:\n  enabled: false\n  excluded_providers: [opencode-free]\n"
            f"platform_toolsets:\n  acp: [{', '.join(cell.get('toolsets', ['todo', 'context_engine', 'file']))}]\n")


def phase_lcm_env(cell: dict, phase: str) -> dict:
    """Phase defaults (Fixture B), with the global override winning in every phase."""
    return {**cell["lcm_env"], **((cell.get("drain") or {}).get("phase2_lcm_env") or {} if phase != "A" else {}),
            **(cell.get("global_lcm_env") or {})}


def forget_host_rows(state_db: Path, session_id: str) -> int:
    """Fixture B, harness state only (no host or plugin code): the host's own soft-archive (active=0, as its in-place
    compaction ``archive_and_compact`` leaves superseded rows) on every active row of the ACP session, so the next ACP
    restore starts from an empty list while LCM's stored rows stay unsummarized above the frontier (hidden)."""
    con = sqlite3.connect(state_db)
    try:
        n = con.execute("UPDATE messages SET active = 0 WHERE session_id = ? AND active = 1", (session_id,)).rowcount
        con.commit()
        return n
    finally:
        con.close()


def reply_text(cell: dict, prefix: str, t: int) -> str:
    asst = cell["assistant"]
    if asst.get("mode") == "repeat-identical" and t in asst.get("repeat_turns", []):
        return "noted, the same as before."
    return f"reply to {prefix}{t:02d}: noted item {t}."


def cite_all(src: str, anchors: dict = P1.ANCHORS) -> dict:
    out = {}
    for key, (rel, needle) in anchors.items():
        try:
            lines = (Path(src) / rel).read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        out[key] = next((f"{rel}:{i}" for i, line in enumerate(lines, 1) if needle in line), None)
    return out


class Scenario:
    """The ``main`` role: R1's scripted provider (probe.py ``scripted``), driven by the turn the client is on."""

    def __init__(self, run: "ProcessCell"):
        self.run, self.cell = run, run.cell
        self.st = {"t": None, "kind": None, "step": 0, "issued": set(), "seen": set(), "phase": "A", "first": 1}
        self.unexpected = []

    def begin(self, t: int, kind: str, phase: str, first: int) -> None:
        self.st.update(t=t, kind=kind, step=0, issued=set(), seen=set(), phase=phase, first=first)

    def plan(self, t: int) -> list:
        phase, first, out = self.st["phase"], self.st["first"], []
        for group in self.cell.get("tool_plan", []):
            turns = ([first] if phase != "A" else []) if group["turns"] == "restart" else group["turns"]
            out += [group["calls"]] if t in turns else []
        return out

    def main(self, messages: list[dict]) -> dict:
        st, run = self.st, self.run
        t, text_tag = st["t"], run.text_tag
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if t is None or (f"[{text_tag}]" not in last_user if text_tag else "continue" not in last_user):
            self.unexpected.append({"turn": t, "last_user": last_user[:120]})
            return {"content": "ok", "log": {"unexpected": True}}
        step, st["step"] = st["step"], st["step"] + 1
        for m in messages:  # a tool result as the model receives it (post host transform)
            cid = m.get("tool_call_id") if m["role"] == "tool" else None
            if cid in st["issued"] and cid not in st["seen"]:
                st["seen"].add(cid)
                body = m.get("content") or ""
                run.event(turn=t, event="tool_seen", id=cid, sha=hashlib.sha256(body.encode()).hexdigest(), chars=len(body))
        log = {"turn": t, "kind": st["kind"], "step": step}
        if run.crash_due(t):
            return {"hold_until_killed": True, "on_hold": lambda: run.crash(t), "log": log}
        groups = self.plan(t)
        cancel = st["kind"] == "cancel" and step == 0
        extra = {"hold": float(self.cell.get("cancel_wait", 5.0)), "on_hold": run.cancel} if cancel else {}
        if step < len(groups):
            calls = []
            for k, c in enumerate(groups[step]):
                args = json.loads(json.dumps(c.get("args", {})).replace("{files}", str(run.files)))
                cid = f"call_T{t:02d}_{step}_{k}"
                st["issued"].add(cid)
                run.event(turn=t, event="tool_issue", tag=f"T{t:02d}", id=cid, name=c["name"], args=args,
                          expect=c.get("expect") or {})
                run.event(turn=t, event="tool_call", name=c["name"], session_prefix="T")
                calls.append({"id": cid, "name": c["name"], "arguments": json.dumps(args)})
            return {"tool_calls": calls, "log": log, **extra}
        text = reply_text(self.cell, "T", t)
        run.event(turn=t, event="emit", tag=f"T{t:02d}", text=text)
        return {"content": text, "log": log, **extra}


class PhaseDeadline(AD.DriverError):
    """The overall phase deadline (run_matrix ``--timeout``) expired."""


class ProcessCell:
    def __init__(self, cell, d: Path, host: dict, transport: str, turn_timeout: float, phase_timeout: float | None = None,
                 scratch: Path | None = None):
        """``d`` gets the records; ``scratch`` (default ``d``) the host's HERMES_HOME and HOME/TMPDIR."""
        self.cell, self.d, self.host, self.transport, self.turn_timeout = cell, d, host, transport, turn_timeout
        self.phase_timeout, self.deadline, self.scratch = phase_timeout, None, scratch or d
        self.home, self.files, self.work = self.scratch / "hermes-home", d / "files", self.scratch / "home" / "work"
        self.transcript, self.proc, self.sid, self.text_tag = d / "transcript.jsonl", None, None, None
        self.phase, self.log_mark, self.fired, self.backlog_log, self.last_turn = "A", 0, set(), [], 0
        self.scenario = Scenario(self)
        self.provider = FP.FakeProvider(d / "provider-requests.jsonl", main=self.scenario.main,
                                        usage_scale=float(cell["assistant"].get("usage_scale", 1.0)))
        self.proxy = FP.ProxySink(d / "proxy-attempts.jsonl")

    # -- records ---------------------------------------------------------------------------------------------
    def event(self, **ev) -> None:
        append(self.transcript, {"phase": self.phase, **ev})

    def host_log(self) -> str:
        path = self.d / f"probe-{self.phase}.hermes.log"
        return path.read_text(errors="replace") if path.exists() else ""

    def notes(self, kind: str) -> list[dict]:
        return [n for n in read_jsonl(self.d / "observer.jsonl") if n["kind"] == kind]

    def fire(self, kind: str, turn: int, **extra) -> None:
        append(self.d / "faults-fired.jsonl", {"kind": kind, "phase": self.phase, "turn": turn})
        self.fired.add(kind)
        self.event(turn=turn, event="crash" if kind.startswith("crash") else kind, fault=kind, session_prefix="T",
                   host_commits=self.host_log()[self.log_mark:].count(P1.COMMITTED), **extra)

    # -- faults ----------------------------------------------------------------------------------------------
    def crash_due(self, t: int) -> bool:
        kind = "crash_after_compaction_before_reply"
        if self.phase != "A" or kind in self.fired or kind not in {f["kind"] for f in self.cell["faults"]}:
            return False
        return any(n["phase"] == "A" and n["turn"] == t and not n.get("final") for n in self.notes("compaction_committed"))

    def crash(self, t: int) -> None:
        """Called by the provider while it holds turn t's request: a TRUE mid-turn crash of the host."""
        self.fire("crash_after_compaction_before_reply", t)
        self.proc.kill()

    def cancel(self) -> None:
        self.proc.notify("session/cancel", {"sessionId": self.sid})
        self.event(turn=self.scenario.st["t"], event="cancel")

    # -- the host process ------------------------------------------------------------------------------------
    def budget(self) -> float:
        """One ACP request's timeout: the per-request timeout, capped by what is left of the phase deadline."""
        if self.deadline is None:
            return self.turn_timeout
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise PhaseDeadline(f"phase {self.phase} exceeded its {self.phase_timeout}s deadline")
        return min(self.turn_timeout, left)

    def env(self) -> dict:
        env = {"HOME": str(self.scratch / "home"), "PATH": "/usr/bin:/bin", "HERMES_HOME": str(self.home),
               "TMPDIR": str(self.scratch / "home"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(self.d / "pycache"), "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": str(OBSERVER), "REL_OBSERVER_DIR": str(self.d), "REL_PHASE": self.phase,
               "HERMES_ACP_SKIP_CONFIGURED_MCP": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
               "LCM_SUMMARY_MODEL": "rel/lcm-summary",
               # Time compression: a cell runs 60 turns in ~20 s, which trips the per-window summary spend guard
               # that a real session spreads over hours; without this the "real" summariser path is mostly skipped.
               "LCM_SUMMARY_SPEND_MAX_CALLS": "100000", **phase_lcm_env(self.cell, self.phase),
               "LCM_NATIVE_RECOVERY": "true" if self.cell["native_recovery"] else "false"}
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env[k] = self.proxy.url  # a recording sink that refuses everything
        return env

    def argv(self) -> list[str]:
        base = [str(Path(self.host["python"]).parent / "hermes"), "acp"]
        return [SANDBOX_EXEC, "-p", SANDBOX, *base] if os.path.exists(SANDBOX_EXEC) else base

    def turn(self, t: int, kind: str = "normal") -> str | None:
        text = P1.user_text(self.cell, "T", t)
        n = int(self.cell["user_text"].get("identical_turns", {}).get(str(t), t))
        self.text_tag, self.last_turn = None if text == "continue" else f"T{n:02d}", t
        (self.d / "turn.json").write_text(json.dumps({"prefix": "T", "t": t, "kind": kind, "text": text,
                                                      "text_tag": self.text_tag, "reply": reply_text(self.cell, "T", t)}))
        self.scenario.begin(t, kind, self.phase, self.first)
        self.log_mark = len(self.host_log())
        self.event(turn=t, event="user_sent" if kind != "retry" else "retry", tag=f"T{t:02d}", role="user",
                   session_prefix="T", content=text, persist=None, content_sha256=hashlib.sha256(text.encode()).hexdigest())
        kind_rot = "crash_after_rotation_before_child_row"
        killer = AD.RotationKiller(self.proc, self.home / "lcm.db", self.sid, lambda ev: self.fire(kind_rot, t, **ev)) \
            if self.phase == "A" and kind_rot not in self.fired and kind_rot in {f["kind"] for f in self.cell["faults"]} \
            else None
        if killer:  # --kill-after-rotation: SIGKILL while the rotated child session has no lcm rows yet
            killer.start()
        try:
            answer, stop = self.proc.prompt(self.sid, text, self.budget())
        finally:
            if killer:
                killer.stop.set()
                killer.join()
        self.event(turn=t, event="acp_response", tag=f"T{t:02d}", stop_reason=stop, chars=len(answer))
        return stop

    def final_check(self) -> dict:
        """The ACP ``/compress`` command, re-sent once after a cleanup-only ``sanitized`` pass (R1 final_check)."""
        attempts, t = [], self.last_turn
        for _ in range(2):
            (self.d / "turn.json").write_text(json.dumps({"prefix": "T", "t": t, "kind": "final", "text": "/compress"}))
            self.scenario.begin(None, "final", self.phase, self.first)
            before, mark = len(self.notes("final_attempt")), len(self.host_log())
            answer, _stop = self.proc.prompt(self.sid, "/compress", self.budget())
            got = self.notes("final_attempt")[before:]
            rec = {k: v for k, v in got[-1].items() if k not in ("phase", "kind", "ts")} if got else \
                {"entry": None, "engine_calls": 0, "engine_status": None, "not_invoked": True}
            rec.update(acp_answer=answer[:300], conflicts=self.host_log()[mark:].count("publication_invariant_conflict"))
            attempts.append(rec)
            if rec.get("engine_status") != "sanitized" or "exception" in rec:
                break
        last = attempts[-1]
        ok = ("compacted", "host_native") if self.cell["native_recovery"] else ("compacted",)
        conflicts = sum(a["conflicts"] for a in attempts)
        published = (last.get("host_status") == "compressed" and last["engine_calls"] > 0 and last["engine_status"] in ok
                     and not conflicts and "exception" not in last)
        failed = bool(conflicts or "exception" in last or last["engine_status"] == "error")
        return {**last, "attempts": attempts, "conflicts": conflicts, "published": published,
                "outcome": "published" if published else "failed" if failed else "inconclusive"}

    def run_phase(self, first: int) -> dict:
        self.first = first
        self.deadline = time.monotonic() + self.phase_timeout if self.phase_timeout else None
        self.proc = AD.AcpProcess(self.argv(), self.env(), self.work, self.d / f"host-{self.phase}.stderr")
        closed = False
        try:
            self.proc.initialize(self.budget())
            if self.phase == "A":
                self.sid = self.proc.new_session(self.files, self.budget())
            else:  # ACP _restore: the stable ACP id, restored from state.db by the fresh process
                self.proc.load_session(self.sid, self.files, self.budget())
            cancel = next((f for f in self.cell["faults"] if f["kind"] == "cancel_then_retry"), None)
            clean = next((f for f in self.cell["faults"] if f["kind"] == "clean_exit_before_turn"), None)
            for t in P1.extend_turns(self.cell, first, self.low_backlog):
                if clean and t == clean.get("turn") and t != first and "clean_exit_before_turn" not in self.fired:
                    rc = self.proc.close()
                    closed = True
                    if not graceful_exit(rc, self.proc.killed, getattr(self.proc, "sent_term", False)):
                        return {"exit": "error", "reason": f"clean exit failed: returncode={rc}, killed={self.proc.killed}"}
                    self.fire("clean_exit_before_turn", t)  # recorded only after a graceful host exit
                    return {"exit": "clean_exit", "next_turn": t}
                if cancel and t == cancel["turn"] and "cancel_then_retry" not in self.fired:
                    self.turn(t, "cancel")
                    end = [n for n in self.notes("turn_end") if n["tag"] == f"T{t:02d}" and n["turn_kind"] == "cancel"]
                    if end and end[-1]["interrupted"]:  # the fault counts only once the host really interrupted
                        self.fire("cancel_then_retry", t)
                        self.turn(t, "retry")
                else:
                    self.turn(t)
            final = {**self.final_check(), "backlog_checks": self.backlog_log} \
                if self.cell.get("final_compaction_check", True) else None
            self.budget()  # work after the last request may have crossed the deadline: never a late "done"
            return {"exit": "done", "next_turn": None, **({"final_check": final} if final else {})}
        except AD.ProcessGone as exc:
            if self.proc.killed and any(k.startswith("crash") for k in self.fired):
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    return {"exit": "error", "reason": f"PhaseDeadline: phase {self.phase} exceeded its "
                                                       f"{self.phase_timeout}s deadline before the crash was recorded"}
                t = self.last_turn
                return {"exit": "crash", "next_turn": t + 1, "turn": t}
            return {"exit": "error", "reason": f"host process ended: {exc}; stderr: {self.stderr_tail()}"}
        except (AD.DriverError, OSError, ValueError) as exc:
            if self.deadline is not None and time.monotonic() >= self.deadline and not isinstance(exc, PhaseDeadline):
                exc = PhaseDeadline(f"phase {self.phase} exceeded its {self.phase_timeout}s deadline ({exc})")
            return {"exit": "error", "reason": f"{type(exc).__name__}: {exc}; stderr: {self.stderr_tail()}"[:600]}
        finally:
            if not closed:
                rc = self.proc.close()
            self.event(event="host_exit", returncode=rc, killed=self.proc.killed)

    def low_backlog(self, last_turn: int) -> bool:
        return P1.backlog_low(self.d, last_turn, self.backlog_log)

    def stderr_tail(self) -> str:
        path = self.d / f"host-{self.phase}.stderr"
        return path.read_text(errors="replace")[-300:] if path.exists() else ""

    def phase_record(self, first: int, result: dict) -> dict:
        notes = [n for n in read_jsonl(self.d / "observer.jsonl") if n["phase"] == self.phase]
        snap = next((n for n in reversed(notes) if n["kind"] == "counters"), {})
        prov = next((n["provenance"] for n in reversed(notes) if n["kind"] == "counters" and n.get("provenance")), None)
        log = self.host_log()
        rec = {"phase": self.phase, "start_turn": first, "transport": self.transport, "acp_session": self.sid,
               "citations": cite_all(self.host["src"]),
               "counters": snap.get("counters") or {"compacted_turns": [], "failed": [], "orphan_drops": 0, "native_max": 0},
               "compactions_logged": len(re.findall(r"LCM compaction #\d+", log)),
               # #714: the same exit-fit normalization as in-process cells
               "log_counts": P1._log_counts(log), "session_count": session_count(self.home),
               "failed_turn_notices": next((n["failed_turn_notices"] for n in reversed(notes) if n.get("failed_turn_notices")), []),
               "provenance": prov, "network_attempts": [n for n in notes if n["kind"] == "network_attempt"][:20],
               "observer_errors": [n for n in notes if n["kind"] in ("observer_error", "refused")][:10], **result}
        if snap.get("orphan_hook"):
            rec["orphan_hook"] = snap["orphan_hook"]
        return rec


def accounting(d: Path, events: list[dict]) -> dict:
    """The provider log is the complete request set. Every distinct request is a model-role completion, a
    ``/models`` catalog read, or unexpected (an unrouted path/method or an unknown role: the cell ERRORs). Each main
    request is exactly one scripted step (a reply, a tool-call step, a crash hold or an unexpected main request)."""
    reqs = read_jsonl(d / "provider-requests.jsonl")
    by_role, by_route, unexpected = {}, {}, []
    for r in {r["rid"]: r for r in reqs}.values():
        route = FP.route_of(r.get("method"), r.get("path"))  # re-derived, not trusted from the log
        by_route[route] = by_route.get(route, 0) + 1
        if route == "completion":
            by_role[r["role"]] = by_role.get(r["role"], 0) + 1
        if route not in ("completion", "models") or (route == "completion" and r["role"] not in FP.ROLES):
            unexpected.append({k: r.get(k) for k in ("rid", "method", "path", "role", "route")})
    implied = {"emit": sum(1 for e in events if e["event"] == "emit"),
               "tool_steps": len({(e["phase"], e["id"].rsplit("_", 1)[0]) for e in events if e["event"] == "tool_issue"}),
               "crash_holds": sum(1 for e in events if e.get("fault") == "crash_after_compaction_before_reply"),
               "unexpected": len({r["rid"] for r in reqs if r.get("unexpected")})}
    return {"requests_by_role": by_role, "requests_by_route": by_route, "unexpected_requests": unexpected[:20],
            "main_implied": implied, "faults": sorted({(r["rid"], r["fault"]) for r in reqs if r.get("fault")}),
            "ok": by_role.get("main", 0) == sum(implied.values()) and not unexpected}


def has_dist(python: str, name: str) -> bool:
    """Package metadata only (a ``<name>-*.dist-info`` in the venv's site-packages): the host interpreter never runs
    outside the contained cell environment. ``python`` is not resolved: a venv's bin/python is a symlink."""
    venv = Path(os.path.abspath(python)).parent.parent
    return any(venv.glob(f"lib/python3*/site-packages/{name}-*.dist-info/METADATA"))


def session_count(home: Path) -> int | None:
    try:
        con = sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)
        try:
            return con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        finally:
            con.close()
    except sqlite3.Error:
        return None


def run_cell_process(cell: dict, host_name: str, host: dict, plugin: dict, out: Path, timeout: int, keep: bool,
                     keep_dbs: str = "none", lcm_env: dict | None = None, identity: dict | None = None,
                     transport: str = "acp-process", turn_timeout: float = 300.0,
                     scratch_root: str | Path | None = None) -> dict:
    cell = RM.with_lcm_env(cell, plugin, lcm_env)
    d = RM.checked_cell_dir(out, out / "cells" / host_name / plugin["sha"][:12] / f"{RM.slug(cell['id'])}@{transport}")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rec = {"cell": cell["id"], "transport": transport, "host": host_name, "host_sha": host["sha"], "plugin_ref": plugin["ref"],
           "plugin_sha": plugin["sha"], "targets": cell["targets"], "dir": str(d), "host_identity": identity}

    def done(**fields):
        rec.update(fields)
        (d / "verdict.json").write_text(json.dumps(rec, indent=1, default=str))
        return rec
    if not identity or "error" in identity:  # before anything reads or runs the host tree
        return done(verdict="ERROR", reason=f"host identity not verified: {(identity or {}).get('error')}")
    if why := unsupported(cell, transport):
        return done(verdict="UNSUPPORTED", reason=why)
    if transport == "gateway-process":
        return done(verdict="UNSUPPORTED", reason=gateway_unsupported(host["src"]))
    if cell.get("api") == "anthropic" and not has_dist(host["python"], "anthropic"):
        return done(verdict="UNSUPPORTED", reason="the host venv has no anthropic SDK (hermes-agent[anthropic] not installed)")
    s = RM.scratch_dir(scratch_root)
    try:  # every exit below, ERROR included, releases the scratch: no Hermes DB outlives its cell by default
        home = s / "hermes-home"
        for sub in (home / "plugins", s / "home" / "work", s / "db", d / "files"):
            sub.mkdir(parents=True)
        (home / "plugins" / plugin["dir"]).symlink_to(plugin["tree"])
        # A fresh, non-empty models.dev disk cache (agent/models_dev.py serves it for 4 h), so model metadata
        # resolution never fetches models.dev; the main model's context_length is pinned in config.yaml.
        (home / "models_dev_cache.json").write_text(json.dumps({"rel": {"id": "rel", "name": "reliability fake", "models": {}}}))
        (d / "files" / "small.txt").write_text("small deterministic file\n")
        (d / "files" / "big.txt").write_text("".join(f"line {i:05d}: " + P1.FILLER * 8 + "\n" for i in range(cell.get("big_lines", 400))))
        run = ProcessCell(cell, d, host, transport, turn_timeout, phase_timeout=timeout, scratch=s)
        run.provider.start()
        (home / "config.yaml").write_text(config_yaml(cell, plugin, run.provider.base_url))
        (d / "cell.json").write_text(json.dumps({**cell, "plugin": plugin, "host": host_name, "host_src": host["src"],
                                                 "host_python": host["python"], "transport": transport}, indent=1))
        started, first, last, phases = time.time(), 1, {}, []
        try:
            for phase in RM.PHASES:
                run.phase = phase
                last = run.run_phase(first)
                (d / f"phase-{phase}.json").write_text(json.dumps(run.phase_record(first, last), indent=1, default=str))
                phases.append(phase)
                if last["exit"] not in ("crash", "clean_exit"):
                    break
                first = last["next_turn"]
                drain = cell.get("drain") or {}
                if drain.get("phase2_lcm_env"):  # the next phase's config carries its LCM threshold too
                    (home / "config.yaml").write_text(config_yaml({**cell, "lcm_env": phase_lcm_env(cell, "B")}, plugin,
                                                                  run.provider.base_url))
                if last["exit"] == "clean_exit" and drain.get("forget_host_rows"):
                    append(d / "fixture.jsonl", {"kind": "forget_host_rows", "after_phase": phase, "before_turn": first,
                                                 "rows": forget_host_rows(home / "state.db", run.sid)})
        finally:
            run.provider.stop()
            run.proxy.stop()
        backup_errors = RM.copy_dbs(home, s / "db")
        events = read_jsonl(d / "transcript.jsonl")
        records = [json.loads(p.read_text()) for p in sorted(d.glob("phase-*.json"))]
        acct = accounting(d, events)
        network = [n for r in records for n in r.get("network_attempts", [])] + read_jsonl(d / "proxy-attempts.jsonl")
        prov = [v for r in records for v in ((r.get("provenance") or {}).get("violations") or [])]
        observer = [n for r in records for n in r.get("observer_errors", [])]
        rec.update(phases=phases, wall_s=round(time.time() - started, 1), acp_session=run.sid, accounting=acct,
                   citations=records[0]["citations"] if records else {}, sandbox=os.path.exists(SANDBOX_EXEC))
        if network:  # stop condition: a host process tried to leave localhost
            return done(verdict="ERROR", reason=f"STOP: host attempted non-localhost network access: {network[:3]}")
        if prov or observer:
            return done(verdict="ERROR", reason=f"import provenance / observer failure: {(prov + observer)[:3]}")
        if acct["unexpected_requests"]:
            return done(verdict="ERROR", reason=f"unexpected provider requests: {acct['unexpected_requests'][:3]}")
        if run.scenario.unexpected:
            return done(verdict="ERROR", reason=f"unexpected main-model requests: {run.scenario.unexpected[:3]}")
        if last.get("exit") == "done" and not acct["ok"]:
            return done(verdict="ERROR", reason=f"provider request log does not account for the transcript: {acct}")
        rec.update(RM.verdict_fields({**cell, "chat_root": run.sid}, d, last, run.fired, rec["citations"], backup_errors,
                                     s / "db"))
        return done()
    finally:
        RM.release_scratch(s, d, rec, keep_dbs, keep)
