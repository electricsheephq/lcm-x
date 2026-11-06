"""Gate-kit driver: the planner (pure), the executor's events against stub host processes, the events contract with
audit.py --events, and a host-backed smoke (a real ``hermes acp`` and TUI gateway against the localhost fake provider)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from bench.instruments.gate_kit import audit, drive  # noqa: E402

S = "ZQX-SENTINEL"


# -- planner -------------------------------------------------------------------------------------------------------
def slot_ops(ops):
    out = {}
    for o in ops:
        out.setdefault(o["slot"], []).append(o)
    return out


def test_plan_meets_every_floor_with_margin_and_is_deterministic():
    tg = drive.targets()
    assert tg["turns"] == 240 and tg["delegated_children"] == 12 and tg["restarts_cold"] == 6
    ops = drive.plan(2055, tg)
    got = drive.planned_counts(ops)
    for k in drive.PLANNED_FLOORS:
        assert got[k] >= tg[k] >= -(-audit.FLOORS[k] * 6 // 5), k
    assert got["restarts_cold"] + got["restarts_warm"] >= -(-audit.FLOORS["restarts"] * 6 // 5)
    assert got["host_sources"] == 2
    assert drive.plan(2055, tg) == ops and drive.plan(2056, tg) != ops
    assert drive.targets(turns=100, children=30)["turns"] == 240 and drive.targets(children=30)["delegated_children"] == 30


def test_plan_repeats_are_word_for_word_and_restarts_are_never_last():
    ops = drive.plan(7, drive.targets())
    for slot, seq in slot_ops(ops).items():
        tools = [o["prompt"] for o in seq if o["op"] == "tool"]
        for i, o in enumerate(seq):
            if o["op"] == "repeat":
                assert o["prompt"] in [p["prompt"] for p in seq[:i] if p["op"] == "tool"], slot
            if o["op"].startswith("restart"):
                assert i < len(seq) - 1 and seq[i - 1]["op"] == "compress", slot
        assert len(set(tools)) == len(tools)  # originals are distinct, so planned repeats are exact
        assert all(drive.BANK[o["prompt"]][0] == "tool" for o in seq if o["op"] in ("tool", "repeat"))
    assert sum(o["op"] == "repeat" for o in ops) == 12
    assert {o["slot"][0] for o in ops if o["op"] == "restart_warm"} == {"T"}
    assert {o["slot"][0] for o in ops if o["op"] == "restart_cold"} == {"A"}


def test_dry_run_prints_counts(capsys):
    assert drive.main(["--hermes-python", "x", "--hermes-src", "x", "--home", "h", "--workdir", "w",
                       "--events", "e.jsonl", "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["planned"]["turns"] >= 240 and out["targets"]["undo_resend"] == 12


def test_live_home_is_refused(tmp_path):
    from bench.instruments.reliability.hosts import real_hermes_dir
    with pytest.raises(SystemExit):
        drive.main(["--hermes-python", "x", "--hermes-src", "x", "--home", str(real_hermes_dir() / "p"),
                    "--workdir", str(tmp_path), "--events", str(tmp_path / "e.jsonl"), "--dry-run"])


# -- executor against stub processes -------------------------------------------------------------------------------
class StubAcp:
    sessions = Counter()

    def __init__(self, log):
        self.log, self.cancel = log, threading.Event()
        self.log.append("start")

    def initialize(self, _t):
        pass

    def new_session(self, _cwd, _t):
        StubAcp.sessions["n"] += 1
        return f"acp-{StubAcp.sessions['n']}"

    def load_session(self, sid, _cwd, _t):
        self.log.append(f"load {sid}")

    def notify(self, method, _params):
        if method == "session/cancel":
            self.cancel.set()

    def prompt(self, sid, text, _t):
        if text == "/context":
            return f"{S} Context usage: ~1,000 / 32,000 tokens (3.1%)", "end_turn"
        if text == "/compress":
            return f"Context compressed: 9 -> 3 messages {S}", "end_turn"
        if "600-word" in text and not self.cancel.wait(0.5):
            return f"{S} reply", "end_turn"
        stop = "cancelled" if self.cancel.is_set() else "end_turn"
        self.cancel.clear()
        return f"{S} reply to {text[:20]}", stop

    def kill(self):
        self.log.append("kill")

    def close(self):
        return 0


class StubTui:
    def __init__(self):
        self.n, self.live, self.events = 0, {}, []

    def ready(self, _t):
        pass

    def call(self, method, params, _t):
        if method == "session.create":
            self.n += 1
            self.live[f"rt{self.n}"] = f"tui-{self.n}"
            return {"session_id": f"rt{self.n}", "stored_session_id": f"tui-{self.n}"}
        if method == "session.resume":
            self.n += 1
            self.live[f"rt{self.n}"] = params["session_id"]
            return {"session_id": f"rt{self.n}", "stored_session_id": params["session_id"]}
        if method == "session.close":
            return {"closed": self.live.pop(params["session_id"], None) is not None}
        if method == "session.undo":
            return {"removed": 2}
        if method == "session.compress":
            return {"compressed": True}
        assert method == "prompt.submit" and params["session_id"] in self.live
        return {"status": "streaming"}

    def wait(self, kinds, sid, _t):
        if "message.complete" in kinds:
            return {"type": "message.complete", "session_id": sid, "payload": {"status": "complete", "text": S}}
        return {"type": "session.info", "session_id": sid, "payload": {"running": False}}

    def kill(self):
        pass

    def close(self):
        return 0


def stub_driver(tmp_path, workdir_name=S):
    workdir = tmp_path / workdir_name  # the prompts name files under it: the sentinel is in every tool prompt
    workdir.mkdir()
    ev = drive.Events(tmp_path / "events.jsonl")
    log = []
    d = drive.Driver(python="unused", src=tmp_path, home=tmp_path, workdir=workdir, events=ev, env={},
                     cancel_after=0.05, acp_factory=lambda: StubAcp(log), tui_factory=StubTui)
    return d, ev, log


def test_events_carry_no_prompt_or_reply_text(tmp_path):
    d, ev, log = stub_driver(tmp_path)
    tg = drive.targets()
    ops = drive.plan(3, tg)
    summary = d.run(ops, tg)
    raw = ev.path.read_text()
    assert S not in raw
    texts = {drive.prompt_text(i, d.workdir) for i in range(len(drive.BANK))}
    lines = [json.loads(x) for x in raw.splitlines()]
    assert not any(t[:40] in raw for t in texts)
    assert all(set(e) >= {"t", "kind", "session", "detail"} for e in lines)
    assert summary["executed"] == summary["planned"]["ops"] and summary["errors"] == {} and summary["exit"] == 0
    assert summary["context_length"] == 32000
    kinds = Counter(e["kind"] for e in lines)
    assert kinds["restart_cold"] == 6 and kinds["restart_warm"] == 6 and kinds["undo_resend"] == 12
    assert kinds["source2_session"] == drive.TUI_SLOTS and kinds["delegate_prompt"] == 12
    assert log.count("kill") >= 6 and lines[-1]["kind"] == "summary"


@pytest.mark.parametrize("where,line", [
    ("host-logs/tui.stderr", "LCM-X is NOT active in this process: Hermes ignored register_context_engine()"),
    ("agent.log", "2026-11-07 02:42:50,585 WARNING run_agent: Context engine 'lcm-x' not found — falling back"),
])
def test_a_host_without_lcmx_stops_the_run(tmp_path, where, line):
    d, ev, _log = stub_driver(tmp_path)
    target = d.workdir / where if where.startswith("host-logs") else tmp_path / "logs" / where  # home = tmp_path

    class LoudTui(StubTui):
        def ready(self, _t):  # the host writes the notice while it loads, before its first turn completes
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(line + "\n")
    d.make_tui = LoudTui
    ops = [{"op": "turn", "slot": "A1", "prompt": 0}, {"op": "turn", "slot": "T1", "prompt": 1},
           {"op": "turn", "slot": "T1", "prompt": 2}, {"op": "turn", "slot": "A1", "prompt": 3}]
    summary = d.run(ops, drive.targets())
    lines = [json.loads(x) for x in ev.path.read_text().splitlines()]
    bad = [e["detail"] for e in lines if e["kind"] == "error"]
    assert bad == [f"lcm_inactive entry=tui source={Path(where).name}"]
    assert summary["stopped"] == "lcm_inactive" and summary["exit"] == drive.EXIT_LCM_INACTIVE != 0
    assert summary["executed"] == {"turn": 2}  # nothing after the first T1 turn ran
    assert summary["errors"] == {"lcm_inactive": 1}


def test_a_late_cancel_is_an_error_not_an_undo(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    d.cancel_after = 5.0  # the stub ends the prompt first
    d.run([{"op": "undo_resend", "slot": "A1", "prompt": drive.CATS["undo"][0]}], drive.targets())
    kinds = [json.loads(x)["kind"] for x in ev.path.read_text().splitlines()]
    assert "undo_resend" not in kinds and "error" in kinds


# -- the events contract with audit.py ------------------------------------------------------------------------------
def minimal_home(tmp_path, sessions):
    """A state.db with the tables audit.py reads (one user/assistant turn per session) and an empty LCM-X store."""
    import benchmarking.longmemeval as lme
    lme._ensure_hermes_lcm_package()
    from hermes_lcm.store import MessageStore
    home = tmp_path / "home"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as con:
        con.execute("create table sessions (id text primary key, source text not null, parent_session_id text,"
                    " started_at real not null, ended_at real, end_reason text)")
        con.execute("create table messages (id integer primary key autoincrement, session_id text not null,"
                    " role text not null, content text, tool_call_id text, tool_calls text, timestamp real not null,"
                    " active integer not null default 1)")
        for sid, source in sessions.items():
            con.execute("insert into sessions values (?,?,?,?,?,?)", (sid, source, None, 1000.0, None, None))
            for role in ("user", "assistant"):
                con.execute("insert into messages (session_id, role, content, timestamp) values (?,?,?,?)",
                            (sid, role, f"{role} {sid}", 1.0))
    MessageStore(home / "lcm.db").close()
    return home


def test_driver_events_feed_audit_events(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    tg = drive.targets()
    summary = d.run(drive.plan(11, tg), tg)
    sources = {s["session"]: ("acp" if slot.startswith("A") else "tui") for slot, s in d.slots.items()}
    home = minimal_home(tmp_path, sources)
    out = tmp_path / "audit.json"
    assert audit.main(["--state-db", str(home / "state.db"), "--lcm-db", str(home / "lcm.db"),
                       "--events", str(ev.path), "--out", str(out)]) == 0
    cov = json.loads(out.read_text())["coverage"]
    ex = summary["executed"]
    assert cov["restarts_cold"]["value"] == ex["restart_cold"] == 6
    assert cov["restarts_warm"]["value"] == ex["restart_warm"] == 6
    assert cov["undo_resend"]["value"] == ex["undo_resend"] == 12
    assert cov["restarts"]["met"] and cov["host_sources"]["value"] == 2
    # every kind audit.py knows is one the driver writes
    written = {json.loads(x)["kind"] for x in ev.path.read_text().splitlines()}
    assert set(audit.EVENT_KINDS) <= written


# -- host-backed smoke ---------------------------------------------------------------------------------------------
def smoke_host():
    """A reliability host (hosts file: --hosts-file / $LCM_RELIABILITY_HOSTS / the default) whose tree has both entry
    points; $LCM_GATE_KIT_HOST picks one by name."""
    from bench.instruments.reliability import hosts
    path = hosts.hosts_file(None)
    if not path.is_file():
        return None, f"no reliability hosts file at $LCM_RELIABILITY_HOSTS or the default path ({path.name})"
    try:
        found = hosts.load(path)
    except (OSError, ValueError) as exc:
        return None, f"hosts file unusable: {exc}"
    want = os.environ.get("LCM_GATE_KIT_HOST")
    for name, h in sorted(found.items(), key=lambda kv: (kv[0] != want, kv[0] != "r34.15-0.21.5", kv[0])):
        src = Path(h["src"])
        undo = src / "tui_gateway" / "methods_session.py"
        if (Path(h["python"]).parent / "hermes").exists() and undo.is_file() and "session.undo" in undo.read_text():
            return (name, h), None
    return None, "no configured host has both `hermes acp` and a TUI gateway with session.undo"


HOST, WHY = smoke_host()


def fake_main(hold_text):
    lock, n, held = threading.Lock(), Counter(), set()

    def main(messages):
        last = messages[-1]
        if last["role"] == "tool":
            return {"content": "done."}
        user = last["content"] if last["role"] == "user" else ""
        with lock:
            n["call"] += 1
            cid = f"call_{n['call']:03d}"
        if m := re.search(r"read_file tool to read (\S+\.txt)", user):
            return {"tool_calls": [{"id": cid, "name": "read_file", "arguments": json.dumps({"path": m.group(1)})}]}
        if "delegate_task tool" in user:
            goal = "Compute 11 + 40 and reply with the number only."
            return {"tool_calls": [{"id": cid, "name": "delegate_task", "arguments": json.dumps({"goal": goal})}]}
        if hold_text in user and hold_text not in held:
            held.add(hold_text)
            return {"content": "late.", "hold": 30.0}  # in flight until the host drops the request
        return {"content": "ok."}
    return main


SMOKE_TOOL, SMOKE_UNDO_ACP, SMOKE_UNDO_TUI = drive.CATS["tool"][3], drive.CATS["undo"][0], drive.CATS["undo"][1]
SMOKE = [{"op": "turn", "slot": "A1", "prompt": drive.CATS["plain"][0]},
         {"op": "tool", "slot": "A1", "prompt": SMOKE_TOOL},
         {"op": "repeat", "slot": "A1", "prompt": SMOKE_TOOL},
         {"op": "compress", "slot": "A1", "prompt": drive.CATS["long"][0]},
         {"op": "delegate", "slot": "A1", "prompt": drive.CATS["delegate"][0]},
         {"op": "undo_resend", "slot": "A1", "prompt": SMOKE_UNDO_ACP},
         {"op": "restart_cold", "slot": "A1"},
         {"op": "turn", "slot": "A1", "prompt": drive.CATS["plain"][1]},
         {"op": "turn", "slot": "T1", "prompt": drive.CATS["plain"][2]},
         {"op": "restart_warm", "slot": "T1"},
         {"op": "undo_resend", "slot": "T1", "prompt": SMOKE_UNDO_TUI},
         {"op": "turn", "slot": "T1", "prompt": drive.CATS["plain"][3]}]


@pytest.mark.skipif(HOST is None, reason=f"host-backed smoke: {WHY}")
def test_smoke_against_fake_provider_then_audit(tmp_path):
    from bench.instruments.reliability import fake_provider as FP, plugin_tree, process_cell as PC
    name, host = HOST
    plugin = {**plugin_tree.identity(REPO), "tree": str(REPO)}
    home, workdir, art = tmp_path / "home", tmp_path / "work", tmp_path / "records"
    for p in (home / "plugins", workdir, art):
        p.mkdir(parents=True)
    (home / "plugins" / plugin["dir"]).symlink_to(REPO)
    (home / "models_dev_cache.json").write_text(json.dumps({"rel": {"id": "rel", "name": "reliability fake",
                                                                     "models": {}}}))
    drive.write_notes(workdir)
    hold = drive.prompt_text(SMOKE_UNDO_ACP, workdir)
    provider = FP.FakeProvider(art / "provider-requests.jsonl", main=fake_main(hold)).start()
    proxy = FP.ProxySink(art / "proxy-attempts.jsonl")
    cell = {"id": "gate-kit/smoke", "in_place": True, "lcm_env": {}, "window": 64000,
            "toolsets": ["todo", "context_engine", "file", "delegation"]}
    config = PC.config_yaml(cell, plugin, provider.base_url).replace("plugins:\n", "plugins:\n  load_timeout_seconds: 60\n")
    (home / "config.yaml").write_text(config + "  tui: [todo, context_engine, file, delegation]\n")
    base = {"PATH": "/usr/bin:/bin", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
            "LCM_SUMMARY_MODEL": "rel/lcm-summary", "LCM_SUMMARY_SPEND_MAX_CALLS": "100000",
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
            **{k: proxy.url for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")}}
    wrap = (PC.SANDBOX_EXEC, "-p", PC.SANDBOX) if os.path.exists(PC.SANDBOX_EXEC) else ()
    ev = drive.Events(art / "events.jsonl")
    d = drive.Driver(python=host["python"], src=Path(host["src"]), home=home, workdir=workdir, events=ev,
                     env=drive.host_env(base, home, workdir, Path(host["src"])), prompt_timeout=180, max_hours=0.5,
                     cancel_after=3.0, wrap=wrap)
    tg = {**{k: 0 for k in drive.PLANNED_FLOORS}, "delegated_children": 1}
    try:
        summary = d.run(SMOKE, tg)
    finally:
        provider.stop()
        proxy.stop()
    lines = [json.loads(x) for x in ev.path.read_text().splitlines()]
    detail = [(e["kind"], e["detail"]) for e in lines if e["kind"] != "summary"]
    assert summary["errors"] == {} and summary["exit"] == 0, (summary, detail, (workdir / "host-logs" / "acp.stderr").read_text()[-3000:])
    assert summary["executed"] == summary["planned"]["ops"], detail
    assert not (art / "proxy-attempts.jsonl").exists() or not (art / "proxy-attempts.jsonl").read_text().strip()
    assert S not in ev.path.read_text() and hold[:40] not in ev.path.read_text()
    exits = sorted(e["detail"] for e in lines if e["kind"] == "host_exit")
    assert exits == ["entry=acp returncode=0", "entry=tui returncode=0"], exits  # both ended their sessions by EOF
    kinds = Counter(e["kind"] for e in lines)
    assert {k: kinds[k] for k in audit.EVENT_KINDS} == {"restart_cold": 1, "restart_warm": 1, "undo_resend": 2,
                                                        "delegate_prompt": 1, "compress_request": 1}
    assert summary["delegate"]["children_in_state_db"] >= 1 and summary["context_length"] == 64000
    logs = sorted((home / "logs").glob("*.log"))
    out = tmp_path / "audit.json"
    assert audit.main(["--state-db", str(home / "state.db"), "--lcm-db", str(home / "lcm.db"), "--events", str(ev.path),
                       *[a for p in logs for a in ("--log", str(p))], "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    sessions = {s["session"] for s in d.slots.values()}
    lineages = {result["parity"]["sessions"][s]["lineage"] for s in sessions if s in result["parity"]["sessions"]}
    assert len(lineages) == 2 and lineages <= set(result["parity"]["lineages"]), result["parity"]["sessions"]
    sources = {result["parity"]["sessions"][s]["source"] for s in sessions}
    assert sources == {"acp", "tui"}
    cov = result["coverage"]
    assert (cov["restarts_cold"]["value"], cov["restarts_warm"]["value"], cov["undo_resend"]["value"]) == (1, 1, 2)
    per_source = {result["parity"]["sessions"][s]["source"]: {k: result["parity"]["lineages"][g][k] for k in (
        "expected_rows", "stored_rows", "missing_rows", "silent_surplus_rows")} for s in sessions
        for g in [result["parity"]["sessions"][s]["lineage"]]}
    print(json.dumps({"host": name, "parity_per_source": per_source}))  # reported, not a bar: the gate is audit's
