"""Gate-kit driver: the planner (pure), the executor's events against stub host processes, the events contract with
audit.py --events, and a host-backed smoke (a real ``hermes acp`` and TUI gateway against the localhost fake provider)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
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
    # ACP has no undo: every undo-and-resend is on TUI; ACP cancel-and-resend is its own kind and never counted
    assert {o["slot"][0] for o in ops if o["op"] == "undo_resend"} == {"T"}
    assert {o["slot"][0] for o in ops if o["op"] == "cancel_resend"} == {"A"}
    assert drive.planned_counts(ops)["undo_resend"] == sum(o["op"] == "undo_resend" for o in ops) == 12


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
    workdir.mkdir(exist_ok=True)
    ev = drive.Events(tmp_path / "events.jsonl")
    log = []
    d = drive.Driver(python="unused", src=tmp_path, home=tmp_path, workdir=workdir, events=ev, env={},
                     cancel_after=0.05, acp_factory=lambda: StubAcp(log), tui_factory=StubTui)
    return d, ev, log


def test_gateway_replacement_resumes_each_stored_session(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    d.send("T1", 0)
    d.send("T2", 1)
    stored = {slot: s["session"] for slot, s in d.slots.items()}
    old = d.tui

    def gone(*_args):
        raise drive.AD.ProcessGone("gateway exited")

    old.call = gone
    d.guarded({"op": "turn", "slot": "T1", "prompt": 2})
    assert d.tui is None
    for slot in stored:
        d.send(slot, 3)
        assert d.tui.live[d.slots[slot]["rt"]] == stored[slot]
    assert sum(json.loads(x)["kind"] == "source2_session" for x in ev.path.read_text().splitlines()) == 2
    shared = [e for e in audit.read_events(ev.path) if e["kind"] == "restart_cold_shared"]
    assert Counter(e["session"] for e in shared) == Counter(stored.values())


def test_failed_delegates_consume_plan_and_topup_attempt_budget(tmp_path, monkeypatch):
    d, _ev, _log = stub_driver(tmp_path)
    attempts = []

    def fail(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 5:  # bound the broken implementation without waiting for its deadline
            d.stop_at = 0
        raise drive.AD.DriverError("delegate failed")

    monkeypatch.setattr(d, "send", fail)
    monkeypatch.setattr(d, "children", lambda: 0)
    summary = d.run([{"op": "delegate", "slot": "A1", "prompt": drive.CATS["delegate"][0]}],
                    {"delegated_children": 1})
    assert len(attempts) == 2 and summary["stopped"] is None
    assert summary["delegate"]["extra_prompts"] == 1
    assert summary["delegate"]["attempts"] == 2 and summary["delegate"]["prompts"] == 0


@pytest.mark.parametrize("shape", ["unfinished", "rotation", "auxiliary", "previous_run", "ended"])
def test_delegate_topup_counts_only_audit_eligible_children(tmp_path, shape):
    d, _ev, _log = stub_driver(tmp_path)
    t0 = d.run_start - 100 if shape == "previous_run" else d.run_start  # a retained home's earlier run
    with sqlite3.connect(tmp_path / "state.db") as con:
        con.execute("create table sessions (id text, source text, parent_session_id text, started_at real,"
                    " ended_at real, end_reason text)")
        con.execute("insert into sessions values ('parent', 'subagent', NULL, ?, ?, 'compression')", (t0, t0 + 10))
        con.executemany("insert into sessions values (?, 'subagent', 'parent', ?, ?, NULL)",
                        [(f"child{i}", t0 + (10 if shape == "rotation" else 1),
                          None if shape == "unfinished" else t0 + 20) for i in range(12)])
    if shape == "auxiliary":
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "agent.log").write_text("".join(
            f"2026-11-06 10:00:00 INFO hermes_lcm.engine: LCM reclassified session child{i} "
            "as auxiliary at first ingest\n" for i in range(12)))
    expected = 12 if shape == "ended" else 0
    count = d.children()
    summary = d.run([], {"delegated_children": 12})
    assert count == summary["delegate"]["children_in_state_db"] == expected
    assert summary["delegate"]["extra_prompts"] == (0 if expected else 24)


def test_cli_paths_are_absolute_for_hosts_and_prompts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "config.yaml").touch()
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "host-python").touch()
    (tmp_path / "venv" / "bin" / "python").symlink_to(tmp_path / "host-python")

    def run(d, _ops, _tg):
        assert all(p.is_absolute() for p in (d.src, d.home, d.workdir, d.ev.path))
        assert d.src == tmp_path / "src" and d.home == tmp_path / "home"
        assert d.ev.path == tmp_path / "events.jsonl"
        made = []
        monkeypatch.setattr(drive.AD, "AcpProcess", lambda *a: made.append(a))
        monkeypatch.setattr(drive, "Gateway", lambda *a: made.append(a))
        d.make_acp()
        d.make_tui()
        assert made[0][0][0] == str(tmp_path / "venv" / "bin" / "hermes")
        assert made[1][0][0] == str(tmp_path / "venv" / "bin" / "python")
        for argv, env, cwd, stderr in made:
            assert Path(argv[0]).is_absolute() and cwd.is_absolute() and stderr.is_absolute()
            assert all(Path(env[k]).is_absolute() for k in ("HERMES_HOME", "HOME", "TMPDIR", "HERMES_PYTHON_SRC_ROOT"))
        assert made[1][1]["PYTHONPATH"] == str(tmp_path / "src")
        assert str(tmp_path / "run" / "notes" / "note-00.txt") in drive.prompt_text(drive.CATS["tool"][0], d.workdir)
        return dict(executed={}, errors={}, delegate={}, context_length=None, stopped=None, exit=0)

    monkeypatch.setattr(drive.Driver, "run", run)
    assert drive.main(["--hermes-python", "venv/bin/python", "--hermes-src", "src", "--home", "home",
                       "--workdir", "run", "--events", "events.jsonl"]) == 0


@pytest.mark.parametrize("where", ["acp.stderr", "tui.stderr", "agent.log"])
def test_inactive_scan_ignores_old_lines_but_reads_new_ones(tmp_path, where):
    target = tmp_path / ("logs" if where == "agent.log" else "work/host-logs") / where
    target.parent.mkdir(parents=True)
    warning = "LCM-X is NOT active in this process\n"
    target.write_text(warning)
    d, _ev, _log = stub_driver(tmp_path, "work")
    assert d.inactive("tui") == []
    with target.open("a") as fh:
        fh.write(warning)
    assert d.inactive("tui") == [("acp" if where == "acp.stderr" else "tui", where)]
    assert d.inactive("tui") == []


@pytest.mark.parametrize("slot", ["A1", "T1"])
def test_compress_request_precedes_triggering_turn(tmp_path, monkeypatch, slot):
    d, ev, _log = stub_driver(tmp_path)
    send = d.send

    def trigger(*args):
        assert ev.path.exists(), "compress_request must be written before the triggering turn"
        requests = [json.loads(x) for x in ev.path.read_text().splitlines() if json.loads(x)["kind"] == "compress_request"]
        assert len(requests) == 1 and requests[0]["session"] == d.slots[slot]["session"]
        return send(*args)

    monkeypatch.setattr(d, "send", trigger)
    d.op({"op": "compress", "slot": slot, "prompt": drive.CATS["long"][0]})
    events = [json.loads(x) for x in ev.path.read_text().splitlines()]
    assert sum(e["kind"] == "compress_request" for e in events) == 1
    assert any(e["kind"] == "compress_result" and "manual=compressed" in e["detail"] for e in events)


def test_shared_acp_restart_records_all_bound_sessions(tmp_path):
    d, ev, log = stub_driver(tmp_path)
    slots = [f"A{i}" for i in range(1, 7)]
    for slot in slots:
        d.send(slot, 0)
    d.op({"op": "restart_cold", "slot": "A1"})
    for slot in slots[1:]:
        d.send(slot, 1)
    events = audit.read_events(ev.path)
    assert [e["session"] for e in events if e["kind"] == "restart_cold"] == [d.slots["A1"]["session"]]
    restarts = [e for e in events if e["kind"] in ("restart_cold", "restart_cold_shared")]
    assert Counter(e["session"] for e in restarts) == Counter(s["session"] for s in d.slots.values())
    assert all(f"load {s['session']}" in log for s in d.slots.values())
    home = minimal_home(tmp_path, {s["session"]: "acp" for s in d.slots.values()})
    out = audit.audit(home / "state.db", home / "lcm.db", events_path=ev.path)
    assert out["coverage"]["restarts_cold"]["value"] == 1


def test_failed_planned_load_has_no_cold_restart(tmp_path):
    d, ev, log = stub_driver(tmp_path)
    d.send("A1", 0)

    class FailedLoad(StubAcp):
        def load_session(self, *_args):
            raise drive.AD.DriverError("load failed")

    d.make_acp = lambda: FailedLoad(log)
    d.guarded({"op": "restart_cold", "slot": "A1"})
    assert not any(e["kind"] == "restart_cold" for e in audit.read_events(ev.path))


def test_acp_crash_records_shared_only_after_reload(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    for slot in ("A1", "A2"):
        d.send(slot, 0)

    def gone(*_args):
        raise drive.AD.ProcessGone("host exited")

    d.acp.prompt = gone
    d.guarded({"op": "turn", "slot": "A1", "prompt": 0})
    assert not any(e["kind"] in audit.RESTARTS for e in audit.read_events(ev.path))
    for slot in ("A1", "A2"):
        d.send(slot, 0)
    assert Counter(e["kind"] for e in audit.read_events(ev.path)) == {"restart_cold_shared": 2}


@pytest.mark.parametrize("slot", ["A1", "T1"])
@pytest.mark.parametrize("stop", ["submit_busy", "session_error", "cancelled"])
def test_failed_resend_has_no_undo_event(tmp_path, monkeypatch, slot, stop):
    d, ev, _log = stub_driver(tmp_path)
    d.send(slot, 0)
    stops = iter(["cancelled" if slot.startswith("A") else "complete", stop])
    monkeypatch.setattr(d, "send", lambda *_a, **_kw: next(stops))
    kind = "cancel_resend" if slot.startswith("A") else "undo_resend"
    d.op({"op": kind, "slot": slot, "prompt": drive.CATS["undo"][0]})
    assert not any(e["kind"] == kind for e in audit.read_events(ev.path))


def test_busy_tui_resend_has_no_undo_event(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    d.tui_rt("T1")
    call, submits = d.tui.call, []

    def busy(method, params, timeout):
        if method == "prompt.submit":
            submits.append(1)
            if len(submits) == 2:
                return {"status": "busy"}
        return call(method, params, timeout)

    d.tui.call = busy
    d.op({"op": "undo_resend", "slot": "T1", "prompt": drive.CATS["undo"][0]})
    assert len(submits) == 2 and not any(e["kind"] == "undo_resend" for e in audit.read_events(ev.path))


@pytest.mark.parametrize("status", ["busy", "error"])
def test_failed_first_tui_send_skips_undo(tmp_path, status):
    d, ev, _log = stub_driver(tmp_path)
    d.tui_rt("T1")
    call, calls = d.tui.call, []

    def reject(method, params, timeout):
        calls.append(method)
        if method == "prompt.submit" and calls.count(method) == 1:
            return {"status": status}
        return call(method, params, timeout)

    d.tui.call = reject
    d.op({"op": "undo_resend", "slot": "T1", "prompt": drive.CATS["undo"][0]})
    assert "session.undo" not in calls and calls.count("prompt.submit") == 1
    assert ev.errors[f"turn_submit_{status}"] == 1
    assert not any(e["kind"] == "undo_resend" for e in audit.read_events(ev.path))


@pytest.mark.parametrize("stop", ["busy", "error"])
def test_failed_first_acp_send_skips_resend(tmp_path, stop):
    d, ev, _log = stub_driver(tmp_path)
    d.acp_sid("A1")
    calls = []
    d.acp.prompt = lambda *_a: (calls.append(1) or "", stop)
    d.op({"op": "cancel_resend", "slot": "A1", "prompt": drive.CATS["undo"][0]})
    assert calls == [1] and ev.errors[f"turn_{stop}"] == 1
    assert not any(e["kind"] == "cancel_resend" for e in audit.read_events(ev.path))


def test_shared_kill_sibling_restarts_only_after_its_reload(tmp_path):
    """A sibling killed with the planned slot's process has no restart until it reloads, so a compaction in a
    sibling that never reloads cannot count toward compactions_then_restart."""
    d, ev, _log = stub_driver(tmp_path)
    for slot in ("A1", "A2", "A3"):
        d.acp_sid(slot)
    before = time.time()
    d.op({"op": "restart_cold", "slot": "A1"})
    shared = lambda: [e for e in audit.read_events(ev.path) if e["kind"] == "restart_cold_shared"]  # noqa: E731
    assert [e["kind"] for e in audit.read_events(ev.path) if e["kind"].startswith("restart")] == ["restart_cold"]
    assert shared() == []  # A2 and A3 were killed but have not reloaded
    d.send("A2", 0)
    assert [e["detail"] for e in shared()] == ["slot=A2"] and shared()[0]["killed_at"] >= before
    assert "A3" in d.cold_pending  # never reloaded: no restart recorded


def test_event_log_failure_stops_the_run_and_shuts_down_both_hosts(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    d.acp_sid("A1")
    d.tui_rt("T1")
    ev.path = tmp_path / "unwritable"
    ev.path.mkdir()  # opening a directory for append fails
    with pytest.raises(drive.EventLogError):
        d.run([{"op": "turn", "slot": "A1", "prompt": drive.CATS["plain"][0]}], drive.targets())
    assert d.acp is None and d.tui is None


def test_events_parent_is_created(tmp_path):
    ev = drive.Events(tmp_path / "new" / "dir" / "events.jsonl")
    ev.write("turn", "s", "x")
    assert ev.path.is_file()


def test_unreadable_state_db_is_an_error_event(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    (tmp_path / "state.db").write_bytes(b"not a database")
    assert d.children() is None and ev.errors["child_count_failed"] == 1


def test_failed_warm_resume_retries_before_next_turn(tmp_path):
    d, ev, _log = stub_driver(tmp_path)
    d.tui_rt("T1")
    call, resumes = d.tui.call, []

    def fail_once(method, params, timeout):
        if method == "session.resume":
            resumes.append(1)
            if len(resumes) == 1:
                raise drive.AD.JsonRpcError(method, {"code": -32000, "message": "resume failed"})
        return call(method, params, timeout)

    d.tui.call = fail_once
    d.guarded({"op": "restart_warm", "slot": "T1"})
    assert not d.tui.live
    assert d.send("T1", 0) == "complete"
    assert len(resumes) == 2 and "T1" in d.tui_loaded
    assert d.tui.live[d.slots["T1"]["rt"]] == d.slots["T1"]["session"]
    assert not any(e["kind"] == "restart_warm" for e in audit.read_events(ev.path))


@pytest.mark.parametrize("closed", [False, True])
def test_failed_warm_restart_has_no_event(tmp_path, closed):
    d, ev, _log = stub_driver(tmp_path)
    d.tui_rt("T1")

    def fail(method, *_args):
        if method == "session.close":
            return {"closed": closed}
        if not closed:
            return {"session_id": "resumed"}
        raise drive.AD.DriverError("resume failed")

    d.tui.call = fail
    d.guarded({"op": "restart_warm", "slot": "T1"})
    assert not any(e["kind"] == "restart_warm" for e in audit.read_events(ev.path))
    assert ev.errors


@pytest.mark.parametrize("slot", ["A1", "T1"])
@pytest.mark.parametrize("raises", [False, True])
def test_failed_compress_request_is_not_attributed(tmp_path, monkeypatch, slot, raises):
    d, ev, _log = stub_driver(tmp_path)

    def fail(*_args):
        if raises:
            raise drive.AD.ProcessGone("host exited")
        return "submit_busy"

    monkeypatch.setattr(d, "send", fail)
    d.guarded({"op": "compress", "slot": slot, "prompt": drive.CATS["long"][0]})
    events = [json.loads(x) for x in ev.path.read_text().splitlines()]
    request = next(e for e in events if e["kind"] == "compress_request")
    assert any(e["kind"] == "compress_request_failed" and e["session"] == request["session"] for e in events)
    assert audit.attribute_compactions([(None, request["t"])], audit.read_events(ev.path)) == ([(None, request["t"])], 1)
    ev.write("compress_request", request["session"], t=request["t"] + 300)
    assert audit.attribute_compactions([(None, request["t"] + 300)], audit.read_events(ev.path)) == (
        [(request["session"], request["t"] + 300)], 0)


def test_inactive_scan_rescans_a_truncated_log(tmp_path):
    d, _ev, _log = stub_driver(tmp_path)
    path = d.logs / "acp.stderr"
    path.write_text("old log\n" * 100)
    assert d.inactive("acp") == []
    path.write_text("LCM-X is NOT active in this process\n")
    assert d.inactive("acp") == [("acp", "acp.stderr")]
    assert d.inactive("acp") == []


def test_nonempty_events_refused_unless_append_is_explicit(tmp_path, monkeypatch, capsys):
    path = tmp_path / "events.jsonl"
    stale = '{"kind":"restart_cold","session":"stale","t":1}\n'
    path.write_text(stale)
    (tmp_path / "config.yaml").touch()
    monkeypatch.setattr(drive, "plan", lambda *_args: [])
    argv = ["--hermes-python", "unused", "--hermes-src", str(tmp_path), "--home", str(tmp_path),
            "--workdir", str(tmp_path / "work"), "--events", str(path)]
    with pytest.raises(SystemExit) as exc:
        drive.main(argv)
    assert exc.value.code == 2 and "--append-events" in capsys.readouterr().err
    assert path.read_text() == stale and not (tmp_path / "work").exists()
    assert drive.main([*argv, "--append-events"]) == 0
    assert path.read_text().startswith(stale)
    assert json.loads(path.read_text().splitlines()[-1])["kind"] == "summary"


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
    d.run([{"op": "cancel_resend", "slot": "A1", "prompt": drive.CATS["undo"][0]}], drive.targets())
    kinds = [json.loads(x)["kind"] for x in ev.path.read_text().splitlines()]
    assert "cancel_resend" not in kinds and "undo_resend" not in kinds and "error" in kinds


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
    cold_events = sum(json.loads(x)["kind"] == "restart_cold" for x in ev.path.read_text().splitlines())
    assert cov["restarts_cold"]["value"] == cold_events == ex["restart_cold"] == 6
    assert cov["restarts_warm"]["value"] == ex["restart_warm"] == 6
    assert cov["undo_resend"]["value"] == ex["undo_resend"] == 12
    assert cov["restarts"]["met"] and cov["host_sources"]["value"] == 2
    # every kind audit.py knows is one the driver writes
    written = {json.loads(x)["kind"] for x in ev.path.read_text().splitlines()}
    assert set(audit.EVENT_KINDS) - {"compress_request_failed"} <= written


# -- host-backed smoke ---------------------------------------------------------------------------------------------
def smoke_host():
    """A reliability host (hosts file: --hosts-file / $LCM_RELIABILITY_HOSTS / the default) whose tree has both entry
    points; $LCM_GATE_KIT_HOST names it (the smoke skips without it)."""
    from bench.instruments.reliability import hosts
    path = hosts.hosts_file(None)
    if not path.is_file():
        return None, f"no reliability hosts file at $LCM_RELIABILITY_HOSTS or the default path ({path.name})"
    try:
        found = hosts.load(path)
    except (OSError, ValueError) as exc:
        return None, f"hosts file unusable: {exc}"
    want = os.environ.get("LCM_GATE_KIT_HOST")
    if not want:
        # Some hosts reach the network at start (blocked by the smoke's sandbox) or cannot load LCM-X; name one.
        return None, "set LCM_GATE_KIT_HOST to a reliability host that runs without network and loads LCM-X"
    for name, h in sorted(found.items(), key=lambda kv: (kv[0] != want, kv[0])):
        if name != want:
            continue
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
         {"op": "cancel_resend", "slot": "A1", "prompt": SMOKE_UNDO_ACP},
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
    assert {k: kinds[k] for k in audit.EVENT_KINDS if kinds[k]} == {"restart_cold": 1, "restart_warm": 1, "undo_resend": 1,
                                                        "delegate_prompt": 1, "compress_request": 1}
    assert kinds["cancel_resend"] == 1  # recorded, never counted as an undo
    assert summary["delegate"]["children_in_state_db"] >= 1 and summary["context_length"] == 64000
    for db in ("state.db", "lcm.db"):  # the audit reads immutable files: open each once after the run to checkpoint
        con = sqlite3.connect(home / db)
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            con.close()
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
    assert (cov["restarts_cold"]["value"], cov["restarts_warm"]["value"], cov["undo_resend"]["value"]) == (1, 1, 1)
    per_source = {result["parity"]["sessions"][s]["source"]: {k: result["parity"]["lineages"][g][k] for k in (
        "expected_rows", "stored_rows", "missing_rows", "silent_surplus_rows")} for s in sessions
        for g in [result["parity"]["sessions"][s]["lineage"]]}
    print(json.dumps({"host": name, "parity_per_source": per_source}))  # reported, not a bar: the gate is audit's
