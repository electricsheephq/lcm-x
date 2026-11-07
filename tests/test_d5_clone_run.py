"""The whole run on a synthetic snapshot with a fake host and a fake edition; no gateway, no network."""

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

from bench.instruments.d5_clone import bars, run as r

MARKER = "synthetic-marker-7f3a9c"


def make_snapshot(root):
    root.mkdir()
    (root / "lcm-large-outputs").mkdir()
    (root / "lcm-large-outputs" / "out-1.json").write_text(MARKER)
    sessions = {"sess-l": 1000, "sess-a": 25, "sess-b": 22}
    with closing(sqlite3.connect(root / "store.db")) as lcm, closing(sqlite3.connect(root / "host-0.db")) as host:
        lcm.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,"
                    " observed_at REAL)")
        host.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, end_reason TEXT)")
        host.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)")
        for sid, count in sessions.items():
            host.execute("INSERT INTO sessions VALUES (?, NULL)", (sid,))
            rows = [(sid, "user" if i % 2 else "assistant", f"{MARKER} {sid} {i}") for i in range(count)]
            lcm.executemany("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", rows)
            host.executemany("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", rows)
        lcm.commit()
        host.commit()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    pin = {"schema": "d5-clone-snapshot-pin/1", "files": [
        {"name": p.relative_to(root).as_posix(), "size": p.stat().st_size,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (root.parent / "snapshot-pin.json").write_text(json.dumps(pin))
    return root.parent / "snapshot-pin.json"


def inventory(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}


class FakeHost(r.LocalHost):
    """Real clones, planning, rows, captures and capture removal; fake gateway, provider, edition and logs."""

    def __init__(self, config, run_dir, fail=None):
        super().__init__(config, run_dir)
        self.fail, self.armed, self.replays, self.calls = fail, [], [], []

    def free_bytes(self):
        return 0 if self.fail == "free_space" else r.NEED_FREE

    def t0_evidence(self, ctx, out):  # No gateway env or arm tree here: the files the edition reads, synthetic.
        return {"filters.json": b'{"ignore_session_patterns": [], "stateless_session_patterns": []}\n',
                "doctor.log": b"host_uid_mode: synthetic\n"}

    def pins(self):
        if self.fail == "P1":
            return {"verdict": "FAIL", "mismatches": ["manifest"]}  # As pins.verify: no hashes on a failure.
        return {"verdict": "PASS", "mismatches": [], "rules_sha256": "f" * 64}

    def rules(self):  # The real configure seams call these, so the test sees the order and the inputs.
        return {"identity": lambda row: (row["role"], row["content"]), "credits": (),
                "covered": lambda sid: self.calls.append(("covered", sid)) or True,
                "configure_planning": lambda **kw: self.calls.append(("configure_planning", kw)),
                "configure_bar1": lambda **kw: self.calls.append(("configure_bar1", kw))}

    def preflight(self, name, plan):
        return {"verdict": "FAIL" if name == self.fail else "PASS"}

    def arm(self, name, home, plan):
        self.armed.append(name)
        for old, new in (("store.db", "lcm.db"), ("host-0.db", "state.db")):
            (home / old).rename(home / new)
        sessions = {c["name"]: c["session_id"] for c in plan["conversations"]}
        return {"name": name, "home": home, "sessions": sessions, "epoch": -1}

    def start(self, ctx):
        ctx["epoch"] += 1

    def stop(self, ctx):
        pass

    def turn(self, ctx, step):
        if self.fail == "warm" and step["kind"] != "cold" and not step["tools"]:
            return False  # Timed out or not completed: nothing stored.
        if self.fail == "ga_hollow" and "ga-rerun" in str(ctx["home"]):
            return False  # A GA re-run that completes no turn.
        if self.fail == "raise" and step["id"] == "e2-S2-2":
            raise RuntimeError("synthetic turn failure")
        sid = ctx["sessions"][step["conversation"]]
        with closing(sqlite3.connect(ctx["home"] / "lcm.db")) as lcm, closing(sqlite3.connect(ctx["home"] / "state.db")) as host:
            if step["kind"] == "cold" and ctx["epoch"]:  # A cold-bind re-store of one old row (a bar-2 copy).
                lcm.execute("INSERT INTO messages (session_id, role, content) SELECT session_id, role, content FROM"
                            " messages WHERE session_id = ? ORDER BY store_id LIMIT 1", (sid,))
                ctx.setdefault("copies", {})[step["id"]] = [lcm.execute("SELECT max(store_id) FROM messages").fetchone()[0]]
            rows = [(sid, "user", step["user_text"]), (sid, "assistant", f"{MARKER} reply {step['id']}")]
            for db in (host,) if self.fail == "lose" and ctx["name"] == "rc" and step["id"] == "e1-S1-1" else (lcm, host):
                db.executemany("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", rows)
                db.commit()
        return True

    def replay(self, ctx, step, t0, h8):
        self.replays.append(step["id"])
        if self.fail == "no_receipt" and step["id"] == "e1-S1-0":
            return None, True, None  # The previous-GA replay left no receipt.
        assert (Path(t0) / "capture-hashes.json").is_file() and (Path(h8) / "RC3").exists() is False
        checks = {k: {"verdict": "PASS"} for k in (bars.VALIDITY, *bars.COMPARISONS)}
        return {"checks": checks, "bind_identity": {"dup_store_ids": ctx.get("copies", {}).get(step["id"], [])}}, False, 0

    def egress(self, ctx):
        return 0

    def wait(self, seconds):
        pass

    def logs(self, ctx):
        return [f"synthetic log {MARKER}"]

    def roots(self, ctx):
        return [str(ctx["home"] / "plugins" / "hermes-lcm-x")]

    def checks(self, ctx):
        self.ga_checks = getattr(self, "ga_checks", 0) + (ctx["name"] == "ga")
        broken = ctx["name"] == "ga" and (self.fail == "ga_twice" or (self.fail in ("ga_once", "ga_hollow") and self.ga_checks == 1))
        return [{"kind": k, "check": k, "status": "fail" if broken and k == "schema" else "ok"}
                for k in bars.CHECK_KINDS]

    def facts(self, ctx):
        return {"cron_fired": 1, "payload_files_created": 1, "compactions_published": 1, "whitespace_rewrites": 1,
                "payload_swaps": 1}

    def stats(self, ctx):
        return {"replay_misses": {}, "calls_by_model": {"d5-agent": 36}}

    def close(self, ctx):
        pass


@pytest.fixture
def setup(tmp_path):
    snap = tmp_path / "snapshot"
    pin = make_snapshot(snap)
    config = {"snapshot_dir": str(snap), "snapshot_pin": str(pin), "nonce": "synthetic"}
    return snap, config, tmp_path / "run"


def test_full_fake_run_passes_reads_only_clones_and_carries_no_content(setup):
    snap, config, run_dir = setup
    before = inventory(snap)
    host = FakeHost(config, run_dir)
    receipt = r.run(host, rc_id="rc-synthetic")
    assert inventory(snap) == before  # The pinned snapshot is never opened, only cloned (planning included).
    assert host.armed == ["rc", "ga"] and len(host.replays) == 12
    assert receipt["schema"] == "d5-clone-receipt/1" and receipt["verdict"]["raw"] == bars.PASS, receipt["bars"]
    assert receipt["bars"]["bar1"]["copies_by_bind"] and receipt["bars"]["bar1"]["lost"] == 0
    assert receipt["preflights"]["P2"]["verdict"] == "PASS" and receipt["plan"]["sha256"]
    assert receipt["arms"]["ga"]["checks"]["by_kind_status"]["schema:ok"] == 1
    assert receipt["arms"]["rc"]["workload"]["turns_completed"] == 36
    assert not list((run_dir / "rc" / "captures").iterdir())  # Each bind's captures go after its replay.
    text = json.dumps(receipt)
    assert MARKER not in text and "synthetic log" not in text


def test_a_lost_rc_row_fails_the_run(setup):
    _, config, run_dir = setup
    receipt = r.run(FakeHost(config, run_dir, fail="lose"), rc_id="rc-synthetic")
    assert receipt["verdict"]["raw"] == bars.FAIL and receipt["bars"]["bar1"]["lost"] == 2


@pytest.mark.parametrize("fail", ["free_space", "P1", "P3", "P6", "P7"])
def test_any_preflight_failure_stops_before_either_arm(setup, fail):
    _, config, run_dir = setup
    host = FakeHost(config, run_dir, fail=fail)
    receipt = r.run(host, rc_id="rc-synthetic")
    assert receipt["stopped_at"] == fail and "verdict" not in receipt
    assert host.armed == [] and not (run_dir / "rc").exists() and not (run_dir / "ga").exists()


def test_rerun_rules(setup):
    _, config, run_dir = setup
    prior = {"rc": "rc-synthetic", "verdict": {"raw": bars.FAIL}}
    assert r.run(FakeHost(config, run_dir), rc_id="rc-synthetic", prior=[prior])["stopped_at"].startswith("a FAIL")
    twice = [{"rc": "rc-synthetic", "verdict": {"raw": bars.INCONCLUSIVE}, "pins": {"rules_sha256": "e" * 64}}] * 2
    assert "defect" in r.run(FakeHost(config, run_dir), rc_id="rc-synthetic", prior=twice)["stopped_at"]
    fixed = r.run(FakeHost(config, run_dir), rc_id="rc-synthetic", prior=twice, instrument_fix="fix-1")
    assert "rule code changed" in fixed["stopped_at"]
    once = twice[:1]  # An ordinary second attempt, with no instrument fix, may not change the rules either.
    assert "rule code changed" in r.run(FakeHost(config, run_dir), rc_id="rc-synthetic", prior=once)["stopped_at"]
    same = [{**once[0], "pins": {"rules_sha256": "f" * 64}}]
    assert r.run(FakeHost(config, run_dir), rc_id="rc-synthetic", prior=same).get("stopped_at") is None


def test_the_rule_fingerprint_freezes_every_pinned_file(tmp_path, monkeypatch):
    manifest = {"files": [{"name": "rules.py", "sha256": "a" * 64, "loads": [{"name": "helper.py", "sha256": "b" * 64}]}],
                "functions": [{"file": "rules.py", "qualname": "identity", "sha256": "c" * 64}], "extra_hashes": []}
    path = tmp_path / "manifest.json"
    monkeypatch.setattr(r.pins, "require", lambda *a, **k: None)
    monkeypatch.setattr(r.pins, "verify", lambda *a, **k: {"verdict": "PASS", "mismatches": []})
    host = r.LocalHost({"manifest": str(path), "instrument_dir": str(tmp_path)}, tmp_path / "run")
    path.write_text(json.dumps(manifest))
    first = host.pins()["rules_sha256"]
    path.write_text(json.dumps({**manifest, "functions": []}))
    assert host.pins()["rules_sha256"] == first  # The closure layer is for parity; the freeze is the files.
    changed = {**manifest, "files": [{**manifest["files"][0], "loads": [{"name": "helper.py", "sha256": "d" * 64}]}]}
    path.write_text(json.dumps(changed))
    assert host.pins()["rules_sha256"] != first  # Any pinned file, a plumbing load included, is frozen.
    path.write_text(json.dumps({**manifest, "rule_files": [{"name": "script.py", "sha256": "e" * 64}]}))
    assert host.pins()["rules_sha256"] != first


@pytest.mark.parametrize("result, done", [({"completed": True, "outcome": "success"}, True),
                                          ({"completed": True, "outcome": "failure"}, False),
                                          ({"completed": True}, False), ({"completed": False, "outcome": "success"}, False)])
def test_a_turn_counts_only_when_it_completed_successfully(tmp_path, result, done):
    import threading
    from types import SimpleNamespace
    from collections import Counter
    ctx = {"provider": SimpleNamespace(lock=threading.Lock(), step=None, stats={"calls_per_step": Counter({"e0-L-0": 1})}),
           "chats": {"L": "chat-l"},
           "gateway": SimpleNamespace(turn=lambda chat, step, text: result)}
    assert r.LocalHost({}, tmp_path).turn(ctx, {"id": "e0-L-0", "conversation": "L", "user_text": "x"}) is done


def test_replay_paths_are_absolute_whatever_the_cli_was_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    host = r.LocalHost({"interpreter": "venv/bin/python", "instrument_dir": "inst", "rc": {"lcm_tree": "rc"},
                        "ga": {"lcm_tree": "ga"}, "constants_argv": ["{python}", "{instrument_dir}", "{rc_tree}",
                        "{ga_tree}", "{out}"], "replay_argv": ["true"]}, "run")
    seen = []
    monkeypatch.setattr(r.sandbox, "wrap", lambda argv, **k: seen.append(argv) or ["/usr/bin/false"])
    host.replay({"name": "rc", "port": 1, "sessions": {"L": "s"}}, {"id": "e0-L-0", "conversation": "L"}, "t0", "h8")
    assert seen and all(os.path.isabs(a) for a in seen[0]), seen


def test_the_rules_module_is_configured_before_planning_and_before_bar1(setup):
    _, config, run_dir = setup
    host = FakeHost(config, run_dir)
    receipt = r.run(host, rc_id="rc-synthetic")
    names = [c[0] for c in host.calls]
    assert names[0] == "configure_planning" and names.count("covered") == 3 and names[-1] == "configure_bar1"
    planning, bar1 = host.calls[0][1], host.calls[-1][1]
    assert planning["plan_home"] == run_dir / "plan" / "home" and planning["scratch"] == run_dir / "planning"
    assert bar1["snapshot_store"] == run_dir / "plan" / "home" / "store.db"  # A clone, never the snapshot.
    assert len(bar1["rc_rows"]) == receipt["bars"]["bar1"]["rows"]
    assert len(bar1["host_rows"]) == receipt["bars"]["bar1"]["host_occurrences"]
    assert len(bar1["bind_receipts"]) == 12 and bar1["window_start"] == max(b["t0_at"] for b in receipt["binds"])


def test_a_bind_without_its_replay_receipt_makes_bar1_inconclusive(setup):
    _, config, run_dir = setup
    receipt = r.run(FakeHost(config, run_dir, fail="no_receipt"), rc_id="rc-synthetic")
    bar1 = receipt["bars"]["bar1"]
    assert (bar1["verdict"], bar1["binds_missing_receipt"]) == (bars.INCONCLUSIVE, ["e1-S1-0"])
    assert receipt["verdict"]["raw"] == bars.INCONCLUSIVE


@pytest.mark.parametrize("fail, missing", [("warm", 20), ("raise", 1)])
def test_a_turn_not_completed_is_never_swallowed(setup, fail, missing):
    _, config, run_dir = setup
    receipt = r.run(FakeHost(config, run_dir, fail=fail), rc_id="rc-synthetic")
    assert receipt["verdict"]["raw"] != bars.PASS and receipt["bars"]["bar6"]["verdict"] == bars.INCONCLUSIVE
    work = receipt["arms"]["rc"]["workload"]
    assert (work["turns_planned"], len(work["turns_not_completed"])) == (36, missing)
    assert receipt["bars"]["bar6"]["items"]["ga"]["turns_completed"] is False


def test_p1_failure_with_an_instrument_fix_stops_cleanly(setup):
    _, config, run_dir = setup
    twice = [{"rc": "rc-synthetic", "verdict": {"raw": bars.INCONCLUSIVE}, "pins": {"rules_sha256": "f" * 64}}] * 2
    receipt = r.run(FakeHost(config, run_dir, fail="P1"), rc_id="rc-synthetic", prior=twice, instrument_fix="fix-1")
    assert receipt["stopped_at"] == "P1" and "verdict" not in receipt


@pytest.mark.parametrize("name", ["/abs/rules.py", "../rules.py", "sub/../../rules.py"])
def test_rules_module_must_stay_inside_the_instrument_dir(tmp_path, name):
    with pytest.raises(ValueError):
        r.LocalHost({"instrument_dir": str(tmp_path), "rules_module": name}, tmp_path / "run").rules()


def test_replay_runs_the_edition_under_the_sandbox(tmp_path, monkeypatch):
    seen, real = [], r.subprocess.run

    def fake_run(argv, **kwargs):
        if argv[0] != "/usr/bin/nice":
            return real(argv, **kwargs)  # The sandbox's own probes run for real.
        seen.append(argv)
        out = Path(kwargs["cwd"])
        (out / "constants.json").write_text(json.dumps({"range_new_rows": 0}))
        (out / "receipt.json").write_text(json.dumps({"checks": {}}))
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(r.subprocess, "run", fake_run)
    denied = []
    monkeypatch.setattr(r.sandbox, "profile", lambda ports, socket_roots=(), deny_paths=(): denied.append(list(deny_paths))
                        or "".join(  # tmp_path may sit under /tmp.
        f'(allow network-outbound (remote ip "localhost:{port}"))' for port in ports))
    config = {"interpreter": "python3", "instrument_dir": "kit", "rc": {"lcm_tree": "rc"}, "ga": {"lcm_tree": "ga"},
              "constants_argv": ["{python}", "gen", "{t0}", "{h8}", "{out}"], "replay_argv": ["{python}", "replay"]}
    host = r.LocalHost(config, tmp_path / "run")
    receipt, error, range_new = host.replay({"name": "rc", "port": 4242, "sessions": {"L": "s"}},
                                            {"id": "e0-L-0", "conversation": "L"}, "T0", "H8")
    assert (receipt, error, range_new) == ({"checks": {}}, False, 0)
    assert denied and all(d == [r.sandbox.hermes_home()] for d in denied)  # The live agent home is denied to both steps.
    assert [argv[:4] for argv in seen] == [["/usr/bin/nice", "-n", "19", "/usr/bin/sandbox-exec"]] * 2
    assert seen[0][-5:] == ["python3", "gen", "T0", "H8", os.fspath(tmp_path / "run" / "rc" / "replays" / "e0-L-0")]
    assert 'localhost:4242' in seen[0][5]


@pytest.mark.parametrize("fail, verdict", [("ga_once", bars.PASS), ("ga_twice", bars.INCONCLUSIVE)])
def test_a_failed_ga_check_reruns_the_ga_arm_once(setup, fail, verdict):
    _, config, run_dir = setup
    host = FakeHost(config, run_dir, fail=fail)
    receipt = r.run(host, rc_id="rc-synthetic")
    assert host.armed == ["rc", "ga", "ga"] and receipt["ga_rerun"] is not None
    bar5 = receipt["bars"]["bar5"]
    assert bar5["verdict"] == verdict and bar5["ga_failed"] == ["schema"]
    assert bar5["ga_disposition_needed"] is (fail == "ga_twice") and bar5["ga_rerun_needed"] is False


def test_a_ga_rerun_that_completes_no_turn_is_not_clean(setup):
    _, config, run_dir = setup
    receipt = r.run(FakeHost(config, run_dir, fail="ga_hollow"), rc_id="rc-synthetic")
    bar5 = receipt["bars"]["bar5"]
    assert bar5["ga_failed"] == ["schema"] and bar5["ga_rerun_failed"] == [] and bar5["ga_rerun_turns_complete"] is False
    assert bar5["verdict"] == bars.INCONCLUSIVE and receipt["verdict"]["raw"] != bars.PASS


def test_a_store_too_damaged_to_check_is_a_failed_check(tmp_path, monkeypatch):
    monkeypatch.setattr(r.LocalHost, "probe", lambda self, ctx: None)  # The doctor probe has its own tests.
    home = tmp_path / "home"
    home.mkdir()
    for name in ("lcm.db", "state.db"):
        with closing(sqlite3.connect(home / name)) as db:
            db.execute("CREATE TABLE t (x)")
            db.commit()
    with open(home / "lcm.db", "r+b") as f:  # A damaged header: SQLite refuses to read the file.
        f.write(b"not a sqlite database at all")
    checks = r.LocalHost({}, tmp_path / "run").checks({"home": home, "session_ids": []})
    assert {(c["kind"], c["check"]): c["status"] for c in checks} == {
        ("quick_check", "lcm.db"): "fail", ("integrity", "lcm.db"): "fail",
        ("quick_check", "state.db"): "ok", ("integrity", "state.db"): "ok"}
    with closing(sqlite3.connect(home / "state.db")) as db:  # A -wal with no -shm: refused read-only, a failed check.
        db.execute("PRAGMA journal_mode=WAL")
    (home / "state.db-shm").unlink(missing_ok=True)
    (home / "state.db-wal").write_bytes(b"")
    assert {c["check"] for c in r.LocalHost({}, tmp_path / "run").checks({"home": home, "session_ids": []})
            if c["status"] == "fail"} == {
        "lcm.db", "state.db"}


@pytest.mark.parametrize("codes, receipt_written, expected", [
    ((0, 0), True, ({"checks": {}}, False, 0)),     # the edition's receipt, read by its declared name
    ((1,), False, (None, True, 3)),                 # constants exit 1: NOT SHOWN, its report's range_new_rows kept
    ((2,), False, (None, True, None)),              # refused: no constants, no report
    ((0, 1), False, (None, True, 0)),               # a replay that fails is an error, range_new_rows still kept
    ((0, 0), False, (None, True, 0))])              # no receipt under the declared name: an error
def test_replay_feeds_the_edition_and_reads_its_outcome(tmp_path, monkeypatch, codes, receipt_written, expected):
    seen, results = [], list(codes)

    def fake_run(argv, **kwargs):
        seen.append(argv)
        out, code = Path(kwargs["cwd"]), results.pop(0)
        if len(seen) == 1 and code in (0, 1):
            (out / "constants.json").write_text(json.dumps({"range_new_rows": 3 if code else 0}))
        if len(seen) == 2 and receipt_written:
            (out / "p-s6r-d5.json").write_text(json.dumps({"checks": {}}))
        return type("Done", (), {"returncode": code})()

    monkeypatch.setattr(r.subprocess, "run", fake_run)
    monkeypatch.setattr(r.sandbox, "wrap", lambda argv, **k: argv)
    config = {"interpreter": "python3", "instrument_dir": "kit", "rc": {"lcm_tree": "rc"}, "ga": {"lcm_tree": "ga"},
              "replay_receipt": "p-s6r-d5.json", "constants_argv": ["gen", "--session-hash", "{session_h12}"],
              "replay_argv": ["replay"]}
    ctx = {"name": "rc", "port": 1, "sessions": {"L": "host-session-1"}}
    assert r.LocalHost(config, tmp_path / "run").replay(ctx, {"id": "e0-L-0", "conversation": "L"}, "T0", "H8") == expected
    assert seen[0][-1] == hashlib.sha256(b"host-session-1").hexdigest()[:12]
    assert len(seen) == len(codes)  # NOT SHOWN or refused: the replay never runs.


def test_replay_receipt_must_be_a_bare_name(tmp_path):
    host = r.LocalHost({"interpreter": "python3", "instrument_dir": "kit", "rc": {"lcm_tree": "rc"},
                        "ga": {"lcm_tree": "ga"}, "replay_receipt": "../receipt.json",
                        "constants_argv": ["gen"], "replay_argv": ["replay"]}, tmp_path / "run")
    with pytest.raises(ValueError):
        host.replay({"name": "rc", "port": 1, "sessions": {"L": "s"}}, {"id": "e0-L-0", "conversation": "L"}, "T0", "H8")


def test_an_h8_capture_carries_the_window_host_logs(tmp_path, monkeypatch):
    home = tmp_path / "arm" / "home"
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "agent.log").write_text("synthetic agent line\n")
    seen = {}
    monkeypatch.setattr(r.capture, "capture", lambda kind, out, **k: seen.setdefault(kind, k["extras"]) and {})
    host = r.LocalHost({}, tmp_path / "run")
    for kind in ("T0", "H8"):
        host.capture({"name": "rc", "home": home}, kind, {"id": f"e0-L-{kind}"}, {"RC3/store.db": "x"})
    assert seen["T0"] == {"RC3/store.db": "x"}
    assert seen["H8"] == {"RC3/store.db": "x", "native-agent.log.log": home / "logs" / "agent.log"}  # gateway.log absent


@pytest.mark.parametrize("range_new", [0, 3, None])
def test_a_not_shown_bind_is_inconclusive_never_pass(tmp_path, monkeypatch, range_new):
    def fake_run(argv, **kwargs):  # The generator exits 1 (NOT SHOWN) and leaves its report; no replay runs.
        if range_new is not None:
            (Path(kwargs["cwd"]) / "constants.json").write_text(json.dumps({"range_new_rows": range_new}))
        return type("Done", (), {"returncode": 1})()

    monkeypatch.setattr(r.subprocess, "run", fake_run)
    monkeypatch.setattr(r.sandbox, "wrap", lambda argv, **k: argv)
    config = {"interpreter": "python3", "instrument_dir": "kit", "rc": {"lcm_tree": "rc"}, "ga": {"lcm_tree": "ga"},
              "constants_argv": ["gen"], "replay_argv": ["replay"]}
    ctx = {"name": "rc", "port": 1, "sessions": {"L": "s"}}
    receipt, error, kept = r.LocalHost(config, tmp_path / "run").replay(ctx, {"id": "e0-L-0", "conversation": "L"},
                                                                      "T0", "H8")
    result = bars.bar2_bind(receipt, range_new_rows=kept, error=error)
    assert (result["verdict"], result["range_new_rows"]) == (bars.INCONCLUSIVE, range_new)


class UnreadableFinal(FakeHost):
    """The arm's store goes bad after its checks: its final rows read raises, and its checks report the failure."""

    def __init__(self, config, run_dir, arm, fail=None):
        super().__init__(config, run_dir, fail=fail)
        self.bad = arm

    def rows(self, path):
        if Path(path).parent.name == self.bad:
            raise sqlite3.DatabaseError("synthetic: malformed")
        return super().rows(path)

    def checks(self, ctx):
        return [{**c, "status": "fail" if ctx["name"] == self.bad and c["kind"] == "integrity" else c["status"]}
                for c in super().checks(ctx)]


@pytest.mark.parametrize("arm", ["rc", "ga"])
def test_an_unreadable_final_store_keeps_its_failed_checks_and_never_passes(setup, arm):
    _, config, run_dir = setup
    host = UnreadableFinal(config, run_dir, arm)
    receipt = r.run(host, rc_id="rc-synthetic")  # A receipt, not a crash.
    if arm == "rc":
        for bar in ("bar1", "bar3"):
            assert receipt["bars"][bar] == {"verdict": bars.INCONCLUSIVE, "rows_unreadable": ["rc"]}
        assert receipt["bars"]["bar5"]["verdict"] == bars.FAIL and receipt["verdict"]["raw"] == bars.FAIL
    else:
        assert (receipt["bars"]["bar1"]["verdict"], receipt["bars"]["bar1"]["rows_unreadable"]) == (
            bars.INCONCLUSIVE, ["ga"])
        assert receipt["bars"]["bar3"]["verdict"] == bars.PASS  # Bar 3 never reads GA's rows.
        assert host.armed == ["rc", "ga", "ga"] and receipt["ga_rerun"] is not None  # The GA failure re-runs it.
        assert receipt["verdict"]["raw"] != bars.PASS


class RefusedFacts(UnreadableFinal):
    """The stopped arm's store is one the read-only open refuses (a -wal with no -shm): its facts read raises."""

    def rows(self, path):
        return FakeHost.rows(self, path)

    def facts(self, ctx):
        if ctx["name"] == self.bad:
            raise r.capture.CaptureError("synthetic: -wal without -shm")
        return super().facts(ctx)


@pytest.mark.parametrize("arm", ["rc", "ga"])
def test_a_refused_workload_read_keeps_the_receipt_and_its_failed_checks(setup, arm):  # #2453 item 1
    _, config, run_dir = setup
    receipt = r.run(RefusedFacts(config, run_dir, arm), rc_id="rc-synthetic")  # A receipt, not a crash.
    if arm == "rc":
        assert receipt["bars"]["bar5"]["verdict"] == bars.FAIL and receipt["verdict"]["raw"] == bars.FAIL
    else:  # GA's failed checks are kept and request the GA re-run; they never pass bar 5.
        assert receipt["ga_rerun"] is not None and receipt["bars"]["bar5"]["verdict"] == bars.INCONCLUSIVE
    bar6 = receipt["bars"]["bar6"]
    assert bar6["verdict"] == bars.INCONCLUSIVE and not bar6["items"][arm]["cron_fired"]
    assert all(bar6["items"]["ga" if arm == "rc" else "rc"].values())  # The other arm's facts are kept.


class UnreadableGaRowsOnly(UnreadableFinal):
    def checks(self, ctx):  # GA's checks stay ok: no rerun, and only its rows are missing.
        return FakeHost.checks(self, ctx)


def test_unreadable_ga_rows_never_hide_an_rc_failure(setup):
    _, config, run_dir = setup
    receipt = r.run(UnreadableGaRowsOnly(config, run_dir, "ga", fail="lose"), rc_id="rc-synthetic")
    assert receipt["bars"]["bar1"]["verdict"] == bars.FAIL and receipt["bars"]["bar1"]["lost"] > 0
    assert receipt["verdict"]["raw"] == bars.FAIL


def test_the_rules_module_can_import_its_pinned_siblings(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "sibling_helper.py").write_text("def norm(x):\n    return x\n")
    (kit / "rules.py").write_text(
        "from sibling_helper import norm\n"
        "def identity(row):\n    return norm(row)\n"
        "credits = ()\n"
        "def covered(session_id):\n    return True\n"
        "def configure_planning(**kwargs):\n    pass\n"
        "def configure_bar1(**kwargs):\n    pass\n")
    before = list(sys.path)
    rules = r.LocalHost({"instrument_dir": str(kit), "rules_module": "rules.py"}, tmp_path / "run").rules()
    assert rules["identity"]("x") == "x"
    assert sys.path == before  # The kit dir is on the path only while the module loads.
    sys.modules.pop("sibling_helper", None)


@pytest.mark.parametrize("calls, outcome, done", [(0, "success", False), (1, "success", True), (2, "failure", False)])
def test_a_turn_the_agent_never_ran_is_not_completed(tmp_path, calls, outcome, done):
    """The host reports a turn it only queued (startup restore) as completed with success; the scripted provider's
    agent calls for the step are the ground truth, so a step with no agent call never counts as a completed turn."""
    import threading
    from collections import Counter

    class Provider:
        lock, step = threading.Lock(), None
        stats = {"calls_per_step": Counter()}

    provider = Provider()

    class Gateway:
        def turn(self, chat, step_id, text):
            provider.stats["calls_per_step"][step_id] += calls
            return {"step": step_id, "completed": True, "outcome": outcome}

    ctx = {"provider": provider, "gateway": Gateway(), "chats": {"L": "d5-L"}}
    assert r.LocalHost({}, tmp_path).turn(ctx, {"id": "e1-L-0", "conversation": "L", "user_text": "x"}) is done
    assert provider.step is None


@pytest.fixture
def rekey_copy(tmp_path):
    home = tmp_path / "arm"
    home.mkdir()
    path = home / "lcm.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, conversation_id TEXT, content TEXT);"
                         "CREATE TABLE lcm_lifecycle_state (conversation_id TEXT PRIMARY KEY, last_finalized_session_id TEXT);")
        db.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)",
                       [(1, "s1", "old", "first"), (2, "s2", "old", "second"), (3, "other", "unrelated", "third")])
        db.executemany("INSERT INTO lcm_lifecycle_state VALUES (?, ?)", [("old", "s2"), ("unrelated", "other")])
    return path, [{"name": "L", "session_id": "s1", "chat_id": "d5-L"}]


def test_rekey_copy_moves_the_whole_conversation_and_receipts_counts(rekey_copy, tmp_path):
    path, conversations = rekey_copy
    key = r.gateway.session_key("d5-L")
    receipt = r.rekey_copy_conversations(path, conversations, run_dir=tmp_path)
    assert receipt == [{"name": "L", "old_h12": r._sha("old")[:12], "new_h12": r._sha(key)[:12],
                        "tables": {"messages": {"before": 2, "after": 2, "old_remaining": 0},
                                   "lcm_lifecycle_state": {"before": 1, "after": 1, "old_remaining": 0}}}]
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == [
            (1, "s1", key, "first"), (2, "s2", key, "second"), (3, "other", "unrelated", "third")]
        assert db.execute("SELECT * FROM lcm_lifecycle_state ORDER BY conversation_id").fetchall() == [
            (key, "s2"), ("unrelated", "other")]


@pytest.mark.parametrize("fault", ["messages_destination", "lifecycle_destination", "ambiguous", "outside", "third_table"])
def test_rekey_copy_refuses_unsafe_inputs_without_updates(rekey_copy, tmp_path, fault):
    path, conversations = rekey_copy
    with closing(sqlite3.connect(path)) as db, db:
        if fault == "messages_destination":
            db.execute("INSERT INTO messages VALUES (4, 'other', ?, 'collision')", (r.gateway.session_key("d5-L"),))
        elif fault == "lifecycle_destination":
            db.execute("INSERT INTO lcm_lifecycle_state VALUES (?, 'other')", (r.gateway.session_key("d5-L"),))
        elif fault == "ambiguous":
            db.execute("INSERT INTO messages VALUES (4, 's1', 'second-key', 'ambiguous')")
        elif fault == "third_table":
            db.execute("CREATE TABLE unexpected (conversation_id TEXT)")
    before = inventory(path.parent)
    with pytest.raises(ValueError, match="unexpected" if fault == "third_table" else None):
        r.rekey_copy_conversations(path, conversations, run_dir=tmp_path / "elsewhere" if fault == "outside" else tmp_path)
    assert inventory(path.parent) == before


def test_rekey_copy_is_a_noop_when_keys_match(rekey_copy, tmp_path):
    path, conversations = rekey_copy
    with closing(sqlite3.connect(path)) as db, db:
        for table in ("messages", "lcm_lifecycle_state"):
            db.execute(f"UPDATE {table} SET conversation_id=? WHERE conversation_id='old'", (r.gateway.session_key("d5-L"),))
    before = inventory(path.parent)
    assert r.rekey_copy_conversations(path, conversations, run_dir=tmp_path) == []
    assert inventory(path.parent) == before


@pytest.mark.parametrize("appears", [True, False])
def test_cold_bind_waits_for_durable_ingest_before_h8(setup, monkeypatch, appears):
    _, config, run_dir = setup
    config["cold_ingest_timeout"] = 1
    clock = [0.0]
    monkeypatch.setattr(r.time, "monotonic", lambda: clock[0])

    class DelayedHost(FakeHost):
        def turn(self, ctx, step):
            self.pending = (ctx, step)
            # Wrong role, text, session, and pre-T0 rows cannot satisfy the wait.
            with closing(sqlite3.connect(ctx["home"] / "lcm.db")) as db, db:
                db.executemany("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                               [("sess-l", "assistant", step["user_text"]), ("sess-l", "user", "other"),
                                ("sess-a", "user", step["user_text"])])
            return True

        def capture(self, ctx, kind, step, extras=None):
            self.calls.append(kind)
            if kind == "T0":
                with closing(sqlite3.connect(ctx["home"] / "lcm.db")) as db, db:
                    db.execute("INSERT INTO messages (session_id, role, content) VALUES ('sess-l', 'user', ?)",
                               (step["user_text"],))
            if kind == "H8":
                assert clock[0] >= 0.5
            return super().capture(ctx, kind, step, extras)

    host = DelayedHost(config, run_dir)

    def tick(seconds):
        assert seconds == 0.5
        clock[0] += seconds
        if appears:
            FakeHost.turn(host, *host.pending)

    monkeypatch.setattr(r.time, "sleep", tick)
    plan = r.planner.build_plan([{"name": "L", "session_id": "sess-l", "chat_id": "d5-L", "lcm_rows": 1000}],
                                epochs=1, warm_turns=0, nonce="synthetic")
    result = r.run_arm(host, "rc", host.clone("rc"), plan, {})
    bind = result["binds"][0]
    assert bind["cold_ingest_wait_s"] == (0.5 if appears else 1.0)
    assert host.calls.count("H8") == int(appears)
    assert host.replays == ([plan["epochs"][0]["steps"][0]["id"]] if appears else [])
    if appears:
        assert bind["bar2"]["verdict"] == bars.PASS
    else:
        assert bind["bar2"]["verdict"] == bars.INCONCLUSIVE
        assert bind["bar2"]["error"] is True
        assert bind["bar2"]["reason"] == "cold-turn ingest not durable within 1 s"
