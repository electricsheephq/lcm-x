"""The whole run on a synthetic snapshot with a fake host and a fake edition; no gateway, no network."""

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

from bench.instruments.d5_clone import bars, capture, plan as planner, run as r

MARKER = "synthetic-marker-7f3a9c"


def make_snapshot(root, flagged=()):
    """*flagged*: sessions given one summary node the doctor's summary_quality check flags (#2471)."""
    root.mkdir()
    (root / "lcm-large-outputs").mkdir()
    (root / "lcm-large-outputs" / "out-1.json").write_text(MARKER)
    sessions = {"sess-l": 1000, "sess-a": 25, "sess-b": 22}
    with closing(sqlite3.connect(root / "store.db")) as lcm, closing(sqlite3.connect(root / "host-0.db")) as host:
        lcm.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,"
                    " observed_at REAL)")
        host.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, end_reason TEXT)")
        host.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)")
        lcm.execute("CREATE TABLE summary_nodes (node_id INTEGER PRIMARY KEY, session_id TEXT, token_count INTEGER,"
                    " source_token_count INTEGER)")
        lcm.executemany("INSERT INTO summary_nodes (session_id, token_count, source_token_count) VALUES (?, ?, ?)",
                        [(sid, 100, 4_000) for sid in sessions] + [(sid, 10, 4_000) for sid in flagged])
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


@pytest.mark.parametrize("fail", ["P3", "P4", "P6", "P7"])
def test_without_the_informational_flag_every_recorded_preflight_still_gates_and_the_receipt_is_unchanged(setup, fail):
    """(a) The default path: each recorded preflight still stops the run before either arm, and the D5 receipt has the
    same shape and verdict whether the flag is left out or passed as False."""
    _, config, run_dir = setup
    stops = [r.run(FakeHost(config, run_dir / name, fail=fail), rc_id="rc-synthetic", **kw)
             for name, kw in (("a", {}), ("b", {"informational": False}))]
    for receipt in stops:
        assert receipt["stopped_at"] == fail and receipt["schema"] == r.SCHEMA
        assert "verdict" not in receipt and "instrument" not in receipt
    assert [{k: v for k, v in x.items() if k != "wall_s"} for x in stops][0] == \
        [{k: v for k, v in x.items() if k != "wall_s"} for x in stops][1]
    hosts = [FakeHost(config, run_dir / name) for name in ("c", "d")]
    full = [r.run(hosts[0], rc_id="rc-synthetic"), r.run(hosts[1], rc_id="rc-synthetic", informational=False)]
    assert sorted(full[0]) == sorted(full[1]) and sorted(full[0]["bars"]) == [f"bar{i}" for i in range(1, 7)]
    assert full[0]["verdict"] == full[1]["verdict"] and full[0]["verdict"]["raw"] == bars.PASS
    assert [len(h.replays) for h in hosts] == [12, 12] and full[0]["schema"] == r.SCHEMA


class RecordingHost(FakeHost):
    def preflight(self, name, plan):
        self.calls.append(("preflight", name))
        return super().preflight(name, plan)

    def capture(self, *args, **kwargs):
        self.calls.append(("capture",))
        return super().capture(*args, **kwargs)


@pytest.mark.parametrize("fail", ["P4", "ga_once"])
def test_the_informational_check_reads_no_recorded_preflight_makes_no_bind_and_is_never_d5(setup, fail):
    """(b) With the flag: no recorded preflight is read or gated (P5 still runs live), no T0/H8 capture and no replay,
    a failed GA check is recorded and not re-run, and the receipt has its own schema, bars 3 to 6 and no verdict."""
    snap, config, run_dir = setup
    before = inventory(snap)
    host = RecordingHost(config, run_dir, fail=fail)
    receipt = r.run(host, rc_id="rc-synthetic", informational=True)
    assert inventory(snap) == before
    assert receipt["schema"] == "informational, not D5" == r.INFORMATIONAL_SCHEMA
    assert "verdict" not in receipt and "stopped_at" not in receipt and sorted(receipt["bars"]) == ["bar3", "bar4", "bar5", "bar6"]
    assert {n: receipt["preflights"][n]["verdict"] for n in ("P3", "P4", "P6", "P7")} == dict.fromkeys(
        ("P3", "P4", "P6", "P7"), r.NOT_RUN) and receipt["preflights"]["P5"]["verdict"] == "PASS"
    assert [c for c in host.calls if c[0] == "preflight"] == [("preflight", "P5")]
    assert not any(c[0] in ("capture", "configure_bar1") for c in host.calls)
    assert host.replays == [] and receipt["binds"] == [] and not list((run_dir / "rc").glob("captures/*"))
    assert host.armed == ["rc", "ga"] and receipt["ga_rerun"] is None  # ga_once: recorded, never re-run.
    assert receipt["arms"]["ga"]["checks"]["by_kind_status"].get("schema:fail", 0) == (fail == "ga_once")
    assert receipt["arms"]["rc"]["workload"]["turns_completed"] == 36
    assert receipt["instrument"]["instrument_sha256"] == r.instrument_identity()["instrument_sha256"]
    assert MARKER not in json.dumps(receipt)


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
    assert receipt["bars"]["bar5"]["ga_turns_complete"] is False  # The runner passes the GA arm's completion to bar 5.


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
    assert denied and all(d == r.sandbox.live_homes() for d in denied)  # The live agent home is denied to both steps.
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
            assert {k: v for k, v in receipt["bars"][bar].items() if k != "known_risk"} == {
                "verdict": bars.INCONCLUSIVE, "rows_unreadable": ["rc"]}
        assert "known_risk" in receipt["bars"]["bar1"]  # Kept in every receipt.
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


@pytest.mark.parametrize("fault", ["messages_destination", "lifecycle_destination", "ambiguous", "outside", "third_table",
                                   "blank_key", "null_key", "space_key", "old_variant", "destination_variant"])
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
        elif fault == "blank_key":  # Legacy blank rows belong to every conversation, one of them outside the plan.
            db.execute("UPDATE messages SET conversation_id='' WHERE session_id='s1'")
            db.execute("INSERT INTO messages VALUES (4, 'other2', '', 'legacy')")
        elif fault == "null_key":
            db.execute("UPDATE messages SET conversation_id=NULL WHERE session_id='s1'")
        elif fault == "space_key":  # Whitespace-only reads as blank, shared here with an unrelated session.
            db.execute("UPDATE messages SET conversation_id=' \t' WHERE session_id='s1'")
            db.execute("INSERT INTO messages VALUES (4, 'other2', ' \t', 'legacy')")
        elif fault == "old_variant":  # A padded legacy copy of the key, owned by the same conversation.
            db.execute("INSERT INTO messages VALUES (4, 'other2', ' old', 'legacy')")
        elif fault == "destination_variant":
            db.execute("INSERT INTO lcm_lifecycle_state VALUES (?, 'other')", (" " + r.gateway.session_key("d5-L") + " ",))
    before = inventory(path.parent)
    with pytest.raises(ValueError, match={"third_table": "unexpected conversation_id table: sha256-12:[0-9a-f]{12}$", "blank_key": "legacy blank conversation_id: L$",
                                          "null_key": "legacy blank conversation_id: L$",
                                          "space_key": "legacy blank conversation_id: L$",
                                          "old_variant": "legacy variants: L$",
                                          "destination_variant": "destination already has rows"}.get(fault)):
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


@pytest.mark.parametrize("same_destination", [False, True])
def test_rekey_copy_resolves_shared_keys_before_any_update(rekey_copy, tmp_path, monkeypatch, same_destination):
    path, conversations = rekey_copy
    conversations.append({"name": "S1", "session_id": "s2",
                          "chat_id": "d5-L" if same_destination else "d5-S1"})
    connect, updates = sqlite3.connect, []

    def traced_connect(*args, **kwargs):
        db = connect(*args, **kwargs)
        db.set_trace_callback(lambda sql: updates.append(sql) if sql.startswith("UPDATE ") else None)
        return db

    monkeypatch.setattr(r.sqlite3, "connect", traced_connect)
    before = inventory(path.parent)
    if not same_destination:
        with pytest.raises(ValueError, match="L.*S1") as error:
            r.rekey_copy_conversations(path, conversations, run_dir=tmp_path)
        assert "old" not in str(error.value) and "agent:" not in str(error.value)
        assert updates == []
        assert inventory(path.parent) == before
        return

    receipt = r.rekey_copy_conversations(path, conversations, run_dir=tmp_path)
    key = r.gateway.session_key("d5-L")
    counts = {"messages": {"before": 2, "after": 2, "old_remaining": 0},
              "lcm_lifecycle_state": {"before": 1, "after": 1, "old_remaining": 0}}
    assert receipt == [{"name": name, "old_h12": r._sha("old")[:12], "new_h12": r._sha(key)[:12],
                        "tables": counts, "shared_k1_with": [other]} for name, other in (("L", "S1"), ("S1", "L"))]
    assert len(updates) == 2  # One rename per table, even with two plan entries.
    with closing(connect(path)) as db:
        assert db.execute("SELECT conversation_id FROM messages ORDER BY store_id").fetchall() == [
            (key,), (key,), ("unrelated",)]
        assert db.execute("SELECT * FROM lcm_lifecycle_state ORDER BY conversation_id").fetchall() == [
            (key, "s2"), ("unrelated", "other")]


@pytest.mark.parametrize("empty_lifecycle", [False, True])
def test_rekey_copy_refuses_two_conversations_with_one_destination(rekey_copy, tmp_path, monkeypatch, empty_lifecycle):
    path, conversations = rekey_copy
    conversations.append({"name": "S1", "session_id": "other", "chat_id": "d5-L"})
    if empty_lifecycle:  # With no lifecycle rows, no unique key stops the second rename.
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("DELETE FROM lcm_lifecycle_state")
    connect, updates = sqlite3.connect, []

    def traced_connect(*args, **kwargs):
        db = connect(*args, **kwargs)
        db.set_trace_callback(lambda sql: updates.append(sql) if sql.startswith("UPDATE ") else None)
        return db

    monkeypatch.setattr(r.sqlite3, "connect", traced_connect)
    before = inventory(path.parent)
    with pytest.raises(ValueError, match="L.*S1") as error:
        r.rekey_copy_conversations(path, conversations, run_dir=tmp_path)
    assert "old" not in str(error.value) and "unrelated" not in str(error.value) and "agent:" not in str(error.value)
    assert updates == []
    assert inventory(path.parent) == before


def test_local_arm_rekeys_after_routing_before_gateway_start(rekey_copy, tmp_path, monkeypatch):
    from types import SimpleNamespace

    try:
        import yaml  # noqa: F401
    except ImportError:  # CI's main test job has no PyYAML; JSON is valid YAML for the config the arm writes.
        monkeypatch.setitem(sys.modules, "yaml", SimpleNamespace(safe_dump=lambda data, **_: json.dumps(data)))
    path, conversations = rekey_copy
    home = path.parent
    path.rename(home / "store.db")
    with closing(sqlite3.connect(home / "host-0.db")) as db, db:
        db.executescript("CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL, session_key TEXT);"
                         "CREATE TABLE gateway_routing (scope TEXT, session_key TEXT, entry_json TEXT, updated_at REAL,"
                         " PRIMARY KEY (scope, session_key));"
                         "INSERT INTO sessions VALUES ('s1', 0, 'old');")
    host_src = tmp_path / "host"
    (host_src / "hermes_cli").mkdir(parents=True)
    (host_src / "hermes_cli" / "config.py").write_text("DEFAULT_CONFIG = {'auxiliary': {}}\n")
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: synthetic\n")
    server = SimpleNamespace(url="http://127.0.0.1:12345/v1",
                             server=SimpleNamespace(server_address=("127.0.0.1", 12345)))
    monkeypatch.setattr(r.prov, "Provider", lambda plan: SimpleNamespace(start=lambda: server))
    connect, events = sqlite3.connect, []
    key = r.gateway.session_key("d5-L")

    def traced_connect(*args, **kwargs):
        db = connect(*args, **kwargs)

        def trace(sql):
            if sql.startswith("UPDATE sessions "):
                events.append("routing")
            if sql.startswith("UPDATE messages "):
                # The routing transaction must already be committed when re-keying starts.
                with closing(connect(home / "state.db")) as host:
                    events.append(("rekey", host.execute("SELECT session_key FROM sessions").fetchone()[0]))

        db.set_trace_callback(trace)
        return db

    def start(gateway, timeout):
        with closing(connect(home / "lcm.db")) as db:
            assert db.execute("SELECT DISTINCT conversation_id FROM messages WHERE session_id IN ('s1', 's2')").fetchall() == [(key,)]
        events.append("start")

    monkeypatch.setattr(r.sqlite3, "connect", traced_connect)
    monkeypatch.setattr(r.gateway.Gateway, "start", start)
    host = r.LocalHost({"rc": {"host_src": host_src, "host_bin": tmp_path / "bin", "lcm_tree": plugin},
                        "settings": {"compression": {}, "lcm": {}}, "context_length": 32000, "lcm_env": {}}, tmp_path)
    ctx = host.arm("rc", home, {"conversations": conversations, "provider": {"steps": {}, "cron": []}})
    assert ctx["routing"]["sessions"] == 1
    assert ctx["routing"]["conversation_rekeys"][0]["tables"]["messages"]["after"] == 2
    assert events == ["routing", ("rekey", key)]
    host.start(ctx)
    assert events == ["routing", ("rekey", key), "start"]


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


@pytest.mark.parametrize("failure", ["false", "raises"])
def test_a_cold_bind_whose_turn_did_not_complete_skips_the_wait(setup, monkeypatch, failure):
    _, config, run_dir = setup
    config["cold_ingest_timeout"] = 1
    monkeypatch.setattr(r.time, "sleep", lambda seconds: pytest.fail("a cold turn that did not complete must not wait"))

    class FailedHost(FakeHost):
        def turn(self, ctx, step):
            if failure == "raises":
                raise TimeoutError("the turn timed out")
            return False

    host = FailedHost(config, run_dir)
    plan = r.planner.build_plan([{"name": "L", "session_id": "sess-l", "chat_id": "d5-L", "lcm_rows": 1000}],
                                epochs=1, warm_turns=0, nonce="synthetic")
    result = r.run_arm(host, "rc", host.clone("rc"), plan, {})
    bind, step_id = result["binds"][0], plan["epochs"][0]["steps"][0]["id"]
    assert "H8" not in host.calls and host.replays == []
    assert bind["cold_ingest_wait_s"] is None
    assert bind["bar2"]["verdict"] == bars.INCONCLUSIVE and bind["bar2"]["error"] is True
    assert bind["bar2"]["reason"] == "cold turn not completed"
    assert result["workload"]["turns_not_completed"] == [step_id]


def test_flagged_summary_nodes_counts_what_the_doctor_flags(tmp_path):
    """The doctor's summary_quality predicate (tools._summary_quality_stats), at its boundaries."""
    store = tmp_path / "store.db"
    with closing(sqlite3.connect(store)) as db, db:
        db.execute("CREATE TABLE summary_nodes (node_id INTEGER PRIMARY KEY, session_id TEXT, token_count INTEGER,"
                   " source_token_count INTEGER)")
        db.executemany("INSERT INTO summary_nodes (session_id, token_count, source_token_count) VALUES (?, ?, ?)", [
            ("s", 10, 4_000), ("s", 10, 3_999), ("s", 499, 100_000), ("s", 500, 100_000), ("s", 499, 99_999),
            ("s", 0, 50_000), ("s", 0, 100_000), ("t", 10, 4_000)])
    assert r.flagged_summary_nodes(store, ["s", "t", "u"]) == {"s": 3, "t": 1, "u": 0}
    assert r.flagged_summary_nodes(tmp_path / "missing.db", ["s"]) is None


@pytest.mark.parametrize("flagged", [(), ("sess-a",)])
def test_a_driven_session_that_starts_flagged_is_judged_by_its_increase(tmp_path, flagged):
    """#2471, ruling R5-4: the plan records each driven session's starting count of flagged summary nodes, each arm
    records its own count before its first turn and at its end, and a session that starts flagged is judged by the
    increase (here none in either arm: it holds)."""
    snap = tmp_path / "snapshot"
    config = {"snapshot_dir": str(snap), "snapshot_pin": str(make_snapshot(snap, flagged)), "nonce": "synthetic"}
    receipt = r.run(FakeHost(config, tmp_path / "run"), rc_id="rc-synthetic")
    keys = {sid: r._sha(sid)[:12] for sid in ("sess-l", "sess-a", "sess-b")}
    counts = receipt["bars"]["bar5"]["flagged_summary_nodes"]
    assert counts["start"] == {keys[s]: int(s in flagged) for s in keys}
    assert set(counts) == {"start", "rc", "ga"}
    assert {c["name"]: c["flagged_summary_nodes"] for c in receipt["plan"]["declaration"]["conversations"]} == {
        "L": 0, "S1": int(bool(flagged)), "S2": 0}  # S1 is sess-a, the larger of the two small sessions.
    inherited = receipt["bars"]["bar5"]["inherited_flags"]
    assert list(inherited) == ([f"summary_quality@{keys['sess-a']}"] if flagged else [])
    for f in inherited.values():
        assert (f["s"], f["r"], f["g"], f["rc_start"], f["ga_start"], f["verdict"]) == (1, 1, 1, 1, 1, bars.PASS)
    assert receipt["bars"]["bar5"]["verdict"] == bars.PASS, receipt["bars"]["bar5"]


class FlagsOnFirstStart(FakeHost):
    """Each arm's first gateway start adds one flagged summary node to sess-a in that arm's own store."""

    def start(self, ctx):
        super().start(ctx)
        if ctx["epoch"] == 0:
            with closing(sqlite3.connect(ctx["home"] / "lcm.db")) as db, db:
                db.execute("INSERT INTO summary_nodes (session_id, token_count, source_token_count)"
                           " VALUES ('sess-a', 10, 4000)")


def test_each_arm_reads_its_start_count_from_its_own_store_before_its_first_turn(tmp_path):
    """Ruling R5-4: rc_start and ga_start are read before the first gateway start, so a node flagged during the arm
    counts in its increase (here 1 in each arm: equal increases hold) and not in its start."""
    snap = tmp_path / "snapshot"
    config = {"snapshot_dir": str(snap), "snapshot_pin": str(make_snapshot(snap, ("sess-a",))), "nonce": "synthetic"}
    receipt = r.run(FlagsOnFirstStart(config, tmp_path / "run"), rc_id="rc-synthetic")
    (f,) = receipt["bars"]["bar5"]["inherited_flags"].values()
    assert (f["s"], f["rc_start"], f["ga_start"], f["r"], f["g"], f["verdict"]) == (1, 1, 1, 2, 2, bars.PASS)


def test_inherited_flags_take_the_rerun_counts_only_when_a_rerun_ran():
    plan = {"conversations": [{"session_id": "sess-a", "flagged_summary_nodes": 1},
                              {"session_id": "sess-b", "flagged_summary_nodes": 0}]}
    h = r._sha("sess-a")[:12]
    arm = {"flagged_summary_nodes_start": {h: 1}, "flagged_summary_nodes": {h: 2}}
    rerun = {"flagged_summary_nodes_start": {h: 1}, "flagged_summary_nodes": {h: 3}}
    assert r.inherited_flags(plan, {"rc": arm, "ga": arm}) == {f"summary_quality@{h}": {
        "start": 1, "rc_start": 1, "rc": 2, "ga_start": 1, "ga": 2}}
    with_rerun = r.inherited_flags(plan, {"rc": arm, "ga": arm}, rerun)[f"summary_quality@{h}"]
    assert (with_rerun["ga_rerun_start"], with_rerun["ga_rerun"]) == (1, 3)
    assert r.inherited_flags(plan, {"rc": None, "ga": None})[f"summary_quality@{h}"]["rc"] is None

class NamesFromTheSnapshot(FakeHost):
    """An arm whose copied routes named a provider of the agent's own, and whose gateway asked for a model of its own."""

    def arm(self, name, home, plan):
        ctx = super().arm(name, home, plan)
        ctx["routing"] = {"sessions": 3, "routing_rows": 3, "changes": [],
                          "stored_route_providers": ["acme-private-provider"]}
        return ctx

    def stats(self, ctx):
        return {"replay_misses": {}, "calls_by_model": {"d5-agent": 36, "acme-private-model": 2},
                "unknown_models": {"acme-private-model": 2}}


@pytest.mark.parametrize("fail", [None, "ga_once"])
def test_the_receipt_never_names_a_provider_or_model_from_the_snapshot(tmp_path, fail):
    """Receipts carry counts, ids and hashes only (release manager, 2026-11-10): a stored route's provider name and
    any model name that is not one of the provider's own d5-* names are written as their sha256 12."""
    snap = tmp_path / "snapshot"
    config = {"snapshot_dir": str(snap), "snapshot_pin": str(make_snapshot(snap)), "nonce": "synthetic"}
    receipt = r.run(NamesFromTheSnapshot(config, tmp_path / "run", fail=fail), rc_id="rc-synthetic")
    assert "acme-private" not in json.dumps(receipt, sort_keys=True, default=str)
    assert bool(receipt["ga_rerun"]) == (fail == "ga_once")  # The re-run's provider stats are hashed too.
    h = {n: hashlib.sha256(n.encode()).hexdigest()[:12] for n in ("acme-private-provider", "acme-private-model")}
    for arm in receipt["arms"].values():
        assert (arm["routing"]["stored_route_providers"], arm["routing"]["stored_route_provider_sha256_12"]) == (
            1, [h["acme-private-provider"]])
        assert arm["routing"]["sessions"] == 3
        assert arm["provider"]["calls_by_model"] == {"d5-agent": 36, f"sha256-12:{h['acme-private-model']}": 2}
        assert arm["provider"]["unknown_models"] == {f"sha256-12:{h['acme-private-model']}": 2}


def test_private_models_keeps_d5_names_and_merges_nothing_else():
    assert r._private_models({"d5-agent": 1, "d5-summary": 2}) == {"d5-agent": 1, "d5-summary": 2}
    assert r.OWN_MODELS == {"d5-agent", "d5-aux", "d5-cron", "d5-summary"}
    for other in ("d5-internal-x", None, 7):  # Only the provider's own four names stay raw; nothing crashes.
        assert list(r._private_models({other: 1})) == [f"sha256-12:{hashlib.sha256(str(other).encode()).hexdigest()[:12]}"]
    assert r._private_models(None) == {} and r._private_models({}) == {}
    assert list(r._private_models({"x": 1})) == [f"sha256-12:{hashlib.sha256(b'x').hexdigest()[:12]}"]
    assert r._private_arm({"calls_by_model": {"x": 1}}, None)[1] is None

def _bound_host(tmp_path):
    (tmp_path / "pin.json").write_text("{}")
    config = {"snapshot_pin": str(tmp_path / "pin.json"), "rc": {"commit": "c" * 40}, "ga": {"commit": "a" * 40},
              "preflights": {}}
    plan = planner.build_plan([dict(session_id=s, name=n, chat_id=f"d5-{n}", lcm_rows=1, host_rows=1)
                               for n, s in (("L", "sess-l"), ("S1", "sess-a"), ("S2", "sess-b"))], nonce="t")
    return r.LocalHost(config, tmp_path / "run"), plan


def _receipt(host, plan, preflight, **changes):
    ga = host.c["ga"]["commit"]
    receipt = {"schema": r.PREFLIGHT_SCHEMA, "name": preflight, "verdict": "PASS", "plan_sha256": planner.plan_sha256(plan),
               "snapshot_pin_sha256": capture._sha256(host.c["snapshot_pin"]),
               "instrument_sha256": r.instrument_identity()["instrument_sha256"],
               "trees": {"rc": host.c["rc"]["commit"] if preflight in r.TREE_BOUND_PREFLIGHTS else ga, "ga": ga},
               "config_sha256": r.config_identity(host.c), **changes}
    path = host.run_dir.parent / f"{preflight}-{len(list(host.run_dir.parent.glob('P*.json')))}.json"
    path.write_text(json.dumps(receipt))
    host.c["preflights"][preflight] = str(path)
    return host.preflight(preflight, plan)


@pytest.mark.parametrize("name", ["P3", "P4", "P6", "P7"])
def test_a_preflight_receipt_counts_only_for_this_run(tmp_path, name):
    """Ruling R5-5: a PASS receipt counts only when its plan, snapshot pin and instrument are this run's; P4 also binds
    the run's two trees, and the others bind the previous GA in both slots. Anything else is treated as missing."""
    host, plan = _bound_host(tmp_path)
    assert _receipt(host, plan, name)["verdict"] == "PASS"
    other_trees = {"rc": "a" * 40, "ga": "a" * 40} if name in r.TREE_BOUND_PREFLIGHTS else {"rc": "c" * 40, "ga": "a" * 40}
    for field, value in (("plan_sha256", "0" * 64), ("snapshot_pin_sha256", "0" * 64), ("instrument_sha256", "0" * 64),
                         ("schema", "other/1"), ("name", "P0"), ("trees", other_trees), ("config_sha256", "0" * 64)):
        result = _receipt(host, plan, name, **{field: value})
        assert (result["verdict"], result["mismatch"]) == ("MISSING", [field]), field
    for field in ("plan_sha256", "instrument_sha256", "trees", "config_sha256"):
        result = _receipt(host, plan, name, **{field: None})
        assert result["verdict"] == "MISSING" and field in result["mismatch"]


@pytest.mark.parametrize("name", ["P3", "P4"])
def test_a_preflight_receipt_does_not_count_after_a_declared_setting_changes(tmp_path, name):
    """A receipt made under one declaration never counts for a run under another (release manager, 2026-11-11), with
    the snapshot, plan and trees unchanged; a field that differs by stage by design leaves it counting."""
    host, plan = _bound_host(tmp_path)
    host.c.update(settings={"compression": {"enabled": False}}, lcm_env={"K": "v"}, context_length=200000)
    path = tmp_path / f"{name}-made.json"
    _receipt(host, plan, name)  # Writes a receipt bound to this declaration; the next lines re-point to a copy.
    path.write_text(Path(host.c["preflights"][name]).read_text())
    host.c["preflights"][name] = str(path)
    assert host.preflight(name, plan)["verdict"] == "PASS"  # Positive control: nothing changed.
    host.c.update(preflights={name: str(path)}, settings_as_of="2026-11-11T00:00:00Z", instrument_dir=str(tmp_path))
    assert host.preflight(name, plan)["verdict"] == "PASS"  # Unbound by name: receipt paths, the time, the kit path.
    for field, value in (("settings", {"compression": {"enabled": True}}), ("lcm_env", {"K": "w"}),
                         ("context_length", 100000), ("p3_max_turns", 7), ("added_later", 1)):
        changed = r.LocalHost({**host.c, field: value}, host.run_dir)
        result = changed.preflight(name, plan)
        assert (result["verdict"], result["mismatch"]) == ("MISSING", ["config_sha256"]), field


def _stage_declaration(stage, pin):
    """A declaration shaped as the lane's declaration builder writes one per stage (preflights, dryrun, run): the
    stage folder, its tree extracts, its arm file, its run label and receipt name differ; the rest does not."""
    ga, rc = "a" * 40, ("c" * 40 if stage == "run" else "a" * 40)
    d, label = f"/scratch/{stage}", {"preflights": "d5pf", "dryrun": "d5dryrun", "run": "d5run"}[stage]
    return {
        "manifest": "/kit/d5-clone-pins.json", "rules_module": "bar1_rules.py", "interpreter": "/host/py",
        "host_py": "/host/py", "host_pin": "/host/src", "snapshot_dir": "/capture/snapshot", "snapshot_pin": str(pin),
        **{arm: {"host_src": "/host/src", "host_bin": "/host/bin", "lcm_tree": f"{d}/trees/{arm}-{c[:8]}", "commit": c}
           for arm, c in (("rc", rc), ("ga", ga))},
        "settings": {"compression": {"enabled": True}}, "lcm_env": {"LCM_X": "1"}, "context_length": 1000000,
        "p3_context_length": 64000, "p3_max_turns": 6, "nonce": "d5-v030-clone", "start_timeout": 600,
        "cold_ingest_timeout": 600, "settings_as_of": "2026-11-07T00:00:00Z",
        "constants_argv": ["{python}", "-B", "{instrument_dir}/gen_constants.py", "--arm", f"{d}/arm.json",
                           "--out", "{out}/s7-constants.json"],
        "replay_argv": ["{python}", "-B", "{instrument_dir}/p_s6r_clone.py", "--shape", "S7", "--label", label,
                        "--tree-base", ga, "--tree-candidate", rc, "--context-length", "1000000", "--out-dir", "{out}"],
        "replay_receipt": f"p-s6r-{label}.json",
        "preflights": {n: f"/scratch/{'run-p4' if stage == 'run' and n == 'P4' else 'preflights'}/receipts/{n}.json"
                       for n in ("P3", "P4", "P6", "P7")}}


def test_the_preflights_stage_receipts_count_in_the_dry_run_and_the_run_and_nothing_else_does(tmp_path):
    """The configuration binding across the builder's stages: P3, P6 and P7 made in the preflights stage count in the
    dry run and the run; P4 counts only with the run's own trees; a changed host build or replay command never counts."""
    _, plan = _bound_host(tmp_path)
    pin = tmp_path / "pin.json"
    decl = {stage: _stage_declaration(stage, pin) for stage in ("preflights", "dryrun", "run")}
    assert len({r.config_identity(c) for c in decl.values()}) == 1

    def made(stage, name):
        c = decl[stage]
        trees = {"rc": r.tree_id(c, "rc") if name in r.TREE_BOUND_PREFLIGHTS else c["ga"]["commit"],
                 "ga": c["ga"]["commit"]}
        path = tmp_path / f"{stage}-{name}.json"
        path.write_text(json.dumps({"schema": r.PREFLIGHT_SCHEMA, "name": name, "verdict": "PASS",
                                    "plan_sha256": planner.plan_sha256(plan), "snapshot_pin_sha256": capture._sha256(pin),
                                    "instrument_sha256": r.instrument_identity()["instrument_sha256"], "trees": trees,
                                    "config_sha256": r.config_identity(c)}))
        return str(path)

    def verdict(c, name, path):
        return r.LocalHost({**c, "preflights": {name: path}}, tmp_path / "run").preflight(name, plan)["verdict"]

    for name in ("P3", "P6", "P7"):
        assert verdict(decl["dryrun"], name, made("preflights", name)) == "PASS"
        assert verdict(decl["run"], name, made("preflights", name)) == "PASS"
    assert verdict(decl["run"], "P4", made("preflights", "P4")) == "MISSING"  # The run's P4 binds rc5's tree.
    assert verdict(decl["run"], "P4", made("run", "P4")) == "PASS"
    run = decl["run"]
    replay = run["replay_argv"]
    changed = {
        "host build": {**run, "rc": {**run["rc"], "host_bin": "/other/bin"}},
        "replay context": {**run, "replay_argv": [("64000" if v == "1000000" else v) for v in replay]},
        "replay shape": {**run, "replay_argv": [("S6" if v == "S7" else v) for v in replay]},
        "candidate not the rc commit": {**run, "replay_argv": [("e" * 40 if v == "c" * 40 else v) for v in replay]},
        "constants command": {**run, "constants_argv": run["constants_argv"] + ["--extra"]},
    }
    for what, c in changed.items():
        result = r.LocalHost({**c, "preflights": {"P6": made("preflights", "P6")}}, tmp_path / "run").preflight("P6", plan)
        assert (result["verdict"], result["mismatch"]) == ("MISSING", ["config_sha256"]), what


def test_an_unreadable_preflight_receipt_is_missing(tmp_path):
    host, plan = _bound_host(tmp_path)
    (tmp_path / "bad.json").write_text("{not json")
    host.c["preflights"]["P3"] = str(tmp_path / "bad.json")
    assert host.preflight("P3", plan) == {"verdict": "MISSING", "mismatch": ["unreadable"]}


def test_the_instrument_identity_is_the_package_content():
    one, two = r.instrument_identity(), r.instrument_identity()
    assert one == two and len(one["instrument_sha256"]) == 64


@pytest.mark.parametrize("as_of, written", [("2026-11-07T01:02:03Z", "2026-11-07T01:02:03Z"),
                                             ("2026-11-07T01:02Z", "2026-11-07T01:02Z"),
                                             ("not a time", None), (12, None), (None, None),
                                             ("2026-11-07T01:02:03", None), ("2026-11-07T01:02:03+00:00", None),
                                             ("2026-99-99T99:99Z", None), ("\u0662026-11-07T01:02Z", None)])
def test_the_receipt_discloses_the_settings_time_and_nothing_else(tmp_path, as_of, written):
    """The settings snapshot's time is recorded as a disclosed fact (release manager, 2026-11-10); a value that is not a
    UTC time is never written."""
    snap = tmp_path / "snapshot"
    config = {"snapshot_dir": str(snap), "snapshot_pin": str(make_snapshot(snap)), "nonce": "synthetic"}
    if as_of is not None:
        config["settings_as_of"] = as_of
    receipt = r.run(FakeHost(config, tmp_path / "run"), rc_id="rc-synthetic")
    assert receipt["settings"]["as_of"] == written
    assert "not a time" not in json.dumps(receipt, sort_keys=True, default=str)


def test_a_host_without_a_config_writes_no_settings_time():
    class Bare:
        pass
    assert r._utc_time(getattr(Bare(), "c", {}).get("settings_as_of")) is None


def test_the_instrument_hash_follows_source_and_ignores_bytecode(tmp_path):
    """R5-5's instrument identity is computed from the package's files: a source edit changes it, a bytecode cache does
    not, and a different package root gives a different hash."""
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.json").write_text("{}")
    first = r.instrument_identity(pkg)["instrument_sha256"]
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"cache")
    (pkg / "sub" / "c.pyc").write_bytes(b"cache")
    assert r.instrument_identity(pkg)["instrument_sha256"] == first
    (pkg / "a.py").write_text("x = 2\n")
    assert r.instrument_identity(pkg)["instrument_sha256"] != first
    assert r.instrument_identity()["instrument_sha256"] != first  # The real package is hashed by default.


def test_a_run_dir_inside_the_instrument_package_is_refused(tmp_path):
    with pytest.raises(ValueError, match="outside the instrument package"):
        r.outside_package(r.PACKAGE / "runs" / "x")
    assert r.outside_package(tmp_path / "run") == tmp_path / "run"


@pytest.mark.parametrize("commit, bound", [("a" * 40, True), (None, False), ("", False), ("main", False),
                                           ("A" * 40, False)])
def test_only_a_declared_commit_identifies_a_tree(tmp_path, commit, bound):
    """A path is no tree identity (another tree can take the same path); an undeclared tree never binds a receipt."""
    host, plan = _bound_host(tmp_path)
    for t in ("rc", "ga"):
        host.c[t] = {"lcm_tree": str(tmp_path), **({"commit": commit} if commit is not None else {})}
    assert r.tree_id(host.c, "rc") == (commit if bound else None)
    receipt = {"schema": r.PREFLIGHT_SCHEMA, "name": "P4", "verdict": "PASS", "plan_sha256": r.planner.plan_sha256(plan),
               "snapshot_pin_sha256": r.capture._sha256(host.c["snapshot_pin"]),
               "instrument_sha256": r.instrument_identity()["instrument_sha256"],
               "trees": {"rc": r.tree_id(host.c, "rc"), "ga": r.tree_id(host.c, "ga")},
               "config_sha256": r.config_identity(host.c)}
    path = tmp_path / "P4.json"
    path.write_text(json.dumps(receipt))
    host.c["preflights"] = {"P4": str(path)}
    assert host.preflight("P4", plan)["verdict"] == ("PASS" if bound else "MISSING")


def test_the_run_entry_point_refuses_a_run_dir_inside_the_package(tmp_path):
    (tmp_path / "decl.json").write_text("{}")
    inside = r.PACKAGE / f"refused-{tmp_path.name}"
    with pytest.raises(ValueError, match="outside the instrument package"):
        r.main(["--config", str(tmp_path / "decl.json"), "--run-dir", str(inside), "--instrument-dir", str(tmp_path),
                "--rc-id", "rc-synthetic"])
    assert not inside.exists()
