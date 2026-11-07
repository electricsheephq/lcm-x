"""The whole run on a synthetic snapshot with a fake host and a fake edition; no gateway, no network."""

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3

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
        self.fail, self.armed, self.replays = fail, [], []

    def free_bytes(self):
        return 0 if self.fail == "free_space" else r.NEED_FREE

    def pins(self):
        return {"verdict": "FAIL" if self.fail == "P1" else "PASS", "mismatches": [], "functions_sha256": "f" * 64}

    def rules(self):
        return {"identity": lambda row: (row["role"], row["content"]), "credits": (), "covered": lambda sid: True}

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
        sid = ctx["sessions"][step["conversation"]]
        with closing(sqlite3.connect(ctx["home"] / "lcm.db")) as lcm, closing(sqlite3.connect(ctx["home"] / "state.db")) as host:
            if step["kind"] == "cold" and ctx["epoch"]:  # A cold-bind re-store of one old row (a bar-2 copy).
                lcm.execute("INSERT INTO messages (session_id, role, content) SELECT session_id, role, content FROM"
                            " messages WHERE session_id = ? ORDER BY store_id LIMIT 1", (sid,))
            rows = [(sid, "user", f"{MARKER} {step['id']}"), (sid, "assistant", f"{MARKER} reply {step['id']}")]
            for db in (host,) if self.fail == "lose" and ctx["name"] == "rc" and step["id"] == "e1-S1-1" else (lcm, host):
                db.executemany("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", rows)
                db.commit()
        return True

    def replay(self, ctx, step, t0, h8):
        self.replays.append(step["id"])
        assert (Path(t0) / "capture-hashes.json").is_file() and (Path(h8) / "RC3").exists() is False
        checks = {k: {"verdict": "PASS"} for k in (bars.VALIDITY, *bars.COMPARISONS)}
        return {"checks": checks, "bind_identity": {"dup_store_ids": [1]}}, False, 0

    def egress(self, ctx):
        return 0

    def wait(self, seconds):
        pass

    def logs(self, ctx):
        return [f"synthetic log {MARKER}"]

    def roots(self, ctx):
        return [str(ctx["home"] / "plugins" / "hermes-lcm-x")]

    def checks(self, ctx):
        return [{"kind": k, "check": k, "status": "ok"} for k in bars.CHECK_KINDS]

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
    twice = [{"rc": "rc-synthetic", "verdict": {"raw": bars.INCONCLUSIVE}, "pins": {"functions_sha256": "e" * 64}}] * 2
    assert "defect" in r.run(FakeHost(config, run_dir), rc_id="rc-synthetic", prior=twice)["stopped_at"]
    fixed = r.run(FakeHost(config, run_dir), rc_id="rc-synthetic", prior=twice, instrument_fix="fix-1")
    assert "rule-function" in fixed["stopped_at"]


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
    config = {"interpreter": "python3", "instrument_dir": "kit", "rc": {"lcm_tree": "rc"}, "ga": {"lcm_tree": "ga"},
              "constants_argv": ["{python}", "gen", "{t0}", "{h8}", "{out}"], "replay_argv": ["{python}", "replay"]}
    host = r.LocalHost(config, tmp_path / "run")
    receipt, error, range_new = host.replay({"name": "rc", "port": 4242}, {"id": "e0-L-0"}, "T0", "H8")
    assert (receipt, error, range_new) == ({"checks": {}}, False, 0)
    assert [argv[:4] for argv in seen] == [["/usr/bin/nice", "-n", "19", "/usr/bin/sandbox-exec"]] * 2
    assert seen[0][-5:] == ["python3", "gen", "T0", "H8", os.fspath(tmp_path / "run" / "rc" / "replays" / "e0-L-0")]
    assert 'localhost:4242' in seen[0][5]
