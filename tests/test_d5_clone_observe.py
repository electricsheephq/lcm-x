"""The host observations bars 5 and 6 read: doctor and schema checks, and the workload facts, on synthetic homes."""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from bench.instruments.d5_clone import bars, run as r
from tests.test_d5_clone_run import FakeHost, setup  # noqa: F401  (setup is the shared fixture)

STUB = "[Externalized tool output: terminal; chars=204800; ref=20261107_120000_s_0123456789ab_1.json]"
OBSERVATIONS = ("compaction", "whitespace", "swap", "cron")
OK_SCHEMA = {"version": 5, "expected": 5, "missing_tables": [], "error": False}


def stores(home):
    home.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(home / "lcm.db")) as lcm, closing(sqlite3.connect(home / "state.db")) as host:
        lcm.executescript("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,"
                          " tool_call_id TEXT); CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);"
                          " CREATE TABLE summary_nodes (node_id INTEGER PRIMARY KEY, session_id TEXT);")
        host.executescript("CREATE TABLE sessions (id TEXT PRIMARY KEY); CREATE TABLE messages (id INTEGER PRIMARY KEY,"
                           " session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT);")
    return home


def observe(home, name, sid="S", cron_id=None, *, native=False, stripped="  \r\n", host_text="d" * 64):
    """Write one observation into a home's stores, as the host and LCM-X would."""
    with closing(sqlite3.connect(home / "lcm.db")) as lcm, closing(sqlite3.connect(home / "state.db")) as host, lcm, host:
        if name == "compaction":
            lcm.execute("INSERT INTO summary_nodes (session_id) VALUES (?)", (sid,))
            lcm.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)",
                        (f"compaction_commit_proof:{sid}", json.dumps({"native": native, "session_id": sid})))
        elif name == "whitespace":
            store_id = lcm.execute("INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
                                   (sid, "hello" + stripped)).lastrowid
            lcm.execute("INSERT INTO metadata VALUES (?, ?)", (f"host_rewrite_identity:{store_id}",
                        json.dumps({"version": 1, "content": "hello", "stripped": ["", stripped]})))
        elif name == "swap":
            lcm.execute("INSERT INTO messages (session_id, role, content, tool_call_id) VALUES (?, 'tool', ?, 'c1')",
                        (sid, STUB))
            host.execute("INSERT INTO messages (session_id, role, content, tool_call_id) VALUES (?, 'tool', ?, 'c1')",
                         (sid, host_text))
        elif name == "cron":
            host.execute("INSERT INTO sessions (id) VALUES (?)", (f"cron_{cron_id}_20261107_120000",))


def test_each_workload_fact_counts_on_a_positive_home_and_is_zero_on_a_negative_one(tmp_path):
    home = stores(tmp_path / "home")
    before = r.marks(home)
    assert r.workload_facts(home, before, "0123456789ab") == {
        "cron_fired": 0, "compactions_published": 0, "whitespace_rewrites": 0, "payload_swaps": 0}
    for name in OBSERVATIONS:
        observe(home, name, cron_id="0123456789ab")
    assert r.workload_facts(home, before, "0123456789ab") == {
        "cron_fired": 1, "compactions_published": 1, "whitespace_rewrites": 1, "payload_swaps": 1}
    assert r.workload_facts(home, r.marks(home), "0123456789ab")["compactions_published"] == 0  # Present before the run.
    assert r.workload_facts(tmp_path / "absent", {}, None) == {
        "cron_fired": 0, "compactions_published": 0, "whitespace_rewrites": 0, "payload_swaps": 0}


@pytest.mark.parametrize("name, kwargs, key", [
    ("compaction", {"native": True}, "compactions_published"),  # A native proof is not an LCM-X publication.
    ("whitespace", {"stripped": ""}, "whitespace_rewrites"),  # No edge whitespace recorded.
    ("swap", {"host_text": STUB}, "payload_swaps"),  # The host row holds the same text: nothing to credit.
    ("cron", {"cron_id": "ffffffffffff"}, "cron_fired"),  # Another job's run.
])
def test_a_near_miss_is_not_counted(tmp_path, name, kwargs, key):
    home = stores(tmp_path / "home")
    before = r.marks(home)
    observe(home, name, **{"cron_id": "0123456789ab", **kwargs})
    assert r.workload_facts(home, before, "0123456789ab")[key] == 0


def probe_out(path, doctor=None, schema=None):
    path.mkdir(parents=True, exist_ok=True)
    if doctor is not None:
        (path / "doctor.json").write_text(json.dumps(doctor))
    if schema is not None:
        (path / "schema.json").write_text(json.dumps(schema))
    return path


def doctor(*statuses):
    return {"overall": "healthy", "checks": [{"check": f"c{i}", "status": s, "detail": "x"} for i, s in enumerate(statuses)]}


def test_doctor_checks_map_to_bar5_statuses(tmp_path):
    out = probe_out(tmp_path, doctor("pass", "warn"), {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA})
    checks = r.probe_checks(out)
    assert [c for c in checks if c["kind"] == "doctor"] == [
        {"kind": "doctor", "check": "c0", "status": "ok"}, {"kind": "doctor", "check": "c1", "status": "warn"}]
    assert [c["status"] for c in checks if c["kind"] == "schema"] == ["ok", "ok"]
    assert bars.failed_checks(checks) == [] and not {"doctor", "schema"} & set(bars.kinds_missing(checks))


@pytest.mark.parametrize("status", ["fail", "FAIL", "degraded"])
def test_a_doctor_fail_or_an_unknown_status_is_a_failed_check(tmp_path, status):
    assert bars.failed_checks(r.probe_checks(probe_out(tmp_path, doctor("pass", status)))) == ["c1"]


@pytest.mark.parametrize("value", [None, {"error": "LCM engine not initialized"}, {"checks": []}, "not json"])
def test_a_doctor_that_did_not_run_leaves_the_kind_missing(tmp_path, value):
    out = probe_out(tmp_path)
    if value is not None:
        (out / "doctor.json").write_text(value if isinstance(value, str) else json.dumps(value))
    assert "doctor" in bars.kinds_missing(r.probe_checks(out)) and "doctor" in bars.kinds_missing(r.probe_checks(None))


@pytest.mark.parametrize("bad", [{"version": 4}, {"missing_tables": ["messages"]}, {"error": True}, {"version": None}])
def test_a_schema_mismatch_fails_that_db(tmp_path, bad):
    checks = r.probe_checks(probe_out(tmp_path, schema={"lcm.db": {**OK_SCHEMA, **bad}, "state.db": OK_SCHEMA}))
    assert bars.failed_checks(checks) == ["lcm.db"]


@pytest.mark.parametrize("schema", [None, {"lcm.db": OK_SCHEMA}])
def test_an_unread_schema_leaves_the_kind_missing(tmp_path, schema):
    assert "schema" in bars.kinds_missing(r.probe_checks(probe_out(tmp_path, schema=schema)))


def test_the_cron_job_is_created_once_before_epoch_2_identically_in_both_arms(tmp_path):
    plan = {"cron": {"schedule": "every 2m", "repeat": 1, "created_before_epoch": 2, "prompt": "d5 cron abc"}}
    jobs = []
    for arm in ("rc", "ga"):
        home = tmp_path / arm
        (home / "cron").mkdir(parents=True)
        (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{"id": "kept"}]}))
        sha = r.LocalHost({}, tmp_path).schedule({"home": home}, plan["cron"])
        data = json.loads((home / "cron" / "jobs.json").read_text())
        assert [j["id"] for j in data["jobs"]][0] == "kept" and sha == r._sha(data["jobs"][1]["id"])
        jobs.append({k: v for k, v in data["jobs"][1].items() if k not in ("created_at", "next_run_at")})
    assert jobs[0] == jobs[1] and jobs[0]["model"] == "d5-cron" and jobs[0]["repeat"] == {"times": 1, "completed": 0}
    assert jobs[0]["schedule"] == {"kind": "interval", "minutes": 2, "display": "every 2m"}


class ObservingHost(FakeHost):
    """The fake run with the real checks() and facts(): the fake turns leave each observation in the arm's stores and
    a synthetic probe output stands in for the doctor and schema probe, unless *absent* names it."""

    checks, facts = r.LocalHost.checks, r.LocalHost.facts

    def __init__(self, config, run_dir, absent=None):
        super().__init__(config, run_dir)
        self.absent, self.cron_steps = absent, []

    def arm(self, name, home, plan):
        ctx = super().arm(name, home, plan)
        with closing(sqlite3.connect(home / "lcm.db")) as lcm, closing(sqlite3.connect(home / "state.db")) as host:
            for db in (lcm, host):
                db.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
            lcm.executescript("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);"
                              " CREATE TABLE summary_nodes (node_id INTEGER PRIMARY KEY, session_id TEXT);")
        payloads = sum(p.is_file() for p in (home / "lcm-large-outputs").rglob("*"))
        return {**ctx, "marks": r.marks(home), "payloads": payloads}

    def rules(self):  # The swap's two rows are one occurrence, as the pinned payload credit makes them.
        return {**super().rules(), "credits": (lambda row, host: row["role"] == host["role"] == "tool"
                                               and row["tool_call_id"] is not None
                                               and row["tool_call_id"] == host["tool_call_id"],)}

    def turn(self, ctx, step):
        ok = super().turn(ctx, step)
        sid = ctx["sessions"][step["conversation"]]
        names = {"e0-S1-1": "whitespace", "e1-L-1": "swap", "e1-L-2": "compaction"}
        if step["id"] in names and names[step["id"]] != self.absent:
            observe(ctx["home"], names[step["id"]], sid)
        if step["id"] == "e1-L-1":
            (ctx["home"] / "lcm-large-outputs" / f"{ctx['name']}-payload.json").write_text("{}")
        if step["id"] == "e2-L-0" and self.absent != "cron":
            observe(ctx["home"], "cron", cron_id=ctx["cron_id"])
        return ok

    def probe(self, ctx):
        out = ctx["home"].parent / "doctor" / "out"
        return probe_out(out, None if self.absent == "doctor" else doctor("pass", "warn"),
                         None if self.absent == "schema" else {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA})


def test_a_full_fake_run_with_every_observation_passes_bars_5_and_6(setup):  # noqa: F811
    _, config, run_dir = setup
    receipt = r.run(ObservingHost(config, run_dir), rc_id="rc-synthetic")
    assert receipt["bars"]["bar5"]["verdict"] == bars.PASS and receipt["bars"]["bar6"]["verdict"] == bars.PASS
    assert receipt["verdict"]["raw"] == bars.PASS, receipt["bars"]
    work = receipt["arms"]["rc"]["workload"]
    assert work["cron_job_sha256"] == receipt["arms"]["ga"]["workload"]["cron_job_sha256"]
    assert receipt["arms"]["rc"]["checks"]["by_kind_status"]["doctor:warn"] == 1


@pytest.mark.parametrize("absent, bar, item", [
    ("doctor", "bar5", "doctor"), ("schema", "bar5", "schema"), ("cron", "bar6", "cron_fired"),
    ("compaction", "bar6", "compaction_published"), ("whitespace", "bar6", "whitespace_rewrite"),
    ("swap", "bar6", "payload_swap")])
def test_each_missing_observation_makes_its_bar_inconclusive(setup, absent, bar, item):  # noqa: F811
    _, config, run_dir = setup
    receipt = r.run(ObservingHost(config, run_dir, absent=absent), rc_id="rc-synthetic")
    result = receipt["bars"][bar]
    assert result["verdict"] == bars.INCONCLUSIVE and receipt["verdict"]["raw"] == bars.INCONCLUSIVE
    assert (result["kinds_missing"] == [item]) if bar == "bar5" else result["items"]["rc"][item] is False


PINNED = pytest.mark.skipif(not (os.environ.get("D5_HOST_SRC") and os.environ.get("D5_HOST_BIN")),
                            reason="D5_HOST_SRC and D5_HOST_BIN not set")


def host_python(code, env, cwd):
    """Run *code* with the pinned host's interpreter; the host's own dependencies are only in its venv."""
    return subprocess.run([str(Path(os.environ["D5_HOST_BIN"]) / "python"), "-P", "-B", "-c", code], env=env, cwd=cwd,
                          check=True, timeout=300, capture_output=True, text=True).stdout


@PINNED
def test_the_cron_job_parses_with_the_pinned_hosts_own_loader(tmp_path):
    home = tmp_path / "home"
    cron = {"schedule": "every 2m", "repeat": 1, "created_before_epoch": 2, "prompt": "d5 cron abc"}
    r.LocalHost({}, tmp_path).schedule({"home": home}, cron)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HERMES_HOME": str(home), "TMPDIR": str(tmp_path)}
    code = ("import json; from cron import jobs\nwith jobs.use_cron_store(" + repr(str(home)) + "):\n"
            "    (job,) = jobs.load_jobs()\n    state = [jobs.is_job_runnable(job), jobs.effective_job_state(job)]\n"
            "print(json.dumps([job, state, jobs.parse_schedule('every 2m'), jobs.normalize_repeat_value(1)]))")
    job, state, schedule, repeat = json.loads(host_python(code, env, tmp_path))
    assert state == [True, "scheduled"] and job["schedule"] == schedule and job["repeat"]["times"] == repeat
    assert job["model"] == "d5-cron" and len(job["id"]) == 12


@PINNED
def test_the_real_probe_runs_the_doctor_on_a_clone_and_never_writes_the_arm_home(tmp_path):
    """The arm's stores are made by the pinned host and this tree's engine; the probe runs on a clone of them."""
    home, tree = tmp_path / "arm" / "home", Path(r.__file__).resolve().parents[3]
    home.mkdir(parents=True)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "HERMES_HOME": str(home), "LCM_DATABASE_PATH": str(home / "lcm.db"),
           "LCM_EMBEDDINGS_ENABLED": "0", "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path)}
    make = ("import importlib, importlib.util, os, sys; import hermes_state; hermes_state.SessionDB("
            f"db_path=__import__('pathlib').Path({str(home / 'state.db')!r})).close(); t = {str(tree)!r}; s = importlib.util.spec_from_file_location("
            "'hermes_lcm_x', t + '/__init__.py', submodule_search_locations=[t]); sys.modules['hermes_lcm_x'] ="
            " importlib.util.module_from_spec(s); c = importlib.import_module('hermes_lcm_x.config');"
            " e = importlib.import_module('hermes_lcm_x.engine'); e.LCMEngine(config=c.LCMConfig.from_env(),"
            f" hermes_home={str(home)!r})._store.connection; sys.stdout.flush(); os._exit(0)")
    host_python(make, env, tmp_path)
    before = {p.relative_to(home): p.stat().st_mtime_ns for p in home.rglob("*") if p.is_file()}
    ctx = {"home": home, "gateway": type("G", (), {"env": env})(),
           "tree": {"host_bin": os.environ["D5_HOST_BIN"], "lcm_tree": str(tree)}}
    checks = r.probe_checks(r.LocalHost({}, tmp_path).probe(ctx))
    assert {p.relative_to(home): p.stat().st_mtime_ns for p in home.rglob("*") if p.is_file()} == before
    assert {c["kind"] for c in checks} == {"doctor", "schema"} and bars.failed_checks(checks) == []
    assert any(c["check"] == "schema_core_tables" and c["status"] == "ok" for c in checks)
