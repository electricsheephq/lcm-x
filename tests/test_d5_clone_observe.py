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

REF = "20261107_120000_s_0123456789ab_1.json"
STUB = f"[Externalized tool output: terminal; chars=204800; ref={REF}]"
RAW = "d" * 2048
GC = STUB.replace("Externalized", "GC'd externalized")  # LCM-X's other reference form, a second row of the same call.
KEY = r._sha("S")[:12]
OBSERVATIONS = ("compaction", "whitespace", "swap", "cron")
OK_SCHEMA = {"version": 5, "expected": 5, "missing_tables": [], "error": False}


def stores(home):
    (home / "lcm-large-outputs").mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(home / "lcm.db")) as lcm, closing(sqlite3.connect(home / "state.db")) as host:
        lcm.executescript("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,"
                          " tool_call_id TEXT); CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);"
                          " CREATE TABLE summary_nodes (node_id INTEGER PRIMARY KEY, session_id TEXT);")
        host.executescript("CREATE TABLE sessions (id TEXT PRIMARY KEY); CREATE TABLE messages (id INTEGER PRIMARY KEY,"
                           " session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT);")
    return home


def observe(home, name, sid="S", cron_id=None, *, native=False, stripped="  \r\n", host_text=STUB, lcm_texts=(RAW,),
            payload=RAW):
    """Write one observation into a home's stores, as the host and LCM-X would. A payload swap: the host row holds the
    externalized stub, the payload file holds the output, and LCM-X stores the output itself."""
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
            (home / "lcm-large-outputs" / REF).write_text(json.dumps({"content": payload}))
            lcm.executemany("INSERT INTO messages (session_id, role, content, tool_call_id) VALUES (?, 'tool', ?, 'c1')",
                            [(sid, text) for text in lcm_texts])
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
    ("cron", {"cron_id": "ffffffffffff"}, "cron_fired"),  # Another job's run.
])
def test_a_near_miss_is_not_counted(tmp_path, name, kwargs, key):
    home = stores(tmp_path / "home")
    before = r.marks(home)
    observe(home, name, **{"cron_id": "0123456789ab", **kwargs})
    assert r.workload_facts(home, before, "0123456789ab")[key] == 0


@pytest.mark.parametrize("kwargs", [  # Each holds an LCM-X reference row that the round-0 rule counted.
    {"host_text": RAW, "lcm_texts": (STUB,)},  # LCM-X externalized it and the host kept the output: no host swap.
    {"host_text": RAW[:64], "lcm_texts": (STUB,)},  # A host truncation.
    {"host_text": STUB + " " + STUB, "lcm_texts": (RAW, GC)},  # Two references in one host row.
    {"payload": RAW[:-1] + "e", "lcm_texts": (RAW, GC)},  # The payload differs from LCM-X's row by one byte.
    {"host_text": STUB.replace(REF, "sub/" + REF), "lcm_texts": (RAW, GC)},  # A reference the engine cannot open.
])
def test_only_the_pinned_credits_payload_swap_shape_counts(tmp_path, kwargs):
    home = stores(tmp_path / "home")
    before = r.marks(home)
    observe(home, "swap", **kwargs)
    assert r.workload_facts(home, before)["payload_swaps"] == 0


PRE_OPEN_OK = {"overall": "healthy", "checks": [{"check": c, "status": "pass", "detail": "x"} for c in r.DOCTOR_PRE_OPEN]}


def probe_out(path, doctor=None, schema=None, pre_open=PRE_OPEN_OK):
    path.mkdir(parents=True, exist_ok=True)
    if doctor is not None:
        (path / "doctor.json").write_text(json.dumps(doctor))
    if pre_open is not None:
        (path / "fts-pre-open.json").write_text(json.dumps(pre_open))
    if schema is not None:
        (path / "schema.json").write_text(json.dumps(schema))
    return path


def doctor(*statuses, names=("database_integrity", "payload_storage")):
    return {"overall": "healthy", "checks": [{"check": n, "status": s, "detail": "x"} for n, s in zip(names, statuses)]}


def per_session(out, key=KEY, *statuses, bound=True):
    """The probe's run bound to one session, and its sessions.json."""
    (out / f"doctor-{key}.json").write_text(json.dumps(doctor(*statuses, names=r.DOCTOR_PER_SESSION)))
    (out / "sessions.json").write_text(json.dumps({key: bound}))


def test_doctor_checks_map_to_bar5_statuses(tmp_path):
    out = probe_out(tmp_path, doctor("pass", "warn"), {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA})
    checks, skipped = r.probe_checks(out, [])
    assert [c for c in checks if c["kind"] == "doctor"] == [{"kind": "doctor", "check": "database_integrity", "status": "ok"},
                                                            {"kind": "doctor", "check": "payload_storage", "status": "warn"},
                                                            *({"kind": "doctor", "check": c, "status": "ok"} for c in r.DOCTOR_PRE_OPEN)]
    assert [c["status"] for c in checks if c["kind"] == "schema"] == ["ok", "ok"] and skipped == []
    assert bars.failed_checks(checks) == [] and not {"doctor", "schema"} & set(bars.kinds_missing(checks))


@pytest.mark.parametrize("status", ["fail", "FAIL", "degraded"])
def test_a_doctor_fail_or_an_unknown_status_is_a_failed_check(tmp_path, status):
    out = probe_out(tmp_path, doctor("pass", status))
    assert bars.failed_checks(r.probe_checks(out, [])[0]) == ["payload_storage"]
    per_session(out, KEY, "pass", status, "pass")
    assert bars.failed_checks(r.probe_checks(out, [KEY])[0]) == [f"orphaned_dag_nodes@{KEY}", "payload_storage"]


def test_checks_the_probe_cannot_exercise_are_reported_never_rows(tmp_path):
    names = ("database_integrity", "identity_migration", "ingest_health", "context_pressure", "new_check",
             *r.DOCTOR_PER_SESSION)
    out = probe_out(tmp_path, doctor(*["pass"] * len(names), names=names))
    per_session(out, KEY, "pass", "warn", "pass")
    checks, skipped = r.probe_checks(out, [KEY])
    assert [c["check"] for c in checks] == ["database_integrity", *r.DOCTOR_PRE_OPEN,
                                            *(f"{c}@{KEY}" for c in r.DOCTOR_PER_SESSION)]
    assert checks[4]["status"] == "warn"  # The bound session's own result.
    not_exercised = {"context_pressure": "the probe process's own counter",
                     "identity_migration": "no plugin registration step in the probe",
                     "ingest_health": "the probe process's own counter", "new_check": "not classified for the probe"}
    assert {s["check"]: s["reason"] for s in skipped} == not_exercised
    checks, skipped = r.probe_checks(out, [KEY, "000000000000"])  # The second session never bound.
    assert checks == [] and "doctor" in bars.kinds_missing(checks)  # A session not looked at is never a PASS.
    assert {s["check"]: s["reason"] for s in skipped} == {
        **not_exercised, **{f"{c}@000000000000": "no session binding in this tree" for c in r.DOCTOR_PER_SESSION}}


def test_a_failing_unclassified_check_fails_bar5(tmp_path):
    names = ("database_integrity", "messages_fts_integrity_background_flag", "another_new_check", "identity_migration")
    out = probe_out(tmp_path, doctor("pass", "fail", "pass", "fail", names=names))
    checks, skipped = r.probe_checks(out, [])
    assert bars.failed_checks(checks) == ["messages_fts_integrity_background_flag"]
    assert {s["check"] for s in skipped} == {"another_new_check", "identity_migration"}  # Ruled not exercised.



def test_an_unclassified_check_that_warns_is_a_row_judged_against_ga(tmp_path):
    names = ("database_integrity", "brand_new_check", "identity_migration")
    rc = r.probe_checks(probe_out(tmp_path / "rc", doctor("pass", "warn", "warn", names=names)), [])[0]
    assert {c["check"] for c in rc} == {"database_integrity", "brand_new_check", *r.DOCTOR_PRE_OPEN}  # Ruled one out.
    sqlite = [{"kind": k, "check": "lcm.db", "status": "ok"} for k in ("quick_check", "integrity", "schema")]
    for ga_status, verdict in (("pass", bars.FAIL), ("warn", bars.PASS)):  # Only the rc warns: over GA's count.
        ga = r.probe_checks(probe_out(tmp_path / ga_status, doctor("pass", ga_status, names=names[:2])), [])[0]
        assert bars.bar5(sqlite + rc, sqlite + ga)["verdict"] == verdict


@pytest.mark.parametrize("pre_open, verdict", [
    (None, bars.INCONCLUSIVE),  # The probe never checked FTS before the engine opened (and repaired) the store.
    ({"overall": "healthy", "checks": [{"check": r.DOCTOR_PRE_OPEN[0], "status": "pass", "detail": "x"}]}, bars.INCONCLUSIVE),
    ({"overall": "healthy", "checks": [{"check": c, "status": s, "detail": "x"} for c, s in
                                       zip(r.DOCTOR_PRE_OPEN, ("fail", "pass"))]}, bars.FAIL)])
def test_fts_damage_the_engine_would_repair_is_still_seen(tmp_path, pre_open, verdict):
    out = probe_out(tmp_path, doctor("pass", "pass"), {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA}, pre_open=pre_open)
    checks, _ = r.probe_checks(out, [])  # The doctor's own run saw the index after the engine rebuilt it: all pass.
    sqlite = [{"kind": k, "check": "lcm.db", "status": "ok"} for k in ("quick_check", "integrity")]
    assert bars.bar5(sqlite + checks, sqlite + checks)["verdict"] == verdict

def test_an_unbound_plan_session_leaves_the_doctor_unshown_but_a_fail_still_fails(tmp_path):
    out = probe_out(tmp_path, doctor("pass", "warn"), {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA})
    per_session(out, KEY, "pass", "pass", "pass", bound=False)
    checks, _ = r.probe_checks(out, [KEY])
    assert [c["kind"] for c in checks] == ["schema", "schema"] and "doctor" in bars.kinds_missing(checks)
    out = probe_out(tmp_path / "f", doctor("pass", "fail"))
    per_session(out, KEY, "pass", "pass", "pass", bound=False)
    assert bars.failed_checks(r.probe_checks(out, [KEY])[0]) == ["payload_storage"]



def test_a_bound_session_missing_a_per_session_check_is_not_looked_at(tmp_path):
    out = probe_out(tmp_path, doctor("pass", "warn"))
    (out / f"doctor-{KEY}.json").write_text(json.dumps(doctor("pass", "pass", names=r.DOCTOR_PER_SESSION[:2])))
    (out / "sessions.json").write_text(json.dumps({KEY: True}))
    checks, skipped = r.probe_checks(out, [KEY])
    assert checks == [] and "doctor" in bars.kinds_missing(checks)
    assert {s["check"] for s in skipped} == {f"{r.DOCTOR_PER_SESSION[2]}@{KEY}"}
    (out / f"doctor-{KEY}.json").write_text(json.dumps(doctor("fail", "pass", names=r.DOCTOR_PER_SESSION[:2])))
    checks, _ = r.probe_checks(out, [KEY])  # An incomplete report that fails a check still fails it.
    assert bars.failed_checks(checks) == [f"{r.DOCTOR_PER_SESSION[0]}@{KEY}"]
    sqlite = [{"kind": k, "check": "lcm.db", "status": "ok"} for k in ("quick_check", "integrity", "schema")]
    ok = [{"kind": "doctor", "check": "database_integrity", "status": "ok"}]
    assert bars.bar5(sqlite + ok, sqlite + checks)["ga_rerun_needed"] is True  # As the GA arm: it re-runs, never PASS.


def test_a_ga_arm_whose_doctor_did_not_run_leaves_bar5_unjudged(tmp_path):
    rc = r.probe_checks(probe_out(tmp_path / "rc", doctor("pass", "warn"), {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA}), [])[0]
    ga = r.probe_checks(probe_out(tmp_path / "ga", None, {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA}), [])[0]
    sqlite = [{"kind": k, "check": "lcm.db", "status": "ok"} for k in ("quick_check", "integrity")]
    result = bars.bar5(sqlite + rc, sqlite + ga)  # The GA probe crashed or timed out: no doctor rows in that arm.
    assert result["verdict"] == bars.INCONCLUSIVE and result["ga_kinds_missing"] == ["doctor"]

@pytest.mark.parametrize("value", [None, {"error": "LCM engine not initialized"}, {"checks": []}, "not json"])
def test_a_doctor_that_did_not_run_leaves_the_kind_missing(tmp_path, value):
    out = probe_out(tmp_path)
    if value is not None:
        (out / "doctor.json").write_text(value if isinstance(value, str) else json.dumps(value))
    assert "doctor" in bars.kinds_missing(r.probe_checks(out, [KEY])[0])
    assert r.probe_checks(None, [KEY]) == ([], [])


@pytest.mark.parametrize("value", [None, {"error": "LCM engine not initialized"}, "not json"])
@pytest.mark.parametrize("statuses, verdict", [(("fail", "pass"), bars.FAIL), (("pass", "pass"), bars.INCONCLUSIVE),
                                               (("warn", "pass"), bars.INCONCLUSIVE)])
def test_a_pre_open_fts_failure_counts_when_the_doctor_did_not_report(tmp_path, value, statuses, verdict):  # #2453 item 2
    pre_open = {"overall": "healthy", "checks": [{"check": c, "status": s, "detail": "x"}
                                                 for c, s in zip(r.DOCTOR_PRE_OPEN, statuses)]}
    out = probe_out(tmp_path, None, {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA}, pre_open=pre_open)
    if value is not None:
        (out / "doctor.json").write_text(value if isinstance(value, str) else json.dumps(value))
    checks, _ = r.probe_checks(out, [KEY])
    sqlite = [{"kind": k, "check": "lcm.db", "status": "ok"} for k in ("quick_check", "integrity")]
    assert bars.bar5(sqlite + checks, sqlite + checks)["verdict"] == verdict
    assert bars.failed_checks(checks) == ([r.DOCTOR_PRE_OPEN[0]] if verdict == bars.FAIL else [])


@pytest.mark.parametrize("bad", [{"version": 4}, {"missing_tables": ["messages"]}, {"error": True}, {"version": None}])
def test_a_schema_mismatch_fails_that_db(tmp_path, bad):
    checks, _ = r.probe_checks(probe_out(tmp_path, schema={"lcm.db": {**OK_SCHEMA, **bad}, "state.db": OK_SCHEMA}), [])
    assert bars.failed_checks(checks) == ["lcm.db"]


@pytest.mark.parametrize("schema", [None, {"lcm.db": OK_SCHEMA}])
def test_an_unread_schema_leaves_the_kind_missing(tmp_path, schema):
    assert "schema" in bars.kinds_missing(r.probe_checks(probe_out(tmp_path, schema=schema), [])[0])


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


@pytest.mark.parametrize("shape", ["dict", "list"])
def test_existing_jobs_are_paused_and_the_file_keeps_its_shape(tmp_path, shape):
    cron = {"schedule": "every 2m", "repeat": 1, "created_before_epoch": 2, "prompt": "d5 cron abc"}
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    jobs = [{"id": "b2", "enabled": True, "state": "scheduled"}, {"id": "a1", "enabled": True}]
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": jobs} if shape == "dict" else jobs))
    host = r.LocalHost({}, tmp_path)
    assert host.pause_cron({"home": home}) == {"job_ids": ["a1", "b2"], "count": 2}
    host.schedule({"home": home}, cron)
    data = json.loads((home / "cron" / "jobs.json").read_text())
    assert isinstance(data, dict if shape == "dict" else list)
    *old, new = data["jobs"] if shape == "dict" else data
    assert all(j["enabled"] is False and j["state"] == "paused" and j["paused_at"]
               and j["paused_reason"] == "d5 clone run: paused before the run" for j in old)
    assert new["enabled"] is True and new["state"] == "scheduled"  # The plan's own job stays enabled.


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
        (home / "cron").mkdir()
        (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [{"id": "snap-job", "enabled": True}]}))
        payloads = sum(p.is_file() for p in (home / "lcm-large-outputs").rglob("*"))
        return {**ctx, "marks": r.marks(home), "payloads": payloads, "session_ids": list(ctx["sessions"].values())}

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
        if step["id"] == "e2-L-0" and self.absent != "cron":
            observe(ctx["home"], "cron", cron_id=ctx["cron_id"])
        return ok

    def probe(self, ctx):
        out = ctx["home"].parent / "doctor" / "out"
        names = ("database_integrity", "payload_storage", "identity_migration", *r.DOCTOR_PER_SESSION)
        probe_out(out, None if self.absent == "doctor" else doctor("pass", "warn", "pass", "pass", "pass", "pass",
                                                                    names=names),
                  None if self.absent == "schema" else {"lcm.db": OK_SCHEMA, "state.db": OK_SCHEMA})
        keys = [r._sha(s)[:12] for s in ctx["session_ids"]]
        for key in keys:
            per_session(out, key, "pass", "pass", "pass")
        (out / "sessions.json").write_text(json.dumps(dict.fromkeys(keys, True)))
        return out


def test_a_full_fake_run_with_every_observation_passes_bars_5_and_6(setup):  # noqa: F811
    _, config, run_dir = setup
    receipt = r.run(ObservingHost(config, run_dir), rc_id="rc-synthetic")
    assert receipt["bars"]["bar5"]["verdict"] == bars.PASS and receipt["bars"]["bar6"]["verdict"] == bars.PASS
    assert receipt["verdict"]["raw"] == bars.PASS, receipt["bars"]
    work = receipt["arms"]["rc"]["workload"]
    assert work["cron_job_sha256"] == receipt["arms"]["ga"]["workload"]["cron_job_sha256"]
    for arm in ("rc", "ga"):
        assert receipt["arms"][arm]["checks"]["by_kind_status"] == {
            "doctor:ok": 12, "doctor:warn": 1, "integrity:ok": 2, "quick_check:ok": 2, "schema:ok": 2}  # 1 + 2 + 3 x 3 ok
        assert receipt["arms"][arm]["doctor_not_exercised"] == [
            {"check": "identity_migration", "reason": "no plugin registration step in the probe"}]
        assert receipt["arms"][arm]["cron_paused"] == {"job_ids": ["snap-job"], "count": 1}


@pytest.mark.parametrize("absent, bar, item", [
    ("doctor", "bar5", "doctor"), ("schema", "bar5", "schema"), ("cron", "bar6", "cron_fired"),
    ("compaction", "bar6", "compaction_published"), ("whitespace", "bar6", "whitespace_rewrite"),
    ("swap", "bar6", "payload_swap")])
def test_each_missing_observation_makes_its_bar_inconclusive(setup, absent, bar, item):  # noqa: F811
    _, config, run_dir = setup
    receipt = r.run(ObservingHost(config, run_dir, absent=absent), rc_id="rc-synthetic")
    result = receipt["bars"][bar]
    assert receipt["arms"]["rc"]["cron_paused"]["count"] == 1  # The snapshot's job never fires in any variant.
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
def test_the_hosts_loader_reads_paused_jobs_as_not_due(tmp_path):
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    due = {"id": "aaaaaaaaaaaa", "name": "old", "prompt": "old", "schedule": {"kind": "interval", "minutes": 1,
           "display": "every 1m"}, "repeat": {"times": None, "completed": 0}, "enabled": True, "state": "scheduled",
           "next_run_at": "2026-01-01T00:00:00+00:00", "deliver": "local"}
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [due]}))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "HERMES_HOME": str(home), "TMPDIR": str(tmp_path)}
    code = ("import json; from cron import jobs\nwith jobs.use_cron_store(" + repr(str(home)) + "):\n"
            "    print(json.dumps([[j['id'] for j in jobs.get_due_jobs()],"
            " [jobs.is_job_runnable(j) for j in jobs.load_jobs()]]))")
    assert json.loads(host_python(code, env, tmp_path))[1] == [True]  # The host would run it.
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [due]}))
    r.LocalHost({}, tmp_path).pause_cron({"home": home})
    assert json.loads(host_python(code, env, tmp_path)) == [[], [False]]


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
            " e = importlib.import_module('hermes_lcm_x.engine'); x = e.LCMEngine(config=c.LCMConfig.from_env(),"
            f" hermes_home={str(home)!r}); x.on_session_start('S', platform='d5-local'); x._ingest_messages(["
            "{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'hi'}]); sys.stdout.flush(); os._exit(0)")
    host_python(make, env, tmp_path)
    before = {p.relative_to(home): p.stat().st_mtime_ns for p in home.rglob("*") if p.is_file()}
    ctx = {"home": home, "gateway": type("G", (), {"env": env})(), "session_ids": ["S"],
           "tree": {"host_bin": os.environ["D5_HOST_BIN"], "lcm_tree": str(tree)}}
    out = r.LocalHost({}, tmp_path).probe(ctx)
    checks, skipped = r.probe_checks(out, [KEY])
    assert {p.relative_to(home): p.stat().st_mtime_ns for p in home.rglob("*") if p.is_file()} == before
    assert {c["kind"] for c in checks} == {"doctor", "schema"} and bars.failed_checks(checks) == []
    names = {c["check"] for c in checks}
    assert "schema_core_tables" in names and {f"{c}@{KEY}" for c in r.DOCTOR_PER_SESSION} <= names
    assert {s["check"] for s in skipped} == set(r.DOCTOR_NOT_EXERCISED) - {"context_pressure"}  # Not emitted here.
    scoped = json.loads((out / f"doctor-{KEY}.json").read_text())["checks"]
    assert next(c for c in scoped if c["check"] == "fts_index_sync")["detail"].startswith("2 session FTS rows")
    assert set(r.DOCTOR_PRE_OPEN) <= names
    damaged = tmp_path / "arm2" / "home"  # A second arm (each arm probes once), with FTS damage that opening repairs.
    damaged.parent.mkdir()
    r.capture.clone_tree(home, damaged)
    with closing(sqlite3.connect(damaged / "lcm.db")) as db:
        db.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
        db.commit()
    checks, _ = r.probe_checks(r.LocalHost({}, tmp_path).probe({**ctx, "home": damaged}), [KEY])
    assert bars.failed_checks(checks) == ["messages_fts_integrity@pre_open"]


def test_a_store_the_read_only_open_refuses_is_never_read_as_empty(tmp_path):
    db = tmp_path / "lcm.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE messages (store_id INTEGER)")
    Path(str(db) + "-shm").unlink(missing_ok=True)
    Path(str(db) + "-wal").write_bytes(b"")
    with pytest.raises(r.capture.CaptureError):
        r._rows(db, "SELECT max(store_id) FROM messages")
