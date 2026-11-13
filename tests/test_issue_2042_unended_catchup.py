"""#2042 step 1: forward-only, off-turn failed-turn catch-up and its proof limits."""
from __future__ import annotations

import copy
import json
import multiprocessing
import sqlite3
import sys
import traceback
from collections import Counter
from pathlib import Path

import pytest

# Spawned interpreters need the same package bootstrap as the pytest parent.
if "hermes_lcm" not in sys.modules:
    from tests import conftest as _bootstrap  # noqa: F401

from hermes_lcm import engine_registry, unended_catchup as catchup
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_issue_2042_unended_failed_turn import FAILED_TURN_NOTICE, _quiesce

MARKER = 1_900_000_000
BOUNDARY = MARKER + 100
NOW = BOUNDARY + 1000
SID = "failed-2042"


def failed_rows(stamped=True, *, boundary=BOUNDARY):
    return [dict(role=role, content=text, **({"timestamp": boundary - 1 + i} if stamped else {}),
                 **({"display_kind": "failed_turn"} if i else {}))
            for i, (role, text) in enumerate((("user", "synthetic failed request 2042"),
                                             ("assistant", FAILED_TURN_NOTICE)))]


class Host:
    """Separate SQL eligibility rows and host-reader dictionaries, as on Hermes."""

    def __init__(self, home, *, create=False, archive_columns=True):
        self.path = Path(home) / "state.db"
        if create:
            with sqlite3.connect(self.path) as conn:
                conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, parent_session_id TEXT, "
                             "started_at REAL, ended_at REAL, end_reason TEXT, model_config TEXT, "
                             "profile_name TEXT, session_key TEXT, transcript_revision INTEGER DEFAULT 0)")
                archive = ", compacted INTEGER DEFAULT 0, _compressed_summary INTEGER DEFAULT 0" if archive_columns else ""
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
                             "role TEXT, display_kind TEXT, active INTEGER DEFAULT 1, timestamp REAL, payload TEXT"
                             + archive + ")")

    def write(self, sid=SID, rows=None, *, boundary=BOUNDARY, **session):
        rows = failed_rows() if rows is None else rows
        values = dict(id=sid, source="desktop", parent_session_id=None, started_at=boundary - len(rows),
                      ended_at=None, end_reason=None, model_config=None, profile_name=None,
                      session_key=None, transcript_revision=0)
        values.update(session)
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)", tuple(values.values()))
            conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            conn.executemany("INSERT INTO messages (session_id,role,display_kind,timestamp,payload) VALUES (?,?,?,?,?)",
                             [(sid, row["role"], row.get("display_kind"), boundary - len(rows) + i + 1,
                               json.dumps(row)) for i, row in enumerate(rows)])

    def get_messages_as_conversation(self, session_id, *, include_ancestors=True, include_inactive=False,
                                     repair_alternation=True, include_row_ids=False, include_compacted=False):
        with sqlite3.connect(self.path) as conn:
            return [json.loads(row[0]) for row in conn.execute(
                "SELECT payload FROM messages WHERE session_id=? AND active=1 ORDER BY id", (session_id,))]

    def get_compression_tip(self, session_id):
        return session_id


class Case:
    def __init__(self, root, *, archive_columns=True, db_path=None):
        self.home = root / "home"
        self.home.mkdir(parents=True)
        self.host = Host(self.home, create=True, archive_columns=archive_columns)
        self.db_path = str((db_path or self.home / "lcm.db").resolve())
        self.config = LCMConfig(database_path=self.db_path, embeddings_enabled=False, temporal_rollups_enabled=False)
        self.engines = []
        self.engine = self.new_engine()
        self.engine._store  # Create the LCM schema without binding or activating catch-up.

    def new_engine(self):
        engine = LCMEngine(config=copy.deepcopy(self.config), hermes_home=str(self.home))
        self.engines.append(engine)
        return engine

    def bind(self, sid="next-2042", *, conversation_id=None):
        engine = self.new_engine()
        engine.bind_session_state(self.host, sid)
        engine.on_session_start(sid, hermes_home=str(self.home), platform="desktop", context_length=128000,
                                **({"conversation_id": conversation_id} if conversation_id else {}))
        _quiesce()
        return engine

    def run(self, *, bind_time=MARKER, profile=None):
        before = Counter(catchup.unended_catchup_counts())
        catchup._catch_up(LCMEngine, self.config, str(self.home.resolve()), self.db_path, str(self.host.path),
                         "next-2042", "desktop", 128000, self.host,
                         profile or self.engine._entity_profile_key(self.db_path), bind_time)
        delta = Counter(catchup.unended_catchup_counts()) - before
        assert not delta["worker_error"], delta
        return delta

    def rows(self, sid=SID):
        return self.engine._store.get_session_messages(sid)

    def assert_rows(self, expected, sid=SID):
        rows = self.rows(sid)
        assert [(r["role"], r["content"]) for r in rows] == [(r["role"], r["content"]) for r in expected]
        assert len({r["store_id"] for r in rows}) == len(expected)
        return rows

    def close(self):
        _quiesce()
        for engine in self.engines:
            engine._unregister_active_engine_binding()
            engine._close_storage()


@pytest.fixture(autouse=True)
def fresh_process_caches(monkeypatch):
    # Process caches are keyed by path; pytest may reuse a removed tmp directory name.
    monkeypatch.setattr(catchup, "_MARKERS", {})
    monkeypatch.setattr(catchup, "_SEEN", {})


@pytest.fixture
def case(tmp_path, monkeypatch):
    _quiesce()
    monkeypatch.setattr(catchup, "_now", lambda: NOW)
    case = Case(tmp_path)
    yield case
    case.close()


@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "unstamped"])
@pytest.mark.parametrize("different_conversation", [False, True], ids=["same-conversation", "different-conversation"])
def test_resume_after_catchup_stores_each_row_once(case, stamped, different_conversation):
    history = failed_rows(stamped)
    case.host.write(rows=history, session_key="original-conversation")
    assert case.run() == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}
    recovered = case.assert_rows(history)
    assert all(r["conversation_id"] == "original-conversation" for r in recovered)
    assert all(case.engine._store.read_metadata_json("unended_turn_recovered:" + str(r["store_id"]))
               == {"version": 1, "issue": 2042} for r in recovered)
    history += [dict(role="user", content="next request 2042", **({"timestamp": NOW} if stamped else {})),
                dict(role="assistant", content="next answer 2042", **({"timestamp": NOW + 1} if stamped else {}))]
    case.host.write(rows=history)
    live = case.bind(SID, conversation_id="new-conversation" if different_conversation else "original-conversation")
    live.ingest(copy.deepcopy(history))
    case.assert_rows(history)
    live.ingest(copy.deepcopy(history))
    case.assert_rows(history)
    assert case.run() == {}  # The completed tail is no longer a failed-turn candidate.


def test_shared_store_has_separate_activation_marker_for_each_home(case, tmp_path):
    assert case.run() == {}
    other = Case(tmp_path / "profile-b", db_path=Path(case.db_path))
    try:
        other.host.write()
        assert other.run(bind_time=BOUNDARY + 1) == {}
        assert other.rows() == []
        other.host.write("after-b-marker", boundary=BOUNDARY + 2)
        assert other.run() == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}
        other.assert_rows(failed_rows(), "after-b-marker")
        assert other.rows() == []
        with sqlite3.connect(case.db_path) as conn:
            markers = conn.execute("SELECT value FROM metadata WHERE key LIKE 'unended_catchup_since:%'").fetchall()
        assert sorted(float(r[0]) for r in markers) == [MARKER, BOUNDARY + 1]
    finally:
        other.close()


def tool_rows(results=2):
    return [failed_rows()[0], {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call-2042-" + str(i), "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}} for i in range(2)]},
        *[{"role": "tool", "content": "synthetic result " + str(i), "tool_call_id": "call-2042-" + str(i)}
          for i in range(results)], failed_rows()[1]]


@pytest.mark.parametrize("results", [0, 1, 2])
def test_complete_tool_groups_required(case, results):
    rows = tool_rows(results)
    case.host.write(rows=rows)
    if results == 2:
        assert case.run() == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": len(rows)}
        stored = case.assert_rows(rows)
        actual = [case.engine._store.to_openai_msg(r) for r in stored]
        assert [r.get("tool_calls") for r in actual] == [r.get("tool_calls") for r in rows]
        assert [r.get("tool_call_id") for r in actual] == [r.get("tool_call_id") for r in rows]
    else:
        assert case.run() == {"candidates": 1, "incomplete_tool_group": 1}
        assert case.rows() == []


@pytest.mark.parametrize("mutation", ["archive-and-carry", "in-place-rewrite", "reader-count", "reader-boundary"])
def test_host_generation_must_match_before_and_after_reader(case, monkeypatch, mutation):
    case.host.write()
    original = case.host.get_messages_as_conversation

    def changed(session_id):
        with sqlite3.connect(case.host.path) as conn:
            if mutation == "archive-and-carry":
                conn.execute("UPDATE messages SET active=0, compacted=1 WHERE session_id=?", (session_id,))
                conn.execute("INSERT INTO messages (session_id,role,timestamp,payload) VALUES (?,?,?,?)",
                             (session_id, "assistant", BOUNDARY, json.dumps({"role": "assistant", "content": "summary"})))
            elif mutation == "in-place-rewrite":
                row = {**failed_rows()[0], "content": "edited synthetic request"}
                conn.execute("UPDATE messages SET payload=? WHERE session_id=? AND role='user'",
                             (json.dumps(row), session_id))
                conn.execute("UPDATE sessions SET transcript_revision=transcript_revision+1 WHERE id=?", (session_id,))
        rows = original(session_id)
        if mutation == "reader-count":
            rows = rows[1:]
        if mutation == "reader-boundary":
            rows[-1].pop("display_kind")
        return rows

    monkeypatch.setattr(case.host, "get_messages_as_conversation", changed)
    assert case.run() == {"candidates": 1, "generation_changed": 1}
    assert case.rows() == []


def test_host_without_archive_columns_still_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(catchup, "_now", lambda: NOW)
    case = Case(tmp_path, archive_columns=False)
    try:
        case.host.write()
        assert case.run() == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}
        case.assert_rows(failed_rows())
    finally:
        case.close()


@pytest.mark.parametrize("role", ["system", "session_meta"])
@pytest.mark.parametrize("position", ["before", "after"])
def test_non_turn_roles_skip_whole_session(case, role, position):
    rows = failed_rows()
    rows.insert(0 if position == "before" else len(rows), {"role": role, "content": "synthetic metadata"})
    case.host.write(rows=rows)
    assert case.run() == {"candidates": 1, "non_turn_role": 1}
    assert case.rows() == []


@pytest.mark.parametrize("shape,reason", [
    ("before-marker", None), ("too-recent", None), ("too-old", None), ("child", None),
    ("subagent", None), ("fork-null-parent", "fork_child"), ("compression", None),
    ("other-profile", None), ("ignored", "session_filtered"), ("stateless", "session_filtered"),
    ("existing", "existing_rows"), ("live-bound", "live_bound"), ("compacted", None),
    ("compressed-summary", None), ("inactive", "generation_changed"), ("ignored-message", "ignored_in_suffix"),
])
def test_ineligible_sessions_are_never_stored(case, shape, reason, monkeypatch):
    assert case.run() == {}  # Activate before any bind can write a later marker.
    boundary, session = BOUNDARY, {}
    if shape == "before-marker":
        boundary = MARKER - 1
    elif shape == "too-recent":
        boundary = NOW - 599
    elif shape == "too-old":
        boundary = NOW - 7 * 86400 - 1
    elif shape == "child":
        session["parent_session_id"] = "parent-2042"
    elif shape == "subagent":
        session["source"] = "subagent"
    elif shape == "fork-null-parent":
        session["model_config"] = json.dumps({"_branched_from": "parent-2042"})
    elif shape == "compression":
        session.update(ended_at=BOUNDARY, end_reason="compression")
    elif shape == "other-profile":
        session["profile_name"] = "a-different-profile"
    elif shape == "ignored":
        case.config.ignore_session_patterns = [SID]
    elif shape == "stateless":
        case.config.stateless_session_patterns = [SID]
    elif shape == "ignored-message":
        # Exercise the gate without the optional regex dependency (as the #2624 tests do).
        monkeypatch.setattr(LCMEngine, "_matches_ignore_message_patterns",
                            lambda det, m, **kw: m.get("content") == "synthetic failed request 2042")
    case.host.write(boundary=boundary, **session)
    if shape in ("existing", "live-bound"):
        live = case.bind(SID)
        if shape == "existing":
            live.ingest([{"role": "user", "content": "existing 2042"}, {"role": "assistant", "content": "ok"}])
    if shape in ("compacted", "compressed-summary", "inactive"):
        column = {"compacted": "compacted", "compressed-summary": "_compressed_summary", "inactive": "active"}[shape]
        with sqlite3.connect(case.host.path) as conn:
            conn.execute(f"UPDATE messages SET {column}=? WHERE session_id=? AND role='user'",
                         (0 if shape == "inactive" else 1, SID))
    before = case.rows()
    assert case.run() == ({"candidates": 1, reason: 1} if reason else {})
    assert case.rows() == before


@pytest.mark.parametrize("boundary", [MARKER, NOW - 600, NOW - 7 * 86400])
def test_time_limits_are_inclusive(case, boundary):
    case.host.write(boundary=boundary)
    assert case.run(bind_time=min(MARKER, boundary)) == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}
    case.assert_rows(failed_rows())


@pytest.mark.parametrize("nrows", [500, 501])
def test_row_limit(case, nrows):
    if nrows == 500:
        rows = [{"role": role, "content": f"{role} row {i}"} for i in range(250) for role in ("user", "assistant")]
    else:
        rows = tool_rows(1)[:-1] + [
            {"role": role, "content": f"{role} row {i}"} for i in range(249) for role in ("user", "assistant")]
        rows[1]["tool_calls"] = rows[1]["tool_calls"][:1]
    assert len(rows) == nrows
    rows[-1] = failed_rows()[1]
    case.host.write(rows=rows)
    if nrows == 500:
        assert case.run() == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 500}
        case.assert_rows(rows)
    else:
        assert case.run() == {"candidates": 1, "suffix_rejected": 1}
        assert case.rows() == []


def test_job_stores_at_most_sixteen_and_query_reads_newest_sixty_four(case):
    for i in range(70):
        case.host.write(f"candidate-{i:02}", boundary=BOUNDARY + i)
    for batch in range(4):
        assert case.run() == Counter({"candidates": 16 * (batch + 1), "existing_rows": 16 * batch,
                                       "recovered_candidates": 16, "recovered_rows": 32})
        for i in range(70):
            case.assert_rows(failed_rows() if i >= 70 - 16 * (batch + 1) else [], f"candidate-{i:02}")
    assert case.run() == {"candidates": 64, "existing_rows": 64}
    with sqlite3.connect(case.db_path) as conn:
        ids = [r[0] for r in conn.execute("SELECT session_id FROM messages ORDER BY store_id")]
    assert ids == [f"candidate-{i:02}" for i in range(69, 5, -1) for _ in range(2)]


def test_live_conversation_lifecycle_registry_and_frontier_unchanged(case):
    assert case.run() == {}
    live = case.bind("live-2042", conversation_id="shared-conversation")
    live.ingest([{"role": "user", "content": "live question"}, {"role": "assistant", "content": "live answer"}])
    state = copy.deepcopy(live._lifecycle.get_by_conversation("shared-conversation"))
    frontier = live._store_complete_frontier()
    registry = engine_registry.resolve_active_lcm_engine(session_id="live-2042", conversation_id="shared-conversation")
    live_rows = case.rows("live-2042")
    case.host.write(session_key="shared-conversation")
    assert case.run() == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}
    case.assert_rows(failed_rows())
    assert live._lifecycle.get_by_conversation("shared-conversation") == state
    assert live._store_complete_frontier() == frontier
    assert case.rows("live-2042") == live_rows
    assert engine_registry.resolve_active_lcm_engine(
        session_id="live-2042", conversation_id="shared-conversation") is registry is live
    assert SID not in engine_registry.live_bound_session_ids()


def test_ten_minute_enqueue_cadence_and_disabled_gate(case, monkeypatch):
    jobs, times = [], [MARKER]
    monkeypatch.setattr(catchup, "_now", lambda: times[0])
    monkeypatch.setattr(catchup._HOST_ROW_RECOVERY_SCHEDULER, "schedule", lambda key, job: jobs.append((key, job)) or True)
    # Suppress the separate #2624 enqueue so only this mixin's cadence is observed.
    monkeypatch.setattr(LCMEngine, "_enqueue_host_row_recovery", lambda *args: None)
    live = case.bind()
    assert len(jobs) == 1
    assert case.engine._store.connection.execute(
        "SELECT count(*) FROM metadata WHERE key LIKE 'unended_catchup_since:%'").fetchone()[0] == 0
    jobs[0][1]()  # Only the off-turn job writes the marker.
    case.host.write()
    before = Counter(catchup.unended_catchup_counts())
    times[0] = MARKER + 599
    live._enqueue_unended_catchup("next-2042", {})
    assert len(jobs) == 1
    times[0] = NOW
    live._config.host_row_recovery_enabled = False
    live._enqueue_unended_catchup("next-2042", {})
    assert len(jobs) == 1
    assert Counter(catchup.unended_catchup_counts()) == before
    assert case.rows() == []
    live._config.host_row_recovery_enabled = True
    times[0] = MARKER + 600
    live._enqueue_unended_catchup("next-2042", {})
    assert len(jobs) == 2 and jobs[0][0] == jobs[1][0]
    # At exactly 600 seconds the failure is still too recent; no delayed retry is scheduled.
    jobs[1][1]()
    assert Counter(catchup.unended_catchup_counts()) == before
    assert case.rows() == []
    times[0] = NOW
    live._enqueue_unended_catchup("next-2042", {})
    assert len(jobs) == 2
    times[0] = MARKER + 1200
    live._enqueue_unended_catchup("next-2042", {})
    assert len(jobs) == 3
    jobs[-1][1]()
    assert Counter(catchup.unended_catchup_counts()) - before == {
        "candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}
    case.assert_rows(failed_rows())


def _spawn_catchup(config, home, db_path, barrier, results):
    original = catchup.protect_messages_for_ingest
    try:
        catchup._now = lambda: NOW

        def synchronized(*args, **kwargs):
            protected = original(*args, **kwargs)
            barrier.wait(timeout=30)  # Both observations said empty, before either BEGIN IMMEDIATE.
            return protected

        catchup.protect_messages_for_ingest = synchronized
        before = Counter(catchup.unended_catchup_counts())
        catchup._catch_up(LCMEngine, config, home, db_path, str(Path(home) / "state.db"),
                         "next-2042", "desktop", 128000, Host(home), home, MARKER)
        results.put({"delta": dict(Counter(catchup.unended_catchup_counts()) - before)})
    except BaseException:
        results.put({"error": traceback.format_exc()})
    finally:
        catchup.protect_messages_for_ingest = original


def _join_processes(processes):
    for process in processes:
        process.join(40)
        if process.is_alive():
            process.terminate()
            process.join(10)
            pytest.fail("spawned worker did not finish")
        assert process.exitcode == 0


def test_two_spawned_processes_store_one_copy_and_count_one_race(case):
    case.host.write()
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_spawn_catchup, args=(
        case.config, str(case.home.resolve()), case.db_path, barrier, results)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        observations = [results.get(timeout=40) for _ in processes]
        _join_processes(processes)
        assert all("error" not in result for result in observations), observations
        deltas = [Counter(result["delta"]) for result in observations]
        assert sorted(deltas, key=lambda d: d["raced"]) == [
            {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}, {"candidates": 1, "raced": 1}]
        case.assert_rows(failed_rows())
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
        results.close()
        results.join_thread()


def _spawn_live_engine(config, home, pipe):
    live = None
    try:
        catchup._now = lambda: NOW
        live = LCMEngine(config=config, hermes_home=home)
        live.bind_session_state(Host(home), SID)
        live.on_session_start(SID, hermes_home=home, platform="desktop", context_length=128000)
        _quiesce()
        pipe.send({"cursor": live._ingest_cursor, "reconcile": live._ingest_cursor_needs_reconcile})
        assert pipe.poll(40), "parent did not deliver the next completed turn"
        live.ingest(pipe.recv())
        pipe.send({"stored": [(r["role"], r["content"]) for r in live._store.get_session_messages(SID)]})
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
    finally:
        if live is not None:
            _quiesce()
            live._unregister_active_engine_binding()
            live._close_storage()
        pipe.close()


@pytest.mark.parametrize("stamped", [
    pytest.param(True, id="stamped"),
    pytest.param(False, id="unstamped", marks=pytest.mark.xfail(
        strict=True, reason="#2042 step 2: ingest guard (after #2293)")),
])
def test_step2_live_engine_in_another_process_bound_before_catchup(case, stamped):
    assert case.run() == {}  # The profile was activated before this failure.
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_spawn_live_engine, args=(case.config, str(case.home.resolve()), child))
    try:
        process.start()
        child.close()
        assert parent.poll(40), "live engine did not bind"
        ready = parent.recv()
        assert ready == {"cursor": 0, "reconcile": False}, ready
        history = failed_rows(stamped)
        case.host.write(rows=history)
        assert case.run() == {"candidates": 1, "recovered_candidates": 1, "recovered_rows": 2}
        case.assert_rows(history)
        # The closer appends the boundary to the live agent's list before its next completed turn.
        history += [{"role": "user", "content": "next live request 2042"},
                    {"role": "assistant", "content": "next live answer 2042"}]
        if stamped:
            for i, row in enumerate(history[2:]):
                row["timestamp"] = NOW + i
        parent.send(history)
        assert parent.poll(40), "live ingest did not finish"
        result = parent.recv()
        _join_processes([process])
        assert "error" not in result, result
        assert result["stored"] == [(r["role"], r["content"]) for r in history]
        case.assert_rows(history)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
