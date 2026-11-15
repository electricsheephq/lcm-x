"""Item 3: recovery beside a warm foreground engine on its compression child.

Engine-level tests with the #2624 stub host. Strict xfails accept only the
known opt-in duplicate shape; the shipped default schedules no recovery.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sqlite3
from collections import Counter

import pytest

from hermes_lcm import host_row_recovery as recovery
from tests import test_issue_2624_host_row_recovery as recovery_helpers
from hermes_lcm.engine import LCMEngine
from tests import test_issue_519_replaced_tail as rotation_helpers
from tests.test_issue_2624_host_row_recovery_review import _run_in_recovery_thread
from tests.test_issue_519_replaced_tail import C, P

_make_case = recovery_helpers._make_case
pair, rows, tools = recovery_helpers.pair, recovery_helpers.rows, recovery_helpers.tools


@pytest.fixture
def live_child_case(tmp_path, monkeypatch, caplog):
    engines = []
    case = _make_case(tmp_path, monkeypatch, caplog, engines)
    yield case
    for engine in engines:
        engine._close_storage()


def _identity(message):
    """Compare payload and tool grouping, independently of product matching."""
    calls = message.get("tool_calls") or []
    if isinstance(calls, str):
        calls = json.loads(calls)
    return (message["role"], message.get("content") or "",
            json.dumps(calls, sort_keys=True), message.get("tool_call_id") or "")


VARIANTS = (
    "V1-unstamped", "V2-anchor-off", "V3-prematch-error", "V4-rewrite",
    "C0-control", "V5-foreground-first", "V6-read-write-race",
)


class LiveChildDuplicate(AssertionError):
    """Every host row is stored, with at least one occurrence stored more than once."""


@pytest.mark.parametrize("variant", [
    pytest.param(variant, marks=pytest.mark.xfail(
        strict=True, raises=LiveChildDuplicate,
        reason="#2624: recovery beside a warm child duplicates host occurrences",
    ) if variant.startswith(("V1-", "V2-", "V3-", "V4-")) else ())
    for variant in VARIANTS
])
@pytest.mark.parametrize("shape", ("tools", "pair"))
@pytest.mark.parametrize("finish", ("ingest", "session-end"))
def test_live_child_exactly_once(live_child_case, monkeypatch, record_property, caplog,
                                 variant, shape, finish):
    case = live_child_case
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "0" if variant == "V2-anchor-off" else "1")
    # Capture P's job explicitly: G's child start also queues an unrelated job.
    root_job = case.jobs[0]
    if variant == "V1-unstamped":
        for message in case.host.lists[C]:
            message.pop("timestamp", None)

    # These are the public binding/start calls made by the host. case.start()
    # constructs a separate engine, bind_session_state(host, P), then starts P.
    g = case.start()
    g.on_session_start(C, boundary_reason="compression", old_session_id=P,
                       platform="acp")
    assert (g._session_id, g._conversation_id) == (C, P)
    assert g._store is not case.live._store
    assert g._store.connection is not case.live._store.connection
    initial = copy.deepcopy(case.host.lists[C][len(case.prefix):])
    assert len(initial) == 2
    g.ingest(copy.deepcopy(case.host.lists[C]))
    assert g._ingest_cursor == len(case.host.lists[C]) > 0
    assert not g._ingest_cursor_needs_reconcile
    assert Counter(map(_identity, rows(case))) == Counter(map(_identity, initial))
    warm_cursor = g._ingest_cursor

    # D resumes P while G stays warm on C. The harness stub reports tip P;
    # model reopen_session(P)'s cleared end markers in its synthetic host DB.
    with sqlite3.connect(case.state) as conn:
        conn.execute("UPDATE sessions SET ended_at=NULL, end_reason=NULL WHERE id=?", (P,))
    case.live.on_session_start(P, platform="acp", context_length=200_000)
    assert case.live._lifecycle.get_by_conversation(P).current_session_id == P
    assert (g._session_id, g._conversation_id, g._ingest_cursor) == (C, P, warm_cursor)

    suffix = copy.deepcopy(tools() if shape == "tools" else pair())
    for index, message in enumerate(suffix):
        if message.get("content"):
            message["content"] += " / item3 new suffix"
        if variant == "V1-unstamped":
            message.pop("timestamp", None)
        else:
            message["timestamp"] = 1900000100 + index
    case.host.lists[C] += suffix
    expected_host = initial + suffix
    assert len(set(map(_identity, expected_host))) == len(expected_host)
    assert g._ingest_cursor == warm_cursor

    def foreground():
        history = copy.deepcopy(case.host.lists[C])
        if finish == "ingest":
            g.ingest(history)
        else:
            g.on_session_end(C, history)

    prematch_faults = []
    if variant == "V3-prematch-error":
        def fail_prematch(*args, **kwargs):
            prematch_faults.append(True)
            raise RuntimeError("item3 injected identity-anchor pre-match fault")
        monkeypatch.setattr(g, "_identity_anchor_prematch", fail_prematch)

    before = recovery.host_row_recovery_counts()
    if variant == "V5-foreground-first":
        foreground()
    elif variant == "V6-read-write-race":
        strict_proof_cursor = recovery._strict_proof_cursor
        raced_at = []

        def foreground_after_read(detached, messages):
            cursor = strict_proof_cursor(detached, messages)
            if not raced_at and not detached._store.connection.in_transaction:
                assert cursor == warm_cursor
                raced_at.append(cursor)
                foreground()
            return cursor

        monkeypatch.setattr(recovery, "_strict_proof_cursor", foreground_after_read)

    if variant == "V6-read-write-race":
        # case.run's all-plugin log privacy check assumes no foreground ingest
        # in its capture window. Use its identical worker runner for this extra
        # concurrent-writer control: foreground logs intentionally contain C.
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            _run_in_recovery_thread(root_job)
        warnings = [record.getMessage() for record in caplog.records
                    if record.levelno == logging.WARNING and record.name == recovery.__name__]
    else:
        warnings = case.run(root_job)
    after_recovery = recovery.host_row_recovery_counts()
    counts = {key: after_recovery[key] - before[key] for key in after_recovery}
    recovered_snapshot = copy.deepcopy(rows(case))
    if variant == "V4-rewrite":
        # Same occurrence stamp, changed bytes after recovery's snapshot/write.
        # Both the recovered old bytes and the current host bytes are audited.
        case.host.lists[C][-1]["content"] += " / host rewrite at same stamp"
        assert case.host.lists[C][-1]["timestamp"] == suffix[-1]["timestamp"]
        expected_host = initial + case.host.lists[C][warm_cursor:]
    if variant not in ("V5-foreground-first", "V6-read-write-race"):
        foreground()

    stored = rows(case)
    actual = Counter(map(_identity, stored))
    expected = Counter(map(_identity, expected_host))
    missing = sum((expected - actual).values())
    exact_payload_extras = sum((actual - expected).values())
    if variant == "V1-unstamped":
        occurrence_counts = actual
    else:
        occurrence_counts = Counter(row["observed_at"] for row in stored)
    duplicate_rows = sum(max(0, n - 1) for n in occurrence_counts.values())
    marked = [row for row in stored if case.live._store.read_metadata_json(
        "host_row_recovered:" + str(row["store_id"]))]
    measurement = dict(
        variant=variant, shape=shape, finish=finish,
        stored_rows=len(stored), expected_rows=len(expected_host),
        recovered_rows=counts["recovered_rows"], recovery_marks=len(marked),
        duplicate=bool(duplicate_rows), duplicate_rows=duplicate_rows,
        missing_rows=missing, extra_payload_rows=exact_payload_extras,
        raced=counts["raced"], empty_suffix=counts["empty_suffix"],
        proof_rejections=counts["proof"], warm_cursor=warm_cursor,
        recovery_snapshot_rows=len(recovered_snapshot),
        final_cursor=g._ingest_cursor, prematch_faults=len(prematch_faults),
        warnings=len(warnings),
        process_nice=os.getpriority(os.PRIO_PROCESS, 0),
        verdict="green" if actual == expected and not duplicate_rows else "red",
    )
    record_property("item3_measurement", json.dumps(measurement, sort_keys=True))
    print("ITEM3_MEASUREMENT " + json.dumps(measurement, sort_keys=True), flush=True)

    # Audit the setup and the worker separately from the exactly-once promise.
    assert all(row["session_id"] == C and row["conversation_id"] == P for row in stored)
    assert len(marked) == counts["recovered_rows"]
    assert counts["worker_error"] == 0
    if variant == "V3-prematch-error":
        assert prematch_faults, "the named pre-match fault must actually fire"
    if variant == "V6-read-write-race":
        assert raced_at, "foreground must run after recovery's proof read"

    # Preserve the original assertions; only the named duplicate shape may xfail.
    assert missing == 0, measurement
    assert Counter(map(_identity, recovered_snapshot)) <= actual, measurement
    if duplicate_rows and not missing:
        raise LiveChildDuplicate(str(measurement))
    assert len(stored) == len(expected_host), measurement
    assert duplicate_rows == 0, measurement
    assert actual == expected, measurement
    assert list(map(_identity, stored)) == list(map(_identity, expected_host)), measurement


def test_live_child_default_off_matches_v0300(tmp_path, monkeypatch):
    """V1's warm unstamped child at the shipped default stores its suffix once."""
    config_factory = rotation_helpers._config
    monkeypatch.setattr(rotation_helpers, "_config", lambda path, **kw: config_factory(
        path, temporal_rollups_enabled=False, embeddings_enabled=False, **kw))
    jobs = []
    monkeypatch.setattr(recovery._HOST_ROW_RECOVERY_SCHEDULER, "schedule",
                        lambda key, job: key[0].startswith("unended-catchup:") or jobs.append(job) or True)
    monkeypatch.setattr(recovery, "_SEEN", set())
    monkeypatch.setattr(recovery, "_WARNED", set())
    before = recovery.host_row_recovery_counts()
    engines = []
    try:
        rotation = rotation_helpers._Rotation(tmp_path, monkeypatch)
        engines.append(rotation.engine)
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        with sqlite3.connect(home / "state.db") as conn:
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, ended_at REAL, "
                         "end_reason TEXT, model_config TEXT, source TEXT)")
            conn.executemany("INSERT INTO sessions VALUES (?,?,?,?,?,?)", [
                (P, None, 1, "compression", None, "acp"), (C, P, None, None, None, "acp")])
        prefix = rotation.restored()
        host = recovery_helpers.Host({C: prefix + pair(stamps=False)})
        for message in host.lists[C]:
            message.pop("timestamp", None)

        def start():
            engine = LCMEngine(config=copy.deepcopy(rotation.engine._config), hermes_home=str(home))
            engines.append(engine)
            engine.bind_session_state(host, P)
            engine.on_session_start(P, platform="acp", context_length=200_000)
            return engine

        rotation.engine._close_storage()
        g = start()
        g.on_session_start(C, boundary_reason="compression", old_session_id=P, platform="acp")
        initial = copy.deepcopy(host.lists[C][len(prefix):])
        g.ingest(copy.deepcopy(host.lists[C]))
        warm_cursor = g._ingest_cursor
        assert warm_cursor == len(host.lists[C]) > 0
        assert not g._ingest_cursor_needs_reconcile
        assert Counter(map(_identity, g._store.get_session_messages(C))) == Counter(map(_identity, initial))
        with sqlite3.connect(home / "state.db") as conn:
            conn.execute("UPDATE sessions SET ended_at=NULL, end_reason=NULL WHERE id=?", (P,))
        resumed = start()
        assert (g._session_id, g._conversation_id, g._ingest_cursor) == (C, P, warm_cursor)
        assert g._store.connection is not resumed._store.connection
        suffix = pair(stamps=False)
        for message in suffix:
            message["content"] += " / item3 new suffix"
        host.lists[C] += suffix
        assert not jobs, "default-off recovery must schedule no job"
        assert recovery.host_row_recovery_counts() == before
        g.ingest(copy.deepcopy(host.lists[C]))
        stored = g._store.get_session_messages(C)
        assert list(map(_identity, stored)) == list(map(_identity, initial + suffix))
        assert Counter(map(_identity, stored)) == Counter(map(_identity, initial + suffix))
        assert recovery.host_row_recovery_counts() == before
    finally:
        for engine in engines:
            engine._unregister_active_engine_binding()
            engine._close_storage()
