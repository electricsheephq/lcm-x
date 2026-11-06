"""P1 store-wide oracle controls and real process-death boundaries (I1/I2/I5)."""

import copy
import os
import sqlite3
import subprocess
import sys

import pytest

from tests.p1_gate import GateFailure, expect_from_view, gate_result, record, store_outcome
from tests.test_host_uid_shadow import _engine, _m, _state_db
from tests.test_issue_2055_ingest_transaction import durable, _no_background_maintenance as _no_background_maintenance
from tests.test_issue_2055_ingest_ledger_writes import engines as engines


@pytest.mark.parametrize("shape,relations,wanted", [
    ("duplicate", [], (0, 0, 1)),
    ("duplicate", [(2, "supersedes", 1)], (0, 1, 0)),
    ("duplicate", [(2, "possible_replay_of", 1)], (0, 1, 0)),
    ("duplicate", [(1, "possible_replay_of", 2)], (0, 1, 0)),
    ("foreign_pair", [(2, "possible_replay_of", 1)], (0, 0, 1)),
    ("duplicate", [(2, "composite", 1)], (0, 0, 1)),
    ("loss", [], (1, 0, 0)),
    ("foreign", [], (0, 0, 1)),
])
def test_store_outcome_controls(shape, relations, wanted):
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE messages(store_id,session_id,conversation_id,role,content,tool_call_id,tool_calls)")
        conn.execute("CREATE TABLE message_relations(store_id,kind,related_store_id)")
        expected = expect_from_view([_m("assistant", "same")], "conv")
        rows = [(1, "S", "conv", "assistant", "same", None, None)]
        if shape == "loss":
            rows = []
        else:
            rows.append((2, "other", "foreign" if shape.startswith("foreign") else "conv",
                         "assistant", "same", None, None))
        conn.executemany("INSERT INTO messages VALUES (?,?,?,?,?,?,?)", rows)
        conn.executemany("INSERT INTO message_relations VALUES (?,?,?)", relations)
        outcome = store_outcome(conn, expected)
        assert tuple(outcome[key] for key in ("loss", "visible", "silent")) == wanted
        assert outcome["silent_store_ids"] == ([2] if wanted[2] else [])
        # Each shape has a negative control: an unlisted expected row must be detected.
        expected["conv"].append(("user", "absent control", None, None))
        assert store_outcome(conn, expected)["loss"] == wanted[0] + 1


def test_real_recovery_duplicate_is_detected(engines, monkeypatch):
    engine = engines()
    view = [_m("user", "seed", 1.), _m("assistant", "reply", 2.)]
    engine.ingest(copy.deepcopy(view))
    ingest = engine._ingest_messages

    def recovery(rows, **kwargs):
        result = ingest(rows, **kwargs)
        engine._store.append("S", rows[-1], source="cli", conversation_id="conv")
        return result

    monkeypatch.setattr(engine, "_ingest_messages", recovery)
    engine.ingest(copy.deepcopy(view))
    outcome = store_outcome(engine._store._conn, expect_from_view(view, "conv"))
    assert outcome["loss"] == 0 and outcome["silent"] == 1
    with pytest.raises(AssertionError):
        assert outcome["loss"] == outcome["silent"] == 0, outcome


PROCESS = r'''
import os, pathlib, sys
import pytest
import tests.conftest
from tests.harness import pin_char_counter
from tests.test_host_uid_shadow import _engine, _m
from tests.p1_gate import install_former_commit_faults
root, boundary = pathlib.Path(sys.argv[1]), sys.argv[2]
patch = pytest.MonkeyPatch()
pin_char_counter(patch)
from hermes_lcm.engine import LCMEngine
for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
    patch.setattr(LCMEngine, name, lambda *args: None)
engine = _engine(root)
prefix = [_m("user", "seed", uid="seed-uid"), _m("assistant", "seed reply", 11., "reply-uid")]
seed = engine._store.get_session_messages("S")[0]["store_id"]
host = [{**prefix[0], "timestamp": 10.}, prefix[1], _m("user", "fresh", 20., "fresh-uid")]
if boundary == "m":
    engine.ingest(host[:2])
    host = [host[0], host[2], host[1]]
engine._identity_anchor_recovery = {"row": host[0]}
install_former_commit_faults(engine, setattr, boundary, lambda point: os._exit(97) if point == boundary else None)
engine._ingest_messages(host)
os._exit(0)
'''


M_XFAIL = pytest.mark.xfail(
    strict=True, raises=GateFailure,
    reason="#2305: the cold bind after a mid-history insert re-stores unstamped rows as silent surplus "
           "(the no-crash control shows the same rows; P1 gate blocker)")
# #2305's silent surplus per m shape; loss is never expected, and a different count is a new failure.
M_SILENT = {"insert": 7, "field_chain_replace": 9}


@pytest.mark.parametrize("boundary,shape", [
    *((point, None) for point in ("b", "c", "d", "e", "f", "INSERT", "h", "i", "j", "k", "l")),
    pytest.param("m", "insert", marks=M_XFAIL, id="m-insert"),
    pytest.param("m", "field_chain_replace", marks=M_XFAIL, id="m-field-chain-replace"),
])
def test_former_commit_process_death(tmp_path, monkeypatch, boundary, shape):
    if boundary == "m":
        _rekey_process_death(tmp_path, shape)
        return
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "on")
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path)
    prefix = [_m("user", "seed", uid="seed-uid"), _m("assistant", "seed reply", 11., "reply-uid")]
    engine.ingest(prefix)
    host = [{**prefix[0], "timestamp": 10.}, prefix[1], _m("user", "fresh", 20., "fresh-uid")]
    if boundary == "m":
        engine.ingest(host[:2])
        host = [host[0], host[2], host[1]]
    engine.shutdown()
    # Compare after binding, just as the child binds; no mutation may survive its uncommitted ingest.
    fresh = _engine(tmp_path)
    try:
        before = durable(fresh)
        env = {**os.environ, "LCM_INGEST_LEDGER": "shadow", "LCM_HOST_MESSAGE_UID": "shadow"}
        result = subprocess.run([sys.executable, "-c", PROCESS, str(tmp_path), boundary],
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 97, result.stderr
        assert durable(fresh) == before
        for _ in range(2):
            fresh.ingest(copy.deepcopy(host))
        assert [row["content"] for row in fresh._store.get_session_messages("S")] == ["seed", "seed reply", "fresh"]
        outcome = store_outcome(fresh._store._conn, expect_from_view(host, "conv"))
        record(dict(family="process_death", boundary=boundary), outcome, fresh)
        assert outcome["loss"] == outcome["silent"] == 0, outcome
    finally:
        fresh.shutdown()


def _rekey_process_death(tmp_path, shape):
    import json
    from tests.p1_gate import prepare_rekey_case

    root = tmp_path / "crash"
    root.mkdir()
    host, view, expected = prepare_rekey_case(root, shape)
    snapshot_path = tmp_path / "prepared.sqlite"
    with sqlite3.connect(snapshot_path) as snapshot:
        host.engine._store._conn.backup(snapshot)

    def restore_prepared():
        # Restore only this test-created DB; every branch uses the identical
        # Hermes home, so the proof's scope remains valid in the cold control.
        with sqlite3.connect(root / "lcm.db") as conn, sqlite3.connect(snapshot_path) as snapshot:
            snapshot.backup(conn)

    if shape == "field_chain_replace":
        warm = host.engine
        warm.ingest(copy.deepcopy(view))
        warm.ingest(copy.deepcopy(view))
        warm_outcome = store_outcome(warm._store._conn, expected)
        assert warm_outcome["loss"] == warm_outcome["silent"] == 0, warm_outcome
    host.close()
    restore_prepared()
    control = host.bind()
    host.engine = control
    try:
        control.ingest(copy.deepcopy(view))
        control.ingest(copy.deepcopy(view))
        control_rows = host.rows()
        control_outcome = store_outcome(control._store._conn, expected)
    finally:
        control.shutdown()
    restore_prepared()
    view_path = root / "view.json"
    view_path.write_text(json.dumps(view))
    fresh = host.bind()
    host.engine = fresh
    try:
        before = durable(fresh)
        script = r'''
import json, os, pathlib, sys
import pytest
import tests.conftest
from tests.harness import pin_char_counter
from tests.test_issue_1286_cleanup_rewrite_proof import Host
from tests.p1_gate import install_former_commit_faults
from hermes_lcm.engine import LCMEngine
patch = pytest.MonkeyPatch()
pin_char_counter(patch)
for name in ("_schedule_rollup_maintenance", "_schedule_embedding_maintenance"):
    patch.setattr(LCMEngine, name, lambda *args: None)
root = pathlib.Path(sys.argv[1])
host = Host(root)
view = json.loads((root / "view.json").read_text())
install_former_commit_faults(host.engine, setattr, "m", lambda point: os._exit(97))
host.engine._ingest_messages(view)
os._exit(0)
'''
        result = subprocess.run([sys.executable, "-c", script, str(root)], env=dict(os.environ),
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 97, result.stderr
        assert durable(fresh) == before
        fresh.ingest(view)
        fresh.ingest(view)
        # Crash adds zero rows: this differential must pass before the expected failure.
        assert host.rows() == control_rows
        outcome = store_outcome(fresh._store._conn, expected)
        assert outcome == control_outcome
        record(dict(family="process_death", boundary="m", shape=shape, strict_xfail=True), outcome, fresh)
        assert outcome["loss"] == 0, outcome  # Loss is never an expected failure.
        assert outcome["silent"] == M_SILENT[shape], outcome  # Only #2305's known rows.
        raise GateFailure(outcome)
    finally:
        fresh.shutdown()


def test_gate_summary_keeps_expected_failures_red():
    clean = dict(cell={}, outcome=dict(loss=0, silent=0))
    assert gate_result([clean]) == "PASS"
    assert gate_result([clean], strict_xfail_cells=["m"]) == "FAIL"
    assert gate_result([{**clean, "cell": {"strict_xfail": True}}]) == "FAIL"
    assert gate_result([{**clean, "result": "FAIL"}]) == "FAIL"
    for key in ("loss", "silent"):
        assert gate_result([{**clean, "outcome": {"loss": 0, "silent": 0, key: 1}}]) == "FAIL"
