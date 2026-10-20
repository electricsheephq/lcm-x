"""#1179 R3: retained occurrences, durable proofs and host metadata failures."""
from copy import deepcopy

import pytest

pytest.importorskip("hypothesis")
from tests.invariants.test_exactly_once_machine import ExactlyOnceMachine, assert_occurrences
from tests.test_compression_boundary import _host_merge_consecutive_users
from tests.test_host_uid_shadow import _RaisingUidDict


@pytest.fixture
def machine():
    machine = ExactlyOnceMachine()
    yield machine
    machine.teardown()


def _compact(machine):
    for text in ("alpha", "same text"):
        machine.new_turn(text, False, 0, False, "lookup")
    machine.compress()
    assert machine.engine._last_compression_status == "compacted"
    assert machine.engine._compress_commit_proof
    return deepcopy(machine.host)


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("compacted", [False, True])
def test_r3_1_retained_suffix(machine, anchor, compacted):
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    if compacted:
        host = _compact(machine)
    else:
        for text in ("alpha", "same text"):
            machine.new_turn(text, False, 0, False, "lookup")
        host = deepcopy(machine.host)
    replacement = machine.stamp(host[-2])
    host[-2] = replacement
    appended = machine.stamp(dict(role="user", content="next"))
    host.append(appended)
    expected = machine.occurrences + [replacement, appended]
    machine.engine._ingest_messages(host)
    assert_occurrences(expected, machine.rows(), engine=machine.engine)
    machine.engine._ingest_messages(deepcopy(host))
    assert_occurrences(expected, machine.rows(), engine=machine.engine)


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("check", ["payload", "replay"])
def test_r3_2_restart_metadata(machine, anchor, check):
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    host = _compact(machine)
    metadata = deepcopy(machine.engine._compress_commit_proof["output_metadata"])
    machine.engine.shutdown()
    machine.engine = machine.start_engine()
    if check == "payload":
        assert machine.engine._durable_commit_proof_payload().get("output_metadata") == metadata
        return
    replacement = machine.stamp(host[-1])
    host[-1] = replacement
    expected = machine.occurrences + [replacement]
    machine.engine._ingest_messages(host)
    assert_occurrences(expected, machine.rows(), engine=machine.engine)
    machine.engine._ingest_messages(deepcopy(host))
    assert_occurrences(expected, machine.rows(), engine=machine.engine)


@pytest.mark.parametrize("field", ["message_uid", "timestamp"])
def test_r3_3_raising_metadata(machine, field):
    class RaisingStampDict(dict):
        def get(self, key, default=None):
            if key == "timestamp":
                raise RuntimeError("host timestamp failure")
            return super().get(key, default)

    machine.new_turn("alpha", False, 0, False, "lookup")
    row_type = _RaisingUidDict if field == "message_uid" else RaisingStampDict
    host = [row_type(row) for row in machine.host]
    appended = machine.stamp(dict(role="user", content="next"))
    host.append(appended)
    machine.engine._ingest_messages(host)
    assert_occurrences(machine.occurrences + [appended], machine.rows(), engine=machine.engine)


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_r3_4_remapped_metadata(machine, anchor, fresh):
    machine.patch.setenv("LCM_IDENTITY_ANCHOR", str(int(anchor)))
    output = _compact(machine)
    host = _host_merge_consecutive_users(output)
    assert len(host) < len(output)
    assert machine.engine._remap_cursor_through_host_merge(host, machine.engine._compress_commit_proof) == len(host)
    if fresh:
        host[-1] = machine.stamp(host[-1])
    expected = machine.occurrences + ([host[-1]] if fresh else [])
    machine.engine._ingest_messages(host)
    assert_occurrences(expected, machine.rows(), engine=machine.engine)
    machine.engine._ingest_messages(deepcopy(host))
    assert_occurrences(expected, machine.rows(), engine=machine.engine)
