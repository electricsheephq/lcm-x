"""I1/I2: a durable compaction proof cannot hide a newly stamped result."""

from copy import deepcopy

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored


def test_cold_compaction_cursor_keeps_changed_result_timestamp(build, monkeypatch):
    plain_fixture = shapes.fixture

    def stamped_fixture(*, uids=False):
        rows = plain_fixture(uids=uids)
        rows[-2]["timestamp"], rows[-1]["timestamp"] = 11, 12
        return rows

    monkeypatch.setattr(shapes, "fixture", stamped_fixture)
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    assert engine._compress_commit_proof is None
    assert engine._durable_commit_proof_payload() is not None
    previous = deepcopy(host)
    result_index = len(host) - 1
    host[result_index]["timestamp"] = 22
    assert host[:result_index] == previous[:result_index]
    assert engine._host_occurrence_changed(host[result_index], previous[result_index])
    scans = []
    helper = engine._replayed_tool_segment_indexes_after_cursor

    def observe(messages, cursor):
        selected = helper(messages, cursor)
        scans.append((cursor, selected))
        return selected

    monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", observe)
    engine.ingest(deepcopy(host))
    after = stored(engine)
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    assert scans and scans[0][0] == result_index  # Real durable-proof rewind, no cursor stub.
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [(row["role"], row.get("tool_call_id"), row.get("observed_at")) for row in added] == [
        ("tool", "c", 22),
    ]
    assert added[0]["content"] == previous[result_index]["content"]


def test_cold_compaction_cursor_keeps_changed_segment_timestamps(build, monkeypatch):
    plain_fixture = shapes.fixture

    def stamped_fixture(*, uids=False):
        rows = plain_fixture(uids=uids)
        rows[-2]["timestamp"], rows[-1]["timestamp"] = 11, 12
        return rows

    monkeypatch.setattr(shapes, "fixture", stamped_fixture)
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    assert engine._compress_commit_proof is None
    assert engine._durable_commit_proof_payload() is not None
    previous = deepcopy(host)
    call_index = len(host) - 2
    host[-2]["timestamp"], host[-1]["timestamp"] = 21, 22
    assert host[:call_index] == previous[:call_index]
    assert all(engine._host_occurrence_changed(host[i], previous[i])
               for i in (call_index, call_index + 1))
    scans = []
    helper = engine._replayed_tool_segment_indexes_after_cursor

    def observe(messages, cursor):
        selected = helper(messages, cursor)
        scans.append((cursor, selected.copy()))
        return selected

    monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", observe)
    engine.ingest(deepcopy(host))
    after = stored(engine)
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    # v7 §2.1 rejects the changed stamps at the proof scan itself (A25).
    assert scans == [(call_index, set())]
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [(row["role"], row.get("observed_at")) for row in added] == [
        ("assistant", 21), ("tool", 22),
    ]
    assert added[0]["tool_calls"] == previous[-2]["tool_calls"]
    assert added[1]["content"] == previous[-1]["content"]
    assert added[1]["tool_call_id"] == "c"
