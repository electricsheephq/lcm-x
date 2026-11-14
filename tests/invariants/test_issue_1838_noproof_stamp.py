"""I1/I2/I8: a conflicting user stamp is fresh before first compaction (#1838)."""
from copy import deepcopy
import json

import pytest

from tests.invariants.test_issue_1529_warm_proof_salvage import (
    build as build, fixture, CLEARED, stored as raw_stored,
)
from tests.invariants.test_issue_1540_bound_retry import stored
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("conflict", [False, True], ids=["identical-stamp", "conflicting-stamp"])
def test_no_proof_retry_respects_user_stamp(build, mode, conflict):
    engine = build()
    host = fixture()[-3:]
    engine.ingest(deepcopy(host))
    assert engine._durable_commit_proof_payload() is None
    before = stored(engine)
    if mode == "cold":
        engine.shutdown()
        engine = build()
        assert engine._compress_commit_proof is None
        assert engine._durable_commit_proof_payload() is None
    if conflict:
        host[0]["timestamp"] = 20
    host[-1]["content"] = CLEARED
    for index in range(2):
        host.extend(tool_pair(f"following-{index}", f"fresh {index}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = raw_stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    counts = (
        sum(row.get("role") == "user" for row in added),
        sum(any(call["id"] == "c" for call in row.get("tool_calls") or []) for row in added),
        sum(row.get("tool_call_id") == "c" for row in added),
    )
    expected = ((1, 1, 1) if mode == "cold" else (1, 0, 1)) if conflict else (0, 0, 0)
    print("STAMP_TRACE", json.dumps(dict(mode=mode, conflict=conflict, counts=counts,
          reconciliation=engine._last_ingest_reconciliation)))
    assert counts == expected, "conflicting user and retry segment lost"
    assert len(added) == 4 + sum(expected)
    assert stored(engine) == after
    engine.ingest(deepcopy(host))
    assert stored(engine) == after, "identical retry added rows"
