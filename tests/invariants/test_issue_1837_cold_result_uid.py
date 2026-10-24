"""I1/I2/I8: a new result UID survives cold compaction replay (#1837)."""
from copy import deepcopy
import json

import pytest

from tests.invariants.test_issue_1529_warm_proof_salvage import (
    build as build, compact, CLEARED, LONG, stored as raw_stored,
)
from tests.invariants.test_issue_1540_bound_retry import stored
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("mode", ["warm", "cold"])
@pytest.mark.parametrize("change", ["result-only", "later-pair", "equal-uids"])
def test_retried_result_uid_keeps_complete_segment(build, monkeypatch, mode, change):
    import tests.invariants.test_issue_1529_warm_proof_salvage as shapes

    plain = shapes.fixture

    def two_segments(*, uids=False):
        rows = plain(uids=uids)
        rows.extend(tool_pair("later", LONG + "later", tool_name="lookup"))
        if uids:
            for index, row in enumerate(rows):
                row["message_uid"] = f"{index + 1:032x}"
        return rows

    monkeypatch.setattr(shapes, "fixture", two_segments)
    engine, host = compact(build, mode=mode, uids=True)
    before = stored(engine)
    cid = "later" if change == "later-pair" else "c"
    result_index = next(i for i, row in enumerate(host) if row.get("tool_call_id") == cid)
    call_index = result_index - 1
    if change != "equal-uids":
        changed = [call_index, result_index] if change == "later-pair" else [result_index]
        for index in changed:
            host[index]["message_uid"] = f"{500 + index:032x}"
    host[result_index]["content"] = CLEARED
    if change == "later-pair":
        next(row for row in host if row.get("tool_call_id") == "c")["content"] = CLEARED
    for index in range(2):
        host.extend(tool_pair(f"following-{index}", f"fresh {index}", tool_name="lookup"))
    trace = []
    helper = engine._replayed_tool_segment_indexes_after_cursor

    def traced(messages, cursor):
        replayed = helper(messages, cursor)
        trace.append(dict(cursor=cursor, replayed=sorted(replayed), result_index=result_index))
        return replayed

    monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", traced)
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = raw_stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    counts = (
        sum(any(call["id"] == cid for call in row.get("tool_calls") or []) for row in added),
        sum(row.get("tool_call_id") == cid for row in added),
    )
    print("UID_TRACE", json.dumps(dict(mode=mode, change=change, call_index=call_index,
          result_index=result_index, helper=trace, reconciliation=engine._last_ingest_reconciliation,
          counts=counts)))
    assert counts == ((0, 0) if change == "equal-uids" else (1, 1)), "new UID segment lost"
    assert len(added) == (4 if change == "equal-uids" else 6)
    assert stored(engine) == after
    engine.ingest(deepcopy(host))
    assert stored(engine) == after, "equal-UID repeat was stored again"
