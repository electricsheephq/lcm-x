"""I1/I2: a retried result whose only change is a new UID is stored in cold mode (#1837, a v0.30.0 named loss limit).

The shapes come from the parked fix draft #1935. The cold cells (the loss) and the warm result-only cell (a split
segment) are strict xfails until the v0.30.x fix; the other cells are controls that pass today.
"""
from copy import deepcopy

import pytest

import tests.invariants.test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import (
    build as build, compact, CLEARED, LONG, stored as raw_stored,
)
from tests.invariants.test_issue_1540_bound_retry import stored
from tests.test_issue_671_stub_first_exit import tool_pair

_COLD = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1837: in cold mode the durable-remap cursor advances past a result "
                                                                     "whose only change is a new UID, so the per-row veto never sees it")
_SPLIT = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1837: warm mode stores the new-UID result without its call, so the "
                                                                      "stored segment is split (acceptance: segments stay whole)")


@pytest.mark.parametrize("mode,change", [
    pytest.param("warm", "result-only", marks=_SPLIT), pytest.param("cold", "result-only", marks=_COLD),
    ("warm", "later-pair"), pytest.param("cold", "later-pair", marks=_COLD),
    ("warm", "equal-uids"), ("cold", "equal-uids"),
])
def test_a_retried_result_with_a_new_uid_is_stored(build, monkeypatch, mode, change):
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
    if change != "equal-uids":
        for index in ([result_index - 1, result_index] if change == "later-pair" else [result_index]):
            host[index]["message_uid"] = f"{500 + index:032x}"
    host[result_index]["content"] = CLEARED
    if change == "later-pair":
        next(row for row in host if row.get("tool_call_id") == "c")["content"] = CLEARED
    for index in range(2):
        host.extend(tool_pair(f"following-{index}", f"fresh {index}", tool_name="lookup"))
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = raw_stored(engine)
    assert after[:len(before)] == before
    results = sum(row.get("tool_call_id") == cid for row in after[len(before):])
    assert results == (0 if change == "equal-uids" else 1), "new UID result lost"
    engine.ingest(deepcopy(host))
    assert stored(engine) == after, "equal-UID repeat was stored again"
