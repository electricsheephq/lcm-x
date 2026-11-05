"""I1/I2: compacted history cannot prove a fresh reused-id tool occurrence."""

from copy import deepcopy

import pytest

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored
from tests.test_compression_boundary import _turn
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("dangling", [
    pytest.param(False, id="full-pair", marks=pytest.mark.xfail(
        strict=True,
        reason="#2172 compacted variant: an identical earlier pair compacted out of the host view still proves a replay; planned for v0.30.0",
    )),
    pytest.param(True, id="dangling-call"),
])
def test_compacted_away_identical_pair_is_not_proof(build, monkeypatch, dangling):
    def fixture(*, uids=False):
        rows = [row for i in range(2) for row in _turn(i)]
        rows += [{"role": "user", "content": "poll early"},
                 *tool_pair("call_0", "OK", tool_name="poll")]
        rows += [row for i in range(2, 12) for row in _turn(i)]
        for cid, text, stamp in (("older", "older lookup", 8),
                                 ("middle", "middle lookup", 9), ("c", "repeat", 10)):
            rows += [{"role": "user", "content": text, "timestamp": stamp},
                     *tool_pair(cid, shapes.LONG + cid, tool_name="lookup")]
        return rows

    monkeypatch.setattr(shapes, "fixture", fixture)
    engine, host = compact(build, mode="cold")
    assert not any(row.get("tool_call_id") == "call_0" for row in host)
    call, result = tool_pair("call_0", "OK", tool_name="poll")
    if dangling:
        host = deepcopy(host) + [{"role": "user", "content": "poll again"}, call]
        engine.ingest(deepcopy(host))
        engine.shutdown()
        engine = build()
        new = [result, {"role": "user", "content": "next"}]
    else:
        new = [{"role": "user", "content": "poll again"}, call, result,
               {"role": "user", "content": "next"}]
    before = stored(engine)
    engine.ingest(deepcopy(host) + new)
    after = stored(engine)
    assert after[:len(before)] == before
    added = after[len(before):]
    assert [(row["role"], row["content"]) for row in added] == [
        (row["role"], row["content"]) for row in new
    ]
    assert added[-2]["tool_call_id"] == "call_0"
