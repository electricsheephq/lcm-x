"""I1/I2: compacted history cannot prove a fresh reused-id tool occurrence."""

from copy import deepcopy

import pytest

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored
from tests.test_compression_boundary import _turn
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("dangling", [False, True], ids=["full-pair", "dangling-call"])
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


def test_unique_id_compacted_cursor_result_is_suppressed(build, monkeypatch):
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    cursor = len(host) - 1
    assert host[cursor - 1]["role"] == "assistant"
    assert host[cursor]["role"] == "tool"
    assert sum(any(call.get("id") == host[cursor]["tool_call_id"]
                   for call in row.get("tool_calls") or []) for row in before) == 1
    # Pin the reported #2163 boundary after a real compaction and cold bind.
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", lambda *a, **kw: cursor)
    assert engine._replayed_tool_segment_indexes_after_cursor(host, cursor) == {cursor}
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    assert stored(engine) == before
