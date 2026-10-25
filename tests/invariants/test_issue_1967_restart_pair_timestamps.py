"""C4/#1869: a unique unchanged user must not hide a fresh stamped pair."""

from copy import deepcopy
import json

import pytest

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import (
    CLEARED, build as build, compact, stored,
)
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("case", ["changed-no-uids", "changed-new-uids", "unchanged-no-uids"])
def test_cold_restart_pair_occurrence(build, monkeypatch, capsys, case):
    with_uids = case == "changed-new-uids"
    changed = case != "unchanged-no-uids"
    plain_fixture = shapes.fixture

    def stamped_fixture(*, uids=False):
        rows = plain_fixture(uids=uids)
        rows[-2]["timestamp"], rows[-1]["timestamp"] = 11, 12
        return rows

    monkeypatch.setattr(shapes, "fixture", stamped_fixture)
    engine, host = compact(build, mode="cold", uids=with_uids)
    before = stored(engine)
    previous = deepcopy(host[-3:])
    pair_index = len(host) - 2
    assert engine._compress_commit_proof is None  # Real cold resume, durable proof only.
    assert sum(r["role"] == "user" and r.get("content") == "repeat" for r in host) == 1
    assert sum(r["role"] == "user" and r.get("content") == "repeat" for r in before) == 1
    if changed:
        host[-2]["timestamp"], host[-1]["timestamp"] = 21, 22
    if with_uids:
        for i, row in enumerate(host[-2:]):
            row["message_uid"] = f"{100 + i:032x}"
    else:
        assert not any(r.get("message_uid") for r in host + before)
    assert host[-3] == previous[0]  # User identity, content AND timestamp stay fixed.
    assert all(engine._host_occurrence_changed(row, old) == changed
               for row, old in zip(host[-2:], previous[-2:]))
    host[-1]["content"] = CLEARED
    for i in range(2):  # Age c out of Hermes' last-two-result salvage protection.
        host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))

    replayed = []
    helper = engine._replayed_tool_segment_indexes_after_cursor

    def observe(messages, cursor):
        selected = helper(messages, cursor)
        replayed.append({"cursor": cursor, "pair_selected": [
            pair_index + i in selected for i in range(2)]})
        return selected

    monkeypatch.setattr(engine, "_replayed_tool_segment_indexes_after_cursor", observe)
    engine.ingest(deepcopy(host))
    assert engine._ingest_failure_count == 0, engine._last_ingest_error
    after = stored(engine)
    assert after[:len(before)] == before
    pair_rows = [r for r in after if r.get("tool_call_id") == "c" or any(
        call["id"] == "c" for call in r.get("tool_calls") or [])]
    counts = [sum(r["role"] == role for r in pair_rows) for role in ("assistant", "tool")]
    changed_stored = [any(r["role"] == role and r.get("observed_at") == stamp
                          for r in pair_rows) for role, stamp in (("assistant", 21), ("tool", 22))]
    with capsys.disabled():
        print("\nC4_ROWS " + json.dumps({"case": case, "before_total": len(before),
              "after_total": len(after), "pair_counts": counts,
              "changed_occurrence_stored": changed_stored, "replay_scan": replayed,
              "pair_rows": [{k: r.get(k) for k in
                             ("role", "observed_at", "message_uid", "tool_call_id")} for r in pair_rows]},
              sort_keys=True))
    assert replayed, "cold ingest must reach the residual replay scan"
    assert counts[0] == counts[1], "pair must remain atomic"
    if changed:
        assert counts == [2, 2] and changed_stored == [True, True], "fresh stamped pair was suppressed"
    else:
        # Differential probe: old may retain an ambiguous duplicate; rc2 should keep one.
        assert counts in ([1, 1], [2, 2])
