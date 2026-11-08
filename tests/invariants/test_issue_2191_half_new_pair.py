"""#2191 named loss limit (v0.30.0): when only one half of a replayed tool pair is a new occurrence, ingest stores
that half and keeps suppressing its partner, so this occurrence's partner copy is missing beside it.

The suppressed half's text is already stored (its earlier copy); what is lost is this occurrence's copy next to
its partner. The strict xfail turns green when the whole segment leaves suppression (planned for v0.30.x) and must
then become an ordinary test. ``test_issue_2172_new_occurrence_at_cursor`` pins the both-halves-new control.
"""

from copy import deepcopy

import pytest

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored

pytestmark = pytest.mark.issue(2191)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2191 named loss limit (v0.30.0): when one half of a replayed tool pair is a new occurrence, "
                          "it is stored without its partner's copy for that occurrence")
@pytest.mark.parametrize("new_half", ["call", "result"])
def test_a_half_new_pair_is_stored_whole(build, monkeypatch, new_half):
    plain_fixture = shapes.fixture

    def stamped_fixture(*, uids=False):
        rows = plain_fixture(uids=uids)
        rows[-2]["timestamp"], rows[-1]["timestamp"] = 11, 12
        return rows

    monkeypatch.setattr(shapes, "fixture", stamped_fixture)
    engine, host = compact(build, mode="cold")
    before = stored(engine)
    if engine._compress_commit_proof is not None or engine._durable_commit_proof_payload() is None:
        pytest.fail("setup: the cold engine must bind through the durable compaction proof")
    previous = deepcopy(host)
    call_index = len(host) - 2
    new, old = (call_index, call_index + 1) if new_half == "call" else (call_index + 1, call_index)
    host[new]["timestamp"] = 21  # only this half is a new occurrence; its partner keeps its stamp
    if not engine._host_occurrence_changed(host[new], previous[new]) or (
            engine._host_occurrence_changed(host[old], previous[old])):
        pytest.fail("setup: exactly one half must read as a new occurrence")
    engine.ingest(deepcopy(host))
    if engine._ingest_failure_count:
        pytest.fail(f"setup: ingest failed: {engine._last_ingest_error}")
    after = stored(engine)
    if after[:len(before)] != before:
        pytest.fail("setup: the stored prefix changed")
    added = [(row["role"], row.get("tool_call_id"), row.get("observed_at")) for row in after[len(before):]]
    whole = [("assistant", None, 21 if new_half == "call" else 11), ("tool", "c", 21 if new_half == "result" else 12)]
    if added not in ([whole[0]] if new_half == "call" else [whole[1]], whole):
        pytest.fail(f"not the #2191 shape: {added}")
    # Main stores the new half alone: its partner's copy for this occurrence is missing beside it.
    assert [role for role, _id, _stamp in added] == ["assistant", "tool"], f"#2191 known shape: added={added}"
