"""I1/I2: compacted history cannot prove a fresh reused-id tool occurrence."""

from copy import deepcopy

import pytest

from tests.invariants import test_issue_1529_warm_proof_salvage as shapes
from tests.invariants.test_issue_1529_warm_proof_salvage import build as build, compact, stored
from tests.test_compression_boundary import _turn
from tests.test_issue_671_stub_first_exit import tool_pair


@pytest.mark.parametrize("dangling", [
    pytest.param(False, id="full-pair", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="#2172 compacted variant: an identical earlier pair compacted out of the host view still proves a replay; planned for v0.30.x (design r3.8 named loss limit)",
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
    try:
        engine, host = compact(build, mode="cold")
    except AssertionError:
        pytest.fail("setup: compaction failed")
    if any(row.get("tool_call_id") == "call_0" for row in host):
        pytest.fail("setup: the earlier call_0 pair was not compacted out of the host view")
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
    try:
        before = stored(engine)
    except AssertionError:
        if dangling:
            raise
        pytest.fail("setup: initial tool segments are split")
    if not dangling:
        expected = [_row(row) for row in fixture()]
        _expect_stored_shape([_row(row) for row in before], expected, expected, "setup")
    engine.ingest(deepcopy(host) + new)
    after = stored(engine)
    added = [(row["role"], row["content"], row.get("tool_call_id")) for row in after[len(before):]]
    if dangling:  # the unchanged dangling-call control
        if after[:len(before)] == before and added == [(row["role"], row["content"], row.get("tool_call_id")) for row in new]:
            return
        pytest.fail(f"not the #2172 shape: prefix kept={after[:len(before)] == before}; added={added}")
    _expect_stored_shape([_row(row) for row in after], [_row(row) for row in [*before, *new]],
                         [_row(row) for row in [*before, new[0], new[-1]]],
                         f"#2172 known shape: the new call and result were taken as a replay: {added}")


def _row(row):
    return row["role"], row["content"], row.get("tool_call_id"), row.get("tool_calls")


def _expect_stored_shape(observed, fixed, known, message, error_type=AssertionError):
    """Accept only the complete fixed list or the complete named-limit list, in store order."""
    if observed == fixed:
        return
    if observed == known:
        raise error_type(message)
    differences = []
    for label, expected in (("fixed", fixed), ("known", known)):
        index = next((i for i, (actual, wanted) in enumerate(zip(observed, expected)) if actual != wanted),
                     min(len(observed), len(expected)))
        differences.append(f"{label}: first difference={index}, lengths={len(observed)}/{len(expected)}")
    pytest.fail("unexpected stored shape; " + "; ".join(differences))


@pytest.mark.parametrize("case", ["known", "fixed", "doubled", "swapped"])
def test_stored_shape_comparison(case):
    fixed = [("user", "first"), ("assistant", "second")]
    known = [fixed[0]]
    observed = {"known": known, "fixed": fixed, "doubled": [fixed[0], *fixed], "swapped": fixed[::-1]}[case]
    if case == "known":
        with pytest.raises(AssertionError, match="synthetic known shape"):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    elif case == "fixed":
        _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
    else:
        with pytest.raises(pytest.fail.Exception, match="first difference=.*lengths="):
            _expect_stored_shape(observed, fixed, known, "synthetic known shape", AssertionError)
