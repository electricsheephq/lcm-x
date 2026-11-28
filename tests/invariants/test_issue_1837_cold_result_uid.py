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
from tests.invariants.test_issue_1540_bound_retry import assert_complete_segments, stored
from tests.test_issue_671_stub_first_exit import tool_pair

_COLD = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1837: in cold mode the durable-remap cursor advances past a result "
                                                                     "whose only change is a new UID, so the per-row veto never sees it")
_SPLIT = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1837: warm mode stores the new-UID result without its call, so the "
                                                                      "stored segment is split (acceptance: segments stay whole)")


def _label(row):
    """A stored row as role plus its call ids, result id or content."""
    calls = [call["id"] for call in row.get("tool_calls") or []]
    return f"{row['role']} " + (f"calls {calls}" if calls else row.get("tool_call_id") or row["content"][:40])


def _row(row):
    return row["role"], row.get("content"), row.get("tool_call_id"), row.get("tool_calls")


KNOWN = {("warm", "result-only"), ("cold", "result-only"), ("cold", "later-pair")}


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
    try:
        engine, host = compact(build, mode=mode, uids=True)
        before = stored(engine)
    except AssertionError:
        pytest.fail("setup: compaction or initial segments failed")
    expected = [_row(row) for row in two_segments(uids=True)]
    _expect_stored_shape([_row(row) for row in before], expected, expected, "setup")
    cid = "later" if change == "later-pair" else "c"
    result_index = next(i for i, row in enumerate(host) if row.get("tool_call_id") == cid)
    if change != "equal-uids":
        for index in ([result_index - 1, result_index] if change == "later-pair" else [result_index]):
            host[index]["message_uid"] = f"{500 + index:032x}"
    host[result_index]["content"] = CLEARED
    if change == "later-pair":
        next(row for row in host if row.get("tool_call_id") == "c")["content"] = CLEARED
    following = [row for index in range(2)
                 for row in tool_pair(f"following-{index}", f"fresh {index}", tool_name="lookup")]
    host.extend(following)
    engine.ingest(deepcopy(host))
    if engine._ingest_failure_count:
        pytest.fail(f"setup: the ingest failed: {engine._last_ingest_error}")
    after = raw_stored(engine)
    if (mode, change) not in KNOWN:  # the controls
        assert after[:len(before)] == before
        results = sum(row.get("tool_call_id") == cid for row in after[len(before):])
        assert results == (0 if change == "equal-uids" else 1), "new UID result lost"
        engine.ingest(deepcopy(host))
        assert stored(engine) == after, "equal-UID repeat was stored again"
        return
    engine.ingest(deepcopy(host))
    if engine._ingest_failure_count:
        pytest.fail(f"repeat ingest failed: {engine._last_ingest_error}")
    repeat = raw_stored(engine)
    fixed = [_row(row) for row in [*before, *host[result_index - 1:result_index + 1], *following]]
    known = [_row(row) for row in [*before, *([host[result_index]] if mode == "warm" else []), *following]]
    observed = [(stage, _row(row)) for stage, rows in (("after", after), ("repeat", repeat)) for row in rows]
    fixed = [(stage, row) for stage in ("after", "repeat") for row in fixed]
    known = [(stage, row) for stage in ("after", "repeat") for row in known]
    try:
        assert_complete_segments(repeat)
        whole = True
    except AssertionError:
        whole = False
    diagnostic = dict(prefix_kept=after[:len(before)] == before, new=[_label(row) for row in after[len(before):]],
                      repeat_added=len(repeat) - len(after), whole=whole)
    _expect_stored_shape(observed, fixed, known, f"#1837 known shape: {diagnostic}")


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
