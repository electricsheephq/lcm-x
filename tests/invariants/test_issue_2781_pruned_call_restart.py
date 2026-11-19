"""#2781: after a restart, a pruned tool result makes the next session end store its assistant call again.

The issue's shape (S1): turn 0 is a user row, an assistant call and an unstamped tool result. Ingest it and end the
session with the user row re-stamped. Then (warm, or after a restart) end again with the user row re-stamped again,
the tool result replaced by the host's pruned-tool placeholder, and two more plain turns appended. The acceptance:
the restarted store equals the warm store per occurrence.

Cells (ruling: run on main before any product code):
- red: the S1 shape (restarted == warm; exactly one call row; the placeholder count equals warm's);
- no-swap: the same without the placeholder swap (is the pruned result needed at all);
- second-restart: after S1, restart again and end a third time (user re-stamped again, one more plain turn).
"""

import copy
import json
import os
from collections import Counter

import pytest

from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from tests.invariants.test_issue_2742_warm_tail_refinal import _open, _turn

pytestmark = pytest.mark.issue(2781)

CALL = ("assistant", "", 3.0)
_RESTAMP_GAP = pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="#2987: cold end misses the re-stamped user occurrence",
)


@pytest.fixture(params=[None, "shadow"], ids=["unset", "shadow"])
def uid_mode(request):
    return request.param


def _occurrences(engine):
    return sorted((row["role"], row["content"], row["observed_at"])
                  for row in engine._store.get_session_messages("parent"))


def _run(root, monkeypatch, uid_mode, *, restart, swap, third):
    """Return the stored occurrences after the second end and (if third) after the third end."""
    root.mkdir()
    with monkeypatch.context() as patch:
        engine = _open(root, patch, uid_mode)
        try:
            turn = _turn(0, "plain", uids=False)
            view = [turn[0], turn[2], turn[3]]
            view[2].pop("timestamp")
            view[2]["content"] = "original tool output " * 11
            engine.ingest(copy.deepcopy(view))
            view[0]["timestamp"] = 1001.0
            engine.on_session_end("parent", copy.deepcopy(view))
            if restart:
                engine.shutdown()
                engine = _open(root, patch, uid_mode)
            view[0]["timestamp"] = 2001.0
            if swap:
                view[2]["content"] = _HERMES_PRUNED_TOOL_PLACEHOLDER
            for n in (1, 2):
                view.extend(_turn(n, "plain", uids=False)[:2])
            engine.on_session_end("parent", copy.deepcopy(view))
            second = _occurrences(engine)
            if not third:
                return second, None
            if restart:
                engine.shutdown()
                engine = _open(root, patch, uid_mode)
            view[0]["timestamp"] = 3001.0
            view.extend(_turn(3, "plain", uids=False)[:2])
            engine.on_session_end("parent", copy.deepcopy(view))
            return second, _occurrences(engine)
        finally:
            engine.shutdown()


def _report(name, uid_mode, warm, cold):
    out = os.environ.get("CELL2781_OUT")
    record = {
        "cell": name, "uid_mode": uid_mode, "warm_n": len(warm), "cold_n": len(cold),
        "cold_minus_warm": sorted(map(list, (Counter(cold) - Counter(warm)).elements())),
        "warm_minus_cold": sorted(map(list, (Counter(warm) - Counter(cold)).elements())),
        "warm_call_rows": warm.count(CALL), "cold_call_rows": cold.count(CALL),
        "warm_placeholders": sum(r[1] == _HERMES_PRUNED_TOOL_PLACEHOLDER for r in warm),
        "cold_placeholders": sum(r[1] == _HERMES_PRUNED_TOOL_PLACEHOLDER for r in cold),
    }
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return record


def _assert_equal_to_warm(record, warm, cold):
    assert cold == warm, record
    assert cold.count(CALL) == 1, record
    assert record["cold_placeholders"] == record["warm_placeholders"], record


def test_red_cell_restart_equals_warm(tmp_path, monkeypatch, uid_mode):
    warm, _ = _run(tmp_path / "warm", monkeypatch, uid_mode, restart=False, swap=True, third=False)
    cold, _ = _run(tmp_path / "cold", monkeypatch, uid_mode, restart=True, swap=True, third=False)
    record = _report("red", uid_mode, warm, cold)
    _assert_equal_to_warm(record, warm, cold)


@pytest.fixture
def no_swap(tmp_path, monkeypatch, uid_mode):
    warm, _ = _run(tmp_path / "warm", monkeypatch, uid_mode, restart=False, swap=False, third=False)
    cold, _ = _run(tmp_path / "cold", monkeypatch, uid_mode, restart=True, swap=False, third=False)
    return warm, cold, _report("no-swap", uid_mode, warm, cold)


def test_no_swap_other_occurrences_equal_warm(no_swap):
    warm, cold, record = no_swap
    assert warm.count(CALL) == cold.count(CALL) == 1, record
    assert cold == [row for row in warm if row != ("user", "question-0", 2001.0)], record


@_RESTAMP_GAP
def test_no_swap_restart_equals_warm(no_swap):
    warm, cold, record = no_swap
    assert cold == warm, record


@pytest.fixture
def second_restart(tmp_path, monkeypatch, uid_mode):
    _, warm = _run(tmp_path / "warm", monkeypatch, uid_mode, restart=False, swap=True, third=True)
    _, cold = _run(tmp_path / "cold", monkeypatch, uid_mode, restart=True, swap=True, third=True)
    return warm, cold, _report("second-restart", uid_mode, warm, cold)


def test_second_restart_other_occurrences_equal_warm(second_restart):
    warm, cold, record = second_restart
    assert warm.count(CALL) == cold.count(CALL) == 1, record  # T1-366 STOP check
    assert cold == [row for row in warm if row != ("user", "question-0", 3001.0)], record


@_RESTAMP_GAP
def test_second_restart_equals_warm_third_end(second_restart):
    warm, cold, record = second_restart
    assert cold == warm, record
