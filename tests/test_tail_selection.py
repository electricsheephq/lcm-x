"""Direct selection tables share the engine's pre-extraction characterization."""

from copy import deepcopy

import pytest

from hermes_lcm.assembly_plan import TailSelection, select_tail
from tests.test_tail_selection_contract import CASES, counters, row


def stub(call_id):
    return {"role": "tool", "tool_call_id": call_id,
            "content": "[Result from earlier conversation — see context summary above]"}


def droppable(message):
    return message["role"] in {"assistant", "tool"}


@pytest.mark.parametrize("name,rows,cap,indexes,charged,trace", CASES, ids=[case[0] for case in CASES])
def test_tail_selection_table(name, rows, cap, indexes, charged, trace):
    rows = deepcopy(rows)
    original = deepcopy(rows)
    identities = [id(message) for message in rows]
    events = []
    one, many = counters(events)
    selection = select_tail(
        rows, cap, 2, one, count_messages_tokens=many,
        missing_tool_result_stub=stub, is_budget_droppable_tail_message=droppable,
    )
    assert selection == TailSelection(indexes, charged)
    assert events == trace
    assert rows == original and [id(message) for message in rows] == identities


def test_eviction_policy_is_lazy_and_counter_exception_propagates():
    rows = [row("assistant", "large", 99), row("assistant", "small", 3)]
    events = []
    one, many = counters(events)

    def protected(message):
        events.append(("evict", message["content"]))
        return False

    result = select_tail(rows, 5, 2, one, count_messages_tokens=many,
                         missing_tool_result_stub=stub, is_budget_droppable_tail_message=protected)
    assert result == TailSelection((1,), 3)
    assert events == [("one", "small"), ("one", "large"), ("evict", "large")]

    def fail(message):
        raise ValueError("counter failed")

    with pytest.raises(ValueError, match="counter failed"):
        select_tail(rows, 5, 2, fail, count_messages_tokens=many,
                    missing_tool_result_stub=stub, is_budget_droppable_tail_message=protected)
