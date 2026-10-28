"""Characterize capped-tail selection through the assembly shell before extraction."""

from copy import deepcopy

import pytest

import hermes_lcm.engine as engine_module
from tests.harness.engine import engine_context


def row(role, label, cost, *calls):
    message = {"role": role, "content": label, "_cost": cost}
    if calls:
        message["tool_calls"] = [
            {"id": call, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            for call in calls
        ]
    if role == "tool":
        message["tool_call_id"] = "same"
    return message


# cap includes the two-token system anchor; charge excludes it and includes stubs.
CASES = [
    ("missing-exact", [row("assistant", "call", 3, "same")], 10, (0,), 8,
     [("one", "call"), ("many", ("stub:same",))]),
    ("missing-short", [row("assistant", "call", 3, "same")], 9, (), 0,
     [("one", "call"), ("many", ("stub:same",))]),
    ("repeated", [row("assistant", "call", 3, "same", "same"), row("tool", "result", 4)],
     14, (0, 1), 12, [("one", "result"), ("one", "call"), ("many", ("stub:same",))]),
    ("reused", [row("assistant", "old", 3, "same"), row("user", "continue", 2),
                row("assistant", "new", 3, "same"), row("tool", "result", 4)],
     14, (1, 2, 3), 9, [("one", "result"), ("one", "new"), ("one", "continue"),
                          ("one", "old"), ("many", ("stub:same",))]),
    ("refund", [row("assistant", "call", 3, "same", "missing"), row("tool", "result", 4)],
     7, (), 0, [("one", "result"), ("one", "call"), ("many", ("stub:missing",)), ("one", "result")]),
    ("gap", [row("user", "old", 2), row("assistant", "large", 99), row("user", "new", 2)],
     8, (2,), 2, [("one", "new"), ("one", "large"), ("one", "old")]),
    ("oversized", [row("user", "large", 99)], 8, (), 0, [("one", "large")]),
    ("zero", [row("user", "new", 2)], 0, (), 0, [("one", "new")]),
    ("identity", [row("user", "first", 2), row("assistant", "call", 3, "same"),
                  row("tool", "result", 4), row("user", "last", 2)],
     13, (0, 1, 2, 3), 11, [("one", "last"), ("one", "result"), ("one", "call"), ("one", "first")]),
    ("merge-after-selection", [row("assistant", "large", 99), row("assistant", "small", 3)],
     5, (1,), 3, [("one", "small"), ("one", "large")]),
    ("empty", [], 0, (), 0, []),
]


def label(message):
    if "_cost" not in message:
        return "stub:" + message["tool_call_id"]
    return message["content"]


def counters(events):
    def one(message):
        events.append(("one", label(message)))
        return message.get("_cost", 5)

    def many(messages):
        events.append(("many", tuple(label(message) for message in messages)))
        return sum(message.get("_cost", 5) for message in messages)

    return one, many


@pytest.mark.parametrize("name,rows,cap,indexes,charged,trace", CASES, ids=[case[0] for case in CASES])
def test_engine_tail_contract(tmp_path, monkeypatch, name, rows, cap, indexes, charged, trace):
    rows = deepcopy(rows)
    original = deepcopy(rows)
    events = []
    one, many = counters(events)
    monkeypatch.setattr(engine_module, "count_message_tokens", one)
    monkeypatch.setattr(engine_module, "count_messages_tokens", many)
    with engine_context(tmp_path, temporal_rollups_enabled=False) as engine:
        engine._pending_context_anchor_messages = []
        sanitize = engine._sanitize_active_context_messages
        phases = []

        def observe(messages, **options):
            phases.append((list(messages), options.copy()))
            return sanitize(messages, **options)

        monkeypatch.setattr(engine, "_sanitize_active_context_messages", observe)
        result = engine._assemble_context(
            row("system", "system", 2), rows, assembly_cap_override=cap,
            include_lcm_note=False, persist=False,
        )
        # Observe the real sanitizer, including its unmerged input to the final pass.
        assert len(phases) == 2
        assert phases[0][1] == {"insert_missing_tool_stubs": False, "merge_adjacent_assistants": False}
        assert phases[1][1] == {"generated_rows": []}
        selected = phases[1][0][1:]
        assert len(selected) == len(indexes)
        assert all(message is rows[index] for message, index in zip(selected, indexes))
        assert result == sanitize([row("system", "system", 2), *selected])
        assert events == [("many", ("system",)), *trace]
        assert rows == original
