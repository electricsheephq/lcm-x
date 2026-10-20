"""I8: characterize emitted tool pairs without changing durable history."""

from __future__ import annotations

from copy import deepcopy
import importlib
import logging
import random
from typing import Any, cast

import pytest

Row = dict[str, Any]
engine_module = importlib.import_module("hermes_lcm.engine")


def assistant(*call_ids: object) -> Row:
    return {"role": "assistant", "tool_calls": [{"id": value} for value in call_ids]}


def tool(call_id: object, content: str = "result") -> Row:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def repair(rows: list[Row], insert: bool = True) -> list[Row]:
    engine = engine_module.LCMEngine.__new__(engine_module.LCMEngine)
    return cast(list[Row], engine._sanitize_tool_pairs(rows, insert_missing_tool_stubs=insert))


def missing(call_id: str) -> Row:
    return tool(call_id, "[Result from earlier conversation — see context summary above]")


@pytest.mark.parametrize("insert", [False, True])
@pytest.mark.parametrize(
    ("rows", "kept"),
    [
        ([], []),
        ([tool("orphan")], []),
        ([assistant("a"), tool("a")], [0, 1]),
        ([assistant("a"), tool("a"), tool("a")], [0, 1]),
        ([assistant("a", "b"), tool("a"), tool("b")], [0, 1, 2]),
        ([assistant("a", "b"), tool("b"), tool("a")], [0, 2, "b"]),
        ([assistant("a"), {"role": "user", "content": "next"}, tool("a")], [0, "a", 1]),
        ([assistant("same", "same"), tool("same")], [0, 1, "same"]),
        ([assistant("same"), assistant("same"), tool("same")], [0, "same", 1, 2]),
        ([assistant("same", "same"), tool("same", "first"), tool("same", "second")], [0, 1, 2]),
        ([assistant(" a ", 7), tool(" a "), tool(7)], [0, 1, 2]),
        ([assistant(None, "", 0), tool("orphan")], [0]),
        ([{"role": "assistant", "tool_calls": [None, {}, {"tool_call_id": "a"}]}, tool("a")], [0, 1]),
        ([{"role": "assistant", "tool_calls": None}, tool("orphan")], [0]),
        ([{"role": "system", "content": "rules"}, tool("orphan"), {"content": "no role"}], [0, 2]),
    ],
)
def test_i8_characterized_sequences(rows: list[Row], kept: list[int | str], insert: bool) -> None:
    before = deepcopy(rows)
    expected = [rows[item] if isinstance(item, int) else missing(item)
                for item in kept if isinstance(item, int) or insert]
    result = repair(rows, insert)
    assert result == expected
    assert rows == before
    assert result is not rows
    for row, item in zip(result, (item for item in kept if isinstance(item, int) or insert)):
        if isinstance(item, int):
            assert row is rows[item]


def test_i8_stub_factory_and_engine_logs(caplog: pytest.LogCaptureFixture) -> None:
    engine = engine_module.LCMEngine.__new__(engine_module.LCMEngine)
    calls: list[str] = []

    def stub(call_id: str) -> Row:
        calls.append(call_id)
        return tool(call_id, "injected stub")

    engine._missing_tool_result_stub = stub
    rows = [tool("orphan"), assistant("same", "same"), tool("same")]
    with caplog.at_level(logging.INFO, logger="hermes_lcm.engine"):
        result = engine._sanitize_tool_pairs(rows)
    assert calls == ["same"]
    assert result == [rows[1], rows[2], tool("same", "injected stub")]
    assert caplog.record_tuples == [
        ("hermes_lcm.engine", logging.INFO,
         "LCM tool-pair guardrail: dropped 1 late/orphan/duplicate tool result(s)"),
        ("hermes_lcm.engine", logging.INFO,
         "LCM tool-pair guardrail: inserted 1 missing tool-result stub(s)"),
    ]
    calls.clear()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hermes_lcm.engine"):
        assert engine._sanitize_tool_pairs(rows, insert_missing_tool_stubs=False) == rows[1:]
    assert calls == []
    assert len(caplog.records) == 1
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hermes_lcm.engine"):
        assert engine._sanitize_tool_pairs([assistant("a"), tool("a")])
    assert caplog.records == []


@pytest.mark.parametrize("insert", [False, True])
def test_i8_seeded_occurrence_properties(insert: bool) -> None:
    rng = random.Random(8048)
    for _ in range(300):
        rows: list[Row] = []
        for index in range(rng.randrange(30)):
            kind = rng.choice(["user", "assistant", "tool", "system"])
            if kind == "assistant":
                rows.append(assistant(*(rng.choice(["a", "b"]) for _ in range(rng.randrange(4)))))
            elif kind == "tool":
                rows.append(tool(rng.choice(["a", "b", "orphan"]), str(index)))
            else:
                rows.append({"role": kind, "content": str(index)})
        before = deepcopy(rows)
        result = repair(rows, insert)
        assert rows == before
        assert repair(result, insert) == result
        assert [id(row) for row in result if row["role"] != "tool"] == [
            id(row) for row in rows if row["role"] != "tool"
        ]
        pending: list[str] = []
        for row in result:
            if row["role"] == "tool":
                # Without stubs, absent earlier calls may be skipped.
                if not insert:
                    while pending and pending[0] != row["tool_call_id"]:
                        pending.pop(0)
                assert pending and pending.pop(0) == row["tool_call_id"]
            else:
                if insert:
                    assert not pending
                pending = [call["id"] for call in row.get("tool_calls", [])]
        if insert:
            assert not pending


@pytest.mark.parametrize("insert", [False, True])
def test_i8_pure_repair_counts(insert: bool) -> None:
    pairs = importlib.import_module("hermes_lcm.tool_pairs")
    rows = [tool("orphan"), assistant("a", "a", "b"), tool("a"), tool("unmatched")]
    before = deepcopy(rows)
    calls: list[str] = []

    def stub(call_id: str) -> Row:
        calls.append(call_id)
        return missing(call_id)

    result = pairs.repair_tool_pairs(rows, stub=stub, insert_missing_tool_stubs=insert)
    assert result.rows == repair(rows, insert)
    assert result.dropped == 2
    assert result.inserted == (2 if insert else 0)
    assert calls == (["a", "b"] if insert else [])
    assert rows == before
    assert result.rows[0] is rows[1]
    assert result.rows[1] is rows[2]
