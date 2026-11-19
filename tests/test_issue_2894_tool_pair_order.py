"""Parallel tool results survive repair and are emitted in call order."""

import importlib.util
import json
import os
import random
from collections import Counter
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import pytest

from hermes_lcm.tool_pairs import RepairResult, repair_tool_pairs
from tests import test_issue_2854_no_summary_pass_unchanged as no_summary
from tests.test_issue_671_stub_first_exit import tool_pair
from tests.test_tool_pairs_i8 import assistant, tool


pytestmark = pytest.mark.issue(2894)
open_engine = no_summary.open_engine
summaries = no_summary.summaries


def stub(call_id):
    return tool(call_id, "synthetic missing result")


def is_api_only(row):
    return row.get("role") == "assistant" and bool(row.get("api_content"))


def test_a_reversed_results_are_kept():
    call = assistant("A", "B")
    a, b = tool("A", "real A"), tool("B", "real B")

    assert repair_tool_pairs([call, b, a], stub=stub) == RepairResult([call, a, b], 0, 0)


@pytest.mark.parametrize("order", list(permutations("ABC")), ids=lambda order: "".join(order))
def test_b_every_arrival_order(order):
    call = assistant("A", "B", "C")
    results = {call_id: tool(call_id, f"real {call_id}") for call_id in "ABC"}
    rows = [call, *(results[call_id] for call_id in order)]

    assert repair_tool_pairs(rows, stub=stub) == RepairResult([call, *results.values()], 0, 0)


def test_c_missing_result_is_stubbed_in_call_position():
    call = assistant("A", "B", "C")
    a, c = tool("A", "real A"), tool("C", "real C")

    assert repair_tool_pairs([call, c, a], stub=stub) == RepairResult([call, a, stub("B"), c], 0, 1)


def test_d_orphan_is_dropped_and_counted():
    call = assistant("A", "B")
    a, b = tool("A", "real A"), tool("B", "real B")
    rows = [call, a, tool("orphan"), b]

    assert repair_tool_pairs(rows, stub=stub) == RepairResult([call, a, b], 1, 0)


@pytest.mark.parametrize("repeat_call", [False, True], ids=["one-call", "repeated-id"])
def test_e_results_are_kept_once_per_call_occurrence(repeat_call):
    call = assistant("A", "A") if repeat_call else assistant("A")
    first, second, extra = (tool("A", content) for content in ("first", "second", "extra"))
    rows = [call, first, second, extra] if repeat_call else [call, first, extra]

    assert repair_tool_pairs(rows, stub=stub) == RepairResult(rows[:-1], 1, 0)


def test_e_duplicate_keeps_the_result_head_already_used():
    call = assistant("A", "B")
    a, first_b, second_b = tool("A"), tool("B", "first B"), tool("B", "second B")

    assert repair_tool_pairs([call, first_b, a, second_b], stub=stub) == RepairResult(
        [call, a, second_b], 1, 0,
    )


def test_f_api_only_rows_follow_the_group_in_their_original_order():
    call = assistant("A", "B")
    a, b = tool("A", "real A"), tool("B", "real B")
    first, second = (
        {"role": "assistant", "content": "", "api_content": content}
        for content in ("first API-only row", "second API-only row")
    )
    rows = [call, b, first, a, second]

    assert repair_tool_pairs(rows, stub=stub, is_api_only=is_api_only) == RepairResult(
        [call, a, b, first, second], 0, 0,
    )


def generated_rows(rng, *, ordered, api_only):
    rows = []
    complete_in_order = True
    for group in range(rng.randrange(1, 4)):
        ids = [f"{group}-{index}" for index in range(rng.randrange(1, 6))]
        expected = assistant(*ids)
        expected["tool_calls"].insert(rng.randrange(len(ids) + 1), rng.choice([None, {}, {"id": " "}]))
        rows.extend([{"role": "user", "content": f"request {group}"}, expected])
        results = []
        for call_id in ids:
            if rng.randrange(3) == 0:
                results.append(tool(rng.choice(["orphan", "", " "])))
            if api_only and rng.randrange(3) == 0:
                results.append({"role": "assistant", "content": "", "api_content": f"API-only {call_id}"})
            if ordered or rng.randrange(4) != 0:
                results.append(tool(call_id, f"real {call_id}"))
                if rng.randrange(3) == 0:
                    results.append(tool(call_id, f"duplicate {call_id}"))
        if not ordered:
            rng.shuffle(results)
        seen_ids = dict.fromkeys(row["tool_call_id"] for row in results if row.get("tool_call_id") in ids)
        complete_in_order &= list(seen_ids) == ids
        rows.extend(results)
    if not ordered and rng.randrange(20) == 0:
        rows.insert(rng.randrange(len(rows) + 1), rng.choice([None, "non-dict row", 0]))
        complete_in_order = False
    return rows, complete_in_order


def repair_outcome(repair, rows, *, insert, api_only):
    try:
        return repair(rows, stub=stub, insert_missing_tool_stubs=insert, is_api_only=api_only)
    except AttributeError as error:
        if any(not isinstance(row, dict) for row in rows):
            return error
        raise


def test_g_head_differential(capsys):
    export = os.environ.get("LCM_PAIR_ORDER_BASELINE")
    if not export:
        pytest.skip("Set LCM_PAIR_ORDER_BASELINE to the untouched dd369712a export")
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm._issue_2894_baseline_tool_pairs", Path(export) / "tool_pairs.py",
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    rng = random.Random(2894)
    complete = malformed = differences = 0
    examples = []
    for _ in range(5_000):
        api_only = rng.choice([None, is_api_only])
        options = {"insert": rng.choice([False, True]), "api_only": api_only}
        rows, complete_in_order = generated_rows(rng, ordered=rng.randrange(4) == 0, api_only=api_only)
        before = repair_outcome(baseline.repair_tool_pairs, rows, **options)
        after = repair_outcome(repair_tool_pairs, rows, **options)
        if isinstance(before, AttributeError):
            assert isinstance(after, AttributeError) and str(after) == str(before)
            malformed += 1
            continue
        before_bytes, after_bytes = (json.dumps(result, ensure_ascii=False).encode() for result in (before, after))
        if complete_in_order:
            assert before_bytes == after_bytes
            complete += 1
        input_ids = {id(row) for row in rows}
        kept_before = Counter(id(row) for row in before.rows if id(row) in input_ids)
        kept_after = Counter(id(row) for row in after.rows if id(row) in input_ids)
        assert kept_before <= kept_after
        assert after.inserted <= before.inserted and after.dropped <= before.dropped
        if before_bytes != after_bytes:
            differences += 1
            if len(examples) < 3:
                examples.append((rows, before, after))

    assert complete and malformed and len(examples) == 3
    with capsys.disabled():
        print(f"\nG: {complete} complete in-order, {malformed} matching malformed, {differences}/5000 differ")
        for rows, before, after in examples:
            print(f"Input: {rows!r}\nHEAD: {before!r}\nCandidate: {after!r}")


def test_h_engine_no_summary_pass_keeps_both_results(open_engine, summaries):
    engine = open_engine(fresh_tail_count=64)
    view = no_summary.host_view()
    call, a = tool_pair("A", "real A")
    other_call, b = tool_pair("B", "real B")
    call["tool_calls"].extend(other_call["tool_calls"])
    view.extend([{"role": "user", "content": "parallel calls"}, call, b, a])
    engine.ingest(deepcopy(view))

    output = engine.compress(view, current_tokens=24_000)

    assert not summaries
    assert engine._last_compress_leaves[1] == 0
    assert output[-3:] == [call, a, b]
