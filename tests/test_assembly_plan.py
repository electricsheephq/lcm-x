"""Direct contracts for prefix selection, with no engine, storage, or model calls."""

from dataclasses import replace

import pytest

from hermes_lcm.assembly_plan import PrefixBudgetFacts, default_prefix_budget, select_prefix


def char_count(message):
    return len(message["content"])


def facts(keys, budget, *, cap=None, fit=True):
    return PrefixBudgetFacts(keys, "user", budget, cap, fit, char_count)


@pytest.mark.parametrize("window,fit,cap,tail,budget,start", [
    (32_768, True, None, 3000, 22_252, 35),
    (131_072, True, None, 3000, 105_811, 14),
    (0, True, None, 3000, None, 0),
    (32_768, False, None, 3000, 22_252, 35),
    (32_768, True, None, 40_000, 6963, 39),
    (32_768, False, None, 40_000, 0, 40),
    (131_072, True, None, 140_000, 27_852, 34),
    (32_768, True, 9000, 3000, 9000, 38),
    (131_072, False, 9000, 3000, 9000, 38),
    (0, True, 9000, 3000, 9000, 38),
    (32_768, True, 0, 3000, 0, 40),
])
def test_window_and_explicit_budget_modes(window, fit, cap, tail, budget, start):
    resolved = cap
    if cap is None and window:
        resolved = default_prefix_budget(
            ceiling=int(window * 0.85), overhead=1000, leading_tokens=100,
            tail_tokens=tail, repaired_tail_tokens=tail + 500, recall_budget_tokens=1000,
            proactive_recall_enabled=True, embeddings_enabled=True, survival_fit=fit,
        )
    assert resolved == budget
    parts = [str(i % 10) * 4000 for i in range(40)]
    keys = [(0, i + 1) for i in range(40)]
    before = parts[:], keys[:]
    result = select_prefix(parts, facts(keys, resolved, cap=cap, fit=fit))
    assert result.kept_indexes == tuple(range(start, 40))
    assert result.selected_parts == tuple(parts[start:])
    assert (parts, keys) == before


@pytest.mark.parametrize("recall,embeddings,reserve,expected", [
    (True, True, 1000, 7000),
    (False, True, 1000, 8000),
    (True, False, 1000, 8000),
    (True, True, 0, 8000),
    (True, True, -1000, 8000),
    (True, True, None, 8000),
])
def test_recall_reserve_requires_both_build_gates(recall, embeddings, reserve, expected):
    assert default_prefix_budget(
        ceiling=10_000, overhead=500, leading_tokens=100,
        tail_tokens=1000, repaired_tail_tokens=1400, recall_budget_tokens=reserve,
        proactive_recall_enabled=recall, embeddings_enabled=embeddings, survival_fit=True,
    ) == expected


@pytest.mark.parametrize("tail,repaired", [(1000, 1400), (1400, 1000)])
def test_reserve_larger_of_raw_and_repaired_tail(tail, repaired):
    assert default_prefix_budget(
        ceiling=10_000, overhead=500, leading_tokens=100,
        tail_tokens=tail, repaired_tail_tokens=repaired, recall_budget_tokens=0,
        proactive_recall_enabled=False, embeddings_enabled=False, survival_fit=False,
    ) == 8000


@pytest.mark.parametrize("delta,cap,fit,indexes", [
    (-1, None, True, (0,)),
    (0, None, True, (0,)),
    (1, None, True, (0,)),
    (-1, 100, True, (0,)),
    (1, 100, True, (1, 2)),
    (-1, None, False, (0,)),
    (1, None, False, (1, 2)),
])
def test_objective_just_around_bound(delta, cap, fit, indexes):
    parts = ["O" * (100 + delta), "D" * 20, "S" * 20]
    keys = [(-1, None), (0, 10), (1, 11)]
    result = select_prefix(parts, facts(keys, 100, cap=cap, fit=fit))
    assert result.kept_indexes == indexes
    assert result.selected_parts == tuple(parts[i] for i in indexes)


@pytest.mark.parametrize("budget,indexes", [(None, (0, 1, 2, 3)), (34, (1, 3)), (14, (1,))])
def test_depth_then_newness_priority_renders_in_source_order(budget, indexes):
    parts = ["deep-old" * 10, "deep-new", "shallow-old" * 10, "shallow-new"]
    keys = [(0, 1), (0, 2), (1, 3), (1, 4)]
    result = select_prefix(parts, facts(keys, budget))
    assert result.kept_indexes == indexes
    assert result.selected_parts == tuple(parts[i] for i in indexes)


def test_exempt_objective_strips_injected_blocks_without_mutating_input():
    objective = "O" * 101 + "\n<relevant-memories>discard</relevant-memories>"
    parts = [objective, "summary"]
    result = select_prefix(parts, facts([(-1, None), (0, 1)], 100))
    assert result.kept_indexes == (0,)
    assert result.selected_parts == ("O" * 101,)
    assert parts == [objective, "summary"]


def test_budget_measurement_does_not_rewrite_nonexempt_objective_bytes():
    objective = "objective\n<relevant-memories>discard</relevant-memories>"
    result = select_prefix([objective, "summary"], facts([(-1, None), (0, 1)], 9, cap=9))
    # Existing assembly measures stripped objective text, but retains original
    # bytes here; the engine's later sanitation/warning pass owns any overflow.
    assert result.kept_indexes == (0,)
    assert result.selected_parts == (objective,)


def test_counter_policy_is_an_explicit_input_and_exceptions_propagate():
    budget = facts([(0, 1), (0, 2)], 10)
    assert select_prefix(["one", "two"], budget).kept_indexes == (1,)
    assert select_prefix(["one", "two"], replace(budget, count_message_tokens=lambda row: 1)).kept_indexes == (0, 1)

    def fail(message):
        raise ValueError("counter failed")

    with pytest.raises(ValueError, match="counter failed"):
        select_prefix(["one", "two"], replace(budget, count_message_tokens=fail))
