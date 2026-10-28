"""Assembly prefix/tail boundaries pinned before selection planners move."""

from copy import deepcopy
import logging

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens
from tests.harness import (
    ScriptedSummarizer, assistant, call, engine_context, pin_char_counter, stored_rows, tool, user,
)
from tests.test_issue_1221_objective_default_bound import (
    bounded_objective_scenario, context_digest, objective_scenario,
)
from tests.test_issue_930_prefix_bound import add_nodes, visible_node_ids
from tests.test_overflow_reserves_sanitizer_stubs import (
    OBJECTIVE, SYSTEM, WARNING, add_summaries, full_cap,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    ScriptedSummarizer(fn=lambda *a, **kw: pytest.fail("unexpected model call"),
                       mode="scripted").install(monkeypatch)


@pytest.fixture
def engine(tmp_path):
    with engine_context(tmp_path, window=32_768, temporal_rollups_enabled=False) as engine:
        yield engine


def assemble(engine, system, tail, **kwargs):
    return engine._assemble_context(system, tail, include_lcm_note=False, persist=False, **kwargs)


@pytest.mark.parametrize("window,objective,pairs,nodes,node_tokens,kept,total,digest", [
    (32_768, 18_000, 2, 8, 3000, 2, 27_839,
     "16ca530d97f1ed547eb1af0f5b53f62abad0d9ff3c97f50986b867f61501dd55"),
    (131_072, 8000, 42, 12, 2500, 9, 110_139,
     "1dd0c96c844df000749fec1160361be1acd900722a6cbe17977552231978db0d"),
])
def test_fitting_objective_neighbor_bytes(engine, window, objective, pairs, nodes, node_tokens, kept, total, digest):
    system, tail, source, anchor = bounded_objective_scenario(engine, window, objective, pairs, nodes, node_tokens)
    raw = stored_rows(engine)
    before = deepcopy(source)
    result = assemble(engine, system, tail)
    fitted = engine._survival_fit(source, result, count_messages_tokens(source), "contract")
    assert context_digest(result) == context_digest(fitted) == digest
    assert count_messages_tokens(result) == total
    assert len(visible_node_ids(result)) == kept
    assert sum(anchor["content"] in row["content"] for row in result) == 1
    assert engine._survival_fit_budget(source, count_messages_tokens(source)) == int(window * 0.85)
    assert fitted[-len(tail):] == tail
    assert source == before and stored_rows(engine) == raw


@pytest.mark.parametrize("window,enabled", [(32_768, True), (131_072, True), (0, True), (32_768, False)])
def test_large_objective_default_modes(engine, window, enabled):
    system, tail, source, objective = objective_scenario(engine)
    engine.context_length = window
    engine._config.survival_fit = enabled
    raw = stored_rows(engine)
    result = assemble(engine, system, tail)
    assert result[-len(tail):] == tail
    if window == 32_768 and enabled:
        assert objective["content"] in result[1]["content"]
        assert visible_node_ids(result) == []
        fitted = engine._survival_fit(source, result, count_messages_tokens(source), "contract")
        assert engine._survival_measure(fitted) <= 27_852
        assert engine._last_survival_fit["uncovered_rows"] == 0
    elif not enabled:
        assert context_digest(result) == "66ff945b818c7b7d10c80d35f884d04f75bc8a99d3d4c5bca66edff1b5de172a"
        assert engine._survival_fit(source, result, 0, "contract") is result
    else:
        assert visible_node_ids(result) == [1]
        engine.context_length = 0
        assert context_digest(result) == context_digest(assemble(engine, system, tail))
    assert stored_rows(engine) == raw


@pytest.mark.parametrize("delta", [-1, 1])
def test_objective_immediately_around_default_bound(engine, delta):
    ceiling = engine._survival_ceiling()
    bound = ceiling // 4
    # A heavy tool tail makes the quarter-ceiling floor the binding prefix budget.
    tail = [assistant("", [call("c")]), tool("c", "data" * 24000)]
    objective = user("plan" * (bound - 7 + delta))
    engine._pending_context_anchor_messages = [objective, *tail]
    add_nodes(engine, 0, 2, token_count=200)
    part = engine._build_preserved_objective_summary_part(objective)
    cost = count_message_tokens(user(part))
    # The fallback includes the marker text; calibrate to one token either side.
    objective["content"] = objective["content"][:len(objective["content"]) + 4 * (bound + delta - cost)]
    engine._pending_context_anchor_messages = [objective, *tail]
    part = engine._build_preserved_objective_summary_part(objective)
    assert count_message_tokens(user(part)) == bound + delta
    engine.ingest([SYSTEM, objective, *tail])
    raw = stored_rows(engine)
    result = assemble(engine, SYSTEM, tail)
    assert objective["content"] in result[1]["content"]
    assert visible_node_ids(result) == []
    assert result[-len(tail):] == tail
    assert engine._survival_fit_budget([SYSTEM, *tail], 0) == ceiling
    assert stored_rows(engine) == raw


@pytest.mark.parametrize("prefix_delta", [0, 1], ids=["node-at-floor", "node-one-over-floor"])
def test_objective_and_node_at_quarter_floor(engine, prefix_delta):
    bound = engine._survival_ceiling() // 4
    tail = [assistant("", [call("c")]), tool("c", "data" * 24000)]
    objective = user("plan" * bound)
    ids = add_nodes(engine, 0, 1, token_count=200)
    node_part = engine_module._summary_part_text(engine._dag.get_node(ids[0]))
    part = engine._build_preserved_objective_summary_part(objective)
    cost = count_message_tokens(user(part + "\n\n---\n\n" + node_part))
    # Leave exactly enough room for the node, then move one token past the floor.
    objective["content"] = objective["content"][:len(objective["content"]) + 4 * (bound + prefix_delta - cost)]
    part = engine._build_preserved_objective_summary_part(objective)
    assert count_message_tokens(user(part + "\n\n---\n\n" + node_part)) == bound + prefix_delta
    engine._pending_context_anchor_messages = [objective, *tail]
    engine.ingest([SYSTEM, objective, *tail])
    raw = stored_rows(engine)
    result = assemble(engine, SYSTEM, tail)
    assert visible_node_ids(result) == (ids if prefix_delta == 0 else [])
    assert objective["content"] in result[1]["content"]
    assert result[-len(tail):] == tail
    assert stored_rows(engine) == raw


@pytest.mark.parametrize("window", [32_768, 131_072, 0])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("cap", [0, 12_000])
def test_prefix_selection_modes_and_explicit_cap(engine, window, enabled, cap):
    engine.context_length = window
    engine._config.survival_fit = enabled
    engine._config.max_assembly_tokens = cap
    ids = add_nodes(engine, 1, 70, token_count=2000)
    tail = [user("TAIL_SENTINEL")]
    engine.ingest([SYSTEM, *tail])
    raw = stored_rows(engine)
    original = deepcopy(tail)
    result = assemble(engine, SYSTEM, tail)
    kept = visible_node_ids(result)
    assert kept == ids[-len(kept):] and kept
    assert result[0] == SYSTEM and result[-1] == tail[0]
    assert tail == original and stored_rows(engine) == raw
    if cap:
        assert len(kept) == 5
        assert count_messages_tokens(result) <= cap
        engine.context_length = 0
        assert context_digest(result) == context_digest(assemble(engine, SYSTEM, tail))
    elif not window:
        assert kept == ids
    else:
        assert len(kept) == (13 if window == 32_768 else 55)
        assert count_messages_tokens(result) <= int(window * 0.85)
    assert [node.node_id for node in engine._dag.get_session_nodes("S", limit=1000)] == ids


@pytest.mark.parametrize("shape,missing", [("missing", 3), ("complete", 0), ("reused", 1), ("repeated", 1)])
def test_stub_reservations_preserve_objective_and_priority(engine, shape, missing, caplog):
    nodes = add_summaries(engine)
    calls = assistant("Running.", [call("same"), call("same")])
    if shape == "missing":
        tail = [assistant("Running.", [call(f"c{i}") for i in range(3)])]
    elif shape == "complete":
        tail = [assistant("Running.", [call("a"), call("b")]), tool("a", "result"), tool("b", "result")]
    elif shape == "reused":
        tail = [assistant("Running.", [call("same")]), user("continue"),
                assistant("Running.", [call("same")]), tool("same", "result")]
    else:
        tail = [calls, tool("same", "result")]
    engine.ingest([SYSTEM, user(OBJECTIVE), *tail])
    raw = stored_rows(engine)
    original = deepcopy(tail)
    cap = full_cap(engine, nodes, tail)
    reserved = count_messages_tokens(engine._sanitize_tool_pairs(tail)) - count_messages_tokens(tail)
    if shape in ("reused", "repeated"):
        assert reserved == missing * count_message_tokens(engine._missing_tool_result_stub("same"))
    else:
        assert (reserved > 0) == bool(missing)
    with caplog.at_level(logging.WARNING):
        result = assemble(engine, SYSTEM, tail, assembly_cap_override=cap)
    text = "\n".join(row["content"] for row in result)
    assert OBJECTIVE in text and "DEEPEST" in text
    assert "SHALLOW_NEW" in text
    assert ("SHALLOW_OLD" in text) == (missing == 0)
    assert count_messages_tokens(result) <= cap
    assert WARNING not in caplog.text
    assert tail == original and stored_rows(engine) == raw


@pytest.mark.parametrize("cap", [1, 10_000])
def test_forced_overflow_flag_and_warning(engine, caplog, cap):
    source = [user("USER_SENTINEL " + "data" * 100)]
    engine.ingest(source)
    raw = stored_rows(engine)
    result = assemble(engine, source[0], [], assembly_cap_override=cap)
    with caplog.at_level(logging.WARNING):
        finalized = engine._finalize_forced_overflow_result(source, result, assembly_cap_override=cap)
    failed = cap == 1
    assert finalized == source
    assert engine._last_overflow_recovery_failed is failed
    assert ("returning best-effort context" in caplog.text) is failed
    assert "USER_SENTINEL" not in caplog.text
    assert stored_rows(engine) == raw


class ObjectiveDropped(AssertionError):
    """Only the final objective check may satisfy the #1225 strict xfail."""


@pytest.mark.xfail(strict=True, raises=ObjectiveDropped,
                   reason="#1225: explicit cap gives tail priority over the preserved objective")
def test_1225_objective_over_remaining_room(engine):
    objective = user("OBJECTIVE_SENTINEL " + "plan" * 500)
    tail = [assistant("TAIL_SENTINEL " + "data" * 700)]
    engine.ingest([SYSTEM, objective, *tail])
    engine._pending_context_anchor_messages = [objective, *tail]
    result = assemble(engine, SYSTEM, tail, assembly_cap_override=1000)
    if not any(objective["content"] in row["content"] for row in result):
        raise ObjectiveDropped("objective missing from the capped assembly")


def test_last_resort_warning_after_unbudgeted_sanitizer_addition(engine, monkeypatch, caplog):
    nodes = add_summaries(engine)
    tail = [assistant("Running.", [call("c")]), tool("c", "result")]
    cap = full_cap(engine, nodes, tail)
    original = engine_module.LCMEngine._sanitize_active_context_messages

    def inflate(self, rows, **options):
        result = original(self, rows, **options)
        if options.get("insert_missing_tool_stubs", True) and any(OBJECTIVE in row["content"] for row in result):
            result = [*result, user("UNBUDGETED_SENTINEL " + "data" * 400)]
        return result

    monkeypatch.setattr(engine_module.LCMEngine, "_sanitize_active_context_messages", inflate)
    with caplog.at_level(logging.WARNING):
        result = assemble(engine, SYSTEM, tail, assembly_cap_override=cap)
    assert not any(OBJECTIVE in row["content"] for row in result)
    assert WARNING in caplog.text and "UNBUDGETED_SENTINEL" not in caplog.text


def test_rejected_call_refunds_orphan_result_before_prefix_selection(engine):
    engine._pending_context_anchor_messages = [user(OBJECTIVE)]
    objective = user(engine._build_preserved_objective_summary_part(user(OBJECTIVE)))
    base = count_messages_tokens([SYSTEM, objective])
    result = tool("kept", "abcd" * (count_message_tokens(objective) + 3))
    tail = [assistant("Running.", [call("kept"), call("missing")]), result]
    assert count_messages_tokens([SYSTEM, result]) <= base + 20
    assert count_messages_tokens([SYSTEM, *engine._sanitize_tool_pairs(tail)]) > base + 20
    engine.ingest([SYSTEM, user(OBJECTIVE), *tail])
    raw = stored_rows(engine)
    assembled = assemble(engine, SYSTEM, tail, assembly_cap_override=base + 20)
    assert OBJECTIVE in "\n".join(row["content"] for row in assembled)
    assert not any(row.get("role") == "tool" or row.get("tool_calls") for row in assembled)
    assert count_messages_tokens(assembled) <= base + 20
    assert stored_rows(engine) == raw
