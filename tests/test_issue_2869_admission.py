"""Synthetic threshold-sweep ordering and progress against the pinned main tree."""

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import json
import sys

import pytest

import hermes_lcm.dag as dag
import hermes_lcm.engine as engine_module
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from tests.differential.trees import PrerequisiteUnavailable, materialize
from tests import test_issue_605_foreground_budget as budget_fixtures
from tests.test_issue_605_foreground_budget import (
    SUMMARY,
    _advance_on,
    _engine,
    _leaves,
    _pin_lcm_char_counter,  # noqa: F401 - shared autouse fixture
    _view,
)

clock = budget_fixtures.clock

BASE = "31ac199b1ec52d7b43de852207415552dfe699d5"
PREFIXES = {
    "under-target": [(0, 10_000), (0, 10_000)],
    "one-to-two-targets": [(0, 15_000)] * 4,
    "over-twice-target": [(0, 25_250)] * 4,
    "over-twice-no-group": [(3, 40_000), (2, 35_000), (1, 30_000)],
}
BACKLOG_LEAF = "Earlier synthetic facts. " * 150


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    try:
        root = materialize(BASE, tmp_path_factory.getbasetemp())
    except PrerequisiteUnavailable as exc:
        pytest.skip(str(exc))
    # Use a distinct package so the engine, provider and store imports stay on
    # their own tree while both engines run in one pytest process.
    name = "lcm_issue_2869_base"
    spec = spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
    sys.modules[name] = module_from_spec(spec)
    return {part: import_module(f"{name}.{part}") for part in ("engine", "dag", "escalation", "tokens", "survival_fit")}


CANDIDATE = dict(engine=engine_module, dag=dag, escalation=escalation, tokens=tokens, survival_fit=survival_fit)


def _run(tree, tmp_path, monkeypatch, clock, *, nodes, pre_work=0, pre_work_passes=None, estimate=None, seconds=8,
         leaf=SUMMARY, turns=240, passes=1, window=1_000_000, settings=None, replay=False):
    tmp_path.mkdir()
    settings = {
        "context_threshold": 0.5, "leaf_chunk_tokens": 24_000,
        "max_assembly_tokens": window, "summary_timeout_ms": 120_000, "embeddings_enabled": False,
        **(settings or {}),
    }
    with monkeypatch.context() as patch:
        patch.setattr(clock, "_real", lambda: 0.0)
        patch.setattr(clock, "offset", 0.0)
        # Keep the shared #605 config and use the selected tree's engine and tokenizer.
        patch.setitem(_engine.__globals__, "LCMEngine", tree["engine"].LCMEngine)
        patch.setitem(_engine.__globals__, "LCMConfig", tree["engine"].LCMConfig)
        patch.setattr(tree["survival_fit"], "_host_estimate", lambda messages: None)
        counter = tree["tokens"]
        patch.setattr(counter, "count_tokens", lambda text: 0 if not text else (
            counter._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))
        engine = _engine(tmp_path, **settings)
        engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
        try:
            for i, (depth, size) in enumerate(nodes):
                engine._dag.add_node(tree["dag"].SummaryNode(
                    session_id="S", depth=depth, summary=f"arc {i} " + "history " * 40,
                    token_count=size, source_token_count=size * 4, source_ids=[],
                    source_type="messages", created_at=float(i + 1),
                ))
            def seed_estimate():
                for _ in range(8):
                    engine._foreground_estimates.record_call(engine._primary_summary_route(), estimate)

            if estimate is not None and not replay:
                seed_estimate()
            _advance_on(patch, engine, "_prepare_retained_user_anchor", clock, pre_work, calls=pre_work_passes)
            calls = []
            condensing = False
            condense_nodes = engine._condense_summary_nodes

            def condense(*args, **kwargs):
                nonlocal condensing
                condensing = True
                try:
                    return condense_nodes(*args, **kwargs)
                finally:
                    condensing = False

            patch.setattr(engine, "_condense_summary_nodes", condense)

            def summarize(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
                kind = "condensation" if condensing else "leaf"
                calls.append((kind, round(timeout, 3)))
                if seconds > timeout:
                    clock.offset += timeout
                    raise TimeoutError("synthetic provider timed out")
                clock.offset += seconds
                return leaf if kind == "leaf" else SUMMARY

            patch.setattr(tree["escalation"], "_invoke_summary_llm", summarize)
            original = _view(turns)
            view = original
            trace = []
            for index in range(passes):
                prior_calls, prior_leaves = len(calls), len(_leaves(engine))
                if not replay or index == 0:
                    engine.ingest(view)
                elif estimate is not None:
                    seed_estimate()
                result = engine.compress(view, current_tokens=engine.threshold_tokens + 1)
                status = engine.get_status()["threshold_full_sweep"]
                trace.append(dict(
                    calls=calls[prior_calls:], leaves=len(_leaves(engine)) - prior_leaves,
                    reported_leaves=status["leaf_passes"], stop=status["stop_reason"],
                    target=status["summary_prefix_target_tokens"], frontier=engine._summary_frontier_tokens(),
                    compression_status=engine._last_compression_status,
                ))
                view = original if replay else result
                engine.note_turn_complete()
                clock.offset += 700  # Permit the next eligible sweep after any base hold.
            return trace, result
        finally:
            engine.shutdown()


def _compare(base_tree, tmp_path, monkeypatch, clock, **case):
    base, base_result = _run(base_tree, tmp_path / "base", monkeypatch, clock, **case)
    candidate, candidate_result = _run(CANDIDATE, tmp_path / "candidate", monkeypatch, clock, **case)
    print(json.dumps({"case": {key: value for key, value in case.items() if key != "leaf"},
                      "base": base, "candidate": candidate}, sort_keys=True))
    return base, candidate, base_result, candidate_result


def _assert_progress(base, candidate):
    """Never less progress than the base: at least its leaves on every pass, and a call whenever work remains."""
    available = sum(row["leaves"] for row in base)  # every leaf the base publishes in this cell
    base_leaves = candidate_leaves = 0
    for index, (before, after) in enumerate(zip(base, candidate, strict=True), start=1):
        # The candidate may be ahead: all leaves published and the prefix at or under its target leaves it nothing to call for.
        done = candidate_leaves >= available and after["frontier"] <= after["target"]
        base_leaves += before["leaves"]
        candidate_leaves += after["leaves"]
        assert not before["calls"] or after["calls"] or done, (index, before, after)
        assert candidate_leaves >= base_leaves, (index, base_leaves, candidate_leaves)


@pytest.mark.parametrize("pre_work", [0, 37, 59, 88, 95])
@pytest.mark.parametrize("estimate", [None, 20, 44, 58, 115], ids=["cold", "20", "44", "58", "115"])
@pytest.mark.parametrize("prefix", PREFIXES)
def test_progress_grid(base_tree, tmp_path, monkeypatch, clock, pre_work, estimate, prefix):
    base, candidate, _, _ = _compare(
        base_tree, tmp_path, monkeypatch, clock, nodes=PREFIXES[prefix],
        pre_work=pre_work, estimate=estimate, seconds=8 if estimate is None else estimate, passes=10,
    )
    _assert_progress(base, candidate)


@pytest.mark.parametrize(("pre_work", "estimate", "seconds", "nodes", "turns", "passes", "leaf"), [
    pytest.param(59, 44, 44, [(0, 25_250)] * 4, 240, 4, SUMMARY, id="PROBE2-59-44"),
    pytest.param(37, 44, 44, [(0, 25_250)] * 4, 240, 4, SUMMARY, id="PROBE2-37-44"),
    pytest.param(30, 58, 40, [(0, 25_250)] * 4, 240, 4, SUMMARY, id="PROBE2-30-58"),
    pytest.param(59, 44, 44, [(0, 22_500)] * 4, 240, 4, SUMMARY, id="PROBE2-90k"),
    pytest.param(88, None, 8, [], 240, 3, SUMMARY, id="PROBE3-88-cold"),
    pytest.param(95, None, 8, [], 240, 3, SUMMARY, id="PROBE3-95-cold"),
    pytest.param(0, 200, 20, [(0, 25_250)] * 4, 240, 4, SUMMARY, id="PROBE5-115-stale"),
    pytest.param(50, 44, 44, [(0, 24_000)] * 4, 1720, 6, BACKLOG_LEAF, id="PROBE6-96k"),
    pytest.param(50, 44, 44, [(0, 24_750)] * 4, 1720, 6, BACKLOG_LEAF, id="PROBE6-99k"),
])
def test_review_probes(base_tree, tmp_path, monkeypatch, clock, pre_work, estimate, seconds, nodes, turns, passes, leaf):
    base, candidate, _, _ = _compare(
        base_tree, tmp_path, monkeypatch, clock, nodes=nodes, pre_work=pre_work, estimate=estimate,
        seconds=seconds, turns=turns, passes=passes, leaf=leaf,
    )
    _assert_progress(base, candidate)


@pytest.mark.parametrize(("settings", "target"), [
    ({}, 50_000),
    ({"context_threshold": 0.1}, 25_000),
    ({"context_threshold": 0.05}, 12_500),
    ({"leaf_chunk_tokens": 60_000}, 60_000),
    ({"summary_prefix_target_tokens": 31_000, "summary_prefix_target_ratio": 0.1}, 31_000),
    ({"summary_prefix_target_ratio": 0.06}, 60_000),
])
def test_window_relative_default_and_explicit_precedence(tmp_path, monkeypatch, clock, settings, target):
    trace, _ = _run(CANDIDATE, tmp_path / "candidate", monkeypatch, clock, nodes=[], turns=12, settings=settings)
    print(f"K1 settings={settings} target={trace[0]['target']}")
    assert trace[0]["target"] == target


@pytest.mark.parametrize(("prefix", "pre_work"), [(26_000, 59), (60_000, 59), (60_000, 0)])
def test_leaf_is_first_and_publishes(tmp_path, monkeypatch, clock, prefix, pre_work):
    trace, _ = _run(
        CANDIDATE, tmp_path / "candidate", monkeypatch, clock, nodes=[(0, prefix // 4)] * 4,
        estimate=44, seconds=44, pre_work=pre_work,
    )
    print(f"K2 prefix={prefix} pre_work={pre_work} {trace}")
    assert trace[0]["calls"][0][0] == "leaf"
    assert trace[0]["leaves"] >= 1


def test_condensation_can_follow_the_leaves_in_the_same_pass(tmp_path, monkeypatch, clock):
    trace, _ = _run(CANDIDATE, tmp_path / "candidate", monkeypatch, clock,
                    nodes=PREFIXES["one-to-two-targets"], turns=12, estimate=20, seconds=20)
    print(f"K2 leaf then condensation {trace}")
    kinds = [kind for kind, _ in trace[0]["calls"]]
    assert kinds[0] == "leaf" and "condensation" in kinds[1:]
    assert trace[0]["leaves"] >= 1


def test_prefix_grows_while_draining_then_shrinks_without_a_leaf(tmp_path, monkeypatch, clock):
    trace, _ = _run(CANDIDATE, tmp_path / "candidate", monkeypatch, clock,
                    nodes=PREFIXES["over-twice-target"], turns=12, estimate=44, seconds=44,
                    pre_work=50, pre_work_passes=(1,), leaf=BACKLOG_LEAF, passes=2)
    first, second = trace
    print(f"K2 grow then shrink {trace}")
    assert first["leaves"] == 1 and first["frontier"] > 101_000
    assert second["leaves"] == 0 and second["calls"][0][0] == "condensation"
    assert second["frontier"] < first["frontier"]


@pytest.mark.parametrize(("window", "target"), [
    (64_000, 8_000), (200_000, 24_000), (272_000, 24_000), (480_000, 24_000),
])
@pytest.mark.parametrize("override", [{}, {"leaf_chunk_tokens": 1}, {"summary_prefix_target_ratio": 0.04}])
def test_legacy_window_parity(base_tree, tmp_path, monkeypatch, clock, window, target, override):
    if "summary_prefix_target_ratio" in override:
        target = int(window * override["summary_prefix_target_ratio"])
    elif override:
        target = 1
    base, candidate, before, after = _compare(
        base_tree, tmp_path, monkeypatch, clock, window=window, nodes=[(0, target // 2 + 1000)] * 2,
        turns=80, seconds=44, settings=override,
    )
    assert base == candidate
    assert base[0]["target"] == target
    assert base[0]["calls"][0][0] == "condensation"
    assert json.dumps(before, sort_keys=True).encode() == json.dumps(after, sort_keys=True).encode()


def test_late_entry_adopts_committed_replay(base_tree, tmp_path, monkeypatch, clock):
    base, candidate, before, after = _compare(
        base_tree, tmp_path, monkeypatch, clock, nodes=[], turns=12, estimate=44, seconds=0,
        pre_work=80, pre_work_passes=(2,), passes=2, replay=True,
        settings={"leaf_chunk_tokens": 400, "summary_prefix_target_tokens": 400},
    )
    assert base == candidate
    assert before == after
    assert candidate[0]["leaves"] > 0
    assert candidate[1]["compression_status"] == "compacted"
    assert not candidate[1]["calls"]
