"""#652: no level 3 leaf or node while the survival fit can rescue the request. #653: a threshold sweep keeps
the summary prefix bounded, and assembly keeps the newest summaries.

Engine-level: compress() with a stubbed summariser; the leaf loop, the sweep, the lifecycle frontier and the
assembly are the real ones."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import pytest

import hermes_lcm.compaction as lcm_compaction
import hermes_lcm.engine as lcm_engine
import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
LEAF = "Earlier turns.\nExpand for details about: turns"
MERGED = "Merged arc.\nExpand for details about: arc"
LEVEL_3 = "head text\n\n[...deterministic truncation — details available via lcm_expand...]\n\ntail text"
REJECTED_LINE = "LCM compaction stopped: summary result rejected at level 3"
PART = re.compile(r"\(d(\d+), node (\d+)\)")


@pytest.fixture(autouse=True)
def _pin_lcm_char_counter(monkeypatch):
    """Same counter for every run (#615): LCM's character estimate, no host estimator."""
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))


class _Summaries:
    """Stands in for summarize_with_escalation; records (depth, source_tokens) per call."""

    def __init__(self, level: int = 1, leaf: str = LEAF, merged: str = MERGED, verbatim: bool = False):
        self.level, self.leaf, self.merged, self.verbatim = level, leaf, merged, verbatim
        self.calls: list[tuple[int, int]] = []

    def __call__(self, **kwargs):
        depth = int(kwargs.get("depth", 0))
        self.calls.append((depth, int(kwargs["source_tokens"])))
        if self.level == 3:
            return (kwargs["text"] if self.verbatim else LEVEL_3), 3  # verbatim: the source fits the l3 budget
        return (self.leaf if depth == 0 else self.merged), self.level


def _summaries(monkeypatch, **kwargs) -> _Summaries:
    stub = _Summaries(**kwargs)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", stub)
    return stub


def _engine(tmp_path, context_length: int = 200_000, **config) -> LCMEngine:
    settings = {"fresh_tail_count": 2, "leaf_chunk_tokens": 400, "context_threshold": 0.001,
                "threshold_full_sweep_enabled": True, "max_assembly_tokens": 100_000,
                "database_path": str(tmp_path / "lcm.db"), **config}
    engine = LCMEngine(config=LCMConfig(**settings))
    engine.on_session_start("S", platform="telegram", context_length=context_length, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float) -> list[dict]:
    return [{"role": "user", "content": f"[{tag}] user turn{PAD}", "timestamp": ts},
            {"role": "assistant", "content": f"reply to {tag}{PAD}"}]


def _turns(start: int, count: int) -> list[dict]:
    return [row for i in range(start, start + count) for row in _turn(f"T{i}", 10.0 * (i + 1))]


def _view(turns: int = 6) -> list[dict]:
    return [{"role": "system", "content": "system prompt"}, *_turns(0, turns)]


def _frontier(engine) -> int:
    return int(getattr(engine._lifecycle.get_by_conversation("conv"), "current_frontier_store_id", 0))


def _nodes(engine) -> list:
    return engine._dag.get_session_nodes("S", limit=100_000)


def _frontier_ids(engine) -> list[int]:
    return sorted(node.node_id for node in engine._summary_frontier_nodes())


def _count(caplog, text: str) -> int:
    return sum(text in record.getMessage() for record in caplog.records)


def _compress(engine, view, caplog, **kwargs):
    engine.ingest(view)
    kwargs.setdefault("current_tokens", engine.threshold_tokens + 1)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        return engine.compress(view, **kwargs)


def _depth_0_nodes(engine, count: int, token_count: int = 1000, text: str = "group") -> list[int]:
    ids = []
    for index in range(count):
        ids.append(engine._dag.add_node(SummaryNode(
            session_id="S", depth=0, summary=f"{text} {index}", token_count=token_count, source_token_count=2000,
            source_ids=[], source_type="messages", created_at=float(index + 1))))
    return ids


# -- T1: a level 3 leaf with the fit available is not written ------------------------------------------------

@pytest.mark.parametrize("sweep", [True, False], ids=["sweep", "no-sweep"])
def test_t1_level_3_leaf_is_not_published_when_the_fit_can_rescue(tmp_path, monkeypatch, caplog, sweep):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=sweep)
    stub = _summaries(monkeypatch, level=3)
    view = _view()
    try:
        engine.ingest(view)
        rows = engine._store.get_session_messages("S", limit=100_000)
        frontier, compacted = _frontier(engine), engine._last_compacted_store_id
        result = _compress(engine, view, caplog)
        assert _nodes(engine) == [] and stub.calls and [depth for depth, _ in stub.calls] == [0]
        assert engine._store.get_session_messages("S", limit=100_000) == rows
        assert _frontier(engine) == frontier and engine._last_compacted_store_id == compacted
        assert engine._last_compression_status == "noop" and result == view
        assert engine._last_compression_noop_reason == "summary result rejected"
        if sweep:
            assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "summary_result_rejected"
        assert engine._sweep_budget_hold_until == 0.0  # no #608 hold
        assert _count(caplog, REJECTED_LINE) == 1
        line = next(r.getMessage() for r in caplog.records if REJECTED_LINE in r.getMessage())
        assert line.endswith("; 0 leaves written, backlog kept")
    finally:
        engine.shutdown()


# -- T2: where level 3 still converges ------------------------------------------------------------------------

@pytest.mark.parametrize("escape", ["forced_overflow", "survival_fit_off", "window_unknown", "verbatim"])
def test_t2_level_3_leaf_is_still_written_without_a_rescue(tmp_path, monkeypatch, caplog, escape):
    engine = _engine(tmp_path, **({"survival_fit": False} if escape == "survival_fit_off" else {}))
    if escape == "window_unknown":
        engine.context_length = 0
    _summaries(monkeypatch, level=3, verbatim=escape == "verbatim")
    kwargs = {"current_tokens": 150_000} if escape == "forced_overflow" else {}  # over the 100k assembly cap
    try:
        _compress(engine, _view(), caplog, **kwargs)
        marker = escape != "verbatim"  # a level 3 of a source inside the l3 budget is the source itself
        assert _nodes(engine) and all(("deterministic truncation" in node.summary) is marker
                                      for node in _nodes(engine))
        assert engine._last_compression_status == "compacted" and _count(caplog, REJECTED_LINE) == 0
    finally:
        engine.shutdown()


# -- T3: condensation ------------------------------------------------------------------------------------------

def test_t3_level_3_condensation_writes_no_node_and_forced_still_does(tmp_path, monkeypatch):
    engine = _engine(tmp_path, condensation_fanin=2, threshold_full_sweep_enabled=False)
    stub = _summaries(monkeypatch, level=3)
    _depth_0_nodes(engine, 2)
    try:
        before = _frontier_ids(engine)
        assert engine._maybe_condense(force_overflow=False) == 0
        assert _frontier_ids(engine) == before and len(_nodes(engine)) == 2 and len(stub.calls) == 1
        assert engine._last_condensation_suppressed_reason == "summary_result_rejected"
        assert engine._maybe_condense(force_overflow=True) == 1
        assert len(_nodes(engine)) == 3 and _frontier_ids(engine) != before
    finally:
        engine.shutdown()


def test_t3_verbatim_level_3_condensation_is_still_written(tmp_path, monkeypatch):
    engine = _engine(tmp_path, condensation_fanin=2, threshold_full_sweep_enabled=False)
    _summaries(monkeypatch, level=3, verbatim=True)
    _depth_0_nodes(engine, 2)
    try:
        assert engine._maybe_condense(force_overflow=False) == 1 and len(_nodes(engine)) == 3
        assert engine._last_condensation_suppressed_reason == ""
    finally:
        engine.shutdown()


def test_t3_level_3_sweep_condensation_writes_no_node(tmp_path, monkeypatch):
    engine = _engine(tmp_path, condensation_fanin=2, summary_prefix_target_tokens=100)
    stub = _summaries(monkeypatch, level=3)
    _depth_0_nodes(engine, 2)
    try:
        before = _frontier_ids(engine)
        passes, reason = engine._run_threshold_sweep_condensation(
            target_tokens=100, pass_budget=5, deadline=time.monotonic() + 100.0)
        assert (passes, reason) == (0, "summary_result_rejected")
        assert _frontier_ids(engine) == before and len(stub.calls) == 1
    finally:
        engine.shutdown()


# -- T4: a partial sweep condenses an oversized prefix before its leaves -------------------------------------

def test_t4_partial_sweep_condenses_before_the_leaves(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path, condensation_fanin=2)
    stub = _summaries(monkeypatch)
    _depth_0_nodes(engine, 6)
    try:
        before = engine._summary_frontier_tokens()
        assert before > 400  # over the sweep target (leaf_chunk_tokens)
        _compress(engine, _view(40), caplog)
        assert engine._summary_frontier_tokens() < before
        assert stub.calls[0][0] == 1  # a condensation runs before the first leaf
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["stop_reason"] == "pass_budget_exhausted" and telemetry["status"] == "partial"
        assert telemetry["pre_leaf_condensation_passes"] > 0
        assert telemetry["total_passes"] == telemetry["leaf_passes"] + telemetry["condensation_passes"]
        assert telemetry["total_passes"] <= lcm_compaction._THRESHOLD_FULL_SWEEP_MAX_PASSES
        assert len(stub.calls) == telemetry["total_passes"]
    finally:
        engine.shutdown()


def test_t4_pre_leaf_rejection_skips_the_post_drain_condensation(tmp_path, monkeypatch, caplog):
    engine = _engine(tmp_path, condensation_fanin=2)
    calls: list[int] = []

    def summarize(**kwargs):  # leaves are accepted; every condensation comes back as a level 3 truncation
        depth = int(kwargs.get("depth", 0))
        calls.append(depth)
        return (LEAF, 1) if depth == 0 else (LEVEL_3, 3)

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    _depth_0_nodes(engine, 6)
    try:
        before = _frontier_ids(engine)
        _compress(engine, _view(), caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert telemetry["pre_leaf_condensation_stop_reason"] == "summary_result_rejected"
        assert telemetry["leaf_passes"] > 0 and calls.count(0) == telemetry["leaf_passes"]
        assert [depth for depth in calls if depth > 0] == [1]  # one condensation call in this compress()
        assert telemetry["post_drain_condensation_skipped"] == "pre_leaf_rejected"
        assert telemetry["stop_reason"] == "summary_result_rejected" and telemetry["status"] == "partial"
        assert set(before) <= set(_frontier_ids(engine))  # the rejected group stays on the frontier
    finally:
        engine.shutdown()


# -- T5: at or below the target nothing changes --------------------------------------------------------------

# Recorded at the base of this change (v0.24.6 for these paths): per sweep shape, the summariser calls as
# (depth, source_tokens), the telemetry passes and stop reason, and a digest of the returned context.
# The digests were re-pinned for #680: the LCM system note in the returned context gained one sentence.
# #998: the digests cover the in-context LCM note, which now names lcm_recall.
T5_EXPECTED = {
    "partial": ([(0, 361)] * 12, (12, 0, 12, "pass_budget_exhausted"),
                "0dc25fc014af98cd7a86651406a47c2a278453983eab785b8d2cfce574ca1184"),
    "drained": ([(0, 361)] * 5 + [(1, 260)], (5, 1, 6, "summary_prefix_target_reached"),
                "9947322c653cf1241eba6d07e6f39ce02ca079c32dde8d870933af180ee14c1d"),
}


def _digest(result) -> str:
    return hashlib.sha256(json.dumps([[m.get("role"), str(m.get("content"))] for m in result]).encode()).hexdigest()


@pytest.mark.parametrize("shape", ["partial", "drained"])
def test_t5_at_or_below_the_target_the_sweep_is_unchanged(tmp_path, monkeypatch, caplog, shape):
    engine = _engine(tmp_path, condensation_fanin=2)
    stub = _summaries(monkeypatch)
    _depth_0_nodes(engine, 3, token_count=130)  # 390 tokens: at or below the 400 token sweep target
    try:
        result = _compress(engine, _view(20 if shape == "partial" else 6), caplog)
        telemetry = engine.get_status()["threshold_full_sweep"]
        passes = (telemetry["leaf_passes"], telemetry["condensation_passes"], telemetry["total_passes"],
                  telemetry["stop_reason"])
        assert (stub.calls, passes, _digest(result)) == T5_EXPECTED[shape]
        assert telemetry.get("pre_leaf_condensation_passes", 0) == 0
    finally:
        engine.shutdown()


# -- T6: several partial sweeps keep the prefix bounded ------------------------------------------------------

def test_t6_partial_sweeps_do_not_grow_the_prefix(tmp_path, monkeypatch, caplog):
    """#605: the pre-leaf split is no longer half the passes and half the time; each sweep is partial by its
    60 s soft target, so the summariser takes 19 s per call on the clock (a call that takes no time would let
    every later leaf in under the target while the condensation passes are bounded by time)."""
    # #605 F2: the 150-token leaves are over a lowered level 3 bound, so each one calls (and takes its 19 s).
    engine = _engine(tmp_path, condensation_fanin=2, leaf_chunk_tokens=150, l3_truncate_tokens=2)
    long_leaf = "Earlier turns." + " summary words" * 20 + "\nExpand for details about: turns"
    offset = [0.0]
    real_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: real_monotonic() + offset[0])

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        offset[0] += 19.0
        return long_leaf

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    view = _view(20)
    sizes = []
    try:
        for round_index in range(6):
            view = _compress(engine, view, caplog)
            sizes.append(engine._summary_frontier_tokens())
            assert engine.get_status()["threshold_full_sweep"]["stop_reason"] == "soft_target_reached"
            view = [*view, *_turns(100 + 20 * round_index, 20)]
        target = engine.get_status()["threshold_full_sweep"]["summary_prefix_target_tokens"]
        assert sizes[0] > target  # the first sweep leaves the prefix over its target
        # The first sweep has no prefix to condense, so the first two sweeps set the bound (#605).
        assert all(size <= max(target, *sizes[:2]) for size in sizes), sizes
    finally:
        engine.shutdown()


# -- T7: assembly keeps the newest uncondensed nodes ---------------------------------------------------------

def _rendered(result) -> list[tuple[int, int]]:
    return [(int(d), int(n)) for m in result for d, n in PART.findall(str(m.get("content")))]


def test_t7_assembly_renders_the_newest_nodes_oldest_first(tmp_path, monkeypatch):
    engine = _engine(tmp_path, condensation_fanin=2, threshold_full_sweep_enabled=False)
    _summaries(monkeypatch)
    ids = _depth_0_nodes(engine, 130, token_count=5)
    tail = _turn("tail", 5000.0)
    try:
        rendered = [node_id for _depth, node_id in _rendered(engine._assemble_context(None, tail))]
        assert rendered == ids[-100:]  # the newest 100, oldest first
        assert [node.node_id for node in engine._dag.get_uncondensed_at_depth("S", 0)] == ids[:100]
        assert engine._maybe_condense() >= 1
        condensed = [node for node in _nodes(engine) if node.depth == 1]
        assert condensed[0].source_ids == ids[:2]  # ordinary condensation still takes the oldest fanin
    finally:
        engine.shutdown()


# -- T8: a summary budget keeps the newest parts of a depth --------------------------------------------------

def test_t8_summary_budget_keeps_the_newest_parts(tmp_path, monkeypatch):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=False)
    ids = _depth_0_nodes(engine, 10, token_count=60, text="group" + " summary words" * 40)
    tail = _turn("tail", 5000.0)
    seen: list[set] = []
    original = engine._build_proactive_recall_message
    monkeypatch.setattr(engine, "_build_proactive_recall_message",
                        lambda messages, role, active: seen.append(set(active)) or original(messages, role, active))
    try:
        tail_tokens = tokens.count_messages_tokens(tail)
        result = engine._assemble_context(None, tail, assembly_cap_override=tail_tokens + 450)
        rendered = [node_id for _depth, node_id in _rendered(result)]
        assert 0 < len(rendered) < len(ids)
        assert rendered == ids[-len(rendered):]  # the newest parts, in today's order
        assert seen == [set(rendered)]  # the dedupe set is exactly the rendered nodes
    finally:
        engine.shutdown()
