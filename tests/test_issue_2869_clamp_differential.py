"""#2869: a randomised differential of the clamped summary-prefix target against main's target, on one tree.

Main's target is this tree's with the large-window rule switched off (``_LARGE_SWEEP_WINDOW_TOKENS`` past every window).
Each case seeds a random store: window, threshold, live and condensed history, per-node sizes including 0, ASCII and
CJK text, priced by the default counter or the fallback counter (the node token count publication would record). Each
case captures the target at sweep start and stops the pass there. Bars, on every case:
- the target is never below main's, and equals it at 480,000 and below, forced, overflow and with preparation on;
- at sweep start, more than 80 live frontier nodes always exceed the target (the sweep condenses before any leaf),
  unless main's target binds;
- the one-statement live frontier read equals the frontier the sweep sums (engine._summary_frontier_nodes).
LCMX_2869_DIFF_CASES sets the case count (default 300).
"""

from __future__ import annotations

import os
import random

import pytest

import hermes_lcm.compaction as compaction
import hermes_lcm.engine as engine_module
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

pytestmark = pytest.mark.issue(2869)

CASES = int(os.environ.get("LCMX_2869_DIFF_CASES", "300"))
WINDOWS = [64_000, 272_000, 480_000, 480_001, 600_000, 1_000_000, 2_000_000]
MESSAGES = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"row {i} " + "alpha beta " * 40}
            for i in range(12)]


class _Stop(BaseException):
    """Ends the pass at sweep start: only the target is read."""


def _capture(engine, monkeypatch, *, force=False, current=None, main=False):
    seen, prepare = [], engine._prepare_compression_sweep

    def wrapped(state, *args, **kwargs):
        prepare(state, *args, **kwargs)
        seen.append((state.sweep_target_tokens, state.threshold_full_sweep_active, state.force_overflow))
        raise _Stop

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_prepare_compression_sweep", wrapped)
        if main:
            patch.setattr(compaction, "_LARGE_SWEEP_WINDOW_TOKENS", 10 ** 15)
        with pytest.raises(_Stop):
            engine.compress(MESSAGES, current_tokens=current or engine.threshold_tokens, force=force)
    return seen[0]


def _text(rng):
    size = rng.choice([0, 1, rng.randint(1, 400), rng.randint(400, 4_000), rng.randint(4_000, 12_000)])
    unit = rng.choice(["x", "漢", "a漢"])
    return (unit * size)[:size] or "."


def _seed(engine, rng, count_tokens):
    clock = 1.0
    for _ in range(rng.randint(0, 6)):  # condensed history: leaves under one parent each group
        children = []
        for _ in range(rng.randint(2, 6)):
            text = _text(rng)
            clock += 1
            children.append(engine._dag.add_node(SummaryNode(
                session_id="S", depth=0, summary=text, token_count=count_tokens(text), source_ids=[],
                created_at=clock, expand_hint="turns")))
        text = _text(rng)
        clock += 1
        engine._dag.add_node(SummaryNode(session_id="S", depth=1, summary=text, token_count=count_tokens(text),
                                         source_ids=children, source_type="nodes", created_at=clock,
                                         expand_hint="turns"))
    for _ in range(rng.choice([0, rng.randint(1, 40), rng.randint(60, 140), rng.randint(140, 260)])):
        text = _text(rng)
        clock += 1
        size = 0 if rng.random() < 0.1 else count_tokens(text)  # a node that reads 0
        engine._dag.add_node(SummaryNode(session_id="S", depth=rng.choice([0, 0, 0, 1, 2]), summary=text,
                                         token_count=size, source_ids=[], created_at=clock, expand_hint="turns"))


@pytest.mark.parametrize("seed", range(CASES))
def test_clamp_differential(tmp_path, monkeypatch, seed):
    rng = random.Random(2869 * 100_003 + seed)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: ("Earlier.\nExpand: x", 1))
    count_tokens = tokens._fallback_token_estimate if rng.random() < 0.5 else tokens.count_tokens
    window = rng.choice(WINDOWS)
    mode = rng.choice(["auto"] * 6 + ["forced", "overflow", "prep", "explicit"])
    config = {"database_path": str(tmp_path / "lcm.db"), "context_threshold": rng.choice([0.5, 0.6, 0.75, 0.9]),
              "embeddings_enabled": False, "temporal_rollups_enabled": False, "max_assembly_tokens": window,
              "background_leaf_prep_enabled": mode == "prep"}
    if mode == "explicit":
        config["summary_prefix_target_tokens"] = rng.randint(1_000, 90_000)
    engine = LCMEngine(config=LCMConfig(**config), hermes_home=str(tmp_path))
    try:
        engine.on_session_start("S", platform="cli", context_length=window)
        _seed(engine, rng, count_tokens)
        frontier = engine._summary_frontier_nodes()
        live, total = engine._dag.live_frontier_tokens("S")
        assert (live, total) == (len(frontier), sum(node.token_count for node in frontier))
        kwargs = {"force": mode == "forced", "current": window + 1 if mode == "overflow" else None}
        head, active, overflow = _capture(engine, monkeypatch, **kwargs)
        main = _capture(engine, monkeypatch, main=True, **kwargs)[0]
        case = f"seed {seed}: window {window} mode {mode} live {live} total {total} head {head} main {main}"
        if os.environ.get("LCMX_2869_DIFF_REPORT"):
            print(f"CASE {case} raised {head > main} binds {live > 80 and head != main}")
        assert head >= main, case
        if mode != "auto" or window <= 480_000 or not active or overflow:
            assert head == main, case
        if live > 80 and head != main:
            assert total > head, case  # the sweep condenses before any leaf
    finally:
        engine.shutdown()
