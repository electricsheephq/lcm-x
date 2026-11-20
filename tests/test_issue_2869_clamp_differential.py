"""#2869 K3: a randomised multi-sweep simulation of raised mode, checked at EVERY condensation check.

Each case walks one session through many automatic sweeps: random window, threshold, leaf chunk, live and condensed
history, summary sizes (skewed: big early, small later), stored token counts of 0 on some leaves, and ASCII or CJK text
priced by the default or the fallback counter. The checks (spies on this engine's own methods):
- P1, a HARD bar: every condensation starts with the frontier over today's target, so main would condense there too;
- P2: a check that stops condensing in raised mode has at most 80 live nodes, or the frontier at or under today's target;
- P3: a raised sweep that starts after a normal stop with at most 80 live holds at most 80 plus the leaves stored
  since, which is at most 12 (one sweep's passes), so no level passes 92 there.
- Unengaged sweeps (480,000 and below) never evaluate the raised rule.
LCMX_2869_SIM_CASES sets the case count (default 8; the bench card runs 200).
"""

from __future__ import annotations

import os
import random

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine

pytestmark = pytest.mark.issue(2869)

CASES = int(os.environ.get("LCMX_2869_SIM_CASES", "8"))
MAX_LIVE, SWEEP_LEAVES = 80, 12
QUIET = {"embeddings_enabled": False, "temporal_rollups_enabled": False, "large_output_externalization_enabled": False,
         "large_output_active_replay_stubbing_enabled": False}


def _text(rng, tokens_, script):
    if script == "cjk":
        return "漢" * max(1, int(tokens_ * 1.5))
    return "x" * max(1, 4 * tokens_)


@pytest.mark.parametrize("seed", range(CASES))
def test_k3_count_bound_simulation(tmp_path, monkeypatch, seed):
    rng = random.Random(28_690_000 + seed)
    window = rng.choice([480_000, 600_000, 1_000_000, 1_000_000, 2_000_000])
    script = rng.choice(["ascii", "cjk"])
    if rng.random() < 0.5:
        monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    big_n, big, small = rng.randint(0, 60), rng.choice([600, 1_000, 1_500, 2_500]), rng.choice([50, 150, 300, 500])
    zero_every = rng.choice([0, 0, 5, 10])
    calls = {"leaf": 0, "depth0": 0}

    def summarize(**kwargs):
        if int(kwargs.get("depth", 0) or 0) == 0:
            calls["leaf"] += 1
            size = big if calls["leaf"] <= big_n else small
        else:
            size = rng.choice([small, small, big // 2])
        return _text(rng, max(1, int(size * rng.uniform(0.6, 1.4))), script), 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    add = SummaryDAG.add_node

    def add_node(self, node, *args, **kwargs):
        if int(node.depth) == 0:
            calls["depth0"] += 1
            if zero_every and calls["depth0"] % zero_every == 0:
                node.token_count = 0  # a stored count that reads 0
        return add(self, node, *args, **kwargs)

    monkeypatch.setattr(SummaryDAG, "add_node", add_node)
    chunk = rng.choice([1_500, 4_000, 8_000, 24_000])
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), leaf_chunk_tokens=chunk,
                                        context_threshold=rng.choice([0.5, 0.6, 0.75]), fresh_tail_count=4,
                                        dynamic_leaf_chunk_enabled=False, **QUIET),
                       hermes_home=str(tmp_path))
    log, state = [], {"since": 0, "last": None, "target": 0, "raised": 0}
    prepare, due, publish = engine._prepare_compression_sweep, engine._sweep_condensation_due, engine._publish_leaf_node
    condense, run = engine._condense_summary_nodes, engine._run_threshold_sweep_condensation

    def frontier():
        return engine._summary_frontier_tokens(with_count=True)

    def prepare_spy(sweep_state, *args, **kwargs):
        out = prepare(sweep_state, *args, **kwargs)
        if sweep_state.threshold_full_sweep_active:
            state["target"] = sweep_state.sweep_target_tokens
            state["raised"] = getattr(sweep_state.budget, "sweep_raise_tokens", 0)
            log.append(("start", *frontier(), state["since"], state["last"], state["raised"],
                        len(engine._dag.get_uncondensed_at_depth("S", 0, limit=10 ** 6))))
        return out

    def due_spy(target, raise_tokens):
        result = due(target, raise_tokens)
        tokens_, live = frontier()
        log.append(("check", tokens_, live, target, raise_tokens, result))
        if result:
            state["last"] = "pending"
        else:
            state["since"], state["last"] = 0, "live" if live <= MAX_LIVE else "floor"
        return result

    def publish_spy(*args, **kwargs):
        out = publish(*args, **kwargs)
        state["since"] += 1
        return out

    def condense_spy(*args, **kwargs):
        log.append(("condense", *frontier(), state["target"], state["raised"]))
        return condense(*args, **kwargs)

    def run_spy(*args, **kwargs):
        out = run(*args, **kwargs)
        if out[1] != "summary_prefix_target_reached":
            state["last"] = "abnormal"
        return out

    for name, spy in (("_prepare_compression_sweep", prepare_spy), ("_sweep_condensation_due", due_spy),
                      ("_publish_leaf_node", publish_spy), ("_condense_summary_nodes", condense_spy),
                      ("_run_threshold_sweep_condensation", run_spy)):
        monkeypatch.setattr(engine, name, spy)
    try:
        engine.on_session_start("S", platform="cli", context_length=window, conversation_id="conv")
        clock = 1.0
        for _ in range(rng.randint(0, 5)):  # condensed history
            children = []
            for _ in range(4):
                clock += 1
                text = _text(rng, rng.choice([small, big]), script)
                children.append(engine._dag.add_node(SummaryNode(
                    session_id="S", depth=0, summary=text, token_count=tokens.count_tokens(text), source_ids=[],
                    created_at=clock, expand_hint="turns")))
            clock += 1
            text = _text(rng, small, script)
            engine._dag.add_node(SummaryNode(session_id="S", depth=1, summary=text, token_count=tokens.count_tokens(text),
                                             source_ids=children, source_type="nodes", created_at=clock,
                                             expand_hint="turns"))
        row = " alpha beta gamma delta" * rng.choice([60, 160])
        view, turn = [], 0
        for _sweep in range(rng.randint(8, 16)):
            for _ in range(rng.randint(10, 40)):
                view += [{"role": "user", "content": f"[U{turn}] user turn{row}"},
                         {"role": "assistant", "content": f"[A{turn}] reply{row}"}]
                turn += 1
            engine.ingest(view)
            view = engine.compress(view, current_tokens=engine.threshold_tokens + rng.randint(0, 120_000))
            engine.note_turn_complete()
            engine._clear_persisted_no_progress_hold()
            engine._no_progress_hold = None
    finally:
        engine.shutdown()

    case = f"seed {seed} window {window} script {script} chunk {chunk} big {big_n}x{big} small {small} zero {zero_every}"
    p1 = [e for e in log if e[0] == "condense" and not e[1] > e[3]]
    assert not p1, f"P1 (HARD): a condensation main would not start, {case}: {p1[:3]}"
    p2 = [e for e in log if e[0] == "check" and not e[5] and not (e[2] <= MAX_LIVE or e[1] <= e[3])]
    assert not p2, f"P2, {case}: {p2[:3]}"
    starts = [e for e in log if e[0] == "start"]
    p3 = [e for e in starts if e[5] and e[4] == "live" and not (e[2] <= MAX_LIVE + e[3] and e[3] <= SWEEP_LEAVES)]
    assert not p3, f"P3, {case}: {p3[:3]}"
    assert not any(e[0] == "check" for e in log) or window > 480_000, f"an unengaged sweep evaluated the rule, {case}"
    assert starts, f"no sweep ran, {case}"
    if os.environ.get("LCMX_2869_SIM_REPORT"):
        raised = [e for e in starts if e[5]]
        print(f"CASE {case}: sweeps {len(starts)} raised {len(raised)} condensations "
              f"{sum(e[0] == 'condense' for e in log)} checks {sum(e[0] == 'check' for e in log)} "
              f"P3-applied {sum(e[4] == 'live' for e in raised)} max-start-live {max((e[2] for e in raised), default=0)}")
