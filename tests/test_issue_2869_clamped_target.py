"""#2869 raised mode at engine level: above 480,000-token windows an automatic sweep condenses only while the frontier is
over today's target AND (over the raised target, min(5% of the window, the threshold cap), OR more than 80 nodes are
live), re-evaluated at every check. Off on forced, overflow and recovery passes, with an explicit target and with
background leaf preparation on, where every pass is byte-identical to main.

Main's behaviour is this tree's with raised mode switched off (``_LARGE_SWEEP_WINDOW_TOKENS`` past every window).
K2: four big live leaves condensed under the raised target with 96 live, then five more leaves: never 101 live.
K4: the reference agent's shape gets the raised target and condenses no more than main.
K5: the unchanged paths. K7: the frontier read returns the same tokens and its node count from one read.
"""

from __future__ import annotations

import pytest

import hermes_lcm.compaction as compaction
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

pytestmark = pytest.mark.issue(2869)

MESSAGES = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"FACT-{i} " + "alpha beta gamma delta " * 44}
            for i in range(48)] + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i}"}
                                   for i in range(24)]
QUIET = {"embeddings_enabled": False, "temporal_rollups_enabled": False, "large_output_externalization_enabled": False,
         "large_output_active_replay_stubbing_enabled": False}
PAD = " alpha beta gamma delta" * 160  # about 800 tokens a row


@pytest.fixture
def open_engine(tmp_path, monkeypatch):
    engines, clock = [], [1.0]

    def open_at(window, nodes=(), **config):
        home = tmp_path / f"h{len(engines)}"
        engine = LCMEngine(config=LCMConfig(database_path=str(home / "lcm.db"), **{**QUIET, **config}),
                           hermes_home=str(home))
        engine.on_session_start("S", platform="cli", context_length=window, conversation_id="conv")
        for depth, tokens, sources in nodes:  # (depth, stored token count, the node ids it condenses)
            clock[0] += 1.0
            engine._dag.add_node(SummaryNode(
                session_id="S", depth=depth, summary="x" * (4 * tokens), token_count=tokens, source_token_count=5,
                source_ids=list(sources), source_type="nodes" if sources else "messages", created_at=clock[0],
                expand_hint="turns"))
        engines.append(engine)
        return engine

    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: ("y" * 800, 1))  # 200 tokens
    yield open_at
    for engine in engines:
        engine.shutdown()


def _live(nodes):
    return [(0, tokens, ()) for tokens in nodes]


def _pass(engine, monkeypatch, *, messages=MESSAGES, current=None, main=False, **kwargs):
    """One compress pass: (today's target, raised target, sweep active, overflow, condensations, checks)."""
    seen, checks, condensed = [], [], [0]
    prepare, due, condense = engine._prepare_compression_sweep, engine._sweep_condensation_due, engine._condense_summary_nodes

    def wrapped(state, *args, **kw):
        out = prepare(state, *args, **kw)
        seen.append((state.sweep_target_tokens, getattr(state.budget, "sweep_raise_tokens", 0),
                     state.threshold_full_sweep_active, state.force_overflow))
        return out

    def due_spy(target, raise_tokens):
        result = due(target, raise_tokens)
        checks.append((result, *engine._summary_frontier_tokens(with_count=True), target))
        return result

    def condense_spy(*args, **kw):
        condensed[0] += 1
        return condense(*args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_prepare_compression_sweep", wrapped)
        patch.setattr(engine, "_sweep_condensation_due", due_spy)
        patch.setattr(engine, "_condense_summary_nodes", condense_spy)
        if main:
            patch.setattr(compaction, "_LARGE_SWEEP_WINDOW_TOKENS", 10 ** 15)
        engine.ingest(messages)
        engine.compress(messages, current_tokens=current or engine.threshold_tokens, **kwargs)
    return (*seen[0], condensed[0], checks)


def _today(engine):
    return min(int(engine._config.leaf_chunk_tokens), int(engine.threshold_tokens * 0.25))


def _live_count(engine, depth=0):
    return len(engine._dag.get_uncondensed_at_depth("S", depth, limit=10 ** 6))


# -- K4 and K5: the raised target, and the unchanged paths ---------------------------------------------------------------

@pytest.mark.parametrize("window, raised", [(1_000_000, 50_000), (2_000_000, 100_000)])
def test_k5_raised_target_is_the_window_share(open_engine, monkeypatch, window, raised):
    engine = open_engine(window, _live([1_590] * 10))
    target, raise_tokens, active, overflow, _n, _checks = _pass(engine, monkeypatch)
    assert (target, raise_tokens, active, overflow) == (_today(engine), raised, True, False)


@pytest.mark.parametrize("nodes, head_condensations", [([1_590] * 30, 0), ([1_590] * 33, None)])
def test_k4_reference_shape_condenses_no_more_than_main(open_engine, monkeypatch, nodes, head_condensations):
    """30 live 1,590-token nodes (47,700): main condenses at 24,000, raised mode does not. 33 (52,470): both do, and
    raised mode no more than main."""
    head = _pass(open_engine(1_000_000, _live(nodes)), monkeypatch)
    main = _pass(open_engine(1_000_000, _live(nodes)), monkeypatch, main=True)
    assert main[4] > 0 and head[4] <= main[4], (head[4], main[4])
    if head_condensations is not None:
        assert head[4] == head_condensations
    assert all(tokens > target for due, tokens, _live_n, target in head[5] if due)  # P1 at every check


@pytest.mark.parametrize("window", [64_000, 272_000, 480_000])
def test_k5_unchanged_at_or_below_480k(open_engine, monkeypatch, window):
    engine = open_engine(window, _live([1_590] * 10))
    target, raise_tokens, *_rest, checks = _pass(engine, monkeypatch)
    assert (target, raise_tokens, checks) == (_today(engine), 0, [])


@pytest.mark.parametrize("mode", ["forced", "recovery", "prep"])
def test_k5_unchanged_forced_recovery_and_prep(open_engine, monkeypatch, mode):
    engine = open_engine(1_000_000, _live([1_590] * 10), background_leaf_prep_enabled=mode == "prep")
    kwargs = {"force": True} if mode == "forced" else {"bypass_cooldown": True} if mode == "recovery" else {}
    target, raise_tokens, *_rest, checks = _pass(engine, monkeypatch, **kwargs)
    assert (target, raise_tokens, checks) == (_today(engine), 0, [])


def test_k5_unchanged_overflow(open_engine, monkeypatch):
    engine = open_engine(1_000_000, _live([1_590] * 10), max_assembly_tokens=1_000_000)
    target, raise_tokens, _active, overflow, _n, checks = _pass(engine, monkeypatch, current=1_200_000)
    assert overflow and (target, raise_tokens, checks) == (_today(engine), 0, [])


@pytest.mark.parametrize("config, expected", [({"summary_prefix_target_tokens": 30_000}, 30_000),
                                              ({"summary_prefix_target_ratio": 0.01}, 10_000)])
def test_k5_unchanged_explicit_settings(open_engine, monkeypatch, config, expected):
    engine = open_engine(1_000_000, _live([1_590] * 10), **config)
    target, raise_tokens, *_rest, checks = _pass(engine, monkeypatch)
    assert (target, raise_tokens, checks) == (expected, 0, [])


# -- K2: Astra's case, checked at every check -----------------------------------------------------------------------------

def test_k2_condensed_big_leaves_then_96_live_then_five_leaves_never_101(open_engine, monkeypatch):
    """Two 200-token parents over eight historical leaves, four oldest live 2,500-token leaves, 96 live 300-token
    leaves (102 live, 39,200 tokens), then a pass with five eligible leaf chunks of 200-token leaves."""
    history = [(0, 2_500, ())] * 8
    engine = open_engine(1_000_000, history, context_threshold=0.5, fresh_tail_count=2)
    leaves = [node.node_id for node in engine._dag.get_session_nodes("S", limit=100)]
    for group in range(2):
        engine._dag.add_node(SummaryNode(session_id="S", depth=1, summary="x" * 800, token_count=200,
                                         source_ids=leaves[4 * group:4 * group + 4], source_type="nodes",
                                         created_at=50.0 + group, expand_hint="turns"))
    for index, tokens in enumerate([2_500] * 4 + [300] * 96):
        engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary="x" * (4 * tokens), token_count=tokens,
                                         source_ids=[], created_at=100.0 + index, expand_hint="turns"))
    assert engine._summary_frontier_tokens(with_count=True) == (39_200, 102)
    rows = [row for i in range(76) for row in ({"role": "user", "content": f"[U{i}] question{PAD}"},
                                               {"role": "assistant", "content": f"[A{i}] answer{PAD}"})]
    target, raise_tokens, *_rest, condensed, checks = _pass(engine, monkeypatch, messages=rows,
                                                            current=engine.threshold_tokens + 140_000)
    assert (target, raise_tokens) == (24_000, 50_000) and condensed > 0, (target, raise_tokens, condensed)
    stops = [(tokens, live) for due, tokens, live, _t in checks if not due]
    assert stops and all(live <= 80 or tokens <= target for tokens, live in stops), stops  # P2 at every check
    assert all(tokens > target for due, tokens, _live_n, _t in checks if due)  # P1 at every check
    status = engine.get_status()["threshold_full_sweep"]
    assert status["leaf_passes"] >= 5, status
    assert _live_count(engine) <= 92, _live_count(engine)


# -- K7: the frontier read --------------------------------------------------------------------------------------------

def test_k7_frontier_read_same_tokens_and_its_count(open_engine):
    engine = open_engine(1_000_000, _live([0, 5, 1_590, 300]) + [(1, 700, ())])
    nodes = engine._summary_frontier_nodes()
    tokens = sum(node.token_count for node in nodes)
    assert engine._summary_frontier_tokens() == tokens == 2_595
    assert engine._summary_frontier_tokens(with_count=True) == (tokens, len(nodes)) == (2_595, 5)


def test_k7_unengaged_pass_never_reads_the_count(open_engine, monkeypatch):
    """At 272,000 the frontier is over today's target and the sweep condenses through main's comparison only."""
    engine = open_engine(272_000, _live([1_590] * 40))
    calls = []
    frontier = engine._summary_frontier_tokens
    monkeypatch.setattr(engine, "_summary_frontier_tokens", lambda **kw: (calls.append(kw), frontier(**kw))[1])
    _target, raise_tokens, *_rest, condensed, checks = _pass(engine, monkeypatch)
    assert raise_tokens == 0 and checks == [] and condensed > 0
    assert calls and all(kw == {} for kw in calls), calls
