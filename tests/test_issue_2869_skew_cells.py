"""#2869 review cells (red proof on the clamped target; local): a store whose mean depth-0 summary differs from the live
frontier the sweep compares with its target.

Bar for the count cells: after one automatic sweep at a 1M window, no level holds more than the 100 uncondensed
summaries the view shows (dag.py get_uncondensed_at_depth limit=100); a level past 100 leaves the rows under its oldest
summaries out of the list with no pointer (A0). Bar for the unit cell: the raise fires for a store whose summaries cost
the same in the sweep's own unit (node token counts) as an ASCII store's.
"""

from __future__ import annotations

import pytest

import hermes_lcm.engine as engine_module
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

pytestmark = pytest.mark.issue(2869)

LIMIT = 100
PAD = " alpha beta gamma delta" * 160  # about 800 tokens a row
SMALL = "y" * 1_200  # the stand-in's summary: 301 tokens
QUIET = {"embeddings_enabled": False, "temporal_rollups_enabled": False, "large_output_externalization_enabled": False,
         "large_output_active_replay_stubbing_enabled": False}  # as the grid runner: no stub-first exit


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: (SMALL, 1))
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2, **QUIET),
                         hermes_home=str(tmp_path))
    instance.on_session_start("S", platform="cli", context_length=1_000_000, conversation_id="conv")
    yield instance
    instance.shutdown()


_clock = [1.0]


def _node(engine, depth, tokens_, text=None, sources=(), source_type="messages"):
    _clock[0] += 1.0
    text = text if text is not None else "x" * (4 * tokens_)
    return engine._dag.add_node(SummaryNode(
        session_id="S", depth=depth, summary=text, token_count=tokens_, source_token_count=4 * tokens_,
        source_ids=list(sources), source_type=source_type, created_at=_clock[0], expand_hint="turns"))


def _live(engine, depth):
    return len(engine._dag.get_uncondensed_at_depth("S", depth, limit=10 ** 6))


def _sweep(engine):
    view = [row for i in range(6) for row in ({"role": "user", "content": f"[U{i}] question{PAD}"},
                                              {"role": "assistant", "content": f"[A{i}] answer{PAD}"})]
    engine.ingest(view)
    engine.compress(view, current_tokens=engine.threshold_tokens)
    status = engine.get_status()["threshold_full_sweep"]
    return status["summary_prefix_target_tokens"], status


def test_skewed_history_condensed_big_leaves_then_100_live_small(engine):
    """20 old 2,400-token leaves condensed into five 600-token parents, then 100 live 300-token leaves: a historical
    leaf mean of 650 permits 50,000 against a live frontier of 33,000, so one more leaf passes the view's 100."""
    old = [_node(engine, 0, 2_400) for _ in range(20)]
    for group in range(5):
        _node(engine, 1, 600, sources=old[4 * group:4 * group + 4], source_type="nodes")
    for _ in range(100):
        _node(engine, 0, 300)
    assert _live(engine, 0) == 100 and engine._summary_frontier_tokens() == 33_000
    target, status = _sweep(engine)
    assert status["leaf_passes"] >= 1, status  # premise: the sweep stored a leaf
    assert _live(engine, 0) <= LIMIT, (f"target {target}: {_live(engine, 0)} live depth-0 summaries, "
                                       f"{_live(engine, 0) - LIMIT} out of view; status {status}")


def test_live_skew_old_big_new_small_all_uncondensed(engine):
    """All leaves live: 20 old 2,000-token leaves, then 80 new 100-token leaves (100 live, frontier 48,000)."""
    for _ in range(20):
        _node(engine, 0, 2_000)
    for _ in range(80):
        _node(engine, 0, 100)
    assert _live(engine, 0) == 100 and engine._summary_frontier_tokens() == 48_000
    target, status = _sweep(engine)
    assert status["leaf_passes"] >= 1, status  # premise: the sweep stored a leaf
    assert _live(engine, 0) <= LIMIT, (f"target {target}: {_live(engine, 0)} live depth-0 summaries; status {status}")


def _target_for(tmp_path, monkeypatch, name, text):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: (SMALL, 1))
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / f"{name}.db"), fresh_tail_count=2, **QUIET),
                         hermes_home=str(tmp_path / name))
    try:
        instance.on_session_start("S", platform="cli", context_length=1_000_000, conversation_id="conv")
        cost = tokens.count_tokens(text)  # the sweep's unit: the node token count publication records
        for _ in range(10):
            _node(instance, 0, cost, text=text)
        return _sweep(instance)[0], cost
    finally:
        instance.shutdown()


def test_cjk_summaries_raise_like_an_ascii_store_of_the_same_sweep_cost(tmp_path, monkeypatch):
    """With the fallback counter, 1,200 CJK characters cost about 801 tokens in the sweep's unit; an ASCII store whose
    summaries cost the same must get the same target."""
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    cjk_target, cjk_cost = _target_for(tmp_path, monkeypatch, "cjk", "漢" * 1_200)
    ascii_text = "z" * (4 * (cjk_cost - 1))
    ascii_target, ascii_cost = _target_for(tmp_path, monkeypatch, "ascii", ascii_text)
    assert ascii_cost == cjk_cost > 700, (ascii_cost, cjk_cost)
    assert cjk_target == ascii_target, f"CJK target {cjk_target}, ASCII target {ascii_target} at cost {cjk_cost}"
