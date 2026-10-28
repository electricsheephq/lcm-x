"""rc4: a no-progress hold cannot suppress leaf publication at the ceiling."""

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "count_tokens", tokens._fallback_token_estimate)
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kwargs: (
        "Earlier turns.\nExpand for details about: turns", 1))
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2,
        leaf_chunk_tokens=400, context_threshold=0.001,
        threshold_full_sweep_enabled=True, max_assembly_tokens=100_000))
    instance.on_session_start("parent", platform="acp", conversation_id="conv", context_length=6_000)
    yield instance
    instance.shutdown()


def _turns(start, count):
    pad = " alpha beta gamma delta" * 30
    return [row for i in range(start, start + count) for row in (
        {"role": "user", "content": f"turn {i}{pad}", "timestamp": float(i + 1)},
        {"role": "assistant", "content": f"reply {i}{pad}"})]


def _leaves(engine):
    return [node for node in engine._dag.get_session_nodes(engine._session_id, limit=100_000)
            if node.depth == 0]


def test_host_rejection_at_ceiling_stores_a_leaf(engine):
    view = _turns(0, 40)
    engine.ingest(view)
    engine.record_rejected_compaction()
    request = engine._survival_measure(view)
    assert request >= engine._survival_ceiling()
    assert engine._no_progress_hold_active() and not engine._sweep_budget_hold_active()
    engine.compress(view, current_tokens=request)
    assert _leaves(engine)
    assert (engine._last_compression_status, engine._last_compression_noop_reason) != ("noop", "held")
