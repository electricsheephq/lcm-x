"""A frontier read lock preserves already committed threshold condensation telemetry."""
import sqlite3

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def test_committed_condensation_frontier_lock_reports_completed_pass(tmp_path, monkeypatch):
    import hermes_lcm.engine as engine_module

    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=2,
        threshold_full_sweep_enabled=True, summary_prefix_target_tokens=1500,
        condensation_fanin=4,
    ))
    engine._session_id = "test-session"
    engine.threshold_tokens = 1
    for i in range(4):
        engine._dag.add_node(SummaryNode(
            session_id="test-session", depth=1, summary=f"durable fact {i}",
            token_count=1000, source_token_count=2000, source_ids=[],
            source_type="messages", created_at=i,
        ))
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kwargs: ("condensed facts", 1))
    original_condense = engine._condense_summary_nodes
    original_frontier = engine._summary_frontier_tokens
    committed = False
    lock = sqlite3.OperationalError("database is locked")

    def condense(*args, **kwargs):
        nonlocal committed
        result = original_condense(*args, **kwargs)
        committed = True
        return result

    def frontier():
        nonlocal committed
        if committed:
            committed = False
            raise lock
        return original_frontier()

    monkeypatch.setattr(engine, "_condense_summary_nodes", condense)
    monkeypatch.setattr(engine, "_summary_frontier_tokens", frontier)
    messages = [{"role": "system", "content": "system"},
                {"role": "user", "content": "fresh request"},
                {"role": "assistant", "content": "fresh answer"}]
    try:
        result = engine.compress(messages, current_tokens=count_messages_tokens(messages))
        assert lock.lcm_completed_condensation_passes == 1
        assert engine.get_status()["threshold_full_sweep"]["condensation_passes"] == 1
        assert any(n.depth == 2 for n in engine._dag.get_session_nodes("test-session"))
        assert result[-2:] == messages[-2:]
    finally:
        engine.shutdown()
