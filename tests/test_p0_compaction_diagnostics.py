"""p0f: bounded publication diffs and truthful pre-leaf budget diagnostics."""

from __future__ import annotations

import logging
import time

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.lifecycle_state as lifecycle_state
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError


def _engine(tmp_path, *, sweep=False):
    engine = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
        temporal_rollups_enabled=False,
        fresh_tail_count=2,
        leaf_chunk_tokens=400,
        context_threshold=0.001,
        threshold_full_sweep_enabled=sweep,
        max_assembly_tokens=100_000,
        l3_truncate_tokens=2,
        condensation_fanin=2,
        summary_timeout_ms=60_000,
    ))
    engine.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    return engine


def test_p0_compaction_diagnostics_missing_row_survives_warning_cut(tmp_path):
    engine = _engine(tmp_path)
    try:
        owned = [engine._store.append(
            "S", {"role": "assistant", "content": "synthetic row content"}, conversation_id="conv"
        ) for _ in range(200)]
        missing_id = owned[-2]
        covered = [store_id for store_id in owned if store_id != missing_id]

        def stage(conn, node_id):
            engine._lifecycle.stage_compaction_publication(conn, "conv", "S", node_id, 0, covered)

        with pytest.raises(LifecyclePublicationConflictError, match="not contiguous") as exc:
            engine._dag.add_node(SummaryNode(
                session_id="S", summary="synthetic summary", token_count=1,
                source_token_count=200, source_ids=covered,
            ), before_commit=stage)
        message = str(exc.value)
        assert f"missing=1 [{missing_id}]" in message[:500]
        assert "owned=200 covered=199 excluded=0" in message[:500]
        assert "extra=0 []; double=0 []" in message[:500]
        assert "expected_frontier=0" in message[:500]
        assert f"authoritative={owned}, covered={covered}, excluded=[]" in message
        assert "synthetic row content" not in message
        assert engine._lifecycle.get_by_conversation("conv").current_frontier_store_id == 0
        assert engine._dag.get_session_nodes("S") == []
    finally:
        engine.shutdown()


def test_p0_compaction_diagnostics_extra_and_double_claims():
    # The publication guards deduplicate claims and reject overlap earlier. Exercise
    # its diagnostic formatter directly to cover overlap and repeats in either list.
    owned = list(range(1, 201))
    covered = list(range(1, 161)) + [100, 201, 202, 202]
    excluded = list(range(160, 201)) + [180, 203, 203]
    message = lifecycle_state._compaction_coverage_conflict_message(0, owned, covered, excluded)
    assert "missing=0 []; extra=3 [201, 202, 203]; double=5 [100, 160, 180, 202, 203]" in message[:500]
    assert "owned=200 covered=164 excluded=44" in message[:500]

    # Counts remain complete while each diagnostic sample is capped at eight ids.
    message = lifecycle_state._compaction_coverage_conflict_message(
        0, list(range(1, 21)), list(range(21, 41)) * 2, list(range(41, 61)) * 2,
    )
    assert "missing=20 [1, 2, 3, 4, 5, 6, 7, 8];" in message
    assert "extra=40 [21, 22, 23, 24, 25, 26, 27, 28];" in message
    assert "double=40 [21, 22, 23, 24, 25, 26, 27, 28];" in message


class _Clock:
    def __init__(self):
        self.real = time.monotonic
        self.offset = 0.0

    def __call__(self):
        return self.real() + self.offset


def test_p0_compaction_diagnostics_pre_leaf_soft_budget_stop(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)
    monkeypatch.setattr(tokens, "count_tokens", lambda text: 0 if not text else (
        tokens._fallback_token_estimate(text) if isinstance(text, str) else len(text) // 4 + 1))
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    calls = []

    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        calls.append(clock.offset)
        clock.offset += 31.0
        return "Earlier turns.\nExpand for details about: turns"

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    engine = _engine(tmp_path, sweep=True)
    original_prepare = engine._prepare_retained_user_anchor

    def prepare(*args, **kwargs):
        result = original_prepare(*args, **kwargs)
        clock.offset += 3.0
        return result

    monkeypatch.setattr(engine, "_prepare_retained_user_anchor", prepare)
    try:
        assert engine._config.foreground_soft_seconds == 60
        for _ in range(8):
            engine._foreground_estimates.record_call("", 31.0)
        for index in range(6):
            engine._dag.add_node(SummaryNode(
                session_id="S", depth=0, summary=f"group {index}", token_count=1000,
                source_token_count=2000, source_ids=[], source_type="messages", created_at=float(index + 1),
            ))
        pad = " alpha beta gamma delta" * 30
        view = [{"role": "system", "content": "system prompt"}]
        for index in range(12):
            view.extend([
                {"role": "user", "content": f"[T{index}] user turn{pad}", "timestamp": 10.0 * (index + 1)},
                {"role": "assistant", "content": f"reply to T{index}{pad}"},
            ])
        engine.ingest(view)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        telemetry = engine.get_status()["threshold_full_sweep"]
        assert calls == [3.0]
        assert telemetry["pre_leaf_condensation_passes"] == 1
        assert telemetry["leaf_passes"] == 0
        assert telemetry["status"] == "partial"
        assert telemetry["stop_reason"] == "soft_target_reached"
        assert len([node for node in engine._dag.get_session_nodes("S") if node.depth > 0]) == 1
        reason = "foreground budget stop before a leaf (soft_target_reached)"
        assert engine._last_compression_noop_reason == reason
        assert f"LCM compression no-op: {reason}" in caplog.text
        assert "no eligible raw backlog" not in caplog.text
    finally:
        engine.shutdown()
