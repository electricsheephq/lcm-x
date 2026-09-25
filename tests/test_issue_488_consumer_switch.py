"""#488 PR-B: consumers read the emission projection, never summary shape.

Occurrence-level checks for the consumer switch: an unproven, ambiguous or
out-of-place summary-shaped row keeps FULL identity (store direction), and a
proven emitted occurrence keeps its projected remainder.
"""

from __future__ import annotations

import time

from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from tests.test_compression_boundary import _config

X = "ISSUE-488-X authored remainder"


def _engine_with_verified_block(tmp_path):
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    node = engine._dag.add_node(
        SummaryNode(
            session_id="S0",
            depth=0,
            summary="Verified summary.",
            token_count=3,
            source_token_count=5,
            source_ids=[],
            source_type="messages",
            created_at=time.time(),
            earliest_at=time.time(),
            latest_at=time.time(),
            expand_hint="verified",
        )
    )
    node_id = getattr(node, "node_id", node)
    block = f"[Recent Summary (d0, node {node_id})]\nVerified summary.\n[Expand for details: verified]"
    assert engine._verified_lcm_summary_prefix_end(block) == len(block)
    return engine, block


def test_replay_cache_never_serves_a_carrier_view_for_its_remainder(tmp_path):
    engine, block = _engine_with_verified_block(tmp_path)
    head = {"role": "user", "content": "first"}
    authored = [head, {"role": "user", "content": block + "\n\n" + X}]
    plain = [head, {"role": "user", "content": X}]
    try:
        engine._remember_active_replay_messages(authored, [dict(m) for m in authored])
        assert engine._cached_active_replay_messages([dict(m) for m in authored]) is not None
        assert engine._cached_active_replay_messages(plain) is None
        assert not engine._is_cached_active_replay_message_at_index(1, plain[1])
    finally:
        engine.shutdown()
