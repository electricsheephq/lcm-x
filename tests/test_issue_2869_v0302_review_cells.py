"""#2869 v0.30.2 review cells, tree-agnostic, so the same file runs on main and on the head (red proof; local).

R1, P1 at the pre-leaf gate: at a 1M window, 81 live leaves total exactly today's 24,000 target at the sweep-start
read. A repair raises one stored count by 100 before the pre-leaf gate, and no raw leaf is eligible. Main gates on the
saved sweep-start total and never condenses. Bar: no condensation.
R2, P3 across no-progress exits: 80 live depth-0 leaves at 1M, the eight oldest short CJK text with a stored count of
0, the other 72 costing 500. New leaves cost 300 and four-leaf condensations return 200, through the real escalation
chain, so a zero-count group is kept verbatim at level 3 as a positive-size parent. Each sweep has the backlog and time
for its 12 passes. Bar: depth 0 never holds more than the 100 uncondensed summaries the view shows; main peaks at 94.
"""

from __future__ import annotations

import pytest

import hermes_lcm.escalation as escalation
import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

pytestmark = pytest.mark.issue(2869)

QUIET = {"embeddings_enabled": False, "temporal_rollups_enabled": False, "large_output_externalization_enabled": False,
         "large_output_active_replay_stubbing_enabled": False}
ROW = " alpha beta gamma delta" * 400  # about 2,000 tokens a row
LEAF, CONDENSED = "y" * 1_200, "z" * 800  # 300 and 200 tokens


def _engine(tmp_path, **config):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), **{**QUIET, **config}),
                       hermes_home=str(tmp_path))
    engine.on_session_start("S", platform="cli", context_length=1_000_000, conversation_id="conv")
    return engine


def _node(engine, tokens_, text, created):
    return engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary=text, token_count=tokens_,
                                            source_token_count=4 * max(tokens_, 1), source_ids=[], created_at=created,
                                            expand_hint="turns"))


def _depth0(engine):
    return len(engine._dag.get_uncondensed_at_depth("S", 0, limit=10 ** 6))


def _live(engine):
    return len(engine._summary_frontier_nodes())


def test_r1_a_repair_between_the_reads_never_starts_a_condensation(tmp_path, monkeypatch):
    monkeypatch.setattr(escalation, "_invoke_summary_llm", lambda *args, **kwargs: CONDENSED)
    engine = _engine(tmp_path, fresh_tail_count=24)
    try:
        nodes = [_node(engine, 296, "x" * 1_184, 1.0 + i) for i in range(80)] + [_node(engine, 320, "x" * 1_280, 90.0)]
        assert engine._summary_frontier_tokens() == 24_000 and _live(engine) == 81
        condensed, gate = [0], engine._condense_before_leaves
        condense = engine._condense_summary_nodes

        def repair_then_gate(*args, **kwargs):  # an operator repair replaces a truncated summary between the reads
            with engine._dag._db_lock:
                engine._dag._conn.execute("UPDATE summary_nodes SET token_count = token_count + 100 WHERE node_id = ?",
                                          (nodes[0],))
                engine._dag._conn.commit()
            return gate(*args, **kwargs)

        def counted(*args, **kwargs):
            condensed[0] += 1
            return condense(*args, **kwargs)

        monkeypatch.setattr(engine, "_condense_before_leaves", repair_then_gate)
        monkeypatch.setattr(engine, "_condense_summary_nodes", counted)
        tail = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"tail {i}"} for i in range(20)]
        engine.ingest(tail)
        engine.compress(tail, current_tokens=engine.threshold_tokens)
        status = engine.get_status()["threshold_full_sweep"]
        assert status["summary_prefix_target_tokens"] == 24_000, status
        assert condensed[0] == 0, f"{condensed[0]} condensation(s) where main starts none; status {status}"
    finally:
        engine.shutdown()


def test_r2_no_progress_exits_never_leave_more_than_100_at_depth_0(tmp_path, monkeypatch):
    def provider(prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        return LEAF if "ROW-MARK" in prompt else CONDENSED

    monkeypatch.setattr(escalation, "_invoke_summary_llm", provider)
    engine = _engine(tmp_path, context_threshold=0.5, fresh_tail_count=4, dynamic_leaf_chunk_enabled=False)
    try:
        for i in range(8):
            _node(engine, 0, "漢字", 1.0 + i)  # short CJK, stored count 0
        for i in range(72):
            _node(engine, 500, "x" * 2_000, 20.0 + i)
        assert _live(engine) == 80 and engine._summary_frontier_tokens() == 36_000
        assert tokens.count_tokens(LEAF) == 301 and tokens.count_tokens(CONDENSED) == 201
        view, turn, walk = [], 0, []
        for _sweep in range(4):
            for _ in range(150):  # backlog for twelve 24,000-token leaves
                view.append({"role": "user" if turn % 2 == 0 else "assistant", "content": f"ROW-MARK {turn}{ROW}"})
                turn += 1
            start = (_live(engine), _depth0(engine))
            engine.ingest(view)
            view = engine.compress(view, current_tokens=engine.threshold_tokens + 300_000)
            status = engine.get_status()["threshold_full_sweep"]
            walk.append({"start": start, "end": (_live(engine), _depth0(engine)), "leaves": status["leaf_passes"],
                         "pre_leaf": status.get("pre_leaf_condensation_passes"),
                         "pre_leaf_stop": status.get("pre_leaf_condensation_stop_reason")})
            engine.note_turn_complete()
            engine._clear_persisted_no_progress_hold()
            engine._no_progress_hold = None
        print("WALK", walk)
        assert walk[0]["leaves"] >= 10, walk  # premise: each sweep spends its passes on leaves
        peak = max(max(step["start"][1], step["end"][1]) for step in walk)
        assert peak <= 100, f"depth 0 reached {peak} (view 100): {walk}"
    finally:
        engine.shutdown()
