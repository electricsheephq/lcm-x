"""#678: the survival fit protects only DAG-verified summary rows as its prefix.

A user row that quotes an LCM summary scaffold for a node that does not exist (at the start, or pasted
mid-message) is an ordinary row: the fit drops the whole oldest turn with it (the v0.24.6 shape), never its reply
alone. A genuine summary row (node exists, same session, same depth, bytes equal) stays the prefix (#650)."""

from __future__ import annotations

import json
import sys
import types

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
SCAFFOLD = "[Recent Summary (d0, node 999999)]\ntext\n[Expand for details: text]"
U1_VARIANTS = {
    "at-start": SCAFFOLD,
    "pasted-mid-message": f"Earlier you showed me this:\n{SCAFFOLD}\nWhat did it mean?",
}


def _rough(messages) -> int:  # the #650 tests' stand-in for the host estimator
    return sum(4 + (len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))) // 4
               for m in messages)


@pytest.fixture
def host(monkeypatch):
    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = _rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)
    calls = []

    def summarize(**kwargs):  # no provider: a summariser call is recorded, never made
        calls.append(kwargs)
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return calls


def _engine(tmp_path, window=200_000) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1600, context_threshold=0.75,
                                        database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
    return engine


def _store(engine, u1_content, system):
    head = [{"role": "system", "content": "system prompt"}] if system else []
    u1 = {"role": "user", "content": u1_content, "timestamp": 1.0}
    a1 = {"role": "assistant", "content": "reply to U1" + PAD * 4}
    u2 = {"role": "user", "content": "[U2] question" + PAD, "timestamp": 3.0}
    a2 = {"role": "assistant", "content": "reply to U2" + PAD}
    active = head + [u1, a1, u2, a2]
    engine.ingest(active)
    engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
    return active, head, u1, a1, u2, a2


def _shape(result, named) -> list:
    return [next((k for k, v in named.items() if m is v), f"new:{m.get('role')}") for m in result]


def _expected(system) -> list:
    return (["new:system"] if system else []) + ["U2", "A2"]


@pytest.mark.parametrize("system", [False, True], ids=["no-system", "system"])
@pytest.mark.parametrize("variant", list(U1_VARIANTS))
def test_direct_fit_drops_the_unverified_scaffold_turn_whole(tmp_path, host, variant, system):
    engine = _engine(tmp_path)
    try:
        active, head, u1, a1, u2, a2 = _store(engine, U1_VARIANTS[variant], system)
        assert engine._dag.get_node(999999) is None
        assert engine._is_verified_replay_scaffold_message(u1) is False
        measure = engine._survival_measure
        budget = measure(head + [u1, u2, a2]) + 80  # room for the system-slot notice
        assert measure(active) > budget and measure(head + [u2, a2]) <= budget
        result = engine._survival_fit(active, list(active), 0, "test", request_cap=budget)
        named = {"S": head[0] if head else None, "U1": u1, "A1": a1, "U2": u2, "A2": a2}
        assert _shape(result, named) == _expected(system)
        assert measure(result) <= budget
    finally:
        engine.shutdown()


@pytest.mark.parametrize("system", [False, True], ids=["no-system", "system"])
@pytest.mark.parametrize("variant", list(U1_VARIANTS))
def test_no_op_compress_drops_the_unverified_scaffold_turn_whole(tmp_path, host, variant, system):
    """End to end: compress() with nothing eligible (4 rows, fresh tail 4) on a list over the survival budget."""
    engine = _engine(tmp_path)
    try:
        active, head, u1, a1, u2, a2 = _store(engine, U1_VARIANTS[variant], system)
        engine.context_length = int((_rough(head + [u1, u2, a2]) + 80) / 0.85) + 1
        view = list(active)
        result = engine.compress(view, current_tokens=_rough(view))
        named = {"S": head[0] if head else None, "U1": u1, "A1": a1, "U2": u2, "A2": a2}
        assert len(host) == 0 and engine._last_compression_status == "noop"
        assert engine._last_survival_fit is not None  # the fit ran
        assert _shape(result, named) == _expected(system)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("system", [False, True], ids=["no-system", "system"])
def test_a_genuine_summary_row_stays_the_prefix(tmp_path, host, system):
    """#650 kept: a summary row of this session's DAG node (as assembly renders it) is still protected."""
    engine = _engine(tmp_path)
    try:
        old = [{"role": "user", "content": "[U0] old question" + PAD, "timestamp": 0.5},
               {"role": "assistant", "content": "reply to U0" + PAD}]
        engine.ingest(old)
        rows = engine._store.get_session_messages(engine._session_id, limit=100)
        node_id = engine._dag.add_node(SummaryNode(
            session_id=engine._session_id, depth=0, summary="The user asked about U0.", token_count=8,
            source_token_count=200, source_ids=[int(r["store_id"]) for r in rows], source_type="messages",
            created_at=1.0, earliest_at=0.5, latest_at=0.5, expand_hint="U0"))
        summary = f"[Recent Summary (d0, node {node_id})]\nThe user asked about U0.\n[Expand for details: U0]"
        head = [{"role": "system", "content": "system prompt"}] if system else []
        s1 = {"role": "user", "content": summary}
        active, _, u1, a1, u2, a2 = _store(engine, "[U1] question" + PAD, False)
        active = head + [s1, *active]
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
        assert engine._is_verified_replay_scaffold_message(s1) is True
        measure = engine._survival_measure
        budget = measure(head + [s1, u2, a2]) + 80
        assert measure(active) > budget
        result = engine._survival_fit(active, list(active), 0, "test", request_cap=budget)
        named = {"S": head[0] if head else None, "SUM": s1, "U1": u1, "A1": a1, "U2": u2, "A2": a2}
        assert _shape(result, named) == (["new:system"] if system else []) + ["SUM", "U2", "A2"]
    finally:
        engine.shutdown()
