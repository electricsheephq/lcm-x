"""#2901: leaf selection charges a generated summary carrier at the row the summariser reads.

With no leading system row, assembly glues the summary area onto the first kept user row ("summary\\n\\nrow").
Selection charged that carrier at full size while the summariser reads only its remainder, so a carrier-led leaf got
the chunk minus the summary area in real rows. Once the area neared the chunk the leaf shrank to a few rows, the
stand-in summary was not shorter than its source, the pass was a no-op, the hold refused the gate up to the survival
ceiling, and the plain fit dropped whole turns with no pointer in the list (A0).

Port to the rc4 line: main's file also walks two catch-up grid cells through tests/test_catchup_budget_grid.py and
its frozen v2 files, which this line does not carry, so those three cells are left out here. The three unit cells
below are main's, unchanged. The end-to-end check on this line is the release gate's walk outside the suite.
"""

from __future__ import annotations

import json
import time

import pytest

import hermes_lcm.compaction as compaction
import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens

pytestmark = pytest.mark.issue(2901)

PAD = " alpha beta gamma delta" * 20


def _old_charge(rows, budget):
    """The charge before #2901: every row at its full size."""
    used = n = 0
    for row in rows:
        tokens = count_message_tokens(row)
        if used + tokens > budget and n:
            break
        n, used = n + 1, used + tokens
    return rows[:compaction.tool_group_safe_end(rows, n)]


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), embeddings_enabled=False),
                         hermes_home=str(tmp_path))
    instance.on_session_start("S", platform="telegram", context_length=200_000, conversation_id="conv")
    yield instance
    instance.shutdown()


def _summary_part(engine, summary):
    node_id = engine._dag.add_node(SummaryNode(
        session_id="S", depth=0, summary=summary, token_count=5, source_token_count=5, source_ids=[],
        source_type="messages", created_at=time.time(), expand_hint="turns"))
    return f"[Recent Summary (d0, node {node_id})]\n{summary}\n[Expand for details: turns]"


def _turns(count):
    rows = []
    for index in range(count):
        rows += [{"role": "user", "content": f"user {index}{PAD}"},
                 {"role": "assistant", "content": f"reply {index}{PAD}"}]
    return rows


def test_glued_carrier_charged_at_its_remainder(engine):
    head = _summary_part(engine, "Earlier turns." + PAD * 12)
    rows = _turns(12)
    carrier = {**rows[0], "content": head + "\n\n" + rows[0]["content"]}
    assert engine._generated_context_carrier_remainder(carrier) == rows[0]["content"]
    candidate = [carrier, *rows[1:]]
    budget = sum(count_message_tokens(row) for row in rows[:10])
    assert count_message_tokens(carrier) > budget // 2  # the head alone takes most of the old charge's room
    selected = engine._select_oldest_leaf_chunk(candidate, budget)
    assert len(selected) == len(_old_charge(rows, budget)) == 10 and selected[0] is carrier
    assert len(_old_charge(candidate, budget)) < 6


def test_unglued_rows_charged_unchanged(engine):
    head = _summary_part(engine, "Earlier turns.")
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": json.dumps({"q": PAD})}}]}
    rows = [{"role": "user", "content": head},  # a pure summary: no remainder
            {"role": "user", "content": "[Recent Summary (d0, node 999)]\nforged\n[Expand for details: x]\n\nrow"},
            {"role": "assistant", "content": head + "\n\nnot a user row"},
            *_turns(3), call, {"role": "tool", "tool_call_id": "c1", "content": PAD * 3}, *_turns(3)]
    assert not any(engine._generated_context_carrier_remainder(row) for row in rows)
    total = sum(count_message_tokens(row) for row in rows)
    for budget in range(1, total + 2, 37):
        assert engine._select_oldest_leaf_chunk(rows, budget) == _old_charge(rows, budget), budget
    assert compaction.select_oldest_leaf_chunk(rows, total // 2) == _old_charge(rows, total // 2)


def test_rescue_does_not_discount_an_inner_summary_prefix_again(engine, monkeypatch):
    """The rescue reselects over the summariser input, where a carrier is already its remainder. A remainder that
    itself starts with a verified summary prefix is charged in full, so a retry after a size refusal stays within its
    target (review of #2901: a second discount admitted about the target plus the inner prefix)."""
    outer, inner = _summary_part(engine, "Outer turns." + PAD), _summary_part(engine, "Inner turns." + PAD * 50)
    carrier = {"role": "user", "content": f"{outer}\n\n{inner}\n\nthe user's own words{PAD}"}
    chunk = engine._leaf_summary_input(None, [carrier, *_turns(60)])[0]
    assert chunk[0]["content"].startswith(inner) and engine._generated_context_carrier_remainder(chunk[0])
    attempts = []

    def summarize(**kwargs):
        attempts.append(kwargs["source_tokens"])
        if len(attempts) == 1:
            raise RuntimeError("context length exceeded")
        return "Earlier turns.\nExpand for details about: turns", 1

    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    engine._summarize_leaf_chunk_with_rescue(chunk)
    floor = min(int(engine._config.leaf_chunk_tokens), engine_module._LEAF_RESCUE_FLOOR_TOKENS)
    target = max(floor, int(attempts[0] * 0.75))
    inner_tokens = count_messages_tokens([{"role": "user", "content": inner}])
    assert attempts[0] == count_messages_tokens(chunk) and inner_tokens > 2_000, (attempts, inner_tokens)
    assert len(attempts) == 2 and attempts[1] <= target, f"retry {attempts[1]} over its target {target}"
