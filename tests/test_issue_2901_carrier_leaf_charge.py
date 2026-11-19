"""#2901: leaf selection charges a generated summary carrier at the row the summariser reads.

With no leading system row, assembly glues the summary area onto the first kept user row ("summary\\n\\nrow").
Selection charged that carrier at full size while the summariser reads only its remainder, so a carrier-led leaf got
the chunk minus the summary area in real rows. Once the area neared the chunk the leaf shrank to a few rows, the
stand-in summary was not shorter than its source, the pass was a no-op, the hold refused the gate up to the survival
ceiling, and the plain fit dropped whole turns with no pointer in the list (A0).

The walks use the catch-up grid runner on CAND_TREE (env LCMX_GRID_CAND, default this tree); ``OLD_CHARGE`` restores
the full-size charge in the runner process, so the premise is checked on the same tree.
"""

from __future__ import annotations

import json
import time

import pytest

import hermes_lcm.compaction as compaction
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_message_tokens
from tests import test_catchup_budget_grid as grid

pytestmark = pytest.mark.issue(2901)

TWIN, ORIGINAL = "s-1m-l40-nosys", "s-1m-l40"
PASSES = 26  # main reaches the survival ceiling at pass 24 of the twin (grid v2)
OLD_CHARGE = ("_select = mods['compaction'].select_oldest_leaf_chunk\n"
              "mods['engine'].LCMEngine._select_oldest_leaf_chunk = lambda self, raw, budget: _select(raw, budget)\n")
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


def _fits(records):
    """Passes whose returned list shrank with no node stored: only the survival fit does that."""
    return [r["pass"] for r in records
            if r["triggered"] and not r["leaves"] and not r["condensations"] and r["prompt_after"] < r["prompt_before"]]


def _a0(records):
    return sum(r["loss"]["missing_count"] for r in records if r["loss"])


def _strip(records):
    return [{k: v for k, v in r.items() if k != "tree"} for r in records]


@pytest.fixture(scope="module")
def walks(tmp_path_factory):
    out = tmp_path_factory.mktemp("issue-2901")
    cells = [{**grid.BY_ID[cell], "passes": PASSES} for cell in (TWIN, ORIGINAL)]
    grid.PATCHES.setdefault("old_charge_2901", OLD_CHARGE)
    return {"new": grid._run(grid.CAND_TREE, cells, out, "new")["by_cell"],
            "old": grid._run(grid.CAND_TREE, cells, out, "old", patch="old_charge_2901")["by_cell"]}


def test_twin_premise_the_full_size_charge_stalls_and_fits(walks):
    """Not the fix: with the old charge the twin reaches the plain fit and loses rows with no pointer."""
    old = walks["old"][TWIN]
    assert len(old) == PASSES and _fits(old) and _a0(old) > 0, (
        f"premise lost: fits {_fits(old)}, A0 {_a0(old)}, leaves {[r['leaves'] for r in old]}")
    assert any(r["status"] == "noop" and r["triggered"] for r in old), "no carrier-led no-op pass before the fit"


def test_twin_no_stall_no_fit_no_loss(walks):
    new = walks["new"][TWIN]
    assert len(new) == PASSES
    assert not _fits(new), f"survival fit reached at passes {_fits(new)}"
    assert _a0(new) == 0 and not grid.a0_violations(new), grid.a0_violations(new)[:3]
    assert all(r["leaves"] for r in new if r["triggered"]), [(r["pass"], r["status"]) for r in new if r["triggered"]]
    assert sum(r["leaves"] for r in new) > sum(r["leaves"] for r in walks["old"][TWIN])


def test_system_row_cell_records_unchanged(walks):
    """With a leading system row no carrier forms: every per-pass record equals the old charge's."""
    assert walks["new"][ORIGINAL] and _strip(walks["new"][ORIGINAL]) == _strip(walks["old"][ORIGINAL])


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
