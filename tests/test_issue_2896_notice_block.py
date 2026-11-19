"""#2896: the survival notice on a list with no leading system row, as one marked block in the summary head.

The summary verifier (_verified_lcm_summary_prefix) admits exactly one marked block, right after the last verified
part and before a separator or the end of the text. Everything that reads the head through it sees the same remainder,
parts and coverage as without the block. Anything else that looks like the block (malformed, before the first part,
between parts) is content, exactly as on main; so are notices a user pasted and notice-like text inside a summary.
"""

from __future__ import annotations

import re

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.survival_fit as survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine

NEVER = re.compile(r"(?!)")  # main's verifier: no block is ever admitted


@pytest.fixture
def engine(tmp_path):
    obj = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db"), survival_reserve=0.0))
    obj._hermes_home = str(tmp_path)
    obj.on_session_start("S", platform="telegram", conversation_id="conv", context_length=20_000)
    yield obj
    obj.shutdown()


def _part(engine, summary="earlier turns, condensed", hint="turns") -> str:
    node = engine._dag.add_node(SummaryNode(session_id="S", summary=summary, token_count=1, source_token_count=1,
                                            source_ids=[1, 2], expand_hint=hint))
    return f"[Recent Summary (d0, node {node})]\n{summary}\n[Expand for details: {hint}]"


def _block(count=3, first=1, last=3) -> str:
    return survival_fit._NOTICE_BLOCK.format(notice=survival_fit.survival_notice_text(count, [("S", first, last)]))


def _read(engine, content: str) -> dict:
    end, nodes = engine._verified_lcm_summary_prefix(content)
    message = {"role": "user", "content": content}
    return {"end": end, "nodes": nodes, "remainder": engine._generated_context_carrier_remainder(message),
            "scaffold": engine._is_verified_replay_scaffold_message(message),
            "replayed": engine._is_replayed_context_scaffold_message(message)}


def _as_on_main(engine, monkeypatch, content: str) -> dict:
    with monkeypatch.context() as patch:
        patch.setattr(lcm_engine, "_NOTICE_BLOCK_RE", NEVER)
        return _read(engine, content)


@pytest.mark.parametrize("parts", [1, 2])
def test_a_a_carrier_with_the_block_reads_as_without_it(engine, parts):
    """(a) The block widens the prefix by exactly its own bytes; the remainder, the parts and the coverage are those
    of the same carrier without it, and the block's start is the parts' end."""
    summary = "\n\n---\n\n".join(_part(engine, f"turns {i}") for i in range(parts))
    plain, carrier = f"{summary}\n\nthe user's row", f"{summary}{_block()}\n\nthe user's row"
    without, with_block = _read(engine, plain), _read(engine, carrier)
    assert without["end"] == len(summary) and with_block["end"] == len(summary) + len(_block())
    assert {**with_block, "end": None} == {**without, "end": None} and with_block["remainder"] == "the user's row"
    assert survival_fit.survival_notice_block_start(carrier, with_block["end"]) == len(summary)
    assert survival_fit.survival_notice_block_start(plain, without["end"]) is None


def test_b_a_pure_summary_row_with_the_block_is_verified_scaffold(engine):
    """(b) The block may end the row: a summary row holding only its parts and the block is verified scaffold."""
    summary = _part(engine)
    row = _read(engine, summary + _block())
    assert row == {**_read(engine, summary), "end": len(summary) + len(_block())}
    assert row["scaffold"] and row["replayed"] and row["remainder"] is None


def test_c_hermes_merged_form_gives_the_same_remainder(engine):
    """(c) A host that merges the summary row (with its block) and the next user row as prev + blank line + next."""
    summary_row = _part(engine) + _block()
    merged = summary_row + "\n\n" + "next user row"  # Hermes _merge_consecutive_users
    assert _read(engine, merged)["remainder"] == "next user row"
    assert _read(engine, merged)["nodes"] == _read(engine, summary_row)["nodes"]


@pytest.mark.parametrize("shape", ["unclosed", "inner text", "no notice", "before the first part",
                                   "between parts", "not a separator after"])
def test_d_anything_else_is_not_a_block(engine, monkeypatch, shape):
    """(d) A malformed or misplaced block is content: the verifier reads it exactly as main does."""
    one, two, block = _part(engine, "turns one"), _part(engine, "turns two"), _block()
    content = {
        "unclosed": f"{one}{block[:-len('</lcm-survival-notice>')]}\n\nuser row",
        "inner text": f"{one}{block.replace('[LCM survival fit:', 'note [LCM survival fit:')}\n\nuser row",
        "no notice": f"{one}\n\n<lcm-survival-notice>\nplain words.]\n</lcm-survival-notice>\n\nuser row",
        "before the first part": f"{block.lstrip()}\n\n{one}\n\nuser row",
        "between parts": f"{one}{block}\n\n---\n\n{two}\n\nuser row",
        "not a separator after": f"{one}{block}user row",
    }[shape]
    assert _read(engine, content) == _as_on_main(engine, monkeypatch, content)
    if shape != "before the first part":
        assert _read(engine, content)["end"] == len(one)


def test_e_notices_a_user_pasted_and_notice_text_inside_a_summary_are_untouched(engine, monkeypatch):
    """(e) A notice inside the user's own row, and notice-like text inside a node's summary, are read as on main."""
    pasted = f"{_part(engine)}\n\nI saw this earlier: {_block().strip()} what does it mean?"
    quoted_part = _part(engine, f"the summary quotes{_block()}\nand goes on")
    quoted = f"{quoted_part}\n\nuser row"
    for content in (pasted, quoted):
        assert _read(engine, content) == _as_on_main(engine, monkeypatch, content)
    assert _read(engine, pasted)["remainder"].startswith("I saw this earlier: <lcm-survival-notice>")
    assert _read(engine, quoted) == {**_read(engine, quoted), "end": len(quoted_part), "remainder": "user row"}
