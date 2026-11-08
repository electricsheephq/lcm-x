"""#2116 wait bar: with a reply candidate, the presence audit also normalizes stored and shown reply text, under the
ingest write lock. Each shape fills the 4 MB text budget to just inside it (the audit must still answer) or past it
(main's result, sized in SQL before any row loads). The bar is 150 ms at a load of at most half the cores; PR CI
checks the results, and the nightly latency cell checks the bound times LCM_PRESENCE_TIMING_SLACK (default 4)."""
from __future__ import annotations

import os

import pytest

from hermes_lcm import identity_anchor

from tests.test_2466_presence_wait_bar import _append, _best
from tests.test_anchor_unstored_before_cursor import _engine

BAR = 0.150 * float(os.environ.get("LCM_PRESENCE_TIMING_SLACK", "4"))
TEXT = identity_anchor._PRESENCE_TEXT_BYTES


def _reply(i, size):
    body = f"stored reply {i} "
    return {"role": "assistant", "content": body + "lorem ipsum dolor sit amet " * max(0, (size - len(body)) // 27)}


MIN = identity_anchor._PRESENCE_TEXT_MIN_CHARGE


@pytest.mark.parametrize("shape", ["long-replies-inside", "many-short-replies-inside", "stored-text-past-the-budget"])
@pytest.mark.parametrize("view_kind", ["fresh-only", "history-then-fresh"])
def test_reply_text_at_the_text_budget(tmp_path, shape, view_kind, latency_check):
    """Inside: stored and shown reply text together just inside the budget, as few long or as many short replies as
    fit (each charged at least MIN). Past: stored text alone over it, a trip in SQL before any row loads."""
    shown = 2 if view_kind == "history-then-fresh" else 1
    if shape == "long-replies-inside":
        rows, size, expected = 2_000, int(TEXT * 0.95 / (2_000 * shown)), "answer"
    elif shape == "many-short-replies-inside":
        rows, size, expected = int(TEXT * 0.98 / (MIN * shown)), 120, "answer"
    else:
        rows, size, expected = 2_000, int(TEXT * 1.1 / 2_000), None
    engine = _engine(tmp_path)
    try:
        stored = [_reply(i, size) for i in range(rows)]
        _append(engine, stored)
        history = stored if view_kind == "history-then-fresh" else []
        took, out = _best(engine, [*history, {"role": "assistant", "content": "a brand new reply"}])
        assert out == ([len(history)] if expected else None)
        assert not latency_check or took < BAR, f"{took * 1000:.0f} ms"
    finally:
        engine.shutdown()
