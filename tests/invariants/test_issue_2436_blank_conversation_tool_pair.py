"""#2436, a named surplus limit (v0.30.0): on a legacy store whose rows carry a blank conversation id, a cold
cursor-0 replay of a stored tool call and result is stored again with no ``possible_replay_of`` link.

The tool-link planner keys the blank stored id as '' and the incoming copy as the active conversation, so the copy
never meets its original. Nothing is lost; the copy is silent. The strict xfail turns green with the fix (v0.30.x)
and must then become an ordinary test. The control is the same replay over a named id (``test_18`` in
``tests/test_issue_2305_cursor0_replay_links.py`` is the unmodified case).
"""
from copy import deepcopy

import pytest

from tests.test_issue_2305_cursor0_replay_links import (  # noqa: F401 (fixtures)
    SID, _counts, _no_summaries, _pairs, _pin_lcm_char_counter, host)

pytestmark = pytest.mark.issue(2436)


def _replay_tool_pair_over(host, stored_id):  # noqa: F811
    host.tool_turn()
    originals = [row["store_id"] for row in host.engine._store.get_session_messages(SID)[-3:-1]]
    copies = deepcopy(host.live[-3:-1])
    if not (copies[0]["tool_calls"] and copies[1]["role"] == "tool"):
        pytest.fail("setup: the copies must be the stored call and result")
    conn = host.engine._store._conn
    conn.execute(f"UPDATE messages SET conversation_id = {stored_id} WHERE session_id = ?", (SID,))
    conn.commit()
    host.restart()
    before = host.stored()
    host.engine.ingest([{"role": "user", "content": "different head", "timestamp": host.stamp()}, *copies])
    decision = host.engine._last_ingest_reconciliation
    if (decision["action"], decision["reason"], decision["cursor"]) != ("persisted batch", "persisted ambiguous delta", 0):
        pytest.fail(f"setup: not a cursor-0 persisted batch: {decision}")
    sources = [row["store_id"] for row in host.engine._store.get_session_messages(SID)[before + 1:]]
    if len(sources) != 2:
        pytest.fail(f"not the #2436 shape: {len(sources)} rows stored")
    return sources, originals


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#2436 named surplus limit (v0.30.0): on a store with blank conversation ids, a cold "
                          "cursor-0 replay of a tool pair is stored again with no link")
@pytest.mark.parametrize("stored_id", ["NULL", "''"], ids=["null", "empty"])
def test_a_cursor0_tool_pair_over_a_blank_conversation_id_is_linked(host, stored_id):  # noqa: F811
    sources, originals = _replay_tool_pair_over(host, stored_id)
    pairs = _pairs(host.engine)
    if pairs not in ([], list(zip(sources, originals))):
        pytest.fail(f"not the #2436 shape: relations {pairs}")
    assert pairs == list(zip(sources, originals)), "#2436 known shape: the call and result copies are unlinked"


def test_a_cursor0_tool_pair_over_a_padded_conversation_id_is_linked(host):  # noqa: F811
    """Control (passes on main): a padded id for the same conversation still meets its originals."""
    sources, originals = _replay_tool_pair_over(host, "' ' || conversation_id || ' '")
    assert _pairs(host.engine) == list(zip(sources, originals))
    assert _counts(host.engine)["ledger.replay_pairs.cursor0_unlinked"] == 0
