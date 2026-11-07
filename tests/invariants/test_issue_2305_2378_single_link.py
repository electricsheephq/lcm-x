"""One ingest must give a stored row at most one ``possible_replay_of`` link (#2305 x #2378).

A cold cursor-0 re-ingest of a rotation child can run both annotators in one transaction:
the #2378 re-stamp pairing (``_link_restamped_occurrences``, during the ingest) links the
re-stamped carried row to the parent's original, and the #2305 ambiguous-batch scan
(``_write_ingest_ledger``, after it) starts from an empty ``linked`` set, so it links the same
row again to an unused earlier copy in the child. Host-view sequence:
1. T is ingested in S at the compress request, compress() rotates S into ``rotated``;
2. the first child ingest carries T with its stamp kept (stored once, no link) and appends a
   genuinely new, unstamped user turn with T's text (stored in the child, no link);
3. the child is rebound cold; the host re-sends a list with a new head and T re-stamped at
   commit time, so reconciliation persists the batch at cursor 0 ("persisted ambiguous delta").
No fault is injected.
"""

from collections import Counter
from copy import deepcopy

import pytest

from hermes_lcm.ingest_ledger import stripped_replay_content
from tests.invariants.test_issue_2378_restamp_link import REPLY, T, T_COMMIT, T_REPLY, T_REQ, history
from tests.invariants.test_issue_2378_restamp_link import build as build

pytestmark = pytest.mark.issue(2378)


def _links(engine):
    return list(engine._store._conn.execute(
        "SELECT r.store_id, r.related_store_id, a.content, b.content FROM message_relations r "
        "JOIN messages a ON a.store_id=r.store_id JOIN messages b ON b.store_id=r.related_store_id "
        "WHERE r.kind='possible_replay_of' ORDER BY r.store_id, r.related_store_id"))


def test_restamp_pairing_and_cursor0_scan_link_a_row_once(build, monkeypatch):
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    engine = build(rotating=True)
    hist = history(False)
    engine.ingest(deepcopy(hist))
    user_t = {"role": "user", "content": T, "timestamp": T_REQ}
    reply_t = {"role": "assistant", "content": REPLY, "timestamp": T_REPLY}
    before = hist + [user_t]
    engine.ingest(deepcopy(before))
    output = engine.compress(deepcopy(before), force=True)
    assert engine._last_compression_status == "compacted"
    engine.ingest(deepcopy(before + [reply_t]))
    engine.on_session_end("S", deepcopy(before))
    engine.on_session_start("rotated", boundary_reason="compression", old_session_id="S",
                            conversation_id=engine._conversation_id, platform="cli")
    first = deepcopy(output) + [dict(reply_t),
                                {"role": "assistant", "content": "steps again: build, push, restart", "timestamp": 2161.0},
                                {"role": "user", "content": "thanks, next question", "timestamp": 2170.0},
                                {"role": "assistant", "content": "sure", "timestamp": 2171.0},
                                {"role": "user", "content": T}]
    engine.ingest(deepcopy(first))
    assert _links(engine) == []

    engine.on_session_start("rotated", conversation_id=engine._conversation_id, platform="cli")
    second = [{"role": "user", "content": "different head", "timestamp": 2300.0}] + deepcopy(output)
    for row in second:
        if row.get("role") == "user" and row.get("content") == T:
            row["timestamp"] = T_COMMIT
    second.append({"role": "user", "content": "a further turn", "timestamp": 2310.0})
    engine.ingest(deepcopy(second))
    decision = engine._last_ingest_reconciliation
    assert (decision["action"], decision["reason"], decision["cursor"]) == (
        "persisted batch", "persisted ambiguous delta", 0)

    links = _links(engine)
    assert links, "the re-stamped copy is linked"
    for source, target, source_content, target_content in links:
        assert target_content == source_content or (
            stripped_replay_content(target_content) == stripped_replay_content(source_content)), (source, target)
    per_source = Counter(source for source, *_ in links)
    assert all(count == 1 for count in per_source.values()), [(s, t) for s, t, *_ in links]
