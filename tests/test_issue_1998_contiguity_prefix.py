"""#1998: compaction never formed a leaf ("leaf input cannot form a contiguous stored prefix").

A chunk row stored above a later chunk row (here a user row the host put before the tool group it follows
in the store, the group carried by claims) holds back the rows behind it. When the walk stops first, at
the leaf budget or at a truncated scan, their claims become a hole below the leaf's end and the leaf
empties on every pass. The #2003 budget lift clears the budget stop; a truncated scan needs the prefix
retry, which leaves the cut rows to the #1162 escape. The store-order controls never stop."""

import pytest

from hermes_lcm import store_complete
from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_1162_retained_escape import PAD, _engine, _rows, lines as _shared_lines, \
    summaries as _shared_summaries

lines = _shared_lines
summaries = _shared_summaries

CONTIGUITY = "leaf input cannot form a contiguous stored prefix"


def _call(cid, ts):
    return {"role": "assistant", "content": "", "timestamp": ts, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}


def _res(cid, text, ts):
    return {"role": "tool", "tool_call_id": cid, "tool_name": "read_file", "content": text, "timestamp": ts}


@pytest.mark.parametrize("scan_limit", [None, 4], ids=["budget", "truncated_scan"])
@pytest.mark.parametrize("inverted", [False, True], ids=["store_order", "host_inverted"])
def test_1998_tool_call_only_rows_form_a_contiguous_leaf(tmp_path, summaries, lines, monkeypatch, scan_limit,
                                                         inverted):
    if scan_limit:
        monkeypatch.setattr(store_complete, "_SCAN_LIMIT", scan_limit)
    engine = _engine(tmp_path, False)
    engine.context_length = 1_048_576
    a, t1, h = _call("call_1", 1.0), _res("call_1", "ok", 2.0), _call("call_2", 3.0)
    t2 = _res("call_2", "[Externalized tool output: tool=read_file; chars=90000; ref=x]" + PAD * 3, 4.0)
    u = {"role": "user", "content": "next", "timestamp": 5.0}
    try:
        engine.ingest([a, t1, h, t2, u])
        ids = [int(r["store_id"]) for r in _rows(engine)]
        assert len(ids) == 5
        chunk = [u, a, t1] if inverted else [a, t1, u]
        mapping = {id(u): ids[4]} if inverted else {id(a): ids[0], id(t1): ids[1], id(u): ids[4]}
        claims = {id(a): [ids[0]], id(t1): [ids[1]]} if inverted else {}
        out = engine._store_complete_input(chunk, claims, mapping, chunk, chunk, 0, [],
                                           count_messages_tokens(chunk), allow_retained_escape=True)
        covered = sorted(s for m, c in out or () for s in [*c, *([mapping[id(m)]] if id(m) in mapping else [])])
        assert engine._store_complete_empty_reason != CONTIGUITY, engine._store_complete_empty_reason
        assert out and covered == ids[:len(covered)]
        holes = [msg for _lvl, msg in lines if msg.startswith("LCM contiguity hole:")]
        if inverted and scan_limit:
            # The stop is logged once, by ids, roles and counts only; the retry then forms the leaf.
            assert len(holes) == 1 and f"hole_store_id={ids[0]} role=assistant " in holes[0], holes
            assert "prefix_retry=True" in holes[0] and "truncated=True" in holes[0], holes
            assert engine._store_complete_cut
    finally:
        engine.shutdown()
