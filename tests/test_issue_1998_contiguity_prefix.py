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


def test_1998_prefix_retry_consumes_only_the_host_prefix(tmp_path, summaries, lines, monkeypatch):
    # Stored a1 t1_2 m3 h4 t2_5 u6; the host shows m u a t1 (a, t1 by claims). The walk emits m, stops at u
    # (stored 6) and the truncated scan never emits a or t1: their claims are a hole and the leaf empties at
    # chunk index 1. The retry runs on [m]; a and t1 stay host rows, escaped from the store.
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", 5)
    engine = _engine(tmp_path, False)
    engine.context_length = 1_048_576
    a, t1, h = _call("call_1", 1.0), _res("call_1", "ok", 2.0), _call("call_2", 4.0)
    m = {"role": "user", "content": "earlier question", "timestamp": 3.0}
    t2 = _res("call_2", "[Externalized tool output: tool=read_file; chars=90000; ref=x]" + PAD * 3, 5.0)
    u = {"role": "user", "content": "next", "timestamp": 6.0}
    try:
        engine.ingest([a, t1, m, h, t2, u])
        ids = [int(r["store_id"]) for r in _rows(engine)]
        assert len(ids) == 6
        chunk = [m, u, a, t1]
        mapping = {id(m): ids[2], id(u): ids[5]}
        out = engine._store_complete_input(chunk, {id(a): [ids[0]], id(t1): [ids[1]]}, mapping, chunk, chunk, 0,
                                           [], count_messages_tokens(chunk), allow_retained_escape=True)
        assert engine._store_complete_empty_reason != CONTIGUITY, engine._store_complete_empty_reason
        holes = [msg for _lvl, msg in lines if msg.startswith("LCM contiguity hole:")]
        assert len(holes) == 1 and f"hole_store_id={ids[0]} " in holes[0], holes
        assert "blocking_chunk_index=1/4 " in holes[0] and "prefix_retry=True" in holes[0], holes
        # The leaf covers exactly the stored prefix below u, each row once; of the host rows only m is input.
        covered = [s for msg, c in out for s in [*c, *([mapping[id(msg)]] if id(msg) in mapping else [])]]
        assert sorted(covered) == ids[:3], covered
        assert [msg for msg, _c in out if any(msg is row for row in chunk)] == [m]
        assert engine._store_complete_cut
        # The cut rows' stored rows are escaped (the host still shows a and t1), not silently covered.
        assert engine._store_complete_escaped == [(ids[0], "assistant", "shown"), (ids[1], "tool", "shown")]
        # The caller's consumption (the leaf loop's derivation) takes only the host prefix [m].
        engine._compress_run.store_ids_by_message_id = mapping
        summary_input, sources, anchor_claims = engine._leaf_summary_input(out, chunk)
        _rows_read, positions, consumed_rows, source_ids, consumed_ids = engine._leaf_consumption(
            chunk, summary_input, sources, out, anchor_claims, set())
        assert positions == [0] and consumed_rows == [m]
        assert source_ids == ids[:3] and consumed_ids == ids[:3]
    finally:
        engine.shutdown()


def test_1998_prefix_retry_keeps_retained_composite_claims_pending(tmp_path, summaries, lines, monkeypatch):
    # Stored x1 y2 m3 h4 t5 u6; the host shows m u C12. The truncated walk stops at u,
    # leaving C12's claims as a hole. Retrying on [m] must not summarise x1/y2 as hidden
    # rows while their composite stays in view without an escape or summary lineage.
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", 5)
    engine = _engine(tmp_path, False)
    engine.context_length = 1_048_576
    x = {"role": "user", "content": "first constituent", "timestamp": 1.0}
    y = {"role": "user", "content": "second constituent", "timestamp": 2.0}
    m = {"role": "user", "content": "earlier question", "timestamp": 3.0}
    h = _call("call_2", 4.0)
    t = _res("call_2", "[Externalized tool output: tool=read_file; chars=90000; ref=x]" + PAD * 3, 5.0)
    u = {"role": "user", "content": "next", "timestamp": 6.0}
    composite = {"role": "user", "content": x["content"] + "\n\n" + y["content"], "timestamp": 1.0}
    try:
        engine.ingest([x, y, m, h, t, u])
        ids = [int(row["store_id"]) for row in _rows(engine)]
        assert len(ids) == 6
        chunk = [m, u, composite]
        mapping = {id(m): ids[2], id(u): ids[5]}
        # R2 observes the exact concatenation at x's stamp, records its witness, and
        # supplies C12's constituent claims to the real store-complete walk.
        out = engine._identity_anchor_summary_input(chunk, mapping, chunk, chunk,
                                                    budget=count_messages_tokens(chunk))
        witness = engine._store.get_message_relations([ids[0]], "composite")
        assert [row["related_store_id"] for row in sorted(witness, key=lambda row: row["ordinal"])] == ids[:2]
        covered = [sid for msg, claims in out or ()
                   for sid in [*claims, *([mapping[id(msg)]] if id(msg) in mapping else [])]]
        assert not set(ids[:2]) & set(covered), (
            "retained composite constituents became hidden summary input without lineage",
            covered, engine._store_complete_escaped)
        assert out == [] and engine._store_complete_empty_reason == CONTIGUITY
        holes = [msg for _lvl, msg in lines if msg.startswith("LCM contiguity hole:")]
        assert len(holes) == 1 and f"hole_store_id={ids[0]} " in holes[0], holes
        assert "blocking_chunk_index=1/3 " in holes[0], holes
        assert engine._store_complete_escaped == []
        assert chunk == [m, u, composite]
        assert len(_rows(engine)) == 6  # the witness does not store a duplicate composite row
    finally:
        engine.shutdown()
