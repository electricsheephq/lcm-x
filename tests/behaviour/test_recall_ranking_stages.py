"""Direct contracts for the extracted, deterministic recall ranking stages."""
from copy import deepcopy

import pytest

from hermes_lcm.retrieval_core import apply_prior, fuse_candidates, place_fts_anchor


def hit(sid, **kwargs):
    return {"kind": "message_excerpt", "store_id": sid, "session_id": str(sid), **kwargs}


def test_fusion_ties_follow_arm_order_and_do_not_mutate_inputs():
    arms = {"chunk": [hit(1)], "summary": [hit(2)], "fts": [hit(3)]}
    before = deepcopy(arms)
    entries, order, weights = fuse_candidates(arms, {})
    assert order == ["fts", "summary", "chunk"]
    assert weights == [1.0, 1.0, 1.0]
    assert [e["hit"]["store_id"] for e in entries] == [3, 2, 1]
    assert all(e["rrf_score"] == 1 / 61 for e in entries)
    assert arms == before


@pytest.mark.parametrize("fts_span,chunk_first", [(None, False), ({"char_start": 0}, False), ({"char_start": 0}, True)])
def test_fusion_pairs_snippet_and_span_and_retains_all_representations(fts_span, chunk_first):
    fts = hit(1, snippet="FTS", chunk_span=fts_span, content_offset=0)
    chunk = hit(1, snippet="chunk", chunk_span={"char_start": 7, "char_end": 12}, content_offset=7)
    summary = hit(1, snippet="summary")
    duplicate = {**chunk, "snippet": "worse", "content_offset": 19}
    chunks = [chunk, duplicate] if chunk_first else [hit(2), chunk, duplicate]
    arms = {"fts": [fts], "summary": [summary], "chunk": chunks}
    before = deepcopy(arms)
    entries, _, _ = fuse_candidates(arms, None)
    entry = next(e for e in entries if e["hit"]["store_id"] == 1)
    rank = 1 if chunk_first else 2
    assert entry["ranks"] == {0: 1, 1: 1, 2: rank}
    assert entry["rrf_score"] == 1 / 61 + 1 / 61 + 1 / (60 + rank)
    assert entry["_alternates"] == [summary, chunk, duplicate]
    expected = chunk if not fts_span or chunk_first else fts
    assert entry["hit"]["snippet"] == expected["snippet"]
    assert entry["hit"]["chunk_span"] == expected["chunk_span"]
    assert entry["hit"]["content_offset"] == expected["content_offset"]
    if expected is chunk:
        assert entry["hit"]["expand_hint"] == "lcm_expand(store_id=1, content_offset=7)"
    assert arms == before


def test_zero_weights_keep_candidates_and_tie_order():
    entries, order, weights = fuse_candidates({"fts": [hit(2)], "chunk": [hit(1)]}, {"fts": 0, "chunk": 0})
    assert order == ["fts", "chunk"]
    assert weights == [0.0, 0.0]
    assert [e["hit"]["store_id"] for e in entries] == [2, 1]
    assert [e["rrf_score"] for e in entries] == [0.0, 0.0]


def test_prior_uses_explicit_clock_scope_set_floor_and_identity_ties():
    now = 10_000_000.0
    stamps = [now, now, now + 10, now - 30 * 86400, None, "invalid"]
    entries = [{"hit": hit(i, timestamp=stamp), "rrf_score": 1.0} for i, stamp in enumerate(stamps)]
    before = deepcopy(entries)
    ranked = apply_prior(entries, {"0", "1"}, now, 0.25)
    assert [e["hit"]["store_id"] for e in ranked] == list(range(6))
    assert [e["_final_score"] for e in ranked] == [1.25, 1.25, 1.0, 0.5, 0.5, 0.5]
    assert all(e["rrf_score"] == 1.0 for e in ranked)
    assert entries == before


@pytest.mark.parametrize("weight,limit,fired", [(0.5, 4, True), (0, 4, False), (0.5, 3, False)])
def test_anchor_moves_only_rank_after_rerank_without_replacing_scores(weight, limit, fired):
    arms = {"fts": [hit(7)], "chunk": [hit(i) for i in range(6)]}
    entries, order, weights = fuse_candidates(arms, {"fts": weight})
    entries = apply_prior(entries, set(), 10.0, 0.25)
    # A provider permutation keeps the weak anchor last and reverses the rest.
    entries = list(reversed(entries[:-1])) + entries[-1:]
    before = deepcopy(entries)
    placed, info = place_fts_anchor(entries, arms, order, limit, arm_weights=weights)
    assert info == {"fired": fired, "position": 4 if fired else None}
    assert placed[3 if fired else -1]["hit"]["store_id"] == 7
    assert {e["hit"]["store_id"]: e["_final_score"] for e in placed} == {
        e["hit"]["store_id"]: e["_final_score"] for e in before
    }
    assert entries == before
