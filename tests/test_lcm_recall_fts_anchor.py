"""Tests for the lcm_recall FTS anchor slot (#950).

With two or more arms, the FTS arm's raw position-1 message is kept inside the
delivered window: when fusion pushed it past ``limit`` it moves to the start of
the S-th distinct session (S = min(limit, 10)) or to slot ``limit``. These tests
cover every inert case, the placement rule, the reported-score rule, the kill
switch, provenance, FTS-only byte invariance, and parity with the frozen replay
helper the #950 numbers were computed with.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import Any

import pytest

import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.retrieval_core import _hit_identity
from hermes_lcm.store import MessageStore

anchor = lcm_tools._lcm_recall_fts_anchor


# -- Oracle: the frozen #950 replay helper (anchor_helper.py, sha256 3e0f0f31...),
#    copied verbatim minus its module docstring. Only the primary variant
#    (require_vector_session=False) is product behaviour. --
_LCM_RECALL_FTS_ANCHOR_SLOTS = 10
_LCM_RECALL_FTS_ANCHOR_GATE_DEPTH = 10


def _first_distinct_sessions(hits: list[dict[str, Any]], depth: int) -> set[Any]:
    seen: list[Any] = []
    for hit in hits:
        sid = hit.get("session_id")
        if sid and sid not in seen:
            seen.append(sid)
            if len(seen) >= depth:
                break
    return set(seen)


def _oracle_fts_anchor(
    ordered: list[dict[str, Any]],
    arm_hits: dict[str, list[dict[str, Any]]],
    arm_order: list[str],
    limit: int,
    *,
    hit_identity,
    slots: int = _LCM_RECALL_FTS_ANCHOR_SLOTS,
    require_vector_session: bool = False,
    gate_depth: int = _LCM_RECALL_FTS_ANCHOR_GATE_DEPTH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info: dict[str, Any] = {"fired": False, "position": None}
    if len(arm_order) < 2 or "fts" not in arm_order or limit < 4 or not arm_hits.get("fts"):
        return ordered, info
    anchor_hit = arm_hits["fts"][0]
    anchor = hit_identity(anchor_hit)
    j = next((i for i, entry in enumerate(ordered) if hit_identity(entry["hit"]) == anchor), None)
    if j is None or j < limit:
        return ordered, info
    if require_vector_session:
        sid = anchor_hit.get("session_id")
        gate = set()
        for name in ("summary", "chunk"):
            if name in arm_order:
                gate |= _first_distinct_sessions(arm_hits.get(name) or [], gate_depth)
        if not sid or sid not in gate:
            info["gated"] = True
            return ordered, info
    window = ordered[:limit]
    target = min(limit, slots)
    cut = limit - 1
    seen: list[Any] = []
    for i, entry in enumerate(window):
        sid = entry["hit"].get("session_id")
        if sid and sid not in seen:
            seen.append(sid)
            if len(seen) == target:
                cut = i
                break
    cut = min(cut, limit - 1)
    ordered = list(ordered)
    entry = ordered.pop(j)
    entry["_anchor_reported"] = window[cut - 1]["_final_score"]
    info.update(fired=True, position=cut + 1, from_index=j)
    return ordered[:cut] + [entry] + ordered[cut:], info


# -- hand-built fused lists --


def _entry(kind: str, ident: int, session: str, score: float) -> dict[str, Any]:
    if kind == "node":
        hit = {"kind": "summary", "node_id": ident, "session_id": session}
    else:
        hit = {"kind": "message_excerpt", "store_id": ident, "session_id": session}
    return {"hit": hit, "rrf_score": score, "_final_score": score, "ranks": {}}


def _fused(sessions: list[str], *, anchor_at: int, anchor_session: str = "s-fts"):
    """A descending-score list of nodes, with message 999 (the FTS top-1) at ``anchor_at``."""
    ordered = []
    node = 0
    for index in range(len(sessions) + 1):
        score = 1.0 / (61 + index)
        if index == anchor_at:
            ordered.append(_entry("message", 999, anchor_session, score))
        else:
            ordered.append(_entry("node", node, sessions[node], score))
            node += 1
    fts = [dict(ordered[anchor_at]["hit"]), {"kind": "message_excerpt", "store_id": 5, "session_id": "x"}]
    return ordered, {"fts": fts, "summary": [{"kind": "summary", "node_id": 0}]}


def _ids(ordered):
    return [_hit_identity(entry["hit"]) for entry in ordered]


@pytest.mark.parametrize(
    "arm_order",
    [["fts"], ["summary", "chunk"], ["summary"], []],
    ids=["fts-only", "no-fts-two-arms", "summary-only", "no-arms"],
)
def test_inert_without_two_arms_including_fts(arm_order):
    ordered, arm_hits = _fused([f"s{i}" for i in range(30)], anchor_at=20)
    out, info = anchor(ordered, arm_hits, arm_order, 8)
    assert out is ordered
    assert info == {"fired": False, "position": None}
    assert "_anchor_reported" not in ordered[20]


def test_inert_when_limit_below_four():
    ordered, arm_hits = _fused([f"s{i}" for i in range(30)], anchor_at=20)
    out, info = anchor(ordered, arm_hits, ["fts", "summary"], 3)
    assert out is ordered and info["fired"] is False


def test_inert_when_fts_arm_is_empty():
    ordered, arm_hits = _fused([f"s{i}" for i in range(30)], anchor_at=20)
    arm_hits["fts"] = []
    out, info = anchor(ordered, arm_hits, ["fts", "summary"], 8)
    assert out is ordered and info["fired"] is False


@pytest.mark.parametrize("anchor_at", [0, 3, 7])
def test_inert_when_anchor_already_inside_the_window(anchor_at):
    ordered, arm_hits = _fused([f"s{i}" for i in range(30)], anchor_at=anchor_at)
    out, info = anchor(ordered, arm_hits, ["fts", "summary"], 8)
    assert out is ordered and info["fired"] is False


def test_inert_when_anchor_not_in_fused_list():
    ordered, arm_hits = _fused([f"s{i}" for i in range(30)], anchor_at=20)
    arm_hits["fts"][0] = {"kind": "message_excerpt", "store_id": 12345, "session_id": "gone"}
    out, info = anchor(ordered, arm_hits, ["fts", "summary"], 8)
    assert out is ordered and info["fired"] is False


def test_fires_at_the_tenth_distinct_session_start():
    # Window of 25 entries over 12 sessions: the 10th distinct session first
    # appears at index 13, so the anchor lands there (position 14).
    sessions = ["s0", "s0", "s1", "s2", "s2", "s3", "s4", "s5", "s5", "s6", "s7", "s7", "s8", "s9", "s9"]
    sessions += [f"s{i}" for i in range(10, 40)]
    ordered, arm_hits = _fused(sessions, anchor_at=30)
    before = _ids(ordered)
    out, info = anchor(ordered, arm_hits, ["fts", "summary", "chunk"], 25)
    assert info == {"fired": True, "position": 14}
    assert _ids(out)[13] == ("message", 999)
    assert _ids(out)[:13] == before[:13]
    assert _ids(out)[14:] == [i for i in before[13:] if i != ("message", 999)]
    assert len(out) == len(ordered)
    # Exactly one session leaves the scored window.
    window_sessions = {e["hit"]["session_id"] for e in out[:25]}
    assert len({e["hit"]["session_id"] for e in ordered[:25]} - window_sessions) <= 1


def test_limit_caps_placement_to_the_last_slot():
    # limit 8 -> S = 8; every entry its own session, so the 8th session starts
    # at index 7 == limit - 1: the anchor is the last delivered hit.
    ordered, arm_hits = _fused([f"s{i}" for i in range(30)], anchor_at=12)
    out, info = anchor(ordered, arm_hits, ["fts", "summary"], 8)
    assert info == {"fired": True, "position": 8}
    assert _ids(out)[7] == ("message", 999)


def test_few_distinct_sessions_falls_back_to_limit_minus_one():
    # Only 3 distinct sessions inside a limit-10 window: no S-th session, so the
    # cut is limit - 1.
    sessions = ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"] + [f"z{i}" for i in range(10)]
    ordered, arm_hits = _fused(sessions, anchor_at=15)
    out, info = anchor(ordered, arm_hits, ["fts", "summary"], 10)
    assert info == {"fired": True, "position": 10}
    assert _ids(out)[9] == ("message", 999)


def test_reported_score_is_the_entry_above_and_ranking_scores_untouched():
    ordered, arm_hits = _fused([f"s{i}" for i in range(30)], anchor_at=20)
    moved = ordered[20]
    rrf_before, final_before = moved["rrf_score"], moved["_final_score"]
    out, info = anchor(ordered, arm_hits, ["fts", "summary"], 8)
    cut = info["position"] - 1
    assert out[cut] is moved
    assert moved["rrf_score"] == rrf_before and moved["_final_score"] == final_before
    assert moved["_anchor_reported"] == out[cut - 1]["_final_score"]
    reported = [e.get("_anchor_reported", e["_final_score"]) for e in out]
    assert all(a >= b for a, b in zip(reported, reported[1:]))


def test_matches_the_frozen_replay_helper_on_random_lists():
    rng = random.Random(950)
    fired = 0
    for _case in range(400):
        n = rng.randint(1, 40)
        n_sessions = rng.randint(1, 15)
        ordered = []
        for index in range(n):
            kind = "node" if rng.random() < 0.4 else "message"
            sid = rng.choice([f"s{k}" for k in range(n_sessions)] + [None])
            ordered.append(_entry(kind, index, sid, 1.0 / (61 + index)))
        fts_pick = ordered[rng.randrange(n)]["hit"] if rng.random() < 0.9 else {
            "kind": "message_excerpt", "store_id": -1, "session_id": "s0"}
        arm_hits = {"fts": [dict(fts_pick)], "summary": [], "chunk": []}
        arm_order = rng.choice([["fts"], ["fts", "summary"], ["fts", "chunk"], ["fts", "summary", "chunk"],
                                ["summary", "chunk"]])
        limit = rng.randint(1, 25)
        slots = rng.choice([10, 10, 10, 3])
        expected_list = [dict(e, hit=e["hit"]) for e in ordered]
        actual_list = [dict(e, hit=e["hit"]) for e in ordered]
        exp, exp_info = _oracle_fts_anchor(expected_list, arm_hits, arm_order, limit,
                                           hit_identity=_hit_identity, slots=slots)
        act, act_info = anchor(actual_list, arm_hits, arm_order, limit, slots=slots)
        assert _ids(act) == _ids(exp)
        assert act_info["fired"] == exp_info["fired"]
        assert act_info["position"] == exp_info["position"]
        assert [e.get("_anchor_reported") for e in act] == [e.get("_anchor_reported") for e in exp]
        fired += bool(act_info["fired"])
    assert fired > 20  # the random cases actually exercise the firing path


# -- end-to-end through lcm_recall --

CURRENT = "session-cur"


@pytest.fixture
def recall_engine(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "recall.db"),
        embeddings_enabled=True,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_query_timeout_s=2.0,
        sensitive_patterns_enabled=True,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(tmp_path),
        current_session_id=CURRENT,
    )
    try:
        yield engine
    finally:
        dag.close()
        store.close()


class MockProvider:
    provider_id = "mock"
    model_id = "mock-model"
    dim = 2
    last_usage_tokens = 7

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _seed_two_arm_case(engine, monkeypatch, *, n_summaries=6):
    """Six summary hits in six sessions outrank the lone FTS message (0.5/61)."""
    store_id = engine._store.append("session-fts", {"role": "user", "content": "kanban dashboard sprint anchor"})
    hits = []
    for index in range(n_summaries):
        node_id = engine._dag.add_node(
            SummaryNode(
                session_id=f"session-{index}",
                depth=0,
                summary=f"summary {index}",
                token_count=20,
                source_token_count=40,
                source_ids=[],
                source_type="messages",
                created_at=10.0,
                earliest_at=10.0,
                latest_at=10.0,
                expand_hint="Expand",
            )
        )
        hits.append({
            "kind": "summary",
            "node_id": node_id,
            "session_id": f"session-{index}",
            "timestamp": 10.0,
            "snippet": f"summary {index}",
            "from_current_session": False,
            "expand_hint": "Expand",
        })
    monkeypatch.setattr(
        lcm_tools,
        "_lcm_recall_summary_arm",
        lambda *_args, **_kwargs: (list(hits), "full", len(hits), len(hits), []),
    )
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools.time, "time", lambda: 10.0)
    return store_id, [hit["node_id"] for hit in hits]


def _recall_raw(engine, **args):
    return lcm_tools.lcm_recall({"query": "kanban dashboard sprint", **args}, engine=engine)


def test_lcm_recall_anchor_fires_with_two_arms(recall_engine, monkeypatch):
    store_id, node_ids = _seed_two_arm_case(recall_engine, monkeypatch)
    payload = json.loads(_recall_raw(recall_engine, limit=4, scope_bias=0.0))

    assert payload["provenance"]["arms_run"] == ["fts", "summary"]
    assert payload["provenance"]["fts_anchor"] == {"fired": True, "position": 4}
    hits = payload["hits"]
    assert [h.get("node_id") for h in hits[:3]] == node_ids[:3]
    assert hits[3]["store_id"] == store_id
    scores = [h["score"] for h in hits]
    assert scores[3] == scores[2]
    assert all(a >= b for a, b in zip(scores, scores[1:]))


def test_lcm_recall_kill_switch_restores_fused_order(recall_engine, monkeypatch):
    _store_id, node_ids = _seed_two_arm_case(recall_engine, monkeypatch)
    recall_engine._config.recall_fts_anchor = False
    payload = json.loads(_recall_raw(recall_engine, limit=4, scope_bias=0.0))

    assert "fts_anchor" not in payload["provenance"]
    assert [h.get("node_id") for h in payload["hits"]] == node_ids[:4]


def test_lcm_recall_reports_unfired_anchor_with_two_arms(recall_engine, monkeypatch):
    store_id, _node_ids = _seed_two_arm_case(recall_engine, monkeypatch, n_summaries=2)
    payload = json.loads(_recall_raw(recall_engine, limit=8, scope_bias=0.0))

    assert payload["provenance"]["fts_anchor"] == {"fired": False, "position": None}
    assert payload["hits"][-1]["store_id"] == store_id


def _seed_fts_only(engine, monkeypatch):
    engine._config.embeddings_enabled = False
    for index in range(12):
        engine._store.append(f"session-{index % 5}", {
            "role": "user", "content": f"kanban dashboard sprint note {index}",
        })
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config: MockProvider())
    monkeypatch.setattr(lcm_tools.time, "time", lambda: 10.0)


@pytest.mark.parametrize(
    "args",
    [
        {"limit": 4},
        {"limit": 8},
        {"limit": 25, "include": "verbatim"},
        {"limit": 8, "detail": "answer_ready"},
    ],
)
def test_fts_only_output_is_byte_identical_on_and_off(recall_engine, monkeypatch, args):
    _seed_fts_only(recall_engine, monkeypatch)
    recall_engine._config.recall_fts_anchor = True
    on = _recall_raw(recall_engine, **args)
    recall_engine._config.recall_fts_anchor = False
    off = _recall_raw(recall_engine, **args)

    assert on == off
    payload = json.loads(on)
    assert payload["provenance"]["arms_run"] == ["fts"]
    assert "fts_anchor" not in payload["provenance"]
    assert payload["hits"]


def test_config_default_on_and_env_kill_switch(monkeypatch):
    assert LCMConfig().recall_fts_anchor is True
    monkeypatch.setenv("LCM_RECALL_FTS_ANCHOR", "false")
    assert LCMConfig.from_env().recall_fts_anchor is False
    monkeypatch.setenv("LCM_RECALL_FTS_ANCHOR", "true")
    assert LCMConfig.from_env().recall_fts_anchor is True
