"""Fixed-clock public recall contracts, captured before ranking extraction."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore
import hermes_lcm.tools as tools

NOW = 1_700_000_000.0
QUERIES = [
    "kanban", "dashboard", "sprint", "alpha", "beta", "gamma", "delta",
    "epsilon", "zeta", "anchor", "kanban dashboard", "sprint alpha",
    "beta gamma", "delta epsilon", "zeta anchor", "unknown", "what did we plan",
    "dashboard sprint", "alpha zeta", "kanban anchor",
]
GOLDEN = Path(__file__).with_name("recall_ranking_golden.json")


class Provider:
    provider_id = "voyage"
    model_id = "contract-model"
    dim = 2
    last_usage_tokens = 0

    def __init__(self):
        self.documents = []

    def embed_query(self, text):
        return [1.0, 0.0]

    def rerank(self, query, documents, **kwargs):
        self.documents.append(list(documents))
        return [(i, 0.9) for i in reversed(range(len(documents)))]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(tools.time, "time", lambda: NOW)
    config = LCMConfig(
        database_path=str(tmp_path / "ranking.db"), embeddings_enabled=True,
        embedding_provider="voyage", embedding_model="contract-model",
        sensitive_patterns_enabled=True,
    )
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    engine = SimpleNamespace(
        _config=config, _store=store, _dag=dag, _hermes_home=str(tmp_path),
        current_session_id="session-0",
    )
    hits = []
    for i, word in enumerate(["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "anchor"]):
        text = f"prefix kanban dashboard sprint {word} trailing"
        sid = store.append(f"session-{i}", {"role": "user", "content": text})
        hits.append({
            "kind": "message_excerpt", "store_id": sid, "session_id": f"session-{i}",
            "timestamp": NOW - i * 86400, "snippet": text[7:-9],
            "content_offset": 7,
            "chunk_span": {"char_start": 7, "char_end": len(text) - 9},
            "from_current_session": i == 0,
            "expand_hint": f"lcm_expand(store_id={sid}, content_offset=7)",
        })
    # Distinct representations and duplicate chunks exercise provenance adoption.
    chunks = deepcopy(hits[:6]) + [{**hits[0], "snippet": "trailing", "content_offset": 37}]
    summaries = [{**h, "snippet": h["snippet"] + " trailing"} for h in reversed(hits[:6])]
    provider = Provider()
    monkeypatch.setattr(tools, "resolve_provider", lambda _config: provider)
    monkeypatch.setattr(tools, "_resolve_recall_chunk_provider", lambda *a, **kw: provider)
    monkeypatch.setattr(tools, "_lcm_grep_embed_query", lambda *a, **kw: [1.0, 0.0])
    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", lambda *a, **kw: (
        deepcopy(summaries), "full", len(summaries), len(summaries), [],
    ))
    monkeypatch.setattr(tools, "_lcm_recall_chunk_arm", lambda *a, **kw: (
        deepcopy(chunks), "full", len(chunks), len(chunks),
    ))
    try:
        yield engine, hits, summaries, chunks, provider
    finally:
        dag.close()
        store.close()


def recall(corpus, **args):
    return tools.lcm_recall({"query": "kanban", "limit": 7, **args}, engine=corpus[0])


def golden_responses(corpus):
    engine = corpus[0]
    responses = []
    for i, query in enumerate(QUERIES):
        engine._config.rerank_enabled = i % 4 == 0
        engine._config.recall_arm_weights = {
            "fts": 0.0 if i % 5 == 0 else 0.5, "summary": 1.0, "chunk": 1.0,
        }
        responses.append(recall(
            corpus, query=query, limit=4 if i % 2 else 7,
            scope_bias=0.0 if i % 3 else 0.25,
            include=["all", "verbatim", "summaries"][i % 3],
        ))
    # IDs and times are deterministic from the fresh seeded store and fixed clock.
    # Keep the complete serialized responses as strings: formatting also matters.
    return json.dumps(responses, ensure_ascii=False, indent=2) + "\n"


def test_golden_responses(corpus):
    assert golden_responses(corpus) == GOLDEN.read_text()


def test_duplicate_chunk_adopts_paired_span_and_keeps_provenance(corpus):
    payload = json.loads(recall(corpus, scope_bias=0))
    target = corpus[1][0]
    hits = [h for h in payload["hits"] if h.get("store_id") == target["store_id"]]
    assert len(hits) == 1
    hit = hits[0]
    assert hit["snippet"] == target["snippet"]
    assert hit["chunk_span"] == target["chunk_span"]
    assert hit["expand_hint"] == target["expand_hint"]
    assert hit["arms"] == ["chunk", "fts", "summary"]


def test_prior_precedes_rerank_and_rerank_preserves_scores(corpus):
    baseline = json.loads(recall(corpus, scope_bias=0.8))
    corpus[0]._config.rerank_enabled = True
    result = json.loads(recall(corpus, scope_bias=0.8))
    assert result["provenance"]["rerank"] == "applied"
    assert corpus[4].documents[-1] == [h["snippet"] for h in baseline["hits"]]
    scores = {h["store_id"]: h["score"] for h in baseline["hits"]}
    assert {h["store_id"]: h["score"] for h in result["hits"]} == scores
    assert [h["store_id"] for h in result["hits"]] == list(reversed(list(scores)))


@pytest.mark.parametrize("weight,enabled", [(0.5, True), (0.0, True), (0.5, False)])
def test_anchor_after_rerank_and_zero_weight(corpus, monkeypatch, weight, enabled):
    engine, hits, _, _, _ = corpus
    anchor = deepcopy(hits[-1])
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", lambda *a, **kw: ([anchor], None))
    engine._config.recall_arm_weights = {"fts": weight, "summary": 1.0, "chunk": 1.0}
    engine._config.recall_fts_anchor = enabled
    # No rerank: the weak FTS-only hit sits below six dual-arm candidates.
    payload = json.loads(recall(corpus, limit=4, scope_bias=0))
    if enabled:
        assert payload["provenance"]["fts_anchor"]["fired"] is (weight > 0)
    else:
        assert "fts_anchor" not in payload["provenance"]
    assert (payload["hits"][-1]["store_id"] == anchor["store_id"]) is (enabled and weight > 0)


def test_arm_ties_and_alternates_reach_rerank(corpus, monkeypatch):
    engine, hits, summaries, chunks, _ = corpus
    fts = [{**hits[0], "snippet": "fts preview", "chunk_span": None}]
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", lambda *a, **kw: (fts, None))
    captured = []

    def capture(provider, query, ordered, **kwargs):
        captured.extend(deepcopy(ordered))
        return ordered, "disabled", []

    monkeypatch.setattr(tools, "_lcm_recall_rerank", capture)
    recall(corpus, scope_bias=0)
    first = next(e for e in captured if e["hit"]["store_id"] == hits[0]["store_id"])
    assert first["ranks"] == {0: 1, 1: 6, 2: 1}
    assert first["rrf_score"] == 0.5 / 61 + 1 / 66 + 1 / 61
    assert first["hit"]["snippet"] == chunks[0]["snippet"]
    assert first["_alternates"] == [summaries[-1], chunks[0], chunks[-1]]
