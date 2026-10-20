"""Golden public recall responses with real rows and deterministic retrieval arms."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import tools
from tests.harness import engine_context, pin_char_counter, user


NOW = 1_700_000_000.0
GOLDENS = Path(__file__).with_name("recall_pipeline_golden.json")


def golden(name, raw):
    # All IDs are assigned in fixture order and timestamps pinned before seeding.
    # Compare the entire JSON document, including absence of optional fields.
    expected = json.loads(GOLDENS.read_text())[name]
    assert json.loads(raw) == expected


@pytest.fixture
def engine(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(tools.time, "time", lambda: NOW)
    with engine_context(tmp_path, window=200_000, session_id="current", embeddings_enabled=False) as value:
        yield value


def seed(engine, texts):
    return [engine._store.append(f"session-{i}", user(text)) for i, text in enumerate(texts)]


def excerpt(engine, sid, *, start=0, end=None, chunk=False):
    row = engine._store.get(sid)
    content = row["content"]
    end = len(content) if end is None else end
    hit = dict(kind="message_excerpt", store_id=sid, session_id=row["session_id"],
               timestamp=NOW, snippet=content[start:end], role="user", source="",
               content_offset=start, from_current_session=False,
               expand_hint=f"lcm_expand(store_id={sid}, content_offset={start})")
    if chunk:
        hit["chunk_span"] = {"char_start": start, "char_end": end}
    return hit


def arms(monkeypatch, engine, *, fts=(), summary=(), chunk=(), leads=(), rerank=False):
    """Only retrieval/provider I/O is substituted; fusion and delivery stay real.

    The semantic switch must be on to reach mixed-arm fusion. No real embedding
    provider is loaded or called: query vectors and arm results are fixed values.
    """
    engine._config.embeddings_enabled = True
    engine._config.rerank_enabled = rerank
    provider = SimpleNamespace(provider_id="voyage", model_id="fixture", last_usage_tokens=0)
    provider.rerank = lambda query, documents, **kwargs: [(i, 0.9) for i in reversed(range(len(documents)))]
    monkeypatch.setattr(tools, "_resolve_recall_provider", lambda *a, **k: provider)
    monkeypatch.setattr(tools, "_resolve_recall_chunk_provider", lambda *a, **k: provider)
    monkeypatch.setattr(tools, "_lcm_grep_embed_query", lambda *a, **k: [1.0, 0.0])
    monkeypatch.setattr(tools, "_lcm_recall_has_usable_vector_corpus", lambda *a, **k: False)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", lambda *a, **k: (deepcopy(fts), None))
    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", lambda *a, **k: (
        deepcopy(summary), "full", len(summary), len(summary), deepcopy(leads),
    ))
    monkeypatch.setattr(tools, "_lcm_recall_chunk_arm", lambda *a, **k: (
        deepcopy(chunk), "full", len(chunk), len(chunk),
    ))


def recall(engine, **kwargs):
    return tools.lcm_recall({"query": "cobalt", "scope_bias": 0.0, **kwargs}, engine=engine)


@pytest.mark.parametrize("include", ["all", "summaries", "verbatim"])
def test_fts_only_and_summaries_fallback(engine, include):
    ids = seed(engine, ["cobalt north", "unrelated south"])
    raw = recall(engine, include=include, limit=2)
    payload = json.loads(raw)
    assert [h["store_id"] for h in payload["hits"]] == ids[:1]
    assert payload["metrics"]["embedding_query_calls"] == 0
    assert payload["provenance"]["arms_run"] == ["fts"]
    assert "fts_anchor" not in payload["provenance"]
    golden(f"fts-{include}", raw)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_ties_duplicate_chunks_and_paired_span(engine, monkeypatch, weight):
    a, b = seed(engine, ["prefix cobalt alpha tail", "cobalt beta"])
    fts_a = excerpt(engine, a)
    # A tied chunk must replace snippet AND offset; duplicate chunks count once.
    best = excerpt(engine, a, start=7, end=19, chunk=True)
    worst = excerpt(engine, a, start=20, chunk=True)
    arms(monkeypatch, engine, fts=[fts_a, excerpt(engine, b)], chunk=[best, worst, excerpt(engine, b, chunk=True)])
    engine._config.recall_arm_weights = {"fts": weight, "summary": 1.0, "chunk": weight}
    raw = recall(engine, include="verbatim", limit=2)
    hits = json.loads(raw)["hits"]
    assert [h["store_id"] for h in hits] == [a, b]
    assert hits[0]["snippet"] == "cobalt alpha"
    assert hits[0]["chunk_span"] == {"char_start": 7, "char_end": 19}
    assert "content_offset=7" in hits[0]["expand_hint"]
    assert hits[0]["score"] == round(2 * weight / 61, 6)
    assert hits[1]["score"] == round(weight / 62 + weight / 63, 6)
    golden(f"merge-weight-{weight}", raw)


def test_equal_rank_uses_earliest_arm_representation(engine, monkeypatch):
    a, b = seed(engine, ["cobalt alpha", "cobalt beta"])
    summary_a = {**excerpt(engine, a), "snippet": "summary representation"}
    arms(monkeypatch, engine, fts=[excerpt(engine, a), excerpt(engine, b)],
         summary=[excerpt(engine, b), summary_a])
    engine._config.recall_arm_weights = {"fts": 1.0, "summary": 1.0}
    raw = recall(engine, limit=2)
    hits = json.loads(raw)["hits"]
    assert [h["store_id"] for h in hits] == [a, b]
    assert hits[0]["snippet"] == "cobalt alpha"
    assert hits[0]["score"] == hits[1]["score"] == round(1 / 61 + 1 / 62, 6)
    golden("arm-order-tie", raw)


def test_rerank_changes_order_without_replacing_reported_scores(engine, monkeypatch):
    ids = seed(engine, ["cobalt alpha", "cobalt beta"])
    arms(monkeypatch, engine, fts=[excerpt(engine, i) for i in ids], rerank=True)
    raw = recall(engine, limit=2)
    payload = json.loads(raw)
    assert [h["store_id"] for h in payload["hits"]] == ids[::-1]
    assert [h["score"] for h in payload["hits"]] == [round(0.5 / 62, 6), round(0.5 / 61, 6)]
    assert payload["provenance"]["rerank"] == "applied"
    golden("rerank", raw)


@pytest.mark.parametrize("markers", [False, True])
def test_unicode_full_encoded_cap_and_delta_refs(engine, monkeypatch, markers):
    seed(engine, [f"cobalt {i} " + "記憶🌿" * 650 for i in range(3)])
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 6000)
    options = dict(include="verbatim", detail="answer_ready", limit=3, seen_refs=[],
                   include_occurrence_time=markers)
    raw = recall(engine, **options)
    payload = json.loads(raw)
    assert len(raw) <= 6000  # The contract is Unicode characters, not UTF-8 bytes.
    assert len(raw.encode("utf-8")) > len(raw)
    assert payload["provenance"]["answer_ready"]["response_truncated"]
    assert 0 < len(payload["hits"]) < 3
    refs = [h["exact_ref"] for h in payload["hits"]]
    assert payload["delta"]["novel_refs"] == refs
    assert payload["delta"]["novel_ref_count"] == len(refs)
    for hit in payload["hits"]:
        row = engine._store.get(hit["store_id"])
        offset, length = hit["content_offset"], hit["content_returned_chars"]
        assert hit["content"] == row["content"][offset:offset + length]
        assert ("occurrence_time" in hit) is markers
    golden(f"unicode-cap-markers-{markers}", raw)
    delta_raw = recall(engine, **{**options, "seen_refs": refs})
    delta = json.loads(delta_raw)
    assert not set(refs) & set(delta["delta"]["novel_refs"])
    golden(f"unicode-delta-markers-{markers}", delta_raw)


def test_strict_refill_and_summary_leads(engine, monkeypatch):
    a, b = seed(engine, ["cobalt first", "cobalt second"])
    invalid = {"kind": "summary", "node_id": "lead", "session_id": "summary-session",
               "snippet": "cobalt lead", "timestamp": NOW, "expand_hint": "lcm_expand(node_id='lead')"}
    leads = [{"node_id": "lead", "session_id": "summary-session", "expand_hint": "lcm_expand(node_id='lead')"}]
    arms(monkeypatch, engine, summary=[invalid, excerpt(engine, a), excerpt(engine, b)], leads=leads)
    raw = recall(engine, include="summaries", detail="answer_ready", limit=1, seen_refs=[])
    payload = json.loads(raw)
    assert [h["store_id"] for h in payload["hits"]] == [a]
    assert payload["provenance"]["answer_ready"]["summary_leads"] == leads
    assert payload["provenance"]["answer_ready"]["unreferenced_dropped_count"] == 1
    golden("strict-refill", raw)
    refs = payload["delta"]["novel_refs"]
    raw = recall(engine, include="summaries", detail="answer_ready", limit=1, seen_refs=refs)
    payload = json.loads(raw)
    assert [h["store_id"] for h in payload["hits"]] == [b]
    golden("strict-delta-refill", raw)
    refs += payload["delta"]["novel_refs"]
    raw = recall(engine, include="summaries", detail="answer_ready", limit=1, seen_refs=refs)
    assert json.loads(raw)["delta"]["termination_reason"] == "no_novel_exact_ref"
    golden("strict-delta-exhausted", raw)


def test_summary_leads_evicted_to_fit_full_response(engine, monkeypatch):
    seed(engine, ["cobalt seed"])
    leads = [{"node_id": i, "session_id": "s" * 2000, "expand_hint": "x" * 2000} for i in range(3)]
    arms(monkeypatch, engine, leads=leads)
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 6000)
    raw = recall(engine, include="summaries", detail="answer_ready", seen_refs=[])
    payload = json.loads(raw)
    assert len(raw) <= 6000
    assert payload["hits"] == payload["delta"]["novel_refs"] == []
    assert len(payload["provenance"]["answer_ready"]["summary_leads"]) < len(leads)
    assert payload["provenance"]["answer_ready"]["response_truncated"]
    golden("leads-cap", raw)


@pytest.mark.parametrize("cap", [6000, 3000])
@pytest.mark.parametrize("anchor", [False, True])
def test_anchor_delivery_after_final_cap(engine, monkeypatch, cap, anchor):
    ids = seed(engine, [f"cobalt evidence {i}" for i in range(6)])
    arms(monkeypatch, engine, fts=[excerpt(engine, ids[0])],
         chunk=[excerpt(engine, i, chunk=True) for i in ids[1:]])
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    raw = tools.lcm_recall({"query": "cobalt", "scope_bias": 0.0, "detail": "answer_ready",
                            "limit": 4, "seen_refs": []}, engine=engine, fts_anchor=anchor)
    payload = json.loads(raw)
    delivered = [h["store_id"] for h in payload["hits"]]
    assert len(raw) <= cap
    if anchor:
        info = payload["provenance"]["fts_anchor"]
        assert info["fired"] is True
        assert info["delivered"] is (ids[0] in delivered)
        assert info["delivered"] is (cap == 6000)
        if info["delivered"]:
            assert payload["hits"][-1]["score"] == round(0.5 / 61, 6)
    else:
        assert "fts_anchor" not in payload["provenance"]
        assert ids[0] not in delivered
    assert payload["delta"]["novel_refs"] == [h["exact_ref"] for h in payload["hits"]]
    golden(f"anchor-{anchor}-cap-{cap}", raw)
