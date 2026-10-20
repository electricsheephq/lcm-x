"""Delivery-only duplicate mitigation; all data and model seams are synthetic."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_lcm import adaptive_retrieval, evidence_compiler, host_evidence
from hermes_lcm import evidence_pack, preanswer_evidence, retrieval_core, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"), embeddings_enabled=False)
    store = MessageStore(config.database_path, ingest_protection_config=config)
    dag = SummaryDAG(config.database_path)
    value = SimpleNamespace(
        _config=config, _store=store, _dag=dag, _hermes_home=str(tmp_path),
        current_session_id="s0",
    )
    try:
        yield value
    finally:
        dag.close()
        store.close()


def append(engine, content="quartz reply", session="s0", **fields):
    return engine._store.append(
        session, {"role": "assistant", "content": content, **fields},
        conversation_id=fields.get("conversation_id", ""),
    )


def recall(engine, **args):
    return json.loads(tools.lcm_recall({"query": "quartz", **args}, engine=engine))


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_unstamped_replay_shape_and_expand_own_row(engine, detail):
    # #1674 is unmerged at the base: minimal D5 shape, a kept plain reply
    # appended again after unstamped host history is replayed.
    ids = [append(engine), append(engine)]
    before = engine._store.get_batch(ids)
    payload = recall(engine, detail=detail)
    assert len(payload["hits"]) == 1
    assert payload["hits"][0]["store_id"] == ids[0]
    assert payload["hits"][0]["duplicate_store_ids"] == ids[1:]
    for store_id in ids:
        expanded = json.loads(tools.lcm_expand({"store_id": store_id}, engine=engine))
        assert expanded["store_id"] == store_id
        assert "quartz reply" in expanded["content"]
    assert engine._store.get_batch(ids) == before


@pytest.mark.parametrize("dimension", ["role", "tool_call_id", "session", "conversation_id", "content"])
def test_group_key_uses_full_identity(engine, dimension):
    base = append(engine, "quartz " + "x" * 500, role="tool", tool_call_id="call-a")
    duplicate = append(engine, "quartz " + "x" * 500, role="tool", tool_call_id="call-a")
    fields = {"role": "tool", "tool_call_id": "call-a"}
    session = "s0"
    content = "quartz " + "x" * 500
    if dimension == "session":
        session = "s1"
    elif dimension == "content":
        content += "different suffix beyond the snippet"
    else:
        fields[dimension] = {"role": "assistant", "tool_call_id": "call-b", "conversation_id": "other"}[dimension]
    distinct = append(engine, content, session=session, **fields)
    hits = recall(engine)["hits"]
    assert {hit["store_id"] for hit in hits} == {base, distinct}
    assert next(hit for hit in hits if hit["store_id"] == base)["duplicate_store_ids"] == [duplicate]


@pytest.mark.parametrize("mode", ["full_text", "semantic", "hybrid"])
@pytest.mark.parametrize("sort", ["recency", "relevance", "hybrid"])
def test_grep_groups_before_limit(engine, mode, sort):
    other = append(engine, "quartz other")
    first = append(engine)
    second = append(engine)
    payload = json.loads(tools.lcm_grep({"query": "quartz", "limit": 2, "mode": mode, "sort": sort}, engine=engine))
    assert {hit["store_id"] for hit in payload["results"]} == {other, first}
    assert next(hit for hit in payload["results"] if hit["store_id"] == first)["duplicate_store_ids"] == [second]


@pytest.mark.parametrize("strict", [False, True])
def test_caps_count_groups_once(engine, strict):
    # 30 rows, 10 duplicate pairs, 20 distinct messages in four sessions.
    for session in range(4):
        for index in range(5):
            text = f"quartz fact {session}-{index}"
            append(engine, text, session=f"s{session}")
            if session < 2:
                append(engine, text, session=f"s{session}")
    engine._config.recall_reference_strict = strict
    snippets = recall(engine, limit=25)
    assert len(snippets["hits"]) == 20
    answer = recall(engine, detail="answer_ready", limit=25)
    assert len(answer["hits"]) == (8 if strict else 20)
    assert len({engine._store.get(hit["store_id"])["content"] for hit in answer["hits"]}) == len(answer["hits"])
    assert sum("content" in hit for hit in answer["hits"]) == 8
    assert max(sum(hit["session_id"] == f"s{s}" for hit in answer["hits"]) for s in range(4)) == 5


def test_fusion_max_score_union_arms_earliest_id_and_summary_exemption(engine):
    first, second = append(engine), append(engine)
    rows = engine._store.get_batch([first, second])
    def hit(sid):
        return {"kind": "message_excerpt", "store_id": sid, "snippet": "quartz reply"}
    arms = {"fts": [hit(second), hit(first)], "chunk": [hit(second)]}
    arms["summary"] = [{"kind": "summary", "node_id": n, "store_id": first, "snippet": "same"} for n in (1, 2)]
    original, _, _ = retrieval_core.fuse_candidates(arms, None)
    before = copy.deepcopy(original)
    grouped = retrieval_core.collapse_duplicate_candidates(original, rows)
    messages = [entry for entry in grouped if entry["hit"]["kind"] != "summary"]
    assert len(messages) == 1
    entry = messages[0]
    assert entry["hit"]["store_id"] == first
    assert entry["hit"]["duplicate_store_ids"] == [second]
    assert entry["rrf_score"] == max(e["rrf_score"] for e in original if e["hit"]["kind"] != "summary")
    assert set(entry["ranks"]) == {0, 2}
    assert len([e for e in grouped if e["hit"]["kind"] == "summary"]) == 2
    assert original == before


def test_group_keeps_citable_duplicate_chunk_after_expansion_cap(engine):
    first, second = append(engine), append(engine)
    arms = {
        "fts": [{"kind": "message_excerpt", "store_id": first,
                 "session_id": "s0", "snippet": ">>>quartz<<< reply", "content_offset": 0}],
        "chunk": [{"kind": "message_excerpt", "store_id": second,
                   "session_id": "s0", "snippet": "quartz reply", "content_offset": 0,
                   "chunk_span": {"char_start": 0, "char_end": 12, "chunk_index": 0}}],
    }
    entries, _, _ = retrieval_core.fuse_candidates(arms, None)
    entries = retrieval_core.collapse_duplicate_candidates(entries, engine._store.get_batch([first, second]))
    selector = tools._LcmRecallStrictSelector(entries, engine=engine, per_session_limit=5, expanded_limit=0)
    selected = selector.take(1)
    assert len(selected) == 1
    assert selected[0]["hit"]["store_id"] == first
    assert selected[0]["hit"]["duplicate_store_ids"] == [second]
    assert selected[0]["hit"]["snippet"] == "quartz reply"


def duplicate_free_responses(engine, monkeypatch):
    """Frozen clock and source timestamps make main's exact JSON reproducible."""
    monkeypatch.setattr(tools.time, "time", lambda: 1_700_000_100.0)
    for index, text in enumerate(("quartz café", "quartz second", "unrelated")):
        append(engine, text, timestamp=1_700_000_000 + index)
    monkeypatch.setattr(tools.time, "time", lambda: 1_700_000_100.0)
    output = {}
    for detail in ("snippets", "answer_ready"):
        output[f"recall-{detail}"] = tools.lcm_recall({"query": "quartz", "detail": detail}, engine=engine)
    for mode in ("full_text", "semantic", "hybrid"):
        for sort in ("recency", "relevance", "hybrid"):
            output[f"grep-{mode}-{sort}"] = tools.lcm_grep({"query": "quartz", "mode": mode, "sort": sort}, engine=engine)
    engine._config.embeddings_enabled = True
    provider = SimpleNamespace(provider_id="fixture", model_id="fixture", dim=2)
    monkeypatch.setattr(tools, "_lcm_grep_resolve_provider", lambda *a, **k: provider)
    monkeypatch.setattr(tools, "_lcm_grep_embed_query", lambda *a, **k: [1.0, 0.0])
    class Ranked(list):
        coverage = "full"
    monkeypatch.setattr(tools, "run_knn", lambda *a, **k: Ranked([("7", 0.8, "summary")]))
    node = SimpleNamespace(node_id=7, session_id="s0", depth=0, summary="quartz summary",
                           token_count=3, expand_hint="lcm_expand(node_id=7)", earliest_at=0, latest_at=0)
    monkeypatch.setattr(tools, "hydrate_semantic_nodes", lambda *a, **k: [(node, 0.8)])
    for mode in ("semantic", "hybrid"):
        output[f"grep-active-{mode}"] = tools.lcm_grep({"query": "quartz", "mode": mode}, engine=engine)
    return output


def test_duplicate_free_bytes_match_main(engine, monkeypatch):
    golden = json.loads((Path(__file__).parent / "fixtures/issue_1681_duplicate_free.json").read_text())
    assert duplicate_free_responses(engine, monkeypatch) == golden


@pytest.mark.parametrize("consumer", ["pack", "compiler", "host", "preanswer", "adaptive", "baseline"])
def test_structured_consumers_preserve_duplicate_ids(engine, consumer):
    first, second = append(engine, "I read 5 pages."), append(engine, "I read 5 pages.")
    hit = {"store_id": first, "content": "I read 5 pages.", "content_offset": 0,
           "exact_ref": f"lcm:{first}:0-15", "quote": "I read 5 pages.", "duplicate_store_ids": [second]}
    if consumer == "pack":
        packed = json.loads(evidence_pack.build_evidence_pack({"question": "What did I read?", "baseline_refs": [hit]}, engine=engine))
        result = packed["evidence"][0]
    elif consumer in {"compiler", "host"}:
        prepared = evidence_compiler.prepare_evidence_selector("What did I read?", baseline_refs=[hit])
        assert prepared["selector_request"]["baseline_evidence"][0]["duplicate_store_ids"] == [second]
        proposal = {"version": "evidence-selector-v1", "selections": [{"claim_id": "a", "facet": "answer", "exact_ref": hit["exact_ref"], "quote": hit["quote"]}], "missing_facets": []}
        compiled = evidence_compiler.compile_evidence("What did I read?", engine=engine, baseline_refs=[hit], enabled=True, selector=lambda _: proposal)
        result = (host_evidence._context_payload(compiled) if consumer == "host" else compiled)["evidence"][0]
    elif consumer == "preanswer":
        result = preanswer_evidence._automatic_candidate(hit, engine=engine, plan=SimpleNamespace(operation="sum"))
    elif consumer == "adaptive":
        result = adaptive_retrieval._extract_exact_evidence({"hits": [hit]}, engine=engine)[0].public_dict()
    else:
        spec = importlib.util.spec_from_file_location("hermes_lcm.issue1681_plugin", Path(__file__).parents[1] / "__init__.py")
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        result = plugin._answer_ready_baseline(SimpleNamespace(handle_tool_call=lambda *_: {"hits": [hit]}), "q", {})[0]
    assert result["duplicate_store_ids"] == [second]
