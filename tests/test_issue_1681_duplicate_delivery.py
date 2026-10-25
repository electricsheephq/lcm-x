"""Delivery-only duplicate mitigation; all data and model seams are synthetic."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
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


def stamped_copy(engine, content="quartz reply", **fields):
    """Known same-occurrence copies; plain append deliberately stays unstamped."""
    return append(engine, content, timestamp=1_700_000_000, **fields)


@pytest.mark.parametrize("delivery", ["grep-recency", "grep-hybrid", "snippets", "answer_ready"])
@pytest.mark.parametrize("stamps", [(None, None, None), (1_700_000_000, None, None),
                                    (1_700_000_000, None, 1_700_000_000),
                                    (1_700_000_000,) * 3])
def test_ranked_representative_prefers_best_stamped_member(engine, monkeypatch, delivery, stamps):
    first = append(engine, timestamp=stamps[0])
    other = append(engine, "quartz other")
    second = append(engine, timestamp=stamps[1])
    third = append(engine, timestamp=stamps[2])
    # Storage time controls ranking independently of source occurrence time.
    for store_id, timestamp in [(first, 1_700_000_000), (other, 1_700_000_200),
                                (second, 1_700_000_300), (third, 1_700_000_100)]:
        engine._store._conn.execute("UPDATE messages SET timestamp = ? WHERE store_id = ?",
                                    (timestamp, store_id))
    engine._store._conn.commit()
    before = engine._store.get_batch([first, other, second, third])
    ranked = [second, other, third, first]
    if delivery.startswith("grep"):
        sort = delivery.split("-")[1]
        assert [hit["store_id"] for hit in engine._store.search("quartz", sort=sort)] == ranked
        hits = json.loads(tools.lcm_grep({"query": "quartz", "sort": sort}, engine=engine))["results"]
    else:
        # Exercise recall delivery with a deterministic ranked FTS arm.
        fuse = tools.fuse_candidates
        def ranked_fuse(*args):
            entries, arms, weights = fuse(*args)
            entries.sort(key=lambda entry: ranked.index(entry["hit"]["store_id"]))
            for index, entry in enumerate(entries):
                entry["rrf_score"] = 1.0 / (index + 1)
            return entries, arms, weights
        monkeypatch.setattr(tools, "fuse_candidates", ranked_fuse)
        hits = recall(engine, detail=delivery)["hits"]
    stamped = {sid for sid, stamp in zip((first, second, third), stamps) if stamp is not None}
    expected = next((sid for sid in ranked if sid in stamped), None)
    assert [hit["store_id"] for hit in hits] == [
        sid for sid in ranked if sid not in stamped or sid == expected
    ]
    for hit in hits:
        if hit["store_id"] == expected and len(stamped) > 1:
            assert hit["duplicate_store_ids"] == sorted(stamped - {expected})
        else:
            assert "duplicate_store_ids" not in hit
    assert engine._store.get_batch([first, other, second, third]) == before


@pytest.mark.parametrize("delivery", ["snippets", "answer_ready", "grep"])
def test_distinct_recorded_times_preserve_latest_restatement(engine, delivery):
    ids = [
        append(engine, f"I live in {city} now", role="user", timestamp=1_700_000_000 + day * 86400)
        for city, day in [("Berlin", 0), ("Paris", 10), ("Berlin", 20)]
    ]
    if delivery == "grep":
        hits = json.loads(tools.lcm_grep({"query": "live"}, engine=engine))["results"]
    else:
        hits = json.loads(tools.lcm_recall({"query": "live", "detail": delivery}, engine=engine))["hits"]
    assert {hit["store_id"] for hit in hits} == set(ids)
    assert all("duplicate_store_ids" not in hit for hit in hits)
    # A latest-fact consumer must still receive the day-20 Berlin occurrence.
    latest = max(hits, key=lambda hit: engine._store.get(hit["store_id"])["observed_at"])
    assert latest["store_id"] == ids[-1]
    assert engine._store.get(latest["store_id"])["content"] == "I live in Berlin now"


@pytest.mark.parametrize("delivery", ["snippets", "answer_ready", "grep"])
def test_unstamped_real_user_turns_preserve_repetition_and_order(engine, delivery):
    first = append(engine, "quartz do it again", role="user")
    middle = append(engine, "quartz done", role="assistant")
    last = append(engine, "quartz do it again", role="user")
    ids = [first, middle, last]
    # Distinct storage positions/times are not proof of a replay copy.
    for index, store_id in enumerate(ids):
        engine._store._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE store_id = ?",
            (1_700_000_000 + index, store_id),
        )
    engine._store._conn.commit()
    before = engine._store.get_batch(ids)
    assert all(row["observed_at"] is None for row in before.values())
    if delivery == "grep":
        hits = json.loads(tools.lcm_grep({"query": "quartz"}, engine=engine))["results"]
    else:
        hits = recall(engine, detail=delivery)["hits"]
    assert {hit["store_id"] for hit in hits} == set(ids)
    assert [hit["store_id"] for hit in sorted(hits, key=lambda hit: hit["timestamp"])] == ids
    assert sum("do it again" in hit.get("content", hit.get("snippet", "")) for hit in hits) == 2
    assert all("duplicate_store_ids" not in hit for hit in hits)
    assert engine._store.get_batch(ids) == before


@pytest.mark.parametrize("delivery", ["snippets", "answer_ready", "grep"])
@pytest.mark.parametrize("dimension", ["tool_calls", "source", "tool_name"])
def test_stamped_distinct_provenance_is_not_a_replay(engine, delivery, dimension):
    ids = []
    for index in range(2):
        fields = {"role": "assistant", "content": "quartz reply", "timestamp": 1_700_000_000}
        if dimension == "tool_calls":
            fields["tool_calls"] = [{"id": f"call-{index}", "type": "function",
                                     "function": {"name": "lookup", "arguments": "{}"}}]
        if dimension == "tool_name":
            fields.update(role="tool", tool_name=f"lookup-{index}")
        ids.append(engine._store.append(
            "s0", fields, source=f"source-{index}" if dimension == "source" else "",
        ))
    before = engine._store.get_batch(ids)
    if delivery == "grep":
        hits = json.loads(tools.lcm_grep({"query": "quartz"}, engine=engine))["results"]
    else:
        hits = recall(engine, detail=delivery)["hits"]
    assert {hit["store_id"] for hit in hits} == set(ids)
    assert all("duplicate_store_ids" not in hit for hit in hits)
    assert engine._store.get_batch(ids) == before


@pytest.mark.parametrize("delivery", ["snippets", "answer_ready", "grep"])
@pytest.mark.parametrize("copy_timestamp", [None, 1_700_000_000])
def test_only_stamped_same_occurrence_copies_collapse(engine, delivery, copy_timestamp):
    first = append(engine, timestamp=1_700_000_000)
    second = append(engine, timestamp=copy_timestamp)
    before = engine._store.get_batch([first, second])
    assert before[first]["observed_at"] == 1_700_000_000
    assert before[second]["observed_at"] == copy_timestamp
    if delivery == "grep":
        hits = json.loads(tools.lcm_grep({"query": "quartz"}, engine=engine))["results"]
    else:
        hits = recall(engine, detail=delivery)["hits"]
    if copy_timestamp is None:
        assert {hit["store_id"] for hit in hits} == {first, second}
        assert all("duplicate_store_ids" not in hit for hit in hits)
    else:
        assert len(hits) == 1
        assert hits[0]["store_id"] == second
        assert hits[0]["duplicate_store_ids"] == [first]
    assert engine._store.get_batch([first, second]) == before


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_unstamped_replay_shape_is_ambiguous_and_expand_keeps_own_row(engine, detail):
    # #1674 is unmerged at the base: minimal D5 shape, a kept plain reply
    # appended again after unstamped host history is replayed.
    ids = [append(engine), append(engine)]
    before = engine._store.get_batch(ids)
    payload = recall(engine, detail=detail)
    assert {hit["store_id"] for hit in payload["hits"]} == set(ids)
    assert all("duplicate_store_ids" not in hit for hit in payload["hits"])
    for store_id in ids:
        expanded = json.loads(tools.lcm_expand({"store_id": store_id}, engine=engine))
        assert expanded["store_id"] == store_id
        assert "quartz reply" in expanded["content"]
    assert engine._store.get_batch(ids) == before


@pytest.mark.parametrize("dimension", ["role", "tool_call_id", "session", "conversation_id", "content"])
def test_group_key_uses_full_identity(engine, dimension):
    base = stamped_copy(engine, "quartz " + "x" * 500, role="tool", tool_call_id="call-a")
    duplicate = stamped_copy(engine, "quartz " + "x" * 500, role="tool", tool_call_id="call-a")
    fields = {"role": "tool", "tool_call_id": "call-a"}
    session = "s0"
    content = "quartz " + "x" * 500
    if dimension == "session":
        session = "s1"
    elif dimension == "content":
        content += "different suffix beyond the snippet"
    else:
        fields[dimension] = {"role": "assistant", "tool_call_id": "call-b", "conversation_id": "other"}[dimension]
    distinct = stamped_copy(engine, content, session=session, **fields)
    hits = recall(engine)["hits"]
    assert {hit["store_id"] for hit in hits} == {duplicate, distinct}
    assert next(hit for hit in hits if hit["store_id"] == duplicate)["duplicate_store_ids"] == [base]


@pytest.mark.parametrize("mode", ["full_text", "semantic", "hybrid"])
@pytest.mark.parametrize("sort", ["recency", "relevance", "hybrid"])
def test_grep_groups_before_limit(engine, mode, sort):
    other = append(engine, "quartz other")
    first = stamped_copy(engine)
    second = stamped_copy(engine)
    payload = json.loads(tools.lcm_grep({"query": "quartz", "limit": 2, "mode": mode, "sort": sort}, engine=engine))
    assert {hit["store_id"] for hit in payload["results"]} == {other, second}
    assert next(hit for hit in payload["results"] if hit["store_id"] == second)["duplicate_store_ids"] == [first]


@pytest.mark.parametrize("strict", [False, True])
def test_caps_count_groups_once(engine, strict):
    # 30 rows, 10 duplicate pairs, 20 distinct messages in four sessions.
    for session in range(4):
        for index in range(5):
            text = f"quartz fact {session}-{index}"
            stamped_copy(engine, text, session=f"s{session}")
            if session < 2:
                stamped_copy(engine, text, session=f"s{session}")
    engine._config.recall_reference_strict = strict
    snippets = recall(engine, limit=25)
    assert len(snippets["hits"]) == 20
    answer = recall(engine, detail="answer_ready", limit=25)
    # Since #1679, strict answer-ready delivers every ranked group; hydration stays capped at 8.
    assert len(answer["hits"]) == 20
    assert len({engine._store.get(hit["store_id"])["content"] for hit in answer["hits"]}) == len(answer["hits"])
    assert sum("content" in hit for hit in answer["hits"]) == 8
    assert max(sum(hit["session_id"] == f"s{s}" for hit in answer["hits"]) for s in range(4)) == 5


def test_fusion_max_score_union_arms_best_rank_and_summary_exemption(engine):
    first, second = stamped_copy(engine), stamped_copy(engine)
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
    assert entry["hit"]["store_id"] == second
    assert entry["hit"]["duplicate_store_ids"] == [first]
    assert entry["rrf_score"] == max(e["rrf_score"] for e in original if e["hit"]["kind"] != "summary")
    assert set(entry["ranks"]) == {0, 2}
    assert len([e for e in grouped if e["hit"]["kind"] == "summary"]) == 2
    assert original == before


def test_group_keeps_citable_duplicate_chunk_after_expansion_cap(engine):
    first, second = stamped_copy(engine), stamped_copy(engine)
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
    path = Path(__file__).parent / "fixtures/issue_1681_duplicate_free.json"
    actual = duplicate_free_responses(engine, monkeypatch)
    if os.environ.get("ISSUE_1681_CAPTURE_BASELINE") == "1":
        path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n")
        return
    assert actual == json.loads(path.read_text())


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
        result = plugin._answer_ready_baseline(SimpleNamespace(handle_tool_call=lambda *_, **kw: {"hits": [hit]}), "q", {})[0]
    assert result["duplicate_store_ids"] == [second]
