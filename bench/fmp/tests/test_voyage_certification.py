"""Voyage store acceptance through production recall with an offline provider."""
import json
import socket
from unittest.mock import Mock, patch

import pytest

from bench.fmp import delivery, mcp_reader
from bench.fmp.common import write_json
from bench.fmp.stores import Stores
from bench.fmp.tests import test_fmp

REPO = test_fmp.REPO
lme = test_fmp.lme
question = test_fmp.question


class VoyageProvider:
    provider_id = "voyage"
    model_id = "voyage-context-3"
    dim = 2

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def rerank(self, query, documents, *, top_k, timeout, model):
        return [(index, 1.0) for index in range(top_k)]


@pytest.fixture
def voyage_store(tmp_path, lme, monkeypatch):
    from hermes_lcm import tools

    provider = VoyageProvider()
    monkeypatch.setattr(provider, "rerank", Mock(wraps=provider.rerank))
    resolver = Mock(return_value=provider)
    no_new_provider = Mock(side_effect=AssertionError("unexpected provider resolution"))
    monkeypatch.setattr(lme, "resolve_harness_provider", resolver)
    monkeypatch.setattr(tools, "resolve_provider", no_new_provider)
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))

    def create(config="voyage", summaries="deterministic"):
        return Stores(tmp_path / "stores", config, REPO, "synthetic-sha", lme, summaries)

    yield create
    no_new_provider.assert_not_called()


def revisions(vectors):
    return dict(vectors._conn.execute("SELECT task, revision FROM lcm_embedding_profile WHERE active=1"))


@pytest.mark.parametrize("rerank", ["", "+rerank:rerank-synthetic"])
def test_voyage_profiles_and_frozen_semantic_recall(voyage_store, question, rerank):
    from hermes_lcm import ingest_policy, tools

    store = voyage_store("voyage" + rerank)
    with patch.object(tools, "lcm_recall", wraps=tools.lcm_recall) as recall:
        assert store.build(question) == "built"
    with store.engine(question) as (engine, vectors):
        raw = delivery.dispatch("lcm_recall", {"query": question.question}, engine)
        reply = json.loads(raw)
        assert "error" not in reply
        assert reply["provenance"]["coverage"] == {
            "fts": "ok", "summary": "full", "chunk": "full"
        }, reply.get("degraded_reason")
        assert {"summary", "chunk"} <= set(reply["provenance"]["arms_run"])
        revision = ingest_policy.embedding_privacy_revision(engine._config)
        assert store.expected(question)["embedding_revision"] == revision
        assert revisions(vectors) == {"chunk": revision, "summary": revision}
        assert engine._fmp_voyage_recall_guard
    assert recall.call_count == 1
    probe = recall.call_args.kwargs["engine"]
    assert probe._lcm_embedding_provider_cache[1] is store.provider
    assert not probe._fmp_voyage_recall_guard
    if rerank:
        assert reply["provenance"]["rerank"].startswith("applied")
        assert {call.kwargs["model"] for call in store.provider.rerank.call_args_list} == {"rerank-synthetic"}


def test_empty_revision_refused_at_build_and_frozen_recall(voyage_store, question, monkeypatch, tmp_path):
    from hermes_lcm import tools

    store = voyage_store()
    monkeypatch.setattr(store, "_embedding_revision", lambda config: "")
    with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$") as build:
        store.build(question)
    path = store.directory(question)
    assert not path.exists()
    stage, = path.parent.glob(path.name + ".building-*")
    with store.engine(question, stage) as (engine, vectors):
        assert revisions(vectors) == {"chunk": "", "summary": ""}
        raw = tools.lcm_recall({"query": question.question}, engine=engine)
        reply = json.loads(raw)
        assert reply["provenance"]["coverage"] == {"fts": "ok", "summary": "none", "chunk": "none"}
        assert tools._EMBEDDING_IDENTITY_STALE_REASON in reply["degraded_reason"]
        assert delivery.recall_metadata(raw)["reason_class"] == "embedding_identity_stale"
        assert build.value.recall_metadata["reason_class"] == "embedding_identity_stale"
        with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$"):
            delivery.single(question, engine)

        log = tmp_path / "mcp.jsonl"
        spec = {"max_calls": 1, "question": {"question_date": question.question_date}}
        reader = mcp_reader.Reader(spec, engine, delivery.schemas(engine), log)

        def call_reader(model, messages, records, server, **kwargs):
            records.append({"mcp_log": str(log)})
            reader.call("lcm_recall", {"query": question.question})

        client = Mock()
        client.agentic.side_effect = call_reader
        with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$"):
            delivery.harness_agentic(question, client, "fixture", {}, [])
        client.agentic.assert_called_once()
        record = json.loads(log.read_text())
        assert record["fatal"] == "VoyageRecallUnavailable"
        assert record["reason_class"] == "embedding_identity_stale"
        assert record["output_bytes"] == 0
        assert "degraded_reason" not in record and "query" not in record and "hits" not in record


@pytest.mark.parametrize("invalid", [
    {"query": ""},
    {"limit": 0},
    {"include": "bogus"},
    {"seen_refs": ["synthetic-ref"]},
])
def test_voyage_validation_reply_reaches_reader(voyage_store, question, tmp_path, invalid):
    from hermes_lcm import tools

    store = voyage_store()
    assert store.build(question) == "built"
    log = tmp_path / "mcp.jsonl"
    valid = {"query": question.question}
    arguments = {**valid, **invalid}
    with store.engine(question) as (engine, _):
        expected = tools.lcm_recall(arguments, engine=engine)
        assert "error" in json.loads(expected)
        spec = {"max_calls": 2, "question": {"question_date": question.question_date}}
        reader = mcp_reader.Reader(spec, engine, delivery.schemas(engine), log)

        def call_reader(model, messages, records, server, **kwargs):
            records.append({"mcp_log": str(log)})
            reply = reader.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "lcm_recall", "arguments": arguments}})["result"]
            assert reply["isError"] and reply["content"][0]["text"] == expected
            raw, error = reader.call("lcm_recall", valid)
            assert not error
            assert json.loads(raw)["provenance"]["coverage"]["chunk"] == "full"
            return {"content": "synthetic answer"}

        client = Mock()
        client.agentic.side_effect = call_reader
        result = delivery.harness_agentic(question, client, "fixture", {}, [])
        assert result["status"] == "ok" and result["hypothesis"] == "synthetic answer"
        assert result["tool_calls"] == 2
        assert result["trace"][0]["reason_class"] == "error"
        assert all("fatal" not in event for event in result["trace"])
        assert delivery.dispatch("lcm_recall", arguments, engine) == expected


def test_voyage_raised_product_exception_stops(voyage_store, question, monkeypatch):
    from hermes_lcm import tools
    from hermes_lcm.ingest_policy import EmbeddingPrivacyPolicyError

    store = voyage_store()
    assert store.build(question) == "built"
    monkeypatch.setattr(tools, "_lcm_grep_embed_query",
                        Mock(side_effect=EmbeddingPrivacyPolicyError("synthetic privacy failure")))
    with store.engine(question) as (engine, _):
        with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$") as run:
            delivery.single(question, engine)
    assert run.value.recall_metadata["reason_class"] == "error"


def test_voyage_build_probe_requires_success(voyage_store, question, monkeypatch):
    from hermes_lcm import tools

    store = voyage_store()
    monkeypatch.setattr(tools, "lcm_recall", Mock(return_value='{"error": "synthetic"}'))
    with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$") as build:
        store.build(question)
    assert build.value.recall_metadata["reason_class"] == "error"
    assert not store.directory(question).exists()


@pytest.mark.parametrize("coverage, reason, expected", [
    ({"summary": "full", "chunk": "none"}, "chunk vectors are unavailable", "other"),
    ({"summary": "none", "chunk": "full"}, "summary vectors are unavailable", "summary_vectors_unavailable"),
    ({}, "embedding provider unavailable: synthetic", "provider_unavailable"),
    ({"summary": "disabled", "chunk": "disabled"}, "semantic retrieval is disabled", "other"),
    ({"summary": "full", "chunk": "full"}, "prefix_stale", "prefix_stale"),
])
def test_voyage_certification_and_guard_refuse_incomplete_replies(voyage_store, question, monkeypatch,
                                                                 coverage, reason, expected):
    from hermes_lcm import tools

    store = voyage_store()
    payload = {
        "provenance": {"coverage": coverage},
        "degraded_reason": tools._EMBEDDING_PREFIX_STALE_REASON if reason == "prefix_stale" else reason,
    }
    monkeypatch.setattr(tools, "lcm_recall", Mock(return_value=json.dumps(payload)))
    with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$") as build:
        store.build(question)
    path = store.directory(question)
    assert not path.exists()
    stage, = path.parent.glob(path.name + ".building-*")
    with store.engine(question, stage) as (engine, _):
        with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$") as run:
            delivery.single(question, engine)
    assert build.value.recall_metadata["reason_class"] == run.value.recall_metadata["reason_class"] == expected


def test_voyage_without_summaries_and_empty_scope(voyage_store, question):
    store = voyage_store(summaries="none")
    assert store.build(question) == "built"
    with store.engine(question) as (engine, _):
        def recall(**kwargs):
            return json.loads(delivery.dispatch("lcm_recall", {"query": question.question, **kwargs}, engine))

        assert recall()["provenance"]["coverage"] == {"fts": "ok", "chunk": "full", "summary": "none"}
        assert recall(include="verbatim")["provenance"]["coverage"] == {"fts": "ok", "chunk": "full"}
        summary_reply = recall(include="summaries")
        assert summary_reply["provenance"]["coverage"] == {"fts": "ok", "summary": "none"}
        assert summary_reply["degraded_reason"] == "summary vectors are unavailable"
        reply = recall(exclude_session_ids=[session["session_id"] for session in question.sessions])
        assert reply["hits"] == [] and reply["provenance"]["coverage"]["chunk"] == "full"


def test_voyage_summaries_only_requires_built_summary_coverage(voyage_store, question, monkeypatch):
    from hermes_lcm import tools

    store = voyage_store()
    assert store.build(question) == "built"
    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", Mock(return_value=([], "none", 0, 0, [])))
    with store.engine(question) as (engine, _):
        args = {"query": question.question, "include": "summaries"}
        with pytest.raises(delivery.VoyageRecallUnavailable, match="^voyage_recall_semantic_unavailable$") as run:
            delivery.dispatch("lcm_recall", args, engine)
    assert run.value.recall_metadata["reason_class"] == "summary_vectors_unavailable"


def test_old_voyage_manifest_rebuilds(voyage_store, question):
    store = voyage_store()
    expected = store.expected(question)
    legacy_manifest = {key: value for key, value in expected.items() if key != "embedding_revision"}
    path = store.directory(question)
    assert store.build(question) == "built"
    write_json(path / "build-manifest.json", legacy_manifest)
    with pytest.raises(RuntimeError, match="store manifest mismatch"):
        with store.engine(question):
            pass
    assert store.build(question) == "built"
    assert json.loads((path / "build-manifest.json").read_text()) == expected
    prior, = path.parent.glob(path.name + ".superseded-*")
    assert json.loads((prior / "build-manifest.json").read_text()) == legacy_manifest
    assert store.build(question) == "reused"
