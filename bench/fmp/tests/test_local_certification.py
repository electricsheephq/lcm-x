"""LOCAL build/run acceptance with production certification and offline vectors."""
import json
import socket
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.fmp import delivery, derived_panel, mcp_reader, models, run, stores as store_module
from bench.fmp.tests import test_fmp

REPO = test_fmp.REPO
lme = test_fmp.lme
question = test_fmp.question


class LocalProvider:
    provider_id = "fastembed"
    model_id = "BAAI/bge-small-en-v1.5"
    dim = 2

    def _ensure_local(self):
        pass

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def local_store(tmp_path, lme, monkeypatch):
    from hermes_lcm import command, embedding_maintenance, tools
    provider = LocalProvider()
    version = store_module.importlib.metadata.version
    monkeypatch.setattr(store_module.importlib.metadata, "version",
                        lambda name: "0.8.0" if name == "fastembed" else version(name))
    monkeypatch.setattr(lme, "resolve_harness_provider", lambda *a, **kw: provider)
    monkeypatch.setattr(command, "resolve_provider", lambda *a, **kw: provider)
    monkeypatch.setattr(tools, "probe_provider_availability", lambda config: {"available": True})
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda config: ("present", "synthetic"))
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden")))
    return store_module.Stores(tmp_path / "stores", "local", REPO, "synthetic-sha", lme)


@pytest.mark.parametrize("summaries", ["none", "deterministic"])
def test_local_build_replay_runs_chunk(local_store, question, summaries):
    local_store.summaries = summaries
    assert local_store.build(question) == "built"
    with local_store.engine(question) as (engine, _):
        body = json.loads(delivery.dispatch("lcm_recall", {"query": question.question}, engine))
    assert "chunk" in body["provenance"]["arms_run"]
    assert "coverage not certified" not in body.get("degraded_reason", "")
    if summaries == "none":
        assert body["degraded"] and body["degraded_reason"] == "summary vectors are unavailable"
    else:
        assert body["provenance"]["coverage"]["summary"] == "full"


@pytest.mark.parametrize("base", ["local", "off"])
def test_format_three_store_is_rejected(local_store, question, monkeypatch, base):
    store = local_store
    if base == "off":
        from hermes_lcm import command
        from benchmarking import longmemeval
        forbidden = Mock(side_effect=AssertionError("off must not register, certify or guard"))
        monkeypatch.setattr(command, "_embedding_register_profiles", forbidden)
        monkeypatch.setattr(longmemeval, "certify_local_recall_store", forbidden)
        monkeypatch.setattr(delivery, "dispatch", forbidden)
        store = store_module.Stores(local_store.root, "off", REPO, "synthetic-sha", local_store.lme)
    store.build(question)
    manifest = store.directory(question) / "build-manifest.json"
    identity = json.loads(manifest.read_text())
    identity["format"] = 3
    manifest.write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="store manifest mismatch"):
        with store.engine(question):
            pass
    assert store.identity["format"] == 4
    if base == "off":
        forbidden.assert_not_called()


def test_uncertified_build_is_never_published(local_store, question, monkeypatch):
    from hermes_lcm import command
    from benchmarking import longmemeval
    monkeypatch.setattr(command, "_embedding_register_profiles", lambda *a, **kw: {})
    monkeypatch.setattr(longmemeval, "certify_local_recall_store", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="local_recall_semantic_unavailable"):
        local_store.build(question)
    assert not local_store.directory(question).exists()


@pytest.mark.parametrize("payload", [
    {"provenance": {"arms_run": ["fts"]}},
    {"provenance": {"arms_run": ["fts", "chunk"]},
     "degraded_reason": "summary semantic index catching up (coverage not certified; catch-up receipt missing)"},
])
def test_build_guard_checks_both_conditions(local_store, question, monkeypatch, payload):
    monkeypatch.setattr(delivery, "dispatch", lambda *a, **kw: json.dumps(payload))
    with pytest.raises(RuntimeError, match="local_recall_semantic_unavailable"):
        local_store.build(question)
    assert not local_store.directory(question).exists()


@pytest.mark.parametrize("failure", ["register", "certify"])
def test_failed_certification_is_never_published(local_store, question, monkeypatch, failure):
    from hermes_lcm import command
    from benchmarking import longmemeval
    if failure == "register":
        monkeypatch.setattr(command, "_embedding_register_profiles", lambda *a, **kw: "status: error")
    else:
        monkeypatch.setattr(longmemeval, "certify_local_recall_store",
                            Mock(side_effect=RuntimeError("synthetic certification failure")))
    error = delivery.LocalRecallUnavailable if failure == "register" else RuntimeError
    reason = "store_embedding_registration_failed" if failure == "register" else "synthetic certification failure"
    with pytest.raises(error, match=reason):
        local_store.build(question)
    assert not local_store.directory(question).exists()


def test_uncertified_run_stops_before_reader(local_store, question, monkeypatch, tmp_path):
    local_store.build(question)
    with sqlite3.connect(local_store.directory(question) / "lcm.db") as con:
        con.execute("DELETE FROM metadata WHERE key LIKE 'embedding_catchup:%'")
    monkeypatch.setattr(local_store, "build", lambda q: "reused")
    monkeypatch.setattr(run, "Stores", lambda *a, **kw: local_store)
    monkeypatch.setattr(run, "load_lcm", lambda *a: local_store.lme)
    monkeypatch.setattr(run.datasets, "load", lambda *a: ([question], "synthetic-sha", "ids", {}))
    reader_started = Mock()
    monkeypatch.setattr(run, "Backends", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(run, "versions", lambda: {})
    def answer(q, mode, client, reader, engine, *a, **kw):
        delivery.dispatch("lcm_recall", {"query": q.question}, engine)
        reader_started()
        return {"status": "ok", "hypothesis": "synthetic"}
    monkeypatch.setattr(delivery, "answer", answer)
    args = run.parser().parse_args(["answer", "--split-label", "synthetic", "--config", "local",
        "--data", str(tmp_path / "unused.json"), "--store-root", str(local_store.root),
        "--output", str(tmp_path / "output"), "--allow-dirty", "--delivery", "single", "--workers", "1"])
    with pytest.raises(RuntimeError, match="local_recall_semantic_unavailable"):
        run._execute(args)
    reader_started.assert_not_called()


def test_uncertified_mcp_recall_fails(local_store, question, tmp_path):
    local_store.build(question)
    with sqlite3.connect(local_store.directory(question) / "lcm.db") as con:
        con.execute("DELETE FROM metadata WHERE key LIKE 'embedding_catchup:%'")
    with local_store.engine(question) as (engine, _):
        reader = mcp_reader.Reader({"max_calls": 1, "question": {"question_date": question.question_date}},
                                   engine, delivery.schemas(engine), tmp_path / "mcp.jsonl")
        with pytest.raises(RuntimeError, match="local_recall_semantic_unavailable"):
            reader.call("lcm_recall", {"query": question.question})


@pytest.mark.parametrize("backend_raises", [False, True])
def test_mcp_semantic_failure_stops_harness(question, tmp_path, monkeypatch, backend_raises):
    log = tmp_path / "mcp.jsonl"
    log.write_text('{"at":"2026-11-17T00:00:00Z","fatal":"LocalRecallUnavailable"}\n')
    monkeypatch.setattr(models.time, "sleep", lambda seconds: None)
    launches = Mock()
    class Client:
        def agentic(self, reader, messages, records, server, **kw):
            launches()
            records.append({"mcp_log": str(log), "error": "codex_failed" if backend_raises else None})
            if backend_raises:
                raise RuntimeError("synthetic MCP disconnection")
            return {"content": "synthetic answer"}
    with pytest.raises(delivery.LocalRecallUnavailable, match="local_recall_semantic_unavailable"):
        delivery.harness_agentic(question, Client(), "fixture", {}, [])
    assert launches.call_count == 1


@pytest.mark.parametrize("summaries", ["none", "deterministic"])
def test_certified_recall_passes_product_replies(local_store, question, monkeypatch, summaries):
    from hermes_lcm import tools
    local_store.summaries = summaries
    assert local_store.build(question) == "built"
    product = tools.lcm_recall
    replies = []
    def capture(args, *, engine):
        raw = product(args, engine=engine)
        replies.append(raw)
        return raw
    monkeypatch.setattr(tools, "lcm_recall", capture)
    calls = [{"include": "summaries"}, {}, {"include": "verbatim"},
             {"exclude_session_ids": list(question.session_map)}, {"query": ""},
             {"limit": 0}, {"include": "bogus"}]
    for extra in calls:
        args = {"query": question.question, **extra}
        with local_store.engine(question, frozen=True) as (engine, _):
            raw = delivery.dispatch("lcm_recall", args, engine)
        assert raw == replies[-1]
        if extra.get("query") == "" or extra.get("limit") == 0 or extra.get("include") == "bogus":
            assert json.loads(raw)["error"] == json.loads(replies[-1])["error"]


@pytest.mark.parametrize("include", [None, "summaries", "verbatim", "empty-scope"])
def test_uncertified_recall_checked_after_passing_call(local_store, question, include):
    assert local_store.build(question) == "built"
    args = {"query": question.question}
    if include == "empty-scope":
        args["exclude_session_ids"] = list(question.session_map)
    elif include is not None:
        args["include"] = include
    with local_store.engine(question, frozen=True) as (engine, _):
        delivery.dispatch("lcm_recall", {"query": question.question}, engine)
        with sqlite3.connect(local_store.directory(question) / "lcm.db") as con:
            con.execute("DELETE FROM metadata WHERE key LIKE 'embedding_catchup:%'")
        with pytest.raises(delivery.LocalRecallUnavailable, match="local_recall_semantic_unavailable"):
            delivery.dispatch("lcm_recall", args, engine)
    with local_store.engine(question, frozen=True) as (engine, _):
        with pytest.raises(delivery.LocalRecallUnavailable, match="local_recall_semantic_unavailable"):
            delivery.dispatch("lcm_recall", args, engine)


def test_certified_mcp_summaries_returns_result(local_store, question, tmp_path):
    assert local_store.build(question) == "built"
    with local_store.engine(question, frozen=True) as (engine, _):
        reader = mcp_reader.Reader({"max_calls": 2, "question": {"question_date": question.question_date}},
                                   engine, delivery.schemas(engine), tmp_path / "mcp.jsonl")
        reply = reader.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "lcm_recall", "arguments": {"query": question.question, "include": "summaries"}}})
        assert reply["id"] == 1 and not reply["result"]["isError"]
        assert "error" not in json.loads(reply["result"]["content"][0]["text"])
        assert reader.handle({"id": 2, "method": "ping"})["result"] == {}


def test_voyage_build_does_not_certify(local_store, question, monkeypatch):
    from hermes_lcm import command
    from benchmarking import longmemeval
    provider = LocalProvider()
    provider.provider_id, provider.model_id = "voyage", "voyage-context-3"
    monkeypatch.setattr(local_store.lme, "resolve_harness_provider", lambda *a, **kw: provider)
    register, certify = Mock(return_value={}), Mock()
    monkeypatch.setattr(command, "_embedding_register_profiles", register)
    monkeypatch.setattr(longmemeval, "certify_local_recall_store", certify)
    voyage = store_module.Stores(local_store.root, "voyage", REPO, "synthetic-sha", local_store.lme)
    assert voyage.build(question) == "built"
    register.assert_not_called()
    certify.assert_not_called()
    with voyage.engine(question, frozen=True) as (engine, _):
        assert not getattr(engine, "_fmp_local_recall_guard", False)


@pytest.mark.parametrize("backend_raises", [False, True])
def test_real_canary_surfaces_mcp_fatal(question, tmp_path, monkeypatch, backend_raises):
    args = SimpleNamespace(output=tmp_path, data=tmp_path / "unused.json", reader="fixture")
    monkeypatch.setattr(run, "server_spec", lambda *a: {})
    log = tmp_path / "canary-mcp.jsonl"
    class Client:
        def call(self, reader, messages, records):
            return {"content": "unknown"}

        def agentic(self, reader, messages, records, server):
            log.write_text(json.dumps({"at": "2026-11-17T00:00:00Z", "fatal": "LocalRecallUnavailable"}) + "\n")
            records.append({"mcp_log": str(log)})
            if backend_raises:
                raise RuntimeError("synthetic MCP disconnection")
            return {"content": "unknown"}
    with pytest.raises(delivery.LocalRecallUnavailable, match="local_recall_semantic_unavailable"):
        run.canary(args, Client(), "synthetic-sha", question)


def test_local25_rejects_format_three(tmp_path):
    panel = derived_panel
    directory = tmp_path / "stores" / "local-none-old" / "synthetic"
    directory.mkdir(parents=True)
    (directory / "build-manifest.json").write_text(json.dumps({"config": "local", "format": 3}))
    (directory / "lcm.db").write_bytes(b"synthetic")
    (directory / "speakers.json").write_text("{}")
    sources = {key: panel.digest_file(directory / name) for name, key in (
        ("build-manifest.json", "manifest_sha256"), ("lcm.db", "database_sha256"),
        ("speakers.json", "speakers_sha256"))}
    product, value = "synthetic-product", {"evidence": "synthetic"}
    identity = {"store": sources, "product": product, "config": {"recall_depth": 25}}
    key = panel.sha(panel.encoded(identity))
    retrieval = tmp_path / "retrieval"
    retrieval.mkdir()
    (retrieval / (key + ".json")).write_text(json.dumps({"identity": identity, "value": value,
        "value_sha256": panel.sha(panel.encoded(value))}))
    receipt = {"status": "COMPLETE", "product": product, "rows": [{"id": "q1", "sources": sources,
        "cache_key": key, "observed": {"limit": 25}, "delivered_sha256": panel.sha(b"synthetic")}]}
    with pytest.raises(panel.Incomplete, match=r"W1 store is not certified \(format\)"):
        panel.local25(receipt, product, tmp_path, population=None)


def test_format_three_store_rebuilt_in_place(local_store, question):
    assert local_store.build(question) == "built"
    directory = local_store.directory(question)
    manifest = directory / "build-manifest.json"
    old = json.loads(manifest.read_text())
    old["format"] = 3
    manifest.write_text(json.dumps(old))
    assert local_store.build(question) == "built"
    assert json.loads(manifest.read_text())["format"] == 4
    superseded = list(directory.parent.glob(directory.name + ".superseded-*"))
    assert len(superseded) == 1
    assert json.loads((superseded[0] / "build-manifest.json").read_text())["format"] == 3
