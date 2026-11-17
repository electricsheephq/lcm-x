"""LOCAL build/run acceptance with production certification and offline vectors."""
import json
import socket
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.fmp import delivery, mcp_reader, run, stores as store_module
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
    with pytest.raises(RuntimeError):
        local_store.build(question)
    assert not local_store.directory(question).exists()


@pytest.mark.parametrize("command", ["canary", "answer"])
def test_uncertified_run_stops_before_reader(local_store, question, monkeypatch, tmp_path, command):
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
    def canary(*a):
        with local_store.engine(question) as (engine, _):
            answer(question, None, None, None, engine)
    monkeypatch.setattr(run, "canary", canary)
    monkeypatch.setattr(delivery, "answer", answer)
    args = run.parser().parse_args([command, "--split-label", "synthetic", "--config", "local",
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
def test_mcp_semantic_failure_stops_harness(question, tmp_path, backend_raises):
    log = tmp_path / "mcp.jsonl"
    log.write_text('{"fatal":"LocalRecallUnavailable","counted":true}\n')
    class Client:
        def agentic(self, reader, messages, records, server, **kw):
            records.append({"mcp_log": str(log)})
            if backend_raises:
                raise RuntimeError("synthetic MCP disconnection")
            return {"content": "synthetic answer"}
    with pytest.raises(RuntimeError, match="local_recall_semantic_unavailable"):
        delivery.harness_agentic(question, Client(), "fixture", {}, [])
