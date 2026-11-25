"""Round 3: runtime fallback stops and content-free recall accounting, offline."""
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bench.fmp import delivery, dev, mcp_reader, run
from bench.fmp.tests import test_local_certification
from bench.fmp.tests.test_fmp import Scripted, tool

lme = test_local_certification.lme
local_store = test_local_certification.local_store
question = test_local_certification.question


def inject(monkeypatch, store, q, fault):
    from hermes_lcm import embedding_maintenance, tools
    if fault in {"not_certified", "catching_up"}:
        with sqlite3.connect(store.directory(q) / "lcm.db") as con:
            if fault == "not_certified":
                con.execute("DELETE FROM metadata WHERE key LIKE 'embedding_catchup:%'")
            else:
                for key, value in con.execute("SELECT key,value FROM metadata WHERE key LIKE 'embedding_catchup:%'"):
                    con.execute("UPDATE metadata SET value=? WHERE key=?",
                                (json.dumps({**json.loads(value), "remaining": 5}), key))
    elif fault == "prefix_stale":
        # Reword the product constant: the guard must follow the runtime value.
        monkeypatch.setattr(tools, "_EMBEDDING_PREFIX_STALE_REASON", "synthetic changed prefix remedy")
        monkeypatch.setattr(tools, "_lcm_recall_catchup_reason",
                            lambda *a: tools._EMBEDDING_PREFIX_STALE_REASON)
    elif fault in {"unavailable", "preparing"}:
        monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda c: (fault, "synthetic"))
    elif fault == "probe_unavailable":
        monkeypatch.setattr(tools, "probe_provider_availability", lambda c: {"available": False})
    else:
        target = {"provider_unavailable": "_resolve_recall_provider",
                  "chunk_provider_unavailable": "_resolve_recall_chunk_provider",
                  "worker_capacity": "_resolve_recall_provider",
                  "chunk_arm": "_lcm_recall_chunk_arm", "summary_arm": "_lcm_recall_summary_arm",
                  "timeout": "_lcm_recall_chunk_arm"}[fault]
        error = (TimeoutError("synthetic private timeout") if fault == "timeout" else
                 tools._WorkerCapacityError("lcm-provider-resolution worker capacity is exhausted")
                 if fault == "worker_capacity" else RuntimeError("synthetic private failure"))
        monkeypatch.setattr(tools, target, Mock(side_effect=error))


def capture(monkeypatch):
    from hermes_lcm import tools
    product, replies = tools.lcm_recall, []
    def recall(*a, **kw):
        raw = product(*a, **kw)
        replies.append(raw)
        return raw
    monkeypatch.setattr(tools, "lcm_recall", recall)
    return replies


def reader(engine, q, tmp_path, calls=2):
    return mcp_reader.Reader({"max_calls": calls, "question": {"question_date": q.question_date}},
                             engine, delivery.schemas(engine), tmp_path / "mcp.jsonl")


def assert_metadata(record, payload, reason_class):
    assert record.get("reason_class") == reason_class
    assert record.get("arms_run") == payload.get("provenance", {}).get("arms_run", [])
    assert isinstance(record["arms_run"], list)


@pytest.mark.parametrize("path", ["mcp", "mcp_session", "session"])
@pytest.mark.parametrize("fault,reason_class", [
    ("not_certified", "not_certified"), ("catching_up", "catching_up"),
    ("prefix_stale", "prefix_stale"), ("provider_unavailable", "provider_unavailable"),
    ("chunk_provider_unavailable", "provider_unavailable"),
    ("unavailable", "unavailable"), ("probe_unavailable", "unavailable"),
])
def test_stop_classes(local_store, question, tmp_path, monkeypatch, path, fault, reason_class):
    local_store.summaries = "deterministic"
    local_store.build(question)
    inject(monkeypatch, local_store, question, fault)
    replies = capture(monkeypatch)
    with local_store.engine(question) as (engine, _):
        if path != "mcp":
            delivery.session_mode(engine)
        mcp = reader(engine, question, tmp_path)
        with pytest.raises(delivery.LocalRecallUnavailable, match="^local_recall_semantic_unavailable$") as caught:
            if path.startswith("mcp"):
                mcp.call("lcm_recall", {"query": question.question})
            else:
                client = Scripted([{"role": "assistant", "content": "synthetic"}])
                delivery.answer(question, "single", client, "fixture", engine)
    payload = json.loads(replies[-1])
    assert hasattr(caught.value, "recall_metadata")
    assert_metadata(caught.value.recall_metadata, payload, reason_class)
    if path.startswith("mcp"):
        records = [json.loads(line) for line in mcp.log.read_text().splitlines()]
        assert len(records) == 1
        assert_metadata(records[0], payload, reason_class)
        assert records[0]["fatal"] == "LocalRecallUnavailable"
        assert "degraded_reason" not in records[0] and "hits" not in records[0] and "query" not in records[0]
        assert "synthetic private" not in mcp.log.read_text()


@pytest.mark.parametrize("path", ["mcp", "session_single", "session_agentic"])
@pytest.mark.parametrize("fault,reason_class", [
    ("preparing", "preparing"), ("worker_capacity", "worker_capacity"),
    ("chunk_arm", "arm_failed"), ("summary_arm", "arm_failed"), ("timeout", "timeout"),
])
def test_transient_recall_counted(local_store, question, tmp_path, monkeypatch, path, fault, reason_class):
    local_store.summaries = "deterministic"
    local_store.build(question)
    inject(monkeypatch, local_store, question, fault)
    replies = capture(monkeypatch)
    with local_store.engine(question) as (engine, _):
        if path == "mcp":
            mcp = reader(engine, question, tmp_path)
            raw, error = mcp.call("lcm_recall", {"query": question.question})
            assert raw == replies[-1] and not error and mcp.counted == 1
            record = json.loads(mcp.log.read_text())
            assert "synthetic private" not in mcp.log.read_text()
            assert not {"query", "hits", "degraded_reason"} & record.keys()
        else:
            delivery.session_mode(engine)
            mode = path.removeprefix("session_")
            client = Scripted(([tool("lcm_recall", {"query": question.question})] if mode == "agentic" else [])
                              + [{"role": "assistant", "content": "synthetic"}])
            result = delivery.answer(question, mode, client, "fixture", engine)
            trace = result["trace"][0]
            expected = {"tool": "lcm_recall", "result": replies[-1]}
            if mode == "agentic":
                expected["arguments"] = json.dumps({"query": question.question})
            assert trace == expected
            assert result["tool_calls"] == (1 if mode == "agentic" else 0)
            mcp = reader(engine, question, tmp_path)
            raw, error = mcp.call("lcm_recall", {"query": question.question})
            assert raw == replies[-1] and not error and mcp.counted == 1
            record = json.loads(mcp.log.read_text())
    assert_metadata(record, json.loads(replies[-1]), reason_class)


def test_product_prefix_stale_constant_contract(lme, monkeypatch):
    from hermes_lcm import tools
    constant = getattr(tools, "_EMBEDDING_PREFIX_STALE_REASON", None)
    assert isinstance(constant, str) and constant
    assert_metadata(delivery.recall_metadata(json.dumps({"degraded_reason": constant})), {}, "prefix_stale")
    # Synthetic tools modules can omit it; absence must skip only this class.
    monkeypatch.delattr(tools, "_EMBEDDING_PREFIX_STALE_REASON")
    assert_metadata(delivery.recall_metadata("{}"), {}, "none")
    assert_metadata(delivery.recall_metadata('{"degraded_reason":"synthetic unknown"}'), {}, "other")


@pytest.mark.parametrize("summaries", ["none", "deterministic"])
@pytest.mark.parametrize("unit", ["hits", "session"])
@pytest.mark.parametrize("shape", ["default", "summaries", "verbatim", "empty_scope", "empty_query", "zero",
                                   "bad_include", "chunk_arm", "timeout", "summary_arm", "answer_ready"])
def test_round_two_shapes_accounted(local_store, question, tmp_path, monkeypatch, summaries, unit, shape):
    local_store.summaries = summaries
    local_store.build(question)
    if shape in {"chunk_arm", "summary_arm", "timeout"}:
        inject(monkeypatch, local_store, question, shape)
    replies = capture(monkeypatch)
    extra = {"summaries": {"include": "summaries"}, "verbatim": {"include": "verbatim"},
             "empty_scope": {"exclude_session_ids": list(question.session_map)}, "empty_query": {"query": ""},
             "zero": {"limit": 0}, "bad_include": {"include": "bogus"},
             "answer_ready": {"detail": "answer_ready"}}.get(shape, {})
    with local_store.engine(question) as (engine, _):
        if unit == "session":
            delivery.session_mode(engine)
        mcp = reader(engine, question, tmp_path)
        raw, error = mcp.call("lcm_recall", {"query": question.question, **extra})
    assert raw == replies[-1]
    payload = json.loads(raw)
    expected = ("error" if shape in {"empty_query", "zero", "bad_include"} else
                "arm_failed" if shape in {"chunk_arm", "summary_arm"} else "timeout" if shape == "timeout" else
                "summary_vectors_unavailable" if summaries == "none" and shape not in {"verbatim", "empty_scope"}
                else "none")
    assert error == (expected == "error")
    assert_metadata(json.loads(mcp.log.read_text()), payload, expected)


@pytest.mark.parametrize("payload,reason_class", [
    ({"degraded": True, "degraded_reason": "synthetic private unclassified degradation"}, "other"),
    ({"degraded_reason": "chunk embedding provider unavailable: synthetic private failure; "
      "summary arm failed: worker capacity is exhausted"}, "provider_unavailable"),
    ({"error": "synthetic private error", "degraded_reason": "coverage not certified"}, "error"),
    ({"degraded_reason": "summary semantic index catching up (coverage not certified; catch-up receipt missing)"},
     "not_certified"),
])
def test_unguarded_metadata_and_precedence(lme, question, tmp_path, monkeypatch, payload, reason_class):
    from hermes_lcm import tools
    payload = {"provenance": {"arms_run": ["fts", "chunk"]}, "hits": [{"snippet": "private hit"}], **payload}
    raw = json.dumps(payload)
    monkeypatch.setattr(tools, "lcm_recall", lambda *a, **kw: raw)
    engine = SimpleNamespace(_config=SimpleNamespace(), _fmp_local_recall_guard=False)
    mcp = mcp_reader.Reader({"max_calls": 1, "question": {"question_date": question.question_date}},
                            engine, [{"function": {"name": "lcm_recall"}}], tmp_path / "mcp.jsonl")
    assert mcp.call("lcm_recall", {"query": "private query"})[0] == raw
    record = json.loads(mcp.log.read_text())
    assert_metadata(record, payload, reason_class)
    assert "private" not in mcp.log.read_text()


def test_budget_rejected_recall_logged(local_store, question, tmp_path):
    local_store.build(question)
    with local_store.engine(question) as (engine, _):
        mcp = reader(engine, question, tmp_path, calls=0)
        assert mcp.call("lcm_recall", {"query": question.question}) == (mcp_reader.OVER_BUDGET, True)
    record = json.loads(mcp.log.read_text())
    assert not record["counted"]
    assert_metadata(record, {"error": "budget"}, "error")


@pytest.mark.parametrize("summaries", ["none", "deterministic"])
def test_dev_retrieval_metadata(local_store, question, monkeypatch, summaries):
    local_store.summaries = summaries
    local_store.build(question)
    replies = capture(monkeypatch)
    value = dev.retrieve(question, local_store, {"delivery_unit": "session"})
    assert_metadata(value["metadata"], json.loads(replies[-1]),
                    "summary_vectors_unavailable" if summaries == "none" else "none")


@pytest.mark.parametrize("module", [run, dev], ids=["run", "dev"])
@pytest.mark.parametrize("reason", ["store_embedding_registration_failed", "local_recall_semantic_unavailable"])
def test_cli_fixed_local_reason(module, reason, tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(output=tmp_path, worker=None)
    monkeypatch.setattr(module, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(module, "execute", Mock(side_effect=delivery.LocalRecallUnavailable(reason)))
    monkeypatch.setattr(module.sys, "argv", ["fmp"])
    assert module.main() == (1 if module is run else 2)
    assert reason in capsys.readouterr().err
    if module is run:
        assert json.loads((tmp_path / "failure.json").read_text()).get("reason") == reason
