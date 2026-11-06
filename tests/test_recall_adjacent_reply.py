"""Opt-in adjacent turns change only the read-side recall candidate pool."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_lcm import config, recall_delivery, schemas, tools
from hermes_lcm.engine import LCMEngine
from tests import test_lcm_recall, test_recall_session_delivery

recall_engine = test_lcm_recall.recall_engine
internal_recall = test_recall_session_delivery.internal_recall


def ids(payload):
    return [h["store_id"] for h in payload["hits"]]


@pytest.fixture
def ranked(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.recall_adjacent_reply = True
    hits, replies = [], []
    for i in range(6):
        # The full row, rather than the bounded search snippet, ends in '?'.
        text = "needle " + ("long question " * 50 + "?  " if i == 2 else f"value {i}")
        hits.append(e._store.append(f"s{i}", {"role": "user", "content": text}))
        replies.append(e._store.append(f"s{i}", {"role": "assistant", "content": f"answer {i}"}))
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    original = tools._lcm_recall_rerank

    def rank(*a, **kw):
        entries, status, scores = original(*a, **kw)
        entries.sort(key=lambda entry: hits.index(entry["hit"]["store_id"]))
        return entries, status, scores

    monkeypatch.setattr(tools, "_lcm_recall_rerank", rank)
    return e, hits, replies


def invoke(e, **args):
    return tools.lcm_recall({"query": "needle", "limit": 6, **args}, engine=e)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_off_identical_to_unset_and_internal_override(ranked, monkeypatch, detail):
    e, _, _ = ranked
    e._config = SimpleNamespace(**{k: v for k, v in vars(e._config).items()
                                  if k != "recall_adjacent_reply"})
    read = Mock(side_effect=AssertionError("flag-off neighbour read"))
    monkeypatch.setattr(e._store, "get_session_messages_after", read)
    unset = invoke(e, detail=detail)
    e._config.recall_adjacent_reply = False
    assert invoke(e, detail=detail) == unset
    e._config.recall_adjacent_reply = True
    assert tools.lcm_recall({"query": "needle", "limit": 6, "detail": detail},
                            engine=e, adjacent_reply=False) == unset
    assert not any(k.startswith("adjacent_reply") for k in json.loads(unset)["provenance"])
    read.assert_not_called()


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_rank_one_and_third_question_keep_limit_and_pool(ranked, monkeypatch, detail):
    e, hits, replies = ranked
    pool = []
    original = recall_delivery.deliver_recall

    def capture(ordered, **kw):
        pool.extend(ordered)
        return original(ordered, **kw)

    monkeypatch.setattr(recall_delivery, "deliver_recall", capture)
    read = Mock(wraps=e._store.get_session_messages_after)
    monkeypatch.setattr(e._store, "get_session_messages_after", read)
    statements = []
    e._store.connection.set_trace_callback(statements.append)
    changes = e._store.connection.total_changes
    raw = invoke(e, detail=detail)
    payload = json.loads(raw)
    assert ids(payload) == [hits[0], replies[0], hits[1], hits[2], replies[2], hits[3]]
    assert payload["total_results"] == 6 and hits[-1] not in ids(payload)
    assert [x["hit"]["store_id"] for x in pool][-2:] == hits[-2:]
    assert replies[1] not in ids(payload)
    assert payload["provenance"]["adjacent_reply"] == [
        {"store_id": replies[i], "after_store_id": hits[i]} for i in (0, 2)]
    for index, source in ((1, 0), (4, 3)):
        assert payload["hits"][index]["score"] == payload["hits"][source]["score"]
        assert payload["hits"][index]["adjacent_to"] == payload["hits"][source]["store_id"]
        assert payload["hits"][index]["arms"] == ["adjacent"]
    assert read.call_count == 2
    assert all(call.kwargs == {"limit": 1} for call in read.call_args_list)
    assert e._store.connection.total_changes == changes
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in statements)
    assert len(raw) <= tools._LCM_RECALL_RESPONSE_CHAR_CAP


def unit_case():
    rows = {i: {"store_id": i, "session_id": "s", "conversation_id": "c",
                "role": "user", "content": "question?"} for i in (1, 3, 5, 7)}
    entries = [{"hit": {"kind": "message_excerpt", "store_id": i, "session_id": "s"},
                "ranks": {0: n}, "rrf_score": 1 / n, "_final_score": 2 / n}
               for n, i in enumerate(rows, 1)]
    neighbour = {"store_id": 9, "session_id": "s", "conversation_id": "c",
                 "role": "assistant", "content": "reply"}
    store = SimpleNamespace(get_session_messages_after=Mock(return_value=[neighbour]))
    return SimpleNamespace(_store=store, current_session_id="current"), entries, rows, neighbour


def apply(e, entries, rows, **kw):
    return tools._lcm_recall_adjacent_reply(e, entries, rows, **{
        "limit": 3, "deadline": tools.time.monotonic() + 10, "excluded": set(), "allowed": None, **kw})


@pytest.mark.parametrize("delivered", [True, False])
def test_existing_neighbour_stays_or_moves_without_duplication(delivered):
    e, entries, rows, neighbour = unit_case()
    entries[1 if delivered else 3]["hit"]["store_id"] = neighbour["store_id"]
    for row in rows.values():
        row["content"] = "not a question"
    before = json.dumps(entries)
    result, pairs, reason = apply(e, entries, rows)
    assert reason is None and json.dumps(entries) == before
    if delivered:
        assert result == entries and pairs == []
    else:
        assert [x["hit"]["store_id"] for x in result] == [1, 9, 3, 5]
        assert pairs == [{"store_id": 9, "after_store_id": 1}]
        assert result[1]["_final_score"] == entries[0]["_final_score"]
        assert result[1]["rrf_score"] == entries[0]["rrf_score"]


@pytest.mark.parametrize("case", ["tool", "tool_calls", "empty", "subagent", "end",
                                  "excluded", "allowed", "wrong_session", "conversation", "summary",
                                  "chunk_only", "tool_source"])
def test_ineligible_is_noop(case):
    e, entries, rows, neighbour = unit_case()
    entries = entries[:1]
    if case == "end":
        e._store.get_session_messages_after.return_value = []
    elif case == "summary":
        entries[0]["hit"]["kind"] = "summary"
    elif case == "chunk_only":
        entries[0]["hit"].pop("store_id")
    elif case == "tool_source":
        rows[1]["role"] = "tool"
    elif case == "tool":
        neighbour["role"] = "tool"
    elif case == "tool_calls":
        neighbour["tool_calls"] = [{"id": "fixture-call"}]
    elif case == "empty":
        neighbour["content"] = "  "
    elif case == "subagent":
        neighbour["source"] = tools._SUBAGENT_SOURCE
    elif case == "wrong_session":
        neighbour["session_id"] = "excluded"
    elif case == "conversation":
        neighbour["conversation_id"] = "other"
    kw = {"excluded": {"s"}} if case == "excluded" else {"allowed": []} if case == "allowed" else {}
    result, pairs, reason = apply(e, entries, rows, **kw)
    assert result == entries and pairs == [] and reason is None
    assert e._store.get_session_messages_after.call_count <= 1


def test_max_three_reads_and_original_top_three_sources():
    e, entries, rows, neighbour = unit_case()
    def read(sid, source, *, limit):
        return [{**neighbour, "store_id": source + 1}]
    e._store.get_session_messages_after.side_effect = read
    result, pairs, reason = apply(e, entries, rows, limit=10)
    assert reason is None
    assert e._store.get_session_messages_after.call_count == 3
    assert pairs == [{"store_id": i + 1, "after_store_id": i} for i in (1, 3, 5)]
    assert [x["hit"]["store_id"] for x in result] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("failure", ["exception", "deadline", "expired", "overrun"])
def test_failure_rolls_back_whole_step(failure, monkeypatch):
    e, entries, rows, neighbour = unit_case()
    before = json.dumps(entries)
    e._store.get_session_messages_after.side_effect = [[neighbour], RuntimeError("private error must not escape")]
    kw = {}
    if failure == "deadline":
        e._store.get_session_messages_after.side_effect = [[neighbour], TimeoutError()]
    elif failure == "expired":
        kw["deadline"] = -1
    elif failure == "overrun":
        clock = iter([0, 0, 2])
        monkeypatch.setattr(tools.time, "monotonic", lambda: next(clock, 2))
        kw["deadline"] = 1
    result, pairs, reason = apply(e, entries, rows, **kw)
    assert result is entries and pairs == [] and json.dumps(entries) == before
    assert reason == ("read_failed:RuntimeError" if failure == "exception" else "deadline")


@pytest.mark.parametrize("error,reason", [(RuntimeError("private"), "read_failed:RuntimeError"),
                                         (TimeoutError(), "deadline")])
def test_failure_provenance_no_ranking_change(ranked, monkeypatch, error, reason):
    e, hits, _ = ranked
    monkeypatch.setattr(e._store, "get_session_messages_after", Mock(side_effect=error))
    payload = json.loads(invoke(e))
    assert ids(payload) == hits
    assert payload["provenance"]["adjacent_reply_unavailable"] == reason
    assert "adjacent_reply" not in payload["provenance"]


def test_session_delivery_is_byte_identical(ranked, monkeypatch):
    e, _, _ = ranked
    from hermes_lcm import tokens
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    e._config.recall_delivery_unit = "session"
    e._config.recall_adjacent_reply = False
    before = invoke(e)
    e._config.recall_adjacent_reply = True
    monkeypatch.setattr(e._store, "get_session_messages_after", Mock(side_effect=AssertionError("read")))
    assert invoke(e) == before


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_existing_caps_still_apply(ranked, monkeypatch, detail):
    e, _, _ = ranked
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 4500)
    payload_raw = invoke(e, detail=detail)
    assert len(payload_raw) <= 4500
    assert len(json.loads(payload_raw)["hits"]) <= 6


@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto", "requirements", "adaptive", "baseline"])
def test_internal_consumers_pin_off_with_model_positive_control(internal_recall, monkeypatch, consumer):
    from hermes_lcm import answer_contract, evidence_compiler, evidence_pack
    e, plugin = internal_recall
    args = {"query": "needle", "detail": "answer_ready"}
    def consume(*a, retrieve, **kw):
        return {"context": retrieve(args)}
    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    monkeypatch.setattr(answer_contract, "compile_answer_contract", lambda *a: SimpleNamespace(status="planned"))
    e._config.preanswer_evidence_enabled = True
    e._config.preanswer_evidence_mode = "requirements_v1"
    e._adaptive_retrieval = SimpleNamespace(search=lambda dispatch, **kw:
        consume(retrieve=lambda recall_args: dispatch("lcm_recall", recall_args)))
    original = tools.lcm_recall
    calls = []
    def capture(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)
    monkeypatch.setattr(tools, "lcm_recall", capture)
    def consume_internal():
        if consumer == "baseline":
            return plugin._answer_ready_baseline(e, "needle", {})
        if consumer == "adaptive":
            return json.loads(tools.lcm_retrieve({"action": "search"}, engine=e))
        if consumer == "requirements":
            return plugin._pre_llm_context(e, {"user_message": "needle", "baseline_refs": []})
        if consumer == "pack":
            return tools.lcm_evidence_pack({}, engine=e)
        return json.loads(tools.lcm_compile_evidence({"question": "needle", "mode": consumer}, engine=e))
    before = consume_internal()
    e._config.recall_adjacent_reply = True
    own = json.loads(original(args, engine=e))
    assert own["provenance"]["adjacent_reply"]
    assert consume_internal() == before
    assert calls and all(c["adjacent_reply"] is False for c in calls)


def test_proactive_pins_off_with_model_positive_control(internal_recall, monkeypatch):
    e, _ = internal_recall
    e._config.proactive_recall_min_score = 0
    monkeypatch.setattr(e, "_proactive_recall_eligible", lambda: True, raising=False)
    monkeypatch.setattr(e, "_newest_user_message_text", lambda _: "needle", raising=False)
    e._proactive_recall_privacy_warned = e._proactive_recall_trust_warned = False
    e._wrap_relevant_memories = LCMEngine._wrap_relevant_memories
    e._proactive_recall_injected_count = e._proactive_recall_skipped_count = 0
    original = tools.lcm_recall
    calls = []
    def capture(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)
    monkeypatch.setattr(tools, "lcm_recall", capture)
    before = LCMEngine._build_proactive_recall_message(e, [], "system", set())
    assert before is not None
    e._config.recall_adjacent_reply = True
    assert json.loads(original({"query": "needle"}, engine=e))["provenance"]["adjacent_reply"]
    assert LCMEngine._build_proactive_recall_message(e, [], "system", set()) == before
    assert len(calls) == 2 and all(c["adjacent_reply"] is False for c in calls)


def test_all_pinned_call_sites_and_no_model_argument():
    root = Path(__file__).resolve().parents[1]
    paths = ["engine.py", "__init__.py", "evidence_tools.py", "bench/instruments/scale389/phase1a.py",
             "benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py"]
    for path in paths:
        calls = [node for node in ast.walk(ast.parse((root / path).read_text()))
                 if isinstance(node, ast.Call) and any(kw.arg == "delivery_order"
                 and isinstance(kw.value, ast.Constant) and kw.value.value == "score" for kw in node.keywords)]
        assert calls
        assert all(any(kw.arg == "adjacent_reply" and isinstance(kw.value, ast.Constant)
                       and kw.value.value is False for kw in node.keywords) for node in calls)
    assert "adjacent_reply" not in json.dumps(schemas.LCM_RECALL)


@pytest.mark.parametrize("yaml_value,env,expected", [(True, None, True), (False, None, False),
    (True, "false", False), (False, "true", True), ("false", None, False), (1, None, False),
    (None, "garbage", False)])
def test_yaml_env_bool_validation(monkeypatch, yaml_value, env, expected):
    for key in tuple(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {"lcm": {"recall_adjacent_reply": yaml_value}})
    if env is not None:
        monkeypatch.setenv("LCM_RECALL_ADJACENT_REPLY", env)
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_adjacent_reply is expected
    assert "recall_adjacent_reply" not in cfg.ignored_config_yaml_lcm_keys
    if not isinstance(yaml_value, bool):
        assert any("expected bool" in w for w in cfg.config_source_warnings)


def test_no_default_or_preset_reaches_the_flag(monkeypatch):
    """Only an explicit operator setting can turn the flag on (no default, preset or preset override)."""
    from hermes_lcm import presets
    for key in tuple(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {})
    assert config.LCMConfig().recall_adjacent_reply is False
    assert config.LCMConfig.from_env().recall_adjacent_reply is False
    assert "recall_adjacent_reply" not in presets._PRESET_FIELDS
    assert "recall_adjacent_reply" not in config._PRESET_ENV_FIELDS
    for preset in presets.shipped_presets():
        assert "recall_adjacent_reply" not in preset.runtime_env
        assert "LCM_RECALL_ADJACENT_REPLY" not in json.dumps(preset.runtime_env)
