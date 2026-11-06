"""Delivery order changes presentation only, after score-based selection/caps."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_lcm import config, recall_delivery, tools
from hermes_lcm.engine import LCMEngine
from tests import test_lcm_recall, test_recall_session_delivery

recall_engine = test_lcm_recall.recall_engine
internal_recall = test_recall_session_delivery.internal_recall


@pytest.fixture
def delivery():
    # Score order deliberately disagrees with observation AND write time.
    times = [300.75, None, 100.25, 100.25, 200.5, None]
    rows = {i: {"observed_at": value} for i, value in enumerate(times, 1)}
    entries = [
        {"hit": {"kind": "summary" if i == 2 else "message", "store_id": i,
                 "node_id": i if i == 2 else None, "session_id": f"session-{i}",
                 "timestamp": 700 - i, "snippet": f"needle {i} " + "x" * 300,
                 "expand_hint": f"lcm_expand(store_id={i})"},
         "ranks": {0: i}, "_final_score": 1 / i}
        for i in rows
    ]
    read = Mock(return_value=(rows, None))
    options = dict(
        query="needle", limit=6, requested_limit=6, scope_bias=0.0,
        include="all", detail="snippets", reference_strict=False,
        delta_requested=False, seen_refs=set(), include_occurrence_time=False,
        deadline=float("inf"), fts_arm_index=0, arm_order=["fts"],
        arm_weights=[1.0], arm_hits={}, coverage={"fts": "full"},
        served_fts_index="stem", rerank_status="disabled", rerank_scores=[],
        embedding_query_metrics=[], degraded_reasons=[], timed_out=False,
        fts_anchor_info=None, summary_leads=[], limit_cap=25, snippet_chars=400,
        response_char_cap=64000, per_session_limit=3, expanded_hit_limit=6,
        content_chars=4000, strict_selector_factory=Mock(),
        hydrate=Mock(return_value={}), read_event_time=read,
        event_time_fields=tools._lcm_recall_event_time,
        read_occurrence_source=Mock(side_effect=AssertionError("extra read")),
        rerank_enabled=lambda: False,
    )

    def invoke(**kwargs):
        return recall_delivery.deliver_recall(entries, **{**options, **kwargs})

    return invoke, read


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_score_is_byte_identical_when_order_unset(delivery, detail):
    invoke, _ = delivery
    assert invoke(detail=detail) == invoke(detail=detail, delivery_order="score")
    assert "delivery_order" not in json.loads(invoke(detail=detail))["provenance"]


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_time_same_hits_stable_ties_untimed_last(delivery, detail):
    invoke, read = delivery
    score = json.loads(invoke(detail=detail))
    read.reset_mock()
    timed = json.loads(invoke(detail=detail, delivery_order="time"))
    read.assert_called_once_with([1, 3, 4, 5, 6], deadline=float("inf"))
    assert [hit["store_id"] for hit in timed["hits"]] == [3, 4, 5, 1, 2, 6]
    assert sorted(score["hits"], key=lambda h: h["store_id"]) == sorted(
        timed["hits"], key=lambda h: h["store_id"])
    assert timed["provenance"].pop("delivery_order") == "time"
    timed["hits"] = score["hits"]
    assert timed == score


@pytest.mark.parametrize("reason", ["deadline", "read_failed"])
def test_unread_rows_keep_score_order(delivery, reason):
    invoke, read = delivery
    read.return_value = ({1: {"observed_at": 300}}, reason)
    payload = json.loads(invoke(delivery_order="time"))
    assert [hit["store_id"] for hit in payload["hits"]] == list(range(1, 7))
    assert payload["event_time_unavailable"] is True


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_response_cap_keeps_score_survivors(delivery, detail):
    invoke, _ = delivery
    # The first case truncates whole hits during construction. The second
    # reaches answer_ready's full-envelope fit after hit construction fits.
    full = json.loads(invoke(detail=detail))
    final_fit_cap = len(invoke(detail=detail)) - 2 * len(json.dumps(full["hits"][-1]))
    caps = [2200, final_fit_cap] if detail == "answer_ready" else [1300]
    for cap in caps:
        score_raw = invoke(detail=detail, response_char_cap=cap)
        score = json.loads(score_raw)
        timed_raw = invoke(detail=detail, response_char_cap=cap, delivery_order="time")
        # answer_ready caps the whole envelope; snippets budgets hits only, so
        # there time mode is score mode plus the provenance marker exactly.
        if detail == "answer_ready":
            assert len(timed_raw) <= cap
        else:
            assert len(timed_raw) == len(score_raw) + len(', "delivery_order": "time"')
        timed = json.loads(timed_raw)
        ids = [hit["store_id"] for hit in score["hits"]]
        assert 0 < len(ids) < 6
        assert ids == list(range(1, len(ids) + 1))
        assert sorted(ids) == sorted(hit["store_id"] for hit in timed["hits"])
        assert sorted(score["hits"], key=lambda h: h["store_id"]) == sorted(
            timed["hits"], key=lambda h: h["store_id"])


def test_stripped_dates_still_order_by_existing_rows(delivery):
    invoke, _ = delivery
    # All six hit bodies fit, but their optional dates exceed this cap.
    full = json.loads(invoke())
    cap = sum(len(json.dumps({k: v for k, v in hit.items()
                              if k not in {"event_time", "event_time_source", "time_uncertain"}}))
              for hit in full["hits"])
    payload = json.loads(invoke(response_char_cap=cap, delivery_order="time"))
    assert [h["store_id"] for h in payload["hits"]] == [3, 4, 5, 1, 2, 6]
    assert any("event_time" not in h for h in payload["hits"] if h["store_id"] in {1, 3, 4, 5})


@pytest.mark.parametrize("raw,expected", [(None, "score"), ("time", "time"),
                                          (" TIME ", "time"), ("bad", "score")])
def test_env_choice(monkeypatch, raw, expected):
    for key in tuple(config.os.environ):
        if key.startswith("LCM_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(config, "_load_hermes_config_yaml", lambda: {})
    if raw is not None:
        monkeypatch.setenv("LCM_RECALL_DELIVERY_ORDER", raw)
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_delivery_order == expected
    warnings = [w for w in cfg.config_source_warnings if "LCM_RECALL_DELIVERY_ORDER" in w]
    assert bool(warnings) == (raw == "bad")
    if warnings:
        assert "ignored; using 'score'" in warnings[0]


def test_config_constructor():
    assert config.LCMConfig().recall_delivery_order == "score"
    assert config.LCMConfig(recall_delivery_order="time").recall_delivery_order == "time"

@pytest.fixture
def ranked_engine(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    ids = [e._store.append(f"s{i}", {"role": "user", "content": f"needle value {i}",
                                    "timestamp": observed})
           for i, observed in enumerate([300, 100, 200])]
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    original = tools._lcm_recall_rerank

    def rank(*args, **kwargs):
        entries, status, scores = original(*args, **kwargs)
        entries.sort(key=lambda entry: ids.index(entry["hit"]["store_id"]))
        return entries, status, scores

    monkeypatch.setattr(tools, "_lcm_recall_rerank", rank)
    return e, ids


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_tools_config_and_internal_override(ranked_engine, detail):
    e, ids = ranked_engine
    args = {"query": "needle", "detail": detail}
    e._config = SimpleNamespace(**{k: v for k, v in vars(e._config).items()
                                  if k != "recall_delivery_order"})
    unset = tools.lcm_recall(args, engine=e)
    e._config.recall_delivery_order = "score"
    assert tools.lcm_recall(args, engine=e) == unset
    e._config.recall_delivery_order = "time"
    timed = json.loads(tools.lcm_recall(args, engine=e))
    assert [h["store_id"] for h in timed["hits"]] == [ids[1], ids[2], ids[0]]
    assert tools.lcm_recall(args, engine=e, delivery_order="score") == unset


def test_session_delivery_ignores_time(ranked_engine, monkeypatch):
    e, _ = ranked_engine
    from hermes_lcm import tokens
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    e._config.recall_delivery_unit = "session"
    args = {"query": "needle"}
    before = tools.lcm_recall(args, engine=e)
    e._config.recall_delivery_order = "time"
    assert tools.lcm_recall(args, engine=e) == before
    assert "delivery_order" not in json.loads(before)["provenance"]


@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto", "requirements", "adaptive", "baseline"])
def test_internal_consumers_ignore_time(internal_recall, monkeypatch, consumer):
    from hermes_lcm import answer_contract, evidence_compiler, evidence_pack
    e, plugin = internal_recall
    e._store.append("other", {"role": "user", "content": "needle second", "timestamp": 100})
    args = {"query": "needle", "detail": "answer_ready"}

    def consume(*a, retrieve, **kw):
        return {"context": retrieve(args)}

    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    monkeypatch.setattr(answer_contract, "compile_answer_contract",
                        lambda *a: SimpleNamespace(status="planned"))
    e._config.preanswer_evidence_enabled = True
    e._config.preanswer_evidence_mode = "requirements_v1"
    e._adaptive_retrieval = SimpleNamespace(search=lambda dispatch, **kw:
        consume(retrieve=lambda recall_args: dispatch("lcm_recall", recall_args)))

    def invoke():
        if consumer == "baseline":
            return plugin._answer_ready_baseline(e, "needle", {})
        if consumer == "adaptive":
            return json.loads(tools.lcm_retrieve({"action": "search"}, engine=e))
        if consumer == "requirements":
            return plugin._pre_llm_context(e, {"user_message": "needle", "baseline_refs": []})
        if consumer == "pack":
            return tools.lcm_evidence_pack({}, engine=e)
        return json.loads(tools.lcm_compile_evidence(
            {"question": "needle", "mode": consumer}, engine=e))

    before = invoke()
    e._config.recall_delivery_order = "time"
    # Positive control: on this fixture the model's own call does reorder.
    own = json.loads(tools.lcm_recall(args, engine=e))
    own_score = json.loads(tools.lcm_recall(args, engine=e, delivery_order="score"))
    assert [h.get("store_id") for h in own["hits"]] != [h.get("store_id") for h in own_score["hits"]]
    assert invoke() == before


def test_proactive_ignores_time(ranked_engine, monkeypatch):
    e, _ = ranked_engine
    e._config.proactive_recall_min_score = 0
    monkeypatch.setattr(e, "_proactive_recall_eligible", lambda: True, raising=False)
    monkeypatch.setattr(e, "_newest_user_message_text", lambda _: "needle", raising=False)
    monkeypatch.setattr(e, "_proactive_recall_privacy_warned", False, raising=False)
    monkeypatch.setattr(e, "_proactive_recall_trust_warned", False, raising=False)
    e._wrap_relevant_memories = LCMEngine._wrap_relevant_memories
    e._proactive_recall_injected_count = 0
    e._proactive_recall_skipped_count = 0
    # Record real recall and exercise the rendered proactive block too.
    original = tools.lcm_recall
    responses = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        responses.append(result)
        return result

    monkeypatch.setattr(tools, "lcm_recall", capture)
    before = LCMEngine._build_proactive_recall_message(e, [], "system", set())
    assert before is not None
    e._config.recall_delivery_order = "time"
    assert LCMEngine._build_proactive_recall_message(e, [], "system", set()) == before
    assert len(responses) == 2
    assert responses[0] == responses[1]
    assert "delivery_order" not in json.loads(responses[1])["provenance"]


def test_benchmark_callers_pin_score():
    root = Path(__file__).resolve().parents[1]
    paths = ["bench/instruments/scale389/phase1a.py", "bench/instruments/local_catchup/recall.py",
             "benchmarking/longmemeval.py",
             "benchmarks/qa-harness/src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py"]
    for path in paths:
        calls = [node for node in ast.walk(ast.parse((root / path).read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "lcm_recall"]
        assert calls
        assert all(any(kw.arg == "delivery_order" and isinstance(kw.value, ast.Constant)
                       and kw.value.value == "score" for kw in call.keywords) for call in calls)


def test_time_marker_never_exceeds_the_cap(delivery):
    invoke, _ = delivery
    full = invoke(detail="answer_ready", delivery_order="time")
    # A cap that fits the unmarked response exactly: the marker must be omitted, no hit dropped.
    unmarked = json.loads(full)
    unmarked["provenance"].pop("delivery_order")
    cap = len(json.dumps(unmarked, ensure_ascii=False))
    raw = invoke(detail="answer_ready", response_char_cap=cap, delivery_order="time")
    assert len(raw) <= cap
    assert len(json.loads(raw)["hits"]) == len(unmarked["hits"])
