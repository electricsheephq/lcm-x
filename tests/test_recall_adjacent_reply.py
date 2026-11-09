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
    monkeypatch.setattr(tools, "_lcm_recall_next_row", read)
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
    read = Mock(wraps=tools._lcm_recall_next_row)
    monkeypatch.setattr(tools, "_lcm_recall_next_row", read)
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
    assert [call.args[2] for call in read.call_args_list] == [hits[0], hits[2]]
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
    read = Mock(return_value=neighbour)
    return SimpleNamespace(_store=SimpleNamespace(), current_session_id="current", _read=read), entries, rows, neighbour


def apply(e, entries, rows, **kw):
    return tools._lcm_recall_adjacent_reply(e, entries, rows, **{
        "limit": 3, "deadline": tools.time.monotonic() + 10, "excluded": set(), "allowed": None, "read_next": e._read, **kw})


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
        # A moved pool entry keeps its own scores and ranks; it is only repositioned.
        assert result[1]["_final_score"] == entries[3]["_final_score"]
        assert result[1]["rrf_score"] == entries[3]["rrf_score"]
        assert result[1]["ranks"] == entries[3]["ranks"]


@pytest.mark.parametrize("case", ["tool", "tool_calls", "empty", "subagent", "end",
                                  "excluded", "allowed", "wrong_session", "conversation", "summary",
                                  "chunk_only", "tool_source"])
def test_ineligible_is_noop(case):
    e, entries, rows, neighbour = unit_case()
    entries = entries[:1]
    if case == "end":
        e._read.return_value = None
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
    assert e._read.call_count <= 1


def test_max_three_reads_and_original_top_three_sources():
    e, entries, rows, neighbour = unit_case()
    def read(sid, source):
        return {**neighbour, "store_id": source + 1}
    e._read.side_effect = read
    result, pairs, reason = apply(e, entries, rows, limit=10)
    assert reason is None
    assert e._read.call_count == 3
    assert pairs == [{"store_id": i + 1, "after_store_id": i} for i in (1, 3, 5)]
    assert [x["hit"]["store_id"] for x in result] == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("failure", ["exception", "deadline", "expired", "overrun"])
def test_failure_rolls_back_whole_step(failure, monkeypatch):
    e, entries, rows, neighbour = unit_case()
    before = json.dumps(entries)
    e._read.side_effect = [neighbour, RuntimeError("private error must not escape")]
    kw = {}
    if failure == "deadline":
        e._read.side_effect = [neighbour, TimeoutError()]
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
    monkeypatch.setattr(tools, "_lcm_recall_next_row", Mock(side_effect=error))
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
    monkeypatch.setattr(tools, "_lcm_recall_next_row", Mock(side_effect=AssertionError("read")))
    assert invoke(e) == before


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_existing_caps_still_apply(ranked, monkeypatch, detail):
    e, _, _ = ranked
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 4500)
    payload_raw = invoke(e, detail=detail)
    assert len(payload_raw) <= 4500
    assert len(json.loads(payload_raw)["hits"]) <= 6


def reply_from_other_role(e):
    """The shared fixture stores both rows as user turns; a reply must come from the other role."""
    e._store.connection.execute(
        "UPDATE messages SET role = 'assistant' WHERE content LIKE 'unrelated neighbor%'")
    e._store.connection.commit()


@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto", "requirements", "adaptive", "baseline"])
def test_internal_consumers_pin_off_with_model_positive_control(internal_recall, monkeypatch, consumer):
    from hermes_lcm import answer_contract, evidence_compiler, evidence_pack
    e, plugin = internal_recall
    reply_from_other_role(e)
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
    reply_from_other_role(e)
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


# -- Review round 1 regressions -------------------------------------------------

def _entry(sid, store, ranks, score):
    return {"hit": {"kind": "message_excerpt", "store_id": store, "session_id": sid},
            "ranks": ranks, "rrf_score": score, "_final_score": score}


def _step(ordered, rows, read, *, limit):
    return tools._lcm_recall_adjacent_reply(
        SimpleNamespace(_store=SimpleNamespace(), current_session_id="cur"), ordered, rows,
        limit=limit, deadline=tools.time.monotonic() + 5, excluded=set(), allowed=None,
        read_next=read)


def test_fts_anchor_still_inside_window_after_replies():
    from hermes_lcm.retrieval_core import place_fts_anchor
    limit = 4
    ordered = [_entry(f"s{i}", 10 * i, {1: i}, 1.0 / (i + 1)) for i in range(1, 6)]
    ordered.append(_entry("s9", 50, {0: 1}, 0.001))
    rows = {e["hit"]["store_id"]: {"session_id": e["hit"]["session_id"], "conversation_id": "c",
                                   "role": "user", "content": "q?"} for e in ordered}
    read = Mock(side_effect=lambda sid, after: {"store_id": after + 1, "session_id": sid,
                                                "conversation_id": "c", "role": "assistant",
                                                "content": "a"})
    placed, pairs, reason = _step(ordered, rows, read, limit=limit)
    assert reason is None and pairs
    arm_hits = {"fts": [{"kind": "message_excerpt", "store_id": 50}], "chunk": [{}]}
    anchored, info = place_fts_anchor(placed, arm_hits, ["fts", "chunk"], limit)
    assert info["fired"]
    assert [e["hit"]["store_id"] for e in anchored].index(50) < limit
    # lcm_recall places replies before the anchor step, so the #950 guarantee holds.
    source = Path(tools.__file__).read_text()
    body = source[source.index("def lcm_recall("):]
    assert body.index("_lcm_recall_adjacent_reply(") < body.index("place_fts_anchor(")


@pytest.mark.parametrize("case", ["older", "same_role", "user_after_user"])
def test_reply_must_be_other_role_and_not_older(case):
    ordered = [_entry("s", 100, {0: 1}, 3)]
    rows = {100: {"session_id": "s", "conversation_id": "", "role": "user", "content": "x",
                  "observed_at": 2_000_000_000.0}}
    reply = {"store_id": 101, "session_id": "s", "conversation_id": "", "role": "assistant",
             "content": "reply", "observed_at": 2_000_000_100.0}
    if case == "older":
        reply["observed_at"] = 1_000_000_000.0  # a re-stored or imported row, years older
    elif case == "same_role":
        rows[100]["role"] = "assistant"
    else:
        reply["role"] = "user"
    placed, pairs, reason = _step(ordered, rows, Mock(return_value=reply), limit=4)
    assert placed == ordered and pairs == [] and reason is None
    reply.update(role="assistant", observed_at=2_000_000_100.0)
    rows[100]["role"] = "user"
    placed, pairs, reason = _step(ordered, rows, Mock(return_value=reply), limit=4)
    assert pairs == [{"store_id": 101, "after_store_id": 100}]


def test_small_limit_sources_stay_in_window_without_error():
    ordered = [_entry("s", 1, {0: 1}, 3), _entry("t", 9, {0: 2}, 2), _entry("s", 2, {0: 3}, 1)]
    rows = {e["hit"]["store_id"]: {"session_id": e["hit"]["session_id"], "conversation_id": "",
                                   "role": "user", "content": "q?"} for e in ordered}
    read = Mock(side_effect=lambda sid, after: {"store_id": after + 1, "session_id": sid,
                                                "conversation_id": "", "role": "assistant",
                                                "content": "a"})
    placed, pairs, reason = _step(ordered, rows, read, limit=2)
    assert reason is None
    assert read.call_count == 2  # only the two delivered entries are sources
    assert [e["hit"]["store_id"] for e in placed][:2] == [1, 2]
    # The step returns every placed pair; delivery keeps those the final hits carry.
    assert pairs == [{"store_id": 2, "after_store_id": 1}, {"store_id": 10, "after_store_id": 9}]
    window = [{"kind": "message_excerpt", "store_id": e["hit"]["store_id"]} for e in placed[:2]]
    assert recall_delivery._delivered_adjacent_receipts(window, pairs) == pairs[:1]


def test_receipts_only_for_delivered_replies():
    ordered = [_entry("s", 1, {0: 1}, 3), _entry("t", 9, {0: 2}, 2)]
    rows = {1: {"session_id": "s", "conversation_id": "", "role": "user", "content": "x"}}
    read = Mock(return_value={"store_id": 2, "session_id": "s", "conversation_id": "",
                              "role": "assistant", "content": "a"})
    placed, pairs, reason = _step(ordered, rows, read, limit=1)
    assert reason is None and pairs == [{"store_id": 2, "after_store_id": 1}]
    assert [e["hit"]["store_id"] for e in placed][:1] == [1]
    window = [{"kind": "message_excerpt", "store_id": 1}]
    assert recall_delivery._delivered_adjacent_receipts(window, pairs) == []


def test_reply_already_delivered_as_collapsed_duplicate_is_not_repeated():
    ordered = [_entry("s", 1, {0: 1}, 3), _entry("t", 9, {0: 2}, 2)]
    ordered[1]["hit"]["duplicate_store_ids"] = [2]
    rows = {1: {"session_id": "s", "conversation_id": "", "role": "user", "content": "x"}}
    read = Mock(return_value={"store_id": 2, "session_id": "s", "conversation_id": "",
                              "role": "assistant", "content": "same text"})
    placed, pairs, reason = _step(ordered, rows, read, limit=4)
    assert placed == ordered and pairs == [] and reason is None


def test_reads_use_a_private_read_only_connection(ranked, monkeypatch):
    e, hits, replies = ranked
    seen = []
    original = tools._lcm_recall_next_row

    def spy(conn, sid, store_id):
        seen.append(conn)
        return original(conn, sid, store_id)

    monkeypatch.setattr(tools, "_lcm_recall_next_row", spy)
    payload = json.loads(invoke(e))
    assert payload["provenance"]["adjacent_reply"]
    assert seen and all(conn is not e._store.connection for conn in seen)
    assert len({id(conn) for conn in seen}) == 1


def test_receipt_kept_when_reply_is_delivered_as_a_collapsed_duplicate():
    ordered = [_entry("s", 10, {0: 1}, 3), _entry("t", 30, {0: 2}, 2), _entry("t", 40, {0: 3}, 1),
               _entry("u", 99, {1: 9}, 0.1)]
    ordered[3]["hit"]["duplicate_store_ids"] = [11]
    rows = {10: {"session_id": "s", "conversation_id": "", "role": "user", "content": "x"}}
    read = Mock(return_value={"store_id": 11, "session_id": "s", "conversation_id": "",
                              "role": "assistant", "content": "a"})
    placed, pairs, reason = _step(ordered, rows, read, limit=3)
    assert reason is None
    assert [e["hit"]["store_id"] for e in placed][:2] == [10, 99]
    assert pairs == [{"store_id": 11, "after_store_id": 10}]


def test_text_or_invalid_observed_at_skips_only_the_time_check():
    ordered = [_entry("s", 100, {0: 1}, 3)]
    rows = {100: {"session_id": "s", "conversation_id": "", "role": "user", "content": "x",
                  "observed_at": "not a time"}}
    reply = {"store_id": 101, "session_id": "s", "conversation_id": "", "role": "assistant",
             "content": "reply", "observed_at": 1_000_000_000.0}
    placed, pairs, reason = _step(ordered, rows, Mock(return_value=reply), limit=4)
    assert reason is None and pairs == [{"store_id": 101, "after_store_id": 100}]
    rows[100]["observed_at"] = "2033-05-18T03:33:20+00:00"  # aware ISO text is honoured
    placed, pairs, reason = _step(ordered, rows, Mock(return_value=reply), limit=4)
    assert reason is None and pairs == []


# -- Review round 2: receipts follow the final hits; no pool candidate is lost ----

def receipts(payload):
    return payload["provenance"].get("adjacent_reply", [])


def assert_receipts_match_hits(payload):
    """Every receipt names a delivered reply, and every delivered reply has one."""
    delivered = {h["store_id"]: h for h in payload["hits"]}
    for pair in receipts(payload):
        assert pair["store_id"] in delivered or any(
            pair["store_id"] in h.get("duplicate_store_ids", []) for h in payload["hits"])
    marked = {h["store_id"] for h in payload["hits"] if "adjacent_to" in h}
    assert marked <= {pair["store_id"] for pair in receipts(payload)} | {
        h["store_id"] for h in payload["hits"]
        if set(h.get("duplicate_store_ids", [])) & {p["store_id"] for p in receipts(payload)}}


def test_inserted_reply_removed_by_seen_refs_has_no_receipt(ranked):
    e, hits, replies = ranked
    first = json.loads(invoke(e, detail="answer_ready", seen_refs=[]))
    assert {"store_id": replies[0], "after_store_id": hits[0]} in receipts(first)
    ref = next(h["exact_ref"] for h in first["hits"] if h["store_id"] == replies[0])
    payload = json.loads(invoke(e, detail="answer_ready", seen_refs=[ref]))
    assert replies[0] not in ids(payload)
    assert all(pair["store_id"] != replies[0] for pair in receipts(payload))
    assert {"store_id": replies[2], "after_store_id": hits[2]} in receipts(payload)
    assert_receipts_match_hits(payload)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_reply_removed_by_response_cap_has_no_receipt(ranked, monkeypatch, detail):
    e, hits, replies = ranked
    full = invoke(e, detail=detail)
    assert len(receipts(json.loads(full))) == 2
    for cap in range(len(full), 0, -25):
        monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
        raw = invoke(e, detail=detail)
        payload = json.loads(raw)
        if replies[2] not in ids(payload):
            break
    assert replies[0] in ids(payload) and replies[2] not in ids(payload)
    assert receipts(payload) == [{"store_id": replies[0], "after_store_id": hits[0]}]
    assert_receipts_match_hits(payload)
    if detail == "answer_ready":
        assert len(raw) <= cap


def test_receipt_never_evicts_a_hit():
    hit = {"kind": "message_excerpt", "store_id": 2, "adjacent_to": 1}
    pairs = [{"store_id": 2, "after_store_id": 1}]

    def fit(cap):
        response = {"query": "q", "total_results": 1, "hits": [dict(hit)], "provenance": {}}
        cap = cap if cap is not None else len(json.dumps(response, ensure_ascii=False))
        encoded = recall_delivery._fit_answer_ready_response(
            fts_anchor_info=None, response=response, response_char_cap=cap, expansion={},
            delta_requested=False, arm_hits={}, adjacent_reply=pairs)
        assert len(encoded) <= cap
        return json.loads(encoded)

    tight = fit(None)  # the hit fits exactly; the receipt does not
    assert tight["hits"] == [hit] and "adjacent_reply" not in tight["provenance"]
    assert fit(10_000)["provenance"]["adjacent_reply"] == pairs


@pytest.fixture
def pooled(recall_engine, monkeypatch):
    """Rank-2 question's reply is itself a retrieved pool candidate, below the window."""
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.recall_adjacent_reply = True
    hits, replies = [], []
    for i in range(4):
        hits.append(e._store.append(f"s{i}", {"role": "user", "content":
                                              f"needle value {i}" + (" ?" if i == 2 else "")}))
        replies.append(e._store.append(f"s{i}", {"role": "assistant", "content":
                                                 "needle answer 2" if i == 2 else f"answer {i}"}))
    order = [*hits, replies[2]]
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    original = tools._lcm_recall_rerank

    def rank(*a, **kw):
        entries, status, scores = original(*a, **kw)
        entries.sort(key=lambda entry: order.index(entry["hit"]["store_id"]))
        return entries, status, scores

    monkeypatch.setattr(tools, "_lcm_recall_rerank", rank)
    return e, hits, replies


def test_moved_reply_delivered_through_refill_has_receipt(pooled):
    e, hits, replies = pooled
    args = {"detail": "answer_ready", "limit": 3}
    first = json.loads(invoke(e, **args, seen_refs=[]))
    assert ids(first) == [hits[0], replies[0], hits[1]]
    assert receipts(first) == [{"store_id": replies[0], "after_store_id": hits[0]}]
    seen = [h["exact_ref"] for h in first["hits"] if h["store_id"] in (hits[0], hits[1])]
    payload = json.loads(invoke(e, **args, seen_refs=seen))
    # Refill reaches the moved reply (a retrieved candidate) right after its source.
    assert ids(payload) == [replies[0], hits[2], replies[2]]
    moved = payload["hits"][2]
    assert moved["adjacent_to"] == hits[2] and "adjacent" in moved["arms"] and moved["arms"] != ["adjacent"]
    assert receipts(payload) == [{"store_id": replies[0], "after_store_id": hits[0]},
                                 {"store_id": replies[2], "after_store_id": hits[2]}]
    assert_receipts_match_hits(payload)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_collapsed_duplicate_reply_has_receipt(ranked, monkeypatch, detail):
    e, hits, replies = ranked
    original = tools.collapse_duplicate_candidates

    def collapse(entries, rows):
        out = original(entries, rows)
        for entry in out:
            if entry["hit"].get("store_id") == hits[3]:
                entry["hit"]["duplicate_store_ids"] = [replies[0]]
        return out

    monkeypatch.setattr(tools, "collapse_duplicate_candidates", collapse)
    payload = json.loads(invoke(e, detail=detail, limit=3))
    # hits[3] carries hits[0]'s reply as a collapsed copy, so it moves after hits[0].
    assert ids(payload) == [hits[0], hits[3], hits[1]]
    assert payload["hits"][1]["adjacent_to"] == hits[0]
    assert receipts(payload) == [{"store_id": replies[0], "after_store_id": hits[0]}]


def _store_ids(entries):
    return [entry["hit"]["store_id"] for entry in entries]


def test_chained_moved_source_keeps_its_reply_in_the_pool():
    # limit=3, ranked [A, B, C, D]: A's reply is a new row X, B's reply is C,
    # C's reply is D; B and C are questions. Inserting X pushes C out of the
    # window, B moves C (a copy) after itself, then C's reply D is moved after
    # C's copy. D used to be popped and dropped because the copy was not found.
    a, b, c, d, x = 10, 20, 21, 22, 11
    ordered = [_entry("a", a, {0: 1}, 4), _entry("b", b, {0: 2}, 3),
               _entry("b", c, {0: 3}, 2), _entry("b", d, {0: 4}, 1)]
    rows = {a: {"session_id": "a", "conversation_id": "", "role": "user", "content": "x"},
            b: {"session_id": "b", "conversation_id": "", "role": "user", "content": "b?"},
            c: {"session_id": "b", "conversation_id": "", "role": "assistant", "content": "c?"}}
    replies = {(a): (x, "assistant"), (b): (c, "assistant"), (c): (d, "user")}
    read = Mock(side_effect=lambda sid, after: {
        "store_id": replies[after][0], "session_id": sid, "conversation_id": "",
        "role": replies[after][1], "content": "r"})
    before = json.dumps(ordered)
    placed, pairs, reason = _step(ordered, rows, read, limit=3)
    assert reason is None and json.dumps(ordered) == before
    assert _store_ids(placed) == [a, x, b, c, d]
    assert pairs == [{"store_id": x, "after_store_id": a}, {"store_id": c, "after_store_id": b},
                     {"store_id": d, "after_store_id": c}]
    assert sorted(_store_ids(placed)) == sorted(_store_ids(ordered) + [x])


def test_source_whose_reply_is_its_own_collapsed_copy_is_skipped():
    ordered = [_entry("t", 9, {0: 1}, 3), _entry("s", 1, {0: 2}, 2)]
    ordered[1]["hit"]["duplicate_store_ids"] = [2]
    rows = {9: {"session_id": "t", "conversation_id": "", "role": "user", "content": "x"},
            1: {"session_id": "s", "conversation_id": "", "role": "user", "content": "q?"}}
    read = Mock(side_effect=lambda sid, after: {"store_id": after + 1, "session_id": sid,
                                                "conversation_id": "", "role": "assistant",
                                                "content": "a"})
    placed, pairs, reason = _step(ordered, rows, read, limit=2)
    assert reason is None
    # 9's new reply pushes 1 out of the window; 1's reply is 1's own collapsed copy.
    assert _store_ids(placed) == [9, 10, 1]
    assert pairs == [{"store_id": 10, "after_store_id": 9}]


def test_property_no_candidate_is_lost():
    import random
    from collections import Counter

    rng = random.Random(2336)
    shapes = Counter()
    for case in range(500):
        sessions = {f"s{n}": [100 * n + k for k in range(rng.randint(2, 6))] for n in range(rng.randint(1, 3))}
        all_rows = {store_id: {"store_id": store_id, "session_id": sid, "conversation_id": "",
                               "role": "user" if k % 2 == 0 else "assistant",
                               "content": "q?" if rng.random() < 0.6 else "statement"}
                    for sid, store_ids in sessions.items() for k, store_id in enumerate(store_ids)}
        pool_ids = rng.sample(sorted(all_rows), rng.randint(1, len(all_rows)))
        # Bias toward the chained shape: three consecutive rows of one session at ranks 1-3.
        triples = [store_ids[j:j + 3] for store_ids in sessions.values()
                   for j in range(len(store_ids) - 2) if set(store_ids[j:j + 3]) <= set(pool_ids)]
        if triples and rng.random() < 0.5:
            triple = rng.choice(triples)
            others = [i for i in pool_ids if i not in triple]
            pool_ids = others[:1] + triple + others[1:]
        ordered = [_entry(all_rows[i]["session_id"], i, {0: n}, 1.0 / n)
                   for n, i in enumerate(pool_ids, 1)]
        absent = [i for i in all_rows if i not in pool_ids]
        if absent and rng.random() < 0.2:
            rng.choice(ordered)["hit"]["duplicate_store_ids"] = [rng.choice(absent)]
        rows = {i: dict(all_rows[i]) for i in pool_ids}

        def read(sid, after):
            later = [i for i in sessions[sid] if i > after]
            return dict(all_rows[later[0]]) if later else None

        limit = rng.choice([1, 2, 3, 3, 3, 4, 5])
        before = json.dumps(ordered)
        placed, pairs, reason = _step(ordered, rows, read, limit=limit)
        assert reason is None and json.dumps(ordered) == before, case

        def holder(store_id, entries):
            return next((n for n, entry in enumerate(entries)
                         if entry["hit"]["store_id"] == store_id
                         or store_id in entry["hit"].get("duplicate_store_ids", [])), None)

        inserted = [p["store_id"] for p in pairs if holder(p["store_id"], ordered) is None]
        assert Counter(_store_ids(placed)) == Counter(_store_ids(ordered) + inserted), case
        paired = set()
        for k, pair in enumerate(pairs):
            reply_at = holder(pair["store_id"], placed)
            assert _store_ids(placed).index(pair["after_store_id"]) < reply_at, case
            assert placed[reply_at]["hit"]["adjacent_to"] == pair["after_store_id"], case
            original = holder(pair["store_id"], ordered)
            shapes["inserted" if original is None else "moved"] += 1
            if pair["after_store_id"] in {p["store_id"] for p in pairs[:k]}:
                shapes["moved_source"] += 1
            if original is not None:
                # It was below the window when moved: each earlier placement shifts it by at most one.
                assert original + k >= limit, case
                paired.add(placed[reply_at]["hit"]["store_id"])
        # Nothing else moves: unpaired pool entries keep their relative order and carry no marker.
        rest = [entry for entry in placed if entry["hit"]["store_id"] in set(pool_ids) - paired]
        assert _store_ids(rest) == [i for i in pool_ids if i not in paired], case
        assert all("adjacent_to" not in entry["hit"] for entry in rest), case
    # Positive control: the generator reaches every shape, including a moved source's reply.
    assert min(shapes[k] for k in ("inserted", "moved", "moved_source")) >= 5, shapes
