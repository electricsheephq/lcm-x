"""Opt-in snippet windows are presentation-only, using the existing exact loader."""
from __future__ import annotations

from functools import partial
import json
import sqlite3
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_lcm import config, recall_delivery, tools
from hermes_lcm.engine import LCMEngine
from tests import test_lcm_recall

recall_engine = test_lcm_recall.recall_engine


def make_delivery(lengths=(1300,) * 7):
    rows, entries = {}, []
    for i, length in enumerate(lengths, 1):
        text = "x" * 799 + " ANSWER_TOKEN " + "y" * (length - 813)
        rows[i] = {"content": text, "role": "assistant", "source": "chat"}
        entries.append({
            "hit": {"kind": "message_excerpt", "store_id": i, "session_id": f"s{i}",
                    "timestamp": 100, "snippet": text[:300],
                    "chunk_span": {"char_start": 800, "char_end": 812},
                    "expand_hint": f"lcm_expand(store_id={i})"},
            "ranks": {0: i}, "_final_score": 1 / i,
        })
    batch = Mock(side_effect=lambda ids: {i: rows[i] for i in ids if i in rows})
    engine = SimpleNamespace(_store=SimpleNamespace(get_batch=batch))
    hydrate = Mock(wraps=partial(tools._lcm_recall_answer_ready_content, engine))
    read_time = Mock(return_value=(rows, None))
    read_windows = Mock(side_effect=lambda entries, deadline, **kw: (hydrate(entries, **kw), None))
    options = dict(
        query="ANSWER_TOKEN", limit=len(entries), requested_limit=len(entries), scope_bias=0.0,
        include="all", detail="snippets", reference_strict=False, delta_requested=False,
        seen_refs=set(), include_occurrence_time=False, deadline=float("inf"),
        fts_arm_index=0, arm_order=["chunk"], arm_weights=[1.0], arm_hits={},
        coverage={"chunk": "full"}, served_fts_index="stem", rerank_status="disabled",
        rerank_scores=[], embedding_query_metrics=[], degraded_reasons=[], timed_out=False,
        fts_anchor_info=None, summary_leads=[], limit_cap=tools._LCM_RECALL_LIMIT_CAP,
        snippet_chars=tools._LCM_RECALL_SNIPPET_CHARS,
        response_char_cap=tools._LCM_RECALL_RESPONSE_CHAR_CAP,
        per_session_limit=tools._LCM_RECALL_ANSWER_READY_PER_SESSION_LIMIT,
        expanded_hit_limit=tools._LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT,
        content_chars=tools._LCM_RECALL_ANSWER_READY_CONTENT_CHARS,
        strict_selector_factory=Mock(), hydrate=hydrate, read_event_time=read_time,
        read_snippet_windows=read_windows,
        event_time_fields=tools._lcm_recall_event_time, read_occurrence_source=Mock(),
        rerank_enabled=lambda: False,
    )

    def invoke(**kwargs):
        return recall_delivery.deliver_recall(entries, **{**options, **kwargs})

    return SimpleNamespace(invoke=invoke, entries=entries, rows=rows, hydrate=hydrate,
                           batch=batch, read_time=read_time, read_windows=read_windows)


def fts_hit(content, *, store_id=1):
    # Match the real _lcm_recall_fts_arm producer, including its marked excerpt.
    match = content.index("ANSWER_TOKEN")
    snippet = content[match - 100:match] + ">>>ANSWER_TOKEN<<<" + content[match + 12:match + 112]
    return {
        "kind": "message_excerpt", "store_id": store_id, "session_id": f"s{store_id}",
        "source": "chat", "role": "assistant", "timestamp": 100, "content_offset": 0,
        "snippet": snippet[:300], "_fts_snippet": snippet, "from_current_session": False,
        "expand_hint": f"lcm_expand(store_id={store_id})",
    }


# Captured from the unchanged base, before implementing the flag. This literal
# pins JSON bytes (including key order), rather than comparing two new-code runs.
BASELINE_JSON = '{"query": "ANSWER_TOKEN", "limit": 1, "scope_bias": 0.0, "include": "all", "total_results": 1, "hits": [{"kind": "message_excerpt", "session_id": "s1", "timestamp": 100, "snippet": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "score": 1.0, "expand_hint": "lcm_expand(store_id=1)", "from_current_session": false, "arms": ["chunk"], "store_id": 1, "chunk_span": {"char_start": 800, "char_end": 812}}], "provenance": {"arms_run": ["chunk"], "arm_weights": {"chunk": 1.0}, "coverage": {"chunk": "full"}, "fts_index": "stem", "rerank": "disabled", "ordering": "rrf-fusion -> scope/recency prior -> rerank reorder (top window); the reported score is the scope/recency-adjusted RRF score, and rerank (when applied) only permutes the top window without replacing that score"}, "metrics": {"embedding_query_calls": 0, "embedding_query_tokens": 0, "embedding_query_tokens_complete": true, "embedding_queries": []}, "degraded": false}'


def test_default_bytes_match_base():
    d = make_delivery((1300,))
    assert d.invoke() == BASELINE_JSON
    assert "snippet_window" not in d.invoke()
    d.hydrate.assert_not_called()


def test_first_five_windows_and_one_batch():
    d = make_delivery()
    before = [entry["hit"]["snippet"] for entry in d.entries]
    payload = json.loads(d.invoke(snippet_window_hits=5))
    for index, hit in enumerate(payload["hits"]):
        if index < 5:
            assert "ANSWER_TOKEN" in hit["snippet"]
            assert hit["snippet_window"] == {
                "content_offset": 0, "content_returned_chars": 1300,
                "content_chars": 1300, "content_truncated": False,
            }
        else:
            assert len(hit["snippet"]) == 300
            assert "snippet_window" not in hit
    d.hydrate.assert_called_once_with(d.entries[:5], query="ANSWER_TOKEN",
                                    expanded_limit=5, fts_arm_index=0)
    d.batch.assert_called_once_with([1, 2, 3, 4, 5])
    assert [entry["hit"]["snippet"] for entry in d.entries] == before
    d.read_time.assert_called_once_with(list(range(1, 8)), deadline=float("inf"))


def test_whole_short_message_and_centered_long_window():
    d = make_delivery((1500, 3000))
    d.entries[1]["hit"]["chunk_span"] = {"char_start": 1500, "char_end": 1512}
    hits = json.loads(d.invoke(snippet_window_hits=5))["hits"]
    assert hits[0]["snippet"] == d.rows[1]["content"]
    assert hits[0]["snippet_window"]["content_truncated"] is False
    assert hits[1]["snippet"] == d.rows[2]["content"][306:2706]
    assert hits[1]["snippet_window"] == {
        "content_offset": 306, "content_returned_chars": 2400,
        "content_chars": 3000, "content_truncated": True,
    }


def test_other_kinds_adjacent_and_missing_rows_keep_original_snippets():
    d = make_delivery()
    d.entries[0]["hit"].update(kind="summary", node_id=1)
    d.entries[1]["hit"] = fts_hit(d.rows[2]["content"], store_id=2)
    d.entries[2]["hit"]["store_id"] = None
    d.entries[3]["hit"]["adjacent_to"] = 1
    del d.rows[5]  # The existing loader omits a disappeared row.
    hits = json.loads(d.invoke(snippet_window_hits=2, arm_order=["fts"]))["hits"]
    assert [h["store_id"] for h in hits if "snippet_window" in h] == [2]
    assert "ANSWER_TOKEN" in hits[1]["snippet"]
    assert hits[1]["arms"] == ["fts"]
    assert all(len(h["snippet"]) == 300 for h in hits if "snippet_window" not in h)
    d.batch.assert_called_once_with([2, 5])


def test_answer_ready_bytes_unchanged():
    d = make_delivery()
    off = d.invoke(detail="answer_ready")
    assert d.invoke(detail="answer_ready", snippet_window_hits=5) == off
    assert "snippet_window" not in off


def test_windowed_items_still_obey_existing_hit_cap():
    d = make_delivery()
    hits = json.loads(d.invoke(snippet_window_hits=5, response_char_cap=3500))["hits"]
    assert 0 < len(hits) < 7
    assert sum(len(json.dumps(h)) for h in hits) <= 3500


@pytest.fixture
def ranked(recall_engine, monkeypatch):
    e = recall_engine
    e._config.embeddings_enabled = False
    d = make_delivery()
    for i, entry in enumerate(d.entries, 1):
        row = d.rows[i]
        store_id = e._store.append(f"s{i}", {"role": "assistant", "content": row["content"]})
        entry["hit"]["store_id"] = store_id
        entry["rrf_score"] = entry["_final_score"]
        if i == 1:
            e.reply_id = e._store.append("s1", {"role": "user", "content": "reply " + "z" * 1200})
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    monkeypatch.setattr(tools, "_lcm_recall_rerank", lambda *a, **kw: (d.entries, "disabled", []))
    return e, d.entries


@pytest.mark.parametrize("configured,override,expected", [(0, 5, 5), (5, 0, 0), (5, 2, 2), (5, 99, 7)])
def test_per_call_overrides_config(ranked, configured, override, expected):
    e, _ = ranked
    e._config.recall_snippet_window_hits = configured
    payload = json.loads(tools.lcm_recall({"query": "ANSWER_TOKEN", "limit": 7}, engine=e,
                                        snippet_window_hits=override))
    assert sum("snippet_window" in h for h in payload["hits"]) == expected


def test_config_enables_windows_and_answer_ready_stays_identical(ranked):
    e, _ = ranked
    args = {"query": "ANSWER_TOKEN", "limit": 7, "detail": "answer_ready"}
    off = tools.lcm_recall(args, engine=e)
    e._config.recall_snippet_window_hits = 5
    assert tools.lcm_recall(args, engine=e) == off
    args["detail"] = "snippets"
    assert sum("snippet_window" in h for h in json.loads(tools.lcm_recall(args, engine=e))["hits"]) == 5


def test_adjacent_reply_keeps_its_snippet(ranked):
    e, _ = ranked
    e._config.recall_adjacent_reply = True
    args = {"query": "ANSWER_TOKEN", "limit": 7}
    off = json.loads(tools.lcm_recall(args, engine=e))
    e._config.recall_snippet_window_hits = 5
    on = json.loads(tools.lcm_recall(args, engine=e))
    adjacent_off = [h for h in off["hits"] if h.get("adjacent_to") is not None]
    assert [h["store_id"] for h in adjacent_off] == [e.reply_id]
    assert [h for h in on["hits"] if h.get("adjacent_to") is not None] == adjacent_off
    assert sum("snippet_window" in h for h in on["hits"]) == 5


def test_reranker_documents_stay_unchanged():
    d = make_delivery()
    config = SimpleNamespace(rerank_enabled=True, rerank_margin=0)
    provider = SimpleNamespace(provider_id="voyage", rerank=Mock(return_value=[(i, 1.0) for i in range(7)]))
    kwargs = dict(window=7, deadline=float("inf"), config=config)
    # Exercise the real rerank document builder with privacy dispatch replaced
    # by an inert local policy; no provider call can leave this process.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tools, "embedding_provider_requires_privacy", lambda _: False)
        tools._lcm_recall_rerank(provider, "ANSWER_TOKEN", d.entries, **kwargs)
        documents_off = provider.rerank.call_args.args[1]
        d.invoke(snippet_window_hits=5)
        tools._lcm_recall_rerank(provider, "ANSWER_TOKEN", d.entries, **kwargs)
    assert provider.rerank.call_count == 2
    assert provider.rerank.call_args.args[1] == documents_off
    assert documents_off == [entry["hit"]["snippet"] for entry in d.entries]
    assert all(len(text) == 300 and "ANSWER_TOKEN" not in text for text in documents_off)


def test_proactive_explicitly_pins_windows_off(ranked, monkeypatch):
    e, _ = ranked
    e._config.proactive_recall_min_score = 0
    e._config.recall_snippet_window_hits = 5
    e._proactive_recall_eligible = lambda: True
    e._newest_user_message_text = lambda _: "ANSWER_TOKEN"
    e._proactive_recall_privacy_warned = False
    e._proactive_recall_trust_warned = False
    e._wrap_relevant_memories = LCMEngine._wrap_relevant_memories
    e._proactive_recall_injected_count = e._proactive_recall_skipped_count = 0
    original = tools.lcm_recall
    payloads = []

    def capture(*args, **kwargs):
        assert kwargs["snippet_window_hits"] == 0
        raw = original(*args, **kwargs)
        payloads.append(raw)
        return raw

    monkeypatch.setattr(tools, "lcm_recall", capture)
    before = LCMEngine._build_proactive_recall_message(e, [], "system", set())
    assert before is not None
    e._config.recall_snippet_window_hits = 0
    assert LCMEngine._build_proactive_recall_message(e, [], "system", set()) == before
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    assert "snippet_window" not in payloads[0]


@pytest.mark.parametrize("length", [1300, 4000])
def test_fts_producer_shape_gets_a_matched_window(length):
    d = make_delivery((length,))
    d.entries[0]["hit"] = fts_hit(d.rows[1]["content"])
    before = dict(d.entries[0]["hit"])
    hit, = json.loads(d.invoke(snippet_window_hits=1, arm_order=["fts"]))["hits"]
    assert hit["kind"] == "message_excerpt"
    assert hit["arms"] == ["fts"]
    assert hit["store_id"] == 1
    assert "ANSWER_TOKEN" in hit["snippet"]
    metadata = hit["snippet_window"]
    assert metadata["content_chars"] == length
    assert metadata["content_returned_chars"] == min(length, 2400)
    assert metadata["content_truncated"] == (length > 2400)
    offset = metadata["content_offset"]
    assert hit["snippet"] == d.rows[1]["content"][offset:offset + 2400]
    assert d.entries[0]["hit"] == before


def test_real_fts_pipeline_gets_window(recall_engine):
    e = recall_engine
    e._config.embeddings_enabled = False
    content = "prefix " * 400 + "ANSWER_TOKEN " + "suffix " * 200
    store_id = e._store.append("s1", {"role": "assistant", "content": content})
    args = {"query": "ANSWER_TOKEN", "limit": 1}
    off = json.loads(tools.lcm_recall(args, engine=e))
    on = json.loads(tools.lcm_recall(args, engine=e, snippet_window_hits=1))
    hit, = on["hits"]
    assert hit["kind"] == "message_excerpt" and hit["store_id"] == store_id
    assert hit["arms"] == ["fts"]
    assert "ANSWER_TOKEN" in hit["snippet"]
    assert hit["snippet_window"]["content_returned_chars"] == 2400
    assert hit["snippet_window"]["content_offset"] > 0
    assert "snippet_window" not in off["hits"][0]


# Frozen from unchanged base delivery with the real FTS producer fields.
FTS_BASELINE_JSON = '{"query": "ANSWER_TOKEN", "limit": 1, "scope_bias": 0.0, "include": "all", "total_results": 1, "hits": [{"kind": "message_excerpt", "session_id": "s1", "timestamp": 100, "snippet": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx >>>ANSWER_TOKEN<<< yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy", "score": 1.0, "expand_hint": "lcm_expand(store_id=1)", "from_current_session": false, "arms": ["fts"], "store_id": 1}], "provenance": {"arms_run": ["fts"], "arm_weights": {"fts": 1.0}, "coverage": {"chunk": "full"}, "fts_index": "stem", "rerank": "disabled", "ordering": "rrf-fusion -> scope/recency prior -> rerank reorder (top window); the reported score is the scope/recency-adjusted RRF score, and rerank (when applied) only permutes the top window without replacing that score"}, "metrics": {"embedding_query_calls": 0, "embedding_query_tokens": 0, "embedding_query_tokens_complete": true, "embedding_queries": []}, "degraded": false}'


def test_fts_default_bytes_match_base():
    d = make_delivery((1300,))
    d.entries[0]["hit"] = fts_hit(d.rows[1]["content"])
    assert d.invoke(snippet_window_hits=0, arm_order=["fts"]) == FTS_BASELINE_JSON
    d.hydrate.assert_not_called()


def test_window_read_error_preserves_switch_off_items(ranked, monkeypatch, caplog):
    e, _ = ranked
    args = {"query": "ANSWER_TOKEN", "limit": 7}
    off = json.loads(tools.lcm_recall(args, engine=e, snippet_window_hits=0))
    hydrate = Mock(side_effect=sqlite3.OperationalError("database is locked"))
    monkeypatch.setattr(tools, "_lcm_recall_answer_ready_content", hydrate)
    on = json.loads(tools.lcm_recall(args, engine=e, snippet_window_hits=3))
    assert json.dumps(on["hits"]) == json.dumps(off["hits"])
    assert on.pop("snippet_window_skipped") == "read_error"
    assert on == off
    hydrate.assert_called_once()
    warnings = [r.getMessage() for r in caplog.records if "snippet window read failed" in r.getMessage()]
    assert warnings == ["lcm_recall snippet window read failed: OperationalError"]
    assert "database is locked" not in caplog.text


def test_expired_window_step_never_opens_database(ranked, monkeypatch):
    e, _ = ranked
    d = make_delivery()
    off = json.loads(d.invoke())
    read = partial(tools._lcm_recall_snippet_windows, e)
    connect = Mock(side_effect=AssertionError("expired read"))
    monkeypatch.setattr(tools.sqlite3, "connect", connect)
    on = json.loads(d.invoke(snippet_window_hits=3, deadline=0, read_snippet_windows=read))
    assert on.pop("snippet_window_skipped") == "deadline"
    assert on == off
    connect.assert_not_called()


def test_window_read_only_covers_deliverable_candidates():
    d = make_delivery((1300,) * 20)
    hits = json.loads(d.invoke(snippet_window_hits=5, limit=3))["hits"]
    d.batch.assert_called_once_with([1, 2, 3])
    assert [h["store_id"] for h in hits] == [1, 2, 3]
    assert all("snippet_window" in h for h in hits)


# Frozen r2 output for the non-ASCII cap fixture: original 300-char previews
# remain ASCII while the expanded content is CJK.
CJK_BASELINE_JSON = '{"query": "ANSWER_TOKEN", "limit": 2, "scope_bias": 0.0, "include": "all", "total_results": 2, "hits": [{"kind": "message_excerpt", "session_id": "s1", "timestamp": 100, "snippet": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "score": 1.0, "expand_hint": "lcm_expand(store_id=1)", "from_current_session": false, "arms": ["chunk"], "store_id": 1, "chunk_span": {"char_start": 800, "char_end": 812}}, {"kind": "message_excerpt", "session_id": "s2", "timestamp": 100, "snippet": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "score": 0.5, "expand_hint": "lcm_expand(store_id=2)", "from_current_session": false, "arms": ["chunk"], "store_id": 2, "chunk_span": {"char_start": 800, "char_end": 812}}], "provenance": {"arms_run": ["chunk"], "arm_weights": {"chunk": 1.0}, "coverage": {"chunk": "full"}, "fts_index": "stem", "rerank": "disabled", "ordering": "rrf-fusion -> scope/recency prior -> rerank reorder (top window); the reported score is the scope/recency-adjusted RRF score, and rerank (when applied) only permutes the top window without replacing that score"}, "metrics": {"embedding_query_calls": 0, "embedding_query_tokens": 0, "embedding_query_tokens_complete": true, "embedding_queries": []}, "degraded": false}'


@pytest.mark.parametrize("delivery_order", ["score", "time"])
def test_non_ascii_windows_fit_serialized_response_cap(delivery_order):
    d = make_delivery((2400, 2400))
    for row in d.rows.values():
        row["content"] = "証" * 2400
    off = d.invoke(snippet_window_hits=0, response_char_cap=17000)
    assert off == CJK_BASELINE_JSON
    raw = d.invoke(snippet_window_hits=2, response_char_cap=17000, delivery_order=delivery_order)
    on = json.loads(raw)
    assert len(raw) <= 17000
    assert on["response_cap_truncated"] is True
    assert len(on["hits"]) == 1
    assert on["hits"][0]["snippet"] == "証" * 2400
    assert "response_cap_truncated" not in json.loads(off)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_session_delivery_ignores_windows(ranked, monkeypatch, detail):
    e, _ = ranked
    args = {"query": "ANSWER_TOKEN", "limit": 3, "detail": detail}
    off = tools.lcm_recall(args, engine=e, delivery_unit="session", snippet_window_hits=0)
    read = Mock(side_effect=AssertionError("session delivery must bypass windows"))
    monkeypatch.setattr(tools, "_lcm_recall_snippet_windows", read)
    assert tools.lcm_recall(args, engine=e, delivery_unit="session", snippet_window_hits=5) == off
    read.assert_not_called()


@pytest.mark.parametrize("raw,expected,source", [
    (None, 0, "default"), ("", 0, "default"), ("   ", 0, "default"),
    ("5", 5, "env:LCM_RECALL_SNIPPET_WINDOW_HITS"), ("bad", 0, "default"),
])
def test_window_env_warning_and_provenance(monkeypatch, tmp_path, raw, expected, source):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("LCM_RECALL_SNIPPET_WINDOW_HITS", raising=False)
    if raw is not None:
        monkeypatch.setenv("LCM_RECALL_SNIPPET_WINDOW_HITS", raw)
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_snippet_window_hits == expected
    assert cfg.config_sources["recall_snippet_window_hits"] == source
    warnings = [w for w in cfg.config_source_warnings if "recall_snippet_window_hits" in w]
    assert warnings == (["invalid env LCM_RECALL_SNIPPET_WINDOW_HITS='bad' ignored; "
                         "using recall_snippet_window_hits=0"] if raw == "bad" else [])


@pytest.mark.parametrize("raw,expected,source", [
    (5, 5, "config_yaml:lcm.recall_snippet_window_hits"), (True, 0, "default"),
])
def test_window_yaml_warning_and_provenance(monkeypatch, tmp_path, raw, expected, source):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LCM_RECALL_SNIPPET_WINDOW_HITS", "")
    (tmp_path / "config.yaml").write_text("lcm:\n  recall_snippet_window_hits: " + json.dumps(raw))
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_snippet_window_hits == expected
    assert cfg.config_sources["recall_snippet_window_hits"] == source
    warnings = [w for w in cfg.config_source_warnings if "recall_snippet_window_hits" in w]
    assert warnings == (["invalid config.yaml lcm.recall_snippet_window_hits=True ignored; "
                         "using recall_snippet_window_hits=0"] if raw is True else [])


def test_handle_tool_call_pins_windows_off(ranked):
    e, _ = ranked
    e._config.recall_snippet_window_hits = 5
    e._disabled_tool_names = lambda: set()
    args = {"query": "ANSWER_TOKEN", "limit": 7}
    off = tools.lcm_recall(args, engine=e, snippet_window_hits=0)
    assert LCMEngine.handle_tool_call(e, "lcm_recall", args, snippet_window_hits=0) == off
    assert "snippet_window" not in off


def test_private_window_read_avoids_shared_store_connection(ranked, monkeypatch):
    e, _ = ranked
    monkeypatch.setattr(e._store, "get_batch", Mock(side_effect=AssertionError("shared read")))
    hits = json.loads(tools.lcm_recall({"query": "ANSWER_TOKEN", "limit": 3}, engine=e,
                                    snippet_window_hits=3))["hits"]
    assert sum("snippet_window" in h for h in hits) == 3
    e._store.get_batch.assert_not_called()


def test_private_window_busy_wait_is_deadline_bounded(tmp_path, caplog):
    path = tmp_path / "locked.db"
    writer = sqlite3.connect(path)
    writer.execute("CREATE TABLE messages (store_id INTEGER, content TEXT, role TEXT, source TEXT)")
    writer.commit()
    writer.execute("BEGIN EXCLUSIVE")
    e = SimpleNamespace(_store=SimpleNamespace(db_path=path))
    d = make_delivery((1300,))
    before = json.loads(d.invoke())
    started = time.monotonic()
    try:
        raw = d.invoke(snippet_window_hits=3, deadline=started + 0.05,
                       read_snippet_windows=partial(tools._lcm_recall_snippet_windows, e))
    finally:
        writer.rollback()
        writer.close()
    assert time.monotonic() - started < 0.5
    after = json.loads(raw)
    assert after.pop("snippet_window_skipped") in {"deadline", "read_error"}
    assert after == before
    assert "OperationalError" in caplog.text
    assert "database is locked" not in caplog.text


def test_window_cap_can_drop_every_hit_and_clears_delivery_receipts():
    d = make_delivery((2400, 2400))
    d.entries[1]["hit"]["adjacent_to"] = 1
    raw = d.invoke(
        snippet_window_hits=2, response_char_cap=1300,
        adjacent_reply=[{"store_id": 2, "after_store_id": 1}],
        arm_order=["chunk", "fts"], arm_weights=[1.0, 1.0],
        arm_hits={"fts": [d.entries[0]["hit"]]},
        fts_anchor_info={"fired": True, "delivered": None},
    )
    on = json.loads(raw)
    assert len(raw) <= 1300
    assert on["hits"] == [] and on["total_results"] == 0
    assert on["response_cap_truncated"] is True
    assert "adjacent_reply" not in on["provenance"]
    assert on["provenance"]["fts_anchor"]["delivered"] is False


def test_window_read_finished_past_deadline_is_discarded(ranked, monkeypatch):
    e, entries = ranked
    # The three clock reads are before connect, for its timeout, and after close.
    with monkeypatch.context() as patch:
        patch.setattr(tools.time, "monotonic", Mock(side_effect=[0.0, 0.0, 2.0]))
        assert tools._lcm_recall_snippet_windows(
            e, entries[:1], deadline=1.0, query="ANSWER_TOKEN", expanded_limit=1, fts_arm_index=0,
        ) == ({}, "deadline")
