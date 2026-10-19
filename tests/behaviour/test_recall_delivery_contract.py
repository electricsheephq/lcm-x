"""Serialized delivery contracts captured on the ranking-only extraction."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import externalize, tools
from hermes_lcm.externalize import maybe_externalize_tool_output
from test_recall_ranking_contract import NOW, corpus as corpus, recall

GOLDEN = Path(__file__).with_name("recall_delivery_golden.json")


def check_golden(name, raw):
    assert raw == json.loads(GOLDEN.read_text())[name]


def set_arms(monkeypatch, *, fts=(), summary=(), chunk=(), leads=()):
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", lambda *a, **kw: (deepcopy(fts), None))
    monkeypatch.setattr(tools, "_lcm_recall_summary_arm", lambda *a, **kw: (
        deepcopy(summary), "full", len(summary), len(summary), deepcopy(leads),
    ))
    monkeypatch.setattr(tools, "_lcm_recall_chunk_arm", lambda *a, **kw: (
        deepcopy(chunk), "full", len(chunk), len(chunk),
    ))


def append_hit(corpus, text, index, *, session=None):
    engine = corpus[0]
    session = session or f"extra-{index}"
    sid = engine._store.append(session, {"role": "user", "content": text})
    return dict(kind="message_excerpt", store_id=sid, session_id=session,
                timestamp=NOW, snippet=text[:300], content_offset=0,
                chunk_span={"char_start": 0, "char_end": min(300, len(text))},
                expand_hint=f"lcm_expand(store_id={sid}, content_offset=0)")


@pytest.mark.parametrize("markers", [False, True])
def test_unicode_cap(corpus, monkeypatch, markers):
    hits = [append_hit(corpus, f"kanban {i} " + "記憶🌿" * 650, i) for i in range(3)]
    set_arms(monkeypatch, chunk=hits)
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 6000)
    raw = recall(corpus, detail="answer_ready", limit=3, seen_refs=[],
                 include_occurrence_time=markers, scope_bias=0)
    result = json.loads(raw)
    assert len(raw) <= 6000 < len(raw.encode())
    assert 0 < len(result["hits"]) < 3
    assert result["provenance"]["answer_ready"]["response_truncated"]
    assert all(("occurrence_time" in h) is markers for h in result["hits"])
    check_golden(f"unicode-cap-{markers}", raw)


@pytest.mark.parametrize("cap", [6000, 3000])
def test_fired_anchor_cap(corpus, monkeypatch, cap):
    _, hits, _, _, _ = corpus
    set_arms(monkeypatch, fts=hits[-1:], chunk=hits[:6])
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap)
    raw = recall(corpus, detail="answer_ready", limit=4, scope_bias=0)
    info = json.loads(raw)["provenance"]["fts_anchor"]
    assert info["fired"]
    assert info["delivered"] is (cap == 6000)
    assert len(raw) <= cap
    check_golden(f"anchor-cap-{cap}", raw)


def test_delta_refill_and_exhaustion(corpus, monkeypatch):
    hits = [append_hit(corpus, f"kanban delta {i}", i, session="delta-session") for i in range(3)]
    set_arms(monkeypatch, chunk=hits)
    seen = []
    for i in range(4):
        raw = recall(corpus, detail="answer_ready", limit=1, seen_refs=seen, scope_bias=0)
        result = json.loads(raw)
        assert result["delta"]["progress"] is (i < 3)
        if i < 3:
            assert result["hits"][0]["store_id"] == hits[i]["store_id"]
        else:
            assert result["delta"]["termination_reason"] == "no_novel_exact_ref"
        seen += result["delta"]["novel_refs"]
        check_golden(f"delta-wave-{i}", raw)


def test_strict_refill(corpus, monkeypatch):
    invalid = {**corpus[1][0], "store_id": 999, "snippet": "missing"}
    set_arms(monkeypatch, chunk=[invalid, *corpus[1][1:]])
    raw = recall(corpus, detail="answer_ready", limit=2, scope_bias=0)
    result = json.loads(raw)
    assert [h["store_id"] for h in result["hits"]] == [h["store_id"] for h in corpus[1][1:3]]
    assert result["provenance"]["answer_ready"]["unreferenced_dropped_count"] == 1
    check_golden("strict-refill", raw)


@pytest.mark.parametrize("strict", [False, True])
def test_hydration_limit(corpus, monkeypatch, strict):
    corpus[0]._config.recall_reference_strict = strict
    hits = [append_hit(corpus, f"kanban evidence {i}", i) for i in range(10)]
    set_arms(monkeypatch, chunk=hits)
    raw = recall(corpus, detail="answer_ready", limit=10, scope_bias=0)
    result = json.loads(raw)
    assert len(result["hits"]) == 10
    assert ["content" in h for h in result["hits"]] == [True] * 8 + [False] * 2
    check_golden(f"hydration-limit-{strict}", raw)


@pytest.mark.parametrize("mode", ["present", "cap", "read_error", "deadline"])
def test_event_time(corpus, monkeypatch, mode):
    engine, hits, _, _, _ = corpus
    engine._store.connection.execute(
        "UPDATE messages SET observed_at = ?, observed_at_source = ?",
        (NOW - 86400, "host_message_timestamp"),
    )
    engine._store.connection.commit()
    set_arms(monkeypatch, chunk=hits)
    if mode == "cap":
        # Item-only snippets cap: keep all seven hits, but strip optional fields
        # from the lowest ranked entries first.
        full = json.loads(recall(corpus, scope_bias=0))
        size = sum(len(json.dumps(h, ensure_ascii=False)) for h in full["hits"])
        monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", size - 1)
    if mode in {"read_error", "deadline"}:
        real_read = tools._lcm_recall_event_time_rows

        def read(engine, ids, *, deadline):
            # Inject at the private SQLite connection, not at the result helper.
            def fail_connect(*a, **kw):
                if mode == "deadline":
                    patch.setattr(tools, "time", SimpleNamespace(monotonic=lambda: deadline + 1))
                raise sqlite3.OperationalError("synthetic read failure")

            with monkeypatch.context() as patch:
                patch.setattr(tools, "sqlite3", SimpleNamespace(
                    connect=fail_connect, Error=sqlite3.Error,
                ))
                return real_read(engine, ids, deadline=deadline)

        monkeypatch.setattr(tools, "_lcm_recall_event_time_rows", read)
    raw = recall(corpus, scope_bias=0)
    result = json.loads(raw)
    present = ["event_time" in h for h in result["hits"]]
    assert len(present) == 7
    if mode == "present":
        assert all(present)
        assert "event_time_unavailable" not in result
    elif mode == "cap":
        assert present == [True] * 6 + [False]
        assert result["event_time_unavailable_reason"] == "response_cap"
    else:
        assert not any(present)
        assert result["event_time_unavailable_reason"] == mode
    check_golden(f"event-time-{mode}", raw)


def test_answer_ready_event_time_cap(corpus, monkeypatch):
    engine = corpus[0]
    engine._store.connection.execute(
        "UPDATE messages SET observed_at = ?, observed_at_source = ?",
        (NOW - 86400, "host_message_timestamp"),
    )
    engine._store.connection.commit()
    set_arms(monkeypatch, chunk=corpus[1])
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 9000)
    baseline = recall(corpus, detail="answer_ready", scope_bias=0)
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", len(baseline) - 10)
    raw = recall(corpus, detail="answer_ready", scope_bias=0)
    result = json.loads(raw)
    fields = ["event_time" in h for h in result["hits"]]
    assert len(fields) == 7 and any(fields) and not all(fields)
    assert fields == sorted(fields, reverse=True)
    assert result["event_time_unavailable_reason"] == "response_cap"
    assert len(raw) <= len(baseline) - 10
    check_golden("answer-ready-event-time-cap", raw)


def test_summary_lead_cap(corpus, monkeypatch):
    leads = [dict(node_id=i, session_id="s" * 2000, expand_hint="x" * 2000) for i in range(3)]
    set_arms(monkeypatch, leads=leads)
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 6000)
    raw = recall(corpus, detail="answer_ready", include="summaries", seen_refs=[])
    result = json.loads(raw)
    assert len(raw) <= 6000
    assert result["hits"] == []
    assert len(result["provenance"]["answer_ready"]["summary_leads"]) < 3
    check_golden("summary-lead-cap", raw)


def test_externalized_payload_marker(corpus, monkeypatch):
    engine = corpus[0]
    monkeypatch.setattr(externalize, "time", SimpleNamespace(
        time=lambda: NOW, time_ns=lambda: int(NOW * 1_000_000_000),
        gmtime=lambda: time.gmtime(NOW), strftime=time.strftime,
    ))
    engine._config.large_output_externalization_threshold_chars = 100
    engine._config.large_output_externalization_enabled = True
    payload = maybe_externalize_tool_output(
        "kanban evidence " * 1000, tool_call_id="contract-call", session_id="session-0",
        config=engine._config, hermes_home=engine._hermes_home,
    )
    assert payload is not None
    hit = append_hit(corpus, payload["placeholder"], 0)
    set_arms(monkeypatch, chunk=[hit])
    raw = recall(corpus, detail="answer_ready", limit=1)
    result = json.loads(raw)
    assert result["hits"][0]["content"] == payload["placeholder"]
    check_golden("externalized-marker", raw)


@pytest.mark.parametrize("case", ["chunk-loses", "summary-kind", "recency", "default-weight", "tie"])
def test_ranking_gaps(corpus, monkeypatch, case):
    engine, hits, _, _, _ = corpus
    fts, summary, chunk = deepcopy(hits[:2]), [], []
    engine._config.recall_arm_weights = None
    if case == "chunk-loses":
        chunk = [deepcopy(hits[1]), {**hits[0], "snippet": "trailing", "content_offset": 37,
                                   "chunk_span": {"char_start": 37, "char_end": 45}}]
    elif case == "summary-kind":
        summary = [dict(kind="summary", node_id=99, session_id="summary-session",
                        timestamp=NOW, snippet="kanban summary", expand_hint="lcm_expand(node_id=99)"),
                   {**hits[0], "snippet": "alternate summary source"}]
    elif case == "recency":
        fts[0]["timestamp"] = NOW - 365 * 86400
        fts[1]["timestamp"] = "invalid"
    elif case == "tie":
        fts[1]["timestamp"] = NOW
        summary = list(reversed(deepcopy(fts)))
    set_arms(monkeypatch, fts=fts, summary=summary, chunk=chunk)
    raw = recall(corpus, scope_bias=0)
    result = json.loads(raw)
    if case == "chunk-loses":
        assert result["hits"][0]["snippet"] == hits[0]["snippet"]
        assert result["hits"][0]["chunk_span"] == hits[0]["chunk_span"]
    elif case == "summary-kind":
        assert any(h["kind"] == "summary" for h in result["hits"])
    elif case == "recency":
        assert [h["score"] for h in result["hits"]] == [round(0.5 / 61, 6), round(0.5 / 62, 6)]
    elif case == "default-weight":
        assert result["provenance"]["arm_weights"] == {"fts": 1.0}
    elif case == "tie":
        assert result["hits"][0]["score"] == result["hits"][1]["score"]
        assert result["provenance"]["rerank"] == "disabled"
    check_golden(case, raw)


@pytest.mark.parametrize("case", ["short", "past-slots", "fallback-cut"])
def test_anchor_window_gaps(corpus, monkeypatch, case):
    hits = [append_hit(corpus, f"kanban anchor window {i}", i,
                       session="same-session" if case == "fallback-cut" else None)
            for i in range(14)]
    set_arms(monkeypatch, fts=hits[-1:], chunk=hits[:-1])
    limit = 3 if case == "short" else 12 if case == "past-slots" else 4
    raw = recall(corpus, limit=limit, scope_bias=0)
    info = json.loads(raw)["provenance"]["fts_anchor"]
    assert info["fired"] is (case != "short")
    assert info["position"] == {"short": None, "past-slots": 10, "fallback-cut": 4}[case]
    check_golden(f"anchor-{case}", raw)


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("deficit", [0, 1])
def test_event_time_item_cap_exact_edge(corpus, monkeypatch, count, deficit):
    engine, hits, _, _, _ = corpus
    engine._store.connection.execute(
        "UPDATE messages SET observed_at = ?, observed_at_source = ?",
        (NOW - 86400, "host_message_timestamp"),
    )
    engine._store.connection.commit()
    chosen = hits[:count]
    # The first hit arrives through two arms but occupies one delivery slot.
    set_arms(monkeypatch, fts=chosen[:1], chunk=chosen)
    baseline = json.loads(recall(corpus, scope_bias=0))
    assert len(baseline["hits"]) == count
    assert all("event_time" in hit for hit in baseline["hits"])
    if count:
        assert baseline["hits"][0]["arms"] == ["chunk", "fts"]
    cap = sum(len(json.dumps(hit, ensure_ascii=False)) for hit in baseline["hits"])
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", max(1, cap - deficit))
    calls, fetched, snapshots = [], {}, {}
    real_read = tools._lcm_recall_event_time_rows
    real_fields = tools._lcm_recall_event_time

    def read(engine, ids, *, deadline):
        calls.append(("read", list(ids)))
        rows, reason = real_read(engine, ids, deadline=deadline)
        fetched.update(rows)
        snapshots.update(deepcopy(rows))
        return rows, reason

    def fields(row):
        store_id = next(key for key, value in fetched.items() if value is row)
        calls.append(("fields", store_id))
        return real_fields(row)

    monkeypatch.setattr(tools, "_lcm_recall_event_time_rows", read)
    monkeypatch.setattr(tools, "_lcm_recall_event_time", fields)
    raw = recall(corpus, scope_bias=0)
    expected = deepcopy(baseline)
    if count and deficit:
        expected["hits"][-1].pop("event_time")
        expected["hits"][-1].pop("event_time_source")
        expected["event_time_unavailable"] = True
        expected["event_time_unavailable_reason"] = "response_cap"
    assert raw == json.dumps(expected)
    ids = [hit["store_id"] for hit in baseline["hits"]]
    assert calls == ([("read", ids)] + [("fields", sid) for sid in ids] if ids else [])
    # Delivery edits its own hit dicts, not the source rows supplied by the read.
    assert fetched == snapshots


def test_event_time_item_cap_bare_oversized_hit(corpus, monkeypatch):
    engine, hits, _, _, _ = corpus
    engine._store.connection.execute(
        "UPDATE messages SET observed_at = NULL, observed_at_source = NULL"
    )
    engine._store.connection.commit()
    set_arms(monkeypatch, chunk=hits[:1])
    baseline = recall(corpus, scope_bias=0)
    assert len(json.loads(baseline)["hits"]) == 1
    assert "event_time" not in json.loads(baseline)["hits"][0]
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", 1)
    # The first hit remains deliverable; stripping zero optional characters
    # must not invent an unavailable marker or drop the hit.
    assert recall(corpus, scope_bias=0) == baseline


def test_event_time_item_cap_skips_bare_tail(corpus, monkeypatch):
    engine, hits, _, _, _ = corpus
    engine._store.connection.execute(
        "UPDATE messages SET observed_at = ?, observed_at_source = ? WHERE store_id IN (?, ?)",
        (NOW - 86400, "host_message_timestamp", hits[0]["store_id"], hits[1]["store_id"]),
    )
    engine._store.connection.commit()
    set_arms(monkeypatch, chunk=hits[:3])
    expected = json.loads(recall(corpus, scope_bias=0))
    assert ["event_time" in hit for hit in expected["hits"]] == [True, True, False]
    cap = sum(len(json.dumps(hit, ensure_ascii=False)) for hit in expected["hits"])
    monkeypatch.setattr(tools, "_LCM_RECALL_RESPONSE_CHAR_CAP", cap - 1)
    expected["hits"][1].pop("event_time")
    expected["hits"][1].pop("event_time_source")
    expected["event_time_unavailable"] = True
    expected["event_time_unavailable_reason"] = "response_cap"
    assert recall(corpus, scope_bias=0) == json.dumps(expected)
