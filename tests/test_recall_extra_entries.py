"""Extra entries preserve full hits and use only bounded private exact reads."""
from functools import partial
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_lcm import config, recall_delivery, schemas, tools
from hermes_lcm.engine import LCMEngine
from tests import test_recall_snippet_windows as windows

make_delivery = windows.make_delivery
recall_engine = windows.recall_engine
ranked = windows.ranked

GOLDEN = Path(__file__).parent / "fixtures/recall_extra_entries_golden.json"


def shaped(name):
    limit = 25 if name == "locomo" else 20
    d = make_delivery((1300 if name == "lme" else 80,) * 55)
    for i, entry in enumerate(d.entries):
        if name != "lme":
            d.rows[i + 1]["content"] = "ANSWER_TOKEN turn " + str(i)
            entry["hit"]["snippet"] = d.rows[i + 1]["content"]
        if name == "locomo":
            entry["hit"]["session_id"] = "one-store"
            if i % 7 == 0:
                entry["hit"].update(kind="summary", node_id=i + 1, store_id=None)
        if name == "pmv2":
            entry["hit"]["session_id"] = f"batch-{i // 20}"
            if i < limit:
                entry["hit"]["duplicate_store_ids"] = [i + 21]
    return d, limit


def real_payload(e, monkeypatch, **kwargs):
    e._config.embeddings_enabled = False
    if not e._store.get_batch([1]):
        for i in range(4):
            e._store.append(f"s{i}", {"role": "assistant", "content": "ANSWER_TOKEN " + str(i)})
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    original = tools._lcm_recall_fts_arm

    def fixed(*args, **kw):
        hits, error = original(*args, **kw)
        for hit in hits:
            hit["timestamp"] = 100
        return hits, error

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", fixed)
    return tools.lcm_recall({"query": "ANSWER_TOKEN", "limit": 2}, engine=e, **kwargs)


def extra_reader(d):
    engine = SimpleNamespace(_store=SimpleNamespace(get_batch=d.batch))

    def read(entries, deadline, **kw):
        messages = [e for e in entries if e["hit"]["kind"] == "message_excerpt"]
        content = tools._lcm_recall_answer_ready_content(engine, messages, **kw)
        for entry in entries:
            hit = entry["hit"]
            content.setdefault(tools._hit_identity(hit), tools._lcm_recall_content_window(
                hit["snippet"], match_start=0, match_end=len(hit["snippet"]), char_cap=300,
            ))
        return content, None

    return Mock(side_effect=read)


@pytest.mark.parametrize("name", ["lme", "locomo", "pmv2"])
def test_zero_golden(name):
    d, limit = shaped(name)
    reader = Mock(side_effect=AssertionError("default read"))
    import inspect
    options = {"extra_entries": 0, "read_extra_entries": reader} if "extra_entries" in inspect.signature(recall_delivery.deliver_recall).parameters else {}
    assert d.invoke(limit=limit, **options) == json.loads(GOLDEN.read_text())[name]
    reader.assert_not_called()


def test_real_zero_golden(recall_engine, monkeypatch):
    recall_engine._config.recall_extra_entries = 0
    reader = Mock(side_effect=AssertionError("default read"))
    monkeypatch.setattr(tools, "_lcm_recall_snippet_windows", reader)
    assert real_payload(recall_engine, monkeypatch, extra_entries=0) == json.loads(GOLDEN.read_text())["real"]
    assert real_payload(recall_engine, monkeypatch) == json.loads(GOLDEN.read_text())["real"]
    reader.assert_not_called()


@pytest.mark.parametrize("name", ["lme", "locomo", "pmv2"])
@pytest.mark.parametrize("order", ["score", "time"])
@pytest.mark.parametrize("cap", [5, 25])
def test_prefix_envelope_ranks_and_six_keys(name, order, cap):
    d, limit = shaped(name)
    off = json.loads(d.invoke(limit=limit, delivery_order=order))
    reader = extra_reader(d)
    on = json.loads(d.invoke(limit=limit, delivery_order=order, extra_entries=cap, read_extra_entries=reader))
    extras = on["hits"][limit:]
    assert extras or name == "pmv2"
    expected = [] if name == "pmv2" else [e["hit"]["store_id"] for e in d.entries[limit:limit + min(cap, limit // 2)]]
    assert [h["store_id"] for h in extras] == expected
    for hit in extras:
        assert list(hit) == ["kind", "store_id", "session_id", "event_time", "window", "extra"]
        assert hit["extra"] is True and hit["event_time"] is None
    on["hits"] = on["hits"][:limit]
    assert on == off


def test_duplicate_skip_without_refill():
    d = make_delivery((1300,) * 16)
    d.entries[0]["hit"]["duplicate_store_ids"] = [10]
    d.entries[8]["hit"]["store_id"] = 1
    d.entries[11]["hit"]["store_id"] = 11
    on = json.loads(d.invoke(limit=8, extra_entries=4, read_extra_entries=extra_reader(d)))
    assert [h["store_id"] for h in on["hits"][8:]] == [11]


@pytest.mark.parametrize("length,span", [(80, (0, 12)), (5000, (3800, 3812))])
def test_window_uses_hit_span_and_not_alternate(length, span):
    d = make_delivery((80, 80, length))
    if length == 80:
        d.rows[3]["content"] = "short ANSWER_TOKEN"
    d.entries[2]["hit"]["chunk_span"] = dict(zip(("char_start", "char_end"), span))
    d.entries[2]["_alternates"] = [{"chunk_span": {"char_start": 0, "char_end": 12}}]
    extra, = json.loads(d.invoke(limit=2, extra_entries=1, read_extra_entries=extra_reader(d)))["hits"][2:]
    assert extra["window"] == tools._lcm_recall_content_window(
        d.rows[3]["content"], match_start=span[0], match_end=span[1], char_cap=300,
    )["content"]


@pytest.mark.parametrize("missing", [False, True])
def test_summary_and_missing_row_fallback_without_dag(recall_engine, missing, monkeypatch):
    d = make_delivery((80,) * 3)
    hit = d.entries[2]["hit"]
    hit["snippet"] = "証" * 800
    if not missing:
        hit.update(kind="summary", store_id=None, node_id=999)
        monkeypatch.setattr(sqlite3, "connect", Mock(side_effect=AssertionError("summary read")))
    reader = partial(tools._lcm_recall_snippet_windows, recall_engine, extra_entries=True)
    recall_engine._dag = Mock()
    extra, = json.loads(d.invoke(limit=2, extra_entries=1, read_extra_entries=reader))["hits"][2:]
    assert extra["window"] == tools._lcm_recall_content_window(
        hit["snippet"], match_start=0, match_end=len(hit["snippet"]), char_cap=300,
    )["content"]
    assert extra["event_time"] is None
    recall_engine._dag.get_node.assert_not_called()


@pytest.mark.parametrize("mode", ["expired", "late", "error", "missing", "callable_error"])
def test_read_failure_preserves_response(recall_engine, monkeypatch, caplog, mode):
    d = make_delivery((1300,) * 4)
    off = json.loads(d.invoke(limit=2))
    reader = partial(tools._lcm_recall_snippet_windows, recall_engine, extra_entries=True)
    if mode == "missing":
        reader = None
    elif mode == "callable_error":
        reader = Mock(side_effect=RuntimeError("private detail"))
    elif mode == "expired":
        monkeypatch.setattr(tools.time, "monotonic", lambda: 2)
        connect = Mock(side_effect=AssertionError("expired connect"))
        monkeypatch.setattr(tools.sqlite3, "connect", connect)
    elif mode == "late":
        monkeypatch.setattr(tools.time, "monotonic", Mock(side_effect=lambda values=iter([0, 0, 2]): next(values, 2)))
    else:
        monkeypatch.setattr(tools.sqlite3, "connect", Mock(side_effect=sqlite3.OperationalError("private detail")))
    on = json.loads(d.invoke(limit=2, extra_entries=2, read_extra_entries=reader, deadline=1 if mode in {"expired", "late"} else float("inf")))
    assert on.pop("extra_entries_skipped") == ("deadline" if mode in {"expired", "late"} else "read_error")
    assert on == off
    monkeypatch.undo()
    assert "private detail" not in caplog.text
    if mode in {"error", "callable_error"}:
        assert any(("OperationalError" if mode == "error" else "RuntimeError") in r.message for r in caplog.records)
    if mode == "expired":
        connect.assert_not_called()


@pytest.mark.parametrize("limit", [20, 25])
def test_kwarg_clamps_and_override_config(recall_engine, monkeypatch, limit):
    e = recall_engine
    d = make_delivery((80,) * 55)
    e._config.embeddings_enabled = False
    e._config.recall_extra_entries = 2
    monkeypatch.setattr(tools, "_lcm_recall_rerank", lambda *a, **kw: (d.entries, "disabled", []))
    for entry in d.entries:
        entry["rrf_score"] = entry["_final_score"]
        e._store.append(entry["hit"]["session_id"], {"role": "assistant", "content": "ANSWER_TOKEN"})
    args = {"query": "ANSWER_TOKEN", "limit": limit}
    off = json.loads(tools.lcm_recall(args, engine=e, extra_entries=0))
    assert len(off["hits"]) == limit
    assert len(json.loads(tools.lcm_recall(args, engine=e))["hits"]) == limit + 2
    on = json.loads(tools.lcm_recall(args, engine=e, extra_entries=40))
    assert len(on["hits"]) == limit + limit // 2
    e._disabled_tool_names = lambda: set()
    assert json.loads(LCMEngine.handle_tool_call(e, "lcm_recall", args, extra_entries=0)) == off


def test_cap_drops_only_extras_and_no_read_after_full_hit_truncation():
    d = make_delivery((1300,) * 12)
    off = d.invoke(limit=6)
    cap = len(off) + 490
    on = json.loads(d.invoke(limit=6, extra_entries=5, read_extra_entries=extra_reader(d), response_char_cap=cap))
    assert 6 < len(on["hits"]) < 9
    assert len(json.dumps(on)) <= cap
    on["hits"] = on["hits"][:6]
    assert on == json.loads(off)
    for kw in ({"response_char_cap": 600}, {"response_char_cap": 1600, "snippet_window_hits": 2}):
        reader = Mock(side_effect=AssertionError("truncated"))
        base = d.invoke(limit=2, **kw)
        assert d.invoke(limit=2, extra_entries=3, read_extra_entries=reader, **kw) == base
        reader.assert_not_called()
    reader = Mock(side_effect=AssertionError("exhausted"))
    assert d.invoke(extra_entries=3, read_extra_entries=reader) == d.invoke()
    reader.assert_not_called()


@pytest.mark.parametrize("delivery", [{"detail": "answer_ready"}, {"session_delivery": lambda entries, response: json.dumps(response)}])
def test_other_delivery_ignores_setting(delivery):
    d = make_delivery()
    reader = Mock(side_effect=AssertionError("other delivery"))
    assert d.invoke(extra_entries=3, read_extra_entries=reader, **delivery) == d.invoke(**delivery)
    reader.assert_not_called()


@pytest.mark.parametrize("consumer", ["pack", "proposal", "auto"])
def test_evidence_callers_pin_off(ranked, monkeypatch, consumer):
    from hermes_lcm import evidence_pack, evidence_compiler
    e, _ = ranked
    e._config.recall_extra_entries = 3
    args = {"query": "ANSWER_TOKEN", "limit": 2}
    original = tools.lcm_recall
    calls = []

    def capture(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)

    def consume(*a, retrieve, **kw):
        return {"context": retrieve(args)}

    monkeypatch.setattr(tools, "lcm_recall", capture)
    monkeypatch.setattr(evidence_pack, "build_evidence_pack", consume)
    monkeypatch.setattr(evidence_compiler, "compile_evidence", consume)
    monkeypatch.setattr(evidence_compiler, "compile_preanswer_evidence", consume)
    result = (tools.lcm_evidence_pack({}, engine=e) if consumer == "pack" else
              json.loads(tools.lcm_compile_evidence({"mode": consumer}, engine=e)))
    assert calls and all(c["extra_entries"] == 0 for c in calls)
    assert len(json.loads(result["context"])["hits"]) == 2
    assert len(json.loads(original(args, engine=e))["hits"]) == 3


def test_proactive_pins_off(ranked, monkeypatch):
    e, _ = ranked
    e._config.proactive_recall_min_score = 0
    e._config.recall_extra_entries = 3
    e._proactive_recall_eligible = lambda: True
    e._newest_user_message_text = lambda _: "ANSWER_TOKEN"
    e._proactive_recall_privacy_warned = e._proactive_recall_trust_warned = False
    e._wrap_relevant_memories = LCMEngine._wrap_relevant_memories
    e._proactive_recall_injected_count = e._proactive_recall_skipped_count = 0
    original = tools.lcm_recall
    calls = []

    def capture(*a, **kw):
        calls.append(kw)
        assert kw["extra_entries"] == 0
        return original(*a, **kw)

    monkeypatch.setattr(tools, "lcm_recall", capture)
    before = LCMEngine._build_proactive_recall_message(e, [], "system", set())
    assert before is not None
    e._config.recall_extra_entries = 0
    assert LCMEngine._build_proactive_recall_message(e, [], "system", set()) == before
    assert len(calls) == 2


def test_store_reader_defaults_and_readonly_extras(recall_engine, monkeypatch, tmp_path):
    from hermes_lcm.mcp_reader import StoreReader
    from hermes_lcm.db_bootstrap import ensure_store_uuid
    e = recall_engine
    for i in range(5):
        e._store.append(f"s{i}", {"role": "assistant", "content": f"ANSWER_TOKEN {i}", "timestamp": 1700000000 + i})
    ensure_store_uuid(e._store._conn)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LCM_EMBEDDINGS_ENABLED", "false")
    monkeypatch.delenv("LCM_RECALL_EXTRA_ENTRIES", raising=False)
    reader = StoreReader(e._store.db_path, tmp_path, config=config.LCMConfig.from_env())
    args = {"query": "ANSWER_TOKEN", "limit": 2}
    off, error = reader.call("lcm_recall", args)
    assert not error
    assert "extra_entries_skipped" not in off and '"extra": true' not in off
    monkeypatch.setenv("LCM_RECALL_EXTRA_ENTRIES", "2")
    reader = StoreReader(e._store.db_path, tmp_path, config=config.LCMConfig.from_env())
    original = sqlite3.connect
    connections, statements = [], []

    def connect(path, **kw):
        connections.append((str(path), kw))
        conn = original(path, **kw)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    raw, error = reader.call("lcm_recall", args)
    assert not error
    extras = json.loads(raw)["hits"][2:]
    assert len(extras) == 1 and all(h["event_time"] for h in extras)
    assert all("mode=ro" in path and kw.get("uri") for path, kw in connections)
    assert any("observed_at, observed_at_source" in sql and "content" in sql for sql in statements)
    assert any("PRAGMA query_only=ON" == sql for sql in statements)


@pytest.mark.parametrize("yaml", [False, True])
@pytest.mark.parametrize("raw,expected", [(0, 0), (5, 5), (40, 25), (-1, 0), (True, 0), (False, 0), (1.5, 0), ("bad", 0), ("5", 0)])
def test_config_parsing_and_provenance(monkeypatch, tmp_path, yaml, raw, expected):
    if yaml:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("lcm:\n  recall_extra_entries: " + json.dumps(raw))
    else:
        monkeypatch.setenv("LCM_RECALL_EXTRA_ENTRIES", str(raw))
        expected = 5 if raw == "5" else expected
    cfg = config.LCMConfig.from_env()
    assert cfg.recall_extra_entries == expected
    invalid = isinstance(raw, (bool, float)) or raw == "bad" or raw == -1 or (yaml and raw == "5")
    assert bool([w for w in cfg.config_source_warnings if "recall_extra_entries" in w]) == invalid
    source = "default" if invalid else ("config_yaml:lcm.recall_extra_entries" if yaml else "env:LCM_RECALL_EXTRA_ENTRIES")
    assert cfg.config_sources["recall_extra_entries"] == source
    monkeypatch.setenv("LCM_RECALL_EXTRA_ENTRIES", "2")
    assert config.LCMConfig.from_env().recall_extra_entries == 2


@pytest.mark.parametrize("order", ["score", "time"])
def test_existing_receipts_windows_and_event_time_are_unchanged(order):
    d = make_delivery((1300,) * 8)
    times = {i: {"observed_at": 1700000000 - i, "observed_at_source": "host_message_timestamp"} for i in range(1, 9)}
    d.read_time.return_value = (times, None)
    d.entries[1]["hit"]["adjacent_to"] = 1
    options = dict(limit=3, delivery_order=order, snippet_window_hits=1,
                   adjacent_reply=[{"store_id": 2, "after_store_id": 1}],
                   arm_order=["chunk", "fts"], arm_weights=[1.0, 1.0],
                   arm_hits={"fts": [d.entries[0]["hit"]]})
    off = json.loads(d.invoke(**options, fts_anchor_info={"fired": True, "delivered": None}))
    on = json.loads(d.invoke(**options, fts_anchor_info={"fired": True, "delivered": None},
                             extra_entries=3, read_extra_entries=extra_reader(d)))
    assert len(on["hits"]) == 4
    del on["hits"][3:]
    assert on == off


def test_fts_and_query_window_span_precedence():
    d = make_delivery((1300,) * 3)
    content = d.rows[3]["content"]
    d.entries[2]["hit"] = windows.fts_hit(content, store_id=3)
    for fts_index in (0, None):
        on = json.loads(d.invoke(limit=2, extra_entries=1, fts_arm_index=fts_index,
                                 read_extra_entries=extra_reader(d)))
        extra, = on["hits"][2:]
        assert extra["window"] == tools._lcm_recall_content_window(
            content, match_start=800, match_end=812, char_cap=300,
        )["content"]


def test_private_read_is_one_batch_and_uses_its_own_event_time(recall_engine, monkeypatch):
    e = recall_engine
    d = make_delivery((1300,) * 9)
    for i in range(1, 10):
        e._store.append(f"s{i}", {"role": "assistant", "content": d.rows[i]["content"], "timestamp": 1700000000 + i})
    monkeypatch.setattr(e._store, "get_batch", Mock(side_effect=AssertionError("shared read")))
    original = sqlite3.connect
    statements = []

    def connect(*a, **kw):
        conn = original(*a, **kw)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    on = json.loads(d.invoke(limit=6, extra_entries=3, read_extra_entries=partial(
        tools._lcm_recall_snippet_windows, e, extra_entries=True)))
    assert len([sql for sql in statements if sql.startswith("SELECT")]) == 1
    assert [h["event_time"] for h in on["hits"][6:]] == [
        tools._lcm_recall_event_time({"observed_at": 1700000000 + i})["event_time"] for i in range(7, 10)
    ]


def test_null_summary_ids_do_not_match_top_hits_or_each_other():
    d = make_delivery((80,) * 6)
    for i, entry in enumerate(d.entries):
        entry["hit"].update(kind="summary", node_id=i + 1, store_id=None)
    on = json.loads(d.invoke(limit=4, extra_entries=3, read_extra_entries=extra_reader(d)))
    assert len(on["hits"]) == 6
    assert [h["store_id"] for h in on["hits"][4:]] == [None, None]
    assert [h["window"] for h in on["hits"][4:]] == [e["hit"]["snippet"] for e in d.entries[4:]]


@pytest.mark.parametrize("limit,cap,count", [(20, 25, 10), (25, 25, 12), (1, 25, 0), (20, 5, 5)])
def test_effective_cap_math(limit, cap, count):
    d = make_delivery((80,) * 55)
    reader = extra_reader(d)
    on = json.loads(d.invoke(limit=limit, extra_entries=cap, read_extra_entries=reader))
    assert len(on["hits"][limit:]) == count
    if count:
        assert reader.call_args.args[0] == d.entries[limit:limit + count]
        reader.assert_called_once()
    else:
        reader.assert_not_called()


def test_recall_schema_matches_origin_main():
    encoded = json.dumps(schemas.LCM_RECALL, sort_keys=True)
    # Captured by executing the unchanged origin/main:schemas.py module.
    assert hashlib.sha256(encoded.encode()).hexdigest() == "2489582d63ae6365e7814027170a946f02f72d2485bcc38584ad442a00cf6013"
    assert '"extra_entries"' not in encoded
    assert '"recall_extra_entries"' not in encoded


@pytest.mark.parametrize("limit", [20, 25])
def test_setting_keeps_candidate_pool_and_rerank_window(recall_engine, monkeypatch, limit):
    e = recall_engine
    e._config.embeddings_enabled = False
    e._config.rerank_enabled = False
    for i in range(55):
        e._store.append(f"s{i}", {"role": "assistant", "content": f"ANSWER_TOKEN turn {i}"})
    arm = Mock(wraps=tools._lcm_recall_fts_arm)
    rerank = Mock(wraps=tools._lcm_recall_rerank)
    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", arm)
    monkeypatch.setattr(tools, "_lcm_recall_rerank", rerank)
    for cap in (0, 25):
        e._config.recall_extra_entries = cap
        payload = json.loads(tools.lcm_recall({"query": "ANSWER_TOKEN", "limit": limit}, engine=e))
        assert len(payload["hits"]) == limit + min(cap, limit // 2)
    assert [call.kwargs["candidate_limit"] for call in arm.call_args_list] == [limit * 4] * 2
    assert [call.kwargs["window"] for call in rerank.call_args_list] == [50, 50]


def test_short_pool_stops_without_more_reads():
    d = make_delivery((80,) * 24)
    off = json.loads(d.invoke(limit=20))
    d.read_time.reset_mock()
    reader = extra_reader(d)
    on = json.loads(d.invoke(limit=20, extra_entries=25, read_extra_entries=reader))
    assert [h["store_id"] for h in on["hits"][20:]] == [21, 22, 23, 24]
    reader.assert_called_once_with(d.entries[20:], query="ANSWER_TOKEN", expanded_limit=4,
                                   fts_arm_index=0, deadline=float("inf"), char_cap=300)
    d.batch.assert_called_once_with([21, 22, 23, 24])
    d.read_time.assert_called_once_with(list(range(1, 21)), deadline=float("inf"))
    d.hydrate.assert_not_called()
    d.read_windows.assert_not_called()
    del on["hits"][20:]
    assert on == off
