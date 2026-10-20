"""#1188 session-pack probes. Hits controls are intentionally GREEN on base."""
import json
from unittest.mock import Mock

import pytest

from hermes_lcm import recall_delivery, tokens, tools
from hermes_lcm.config import LCMConfig
from tests import test_lcm_recall

recall_engine = test_lcm_recall.recall_engine


def add(engine, text, sid="one", role="user", **kw):
    if "observed_at" in kw:
        kw["timestamp"] = kw.pop("observed_at")
    return engine._store.append(sid, {"role": role, "content": text, **kw})


def recall(engine, **kw):
    return json.loads(tools.lcm_recall({"query": "needle", "limit": 1, **kw}, engine=engine))


@pytest.fixture
def session_engine(recall_engine):
    recall_engine._config.embeddings_enabled = False
    recall_engine._config.recall_delivery_unit = "session"
    return recall_engine


def test_config(monkeypatch):
    # RED: base has no field/choice validation.
    assert LCMConfig().recall_delivery_unit == "hits"
    monkeypatch.setenv("LCM_RECALL_DELIVERY_UNIT", "session")
    assert LCMConfig.from_env().recall_delivery_unit == "session"
    monkeypatch.setenv("LCM_RECALL_DELIVERY_UNIT", "bad")
    config = LCMConfig.from_env()
    assert config.recall_delivery_unit == "hits"
    assert any("LCM_RECALL_DELIVERY_UNIT" in w for w in config.config_source_warnings)


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_hits_green_control(recall_engine, monkeypatch, detail):
    # GREEN control: identical raw bytes and no new session/token work.
    engine = recall_engine
    engine._config.embeddings_enabled = False
    add(engine, "needle")
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    monkeypatch.setattr(engine._store, "load_session_window", Mock(side_effect=AssertionError("session read")))
    monkeypatch.setattr(tokens, "count_tokens", Mock(side_effect=AssertionError("token count")))
    args = {"query": "needle", "detail": detail, **({"seen_refs": []} if detail == "answer_ready" else {})}
    before = tools.lcm_recall(args, engine=engine)
    engine._config.recall_delivery_unit = "hits"
    assert tools.lcm_recall(args, engine=engine) == before


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_whole_session_rank_and_dates(session_engine, monkeypatch, detail):
    # RED: base returns one hit; no neighbors or session metadata.
    engine = session_engine
    first = add(engine, "Alice: before", observed_at=1600000000)
    anchor = add(engine, "needle", role="assistant")
    last = add(engine, "[image: a red kite]", role="user")
    add(engine, "needle", sid="two")
    original = tools._lcm_recall_rerank

    def rank(*a, **kw):
        ordered, status, scores = original(*a, **kw)
        ordered.sort(key=lambda e: e["hit"]["session_id"])
        for i, entry in enumerate(ordered):
            entry["_final_score"] = i + 1
        return ordered, status, scores

    monkeypatch.setattr(tools, "_lcm_recall_rerank", rank)
    result = recall(engine, detail=detail)
    assert [h["store_id"] for h in result["hits"]] == [first, anchor, last]
    assert result["hits"][0]["event_time"].endswith("Z")
    assert "event_time" not in result["hits"][1]
    assert result["hits"][0]["delivery_origin"] == "session_neighbor"
    assert result["hits"][0]["score"] == 0 and result["hits"][0]["arms"] == []
    assert result["provenance"]["delivery"]["sessions"][0]["complete"]
    for h in result["hits"]:
        row = engine._store.get(h["store_id"])
        assert row["content"][h["content_offset"]:h["content_returned_chars"]] == h["content"]


@pytest.mark.parametrize("encoder", [False, True])
def test_budget_late_anchor_and_multiple_windows(session_engine, monkeypatch, encoder):
    # RED: no late-hit neighborhoods, fixed whole-JSON budget or partial ranges.
    engine = session_engine
    monkeypatch.setattr(tokens, "_get_encoder", lambda: type("Encoder", (), {
        "encode": lambda self, s, **kw: list(range(len(s)))})() if encoder else None)
    original = recall_delivery.deliver_recall
    def giant_echo(ordered, **kw):
        kw["query"] = "needle " + "界" * 100000
        return original(ordered, **kw)
    monkeypatch.setattr(recall_delivery, "deliver_recall", giant_echo)
    ids = [add(engine, ("needle " if i in (60, 90) else "context ") + "界" * 600) for i in range(100)]
    result = recall(engine, limit=2)
    raw = json.dumps(result, ensure_ascii=False)
    assert len(raw) <= 64000
    assert result["provenance"]["delivery"]["response_tokens"] <= 16000
    out = [h["store_id"] for h in result["hits"]]
    assert ids[60] in out and ids[90] in out and ids[59] in out
    assert out == sorted(set(out)) and ids[0] not in out
    assert not result["provenance"]["delivery"]["sessions"][0]["complete"]


def test_oversized_tool_group_and_next_session(session_engine, monkeypatch):
    # RED: base splits matching tool hits and has no omission reasons.
    e = session_engine
    e._config.large_output_externalization_enabled = False
    call = {"id": "call", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    add(e, "needle", tool_calls=[call], role="assistant")
    add(e, "x" * 65000, role="tool", tool_call_id="call")
    other = add(e, "needle", sid="two")
    original = tools._lcm_recall_rerank
    def rank(*a, **kw):
        ordered, status, scores = original(*a, **kw)
        ordered.sort(key=lambda e: e["hit"]["store_id"])
        return ordered, status, scores
    monkeypatch.setattr(tools, "_lcm_recall_rerank", rank)
    result = recall(e, limit=2)
    assert [h["store_id"] for h in result["hits"]] == [other]
    assert any(s["omission_reason"] == "oversized_group" for s in result["provenance"]["delivery"]["sessions"])


def test_atomic_tool_delta_and_duplicates(session_engine):
    # RED: no atomic source groups / full-row delta after filtering.
    e = session_engine
    call = {"id": "c", "function": {"name": "read", "arguments": "{}"}}
    a = add(e, "needle", role="assistant", tool_calls=[call])
    b = add(e, "needle", role="tool", tool_call_id="c")
    c = add(e, "same")
    d = add(e, "needle same", sid="two")
    result = recall(e, limit=3, detail="answer_ready", seen_refs=[])
    assert [h["store_id"] for h in result["hits"]] == [a, b, c, d]
    assert result["hits"][0]["tool_calls"] == [call]
    assert result["hits"][1]["tool_call_id"] == "c"
    refs = result["delta"]["novel_refs"]
    assert refs == [h["exact_ref"] for h in result["hits"]]
    delta = recall(e, limit=3, detail="answer_ready", seen_refs=[refs[0]])
    assert not {a, b} & {h["store_id"] for h in delta["hits"]}
    assert not delta["provenance"]["delivery"]["sessions"][0]["complete"]


def test_scope_stale_markers_and_equal_text(session_engine, monkeypatch):
    # RED: source verification plus expansion must retain real repeated rows.
    e = session_engine
    a = add(e, "needle")
    b = add(e, "same")
    c = add(e, "same")
    add(e, "needle", sid="excluded")
    marker = add(e, "[externalized: keep marker]")
    result = recall(e, exclude_session_ids=["excluded"])
    assert [h["store_id"] for h in result["hits"]] == [a, b, c, marker]
    assert result["hits"][-1]["content"] == "[externalized: keep marker]"
    original = tools._lcm_recall_rerank

    def stale(*a, **kw):
        rows, status, scores = original(*a, **kw)
        for entry in rows:
            entry["hit"]["session_id"] = "foreign"
        return rows, status, scores

    monkeypatch.setattr(tools, "_lcm_recall_rerank", stale)
    assert recall(e)["hits"] == []


def test_deadline_and_read_failure(session_engine, monkeypatch):
    # RED: no completed-group-only partial delivery.
    e = session_engine
    add(e, "needle")
    add(e, "neighbor")
    original = e._store.load_session_window
    calls = []

    def fail(*a, **kw):
        calls.append(1)
        if len(calls) > 1:
            raise TimeoutError
        return original(*a, **kw)

    monkeypatch.setattr(e._store, "load_session_window", fail)
    result = recall(e)
    assert result["provenance"]["delivery"]["truncated"]
    assert result["timeout"]
    assert len(calls) == 2


def test_summary_lineage_and_internal_hits_override(session_engine):
    # RED: P metadata absent; internal proactive override stays hit-only.
    e = session_engine
    add(e, "needle")
    add(e, "neighbor")
    result = json.loads(tools.lcm_recall({"query": "needle"}, engine=e, delivery_unit="hits"))
    assert "delivery" not in result["provenance"]
    assert recall(e)["total_results"] == 2


def test_summary_seed_verified_lineage(session_engine, monkeypatch):
    # RED: snippets base returns a summary, not its source neighborhood.
    from hermes_lcm.dag import SummaryNode
    e = session_engine
    a = add(e, "needle source row")
    b = add(e, "neighbor")
    node = e._dag.add_node(SummaryNode(session_id="one", depth=0, summary="needle", source_ids=[0, a]))
    def ranked(*args, **kw):
        return [dict(hit=dict(kind="summary", session_id="one", node_id=node, snippet="needle"),
            ranks={0: 1}, _final_score=1)], "off", []
    monkeypatch.setattr(tools, "_lcm_recall_rerank", ranked)
    assert [h["store_id"] for h in recall(e)["hits"]] == [a, b]
    e._dag.source_message_ids = lambda *a, **kw: []
    result = recall(e)
    assert result["hits"] == []
    assert result["provenance"]["delivery"]["summary_leads"][0]["node_id"] == node


def test_same_numeric_ids_other_store_not_read(session_engine, tmp_path):
    # RED: a non-hit neighbor must come exclusively from the engine's store (I2).
    from hermes_lcm.store import MessageStore
    other = MessageStore(str(tmp_path / "other.db"))
    try:
        other.append("one", {"role": "user", "content": "foreign"})
        other.append("one", {"role": "user", "content": "foreign neighbor"})
        add(session_engine, "needle")
        add(session_engine, "local neighbor")
        before = session_engine._store.get_batch([1, 2])
        result = recall(session_engine)
        assert [h["content"] for h in result["hits"]] == ["needle", "local neighbor"]
        assert before == session_engine._store.get_batch([1, 2])
    finally:
        other.close()


@pytest.mark.parametrize("failure", ["clock", "error"])
def test_bounded_read_stop(session_engine, monkeypatch, failure):
    # RED: base neither expands neighbors nor publishes partial stop reasons.
    e = session_engine
    add(e, "needle")
    add(e, "neighbor")
    original = e._store.load_session_window
    clock = tools.time.monotonic
    expired = [False]
    calls = []
    def read(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            if failure == "error":
                raise ValueError("unreadable fixture")
            expired[0] = True
        return original(*a, **kw)
    monkeypatch.setattr(e._store, "load_session_window", read)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock() + (100 if expired[0] else 0))
    result = recall(e)
    assert result["provenance"]["delivery"]["stop_reason"] == ("deadline" if failure == "clock" else "read_error")
    assert len(calls) == 2
    assert [h["content"] for h in result["hits"]] == ["needle"]
