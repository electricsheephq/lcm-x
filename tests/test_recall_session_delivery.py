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
@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_budget_late_anchor_and_multiple_windows(session_engine, monkeypatch, encoder, detail):
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
    result = recall(engine, limit=2, detail=detail)
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


def test_atomic_tool_delta_and_duplicates(session_engine, monkeypatch):
    # RED: no atomic source groups / full-row delta after filtering.
    e = session_engine
    call = {"id": "c", "function": {"name": "read", "arguments": "{}"}}
    a = add(e, "needle", role="assistant", tool_calls=[call])
    b = add(e, "needle", role="tool", tool_call_id="c")
    c = add(e, "same")
    d = add(e, "needle same", sid="two")
    equal = add(e, "same", sid="two")
    original = tools._lcm_recall_rerank
    def chunks(*args, **kw):
        ordered, status, scores = original(*args, **kw)
        duplicate = dict(ordered[0], hit={**ordered[0]["hit"], "chunk_span": [0, 6]})
        return [ordered[0], duplicate, *ordered[1:]], status, scores
    monkeypatch.setattr(tools, "_lcm_recall_rerank", chunks)
    result = recall(e, limit=3, detail="answer_ready", seen_refs=[])
    assert [h["store_id"] for h in result["hits"]] == [a, b, c, d, equal]
    assert result["hits"][2]["content"] == result["hits"][-1]["content"] == "same"
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


@pytest.mark.parametrize("calls", [["bad"], {"id": "x"}, "bad", [{"id": []}]])
def test_fix1_malformed_group(session_engine, calls):
    add(session_engine, "needle")
    add(session_engine, "answer", role="assistant", tool_calls=calls)
    result = recall(session_engine)
    meta = result["provenance"]["delivery"]["sessions"][0]
    assert meta["omission_reason"] == "unreadable_group" and meta["expand_hint"]


@pytest.mark.parametrize("key", ["id", "tool_call_id"])
def test_fix1_ids_and_empty_refs(session_engine, key):
    a = add(session_engine, "", role="assistant", tool_calls=[{key: " 7 "}])
    b = add(session_engine, "needle", role="tool", tool_call_id=" 7 ")
    result = recall(session_engine, detail="answer_ready", seen_refs=[])
    assert [h["store_id"] for h in result["hits"]] == [a, b]
    assert "exact_ref" not in result["hits"][0]
    assert result["delta"]["novel_refs"] == [f"lcm:{b}:0-6"]


@pytest.mark.parametrize("failure", ["deadline", "read_error"])
def test_fix1_preserve_omission(session_engine, monkeypatch, failure):
    add(session_engine, "needle orphan", role="tool", tool_call_id="orphan")
    add(session_engine, "needle two", sid="two")
    original = session_engine._store.load_session_window
    rank = tools._lcm_recall_rerank
    def ranked(*a, **kw):
        ordered, status, scores = rank(*a, **kw)
        return sorted(ordered, key=lambda e: e["hit"]["store_id"]), status, scores
    monkeypatch.setattr(tools, "_lcm_recall_rerank", ranked)
    def read(sid, **kw):
        if sid == "two":
            raise TimeoutError if failure == "deadline" else OSError
        return original(sid, **kw)
    monkeypatch.setattr(session_engine._store, "load_session_window", read)
    result = recall(session_engine, limit=2)
    metas = {s["session_id"]: s for s in result["provenance"]["delivery"]["sessions"]}
    assert metas["one"]["omission_reason"] == "unreadable_group"
    assert metas["two"]["omission_reason"] == failure


@pytest.mark.parametrize("entry", [False, True])
def test_fix1_deadline_seeds(session_engine, monkeypatch, entry):
    from hermes_lcm import recall_session
    seed = add(session_engine, "needle")
    original, get = recall_session.deliver_session, session_engine._store.get
    clock, expired = tools.time.monotonic, [False]
    def read(*a, **kw):
        row = get(*a, **kw)
        expired[0] = True
        return row
    def deliver(*a, **kw):
        expired[0] = entry
        monkeypatch.setattr(session_engine._store, "get", read)
        return original(*a, **kw)
    monkeypatch.setattr(recall_session, "deliver_session", deliver)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock() + (100 if expired[0] else 0))
    result = recall(session_engine)
    meta = result["provenance"]["delivery"]
    leads = meta["sessions"] + meta.get("seed_leads", [])
    assert result["timeout"] and any(seed in s.get("seed_ids", []) and s.get("expand_hint") for s in leads)


@pytest.mark.parametrize("nonascii,anchor", [(5000, False), (8000, True)])
def test_fix1_fallback_final_bytes(session_engine, monkeypatch, nonascii, anchor):
    from hermes_lcm import recall_session
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    session_engine._config.large_output_externalization_enabled = False
    original = recall_session.deliver_session
    def deliver(ordered, response, **kw):
        if anchor:
            response["provenance"]["fts_anchor"] = dict(position=1, fired=True, delivered=False)
        return original(ordered, response, **kw)
    monkeypatch.setattr(recall_session, "deliver_session", deliver)
    sid = add(session_engine, "needle " + "a" * 400 + "界" * nonascii)
    raw = tools.lcm_recall({"query": "needle", "limit": 1}, engine=session_engine)
    pad = 400 + nonascii * 5 - len(raw)
    for delta in range(-20, 21):
        with session_engine._store._write_lock:
            session_engine._store._conn.execute("UPDATE messages SET content=? WHERE store_id=?",
                ("needle " + "a" * (pad + delta) + "界" * nonascii, sid))
            session_engine._store._conn.commit()
        raw = tools.lcm_recall({"query": "needle", "limit": 1}, engine=session_engine)
        d = json.loads(raw)["provenance"]["delivery"]
        assert d.get("stop_reason") != "read_error"
        assert tokens._fallback_token_estimate(raw) == d["response_tokens"] <= 16000
        assert len(raw) == d["response_chars"] <= 64000


@pytest.mark.parametrize("size", [56000, 59000])
def test_fix2_deadline_keeps_evidence_before_hints(session_engine, monkeypatch, size):
    e = session_engine
    e._config.large_output_externalization_enabled = False
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    seed = add(e, "needle " + "a" * size, sid="big")
    for i in range(20):
        add(e, "needle", sid=f"s{i:02d}")
    rank, read = tools._lcm_recall_rerank, e._store.load_session_window
    def ranked(*a, **kw):
        ordered, status, scores = rank(*a, **kw)
        return sorted(ordered, key=lambda x: x["hit"]["session_id"] != "big"), status, scores
    def window(sid, **kw):
        if sid != "big":
            raise TimeoutError
        return read(sid, **kw)
    monkeypatch.setattr(tools, "_lcm_recall_rerank", ranked)
    monkeypatch.setattr(e._store, "load_session_window", window)
    result = recall(e, limit=21)
    d = result["provenance"]["delivery"]
    assert [h["store_id"] for h in result["hits"]] == [seed]
    known = {i for s in d["sessions"] for i in s["seed_ids"]} | {seed}
    leads = [i for lead in d.get("seed_leads", []) for i in lead["seed_ids"]]
    assert not known.intersection(leads) and len(leads) == len(set(leads))
    assert result["timeout"] and d["response_tokens"] <= 16000 and d["response_chars"] <= 64000


@pytest.mark.parametrize("entry", [False, True])
def test_fix2_deadline_summary_lead(session_engine, monkeypatch, entry):
    from hermes_lcm import recall_session
    from hermes_lcm.dag import SummaryNode
    e = session_engine
    source = add(e, "needle source")
    node = e._dag.add_node(SummaryNode(session_id="one", depth=0, summary="needle", source_ids=[source]))
    seed = dict(hit=dict(kind="summary", session_id="one", node_id=node, snippet="needle"),
                ranks={0: 1}, _final_score=1)
    monkeypatch.setattr(tools, "_lcm_recall_rerank", lambda *a, **kw: ([seed, seed], "off", []))
    original, clock, expired = recall_session.deliver_session, tools.time.monotonic, [False]
    def deliver(*a, **kw):
        expired[0] = entry
        return original(*a, **kw)
    def sources(*a, **kw):
        raise TimeoutError
    monkeypatch.setattr(recall_session, "deliver_session", deliver)
    monkeypatch.setattr(e._dag, "source_message_ids", sources)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock() + (100 if expired[0] else 0))
    result = recall(e, limit=2)
    leads = result["provenance"]["delivery"]["summary_leads"]
    assert result["timeout"] and result["hits"] == []
    assert leads == [dict(node_id=node, session_id="one",
                         expand_hint=f'lcm_expand(node_id={node}, session_id="one")')]


def test_fix2_nonconvergence_omission(session_engine, monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(tools.time, "time", lambda: 1700000000)
    e = session_engine
    e._config.large_output_externalization_enabled = False
    sid = add(e, "needle " + "a" * 400 + "界" * 5000)
    raw = tools.lcm_recall({"query": "needle", "limit": 1}, engine=e)
    pad, observed = 400 + 25000 - len(raw), []
    for delta in range(-20, 21):
        with e._store._write_lock:
            e._store._conn.execute("UPDATE messages SET content=? WHERE store_id=?",
                ("needle " + "a" * (pad + delta) + "界" * 5000, sid))
            e._store._conn.commit()
        raw = tools.lcm_recall({"query": "needle", "limit": 1}, engine=e)
        d = json.loads(raw)["provenance"]["delivery"]
        assert tokens._fallback_token_estimate(raw) == d["response_tokens"] <= 16000
        assert len(raw) == d["response_chars"] <= 64000
        if d.get("stop_reason") == "size_nonconvergence":
            observed.append(delta)
            assert {s["omission_reason"] for s in d["sessions"]} == {"size_nonconvergence"}
    assert observed, "fixture must exercise the fallback estimator discontinuity"


def test_fix3_memory_store(session_engine, monkeypatch):
    from hermes_lcm.store import MessageStore
    from hermes_lcm.dag import SummaryDAG
    from hermes_lcm import recall_session
    store, dag = MessageStore(":memory:"), SummaryDAG(":memory:")
    monkeypatch.setattr(session_engine, "_store", store)
    monkeypatch.setattr(session_engine, "_dag", dag)
    try:
        seed = add(session_engine, "needle")
        neighbor = add(session_engine, "local neighbor")
        ranked = dict(hit=dict(kind="message_excerpt", session_id="one", store_id=seed),
                      ranks={}, _final_score=1)
        monkeypatch.setattr(tools, "_lcm_recall_rerank", lambda *a, **kw: ([ranked], "off", []))
        connections = (store._conn, dag._conn)
        before = [(c.total_changes, c.in_transaction, c.execute("PRAGMA query_only").fetchone()[0])
                  for c in connections]
        monkeypatch.setattr(recall_session.sqlite3, "connect", Mock(side_effect=AssertionError("wrong file")))
        result = recall(session_engine)
        assert [h["store_id"] for h in result["hits"]] == [seed, neighbor]
        assert before == [(c.total_changes, c.in_transaction, c.execute("PRAGMA query_only").fetchone()[0])
                          for c in connections]
    finally:
        dag.close()
        store.close()


@pytest.mark.parametrize("phase", ["admission", "refit"])
def test_fix3_evidence_before_pointers(session_engine, monkeypatch, phase):
    from hermes_lcm import recall_session
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    session_engine._config.large_output_externalization_enabled = False
    seed = add(session_engine, "needle " + "a" * 2000)
    original = recall_session.deliver_session
    def deliver(ordered, response, **kw):
        kw["summary_leads"] = [dict(session_id="one", node_id=99, expand_hint="x" * 64000)]
        if phase == "refit":
            # Simulate the final deadline metadata arriving after admission.
            def count(raw, **unused):
                payload = json.loads(raw)
                final = payload["provenance"]["delivery"]["sessions"][0]["complete"]
                return 16001 if final and payload["provenance"]["delivery"]["summary_leads"] else 100
            monkeypatch.setattr(tokens, "count_tokens_with_encoder", count)
            kw["summary_leads"][0]["expand_hint"] = "hint"
        return original(ordered, response, **kw)
    monkeypatch.setattr(recall_session, "deliver_session", deliver)
    result = recall(session_engine)
    assert [h["store_id"] for h in result["hits"]] == [seed]
    assert result["provenance"]["delivery"]["summary_leads"] == []


def test_fix3_oversized_base_envelope(session_engine, monkeypatch):
    from hermes_lcm import recall_session
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    add(session_engine, "needle")
    original = recall_session.deliver_session
    def deliver(ordered, response, **kw):
        response["degraded_reason"] = "x" * 100000
        return original(ordered, response, **kw)
    monkeypatch.setattr(recall_session, "deliver_session", deliver)
    result = recall(session_engine)
    raw = json.dumps(result, ensure_ascii=False)
    d = result["provenance"]["delivery"]
    assert len(raw) == d["response_chars"] <= 64000
    assert tokens._fallback_token_estimate(raw) == d["response_tokens"] <= 16000
    assert result["error"] == "session_base_envelope_over_budget"
