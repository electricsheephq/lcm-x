"""Read-path integration with a real, synthetic store and no semantic provider."""
import json
import sqlite3
from types import SimpleNamespace
import time

import pytest

from hermes_lcm import forget_requests as forget, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore
from tests.test_forget_requests import POSITIVE_EXAMPLES


@pytest.fixture
def engine(tmp_path):
    cfg = LCMConfig(database_path=str(tmp_path / "forget.db"), recall_honour_forget=True)
    store, dag = MessageStore(cfg.database_path), SummaryDAG(cfg.database_path)
    e = SimpleNamespace(_config=cfg, _store=store, _dag=dag, current_session_id="s", _hermes_home=str(tmp_path))
    yield e
    store.close()
    dag.close()


def add(e, text, role="user", session="s", stamp=1700000000, source=""):
    return e._store.append(session, dict(role=role, content=text, timestamp=stamp), source=source)


def recall(e, **args):
    return json.loads(tools.lcm_recall({"query": "green tea", "limit": 2, **args}, engine=e))


def seeded(e):
    old = add(e, "I prefer green tea")
    reply = add(e, "Your green tea preference is noted", "assistant")
    f = add(e, "Please forget my green tea preference", stamp=1700100000)
    return old, reply, f


def test_adjacent_replacement_count_provenance_and_no_writes(engine):
    old, reply, f = seeded(engine)
    add(engine, "Green gardens unrelated", stamp=1700200000)
    before = engine._store.connection.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
    engine._config.recall_honour_forget = False
    window = {h["store_id"] for h in recall(engine)["hits"]}
    engine._config.recall_honour_forget = True
    result = recall(engine)
    assert f in [h["store_id"] for h in result["hits"]]
    assert not {old, reply} & {h["store_id"] for h in result["hits"]}
    assert result["total_results"] == 2
    assert {r["store_id"] for r in result["provenance"]["forget_withheld"]} == {old, reply} & window
    assert all(r["forget_store_id"] == f and r["forget_event_time"] == "2023-11-16T02:00:00Z"
               for r in result["provenance"]["forget_withheld"])
    assert next(h for h in result["hits"] if h["store_id"] == f)["event_time"] == "2023-11-16T02:00:00Z"
    assert engine._store.connection.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == before


def test_far_and_one_word_and_later(engine):
    far = add(engine, "green tea", stamp=1600000000)
    old = add(engine, "green tea", session="old", stamp=1600000000)
    one = add(engine, "green coffee", session="old", stamp=1600000001)
    for _ in range(4):
        add(engine, "unrelated", stamp=1700000000)
    f = add(engine, "Forget my green tea", stamp=1700100000)
    later = add(engine, "green tea again", stamp=1700200000)
    result = recall(engine, limit=10)
    ids = {h["store_id"] for h in result["hits"]}
    assert {far, old, f, one, later} <= ids


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
@pytest.mark.parametrize("unit", ["hits", "session"])
def test_off_is_whole_json_identical_to_unset(engine, monkeypatch, detail, unit):
    seeded(engine)
    engine._config.recall_delivery_unit = unit
    # SimpleNamespace gives a genuinely absent key, unlike a dataclass class default.
    config = vars(engine._config).copy()
    config.pop("recall_honour_forget")
    engine._config = SimpleNamespace(**config)
    args = {"query": "green tea", "detail": detail}
    monkeypatch.setattr(forget, "read_forget_rows", lambda *a, **k: pytest.fail("flag-off read"))
    before = tools.lcm_recall(args, engine=engine)
    grep = tools.lcm_grep({"query": "green tea"}, engine=engine)
    expand = tools.lcm_expand({"store_id": 1}, engine=engine)
    engine._config.recall_honour_forget = False
    assert tools.lcm_recall(args, engine=engine) == before
    assert tools.lcm_grep({"query": "green tea"}, engine=engine) == grep
    assert tools.lcm_expand({"store_id": 1}, engine=engine) == expand


def test_explicit_tools_return_covered_rows(engine):
    old, reply, f = seeded(engine)
    payload = json.loads(tools.lcm_grep({"query": "green tea"}, engine=engine))
    covered = [h for h in payload["results"] if h.get("store_id") in {old, reply}]
    assert len(covered) == 2
    for h in covered:
        assert h["forget_requested"] == {"forget_store_id": f, "on": "2023-11-16"}
    expanded = json.loads(tools.lcm_expand({"store_id": reply}, engine=engine))
    assert expanded["content"] == "Your green tea preference is noted"
    assert expanded["forget_requested"]["forget_store_id"] == f


@pytest.mark.parametrize("reason", ["deadline", "read_error"])
def test_failure_noop(engine, monkeypatch, reason):
    seeded(engine)
    engine._config.recall_honour_forget = False
    before = recall(engine)
    engine._config.recall_honour_forget = True
    monkeypatch.setattr(forget, "read_forget_rows", lambda *a, **k: ([], {}, reason))
    result = recall(engine)
    assert result["hits"] == before["hits"]
    assert result["provenance"].pop("forget_unavailable") == reason
    assert result == before


def test_bounded_read_failures(engine, monkeypatch):
    assert forget.read_forget_rows(engine._store.db_path, [], deadline=0) == ([], {}, "deadline")
    assert forget.read_forget_rows("/nonexistent/forget.db", [], deadline=time.monotonic() + 1)[2] == "read_error"
    seeded(engine)
    for i in range(205):
        add(engine, f"Remember a narrative {i}", stamp=1700200000 + i)
    fs, _, reason = forget.read_forget_rows(engine._store.db_path, [], deadline=time.monotonic() + 1)
    assert reason is None and not fs  # real forget is outside the 200-row prefilter window


def test_scope_and_summary_pool(engine):
    old, _, f = seeded(engine)
    fs, rows, reason = forget.read_forget_rows(engine._store.db_path, [old], allowed=[], deadline=time.monotonic() + 1)
    assert fs == [] and reason is None
    fs, rows, _ = forget.read_forget_rows(engine._store.db_path, [old], deadline=time.monotonic() + 1)
    summary = {"hit": {"kind": "summary", "node_id": 1, "store_id": old}, "ranks": {}, "_final_score": .2}
    entry = {"hit": {"kind": "message_excerpt", "store_id": old}, "ranks": {}, "_final_score": .1}
    ordered, safe, receipts = forget.withhold([summary, entry], rows, fs, 2, "s")
    assert ordered[0] is summary and ordered[-1] is entry
    assert ordered[1]["hit"]["store_id"] == f and safe == 2 and receipts


def test_summary_hit_untouched_in_recall(engine, monkeypatch):
    seeded(engine)
    real = tools.fuse_candidates

    def fuse(*args, **kwargs):
        ordered, arms, weights = real(*args, **kwargs)
        ordered.insert(0, {"hit": dict(kind="summary", node_id=99, session_id="s", snippet="green tea"),
                           "ranks": {}, "rrf_score": .9, "_final_score": .9})
        return ordered, arms, weights

    monkeypatch.setattr(tools, "fuse_candidates", fuse)
    result = recall(engine, limit=5)
    assert any(h.get("node_id") == 99 for h in result["hits"])


def test_session_neighbours_cannot_reintroduce_covered_rows(engine):
    old, reply, f = seeded(engine)
    engine._config.recall_delivery_unit = "session"
    result = recall(engine)
    assert f in {h["store_id"] for h in result["hits"]}
    assert not {old, reply} & {h["store_id"] for h in result["hits"]}
    delivery = result["provenance"]["delivery"]
    assert delivery["sessions"][0]["complete"] is False
    assert delivery["sessions"][0]["omission_reason"] == "forget_requested"
    assert delivery["truncated"] == (bool(delivery.get("stop_reason")) or any(
        not s["complete"] for s in delivery["sessions"]))
    assert delivery["truncated"] is True


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_forget_not_in_ranked_pool_is_inserted(engine, monkeypatch, detail):
    old, reply, f = seeded(engine)
    real = tools._lcm_recall_fts_arm

    def without_forget(*args, **kwargs):
        hits, error = real(*args, **kwargs)
        hits[:] = [h for h in hits if h["store_id"] != f]
        return hits, error

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", without_forget)
    result = recall(engine, detail=detail)
    assert [h["store_id"] for h in result["hits"]] == [f]
    assert result["hits"][0]["event_time"] == "2023-11-16T02:00:00Z"
    assert {h["store_id"] for h in result["provenance"]["forget_withheld"]} == {old, reply}
    if detail == "answer_ready":
        assert result["hits"][0]["content"] == "Please forget my green tea preference"


def test_expand_summary_source_rows_stay_truthful(engine):
    old, reply, f = seeded(engine)
    node = SummaryNode(session_id="s", depth=0, summary="green tea",
                       source_type="messages", source_ids=[old, reply])
    engine._dag.add_node(node)
    result = json.loads(tools.lcm_expand({"node_id": node.node_id}, engine=engine))
    assert [h["store_id"] for h in result["expanded"]] == [old, reply]
    assert all(h["forget_requested"]["forget_store_id"] == f for h in result["expanded"])


def test_one_snapshot_read_and_unknown_date(engine, monkeypatch):
    old = add(engine, "green tea", stamp=None)
    f = add(engine, "Forget my green tea", stamp=None)
    statements, connections = [], []
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        connections.append((args, kwargs))
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(forget.sqlite3, "connect", connect)
    result = json.loads(tools.lcm_expand({"store_id": old}, engine=engine))
    assert result["forget_requested"] == {"forget_store_id": f, "on": None}
    assert len(connections) == 1 and "mode=ro" in connections[0][0][0]
    assert len([s for s in statements if s.startswith("WITH requests")]) == 1
    assert not any(s.startswith(("INSERT", "UPDATE", "DELETE")) for s in statements)


def test_subagent_user_rows_never_request_forgetting(engine):
    old = add(engine, "green tea")
    delegated = add(engine, "Forget my green tea", source=tools._SUBAGENT_SOURCE)
    fs, _, reason = forget.read_forget_rows(engine._store.db_path, [old], deadline=time.monotonic() + 1)
    assert reason is None and fs == []
    assert {old, delegated} <= {h["store_id"] for h in recall(engine)["hits"]}


def test_only_delivered_window_has_receipts_but_entire_pool_is_withheld(engine):
    ids = []
    for i in range(4):
        ids.extend(add(engine, f"green tea detail {i * 2 + j}") for j in range(2))
        add(engine, "Forget my green tea", stamp=1700100000 + i)
    engine._config.recall_honour_forget = False
    window = {h["store_id"] for h in recall(engine, limit=2)["hits"]}
    engine._config.recall_honour_forget = True
    result = recall(engine, limit=2)
    receipts = result["provenance"]["forget_withheld"]
    assert 0 < len(receipts) <= 2
    assert {r["store_id"] for r in receipts} == window & set(ids)
    assert not set(ids) & {h["store_id"] for h in result["hits"]}


def test_session_only_removes_rows_and_preserves_unaffected_provenance(engine, monkeypatch):
    from hermes_lcm import recall_session

    old = add(engine, "green tea", session="covered")
    f = add(engine, "Forget my green tea. " + "padding " * 5000,
            session="covered", stamp=1700100000)
    healthy = add(engine, "green tea", session="healthy", stamp=1700200000)
    engine._config.recall_delivery_unit = "session"
    original_fts = tools._lcm_recall_fts_arm
    original_delivery = recall_session.deliver_session
    raw_payloads = []

    def without_forget(*a, **kw):
        hits, error = original_fts(*a, **kw)
        hits[:] = [h for h in hits if h["store_id"] != f]
        return hits, error

    def capture(*a, **kw):
        raw = original_delivery(*a, **kw)
        raw_payloads.append(json.loads(raw))
        return raw

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", without_forget)
    monkeypatch.setattr(recall_session, "deliver_session", capture)
    result = recall(engine)
    assert {h["store_id"] for h in result["hits"]} == {f, healthy}
    assert {h["store_id"] for h in raw_payloads[0]["hits"]} == {old, f, healthy}
    before = {s["session_id"]: s for s in raw_payloads[0]["provenance"]["delivery"]["sessions"]}
    after = {s["session_id"]: s for s in result["provenance"]["delivery"]["sessions"]}
    assert after["healthy"] == before["healthy"]
    assert after["covered"]["complete"] is False
    assert after["covered"]["omission_reason"] == "forget_requested"
    assert after["covered"]["returned_id_ranges"] == [[f, f]]
    assert result["provenance"]["delivery"]["response_chars"] < result["provenance"]["delivery"]["char_budget"]


def test_seen_forget_replacement_is_not_novel_in_answer_ready_hits(engine, monkeypatch):
    old, reply, f = seeded(engine)
    real = tools._lcm_recall_fts_arm

    def without_forget(*a, **kw):
        hits, error = real(*a, **kw)
        hits[:] = [h for h in hits if h["store_id"] != f]
        return hits, error

    monkeypatch.setattr(tools, "_lcm_recall_fts_arm", without_forget)
    ref = f"lcm:{f}:0-{len('Please forget my green tea preference')}"
    result = recall(engine, detail="answer_ready", seen_refs=[ref])
    assert result["hits"] == []
    assert result["delta"]["novel_refs"] == []
    assert result["delta"]["novel_ref_count"] == 0 and not result["delta"]["progress"]
    assert result["delta"]["termination_reason"] == "no_novel_exact_ref"


def test_session_delta_rebuild_excludes_seen_refs():
    old = dict(store_id=1, session_id="s", content="green tea")
    f = dict(store_id=2, session_id="s", content="Forget my green tea", role="user", previous_ids=[1])
    hits = [dict(**r, score=1, exact_ref=f"lcm:{r['store_id']}:0-{len(r['content'])}") for r in [old, f]]
    ref = hits[1]["exact_ref"]
    payload = dict(hits=hits, delta={}, provenance=dict(delivery=dict(sessions=[dict(session_id="s")])))
    forget.filter_session_payload(payload, {1: old, 2: f}, [f], "s", seen_refs={ref})
    assert payload["delta"]["novel_refs"] == []
    assert payload["delta"]["termination_reason"] == "no_novel_exact_ref"


def test_fts_candidate_path_is_used_scoped_and_read_only(engine, monkeypatch):
    old, _, f = seeded(engine)
    add(engine, "Forget my green tea", session="excluded", stamp=1700300000)
    add(engine, "Forget my green tea", session="outside", stamp=1700400000)
    statements = []
    real_connect = sqlite3.connect

    def connect(*a, **kw):
        conn = real_connect(*a, **kw)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(forget.sqlite3, "connect", connect)
    fs, _, reason = forget.read_forget_rows(engine._store.db_path, [old], allowed=["s", "excluded"],
                                           excluded=["excluded"], deadline=time.monotonic() + 1)
    assert reason is None and [r["store_id"] for r in fs] == [f]
    query = next(s for s in statements if s.startswith("WITH requests"))
    assert "messages_fts MATCH" in query and "instr(lower" not in query
    assert forget._FTS_QUERY in query
    assert not any(s.startswith(("INSERT", "UPDATE", "DELETE", "CREATE")) for s in statements)


def test_scan_fallback_when_fts_is_absent(tmp_path):
    path = tmp_path / "no-fts.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
                     "content TEXT, source TEXT, observed_at REAL, timestamp REAL)")
        conn.execute("INSERT INTO messages VALUES (1, 's', 'user', 'Forget my green tea', '', NULL, 1)")
    fs, _, reason = forget.read_forget_rows(path, [], deadline=time.monotonic() + 1)
    assert reason is None and [r["store_id"] for r in fs] == [1]


@pytest.mark.parametrize("text", POSITIVE_EXAMPLES)
def test_every_detected_form_is_read_through_the_index(engine, text):
    old = add(engine, "My home address is 12 Oak Street")
    add(engine, "Noted your address", "assistant")
    f = add(engine, text, stamp=1700100000)
    assert forget.is_forget_request(text, "user")
    fs, _, reason = forget.read_forget_rows(engine._store.db_path, [old], deadline=time.monotonic() + 1)
    assert reason is None and f in [r["store_id"] for r in fs]


def test_tool_call_and_result_rows_are_never_covered():
    f = dict(store_id=10, session_id="s", content="Please forget my home address on Oak Street",
             previous_ids=(8, 9), observed_at=None)
    call = dict(store_id=8, session_id="s", role="assistant", content="Looking up your home address on Oak Street",
                tool_calls='[{"id":"1"}]', observed_at=None)
    result = dict(store_id=9, session_id="s", role="tool", content="home address Oak Street", observed_at=None)
    plain = dict(store_id=8, session_id="s", role="assistant", content="Your home address on Oak Street", observed_at=None)
    assert not forget.covers(call, f) and not forget.covers(result, f)
    assert forget.covers(plain, f)


@pytest.mark.parametrize("unit", ["hits", "session"])
def test_near_cap_forget_receipts_are_trimmed(engine, monkeypatch, unit):
    ids = {add(engine, f"green tea detail {i}") for i in range(2)}
    f = add(engine, "Forget my green tea", stamp=1700100000)
    engine._config.recall_delivery_unit = unit
    original = tools.recall_delivery.deliver_recall
    delivered = []

    def near_cap(*a, **kw):
        payload = json.loads(original(*a, **kw))
        cap = (payload["provenance"]["delivery"]["char_budget"] if unit == "session"
               else tools._LCM_RECALL_RESPONSE_CHAR_CAP)
        payload["provenance"]["padding"] = ""
        payload["provenance"]["padding"] = "x" * (cap - 150 - len(json.dumps(payload, ensure_ascii=False)))
        raw = json.dumps(payload, ensure_ascii=False)
        assert len(raw) == cap - 150
        delivered.append(payload)
        return raw

    monkeypatch.setattr(tools.recall_delivery, "deliver_recall", near_cap)
    raw = tools.lcm_recall({"query": "green tea", "limit": 25, "detail": "answer_ready"}, engine=engine)
    result = json.loads(raw)
    provenance = result["provenance"]
    cap = provenance["delivery"]["char_budget"] if unit == "session" else tools._LCM_RECALL_RESPONSE_CHAR_CAP
    assert len(raw) <= cap
    assert provenance["forget_withheld_omitted"] > 0
    assert result["hits"] == delivered[0]["hits"]
    assert {h["store_id"] for h in result["hits"]} == {f}
    assert not ids & {h["store_id"] for h in result["hits"]}
    if unit == "session":
        assert provenance["delivery"]["response_chars"] == len(raw)


@pytest.mark.parametrize("ids,ranges,expected", [
    ([1, 2, 3, 4, 5], [[1, 5]], [[1, 2], [4, 5]]),
    ([1, 2, 3, 4, 5], [[1, 2], [3, 3], [4, 4], [5, 5]], [[1, 2], [4, 4], [5, 5]]),
    ([1, 2, 3, 5, 7], [[1, 7]], [[1, 2], [5, 7]]),
])
def test_session_ranges_preserve_source_runs_after_filtering(ids, ranges, expected):
    rows = {i: dict(store_id=i, session_id="s", content="green tea" if i == 3 else "unrelated") for i in ids}
    f = dict(store_id=10, session_id="s", content="Forget my green tea", previous_ids=[3])
    meta = dict(session_id="s", complete=True, returned_id_ranges=ranges)
    payload = dict(hits=list(rows.values()), provenance=dict(delivery=dict(sessions=[meta])))
    forget.filter_session_payload(payload, rows, [f], "s")
    assert meta["returned_id_ranges"] == expected


def test_receipt_budget_preserves_unavailable_and_prefix():
    receipts = [dict(store_id=i, forget_store_id=99, forget_event_time=None) for i in range(10)]
    payload = dict(hits=[], provenance=dict(forget_withheld=receipts.copy(), forget_unavailable="deadline"))
    cap = len(json.dumps(dict(hits=[], provenance=dict(
        forget_withheld=receipts[:2], forget_unavailable="deadline", forget_withheld_omitted=8)), ensure_ascii=False))
    raw = forget.encode_payload(payload, char_cap=cap)
    assert len(raw) <= cap
    assert payload["provenance"]["forget_withheld"] == receipts[:2]
    assert payload["provenance"]["forget_withheld_omitted"] == 8
    assert payload["provenance"]["forget_unavailable"] == "deadline"
    # A base envelope that cannot fit still returns, without dropping its failure reason.
    raw = forget.encode_payload(payload, char_cap=1)
    assert len(raw) > 1
    assert payload["provenance"]["forget_unavailable"] == "deadline"
    assert payload["provenance"]["forget_withheld"] == []
    assert payload["provenance"]["forget_withheld_omitted"] == 10
