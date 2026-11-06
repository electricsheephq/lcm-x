"""Read-path integration with a real, synthetic store and no semantic provider."""
import json
from types import SimpleNamespace
import time

import pytest

from hermes_lcm import forget_requests as forget, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    cfg = LCMConfig(database_path=str(tmp_path / "forget.db"), recall_honour_forget=True)
    store, dag = MessageStore(cfg.database_path), SummaryDAG(cfg.database_path)
    e = SimpleNamespace(_config=cfg, _store=store, _dag=dag, current_session_id="s", _hermes_home=str(tmp_path))
    yield e
    store.close()
    dag.close()


def add(e, text, role="user", session="s", stamp=1700000000):
    return e._store.append(session, dict(role=role, content=text, timestamp=stamp))


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
    result = recall(engine)
    assert f in [h["store_id"] for h in result["hits"]]
    assert not {old, reply} & {h["store_id"] for h in result["hits"]}
    assert result["total_results"] == 2
    assert {r["store_id"] for r in result["provenance"]["forget_withheld"]} == {old, reply}
    assert all(r["forget_store_id"] == f and r["forget_event_time"] == "2023-11-16T02:00:00Z"
               for r in result["provenance"]["forget_withheld"])
    assert next(h for h in result["hits"] if h["store_id"] == f)["event_time"] == "2023-11-16T02:00:00Z"
    assert engine._store.connection.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == before


def test_far_and_one_word_and_later(engine):
    old = add(engine, "green tea", session="old", stamp=1600000000)
    one = add(engine, "green coffee", session="old", stamp=1600000001)
    for _ in range(4):
        add(engine, "unrelated", stamp=1700000000)
    f = add(engine, "Forget my green tea", stamp=1700100000)
    later = add(engine, "green tea again", stamp=1700200000)
    result = recall(engine, limit=10)
    ids = {h["store_id"] for h in result["hits"]}
    assert old not in ids
    assert {f, one, later} <= ids


@pytest.mark.parametrize("detail", ["snippets", "answer_ready"])
def test_off_is_whole_json_identical_to_unset(engine, monkeypatch, detail):
    seeded(engine)
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
    import sqlite3

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
