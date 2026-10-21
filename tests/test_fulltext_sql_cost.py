"""Deterministic #1586 SQL work, ranking, and private deadline ownership guards."""
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hermes_lcm import db_bootstrap, store as sm, tools
from hermes_lcm.dag import SummaryDAG


class CountedCursor(sqlite3.Cursor):
    def fetchall(self):
        rows = super().fetchall()
        self.connection.fetched.append(len(rows))
        return rows

    def fetchone(self):
        row = super().fetchone()
        self.connection.fetched.append(int(row is not None))
        return row


class CountedConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statements = []
        self.fetched = []
        self.materialized = {"content": 0, "snippet": 0}
        self.count_materialization = False
        self.fail_code = None
        self.handlers = []
        for name in self.materialized:
            def count(value, name=name):
                self.materialized[name] += 1
                return value
            self.create_function("counted_" + name, 1, count)

    def execute(self, sql, parameters=()):
        if "MATCH" in sql:
            self.statements.append((sql, list(parameters)))
            if self.fail_code is not None:
                error = sqlite3.OperationalError("synthetic SQL failure")
                error.sqlite_errorcode = self.fail_code
                raise error
        if "snippet(" in sql and self.count_materialization:
            sql = sql.replace("m.content,", "counted_content(m.content) AS content,")
            sql = sql.replace("snippet(", "counted_snippet(snippet(")
            sql = sql.replace("40) as snippet", "40)) as snippet")
        return self.cursor(CountedCursor).execute(sql, parameters)

    def set_progress_handler(self, callback, n):
        self.handlers.append((callback, n))
        return super().set_progress_handler(callback, n)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(db_bootstrap, "dispatch_background_stem_backfill", lambda *a: None)
    result = sm.MessageStore(tmp_path / "sql-cost.db")
    result.connection.close()
    result._conn = sqlite3.connect(result.db_path, factory=CountedConnection)
    result._conn.row_factory = sqlite3.Row
    result._conn.executemany(
        "INSERT INTO messages(session_id,source,conversation_id,role,content,timestamp,observed_at) "
        "VALUES(?,?,?,?,?,?,?)",
        [(f"session-{i % 3}", "telegram" if i % 2 else "unknown", f"conv-{i % 2}",
          ("user", "assistant", "tool")[i % 3], "needle alpha beta filler", float(i // 3),
          1700000000.1234567) for i in range(400)],
    )
    result._conn.commit()
    yield result
    result.close()


def test_materialization_is_bounded_by_ranked_window(store, monkeypatch):
    monkeypatch.setattr(sm, "should_apply_directness_rank_adjustment", lambda *a: False)
    conn = store.connection
    conn.count_materialization = True
    assert store._search_fts("needle", limit=3, sort="relevance")
    window = sm.compute_search_fetch_limit(3, ["needle"], [])
    assert conn.materialized == {"content": window, "snippet": window}


def test_ranked_fetch_crosses_python_in_one_row(store, monkeypatch):
    monkeypatch.setattr(sm, "should_apply_directness_rank_adjustment", lambda *a: False)
    conn = store.connection
    conn.fetched.clear()
    assert store._search_fts("needle", limit=3, sort="relevance")
    # No row-by-row Python fetch after the SQL ranking/materialization step.
    assert conn.fetched[-1:] == [1]


def test_interrupt_is_terminal_before_plain_or_like_retry(store, monkeypatch):
    conn = store.connection
    conn.fail_code = sqlite3.SQLITE_INTERRUPT
    monkeypatch.setattr(store, "_stem_state", lambda: "ready")
    like_calls = []
    monkeypatch.setattr(store, "_search_like", lambda *a, **kw: like_calls.append(kw) or [])
    with pytest.raises(sqlite3.OperationalError) as caught:
        store._search_fts("needle", stemmed=True)
    assert caught.value.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT
    assert len(conn.statements) == 1
    assert "messages_fts_stem MATCH" in conn.statements[0][0]
    assert like_calls == []


@pytest.mark.parametrize("sort", ["relevance", "recency", "hybrid"])
def test_ranked_windows_match_frozen_stock_payload_with_filters(store, monkeypatch, sort):
    # Freeze time-dependent hybrid ordering; all stored messages are in its past.
    monkeypatch.setattr(sm.time, "time", lambda: 1800000000.0)
    namespace = vars(sm).copy()
    source = (Path(__file__).parent / "fixtures" / "issue_1586_stock_fts.txt").read_text()
    exec(compile(source, "<stock-1586>", "exec"), namespace)
    stock = namespace["_search_fts"]
    # Deliberately exercise multiple offsets and directness/rank ties without
    # requiring a large corpus. Both paths consume these same ranking inputs.
    monkeypatch.setattr(sm, "compute_search_fetch_limit", lambda *a: 3)
    monkeypatch.setattr(sm, "compute_search_candidate_cap", lambda *a: 30)
    namespace.update(compute_search_fetch_limit=sm.compute_search_fetch_limit,
                     compute_search_candidate_cap=sm.compute_search_candidate_cap)
    page_counts = []
    for filters in ({}, {"exclude_session_ids": {"session-0"}},
                    {"session_id": "session-1", "source": "telegram", "role": "assistant",
                     "conversation_id": "conv-1", "time_from": 5.0, "time_to": 90.0}):
        kwargs = dict(limit=2, sort=sort, allow_operators=True, **filters)
        expected = stock(store, "needle", **kwargs)
        conn = store.connection
        conn.statements.clear()
        actual = store._search_fts("needle", **kwargs)
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        page_counts.append(len(conn.statements))
        assert all("ranked_keys AS MATERIALIZED" in sql for sql, _ in conn.statements)
    if sort == "relevance":
        assert max(page_counts) > 1


def test_private_deadline_has_no_hot_progress_handler(store, monkeypatch):
    dag = SummaryDAG(store.db_path)
    engine = SimpleNamespace(_store=store, _dag=dag)
    original_connect = sqlite3.connect
    opened = []

    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs, factory=CountedConnection)
        opened.append(conn)
        return conn

    monkeypatch.setattr(tools.sqlite3, "connect", connect)
    monkeypatch.setattr(tools, "_run_within_deadline", lambda invoke, **kw: invoke())
    monkeypatch.setattr(tools, "_lcm_grep_full_text", lambda *a, **kw: '{"results": []}')
    try:
        payload = tools._lcm_grep_full_text_with_deadline(
            {"query": "needle"}, engine=engine, deadline=tools.time.monotonic() + 5,
        )
        assert payload == {"results": []}
        assert len(opened) == 2
        assert all(not any(callback is not None and n < 1000000 for callback, n in c.handlers)
                   for c in opened)
        assert not any(t.name == "full-text-deadline" for t in threading.enumerate())
    finally:
        dag.close()
