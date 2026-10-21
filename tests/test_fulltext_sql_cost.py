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
        if rows and len(rows[0]) == 17:
            self.connection.windows.append([tuple(row) for row in rows])
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
        self.aggregate_fail_code = None
        self.aggregate_errors = []
        self.windows = []
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
        try:
            if "json_group_array" in sql and self.aggregate_fail_code is not None:
                error = sqlite3.OperationalError("synthetic aggregate failure")
                error.sqlite_errorcode = self.aggregate_fail_code
                raise error
            return self.cursor(CountedCursor).execute(sql, parameters)
        except sqlite3.Error as exc:
            if "json_group_array" in sql:
                self.aggregate_errors.append(exc.sqlite_errorcode)
            raise

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
        conn.materialized = {"content": 0, "snippet": 0}
        conn.count_materialization = True
        actual = store._search_fts("needle", **kwargs)
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        page_counts.append(len(conn.statements))
        assert all(0 < count <= sm.compute_search_candidate_cap(2)
                   for count in conn.materialized.values())
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


def frozen_stock():
    namespace = vars(sm).copy()
    source = (Path(__file__).parent / "fixtures" / "issue_1586_stock_fts.txt").read_text()
    exec(compile(source, "<stock-1586>", "exec"), namespace)
    return namespace["_search_fts"]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("sort", ["relevance", "recency", "hybrid"])
def test_reverse_scan_matches_stock_windows_and_directness_pages(store, monkeypatch, sort, tied):
    conn = store.connection
    if tied:
        conn.execute("UPDATE messages SET timestamp = 1.0")
        conn.commit()
    else:
        conn.execute("UPDATE messages SET timestamp = store_id")
        conn.commit()
    conn.execute("PRAGMA reverse_unordered_selects=ON")
    monkeypatch.setattr(sm, "compute_search_fetch_limit", lambda *a: 3)
    monkeypatch.setattr(sm, "compute_search_candidate_cap", lambda *a: 30)
    stock = frozen_stock()
    conn.windows.clear()
    expected = stock(store, "needle", limit=2, sort=sort)
    expected_windows = list(conn.windows)
    actual_windows = []
    original = sm._fetch_ranked_search_rows

    def capture(*args):
        rows = original(*args)
        actual_windows.append([tuple(row) for row in rows])
        return rows

    monkeypatch.setattr(sm, "_fetch_ranked_search_rows", capture)
    actual = store._search_fts("needle", limit=2, sort=sort)
    assert actual_windows == expected_windows
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    if sort == "relevance":
        assert len(actual_windows) > 1


@pytest.mark.parametrize("error_code", [sqlite3.SQLITE_TOOBIG, sqlite3.SQLITE_NOMEM])
@pytest.mark.parametrize("stemmed", [False, True])
def test_aggregate_resource_error_keeps_stock_fulltext_window(store, monkeypatch, error_code, stemmed):
    conn = store.connection
    if stemmed:
        conn.execute("INSERT INTO messages_fts_stem(messages_fts_stem) VALUES('rebuild')")
        monkeypatch.setattr(store, "_stem_state", lambda: "ready")
    # Every individual stock row fits; the combined JSON window does not.
    old_limit = conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 16384)
    if error_code == sqlite3.SQLITE_NOMEM:
        conn.aggregate_fail_code = error_code
    like_calls = []
    monkeypatch.setattr(store, "_search_like", lambda *a, **kw: like_calls.append(kw) or [])
    try:
        conn.windows.clear()
        expected = frozen_stock()(store, "needle", limit=20, stemmed=stemmed)
        window_sizes = [len(page) for page in conn.windows]
        conn.statements.clear()
        conn.fetched.clear()
        actual = store._search_fts("needle", limit=20, stemmed=stemmed)
        assert like_calls == []
        assert conn.aggregate_errors == [error_code] * len(window_sizes)
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        assert conn.fetched == window_sizes
        assert len(conn.statements) == 2 * len(window_sizes)
        table = "messages_fts_stem" if stemmed else "messages_fts"
        assert all(f"{table} MATCH" in sql for sql, _ in conn.statements)
        assert store._last_fts_index == ("stem" if stemmed else "plain")
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, old_limit)
