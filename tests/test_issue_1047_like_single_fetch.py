"""Seeded identity and SQL-count regression for the LIKE candidate window."""
import random
import sqlite3

import pytest

from hermes_lcm import store as store_module
from hermes_lcm.search_query import compute_search_candidate_cap
from hermes_lcm.store import MessageStore
from ._reference_like_search_1047 import ReferenceLikeSearch


OLD_TIME = 1_600_000_000.0
QUERIES = (
    "记忆", "北京", "上海 计划", "东京 中文", '"中文 计划"',
    "foo_bar", "100%", "path/to", "🚀", "alpha alpha", '"alpha beta"',
    "不存在 missing_term",
)
SCOPES = (
    {},
    {"session_id": "session-1"},
    {"role": "user", "time_from": OLD_TIME - 3600, "time_to": OLD_TIME},
    {"source": "synthetic-a", "conversation_id": "conversation-0",
     "exclude_session_ids": ["session-2"], "time_to": OLD_TIME},
)
CASES = [
    pytest.param(query, scope, (1, 26, 40, 65)[scope_index],
                 id=f"query-{query_index}-scope-{scope_index}")
    for query_index, query in enumerate(QUERIES)
    for scope_index, scope in enumerate(SCOPES)
]


@pytest.fixture(scope="module")
def corpus():
    # Search-only fixture: the actual message columns, no ingestion/FTS side effects.
    store = MessageStore.__new__(MessageStore)
    store._conn = sqlite3.connect(":memory:")
    store._conn.execute("""
        CREATE TABLE messages (
            store_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
            source TEXT DEFAULT '', role TEXT NOT NULL, content TEXT,
            tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
            timestamp REAL NOT NULL, token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0, conversation_id TEXT DEFAULT '',
            ingested_at REAL, observed_at REAL, observed_at_source TEXT
        )
    """)
    rng = random.Random(1047)
    rows = []
    for index in range(3000):
        # 1,200 newest user rows put caps 500/520/800 inside the same group.
        timestamp = OLD_TIME - (index // 1500) * 3600
        role = "user" if index % 5 != 4 else rng.choice(("assistant", "tool", "system"))
        terms = rng.sample(("北京", "上海", "东京", "中文 计划", "alpha beta"), 3)
        content = "记忆 foo_bar 100% path/to 🚀 " + " ".join(terms)
        content += " 记忆" * rng.randrange(4)
        rows.append((index + 1, f"session-{index % 4}",
                     f"synthetic-{'ab'[index % 2]}", role, content, timestamp,
                     f"conversation-{index % 3}"))
    store._conn.executemany(
        "INSERT INTO messages "
        "(store_id, session_id, source, role, content, timestamp, conversation_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", rows,
    )
    store._conn.commit()
    yield store
    store._conn.close()


def _run_with_candidate_ids(store, method, query, **kwargs):
    ids = []
    original = store._row_to_dict

    def record(row):
        ids.append(row[0])
        return original(row)

    store._row_to_dict = record
    try:
        results = method(store, query, **kwargs)
    finally:
        store._row_to_dict = original
    return results, ids


@pytest.mark.parametrize("sort", ("recency", "relevance", "hybrid"))
@pytest.mark.parametrize("query,scope,limit", CASES)
def test_like_differential(corpus, monkeypatch, sort, query, scope, limit):
    # Fixed old timestamps keep SQL hybrid ordering stable across wall-clock seconds.
    # Freeze Python's age clock too; SQLite strftime('now') remains the real clock.
    monkeypatch.setattr(store_module.time, "time", lambda: 1_900_000_000.0)
    kwargs = {"sort": sort, "limit": limit, **scope}
    expected, expected_ids = _run_with_candidate_ids(
        corpus, ReferenceLikeSearch._search_like, query, **kwargs,
    )
    actual, actual_ids = _run_with_candidate_ids(
        corpus, MessageStore._search_like, query, **kwargs,
    )
    assert actual_ids == expected_ids
    # Full dictionaries include store_id order, search_rank and snippet.
    assert actual == expected
    if sort == "recency" and query == "记忆" and not scope:
        assert len(actual_ids) > compute_search_candidate_cap(limit)


@pytest.mark.parametrize("sort,expected_count", (("recency", 2), ("relevance", 1), ("hybrid", 1)))
def test_like_query_count(corpus, sort, expected_count):
    statements = []
    corpus._conn.set_trace_callback(statements.append)
    try:
        results = corpus._search_like("记忆", limit=26, sort=sort)
    finally:
        corpus._conn.set_trace_callback(None)
    queries = [sql for sql in statements if "FROM messages" in sql]
    assert len(results) == 26
    assert len(queries) == expected_count, f"{sort}: {len(queries)} message statements"
    assert "LIMIT 520 OFFSET 0" in queries[0]
    if sort == "recency":
        assert "AND timestamp = 1600000000.0" in queries[1]
        # Only the boundary role-bias group can continue the window; other roles are never fetched.
        assert "AND (CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 WHEN 'tool' THEN 2 ELSE 1 END) = " in queries[1]
        assert "OFFSET" not in queries[1]
        assert "LIMIT" not in queries[1]
