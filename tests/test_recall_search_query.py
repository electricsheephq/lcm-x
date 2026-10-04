"""Recall's private OR query uses the harness's content-term spelling."""
import pytest


@pytest.mark.parametrize(("query", "expected"), [
    ("beagle, puppy!", "beagle OR puppy"),
    ("What name did we give the beagle puppy?", "name OR we OR give OR beagle OR puppy"),
    ("The BEAGLE Puppy", "beagle OR puppy"),
    ("NOT", "not"),
    ("NEAR", "near"),
    ("beagle puppy beagle puppy?", "beagle OR puppy"),
    ("art-related art", "artrelated OR art OR related"),
    ("", ""),
    ("what did the?", ""),
    ("?!,", ""),
])
def test_build_recall_or_query(query, expected):
    from hermes_lcm.search_query import build_recall_or_query

    assert build_recall_or_query(query) == expected


@pytest.mark.parametrize("word", ["NOT", "NEAR"])
def test_recall_or_query_never_emits_a_bare_operator(word):
    import sqlite3

    from hermes_lcm.search_query import build_recall_or_query, sanitize_fts5_query

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.execute("INSERT INTO t VALUES ('we did not deploy near the bridge')")
    query = sanitize_fts5_query(build_recall_or_query(word), allow_operators=True)
    assert conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (query,)).fetchone()[0] == 1
