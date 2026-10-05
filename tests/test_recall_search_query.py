"""Recall's private OR query uses the harness's content-term spelling."""
import pytest


@pytest.mark.parametrize(("query", "expected"), [
    ("beagle, puppy!", "beagle OR puppy"),
    ("What name did we give the beagle puppy?", "name OR we OR give OR beagle OR puppy"),
    ("The BEAGLE Puppy", "BEAGLE OR Puppy"),
    ("\u13a0 \u1c90", "\u13a0 OR \u1c90"),
    ("NOT", '"NOT"'),
    ("NEAR", '"NEAR"'),
    ("beagle puppy beagle puppy?", "beagle OR puppy"),
    ("art-related art", "art OR related"),
    ("What is Alice's birthday?", "Alice OR birthday"),
    ("pi is 3.14", "pi OR 3 OR 14"),
    ("don't forget it's due", "don OR forget OR due"),
    ('meet in "New York" soon', '"New York" OR meet OR soon'),
    ('"the of"', '"the of"'),
    ("", ""),
    ("what did the?", ""),
    ("?!,", ""),
])
def test_build_recall_or_query(query, expected):
    from hermes_lcm.search_query import build_recall_or_query

    assert build_recall_or_query(query) == expected


def test_recall_or_stopword_phrase_survives():
    from hermes_lcm.search_query import build_recall_or_query

    assert build_recall_or_query('"The Who" band') == '"The Who" OR band'


def test_recall_or_dedupes_ascii_case_only():
    from hermes_lcm.search_query import build_recall_or_query

    assert build_recall_or_query("Project project Zebra") == "Project OR Zebra"
    assert build_recall_or_query("Project Zebra project") == "Project OR Zebra"
    assert build_recall_or_query("\u13a0 \uab70 \u1c90 \u10d0") == "\u13a0 OR \uab70 OR \u1c90 OR \u10d0"


@pytest.mark.parametrize("word", ["NOT", "NEAR"])
def test_recall_or_mixed_operator_is_literal(word):
    from hermes_lcm.search_query import build_recall_or_query, extract_search_terms

    assert set(build_recall_or_query(f"{word} status").split(" OR ")) == {f'"{word}"', "status"}
    assert extract_search_terms(f"{word} status") == ["status"]


def test_bundled_recall_policy_distinguishes_or_and():
    from pathlib import Path

    policy = (Path(__file__).resolve().parents[1] / "skills/hermes-lcm/references/recall-policy.md").read_text()
    assert "`lcm_recall`'s full-text arm ORs content words" in policy
    assert "stop words dropped, quoted phrases kept" in policy
    assert "`lcm_grep`, its fallbacks, and `lcm_expand_query` still AND their terms" in policy
    assert "Do not pad a query with synonyms." in policy


@pytest.mark.parametrize("word", ["NOT", "NEAR"])
def test_recall_or_query_never_emits_a_bare_operator(word):
    import sqlite3

    from hermes_lcm.search_query import build_recall_or_query, sanitize_fts5_query

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.execute("INSERT INTO t VALUES ('we did not deploy near the bridge')")
    query = sanitize_fts5_query(build_recall_or_query(word), allow_operators=True)
    assert conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (query,)).fetchone()[0] == 1


@pytest.mark.parametrize("word", ["\u13a0", "\u1c90"])
def test_recall_or_query_keeps_terms_whose_case_the_tokenizer_does_not_fold(word):
    # Cherokee and Georgian Mtavruli capitals: Python lower-cases them, unicode61 (Unicode 6.1) does not.
    import sqlite3

    from hermes_lcm.search_query import build_recall_or_query, sanitize_fts5_query

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.execute("INSERT INTO t VALUES (?)", (f"note {word} here",))
    query = sanitize_fts5_query(build_recall_or_query(word), allow_operators=True)
    assert conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (query,)).fetchone()[0] == 1


@pytest.mark.parametrize(("stored", "query"), [
    ("Alice's party is on Friday", "When is Alice's birthday?"),
    ("pi is roughly 3.14 here", "the value 3.14"),
    ("we met in New York last May", 'trip to "New York"'),
])
def test_recall_or_query_keeps_the_tokenizer_boundaries(stored, query):
    import sqlite3

    from hermes_lcm.search_query import build_recall_or_query, sanitize_fts5_query

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.execute("INSERT INTO t VALUES (?)", (stored,))
    conn.execute("INSERT INTO t VALUES ('an unrelated row about lunch')")
    built = build_recall_or_query(query)
    assert built
    match = sanitize_fts5_query(built, allow_operators=True)
    assert conn.execute("SELECT x FROM t WHERE t MATCH ? ORDER BY rank", (match,)).fetchone()[0] == stored
