"""Policy thresholds, command-form detection and candidate-path parity."""
import sqlite3
import time

import pytest

from hermes_lcm.forget_requests import covers, forget_content_words, is_forget_request, read_forget_rows
from hermes_lcm.store import MessageStore


POSITIVE_EXAMPLES = [
    "Please forget that I prefer green tea.", "Please forget my preference for window seats.",
    "Please forget the detail about my brother's job.", "Forget that I'm vegetarian.",
    "Don't remember that I live in Austin.", "Please delete my address from your memory.",
    "I don't want you to remember my phone number.",
    "A new thought. Could you forget my green tea preference?",
    "Don't remember my green tea preference.", "Do not keep in mind my green tea preference.",
    "Stop retaining my green tea preference.", "No longer store what I said about green tea.",
    "Disregard what I said about green tea.", "Ignore what I shared about green tea.",
    "Erase my preference for green tea from memory.", "Remove from your memory green tea.",
    "I'd rather you not remember my green tea preference.",
    "I'd rather you didn't know about my green tea preference.",
    "Kindly forget about my green tea preference.", "FORGET THAT I'VE MOVED TO AUSTIN.",
    "I’d like you to forget that I’m vegetarian.",
    "Can you erase my phone number from your memory?",
    "Stop remembering my home address on Oak Street",
    "Don't store my home address on Oak Street",
    "I don't want you to know my home address on Oak Street",
    "Ignore what I said about my home address on Oak Street",
    "I'd rather you not store my home address on Oak Street",
    "Stop memorize my green tea preference.",
    "No longer store about me green tea.",
    "Stop keep in minding my green tea preference.",
]


@pytest.mark.parametrize("text", POSITIVE_EXAMPLES)
def test_commands(text):
    assert is_forget_request(text)
    assert not is_forget_request(text, "assistant")


@pytest.mark.parametrize("text", [
    "it made me forget the time", "I won't forget this day", "don't forget to taste as you go",
    "I may forget that tomorrow", "Please forget to buy tea", "Remember green tea",
    "Delete " + "x" * 41 + " my data",
    "Can you remove the duplicate rows I mentioned earlier?", "forget it, let's move on",
    "Do not save the file yet", "Don't use that library for date parsing",
    "Stop tracking the time spent", "Delete the previous migration file",
    "I don't want you to use jQuery here", "Never mention it again lol",
    'She said: "Forget about the past."', "lyric:\n> forget about the boy",
    "```\nforget the cache\n```", "Kindly forget about green tea.",
    "She said: “Forget my green tea preference.”", '"Please forget that I like green tea."',
    "```\nPlease forget that I like green tea.\n```", "`Forget my green tea preference.`",
    'She said: "First line.\nPlease forget my green tea preference."',
    "She said: ‘First line. Please forget my green tea preference.’",
    "> Please forget that I like green tea.\nordinary text",
    "Erase my green tea preference.", "I'd rather you not remember green tea.",
    "Delete " + "x" * 41 + " from your memory.",
    "Don't delete my green tea preference from your memory.",
    "Do not delete my green tea preference from your memory.",
    "Never erase my address from your memory.",
    "No need to remove my phone number from memory.",
    "Please don't forget that I prefer green tea.",
    "I don't want you to forget my birthday.",
    "Not: delete my green tea preference from memory.",
    "You shouldn't: forget my green tea preference.",
    "She might delete my green tea preference from memory.",
])
def test_non_commands(text):
    assert not is_forget_request(text)


def test_detail_words():
    assert forget_content_words("Please forget that I shared my preference: green tea!") == {"green", "tea"}
    assert forget_content_words("I'm I've I'd I'll it's let's that's what's don't Jane’s party a x 12") == {"jane", "party"}


def test_every_positive_through_fts_and_scan(tmp_path):
    path = tmp_path / "indexed.db"
    store = MessageStore(str(path))
    try:
        assert store.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name='messages_fts'"
        ).fetchone()
        ids = [store.append("s", dict(role="user", content=text, timestamp=1)) for text in POSITIVE_EXAMPLES]
        fs, _, reason = read_forget_rows(path, [], deadline=time.monotonic() + 5)
        assert reason is None
        assert {r["store_id"] for r in fs} == set(ids)
        indexed = {r["content"] for r in fs}
    finally:
        store.close()
    scan_path = tmp_path / "scan.db"
    with sqlite3.connect(scan_path) as conn:
        conn.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
                     "content TEXT, source TEXT, observed_at REAL, timestamp REAL)")
        conn.executemany("INSERT INTO messages VALUES (?, 's', 'user', ?, '', NULL, 1)",
                         enumerate(POSITIVE_EXAMPLES, 1))
    fs, _, reason = read_forget_rows(scan_path, [], deadline=time.monotonic() + 5)
    assert reason is None
    assert {r["content"] for r in fs} == indexed


@pytest.mark.parametrize("excluded", [
    "```\nalpha beta gamma delta epsilon zeta\n```",
    '"alpha beta gamma delta epsilon zeta"',
    "> alpha beta gamma delta epsilon zeta",
    "`alpha beta gamma delta epsilon zeta`",
])
def test_coverage_uses_cleaned_request_text(excluded):
    f = row(10, "Please forget my green tea preference\n" + excluded, previous_ids=[9])
    assert is_forget_request(f["content"])
    assert covers(row(9, "I love green tea in the morning"), f)
    assert not covers(row(1, "alpha beta gamma delta epsilon zeta"), f)


@pytest.mark.parametrize("forgotten,other,adjacent", [
    ("Forget about Jane's party", "Let's go to the party tonight", False),
    ("Please forget that I'm vegetarian", "I'm going to the gym", True),
    ("Please forget what I said about my brother's job", "What's your job? It's busy", False),
])
def test_contractions_do_not_create_coverage(forgotten, other, adjacent):
    assert not covers(row(9, other), row(10, forgotten, previous_ids=[9] if adjacent else []))


def row(ident, text, session="s", **kw):
    return dict(store_id=ident, session_id=session, content=text, **{"session_order": 1, **kw})


def test_adjacent_and_far_thresholds():
    f = row(10, "Forget my green tea lemon honey preference", previous_ids=[8, 9])
    assert covers(row(8, "green tea"), f)
    assert covers(row(9, "green tea"), f)
    assert not covers(row(9, "green"), f)  # 25%, adjacent
    assert covers(row(2, "green tea"), f)  # two of four, far
    assert not covers(row(2, "green"), f)
    f["content"] += " sugar"
    assert covers(row(8, "green tea"), f)  # 40%, adjacent
    assert not covers(row(2, "green tea"), f)  # 40%, far


def test_order_scope_and_precision():
    f = row(10, "Forget my green tea", previous_ids=[9], observed_at=200)
    for r in (row(10, "green tea"), row(11, "green tea"),
              row(1, "green tea", scope="other"), row(1, "green tea", "other", observed_at=300)):
        assert not covers(r, f)
    assert not covers(row(50, "green tea", "other", observed_at=100), f)
    assert covers(row(1, "green tea", "other", observed_at=100), f)
    assert covers(row(1, "green tea", "other", observed_at=200), f)
    assert covers(row(1, "green tea", "other"), f)
    f["session_order"] = 5
    assert covers(row(1, "green tea", "other"), f)
    for text in ("Forget it", "Forget my tea"):
        f["content"] = text
        assert not covers(row(9, "green tea"), f)


def test_later_restatement_in_old_session_without_observed_at():
    f = row(100, "Forget my green tea", "B", session_order=100)
    assert not covers(row(150, "green tea", "A", session_order=1), f)


def test_store_prose_is_not_a_candidate_term():
    """Only store-request phrasings reach the bounded candidate set, so unrelated
    store/storage prose cannot crowd real requests out of the 200-row cap."""
    from hermes_lcm import forget_requests
    assert "store*" not in forget_requests._FTS_QUERY
    assert '"store what"' in forget_requests._FTS_QUERY
