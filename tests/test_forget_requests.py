"""Pure policy thresholds and command-form detection."""
import pytest

from hermes_lcm.forget_requests import covers, forget_content_words, is_forget_request


@pytest.mark.parametrize("text", [
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
])
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
])
def test_non_commands(text):
    assert not is_forget_request(text)


def test_detail_words():
    assert forget_content_words("Please forget that I shared my preference: green tea!") == {"green", "tea"}
    assert forget_content_words("I'm I've I'd I'll it's let's that's what's don't Jane’s party a x 12") == {"jane", "party"}


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
    assert not covers(row(2, "green tea"), f)  # two of four, far
    assert not covers(row(2, "green"), f)
    assert not covers(row(2, "green tea lemon honey"), f)  # 100%, far
    f["content"] += " sugar"
    assert covers(row(8, "green tea"), f)  # 40%, adjacent
    assert not covers(row(2, "green tea"), f)  # 40%, far


def test_order_scope_and_precision():
    f = row(10, "Forget my green tea", previous_ids=[9], observed_at=200)
    for ident in (10, 11, 50):
        assert not covers(row(ident, "green tea", observed_at=100),
                          {**f, "previous_ids": [ident]})
    assert not covers(row(9, "green tea", scope="other"), f)
    assert not covers(row(9, "green tea", observed_at=300), f)
    assert covers(row(9, "green tea", observed_at=100), f)
    assert covers(row(9, "green tea", observed_at=200), f)
    assert covers(row(9, "green tea"), f)
    for stamp in (100, 200, 300, None):
        assert not covers(row(9, "green tea", "other", observed_at=stamp), f)
    f["session_order"] = 5
    assert covers(row(9, "green tea"), f)
    for text in ("Forget it", "Forget my tea"):
        f["content"] = text
        assert not covers(row(9, "green tea"), f)


def test_question_rows_are_never_covered():
    f = row(10, "Please forget my green tea preference", previous_ids=[8, 9])
    question = "Can you suggest some green tea brands?"
    assert not covers(row(8, question, role="user"), f)
    assert not covers(row(8, question + "  ", role="user"), f)
    assert not covers(row(9, question, role="assistant"), f)
    assert covers(row(9, "Your green tea preference is noted", role="assistant"), f)


def test_later_restatement_in_old_session_without_observed_at():
    f = row(100, "Forget my green tea", "B", session_order=100)
    assert not covers(row(150, "green tea", "A", session_order=1), f)
