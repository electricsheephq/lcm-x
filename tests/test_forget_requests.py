"""Pure policy thresholds and command-form detection."""
import pytest

from hermes_lcm.forget_requests import covers, forget_content_words, is_forget_request


@pytest.mark.parametrize("text", [
    "Please forget that I prefer green tea.", "Kindly forget about green tea.",
    'A new thought. "Could you forget my green tea preference?',
    "Don't remember my green tea preference.", "Do not keep that green tea detail.",
    "Stop tracking my green tea preference.", "No longer store what I said about green tea.",
    "Never bring up my green tea preference.", "Please don't use what I shared about green tea.",
    "Disregard what I said about green tea.", "Erase my preference for green tea.",
    "Delete that detail about green tea.", "Remove from your memory green tea.",
    "Wipe my history of green tea.", "Scratch what I mentioned about green tea.",
    "Discard that info about green tea.", "Purge my data about green tea.",
    "I'd rather you not remember green tea.", "I'd rather you didn't mention green tea.",
    "I don't want you to use green tea.",
])
def test_commands(text):
    assert is_forget_request(text)
    assert not is_forget_request(text, "assistant")


@pytest.mark.parametrize("text", [
    "it made me forget the time", "I won't forget this day", "don't forget to taste as you go",
    "I may forget that tomorrow", "Please forget to buy tea", "Remember green tea",
    "Delete " + "x" * 41 + " my data",
])
def test_non_commands(text):
    assert not is_forget_request(text)


def test_detail_words():
    assert forget_content_words("Please forget that I shared my preference: green tea!") == {"green", "tea"}


def row(ident, text, session="s", **kw):
    return dict(store_id=ident, session_id=session, content=text, session_order=1, **kw)


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
    assert covers(row(50, "green tea", "other", observed_at=100), f)
    assert not covers(row(1, "green tea", "other"), f)  # equal/unknown session order
    f["session_order"] = 5
    assert covers(row(1, "green tea", "other"), f)
    for text in ("Forget it", "Forget my tea"):
        f["content"] = text
        assert not covers(row(9, "green tea"), f)
