"""The newest auto-derived focus bullet must survive the prompt head window (#613)."""

import inspect

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm import escalation
from hermes_lcm.engine import LCMEngine
from hermes_lcm.escalation import (
    _build_l1_prompt,
    _build_l1_prompt_v2,
    _build_l2_prompt,
)


A = "a" * 80
B = "b" * 80
NEW = "NEWEST-REQUEST " + "c" * 65


@pytest.fixture
def engine(tmp_path):
    instance = LCMEngine(config=None, hermes_home=str(tmp_path))
    try:
        yield instance
    finally:
        instance.shutdown()


def _derive(engine, turns):
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn})
        messages.append({"role": "assistant", "content": "Acknowledged."})
    derived = engine._derive_auto_focus_topic(messages)
    assert derived is not None
    return derived


def _primary_focus(prompt):
    return next(line for line in prompt.splitlines() if line.startswith("Primary focus:"))


def test_three_80_char_turns_newest_in_v1_l1_primary_focus(engine):
    derived = _derive(engine, [A, B, NEW])
    prompt = _build_l1_prompt("x", 500, depth=0, focus_topic=derived)
    assert NEW in _primary_focus(prompt)


def test_three_80_char_turns_newest_in_v1_l2_primary_focus(engine):
    derived = _derive(engine, [A, B, NEW])
    prompt = _build_l2_prompt("x", 500, focus_topic=derived)
    assert NEW in _primary_focus(prompt)


def test_three_80_char_turns_newest_in_v2_focus_tag(engine):
    derived = _derive(engine, [A, B, NEW])
    prompt = _build_l1_prompt_v2("x", 500, 0, focus_topic=derived)
    topic = prompt.rsplit("<lcm-focus-topic>", 1)[1].split("</lcm-focus-topic>", 1)[0]
    assert NEW in topic


def test_host_composed_newest_request_survives_prompt_window(engine):
    request = "draft the Q3 vendor renewal email"
    newest = "[auto-loaded skill: x]\n" + "Guidance text. " * 60 + "\n\n" + request
    derived = _derive(engine, ["Check the vendor list", "Review the renewal date", newest])
    prompt = _build_l1_prompt("x", 500, depth=0, focus_topic=derived)
    newest_bullet = derived.splitlines()[1]
    assert request in _primary_focus(prompt)
    assert newest_bullet in _primary_focus(prompt)
    assert "[auto-loaded skill: x]" in newest_bullet
    assert "…" in newest_bullet


def test_auto_focus_lists_newest_turn_first(engine):
    for turns in ([NEW], [B, NEW], [A, B, NEW]):
        derived = _derive(engine, turns)
        assert derived.startswith("Recent user focus:")
        assert "-" not in derived.splitlines()[0]
        assert derived.splitlines()[1:] == [f"- {turn}" for turn in reversed(turns)]
    assert derived.index(NEW) < derived.index(B) < derived.index(A)


def test_prompt_window_constant_matches_normalizer_default():
    default = inspect.signature(escalation._normalized_focus_topic).parameters["max_chars"].default
    assert engine_module._AUTO_FOCUS_PROMPT_MAX_CHARS == default


def test_explicit_focus_topic_is_unchanged():
    prompt = _build_l1_prompt("x", 500, depth=0, focus_topic="deploy the billing service")
    assert _primary_focus(prompt) == "Primary focus: deploy the billing service"

    explicit = "migration very-long-topic " + "x" * 274
    assert len(explicit) == 300
    prompt = _build_l1_prompt("x", 500, depth=0, focus_topic=explicit)
    assert _primary_focus(prompt) == "Primary focus: " + explicit[:159] + "…"
