"""#627: structured image parts count at the host's flat per-image price.

A message whose content holds an inline image used to count by the characters
of the encoded image: one 0.5 MB screenshot counted about 130,000 tokens while
the host and the provider price it at a flat figure (host default 1,500). The
over-count forced compactions the host did not ask for, cut the fresh tail at
the image row and could start the forced overflow recovery.
"""

import base64
import importlib
import json
import random
import sys
import types

import pytest

from hermes_lcm import tokens as tokens_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.fresh_tail import resolve_fresh_tail_boundary
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens, count_tokens

PRICE = 1500
_WORDS = (
    "context window summary engine message archive session review budget tail "
    "provider screenshot result compaction token assistant request history recall"
).split()


def _image_data_url(chars: int = 500_000, seed: int = 627) -> str:
    # Random bytes, so no token counter can compress the payload.
    raw = random.Random(seed).randbytes(chars * 3 // 4)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _prose(words: int, seed: int = 1) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(words))


def _stripped_json(parts) -> str:
    return json.dumps(parts, ensure_ascii=False, sort_keys=True)


def _image_url_part(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _placeholder(part_type: str) -> dict:
    return {"type": part_type, "image": "[stripped]"}


@pytest.fixture
def price(monkeypatch):
    monkeypatch.setattr(tokens_mod, "image_token_cost", lambda: PRICE, raising=False)
    return PRICE


def _no_price_call():
    raise AssertionError("image_token_cost must not be called for content without image parts")


# -- T1: measured shape ------------------------------------------------------


def test_t1_tool_row_with_text_and_inline_image_counts_below_60k(price):
    text = _prose(25_000)
    text_part = {"type": "text", "text": text}
    content = [text_part, _image_url_part(_image_data_url())]
    msg = {"role": "tool", "tool_call_id": "call-1", "content": content}

    text_tokens = count_tokens(_stripped_json([text_part, _placeholder("image_url")]))
    assert 20_000 < text_tokens < 55_000  # setup: about 50,000 tokens of text
    assert len(normalize_content_value(content)) > 500_000

    count = count_message_tokens(msg)
    assert count < 60_000
    assert count == 4 + text_tokens + price


# -- T2: one test per type -----------------------------------------------------


@pytest.mark.parametrize(
    "image_part",
    [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD" * 50_000}},
        _image_url_part("data:image/png;base64," + "QUJD" * 50_000),
        {"type": "input_image", "image_url": "data:image/png;base64," + "QUJD" * 50_000},
    ],
    ids=["image", "image_url", "input_image"],
)
def test_t2_each_image_part_type_is_priced_flat(price, image_part):
    text_part = {"type": "text", "text": "what does this screenshot show?"}
    msg = {"role": "user", "content": [text_part, image_part]}

    expected_text = count_tokens(_stripped_json([text_part, _placeholder(image_part["type"])]))
    assert count_message_tokens(msg) == 4 + expected_text + price


def test_t2_multimodal_tool_result_counts_text_summary_plus_price(price):
    summary = "Screenshot of the settings page with the save button highlighted."
    content = {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "captured"},
            _image_url_part(_image_data_url(200_000)),
        ],
        "text_summary": summary,
    }
    msg = {"role": "tool", "tool_call_id": "call-2", "content": content}

    assert count_message_tokens(msg) == 4 + count_tokens(summary) + price


# -- T3: price per part ----------------------------------------------------------


def test_t3_three_image_parts_add_exactly_two_prices_over_one(price, monkeypatch):
    # Fallback counter with a private cache generation: equal-length ASCII text
    # counts exactly equal, so the difference is the image price alone.
    monkeypatch.setattr(tokens_mod, "_get_encoder", lambda: None)
    monkeypatch.setattr(tokens_mod, "_encoder_generation", -627)

    text_part = {"type": "text", "text": "compare these three screenshots"}
    image = _image_url_part("data:image/png;base64," + "QUJD" * 10_000)
    stripped_len = len(_stripped_json(_placeholder("image_url")))
    filler = {"type": "text", "text": ""}
    filler["text"] = "x" * (stripped_len - len(_stripped_json(filler)))
    assert len(_stripped_json(filler)) == stripped_len

    three = {"role": "user", "content": [text_part, image, image, image]}
    one = {"role": "user", "content": [text_part, image, filler, filler]}

    assert count_message_tokens(three) - count_message_tokens(one) == 2 * price


# -- T4: zero moves ----------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "plain string content " * 50,
        json.dumps([{"type": "text", "text": "hi"}, _image_url_part("data:image/png;base64," + "QUJD" * 5_000)]),
        [{"type": "text", "text": "no image here"}, {"type": "tool_result", "content": "ok"}],
        {"type": "text", "text": "a dict without _multimodal", "image_url": "data:image/png;base64,QUJD"},
        {"_multimodal": True, "content": [{"type": "text", "text": "only text"}], "text_summary": "s"},
        None,
    ],
    ids=["string", "json-string-with-image", "list-without-image", "dict", "multimodal-without-image", "none"],
)
def test_t4_content_without_structured_image_parts_counts_as_before(monkeypatch, content):
    monkeypatch.setattr(tokens_mod, "image_token_cost", _no_price_call, raising=False)
    msg = {"role": "user", "content": content}

    assert count_message_tokens(msg) == 4 + count_tokens(normalize_content_value(content) or "")


# -- T5: forced overflow recovery -------------------------------------------------


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm-627.db"), reserve_tokens_floor=40_000)
    eng = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    eng.on_session_start("issue-627", conversation_id="issue-627-conversation", context_length=272_000)
    yield eng
    eng.shutdown()


def _text_rows(target_tokens: int) -> list:
    rows = []
    total = 0
    index = 0
    while total < target_tokens:
        row = {"role": "user" if index % 2 == 0 else "assistant", "content": _prose(1_500, seed=index)}
        rows.append(row)
        total += count_message_tokens(row)
        index += 1
    return rows


def _image_pair(call_id: str, seed: int) -> list:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": [{"type": "text", "text": "captured"}, _image_url_part(_image_data_url(seed=seed))],
        },
    ]


def test_t5_image_rows_do_not_force_overflow_recovery(engine, price):
    assert engine._effective_assembly_token_cap() == 232_000
    rows = _text_rows(133_000)
    assert 133_000 <= count_messages_tokens(rows) < 140_000
    messages = [{"role": "system", "content": "system"}, *rows, *_image_pair("img-1", 1), *_image_pair("img-2", 2)]

    assert engine._should_force_overflow_recovery(messages=messages) is False
    assert count_messages_tokens(messages) < 150_000


def test_t5_text_only_over_cap_still_forces_overflow_recovery(engine, price):
    assert engine._effective_assembly_token_cap() == 232_000
    messages = [{"role": "system", "content": "system"}, *_text_rows(233_000)]

    assert count_messages_tokens(messages) > 232_000
    assert engine._should_force_overflow_recovery(messages=messages) is True


# -- T6: fresh tail -------------------------------------------------------------------


def test_t6_image_row_does_not_end_the_protected_tail(price):
    messages = []
    for index in range(20):
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": _prose(200, seed=index)})
    messages.extend(_image_pair("img-tail", 3))  # indices 20, 21: inside the last 5 rows
    messages.append({"role": "assistant", "content": _prose(200, seed=22)})
    messages.append({"role": "user", "content": _prose(200, seed=23)})
    assert len(messages) == 24

    boundary = resolve_fresh_tail_boundary(messages, fresh_tail_count=24, fresh_tail_max_tokens=24_000)

    assert boundary.start == 0
    assert boundary.count == 24
    assert boundary.token_limited is False
    assert boundary.tokens <= 24_000


# -- T7: price helper -------------------------------------------------------------------


def _stub_host(monkeypatch, fn):
    stub = types.ModuleType("agent.image_token_cost")
    stub.current_image_token_cost = fn
    monkeypatch.setitem(sys.modules, "agent.image_token_cost", stub)


def test_t7_price_helper_host_module_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent.image_token_cost", None)
    assert tokens_mod.image_token_cost() == 1500


def test_t7_price_helper_uses_host_value(monkeypatch):
    _stub_host(monkeypatch, lambda: 1070)
    assert tokens_mod.image_token_cost() == 1070


def _raise():
    raise RuntimeError("host price unavailable")


@pytest.mark.parametrize("fn", [lambda: 0, lambda: -5, _raise], ids=["zero", "negative", "raises"])
def test_t7_price_helper_falls_back_to_default(monkeypatch, fn):
    _stub_host(monkeypatch, fn)
    assert tokens_mod.image_token_cost() == 1500


# -- T8: real host ----------------------------------------------------------------------


def test_t8_real_host_price_is_used():
    try:
        host = importlib.import_module("agent.image_token_cost")
    except Exception:
        pytest.skip("host agent.image_token_cost is not importable")
    text_part = {"type": "text", "text": "a screenshot from the real host"}
    msg = {"role": "tool", "tool_call_id": "call-8", "content": [text_part, _image_url_part(_image_data_url(100_000))]}

    expected_text = count_tokens(_stripped_json([text_part, _placeholder("image_url")]))
    assert count_message_tokens(msg) == 4 + expected_text + host.current_image_token_cost()
