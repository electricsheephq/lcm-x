"""#931 amendment 1: tolerate final hint formatting and bare trailing controls only."""

import logging
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.escalation as escalation

NONCE = "0123456789abcdef0123456789abcdef"
OPEN = f'<lcm-summary nonce="{NONCE}">'
CLOSE = "</lcm-summary>"
LINES = "Decision: preserve raw history and the migration approval gate."
HINT = "Expand for details about: migration gate and raw history"
BODY = f"{LINES}\n{HINT}"
SOURCE = "ordinary historical transcript " * 200


def check(body):
    return escalation._check_summary_contract(f"{OPEN}\n{body}\n{CLOSE}", NONCE, 160)


def install_provider(monkeypatch, make_reply):
    module = ModuleType("agent.auxiliary_client")

    def call_llm(**kwargs):
        nonce = re.search(r'nonce="([0-9a-f]{32})"', kwargs["messages"][0]["content"]).group(1)
        reply = make_reply(f'<lcm-summary nonce="{nonce}">')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)


@pytest.mark.parametrize(("hint", "tag"), [
    (f"- {HINT}", "hint_list"),
    (f"* {HINT}", "hint_list"),
    (f"• {HINT}", "hint_list"),
    (f"1. {HINT}", "hint_list"),
    (f"## {HINT}", "hint_heading"),
    ("**Expand for details about**: migration gate and raw history", "hint_bold_label"),
    # This already-supported #612 form keeps its existing decoration tag.
    ("**Expand for details about:** migration gate and raw history", "hint_decoration"),
    (f'- "{HINT}"', "hint_list"),
])
def test_near_miss_is_normalized_at_level_one(monkeypatch, caplog, hint, tag):
    body, failed, tolerated = check(f"{LINES}\n{hint}")
    assert body == BODY and failed == "" and tag in tolerated
    install_provider(monkeypatch, lambda opening: f"{opening}\n{LINES}\n{hint}\n{CLOSE}")
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result, level = escalation.summarize_with_escalation(
            SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=80)
    assert (result, level) == (BODY, 1)
    assert any(tag in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("tokens", ["[SILENT]", "NO_REPLY", "[silent]\n\nno_reply"])
def test_trailing_controls_are_dropped(monkeypatch, tokens):
    body, failed, tolerated = check(f"{BODY}\n{tokens}")
    assert (body, failed, tolerated) == (BODY, "", ("hint_trailing_control",))
    install_provider(monkeypatch, lambda opening: f"{opening}\n{BODY}\n{tokens}\n{CLOSE}")
    assert escalation.summarize_with_escalation(
        SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=80) == (BODY, 1)


def test_near_miss_before_controls_combines_tags():
    body, failed, tolerated = check(f"{LINES}\n- {HINT}\n[SILENT]\nNO_REPLY")
    assert body == BODY and failed == ""
    assert {"hint_list", "hint_trailing_control"} <= set(tolerated)


@pytest.mark.parametrize("tail", [
    "content after the hint", "[SILENT] extra",
    "[SILENT]\nNO_REPLY\nSILENT", "content\n[SILENT]", "[SILENT]\ncontent",
    "[SILENT", "NO_REPLY]",
])
def test_other_trailing_lines_still_reject(tail):
    assert check(f"{BODY}\n{tail}") == ("", "closing_hint", ())


def test_missing_hint_and_envelope_still_reject():
    assert check(LINES) == ("", "closing_hint", ())
    assert escalation._check_summary_contract(BODY, NONCE, 160) == ("", "envelope", ())


@pytest.mark.parametrize("body", [
    "- Expand for details about: x", "## Expand for details about: x",
    "**Expand for details about**: x", "Expand for details about: x\n[SILENT]\nNO_REPLY",
])
def test_normalized_short_body_still_rejects(body):
    assert check(body) == ("", "short_body", ())


@pytest.mark.parametrize(("hint", "shape"), [
    ("- Expand for details about:PRIVATE", "list"),
    ("## Expand for details about:PRIVATE", "heading"),
    ("**Expand for details about**:PRIVATE", "bold"),
    (f"{HINT}\nPRIVATE after the hint", "trailing"),
    (f"{HINT}\n\nPRIVATE first line\n\nPRIVATE last line", "trailing"),
    (f"{HINT}\nPRIVATE one\nPRIVATE two\nPRIVATE three", "missing"),
    (f"{HINT}\nPRIVATE one\nPRIVATE two\nPRIVATE three\nPRIVATE four\nPRIVATE five", "missing"),
    ("PRIVATE no hint", "missing"),
    ("Expand for details about:PRIVATE", "other"),
])
def test_shape_info_is_content_free_and_warning_is_unchanged(monkeypatch, caplog, hint, shape):
    install_provider(monkeypatch, lambda opening: f"{opening}\n{LINES}\n{hint}\n{CLOSE}")
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        prompt = escalation._build_l1_prompt(SOURCE, token_budget=80, depth=0)
        assert escalation._call_llm_for_summary(prompt, 160, model="tier-x") == ""
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "LCM summary discarded output that violated the integrity contract "
        "(model=tier-x, check=closing_hint); escalating",
        f"LCM summary contract shape: check=closing_hint last_line={shape}",
    ]
    assert caplog.records[1].levelno == logging.INFO
    assert all("PRIVATE" not in message and LINES not in message for message in messages)


def test_chain_appends_contract_check_but_empty_output_does_not(monkeypatch, caplog):
    install_provider(monkeypatch, lambda opening: f"{opening}\n{LINES}\n{CLOSE}")
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        escalation.summarize_with_escalation(
            SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=80)
    chain = [record.getMessage() for record in caplog.records if "LCM summary result rejected" in record.getMessage()]
    assert len(chain) == 2 and all("reason=no_content contract=closing_hint" in line for line in chain)
    caplog.clear()
    install_provider(monkeypatch, lambda opening: "")
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        escalation.summarize_with_escalation(
            SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=80)
    chain = [record.getMessage() for record in caplog.records if "LCM summary result rejected" in record.getMessage()]
    assert len(chain) == 2 and all("reason=no_content" in line and "contract=" not in line for line in chain)
