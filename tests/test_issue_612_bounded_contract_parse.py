"""#612: the summary envelope accepts three named formatting mistakes and names the check a rejection failed.

Accepted: the ``</summary>`` closer instead of ``</lcm-summary>``, one ``<summary>`` wrapper around the whole
body, and one layer of quotes or emphasis on the closing hint. The exact nonce opening tag, its uniqueness, the
body minimum and a recognised closing hint are still required."""

from __future__ import annotations

import logging
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

import hermes_lcm.escalation as escalation

NONCE = "0123456789abcdef0123456789abcdef"
OPEN = f'<lcm-summary nonce="{NONCE}">'
CLOSE = "</lcm-summary>"
MAX_TOKENS = 160
LINES = (
    "- Decision: keep the owner approval gate before the schema migration runs.\n"
    "- Current state: raw history remains available for exact recovery."
)
HINT = "Expand for details about: migration gate and raw history"
PLAIN_BODY = f"{LINES}\n{HINT}"
PLACEHOLDER = "<summary body ending with the required 'Expand for details about:' line>"


def _check(content: str) -> tuple[str, str, tuple[str, ...]]:
    return escalation._check_summary_contract(content, NONCE, MAX_TOKENS)


# -- T1: each named mistake is accepted and the body ends with the plain hint ---------------------------------

@pytest.mark.parametrize(("reply", "variant"), [
    (f"{OPEN}\n{PLAIN_BODY}\n</summary>", "summary_closer"),
    (f"{OPEN}\n<summary>\n{PLAIN_BODY}\n</summary>\n{CLOSE}", "summary_wrapper"),
    (f'{OPEN}\n{LINES}\n"{HINT}"\n{CLOSE}', "hint_decoration"),
    (f"{OPEN}\n{LINES}\n'{HINT}'\n{CLOSE}", "hint_decoration"),
    (f"{OPEN}\n{LINES}\n**Expand for details about:** migration gate and raw history\n{CLOSE}", "hint_decoration"),
    (f"{OPEN}\n{LINES}\n`{HINT}`\n{CLOSE}", "hint_decoration"),
    (f"{OPEN}\n{LINES}\n*{HINT}*\n{CLOSE}", "hint_decoration"),
    (f"{OPEN}\n{LINES}\n__Expand for details about:__ migration gate and raw history\n{CLOSE}", "hint_decoration"),
], ids=["closer", "wrapper", "double-quoted", "single-quoted", "bold-label", "backtick", "italic", "underscore-label"])
def test_t1_named_mistake_is_accepted_with_a_plain_hint(reply, variant):
    body, check, tolerated = _check(reply)
    assert check == "" and tolerated == (variant,)
    assert body == PLAIN_BODY
    assert escalation._unwrap_summary_contract(reply, NONCE, MAX_TOKENS) == PLAIN_BODY


def test_t1_the_three_mistakes_combine():
    reply = f'{OPEN}\n<summary>\n{LINES}\n"{HINT}"\n</summary>\n</summary>'
    # The closer is taken first, so the remaining body is a single wrapper pair around quoted-hint text.
    body, check, tolerated = _check(reply)
    assert check == "" and body == PLAIN_BODY
    assert tolerated == ("summary_closer", "summary_wrapper", "hint_decoration")


def test_t1_a_valid_reply_is_unchanged_and_tolerates_nothing():
    assert _check(f"{OPEN}\n{PLAIN_BODY}\n{CLOSE}") == (PLAIN_BODY, "", ())


# -- T2: everything else is still rejected, naming the failed check -------------------------------------------

@pytest.mark.parametrize(("reply", "expected"), [
    (PLAIN_BODY, "envelope"),
    (f'<lcm-summary nonce="{"f" * 32}">\n{PLAIN_BODY}\n{CLOSE}', "envelope"),
    (f"preamble\n{OPEN}\n{PLAIN_BODY}\n{CLOSE}", "envelope"),
    (f"{OPEN}\n{OPEN}\n{PLAIN_BODY}\n{CLOSE}", "nonce_count"),
    (f"{OPEN}\n{PLAIN_BODY}\n</lcm>", "envelope"),
    (f"{OPEN}\n{PLAIN_BODY}", "envelope"),
    (f"{OPEN}\n{PLAIN_BODY}\n</summary>trailing", "envelope"),
    (f"{OPEN}\nExpand for details about: x\n{CLOSE}", "short_body"),
    (f"{OPEN}\n{LINES}\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n{HINT}\n{LINES}\n{CLOSE}", "closing_hint"),
    (f'{OPEN}\n{LINES}\n"{HINT}\n{CLOSE}', "closing_hint"),
    (f"{OPEN}\n{LINES}\n{PLACEHOLDER}\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n{PLACEHOLDER}\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n<summary>\nshort\n</summary>\n{CLOSE}", "short_body"),
    (f"{OPEN}\n<summary>\n{LINES}\n</summary>\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n<summary>\n<summary>\n{PLAIN_BODY}\n</summary>\n</summary>\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n{LINES}\n**Expand for details about:**migration gate\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n{LINES}\n**{HINT}*\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n{LINES}\n\"\"{HINT}\"\"\"\n{CLOSE}", "closing_hint"),
    (f"{OPEN}\n{LINES}\n{'_' * 50_000}\n{CLOSE}", "closing_hint"),
], ids=["no-envelope", "wrong-nonce", "outside-text", "two-opening-tags", "closer-lcm", "no-closer",
        "closer-not-last", "short-body", "missing-hint", "hint-not-last", "unbalanced-quote", "placeholder-echo",
        "placeholder-only", "wrapper-short-body", "wrapper-missing-hint", "nested-wrapper", "label-without-space",
        "unequal-emphasis", "unequal-quotes", "long-rule-line"])
def test_t2_other_violations_are_rejected_with_their_check(reply, expected):
    assert _check(reply) == ("", expected, ())
    assert escalation._unwrap_summary_contract(reply, NONCE, MAX_TOKENS) == ""


def test_t2_a_decorated_reply_is_accepted_only_if_its_plain_form_would_be():
    quotes = '"' * 32
    decorated = f"{OPEN}\n{quotes}Expand for details about: x{quotes}\n{CLOSE}"
    assert escalation.count_tokens(f"{quotes}Expand for details about: x{quotes}") >= 10  # passes the minimum
    assert _check(f"{OPEN}\nExpand for details about: x\n{CLOSE}") == ("", "short_body", ())
    assert _check(decorated) == ("", "short_body", ())
    # A real body before the decorated hint is still accepted with the same plain body.
    assert _check(f'{OPEN}\n{LINES}\n{quotes}{HINT}{quotes}\n{CLOSE}') == (PLAIN_BODY, "", ("hint_decoration",))


def test_t2_the_no_nonce_path_is_unchanged():
    assert escalation._unwrap_summary_contract("anything at all", "", MAX_TOKENS) == "anything at all"
    assert escalation._check_summary_contract("anything at all", "", MAX_TOKENS) == ("anything at all", "", ())


# -- T3: through _call_llm_for_summary with a stubbed call_llm -----------------------------------------------

SECRET = "SECRET_TRANSCRIPT_MARKER"


def _install_call_llm(monkeypatch, make_reply):
    module = ModuleType("agent.auxiliary_client")

    def call_llm(**kwargs):
        nonce = re.search(r'<lcm-summary nonce="([0-9a-f]{32})">', kwargs["messages"][0]["content"]).group(1)
        reply = make_reply(f'<lcm-summary nonce="{nonce}">')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)


def _summarize(caplog) -> str:
    prompt = escalation._build_l1_prompt(f"{SECRET} ordinary historical transcript " * 30, token_budget=80, depth=0)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        return escalation._call_llm_for_summary(prompt, MAX_TOKENS, model="tier-x")


def test_t3_tolerated_reply_is_returned_and_logs_one_info_line(monkeypatch, caplog):
    body = f"{LINES} {SECRET}\n\"{HINT}\""
    _install_call_llm(monkeypatch, lambda opening: f"{opening}\n{body}\n</summary>")
    result = _summarize(caplog)
    assert result == f"{LINES} {SECRET}\n{HINT}"
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["LCM summary contract: tolerated summary_closer+hint_decoration (model=tier-x)"]
    assert caplog.records[0].levelno == logging.INFO


def test_t3_violating_reply_logs_the_warning_with_the_check(monkeypatch, caplog):
    _install_call_llm(monkeypatch, lambda opening: f"{opening}\n{LINES} {SECRET}\n</lcm>")
    assert _summarize(caplog) == ""
    # #931: the shape line is a separate INFO record
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert messages == ["LCM summary discarded output that violated the integrity contract "
                        "(model=tier-x, check=envelope); escalating"]
    assert all(SECRET not in m and "Decision" not in m for m in messages)


def test_t3_short_and_hintless_replies_name_their_checks(monkeypatch, caplog):
    _install_call_llm(monkeypatch, lambda opening: f"{opening}\n{LINES} {SECRET}\n{CLOSE}")
    assert _summarize(caplog) == ""
    # #931: the shape line is a separate INFO record
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [
        "LCM summary discarded output that violated the integrity contract (model=tier-x, check=closing_hint); "
        "escalating"]
