"""#612: the summary envelope accepts three named formatting mistakes and names the check a rejection failed.

Accepted: the ``</summary>`` closer instead of ``</lcm-summary>``, one ``<summary>`` wrapper around the whole
body, and one layer of quotes or emphasis on the closing hint. The exact nonce opening tag, its uniqueness, the
body minimum and a recognised closing hint are still required.
#1163 also accepts ``unclosed_envelope`` only when every remaining check passes."""

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
], ids=["no-envelope", "wrong-nonce", "outside-text", "two-opening-tags", "closer-lcm",
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


# -- #1163: a missing closer is tolerated only when all remaining checks pass --------------------------------

@pytest.mark.parametrize(("reply", "body", "tolerated"), [
    (f"{OPEN}\n{PLAIN_BODY}", PLAIN_BODY, ("unclosed_envelope",)),
    (f'{OPEN}\n{LINES}\n"{HINT}"', PLAIN_BODY, ("unclosed_envelope", "hint_decoration")),
    # The existing </summary> closer path takes precedence, leaving the unmatched <summary> in the body.
    (f"{OPEN}\n<summary>\n{PLAIN_BODY}\n</summary>", f"<summary>\n{PLAIN_BODY}", ("summary_closer",)),
    (f"{OPEN}\n{PLAIN_BODY}\n[SILENT]", PLAIN_BODY, ("unclosed_envelope", "hint_trailing_control")),
], ids=["no-closer", "decorated-hint", "summary-closer-precedence", "trailing-control"])
def test_1163_unclosed_valid_reply_is_accepted(reply, body, tolerated):
    assert _check(reply) == (body, "", tolerated)
    assert escalation._unwrap_summary_contract(reply, NONCE, MAX_TOKENS) == body


@pytest.mark.parametrize("reply", [
    f"{OPEN}\n{LINES}",
    f"{OPEN}\n{LINES[:len(LINES) // 2]}",
    f"{OPEN}\n{PLAIN_BODY}\nThanks for reading.",
    f"{OPEN}\n{OPEN}\n{PLAIN_BODY}",
    f'<lcm-summary nonce="{"f" * 32}">\n{PLAIN_BODY}',
    f"{OPEN}\nExpand for details about: x",
    f"{OPEN}\n{PLAIN_BODY}\n{CLOSE}\nappended after the envelope\n{HINT}",
    f"{OPEN}\n{PLAIN_BODY}\n</summary>\nappended after the envelope\n{HINT}",
    f"{OPEN}\n<think>\n{PLAIN_BODY}",
    f"{OPEN}\nThinking process:\n{PLAIN_BODY}",
    f"{OPEN}\n<summary>\n<think>\n{PLAIN_BODY}",
    f"{OPEN}\n<summary>\n{PLAIN_BODY}",
    f"{OPEN}\nSummary follows:\n<think>\n{PLAIN_BODY}",
    f"{OPEN}\n{LINES}\nThinking process:\n{HINT}",
    f"{OPEN}\n**Thinking Process:**\n{PLAIN_BODY}",
    f"{OPEN}\n{LINES}\n> Reasoning process:\n{HINT}",
    f"{OPEN}\n- <think>\n{PLAIN_BODY}",
    f"{OPEN}\n1. _Chain of thought:_\n{PLAIN_BODY}",
    f"{OPEN}\nSummary: <think>\n{PLAIN_BODY}",
    f"{OPEN}\n{LINES}\n<analysis>\nThe system prompt says.\n{HINT}",
    f"{OPEN}\n{LINES}\n<scratchpad note=\"x\">\n{HINT}",
    f"{OPEN}\n{LINES}\n<|channel|>analysis\n{HINT}",
    f"{OPEN}\n{LINES}\n[thinking]\n{HINT}",
    f'{OPEN}\n{LINES}\n<lcm-summary nonce="{"f" * 32}">\n{PLAIN_BODY}',
    f"{OPEN}\nSummary: < think>\n{PLAIN_BODY}",
    f"{OPEN}\n{LINES}\nNotes: < Thinking >\n{HINT}",
    f"{OPEN}\n{LINES}\nThen <| think |> it went on\n{HINT}",
    f"{OPEN}\n{LINES}\nDone. </ think> and more\n{HINT}",
], ids=["missing-hint", "truncated-mid-body", "chatter-after-hint", "two-opening-tags", "wrong-nonce", "short-body",
        "text-after-closer", "text-after-summary-closer", "unclosed-think", "reasoning-header", "wrapper-hides-think",
        "unmatched-wrapper", "think-after-a-line", "reasoning-header-before-hint",
        "bold-reasoning-header", "quoted-reasoning-header", "listed-think", "numbered-emphasis-header",
        "inline-think", "analysis-tag", "scratchpad-tag", "channel-token", "bracket-thinking", "other-nonce-opener",
        "inline-spaced-think", "inline-spaced-thinking", "inline-spaced-pipe-think", "inline-spaced-think-closer"])
def test_1163_unclosed_invalid_reply_keeps_envelope_failure(reply):
    assert _check(reply) == ("", "envelope", ())
    assert escalation._unwrap_summary_contract(reply, NONCE, MAX_TOKENS) == ""


def test_1163_unclosed_reply_logs_one_info_line(monkeypatch, caplog):
    _install_call_llm(monkeypatch, lambda opening: f"{opening}\n{PLAIN_BODY}")
    assert _summarize(caplog) == PLAIN_BODY
    assert [r.getMessage() for r in caplog.records] == [
        "LCM summary contract: tolerated unclosed_envelope (model=tier-x)"]
    assert caplog.records[0].levelno == logging.INFO


def test_1163_unclosed_reasoning_inside_the_envelope_is_discarded(monkeypatch, caplog):
    _install_call_llm(monkeypatch, lambda opening: f"{opening}\n<think>\nThe system prompt says to summarise.\n{PLAIN_BODY}")
    assert _summarize(caplog) == ""
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == [
        "LCM summary discarded output that violated the integrity contract (model=tier-x, check=envelope); "
        "escalating"]


def test_1163_closer_quoted_inside_a_closed_reasoning_block_is_not_a_close(monkeypatch, caplog):
    # The model never closed the envelope; the closed reasoning block is stripped before the contract check.
    _install_call_llm(monkeypatch, lambda opening: f"{opening}\n{LINES}\n<think>quoted token: {CLOSE}</think>\n{HINT}")
    result = _summarize(caplog)
    assert result.endswith(HINT) and LINES in result and "quoted token" not in result


@pytest.mark.parametrize("line", ["Keep x < y and y > z in the budget check.", "Use a -> b, then b <= c.",
                                  "Compare a < thinker > b in the ranking."])
def test_1163_unclosed_body_with_plain_comparisons_is_still_accepted(line):
    body = f"{LINES}\n- {line}\n{HINT}"
    assert _check(f"{OPEN}\n{body}") == (body, "", ("unclosed_envelope",))
