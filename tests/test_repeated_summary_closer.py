"""#1948: a repeated ``</lcm-summary>`` closer is tolerated; every other envelope check is unchanged.

Small summary models (observed: a 4B hybrid-thinking model served through LM Studio) often end an
otherwise valid reply with the closer twice. Only one closer was sliced off, so the second became the
body's last line and the reply failed ``closing_hint``, logged as ``reason=no_content``. The closer
carries no nonce, so a run of identical closers at the very end is as safe as the single closer, or as
the missing closer #1163 already tolerates. The closing-hint check still anchors the body.
"""

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
    "Task and current state · The Longhorn health check finished; all volumes are healthy.\n"
    "Open items, blockers and the next step · Confirm the recurring backup job ran overnight."
)
HINT = "Expand for details about: Longhorn volume health and the recurring backup job"
BODY = f"{LINES}\n{HINT}"
SOURCE = "ordinary historical transcript " * 200


def _check(content: str) -> tuple[str, str, tuple[str, ...]]:
    return escalation._check_summary_contract(content, NONCE, MAX_TOKENS)


@pytest.mark.parametrize("reply", [
    f"{OPEN}\n{BODY}\n{CLOSE}\n{CLOSE}",
    f"{OPEN}\n{BODY}\n\n{CLOSE}\n{CLOSE}",
    f"{OPEN}\n{BODY}\n{CLOSE}\n\n{CLOSE}\n",
    f"{OPEN}\n{BODY}\n{CLOSE}{CLOSE}",
    f"{OPEN}\n{BODY}\n{CLOSE}\n{CLOSE}\n{CLOSE}",
], ids=["doubled", "blank-before", "blank-between", "same-line", "tripled"])
def test_repeated_closer_is_accepted_with_the_plain_body(reply):
    body, check, tolerated = _check(reply)
    assert (body, check, tolerated) == (BODY, "", ("repeated_closer",))
    assert escalation._unwrap_summary_contract(reply, NONCE, MAX_TOKENS) == BODY


def test_repeated_closer_combines_with_a_decorated_hint():
    body, check, tolerated = _check(f'{OPEN}\n{LINES}\n"{HINT}"\n{CLOSE}\n{CLOSE}')
    assert (body, check) == (BODY, "")
    assert tolerated == ("repeated_closer", "hint_decoration")


def test_a_single_closer_still_tolerates_nothing():
    assert _check(f"{OPEN}\n{BODY}\n{CLOSE}") == (BODY, "", ())


@pytest.mark.parametrize(("reply", "expected"), [
    # The hint must still be the body's final line.
    (f"{OPEN}\n{LINES}\n{CLOSE}\n{CLOSE}", "closing_hint"),
    # Text between closers is text written past a closed envelope, not a repeated closer.
    (f"{OPEN}\n{BODY}\n{CLOSE}\ntrailing\n{CLOSE}", "closing_hint"),
    # Mixed closers keep their existing outcome: only identical </lcm-summary> runs collapse.
    (f"{OPEN}\n{BODY}\n{CLOSE}\n</summary>", "closing_hint"),
    # A repeated closer never rescues a body that is too short or a missing/duplicated opener.
    (f"{OPEN}\nExpand for details about: x\n{CLOSE}\n{CLOSE}", "short_body"),
    (f"{BODY}\n{CLOSE}\n{CLOSE}", "envelope"),
    (f"{OPEN}\n{OPEN}\n{BODY}\n{CLOSE}\n{CLOSE}", "nonce_count"),
])
def test_other_violations_still_reject_with_their_check(reply, expected):
    body, check, tolerated = _check(reply)
    assert (body, check, tolerated) == ("", expected, ())


def test_summary_wrapper_closer_pair_is_not_collapsed():
    # #612: `</summary>\n</summary>` is the wrapper closer followed by the tolerated closer, not a repeat.
    reply = f"{OPEN}\n<summary>\n{BODY}\n</summary>\n</summary>"
    assert _check(reply) == (BODY, "", ("summary_closer", "summary_wrapper"))


def _install_call_llm(monkeypatch, make_reply):
    module = ModuleType("agent.auxiliary_client")

    def call_llm(**kwargs):
        nonce = re.search(r'nonce="([0-9a-f]{32})"', kwargs["messages"][0]["content"]).group(1)
        reply = make_reply(f'<lcm-summary nonce="{nonce}">')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    module.call_llm = call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", module)


def test_repeated_closer_summarizes_at_level_one_and_logs_one_info_line(monkeypatch, caplog):
    _install_call_llm(monkeypatch, lambda opening: f"{opening}\n{BODY}\n\n{CLOSE}\n{CLOSE}")
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result, level = escalation.summarize_with_escalation(
            SOURCE, source_tokens=escalation.count_tokens(SOURCE), token_budget=80)
    assert (result, level) == (BODY, 1)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [r.getMessage() for r in caplog.records if "tolerated" in r.getMessage()] == [
        "LCM summary contract: tolerated repeated_closer (model=<default>)"]
