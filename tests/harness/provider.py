"""Offline token counting and provider-only summary scripting."""

from functools import lru_cache
from typing import NamedTuple

import hermes_lcm.escalation as escalation
import hermes_lcm.survival_fit as survival_fit
import hermes_lcm.tokens as tokens


def pin_char_counter(monkeypatch):
    """Use the real fallback through every imported count_tokens alias.

    As in test_issue_947_clipped_no_verbatim's estimate case, pin encoder state;
    as in test_issue_628_summary_circuit, disable the host estimator. An isolated
    cache prevents earlier encoder counts and later fallback counts leaking.
    """
    monkeypatch.setattr(tokens, "_encoder_ready", True)
    monkeypatch.setattr(tokens, "_encoder", None)
    # An encoder load still running from an earlier test can set _encoder later; the seam stays None.
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(tokens, "_count_tokens_cached",
                        lru_cache(maxsize=2048)(tokens._count_tokens_cached.__wrapped__))
    monkeypatch.setattr(survival_fit, "_host_estimate", lambda messages: None)


class SummaryCall(NamedTuple):
    model: str
    max_tokens: int
    prompt_length: int


class ScriptedSummarizer:
    """Provider seam fake; escalation, circuit, leaf loop, and logging stay real.

    Scripted replies repeat the final value; None represents provider failure.
    Alternatively ``fn`` receives the provider's arguments and supplies replies.
    Healthy text is the accepted provider result from test_issue_628_summary_circuit.
    """

    healthy_reply = "Earlier turns.\nExpand for details about: turns"

    def __init__(self, mode="healthy", replies=None, fn=None):
        if mode not in {"healthy", "outage", "scripted"}:
            raise ValueError(f"Unknown summarizer mode: {mode}")
        self.mode = mode
        self.replies = list(replies) if replies is not None else []
        self.fn = fn
        if mode == "scripted" and not self.replies and fn is None:
            raise ValueError("scripted mode needs replies or fn")
        self.calls: list[SummaryCall] = []

    def install(self, monkeypatch):
        monkeypatch.setattr(escalation, "_call_llm_for_summary", self)
        return self

    def __call__(self, prompt, max_tokens, model="", timeout=None, reasoning_effort=""):
        index = len(self.calls)
        self.calls.append(SummaryCall(model, max_tokens, len(prompt)))
        if self.mode == "outage":
            return None
        if self.mode == "healthy":
            return self.healthy_reply
        if self.fn is not None:
            return self.fn(prompt, max_tokens, model=model, timeout=timeout,
                           reasoning_effort=reasoning_effort)
        return self.replies[min(index, len(self.replies) - 1)]
