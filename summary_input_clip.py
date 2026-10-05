"""Summariser input clip (#611 / #440): how much of each message text reaches the summariser.

``LCM_SUMMARY_INPUT_CLIP`` selects the arm; every ``_serialize_messages`` consumer (the leaf call, extraction,
the route-stop ``verbatim_source`` check and level-3 repair) goes through it.

- ``legacy`` (default): a message over 3,000 characters keeps its first 2,000 and last 800; tool-call
  arguments over 500 characters keep their first 400.
- ``whole12k``: a message of at most 12,000 characters goes whole; above that, the first 8,000 and the last
  2,000. An externalised tool result gets a head/tail preview after its placeholder. Arguments keep the legacy rule.
- ``budget``: the chunk's texts (message contents and tool-call arguments) share ``leaf_chunk_tokens``; when they
  exceed it, every text keeps the same proportional share of its tokens, as head and tail (the legacy 5:2 split).
"""
from __future__ import annotations

from typing import List

from .tokens import count_tokens

SUMMARY_INPUT_CLIP_MODES = ("legacy", "whole12k", "budget")
CLIP_MARKER = "\n...[truncated]...\n"


def summary_input_clip_mode(config) -> str:
    mode = str(getattr(config, "summary_input_clip", "legacy") or "legacy").strip().lower()
    return mode if mode in SUMMARY_INPUT_CLIP_MODES else "legacy"


def clip_message_text(text: str, mode: str) -> str:
    """The per-message rule of ``legacy`` and ``whole12k`` (``budget`` clips in :func:`clip_to_budget`)."""
    if mode == "whole12k":
        return text if len(text) <= 12_000 else text[:8_000] + CLIP_MARKER + text[-2_000:]
    if mode == "budget":
        return text
    return text if len(text) <= 3_000 else text[:2_000] + CLIP_MARKER + text[-800:]


def clip_tool_arguments(args: str, mode: str) -> str:
    if mode == "budget":
        return args
    return args if len(args) <= 500 else args[:400] + "..."


def externalized_preview(content: str, mode: str) -> str:
    """``whole12k`` only: the text appended after (never inside) an externalised result's placeholder."""
    if mode != "whole12k" or not content:
        return ""
    return f"\n[preview: {content[:600]} … {content[-300:]}]"


def clip_to_budget(texts: List[str], budget_tokens: int) -> List[str]:
    """``budget``: when the texts exceed ``budget_tokens`` together, each keeps ``budget_tokens / total`` of its
    tokens (by character share), as head and tail. Labels, placeholders and markers are outside the budget."""
    counts = [count_tokens(text) if text else 0 for text in texts]
    total, budget = sum(counts), max(1, int(budget_tokens))
    if total <= budget:
        return list(texts)
    out = []
    for text, tokens in zip(texts, counts):
        keep = len(text) * budget // total if tokens else 0
        if keep >= len(text):
            out.append(text)
            continue
        head = keep * 5 // 7
        tail = keep - head
        out.append(text[:head] + CLIP_MARKER + (text[-tail:] if tail else ""))
    return out
