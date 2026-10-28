"""Shared summariser input budget (#611 / #440), including externalised previews."""
from __future__ import annotations

from typing import List, Optional

from .tokens import count_tokens

CLIP_MARKER = "\n...[truncated]...\n"


def externalized_preview(content: str) -> str:
    """Budgeted text appended after, never inside, an externalised placeholder."""
    if not content:
        return ""
    if len(content) <= 900:
        return f"\n[preview: {content}]"
    return f"\n[preview: {content[:600]} … {content[-300:]}]"


def clip_to_budget(texts: List[str], budget_tokens: int, kinds: Optional[List[str]] = None) -> List[str]:
    """Keep message/argument legacy floors; previews share the budget with no floor.

    ``kinds`` distinguishes message contents, arguments and externalized previews.
    Labels, placeholders and share clip markers sit outside the text budget.
    """
    # Messages and arguments keep at least as many characters as v0.26.0;
    # previews have no legacy floor. Text tokens are bounded by the legacy
    # total plus leaf_chunk_tokens (2% token-share allowance).
    # #899: a single proportional clip targets leaf_chunk_tokens, outside the #722
    # verbatim window while leaf_chunk_tokens > 2 * l3_truncate_tokens
    # (defaults: 20,000 > 1,024; fleet: 8,000 > 1,024).
    counts = [count_tokens(text) if text else 0 for text in texts]
    total, budget = sum(counts), max(1, int(budget_tokens))
    if total <= budget:
        return list(texts)
    out = []
    for index, (text, tokens) in enumerate(zip(texts, counts)):
        kind = kinds[index] if kinds is not None else "message"
        if kind == "preview":
            floor, floor_chars = "", 0
        elif len(text) <= (500 if kind == "arguments" else 3_000):
            floor, floor_chars = text, len(text)
        elif kind == "arguments":
            floor, floor_chars = text[:400] + "...", 400
        else:
            floor, floor_chars = text[:2_000] + CLIP_MARKER + text[-800:], 2_800
        keep = len(text) * budget // total if tokens else 0
        if kind == "preview" and keep < 200:
            out.append("")
            continue
        if keep <= floor_chars:
            out.append(floor)
            continue
        if keep >= len(text):
            out.append(text)
            continue
        head = keep * 5 // 7
        tail = keep - head
        share = (tokens * budget + total - 1) // total
        target = share * 102 // 100
        if count_tokens(text[:head] + (text[-tail:] if tail else "")) > target:
            floor_text = floor if kind in ("arguments", "preview") else text[:2_000] + text[-800:]
            if count_tokens(floor_text) > target:
                out.append(floor)
                continue
            low, high = floor_chars, keep
            while high - low > 1:
                keep = (low + high) // 2
                head = keep * 5 // 7
                tail = keep - head
                if count_tokens(text[:head] + (text[-tail:] if tail else "")) <= target:
                    low = keep
                else:
                    high = keep
            if low == floor_chars or (kind == "preview" and low < 200):
                out.append(floor)
                continue
            head = low * 5 // 7
            tail = low - head
        out.append(text[:head] + CLIP_MARKER + (text[-tail:] if tail else ""))
    return out
