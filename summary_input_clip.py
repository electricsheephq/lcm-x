"""Shared summariser input budget (#611 / #440), including externalised previews."""
from __future__ import annotations

from typing import List

from .tokens import count_tokens

CLIP_MARKER = "\n...[truncated]...\n"


def externalized_preview(content: str) -> str:
    """Budgeted text appended after, never inside, an externalised placeholder."""
    return f"\n[preview: {content[:600]} … {content[-300:]}]" if content else ""


def clip_to_budget(texts: List[str], budget_tokens: int) -> List[str]:
    """When the texts exceed ``budget_tokens`` together, each keeps ``budget_tokens / total`` of its
    tokens (by character share), as head and tail. Labels, placeholders and markers are outside the budget."""
    # #899: a single clipped text fills leaf_chunk_tokens, outside the #722
    # verbatim window while leaf_chunk_tokens > 2 * l3_truncate_tokens
    # (defaults: 20,000 > 1,024; fleet: 8,000 > 1,024).
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
