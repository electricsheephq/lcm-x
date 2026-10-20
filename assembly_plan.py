"""Pure prefix and tail budgeting and selection; assembly owns reads and payload writes."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .extraction import strip_injected_context_blocks
from .message_analysis import _tool_call_id


def default_prefix_budget(
    *, ceiling: int, overhead: int, leading_tokens: int, tail_tokens: int,
    repaired_tail_tokens: int, recall_budget_tokens: int | None,
    proactive_recall_enabled: bool, embeddings_enabled: bool, survival_fit: bool,
) -> int:
    """Reserve the measured tail/recall; the quarter floor requires later fitting."""
    return max(
        ceiling - overhead - leading_tokens - max(tail_tokens, repaired_tail_tokens)
        - (max(0, int(recall_budget_tokens or 0))
           if proactive_recall_enabled and embeddings_enabled else 0),
        ceiling // 4 if survival_fit else 0,
    )


@dataclass(frozen=True)
class PrefixBudgetFacts:
    """Resolved budget and ordering facts, after lazy objective fallback preparation.

    None means unbounded; zero is a real budget. The counter is explicit so the
    caller's token-counting policy applies to the planner and the assembly shell.
    """

    part_keys: Sequence[tuple[int, int | None]]
    summary_role: str
    summary_budget: int | None
    assembly_cap: int | None
    survival_fit: bool
    count_message_tokens: Callable[[dict], int]


@dataclass(frozen=True)
class PrefixSelection:
    kept_indexes: tuple[int, ...]
    selected_parts: tuple[str, ...]


def select_prefix(parts: Sequence[str], budget_facts: PrefixBudgetFacts) -> PrefixSelection:
    """Select by group/newness, render in source order, and leave inputs intact.

    The engine resolves the default budget and tries objective fallbacks first:
    preparing those fallbacks can externalize payloads and must remain lazy.
    """
    summary_parts = list(parts)
    part_keys = budget_facts.part_keys
    summary_role = budget_facts.summary_role
    summary_budget = budget_facts.summary_budget
    count_message_tokens = budget_facts.count_message_tokens
    anchor_exempt = (
        summary_budget is not None and budget_facts.assembly_cap is None and budget_facts.survival_fit
        and bool(summary_parts) and part_keys[0] == (-1, None) and count_message_tokens({
            "role": summary_role, "content": strip_injected_context_blocks(summary_parts[0]),
        }) > summary_budget
    )
    kept_indexes = list(range(len(summary_parts)))
    if anchor_exempt:
        summary_parts[0] = strip_injected_context_blocks(summary_parts[0])
        kept_indexes = [0]
    elif summary_budget is not None:
        # Within a depth keep newest parts first; render in the original order.
        kept_indexes = []
        for index in sorted(range(len(summary_parts)), key=lambda i: (part_keys[i][0], -i)):
            candidate = "\n\n---\n\n".join(
                strip_injected_context_blocks(summary_parts[i])
                if part_keys[i] == (-1, None) and count_message_tokens({
                    "role": summary_role, "content": summary_parts[i],
                }) > summary_budget else summary_parts[i]
                for i in sorted(kept_indexes + [index]))
            candidate_msg = {"role": summary_role, "content": candidate}
            if count_message_tokens(candidate_msg) > summary_budget:
                continue
            kept_indexes.append(index)
        kept_indexes.sort()
    return PrefixSelection(tuple(kept_indexes), tuple(summary_parts[i] for i in kept_indexes))


@dataclass(frozen=True)
class TailSelection:
    """Source-order indexes and their charge, including reserved missing results."""

    indexes: tuple[int, ...]
    charged_tokens: int


def select_tail(
    prepared_rows: Sequence[dict], cap: int, anchor_cost: int,
    count_message_tokens: Callable[[dict], int], *,
    count_messages_tokens: Callable[[list[dict]], int],
    missing_tool_result_stub: Callable[[str], dict],
    is_budget_droppable_tail_message: Callable[[dict], bool],
) -> TailSelection:
    """Select an ordinary capped tail without copying or changing its rows.

    Rows have been sanitized without merging adjacent assistants or inserting
    missing results. The shell supplies pure stub/eviction policies and both
    counters, preserving its measurement order and batch-counting policy.
    Forced-recovery stubbing and final sanitation remain in the shell.
    """
    kept_reversed: list[int] = []
    tail_token_total = 0
    skipped_tail_gap = False
    for index in range(len(prepared_rows) - 1, -1, -1):
        msg = prepared_rows[index]
        msg_tokens = count_message_tokens(msg)
        own_results_start: int | None = None
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # The final sanitizer inserts a plain stub for each call whose result is not
            # in this turn's own result group (the tool rows right after it). Reserve them
            # too, or a full summary budget overflows after sanitizing and the over-cap
            # pass drops the preserved objective first (#1048).
            own_results_start = len(kept_reversed)
            while own_results_start and prepared_rows[kept_reversed[own_results_start - 1]].get("role") == "tool":
                own_results_start -= 1
            # Count, don't collapse: the sanitizer pairs each call occurrence separately.
            unmatched_results = Counter(
                str(prepared_rows[i].get("tool_call_id") or "").strip()
                for i in kept_reversed[own_results_start:]
            )
            missing_ids = []
            for call_id in (_tool_call_id(tc) for tc in msg["tool_calls"]):
                if not call_id:
                    continue
                if unmatched_results[call_id]:
                    unmatched_results[call_id] -= 1
                else:
                    missing_ids.append(call_id)
            if missing_ids:
                msg_tokens += count_messages_tokens(
                    [missing_tool_result_stub(call_id) for call_id in missing_ids])
        if anchor_cost + tail_token_total + msg_tokens > cap:
            if own_results_start is not None and own_results_start < len(kept_reversed):
                # The turn is rejected: its kept results would be orphans the final
                # sanitizer drops, so stop charging them to the tail budget.
                tail_token_total -= sum(count_message_tokens(prepared_rows[i]) for i in kept_reversed[own_results_start:])
                del kept_reversed[own_results_start:]
            if is_budget_droppable_tail_message(msg):
                skipped_tail_gap = True
                continue
            break
        if skipped_tail_gap:
            break
        kept_reversed.append(index)
        tail_token_total += msg_tokens
    return TailSelection(tuple(reversed(kept_reversed)), tail_token_total)
