"""Pure summary-chain decisions; the caller owns clocks and circuit effects (I7)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

from .summary_attempt import (
    SummaryAttempt,
    is_summary_content_filter_error,
    is_summary_context_length_error,
    is_summary_route_config_error,
)


class SummaryOutcomeKind(Enum):
    BUDGET_CUT = "budget_cut"
    ACCEPTED = "accepted"
    VERBATIM_SMALL_SOURCE = "verbatim_small_source"
    REJECTED = "rejected"
    CONFIG_ERROR = "config_error"
    CONTENT_FILTER = "content_filter"
    CONTEXT_LENGTH = "context_length"
    FAILURE = "failure"
    REJECTION = "rejection"


@dataclass(frozen=True)
class SummaryOutcome:
    kind: SummaryOutcomeKind
    reason: Literal["", "not_shorter", "no_content"] = ""
    hang: bool = False
    record_hang: bool = False


@dataclass(frozen=True)
class SummaryOutcomeFacts:
    budget_present: bool
    budget_bound: bool
    call_timeout: float | None
    configured_timeout: float
    hard: float
    reserve: float
    leaf_reserve: float
    usable_deadline: float
    now: float
    breaker_present: bool
    accepts_result: bool
    verbatim_small_source: bool
    source_tokens: int | None
    result_tokens: int
    context_length_rescue: bool


def is_budget_cut(
    error: BaseException | None, *, budget_bound: bool, wall: float, call_timeout: float | None,
    deadline_installed: bool, now: float, usable_deadline: float,
) -> bool:
    """#605 F4: a timeout is a budget cut when the budget's limit bound the call (its timeout was the usable
    time left, below the configured one, and it ran out or the usable deadline passed), or the host deadline
    fired. A configured-timeout hit stays an ordinary route failure, even one that ends near the deadline.
    A call that ends at the usable deadline under a budget-bound timeout or an installed host deadline is a
    budget cut whatever its error says, because a host fallback can mask the timeout (#1120).
    A call that outlives its configured timeout and ends at the usable deadline was bounded by the budget, for
    example on a host whose timeout is a no-progress window; only where that deadline was installed in the host
    (#1099), so a provider that ignores its timeout on a host without the seam stays a route failure."""
    message = str(error or "").lower()
    if error is None:
        return False
    timeout_like = isinstance(error, TimeoutError) or "timed out" in message or "timeout" in message
    at_deadline = now >= usable_deadline - 1.0
    return bool(
        (timeout_like and "host compression deadline" in message)
        or (budget_bound and at_deadline)
        or (timeout_like and budget_bound and call_timeout is not None and wall >= call_timeout - 1.0)
        or (deadline_installed and call_timeout is not None and wall > call_timeout + 1.0
            and at_deadline)
    )


def classify_summary_outcome(attempt: SummaryAttempt, facts: SummaryOutcomeFacts) -> SummaryOutcome:
    """Classify captured values without logging, consulting a clock or changing a circuit.

    REJECTED and REJECTION share rejection reasons; the latter requests a breaker
    rejection event. Budgeted attempts always have an ended clock observation.
    """
    result, error = attempt.text, attempt.error
    if facts.budget_present and result is None:
        started, ended = attempt.timing.started, cast(float, attempt.timing.ended)
        deadline_installed = attempt.timing.deadline_installed
        wall, call_timeout, budget_bound = ended - started, facts.call_timeout, facts.budget_bound
        budget_cut = is_budget_cut(
            error, budget_bound=budget_bound, wall=wall, call_timeout=call_timeout,
            deadline_installed=deadline_installed, now=facts.now, usable_deadline=facts.usable_deadline,
        )
        if budget_cut:
            full_window = min(facts.configured_timeout, facts.hard - facts.reserve - facts.leaf_reserve)
            if not budget_bound:  # host timeout may be a shorter no-progress window
                full_window = facts.hard - facts.reserve - facts.leaf_reserve
            used_window = (
                (budget_bound and cast(float, call_timeout) >= 0.9 * full_window)
                or (deadline_installed and ended >= facts.usable_deadline - 1.0)
            )
            hang = bool(used_window and ended - started >= 0.9 * full_window)
            return SummaryOutcome(SummaryOutcomeKind.BUDGET_CUT, hang=hang,
                                  record_hang=facts.breaker_present and hang)
    if result and facts.accepts_result:
        return SummaryOutcome(SummaryOutcomeKind.ACCEPTED)
    if (result and facts.verbatim_small_source and facts.source_tokens is not None
            and facts.result_tokens >= facts.source_tokens):
        return SummaryOutcome(SummaryOutcomeKind.VERBATIM_SMALL_SOURCE)
    if result is not None:
        return SummaryOutcome(
            SummaryOutcomeKind.REJECTION if facts.breaker_present else SummaryOutcomeKind.REJECTED,
            reason="not_shorter" if result else "no_content",
        )
    if is_summary_route_config_error(error):
        return SummaryOutcome(SummaryOutcomeKind.CONFIG_ERROR)
    if is_summary_content_filter_error(error):
        return SummaryOutcome(SummaryOutcomeKind.CONTENT_FILTER)
    if facts.context_length_rescue and is_summary_context_length_error(error):
        return SummaryOutcome(SummaryOutcomeKind.CONTEXT_LENGTH)
    return SummaryOutcome(SummaryOutcomeKind.FAILURE)
