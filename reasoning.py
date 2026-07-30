"""Provider-neutral typed planning, grounded computation, and answer verification.

Runtime inputs are limited to the question, optional question date, and exact
retrieved evidence references.  The compiler never inspects benchmark IDs,
reference answers, categories, or audit labels. Unsupported or ambiguous work
returns a typed fallback instead of a partial computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import calendar
import math
import re
from typing import Any, Literal, Mapping, Sequence

from .assertion_state import query_assertion_state
from .assertion_store import AssertionStore
from .store import MessageStore


ComputationOperation = Literal[
    "date_interval",
    "date_filter",
    "count_distinct",
    "sum",
    "difference",
    "order",
    "latest_fact",
]
PlanStatus = Literal["planned", "not_applicable", "fallback"]

_MAX_OPERANDS = 50
_MAX_QUESTION_CHARS = 4_000
_MAX_QUOTE_CHARS = 24_000
_MAX_LABEL_CHARS = 300
_MAX_CALLER_SESSION_DATE_NOTE_CHARS = 64
_MAX_TEMPORAL_TRUST_NOTES = 8
_MAX_TEMPORAL_TRUST_NOTE_CHARS = 512
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_COMPUTATION_TRIGGER_RE = re.compile(
    r"\b(how many|how much|count|total|sum|difference|more than|less than|ago|"
    r"how long|since|between|before|after|last|latest|previous|earliest|first|"
    r"second|third|chronolog|order|most recent|current|currently|now|usually|"
    r"average|mean|median|percentage|percent|rate)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_OPERATION_RE = re.compile(
    r"\b(average|mean|median|percentage|percent|rate|ratio|multiply|product|divide)\b",
    re.IGNORECASE,
)
_WORD_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_MONTHS = {
    name.lower(): number
    for number, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update({
    name.lower(): number
    for number, name in enumerate(calendar.month_abbr)
    if name
})


def resolve_occurrence_time_with_trust(
    text: Any,
    *,
    observed_at: Any,
    session_date: Any = None,
    engine: Any = None,
    session_id: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lazy import keeps plugin bootstrap order independent of this optional parser."""
    from .occurrence_time import resolve_occurrence_time as _resolve

    caller_session_date = str(session_date or "").strip() or None
    sidecar_dates = getattr(engine, "_session_occurrence_dates", {}) or {}
    session_key = str(session_id or "")
    sidecar_session_date = (
        str(sidecar_dates.get(session_key) or "").strip() or None
        if isinstance(sidecar_dates, Mapping) and session_key
        else None
    )
    overridden = bool(
        sidecar_session_date
        and caller_session_date
        and caller_session_date != sidecar_session_date
    )
    caller_session_date_note = caller_session_date
    if (
        caller_session_date_note
        and len(caller_session_date_note) > _MAX_CALLER_SESSION_DATE_NOTE_CHARS
    ):
        caller_session_date_note = (
            caller_session_date_note[:_MAX_CALLER_SESSION_DATE_NOTE_CHARS - 3]
            + "..."
        )
    anchor = sidecar_session_date or caller_session_date
    result = _resolve(text, observed_at=observed_at, session_date=anchor)
    source = str(result.get("event_time_source") or "unknown")
    reason = str(result.get("reason") or "")
    trust: dict[str, Any] = {
        "anchor_trust": "not_applicable",
        "temporal_certified": None,
        "session_date_overridden": False,
    }
    if source == "explicit":
        # An explicit date is self-anchoring. Sidecar availability or validity
        # cannot change its certification.
        trust["temporal_certified"] = True
    elif source == "relative_to_session" and sidecar_session_date:
        trust["anchor_trust"] = "engine_sidecar"
        trust["temporal_certified"] = True
        trust["session_date_overridden"] = overridden
        if overridden:
            trust["trust_note"] = (
                f"caller session_date {caller_session_date_note} overridden by "
                f"engine sidecar {sidecar_session_date}"
            )
    elif (
        source == "relative_to_session"
        or reason == "relative_expression_without_session_date"
    ):
        trust["anchor_trust"] = "low_trust"
        trust["temporal_certified"] = False
        if sidecar_session_date:
            trust["trust_note"] = (
                f"engine occurrence-date sidecar invalid for session "
                f"{session_key or '<unknown>'}; temporal result is low-trust"
            )
        else:
            trust["trust_note"] = (
                f"engine occurrence-date sidecar absent for session "
                f"{session_key or '<unknown>'}; temporal result is low-trust"
            )
    return result, trust


def temporal_trust_wire(
    status: Any,
    certified: Any,
    notes: Sequence[Any] = (),
) -> dict[str, Any]:
    """Return the one bounded temporal-trust shape used on public tool wires."""
    normalized_status = str(status or "not_applicable")
    if normalized_status not in {
        "not_applicable",
        "engine_sidecar",
        "low_trust",
    }:
        normalized_status = "low_trust"
    normalized_certified = certified if isinstance(certified, bool) else None
    normalized_notes: list[str] = []
    for value in notes:
        note = str(value or "").strip()[:_MAX_TEMPORAL_TRUST_NOTE_CHARS]
        if note and note not in normalized_notes:
            normalized_notes.append(note)
        if len(normalized_notes) >= _MAX_TEMPORAL_TRUST_NOTES:
            break
    return {
        "status": normalized_status,
        "certified": normalized_certified,
        "notes": normalized_notes,
    }


def resolve_occurrence_time(
    text: Any,
    *,
    observed_at: Any,
    session_date: Any = None,
    engine: Any = None,
    session_id: Any = None,
) -> dict[str, Any]:
    """Return only the operand-shaped occurrence object.

    Trust metadata is deliberately kept out of this object so an answer-ready
    recall hit can pass it to ``lcm_compute`` without changing wire shape.
    """
    occurrence, _trust = resolve_occurrence_time_with_trust(
        text,
        observed_at=observed_at,
        session_date=session_date,
        engine=engine,
        session_id=session_id,
    )
    return occurrence


@dataclass(frozen=True)
class TemporalWindow:
    start: date
    end: date
    description: str

    def as_dict(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "description": self.description,
        }


@dataclass(frozen=True)
class EvidencePlan:
    operation: ComputationOperation
    minimum_operands: int
    maximum_operands: int
    exact_operands: int | None = None
    temporal_window: TemporalWindow | None = None
    question_anchor: date | None = None
    interval_unit: Literal["auto", "day", "week", "month", "year"] = "day"
    difference_direction: Literal[
        "absolute", "first_minus_second", "second_minus_first"
    ] | None = None
    order_direction: Literal["ascending", "descending"] | None = None
    requires_complete_evidence: bool = False
    result_unit: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "minimum_operands": self.minimum_operands,
            "maximum_operands": self.maximum_operands,
            "exact_operands": self.exact_operands,
            "temporal_window": (
                self.temporal_window.as_dict() if self.temporal_window else None
            ),
            "question_anchor": (
                self.question_anchor.isoformat() if self.question_anchor else None
            ),
            "interval_unit": self.interval_unit,
            "difference_direction": self.difference_direction,
            "order_direction": self.order_direction,
            "requires_complete_evidence": self.requires_complete_evidence,
            "result_unit": self.result_unit,
        }


@dataclass(frozen=True)
class PlanDecision:
    status: PlanStatus
    plan: EvidencePlan | None = None
    reason: str = ""


@dataclass(frozen=True)
class GroundedEvidence:
    citation: str
    store_id: int
    span_start: int
    span_end: int
    quote: str
    value: int | float | str | None
    unit: str | None
    key: str | None
    label: str | None
    evidence_date: date | None
    assertion_id: str | None
    assertion_kind: str | None
    subject_key: str | None
    predicate_key: str | None
    scope_key: str | None
    active: bool | None
    unresolved_conflict: bool | None
    group_active_assertion_ids: tuple[str, ...]
    group_state_truncated: bool
    temporal_trust: Literal[
        "not_applicable", "engine_sidecar", "low_trust"
    ] = "not_applicable"
    temporal_certified: bool | None = None
    temporal_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingDecision:
    status: Literal["grounded", "fallback"]
    operands: tuple[GroundedEvidence, ...] = ()
    reason: str = ""
    temporal_trust: Literal[
        "not_applicable", "engine_sidecar", "low_trust"
    ] = "not_applicable"
    temporal_certified: bool | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComputationTrace:
    operation: ComputationOperation
    result: str
    result_value: int | float | str | tuple[str, ...]
    unit: str | None
    citations: tuple[str, ...]
    entities: tuple[str, ...]
    evidence_dates: tuple[str, ...]
    steps: tuple[str, ...]
    answer: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "result": self.result,
            "result_value": self.result_value,
            "unit": self.unit,
            "citations": list(self.citations),
            "entities": list(self.entities),
            "evidence_dates": list(self.evidence_dates),
            "steps": list(self.steps),
            "answer": self.answer,
        }


@dataclass(frozen=True)
class ComputationDecision:
    status: Literal["computed", "fallback"]
    trace: ComputationTrace | None = None
    reason: str = ""


@dataclass(frozen=True)
class VerificationDecision:
    status: Literal["verified", "fallback"]
    reason: str = ""


def _parse_day(value: Any, *, require_timezone_for_datetime: bool = True) -> date | None:
    if isinstance(value, datetime):
        if require_timezone_for_datetime and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            return None
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("/", "-")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        try:
            return date.fromisoformat(normalized)
        except ValueError:
            return None
    iso_text = normalized[:-1] + "+00:00" if normalized.endswith("Z") else normalized
    try:
        parsed = datetime.fromisoformat(iso_text)
    except ValueError:
        return None
    if require_timezone_for_datetime and (
        parsed.tzinfo is None or parsed.utcoffset() is None
    ):
        return None
    return parsed.astimezone(timezone.utc).date() if parsed.tzinfo else parsed.date()


def _day_from_epoch(value: Any) -> date | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(seconds):
        return None
    return datetime.fromtimestamp(seconds, tz=timezone.utc).date()


def question_date_as_of_epoch(value: Any) -> float | None:
    """Return the inclusive end-of-day UTC boundary for a question date."""
    parsed = _parse_day(value)
    if parsed is None:
        return None
    next_day = datetime.combine(
        parsed + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )
    return next_day.timestamp() - 1e-6


def _subtract_months_clamped(anchor: date, months: int) -> date:
    zero_based = anchor.year * 12 + (anchor.month - 1) - months
    year, month_zero = divmod(zero_based, 12)
    month = month_zero + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(anchor.day, last_day))


def resolve_temporal_window(question: str, question_date: Any) -> TemporalWindow | None:
    anchor = _parse_day(question_date)
    if anchor is None:
        return None
    normalized = question.casefold()
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(day|week|month)s?\s+ago\b",
        normalized,
    )
    if match:
        amount = (
            int(match.group(1))
            if match.group(1).isdigit()
            else _WORD_NUMBERS[match.group(1)]
        )
        unit = match.group(2)
        try:
            if unit == "day":
                target = anchor - timedelta(days=amount)
            elif unit == "week":
                target = anchor - timedelta(days=amount * 7)
            else:
                target = _subtract_months_clamped(anchor, amount)
        except (OverflowError, ValueError):
            return None
        return TemporalWindow(
            target,
            target + timedelta(days=1),
            f"{amount} {unit}{'' if amount == 1 else 's'} before the question date",
        )
    if re.search(r"\byesterday\b", normalized):
        target = anchor - timedelta(days=1)
        return TemporalWindow(target, anchor, "yesterday")
    weekday_match = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        normalized,
    )
    if weekday_match:
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        delta = (anchor.weekday() - weekdays[weekday_match.group(1)]) % 7 or 7
        target = anchor - timedelta(days=delta)
        return TemporalWindow(
            target,
            target + timedelta(days=1),
            f"the most recent {weekday_match.group(1)} before the question date",
        )
    if re.search(r"\blast week\b", normalized):
        start = anchor - timedelta(days=7)
        return TemporalWindow(start, anchor, "the seven days preceding the question date")
    if re.search(r"\blast month\b", normalized):
        end = anchor.replace(day=1)
        start = _subtract_months_clamped(end, 1)
        return TemporalWindow(start, end, "the previous calendar month")
    return None


def _bounded_sum_count(question: str) -> int | None:
    match = re.search(
        r"\b(?:of\s+(?:the\s+)?|(?:the\s+)?)(one|two|three|four|five|six|"
        r"seven|eight|nine|ten|\d+)\s+(?:actual\s+)?(?:novels?|books?|items?|"
        r"events?|purchases?|trips?|sessions?|amounts?|values?|scores?|costs?|"
        r"prices?|durations?|vacations?|holidays?|people|persons?)\b",
        question.casefold(),
    )
    if not match:
        return None
    return _WORD_NUMBERS.get(match.group(1), int(match.group(1)) if match.group(1).isdigit() else 0) or None


def compile_evidence_plan(question: str, question_date: Any = None) -> PlanDecision:
    text = str(question or "").strip()
    if not text:
        return PlanDecision("fallback", reason="question is required")
    if len(text) > _MAX_QUESTION_CHARS:
        return PlanDecision("fallback", reason="question exceeds the bounded planner input")
    normalized = text.casefold()
    temporal = resolve_temporal_window(text, question_date)
    question_anchor = _parse_day(question_date)
    if _UNSUPPORTED_OPERATION_RE.search(normalized):
        return PlanDecision("fallback", reason="question requests an unsupported operation")

    operation: ComputationOperation | None = None
    exact: int | None = None
    minimum = 1
    direction: Literal[
        "absolute", "first_minus_second", "second_minus_first"
    ] | None = None
    order: Literal["ascending", "descending"] | None = None
    requires_complete = False
    interval_unit: Literal["auto", "day", "week", "month", "year"] = "day"
    if re.search(r"\bhow long ago\b", normalized):
        if question_anchor is None:
            return PlanDecision(
                "fallback",
                reason="question needs a valid question date for deterministic temporal planning",
            )
        requested_interval_unit = re.search(
            r"\bin\s+(?:calendar\s+)?(days?|weeks?|months?|years?)\b",
            normalized,
        )
        interval_unit = (
            requested_interval_unit.group(1).rstrip("s")
            if requested_interval_unit
            else "auto"
        )  # type: ignore[assignment]
        operation, exact, minimum = "date_interval", 1, 1
    elif re.search(
        r"\b(how long|time between|interval between|how many\s+(?:calendar\s+)?days?\s+between|since when)\b",
        normalized,
    ):
        operation, exact, minimum = "date_interval", 2, 2
    elif (interval_match := re.search(
        r"\bhow many\s+(?:calendar\s+)?(days?|weeks?|months?)\b.*\b(?:ago|since|between|passed)\b",
        normalized,
    )):
        interval_unit = interval_match.group(1).rstrip("s")  # type: ignore[assignment]
        if question_anchor is None:
            return PlanDecision(
                "fallback",
                reason="question needs a valid question date for deterministic temporal planning",
            )
        if re.search(r"\b(?:between|since)\b.*\b(?:when|and)\b", normalized):
            operation, exact, minimum = "date_interval", 2, 2
        else:
            operation, exact, minimum = "date_interval", 1, 1
    elif re.search(
        r"\b(difference|how much (?:more|less)|how much .*\b(?:save|saved|short|"
        r"need|left|remain)|more than|less than)\b",
        normalized,
    ):
        operation, exact, minimum = "difference", 2, 2
        if re.search(r"\bhow much\s+less\b|\bless than\b", normalized):
            direction = "second_minus_first"
        elif re.search(r"\bhow much\s+more\b|\bmore than\b", normalized):
            direction = "first_minus_second"
        else:
            direction = "absolute"
    elif (
        re.search(r"\b(total|combined|altogether|in all|sum(?:med)?|add(?:ed)? together)\b", normalized)
        or re.search(r"\bpage count of (?:the )?(?:two|three|four|five|six|seven|eight|nine|\d+)\b", normalized)
    ):
        operation = "sum"
        exact = _bounded_sum_count(text)
        minimum = exact or 2
        requires_complete = exact is None
    elif re.search(r"\b(how many|count|number of)\b", normalized):
        operation = "count_distinct"
        exact = _bounded_sum_count(text)
        minimum = exact or 1
        requires_complete = exact is None
    elif (
        re.search(r"\b(current|currently|now|latest|most recent|usually|these days)\b", normalized)
        and not re.search(
            r"\b(chronolog(?:ical|ically|y)?|in order|order of|earliest|first|second|third)\b",
            normalized,
        )
    ):
        operation, minimum = "latest_fact", 1
        requires_complete = True
    elif re.search(
        r"\b(chronolog(?:ical|ically|y)?|in order|order of|earliest|previous|first|second|third)\b",
        normalized,
    ):
        operation, minimum = "order", 2
        requires_complete = True
        exact = _bounded_sum_count(text)
        if exact is not None:
            minimum = exact
            requires_complete = False
        order = "descending" if re.search(r"\b(reverse|latest first|newest first|descending)\b", normalized) else "ascending"
    elif temporal is not None:
        operation = "date_filter"
        exact = _bounded_sum_count(text)
        minimum = exact or 1
        # Grammar such as "what happened" or singular "who" never proves
        # that a temporal window contains only one event/person. The filter is
        # open unless the question itself declares a finite cardinality.
        requires_complete = exact is None

    if operation is None:
        if _COMPUTATION_TRIGGER_RE.search(normalized):
            if re.search(r"\b(ago|yesterday|last week|last month|last (?:mon|tues|wednes|thurs|fri|satur|sun)day)\b", normalized):
                return PlanDecision(
                    "fallback",
                    reason="question needs a valid question date for deterministic temporal planning",
                )
            return PlanDecision("fallback", reason="question's computation form is ambiguous")
        return PlanDecision("not_applicable", reason="no supported deterministic operation detected")

    return PlanDecision("planned", plan=EvidencePlan(
        operation=operation,
        minimum_operands=minimum,
        maximum_operands=_MAX_OPERANDS,
        exact_operands=exact,
        temporal_window=temporal,
        question_anchor=question_anchor,
        interval_unit=interval_unit,
        difference_direction=direction,
        order_direction=order,
        requires_complete_evidence=requires_complete,
    ))


def normalize_unit(value: Any) -> str | None:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return None
    if normalized.endswith("s") and normalized not in {"usd"}:
        normalized = normalized[:-1]
    aliases = {
        "$": "usd",
        "dollar": "usd",
        "usd": "usd",
        "min": "minute",
        "minute": "minute",
        "hr": "hour",
        "hour": "hour",
        "day": "day",
        "week": "week",
        "month": "month",
        "page": "page",
        "item": "item",
        "event": "event",
        "point": "point",
    }
    return aliases.get(normalized, normalized)


def _explicit_units(text: str) -> set[str]:
    checks = (
        (r"(?:\$\s*-?\d|\b(?:usd|dollars?)\b)", "usd"),
        (r"\bpages?\b", "page"),
        (r"\b(?:hours?|hrs?)\b", "hour"),
        (r"\b(?:minutes?|mins?)\b", "minute"),
        (r"\bdays?\b", "day"),
        (r"\bweeks?\b", "week"),
        (r"\bmonths?\b", "month"),
        (r"\bpoints?\b", "point"),
        (r"\bitems?\b", "item"),
        (r"\bevents?\b", "event"),
    )
    return {unit for pattern, unit in checks if re.search(pattern, text, re.IGNORECASE)}


def _explicit_numbers(text: str) -> list[float]:
    values = [
        float(match.group(0).replace(",", ""))
        for match in _NUMBER_RE.finditer(text)
        if math.isfinite(float(match.group(0).replace(",", "")))
    ]
    values.extend(
        float(_WORD_NUMBERS[match.group(0).casefold()])
        for match in re.finditer(
            r"\b(?:" + "|".join(_WORD_NUMBERS) + r")\b",
            text,
            re.IGNORECASE,
        )
    )
    return values


def _numeric_value_has_unit(value: float, unit: str, quote: str) -> bool:
    unit_patterns = {
        "page": r"pages?",
        "hour": r"(?:hours?|hrs?)",
        "minute": r"(?:minutes?|mins?)",
        "day": r"days?",
        "week": r"weeks?",
        "month": r"months?",
        "point": r"points?",
        "item": r"items?",
        "event": r"events?",
    }
    mentions = [
        (match, float(match.group(0).replace(",", "")))
        for match in _NUMBER_RE.finditer(quote)
    ]
    mentions.extend(
        (match, float(_WORD_NUMBERS[match.group(0).casefold()]))
        for match in re.finditer(
            r"\b(?:" + "|".join(_WORD_NUMBERS) + r")\b",
            quote,
            re.IGNORECASE,
        )
    )
    for match, parsed in mentions:
        if abs(parsed - value) > 1e-9:
            continue
        before = quote[max(0, match.start() - 24):match.start()]
        after = quote[match.end():match.end() + 24]
        if unit == "usd" and (
            re.search(r"(?:\$|\busd)\s*$", before, re.IGNORECASE)
            or re.match(r"\s*(?:usd\b|dollars?\b)", after, re.IGNORECASE)
        ):
            return True
        pattern = unit_patterns.get(
            unit, re.escape(unit).replace(r"\_", r"[ _]") + "s?"
        )
        if pattern and (
            re.match(rf"\s*{pattern}\b", after, re.IGNORECASE)
            or re.search(rf"\b{pattern}\s*[:=~-]?\s*$", before, re.IGNORECASE)
        ):
            return True
    return False


def _quote_supports_day(quote: str, expected: date, raw_date: str) -> bool:
    if raw_date and raw_date in quote:
        return True
    for match in re.finditer(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", quote):
        try:
            if date(int(match.group(1)), int(match.group(2)), int(match.group(3))) == expected:
                return True
        except ValueError:
            continue
    for match in re.finditer(
        r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        quote,
        re.IGNORECASE,
    ):
        month = _MONTHS.get(match.group(1).casefold())
        if month:
            try:
                if date(int(match.group(3)), month, int(match.group(2))) == expected:
                    return True
            except ValueError:
                continue
    return False


def _bounded_optional_text(value: Any, field: str) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, f"{field} must not be empty"
    if len(text) > _MAX_LABEL_CHARS:
        return None, f"{field} exceeds {_MAX_LABEL_CHARS} characters"
    return text, None


def _normalized_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\w]+", value.casefold(), re.UNICODE))


def _quote_supports_key(quote: str, key: str) -> bool:
    """Fail closed unless every canonical-key token is present in the quote."""
    quote_tokens = set(_normalized_tokens(quote))
    key_tokens = _normalized_tokens(key)
    return bool(key_tokens) and all(token in quote_tokens for token in key_tokens)


def _ground_one(
    raw: Any,
    *,
    messages: MessageStore,
    assertions: AssertionStore | None,
    as_of: float | None,
    engine: Any = None,
) -> tuple[GroundedEvidence | None, str | None]:
    if not isinstance(raw, dict):
        return None, "each operand must be an object"
    assertion_id = str(raw.get("assertion_id") or "").strip().casefold() or None
    assertion_row: dict[str, Any] | None = None
    active: bool | None = None
    conflict: bool | None = None
    group_active_ids: tuple[str, ...] = ()
    group_truncated = False
    if assertion_id:
        if assertions is None:
            return None, "assertion_id requires the V4 assertion store"
        try:
            rows = assertions.query_assertions(
                assertion_id=assertion_id,
                as_of=as_of,
                limit=1,
            )
        except ValueError as exc:
            return None, str(exc)
        if not rows:
            return None, "assertion_id is missing or source-invalidated"
        assertion_row = rows[0]
        store_id = int(assertion_row["source_store_id"])
        span_start = int(assertion_row["source_span_start"])
        span_end = int(assertion_row["source_span_end"])
        quote = str(assertion_row["source_quote"])
        state = query_assertion_state(
            assertions,
            subject_key=str(assertion_row["subject_key"]),
            predicate_key=str(assertion_row["predicate_key"]),
            kinds=[str(assertion_row["kind"])],
            scope_key=str(assertion_row["scope_key"]),
            as_of=as_of,
            limit=500,
        )
        state_row = next(
            (row for row in state.assertions if row["assertion_id"] == assertion_id),
            None,
        )
        if state_row is None:
            return None, "assertion state is unavailable"
        active = bool(state_row["active"])
        conflict = bool(state_row["unresolved_conflict"])
        group_active_ids = state.active_assertion_ids
        group_truncated = state.assertions_truncated or state.relations_truncated
    else:
        for field in ("store_id", "span_start", "span_end", "quote"):
            if field not in raw:
                return None, "raw operands require store_id, span_start, span_end, and quote"
        try:
            store_id = int(raw["store_id"])
            span_start = int(raw["span_start"])
            span_end = int(raw["span_end"])
        except (TypeError, ValueError, OverflowError):
            return None, "raw operand refs require integer store_id/span offsets"
        quote = str(raw.get("quote") or "")

    if not quote or len(quote) > _MAX_QUOTE_CHARS:
        return None, f"operand quote must contain 1..{_MAX_QUOTE_CHARS} characters"
    stored = messages.get(store_id)
    if not stored:
        return None, f"message store_id {store_id} does not exist"
    content = str(stored.get("content") or "")
    if span_start < 0 or span_end <= span_start or span_end > len(content):
        return None, "operand span is outside the exact source row"
    if content[span_start:span_end] != quote:
        return None, "operand quote does not match the exact source span"
    if assertion_row is not None:
        supplied_ref = any(field in raw for field in ("store_id", "span_start", "span_end", "quote"))
        if supplied_ref:
            try:
                supplied_store_id = int(raw.get("store_id", store_id))
                supplied_span_start = int(raw.get("span_start", span_start))
                supplied_span_end = int(raw.get("span_end", span_end))
            except (TypeError, ValueError, OverflowError):
                return None, "assertion_id supplied ref requires integer store_id/span offsets"
            if (
                supplied_store_id != store_id
                or supplied_span_start != span_start
                or supplied_span_end != span_end
                or str(raw.get("quote", quote)) != quote
            ):
                return None, "assertion_id and supplied exact ref disagree"

    value = raw.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float, str, type(None))):
        return None, "operand value must be a number, string, or null"
    if isinstance(value, float) and not math.isfinite(value):
        return None, "operand numeric value must be finite"
    if isinstance(value, (int, float)):
        if not any(abs(number - float(value)) < 1e-9 for number in _explicit_numbers(quote)):
            return None, f"numeric value {value} is not explicit in its exact quote"
    elif isinstance(value, str) and value not in quote:
        return None, "string value is not explicit in its exact quote"

    unit = normalize_unit(raw.get("unit"))
    if unit and len(unit) > 100:
        return None, "unit exceeds 100 characters"
    if unit and isinstance(value, (int, float)) and not _numeric_value_has_unit(
        float(value), unit, quote
    ):
        return None, f"unit {unit} is not attached to value {value} in its exact quote"
    label, error = _bounded_optional_text(raw.get("label"), "label")
    if error:
        return None, error
    if label and label.casefold() not in quote.casefold():
        return None, "label is not explicit in its exact quote"
    key, error = _bounded_optional_text(raw.get("key"), "key")
    if error:
        return None, error
    if key:
        key = " ".join(key.casefold().split())
        if not _quote_supports_key(quote, key):
            return None, "canonical key is not token-supported by its exact quote"

    assertion_day = None
    if assertion_row is not None:
        assertion_day = _day_from_epoch(assertion_row.get("event_at"))
    evidence_day = assertion_day

    occurrence_day = None
    resolved_occurrence: Mapping[str, Any] | None = None
    temporal_trust: Literal[
        "not_applicable", "engine_sidecar", "low_trust"
    ] = "not_applicable"
    temporal_certified: bool | None = None
    temporal_notes: tuple[str, ...] = ()
    raw_occurrence = raw.get("occurrence_time")
    if raw_occurrence is not None:
        if not isinstance(raw_occurrence, dict):
            return None, "occurrence_time must be an object"
        resolved, resolved_trust = resolve_occurrence_time_with_trust(
            quote,
            observed_at=(
                stored.get("observed_at")
                if stored.get("observed_at") is not None
                else stored.get("timestamp")
            ),
            session_date=raw_occurrence.get("session_date"),
            engine=engine,
            session_id=stored.get("session_id"),
        )
        resolved_occurrence = resolved
        temporal_trust = str(  # type: ignore[assignment]
            resolved_trust["anchor_trust"]
        )
        temporal_certified = resolved_trust["temporal_certified"]
        note = str(resolved_trust.get("trust_note") or "").strip()
        temporal_notes = (note,) if note else ()
        supplied_source = str(raw_occurrence.get("event_time_source") or "")
        if supplied_source != resolved["event_time_source"]:
            return None, "occurrence_time source is not supported by the exact quote"
        supplied_date = str(raw_occurrence.get("event_date") or "").strip()
        if (
            supplied_date
            and supplied_date != str(resolved.get("event_date") or "")
            and not resolved_trust.get("session_date_overridden")
        ):
            return None, "occurrence_time date is not supported by the exact quote"
        occurrence_day = _parse_day(str(resolved.get("event_date") or ""))
        evidence_day = occurrence_day or evidence_day

    raw_date = str(raw.get("date") or "").strip()
    if raw_date:
        claimed_day = _parse_day(raw_date)
        if claimed_day is None:
            return None, f"date {raw_date!r} is invalid or timezone-ambiguous"
        claimed_supported = (
            claimed_day in {assertion_day, occurrence_day}
            or _quote_supports_day(quote, claimed_day, raw_date)
        )
        occurrence_overridden = bool(
            resolved_occurrence
            and resolved_trust.get("session_date_overridden")
        )
        if not claimed_supported and not occurrence_overridden:
            return None, f"date {raw_date!r} is not supported by metadata or exact quote"
        if not occurrence_overridden:
            evidence_day = claimed_day

    if as_of is not None:
        source_observation_grounded = False
        if assertion_row is not None:
            try:
                assertion_observed_at = float(assertion_row.get("observed_at"))
            except (TypeError, ValueError, OverflowError):
                return None, "assertion observation timestamp is invalid"
            if not math.isfinite(assertion_observed_at) or assertion_observed_at > as_of:
                return None, "assertion was observed after the question-date boundary"
            source_observation_grounded = True
        elif stored.get("observed_at") is not None:
            try:
                stored_observed_at = float(stored.get("observed_at"))
            except (TypeError, ValueError, OverflowError):
                return None, "source observation timestamp is invalid"
            if not math.isfinite(stored_observed_at) or stored_observed_at > as_of:
                return None, "source was observed after the question-date boundary"
            source_observation_grounded = True
        elif resolved_occurrence is not None and resolved_occurrence.get("session_date"):
            source_observed_day = _parse_day(
                str(resolved_occurrence.get("session_date"))
            )
            if source_observed_day is None:
                return None, "occurrence_time session_date is invalid"
            source_observed_epoch = datetime.combine(
                source_observed_day, datetime.max.time(), tzinfo=timezone.utc
            ).timestamp()
            if source_observed_epoch > as_of:
                return None, "source was observed after the question-date boundary"
            source_observation_grounded = True
        if not source_observation_grounded:
            source_observed_at = (
                stored.get("observed_at")
                if stored.get("observed_at") is not None
                else stored.get("ingested_at", stored.get("timestamp"))
            )
            try:
                observed_at = float(source_observed_at)
            except (TypeError, ValueError, OverflowError):
                return None, "source observation timestamp is invalid"
            if not math.isfinite(observed_at) or observed_at > as_of:
                return None, "source was observed after the question-date boundary"
        if evidence_day is not None:
            evidence_epoch = datetime.combine(
                evidence_day, datetime.max.time(), tzinfo=timezone.utc
            ).timestamp()
            if evidence_epoch > as_of:
                return None, "source occurrence was after the question-date boundary"

    return GroundedEvidence(
        citation=f"lcm:{store_id}:{span_start}-{span_end}",
        store_id=store_id,
        span_start=span_start,
        span_end=span_end,
        quote=quote,
        value=value,
        unit=unit,
        key=key,
        label=label,
        evidence_date=evidence_day,
        assertion_id=assertion_id,
        assertion_kind=str(assertion_row["kind"]) if assertion_row else None,
        subject_key=str(assertion_row["subject_key"]) if assertion_row else None,
        predicate_key=str(assertion_row["predicate_key"]) if assertion_row else None,
        scope_key=str(assertion_row["scope_key"]) if assertion_row else None,
        active=active,
        unresolved_conflict=conflict,
        group_active_assertion_ids=group_active_ids,
        group_state_truncated=group_truncated,
        temporal_trust=temporal_trust,
        temporal_certified=temporal_certified,
        temporal_notes=temporal_notes,
    ), None


def ground_evidence(
    raw_operands: Any,
    *,
    messages: MessageStore,
    assertions: AssertionStore | None,
    as_of: float | None = None,
    engine: Any = None,
) -> GroundingDecision:
    if not isinstance(raw_operands, list):
        return GroundingDecision("fallback", reason="operands must be an array")
    if not 1 <= len(raw_operands) <= _MAX_OPERANDS:
        return GroundingDecision(
            "fallback", reason=f"operands must contain between 1 and {_MAX_OPERANDS} items"
        )
    grounded: list[GroundedEvidence] = []
    for index, raw in enumerate(raw_operands):
        operand, error = _ground_one(
            raw,
            messages=messages,
            assertions=assertions,
            as_of=as_of,
            engine=engine,
        )
        if error:
            return GroundingDecision("fallback", reason=f"operands[{index}]: {error}")
        grounded.append(operand)  # type: ignore[arg-type]
    temporal = [
        operand
        for operand in grounded
        if operand.temporal_certified is not None
    ]
    notes = tuple(dict.fromkeys(
        note
        for operand in temporal
        for note in operand.temporal_notes
    ))
    if any(operand.temporal_certified is False for operand in temporal):
        trust = "low_trust"
        certified = False
    elif temporal:
        trust = (
            "engine_sidecar"
            if any(
                operand.temporal_trust == "engine_sidecar"
                for operand in temporal
            )
            else "not_applicable"
        )
        certified = True
    else:
        trust = "not_applicable"
        certified = None
    return GroundingDecision(
        "grounded",
        operands=tuple(grounded),
        temporal_trust=trust,
        temporal_certified=certified,
        notes=notes,
    )


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(round(value, 6)).rstrip("0").rstrip(".")


def _format_quantity(value: float, unit: str | None) -> str:
    number = _format_number(value)
    if not unit:
        return number
    if unit == "usd":
        return f"${number}"
    return f"{number} {unit}{'' if abs(value) == 1 else 's'}"


def _answer(result: str, citations: Sequence[str]) -> str:
    return f"{result} " + " ".join(f"[{citation}]" for citation in citations)


def _trace(
    plan: EvidencePlan,
    operands: Sequence[GroundedEvidence],
    *,
    result: str,
    result_value: int | float | str | tuple[str, ...],
    unit: str | None,
    steps: Sequence[str],
) -> ComputationDecision:
    citations = tuple(dict.fromkeys(operand.citation for operand in operands))
    entities = tuple(dict.fromkeys(
        label
        for operand in operands
        if (label := (operand.label or operand.key))
    ))
    evidence_dates = tuple(dict.fromkeys(
        operand.evidence_date.isoformat()
        for operand in operands
        if operand.evidence_date is not None
    ))
    return ComputationDecision("computed", trace=ComputationTrace(
        operation=plan.operation,
        result=result,
        result_value=result_value,
        unit=unit,
        citations=citations,
        entities=entities,
        evidence_dates=evidence_dates,
        steps=tuple(steps),
        answer=_answer(result, citations),
    ))


def _label(operand: GroundedEvidence) -> str | None:
    if operand.label:
        return operand.label
    if isinstance(operand.value, str):
        return operand.value
    if operand.key:
        return operand.key
    return None


def validate_selector_alignment(
    question: str,
    plan: EvidencePlan,
    operands: Sequence[GroundedEvidence],
) -> str | None:
    """Validate the host selector's only semantic ordering obligation."""
    if plan.operation != "difference" or plan.difference_direction == "absolute":
        return None
    if len(operands) != 2 or any(not operand.label for operand in operands):
        return "directed difference requires two explicit operand labels"
    normalized = question.casefold()
    labels = [str(operand.label).casefold() for operand in operands]
    if labels[0] == labels[1]:
        return "directed difference labels must be distinct"
    positions = [normalized.find(label) for label in labels]
    if any(position < 0 for position in positions):
        return "directed difference labels must appear literally in the question"
    if positions != sorted(positions):
        return "directed difference operands must follow question mention order"
    return None


def execute_plan(
    plan: EvidencePlan,
    operands: Sequence[GroundedEvidence],
) -> ComputationDecision:
    if plan.exact_operands is not None and len(operands) != plan.exact_operands:
        return ComputationDecision(
            "fallback", reason=f"{plan.operation} requires exactly {plan.exact_operands} operands"
        )
    if not plan.minimum_operands <= len(operands) <= plan.maximum_operands:
        return ComputationDecision(
            "fallback",
            reason=(
                f"{plan.operation} requires between {plan.minimum_operands} "
                f"and {plan.maximum_operands} operands"
            ),
        )
    selected = list(operands)
    temporal_steps: list[str] = []
    if plan.temporal_window and plan.operation != "date_interval":
        if any(operand.evidence_date is None for operand in selected):
            return ComputationDecision(
                "fallback", reason="temporal plan requires a grounded date for every operand"
            )
        before_count = len(selected)
        selected = [
            operand
            for operand in selected
            if plan.temporal_window.start
            <= operand.evidence_date  # type: ignore[operator]
            < plan.temporal_window.end
        ]
        temporal_steps.append(
            f"Resolved {plan.temporal_window.description} as "
            f"[{plan.temporal_window.start.isoformat()}, {plan.temporal_window.end.isoformat()})"
        )
        temporal_steps.append(
            f"Selected {len(selected)} of {before_count} dated evidence items"
        )
        if not selected:
            return ComputationDecision(
                "fallback", reason="no grounded evidence falls inside the resolved date window"
            )

    citations = [operand.citation for operand in selected]
    if len(citations) != len(set(citations)):
        return ComputationDecision("fallback", reason="duplicate exact refs are not valid operands")

    if plan.operation == "count_distinct":
        if any(not operand.key for operand in selected):
            return ComputationDecision(
                "fallback", reason="count_distinct requires one canonical key per operand"
            )
        keys = tuple(dict.fromkeys(operand.key for operand in selected if operand.key))
        units = {operand.unit for operand in selected if operand.unit}
        if len(units) > 1:
            return ComputationDecision("fallback", reason="count_distinct has mixed units")
        if units and any(operand.unit is None for operand in selected):
            return ComputationDecision(
                "fallback", reason="every counted operand must carry the compatible unit"
            )
        unit = next(iter(units), "item")
        result = _format_quantity(float(len(keys)), unit)
        return _trace(
            plan,
            selected,
            result=result,
            result_value=len(keys),
            unit=unit,
            steps=[
                *temporal_steps,
                f"Deduplicated {len(selected)} grounded mentions into {len(keys)} keys: "
                + ", ".join(keys),
            ],
        )

    if plan.operation in {"sum", "difference"}:
        if any(not isinstance(operand.value, (int, float)) for operand in selected):
            return ComputationDecision("fallback", reason="numeric operand missing")
        units = {operand.unit for operand in selected if operand.unit}
        requested_unit = normalize_unit(plan.result_unit)
        time_factors = {
            "minute": 1.0,
            "hour": 60.0,
            "day": 1_440.0,
            "week": 10_080.0,
        }
        time_dimension = bool(units) and units.issubset(time_factors)
        if time_dimension and any(operand.unit is None for operand in selected):
            return ComputationDecision(
                "fallback",
                reason="every time operand must carry a compatible unit",
            )
        if len(units) > 1 and not time_dimension:
            return ComputationDecision("fallback", reason="mixed units are not supported")
        if requested_unit and requested_unit not in units:
            if not (time_dimension and requested_unit in time_factors):
                return ComputationDecision(
                    "fallback", reason="requested result unit is incompatible with operands"
                )
        unit = requested_unit or (
            min(units, key=lambda item: time_factors[item])
            if time_dimension and len(units) > 1
            else next(iter(units), None)
        )
        if unit and not time_dimension and any(operand.unit != unit for operand in selected):
            return ComputationDecision(
                "fallback", reason="every numeric operand must carry the compatible unit"
            )
        if not unit:
            quote_units = set().union(
                *(_explicit_units(operand.quote) for operand in selected)
            )
            if quote_units:
                return ComputationDecision(
                    "fallback",
                    reason="numeric units present in evidence must be supplied explicitly",
                )
        values = [float(operand.value) for operand in selected]  # type: ignore[arg-type]
        if time_dimension and unit in time_factors:
            values = [
                value * time_factors[str(operand.unit)] / time_factors[unit]
                for value, operand in zip(values, selected)
            ]
        if plan.operation == "sum":
            result_value = sum(values)
        else:
            if len(values) != 2:
                return ComputationDecision(
                    "fallback", reason="difference requires exactly two operands"
                )
            if plan.difference_direction == "first_minus_second":
                result_value = values[0] - values[1]
            elif plan.difference_direction == "second_minus_first":
                result_value = values[1] - values[0]
            else:
                result_value = abs(values[0] - values[1])
            if plan.difference_direction != "absolute" and result_value < 0:
                return ComputationDecision(
                    "fallback",
                    reason="directed difference premise is contradicted by the operands",
                )
        result = _format_quantity(result_value, unit)
        return _trace(
            plan,
            selected,
            result=result,
            result_value=(int(result_value) if result_value.is_integer() else result_value),
            unit=unit,
            steps=[
                *temporal_steps,
                f"{plan.operation}:{plan.difference_direction or 'n/a'}"
                f"({', '.join(_format_number(value) for value in values)}) "
                f"= {_format_number(result_value)}",
            ],
        )

    if plan.operation == "date_interval":
        if len(selected) not in {1, 2} or any(
            operand.evidence_date is None for operand in selected
        ):
            return ComputationDecision(
                "fallback", reason="date_interval requires one or two grounded dates"
            )
        if len(selected) == 1:
            if plan.question_anchor is None:
                return ComputationDecision(
                    "fallback",
                    reason="one-operand date_interval requires a question-date anchor",
                )
            anchor = plan.question_anchor
            days = abs((anchor - selected[0].evidence_date).days)  # type: ignore[operator]
            interval_step = (
                f"Absolute interval from {selected[0].evidence_date.isoformat()} "
                f"to question date {anchor.isoformat()} is {days} days"
            )
        else:
            days = abs((selected[1].evidence_date - selected[0].evidence_date).days)  # type: ignore[operator]
            interval_step = (
                f"Absolute interval from {selected[0].evidence_date.isoformat()} "
                f"to {selected[1].evidence_date.isoformat()} is {days} days"  # type: ignore[union-attr]
            )
        interval_unit = plan.interval_unit
        complete_months = complete_years = 0
        if interval_unit in {"auto", "month", "year"}:
            if len(selected) == 1:
                first_day, second_day = (
                    selected[0].evidence_date,
                    plan.question_anchor,
                )
            else:
                first_day, second_day = (
                    selected[0].evidence_date,
                    selected[1].evidence_date,
                )
            if first_day is None or second_day is None:
                return ComputationDecision(
                    "fallback",
                    reason="calendar date_interval requires two grounded dates",
                )
            earlier, later = sorted((first_day, second_day))
            complete_months = (
                (later.year - earlier.year) * 12 + later.month - earlier.month
            )
            if later.day < earlier.day:
                complete_months -= 1
            complete_years = later.year - earlier.year
            anniversary_day = min(
                earlier.day,
                calendar.monthrange(later.year, earlier.month)[1],
            )
            if (later.month, later.day) < (earlier.month, anniversary_day):
                complete_years -= 1

        if interval_unit == "auto":
            if complete_years > 0:
                interval_unit = "year"
            elif complete_months > 0:
                interval_unit = "month"
            elif days >= 7:
                interval_unit = "week"
            else:
                interval_unit = "day"

        if interval_unit == "week":
            result_value = days // 7
        elif interval_unit == "month":
            result_value = complete_months
        elif interval_unit == "year":
            result_value = complete_years
        else:
            result_value = days
        result = _format_quantity(float(result_value), interval_unit)
        return _trace(
            plan,
            selected,
            result=result,
            result_value=result_value,
            unit=interval_unit,
            steps=[interval_step, f"Reported interval in {interval_unit}s"],
        )

    if plan.operation == "date_filter":
        labels = tuple(label for operand in selected if (label := _label(operand)))
        if len(labels) != len(selected):
            return ComputationDecision(
                "fallback", reason="date_filter requires a grounded label or string value"
            )
        return _trace(
            plan,
            selected,
            result="; ".join(labels),
            result_value=labels,
            unit=None,
            steps=temporal_steps,
        )

    if plan.operation == "order":
        if any(operand.evidence_date is None for operand in selected):
            return ComputationDecision("fallback", reason="order requires grounded dates")
        selected.sort(key=lambda operand: operand.evidence_date)  # type: ignore[arg-type]
        if plan.order_direction == "descending":
            selected.reverse()
        labels = tuple(label for operand in selected if (label := _label(operand)))
        if len(labels) != len(selected):
            return ComputationDecision("fallback", reason="order requires grounded labels")
        return _trace(
            plan,
            selected,
            result=" -> ".join(labels),
            result_value=labels,
            unit=None,
            steps=[
                f"Sorted {len(labels)} grounded items by date "
                f"({plan.order_direction or 'ascending'})"
            ],
        )

    if any(operand.assertion_id is None for operand in selected):
        return ComputationDecision(
            "fallback", reason="latest_fact requires typed assertion operands"
        )
    identities = {
        (
            operand.subject_key,
            operand.predicate_key,
            operand.scope_key,
            operand.assertion_kind,
        )
        for operand in selected
    }
    if len(identities) != 1:
        return ComputationDecision(
            "fallback",
            reason="latest_fact operands must share subject, predicate, scope, and kind",
        )
    if any(operand.group_state_truncated for operand in selected):
        return ComputationDecision(
            "fallback", reason="latest_fact state was truncated and is not safely complete"
        )
    active_ids = set().union(
        *(set(operand.group_active_assertion_ids) for operand in selected)
    )
    selected_ids = {operand.assertion_id for operand in selected}
    if not active_ids.issubset(selected_ids):
        return ComputationDecision(
            "fallback", reason="latest_fact operands omit active typed state candidates"
        )
    if any(operand.unresolved_conflict for operand in selected if operand.active):
        return ComputationDecision(
            "fallback", reason="latest_fact has unresolved conflicting active state"
        )
    active = [operand for operand in selected if operand.active]
    if not active or any(operand.evidence_date is None for operand in active):
        return ComputationDecision(
            "fallback", reason="latest_fact requires dated active typed state"
        )
    latest_day = max(operand.evidence_date for operand in active)  # type: ignore[type-var]
    latest = [operand for operand in active if operand.evidence_date == latest_day]
    latest_values = {str(operand.value) for operand in latest}
    if len(latest_values) != 1:
        return ComputationDecision(
            "fallback", reason="latest_fact has multiple values at the latest timestamp"
        )
    chosen = latest[0]
    if chosen.value is None:
        return ComputationDecision("fallback", reason="latest_fact value is missing")
    result = (
        _format_quantity(float(chosen.value), chosen.unit)
        if isinstance(chosen.value, (int, float))
        else str(chosen.value)
    )
    return _trace(
        plan,
        selected,
        result=result,
        result_value=chosen.value,
        unit=chosen.unit,
        steps=[f"Selected the only non-conflicting active state dated {latest_day.isoformat()}"],
    )


def verify_final_answer(candidate: Any, trace: ComputationTrace) -> VerificationDecision:
    text = str(candidate or "").strip()
    if not text:
        return VerificationDecision("fallback", "candidate answer is empty")
    if len(text) > 4_000:
        return VerificationDecision("fallback", "candidate answer exceeds verifier bound")
    expected_citations = {f"[{citation}]" for citation in trace.citations}
    actual_citations = set(re.findall(r"\[lcm:\d+:\d+-\d+\]", text))
    if actual_citations != expected_citations:
        return VerificationDecision("fallback", "candidate citations do not match exact operands")
    if text == trace.answer:
        return VerificationDecision("verified")
    without_citations = re.sub(r"\s*\[lcm:\d+:\d+-\d+\]", "", text).strip()
    if trace.result.casefold() not in without_citations.casefold():
        return VerificationDecision("fallback", "candidate does not preserve the verified result")
    expected_numbers = _explicit_numbers(trace.result)
    actual_numbers = _explicit_numbers(without_citations)
    if expected_numbers != actual_numbers:
        return VerificationDecision("fallback", "candidate changes or adds numeric results")
    expected_units = _explicit_units(trace.result)
    actual_units = _explicit_units(without_citations)
    if expected_units != actual_units:
        return VerificationDecision("fallback", "candidate changes the verified unit")
    normalized_candidate = without_citations.casefold()
    missing_entities = [
        entity
        for entity in trace.entities
        if entity.casefold() not in normalized_candidate
    ]
    if missing_entities:
        return VerificationDecision(
            "fallback", "candidate omits or changes a grounded entity"
        )
    return VerificationDecision("verified")
