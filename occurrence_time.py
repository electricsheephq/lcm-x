"""Provider-neutral occurrence-time extraction for bounded exact evidence.

Observation time and occurrence time are deliberately separate.  The parser
only resolves dates supported by the evidence text itself, optionally anchored
to source/session metadata supplied by the host.  Failure and ambiguity are
valid ``unknown`` results; observation time is never reused as event time.
"""

from __future__ import annotations

import calendar
import hashlib
from functools import lru_cache
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any


POLICY_VERSION = "occurrence-time-v1"

_ISO_DATE = re.compile(r"(?<!\d)(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})(?!\d)")
_RELATIVE = re.compile(
    r"\b(?:(?P<count>\d+)\s+(?P<unit>days?|weeks?|months?)\s+ago|(?P<simple>today|yesterday)|last\s+(?P<weekday>monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
    re.IGNORECASE,
)
_RELATIVE_MENTION = re.compile(
    # Whole past-day idioms come before yesterday to avoid partial matches.
    r"\b(?:the\s+day\s+before(?:\s+yesterday)?|the\s+other\s+day|"
    # Past periods point away from the observation day; no today synonyms.
    r"last\s+(?:week|month|year|night|weekend)|"
    # Counts are mentions only; the resolvable numeric grammar stays unchanged.
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"a\s+few|a\s+couple\s+of)\s+(?:days?|weeks?|months?|years?)\s+ago|"
    # Earlier periods, excluding earlier today.
    r"earlier\s+this\s+(?:week|month|year))\b|" + _RELATIVE.pattern,
    re.IGNORECASE,
)
_DATE_TEXT_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
)
_WEEKDAYS = {
    name: index
    for index, name in enumerate(
        ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    )
}


def _date_value(raw: str) -> date | None:
    normalized = " ".join(str(raw or "").strip().split())
    for pattern in _DATE_FORMATS:
        try:
            return datetime.strptime(normalized, pattern).date()
        except ValueError:
            continue
    return None


def _carried_date_text(quote):
    quote = re.sub(r"(\d{1,2})(?:st|nd|rd|th)\b", r"\1", str(quote), flags=re.IGNORECASE)
    quote = re.sub(r"(?<=\d)T(?=\d{2}:)", " ", quote)
    return re.sub(r"(\b[A-Za-z]{3,9}\s+\d{1,2})\s+(\d{4})\b", r"\1, \2", quote)


def _parse_quote_date_info(content):
    days = {_date_value(m.group(0)) for m in _DATE_TEXT_RE.finditer(_carried_date_text(content))} - {None}
    reason = ("ambiguous_multiple_explicit_dates" if len(days) > 1 else
              "mixed_absolute_and_relative_dates" if days and _RELATIVE_MENTION.search(content) else None)
    return (next(iter(days)) if len(days) == 1 else None), reason


_small_quote_date_info = lru_cache(maxsize=4096)(_parse_quote_date_info)
_large_quote_date_info = {}


def _quote_date_info(content):
    """Bound retained text to short quotes; retain only digests for large rows."""
    if len(content) <= 1200:
        return _small_quote_date_info(content)
    key = hashlib.sha256(content.encode("utf-8", "surrogatepass")).digest()
    if key not in _large_quote_date_info:
        value = _parse_quote_date_info(content)
        if len(_large_quote_date_info) >= 4096:
            _large_quote_date_info.clear()
        _large_quote_date_info[key] = value
    return _large_quote_date_info[key]


def _withhold_reason(content):
    return _quote_date_info(content)[1]


def _parse_anchor(value: Any) -> date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10].replace("/", "-"))
    except ValueError:
        return None


def _epoch(day: date) -> float:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()


def _subtract_months(day: date, months: int) -> date:
    absolute = day.year * 12 + (day.month - 1) - months
    year, zero_month = divmod(absolute, 12)
    month = zero_month + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _unknown(
    observed_at: Any,
    *,
    session_date: Any = None,
    reason: str = "no_supported_occurrence_time",
) -> dict[str, Any]:
    try:
        observed = float(observed_at)
    except (TypeError, ValueError, OverflowError):
        observed = 0.0
    anchor = _parse_anchor(session_date)
    return {
        "observed_at": _epoch(anchor) if anchor else observed,
        "stored_at": observed,
        "event_at": None,
        "event_date": None,
        "event_time_source": "unknown",
        "session_date": anchor.isoformat() if anchor else None,
        "precision": "unknown",
        "policy_version": POLICY_VERSION,
        "reason": reason,
    }


def resolve_occurrence_time(
    text: Any,
    *,
    observed_at: Any,
    session_date: Any = None,
) -> dict[str, Any]:
    """Resolve one unambiguous day from exact evidence and source metadata."""
    content = str(text or "")
    explicit: list[tuple[date, re.Match[str]]] = []
    for match in _ISO_DATE.finditer(content):
        try:
            explicit.append(
                (
                    date(
                        int(match.group("year")),
                        int(match.group("month")),
                        int(match.group("day")),
                    ),
                    match,
                )
            )
        except ValueError:
            continue
    distinct_explicit = {item[0] for item in explicit}
    if len(distinct_explicit) > 1:
        return _unknown(
            observed_at,
            session_date=session_date,
            reason="ambiguous_multiple_explicit_dates",
        )
    if explicit:
        if reason := _withhold_reason(content):
            return _unknown(observed_at, session_date=session_date, reason=reason)
        day, match = explicit[0]
        anchor = _parse_anchor(session_date)
        return {
            "observed_at": _epoch(anchor) if anchor else float(observed_at or 0.0),
            "stored_at": float(observed_at or 0.0),
            "event_at": _epoch(day),
            "event_date": day.isoformat(),
            "event_time_source": "explicit",
            "precision": "day",
            "policy_version": POLICY_VERSION,
            "session_date": anchor.isoformat() if anchor else None,
            "support": {
                "quote": match.group(0),
                "char_start": match.start(),
                "char_end": match.end(),
            },
        }

    matches = list(_RELATIVE.finditer(content))
    if len(matches) != 1:
        return _unknown(
            observed_at,
            session_date=session_date,
            reason="ambiguous_relative_expression" if matches else "no_supported_occurrence_time",
        )
    anchor = _parse_anchor(session_date)
    if anchor is None:
        return _unknown(
            observed_at,
            session_date=session_date,
            reason="relative_expression_without_session_date",
        )

    # Only base day-objects may change; unknown reasons and bytes stay intact.
    if reason := _withhold_reason(content):
        return _unknown(observed_at, session_date=session_date, reason=reason)
    if len(list(_RELATIVE_MENTION.finditer(content))) > 1:
        return _unknown(observed_at, session_date=session_date, reason="ambiguous_relative_expression")

    match = matches[0]
    simple = (match.group("simple") or "").casefold()
    if simple == "today":
        day = anchor
    elif simple == "yesterday":
        day = anchor - timedelta(days=1)
    elif match.group("weekday"):
        target = _WEEKDAYS[match.group("weekday").casefold()]
        delta = (anchor.weekday() - target) % 7
        day = anchor - timedelta(days=delta or 7)
    else:
        count = int(match.group("count"))
        unit = match.group("unit").casefold()
        if unit.startswith("day"):
            day = anchor - timedelta(days=count)
        elif unit.startswith("week"):
            day = anchor - timedelta(weeks=count)
        else:
            day = _subtract_months(anchor, count)

    return {
        "observed_at": _epoch(anchor),
        "stored_at": float(observed_at or 0.0),
        "event_at": _epoch(day),
        "event_date": day.isoformat(),
        "event_time_source": "relative_to_session",
        "session_date": anchor.isoformat(),
        "precision": "day",
        "policy_version": POLICY_VERSION,
        "support": {
            "quote": match.group(0),
            "char_start": match.start(),
            "char_end": match.end(),
        },
    }
