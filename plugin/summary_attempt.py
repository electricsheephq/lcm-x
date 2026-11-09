"""Summary attempt values and provider-error predicates; no circuit policy (I7)."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryTiming:
    """Clock observations at the existing provider boundary."""

    started: float
    ended: float | None
    deadline_installed: bool


@dataclass(frozen=True)
class SummaryAttempt:
    """One provider result, detached from the thread-local transport channel."""

    text: str | None
    error: BaseException | None
    route: dict[str, object]
    contract: str
    timing: SummaryTiming


# #682: a summary route that cannot serve its model is host/profile config, not a transient failure.
_STATUS_IN_MESSAGE_RE = re.compile(r"(?:error code|status(?: code)?)\s*[:=]?\s*(\d{3})\b")
_ROUTE_CONFIG_ERROR_TOKENS = (
    "model not found", "model_not_found", "unknown model", "no such model", "is not a valid model",
    "model not supported", "model is not supported", "model_not_supported", "unsupported model",
)
# Generic tokens count only when the message also names a model (a bare 404 or a missing file is not one).
_ROUTE_CONFIG_ERROR_GENERIC_TOKENS = ("not_found_error", "does not exist", "not supported when using")
_NOT_ROUTE_CONFIG_TOKENS = (
    "context length", "context_length", "context window", "maximum context", "too many tokens", "rate limit",
    "rate_limit", "too many requests", "billing", "credits", "insufficient", "quota", "payment", "free tier",
    "timed out", "timeout",
)


def is_summary_route_config_error(exc: BaseException | None) -> bool:
    """#682: a 400/404 saying the route cannot serve its model. Context-length, rate-limit, billing (402),
    timeout and 5xx failures are not. The status is read as the host reads it (``status_code``, else
    ``response.status_code``); an error without one is judged by its text."""
    if exc is None or isinstance(exc, TimeoutError):
        return False
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    message = str(exc).lower()
    if status is None and (match := _STATUS_IN_MESSAGE_RE.search(message)):
        status = int(match.group(1))
    if status not in (400, 404, None) or any(token in message for token in _NOT_ROUTE_CONFIG_TOKENS):
        return False
    return any(token in message for token in _ROUTE_CONFIG_ERROR_TOKENS) or (
        "model" in message and any(token in message for token in _ROUTE_CONFIG_ERROR_GENERIC_TOKENS))


# #751: the provider refused the request for its size; a property of the chunk, not of the route.
_CONTEXT_LENGTH_ERROR_TOKENS = (
    "context length", "context_length", "context window", "maximum context", "max context", "too many tokens",
    "token limit", "prompt is too long", "input too long", "request too large", "context limit",
    "maximum number of tokens",
)
_NOT_CONTEXT_LENGTH_TOKENS = ("rate limit", "rate_limit", "too many requests", "quota")


def is_summary_context_length_error(exc: BaseException | None) -> bool:
    """#751: a summary call refused because the request does not fit the route's window. A timeout, a
    rate limit or a quota error is not one; the status is read as ``is_summary_route_config_error`` reads it."""
    if exc is None or isinstance(exc, TimeoutError):
        return False
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    message = str(exc).lower()
    if status is None and (match := _STATUS_IN_MESSAGE_RE.search(message)):
        status = int(match.group(1))
    if status == 429 or getattr(exc, "status", None) == 429:
        return False
    if any(token in message for token in _NOT_CONTEXT_LENGTH_TOKENS):
        return False
    return any(token in message for token in _CONTEXT_LENGTH_ERROR_TOKENS)


def is_summary_content_filter_error(exc: BaseException | None) -> bool:
    """#1012 F4: an HTTP 400 refusing this chunk's content, not a route failure."""
    if exc is None or isinstance(exc, TimeoutError):
        return False
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    message = " ".join(str(value) for value in (
        exc, getattr(exc, "body", ""), getattr(response, "text", ""),
    )).lower()
    if status is None and (match := _STATUS_IN_MESSAGE_RE.search(message)):
        status = int(match.group(1))
    return status == 400 and (
        any(token in message for token in ("contentfilter", "content_filter", "content_policy_violation"))
        or (re.search(r"\b1301\b", message) is not None and "sensitive" in message)
    )
