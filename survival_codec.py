"""Pure survival projection rendering and replay-marker parsing."""
from __future__ import annotations

import json
import re
from itertools import islice
from typing import Any, Dict

from .message_content import normalize_content_value

_PROJECTED = ("[LCM survival fit: this {role} message ({tokens} tokens) is stored verbatim as store id {store_id}; "
              "lcm_expand / lcm_grep reach the full text. (projection {head}/{tail})]")
_PROJECTED_RE = re.compile(r"\[LCM survival fit: this ([a-z_]+) message \((\d+) tokens\) is stored verbatim as store id "
                           r"(\d+); lcm_expand / lcm_grep reach the full text\. \(projection (\d+)/(\d+)\)\]")
_PROJECTED_PREFIX = "[LCM survival fit: this "
_HEAD, _TAIL = 1200, 600  # a projected text keeps its first _HEAD and last _TAIL characters


def projected_fields(row: Dict[str, Any], tokens: int, head: int, tail: int,
                     *, legacy: bool = False, mark_calls: bool = False) -> Dict[str, Any]:
    """The projection of a stored row: its content as head/tail around the mark, head plus mark when
    shorter, or whole when at most ``head``; tool-call arguments over ``head`` replaced by the mark, and
    stored tool linkage. ``legacy`` gives the mark-only form: survival fit's second pass emits it for short
    rows as a last resort, and replay recomputes it to recognise rows emitted that way.
    ``mark_calls`` marks short content when unbounded stored calls replace the live calls.
    Deterministic in (row bytes, tokens, head, tail) for each form."""
    store_id = int(row["store_id"])
    marker = _PROJECTED.format(role=row.get("role"), tokens=tokens, store_id=store_id, head=head, tail=tail)
    text = normalize_content_value(row.get("content")) or ""
    if len(text) > 2 * head:
        text = f"{text[:head]}\n...\n{marker}\n...\n{text[-tail:]}"
    elif legacy and text:
        text = marker
    elif len(text) > head:
        text = f"{text[:head]}\n...\n{marker}"
    calls = row.get("tool_calls")
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except ValueError:
            calls = None
    fields: Dict[str, Any] = {"content": text,
                             "tool_calls": [bounded_call(call, marker, head)
                                            for call in calls] if isinstance(calls, list) and calls else None}
    if mark_calls and not legacy and len(text) <= head and fields["tool_calls"] == (calls or None):
        fields["content"] = f"{text}\n...\n{marker}" if text else marker
    if row.get("tool_call_id"):
        fields["tool_call_id"] = row["tool_call_id"]
    return fields


def bounded_call(call: Any, marker: str, head: int = _HEAD) -> Any:
    """A view copy of a tool call whose arguments are over ``head`` characters: valid JSON carrying the
    provenance notice. The stored row keeps the verbatim arguments; the call stays data."""
    function = call.get("function") if isinstance(call, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    if not isinstance(arguments, str) or len(arguments) <= head:
        return call
    notice = f"{marker} Its tool-call arguments ({len(arguments)} characters) are not in live context."
    return {**call, "function": {**function, "arguments": json.dumps({"lcm_survival_fit": notice})}}


def project_message(message: Dict[str, Any], row: Dict[str, Any], tokens: int) -> Dict[str, Any]:
    """Build the head/tail candidate while preserving the live message's structured content."""
    fields = projected_fields(row, tokens, _HEAD, _TAIL)
    marked = _PROJECTED_PREFIX in (normalize_content_value(message.get("content")) or "")
    if fields["tool_calls"] != (message.get("tool_calls") or None) or (fields["tool_calls"] and marked):
        # A storage-only call projection keeps a mark when it is projected again.
        fields = projected_fields(row, tokens, _HEAD, _TAIL, mark_calls=True)
    stored_text = normalize_content_value(row.get("content")) or ""
    if (fields["content"] == stored_text and len(stored_text) <= _HEAD
            and normalize_content_value(message.get("content")) == stored_text):
        fields["content"] = message.get("content")  # the same content, kept in its structured form
    projected = {key: value for key, value in {**message, **fields}.items()
                 if key != "tool_calls" or value}
    return projected


def project_message_legacy(current: Dict[str, Any], row: Dict[str, Any], tokens: int) -> Dict[str, Any]:
    """Build the mark-only candidate; the caller decides whether to use it."""
    fields = projected_fields(row, tokens, _HEAD, _TAIL, legacy=True)
    projected = {key: value for key, value in {**current, **fields}.items()
                 if key != "tool_calls" or value}
    return projected


def projection_mark_candidates(content: str, calls: Any) -> list:
    """Parse emitted positions and bounded scans in replay-verification order."""
    haystack = content if _PROJECTED_PREFIX in content else ""
    call_text = ""
    emitted_matches, emitted_keys = [], set()
    if isinstance(calls, list):
        arguments = [str((call.get("function") or {}).get("arguments") or "") for call in calls
                     if isinstance(call, dict) and isinstance(call.get("function"), dict)
                     and "lcm_survival_fit" in str(call["function"].get("arguments") or "")]
        call_text = "".join(arguments)
        for argument in arguments:
            if len(argument) > 1024 or not argument.startswith('{"lcm_survival_fit": "'):
                # Only the emitted bounded form is parsed; it is a short notice.
                continue
            try:
                bounded = json.loads(argument)
            except (ValueError, RecursionError):
                continue
            if (isinstance(bounded, dict) and set(bounded) == {"lcm_survival_fit"}
                    and isinstance(bounded["lcm_survival_fit"], str)):
                match = _PROJECTED_RE.match(bounded["lcm_survival_fit"])
                if match is not None and match.groups() not in emitted_keys:
                    emitted_keys.add(match.groups())
                    emitted_matches.append(match)
                    if len(emitted_matches) == 4:  # the same candidate bound as the scans below
                        break
    if _PROJECTED_PREFIX not in haystack + call_text:
        return []
    # Emitted positions precede the capped scans of pasted content and arguments.
    matches = emitted_matches + list(islice(_PROJECTED_RE.finditer(haystack), 4)) + list(
        islice(_PROJECTED_RE.finditer(call_text), 4))
    emitted = _PROJECTED_RE.match(content, _HEAD + len("\n...\n"))
    if emitted is not None:
        matches.insert(0, emitted)
    separator = content.rfind("\n...\n")
    ending = _PROJECTED_RE.fullmatch(content, separator + len("\n...\n") if separator >= 0 else 0)
    if ending is not None:
        matches.insert(0, ending)
    return matches
