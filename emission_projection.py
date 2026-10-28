"""Proof-bound projection of generated emission spans onto their source rows."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional

from .message_content import normalize_content_value

@dataclass(frozen=True)
class EmissionProjectionEntry:
    full_identity: tuple[str, str, str, str, str]
    effective_identity: tuple[str, str, str, str, str]
    generated_span: Optional[str] = None
    retained_source: Any = None
    kind: Optional[str] = None
    suffix_sha256: Optional[str] = None
    suffix_length: Optional[int] = None
    output_index: Optional[int] = None


@dataclass(frozen=True)
class EmissionProjection:
    entries: tuple[EmissionProjectionEntry, ...]
    offset: int = 0
    complete_length: int = 0

    def slice(self, offset: int, length: Optional[int] = None) -> "EmissionProjection":
        stop = None if length is None else offset + length
        return EmissionProjection(self.entries[offset:stop], self.offset + offset, self.complete_length)


def _emission_identity(message: Mapping[str, Any], content: Optional[str] = None) -> tuple[str, str, str, str, str]:
    role = str(message.get("role") or "unknown")
    normalized = normalize_content_value(message.get("content")) or "" if content is None else content
    tool_calls = json.dumps(
        message.get("tool_calls") or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    tool_name = str(message.get("tool_name") or message.get("name") or "") if role == "tool" else ""
    return role, normalized, str(message.get("tool_call_id") or ""), tool_calls, tool_name


def _emission_candidate_rows(messages: Sequence[dict[str, Any]], role: Any, span: str | int, digest: str | None = None) -> Iterator[tuple[int, dict[str, Any]]]:
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("tool_calls") or message.get("tool_call_id"):
            continue
        content = message.get("content")
        if str(message.get("role") or "unknown") != role or not isinstance(content, str):
            continue
        raw = content.encode("utf-8")
        if (content.startswith(span) if isinstance(span, str)
                else len(raw) >= span and hashlib.sha256(raw[:span]).hexdigest() == digest):
            yield index, message


def _finalize_emission_descriptors(messages: Sequence[dict[str, Any]], candidates: Sequence[Mapping[str, Any]], scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind assembly candidates to the exact returned occurrences."""
    descriptors: list[dict[str, Any]] = []
    search_from = 0
    for candidate in candidates:
        kind, span = candidate.get("kind"), candidate.get("span")
        if kind not in {"summary", "carrier", "objective", "recovery"} or not isinstance(span, str) or not span:
            continue
        expected_identity = candidate.get("full_identity")
        candidate_rows = list(_emission_candidate_rows(messages, expected_identity[0] if isinstance(expected_identity, tuple) else None, span))
        remaining = [(index, message) for index, message in candidate_rows if index >= search_from]
        bound = [(index, message) for index, message in remaining if message is candidate.get("row")]
        if not bound:
            bound = [(index, message) for index, message in remaining
                     if _emission_identity(message) == expected_identity]
        if len(bound) != 1:
            continue
        for index in range(search_from, len(messages)):
            if index != bound[0][0]:
                continue
            message = messages[index]
            content = message.get("content") if isinstance(message, dict) else None
            identity = _emission_identity(message) if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.startswith(span) or (
                expected_identity is not None and identity != expected_identity
                and message is not candidate.get("row")
            ) or message.get("tool_calls") or message.get("tool_call_id"):
                continue
            span_bytes = span.encode("utf-8")
            suffix_bytes = content.encode("utf-8")[len(span_bytes):]
            normalized_suffix = b"" if kind in {"summary", "objective"} and candidate.get("retained_source") is None else suffix_bytes.decode("utf-8").strip().encode("utf-8")
            suffix_sha256 = hashlib.sha256(normalized_suffix).hexdigest()  # the bound row's current suffix
            suffix_length = len(normalized_suffix)
            role = str(message.get("role") or "unknown")
            ordinal = sum(
                _emission_identity(previous) == identity
                for previous in messages[: index + 1]
                if isinstance(previous, dict)
            ) - 1
            same_prefix_ordinal = [row_index for row_index, _ in candidate_rows].index(index)
            descriptors.append({
                "kind": kind,
                "role": role,
                "same_prefix_ordinal": same_prefix_ordinal,
                "output_occurrence": {"index": index, "same_identity_ordinal": ordinal, "of": len(messages)},
                "generated_span_sha256": hashlib.sha256(span_bytes).hexdigest(),
                "generated_span_bytes": len(span_bytes),
                "suffix_sha256": suffix_sha256,
                "suffix_length": suffix_length,
                "retained_source": candidate.get("retained_source"),
                "scope": dict(scope),
                # B2 (additive, optional): the engine uid the bound row carries; readers never require it.
                **({"engine_uid": message["message_uid"]} if candidate.get("engine_uid") is not None
                   and message.get("message_uid") == candidate["engine_uid"] else {}),
            })
            if kind == "recovery" and index > 0:
                previous = messages[index - 1]
                if previous.get("role") == "user" and isinstance(previous.get("content"), str) and not (
                    previous.get("tool_calls") or previous.get("tool_call_id")
                ):
                    base = previous["content"].strip().encode("utf-8")
                    descriptors[-1]["merge_base"] = {"sha256": hashlib.sha256(base).hexdigest(), "bytes": len(base)}
            search_from = index + 1
            break
    return descriptors


def _carrier_emission_key(descriptor_or_entry: Any) -> tuple[Any, Any, Any]:
    """#1000: one carrier emission, from its descriptor or its projected entry: the generated span, the suffix
    and the stored row it was formed around (the process-local retirement key; no output index, so a
    carried-forward copy of a retired carrier is retired too)."""
    if isinstance(descriptor_or_entry, EmissionProjectionEntry):
        span = descriptor_or_entry.generated_span
        span_sha256 = hashlib.sha256(span.encode("utf-8")).hexdigest() if isinstance(span, str) else None
        suffix_sha256, source = descriptor_or_entry.suffix_sha256, descriptor_or_entry.retained_source
    else:
        span_sha256 = descriptor_or_entry.get("generated_span_sha256")
        suffix_sha256, source = descriptor_or_entry.get("suffix_sha256"), descriptor_or_entry.get("retained_source")
    return span_sha256, suffix_sha256, source.get("store_id") if isinstance(source, Mapping) else None


def _descriptor_shape_is_well_formed(descriptor: Any) -> bool:
    """The nested shapes proof consumers index into (#514); projection still declines the rest."""
    occurrence = descriptor.get("output_occurrence") if isinstance(descriptor, Mapping) else None
    if not isinstance(occurrence, Mapping) or not isinstance(descriptor.get("scope"), Mapping):
        return False
    index = occurrence.get("index")
    if type(index) is not int or index < 0:
        return False
    return "of" not in occurrence or (type(occurrence["of"]) is int and occurrence["of"] > index)


def _project_emitted_occurrences(
    messages: Sequence[dict[str, Any]], *, proof: Mapping[str, Any] | None
) -> EmissionProjection:
    """Project proof-bound generated prefixes without changing stored identity."""
    entries = [
        EmissionProjectionEntry(_emission_identity(message), _emission_identity(message))
        for message in messages
    ]
    if not isinstance(proof, Mapping) or proof.get("version") != 4:
        return EmissionProjection(tuple(entries), complete_length=len(entries))
    expected_binding = {key: proof.get(key) for key in ("hermes_home", "session_id", "conversation_id", "reset_epoch")}
    search_from = 0
    for descriptor in proof.get("emissions") or ():
        if not isinstance(descriptor, Mapping) or descriptor.get("kind") not in {
            "summary", "carrier", "objective", "recovery"
        } or descriptor.get("scope") != expected_binding:
            continue
        length = descriptor.get("generated_span_bytes")
        digest = descriptor.get("generated_span_sha256")
        suffix_length = descriptor.get("suffix_length")
        suffix_digest = descriptor.get("suffix_sha256")
        role = descriptor.get("role")
        ordinal = descriptor.get("same_prefix_ordinal")
        occurrence: Any = descriptor.get("output_occurrence")
        output_index = occurrence.get("index") if isinstance(occurrence, Mapping) else None
        has_length = isinstance(occurrence, Mapping) and "of" in occurrence  # recorded by the finalizer
        output_length = occurrence.get("of") if has_length else None
        output = proof.get("output")
        if not isinstance(output, (list, tuple)) or not all(isinstance(item, (list, tuple)) for item in output):
            output = ()  # a malformed output carries no bound and no multiplicity witness
        if not isinstance(length, int) or length <= 0 or not isinstance(digest, str) or (
            not isinstance(role, str) or not role or type(ordinal) is not int or ordinal < 0
        ) or type(suffix_length) is not int or suffix_length < 0 or not isinstance(suffix_digest, str) or (
            type(output_index) is not int or output_index < 0  # malformed (F7): decline, never raise
        ) or (has_length and (type(output_length) is not int or output_index >= output_length)) or (
            output and output_index >= len(output)  # a malformed or exceeded recorded length, or an index beyond the output
        ):
            continue
        candidate_ordinal = -1
        for index, message in _emission_candidate_rows(messages, role, length, digest):
            content = message.get("content")
            if str(message.get("role") or "unknown") != role or message.get("tool_calls") or (
                message.get("tool_call_id")
            ) or not isinstance(content, str):
                continue
            raw = content.encode("utf-8")
            if len(raw) < length or hashlib.sha256(raw[:length]).hexdigest() != digest:
                continue
            candidate_ordinal += 1
            if candidate_ordinal != ordinal:
                continue
            if index < search_from:
                break
            try:
                span, suffix = raw[:length].decode("utf-8"), raw[length:].decode("utf-8")
            except UnicodeDecodeError:
                continue
            normalized_suffix = suffix.strip().encode("utf-8")
            if len(normalized_suffix) < suffix_length or hashlib.sha256(
                normalized_suffix[:suffix_length]
            ).hexdigest() != suffix_digest:
                continue
            if not suffix_length and not normalized_suffix:
                multiplicity = sum(
                    tuple(item) == tuple(output[output_index]) for item in output
                ) if output_index < len(output) else descriptor.get("output_multiplicity")
                if type(multiplicity) is not int or multiplicity > sum(message.get("content") == span for _, message in _emission_candidate_rows(messages, role, length, digest)):
                    continue
            retained_source = descriptor.get("retained_source")
            if isinstance(retained_source, Mapping):
                retained_source = MappingProxyType(dict(retained_source))
            entries[index] = EmissionProjectionEntry(
                entries[index].full_identity,
                _emission_identity(messages[index], suffix),
                span,
                retained_source,
                str(descriptor["kind"]),
                suffix_digest,
                suffix_length,
                output_index,
            )
            search_from = index + 1
            break
        else:
            base = descriptor.get("merge_base")
            if descriptor["kind"] != "recovery" or not isinstance(base, Mapping):
                continue
            for index in range(search_from, min(output_index, len(messages))):
                message = messages[index]
                content = message.get("content")
                if message.get("role") != "user" or not isinstance(content, str) or (
                    message.get("tool_calls") or message.get("tool_call_id")
                ):
                    continue
                raw = content.encode("utf-8")
                if raw[-length - 2:-length] != b"\n\n" or hashlib.sha256(raw[-length:]).hexdigest() != digest:
                    continue
                older = raw[:-length - 2].decode("utf-8")
                normalized = older.strip().encode("utf-8")
                if len(normalized) != base.get("bytes") or hashlib.sha256(normalized).hexdigest() != base.get("sha256"):
                    continue
                entries[index] = EmissionProjectionEntry(
                    entries[index].full_identity, _emission_identity(message, older),
                    raw[-length:].decode("utf-8"), kind="recovery_base", output_index=output_index,
                )
                search_from = index + 1
                break
    return EmissionProjection(tuple(entries), complete_length=len(entries))
