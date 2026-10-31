"""Expansion provenance and continuation payload construction."""

from __future__ import annotations

import hashlib
import json
from typing import Any


_EXPAND_QUERY_EVIDENCE_QUOTE_MAX_CHARS = 500
_EXPAND_QUERY_EVIDENCE_MAX_ITEMS = 24
_EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS = 10_000
_EXPAND_QUERY_EVIDENCE_METADATA_MAX_CHARS = 256
_EXPAND_QUERY_EVIDENCE_SOURCE_PATH_MAX_HOPS = 8
_EXPAND_QUERY_EVIDENCE_SOURCE_PATH_MAX_PATHS = 8


def _expand_query_evidence_serialized_chars(evidence: dict[str, Any]) -> int:
    """Measure the exact JSON representation used by the tool response."""

    return len(json.dumps(evidence))


def _bounded_evidence_text(value: Any) -> tuple[str, dict[str, Any] | None]:
    """Bound metadata while retaining explicit length and digest disclosure."""

    text = str(value or "")
    if len(text) <= _EXPAND_QUERY_EVIDENCE_METADATA_MAX_CHARS:
        return text, None
    return text[:_EXPAND_QUERY_EVIDENCE_METADATA_MAX_CHARS], {
        "original_chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _normalized_evidence_source_path(
    value: Any,
    *,
    source_path_depth: Any = None,
    source_path_truncated: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(value, (list, tuple)):
        return None
    result: list[dict[str, int]] = []
    for part in value[-_EXPAND_QUERY_EVIDENCE_SOURCE_PATH_MAX_HOPS:]:
        if not isinstance(part, dict):
            continue
        node_id = part.get("node_id")
        source_index = part.get("source_index")
        if isinstance(node_id, int) and isinstance(source_index, int):
            result.append({"node_id": node_id, "source_index": source_index})
    if not result:
        return None
    depth = (
        source_path_depth
        if isinstance(source_path_depth, int) and source_path_depth >= len(result)
        else len(result)
    )
    return {
        "path": result,
        "depth": depth,
        "truncated": bool(source_path_truncated or depth > len(result)),
    }


def _evidence_source_path_with_final_edge(
    block: dict[str, Any],
    source_index: Any,
) -> tuple[Any, Any, Any]:
    """Return a block path extended through its selected child/message edge."""

    source_path = block.get("source_path")
    path = list(source_path) if isinstance(source_path, (list, tuple)) else []
    depth = block.get("source_path_depth")
    if not isinstance(depth, int) or depth < len(path):
        depth = len(path)
    node_id = block.get("node_id")
    if isinstance(node_id, int) and isinstance(source_index, int):
        path.append({"node_id": node_id, "source_index": source_index})
        depth += 1
    return path, depth, block.get("source_path_truncated")


def _build_expand_query_evidence(
    context_blocks: list[dict[str, Any]],
    *,
    session_id: str,
    context_truncated: bool,
    synthesis_status: str,
    session_ids: list[str] | None = None,
    scope_kind: str = "current_session",
) -> dict[str, Any]:
    """Return bounded, tool-extracted provenance for expansion synthesis context.

    The bundle identifies excerpts that were actually sent to the auxiliary
    model. It deliberately does not claim that the resulting answer is
    semantically entailed by those excerpts.

    ``session_id`` is the ACTIVE session: it is the per-item attribution
    fallback and decides whether a summary locator needs an explicit session.
    ``session_ids``/``scope_kind`` describe the sessions actually searched, which
    for an explicit cross-session retrieval is not the active session alone.
    """

    synthesis_status = (
        synthesis_status
        if synthesis_status in {"completed", "failed", "not_run"}
        else "unknown"
    )
    candidates: list[dict[str, Any]] = []
    seen_items: dict[str, dict[str, Any]] = {}

    def put_bounded_text(item: dict[str, Any], field: str, value: Any) -> bool:
        bounded, truncation = _bounded_evidence_text(value)
        item[field] = bounded
        if truncation is not None:
            item.setdefault("metadata_truncation", {})[field] = truncation
            return True
        return False

    def add_candidate(
        identity: str,
        item: dict[str, Any],
        *,
        source_path: Any = None,
        source_path_depth: Any = None,
        source_path_truncated: Any = None,
    ) -> None:
        path = _normalized_evidence_source_path(
            source_path,
            source_path_depth=source_path_depth,
            source_path_truncated=source_path_truncated,
        )
        existing = seen_items.get(identity)
        if existing is not None:
            existing["context_occurrence_count"] += 1
            source_paths = existing.setdefault("source_paths", [])
            if path and path not in source_paths:
                if len(source_paths) < _EXPAND_QUERY_EVIDENCE_SOURCE_PATH_MAX_PATHS:
                    source_paths.append(path)
                else:
                    existing["source_paths_truncated"] = True
            return
        put_bounded_text(item, "evidence_id", identity)
        item["context_occurrence_count"] = 1
        if path:
            item["source_paths"] = [path]
        seen_items[identity] = item
        candidates.append(item)

    def summary_expand_args(node_id: int, item: dict[str, Any]) -> dict[str, Any]:
        # Mirrors `_lcm_recall_summary_expand_hint`: a summary locator carries
        # its session only when the node is not the active session's, because a
        # bare node_id replays against whatever session is active now.
        expand_args: dict[str, Any] = {"node_id": node_id}
        item_session_id = item.get("session_id") or ""
        if (
            item_session_id
            and item_session_id != session_id
            and "session_id" not in (item.get("metadata_truncation") or {})
        ):
            expand_args["session_id"] = item_session_id
        return expand_args

    def quote_payload(content: Any) -> dict[str, Any]:
        quote = str(content or "")
        return {
            "quote": quote[:_EXPAND_QUERY_EVIDENCE_QUOTE_MAX_CHARS],
            "quote_chars_before_provenance_cap": len(quote),
            "quote_truncated_by_provenance_cap": len(quote)
            > _EXPAND_QUERY_EVIDENCE_QUOTE_MAX_CHARS,
        }

    for block_index, block in enumerate(context_blocks):
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")

        if block_type == "summary":
            node_id = block.get("node_id")
            identity = (
                f"node:{node_id}"
                if isinstance(node_id, int)
                else f"context:{block_index}:summary"
            )
            item: dict[str, Any] = {
                "source_type": "summary",
                "node_id": node_id if isinstance(node_id, int) else None,
                "locator_present": isinstance(node_id, int),
                "locator_replay_status": "unverified" if isinstance(node_id, int) else "not_available",
                **quote_payload(block.get("summary")),
            }
            put_bounded_text(item, "session_id", block.get("session_id") or session_id)
            if isinstance(node_id, int):
                item["expand_args"] = summary_expand_args(node_id, item)
            add_candidate(
                identity,
                item,
                source_path=block.get("source_path"),
                source_path_depth=block.get("source_path_depth"),
                source_path_truncated=block.get("source_path_truncated"),
            )

        if block_type in {"child_nodes", "descendant_child_nodes"}:
            for child_index, child in enumerate(block.get("children", []) or []):
                if not isinstance(child, dict):
                    continue
                node_id = child.get("node_id")
                identity = (
                    f"node:{node_id}"
                    if isinstance(node_id, int)
                    else f"context:{block_index}:child:{child_index}"
                )
                item = {
                    "source_type": "summary",
                    "node_id": node_id if isinstance(node_id, int) else None,
                    "parent_node_id": (
                        block.get("node_id")
                        if isinstance(block.get("node_id"), int)
                        else None
                    ),
                    "source_index": (
                        child.get("source_index")
                        if isinstance(child.get("source_index"), int)
                        else None
                    ),
                    "locator_present": isinstance(node_id, int),
                    "locator_replay_status": "unverified" if isinstance(node_id, int) else "not_available",
                    **quote_payload(child.get("summary")),
                }
                put_bounded_text(
                    item,
                    "session_id",
                    child.get("session_id") or block.get("session_id") or session_id,
                )
                if isinstance(node_id, int):
                    item["expand_args"] = summary_expand_args(node_id, item)
                source_path, source_path_depth, source_path_truncated = (
                    _evidence_source_path_with_final_edge(
                        block,
                        child.get("source_index"),
                    )
                )
                add_candidate(
                    identity,
                    item,
                    source_path=source_path,
                    source_path_depth=source_path_depth,
                    source_path_truncated=source_path_truncated,
                )

        if block_type not in {"messages", "child_messages", "raw_messages"}:
            continue
        for message_index, message in enumerate(block.get("messages", []) or []):
            if not isinstance(message, dict):
                continue
            store_id = message.get("store_id")
            content_offset = message.get("content_offset")
            if not isinstance(content_offset, int) or content_offset < 0:
                content_offset = 0
            content_source = str(message.get("content_source") or "message")
            externalized = message.get("externalized") or {}
            externalized_ref = externalized.get("ref") if isinstance(externalized, dict) else None
            if content_source == "externalized_payload" and externalized_ref:
                identity = f"externalized:{externalized_ref}:{content_offset}"
                source_type = "externalized_payload"
            elif isinstance(store_id, int):
                identity = f"store:{store_id}:{content_offset}:{content_source}"
                source_type = "raw_message"
            else:
                identity = f"context:{block_index}:message:{message_index}"
                source_type = "raw_message"

            item = {
                "source_type": source_type,
                "store_id": store_id if isinstance(store_id, int) else None,
                "node_id": (
                    block.get("node_id")
                    if isinstance(block.get("node_id"), int)
                    else None
                ),
                "parent_node_id": (
                    block.get("parent_node_id")
                    if isinstance(block.get("parent_node_id"), int)
                    else None
                ),
                "source_index": (
                    message.get("source_index")
                    if isinstance(message.get("source_index"), int)
                    else None
                ),
                "content_offset": content_offset,
                **quote_payload(message.get("content")),
            }
            put_bounded_text(item, "session_id", message.get("session_id") or "")
            put_bounded_text(item, "source", message.get("source") or "")
            put_bounded_text(item, "role", message.get("role"))
            put_bounded_text(item, "content_source", content_source)
            ref_truncated = put_bounded_text(item, "externalized_ref", externalized_ref)
            if content_source == "externalized_payload":
                if externalized_ref and not ref_truncated:
                    item["expand_args"] = {
                        "externalized_ref": str(externalized_ref),
                        "content_offset": content_offset,
                    }
            elif isinstance(store_id, int):
                item["expand_args"] = {"store_id": store_id, "content_offset": content_offset}
            item["locator_present"] = "expand_args" in item
            item["locator_replay_status"] = (
                "unverified" if item["locator_present"] else "not_available"
            )
            source_path = block.get("source_path")
            source_path_depth = block.get("source_path_depth")
            source_path_truncated = block.get("source_path_truncated")
            if block_type in {"messages", "child_messages"}:
                source_path, source_path_depth, source_path_truncated = (
                    _evidence_source_path_with_final_edge(
                        block,
                        message.get("source_index"),
                    )
                )
            add_candidate(
                identity,
                item,
                source_path=source_path,
                source_path_depth=source_path_depth,
                source_path_truncated=source_path_truncated,
            )

            # Externalized/hydrated context blocks can contain both the text
            # exposed as ``content`` and the durable transcript representation
            # exposed as ``transcript_content``. Both fields are serialized into
            # the auxiliary model prompt, so traceability cannot be called
            # complete unless both are represented when they differ.
            transcript_content = message.get("transcript_content")
            if transcript_content is not None and str(transcript_content) != str(
                message.get("content") or ""
            ):
                transcript_identity = (
                    f"store:{store_id}:0:transcript_content"
                    if isinstance(store_id, int)
                    else f"context:{block_index}:message:{message_index}:transcript_content"
                )
                transcript_item = {
                    "source_type": "raw_message",
                    "store_id": store_id if isinstance(store_id, int) else None,
                    "node_id": (
                        block.get("node_id")
                        if isinstance(block.get("node_id"), int)
                        else None
                    ),
                    "parent_node_id": (
                        block.get("parent_node_id")
                        if isinstance(block.get("parent_node_id"), int)
                        else None
                    ),
                    "source_index": (
                        message.get("source_index")
                        if isinstance(message.get("source_index"), int)
                        else None
                    ),
                    "content_offset": 0,
                    "locator_present": isinstance(store_id, int),
                    "locator_replay_status": "unverified" if isinstance(store_id, int) else "not_available",
                    **quote_payload(transcript_content),
                }
                put_bounded_text(transcript_item, "session_id", message.get("session_id") or "")
                put_bounded_text(transcript_item, "source", message.get("source") or "")
                put_bounded_text(transcript_item, "role", message.get("role"))
                put_bounded_text(transcript_item, "content_source", "transcript_content")
                if isinstance(store_id, int):
                    transcript_item["expand_args"] = {
                        "store_id": store_id,
                        "content_offset": 0,
                    }
                add_candidate(
                    transcript_identity,
                    transcript_item,
                    source_path=source_path,
                    source_path_depth=source_path_depth,
                    source_path_truncated=source_path_truncated,
                )

    items = candidates[:_EXPAND_QUERY_EVIDENCE_MAX_ITEMS]
    scope_kind = (
        scope_kind if scope_kind in {"current_session", "explicit_sessions"} else "unknown"
    )
    scope_sessions = list(session_ids) if session_ids else [session_id]
    bounded_scope_sessions: list[str] = []
    scope_truncations: dict[str, Any] = {}
    for scope_index, scope_session in enumerate(scope_sessions):
        bounded_scope_session, scope_truncation = _bounded_evidence_text(scope_session)
        bounded_scope_sessions.append(bounded_scope_session)
        if scope_truncation is not None:
            scope_truncations[f"session_ids[{scope_index}]"] = scope_truncation
    retrieval_scope: dict[str, Any] = {
        "kind": scope_kind,
        "session_ids": bounded_scope_sessions,
    }
    if scope_truncations:
        retrieval_scope["metadata_truncation"] = scope_truncations
    evidence = {
        "retrieval_scope": retrieval_scope,
        "identifiers_are_authority": False,
        "locator_replay_safety": "not_guaranteed",
        "synthesis_status": synthesis_status,
        "semantic_entailment": "not_verified" if synthesis_status == "completed" else "not_applicable",
        "quote_origin": "tool_extracted_from_synthesis_context",
        "context_truncated": bool(context_truncated),
        "context_unique_item_count": len(candidates),
        "context_occurrence_count": sum(
            int(item.get("context_occurrence_count") or 1) for item in candidates
        ),
        "serialized_char_limit": _EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS,
        "serialization": {
            "scope": "evidence_provenance_only",
            "json_ensure_ascii": True,
        },
        "quote_max_chars": _EXPAND_QUERY_EVIDENCE_QUOTE_MAX_CHARS,
        "items": items,
    }

    evidence = _select_expand_query_evidence_budget(
        items=items, candidates=candidates, evidence=evidence, retrieval_scope=retrieval_scope,
        scope_kind=scope_kind, scope_sessions=scope_sessions,
        synthesis_status=synthesis_status, context_truncated=context_truncated,
    )
    return evidence


def _select_expand_query_evidence_budget(
    *,
    items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    evidence: dict[str, Any],
    retrieval_scope: dict[str, Any],
    scope_kind: str,
    scope_sessions: list[str],
    synthesis_status: str,
    context_truncated: bool,
) -> dict[str, Any]:
    def refresh_bounds() -> None:
        items_truncated = len(items) < len(candidates)
        located_items = sum(1 for item in items if item.get("locator_present"))
        if not candidates:
            locator_coverage = "none"
        elif items_truncated or located_items < len(candidates):
            locator_coverage = "partial"
        else:
            locator_coverage = "complete"
        evidence["locator_coverage"] = locator_coverage
        evidence["items_truncated"] = items_truncated
        evidence["quotes_truncated_by_provenance_cap"] = sum(
            1 for item in items if item.get("quote_truncated_by_provenance_cap")
        )
        evidence["metadata_truncated"] = bool(
            retrieval_scope.get("metadata_truncation")
            or any(item.get("metadata_truncation") for item in items)
        )

    refresh_bounds()
    while (
        items
        and _expand_query_evidence_serialized_chars(evidence)
        > _EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS
    ):
        items.pop()
        refresh_bounds()
    if (
        _expand_query_evidence_serialized_chars(evidence)
        > _EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS
    ):
        # Bounded metadata makes this unreachable under normal inputs. Keep a
        # compact fail-safe so even a pathological fixed envelope cannot violate
        # the public nested-object contract.
        evidence = {
            "retrieval_scope": {
                "kind": scope_kind,
                "session_ids": [],
                "session_id_sha256": hashlib.sha256(
                    "\n".join(str(scope or "") for scope in scope_sessions).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "session_id_omitted_due_to_size": True,
            },
            "identifiers_are_authority": False,
            "locator_replay_safety": "not_guaranteed",
            "synthesis_status": synthesis_status,
            "semantic_entailment": (
                "not_verified" if synthesis_status == "completed" else "not_applicable"
            ),
            "quote_origin": "tool_extracted_from_synthesis_context",
            "context_truncated": bool(context_truncated),
            "context_unique_item_count": len(candidates),
            "context_occurrence_count": sum(
                int(item.get("context_occurrence_count") or 1) for item in candidates
            ),
            "serialized_char_limit": _EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS,
            "serialization": {
                "scope": "evidence_provenance_only",
                "json_ensure_ascii": True,
            },
            "quote_max_chars": _EXPAND_QUERY_EVIDENCE_QUOTE_MAX_CHARS,
            "locator_coverage": "none" if not candidates else "partial",
            "items_truncated": bool(candidates),
            "quotes_truncated_by_provenance_cap": 0,
            "metadata_truncated": True,
            "fixed_envelope_compacted": True,
            "items": [],
        }
    return evidence

def _serialized_context_token_count(blocks: list[dict[str, Any]]) -> int:
    from .tokens import count_tokens

    # Budget against the encoding the caller actually receives. Every
    # ``lcm_expand_query`` return goes out as ``json.dumps(payload)`` with the
    # default ``ensure_ascii=True``, which escapes non-ASCII to ``\uXXXX`` (a
    # surrogate pair, 12 characters, per astral emoji). Counting an
    # ``ensure_ascii=False`` serialization here under-counts CJK/emoji evidence
    # by a multiple, so the emitted payload can overrun ``context_max_tokens``
    # while ``context_tokens`` reports it inside budget. ASCII evidence is
    # byte-identical under both encodings, so this only tightens the
    # non-ASCII case.
    return count_tokens(json.dumps(blocks))


def continuations(
    blocks: list[dict[str, Any]],
    omitted_raw_cursor: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return continuation handles in the order exposed by the context blocks."""
    context_pagination = []
    pagination_blocks = blocks
    if omitted_raw_cursor is not None:
        pagination_blocks = [*blocks, omitted_raw_cursor]
    for block in pagination_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "summary" and block.get("summary_truncated"):
            item: dict[str, Any] = {
                "node_id": block.get("node_id"),
                "type": "summary",
                "summary_truncated": True,
                "expand_args": {"node_id": block.get("node_id")},
            }
            if block.get("session_id") is not None:
                item["session_id"] = block["session_id"]
                item["expand_args"]["session_id"] = block["session_id"]
            context_pagination.append(item)
            continue

        if block_type in {"child_nodes", "descendant_child_nodes"}:
            for child in block.get("children", []):
                if child.get("summary_truncated"):
                    child_node_id = child.get("node_id")
                    item = {
                        "node_id": block.get("node_id"),
                        "type": "child_summary"
                        if block_type == "child_nodes"
                        else "descendant_child_summary",
                        "child_node_id": child_node_id,
                        "source_index": child.get("source_index"),
                        "summary_truncated": True,
                        "expand_args": {"node_id": child_node_id},
                    }
                    if block.get("session_id") is not None:
                        item["session_id"] = block["session_id"]
                        item["expand_args"]["session_id"] = block["session_id"]
                    context_pagination.append(item)

        pagination = block.get("pagination")
        if not pagination or not pagination.get("has_more"):
            continue

        item = {
            "node_id": block.get("node_id"),
            "type": block_type,
            "pagination": pagination,
        }
        if block.get("session_id") is not None:
            item["session_id"] = block.get("session_id")
        if block_type in {"messages", "child_messages"}:
            truncated_message = next(
                (
                    message
                    for message in block.get("messages", [])
                    if message.get("content_truncated")
                ),
                None,
            )
            if truncated_message:
                item["source_index"] = truncated_message.get("source_index")
                item["content_source"] = truncated_message.get("content_source")
                externalized = truncated_message.get("externalized") or {}
                externalized_ref = externalized.get("ref")
                if externalized_ref:
                    item["externalized_ref"] = externalized_ref
                    item["tool_call_id"] = externalized.get("tool_call_id")
                if (
                    truncated_message.get("content_source") == "externalized_payload"
                    and externalized_ref
                ):
                    item["expand_args"] = {
                        "externalized_ref": externalized_ref,
                        "content_offset": pagination.get("next_content_offset") or 0,
                    }
                else:
                    item["expand_args"] = {
                        "node_id": block.get("node_id"),
                        "source_offset": pagination.get("next_source_offset") or 0,
                        "content_offset": pagination.get("next_content_offset") or 0,
                    }
            else:
                item["expand_args"] = {
                    "node_id": block.get("node_id"),
                    "source_offset": pagination.get("next_source_offset") or 0,
                    "content_offset": pagination.get("next_content_offset") or 0,
                }
        elif block_type == "raw_messages":
            truncated_message = next(
                (
                    message
                    for message in block.get("messages", [])
                    if message.get("content_truncated")
                ),
                None,
            )
            if truncated_message:
                item["store_id"] = truncated_message.get("store_id")
                item["content_source"] = truncated_message.get("content_source")
                item["expand_args"] = {
                    "store_id": truncated_message.get("store_id"),
                    "content_offset": truncated_message.get("next_content_offset") or 0,
                }
            elif pagination.get("next_store_id"):
                item["store_id"] = pagination.get("next_store_id")
                item["expand_args"] = {"store_id": pagination.get("next_store_id")}
        elif block_type in {"child_nodes", "descendant_child_nodes"}:
            item["expand_args"] = {
                "node_id": block.get("node_id"),
                "source_offset": pagination.get("next_source_offset") or 0,
            }
        if item.get("session_id") is not None and "node_id" in item.get(
            "expand_args", {}
        ):
            item["expand_args"]["session_id"] = item["session_id"]
        context_pagination.append(item)

    return context_pagination


def _pagination_payload(
    *,
    total_sources: int,
    source_offset: int,
    content_offset: int,
    source_limit: int,
    returned_sources: int,
    next_source_offset: int | None,
    next_content_offset: int,
    has_more: bool,
) -> dict[str, Any]:
    if not has_more:
        next_source_offset = None
        next_content_offset = 0
    remaining_sources = 0
    if has_more and next_source_offset is not None:
        remaining_sources = max(0, total_sources - next_source_offset)
    return {
        "source_offset": source_offset,
        "content_offset": content_offset,
        "source_limit": source_limit,
        "returned_sources": returned_sources,
        "total_sources": total_sources,
        "next_source_offset": next_source_offset,
        "next_content_offset": next_content_offset,
        "has_more": has_more,
        "remaining_sources": remaining_sources,
    }


def _bounded_source_path_payload(source_path: list[dict[str, int]]) -> dict[str, Any]:
    path_tail = source_path[-8:]
    payload: dict[str, Any] = {
        "source_path": path_tail,
        "source_path_depth": len(source_path),
    }
    if len(path_tail) < len(source_path):
        payload["source_path_truncated"] = True
    return payload


def _raw_match_cursor_block(raw_block: dict[str, Any]) -> dict[str, Any] | None:
    """Return a content-free ``raw_messages`` block preserving the raw cursor.

    A raw block dropped for budget reasons leaves matched messages unreturned.
    The caller still needs a handle to continue into them, so the first omitted
    ``store_id`` survives without any of the content that blew the budget.
    """
    pagination = raw_block.get("pagination") or {}
    next_store_id = next(
        (
            message.get("store_id")
            for message in raw_block.get("messages") or []
            if isinstance(message.get("store_id"), int)
        ),
        pagination.get("next_store_id"),
    )
    if not isinstance(next_store_id, int):
        return None
    return {
        "type": "raw_messages",
        "messages": [],
        "pagination": {
            "has_more": True,
            "returned_sources": 0,
            "total_sources": pagination.get("total_sources", 0),
            "next_store_id": next_store_id,
        },
    }


def _collect_store_ids_from_context_blocks(blocks: list[dict[str, Any]]) -> set[int]:
    store_ids: set[int] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for message in block.get("messages", []) or []:
            store_id = message.get("store_id")
            if isinstance(store_id, int):
                store_ids.add(store_id)
    return store_ids


def _context_content_token_count(blocks: list[dict[str, Any]]) -> int:
    from .tokens import count_tokens

    total = 0
    for block in blocks:
        if block.get("type") == "summary":
            total += count_tokens(str(block.get("summary") or ""))
        if "source_path" in block:
            total += count_tokens(
                json.dumps(
                    {
                        "source_path": block.get("source_path") or [],
                        "source_path_depth": block.get("source_path_depth"),
                        "source_path_truncated": block.get("source_path_truncated", False),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if block.get("type") in {"messages", "child_messages", "raw_messages"}:
            for message in block.get("messages", []):
                total += count_tokens(str(message.get("content") or ""))
                total += count_tokens(str(message.get("transcript_content") or ""))
        elif block.get("type") in {"child_nodes", "descendant_child_nodes"}:
            total += sum(count_tokens(str(child.get("summary") or "")) for child in block.get("children", []))
    return total
