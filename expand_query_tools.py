"""Question-driven expansion behind the public tools facade."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, TYPE_CHECKING

from . import expansion_evidence
from . import tools as _tools
from .model_routing import apply_lcm_model_route

if TYPE_CHECKING:
    from .engine import LCMEngine

logger = logging.getLogger(f"{__package__}.tools" if __package__ else "tools")


def _collect_descendant_evidence_blocks(
    engine: "LCMEngine",
    node: Any,
    max_tokens: int,
    *,
    hydrate_externalized_content: bool = False,
    visited_node_ids: set[int] | None = None,
    source_path: list[dict[str, int]] | None = None,
    remaining_node_visits: list[int] | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    if max_tokens <= 0 or node.source_type != "nodes":
        return []
    if visited_node_ids is None:
        visited_node_ids = set()
    if source_path is None:
        source_path = []
    if remaining_node_visits is None:
        # Budget and cycle detection are the primary limits. Keep a high,
        # budget-scaled guard so corrupt zero-token DAGs cannot make expansion
        # walk an unbounded number of nodes while normal deep summaries still
        # reach their leaf evidence.
        remaining_node_visits = [max(64, int(max_tokens) * 4)]
    session_id = engine.current_session_id if session_id is None else session_id
    if remaining_node_visits[0] <= 0:
        return []

    blocks: list[dict[str, Any]] = []
    budget_used = 0
    root_node_id = int(node.node_id)
    stack: list[tuple[Any, list[dict[str, int]], set[int], int]] = [
        (node, source_path, {*visited_node_ids, root_node_id}, 0)
    ]

    while stack and budget_used < max_tokens and remaining_node_visits[0] > 0:
        current, current_path, current_visited, source_index = stack.pop()
        if source_index >= len(current.source_ids):
            continue

        stack.append((current, current_path, current_visited, source_index + 1))
        child_id = current.source_ids[source_index]
        child = engine._dag.get_node(child_id)
        if child is None or child.session_id != session_id:
            continue
        child_node_id = int(child.node_id)
        if child_node_id in current_visited:
            continue

        remaining_node_visits[0] -= 1
        child_path = [
            *current_path,
            {"node_id": int(current.node_id), "source_index": source_index},
        ]
        remaining_tokens = max(0, max_tokens - budget_used)
        if child.source_type == "messages":
            messages, pagination = _tools._expand_message_sources(
                engine,
                child,
                max_tokens=remaining_tokens,
                hydrate_externalized_content=hydrate_externalized_content,
                session_id=session_id,
            )
            if messages or pagination.get("has_more"):
                block = {
                    "type": "child_messages",
                    "parent_node_id": current.node_id,
                    "node_id": child.node_id,
                    "depth": child.depth,
                    "source_index": source_index,
                    **expansion_evidence._bounded_source_path_payload(child_path),
                    "messages": messages,
                    "pagination": pagination,
                }
                blocks.append(block)
                budget_used += expansion_evidence._context_content_token_count([block])
            continue

        if child.source_type == "nodes":
            children, pagination = _tools._expand_child_nodes(
                engine,
                child,
                max_tokens=remaining_tokens,
                session_id=session_id,
            )
            if children or pagination.get("has_more"):
                block = {
                    "type": "descendant_child_nodes",
                    "parent_node_id": current.node_id,
                    "node_id": child.node_id,
                    "depth": child.depth,
                    "source_index": source_index,
                    **expansion_evidence._bounded_source_path_payload(child_path),
                    "children": children,
                    "pagination": pagination,
                }
                blocks.append(block)
                budget_used += expansion_evidence._context_content_token_count([block])
            if budget_used < max_tokens and remaining_node_visits[0] > 0:
                stack.append((child, child_path, {*current_visited, child_node_id}, 0))
    return blocks


def _collect_context_blocks_for_node(
    engine: "LCMEngine",
    node: Any,
    max_tokens: int,
    *,
    hydrate_externalized_content: bool = False,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    from .tokens import count_tokens

    session_id = engine.current_session_id if session_id is None else session_id
    summary, summary_truncated = _tools._truncate_text_to_token_budget(node.summary, max_tokens)
    blocks: list[dict[str, Any]] = [
        {
            "type": "summary",
            "node_id": node.node_id,
            "depth": node.depth,
            "summary": summary,
            "summary_truncated": summary_truncated,
            "expand_hint": node.expand_hint,
            "token_count": node.token_count,
        }
    ]
    remaining_tokens = max(0, max_tokens - count_tokens(summary))

    if node.source_type == "messages":
        messages, pagination = _tools._expand_message_sources(
            engine,
            node,
            max_tokens=remaining_tokens,
            hydrate_externalized_content=hydrate_externalized_content,
            session_id=session_id,
        )
        if messages or pagination.get("has_more"):
            block = {
                "type": "messages",
                "node_id": node.node_id,
                "messages": messages,
                "pagination": pagination,
            }
            blocks.append(block)
    elif node.source_type == "nodes":
        children, pagination = _tools._expand_child_nodes(
            engine,
            node,
            max_tokens=remaining_tokens,
            session_id=session_id,
        )
        if children or pagination.get("has_more"):
            blocks.append(
                {
                    "type": "child_nodes",
                    "node_id": node.node_id,
                    "children": children,
                    "pagination": pagination,
                }
            )
        used_tokens = expansion_evidence._context_content_token_count(blocks)
        descendant_tokens = max(0, max_tokens - used_tokens)
        if descendant_tokens > 0:
            blocks.extend(
                _collect_descendant_evidence_blocks(
                    engine,
                    node,
                    max_tokens=descendant_tokens,
                    hydrate_externalized_content=hydrate_externalized_content,
                    session_id=session_id,
                )
            )

    return blocks


def _collect_raw_match_context_block(
    engine: "LCMEngine",
    rows: list[dict[str, Any]],
    max_tokens: int,
    *,
    query: str | None = None,
    exclude_store_ids: set[int] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    from .tokens import count_tokens

    exclude_store_ids = exclude_store_ids or set()
    messages: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    budget_used = 0
    has_more = False
    next_store_id: int | None = None
    for row in rows:
        store_id = row.get("store_id")
        if store_id in exclude_store_ids:
            continue
        remaining_tokens = max(0, max_tokens - budget_used)
        if remaining_tokens <= 0:
            has_more = True
            next_store_id = store_id if isinstance(store_id, int) else None
            break
        content = str(row.get("content") or "")
        match_offset = _tools._content_offset_for_query_match(content, query)
        content_slice = _tools._slice_content_for_response(content, remaining_tokens, content_offset=match_offset)
        content = content_slice["content"]
        item = {
            "store_id": store_id,
            "session_id": row.get("session_id") or "",
            "source": row.get("source") or "",
            "role": row.get("role"),
            "timestamp": row.get("timestamp", 0),
            **content_slice,
            "content_source": "raw_search_hit",
            "search_rank": row.get("search_rank"),
        }
        if row.get("tool_call_id"):
            item["tool_call_id"] = row.get("tool_call_id")
        if match_offset:
            item["match_window_offset"] = match_offset
        if row.get("tool_calls"):
            item["tool_calls_omitted"] = True
        if row.get("tool_name"):
            item["tool_name"] = row.get("tool_name")
        messages.append(item)
        matches.append(
            {
                "store_id": store_id,
                # Session ownership travels with the match, not just with the
                # context-block message. `lcm_expand_query(output='answer')`
                # returns no context blocks, so for a cross-session raw-only
                # hit this is the caller's only provenance channel.
                "session_id": row.get("session_id") or "",
                "role": row.get("role"),
                "snippet": row.get("snippet") or content[:300],
                "search_rank": row.get("search_rank"),
            }
        )
        budget_used += count_tokens(content)
        if content_slice["has_more"]:
            has_more = True
            break

    if not messages and not has_more:
        return None, matches
    block = {
        "type": "raw_messages",
        "messages": messages,
        "pagination": {
            "has_more": has_more,
            "returned_sources": len(messages),
            "total_sources": len(rows),
            "next_store_id": next_store_id,
        },
    }
    return block, matches


class _ExpansionSynthesisError(RuntimeError):
    """Raised when the expansion aux-LLM call returns an unusable response.

    ``call_llm`` can hand back an error-shaped or partial object whose
    ``choices`` is missing, ``None``, or otherwise non-subscriptable. Surfacing
    that as a typed error lets ``lcm_expand_query`` degrade the same way it
    already degrades on a synthesis timeout, instead of leaking an opaque
    ``TypeError``/``AttributeError`` out of the tool call.
    """


def _synthesize_expansion_answer(
    *,
    prompt: str,
    context_blocks: list[dict[str, Any]],
    model: str,
    reasoning_effort: str = "",
    max_tokens: int,
    timeout: float,
) -> str:
    from agent.auxiliary_client import call_llm  # type: ignore[import-not-found]
    from .model_routing import apply_lcm_reasoning_effort

    system_prompt = (
        "You answer questions using expanded LCM retrieval context. "
        "Be concise, factual, and grounded in the provided context. "
        "If the context is insufficient, say so plainly."
    )
    user_prompt = (
        f"QUESTION:\n{prompt}\n\n"
        "EXPANDED CONTEXT:\n"
        f"{json.dumps(context_blocks, ensure_ascii=False, indent=2)}"
    )
    call_kwargs = {
        "task": "compression",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    apply_lcm_model_route(call_kwargs, model)
    apply_lcm_reasoning_effort(call_kwargs, reasoning_effort)
    response = call_llm(**call_kwargs)
    # ``call_llm`` can return an error-shaped or partial object (e.g. when the
    # auxiliary lane collides with the foreground model's own in-flight calls),
    # in which case ``choices`` may be absent, ``None``, empty, or a
    # non-subscriptable sentinel. Guard the access so an unusable response
    # degrades cleanly at the call site instead of raising an opaque
    # ``TypeError``/``AttributeError`` that kills the whole expand call.
    choices = getattr(response, "choices", None)
    first = None
    if choices is not None:
        try:
            first = choices[0]
        except (TypeError, IndexError, KeyError):
            first = None
    if first is None:
        raise _ExpansionSynthesisError(
            "expansion synthesis returned no usable choices "
            f"(response type={type(response).__name__})"
        )
    message = getattr(first, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        content = str(content) if content else ""
    from .escalation import _strip_reasoning_blocks
    return _strip_reasoning_blocks(content).strip()


def lcm_expand_query(args: Dict[str, Any], **kwargs: Any) -> str:
    """Answer a question by expanding matching summaries or explicit node ids."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return json.dumps({"error": "LCM engine not initialized"})
    session_id = engine.current_session_id

    prompt = str(args.get("prompt") or "").strip()
    if not prompt:
        return json.dumps({"error": "prompt is required"})

    raw_output = args.get("output", "answer")
    if not isinstance(raw_output, str):
        return json.dumps({"error": "output must be one of: answer, evidence"})
    output = raw_output.strip().lower()
    if output not in {"answer", "evidence"}:
        return json.dumps({"error": "output must be one of: answer, evidence"})

    raw_session_ids = args.get("session_ids")
    # An empty optional array means "not provided" (#1194).
    session_ids_explicit = "session_ids" in args and raw_session_ids != []
    if not session_ids_explicit:
        session_ids = [engine.current_session_id]
    elif not isinstance(raw_session_ids, list) or not raw_session_ids:
        return json.dumps({"error": "session_ids must be a non-empty array of strings"})
    elif len(raw_session_ids) > _tools._LCM_EXPAND_QUERY_SESSION_ID_CAP:
        return json.dumps({"error": "session_ids accepts at most 20 entries"})
    else:
        session_ids = []
        for raw_session_id in raw_session_ids:
            if not isinstance(raw_session_id, str) or not raw_session_id.strip():
                return json.dumps(
                    {"error": "session_ids must contain only non-empty strings"}
                )
            session_id = raw_session_id.strip()
            if session_id not in session_ids:
                session_ids.append(session_id)

    # `session_id` is reused as a loop variable below, so provenance reads the
    # active session from the engine: an explicit cross-session retrieval must
    # not be described as the last session the search happened to visit.
    active_session_id = engine.current_session_id
    evidence_scope_kind = (
        "explicit_sessions" if session_ids_explicit else "current_session"
    )

    def _parse_int_arg(name: str, default: int) -> tuple[int | None, str | None]:
        raw_value = args.get(name, default)
        try:
            return int(raw_value), None
        except (TypeError, ValueError):
            return None, f"{name} must be an integer"

    max_tokens, max_tokens_error = _parse_int_arg("max_tokens", 2000)
    if max_tokens_error:
        return json.dumps({"error": max_tokens_error})
    max_tokens = max(1, max_tokens)  # type: ignore[type-var]
    context_default = max(  # type: ignore[type-var]
        max_tokens,
        int(getattr(engine._config, "expansion_context_tokens", 32_000) or 32_000),
    )
    context_max_tokens, context_max_tokens_error = _parse_int_arg(
        "context_max_tokens", context_default  # type: ignore[arg-type]
    )
    if context_max_tokens_error:
        return json.dumps({"error": context_max_tokens_error})
    context_max_tokens = max(1, context_max_tokens)  # type: ignore[type-var]

    max_results, max_results_error = _parse_int_arg("max_results", 5)
    if max_results_error:
        return json.dumps({"error": max_results_error})
    max_results = max(1, int(max_results or 5))

    query = _tools._coerce_query_arg(args.get("query"))
    raw_node_ids = args.get("node_ids") or []

    nodes = []
    raw_results: list[dict[str, Any]] = []
    if raw_node_ids:
        allowed_session_ids = set(session_ids)
        for node_id in raw_node_ids:
            try:
                parsed_node_id = int(node_id)
            except (TypeError, ValueError):
                return json.dumps({"error": "node_ids must contain only integers"})
            node = engine._dag.get_node(parsed_node_id)
            if node is not None and node.session_id in allowed_session_ids:
                nodes.append(node)
    elif query:
        for session_id in session_ids:
            nodes.extend(
                engine._dag.search(query, session_id=session_id, limit=max_results)
            )
            raw_results.extend(
                engine._store.search(query, session_id=session_id, limit=max_results)
            )
        if len(session_ids) > 1:
            nodes.sort(
                key=lambda node: (
                    float(node.search_rank)
                    if node.search_rank is not None
                    else float("inf"),
                    -float(node.search_directness or 0),
                    -float(node.latest_at or node.created_at or 0),
                )
            )
            raw_results.sort(
                key=lambda row: _tools._combined_result_sort_key(
                    {
                        "type": "message",
                        "role": row.get("role"),
                        "_sort_rank": row.get("search_rank"),
                        "_sort_directness": row.get("_directness_score"),
                        "_sort_ts": row.get("timestamp"),
                    },
                    "relevance",
                )
            )
        nodes = nodes[:max_results]
        raw_results = raw_results[:max_results]
    else:
        return json.dumps({"error": "Provide either query or node_ids"})

    if not nodes and not raw_results:
        if output == "evidence":
            payload: dict[str, Any] = {
                "prompt": prompt,
                "query": query,
                "output": "evidence",
                "evidence": [],
                "context_max_tokens": context_max_tokens,
                "context_tokens": 0,
                "context_truncated": False,
                "context_pagination": [],
                "node_ids": [],
                "matches": [],
                "raw_matches": [],
            }
            if session_ids_explicit:
                payload["session_ids"] = session_ids
            return json.dumps(payload)
        return json.dumps(
            {
                "prompt": prompt,
                "query": query,
                "answer": (
                    "No matching summaries or raw messages found in the requested sessions."
                    if session_ids_explicit
                    else "No matching summaries or raw messages found in the current session."
                ),
                "node_ids": [],
                "matches": [],
                "raw_matches": [],
                "evidence_provenance": expansion_evidence._build_expand_query_evidence(
                    [],
                    session_id=active_session_id,
                    session_ids=session_ids,
                    scope_kind=evidence_scope_kind,
                    context_truncated=False,
                    synthesis_status="not_run",
                ),
            }
        )

    context_blocks: list[dict[str, Any]] = []
    context_budget_used = 0
    context_budget_truncated = False
    for node in nodes[:max_results]:
        remaining_context_tokens = max(0, context_max_tokens - context_budget_used)  # type: ignore[operator]
        if output == "evidence" and remaining_context_tokens <= 0:
            context_budget_truncated = True
            break
        node_blocks = _collect_context_blocks_for_node(
            engine,
            node,
            max_tokens=remaining_context_tokens,
            hydrate_externalized_content=(
                not session_ids_explicit or node.session_id == engine.current_session_id
            ),
            session_id=node.session_id,
        )
        if session_ids_explicit:
            for block in node_blocks:
                block["session_id"] = node.session_id
        if output == "evidence":
            for block in node_blocks:
                candidate_blocks = [*context_blocks, block]
                candidate_tokens = expansion_evidence._serialized_context_token_count(candidate_blocks)
                if candidate_tokens > context_max_tokens:  # type: ignore[operator]
                    context_budget_truncated = True
                    break
                context_blocks.append(block)
                context_budget_used = candidate_tokens
            if context_budget_truncated:
                break
        else:
            context_blocks.extend(node_blocks)
            context_budget_used += expansion_evidence._context_content_token_count(node_blocks)

    raw_matches: list[dict[str, Any]] = []
    # Summaries can consume the whole budget while distinct raw messages still
    # matched. Those hits are then omitted from the context, but the caller
    # still needs a cursor into them, so the raw block is always collected — at
    # a zero budget it carries no content and only the ``next_store_id`` handle.
    raw_cursor_block: dict[str, Any] | None = None
    if raw_results:
        seen_store_ids = expansion_evidence._collect_store_ids_from_context_blocks(context_blocks)
        remaining_context_tokens = max(0, context_max_tokens - context_budget_used)  # type: ignore[operator]
        raw_block, raw_matches = _collect_raw_match_context_block(
            engine,
            raw_results,
            max_tokens=remaining_context_tokens,
            query=query,
            exclude_store_ids=seen_store_ids,
        )
        if raw_block is not None:
            if output == "evidence":
                candidate_blocks = [*context_blocks, raw_block]
                candidate_tokens = expansion_evidence._serialized_context_token_count(candidate_blocks)
                if candidate_tokens <= context_max_tokens:  # type: ignore[operator]
                    context_blocks.append(raw_block)
                    context_budget_used = candidate_tokens
                else:
                    context_budget_truncated = True
                    raw_cursor_block = expansion_evidence._raw_match_cursor_block(raw_block)
            else:
                context_blocks.append(raw_block)
                context_budget_used += expansion_evidence._context_content_token_count([raw_block])

    context_pagination = expansion_evidence.continuations(context_blocks, raw_cursor_block)

    context_truncated = context_budget_truncated or any(
        bool(item.get("summary_truncated"))
        or bool(item.get("pagination", {}).get("has_more"))
        for item in context_pagination
    )

    selected_nodes = nodes[:max_results]
    matches = [
        {
            "node_id": node.node_id,
            "depth": node.depth,
            "summary": node.summary[:300],
            "expand_hint": node.expand_hint,
        }
        for node in selected_nodes
    ]
    if session_ids_explicit:
        for match, node in zip(matches, selected_nodes):
            match["session_id"] = node.session_id
            match["expand_hint"] = _tools._session_expand_hint(node.node_id, node.session_id)
    node_ids = [node.node_id for node in selected_nodes]

    if output == "evidence":
        payload = {
            "prompt": prompt,
            "query": query,
            "output": "evidence",
            "evidence": context_blocks,
            "context_max_tokens": context_max_tokens,
            "context_tokens": expansion_evidence._serialized_context_token_count(context_blocks),
            "context_truncated": context_truncated,
            "context_pagination": context_pagination,
            "node_ids": node_ids,
            "matches": matches,
            "raw_matches": raw_matches,
        }
        if session_ids_explicit:
            payload["session_ids"] = session_ids
        return json.dumps(payload)

    def _degraded_payload(reason: str, *, include_timeout: bool = False) -> str:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "query": query,
            "error": reason,
            "degraded": True,
            "model": model,
            "max_tokens": max_tokens,
            "context_max_tokens": context_max_tokens,
            "context_truncated": context_truncated,
            "context_pagination": context_pagination,
            "node_ids": node_ids,
            "matches": matches,
            "raw_matches": raw_matches,
            "evidence_provenance": expansion_evidence._build_expand_query_evidence(
                context_blocks,
                session_id=active_session_id,
                session_ids=session_ids,
                scope_kind=evidence_scope_kind,
                context_truncated=context_truncated,
                synthesis_status="failed",
            ),
        }
        if include_timeout:
            payload["timeout_seconds"] = timeout
        return json.dumps(payload)

    model = engine._config.expansion_model or engine._config.summary_model or ""
    timeout = engine._config.expansion_timeout_ms / 1000
    try:
        synthesize_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "context_blocks": context_blocks,
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if engine._config.expansion_reasoning_effort:
            synthesize_kwargs["reasoning_effort"] = engine._config.expansion_reasoning_effort
        answer = _synthesize_expansion_answer(**synthesize_kwargs)
    except TimeoutError:
        logger.warning("LCM expand_query synthesis timed out after %.3fs", timeout)
        return _degraded_payload(
            f"lcm_expand_query synthesis timed out after {timeout:.3g}s",
            include_timeout=True,
        )
    except _ExpansionSynthesisError as exc:
        # Unusable aux-LLM response. Degrade the same way a timeout does: the
        # caller still receives the node/raw matches it asked for, only the
        # synthesized prose answer is unavailable.
        logger.warning("LCM expand_query synthesis failed: %s", exc)
        return _degraded_payload(f"lcm_expand_query synthesis unavailable: {exc}")

    answer = str(answer).strip() if answer is not None else ""
    if not answer:
        logger.warning("LCM expand_query synthesis returned an empty answer")
        return _degraded_payload("lcm_expand_query synthesis returned an empty answer")

    return json.dumps(
        {
            "prompt": prompt,
            "query": query,
            "answer": answer,
            "model": model,
            "max_tokens": max_tokens,
            "context_max_tokens": context_max_tokens,
            "context_truncated": context_truncated,
            "context_pagination": context_pagination,
            "node_ids": node_ids,
            "matches": matches,
            "raw_matches": raw_matches,
            "evidence_provenance": expansion_evidence._build_expand_query_evidence(
                context_blocks,
                session_id=active_session_id,
                session_ids=session_ids,
                scope_kind=evidence_scope_kind,
                context_truncated=context_truncated,
                synthesis_status="completed",
            ),
        }
    )
