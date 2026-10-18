"""Describe and expand handlers behind the public tools facade."""
from __future__ import annotations

from typing import Any, Dict

from . import tools as _tools


def lcm_describe(args: Dict[str, Any], **kwargs: Any) -> str:
    """Inspect a summary node's subtree or get session DAG overview."""
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return _tools.json.dumps({"error": "LCM engine not initialized"})  # type: ignore[attr-defined, no-any-return]

    externalized_ref = str(args.get("externalized_ref") or "").strip()
    if externalized_ref:
        # Externalized payloads are current-session objects; an explicit
        # session_id alongside the ref would be silently ignored below and the
        # response could misattribute current-session content to the requested
        # archive session. Reject the combination before dereferencing, the
        # same way lcm_expand validates its mode arguments.
        if "session_id" in args:
            return _tools.json.dumps(  # type: ignore[attr-defined, no-any-return]
                {
                    "error": "externalized_ref cannot be combined with "
                    "session_id; externalized payloads are current-session "
                    "objects"
                }
            )
        payload = _tools._get_externalized_payload(engine, externalized_ref)
        if payload is None:
            return _tools.json.dumps(  # type: ignore[attr-defined, no-any-return]
                {
                    "error": f"Externalized payload {externalized_ref} not found in current session"
                }
            )
        return _tools.json.dumps(  # type: ignore[attr-defined, no-any-return]
            {
                "externalized_ref": externalized_ref,
                "kind": payload.get("kind", "tool_result"),
                "tool_call_id": payload.get("tool_call_id", ""),
                "role": payload.get("role", ""),
                "session_id": payload.get("session_id", ""),
                "field_path": payload.get("field_path", ""),
                "content_chars": payload.get("content_chars", 0),
                "content_bytes": payload.get("content_bytes", 0),
                "created_at": payload.get("created_at"),
                "content_preview": (payload.get("content") or "")[:500],
            }
        )

    node_id = args.get("node_id")
    session_id_explicit = "session_id" in args
    session_id_arg = args.get("session_id")
    if session_id_explicit:
        if not isinstance(session_id_arg, str) or not session_id_arg.strip():
            return _tools.json.dumps({"error": "session_id must be a non-empty string"})  # type: ignore[attr-defined, no-any-return]
        session_id = session_id_arg.strip()
    else:
        session_id = engine.current_session_id

    if node_id is not None:
        node = _tools._get_session_node(engine, node_id, session_id=session_id)
        if node is None:
            scope = (
                f"session {session_id}" if session_id_explicit else "current session"
            )
            return _tools.json.dumps({"error": f"Node {node_id} not found in {scope}"})  # type: ignore[attr-defined, no-any-return]
        info = engine._dag.describe_subtree(node_id)
        provenance = engine._dag.get_node_provenance(node.node_id)  # #441
        if provenance is not None:
            info.update(provenance)
        if session_id_explicit:
            info["session_id"] = node.session_id
            info["expand_hint"] = _tools._session_expand_hint(node.node_id, node.session_id)
            for child in info.get("children", []):
                child["expand_hint"] = _tools._session_expand_hint(
                    child["node_id"], node.session_id
                )
        return _tools.json.dumps(info)  # type: ignore[attr-defined, no-any-return]

    depth_stats = engine._dag.get_session_depth_stats(session_id)
    depth_samples = engine._dag.get_session_depth_samples(
        session_id,
        per_depth_limit=20,
        depths=list(depth_stats),
    )
    overview: Dict[str, Any] = {
        "session_id": session_id,
        "store_message_count": engine._store.get_session_count(session_id),
        "depths": {},
    }

    for depth, stats in sorted(depth_stats.items()):
        nodes = depth_samples.get(depth, [])
        overview["depths"][f"d{depth}"] = {
            "count": stats["count"],
            "total_tokens": stats["tokens"],
            "total_source_tokens": stats["source_tokens"],
            "nodes": [
                {
                    "node_id": node.node_id,
                    "token_count": node.token_count,
                    "expand_hint": (
                        _tools._session_expand_hint(node.node_id, node.session_id)
                        if session_id_explicit
                        else node.expand_hint
                    ),
                }
                for node in nodes
            ],
        }

    return _tools.json.dumps(overview)  # type: ignore[attr-defined, no-any-return]


def lcm_expand(args: Dict[str, Any], **kwargs: Any) -> str:
    """Expand a summary node, externalized payload, or raw message to its content.

    Mode selection (exactly one is required):
    - ``externalized_ref``: open a stored externalized payload by ref filename (current session only)
    - ``store_id``: fetch a single raw message by store_id; works across sessions
    - ``node_id``: expand a summary node to its source content (explicit ``session_id`` required cross-session)

    Omitting ``session_id`` preserves current-session node lookup. Carried-over
    current-session nodes may reference raw source rows from the previous session.
    """
    engine = _tools._require_engine(kwargs)
    if engine is None:
        return _tools.json.dumps({"error": "LCM engine not initialized"})  # type: ignore[attr-defined, no-any-return]

    externalized_ref = str(args.get("externalized_ref") or "").strip()
    raw_store_id_arg = args.get("store_id")
    raw_node_id_arg = args.get("node_id")

    modes_provided: list[str] = []
    if externalized_ref:
        modes_provided.append("externalized_ref")
    if raw_store_id_arg is not None:
        modes_provided.append("store_id")
    if raw_node_id_arg is not None:
        modes_provided.append("node_id")

    if len(modes_provided) > 1:
        return _tools.json.dumps(  # type: ignore[attr-defined, no-any-return]
            {
                "error": (
                    "Provide only one of node_id, externalized_ref, store_id "
                    f"(got {', '.join(modes_provided)})"
                ),
            }
        )
    if not modes_provided:
        return _tools.json.dumps(  # type: ignore[attr-defined, no-any-return]
            {
                "error": "node_id, externalized_ref, or store_id is required",
            }
        )

    session_id_explicit = "session_id" in args
    session_id_arg = args.get("session_id")
    if session_id_explicit and "node_id" not in modes_provided:
        return _tools.json.dumps({"error": "session_id is only valid with node_id"})  # type: ignore[attr-defined, no-any-return]
    if session_id_explicit:
        if not isinstance(session_id_arg, str) or not session_id_arg.strip():
            return _tools.json.dumps({"error": "session_id must be a non-empty string"})  # type: ignore[attr-defined, no-any-return]
        session_id = session_id_arg.strip()
    else:
        session_id = engine.current_session_id

    max_tokens = _tools._parse_positive_int(args.get("max_tokens", 4000), 4000)
    source_offset = _tools._parse_non_negative_int(args.get("source_offset", 0), 0)
    source_limit_arg = args.get("source_limit")
    source_limit = (
        _tools._parse_positive_int(source_limit_arg, 0)
        if source_limit_arg is not None
        else None
    )
    content_offset = _tools._parse_non_negative_int(args.get("content_offset", 0), 0)
    raw_include_exact_ref = args.get("include_exact_ref", False)
    if not isinstance(raw_include_exact_ref, bool):
        return _tools.json.dumps({"error": "include_exact_ref must be a boolean"})  # type: ignore[attr-defined, no-any-return]
    include_exact_ref = raw_include_exact_ref
    if include_exact_ref and raw_store_id_arg is None:
        return _tools.json.dumps({"error": "include_exact_ref is supported only with store_id mode"})  # type: ignore[attr-defined, no-any-return]

    if externalized_ref:
        payload = _tools._get_externalized_payload(engine, externalized_ref)
        if payload is None:
            return _tools.json.dumps(  # type: ignore[attr-defined, no-any-return]
                {
                    "error": f"Externalized payload {externalized_ref} not found in current session"
                }
            )
        content = payload.get("content", "")
        sliced = _tools._slice_content_for_response(content, max_tokens, content_offset)
        return _tools.json.dumps(  # type: ignore[attr-defined, no-any-return]
            {
                "externalized_ref": externalized_ref,
                "source_type": "externalized_payload",
                "kind": payload.get("kind", "tool_result"),
                "tool_call_id": payload.get("tool_call_id", ""),
                "role": payload.get("role", ""),
                "session_id": payload.get("session_id", ""),
                "field_path": payload.get("field_path", ""),
                "content_chars": payload.get("content_chars", len(content)),
                "content_bytes": payload.get("content_bytes", 0),
                "content": sliced["content"],
                "content_offset": sliced["content_offset"],
                "content_returned_chars": sliced["content_returned_chars"],
                "content_truncated": sliced["content_truncated"],
                "next_content_offset": sliced["next_content_offset"],
                "has_more": sliced["has_more"],
            }
        )

    if raw_store_id_arg is not None:
        try:
            store_id = int(raw_store_id_arg)
        except (TypeError, ValueError, OverflowError):
            return _tools.json.dumps({"error": "store_id must be an integer"})  # type: ignore[attr-defined, no-any-return]
        stored = engine._store.get(store_id)
        if stored is None:
            return _tools.json.dumps({"error": f"Message store_id {store_id} not found"})  # type: ignore[attr-defined, no-any-return]
        transcript_content = stored.get("content", "") or ""
        sliced = _tools._slice_content_for_response(
            transcript_content, max_tokens, content_offset
        )
        engine_session_id = engine.current_session_id
        stored_session_id = stored.get("session_id", "")
        result: Dict[str, Any] = {
            "store_id": store_id,
            "source_type": "raw_message",
            "session_id": stored_session_id,
            "source": stored.get("source") or "",
            "conversation_id": stored.get("conversation_id") or "",
            "role": stored.get("role"),
            "timestamp": stored.get("timestamp", 0),
            "tool_call_id": stored.get("tool_call_id") or "",
            "from_current_session": bool(engine_session_id)
            and stored_session_id == engine_session_id,
            "content": sliced["content"],
            "content_chars": sliced["content_chars"],
            "content_offset": sliced["content_offset"],
            "content_returned_chars": sliced["content_returned_chars"],
            "content_truncated": sliced["content_truncated"],
            "next_content_offset": sliced["next_content_offset"],
            "has_more": sliced["has_more"],
        }
        if include_exact_ref and sliced["content_returned_chars"] > 0:
            exact_start = sliced["content_offset"]
            exact_end = exact_start + sliced["content_returned_chars"]
            result["exact_ref"] = f"lcm:{store_id}:{exact_start}-{exact_end}"
        # Surface externalized-payload metadata when the row references one. Content
        # is not hydrated by default, mirroring the existing _expand_message_sources
        # default. Externalized lookup remains session-scoped (per the existing
        # _get_externalized_payload contract); cross-session rows surface only the
        # ref string, with a hint pointing at the same-session expansion path.
        ref_values = [transcript_content]
        if stored.get("tool_calls"):
            try:
                ref_values.append(
                    _tools.json.dumps(  # type: ignore[attr-defined]
                        stored.get("tool_calls"), ensure_ascii=False, sort_keys=True
                    )
                )
            except (TypeError, ValueError):
                ref_values.append(str(stored.get("tool_calls")))
        refs: list[str] = []
        for value in ref_values:
            if not isinstance(value, str):
                continue
            for found_ref in _tools.extract_ingest_externalized_refs(value):  # type: ignore[attr-defined]
                if found_ref not in refs:
                    refs.append(found_ref)
            legacy_ref = _tools.extract_externalized_ref(value)  # type: ignore[attr-defined]
            if legacy_ref and legacy_ref not in refs:
                refs.append(legacy_ref)
        if refs:
            result["externalized_refs"] = refs
            result["externalized_ref"] = refs[0]
            if bool(engine_session_id) and stored_session_id == engine_session_id:
                payload_summaries = []
                for ref in refs:
                    payload = _tools._get_externalized_payload(engine, ref)
                    if payload is None:
                        continue
                    payload_summary = dict(payload)
                    payload_summary.pop("content", None)
                    payload_summaries.append(payload_summary)
                if payload_summaries:
                    result["externalized_payloads"] = payload_summaries
                    result["externalized"] = payload_summaries[0]
            else:
                result["externalized_note"] = (
                    "Externalized payload metadata is session-scoped; "
                    "cross-session ref is surfaced for traceability only and cannot be expanded in this version."
                )
        return _tools.json.dumps(result)  # type: ignore[attr-defined, no-any-return]

    assert raw_node_id_arg is not None
    node_id = raw_node_id_arg
    node = _tools._get_session_node(engine, node_id, session_id=session_id)
    if node is None:
        scope = f"session {session_id}" if session_id_explicit else "current session"
        return _tools.json.dumps({"error": f"Node {node_id} not found in {scope}"})  # type: ignore[attr-defined, no-any-return]

    if node.source_type == "messages":
        messages, pagination = _tools._expand_message_sources(
            engine,
            node,
            max_tokens=max_tokens,
            source_offset=source_offset,
            source_limit=source_limit,
            content_offset=content_offset,
        )
        result = {
            "node_id": node_id,
            "depth": node.depth,
            "source_type": "messages",
            "expanded": messages,
            "pagination": pagination,
        }
        if session_id_explicit:
            result["session_id"] = node.session_id
        return _tools.json.dumps(result)  # type: ignore[attr-defined, no-any-return]

    if node.source_type == "nodes":
        children, pagination = _tools._expand_child_nodes(
            engine,
            node,
            max_tokens=max_tokens,
            source_offset=source_offset,
            source_limit=source_limit,
            session_id=node.session_id,
        )
        result = {
            "node_id": node_id,
            "depth": node.depth,
            "source_type": "nodes",
            "expanded": children,
            "pagination": pagination,
        }
        if session_id_explicit:
            result["session_id"] = node.session_id
            for child in result["expanded"]:
                child["expand_hint"] = _tools._session_expand_hint(
                    child["node_id"], node.session_id
                )
        return _tools.json.dumps(result)  # type: ignore[attr-defined, no-any-return]

    return _tools.json.dumps({"error": f"Unknown source_type: {node.source_type}"})  # type: ignore[attr-defined, no-any-return]
