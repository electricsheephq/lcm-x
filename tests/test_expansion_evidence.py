"""Pin expansion evidence and continuation payloads before their extraction."""

from __future__ import annotations

import importlib

from hermes_lcm import expand_query_tools
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

lcm_tools = importlib.import_module("hermes_lcm.tools")
expansion_evidence = importlib.import_module("hermes_lcm.expansion_evidence")


def test_expansion_continuation_shapes(monkeypatch: Any) -> None:
    """The public handler preserves cursor order, offsets and session ownership."""
    page = {"has_more": True, "next_source_offset": 3, "next_content_offset": 11}
    blocks: list[dict[str, Any]] = [
        {"type": "summary", "node_id": 1, "summary_truncated": True},
        {"type": "summary", "node_id": 2, "summary_truncated": False},
    ]
    expected: list[dict[str, Any]] = [
        {"node_id": 1, "type": "summary", "summary_truncated": True,
         "session_id": "archive", "expand_args": {"node_id": 1, "session_id": "archive"}},
    ]
    for kind in ("child_nodes", "descendant_child_nodes"):
        blocks.append({"type": kind, "node_id": 3, "children": [
            {"node_id": 4, "source_index": 2, "summary_truncated": True},
            {"node_id": 5, "summary_truncated": False},
        ], "pagination": page})
        expected.extend([
            {"node_id": 3,
             "type": "child_summary" if kind == "child_nodes" else "descendant_child_summary",
             "child_node_id": 4, "source_index": 2, "summary_truncated": True,
             "session_id": "archive", "expand_args": {"node_id": 4, "session_id": "archive"}},
            {"node_id": 3, "type": kind, "pagination": page, "session_id": "archive",
             "expand_args": {"node_id": 3, "source_offset": 3, "session_id": "archive"}},
        ])
    for kind in ("messages", "child_messages"):
        for externalized in (False, True):
            message = {"store_id": 8, "source_index": 2, "content_truncated": True,
                       "content_source": "externalized_payload" if externalized else "message",
                       "externalized": {"ref": "payload-8", "tool_call_id": "call-8"}}
            blocks.append({"type": kind, "node_id": 6, "messages": [message], "pagination": page})
            expected.append({
                "node_id": 6, "type": kind, "pagination": page, "session_id": "archive",
                "source_index": 2, "content_source": message["content_source"],
                "externalized_ref": "payload-8", "tool_call_id": "call-8",
                "expand_args": {"externalized_ref": "payload-8", "content_offset": 11}
                if externalized else {"node_id": 6, "source_offset": 3,
                                      "content_offset": 11, "session_id": "archive"},
            })
        blocks.append({"type": kind, "node_id": 7, "messages": [], "pagination": {"has_more": True}})
        expected.append({"node_id": 7, "type": kind, "pagination": {"has_more": True},
                         "session_id": "archive", "expand_args": {"node_id": 7,
                         "source_offset": 0, "content_offset": 0, "session_id": "archive"}})
    blocks.extend([
        {"type": "raw_messages", "messages": [{"store_id": 8, "content_truncated": True,
         "content_source": "message", "next_content_offset": 13}], "pagination": page},
        {"type": "raw_messages", "messages": [], "pagination": {"has_more": True, "next_store_id": 9}},
        {"type": "raw_messages", "messages": [], "pagination": {"has_more": False}},
    ])
    expected.extend([
        {"node_id": None, "type": "raw_messages", "pagination": page, "session_id": "archive",
         "store_id": 8, "content_source": "message", "expand_args": {"store_id": 8, "content_offset": 13}},
        {"node_id": None, "type": "raw_messages", "pagination": {"has_more": True, "next_store_id": 9},
         "session_id": "archive", "store_id": 9, "expand_args": {"store_id": 9}},
    ])
    node = SimpleNamespace(node_id=1, session_id="archive", depth=0, summary="summary", expand_hint="hint")
    engine = SimpleNamespace(current_session_id="active", _config=SimpleNamespace(),
                             _dag=SimpleNamespace(get_node=lambda _: node))
    monkeypatch.setattr(expand_query_tools, "_collect_context_blocks_for_node", lambda *a, **kw: deepcopy(blocks))
    result = json.loads(lcm_tools.lcm_expand_query(
        {"prompt": "details?", "node_ids": [1], "session_ids": ["archive"],
         "output": "evidence", "context_max_tokens": 32000}, engine=engine,
    ))
    assert result["context_pagination"] == expected
    assert result["context_truncated"] is True
    assert all("session_id" not in block for block in blocks)


def test_expansion_omitted_raw_cursor(monkeypatch: Any) -> None:
    """Even when the raw block cannot fit, the caller receives its next handle."""
    node = SimpleNamespace(node_id=1, session_id="active", depth=0, summary="summary", expand_hint="hint")
    engine = SimpleNamespace(
        current_session_id="active", _config=SimpleNamespace(),
        _dag=SimpleNamespace(search=lambda *a, **kw: [node]),
        _store=SimpleNamespace(search=lambda *a, **kw: [{"store_id": 9}]),
    )
    monkeypatch.setattr(expand_query_tools, "_collect_context_blocks_for_node", lambda *a, **kw: [])
    monkeypatch.setattr(expand_query_tools, "_collect_raw_match_context_block", lambda *a, **kw: (
        {"type": "raw_messages", "messages": [],
         "pagination": {"has_more": True, "next_store_id": 9}}, [],
    ))
    result = json.loads(lcm_tools.lcm_expand_query(
        {"prompt": "details?", "query": "details", "output": "evidence", "context_max_tokens": 1},
        engine=engine,
    ))
    assert result["evidence"] == []
    assert result["context_truncated"] is True
    assert result["context_pagination"] == [{
        "node_id": None, "type": "raw_messages",
        "pagination": {"has_more": True, "returned_sources": 0, "total_sources": 0, "next_store_id": 9},
        "store_id": 9, "expand_args": {"store_id": 9},
    }]


def test_expansion_empty_provenance_bytes() -> None:
    """Pin the serialized envelope and unknown-value normalization."""
    evidence = expansion_evidence._build_expand_query_evidence(
        [], session_id="active", session_ids=["archive"], scope_kind="unexpected",
        context_truncated=True, synthesis_status="unexpected",
    )
    assert json.dumps(evidence) == (
        '{"retrieval_scope": {"kind": "unknown", "session_ids": ["archive"]}, '
        '"identifiers_are_authority": false, "locator_replay_safety": "not_guaranteed", '
        '"synthesis_status": "unknown", "semantic_entailment": "not_applicable", '
        '"quote_origin": "tool_extracted_from_synthesis_context", "context_truncated": true, '
        '"context_unique_item_count": 0, "context_occurrence_count": 0, "serialized_char_limit": 10000, '
        '"serialization": {"scope": "evidence_provenance_only", "json_ensure_ascii": true}, '
        '"quote_max_chars": 500, "items": [], "locator_coverage": "none", '
        '"items_truncated": false, "quotes_truncated_by_provenance_cap": 0, "metadata_truncated": false}'
    )
