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


def test_expansion_budget_exact_serialized_edge(monkeypatch: Any) -> None:
    """An exact fit stays intact; one character less drops the last item."""
    blocks = [{"type": "summary", "node_id": i, "summary": f"quote {i}"} for i in (1, 2)]
    kwargs = dict(session_id="active", context_truncated=False, synthesis_status="not_run")
    original = expansion_evidence._build_expand_query_evidence(blocks, **kwargs)
    # The limit has four digits here instead of the default's five digits.
    limit = len(json.dumps(original)) - 1
    assert 1000 <= limit < 10000
    monkeypatch.setattr(expansion_evidence, "_EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS", limit)
    exact = expansion_evidence._build_expand_query_evidence(blocks, **kwargs)
    expected = deepcopy(original)
    expected["serialized_char_limit"] = limit
    assert json.dumps(exact) == json.dumps(expected)
    assert len(json.dumps(exact)) == limit
    monkeypatch.setattr(expansion_evidence, "_EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS", limit - 1)
    trimmed = expansion_evidence._build_expand_query_evidence(blocks, **kwargs)
    expected["serialized_char_limit"] = limit - 1
    expected["items"] = expected["items"][:1]
    expected["items_truncated"] = True
    expected["locator_coverage"] = "partial"
    assert json.dumps(trimmed) == json.dumps(expected)
    assert trimmed["items"][0]["evidence_id"] == "node:1"


def test_expansion_budget_item_edge(monkeypatch: Any) -> None:
    """The item cap keeps the first 24 sources, including a zero-token quote."""
    monkeypatch.setattr(expansion_evidence, "_EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS", 100000)
    for count in (24, 25):
        blocks = [{"type": "summary", "node_id": i, "summary": "", "token_count": 0}
                  for i in range(count)]
        evidence = expansion_evidence._build_expand_query_evidence(
            blocks, session_id="active", context_truncated=False, synthesis_status="not_run",
        )
        assert [item["evidence_id"] for item in evidence["items"]] == [f"node:{i}" for i in range(24)]
        assert all(item["quote"] == "" and item["quote_chars_before_provenance_cap"] == 0
                   for item in evidence["items"])
        assert evidence["context_unique_item_count"] == count
        assert evidence["context_occurrence_count"] == count
        assert evidence["items_truncated"] is (count == 25)
        assert evidence["locator_coverage"] == ("complete" if count == 24 else "partial")


def test_expansion_budget_repeated_source_deep_paths() -> None:
    """Repeated deep-tree sources retain the first quote and bounded path order."""
    paths = [[{"node_id": 100 * occurrence + depth, "source_index": depth}
              for depth in range(12)] for occurrence in range(10)]
    blocks = [{"type": "summary", "node_id": 1, "summary": "first" if i == 0 else "later",
               "source_path": path, "source_path_depth": 12, "token_count": 0}
              for i, path in enumerate(paths)]
    before = deepcopy(blocks)
    evidence = expansion_evidence._build_expand_query_evidence(
        blocks, session_id="active", context_truncated=False, synthesis_status="not_run",
    )
    assert evidence["context_unique_item_count"] == 1
    assert evidence["context_occurrence_count"] == 10
    assert evidence["items_truncated"] is False
    assert evidence["locator_coverage"] == "complete"
    assert evidence["items"] == [{
        "source_type": "summary", "node_id": 1, "locator_present": True,
        "locator_replay_status": "unverified", "quote": "first",
        "quote_chars_before_provenance_cap": 5, "quote_truncated_by_provenance_cap": False,
        "session_id": "active", "expand_args": {"node_id": 1}, "evidence_id": "node:1",
        "context_occurrence_count": 10,
        "source_paths": [{"path": path[-8:], "depth": 12, "truncated": True} for path in paths[:8]],
        "source_paths_truncated": True,
    }]
    assert blocks == before


def test_expansion_budget_alias_and_read_order(monkeypatch: Any) -> None:
    """Each measurement sees the same envelope/list and refreshed prefix metadata."""
    blocks = [{"type": "summary", "node_id": i, "summary": "q" * 600} for i in (1, 2, 3)]
    envelopes: list[dict[str, Any]] = []
    lists: list[list[dict[str, Any]]] = []
    snapshots: list[tuple[list[str], bool, int]] = []
    first_items: list[dict[str, Any]] = []

    def measured(evidence: dict[str, Any]) -> int:
        envelopes.append(evidence)
        lists.append(evidence["items"])
        if not first_items:
            first_items.extend(evidence["items"])
        snapshots.append(([item["evidence_id"] for item in evidence["items"]],
                          evidence["items_truncated"], evidence["quotes_truncated_by_provenance_cap"]))
        return 10001 if len(evidence["items"]) > 1 else 10000

    monkeypatch.setattr(expansion_evidence, "_expand_query_evidence_serialized_chars", measured)
    result = expansion_evidence._build_expand_query_evidence(
        blocks, session_id="active", context_truncated=False, synthesis_status="completed",
    )
    assert snapshots == [(["node:1", "node:2", "node:3"], False, 3),
                         (["node:1", "node:2"], True, 2),
                         (["node:1"], True, 1), (["node:1"], True, 1)]
    assert all(envelope is result for envelope in envelopes)
    assert all(items is result["items"] for items in lists)
    assert result["items"][0] is first_items[0]
    assert result["items"][0]["quote"] == "q" * 500
    assert result["context_unique_item_count"] == 3
    assert result["locator_coverage"] == "partial"


def test_expansion_budget_omitted_cursor() -> None:
    """An omitted raw cursor stays separate from the zero-token evidence item."""
    blocks = [{"type": "summary", "node_id": 1, "summary": "", "token_count": 0}]
    omitted = {"type": "raw_messages", "messages": [],
               "pagination": {"has_more": True, "next_store_id": 9}}
    before = deepcopy((blocks, omitted))
    evidence = expansion_evidence._build_expand_query_evidence(
        blocks, session_id="active", context_truncated=True, synthesis_status="not_run",
    )
    assert evidence["items"][0]["quote"] == ""
    assert evidence["items"][0]["expand_args"] == {"node_id": 1}
    assert evidence["context_unique_item_count"] == 1
    assert evidence["context_truncated"] is True
    assert expansion_evidence.continuations(blocks, omitted) == [{
        "node_id": None, "type": "raw_messages", "pagination": omitted["pagination"],
        "store_id": 9, "expand_args": {"store_id": 9},
    }]
    assert (blocks, omitted) == before


def test_expansion_budget_fixed_envelope(monkeypatch: Any) -> None:
    """An oversized fixed envelope compacts only after the last item is removed."""
    import hashlib

    monkeypatch.setattr(expansion_evidence, "_EXPAND_QUERY_EVIDENCE_MAX_SERIALIZED_CHARS", 900)
    sessions = [f"session-{i:03}" for i in range(100)]
    evidence = expansion_evidence._build_expand_query_evidence(
        [{"type": "summary", "node_id": 1, "summary": ""}], session_id="active",
        session_ids=sessions, scope_kind="explicit_sessions", context_truncated=True,
        synthesis_status="completed",
    )
    assert evidence["retrieval_scope"] == {
        "kind": "explicit_sessions", "session_ids": [],
        "session_id_sha256": hashlib.sha256("\n".join(sessions).encode()).hexdigest(),
        "session_id_omitted_due_to_size": True,
    }
    assert evidence["fixed_envelope_compacted"] is True
    assert evidence["items"] == []
    assert evidence["items_truncated"] is True
    assert evidence["metadata_truncated"] is True
    assert evidence["locator_coverage"] == "partial"
    assert evidence["context_unique_item_count"] == evidence["context_occurrence_count"] == 1
    assert evidence["semantic_entailment"] == "not_verified"
