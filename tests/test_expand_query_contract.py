"""Pin expansion response bytes, cursor accounting and late-bound dependencies."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_lcm import expansion_evidence, tools, tokens
from tests.harness import pin_char_counter


@pytest.fixture
def inventory(monkeypatch: pytest.MonkeyPatch) -> Any:
    pin_char_counter(monkeypatch)  # type: ignore[no-untyped-call]
    rows = {i: dict(store_id=i, session_id="session", source="cli", role="user",
                    content=f"café evidence {i} " * 8, timestamp=100, search_rank=-1.0)
            for i in (1, 2, 3)}
    nodes = {i: SimpleNamespace(node_id=i, session_id="session", depth=0,
                               summary=f"café summary {i}", token_count=4,
                               source_token_count=32, source_ids=[i], source_type="messages",
                               expand_hint=f"lcm_expand(node_id={i})") for i in (1, 2)}
    nodes[3] = SimpleNamespace(node_id=3, session_id="session", depth=1,
                              summary="café parent", token_count=3, source_token_count=64,
                              source_ids=[1, 2], source_type="nodes",
                              expand_hint="lcm_expand(node_id=3)")
    return SimpleNamespace(
        current_session_id="session",
        _config=SimpleNamespace(expansion_context_tokens=32000, expansion_model="fixture",
                                summary_model="", expansion_timeout_ms=2500,
                                expansion_reasoning_effort=""),
        _dag=SimpleNamespace(get_node=nodes.get, search=lambda *a, **k: [nodes[3]]),
        _store=SimpleNamespace(get_batch=lambda ids: {i: rows[i] for i in ids},
                               search=lambda *a, **k: [rows[3]], get=rows.get),
    )


def response_bytes(engine: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setattr(tools, "_synthesize_expansion_answer", lambda **kw: "  café answer  ")
    args = {"prompt": "What evidence?", "node_ids": [3], "context_max_tokens": 1000}
    result = {
        "missing": tools.lcm_expand_query({}),
        "success": tools.lcm_expand_query(args, engine=engine),
        "truncated": tools.lcm_expand_query(dict(args, context_max_tokens=2), engine=engine),
        "raw_cursor": tools.lcm_expand_query(
            {"prompt": "What evidence?", "query": "café", "output": "evidence",
             "context_max_tokens": 1}, engine=engine),
        "children": tools.lcm_expand({"node_id": 3, "source_limit": 1}, engine=engine),
        "child_page": tools.lcm_expand({"node_id": 3, "source_offset": 1}, engine=engine),
        "message_page": tools.lcm_expand({"node_id": 1, "max_tokens": 4}, engine=engine),
    }
    first = json.loads(result["message_page"])["pagination"]
    result["message_continue"] = tools.lcm_expand(
        {"node_id": 1, "max_tokens": 4, "source_offset": first["next_source_offset"],
         "content_offset": first["next_content_offset"]}, engine=engine)
    return result


def test_expand_query_response_bytes(inventory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = json.loads((Path(__file__).parent / "fixtures" / "expand_query_contract.json").read_text())
    actual = response_bytes(inventory, monkeypatch)
    assert actual == expected
    assert json.loads(actual["success"])["evidence_provenance"]["synthesis_status"] == "completed"
    assert json.loads(actual["raw_cursor"])["context_pagination"]
    assert tools.lcm_expand_query.__module__ == "hermes_lcm.tools"


def test_expand_query_timeout_logger(
    inventory: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def timeout(**kwargs: Any) -> str:
        raise TimeoutError("fixture")

    monkeypatch.setattr(tools, "_synthesize_expansion_answer", timeout)
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.tools"):
        result = json.loads(tools.lcm_expand_query(
            {"prompt": "What evidence?", "node_ids": [1]}, engine=inventory))
    assert result["degraded"] is True
    assert result["evidence_provenance"]["synthesis_status"] == "failed"
    assert [(r.name, r.levelname, r.message) for r in caplog.records] == [
        ("hermes_lcm.tools", "WARNING", "LCM expand_query synthesis timed out after 2.500s"),
    ]


def test_expansion_leaf_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    cursor = expansion_evidence._raw_match_cursor_block({
        "messages": [{"store_id": "skip"}, {"store_id": 7}],
        "pagination": {"next_store_id": 9, "total_sources": 3},
    })
    assert cursor == {"type": "raw_messages", "messages": [], "pagination": {
        "has_more": True, "returned_sources": 0, "total_sources": 3, "next_store_id": 7,
    }}
    assert expansion_evidence._raw_match_cursor_block({"pagination": {"next_store_id": "bad"}}) is None
    assert expansion_evidence._collect_store_ids_from_context_blocks(
        [None, {"messages": [{"store_id": 7}, {"store_id": "skip"}]}],  # type: ignore[list-item]
    ) == {7}
    path = [{"node_id": i, "source_index": i} for i in range(10)]
    assert expansion_evidence._bounded_source_path_payload(path) == {
        "source_path": path[-8:], "source_path_depth": 10, "source_path_truncated": True,
    }
    assert expansion_evidence._pagination_payload(
        total_sources=9, source_offset=1, content_offset=0, source_limit=2,
        returned_sources=2, next_source_offset=3, next_content_offset=0, has_more=True,
    ) == dict(source_offset=1, content_offset=0, source_limit=2, returned_sources=2,
              total_sources=9, next_source_offset=3, next_content_offset=0,
              has_more=True, remaining_sources=6)
    seen: list[str] = []

    def count(text: str) -> int:
        seen.append(text)
        return len(text)

    monkeypatch.setattr(tokens, "count_tokens", count)
    blocks: list[dict[str, Any]] = [{"type": "summary", "summary": "sum", "source_path": path[:1], "source_path_depth": 1}]
    blocks.extend({"type": kind, "messages": [{"content": "body", "transcript_content": "original"}]}
                  for kind in ("messages", "child_messages", "raw_messages"))
    blocks.extend({"type": kind, "children": [{"summary": "child"}]}
                  for kind in ("child_nodes", "descendant_child_nodes"))
    expected = ["sum", '{"source_path":[{"node_id":0,"source_index":0}],"source_path_depth":1,"source_path_truncated":false}',
                "body", "original", "body", "original", "body", "original", "child", "child"]
    assert expansion_evidence._context_content_token_count(blocks) == sum(map(len, expected))
    assert seen == expected


def test_expansion_evidence_keeps_tokens_import_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import sys
    from types import ModuleType

    package = ModuleType("expansion_import_probe")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(
        "expansion_import_probe.expansion_evidence", expansion_evidence.__file__,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "expansion_import_probe.tokens" not in sys.modules
    fake_tokens = ModuleType("expansion_import_probe.tokens")
    monkeypatch.setattr(fake_tokens, "count_tokens", lambda text: 17, raising=False)
    monkeypatch.setitem(sys.modules, fake_tokens.__name__, fake_tokens)
    assert module._context_content_token_count([{"type": "summary", "summary": "marker"}]) == 17
