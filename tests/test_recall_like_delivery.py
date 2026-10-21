"""LIKE excerpts retain exact row locations beyond the hydration budget."""
import json

import pytest

from hermes_lcm import tools
from hermes_lcm.dag import SummaryNode
from hermes_lcm.search_query import build_snippet, build_snippet_with_span
from tests.test_recall_fts_delivery import check_spans, recall, seed
from tests.test_lcm_recall import recall_engine as recall_engine


@pytest.mark.parametrize("query", ["🦄", "東京", "evidence", "🦄" + "x" * 320],
                         ids=["emoji", "cjk", "fts-error", "long-query"])
def test_25_late_like_rows_survive_eight_hydrations(recall_engine, query):
    seed(recall_engine, [
        "padding " * 400 + f"row {i} {query} exact detail " + "tail " * 20
        for i in range(25)
    ])
    if query == "evidence":
        # Exercise the real SQLite-error fallback, not a mocked search arm.
        recall_engine._store.connection.execute("DROP TABLE messages_fts")
        recall_engine._store.connection.execute("DROP TABLE IF EXISTS messages_fts_stem")
    payload = check_spans(recall_engine, recall(recall_engine, query))
    assert len(payload["hits"]) == 25
    assert sum("content" in hit for hit in payload["hits"]) == 8
    for hit in payload["hits"]:
        assert query[:100] in hit.get("content", hit["snippet"])
        assert hit["content_offset"] > 0
        if "content" not in hit:
            assert f"content_offset={hit['content_offset']}" in hit["expand_hint"]
    policy = payload["provenance"]["answer_ready"]
    assert policy["unreferenced_dropped_count"] == 0
    assert policy["unreferenced_omitted_count"] == 0


def test_summary_like_retains_whole_summary_without_raw_reference(recall_engine):
    text = "padding " * 400 + "東京 summary detail"
    node_id = recall_engine._dag.add_node(SummaryNode(
        node_id=0, session_id="summary-only", depth=0, summary=text,
        token_count=900, source_type="messages", source_ids=[],
    ))
    hits = recall_engine._dag.search("東京", session_id="summary-only")
    assert [(node.node_id, node.summary) for node in hits] == [(node_id, text)]
    assert tools._lcm_recall_reference_shape({
        "kind": "summary", "node_id": node_id,
    }, hydratable=False) is None


def test_like_excerpt_is_reverified_against_current_row(recall_engine, monkeypatch):
    seed(recall_engine, ["padding " * 400 + "🦄 exact detail"])
    original = recall_engine._store.get_batch

    def rewritten(ids):
        return {key: {**row, "content": "changed row"}
                for key, row in original(ids).items()}

    monkeypatch.setattr(recall_engine._store, "get_batch", rewritten)
    monkeypatch.setattr(tools, "_LCM_RECALL_ANSWER_READY_EXPANDED_HIT_LIMIT", 0)
    payload = json.loads(recall(recall_engine, "🦄"))
    assert payload["hits"] == []


@pytest.mark.parametrize("text", ["", "🦄", "...🦄...", "prefix " * 400 + "🦄...",
                                  "x" * 100 + "🦄" + "x" * 100 + "🦄" + "x" * 100])
def test_like_span_tracks_undecorated_source(text):
    snippet, start, end = build_snippet_with_span(text, ["🦄"])
    assert snippet == build_snippet(text, ["🦄"])
    assert snippet == ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")
