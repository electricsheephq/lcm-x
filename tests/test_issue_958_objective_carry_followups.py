"""#958: capped objective carry, sanitized-tail dedupe, and linear DAG scans."""

from unittest.mock import Mock

import pytest

import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import _PRESERVED_OBJECTIVE_CONTEXT_PREFIX as PREFIX
from hermes_lcm.tokens import count_message_tokens, count_messages_tokens


SEPARATOR = "\n\n---\n\n"
NEW = "NEW request: continue with the revised plan."


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    tokens._count_tokens_cached.cache_clear()
    instance = LCMEngine(config=LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        embeddings_enabled=False, temporal_rollups_enabled=False,
        empty_lifecycle_gc_enabled=False,
    ), hermes_home=str(tmp_path / "home"))
    instance.on_session_start("issue-958", context_length=128_000)
    try:
        yield instance
    finally:
        instance.shutdown()
        tokens._count_tokens_cached.cache_clear()


def _parts(engine, count=1):
    parts = []
    for index in range(count):
        summary = f"DAG summary {index}: " + "older work " * 80
        node_id = engine._dag.add_node(SummaryNode(
            session_id=engine._session_id, depth=0, summary=summary,
            token_count=tokens.count_tokens(summary), source_token_count=1000,
            source_ids=[], source_type="messages", expand_hint=f"details-{index}",
        ))
        parts.append(
            f"[Recent Summary (d0, node {node_id})]\n{summary}\n"
            f"[Expand for details: details-{index}]"
        )
    return parts


@pytest.mark.parametrize("keep_older", [True, False])
def test_merged_suffix_survives_cap_and_uncapped_bytes_stay_identical(engine, keep_older):
    part = _parts(engine)[0]
    older = "Older objective." if keep_older else "Older objective: " + "detail " * 1000
    objective = PREFIX + "\n" + older
    merged = {"role": "user", "content": objective + SEPARATOR + part + "\n\n" + NEW}
    tail = [{"role": "assistant", "content": "Continuing the tool work."}]
    engine._pending_context_anchor_messages = [merged, *tail]

    # The legacy caller (including the compaction guard) keeps the whole row.
    assert engine._latest_user_context_anchor([merged], []) == merged["content"]
    uncapped = engine._assemble_context(None, tail, include_lcm_note=False, persist=False)
    assert uncapped[0]["content"] == merged["content"] + SEPARATOR + part
    assert uncapped[-1] == tail[0]

    expected = objective + "\n\n" + NEW if keep_older else PREFIX + "\n" + NEW
    budget = count_message_tokens({"role": "user", "content": expected})
    assert count_message_tokens(merged) > budget
    if not keep_older:
        assert count_message_tokens({"role": "user", "content": objective + "\n\n" + NEW}) > budget
    capped = engine._assemble_context(
        None, tail, assembly_cap_override=budget + count_messages_tokens(tail),
        include_lcm_note=False, persist=False,
    )
    text = "\n".join(row.get("content", "") for row in capped)
    assert text.count(NEW) == 1
    assert capped[0]["content"] == expected
    assert part not in capped[0]["content"]
    assert capped[-1] == tail[0]


def test_sanitized_objective_in_capped_fresh_tail_is_emitted_once(engine):
    raw = {"role": "user", "content": (
        PREFIX + "\nKeep this objective.\n"
        "<relevant-memories>\nremovable injected context\n</relevant-memories>"
    )}
    tail = [raw, {"role": "assistant", "content": "Continuing."}]
    clean = engine._sanitize_active_context_messages(tail, merge_adjacent_assistants=False)
    assert clean[0] != raw
    cap = count_messages_tokens(clean) + count_message_tokens(clean[0]) + 100
    assembled = engine._assemble_context(
        None, tail, assembly_cap_override=cap, include_lcm_note=False, persist=False,
    )
    text = "\n".join(row.get("content", "") for row in assembled)
    assert text.count(PREFIX) == 1
    assert text.count("Keep this objective.") == 1
    assert "removable injected context" not in text


@pytest.mark.parametrize("suffix", ["", "\n\n" + NEW])
def test_objective_scan_looks_up_each_verified_part_once(engine, monkeypatch, suffix):
    parts = _parts(engine, count=16)
    objective = PREFIX + "\n" + SEPARATOR.join(["Markdown rule text"] * 200)
    row = {"role": "user", "content": objective + SEPARATOR + SEPARATOR.join(parts) + suffix}
    get_node = Mock(wraps=engine._dag.get_node)
    monkeypatch.setattr(engine._dag, "get_node", get_node)
    expected = row["content"] if suffix else objective
    assert engine._latest_user_context_anchor([row], []) == expected
    assert get_node.call_count == len(parts)
