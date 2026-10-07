"""#958: capped objective carry, sanitized-tail dedupe, and linear DAG scans."""

from unittest.mock import Mock

import pytest

import hermes_lcm.tokens as tokens
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine, strip_injected_context_blocks
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


def _assemble_round2(engine, row, tail, cap=None):
    engine._pending_context_anchor_messages = [row, *tail]
    try:
        return engine._assemble_context(
            None, tail, assembly_cap_override=cap, include_lcm_note=False, persist=False,
        )
    finally:
        engine._pending_context_anchor_messages = None


def _assert_round2_whole_row_bytes(engine, row, tail, monkeypatch, expected_anchor):
    """I1/I2: compare a roomy assembly with the unchanged candidate-0-only path."""
    assert engine._latest_user_context_anchor([row], []) == expected_anchor
    assembled = _assemble_round2(engine, row, tail)
    assert assembled[0]["content"].startswith(strip_injected_context_blocks(expected_anchor))
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_latest_user_context_anchor_candidates", lambda *_: [expected_anchor])
        assert assembled == _assemble_round2(engine, row, tail)
    return assembled


def test_round2_reemitted_chain_retries_fallback_after_trim(engine, monkeypatch):
    # judge-2/test_judge2_chain.py: feed the engine's uncapped row back into assembly.
    part = _parts(engine)[0]
    objective = PREFIX + "\nOlder objective: " + "detail " * 400
    merged = {"role": "user", "content": objective + SEPARATOR + part + "\n\n" + NEW}
    tail = [{"role": "assistant", "content": "Continuing."}]
    first = _assemble_round2(engine, merged, tail)
    assert first[0]["content"] == merged["content"] + SEPARATOR + part
    reemitted = first[0]
    _assert_round2_whole_row_bytes(engine, reemitted, tail, monkeypatch, merged["content"])
    expected = PREFIX + "\n" + NEW
    budget = count_message_tokens({"role": "user", "content": expected}) + 20
    assert count_message_tokens(merged) > budget
    out = _assemble_round2(engine, reemitted, tail, budget + count_messages_tokens(tail))
    assert "\n".join(row["content"] for row in out).count(NEW) == 1
    assert out[0]["content"] == expected


def test_round2_default_survival_budget_retries_fallback(engine, monkeypatch):
    # judge-5/test_judge5_probe.py::test_p3_survival_budget_path.
    part = _parts(engine)[0]
    objective = PREFIX + "\nOlder objective: " + "detail " * 2000
    merged = {"role": "user", "content": objective + SEPARATOR + part + "\n\n" + NEW}
    tail = [{"role": "assistant", "content": "Continuing."}]
    _assert_round2_whole_row_bytes(engine, merged, tail, monkeypatch, merged["content"])
    assert engine._config.max_assembly_tokens == engine._config.reserve_tokens_floor == 0
    engine.context_length = 4000
    assert count_message_tokens(merged) > engine._survival_ceiling()
    out = _assemble_round2(engine, merged, tail)
    assert "\n".join(row["content"] for row in out).count(NEW) == 1
    assert out[0]["content"].startswith(PREFIX + "\n" + NEW)


def test_round2_double_merge_uses_last_resort_only(engine, monkeypatch):
    # judge-7/test_judge7_double_merge.py: keep the first-run fallbacks verbatim.
    first_part, second_part = _parts(engine, count=2)
    objective = PREFIX + "\nOlder objective."
    first_new = "NEW1 first merged" + SEPARATOR + second_part + "\n\nNEW2 latest"
    merged = {"role": "user", "content": objective + SEPARATOR + first_part + "\n\n" + first_new}
    tail = [{"role": "assistant", "content": "Continuing."}]
    _assert_round2_whole_row_bytes(engine, merged, tail, monkeypatch, merged["content"])
    expected = PREFIX + "\nNEW2 latest"
    candidates = engine._latest_user_context_anchor_candidates([merged], [])
    assert candidates == [
        merged["content"], objective + "\n\n" + first_new, PREFIX + "\n" + first_new, expected,
    ]
    budget = count_message_tokens({"role": "user", "content": expected})
    assert all(count_message_tokens({"role": "user", "content": c}) > budget for c in candidates[:-1])
    out = _assemble_round2(engine, merged, tail, budget + count_messages_tokens(tail))
    assert "\n".join(row["content"] for row in out).count("NEW2 latest") == 1
    assert out[0]["content"] == expected


def test_round2_candidate_is_measured_by_emitted_size(engine, monkeypatch):
    # J12 judge-3/test_j3_probe.py::test_j3_block_in_new: metadata is host-appended.
    part = _parts(engine)[0]
    objective = PREFIX + "\nOLDER-OBJECTIVE keep me."
    new_raw = NEW + "\n<relevant-memories>\n" + "mem " * 300 + "\n</relevant-memories>"
    merged = {"role": "user", "content": objective + SEPARATOR + part + "\n\n" + new_raw}
    tail = [{"role": "assistant", "content": "Continuing."}]
    _assert_round2_whole_row_bytes(engine, merged, tail, monkeypatch, merged["content"])
    candidate = objective + "\n\n" + new_raw
    expected = strip_injected_context_blocks(candidate)
    budget = count_message_tokens({"role": "user", "content": expected}) + 5
    assert count_message_tokens({"role": "user", "content": candidate}) > budget
    assert count_message_tokens({"role": "user", "content": strip_injected_context_blocks(merged["content"])}) > budget
    out = _assemble_round2(engine, merged, tail, budget + count_messages_tokens(tail))
    assert out[0]["content"] == expected
    assert "\n".join(row["content"] for row in out).count(NEW) == 1
    assert count_messages_tokens(out) <= budget + count_messages_tokens(tail)


@pytest.mark.parametrize("double_merge", [False, True])
def test_round2_dashed_merge_separator_is_removed(engine, monkeypatch, double_merge):
    # judge-9/test_j9.py; also exercise the same separator after the last verified run.
    parts = _parts(engine, count=2 if double_merge else 1)
    objective = PREFIX + "\nOlder objective: " + "detail " * 400
    content = objective + SEPARATOR + parts[0]
    if double_merge:
        content += "\n\nNEW1 first merged" + SEPARATOR + parts[1]
    newest = "NEW via dashed separator"
    merged = {"role": "user", "content": content + SEPARATOR + newest}
    tail = [{"role": "assistant", "content": "Continuing."}]
    _assert_round2_whole_row_bytes(engine, merged, tail, monkeypatch, merged["content"])
    expected = PREFIX + "\n" + newest
    budget = count_message_tokens({"role": "user", "content": expected}) + 4
    candidates = engine._latest_user_context_anchor_candidates([merged], [])
    assert candidates[-1] == expected
    assert all(count_message_tokens({"role": "user", "content": c}) > budget for c in candidates[:-1])
    out = _assemble_round2(engine, merged, tail, budget + count_messages_tokens(tail))
    assert out[0]["content"] == expected
    assert "\n".join(row["content"] for row in out).count(newest) == 1
