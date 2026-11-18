"""A host-facing rewrite cannot carry API text for different content (I10, I11)."""

from copy import deepcopy

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.tokens import count_messages_tokens
from tests.harness.engine import engine_context
from tests.invariants.test_issue_2855_kept_row_keys import _assert_host_keys


@pytest.fixture
def engine(tmp_path):
    with engine_context(tmp_path, window=32_768, fresh_tail_count=6, embeddings_enabled=False,
                        temporal_rollups_enabled=False) as instance:
        yield instance


def _assistant(content):
    return {"role": "assistant", "content": content}


def _user(content):
    return {"role": "user", "content": content}


def _tool_pair():
    call = {"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
    return [dict(_assistant(""), tool_calls=[call]),
            {"role": "tool", "content": "Synthetic result", "tool_call_id": "call_1"}]


def _compress(engine, rows, **kwargs):
    before = deepcopy(rows)
    output = engine.compress(rows, current_tokens=count_messages_tokens(rows), **kwargs)
    assert rows == before
    return output


def _cleanup(engine, rows):
    engine._preflight_automatic_request = True
    output = _compress(engine, rows)
    assert engine.last_compression_status == "sanitized"
    return output


@pytest.mark.parametrize("kind", ["merge", "clean-string", "clean-structured"])
def test_rewritten_assistant_drops_late_api_text(engine, kind):
    source = _assistant("first sentence")
    rows = [_user("Question"), source]
    expected = "first sentence"
    if kind == "merge":
        rows.append(_assistant("second sentence"))
        expected += "\nsecond sentence"
    elif kind == "clean-string":
        source["content"] = "<think>internal detail</think>first sentence"
    else:
        expected = [{"type": "text", "text": expected}]
        source["content"] = [{"type": "thinking", "thinking": "internal detail"}, *expected]
    engine.ingest(rows)
    source["api_content"] = "previous API text"

    output = _cleanup(engine, rows)

    assistant, = [row for row in output if row["role"] == "assistant"]
    assert assistant["content"] == expected
    assert "api_content" not in assistant


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_public_survival_projection_drops_api_text(engine, role):
    source = {"role": role, "content": "alpha beta gamma " * 450}
    rows = [source] if role == "user" else [_user("Question"), source]
    engine.ingest(rows)
    source["api_content"] = "previous API text"

    output = _compress(engine, rows, bypass_cooldown=True)

    projected, = [row for row in output if row["role"] == role]
    assert projected["content"] != source["content"]
    assert "[LCM survival fit:" in projected["content"]
    if role == "user":
        watch, _ = engine._host_rewrite_state()
        assert any(projected is row for _, _, watched in watch.values() for row in watched)
    assert "api_content" not in projected


def test_exception_survival_return_drops_api_text(engine, monkeypatch):
    source = dict(_user("alpha beta gamma " * 450), api_content="previous API text")
    rows = [source]
    engine.ingest(rows)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic compaction failure")

    monkeypatch.setattr(engine, "_compress_impl", fail)
    projected, = _compress(engine, rows, bypass_cooldown=True)

    assert engine.last_compression_status == "error"
    assert "[LCM survival fit:" in projected["content"]
    assert "api_content" not in projected


def test_bypassed_child_return_drops_api_text(engine, monkeypatch):
    rows = [_user("Question"), dict(_assistant("<think>internal</think>Answer"), api_content="old API text")]
    monkeypatch.setattr(engine, "_get_host_fallback_compressor", lambda: None)
    engine._mark_thread_context_stateless("synthetic-child")
    try:
        output = _compress(engine, rows, force=True)
        assert engine.last_compression_status == "host_fallback"
    finally:
        engine._clear_thread_context_stateless("synthetic-child")

    assert output[-1]["content"] == "Answer"
    assert "api_content" not in output[-1]


def test_merged_head_preserves_api_text_only_for_equal_visible_content(engine):
    head = dict(_assistant("first sentence"), api_content="provider text for first sentence")
    rows = [_user("Question"), head, *_tool_pair()]
    output = _cleanup(engine, rows)
    assert len(output) == 3 and output[1]["tool_calls"] == rows[2]["tool_calls"]
    assert output[1]["content"] == head["content"]
    assert output[1]["api_content"] == head["api_content"]

    rows[2]["content"] = "second sentence"
    output = _cleanup(engine, rows)
    assert len(output) == 3 and output[1]["tool_calls"] == rows[2]["tool_calls"]
    assert output[1]["content"] == "first sentence\nsecond sentence"
    assert "api_content" not in output[1]


def test_returned_api_text_has_an_exact_input_pair(engine):
    parts = [{"type": "text", "text": "unchanged structured content"}]
    kept = dict(_user(parts), api_content={"text": ["structured sidecar"]}, host_extra="kept")
    empty, tool = _tool_pair()
    empty["api_content"] = "API text for empty content"
    matching = dict(_assistant("<think>internal</think>shared text"), api_content="shared API text")
    mismatching = dict(_assistant("<think>internal</think>other text"), api_content="only with old content")
    rows = [kept, empty, tool, _user("First question"), matching,
            dict(_user("shared text"), api_content="shared API text"), mismatching,
            dict(_user("other text"), api_content="other API text")]

    output = _cleanup(engine, rows)

    _assert_host_keys([(kept, output[0]), (empty, output[1])])
    shared, = [row for row in output if row["role"] == "assistant" and row["content"] == "shared text"]
    assert shared["api_content"] == matching["api_content"]  # The matching pair is on a different input row.
    assert len(rows) == len(output)
    assert all(any("api_content" in source and source.get("content") == row.get("content")
                   and source["api_content"] == row["api_content"] for source in rows)
               for row in output if "api_content" in row)


@pytest.mark.parametrize("late_stamp", [False, True], ids=["initially-stamped", "late-stamp"])
def test_no_progress_hold_survives_unchanged_replay_and_clears_on_leaf(engine, monkeypatch, late_stamp):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Earlier synthetic work.", 1))
    engine.threshold_tokens = 100
    engine._config.leaf_chunk_tokens = 400
    engine._config.fresh_tail_pressure_yield_enabled = False
    engine._config.threshold_full_sweep_enabled = False
    rows = [_user("Current objective. " * 100), _assistant("Current reply. " * 80)]
    api_text = "API text for current objective"
    if not late_stamp:
        rows[0]["api_content"] = api_text
    engine.ingest(rows)
    if late_stamp:
        rows[0]["api_content"] = api_text
    prompt_tokens = count_messages_tokens(rows)

    assert _compress(engine, rows) is rows
    assert engine._last_compress_leaves[1] == 0
    key = "compaction_no_progress:v1:S"
    hold = engine._store.read_metadata_json(key)
    assert hold["reason"] == "no_progress"
    snapshot = engine._no_progress_replay
    assert snapshot is not None
    engine.ingest(rows)
    assert engine._no_progress_replay == snapshot
    assert not engine.should_compress(prompt_tokens)
    assert _compress(engine, rows) is rows
    assert engine._store.read_metadata_json(key) == hold

    rows += [row for index in range(5) for row in (
        _user(f"Question {index}. " * 80), _assistant(f"Reply {index}. " * 80),
    )]
    head = rows[-1]
    rows.append(_assistant("A second part of the reply."))
    engine.ingest(rows)
    head["api_content"] = "API text for the unmerged last reply"
    assert engine._no_progress_replay != snapshot
    assert engine._no_progress_fingerprint() != hold["fingerprint"]
    # A same-process hold retains its pacing floor; an explicit pass can publish a leaf and clear it.
    output = _compress(engine, rows, force=True)
    assert engine._last_compress_leaves[1] == 1, engine.last_compression_noop_reason
    assert engine._no_progress_hold is None
    assert engine._store.read_metadata_json(key) is None
    assert output[-1]["content"].endswith("\nA second part of the reply.")
    assert "api_content" not in output[-1]
