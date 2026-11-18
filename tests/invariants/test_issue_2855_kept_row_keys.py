"""The host's current row metadata survives a compaction replay (I1, I2, I8)."""

import copy
import random

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from hermes_lcm.tokens import count_messages_tokens
from tests.harness.engine import engine_context

REWRITTEN_FIELDS = {"role", "content", "tool_calls", "tool_call_id"}


@pytest.fixture(params=[None, "shadow", "on"], ids=["uid-unset", "uid-shadow", "uid-on"])
def uid_mode(request, monkeypatch):
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    if request.param is not None:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", request.param)


@pytest.fixture
def open_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Synthetic leaf summary.", 1))
    return lambda: engine_context(tmp_path, window=200_000, fresh_tail_count=6, leaf_chunk_tokens=400)


def _row(role, content):
    return {"role": role, "content": content}


def _stamp(row, index):
    row.update(message_uid=f"{index:032x}", timestamp=1_700_000_000.0 + index,
               _row_id=index, _db_persisted=True, display_kind="synthetic",
               host_extra={"sequence": [index, None]})
    if row["role"] == "user":
        row["api_content"] = row["content"] + "\nHost prologue text."


def _history():
    return [row for i in range(12) for row in (
        _row("user", f"Question {i}: " + "alpha beta gamma delta " * 40),
        _row("assistant", f"Answer {i}."),
    )]


def _call(engine, rows, *, leaves):
    given = copy.deepcopy(rows)
    if not leaves:
        # An automatic maintenance request below the host threshold takes the
        # cleanup path and emits the replay view without asking for a summary.
        engine._preflight_automatic_request = True
    result = engine.compress(given, current_tokens=count_messages_tokens(given), force=bool(leaves))
    assert given == rows
    assert (engine._last_compress_leaves or (None, 0))[1] == leaves
    assert engine.last_compression_status == ("compacted" if leaves else "sanitized")
    return result


def _kept_rows(given, returned):
    # Unique role/content pairs give an independent match even when a UID was
    # omitted from the return. This also excludes generated summaries and stubs.
    by_content = {(row["role"], row["content"]): row for row in given}
    assert len(by_content) == len(given)
    return [(by_content[key], row) for row in returned
            if (key := (row["role"], row["content"])) in by_content]


def _assert_host_keys(pairs):
    assert pairs
    problems = []
    for source, output in pairs:
        missing = source.keys() - output.keys()
        changed = {key for key in (source.keys() & output.keys()) - REWRITTEN_FIELDS
                   if output[key] != source[key]}
        if missing or changed:
            problems.append((source["role"], source["content"][:28], sorted(missing), sorted(changed)))
    assert not problems, f"(role, row, missing keys, changed keys): {problems}"


@pytest.mark.parametrize("cell, leaves", [
    ("already-stamped", 1),
    ("late-api", 1),
    ("host-order", 1),
    ("bare-then-flushed", 1),
    ("no-leaf", 0),
    ("auto-after-commit", 0),
    ("restart-between-cycles", 1),
])
def test_kept_keys(open_engine, uid_mode, cell, leaves):
    history = _history()
    for i, row in enumerate(history, 1):
        _stamp(row, i)
    with open_engine() as engine:
        engine.ingest(history)
        if cell in {"auto-after-commit", "restart-between-cycles"}:
            adopted = _call(engine, history, leaves=1)
            _assert_host_keys(_kept_rows(history, adopted))
            engine.on_session_end("S", history)
            engine.on_session_start("S", boundary_reason="compression", old_session_id="S", platform="cli")
            history = [dict(row) for row in adopted]
            if cell == "auto-after-commit":
                _exercise_pass(engine, history, cell, leaves)
        else:
            _exercise_pass(engine, history, cell, leaves)
    if cell == "restart-between-cycles":
        with open_engine() as engine:
            history.extend(_history()[0:12])
            _exercise_pass(engine, history, cell, leaves)


def _exercise_pass(engine, history, cell, leaves):
    current = _row("user", "Current question: " + "epsilon zeta eta theta " * 40)
    if cell == "already-stamped":
        _stamp(current, 101)
    elif cell in {"late-api", "host-order"}:
        _stamp(current, 101)
        expected_api = current.pop("api_content")

    history.append(current)
    if cell in {"already-stamped", "late-api"}:
        engine.ingest(history)
    else:
        engine.should_compress_preflight(history)

    if cell in {"late-api", "host-order"}:
        current["api_content"] = expected_api
    elif cell != "already-stamped":
        _stamp(current, 101)
    if cell == "late-api":
        previous = next(row for row in reversed(history[:-1]) if row["role"] == "user")
        del previous["api_content"]

    if cell in {"host-order", "bare-then-flushed"}:
        call = dict(_row("assistant", ""), tool_calls=[{
            "id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"},
        }])
        tool = dict(_row("tool", "Synthetic tool result"), tool_call_id="call_1")
        if cell == "host-order":
            _stamp(call, 102)
            _stamp(tool, 103)
        history.extend([call, tool])
        engine.ingest(history)  # post_llm_call's ingest
        engine.note_turn_complete()
        if cell == "bare-then-flushed":
            _stamp(call, 102)
            _stamp(tool, 103)

    result = _call(engine, history, leaves=leaves)
    pairs = _kept_rows(history, result)
    assert any(source is current for source, _ in pairs)
    _assert_host_keys(pairs)
    if cell == "late-api":
        assert any(source is previous for source, _ in pairs)
        assert all("api_content" not in output for source, output in pairs if source is previous)


@pytest.mark.parametrize("append", [False, True], ids=["no-new-rows", "cached-prefix"])
def test_returned_rows_preserve_host_key_property(open_engine, uid_mode, append):
    rng = random.Random(2855)
    history = _history()
    for row in history:
        row["host_mutable"] = "first sighting"
    with open_engine() as engine:
        engine.ingest(history)
        for index, row in enumerate(history, 1):
            _stamp(row, index)
            row["host_mutable"] = {"current": index}
            for key in rng.sample(range(10), rng.randint(1, 10)):
                row[f"host_extension_{key}"] = rng.choice([None, False, 0, "", [key], {"key": key}])
        if append:
            history.append(_row("user", "New suffix outside the replay cache."))
        result = _call(engine, history, leaves=0)
        pairs = _kept_rows(history, result)
        assert len(pairs) == len(result) == len(history)
        _assert_host_keys(pairs)


def test_generated_replay_row_does_not_inherit_host_keys(open_engine, monkeypatch):
    host = _row("user", "Synthetic ignored input")
    kept = _row("assistant", "Synthetic reply")
    rows = [host, kept]
    with open_engine() as engine:
        # Give the placeholder builder a filtering decision without requiring regex.
        monkeypatch.setattr(engine, "_compiled_ignore_message_patterns", [object()])
        cached = engine._apply_ignored_active_replay_placeholders(rows, rows, ignored_messages=[True, False])
        host["host_extra"] = "belongs to the original input"
        kept["host_extra"] = "belongs to the kept reply"

        generated, returned = engine._keep_host_held_stubs(rows, cached, rows)

        assert "LCM active replay placeholder: message ignored" in generated["content"]
        assert id(generated) in engine._generated_ignored_active_replay_placeholder_message_ids
        assert "host_extra" not in generated
        assert returned["host_extra"] == kept["host_extra"]


def test_host_held_stub_is_skipped_by_key_refresh(open_engine):
    class HostRow(dict):
        def update(self, *args, **kwargs):
            pytest.fail("The refresh must skip the host's own row")

    host = HostRow(role="tool", content=_HERMES_PRUNED_TOOL_PLACEHOLDER, host_extra="host-owned")
    with open_engine() as engine:
        cached = engine._copy_active_replay_messages_preserving_generated_ids([
            _row("tool", "Original synthetic tool result"),
        ])
        returned, = engine._keep_host_held_stubs([host], cached, [host])

    assert returned is host
