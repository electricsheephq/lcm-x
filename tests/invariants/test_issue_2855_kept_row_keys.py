"""Compaction replay refreshes host keys and retains ingest timestamps (I1, I2, I8)."""

import copy
import random

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from hermes_lcm.tokens import count_messages_tokens
from tests.harness.engine import engine_context, stored_rows

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


def _assert_host_keys(pairs, *, unstamped=()):
    assert pairs
    unstamped_ids = {id(row) for row in unstamped}
    problems = []
    for source, output in pairs:
        if id(source) in unstamped_ids:
            assert "timestamp" not in output, source["content"][:28]
            source = {key: value for key, value in source.items() if key != "timestamp"}
        missing = source.keys() - output.keys()
        changed = {key for key in (source.keys() & output.keys()) - REWRITTEN_FIELDS
                   if output[key] != source[key]}
        if missing or changed:
            problems.append((source["role"], source["content"][:28], sorted(missing), sorted(changed)))
    assert not problems, f"(role, row, missing keys, changed keys): {problems}"


def _assert_cached_rows_unstamped(engine, unstamped):
    pairs = _kept_rows(unstamped, engine._last_active_replay_messages)
    assert len(pairs) == len(unstamped)
    _assert_host_keys(pairs, unstamped=unstamped)


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
    unstamped = []
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
        unstamped.append(current)
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
            unstamped.extend([call, tool])
            _stamp(call, 102)
            _stamp(tool, 103)

    result = _call(engine, history, leaves=leaves)
    if unstamped:
        _assert_cached_rows_unstamped(engine, unstamped)
    pairs = _kept_rows(history, result)
    assert any(source is current for source, _ in pairs)
    # Below-threshold cleanup can return the original host list when only stamps differ.
    _assert_host_keys(pairs, unstamped=unstamped if leaves else ())
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
        unstamped = list(history)
        for index, row in enumerate(history, 1):
            _stamp(row, index)
            row["host_mutable"] = {"current": index}
            for key in rng.sample(range(10), rng.randint(1, 10)):
                row[f"host_extension_{key}"] = rng.choice([None, False, 0, "", [key], {"key": key}])
        if append:
            history.append(_row("user", "New suffix outside the replay cache."))
        result = _call(engine, history, leaves=0)
        _assert_cached_rows_unstamped(engine, unstamped)
        pairs = _kept_rows(history, result)
        assert len(pairs) == len(result) == len(history)
        _assert_host_keys(pairs)


@pytest.mark.parametrize("adopt", [False, True], ids=["replay", "adopt"])
@pytest.mark.parametrize("stamp_at_ingest", [False, True], ids=["late-restamp", "stable-first-stamp"])
def test_timestamp_replay_stored_rows_match_main(open_engine, adopt, stamp_at_ingest):
    history = [row for i in range(6) for row in (
        _row("user", f"Question {i}: " + "alpha beta " * 30),
        _row("assistant", f"Answer {i}."),
    )]
    epoch = 1_700_000_000.0
    if stamp_at_ingest:
        for i, row in enumerate(history):
            row["timestamp"] = epoch + i

    with open_engine() as engine:
        engine.ingest(history)
        for turn in range(3):
            current = _row("user", f"Current {turn}: " + "zeta eta " * 20)
            if stamp_at_ingest:
                current["timestamp"] = epoch + 100 + turn
            history.append(current)
            engine.should_compress_preflight(history)
            if not stamp_at_ingest:
                for i, row in enumerate(history):
                    row["timestamp"] = epoch + i
            given = copy.deepcopy(history)
            result = engine.compress(given, current_tokens=count_messages_tokens(given))
            if adopt:
                history = [dict(row) for row in result]

            reply = _row("assistant", f"Reply {turn}.")
            if stamp_at_ingest:
                reply["timestamp"] = epoch + 200 + turn
            history.append(reply)
            engine.ingest(history)
            if not stamp_at_ingest:
                # The next compress sees this reply with its new positional stamp.
                reply["timestamp"] = epoch + 200 + turn

        # The archived main base (0d01c2acdd7e) stores all 18 messages once in each cell.
        assert len(stored_rows(engine)) == 18


def test_timestamp_same_object_stored_rows_match_main(open_engine):
    row = _row("user", "Synthetic same-object replay")
    with open_engine() as engine:
        engine.ingest([row])
        for timestamp in (1_700_000_001.0, 1_700_000_002.0):
            row["timestamp"] = timestamp
            engine.ingest([row])

        # The archived main base (0d01c2acdd7e) stores the bare row once.
        assert len(stored_rows(engine)) == 1


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
