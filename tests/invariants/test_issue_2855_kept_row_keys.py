"""Compaction replay refreshes host keys and retains ingest timestamps (I1, I2, I8)."""

import copy
import random

import pytest

from hermes_lcm import engine as engine_module, host_uid_emit
from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER
from hermes_lcm.tokens import count_messages_tokens
from tests.harness.engine import engine_context, stored_rows
from tests.invariants.test_issue_2815_c8 import _alter_guard_input, _output_sources, _set_frontier
from tests.test_host_uid_shadow import _state_db

REWRITTEN_FIELDS = {"role", "content", "tool_calls", "tool_call_id"}


@pytest.fixture(params=[None, "shadow", "on"], ids=["uid-unset", "uid-shadow", "uid-on"])
def uid_mode(request, monkeypatch):
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", True)
    if request.param is not None:
        monkeypatch.setenv("LCM_HOST_MESSAGE_UID", request.param)
    return request.param


@pytest.fixture
def open_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: ("Synthetic leaf summary.", 1))
    return lambda **kwargs: engine_context(
        tmp_path, window=200_000, fresh_tail_count=6, leaf_chunk_tokens=400, **kwargs,
    )


@pytest.fixture
def open_seeded_engine(tmp_path, open_engine):
    host_home = tmp_path / "home"
    host_home.mkdir()
    _state_db(host_home, [("S", None, None)])
    return lambda: open_engine(conversation_id="conv")


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


@pytest.mark.parametrize("uid_mode", ["on"], ids=["uid-on"], indirect=True)
def test_ingest_binds_host_uid(open_seeded_engine, uid_mode):
    user = _row("user", "Synthetic bound user")
    _stamp(user, 1)
    with open_seeded_engine() as engine:
        engine.ingest([user])
        lineage = engine._host_uid_lineage_key()[0]
        assert lineage is not None
        stored, = stored_rows(engine)
        assert engine._store.host_uid_bindings_for(lineage, [user["message_uid"]]) == {
            user["message_uid"]: [(stored["store_id"], "canonical")],
        }


@pytest.mark.parametrize("uid_mode", ["on"], ids=["uid-on"], indirect=True)
@pytest.mark.parametrize("restore", ["missing-header", "no-leaf"])
def test_uid_trim_restores_late_api_content(open_seeded_engine, uid_mode, monkeypatch, restore):
    publish_leaf = restore == "missing-header"
    history = _history() if publish_leaf else _history()[:8]
    if publish_leaf:
        # Sanitizing the restored view must reduce its token count, so the
        # hidden-only fallback does not substitute the original host list.
        history[-1]["content"] += "<think>Synthetic reasoning to strip.</think>"
    for i, row in enumerate(history, 1):
        _stamp(row, i)
    kept = history[-2]
    api_text = kept.pop("api_content")
    covered_uid = history[0]["message_uid"]
    with open_seeded_engine() as engine:
        engine.ingest(history)
        ids = [row["store_id"] for row in stored_rows(engine)]
        _set_frontier(engine, ids[:1] if publish_leaf else ids[:2])
        nodes_before = engine._dag.get_session_nodes("S")

        kept["api_content"] = api_text
        assert engine._host_uid_resolve([kept])[0]["store_id"] == ids[-2]
        drop = engine._drop_uid_covered_live_rows
        trimmed = []

        def record_trim(rows):
            result = drop(rows)
            trimmed.append({row["message_uid"] for row in rows} - {row["message_uid"] for row in result})
            return result

        monkeypatch.setattr(engine, "_drop_uid_covered_live_rows", record_trim)
        if publish_leaf:
            def remove_header(result, _patch):
                assert all(row.get("message_uid") != covered_uid for row in result)
                assert ids[0] in _output_sources(engine, result)
                return [row for row in result if ids[0] not in _output_sources(engine, [row])]

            _alter_guard_input(engine, monkeypatch, remove_header)

        output = engine.compress(history, current_tokens=count_messages_tokens(history), force=publish_leaf)
        assert len(trimmed) == 1 and covered_uid in trimmed[0]
        assert len(engine._dag.get_session_nodes("S")) == len(nodes_before) + int(publish_leaf)
        assert (engine._last_compress_leaves or (None, 0))[1] == int(publish_leaf)
        assert any(row.get("message_uid") == covered_uid for row in output)
        if publish_leaf:
            assert output[-1]["content"] == history[-1]["content"].split("<think>")[0]
        returned, = [row for row in output if row.get("message_uid") == kept["message_uid"]]
        assert returned.get("api_content") == api_text


@pytest.mark.parametrize("uid_mode", ["off", "shadow", "on"], ids=["uid-off", "uid-shadow", "uid-on"], indirect=True)
def test_lone_surrogate_in_uid_stamped_rows_commits_a_pass_in_every_uid_mode(open_seeded_engine, uid_mode):
    # Hotfix tree (port 7, #2783): uid-stamped rows hold a lone surrogate, one covered by the leaf, one kept.
    history = _history()
    for i, row in enumerate(history, 1):
        _stamp(row, i)
    history[0]["content"] += " \udc00"
    kept = history[-2]
    assert kept["role"] == "user"
    kept["content"] += " \udc01"
    kept["api_content"] += " \udc01"
    given = copy.deepcopy(history)
    with open_seeded_engine() as engine:
        engine.ingest(history)
        lineage = engine._host_uid_lineage_key()[0]
        assert lineage is not None
        surrogate_uids = [history[0]["message_uid"], kept["message_uid"]]
        stored = stored_rows(engine)
        bindings = engine._store.host_uid_bindings_for(lineage, surrogate_uids)
        expected = {uid: [(row["store_id"], "canonical")]
                    for uid, row in zip(surrogate_uids, (stored[0], stored[-2]))}
        assert bindings == ({} if uid_mode == "off" else expected)
        if uid_mode == "on":
            assert engine._host_uid_resolve(history)[0]["store_id"] == stored[0]["store_id"]

        result = _call(engine, history, leaves=1)
        assert engine._ingest_failure_count == 0
        assert stored_rows(engine) == stored
        texts = [row["content"] for row in stored]
        assert not any("\udc00" in text or "\udc01" in text for text in texts)
        assert sum("\ufffd" in text for text in texts) == 2
        output, = [row for row in result if row.get("message_uid") == kept["message_uid"]]
        assert output["content"] == kept["content"].replace("\udc01", "\ufffd")
        assert output["api_content"] == kept["api_content"].replace("\udc01", "\ufffd")
        _assert_host_keys([(dict(kept, api_content=output["api_content"]), output)])  # api_content checked above
        engine.ingest(copy.deepcopy(history))
        assert len(stored) == len(history)
        assert stored_rows(engine) == stored
        assert not {name: count for name, count in (getattr(engine, "_host_uid_counters", None) or {}).items()
                    if "disagree" in name and count}
        assert engine._ingest_failure_count == 0
    assert history == given  # host objects are never mutated


def test_compress_first_watches_raw_surrogate_row_for_in_place_strip(open_seeded_engine):
    user = _row("user", "  Synthetic question \udc00 \n")
    history = [user, _row("assistant", "Synthetic answer")]
    with open_seeded_engine() as engine:
        engine.compress(history, current_tokens=count_messages_tokens(history))
        before = stored_rows(engine)
        assert len(before) == len(history)

        user["content"] = user["content"].strip()
        new = _row("user", "Next synthetic question")
        engine.ingest([*history, new])

        key = f"host_rewrite_identity:{before[0]['store_id']}"
        override = engine._store.read_metadata_json_many([key])
        assert key in override
        assert override[key]["content"] == user["content"].replace("\udc00", "\ufffd")
        *prefix, appended = stored_rows(engine)
        assert prefix == before
        assert appended["content"] == new["content"]
