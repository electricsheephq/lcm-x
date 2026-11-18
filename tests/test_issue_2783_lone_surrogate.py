"""#2783: host-normalized text and exactly-once replay across persistence paths."""

import base64
import json
import logging
from contextlib import closing
from copy import deepcopy
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.claude_code_capture import project_entry
from hermes_lcm import claude_code_store, ingest_ledger, sender_capture
from hermes_lcm.ingest_classifier import SqliteLedgerSnapshot
from tests.harness import engine_context, pin_char_counter, stored_rows
from tests.test_anchor_unstored_before_cursor import _presence
from tests.test_claude_code_store import _entry, _open, _rows, _transcript

SURROGATE = "\ud83d"
REPLACEMENT = "\ufffd"
TOOL_CALL_CELLS = ("tool_args_string", "tool_args_dict", "function_name", "escaped_args_string")
CELLS = ("user_text", "content_part", "tool_result", "tool_call_id", *TOOL_CALL_CELLS)


def host_rewrite(value):
    if isinstance(value, str):
        return "".join(REPLACEMENT if 0xD800 <= ord(char) <= 0xDFFF else char for char in value)
    if isinstance(value, list):
        return [host_rewrite(item) for item in value]
    if isinstance(value, dict):
        return {key: host_rewrite(item) for key, item in value.items()}
    return value


def messages(cell):
    text = f"hello {SURROGATE} there"
    call_id = f"call{SURROGATE}" if cell == "tool_call_id" else "call"
    tool_name = f"lookup{SURROGATE}" if cell == "function_name" else "lookup"
    arguments = {
        "tool_args_string": '{"term": "' + SURROGATE + '"}',
        "tool_args_dict": {"term": SURROGATE},
        "escaped_args_string": r'{"term": "\ud83d"}',
        "colliding_keys": {SURROGATE: "first", "\ud83e": "second", REPLACEMENT: "unchanged"},
    }.get(cell, "{}")
    user_content = {
        "user_text": text,
        "content_part": [{"type": "text", "text": text}],
    }.get(cell, "Find the fixture")
    rows = [{"role": "user", "content": user_content}]
    if cell not in ("user_text", "content_part"):
        rows.extend([
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function", "function": {"name": tool_name, "arguments": arguments},
            }]},
            {"role": "tool", "tool_call_id": call_id, "content": text if cell == "tool_result" else "fixture result"},
        ])
    rows.append({"role": "assistant", "content": "Finished."})
    return [dict(row, timestamp=float(index)) for index, row in enumerate(rows, 1)]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    records = []

    class CollectRecords(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("hermes_lcm.store")
    handler = CollectRecords()
    logger.addHandler(handler)
    try:
        yield
        bad = [record.getMessage() for record in records if "replaced surrogates" in record.getMessage()]
        if bad:
            pytest.fail(f"Unexpected message write-net warnings: {bad}")
    finally:
        logger.removeHandler(handler)


def engine_at(path, **config):
    return engine_context(path, window=128000, conversation_id="conv", embeddings_enabled=False, **config)


def drain_bind_maintenance(engine):
    # The bind-time rollup pass writes on its own connection, and the sender stamp after a
    # pre-ingest is non-blocking by design, so let that pass finish first.
    assert engine.drain_rollup_maintenance(timeout=10)
    assert engine.drain_embedding_maintenance(timeout=10)


def sql_messages(engine):
    return engine._store.connection.execute(
        "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages WHERE session_id = ? ORDER BY store_id",
        ("S",),
    ).fetchall()


def expected_rows(view):
    return [
        (row["role"], row["content"] if isinstance(row["content"], str)
         else json.dumps(row["content"], ensure_ascii=False, sort_keys=True),
         row.get("tool_call_id"), row.get("tool_calls"), row.get("tool_name"))
        for row in host_rewrite(view)
    ]


def assert_escaped_args_readers(engine, view):
    conn = engine._store.connection
    lineage = ingest_ledger.persisted_lineage(conn, "S", ingest_ledger.lineage_key(None, "S"))
    snapshot = SqliteLedgerSnapshot(conn, lineage, session_id="S")
    assert [row["tool_calls"] for row in snapshot.bindings.values() if row.get("role") == "assistant"
            and row.get("tool_calls")] == [view[1]["tool_calls"]]

    # Exercise the presence reader with the outer JSON still serialized and a different inner JSON spelling.
    call = deepcopy(view[1])
    call["tool_calls"][0]["function"]["arguments"] = json.dumps({"term": REPLACEMENT}, ensure_ascii=False)
    call["tool_calls"] = json.dumps(call["tool_calls"], ensure_ascii=False)
    assert _presence(engine, [call, view[2]]) == []


@pytest.mark.parametrize("cell", CELLS)
def test_warm_raw_and_host_rewritten_view(tmp_path, caplog, cell):
    view = messages(cell)
    original = deepcopy(view)
    with engine_at(tmp_path) as engine:
        counts = []
        for incoming in (view, deepcopy(view), host_rewrite(view)):
            engine.ingest(incoming)
            counts.append(len(sql_messages(engine)))
        rows = [(role, text, call_id, json.loads(calls) if calls else None, name)
                for role, text, call_id, calls, name in sql_messages(engine)]
        print(f"warm {cell}: counts={counts} ingest_failures={engine._ingest_failure_count} "
              f"ledger_prepare_failures={engine._ingest_ledger_counters.get('ledger.prepare_failed', 0)}")
        assert counts == [len(view)] * 3
        assert rows == expected_rows(view)
        assert engine._ingest_failure_count == 0
        assert not [r.message for r in caplog.records if "ledger.prepare_failed" in r.message]
    assert view == original


@pytest.mark.parametrize("cell", CELLS)
def test_cold_raw_view(tmp_path, caplog, cell):
    view = messages(cell)
    with engine_at(tmp_path) as engine:
        engine.ingest(view)
        before = sql_messages(engine)
    with engine_at(tmp_path) as resumed:
        resumed.ingest(deepcopy(view))
        after = sql_messages(resumed)
        errors = [record.message for record in caplog.records
                  if "ingest failed" in record.message or "ledger.prepare_failed" in record.message]
        print(f"cold {cell}: before={len(before)} after={len(after)} failures={ascii(errors)}")
        assert not errors
        assert len(before) == len(view)
        assert after == before
        assert resumed._ingest_failure_count == 0


@pytest.mark.parametrize("cell", TOOL_CALL_CELLS)
def test_cold_tool_calls_with_replay_fingerprint(tmp_path, caplog, cell):
    view = messages(cell)
    with engine_at(tmp_path) as engine:
        engine.ingest(view)
        before = sql_messages(engine)
        # A prior recovery fingerprint makes cold reconciliation hash the candidate prefixes.
        engine._remember_native_recovery_replay_snapshot_digest("a" * 64)
    with engine_at(tmp_path) as resumed:
        resumed.ingest(deepcopy(view))
        errors = [r.message for r in caplog.records if "ingest failed" in r.message]
        print(f"cold fingerprint {cell}: before={len(before)} after={len(sql_messages(resumed))} "
              f"failures={ascii(errors)}")
        assert not errors
        assert len(before) == len(view)
        assert sql_messages(resumed) == before
        if cell == "escaped_args_string":
            assert_escaped_args_readers(resumed, view)


def compact_view(tmp_path, monkeypatch, view, summary):
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", lambda **kw: (summary, 1))
    view += [{"role": role, "content": f"turn {index} {role} " + "alpha beta gamma delta " * 40}
             for index in range(8) for role in ("user", "assistant")]
    with engine_at(tmp_path, fresh_tail_count=2, leaf_chunk_tokens=1) as engine:
        engine.compress(view, current_tokens=engine.threshold_tokens + 1, force=True)
        before = sql_messages(engine)
        nodes = engine._dag.get_session_nodes("S")
        engine.ingest(deepcopy(view))
        print(f"compaction: status={engine.last_compression_status} nodes={len(nodes)} rows={len(before)}")
        assert engine.last_compression_status == "compacted"
        assert len(before) == len(view)
        assert sql_messages(engine) == before
        assert engine._ingest_failure_count == 0
        assert nodes
        assert all(node.source_ids for node in nodes)
        return nodes


@pytest.mark.parametrize("cell", ["user_text", "escaped_args_string"])
def test_compaction_then_raw_replay(tmp_path, monkeypatch, caplog, cell):
    compact_view(tmp_path, monkeypatch, messages(cell), "Earlier turns.\nExpand for details about: turns")
    assert not [r.message for r in caplog.records if "ledger.prepare_failed" in r.message]


def test_summary_and_expand_hint_committed_in_host_form(tmp_path, monkeypatch):
    raw = f"Earlier fixture {SURROGATE}.\nExpand for details about: fixture {SURROGATE}"
    nodes = compact_view(tmp_path, monkeypatch, host_rewrite(messages("user_text")), raw)
    assert nodes[0].summary == host_rewrite(raw)
    assert nodes[0].expand_hint == f"fixture {REPLACEMENT}"


def seed_legacy_tool_calls(tmp_path, view):
    with engine_at(tmp_path) as engine:
        conn = engine._store.connection
        conn.executemany(
            "INSERT INTO messages (session_id, source, conversation_id, role, content, tool_call_id, "
            "tool_calls, timestamp, observed_at) VALUES ('S', 'cli', 'conv', ?, ?, ?, ?, ?, ?)",
            [(row["role"], row["content"], row.get("tool_call_id"),
              json.dumps(row["tool_calls"]) if "tool_calls" in row else None,
              row["timestamp"], row["timestamp"]) for row in view],
        )
        conn.commit()
        before = sql_messages(engine)
        assert r"\ud83d" in before[1][3]
        return before


def test_upgrade_legacy_tool_calls_escape(tmp_path):
    view = messages("tool_args_dict")
    before = seed_legacy_tool_calls(tmp_path, view)
    with engine_at(tmp_path) as resumed:
        resumed.ingest(view)
        assert resumed._ingest_failure_count == 0
        assert sql_messages(resumed) == before
        assert stored_rows(resumed)[1]["tool_calls"] == host_rewrite(view[1]["tool_calls"])


def test_ledger_snapshot_canonicalizes_legacy_tool_calls(tmp_path):
    view = messages("tool_args_dict")
    with engine_at(tmp_path) as engine:
        engine.ingest(host_rewrite(view))
        conn = engine._store.connection
        conn.execute("UPDATE messages SET tool_calls = ? WHERE session_id = 'S' AND tool_calls IS NOT NULL",
                     (json.dumps(view[1]["tool_calls"]),))
        lineage = ingest_ledger.persisted_lineage(conn, "S", ingest_ledger.lineage_key(None, "S"))
        snapshot = SqliteLedgerSnapshot(conn, lineage, session_id="S")
        assert [row["tool_calls"] for row in snapshot.bindings.values() if row.get("tool_calls")] == [
            host_rewrite(view[1]["tool_calls"]),
        ]


@pytest.mark.parametrize("escaped_field", ["name", "arguments"])
def test_anchor_presence_accepts_fresh_result_for_stored_escaped_call(tmp_path, escaped_field):
    call, result = messages("escaped_args_string")[1:3]
    function = call["tool_calls"][0]["function"]
    canonical_arguments = json.dumps({"term": REPLACEMENT}, ensure_ascii=False)
    if escaped_field == "name":
        function["name"] += SURROGATE
        function["arguments"] = canonical_arguments
    stored = host_rewrite(call)
    stored["tool_calls"][0]["function"]["arguments"] = canonical_arguments
    incoming = {**call, "tool_calls": json.dumps(call["tool_calls"])}
    with engine_at(tmp_path) as engine:
        engine.ingest([stored])
        # A reused id would suppress the fresh result from the presence audit.
        assert _presence(engine, [incoming, result]) == [1]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Legacy JSON decodes adjacent surrogate halves as one emoji")
@pytest.mark.parametrize("form", ["raw", "host_rewritten"])
def test_known_limit_legacy_tool_calls_surrogate_pair(tmp_path, form):
    view = messages("tool_args_dict")
    view[1]["tool_calls"][0]["function"]["arguments"]["term"] += "\ude00"
    before = seed_legacy_tool_calls(tmp_path, view)
    with engine_at(tmp_path) as resumed:
        resumed.ingest(view if form == "raw" else host_rewrite(view))
        assert sql_messages(resumed) == before


def test_canonical_value_reserves_unchanged_keys():
    from hermes_lcm.message_content import canonical_value

    raw = {SURROGATE: 1, REPLACEMENT: 2, "\ud83e": 3, f"{REPLACEMENT}~3": 4}
    expected = [(REPLACEMENT, 2), (f"{REPLACEMENT}~3", 4), (f"{REPLACEMENT}~2", 1), (f"{REPLACEMENT}~4", 3)]
    assert list(canonical_value(raw).items()) == expected
    assert list(canonical_value(raw).items()) == expected


def test_colliding_keys_survive_warm_cold_and_host_rewritten_replay(tmp_path):
    view = messages("colliding_keys")
    view[0]["content"] += SURROGATE
    original = deepcopy(view)
    with engine_at(tmp_path) as engine:
        engine.ingest(view)
        before = sql_messages(engine)
        assert len(before) == 4
        calls = json.loads(before[1][3])
        assert calls[0]["function"]["arguments"] == {
            REPLACEMENT: "unchanged", f"{REPLACEMENT}~2": "first", f"{REPLACEMENT}~3": "second",
        }
        engine.ingest(view)
        assert sql_messages(engine) == before
        assert engine._ingest_failure_count == 0
    with engine_at(tmp_path) as resumed:
        for replay in (deepcopy(view), host_rewrite(view)):
            resumed.ingest(replay)
            assert sql_messages(resumed) == before
            assert resumed._ingest_failure_count == 0
    assert view == original


@pytest.mark.parametrize("argument_form", ["dict", "escaped_string"])
def test_colliding_keys_cold_replay_ignores_key_order(tmp_path, argument_form):
    first, reordered = messages("colliding_keys"), messages("colliding_keys")
    for view, reverse in ((first, False), (reordered, True)):
        function = view[1]["tool_calls"][0]["function"]
        keys = function["arguments"]
        if reverse:
            keys = dict(reversed(keys.items()))
        function["arguments"] = json.dumps(keys) if argument_form == "escaped_string" else keys
    with engine_at(tmp_path) as engine:
        engine.ingest(first)
        before = sql_messages(engine)
        assert len(before) == len(first)
    with engine_at(tmp_path) as resumed:
        resumed.ingest(reordered)
        assert resumed._ingest_failure_count == 0
        assert sql_messages(resumed) == before


def test_preflight_surrogate_id_preserves_ingest_success(tmp_path, monkeypatch):
    view = messages("tool_call_id")
    view[2]["content"] += (
        "\n\n[OUT-OF-BAND USER MESSAGE — fixture]\nA synthetic follow-up.\n"
        "[/OUT-OF-BAND USER MESSAGE]"
    )
    with engine_at(tmp_path) as engine:
        prepare_anchor = Mock(wraps=engine._prepare_retained_user_anchor)
        monkeypatch.setattr(engine, "_prepare_retained_user_anchor", prepare_anchor)
        engine.should_compress_preflight(view)
        assert len(sql_messages(engine)) == len(view)
        assert engine._ingest_failure_count == 0
        prepare_anchor.assert_called_once()


@pytest.mark.parametrize("entry", ["session_end", "compress", "preflight", "tool_dispatch"])
def test_colliding_keys_at_other_entries(tmp_path, entry):
    view = messages("colliding_keys")
    original = deepcopy(view)
    with engine_at(tmp_path) as engine:
        if entry == "session_end":
            engine.on_session_end("S", view)
        elif entry == "compress":
            engine.compress(view, current_tokens=10)
        elif entry == "preflight":
            engine.should_compress_preflight(view)
        else:
            engine.handle_tool_call("lcm_recall", {"query": "fixture"}, messages=view)
        assert len(sql_messages(engine)) == 4
        assert engine._ingest_failure_count == 0
    assert view == original


@pytest.mark.parametrize("text", ["fixture", f"fixture {SURROGATE}"], ids=["ascii", "surrogate"])
@pytest.mark.parametrize("capture_after_preflight", [False, True], ids=["before_ingest", "after_preflight"])
def test_gateway_sender_survives_canonical_copy(tmp_path, request, text, capture_after_preflight):
    view = [{"role": "user", "content": text}]
    with engine_at(tmp_path) as engine:
        if capture_after_preflight:
            engine.should_compress_preflight(view)
            drain_bind_maintenance(engine)
        pending = sender_capture.create_pending(engine, {
            "session_id": "S", "turn_id": request.node.name, "platform": "telegram",
            "sender_id": "synthetic-a", "conversation_history": view,
        })
        assert pending is not None
        engine.ingest(view)
        assert engine._store.connection.execute(
            "SELECT sender_id, sender_source FROM messages WHERE session_id = 'S'"
        ).fetchall() == [("telegram:synthetic-a", "gateway_principal")]


def test_gateway_sender_does_not_stamp_equal_older_surrogate_row(tmp_path):
    old = {"role": "user", "content": f"fixture {SURROGATE}"}
    with engine_at(tmp_path) as engine:
        engine.should_compress_preflight([old])
        drain_bind_maintenance(engine)
        assert sender_capture.create_pending(engine, {
            "session_id": "S", "turn_id": "turn-b", "platform": "telegram",
            "sender_id": "synthetic-b", "conversation_history": [deepcopy(old)],
        }) is not None
        assert engine._store.connection.execute(
            "SELECT sender_id, sender_source FROM messages WHERE session_id = 'S'"
        ).fetchall() == [(None, None)]


@pytest.mark.parametrize("text", ["fixture", f"fixture {SURROGATE}"], ids=["ascii", "surrogate"])
def test_repeated_surrogate_sender_captured_after_preflight(tmp_path, request, text):
    first = {"role": "user", "content": text}
    original = [first, {"role": "assistant", "content": "Finished."}]
    view = [*original, dict(first)]
    with engine_at(tmp_path) as engine:
        engine.ingest(original)
        engine.should_compress_preflight(view)
        assert len(sql_messages(engine)) == len(view)
        drain_bind_maintenance(engine)
        assert sender_capture.create_pending(engine, {
            "session_id": "S", "turn_id": request.node.name, "platform": "telegram",
            "sender_id": "synthetic-repeat", "conversation_history": view,
        }) is not None
        engine.ingest(view)
        assert len(sql_messages(engine)) == len(view)
        assert engine._store.connection.execute(
            "SELECT sender_id, sender_source FROM messages WHERE session_id = 'S' AND role = 'user' ORDER BY store_id"
        ).fetchall() == [(None, None), ("telegram:synthetic-repeat", "gateway_principal")]


def test_cross_turn_host_whitespace_rewrite_of_surrogate_user(tmp_path):
    user = {"role": "user", "content": f"  fixture {SURROGATE} \n"}
    view = [user, {"role": "assistant", "content": "Finished."}]
    with engine_at(tmp_path) as engine:
        engine.ingest(view)
        before = sql_messages(engine)
        assert len(before) == 2
        user["content"] = user["content"].strip()
        engine.ingest(view + [{"role": "user", "content": "Next turn"}])
        after = sql_messages(engine)
        assert after[:-1] == before
        assert after[-1][:2] == ("user", "Next turn")
        assert engine._ingest_failure_count == 0


def test_answer_ready_tool_reply_is_unicode(tmp_path):
    with engine_at(tmp_path) as engine:
        reply = engine.handle_tool_call("lcm_recall", {
            "query": f"fixture {SURROGATE}", "detail": "answer_ready",
        })
    reply.encode("utf-8")
    assert json.loads(reply)["query"] == f"fixture {REPLACEMENT}"


def test_ascii_entry_keeps_message_objects(tmp_path, monkeypatch):
    with engine_at(tmp_path) as engine:
        messages_seen = Mock(wraps=engine._ingest_messages)
        monkeypatch.setattr(engine, "_ingest_messages", messages_seen)
        view = [{"role": "assistant", "content": ["ASCII", {"text": "ASCII"}]}, {"role": "user", "content": "ASCII"}]
        engine.ingest(view)
        assert messages_seen.call_args.args[0] is view
        assert all(actual is original for actual, original in zip(messages_seen.call_args.args[0], view))


@pytest.mark.parametrize("entry", ["preflight", "session_end", "tool_dispatch", "stopped_turn"])
def test_other_host_message_entries(tmp_path, entry):
    view = messages("tool_call_id")
    view[0]["content"] = "\ud800\udfff\U0001f680"
    view[2]["tool_name"] = f"tool{SURROGATE}"
    original = deepcopy(view)
    with engine_at(tmp_path) as engine:
        if entry == "preflight":
            engine.should_compress_preflight(view)
        elif entry == "session_end":
            engine.on_session_end("S", view)
        elif entry == "tool_dispatch":
            engine.handle_tool_call("lcm_recall", {"query": "fixture"}, messages=view)
        else:
            engine.ingest(view[:1])
            engine.on_turn_complete(view, interrupted=True)
        assert engine._ingest_failure_count == 0
        assert [(r["role"], r["content"], r["tool_call_id"], r["tool_calls"], r["tool_name"])
                for r in stored_rows(engine, "S")] == expected_rows(view)
    assert view == original


def test_claude_code_transcript_projection():
    entry = {"type": "assistant", "uuid": "fixture-uuid", "message": {"content": [
        {"type": "text", "text": f"fixture{SURROGATE}"},
        {"type": "tool_use", "id": f"call{SURROGATE}", "name": "lookup", "input": {"query": SURROGATE}},
    ]}}
    original = deepcopy(entry)
    [(_, row)] = project_entry(entry).rows
    assert row["content"] == f"fixture{REPLACEMENT}"
    [call] = row["tool_calls"]
    assert call["id"] == f"call{REPLACEMENT}"
    assert json.loads(call["function"]["arguments"]) == {"query": REPLACEMENT}
    assert entry == original


@pytest.mark.parametrize("first_entry,edit_entry", [
    ("ingest", "ingest"), ("preflight", "ingest"),
    ("ingest", "preflight"), ("ingest", "tool_dispatch"),
    ("tool_dispatch", "ingest"), ("ingest", "compress"),
    ("compress", "ingest"), ("ingest", "session_end"),
])
@pytest.mark.parametrize("before,after", [
    ("draft", "edited"),
    (f"draft{SURROGATE}", "edited"),
    (f"draft{SURROGATE}", f"edited{SURROGATE}"),
    ("draft", f"edited{SURROGATE}"),
    ("draft", "draft\n\nmore words"),
    (f"draft{SURROGATE}", f"draft{SURROGATE}\n\nmore words"),
], ids=["ascii_edit", "sur_to_ascii", "sur_to_sur", "ascii_to_sur", "merge_ascii", "merge_sur"])
def test_in_place_host_edit_across_entries(tmp_path, first_entry, edit_entry, before, after):
    user = {"role": "user", "content": before}
    view = [{"role": "user", "content": "hi"}, user]
    with engine_at(tmp_path) as engine:
        entries = {
            "ingest": engine.ingest,
            "preflight": engine.should_compress_preflight,
            "tool_dispatch": lambda rows: engine.handle_tool_call("lcm_recall", {"query": "fixture"}, messages=rows),
            "compress": lambda rows: engine.compress(rows, current_tokens=10),
            "session_end": lambda rows: engine.on_session_end("S", rows),
        }
        entries[first_entry](view)
        first = [row[:2] for row in sql_messages(engine)]
        assert view[1] is user
        assert user == {"role": "user", "content": before}
        user["content"] = after
        entries[edit_entry](view)
        final = [row[:2] for row in sql_messages(engine)]
        if edit_entry != "session_end":
            engine.ingest(view)
            assert [row[:2] for row in sql_messages(engine)] == final
        assert view[1] is user
        assert user == {"role": "user", "content": after}
        assert final == [("user", text) for text in host_rewrite(["hi", before, after])]
        assert first == final[:2]
        assert engine._ingest_failure_count == 0


@pytest.mark.parametrize("suffix", ["", SURROGATE], ids=["ascii", "surrogate"])
@pytest.mark.parametrize("entry", ["ingest", "session_end"])
@pytest.mark.parametrize("edit", ["text", "merge", "strip"])
def test_compress_adopted_then_in_place_edit(tmp_path, suffix, entry, edit):
    draft = f"draft{suffix}"
    original = f"  {draft} \n" if edit == "strip" else draft
    user = {"role": "user", "content": original}
    view = [{"role": "user", "content": "hi"}, user, {"role": "assistant", "content": "ok"}]
    with engine_at(tmp_path) as engine:
        engine.ingest(view)
        before = sql_messages(engine)
        assert before == expected_rows(view)
        store_id = stored_rows(engine)[1]["store_id"]

        adopted = engine.compress(view, current_tokens=10)
        adopted_user = adopted[1]
        assert adopted_user["content"] == host_rewrite(original)
        edited = {
            "text": f"edited{suffix}",
            "merge": adopted_user["content"] + "\n\nmore words",
            "strip": adopted_user["content"].strip(),
        }[edit]
        adopted_user["content"] = edited

        expected_tail = [] if edit == "strip" else [{"role": "user", "content": edited}]
        if entry == "ingest":
            next_turn = {"role": "user", "content": "next"}
            adopted.append(next_turn)
            engine.ingest(adopted)
            expected_tail.append(next_turn)
        else:
            engine.on_session_end("S", adopted)

        assert sql_messages(engine) == before + expected_rows(expected_tail)
        assert engine._ingest_failure_count == 0
        key = f"host_rewrite_identity:{store_id}"
        overrides = engine._store.read_metadata_json_many([key])
        if edit == "strip":
            assert overrides[key]["content"] == host_rewrite(draft)
        else:
            assert overrides == {}


def test_stopped_turn_preserves_raw_user_for_next_edit(tmp_path):
    user = {"role": "user", "content": f"draft{SURROGATE}"}
    view = [{"role": "user", "content": "hi"}, user, {"role": "assistant", "content": "Finished."}]
    with engine_at(tmp_path) as engine:
        engine.ingest(view[:1])
        engine.on_turn_complete(view, interrupted=True)
        before = [row[:2] for row in sql_messages(engine)]
        user["content"] = f"edited{SURROGATE}"
        engine.ingest(view)
        assert [row[:2] for row in sql_messages(engine)] == before + [("user", f"edited{REPLACEMENT}")]
        assert before == [("user", "hi"), ("user", f"draft{REPLACEMENT}"), ("assistant", "Finished.")]
        assert engine._ingest_failure_count == 0


def test_claude_code_capture_keeps_colliding_input_keys(tmp_path):
    inputs = dict(zip("\ud800\ud801" + REPLACEMENT, ("first", "second", "unchanged"), strict=True))
    tool = {"type": "tool_use", "id": "t1", "name": "Lookup",
            "input": inputs}
    transcript = _transcript(tmp_path, [_entry("user", "u1", "Find it"), _entry("assistant", "a1", [tool])])
    with closing(_open(tmp_path / "home", tmp_path / "hermes")) as store:
        lineage = claude_code_store.Lineage("s1", None, transcript)
        claude_code_store.drain(store, lineage)
        before = _rows(store)
        assert len(before) == 2
        claude_code_store.drain(store, lineage)
        assert _rows(store) == before
        [calls] = store.connection.execute("SELECT tool_calls FROM messages WHERE tool_calls IS NOT NULL").fetchone()
        arguments = json.loads(json.loads(calls)[0]["function"]["arguments"])
        assert arguments == {REPLACEMENT: "unchanged", f"{REPLACEMENT}~2": "first", f"{REPLACEMENT}~3": "second"}


@pytest.mark.parametrize("redaction", [True, False], ids=["redaction", "payload"])
def test_ingest_reserialization_keeps_colliding_argument_keys(tmp_path, redaction):
    if redaction:
        arguments = r'{"\ud83d":"first","�":"keep","password":"synthetic-password"}'
    else:
        payload = bytes((i * 131 + (i >> 3) * 17 + 7) % 256 for i in range(6000))
        # Escaped slashes break the raw base64 run, so it is only externalized after JSON parsing.
        escaped_payload = base64.b64encode(payload).decode().replace("/", r"\/")
        arguments = rf'{{"\ud800":"{escaped_payload}","\ufffd":"keep"}}'
    view = messages("escaped_args_string")
    view[1]["tool_calls"][0]["function"]["arguments"] = arguments
    original = deepcopy(view)
    with engine_at(tmp_path, sensitive_patterns_enabled=redaction) as engine:
        engine.ingest(view)
        before = sql_messages(engine)
        engine.ingest(deepcopy(view))
        assert sql_messages(engine) == before
        assert engine._ingest_failure_count == 0
    with engine_at(tmp_path, sensitive_patterns_enabled=redaction) as resumed:
        resumed.ingest(deepcopy(view))
        after = sql_messages(resumed)
        assert resumed._ingest_failure_count == 0
    assert len(before) == len(view)
    decoded = json.loads(json.loads(before[1][3])[0]["function"]["arguments"])
    assert decoded[REPLACEMENT] == "keep"
    if redaction:
        assert decoded[f"{REPLACEMENT}~2"] == "first"
        assert decoded["password"].startswith("[LCM sensitive redaction")
    else:
        assert decoded[f"{REPLACEMENT}~2"].startswith("[Externalized LCM ingest payload:")
    assert after == before
    assert view == original
