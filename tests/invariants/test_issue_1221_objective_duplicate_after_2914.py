"""I1: the ACP turn-finalizer replay in reliability cell default-window/objective-1221."""

import hashlib
import json

import pytest

from hermes_lcm import compaction, host_uid_emit, survival_fit
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import extract_externalized_refs, load_externalized_payload
from hermes_lcm.tokens import count_messages_tokens
from tests.test_host_uid_shadow import _state_db
from tests.test_issue_2863_host_contract import force_char_estimate, host_rough

WINDOW = 32_768
SESSION = "S0"
OBJECTIVE = "[T02] user turn 2: Keep the following objective verbatim.\n" + "plan " * 9600 + " end."
PRESERVED = "[Current user objective preserved from compacted history]\n"
TIGHT_FIT_PRESSURES = {
    # The seventh round sets the peak overhead. Calibrate it to the stub summary and host counter
    # so four normal excerpts plus the full latest result sit just above the survival budget.
    "r34.15-0.21.5": {2: (26_329,), 3: (28_372, 30_272), 7: (25_554, 26_451), 11: (25_479, 26_148)},
    "upstream-main": {2: (26_339,), 3: (28_387, 30_282), 7: (25_579, 26_451), 11: (25_504, 26_148)},
}


def read_file_rows(step, size, *, reports_not_found):
    call_id = f"call_T02_{step - 1}_0"
    text = "".join(f"Record {i:04d}: datum for task step {step:02d}.\n" for i in range(size // 30 + 1))[:size]
    lines = text.splitlines()
    payload = {
        "content": "\n".join(f"{i}|{line}" for i, line in enumerate(lines, 1)),
        "total_lines": len(lines), "file_size": size, "truncated": False, "is_binary": False, "is_image": False,
    }
    if reports_not_found:
        payload["not_found"] = False
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "call_id": call_id, "response_item_id": f"fc_T02_{step - 1}_0", "type": "function",
             "function": {"name": "read_file", "arguments": json.dumps({"path": f"~/objective-1221/big-{step:02d}.txt"})}},
        ]},
        {"role": "tool", "tool_call_id": call_id, "tool_name": "read_file", "content": json.dumps(payload)},
    ]


def row_key(engine, row):
    content = row.get("content") or ""
    refs = extract_externalized_refs(content) if row["role"] == "tool" else []
    if refs:
        assert len(refs) == 1
        content = load_externalized_payload(refs[0], config=engine._config, hermes_home=engine._hermes_home)["content"]
    return row["role"], content, row.get("tool_call_id"), row.get("tool_calls") or []


@pytest.mark.parametrize(("uids", "final_attempt", "restamp", "tight_fit_host", "head_identity"), [
    pytest.param(False, True, False, None, False, id="ref-0.21.5"),
    pytest.param(True, True, False, None, False, id="upstream-uid"),
    pytest.param(False, False, False, None, False, id="boundary-only"),
    pytest.param(False, True, True, None, False, id="new-occurrence"),
    pytest.param(False, True, False, "r34.15-0.21.5", False, id="r34.15-0.21.5"),
    pytest.param(False, True, False, "upstream-main", False, id="upstream-main"),
    # #3013 (C1a): the replay identity of the generated head is not its raw span.
    pytest.param(False, True, False, None, True, id="head-identity"),
    pytest.param(True, True, False, None, True, id="head-identity-uid"),
    pytest.param(False, False, False, None, True, id="head-identity-boundary-only"),
    pytest.param(False, True, True, None, True, id="head-identity-new-occurrence"),
])
def test_objective_is_stored_once_after_in_place_compaction(
    tmp_path, monkeypatch, uids, final_attempt, restamp, tight_fit_host, head_identity,
):
    monkeypatch.setattr(compaction, "_host_has_post_tool_reanchor", lambda: True, raising=False)
    force_char_estimate(monkeypatch)
    monkeypatch.setattr(survival_fit, "_host_estimate", host_rough)
    monkeypatch.setattr(host_uid_emit, "_host_uid_capability", uids)
    host_home = tmp_path / "home"
    host_home.mkdir()
    _state_db(host_home, [(SESSION, None, None, "acp")])
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")), hermes_home=str(host_home))
    engine.on_session_start(SESSION, platform="acp", model="test/model", context_length=WINDOW)
    engine.update_model("test/model", WINDOW, provider="openrouter")
    summaries = iter(enumerate(["UT01 UT02", ""], 1))

    def stub_summary(chunk, **kwargs):
        number, tags = next(summaries)
        text = f"Stub summary #{number} covers {tags}.\nExpand for details about: stub"
        return chunk, count_messages_tokens(chunk), text, 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", stub_summary)
    if head_identity:
        # The sole-user cell's live log (#3013): the source cache's identity for the generated head
        # differs from its raw span, so the guard must compare identities computed the same way.
        replay_identity = engine._message_replay_identity

        def diverging_identity(message, **kwargs):
            identity = replay_identity(message, **kwargs)
            content = message.get("content") if isinstance(message, dict) else None
            if (not kwargs.get("stored_row") and message.get("role") == "user" and isinstance(content, str)
                    and content.startswith(PRESERVED)):
                return ("user", "head:" + hashlib.sha256(content.encode()).hexdigest(), *identity[2:])
            return identity

        monkeypatch.setattr(engine, "_message_replay_identity", diverging_identity)
    ordinal = 0

    def stamp(rows):
        nonlocal ordinal
        result = []
        for row in rows:
            ordinal += 1
            item = dict(row)
            item.setdefault("timestamp", 1_795_236_258 + ordinal / 100)
            if uids:
                item.setdefault("message_uid", f"{ordinal:032x}")
            result.append(item)
        return result

    # Transcript compress ingress events 22, 31, 34, 66, 69, 101, 104 on ref-0.21.5.
    # The system prompt contributes to pressure and is outside every engine list.
    pressures = {2: (26_314,), 3: (28_356, 30_256), 7: (25_538, 26_377),
                 11: (25_464, 27_781) if final_attempt else (25_464,)}
    if tight_fit_host:
        pressures = TIGHT_FIT_PRESSURES[tight_fit_host]
    reports_not_found = uids or tight_fit_host == "upstream-main"
    try:
        setup = stamp([{"role": "user", "content": "[T01] user turn 1: Earlier request. end."},
                       {"role": "assistant", "content": "reply to T01: noted item 1."}])
        engine.should_compress_preflight(setup[:1])
        engine.ingest(setup)
        objective, = stamp([{"role": "user", "content": OBJECTIVE}])
        expected = [*setup, objective]
        live = list(expected)
        engine.should_compress_preflight(live)
        for step, size in enumerate([24_000, 16_000] + [7_400] * 9, 1):
            engine.update_from_response({"prompt_tokens": sum(len(row["content"]) for row in live) // 4 + 2_700,
                                         "completion_tokens": 20})
            delta = stamp(read_file_rows(step, size, reports_not_found=reports_not_found))
            expected.extend(delta)
            live.extend(delta)
            for pressure in pressures.get(step, ()):
                assert engine.should_compress(pressure)
                result = engine.compress(live, current_tokens=pressure)
                if result != live:
                    engine.on_session_end(SESSION, live)
                    engine.on_session_start(SESSION, boundary_reason="compression", old_session_id=SESSION, platform="acp")
                    live = stamp(result)
                    engine.awaiting_real_usage_after_compression = True
                    engine._verify_compaction_cleared_threshold = engine._last_compression_made_progress

        assert engine.compression_count == 2
        assert engine._last_compacted_store_id == 15
        if final_attempt:
            assert engine._ingest_cursor == 11
            assert not engine._compress_commit_proof["end_consumed"]
        else:
            assert engine._ingest_cursor_needs_reconcile
            assert engine._compress_commit_proof is None
        durable = engine._durable_commit_proof_payload()
        assert durable["last_store_id"] == 17
        if not tight_fit_host:
            assert [len(row["content"]) for row in live] == [48_368, 0, 1961, 0, 1961, 0, 1961, 0, 1961, 0,
                                                            8423 if uids else 8403]
        assert live[0]["content"].startswith(PRESERVED + OBJECTIVE)
        assert live[0]["timestamp"] != objective["timestamp"]
        if uids:
            assert live[0]["message_uid"] != objective["message_uid"]
            assert durable["emissions"][0]["engine_uid"] == live[0]["message_uid"]
        assert [row["tool_call_id"] for row in live if row["role"] == "tool"] == [
            f"call_T02_{step}_0" for step in range(6, 11)
        ]

        # Hermes re-anchors to the generated user at index 0. The finalizer applies
        # its persist override to content in place, keeping the generated stamp/UID.
        bare_objective = {**live[0], "content": OBJECTIVE}
        if restamp:
            bare_objective, = stamp([{"role": "user", "content": OBJECTIVE}])
            expected.append(bare_objective)
        reply, = stamp([{"role": "assistant", "content": "reply to T02: noted item 2."}])
        final = [bare_objective, *live[1:], reply]
        engine.ingest(final)
        engine.on_turn_complete(final)

        rows = engine._store.get_session_messages(SESSION)
        objective_ids = [row["store_id"] for row in rows if row["role"] == "user" and row["content"] == OBJECTIVE]
        assert objective_ids == ([3, 26] if restamp else [3])
        assert [row_key(engine, row) for row in rows] == [row_key(engine, row) for row in [*expected, reply]]
        if tight_fit_host:
            # B2 compares the next request's result to the durable dispatch. The latest result
            # still fits in full if the fit spends a little less of its budget on older excerpts.
            assert final[-2]["content"] == expected[-1]["content"], (
                f"{final[-2]['tool_call_id']} reached the wire with {len(final[-2]['content'])} characters"
            )

        # Byte-identical text sent again after the reply is a separate user turn.
        engine.ingest([*final, *stamp([{"role": "user", "content": OBJECTIVE}])])
        assert engine._store.get_session_count(SESSION) == len(rows) + 1
        assert engine._store.get_session_messages(SESSION)[-1]["content"] == OBJECTIVE
    finally:
        engine.shutdown()
