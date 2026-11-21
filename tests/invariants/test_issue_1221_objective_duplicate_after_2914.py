"""I1 at the #1221 window with the #2863 host progress and persistence contracts."""

import json

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import extract_externalized_refs, load_externalized_payload
from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_2863_host_contract import ModelClock, force_char_estimate, host_rough

WINDOW = 32_768
SESSION = "objective-1221"
OBJECTIVE = "Keep the following objective verbatim.\n" + "plan " * 9600 + "end."


def read_file_rows(step, size):
    call_id = f"read-{step}"
    text = "".join(f"Record {i:04d}: datum for task step {step:02d}.\n"
                   for i in range(size // 30 + 1))[:size]
    path = f"~/objective-1221/big-{step:02d}.txt"
    content = json.dumps({"content": "\n".join(f"{i:6d}|{line}" for i, line in enumerate(text.splitlines(), 1)),
                          "path": path})
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": path})}},
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


def stub_summary(chunk, **kwargs):
    return chunk, count_messages_tokens(chunk), "The earlier turn set up the task.", 1, 1


def row_key(engine, row):
    content = row.get("content") or ""
    refs = extract_externalized_refs(content) if row["role"] == "tool" else []
    if refs:
        assert len(refs) == 1
        content = load_externalized_payload(refs[0], config=engine._config, hermes_home=engine._hermes_home)["content"]
    return row["role"], content, row.get("tool_call_id"), row.get("tool_calls") or []


def test_objective_is_stored_once_after_in_place_compaction(tmp_path, monkeypatch):
    force_char_estimate(monkeypatch)
    clock = ModelClock(monkeypatch)
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")),
                       hermes_home=str(tmp_path / "home"))
    engine.on_session_start(SESSION, platform="cli", model="probe", context_length=WINDOW)
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", stub_summary)
    try:
        setup = [{"role": "user", "content": "Earlier request."},
                 {"role": "assistant", "content": "Earlier answer."}]
        engine.should_compress_preflight(setup[:1])
        engine.ingest(setup)
        objective = {"role": "user", "content": OBJECTIVE}
        expected = setup + [objective]
        live = list(expected)
        engine.should_compress_preflight(live)
        attempts = 0
        for step, size in enumerate([24_000, 16_000] + [7_400] * 9, start=1):
            clock.tick()
            # The model reports the live rows plus its system prompt and tool schemas.
            prompt = sum(len(row.get("content") or "") for row in live) // 4 + 5_000
            verify = getattr(engine, "_verify_compaction_cleared_threshold", False)
            engine.update_from_response({"prompt_tokens": prompt, "completion_tokens": 20})
            if verify and prompt < engine.threshold_tokens:
                attempts = 0
            delta = read_file_rows(step, size)
            expected.extend(delta)
            live.extend(delta)
            pressure = prompt + 20 + host_rough(delta[1:])
            if attempts == 3 or getattr(engine, "awaiting_real_usage_after_compression", False):
                continue
            if not engine.should_compress(pressure):
                continue
            attempts += 1
            result = engine.compress(live, current_tokens=pressure)
            if result != live:
                engine.on_session_end(SESSION, live)
                engine.on_session_start(SESSION, boundary_reason="compression", old_session_id=SESSION, platform="cli")
                engine.awaiting_real_usage_after_compression = True
                engine._verify_compaction_cleared_threshold = engine._last_compression_made_progress
                live = result

        assert engine.compression_count >= 2
        assert engine._last_survival_fit
        # Hermes re-anchors the current user after compaction, then applies
        # its persist_user_message override before post_llm_call.
        current = next(row for row in reversed(live) if row["role"] == "user")
        assert OBJECTIVE in current["content"] and current["content"] != OBJECTIVE
        current["content"] = OBJECTIVE
        reply = {"role": "assistant", "content": "Finished the task."}
        engine.ingest([*live, reply])

        rows = engine._store.get_session_messages(SESSION)
        objective_ids = [row["store_id"] for row in rows if row["role"] == "user" and row["content"] == OBJECTIVE]
        assert len(objective_ids) == 1, objective_ids
        assert [row_key(engine, row) for row in rows] == [row_key(engine, row) for row in [*expected, reply]]

        # Byte-identical text sent again after the reply is a separate user turn.
        engine.ingest([*live, reply, dict(objective)])
        assert engine._store.get_session_count(SESSION) == len(rows) + 1
        assert engine._store.get_session_messages(SESSION)[-1]["content"] == OBJECTIVE
    finally:
        engine.shutdown()
