"""Pinned timestamp/UID and host-repair shapes through replay and compaction."""
from copy import deepcopy

import pytest
from agent.context_engine import ContextEngine

from tests.harness import ScriptedSummarizer, make_engine, pin_char_counter
from tests.test_issue_1385_r5_host_merge_collapse import _identity


@pytest.mark.parametrize("form", ["floor", "r34", "upstream-uid"])
@pytest.mark.parametrize("shape", ["tool-group", "user-merge", "orphan-prune", "retry"])
def test_stamped_stopped_rows_survive_repair_cold_replay_and_compaction(tmp_path, monkeypatch, form, shape):
    monkeypatch.setattr(ContextEngine, "on_turn_complete", lambda *a, **k: None, raising=False)
    pin_char_counter(monkeypatch)
    ScriptedSummarizer("healthy").install(monkeypatch)
    sequence = 0

    def stamp(role, content, **extra):
        nonlocal sequence
        sequence += 1
        row = dict(role=role, content=content, timestamp=float(1000 + sequence), **extra)
        if form == "upstream-uid":
            row["message_uid"] = f"uid-{sequence}"
        return row

    warm = [stamp("user", "warm question"), stamp("assistant", "warm answer")]
    stopped = [stamp("user", "stopped request")]
    if shape != "user-merge":
        stopped += [stamp("assistant", "", tool_calls=[{
            "id": "call-stopped", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"fixture.txt"}'}}])]
    if shape in {"tool-group", "retry"}:
        stopped += [stamp("tool", "fixture result", tool_call_id="call-stopped"),
                    stamp("assistant", "Operation interrupted.")]
    history = warm + stopped
    engine = make_engine(tmp_path, window=128000, fresh_tail_count=2, leaf_chunk_tokens=16)

    def assert_ledger(expected):
        stored = engine._store.get_session_messages("S")
        assert list(map(_identity, stored)) == list(map(_identity, expected))

    try:
        engine.ingest(warm)
        engine.on_turn_complete(deepcopy(history), interrupted=True)
        engine.on_turn_complete(deepcopy(history), interrupted=True)
        # Only the open user tail is deferred; closed tool-stop shapes persist.
        assert_ledger(warm if shape == "user-merge" else history)
        next_user = stamp("user", "stopped request" if shape == "retry" else "next request")
        answer = stamp("assistant", "next answer")
        if shape in {"user-merge", "orphan-prune"}:
            # Hermes prunes the unanswered call, then merges adjacent users.
            merged = deepcopy(stopped[0])
            merged["content"] += "\n\n" + next_user["content"]
            if form == "r34":
                merged["_merged_turn_prefix"] = stopped[0]["content"] + "\n\n"
            elif form == "upstream-uid":
                merged["_merged_turn_prefix"] = stopped[0]["content"]
                merged["_absorbed_message_uids"] = [next_user["message_uid"]]
            current = warm + [merged, answer]
            expected = current if shape == "user-merge" else history + [next_user, answer]
        else:
            current = (warm if shape == "retry" else history) + [next_user, answer]
            # /retry is a newly stamped user occurrence, as in completed-turn #1179.
            expected = history + [next_user, answer]
        engine.ingest(deepcopy(current))
        engine.ingest(deepcopy(current))
        assert_ledger(expected)
        engine.shutdown()
        engine = make_engine(tmp_path, window=128000, fresh_tail_count=2, leaf_chunk_tokens=16)
        engine.ingest(deepcopy(current))
        assert_ledger(expected)
        view = engine.compress(deepcopy(current), current_tokens=engine.threshold_tokens + 1, force=True)
        assert engine._last_compression_status == "compacted"
        assert_ledger(expected)
        after = [stamp("user", "after compaction"), stamp("assistant", "after answer")]
        engine.ingest(deepcopy(view + after))
        engine.ingest(deepcopy(view + after))
        assert_ledger(expected + after)
        engine.on_session_end("S", deepcopy(view + after))
        assert_ledger(expected + after)
    finally:
        engine.shutdown()
