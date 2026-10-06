"""Plugin-only replay of a host persist override at an emitted scaffold slot."""
import time

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


@pytest.fixture(params=["summary", "objective"])
def compacted_head(tmp_path, request):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    prompt = {"role": "user", "content": "Original sole-user objective", "timestamp": 1000.0}
    tail = [
        {"role": "assistant", "content": "Working", "timestamp": 1001.0,
         "tool_calls": [{"id": "call_923", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "content": "Read result", "tool_call_id": "call_923", "timestamp": 1002.0},
    ]
    older_tail = [
        {"role": "assistant", "content": "Earlier work", "timestamp": 1000.1},
        {"role": "tool", "content": "Earlier result", "tool_call_id": "call_earlier", "timestamp": 1000.2},
    ]
    before = [prompt, *older_tail, *tail]
    engine.ingest(before)
    prompt_id = engine._store.get_session_messages("S0")[0]["store_id"]
    engine._last_compacted_store_id = prompt_id
    engine._dag.add_node(SummaryNode(
        session_id="S0", depth=0, summary="The user requested work.", token_count=6,
        source_token_count=8, source_ids=[prompt_id], source_type="messages",
        created_at=time.time(), earliest_at=time.time(), latest_at=time.time(), expand_hint="original request",
    ))
    engine._pending_context_anchor_messages = before if request.param == "objective" else None
    returned = engine._assemble_context(None, tail, include_lcm_note=False)
    engine._pending_context_anchor_messages = None
    engine._ingest_cursor = len(returned)
    engine._ingest_cursor_needs_reconcile = False
    engine._last_compression_status = "compacted"
    engine._record_compress_commit_proof(before, returned)
    proof = engine._durable_commit_proof_payload()
    assert proof["emissions"][0]["kind"] == request.param
    assert proof["emissions"][0]["output_occurrence"]["index"] == 0
    try:
        yield engine, prompt, returned
    finally:
        engine.shutdown()


def _unstamped(prompt):
    return {key: value for key, value in prompt.items() if key != "timestamp"}


def _users(engine):
    return [row for row in engine._store.get_session_messages("S0") if row["role"] == "user"]


def test_host_head_rewrite_is_replay_then_reply_is_new(compacted_head):
    engine, prompt, returned = compacted_head
    host = [_unstamped(prompt), *returned[1:]]
    count = engine._store.get_session_count("S0")
    engine.ingest(host)
    assert len(_users(engine)) == 1
    assert engine._store.get_session_count("S0") == count
    host.append({"role": "assistant", "content": "Final reply"})
    engine.ingest(host)
    assert len(_users(engine)) == 1
    assert engine._store.get_session_count("S0") == count + 1
    assert engine._store.get_session_tail("S0", limit=1)[0]["content"] == "Final reply"


def test_same_unstamped_prompt_at_non_scaffold_position_is_new(compacted_head):
    engine, prompt, returned = compacted_head
    engine.ingest([*returned, _unstamped(prompt)])
    assert len(_users(engine)) == 2
    assert _users(engine)[-1]["observed_at"] is None


@pytest.mark.parametrize("stamp,expected_users", [(1000.0, 1), (2000.0, 2)])
def test_stamped_rewrite_keeps_existing_identity_anchor_behavior(compacted_head, stamp, expected_users):
    engine, prompt, returned = compacted_head
    engine.ingest([{**prompt, "timestamp": stamp}, *returned[1:]])
    assert len(_users(engine)) == expected_users


@pytest.mark.parametrize("mismatch", ["payload", "tool_calls", "frontier", "session", "proof"])
def test_unproven_head_rewrite_keeps_duplicate_over_loss(compacted_head, mismatch):
    engine, prompt, returned = compacted_head
    rewritten = _unstamped(prompt)
    if mismatch == "payload":
        rewritten["content"] += " with a new instruction"
    elif mismatch == "tool_calls":
        rewritten["tool_calls"] = [{"id": "new", "function": {"name": "read", "arguments": "{}"}}]
    elif mismatch == "frontier":
        engine._last_compacted_store_id = 0
    elif mismatch == "session":
        engine._store._conn.execute("UPDATE messages SET session_id = 'other' WHERE role = 'user'")
    else:
        engine._store._conn.execute("DELETE FROM metadata WHERE key LIKE 'compaction_commit_proof%'")
        engine._compress_commit_proof = None
        engine._last_emission_descriptors = None
        engine._ingest_cursor_needs_reconcile = True
    engine.ingest([rewritten, *returned[1:]])
    expected = 1 if mismatch == "session" else 2
    assert len(_users(engine)) == expected
    assert _users(engine)[-1]["content"] == rewritten["content"]
