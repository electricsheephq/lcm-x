"""I1/I8: a steady retry must retain the results skipped by #1637 (occ-0002)."""

from copy import deepcopy

from hermes_lcm.reconcile import _HERMES_PRUNED_TOOL_PLACEHOLDER as CLEARED
from tests.harness import engine_context
from tests.test_issue_671_stub_first_exit import tool_pair


def test_steady_retry_retains_both_sibling_results(tmp_path, monkeypatch):
    monkeypatch.setenv("LCM_IDENTITY_ANCHOR", "1")
    with engine_context(
        tmp_path, window=200_000, session_id="S", conversation_id="conversation",
        fresh_tail_count=9, leaf_chunk_tokens=400,
        large_output_externalization_enabled=False,
        large_output_active_replay_stubbing_enabled=False,
        temporal_rollups_enabled=False, embeddings_enabled=False,
        deferred_maintenance_enabled=False,
    ) as engine:
        rows = [{"role": "user", "content": "repeat", "timestamp": 10}]
        for call_id in ("c", "d"):
            rows.extend(tool_pair(call_id, "synthetic long lookup output " * 20 + call_id,
                                  tool_name="lookup"))
        engine.ingest(deepcopy(rows))
        before = engine._store.get_session_messages("S")
        host = deepcopy(rows)
        host[0]["timestamp"] = 20
        for row in host:
            if row["role"] == "tool":
                row["content"] = CLEARED
        for i in range(2):
            host.extend(tool_pair(f"following-{i}", f"new result {i}", tool_name="lookup"))

        engine.ingest(host)
        stored = engine._store.get_session_messages("S")
        assert stored[:len(before)] == before
        added = stored[len(before):]
        # Pin the two real result rows restored to GA behaviour. The separate
        # GA call-envelope deficit remains outside this targeted regression.
        assert [(row["tool_call_id"], row["content"]) for row in added
                if row["role"] == "tool"] == [
            ("c", CLEARED), ("d", CLEARED),
            ("following-0", "new result 0"), ("following-1", "new result 1"),
        ]
        assert sum(row["role"] == "user" and row["content"] == "repeat" for row in added) == 1
