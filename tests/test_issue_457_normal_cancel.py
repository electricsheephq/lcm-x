"""Issue 457: existing worker fixture adapted to one normal leaf and cold retry."""

import json
import re
import socket
from collections import Counter
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent.conversation_compression import CompressionCommitFence, _run_summary_dispatch
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, tail=1):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "cancel.db"),
            fresh_tail_count=tail,
            leaf_chunk_tokens=80,
            incremental_max_depth=0,
            context_threshold=0.01,
            threshold_full_sweep_enabled=False,
            summary_prefix_target_tokens=100_000,
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start(
        "synthetic-cancel", platform="cli",
        conversation_id="synthetic-conversation", context_length=200_000,
    )
    return engine


@pytest.mark.parametrize("restart", [False, True], ids=["same-engine", "cold-engine"])
@pytest.mark.parametrize("tail", [1, 24, 32])
def test_normal_leaf_late_cancel_cold_retry(tmp_path, monkeypatch, restart, tail, end_flush=False):
    def deny_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in this synthetic fixture")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    calls = []

    def provider(**kwargs):
        calls.append(1)
        policy = str(kwargs["messages"][0].get("content") or "")
        match = re.search(r'<lcm-summary nonce="[0-9a-f]+">', policy)
        assert match is not None
        content = (
            f"{match.group(0)}\nDurable synthetic summary preserves ordered decisions, "
            "constraints, active work, and the exact recovery boundary.\n"
            "Expand for details about: synthetic cancellation qualification\n</lcm-summary>"
        )
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content)
        )])

    monkeypatch.setattr(aux, "call_llm", provider)
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"owned turn {i} " + "detail " * 60}
        for i in range(7 + tail)
    ]
    if tail > 1:
        messages[-2] = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "synthetic-tool", "type": "function",
             "function": {"name": "synthetic", "arguments": "{}"}}]}
        messages[-1] = {"role": "tool", "tool_call_id": "synthetic-tool", "content": "synthetic result"}
    original = deepcopy(messages)
    engine = _engine(tmp_path, tail)
    fence, stop = CompressionCommitFence(), Event()
    receipt = {"full_sweep": False}

    def rows():
        return list(engine._store._conn.execute("SELECT * FROM messages ORDER BY store_id"))

    def leaves():
        return [n for n in engine._dag.get_session_nodes("synthetic-cancel")
                if n.source_type == "messages"]

    def frontier():
        return engine._lifecycle.get_by_conversation(
            "synthetic-conversation"
        ).current_frontier_store_id

    def dispatch(compress_fn, current_fence, event, generation):
        return _run_summary_dispatch(
            SimpleNamespace(context_compressor=engine, session_id=engine._session_id),
            messages, compress_fn, {"current_tokens": 50_000, "force": True},
            commit_fence=current_fence, attempt_generation=generation,
            hard_cancel_event=event,
        )

    try:
        engine.ingest(messages)
        before = rows()
        receipt["before_raw"] = len(before)

        def cancel_after_compress(*args, **kwargs):
            result = engine.compress(*args, **kwargs)
            assert len(leaves()) == 1
            receipt["after_compress_raw"] = len(rows())
            receipt["committed_frontier"] = frontier()
            assert fence.try_cancel_before_commit() is True
            stop.set()
            return result

        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            dispatch(cancel_after_compress, fence, stop, 1)
        first_leaf = leaves()[0]
        first_sources = list(first_leaf.source_ids)
        receipt["after_cancel_raw"] = len(rows())
        receipt["first_leaf_sources"] = len(first_sources)
        assert rows() == before and messages == original
        assert frontier() == max(first_sources)
        if restart:
            engine.shutdown()
            engine = _engine(tmp_path, tail)
        receipt["cold_open_raw"] = len(rows())
        resumed = dispatch(engine.compress, CompressionCommitFence(), None, 2)
        source_counts = Counter(s for n in leaves() for s in n.source_ids)
        receipt.update(
            after_retry_raw=len(rows()), after_retry_frontier=frontier(),
            leaf_count=len(leaves()), unique_coverage=all(v == 1 for v in source_counts.values()),
            first_leaf_preserved=any(n.node_id == first_leaf.node_id and n.source_ids == first_sources
                                     for n in leaves()),
            original_rows_preserved=rows() == before,
            original_input_preserved=messages == original,
            returned_messages=len(resumed),
            returned_summary=any(m.get("_compressed_summary") is True for m in resumed),
            untouched_suffix=resumed[-tail:] == original[-tail:],
            duplicate_ingestion=len(rows()) - len(before), provider_stub_calls=len(calls),
        )
        print("BOUNDARY_RECEIPT=" + json.dumps(receipt, sort_keys=True))
        assert receipt["original_rows_preserved"]
        assert receipt["first_leaf_preserved"] and receipt["unique_coverage"]
        assert receipt["untouched_suffix"]
        assert receipt["returned_summary"], "Cold retry returned old raw input without the committed summary"
        assert len(calls) == 1, "Recovery must not repeat provider work"
        if end_flush:
            engine.on_session_end("synthetic-cancel", messages)
            after_end = len(rows())
            engine.on_session_start("synthetic-cancel", old_session_id="synthetic-cancel",
                boundary_reason="compression", conversation_id="synthetic-conversation",
                platform="cli", context_length=200_000)
            print("END_FLUSH_RECEIPT=" + json.dumps({"before": len(before),
                "after_end": after_end, "after_boundary": len(rows())}))
            assert after_end == len(rows()) == len(before)
        # New equal text is a new occurrence at a new cumulative position.
        continued = resumed + deepcopy(original[-2:])
        engine.ingest(continued)
        assert len(rows()) == len(before) + 2
        engine.ingest(continued)
        assert len(rows()) == len(before) + 2
    finally:
        engine.shutdown()


def test_cold_retry_original_history_end_flush(tmp_path, monkeypatch):
    test_normal_leaf_late_cancel_cold_retry(tmp_path, monkeypatch, True, 24, end_flush=True)


@pytest.mark.parametrize("fail_at", [1, 2])
def test_pending_record_rolls_back_with_leaf_and_retains_prior_commit(tmp_path, monkeypatch, fail_at):
    from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError

    engine = _engine(tmp_path)
    engine._config.threshold_full_sweep_enabled = True
    messages = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"synthetic ordered {i} " + "detail " * 60} for i in range(8)]
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue",
                        lambda chunk, **k: (chunk, 1000, "Synthetic ordered summary.", 1, 0))
    add_node = engine._dag.add_node
    attempts = []

    def fail_after_staging(node, before_commit=None):
        attempts.append(1)
        def staged(conn, node_id):
            before_commit(conn, node_id)
            if len(attempts) == fail_at:
                raise LifecyclePublicationConflictError("synthetic transaction failure")
        return add_node(node, before_commit=staged)

    monkeypatch.setattr(engine._dag, "add_node", fail_after_staging)
    try:
        engine.compress(messages, current_tokens=50_000, force=True)
        assert len(attempts) == fail_at
        nodes = engine._dag.get_session_nodes("synthetic-cancel")
        row = engine._store._conn.execute("SELECT value FROM metadata WHERE key = ?",
                                         (engine._pending_compaction_key(),)).fetchone()
        assert len(nodes) == fail_at - 1
        if fail_at == 1:
            assert row is None
            assert engine._last_compacted_store_id == 0
        else:
            record = json.loads(row[0])
            assert record["nodes"] == [nodes[0].node_id]
            assert record["frontier"] == max(nodes[0].source_ids)
            pending = engine._resume_pending_compaction(messages)
            assert pending is not None
            consumed = {position for position, _ in record["consumed"]}
            suffix = [message for i, message in enumerate(messages) if i not in consumed]
            assert pending[-len(suffix):] == suffix
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 8
    finally:
        engine.shutdown()


@pytest.mark.parametrize("change", ["extended", "altered", "binding", "corrupt", "source"])
def test_pending_recovery_rejects_changed_proof(tmp_path, monkeypatch, change):
    engine = _engine(tmp_path)
    messages = [{"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"synthetic {i} " + "detail " * 60} for i in range(8)]
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue",
                        lambda chunk, **k: (chunk, 1000,
                            "Durable synthetic summary of the ordered original messages.", 1, 0))
    try:
        engine.compress(messages, current_tokens=50_000, force=True)
        key = engine._pending_compaction_key()
        record_row = engine._store._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        assert record_row is not None
        candidate = deepcopy(messages)
        if change == "extended":
            candidate.append(deepcopy(messages[-1]))
        elif change == "altered":
            candidate[0]["content"] += " changed"
        elif change == "binding":
            engine._conversation_id = "synthetic-other-owner"
        else:
            record = json.loads(record_row[0])
            if change == "corrupt":
                record["consumed"][0][0] = len(messages)
            else:
                record["consumed"][0][1] = 999999
            engine._store._conn.execute("UPDATE metadata SET value = ? WHERE key = ?",
                                        (json.dumps(record), key))
            engine._store._conn.commit()
        count = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert engine._resume_pending_compaction(candidate) is None
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == count
    finally:
        engine.shutdown()
