import json
import re
import sqlite3
import time

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine


def test_compression_boundary_carries_summaries_without_moving_raw_messages(tmp_path):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start(
            "parent-session",
            platform="discord",
            conversation_id="discord-thread",
            context_length=200_000,
        )
        store_ids = engine._store.append_batch(
            "parent-session",
            [
                {"role": "user", "content": "raw parent payload"},
                {"role": "assistant", "content": "raw assistant payload"},
            ],
            source="discord",
        )
        engine._dag.add_node(
            SummaryNode(
                session_id="parent-session",
                depth=0,
                summary="summary carried across compression boundary",
                token_count=8,
                source_token_count=12,
                source_ids=store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=time.time(),
                latest_at=time.time(),
                expand_hint="Expand for raw parent payload",
            )
        )
        externalized_dir = tmp_path / "externalized"
        externalized_dir.mkdir()
        payload_path = externalized_dir / "payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "kind": "ingest_payload",
                    "role": "tool",
                    "session_id": "parent-session",
                    "content": "large raw payload",
                    "created_at": time.time(),
                }
            ),
            encoding="utf-8",
        )

        engine.on_session_start(
            "child-session",
            platform="discord",
            conversation_id="discord-thread",
            context_length=200_000,
            boundary_reason="compression",
            old_session_id="parent-session",
        )

        assert engine._store.get_session_count("parent-session") == 2
        assert engine._store.get_session_count("child-session") == 0
        assert engine._dag.get_session_nodes("parent-session") == []
        child_nodes = engine._dag.get_session_nodes("child-session")
        assert len(child_nodes) == 1
        assert child_nodes[0].source_ids == store_ids
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["session_id"] == "parent-session"
    finally:
        engine.shutdown()


# --- Hermes compaction commit sequence (#483) --------------------------------
# Hermes commits a compaction as compress() -> on_session_end(sid, <compress
# input>) -> on_session_start(new_or_same_sid, boundary_reason="compression",
# old_session_id=sid), then adopts compress()'s output, optionally after
# merging consecutive user rows (agent_runtime_helpers._merge_consecutive_users).
# The driver below replays that exact call order against the real engine with a
# stub summarizer that keeps turn tags, so coverage can be checked strictly.

_CYCLES = 3


def _turn(i):
    return (
        {"role": "user", "content": f"[T{i:02d}] user turn {i}: " + ("alpha beta gamma delta " * 40)},
        {"role": "assistant", "content": f"reply to T{i:02d}: noted item {i}."},
    )


def _host_merge_consecutive_users(messages):
    """Hermes' role-alternation repair: glue consecutive string user rows."""
    merged = []
    for message in messages:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.get("role") == "user"
            and message.get("role") == "user"
            and isinstance(previous.get("content"), str)
            and isinstance(message.get("content"), str)
        ):
            previous["content"] = previous["content"] + "\n\n" + message["content"]
            continue
        merged.append(dict(message))
    return merged


def _stub_summarizer():
    calls = {"n": 0}

    def summarize(*args, **kwargs):
        calls["n"] += 1
        text = kwargs.get("text") if "text" in kwargs else (args[0] if args else "")
        tags = sorted(set(re.findall(r"\[T(\d\d)\] user", text or "")) | set(re.findall(r"U(\d\d)", text or "")))
        return (
            f"Stub summary #{calls['n']} covers " + " ".join("U" + tag for tag in tags) + ".\n"
            "Expand for details about: stub",
            1,
        )

    return summarize


def _run_host_commit_sequence(tmp_path, monkeypatch, *, in_place, merge, tail, restart_after=-1):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    db_path = tmp_path / "lcm.db"

    def config():
        return LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=tail,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        )

    engine = LCMEngine(config=config(), hermes_home=str(tmp_path / "home"))
    sid = "S0"
    engine.on_session_start(sid, platform="acp", context_length=200_000)
    host: list = []
    turn = 0
    statuses = []
    context_missing = []
    try:
        for cycle in range(1, _CYCLES + 1):
            for _ in range(6 if cycle > 1 else 12):
                turn += 1
                user, reply = _turn(turn)
                host.append(user)
                engine.ingest(host)
                host.append(reply)
                engine.ingest(host)
            turn += 1
            host.append(_turn(turn)[0])  # the compacting turn's user row, then preflight
            engine.ingest(host)
            pre = list(host)
            compressed = engine.compress(list(host), force=True)
            statuses.append(engine._last_compression_status)
            engine.on_session_end(sid, pre)  # commit_memory_session (both modes)
            new_sid = sid if in_place else f"S{cycle}"
            engine.on_session_start(
                new_sid, boundary_reason="compression", old_session_id=sid, platform="acp"
            )
            sid = new_sid
            host = list(compressed)
            if merge:
                host = _host_merge_consecutive_users(host)
            host.append(_turn(turn)[1])  # reply to the compacting turn
            engine.ingest(host)
            if cycle == restart_after:  # process restart; the host resumes the same session
                engine.shutdown()
                engine = LCMEngine(config=config(), hermes_home=str(tmp_path / "home"))
                engine.on_session_start(sid, platform="acp", context_length=200_000)
                engine.ingest(host)
            view = " ".join(str(message.get("content")) for message in host)
            view_tags = set(re.findall(r"\[T(\d\d)\] user", view)) | set(re.findall(r"U(\d\d)", view))
            context_missing.append(sorted({f"{i:02d}" for i in range(1, turn + 1)} - view_tags))
    finally:
        engine.shutdown()

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    finally:
        conn.close()
    user_tags = [m.group(1) for role, content in rows if role == "user" for m in [re.match(r"\[T(\d\d)\] user", content or "")] if m]
    reply_tags = [m.group(1) for role, content in rows if role == "assistant" for m in [re.match(r"reply to T(\d\d)", content or "")] if m]
    expected = {f"{i:02d}" for i in range(1, turn + 1)}
    return {
        "statuses": statuses,
        "duplicate_rows": len(rows) - len(set(rows)),
        "scaffold_rows_stored": sum(1 for _role, content in rows if re.search(r"Summary \(d\d+, node \d+\)", content or "")),
        "missing_user_turns": sorted(expected - set(user_tags)),
        "missing_replies": sorted(expected - set(reply_tags)),
        "context_missing": context_missing,
    }


def _assert_clean_commit_sequence(result):
    assert result["statuses"] == ["compacted"] * _CYCLES, result
    assert result["duplicate_rows"] == 0, result
    assert result["scaffold_rows_stored"] == 0, result
    assert result["missing_user_turns"] == [], result
    assert result["missing_replies"] == [], result
    assert result["context_missing"] == [[]] * _CYCLES, result



# A host that glues LCM's user-role summary to a user-leading fresh tail
# (tail=7) needs verified-carrier recognition; restart/resume needs the
# durable proof. Those cases join the matrix with the commits that fix them.
_HOST_MERGE_SEAMS = [(False, 6), (False, 7), (True, 6)]


@pytest.mark.parametrize("merge, tail", _HOST_MERGE_SEAMS)
def test_hermes_in_place_commit_sequence_publishes_repeatedly(tmp_path, monkeypatch, merge, tail):
    """compress -> end(sid, pre) -> start(sid, compression, old=sid), x3 (#483)."""
    result = _run_host_commit_sequence(tmp_path, monkeypatch, in_place=True, merge=merge, tail=tail)
    _assert_clean_commit_sequence(result)


@pytest.mark.parametrize("merge, tail", _HOST_MERGE_SEAMS)
def test_hermes_rotation_commit_sequence_stores_every_turn(tmp_path, monkeypatch, merge, tail):
    """compress -> end(S0, pre) -> start(S1, compression, old=S0): the stale
    end-ingest must not re-store the tail in the parent, and the child must not
    skip its first turns (#483 rotation variant)."""
    result = _run_host_commit_sequence(tmp_path, monkeypatch, in_place=False, merge=merge, tail=tail)
    _assert_clean_commit_sequence(result)


def _compacted_engine(tmp_path, monkeypatch, *, turns=12, tail=6):
    """An engine bound to S0 that just ran compress() over `turns` turns plus the
    compacting turn's user row. Returns (engine, pre, compressed)."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=tail,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    host: list = []
    for i in range(1, turns + 1):
        host.extend(_turn(i))
        engine.ingest(host)
    host.append(_turn(turns + 1)[0])
    engine.ingest(host)
    pre = list(host)
    compressed = engine.compress(list(host), force=True)
    assert engine._last_compression_status == "compacted"
    return engine, pre, compressed


def _row_count(engine, session_id="S0"):
    return engine._store.get_session_count(session_id)


def test_compaction_commit_end_is_not_a_session_end(tmp_path, monkeypatch):
    """C1: end(sid, <exact compress input>) neither re-ingests nor finalizes."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        rows = _row_count(engine)
        frontier = engine._lifecycle.get_by_session("S0").current_frontier_store_id
        assert frontier > 0

        engine.on_session_end("S0", pre)

        state = engine._lifecycle.get_by_session("S0")
        assert state.current_session_id == "S0"
        assert state.current_frontier_store_id == frontier
        assert _row_count(engine) == rows
        assert engine._ingest_cursor == len(compressed)
    finally:
        engine.shutdown()


def test_compaction_commit_end_classification_is_one_shot(tmp_path, monkeypatch):
    """C1: a second identical end is a real end and finalizes the session."""
    engine, pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_end("S0", pre)
        state = engine._lifecycle.get_by_session("S0")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S0"
    finally:
        engine.shutdown()


def test_real_session_end_after_compress_still_finalizes(tmp_path, monkeypatch):
    """C1: an end whose list differs from the compress input (/new, exit) is a
    real end (#247): it finalizes and stores the new turn exactly once."""
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        rows = _row_count(engine)
        final = list(compressed) + [_turn(13)[1]]
        engine.on_session_end("S0", final)
        state = engine._lifecycle.get_by_session("S0")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S0"
        assert _row_count(engine) == rows + 1
    finally:
        engine.shutdown()


def test_compress_records_no_commit_proof_when_cursor_is_not_trusted(tmp_path, monkeypatch):
    """C1: no proof for a reconcile-pending cursor, a bypassed session or a
    no-progress compress (an end with that list must stay a real end)."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        assert engine._compress_commit_proof is not None
        engine._ingest_cursor_needs_reconcile = True
        engine._record_compress_commit_proof(pre, compressed)
        assert engine._compress_commit_proof is None
        engine._ingest_cursor_needs_reconcile = False

        engine._record_compress_commit_proof(compressed, list(compressed))
        assert engine._compress_commit_proof is None

        monkeypatch.setattr(engine, "_bypasses_lcm_context_management", lambda: True)
        engine._record_compress_commit_proof(pre, compressed)
        assert engine._compress_commit_proof is None
    finally:
        engine.shutdown()


def test_in_place_compression_start_keeps_binding_frontier_and_cursor(tmp_path, monkeypatch):
    """C2: start(sid, compression, old=sid) continues the same LCM segment."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        frontier = engine._lifecycle.get_by_session("S0").current_frontier_store_id
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        state = engine._lifecycle.get_by_session("S0")
        assert state.current_session_id == "S0"
        assert state.current_frontier_store_id == frontier
        assert engine._last_compacted_store_id == frontier
        assert engine._ingest_cursor == len(compressed)
    finally:
        engine.shutdown()


def test_first_ingest_after_compaction_reconciles_a_rewritten_prefix(tmp_path, monkeypatch):
    """C4: the positional cursor is trusted only when the host prefix is the
    compress() output; an unrecognized rewrite goes to reconcile, not a skip."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        rewritten = [{"role": "user", "content": "host-rewritten head"}] + list(compressed[1:])
        rewritten.append(_turn(13)[1])
        engine._last_ingest_reconciliation = {"action": "none", "reason": "not run"}
        rows = _row_count(engine)
        engine.ingest(rewritten)
        assert engine._last_ingest_reconciliation["action"] != "none"
        stored = [row["content"] for row in engine._store.get_session_tail("S0", limit=5)]
        assert _turn(13)[1]["content"] in stored  # never a silent skip
        assert _row_count(engine) > rows
        assert engine._compress_commit_proof is not None  # kept until a prefix is proven
    finally:
        engine.shutdown()


def test_first_ingest_after_compaction_trusts_the_exact_output(tmp_path, monkeypatch):
    """C4: the host adopted the output unchanged -> only the new row is stored."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        rows = _row_count(engine)
        engine.ingest(list(compressed) + [_turn(13)[1]])
        assert _row_count(engine) == rows + 1
        assert engine._compress_commit_proof is None
    finally:
        engine.shutdown()


def test_host_that_keeps_the_compress_input_resumes_after_it(tmp_path, monkeypatch):
    """C4b: the host committed, then refused or rolled back the output and kept
    the input -> no row of the input is re-stored and the new turn is kept."""
    engine, pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        rows = _row_count(engine)
        engine.ingest(list(pre) + [_turn(13)[1]])
        assert _row_count(engine) == rows + 1
        assert engine._ingest_cursor == len(pre) + 1
        assert engine._compress_commit_proof is None
    finally:
        engine.shutdown()


class TestBindSessionFrontierAfterOwnFinalize:
    """C3: bind_session resumes a frontier only for the session that finalized it."""

    def _store(self, tmp_path):
        from hermes_lcm.lifecycle_state import LifecycleStateStore

        store = LifecycleStateStore(tmp_path / "lifecycle.db")
        store.bind_session("S0", conversation_id="c1")
        store.advance_frontier("c1", "S0", 42)
        store.finalize_session("c1", "S0", 42)
        return store

    def test_same_session_rebind_restores_its_frontier(self, tmp_path):
        store = self._store(tmp_path)
        try:
            assert store.bind_session("S0", conversation_id="c1").current_frontier_store_id == 42
        finally:
            store.close()

    def test_reset_after_finalize_starts_at_zero(self, tmp_path):
        store = self._store(tmp_path)
        try:
            finalized_at = store.get_by_conversation("c1").last_finalized_at
            store._conn.execute(
                "UPDATE lcm_lifecycle_state SET last_reset_at = ? WHERE conversation_id = 'c1'",
                (finalized_at + 1,),
            )
            store._conn.commit()
            assert store.bind_session("S0", conversation_id="c1").current_frontier_store_id == 0
        finally:
            store.close()

    def test_other_session_never_inherits_the_frontier(self, tmp_path):
        store = self._store(tmp_path)
        try:
            assert store.bind_session("S1", conversation_id="c1").current_frontier_store_id == 0
        finally:
            store.close()
