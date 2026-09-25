import json
import re
import sqlite3
import time

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecyclePublicationConflictError
from hermes_lcm.reconcile import _commit_proof_identity_digest


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


def _run_host_commit_sequence(
    tmp_path, monkeypatch, *, in_place, merge, tail, restart_after=-1, restart_before_first_ingest=-1
):
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
    frontier_regressions = []
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
            published_frontier = engine._last_compacted_store_id
            engine.on_session_end(sid, pre)  # commit_memory_session (both modes)
            new_sid = sid if in_place else f"S{cycle}"
            engine.on_session_start(
                new_sid, boundary_reason="compression", old_session_id=sid, platform="acp"
            )
            sid = new_sid
            state = engine._lifecycle.get_by_session(sid)
            if state is None or state.current_session_id != sid or state.current_frontier_store_id < published_frontier:
                frontier_regressions.append((cycle, published_frontier, state and state.current_frontier_store_id))
            host = list(compressed)
            if merge:
                host = _host_merge_consecutive_users(host)
            if cycle == restart_before_first_ingest:  # restart right after the boundary
                engine.shutdown()
                engine = LCMEngine(config=config(), hermes_home=str(tmp_path / "home"))
                engine.on_session_start(sid, platform="acp", context_length=200_000)
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
        "frontier_regressions": frontier_regressions,
    }


def _assert_clean_commit_sequence(result):
    assert result["statuses"] == ["compacted"] * _CYCLES, result
    assert result["duplicate_rows"] == 0, result
    assert result["scaffold_rows_stored"] == 0, result
    assert result["missing_user_turns"] == [], result
    assert result["missing_replies"] == [], result
    assert result["context_missing"] == [[]] * _CYCLES, result
    assert result["frontier_regressions"] == [], result


# tail=6 leaves an assistant-leading fresh tail; tail=7 a user-leading one, so
# a merging host glues LCM's user-role summary to the first tail user row.
_IN_PLACE_SEAMS = [(False, 6), (False, 7), (True, 6), (True, 7)]
_ROTATION_SEAMS = _IN_PLACE_SEAMS


@pytest.mark.parametrize("merge, tail", _IN_PLACE_SEAMS)
def test_hermes_in_place_commit_sequence_publishes_repeatedly(tmp_path, monkeypatch, merge, tail):
    """compress -> end(sid, pre) -> start(sid, compression, old=sid), x3 (#483)."""
    result = _run_host_commit_sequence(tmp_path, monkeypatch, in_place=True, merge=merge, tail=tail)
    _assert_clean_commit_sequence(result)


@pytest.mark.parametrize("merge, tail", _ROTATION_SEAMS)
def test_hermes_rotation_commit_sequence_stores_every_turn(tmp_path, monkeypatch, merge, tail):
    """compress -> end(S0, pre) -> start(S1, compression, old=S0): the stale
    end-ingest must not re-store the tail in the parent, and the child must not
    skip its first turns (#483 rotation variant)."""
    result = _run_host_commit_sequence(tmp_path, monkeypatch, in_place=False, merge=merge, tail=tail)
    _assert_clean_commit_sequence(result)


@pytest.mark.parametrize("restart_after", [1, 2])
@pytest.mark.parametrize("merge, tail", _IN_PLACE_SEAMS)
@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_resume_after_compaction_does_not_restore_tail(
    tmp_path, monkeypatch, in_place, merge, tail, restart_after
):
    """A fresh process resuming the session after compaction 1 or 2 re-indexes
    the host list through the durable commit proof instead of re-storing the
    fresh tail (C6), in place and across a rotation (C7)."""
    result = _run_host_commit_sequence(
        tmp_path, monkeypatch, in_place=in_place, merge=merge, tail=tail, restart_after=restart_after
    )
    _assert_clean_commit_sequence(result)


@pytest.mark.parametrize("restart_cycle", [1, 2])
@pytest.mark.parametrize("merge, tail", _IN_PLACE_SEAMS)
@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_restart_before_first_post_compaction_ingest(tmp_path, monkeypatch, in_place, merge, tail, restart_cycle):
    """#484 round 1 item 2: a process restart between the boundary start and the
    first ingest. A rotation child is still empty then, so its durable proof must be
    consulted before the empty-session path stores the resumed snapshot from 0."""
    result = _run_host_commit_sequence(
        tmp_path, monkeypatch, in_place=in_place, merge=merge, tail=tail,
        restart_before_first_ingest=restart_cycle,
    )
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


def test_compaction_commit_end_finalizes_without_reingest(tmp_path, monkeypatch):
    """C1: end(sid, <exact compress input>) does not re-ingest, and still finalizes
    the session with its published frontier (#484 round 2 item 17)."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        rows = _row_count(engine)
        frontier = engine._lifecycle.get_by_session("S0").current_frontier_store_id
        assert frontier > 0

        engine.on_session_end("S0", pre)

        state = engine._lifecycle.get_by_session("S0")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S0"
        assert state.last_finalized_frontier_store_id == frontier
        assert engine._compress_commit_proof["end_consumed"] is True
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


@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_commit_proof_is_consulted_once_after_a_reconcile(tmp_path, monkeypatch, in_place):
    """#484 item 11i (#487 note): after the first post-compaction ingest sends a
    rewritten prefix to reconcile, a reconciled cursor that happens to equal
    len(output) must not re-arm the kept proof on the next ingest."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    child = "S0" if in_place else "S1"
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
        host = [{"role": "user", "content": "host-rewritten head"}] + list(compressed[1:])
        engine.ingest(host)  # proof consulted: prefix rewritten -> reconcile
        assert engine._ingest_cursor == len(compressed)
        rows = engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        for i in range(13, 16):
            host.append(_turn(i)[1] if i == 13 else _turn(i)[0])
            engine.ingest(host)
        # The first reconcile's bounded duplicates (#259) are already counted in rows.
        assert engine._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == rows + 3
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

    def test_finalize_records_only_the_finalizing_sessions_frontier(self, tmp_path):
        """#484 round 2 item 16: A finalizes at 0, B finalizes at 42, A rebinds and
        finalizes at 0 again; A's next rebind must not resume B's 42."""
        from hermes_lcm.lifecycle_state import LifecycleStateStore

        store = LifecycleStateStore(tmp_path / "lifecycle.db")
        try:
            store.bind_session("A", conversation_id="c1")
            store.finalize_session("c1", "A", 0)
            store.bind_session("B", conversation_id="c1")
            store.advance_frontier("c1", "B", 42)
            store.finalize_session("c1", "B", 42)
            assert store.bind_session("A", conversation_id="c1").current_frontier_store_id == 0
            state = store.finalize_session("c1", "A", 0)
            assert state.last_finalized_session_id == "A"
            assert state.last_finalized_frontier_store_id == 0
            assert store.bind_session("A", conversation_id="c1").current_frontier_store_id == 0
        finally:
            store.close()

    def test_same_session_refinalize_keeps_its_own_frontier(self, tmp_path):
        store = self._store(tmp_path)
        try:
            store.bind_session("S0", conversation_id="c1")
            assert store.finalize_session("c1", "S0", 0).last_finalized_frontier_store_id == 42
            assert store.bind_session("S0", conversation_id="c1").current_frontier_store_id == 42
        finally:
            store.close()


def _summary_message(compressed):
    for message in compressed:
        if re.search(r"Summary \(d\d+, node \d+\)\]", str(message.get("content"))):
            return message
    raise AssertionError("compress() output carries no LCM summary")


def _summary_carrier_fixture(tmp_path):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "carrier.db"),
            fresh_tail_count=3,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    compacted = {"role": "assistant", "content": "older compacted answer"}
    compacted_id = engine._store.append("S0", compacted)
    engine._last_compacted_store_id = compacted_id
    engine._dag.add_node(
        SummaryNode(
            session_id="S0",
            depth=0,
            summary="Carrier summary.",
            token_count=3,
            source_token_count=5,
            source_ids=[compacted_id],
            source_type="messages",
            created_at=time.time(),
            earliest_at=time.time(),
            latest_at=time.time(),
            expand_hint="carrier summary",
        )
    )
    tail = [
        {"role": "user", "content": "historical user row"},
        {"role": "assistant", "content": "historical assistant row"},
        {"role": "user", "content": "current prompt"},
    ]
    tail_ids = engine._store.append_batch("S0", tail)
    return engine, compacted, tail, tail_ids


def test_assembly_emits_verified_summary_carrier_with_original_row_identity(tmp_path):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        assembled = engine._assemble_context(None, tail)
        carrier = assembled[0]

        assert engine._generated_context_carrier_remainder(carrier) == tail[0]["content"]
        assert engine._proof_replay_identity(carrier) == engine._proof_replay_identity(tail[0])
        assert assembled[1:] == tail[1:]

        engine._ingest_cursor = len(assembled)
        engine._ingest_cursor_needs_reconcile = False
        engine._record_compress_commit_proof([compacted, *tail], assembled)
        proof = engine._compress_commit_proof
        assert proof is not None
        assert proof["output"][0] == engine._proof_replay_identity(carrier, strip_carrier=False)  # #488 F4: full
        assert proof["output_effective"][0] == engine._proof_replay_identity(tail[0])
    finally:
        engine.shutdown()


def test_assembly_never_folds_the_only_tail_user_current_prompt(tmp_path):
    engine, _compacted, _tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    current = {"role": "user", "content": "current prompt"}
    try:
        assembled = engine._assemble_context(None, [current])
        assert len(assembled) == 2
        assert engine._generated_context_carrier_remainder(assembled[0]) is None
        assert assembled[1] == current
    finally:
        engine.shutdown()


def test_assembly_does_not_fold_list_content_user_row(tmp_path):
    engine, _compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    multimodal = {
        "role": "user",
        "content": [{"type": "text", "text": "historical multimodal row"}],
    }
    try:
        assembled = engine._assemble_context(None, [multimodal, *tail[1:]])
        assert engine._generated_context_carrier_remainder(assembled[0]) is None
        assert assembled[1] == multimodal
    finally:
        engine.shutdown()


def test_assembly_with_system_message_keeps_summary_and_tail_separate(tmp_path):
    engine, _compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        assembled = engine._assemble_context(
            {"role": "system", "content": "system prompt"},
            tail,
            include_lcm_note=False,
        )
        assert assembled[0] == {"role": "system", "content": "system prompt"}
        assert engine._generated_context_carrier_remainder(assembled[1]) is None
        assert assembled[2:] == tail
    finally:
        engine.shutdown()


def test_summary_carrier_maps_to_original_store_id_and_carry_range(tmp_path):
    engine, compacted, tail, tail_ids = _summary_carrier_fixture(tmp_path)
    try:
        assembled = engine._assemble_context(None, tail)
        carrier = assembled[0]
        assert engine._get_store_ids_for_messages(assembled)[0] == tail_ids[0]

        engine._ingest_cursor = len(assembled)
        engine._ingest_cursor_needs_reconcile = False
        engine._record_compress_commit_proof([compacted, *tail], assembled)
        assert engine._compress_commit_proof is not None
        assert engine._compress_commit_proof["carry_ranges"] == [
            ("S0", tail_ids[0] - 1, tail_ids[-1])
        ]
        assert engine._generated_context_carrier_remainder(carrier) == tail[0]["content"]
    finally:
        engine.shutdown()


def test_compress_commit_proof_coalesces_contiguous_carry_rows(tmp_path):
    engine = LCMEngine(
        config=_config(tmp_path),
        hermes_home=str(tmp_path / "home"),
    )
    messages = [
        {"role": "user", "content": f"message {index}"}
        for index in range(4)
    ]
    try:
        engine.on_session_start(
            "S0",
            platform="acp",
            conversation_id="carry-coalescing",
            context_length=200_000,
        )
        engine.ingest(messages)
        result = messages[1:]
        engine._ingest_cursor = len(result)
        engine._ingest_cursor_needs_reconcile = False
        engine._record_compress_commit_proof(messages, result)
        assert engine._compress_commit_proof["carry_ranges"] == [("S0", 1, 4)]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", ["none", "before-adoption", "after-adoption"])
def test_native_adoption_with_summary_carrier_survives_restart_without_loss(
    tmp_path, restart
):
    engine, compacted, tail, _tail_ids = _summary_carrier_fixture(tmp_path)
    adopted = [
        {"role": "user", "content": "Synthetic native summary."},
        *engine._assemble_context(None, tail),
    ]
    engine._last_compression_status = "host_native"
    engine._last_native_summary_index = 0
    engine._ingest_cursor = len(adopted)
    engine._ingest_cursor_needs_reconcile = False
    engine._record_compress_commit_proof([compacted, *tail], adopted)
    assert engine._compress_commit_proof is not None
    assert engine._compress_commit_proof["native"] is True
    assert engine._generated_context_carrier_remainder(adopted[1]) == tail[0]["content"]
    assert engine._durable_commit_proof_payload() is not None

    def resumed_engine():
        return LCMEngine(
            config=LCMConfig(
                database_path=str(tmp_path / "carrier.db"),
                fresh_tail_count=3,
                large_output_externalization_path=str(tmp_path / "externalized"),
            ),
            hermes_home=str(tmp_path / "home"),
        )

    if restart == "before-adoption":
        engine.shutdown()
        engine = resumed_engine()
        engine.on_session_start("S0", platform="acp")
        assert engine._durable_commit_proof_payload() is not None
    elif restart == "after-adoption":
        engine.ingest(adopted)
        engine.shutdown()
        engine = resumed_engine()
        engine.on_session_start("S0", platform="acp")
        assert engine._durable_commit_proof_payload() is not None

    appended = {"role": "assistant", "content": "post-adoption reply"}
    try:
        engine.ingest([*adopted, appended])
        stored = engine._store.get_session_messages("S0")
        assert len(stored) == 5
        assert [row["content"] for row in stored] == [
            compacted["content"],
            *(row["content"] for row in tail),
            appended["content"],
        ]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("separator", ["\n\n", "\n\n---\n\n"])
def test_host_merged_summary_carrier_is_identified_by_its_glued_row(tmp_path, monkeypatch, separator):
    """C5: summary + separator + real user row is a carrier, not scaffold, and
    its replay identity is the glued row's identity."""
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        summary = _summary_message(compressed)["content"]
        glued = {"role": "user", "content": "the real next user row"}
        carrier = {"role": "user", "content": summary + separator + glued["content"]}
        assert engine._generated_context_carrier_remainder(carrier) == glued["content"]
        assert not engine._is_replayed_context_scaffold_message(carrier)
        assert engine._message_replay_identity(carrier) == engine._message_replay_identity(glued)
    finally:
        engine.shutdown()


def test_pure_lcm_summary_stays_scaffold(tmp_path, monkeypatch):
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        summary = {"role": "user", "content": _summary_message(compressed)["content"]}
        assert engine._generated_context_carrier_remainder(summary) is None
        assert engine._is_replayed_context_scaffold_message(summary)
    finally:
        engine.shutdown()


def test_unverified_summary_prefix_is_never_stripped(tmp_path, monkeypatch):
    """C5: an edited summary text or a wrong node id does not verify against the
    DAG, so the row keeps its full identity (only LCM-rendered text is stripped)."""
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        summary = _summary_message(compressed)["content"]
        node_id = int(re.search(r"node (\d+)\)\]", summary).group(1))
        edited = summary.replace("Stub summary", "Edited summary", 1)
        wrong_node = summary.replace(f"node {node_id})]", f"node {node_id + 999})]", 1)
        for forged in (edited, wrong_node):
            carrier = {"role": "user", "content": forged + "\n\nthe real next user row"}
            assert engine._generated_context_carrier_remainder(carrier) is None
            identity = engine._message_replay_identity(carrier)
            assert identity != engine._message_replay_identity({"role": "user", "content": "the real next user row"})
    finally:
        engine.shutdown()


def test_another_sessions_summary_part_is_not_a_carrier(tmp_path, monkeypatch):
    """#484 item 11j: a summary part verifies only against a node of the bound
    session. A row in session B quoting session A's exact rendered part plus text
    is real content, so a restart reconcile stores it (duplicates at worst)."""
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    part = _summary_message(compressed)["content"]
    system = {"role": "system", "content": "You are concise."}
    try:
        engine.on_session_start("SB", platform="acp", conversation_id="conv-b", context_length=200_000)
        engine.ingest([system, {"role": "user", "content": "same"}])
        assert engine._store.get_session_count("SB") == 2
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("SB", platform="acp", conversation_id="conv-b", context_length=200_000)
        quoted = {"role": "user", "content": part + "\n\nsame"}
        resumed.ingest([system, quoted])
        stored = [c for (c,) in resumed._store._conn.execute("SELECT content FROM messages WHERE session_id = 'SB'")]
        assert stored.count(quoted["content"]) == 1  # stored, not skipped (a re-stored system row is #259)
        assert resumed._generated_context_carrier_remainder(quoted) is None
    finally:
        resumed.shutdown()


@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_scaffold_only_compaction_restart_before_first_ingest_stores_no_scaffold(tmp_path, monkeypatch, in_place):
    """#484 item 11k: with fresh_tail_count=0 the compress() output can be scaffold
    only (empty effective proof). A restart before the first ingest must not store
    the generated scaffolds as raw rows, and the new user row is stored once."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
    child = "S0" if in_place else "S1"
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host: list = [{"role": "system", "content": "You are concise."}]
        for i in range(1, 13):
            host.extend(_turn(i))
            engine.ingest(host)
        pre = list(host)
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        assert all(engine._is_verified_replay_scaffold_message(m) for m in compressed), [
            (m.get("role"), str(m.get("content"))[:60]) for m in compressed
        ]
        engine.on_session_end("S0", pre)
        engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start(child, platform="acp", context_length=200_000)
        new_user = {"role": "user", "content": "NEW-1147 first turn after the compaction"}
        resumed.ingest(list(compressed) + [new_user])
        rows = resumed._store._conn.execute("SELECT content FROM messages").fetchall()
        assert sum(1 for (c,) in rows if re.search(r"Summary \(d\d+, node \d+\)", c or "")) == 0
        assert sum(1 for (c,) in rows if "Lossless Context Management" in (c or "")) == 0
        assert sum(1 for (c,) in rows if c == new_user["content"]) == 1
    finally:
        resumed.shutdown()


def _scaffold_only_compaction(tmp_path, monkeypatch, *, in_place):
    """fresh_tail_count=0 compaction, commit, boundary start, then a restart before
    the first ingest. Returns (resumed engine, compressed output)."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
    child = "S0" if in_place else "S1"
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host: list = [{"role": "system", "content": "You are concise."}]
        for i in range(1, 13):
            host.extend(_turn(i))
            engine.ingest(host)
        pre = list(host)
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        engine.on_session_end("S0", pre)
        engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path, fresh_tail_count=0), hermes_home=str(tmp_path / "home"))
    resumed.on_session_start(child, platform="acp", context_length=200_000)
    return resumed, compressed


def _raw_scaffold_rows(engine):
    rows = engine._store._conn.execute("SELECT content FROM messages").fetchall()
    return sum(
        1
        for (c,) in rows
        if "Lossless Context Management" in (c or "")
        or (c or "").startswith(("[Current user objective", "[Recent Summary", "[Session Arc Summary", "[Durable Summary"))
    )


# #484 item 11l (a strict xfail until #488): the store matcher no longer skips a
# trailing row that is a verified own summary by shape; only a descriptor does.
@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_scaffold_only_proof_does_not_skip_a_new_row_quoting_the_summary(tmp_path, monkeypatch, in_place):
    """#484 item 11l: the scaffold-only durable proof is bound to the emitted rows,
    not to their shape. A new user row whose whole content is the session's own
    rendered summary block is stored once; no generated scaffold is stored raw."""
    resumed, compressed = _scaffold_only_compaction(tmp_path, monkeypatch, in_place=in_place)
    try:
        block = compressed[-1]["content"]
        block = block[block.index("[Recent Summary"):] if "[Recent Summary" in block else block
        quoted = {"role": "user", "content": block}
        assert resumed._is_verified_replay_scaffold_message(quoted)
        scaffold_before = _raw_scaffold_rows(resumed)
        resumed.ingest(list(compressed) + [quoted])
        stored = [c for (c,) in resumed._store._conn.execute("SELECT content FROM messages")]
        assert stored.count(block) == 1
        assert _raw_scaffold_rows(resumed) == scaffold_before + 1  # only the quoted row
    finally:
        resumed.shutdown()


@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_scaffold_only_proof_refuses_a_host_list_without_the_note(tmp_path, monkeypatch, in_place):
    """#484 item 11l negative: the host dropped the generated note row, so the list
    is not the emitted sequence: the proof returns None and nothing is lost."""
    resumed, compressed = _scaffold_only_compaction(tmp_path, monkeypatch, in_place=in_place)
    try:
        host = list(compressed[1:]) + [{"role": "user", "content": "NEW-2210 after the compaction"}]
        assert resumed._cursor_from_durable_commit_proof(host) is None
        resumed.ingest(host)
        stored = [c for (c,) in resumed._store._conn.execute("SELECT content FROM messages")]
        assert stored.count("NEW-2210 after the compaction") == 1
    finally:
        resumed.shutdown()


def test_list_content_row_is_not_a_carrier(tmp_path, monkeypatch):
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        summary = _summary_message(compressed)["content"]
        row = {"role": "user", "content": [{"type": "text", "text": summary + "\n\nreal row"}]}
        assert engine._generated_context_carrier_remainder(row) is None
    finally:
        engine.shutdown()


def _durable_commit_proof(engine, session_id):
    return engine._store.read_metadata_json(f"compaction_commit_proof:{session_id}")


def test_durable_commit_proof_is_written_only_for_a_published_compaction(tmp_path, monkeypatch):
    """C6: a published compaction persists its output proof; a compress() that
    did not publish leaves no durable proof behind."""
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        payload = _durable_commit_proof(engine, "S0")
        assert payload["version"] == 4
        assert payload["hermes_home"] == str(tmp_path / "home")
        assert len(payload["effective_sha256"]) == len(engine._compress_commit_proof["output_effective"])
        assert payload["last_store_id"] > 0

        engine._store.write_metadata_json(["compaction_commit_proof:S0"], "null")
        engine._last_compression_status = "noop"
        extended = list(compressed) + [_turn(13)[1]]
        engine._ingest_cursor = len(extended)
        engine._record_compress_commit_proof(compressed, extended)
        assert engine._compress_commit_proof["published"] is False
        assert _durable_commit_proof(engine, "S0") is None
    finally:
        engine.shutdown()


def test_durable_commit_proof_only_extends_the_reconciled_cursor(tmp_path, monkeypatch):
    """C6: the durable proof may extend the matcher's cursor, never shrink it: a
    matcher that already proved the whole snapshot keeps its result and label."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host: list = []
        for i in range(1, 4):
            host.extend(_turn(i))
            engine.ingest(host)
        rows = _row_count(engine)
        monkeypatch.setattr(engine, "_cursor_from_durable_commit_proof", lambda messages: 1)
        engine._ingest_cursor = 0
        engine._ingest_cursor_needs_reconcile = True
        engine.ingest(host)
        assert _row_count(engine) == rows
        assert engine._ingest_cursor == len(host)
        assert engine._last_ingest_reconciliation["reason"] == "replayed durable tail"
    finally:
        engine.shutdown()


def test_rotation_moves_the_commit_proof_to_the_child(tmp_path, monkeypatch):
    """C7: after a commit-classified end, start(S1, compression, old=S0) re-keys
    the proof to the child and persists it with no child rows yet."""
    engine, pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start("S1", boundary_reason="compression", old_session_id="S0", platform="acp")
        assert engine._compress_commit_proof["session_id"] == "S1"
        assert engine._compress_commit_proof["input"] is None
        child = _durable_commit_proof(engine, "S1")
        assert child["last_store_id"] == 0
        assert child["effective_sha256"] == _durable_commit_proof(engine, "S0")["effective_sha256"]
    finally:
        engine.shutdown()


def test_proofless_rotation_child_gets_no_carry_authority(tmp_path, monkeypatch):
    engine, pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine._compress_commit_proof = None
        engine.on_session_end("S0", pre)
        engine.on_session_start(
            "S1", boundary_reason="compression", old_session_id="S0", platform="acp"
        )
        assert engine._load_compression_carry_ranges() == []
    finally:
        engine.shutdown()


def test_rotation_inherits_only_open_grandparent_carry_and_reset_clears_it(
    tmp_path, monkeypatch
):
    engine, pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        frontier = engine._last_compacted_store_id
        proof = engine._compress_commit_proof
        proof["carry_ranges"] = [
            ("closed-grandparent", 0, frontier),
            ("open-grandparent", frontier - 1, frontier + 2),
            ("open-grandparent", frontier + 2, frontier + 5),
        ]
        engine._persist_compress_commit_proof(proof)
        engine.on_session_end("S0", pre)
        engine.on_session_start(
            "S1", boundary_reason="compression", old_session_id="S0", platform="acp"
        )
        inherited = engine._load_compression_carry_ranges()
        assert inherited == [("open-grandparent", frontier, frontier + 5)]

        engine.on_session_reset()
        assert engine._compress_commit_proof is None
        assert engine._load_compression_carry_ranges() == []
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart_child", [False, True], ids=["live", "restart"])
def test_rotation_child_can_compact_only_parent_carried_rows(
    tmp_path, monkeypatch, restart_child
):
    """#495: a proof-backed child may publish the parent's still-open rows."""
    engine, pre, compressed = _compacted_engine(
        tmp_path, monkeypatch, turns=12, tail=6
    )
    try:
        frontier = engine._last_compacted_store_id
        carried_ids = {
            int(row[0])
            for row in engine._store._conn.execute(
                "SELECT store_id FROM messages "
                "WHERE session_id = 'S0' AND store_id > ? ORDER BY store_id",
                (frontier,),
            )
        }
        assert carried_ids
        engine.on_session_end("S0", pre)
        engine.on_session_start(
            "S1", boundary_reason="compression", old_session_id="S0", platform="acp"
        )
        engine._config.fresh_tail_count = 2
        engine.protect_last_n = 2
        if restart_child:
            engine.shutdown()
            engine = LCMEngine(
                config=LCMConfig(
                    database_path=str(tmp_path / "lcm.db"),
                    fresh_tail_count=2,
                    leaf_chunk_tokens=400,
                    large_output_externalization_path=str(tmp_path / "externalized"),
                ),
                hermes_home=str(tmp_path / "home"),
            )
            engine.on_session_start("S1", platform="acp", context_length=200_000)

        host = list(compressed)
        host.extend(_turn(14))
        engine.ingest(host)
        existing_node_ids = {
            node.node_id for node in engine._dag.get_session_nodes("S1")
        }

        engine.compress(list(host), force=True)

        assert engine._last_compression_status == "compacted"
        new_nodes = [
            node
            for node in engine._dag.get_session_nodes("S1")
            if node.node_id not in existing_node_ids
        ]
        assert len(new_nodes) == 1
        assert new_nodes[0].source_ids
        assert set(new_nodes[0].source_ids) <= carried_ids
    finally:
        engine.shutdown()


@pytest.mark.parametrize("dropped", ["empty-parent-row", "ignored-carried-row"])
def test_rotation_child_publication_uses_only_rows_the_proof_carried(
    tmp_path, monkeypatch, dropped
):
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=8,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host = []
        marker = "DROP_ME carried parent row"
        for i in range(1, 13):
            user, assistant = _turn(i)
            if i == 10:
                assistant = {
                    "role": "assistant",
                    "content": "" if dropped == "empty-parent-row" else marker,
                }
            host.extend([user, assistant])
            engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        pre = list(host)
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"

        engine.on_session_end("S0", pre)
        engine.on_session_start(
            "S1", boundary_reason="compression", old_session_id="S0", platform="acp"
        )
        if dropped == "ignored-carried-row":
            class DropPattern:
                def search(self, text, timeout=None):
                    del timeout
                    return object() if marker in str(text) else None

            engine._compiled_ignore_message_patterns = [DropPattern()]
        engine._config.fresh_tail_count = 2
        engine.protect_last_n = 2
        child = list(compressed)
        if dropped == "ignored-carried-row":
            child = [
                message
                for message in child
                if marker not in str(message.get("content") or "")
            ]
        child.extend(_turn(14))
        engine.ingest(child)
        result = engine.compress(list(child), force=True)

        assert engine._last_compression_status == "compacted"
        assert len(result) < len(child)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("path", ["fail-open", "sanitized"])
def test_compress_returns_input_object_when_only_host_metadata_differs(
    tmp_path, monkeypatch, path
):
    """#495: metadata-only rewrites are no progress to the Hermes host."""
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    messages = [
        {
            "role": "user",
            "content": "unchanged payload",
            "_row_id": 7,
            "_db_persisted": True,
            "timestamp": 123.0,
        }
    ]
    public_rows = [{"role": "user", "content": "unchanged payload"}]

    def compress_impl(*_args, **_kwargs):
        if path == "fail-open":
            return engine._fail_open_after_publication_failure(
                public_rows,
                LifecyclePublicationConflictError("injected conflict"),
                compress_started=time.perf_counter(),
                threshold_full_sweep_active=False,
                recovery_assembly_cap=None,
                leaf_passes=0,
            )
        engine._last_compression_status = "sanitized"
        return engine._sanitize_active_context_messages(
            public_rows,
            insert_missing_tool_stubs=False,
        )

    monkeypatch.setattr(engine, "_compress_impl", compress_impl)
    try:
        result = engine.compress(messages)
        if path == "fail-open":
            assert engine._last_compression_status == "error"
            assert result is public_rows
        else:
            assert result is messages
    finally:
        engine.shutdown()


def test_durable_commit_proof_checks_every_row_after_the_proof_across_pages(tmp_path, monkeypatch):
    """#484 round 1 item 3: the post-proof row check must page through every row,
    not stop at the store's default page (simulated with a 3-row page)."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        host = list(compressed) + [_turn(13)[1]]
        engine.ingest(host)
        for i in range(14, 18):
            host.extend(_turn(i))
            engine.ingest(host)
    finally:
        engine.shutdown()

    resumed = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=6,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        store = resumed._store
        paged = store.get_session_messages_after

        def small_pages(session_id, after_store_id=0, limit=3):
            return paged(session_id, after_store_id=after_store_id, limit=min(limit, 3))

        monkeypatch.setattr(store, "get_session_messages_after", small_pages)
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        rows = _row_count(resumed)
        host.append(_turn(18)[0])
        resumed.ingest(host)
        assert _row_count(resumed) == rows + 1
        stored = resumed._store._conn.execute("SELECT role, content FROM messages").fetchall()
        assert len(stored) == len(set(stored))
    finally:
        resumed.shutdown()


def test_profile_rebind_clears_the_commit_proof(tmp_path, monkeypatch):
    """#484 round 1 item 4: a same-id session under another Hermes home is a new
    profile; its end with the same list is a real end and must finalize."""
    engine, pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        assert engine._compress_commit_proof is not None
        engine.on_session_start("S0", platform="acp", context_length=200_000, hermes_home=str(tmp_path / "other-home"))
        assert engine._compress_commit_proof is None
        engine.on_session_end("S0", pre)
        state = engine._lifecycle.get_by_session("S0")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S0"
    finally:
        engine.shutdown()


def test_conversation_change_clears_the_commit_proof(tmp_path, monkeypatch):
    """#484 item 11c: a reused engine rebinding the same session id to another
    conversation must not consume the old conversation's commit proof; the end is a
    real end and finalizes in the new conversation."""
    engine, pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        old_conversation = engine._conversation_id
        assert engine._compress_commit_proof is not None
        engine.on_session_start("S0", platform="acp", conversation_id="other-conversation", context_length=200_000)
        assert engine._conversation_id == "other-conversation" != old_conversation
        engine.on_session_end("S0", pre)
        assert engine._compress_commit_proof is None or not engine._compress_commit_proof.get("end_consumed")
        state = engine._lifecycle.get_by_conversation("other-conversation")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S0"
    finally:
        engine.shutdown()


def test_durable_commit_proof_is_bound_to_its_conversation(tmp_path, monkeypatch):
    """#484 item 11c: the durable proof records its conversation and is not used
    after a restart that binds the same session id to another conversation."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    host = list(compressed) + [_turn(13)[1]]
    try:
        assert _durable_commit_proof(engine, "S0")["conversation_id"] == engine._conversation_id
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S0", platform="acp", conversation_id="other-conversation", context_length=200_000)
        assert resumed._cursor_from_durable_commit_proof(host) is None
    finally:
        resumed.shutdown()


def _as_rc3_durable_proof(engine, compressed):
    """Rewrite the durable proof the way rc3 wrote it: version 2, exact identities."""
    payload = _durable_commit_proof(engine, "S0")
    payload["version"] = 2
    payload["effective_sha256"] = [
        _commit_proof_identity_digest(engine._message_replay_identity(m))
        for m in compressed
        if not engine._is_replayed_context_scaffold_message(m)
    ]
    engine._store.write_metadata_json(["compaction_commit_proof:S0"], json.dumps(payload, sort_keys=True))
    return payload


def test_rc3_durable_commit_proof_is_still_honoured_after_upgrade(tmp_path, monkeypatch):
    """#498 P1: an rc3 (version-2) durable proof, verified with the exact identity
    it was hashed with, still re-indexes an in-place commit after a restart: the
    reply is the only new row, and the next compaction publishes (no conflict)."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        _as_rc3_durable_proof(engine, compressed)
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        rows = _row_count(resumed)
        host = list(compressed) + [_turn(13)[1]]
        resumed.ingest(host)
        assert _row_count(resumed) == rows + 1
        for i in range(14, 20):
            host.extend(_turn(i))
            resumed.ingest(host)
        host.append(_turn(20)[0])
        resumed.ingest(host)
        resumed.compress(list(host), force=True)
        assert resumed._last_compression_status == "compacted", resumed._last_compression_noop_reason
        stored = resumed._store._conn.execute("SELECT role, content FROM messages").fetchall()
        assert len(stored) == len(set(stored))
    finally:
        resumed.shutdown()


def test_rc3_durable_commit_proof_stays_exact_for_user_whitespace(tmp_path, monkeypatch):
    """A version-2 proof hashed exact identities, so it never proves a user row
    that differs only by whitespace (the cursor reconciles instead)."""
    engine, _pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    padded = [dict(m) for m in compressed]
    padded[-1]["content"] += "\n"
    try:
        assert engine._cursor_from_durable_commit_proof(padded) is not None  # version 3: proof-bound tolerance
        _as_rc3_durable_proof(engine, compressed)
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        assert resumed._cursor_from_durable_commit_proof(list(compressed)) is not None
        assert resumed._cursor_from_durable_commit_proof(padded) is None
    finally:
        resumed.shutdown()


@pytest.mark.parametrize(("stored", "replayed"), [("retry", " retry\n"), ("", "  ")], ids=["retry", "empty"])
def test_unbound_store_reconciliation_keeps_exact_user_identity(tmp_path, stored, replayed):
    """#498 P0: with no commit proof binding the position, a user row that differs
    from the stored one only by whitespace is a new exchange, not replay."""
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        engine.ingest([{"role": "user", "content": stored}, {"role": "assistant", "content": "OK"}])
        assert _row_count(engine) == 2
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        resumed.ingest([{"role": "user", "content": replayed}, {"role": "assistant", "content": "OK"}])
        assert _row_count(resumed) == 4
    finally:
        resumed.shutdown()


def test_host_trim_of_the_compacted_user_row_after_commit_stores_no_duplicates(tmp_path, monkeypatch):
    """Hermes ACP persists prompt.strip(): at turn end it rewrites the adopted user
    dict in place, after the same-turn compaction stored and proved the raw row.
    The next ingest keeps the proof and stores only the reply."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host: list = []
        for i in range(1, 13):
            host.extend(_turn(i))
            engine.ingest(host)
        host.append({"role": "user", "content": _turn(13)[0]["content"] + "\n"})
        engine.ingest(host)
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        engine.on_session_end("S0", host)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        rows = _row_count(engine)
        assert compressed[-1]["content"].endswith("\n")
        compressed[-1]["content"] = compressed[-1]["content"].strip()  # finalize_turn's rewrite
        engine.ingest(list(compressed) + [_turn(13)[1]])
        assert _row_count(engine) == rows + 1
        stored = engine._store._conn.execute("SELECT role, content FROM messages").fetchall()
        normalized = [(role, (content or "").rstrip()) for role, content in stored]
        assert len(normalized) == len(set(normalized))
    finally:
        engine.shutdown()


def test_ordinary_session_reset_clears_the_commit_proof(tmp_path, monkeypatch):
    engine, _pre, _compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_reset()
        assert engine._compress_commit_proof is None
    finally:
        engine.shutdown()


class _IterationFails(list):
    def __iter__(self):
        raise RuntimeError("injected identity failure")


@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_proof_creation_failure_loses_nothing(tmp_path, monkeypatch, in_place):
    """#484 round 1 item 5, round 2 item 14: if the commit proof cannot be built,
    the commit end is a real end; the in-place or rotation start must reconcile, so
    no turn is lost and the published frontier survives (duplicates are the
    bounded worst case)."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=6,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    original = engine._record_compress_commit_proof
    monkeypatch.setattr(
        engine, "_record_compress_commit_proof", lambda messages, result: original(_IterationFails(messages), result)
    )
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host: list = []
        for i in range(1, 13):
            host.extend(_turn(i))
            engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        pre = list(host)
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        assert engine._compress_commit_proof is None
        frontier = engine._lifecycle.get_by_session("S0").current_frontier_store_id
        engine.on_session_end("S0", pre)
        child = "S0" if in_place else "S1"
        engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
        engine.ingest(list(compressed) + [_turn(13)[1]])
        stored = [content for (content,) in engine._store._conn.execute("SELECT content FROM messages")]
        assert stored.count(_turn(13)[1]["content"]) == 1
        for i in range(1, 14):
            assert _turn(i)[0]["content"] in stored
        assert engine._lifecycle.get_by_session(child).current_frontier_store_id >= frontier
    finally:
        engine.shutdown()


def test_durable_proof_write_failure_loses_nothing_across_restart(tmp_path, monkeypatch):
    """#484 round 1 item 5: without the durable proof a resumed process falls back to
    content reconciliation: no turn is lost and the frontier survives."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        engine._store.write_metadata_json(["compaction_commit_proof:S0"], "null")  # as if the write had failed
        host = list(compressed) + [_turn(13)[1]]
        engine.ingest(host)
        frontier = engine._lifecycle.get_by_session("S0").current_frontier_store_id
    finally:
        engine.shutdown()
    resumed = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=6,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        host.append(_turn(14)[0])
        resumed.ingest(host)
        stored = [content for (content,) in resumed._store._conn.execute("SELECT content FROM messages")]
        for i in range(1, 15):
            assert _turn(i)[0]["content"] in stored
        assert _turn(13)[1]["content"] in stored
        assert resumed._lifecycle.get_by_session("S0").current_frontier_store_id >= frontier
    finally:
        resumed.shutdown()


def test_child_proof_write_failure_then_restart_before_first_ingest_loses_nothing(tmp_path, monkeypatch):
    """#484 round 2 item 14: the rotation re-keys the proof to the child, but the
    child's durable copy is lost; a restart before the child's first ingest must
    still store the reply to the compacting turn (duplicates are the bounded
    worst case)."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start("S1", boundary_reason="compression", old_session_id="S0", platform="acp")
        engine._store.write_metadata_json(["compaction_commit_proof:S1"], "null")  # as if the write had failed
    finally:
        engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S1", platform="acp", context_length=200_000)
        resumed.ingest(list(compressed) + [_turn(13)[1]])
        stored = [content for (content,) in resumed._store._conn.execute("SELECT content FROM messages")]
        assert stored.count(_turn(13)[1]["content"]) == 1
        for i in range(1, 14):
            assert _turn(i)[0]["content"] in stored
    finally:
        resumed.shutdown()


def test_metadata_write_exception_is_contained(tmp_path, monkeypatch):
    """#484 round 1 item 5: a raising write_metadata_json does not fail compress()
    and keeps the process-local proof, so the in-process commit still classifies."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "lcm.db"),
            fresh_tail_count=6,
            leaf_chunk_tokens=400,
            large_output_externalization_path=str(tmp_path / "externalized"),
        ),
        hermes_home=str(tmp_path / "home"),
    )
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host: list = []
        for i in range(1, 13):
            host.extend(_turn(i))
            engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        pre = list(host)
        real_write = engine._store.write_metadata_json

        def failing_write(keys, serialized, **kwargs):
            if any(str(key).startswith("compaction_commit_proof:") for key in keys):
                raise RuntimeError("injected metadata write failure")
            return real_write(keys, serialized, **kwargs)

        monkeypatch.setattr(engine._store, "write_metadata_json", failing_write)
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        assert engine._compress_commit_proof is not None
        rows = _row_count(engine)
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        engine.ingest(list(compressed) + [_turn(13)[1]])
        assert _row_count(engine) == rows + 1
    finally:
        engine.shutdown()


def _config(tmp_path, **overrides):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=6,
        leaf_chunk_tokens=400,
        large_output_externalization_path=str(tmp_path / "externalized"),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _forged_carrier_host(compressed, forgery):
    summary_message = _summary_message(compressed)
    summary = summary_message["content"]
    node_id = int(re.search(r"node (\d+)\)\]", summary).group(1))
    forged = (
        summary.replace("Stub summary", "Edited summary", 1)
        if forgery == "edited-summary"
        else summary.replace(f"node {node_id})]", f"node {node_id + 999})]", 1)
    )
    fresh = "FRESH-8841 the user typed this after the compaction"
    host = [{"role": "user", "content": forged + "\n\n" + fresh}] + [
        m for m in compressed if m is not summary_message
    ] + [_turn(13)[1]]
    return host, fresh


@pytest.mark.parametrize("forgery", ["edited-summary", "wrong-node-id"])
def test_commit_proof_matchers_treat_unverified_summary_text_as_content(tmp_path, monkeypatch, forgery):
    """#484 round 1 item 8: in the commit-proof matchers (C4 remap, C6 durable) a
    summary-shaped row that fails DAG verification is real content, so neither
    proof can advance the cursor past the fresh user text glued behind it."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    try:
        host, _fresh = _forged_carrier_host(compressed, forgery)
        assert engine._remap_cursor_through_host_merge(host, engine._compress_commit_proof) is None
        assert engine._cursor_from_durable_commit_proof(host) is None
        # #484 round 2 item 13: every reconciliation path uses the verified predicate.
        assert engine._is_replayed_context_scaffold_message(host[0])  # compaction/provenance shape test
        assert not engine._is_verified_replay_scaffold_message(host[0])
        assert not engine._is_verified_replay_scaffold_message(
            {"role": "user", "content": host[0]["content"].split("\n\nFRESH")[0]}
        )
        assert engine._is_verified_replay_scaffold_message(_summary_message(compressed))
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["in-process", "after-restart"])
@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
@pytest.mark.parametrize("forgery", ["edited-summary", "wrong-node-id"])
def test_unverified_summary_prefix_does_not_hide_fresh_user_text(tmp_path, monkeypatch, forgery, in_place, restart):
    """End to end (#484 round 1 item 8, round 2 item 13): the fresh user text glued
    behind an unverified summary prefix is stored, in place and across a rotation."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    host, fresh = _forged_carrier_host(compressed, forgery)
    child = "S0" if in_place else "S1"
    try:
        engine.on_session_end("S0", pre)
        engine.on_session_start(child, boundary_reason="compression", old_session_id="S0", platform="acp")
        if restart:
            engine.shutdown()
            engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
            engine.on_session_start(child, platform="acp", context_length=200_000)
        engine.ingest(host)
        stored = [content for (content,) in engine._store._conn.execute("SELECT content FROM messages")]
        assert any(fresh in (content or "") for content in stored)
        assert stored.count(_turn(13)[1]["content"]) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", [False, True], ids=["in-process", "after-restart"])
def test_lossy_password_identity_never_advances_the_proof_cursor(tmp_path, monkeypatch, restart):
    """#484 round 1 item 9: password_assignment placeholders carry no digest, so two
    different same-length values share one replay identity. The commit-proof
    matchers must not treat that identity as proof (synthetic values only)."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    sensitive = {
        "sensitive_patterns_enabled": True,
        "sensitive_patterns": ["password_assignment"],
    }
    engine = LCMEngine(config=_config(tmp_path, **sensitive), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000)
        host: list = []
        for i in range(1, 12):
            host.extend(_turn(i))
            engine.ingest(host)
        host.extend(
            [
                {"role": "user", "content": "[T12] set password=SYNTHaaaa1111 for the test rig"},
                {"role": "assistant", "content": "reply to T12: rig configured."},
            ]
        )
        engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        pre = list(host)
        compressed = engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        engine.on_session_end("S0", pre)
        engine.on_session_start("S0", boundary_reason="compression", old_session_id="S0", platform="acp")
        if restart:
            engine.ingest(list(compressed) + [_turn(13)[1]])
            engine.shutdown()
            engine = LCMEngine(config=_config(tmp_path, **sensitive), hermes_home=str(tmp_path / "home"))
            engine.on_session_start("S0", platform="acp", context_length=200_000)
            changed = list(compressed) + [_turn(13)[1], _turn(14)[0]]
        else:
            changed = list(compressed) + [_turn(13)[1]]
        changed = [
            dict(m, content="[T12] set password=SYNTHbbbb2222 for the test rig")
            if "[T12] set password" in str(m.get("content"))
            else m
            for m in changed
        ]
        assert any("[T12] set password=SYNTHbbbb2222" in str(m.get("content")) for m in changed)

        def placeholder_rows():
            return engine._store._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE content LIKE '[T12] set password=%'"
            ).fetchone()[0]

        before = placeholder_rows()
        engine.ingest(changed)
        assert placeholder_rows() == before + 1  # the changed occurrence is stored, not skipped
    finally:
        engine.shutdown()


@pytest.mark.parametrize("fresh_process", [False, True], ids=["same-process", "fresh-process"])
def test_cancelled_commit_then_exit_finalizes_the_session(tmp_path, monkeypatch, fresh_process):
    """#484 round 1 item 10, round 2 item 17: compress(I) -> O, the host cancels and
    keeps I, the user exits and end(S, I) matches the unconsumed proof. The end is
    not re-ingested (every row of I is durable) but S is finalized; the next session
    in the same conversation binds cleanly, stores its turns once and finalizes."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
    try:
        engine.on_session_start("S0", platform="acp", conversation_id="conv", context_length=200_000)
        host: list = []
        for i in range(1, 13):
            host.extend(_turn(i))
            engine.ingest(host)
        host.append(_turn(13)[0])
        engine.ingest(host)
        pre = list(host)
        engine.compress(list(host), force=True)
        assert engine._last_compression_status == "compacted"
        rows_s0 = engine._store.get_session_count("S0")
        frontier = engine._lifecycle.get_by_conversation("conv").current_frontier_store_id
        engine.on_session_end("S0", pre)  # classified as a commit: no re-ingest
        assert engine._compress_commit_proof["end_consumed"] is True
        assert engine._store.get_session_count("S0") == rows_s0
        state = engine._lifecycle.get_by_conversation("conv")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S0"
        assert state.last_finalized_frontier_store_id == frontier
        if fresh_process:
            engine.shutdown()
            engine = LCMEngine(config=_config(tmp_path), hermes_home=str(tmp_path / "home"))
        engine.on_session_start("S1", platform="acp", conversation_id="conv", context_length=200_000)
        state = engine._lifecycle.get_by_conversation("conv")
        assert state.current_session_id == "S1"
        assert state.current_frontier_store_id == 0  # a new session starts fresh (#247)
        assert engine._compress_commit_proof is None
        s1 = [
            {"role": "user", "content": "S1 first question after the exit"},
            {"role": "assistant", "content": "S1 first answer"},
        ]
        engine.ingest(s1)
        engine.ingest(s1)
        assert engine._store.get_session_count("S1") == 2
        engine.on_session_end("S1", s1)
        state = engine._lifecycle.get_by_conversation("conv")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S1"
        assert engine._store.get_session_count("S0") == rows_s0
        stored = engine._store._conn.execute("SELECT session_id, role, content FROM messages").fetchall()
        assert len(stored) == len(set(stored))
    finally:
        engine.shutdown()


def _password_session(tmp_path, monkeypatch, *, password_turn=True):
    """S0 compacted with a synthetic password in the fresh tail; returns (engine, pre, compressed)."""
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", _stub_summarizer())
    sensitive = {"sensitive_patterns_enabled": True, "sensitive_patterns": ["password_assignment"]}
    engine = LCMEngine(config=_config(tmp_path, **sensitive), hermes_home=str(tmp_path / "home"))
    engine.on_session_start("S0", platform="acp", context_length=200_000)
    host: list = []
    for i in range(1, 12):
        host.extend(_turn(i))
        engine.ingest(host)
    host.extend(
        [
            {"role": "user", "content": "[T12] set password=SYNTHaaaa1111 for the test rig"}
            if password_turn
            else _turn(12)[0],
            {"role": "assistant", "content": "reply to T12: rig configured."},
        ]
    )
    engine.ingest(host)
    host.append(_turn(13)[0])
    engine.ingest(host)
    pre = list(host)
    compressed = engine.compress(list(host), force=True)
    assert engine._last_compression_status == "compacted"
    return engine, pre, compressed


def _changed_password_end(engine, pre):
    changed = [
        dict(m, content="[T12] set password=SYNTHbbbb2222 for the test rig")
        if "[T12] set password" in str(m.get("content"))
        else m
        for m in pre
    ]
    rows = engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE content LIKE '[T12] set password=%'"
    ).fetchone()[0]
    engine.on_session_end("S0", changed)
    return rows, engine._store._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE content LIKE '[T12] set password=%'"
    ).fetchone()[0]


def test_end_with_a_changed_same_length_password_is_a_real_end(tmp_path, monkeypatch):
    """#484 round 2 item 11: C1 compares the RAW end list with the RAW compress()
    input, so a same-length password change is a different identity: the end is a
    real end and the lifecycle finalizes (synthetic values only)."""
    engine, pre, _compressed = _password_session(tmp_path, monkeypatch)
    try:
        assert not any(
            "[LCM sensitive redaction" in identity[1] for identity in engine._compress_commit_proof["input"]
        )
        _changed_password_end(engine, pre)
        state = engine._lifecycle.get_by_session("S0")
        assert state.current_session_id is None
        assert state.last_finalized_session_id == "S0"
    finally:
        engine.shutdown()


def test_end_with_a_changed_same_length_password_stores_the_changed_row(tmp_path, monkeypatch):
    """#484 item 11b: the real end reconciles its redacted list; the store matcher
    ('replayed durable tail') must not skip past a digest-less placeholder, so the
    changed occurrence is stored (duplicates at worst, never loss)."""
    engine, pre, _compressed = _password_session(tmp_path, monkeypatch)
    try:
        before, after = _changed_password_end(engine, pre)
        assert after == before + 1
    finally:
        engine.shutdown()


def test_restart_with_a_changed_same_length_password_stores_the_changed_row(tmp_path, monkeypatch):
    """#484 item 11b, restart sibling: a resumed process reconciles the host list
    against the store; a changed same-length password row is stored, not skipped."""
    engine, pre, _compressed = _password_session(tmp_path, monkeypatch)
    sensitive = {"sensitive_patterns_enabled": True, "sensitive_patterns": ["password_assignment"]}
    engine.shutdown()
    resumed = LCMEngine(config=_config(tmp_path, **sensitive), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        changed = [
            dict(m, content="[T12] set password=SYNTHbbbb2222 for the test rig")
            if "[T12] set password" in str(m.get("content"))
            else m
            for m in pre
        ]

        def password_rows():
            return resumed._store._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE content LIKE '[T12] set password=%'"
            ).fetchone()[0]

        before = password_rows()
        resumed.ingest(changed)
        assert password_rows() == before + 1
    finally:
        resumed.shutdown()


def test_password_free_compaction_still_classifies_its_commit(tmp_path, monkeypatch):
    engine, pre, _compressed = _password_session(tmp_path, monkeypatch, password_turn=False)
    try:
        assert engine._compress_commit_proof is not None
        rows = _row_count(engine)
        engine.on_session_end("S0", pre)
        assert engine._compress_commit_proof["end_consumed"] is True
        assert _row_count(engine) == rows
    finally:
        engine.shutdown()


@pytest.mark.parametrize("retain_depth", [None, -1], ids=["default-retention", "keep-all-nodes"])
def test_durable_proof_from_before_a_reset_is_ignored(tmp_path, monkeypatch, retain_depth):
    """#484 round 2 item 12: a durable proof written before a lifecycle reset of the
    same session id must not advance the cursor after a restart."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    if retain_depth is not None:
        engine._config.new_session_retain_depth = retain_depth
    host = list(compressed) + [_turn(13)[1]]
    try:
        assert engine._store.read_metadata_json("compaction_commit_proof:S0") is not None
        engine.on_session_reset()
    finally:
        engine.shutdown()
    overrides = {} if retain_depth is None else {"new_session_retain_depth": retain_depth}
    resumed = LCMEngine(config=_config(tmp_path, **overrides), hermes_home=str(tmp_path / "home"))
    try:
        resumed.on_session_start("S0", platform="acp", context_length=200_000)
        assert resumed._cursor_from_durable_commit_proof(host) is None
        resumed.ingest(host)
        assert resumed._last_ingest_reconciliation["reason"] != "replayed proven post-compaction continuation"
    finally:
        resumed.shutdown()


def test_durable_proof_does_not_cross_hermes_homes_on_a_shared_database(tmp_path, monkeypatch):
    """#484 round 2 item 12, profile part: with an explicitly configured shared
    database_path, a rebind to another Hermes home keeps the same store, so the
    durable proof of a same-id session from the other home must not be used."""
    engine, pre, compressed = _compacted_engine(tmp_path, monkeypatch)
    host = list(compressed) + [_turn(13)[1]]
    try:
        engine.on_session_start("S0", platform="acp", context_length=200_000, hermes_home=str(tmp_path / "other-home"))
        assert engine._store.db_path == tmp_path / "lcm.db" or str(engine._store.db_path) == str(tmp_path / "lcm.db")
        assert engine._cursor_from_durable_commit_proof(host) is None
    finally:
        engine.shutdown()
