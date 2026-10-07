"""Historical native proofs and replay digests remain readable after writer removal."""
import copy
import json
import random
import time
from collections import Counter
from types import SimpleNamespace

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.reconcile import (
    _PRESERVED_OBJECTIVE_CONTEXT_PREFIX,
    _commit_proof_identity_digest,
    _emission_identity,
)

from tests.legacy_native_state import install_legacy_writer, legacy_commit_proof


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    cfg = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), native_recovery=True,
        fresh_tail_count=2, fresh_tail_max_tokens=2000, leaf_chunk_tokens=400,
        dynamic_leaf_chunk_enabled=False, embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
    )
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start("retained", conversation_id="conversation")
    e.model, e.provider, e.api_mode = "gpt-6-astra", "openai-codex", "codex_responses"
    e.context_length, e.threshold_tokens = 272000, 204000
    e._compression_cancelled_check = lambda: False
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **kw: ("leaf summary", 1))
    monkeypatch.setattr(e, "_record_compress_commit_proof", legacy_commit_proof.__get__(e))
    yield e
    e.shutdown()


def install_native(monkeypatch, behavior=None):
    install_legacy_writer(monkeypatch, LCMEngine)
    calls = []
    class Native:
        _COMPACTION_TAIL_MARKER = "_compaction_tail"
        _DB_PERSISTED_MARKER = "_db_persisted"
        _last_compress_aborted = False
        _last_summary_fallback_used = False
        _last_compression_made_progress = True

        def __init__(self, **kwargs):
            assert kwargs["abort_on_summary_failure"] is True
            assert kwargs["model"] == "gpt-6-astra"
            assert kwargs["provider"] == "openai-codex"
            self.init_kwargs = kwargs
            calls.append(self)

        def on_session_start(self, sid):
            assert sid == "retained"

        @classmethod
        def _strip_context_summary_handoff_message(cls, message):
            if message.get("content") == "native summary":
                return None
            if "_test_native_merged_source" in message:
                projected = copy.deepcopy(message)
                projected["content"] = projected.pop("_test_native_merged_source")
                return projected
            return copy.deepcopy(message)

        @classmethod
        def _derive_auto_focus_topic(cls, messages):
            return next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                None,
            )

        @classmethod
        def _is_synthetic_compression_user_turn(cls, message):
            return False

        def _find_inflight_user_task(self, messages):
            return "stale task"

        def compress(self, messages, **kwargs):
            assert callable(self._compression_cancelled_check)
            self.compress_kwargs = kwargs
            self.incoming = copy.deepcopy(messages)
            if behavior:
                return behavior(self, messages)
            return [{"role": "assistant", "content": "native summary"}] + messages[-3:]

    # Tests run with the repository's minimal Hermes dependency stub.
    import sys
    monkeypatch.setitem(
        sys.modules,
        "agent.context_compressor",
        SimpleNamespace(
            ContextCompressor=Native,
            _COMPACTION_TAIL_MARKER=Native._COMPACTION_TAIL_MARKER,
            _DB_PERSISTED_MARKER=Native._DB_PERSISTED_MARKER,
        ),
    )
    return calls


def history():
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"turn {i}: " + "preserved source " * 70} for i in range(12)]


@pytest.mark.parametrize("rollover", [False, True])
def test_native_recovery_persists_subsequent_turns(candidate, monkeypatch, rollover):
    install_native(monkeypatch)
    messages = history()
    compressed = candidate.compress(messages, current_tokens=250000, force=True)
    before = candidate._store._conn.execute(
        "SELECT * FROM messages ORDER BY store_id"
    ).fetchall()
    if rollover:
        candidate.on_session_start(
            "retained-child", boundary_reason="compression", old_session_id="retained",
        )
    new_turns = [
        {"role": "user", "content": "Post-compaction constraint: preserve ORCHID-937."},
        {"role": "assistant", "content": "Acknowledged ORCHID-937."},
    ]
    candidate._ingest_messages(compressed + new_turns)
    after = candidate._store._conn.execute(
        "SELECT * FROM messages ORDER BY store_id"
    ).fetchall()
    assert after[:len(before)] == before
    assert len(after) == len(before) + len(new_turns)
    assert candidate._store._conn.execute(
        "SELECT role, content FROM messages ORDER BY store_id DESC LIMIT 2"
    ).fetchall() == [(m["role"], m["content"]) for m in reversed(new_turns)]


def test_native_recovery_host_rejection_reconciles_original_snapshot(candidate, monkeypatch):
    install_native(monkeypatch)
    messages = history()
    candidate.compress(messages, current_tokens=250_000, force=True)
    before = candidate._store._conn.execute(
        "SELECT * FROM messages ORDER BY store_id"
    ).fetchall()
    new_turns = [
        {"role": "user", "content": "Host rejected the archive; keep COBALT-418."},
        {"role": "assistant", "content": "COBALT-418 retained."},
    ]

    candidate._ingest_messages(messages + new_turns)

    after = candidate._store._conn.execute(
        "SELECT * FROM messages ORDER BY store_id"
    ).fetchall()
    assert after[:len(before)] == before
    assert len(after) == len(before) + len(new_turns)
    assert candidate._store._conn.execute(
        "SELECT role, content FROM messages ORDER BY store_id DESC LIMIT 2"
    ).fetchall() == [(m["role"], m["content"]) for m in reversed(new_turns)]


def test_native_recovery_missing_commit_proof_reconciles_host_rejection(
    candidate, monkeypatch
):
    install_native(monkeypatch)

    def fail_persist(_proof):
        raise RuntimeError("synthetic proof persistence failure")

    monkeypatch.setattr(candidate, "_persist_compress_commit_proof", fail_persist)
    messages = history()
    candidate.compress(messages, current_tokens=250_000, force=True)

    assert candidate._compress_commit_proof is None
    assert candidate._ingest_cursor == 0
    assert candidate._ingest_cursor_needs_reconcile is True

    before = candidate._store.get_session_count("retained")
    new_turns = [
        {"role": "user", "content": "Rejected recovery constraint: keep JADE-419."},
        {"role": "assistant", "content": "JADE-419 retained."},
    ]
    candidate._ingest_messages(messages + new_turns)

    assert candidate._store.get_session_count("retained") == before + len(new_turns)


def test_native_recovery_replay_proof_survives_cold_restart(tmp_path, monkeypatch):
    cfg = LCMConfig(
        database_path=str(tmp_path / "restart-lcm.db"), native_recovery=True,
        fresh_tail_count=2, fresh_tail_max_tokens=2000, leaf_chunk_tokens=400,
        dynamic_leaf_chunk_enabled=False, embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
    )
    first = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    first.on_session_start("retained", conversation_id="restart-conversation")
    first.model, first.provider, first.api_mode = "gpt-6-astra", "openai-codex", "codex_responses"
    first.context_length, first.threshold_tokens = 272000, 204000
    first._compression_cancelled_check = lambda: False
    calls = install_native(monkeypatch)
    messages = history()
    recovered = first.compress(messages, current_tokens=250_000, force=True)
    assert calls
    first.shutdown()

    second = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    second.on_session_start("retained", conversation_id="restart-conversation")
    new_turns = [
        {"role": "user", "content": "Cold restart constraint: preserve VIOLET-772."},
        {"role": "assistant", "content": "VIOLET-772 retained."},
    ]
    try:
        before_count = second._store.get_session_count("retained")
        second._ingest_messages(recovered + new_turns)
        assert second._store.get_session_count("retained") == before_count + len(new_turns)
        assert second._store._conn.execute(
            "SELECT role, content FROM messages ORDER BY store_id DESC LIMIT 2"
        ).fetchall() == [(m["role"], m["content"]) for m in reversed(new_turns)]
    finally:
        second.shutdown()


def test_config_opt_in(monkeypatch):
    monkeypatch.delenv("LCM_NATIVE_RECOVERY", raising=False)
    assert LCMConfig.from_env().native_recovery is False
    monkeypatch.setenv("LCM_NATIVE_RECOVERY", "true")
    assert LCMConfig.from_env().native_recovery is True


def test_native_recovery_in_place_commit_sequence_keeps_every_new_turn(candidate, monkeypatch):
    """#484 round 1 item 1: with native recovery the compress() records no commit
    proof, so the host's commit end is a real end (ingest + finalize). The in-place
    compression start must then reconcile the recovered list instead of keeping the
    end's len(input) cursor, or the first new turn is treated as already stored."""
    install_native(monkeypatch)
    messages = history()
    compressed = candidate.compress(messages, current_tokens=250_000, force=True)
    assert candidate.last_compression_status == "host_native"
    candidate.on_session_end("retained", messages)
    candidate.on_session_start(
        "retained", boundary_reason="compression", old_session_id="retained",
    )
    before = candidate._store._conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    new_turns = [
        {"role": "user", "content": "After the in-place native commit: keep AMBER-551."},
        {"role": "assistant", "content": "AMBER-551 kept."},
    ]
    candidate._ingest_messages(compressed + new_turns)
    after = candidate._store._conn.execute("SELECT role, content FROM messages ORDER BY store_id").fetchall()
    assert after[: len(before)] == before
    assert after[len(before):] == [(m["role"], m["content"]) for m in new_turns]
    assert len(after) == len(set(after))


def _native_engine(tmp_path, session_id="retained", **overrides):
    cfg = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), native_recovery=True,
        fresh_tail_count=2, fresh_tail_max_tokens=2000, leaf_chunk_tokens=400,
        dynamic_leaf_chunk_enabled=False, embeddings_enabled=False,
        temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    e = LCMEngine(config=cfg, hermes_home=str(tmp_path))
    e.on_session_start(session_id, conversation_id="conversation")
    e.model, e.provider, e.api_mode = "gpt-6-astra", "openai-codex", "codex_responses"
    e.context_length, e.threshold_tokens = 272000, 204000
    e._compression_cancelled_check = lambda: False
    return e


def _native_commit(engine, messages, *, in_place):
    """compress -> end(retained, input) -> start(<same or child>, compression)."""
    recovered = engine.compress(messages, current_tokens=250_000, force=True)
    assert engine.last_compression_status == "host_native"
    engine.on_session_end("retained", messages)
    child = "retained" if in_place else "retained-child"
    engine.on_session_start(child, boundary_reason="compression", old_session_id="retained")
    return recovered, child


@pytest.mark.parametrize("in_place", [True, False], ids=["inplace", "rotation"])
def test_native_recovery_restart_before_first_ingest_keeps_every_new_turn(tmp_path, monkeypatch, in_place):
    """#484 round 2 item 18: a restart between the native compression boundary and
    the first ingest; an empty rotation child must still reconcile against its
    carried native replay proof instead of storing the recovered snapshot again."""
    install_native(monkeypatch)
    engine = _native_engine(tmp_path)
    messages = history()
    try:
        recovered, child = _native_commit(engine, messages, in_place=in_place)
    finally:
        engine.shutdown()
    resumed = _native_engine(tmp_path, session_id=child)
    try:
        before = resumed._store._conn.execute("SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
        new_turns = [
            {"role": "user", "content": "After the restart: keep JADE-204."},
            {"role": "assistant", "content": "JADE-204 kept."},
        ]
        resumed._ingest_messages(recovered + new_turns)
        after = resumed._store._conn.execute("SELECT session_id, role, content FROM messages ORDER BY store_id").fetchall()
        assert after[: len(before)] == before
        assert [(role, content) for _sid, role, content in after[len(before):]] == [
            (m["role"], m["content"]) for m in new_turns
        ]
    finally:
        resumed.shutdown()


_SYNTH_PASSWORD_ROW = "turn 10: set password=SYNTHaaaa1111 for the rig. " + "preserved source " * 70


@pytest.mark.parametrize("path", ["in-process", "rollover-child", "after-restart"])
def test_native_replay_proof_refuses_lossy_password_identities(tmp_path, monkeypatch, path):
    """#484 round 2 item 15: a password_assignment placeholder carries no digest, so
    a recovered snapshot whose retained row changed to a different same-length
    password hashes to the same native replay digest. That match must not skip the
    changed row (synthetic values only)."""
    install_native(monkeypatch)
    sensitive = {"sensitive_patterns_enabled": True, "sensitive_patterns": ["password_assignment"]}
    engine = _native_engine(tmp_path, **sensitive)
    messages = history()
    messages[-2] = {"role": "user", "content": _SYNTH_PASSWORD_ROW}
    try:
        engine.ingest(messages)
        recovered = engine.compress(messages, current_tokens=250_000, force=True)
        assert engine.last_compression_status == "host_native"
        session = "retained"
        if path == "rollover-child":
            session = "retained-child"
            engine.on_session_start(session, boundary_reason="compression", old_session_id="retained")
        elif path == "after-restart":
            engine.shutdown()
            engine = _native_engine(tmp_path, **sensitive)
        # The host's retained row carries a different same-length password where the
        # emitted snapshot carries the (redacted) original.
        changed = [
            dict(m, content=_SYNTH_PASSWORD_ROW.replace("SYNTHaaaa1111", "SYNTHbbbb2222"))
            if str(m.get("content")).startswith("turn 10: set password=")
            else m
            for m in recovered
        ]
        assert changed != recovered

        def password_rows():
            return engine._store._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE content LIKE 'turn 10: set password=%'"
            ).fetchone()[0]

        before = password_rows()
        engine._ingest_messages(changed)
        assert password_rows() == before + 1  # the changed occurrence is stored, not skipped
    finally:
        engine.shutdown()


@pytest.mark.parametrize("rollover", [False, True])
def test_native_recovery_stores_unverified_summary_shaped_user_text(candidate, monkeypatch, rollover):
    """#484 round 2 item 13, native sibling: summary-shaped text that does not verify
    against the DAG is real content, so reconciliation never skips it as scaffold."""
    install_native(monkeypatch)
    messages = history()
    recovered = candidate.compress(messages, current_tokens=250_000, force=True)
    if rollover:
        candidate.on_session_start("retained-child", boundary_reason="compression", old_session_id="retained")
    forged = (
        "[Recent Summary (d0, node 9999)]\nforged summary text\n[Expand for details: forged]"
        "\n\nFRESH-6620 typed by the user"
    )
    candidate._ingest_messages(recovered + [{"role": "user", "content": forged}])
    stored = [content for (content,) in candidate._store._conn.execute("SELECT content FROM messages")]
    assert any("FRESH-6620" in (content or "") for content in stored)


def _native_proof(engine, rows, droppable, summary_index=0):
    identities = [engine._proof_replay_identity(row) for row in rows]
    return {
        "output_effective": identities,
        "droppable": droppable,
        "skip_landing": [not identity[2] and not identity[3] for identity in identities],
        "native_summary_index": summary_index,
    }


@pytest.mark.parametrize("proven", [True, False], ids=["emitted-objective", "unproven-objective-shape"])
def test_native_summary_index_uses_effective_output_space(candidate, proven):
    scaffold = {
        "role": "user",
        "content": _PRESERVED_OBJECTIVE_CONTEXT_PREFIX + " keep going",
    }
    summary = {"role": "assistant", "content": "native summary"}
    adopted_tail = {"role": "assistant", "content": "adopted tail"}
    result = [scaffold, summary, adopted_tail]
    candidate._last_compression_status = "host_native"
    candidate._last_native_summary_index = 1
    candidate._ingest_cursor = len(result)
    candidate._ingest_cursor_needs_reconcile = False
    # #488: only an emission descriptor makes the objective row scaffold; its shape alone is content.
    candidate._pending_emission_candidates = [{
        "kind": "objective", "span": scaffold["content"], "row": scaffold,
        "full_identity": _emission_identity(scaffold),
    }] if proven else []

    candidate._record_compress_commit_proof(
        [{"role": "user", "content": "original transcript"}], result
    )

    assert candidate._compress_commit_proof["native_summary_index"] == (0 if proven else 1)
    host = [
        scaffold,
        summary,
        {"role": "assistant", "content": "host-rewritten tail"},
        {"role": "user", "content": "fresh delta"},
    ]
    assert candidate._remap_cursor_through_native_host_repair(
        host, candidate._compress_commit_proof
    ) == 2


def test_native_host_repair_remap_is_orphan_tool_only(candidate):
    summary = {"role": "assistant", "content": "native summary"}
    orphan = {"role": "tool", "tool_call_id": "orphan", "content": "dropped"}
    kept = {"role": "user", "content": "kept source"}
    fresh = {"role": "assistant", "content": "fresh delta"}
    proof = _native_proof(candidate, [summary, orphan, kept], [False, True, False])

    assert candidate._remap_cursor_through_native_host_repair([summary, orphan, kept], proof) == 3
    assert candidate._remap_cursor_through_native_host_repair([summary, kept, fresh], proof) == 2

    non_tool_gap = _native_proof(candidate, [summary, fresh, kept], [False, False, False])
    assert candidate._remap_cursor_through_native_host_repair([summary, kept], non_tool_gap) == 1
    assert candidate._remap_cursor_through_native_host_repair([fresh, summary, kept], proof) is None


def test_native_host_repair_keeps_new_identity_duplicate_after_orphan_gap(candidate):
    summary = {"role": "assistant", "content": "native summary"}
    orphan = {"role": "tool", "tool_call_id": "orphan", "content": "dropped"}
    adopted = {"role": "user", "content": "same visible row"}
    proof = _native_proof(candidate, [summary, orphan, adopted], [False, True, False])

    # The first copy is the adopted row; the identical second copy is new and
    # must remain the delta start instead of being consumed as replay.
    host = [summary, adopted, dict(adopted)]
    assert candidate._remap_cursor_through_native_host_repair(host, proof) == 2


def test_native_host_repair_keeps_id_bearing_skip_landing_as_delta(candidate):
    summary = {"role": "assistant", "content": "native summary"}
    orphan = {"role": "tool", "tool_call_id": "call9", "content": "dropped"}
    adopted_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call0",
                "type": "function",
                "function": {"name": "list_files", "arguments": "{}"},
            }
        ],
    }
    proof = _native_proof(
        candidate, [summary, orphan, adopted_call], [False, True, False]
    )
    new_call = copy.deepcopy(adopted_call)
    host = [
        summary,
        new_call,
        {"role": "tool", "tool_call_id": "call0", "content": "new result"},
    ]

    assert candidate._remap_cursor_through_native_host_repair(host, proof) == 1

    _write_durable_native_proof(
        candidate, [summary, orphan, adopted_call], [False, True, False]
    )
    assert candidate._cursor_from_durable_commit_proof(host) == 1


def test_native_host_repair_starts_delta_before_new_content_after_missing_id_row(candidate):
    summary = {"role": "assistant", "content": "native summary"}
    adopted_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call0",
                "type": "function",
                "function": {"name": "list_files", "arguments": "{}"},
            }
        ],
    }
    proof = _native_proof(candidate, [summary, adopted_call], [False, False])
    host = [summary, {"role": "user", "content": "different new content"}]

    assert candidate._remap_cursor_through_native_host_repair(host, proof) == 1

    _write_durable_native_proof(candidate, [summary, adopted_call], [False, False])
    assert candidate._cursor_from_durable_commit_proof(host) == 1


def _write_durable_native_proof(
    engine,
    rows,
    droppable,
    *,
    version=2,
    native=True,
    include_skip_landing=True,
):
    identity_for_proof = (
        engine._message_replay_identity
        if version == 2
        else engine._proof_replay_identity
    )
    identities = [identity_for_proof(row) for row in rows]
    payload = {
        "version": version,
        "hermes_home": str(engine._hermes_home or ""),
        "conversation_id": engine._conversation_id,
        "created_at": time.time(),
        "effective_sha256": [
            _commit_proof_identity_digest(identity) for identity in identities
        ],
        "last_store_id": 0,
        "native": native,
        "droppable": droppable,
        "native_summary_index": 0,
    }
    if include_skip_landing:
        payload["skip_landing"] = [
            not identity[2] and not identity[3] for identity in identities
        ]
    engine._store.write_metadata_json(["compaction_commit_proof:retained"], json.dumps(payload))


def test_durable_native_proof_skips_only_orphan_tool_results(candidate):
    summary = {"role": "assistant", "content": "native summary"}
    orphan = {"role": "tool", "tool_call_id": "orphan", "content": "dropped"}
    kept = {"role": "user", "content": "kept source"}
    _write_durable_native_proof(candidate, [summary, orphan, kept], [False, True, False])
    assert candidate._cursor_from_durable_commit_proof([summary, kept]) == 2

    _write_durable_native_proof(
        candidate,
        [summary, orphan, kept],
        [False, True, False],
        include_skip_landing=False,
    )
    # Older payloads cannot prove a safe skip landing, so ``kept`` remains the
    # delta start instead of being consumed as replay.
    assert candidate._cursor_from_durable_commit_proof([summary, kept]) == 1

    _write_durable_native_proof(candidate, [summary, orphan, kept], [False, True, False], native=False)
    assert candidate._cursor_from_durable_commit_proof([summary, kept]) is None


def test_durable_native_proof_refuses_lossy_identity_before_matching(candidate):
    lossy = {
        "role": "user",
        "content": "[LCM sensitive redaction: name=password_assignment; chars=12]",
    }
    _write_durable_native_proof(candidate, [lossy], [False])
    assert candidate._cursor_from_durable_commit_proof([lossy]) is None


def test_native_rejection_prefix_refuses_lossy_identity_before_matching(candidate):
    lossy = {
        "role": "user",
        "content": "[LCM sensitive redaction: name=password_assignment; chars=12]",
    }
    identity = candidate._message_replay_identity(lossy)
    candidate._compress_commit_proof = {
        "session_id": "retained", "conversation_id": "conversation", "consulted": False,
        "output": [identity], "input": [identity], "native": True,
    }
    candidate._ingest_cursor = 1

    candidate._ingest_messages([lossy])

    assert candidate._store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1


def test_consumed_native_proof_rekeys_on_rotation(candidate, monkeypatch):
    install_native(monkeypatch)
    messages = history()
    recovered = candidate.compress(messages, current_tokens=250_000, force=True)
    candidate.on_session_end("retained", messages)
    assert candidate._compress_commit_proof["end_consumed"] is True

    candidate.on_session_start("retained-child", boundary_reason="compression", old_session_id="retained")
    assert candidate._ingest_cursor == len(recovered)
    assert candidate._ingest_cursor_needs_reconcile is False
    assert candidate._compress_commit_proof["session_id"] == "retained-child"
    assert candidate._store.read_metadata_json("compaction_commit_proof:retained-child")["native"] is True


def test_unconsumed_native_proof_still_reconciles_on_rotation(candidate, monkeypatch):
    install_native(monkeypatch)
    messages = history()
    candidate.compress(messages, current_tokens=250_000, force=True)
    assert candidate._compress_commit_proof["end_consumed"] is False

    candidate.on_session_start("retained-child", boundary_reason="compression", old_session_id="retained")
    assert candidate._compress_commit_proof is None
    assert candidate._ingest_cursor == 0
    assert candidate._ingest_cursor_needs_reconcile is True


def _host_repair_for_native_scenario(messages):
    """The Hermes repair passes relevant to native adoption (#479/#500)."""
    merged = []
    for message in copy.deepcopy(messages):
        previous = merged[-1] if merged else None
        if previous and previous.get("role") == message.get("role") == "assistant":
            left, right = previous.get("content") or "", message.get("content") or ""
            previous["content"] = f"{left}\n\n{right}" if left and right else left or right
            if message.get("tool_calls"):
                previous["tool_calls"] = list(previous.get("tool_calls") or []) + list(
                    message["tool_calls"]
                )
            continue
        merged.append(message)

    repaired, known, matched = [], set(), set()
    for message in merged:
        role = message.get("role")
        if role in {"assistant", "user"}:
            known = {
                str(call.get("id") or call.get("call_id") or "").strip()
                for call in (message.get("tool_calls") or [])
            } - {""}
            matched = set()
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "").strip()
            if call_id and (call_id not in known or call_id in matched):
                continue
            matched.add(call_id)
        repaired.append(message)

    output = []
    for index, message in enumerate(repaired):
        calls = message.get("tool_calls") or []
        if message.get("role") == "assistant" and calls:
            answered = {
                str(row.get("tool_call_id") or "").strip()
                for row in repaired[index + 1 :]
                if row.get("role") == "tool"
            }
            kept = [
                call
                for call in calls
                if str(call.get("id") or call.get("call_id") or "").strip()
                in answered
            ]
            if len(kept) != len(calls):
                if not kept and not str(message.get("content") or "").strip():
                    continue
                message = dict(message)
                if kept:
                    message["tool_calls"] = kept
                else:
                    message.pop("tool_calls", None)
        output.append(message)
    return output


def _native_scenario_metrics(engine, expected):
    actual = Counter(
        engine._store._conn.execute(
            "SELECT role, COALESCE(content, '') FROM messages ORDER BY store_id"
        ).fetchall()
    )
    wanted = Counter((row["role"], row.get("content") or "") for row in expected)
    duplicates = sum(max(0, count - wanted[key]) for key, count in actual.items())
    lost = sum(max(0, count - actual[key]) for key, count in wanted.items())
    summaries = sum(
        count for (role, content), count in actual.items() if content == "native summary"
    )
    return duplicates, lost, summaries


def _run_native_repair_scenario(
    tmp_path, monkeypatch, history_rows, first_new, second_new, in_place, restart
):
    def native_behavior(_native, messages):
        return [{"role": "user", "content": "native summary"}] + copy.deepcopy(
            messages[-3:]
        )

    install_native(monkeypatch, native_behavior)
    engine = _native_engine(tmp_path)
    try:
        engine.ingest(history_rows)
        recovered = engine.compress(
            copy.deepcopy(history_rows), current_tokens=250_000, force=True
        )
        assert engine.last_compression_status == "host_native"
        engine.on_session_end("retained", history_rows)
        session_id = "retained" if in_place else "retained-child"
        engine.on_session_start(
            session_id,
            boundary_reason="compression",
            old_session_id="retained",
            conversation_id="conversation",
        )
        host = _host_repair_for_native_scenario(recovered) + copy.deepcopy(first_new)
        if restart == "before":
            engine.shutdown()
            engine = _native_engine(tmp_path, session_id=session_id)
        engine.ingest(host)
        first = _native_scenario_metrics(engine, history_rows + first_new)
        if restart == "after":
            engine.shutdown()
            engine = _native_engine(tmp_path, session_id=session_id)
        host += copy.deepcopy(second_new)
        engine.ingest(host)
        second = _native_scenario_metrics(
            engine, history_rows + first_new + second_new
        )
        return first, second
    finally:
        engine.shutdown()


def _run_native_whitespace_adoption_scenario(tmp_path, monkeypatch, restart):
    def native_behavior(_native, messages):
        return [{"role": "assistant", "content": "native summary"}] + copy.deepcopy(
            messages[-2:]
        )

    install_native(monkeypatch, native_behavior)
    engine = _native_engine(tmp_path)
    history_rows = history()
    history_rows[-2]["content"] = "  ACP adopted user row keeps edge whitespace.  \n"
    first_new = [{"role": "assistant", "content": "Whitespace adoption acknowledged."}]
    second_new = [{"role": "user", "content": "Continue after whitespace adoption."}]
    try:
        engine.ingest(history_rows)
        recovered = engine.compress(
            copy.deepcopy(history_rows), current_tokens=250_000, force=True
        )
        assert engine.last_compression_status == "host_native"

        adopted = copy.deepcopy(recovered)
        adopted_user = next(
            row
            for row in adopted
            if row.get("content") == "  ACP adopted user row keeps edge whitespace.  \n"
        )
        adopted_user["content"] = adopted_user["content"].strip()
        proof = engine._compress_commit_proof
        in_memory_cursor = engine._remap_cursor_through_native_host_repair(
            adopted, proof
        )
        durable_cursor = engine._cursor_from_durable_commit_proof(adopted)
        assert in_memory_cursor == durable_cursor == len(adopted)

        engine.on_session_end("retained", history_rows)
        engine.on_session_start(
            "retained",
            boundary_reason="compression",
            old_session_id="retained",
            conversation_id="conversation",
        )
        host = adopted + copy.deepcopy(first_new)
        if restart == "before":
            engine.shutdown()
            engine = _native_engine(tmp_path)
        engine.ingest(host)
        first = _native_scenario_metrics(engine, history_rows + first_new)
        if restart == "after":
            engine.shutdown()
            engine = _native_engine(tmp_path)
        host += copy.deepcopy(second_new)
        engine.ingest(host)
        second = _native_scenario_metrics(
            engine, history_rows + first_new + second_new
        )
        return first, second
    finally:
        engine.shutdown()


@pytest.mark.parametrize("restart", ["none", "before", "after"])
def test_native_adoption_proof_tolerates_user_edge_whitespace(
    tmp_path, monkeypatch, restart
):
    first, second = _run_native_whitespace_adoption_scenario(
        tmp_path, monkeypatch, restart
    )

    assert first == (0, 0, 0)
    assert second == (0, 0, 0)


@pytest.mark.parametrize("shape", ["call-only", "orphan-tool"])
@pytest.mark.parametrize("restart", ["none", "before", "after"])
@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_s7_host_drops_trailing_identity_row_before_new_content(
    tmp_path, monkeypatch, shape, restart, in_place
):
    history_rows = history()[:6] + [
        {"role": "user", "content": "S7 setup user"},
        {"role": "assistant", "content": "S7 setup reply"},
        {"role": "user", "content": "S7 pending user"},
    ]
    if shape == "call-only":
        history_rows.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "s7-call",
                        "type": "function",
                        "function": {"name": "list_files", "arguments": "{}"},
                    }
                ],
            }
        )
    else:
        history_rows.append(
            {
                "role": "tool",
                "tool_call_id": "s7-orphan",
                "tool_name": "list_files",
                "content": "S7 orphan result",
            }
        )
    first_new = [{"role": "assistant", "content": "S7 fresh reply"}]
    second_new = [
        {"role": "user", "content": "S7 next user"},
        {"role": "assistant", "content": "S7 next reply"},
    ]

    first, second = _run_native_repair_scenario(
        tmp_path,
        monkeypatch,
        history_rows,
        first_new,
        second_new,
        in_place,
        restart,
    )

    assert first == (0, 0, 0)
    assert second == (0, 0, 0)


@pytest.mark.parametrize("restart", ["none", "before", "after"])
@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "rotation"])
def test_s8_host_strips_unanswered_tool_calls_from_content_assistant(
    tmp_path, monkeypatch, restart, in_place
):
    history_rows = history()[:6] + [
        {"role": "user", "content": "S8 setup user"},
        {"role": "assistant", "content": "S8 setup reply"},
        {"role": "user", "content": "S8 pending user"},
        {
            "role": "assistant",
            "content": "S8 checking files",
            "tool_calls": [
                {
                    "id": "s8-call",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": "{}"},
                }
            ],
        },
    ]
    first_new = [{"role": "user", "content": "S8 fresh user"}]
    second_new = [{"role": "assistant", "content": "S8 fresh reply"}]

    first, second = _run_native_repair_scenario(
        tmp_path,
        monkeypatch,
        history_rows,
        first_new,
        second_new,
        in_place,
        restart,
    )

    assert first[0] <= 1 and first[1:] == (0, 0)
    assert second[0] <= 1 and second[1:] == (0, 0)


@pytest.mark.parametrize("proof_version", [2, 3], ids=["v2-exact", "v3-replay"])
def test_native_repair_walks_are_seeded_equivalent(candidate, proof_version):
    rng = random.Random(492_00 + proof_version)

    def row(kind, token):
        if kind == "user":
            content = f"user-{token % 7}"
            if proof_version == 3:
                content = rng.choice(["", " ", "  ", "\n", " \n"]) + content
                content += rng.choice(["", " ", "\n"])
            return {"role": "user", "content": content}
        if kind == "assistant":
            return {"role": "assistant", "content": f"assistant-{token % 7}"}
        if kind == "call":
            return {
                "role": "assistant",
                "content": "" if token % 2 else f"checking-{token % 5}",
                "tool_calls": [
                    {
                        "id": f"call-{token % 5}",
                        "type": "function",
                        "function": {"name": "tool", "arguments": "{}"},
                    }
                ],
            }
        return {
            "role": "tool",
            "tool_call_id": f"call-{token % 5}",
            "tool_name": "tool",
            "content": f"result-{token % 3}",
        }

    kinds = ["user", "assistant", "call", "tool", "tool"]
    for trial in range(2500):
        adopted = [{"role": "assistant", "content": "native summary"}] + [
            row(rng.choice(kinds), rng.randrange(20))
            for _ in range(rng.randint(2, 8))
        ]
        called = {
            str(call.get("id") or "").strip()
            for message in adopted
            for call in (message.get("tool_calls") or [])
        }
        returned = {
            str(message.get("tool_call_id") or "").strip()
            for message in adopted
            if message.get("role") == "tool"
        }
        matched = called & returned
        droppable = [
            message.get("role") == "tool"
            and str(message.get("tool_call_id") or "").strip() not in matched
            for message in adopted
        ]
        proof = _native_proof(candidate, adopted, droppable)
        host_base = [
            copy.deepcopy(message)
            for index, message in enumerate(adopted)
            if not (droppable[index] and rng.random() < 0.7)
        ]
        if proof_version == 3:
            for message in host_base:
                if message.get("role") == "user" and rng.random() < 0.5:
                    message["content"] = message["content"].strip()
        if len(host_base) > 1 and rng.random() < 0.35:
            del host_base[rng.randrange(1, len(host_base))]
        fresh = [
            {"role": "user", "content": f"fresh-{trial}-{index}"}
            for index in range(1 + rng.randrange(3))
        ]
        host = host_base + fresh
        _write_durable_native_proof(
            candidate, adopted, droppable, version=proof_version
        )

        in_memory = candidate._remap_cursor_through_native_host_repair(host, proof)
        durable = candidate._cursor_from_durable_commit_proof(host)

        assert in_memory == durable, (trial, in_memory, durable, adopted, host)
        assert in_memory is None or in_memory <= len(host_base)
