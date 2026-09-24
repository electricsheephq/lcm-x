"""Native recovery must not convert a rejected LCM claim into committed coverage."""
import copy
import json
from types import SimpleNamespace

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.externalize import extract_externalized_ref, load_externalized_payload


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
    yield e
    e.shutdown()


def install_native(monkeypatch, behavior=None):
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


def test_native_recovery_preserves_sources_and_tools(candidate, monkeypatch):
    calls = install_native(monkeypatch)
    def reject_lcm_summary(**kwargs):
        pytest.fail("native recovery must not invoke the LCM summarizer")
    monkeypatch.setattr(engine_module, "summarize_with_escalation", reject_lcm_summary)
    # Stored old history is absent from the active replay. The normal engine
    # cannot prove a new leaf covers those rows. Recovery leaves its frontier intact.
    candidate.ingest([{"role": "user", "content": "old source remains searchable"}])
    messages = history()
    candidate.ingest(messages)
    before = candidate._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall()
    frontier = candidate._lifecycle.get_by_conversation("conversation").current_frontier_store_id
    result = candidate.compress(messages, current_tokens=250000, force=True)
    assert len(calls) == 1
    assert len(result) < len(messages)
    assert [candidate._message_replay_identity(m) for m in result[-2:]] == [
        candidate._message_replay_identity(m) for m in messages[-2:]
    ]
    assert all(m["_compaction_tail"] is True for m in result[-2:])
    assert candidate.last_compression_status == "host_native"
    assert candidate._lifecycle.get_by_conversation("conversation").current_frontier_store_id == frontier
    assert candidate._store._conn.execute("SELECT * FROM messages ORDER BY store_id").fetchall() == before
    assert candidate._dag._conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == 0
    assert any(s.get("name", s.get("function", {}).get("name")) == "lcm_recall" for s in candidate.get_tool_schemas())


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


def test_native_recovery_honors_resolved_tail_and_host_token_observation(candidate, monkeypatch):
    candidate._config.fresh_tail_count = 5
    candidate._config.fresh_tail_max_tokens = 120
    messages = history()
    expected_tail = candidate._fresh_tail_boundary(messages)

    def preserve_resolved_tail(native, incoming):
        return [{"role": "assistant", "content": "native summary"}] + incoming[-3:]

    calls = install_native(monkeypatch, preserve_resolved_tail)
    observed_tokens = 231_337
    recovered = candidate.compress(messages, current_tokens=observed_tokens, force=True)

    assert candidate.last_compression_status == "host_native"
    assert calls[0].init_kwargs["protect_last_n"] == 3
    assert calls[0].tail_token_budget == 1
    assert calls[0].incoming == messages[:expected_tail.start]
    assert calls[0].compress_kwargs["current_tokens"] == observed_tokens
    assert calls[0].compress_kwargs["focus_topic"] == messages[-2]["content"]
    assert [candidate._message_replay_identity(message) for message in recovered[-expected_tail.count:]] == [
        candidate._message_replay_identity(message) for message in messages[expected_tail.start:]
    ]


def test_native_recovery_accepts_summary_merged_into_protected_tail(candidate, monkeypatch):
    messages = history()

    def merge_summary_into_tail(native, incoming):
        tail = copy.deepcopy(incoming[-2:])
        tail[0]["_test_native_merged_source"] = tail[0]["content"]
        tail[0]["content"] = "native summary plus preserved tail carrier"
        return tail

    install_native(monkeypatch, merge_summary_into_tail)
    recovered = candidate.compress(messages, current_tokens=250_000, force=True)

    assert recovered != messages
    assert candidate.last_compression_status == "host_native"


def test_native_recovery_disables_stale_inflight_task_when_suffix_has_user(
    candidate, monkeypatch
):
    calls = install_native(monkeypatch)

    candidate.compress(history(), current_tokens=250_000, force=True)

    assert calls[0]._find_inflight_user_task([]) is None


def test_native_recovery_short_prefix_rejects_without_dispatch(candidate, monkeypatch):
    candidate._config.fresh_tail_count = 8
    calls = install_native(monkeypatch)
    messages = history()

    assert candidate._compress_native_recovery(messages) is messages
    assert calls == []
    assert candidate._last_native_recovery_rejection == "prefix_too_short"
    assert candidate.last_compression_noop_reason == "prefix_too_short"


@pytest.mark.parametrize("missing", ["_COMPACTION_TAIL_MARKER", "_DB_PERSISTED_MARKER"])
def test_native_recovery_missing_marker_uses_legacy_full_transcript(
    candidate, monkeypatch, missing
):
    calls = install_native(monkeypatch)
    module = __import__("agent.context_compressor", fromlist=["ContextCompressor"])
    delattr(module, missing)
    messages = history()
    expected_tail = candidate._fresh_tail_boundary(messages)

    recovered = candidate._compress_native_recovery(messages)

    assert calls[0].incoming == messages
    assert calls[0].init_kwargs["protect_last_n"] == expected_tail.count
    assert calls[0].tail_token_budget == expected_tail.tokens
    assert "focus_topic" not in calls[0].compress_kwargs
    assert [candidate._message_replay_identity(m) for m in recovered[-2:]] == [
        candidate._message_replay_identity(m) for m in messages[-2:]
    ]


def test_cleanup_only_handoff_precedes_native_summary_and_is_consumed(candidate, monkeypatch):
    calls = install_native(monkeypatch)
    candidate._preflight_cleanup_only_due_to_boundary_cooldown = True

    result = candidate.compress(history(), current_tokens=190_000)

    assert calls == []
    assert result == history()
    assert candidate._preflight_cleanup_only_due_to_boundary_cooldown is False
    candidate._native_recovery_preflight_cleanup_only = True
    assert candidate.compress([]) == []
    assert candidate._native_recovery_preflight_cleanup_only is False


def test_native_preflight_admits_threshold_pressure_without_lcm_leaf(candidate):
    candidate.threshold_tokens = 100
    candidate._config.leaf_chunk_tokens = 1_000_000
    candidate._config.threshold_full_sweep_enabled = False
    messages = history()

    eligible, _reason = candidate._leaf_compaction_candidate_status(messages)
    assert eligible is False
    assert candidate.should_compress_preflight(messages) is True


def test_subthreshold_ingest_cleanup_adopts_safe_replay_without_native_summary(candidate, monkeypatch):
    calls = install_native(monkeypatch)
    candidate._config.sensitive_patterns_enabled = True
    candidate._config.sensitive_patterns = ["api_key"]
    candidate._config.large_output_externalization_enabled = True
    candidate._config.large_output_externalization_threshold_chars = 12_000
    secret = "sk-synthetic-cleanup-canary-1234567890-cdef"
    retained_payload = "synthetic observed raw payload " * 2_100
    messages = [
        {
            "role": "user",
            "content": f"api_key={secret} retain this request\n{retained_payload}",
        },
        {"role": "assistant", "content": "calling lookup", "tool_calls": [{
            "id": "call-cleanup", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call-cleanup", "content": "synthetic result"},
        {"role": "assistant", "content": "synthetic acknowledgement"},
    ]
    original = copy.deepcopy(messages)

    assert candidate.should_compress_preflight(messages) is True
    before = candidate._store._conn.execute(
        "SELECT * FROM messages ORDER BY store_id"
    ).fetchall()
    result = candidate.compress(messages, current_tokens=40_000)
    serialized = json.dumps(result, sort_keys=True)

    assert calls == []
    assert secret not in serialized
    assert retained_payload in result[0]["content"]
    stored_content = candidate._store._conn.execute(
        "SELECT content FROM messages WHERE role='user' ORDER BY store_id LIMIT 1"
    ).fetchone()[0]
    ref = extract_externalized_ref(stored_content)
    assert ref is not None
    payload = load_externalized_payload(
        ref,
        config=candidate._config,
        hermes_home=candidate._hermes_home,
    )
    assert payload is not None
    assert secret not in payload["content"]
    assert "[LCM sensitive redaction:" in payload["content"]
    assert retained_payload in payload["content"]
    assert [message["role"] for message in result] == ["user", "assistant", "tool", "assistant"]
    assert result[2]["tool_call_id"] == result[1]["tool_calls"][0]["id"]
    assert messages == original
    assert candidate._store._conn.execute(
        "SELECT * FROM messages ORDER BY store_id"
    ).fetchall() == before


@pytest.mark.parametrize("native_enabled", [True, False])
def test_user_storage_stub_does_not_hide_sanitized_native_input(
    candidate, monkeypatch, native_enabled
):
    candidate._config.native_recovery = native_enabled
    candidate._config.large_output_externalization_enabled = True
    candidate._config.large_output_externalization_threshold_chars = 12_000
    candidate._config.sensitive_patterns_enabled = True
    candidate._config.sensitive_patterns = ["api_key"]
    secret = "sk-synthetic-recall-test-1234567890-cdef"
    fact = "The approved project name is amber-lantern."
    messages = [{"role": "user", "content":
                 f"api_key={secret}\n" + "ordinary retained prose " * 2800 + fact}]
    messages += history()
    original = copy.deepcopy(messages)
    replay = candidate._ingest_messages(messages)
    assert secret not in json.dumps(replay)
    assert (fact in replay[0]["content"]) is native_enabled
    stored = candidate._store._conn.execute(
        "SELECT content FROM messages WHERE role='user' ORDER BY store_id LIMIT 1"
    ).fetchone()[0]
    ref = extract_externalized_ref(stored)
    assert ref is not None
    payload = load_externalized_payload(
        ref, config=candidate._config, hermes_home=candidate._hermes_home
    )
    assert fact in payload["content"] and secret not in payload["content"]
    if native_enabled:
        seen = []
        def capture(native, incoming):
            seen.extend(copy.deepcopy(incoming))
            return [{"role": "assistant", "content": "summary: " + fact}] + incoming[-2:]
        install_native(monkeypatch, capture)
        candidate.compress(replay, current_tokens=250_000, force=True)
        assert fact in seen[0]["content"]
        assert secret not in json.dumps(seen)
    assert messages == original


@pytest.mark.parametrize(
    ("current_tokens", "force"),
    [(250_000, False), (300_000, False), (40_000, True)],
    ids=["threshold", "overflow", "explicit-force"],
)
def test_pressure_overflow_and_force_still_dispatch_native_summary(
    candidate, monkeypatch, current_tokens, force
):
    calls = install_native(monkeypatch)
    result = candidate.compress(history(), current_tokens=current_tokens, force=force)

    assert len(calls) == 1
    assert candidate.last_compression_status == "host_native"
    assert len(result) < len(history())


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("exception", "exception"),
        ("aborted", "native_aborted"),
        ("placeholder", "fallback_used"),
        ("empty", "empty"),
        ("grows", "not_smaller"),
        ("cancelled", "cancelled"),
        ("superseded", "binding_changed"),
        ("prune_only", "no_progress"),
        ("digest_failure", "digest_unavailable"),
    ],
)
def test_failed_recovery_preserves_exact_input(
    candidate, monkeypatch, caplog, kind, reason
):
    original = history()
    def behavior(native, messages):
        if kind == "exception":
            messages[0]["content"] = "mutation stays on copy"
            raise RuntimeError("provider failed")
        if kind == "prune_only":
            native._last_compression_made_progress = False
        if kind == "digest_failure":
            return [{"role": "assistant", "content": "[digest unavailable for segment 1/1 — recover via session_search]"}]
        if kind == "aborted":
            native._last_compress_aborted = True
        if kind == "placeholder":
            native._last_summary_fallback_used = True
        if kind == "empty":
            return []
        if kind == "grows":
            return messages + messages
        if kind == "cancelled":
            cancelled[0] = True
        if kind == "superseded":
            candidate._compression_cancelled_check = lambda: False
        return [{"role": "assistant", "content": "native summary"}] + messages[-2:]
    cancelled = [False]
    candidate._compression_cancelled_check = lambda: cancelled[0]
    install_native(monkeypatch, behavior)
    snapshot = copy.deepcopy(original)
    candidate._ingest_cursor = len(original)
    candidate._ingest_cursor_needs_reconcile = True
    result = candidate._compress_native_recovery(original)
    assert result is original
    assert original == snapshot
    assert candidate._last_compress_aborted is True
    assert candidate._ingest_cursor == len(original)
    assert candidate._ingest_cursor_needs_reconcile is True
    assert candidate._last_native_recovery_rejection == reason
    assert candidate.last_compression_noop_reason == reason
    assert candidate._last_summary_error == "native recovery did not produce a usable summary"
    warning = next(record.message for record in caplog.records if "Native recovery" in record.message)
    if kind == "exception":
        assert "exception_type=RuntimeError" in warning
    else:
        assert f"reason={reason}" in warning
        assert "input_rows=" in warning
        assert "recovered_rows=" in warning
        assert "protected_rows=" in warning


@pytest.mark.parametrize("failure", ["exception", "aborted", "cancelled", "placeholder"])
def test_native_abort_preserves_sanitized_active_replay(candidate, monkeypatch, failure):
    candidate._config.sensitive_patterns_enabled = True
    candidate._config.sensitive_patterns = ["api_key"]
    secret = "sk-synthetic-abort-canary-1234567890-cdef"
    messages = history()
    messages[0]["content"] += f" api_key={secret}"
    original = copy.deepcopy(messages)
    cancelled = [False]
    candidate._compression_cancelled_check = lambda: cancelled[0]

    def fail(native, incoming):
        assert secret not in json.dumps(incoming)
        if failure == "exception":
            raise RuntimeError("synthetic provider failure")
        if failure == "aborted":
            native._last_compress_aborted = True
        if failure == "cancelled":
            cancelled[0] = True
        if failure == "placeholder":
            native._last_summary_fallback_used = True
        return [{"role": "assistant", "content": "rejected summary"}]

    install_native(monkeypatch, fail)
    result = candidate.compress(messages, current_tokens=250_000, force=True)
    assert secret not in json.dumps(result)
    assert "[LCM sensitive redaction:" in result[0]["content"]
    assert result[1:] == original[1:]
    assert len(result) == len(original)
    assert messages == original
    assert candidate._last_compress_aborted is True
    assert candidate.last_compression_status == "error"


@pytest.mark.parametrize("mode", ["disabled", "unfenced", "cancelled"])
def test_no_native_dispatch_without_admission(candidate, monkeypatch, mode):
    calls = install_native(monkeypatch)
    if mode == "disabled":
        candidate._config.native_recovery = False
    elif mode == "unfenced":
        candidate._compression_cancelled_check = None
    else:
        candidate._compression_cancelled_check = lambda: True
    messages = history()
    assert candidate._compress_native_recovery(messages) is messages
    assert not calls


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
