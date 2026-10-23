"""Synthetic #1880 B: ordinary no-progress proofs survive engine reconstruction.

No live host/profile is needed. Cold threshold queries deliberately fail open
until the current ingest has supplied replay/cleanup identity.
"""
import copy
import json
from unittest.mock import Mock

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.dag import SummaryNode
from tests.harness import make_engine, pin_char_counter
from tests.test_issue_1162_retained_escape import _reorder

KEY = "compaction_no_progress:v1:S"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", Mock(return_value=(
        "Earlier turns.\nExpand for details about: synthetic turns", 1)))


def rebuild(path, **config):
    engine = make_engine(path,
                         fresh_tail_count=1, leaf_chunk_tokens=1, context_threshold=0.001,
                         dynamic_leaf_chunk_max=400, threshold_full_sweep_enabled=False,
                         temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
                         fresh_tail_pressure_yield_enabled=False, **config)
    engine._hermes_home = str(path)
    engine._store._hermes_home = str(path)
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    engine.last_prompt_tokens = 10_000  # Same host request-pressure observation as the completed pass.
    return engine


def arm(path):
    engine, view = _reorder(path, False)
    engine._config.fresh_tail_pressure_yield_enabled = False
    engine._config.empty_lifecycle_gc_enabled = False
    # Publish the valid prefix before R, leaving R immediately after frontier.
    view = engine.compress(view, current_tokens=10_000)
    engine._no_progress_hold = None
    engine_module.summarize_with_escalation.reset_mock()
    before = copy.deepcopy(engine._store.get_session_messages("S"))
    result = engine.compress(view, current_tokens=10_000)
    assert result == view
    assert engine._last_compress_leaves[1] == 0
    assert engine._no_progress_hold[1] == "no_progress"
    assert engine._store.get_session_messages("S") == before
    engine.shutdown()
    return view, before


@pytest.mark.parametrize("consumer", ["threshold", "preflight", "direct"])
def test_red1_rebuilt_retained_noop(tmp_path, monkeypatch, consumer):
    view, before = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        assert engine.should_compress(10_000)  # Unknown cold replay must fail open.
        if consumer == "threshold":
            engine._ingest_messages(view)
        scan = Mock(side_effect=AssertionError("held admission repeated attribution"))
        monkeypatch.setattr(engine, "_leaf_compaction_candidate_status", scan)
        monkeypatch.setattr(engine, "_prepare_compression_sweep", scan)
        if consumer == "threshold":
            assert not engine.should_compress(10_000)
            assert engine._automatic_compression_blocked()
            assert engine._compression_block_reason() == "cooldown:lcm_no_progress"
        elif consumer == "preflight":
            assert not engine.should_compress_preflight(view)
        else:
            assert engine.compress(view, current_tokens=10_000) == view
            assert engine._last_compression_status != "error"
        scan.assert_not_called()
        assert engine._store.get_session_messages("S") == before
        assert engine._store_complete_frontier() == 2
        assert engine._store.read_metadata_json(KEY)["reason"] == "no_progress"
        engine_module.summarize_with_escalation.assert_not_called()
    finally:
        engine.shutdown()


def test_red2_fixed_expiry_and_turn_end(tmp_path, monkeypatch):
    real_wall = engine_module.time.time
    real_mono = engine_module.time.monotonic
    wall, mono = real_wall(), real_mono()
    offset = [0]
    monkeypatch.setattr(engine_module.time, "time", lambda: wall + offset[0])
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: mono + offset[0])
    view, _ = arm(tmp_path)
    for age in (1, 300, 599, 601):
        offset[0] = age
        engine = rebuild(tmp_path)
        try:
            engine._ingest_messages(view)
            engine.note_turn_complete()
            assert engine.should_compress(10_000) is (age > 600)
            record = engine._store.read_metadata_json(KEY)
            assert record["expires_at"] == wall + 600
            if age < 600:
                write = Mock(side_effect=AssertionError("a hit rewrote metadata"))
                monkeypatch.setattr(engine._store, "write_metadata_json", write)
                assert not engine.should_compress(10_000)
                write.assert_not_called()
        finally:
            engine.shutdown()


@pytest.mark.parametrize("change", ["history", "config", "summary", "replay", "pressure"])
def test_red3_state_change_releases_same_turn(tmp_path, change):
    view, before = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        engine._ingest_messages(view)
        assert not engine.should_compress(10_000)
        if change == "history":
            view += [{"role": "user", "content": "new synthetic tail " * 100},
                     {"role": "assistant", "content": "new response " * 100}]
            engine._ingest_messages(view)
        elif change == "config":
            engine._config.fresh_tail_count += 1
        elif change == "summary":
            engine._dag.add_node(SummaryNode(session_id="S", depth=0, summary="Synthetic summary",
                                            token_count=2, source_ids=[], source_type="messages"))
        elif change == "replay":
            # Same source max id, different active occurrence order/cleanup identity.
            engine._remember_active_replay_messages(list(reversed(view)), list(reversed(view)))
        else:
            engine.threshold_tokens = 20_000  # discrete below-threshold class
        assert engine.should_compress(10_000) is (change != "pressure")
        assert not engine._automatic_compression_blocked()
        assert engine._store.get_session_messages("S")[:len(before)] == before
        if change == "history":
            engine.compress(view, current_tokens=10_000)
            # #1162 is deferred: a changed state retries, but cannot skip R.
            assert engine._last_compress_leaves[1] == 0
            assert engine._store_complete_frontier() == 2
            assert engine._store.get_session_messages("S")[:len(before)] == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("record", [None, {}, {"version": 2}, "broken", "future", "expired"])
def test_red4_invalid_records_fail_open(tmp_path, record, caplog):
    view, _ = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        valid = engine._store.read_metadata_json(KEY)
        assert valid is not None
        if record in ("future", "expired"):
            record = dict(valid, recorded_at=valid["recorded_at"] + (1000 if record == "future" else -1000),
                          expires_at=valid["expires_at"] + (1000 if record == "future" else -1000))
        serialized = "{" if record == "broken" else json.dumps(record)
        engine._store.write_metadata_json([KEY], serialized)
        engine._ingest_messages(view)
        assert engine.should_compress(10_000)
        assert engine.should_compress(10_000)
        assert not engine._automatic_compression_blocked()
        assert len([r for r in caplog.records if "no-progress metadata unavailable" in r.message]) <= 1
    finally:
        engine.shutdown()


def test_red4_isolation_recovery_and_no_rearm(tmp_path, monkeypatch):
    view, before = arm(tmp_path)
    engine = rebuild(tmp_path)
    other = rebuild(tmp_path / "other")
    try:
        engine._ingest_messages(view)
        assert not engine.should_compress(10_000)
        record = engine._store.read_metadata_json(KEY)
        assert other.should_compress(10_000)
        assert not engine._automatic_compression_blocked(ignore_cooldown=True)
        assert not engine._no_progress_hold_blocks(engine._survival_ceiling())
        scan = Mock(side_effect=RuntimeError("synthetic publication failure"))
        monkeypatch.setattr(engine, "_prepare_compression_sweep", scan)
        with pytest.raises(RuntimeError, match="synthetic publication failure"):
            engine.compress(view, current_tokens=10_000, force=True)
        assert scan.call_count == 1
        engine.compress(view, current_tokens=10_000, bypass_cooldown=True)
        assert engine._last_compression_status == "error"  # Recovery returns its emergency fit.
        assert scan.call_count == 2
        assert engine._store.read_metadata_json(KEY) == record
        assert engine._store.get_session_messages("S") == before
        engine._session_id = "another-synthetic-session"
        engine._no_progress_hold = None
        assert engine.should_compress(10_000)
        assert engine._store.read_metadata_json(KEY) == record
    finally:
        engine.shutdown()
        other.shutdown()


def test_red3_new_tail_makes_previous_history_eligible(tmp_path):
    engine = make_engine(tmp_path, window=200_000, session_id="S", conversation_id="conv",
                         fresh_tail_count=4, leaf_chunk_tokens=100,
                         fresh_tail_pressure_yield_enabled=False, context_threshold=0.001,
                         temporal_rollups_enabled=False, empty_lifecycle_gc_enabled=False,
                         survival_fit=False)
    view = [{"role": "user", "content": "synthetic user objective " * 30},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "synthetic-call",
             "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "synthetic-call", "content": "synthetic result " * 30},
            {"role": "assistant", "content": "synthetic answer " * 30}]
    try:
        assert engine.compress(view, current_tokens=10_000) == view
        assert engine._no_progress_hold[1] == "no_progress"
        before = copy.deepcopy(engine._store.get_session_messages("S"))
        assert not engine.should_compress(10_000)
        view += [{"role": "user" if i % 2 == 0 else "assistant", "content": f"new tail {i} " * 50}
                 for i in range(8)]
        assert engine.should_compress_preflight(view)
        engine.compress(view, current_tokens=10_000)
        assert engine._last_compress_leaves[1] >= 1
        assert engine._store_complete_frontier() >= before[-1]["store_id"]
        assert engine._store.get_session_messages("S")[:len(before)] == before
    finally:
        engine.shutdown()


def test_red4_unreadable_metadata_and_failed_ingest(tmp_path, monkeypatch, caplog):
    view, before = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        engine._ingest_messages(view)
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "read_metadata_json", Mock(side_effect=OSError("synthetic unreadable")))
            assert engine.should_compress(10_000)
            assert engine.should_compress(10_000)
        assert len([r for r in caplog.records if "no-progress metadata unavailable" in r.message]) == 1
        old_record = engine._store.read_metadata_json(KEY)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_ingest_messages", Mock(side_effect=RuntimeError("synthetic ingest failure")))
            assert not engine.should_compress_preflight(view)
        assert engine.should_compress(10_000)  # Failed ingest leaves no valid current proof.
        assert engine._store.read_metadata_json(KEY) == old_record
        assert engine._store.get_session_messages("S") == before
    finally:
        engine.shutdown()


@pytest.mark.parametrize("setting", ["leaf_target_ratio", "leaf_target_min_tokens", "leaf_target_max_tokens"])
def test_leaf_target_invalidation_and_bound_store_identity(tmp_path, monkeypatch, setting):
    view, _ = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        engine._ingest_messages(view)
        resolve = engine_module.Path.resolve
        resolved_paths = []

        def observe_resolve(path, *args, **kwargs):
            resolved_paths.append(str(path))
            return resolve(path, *args, **kwargs)

        monkeypatch.setattr(engine_module.Path, "resolve", observe_resolve)
        metadata_read = Mock(wraps=engine._store.read_metadata_json)
        indexed_read = Mock(wraps=engine._store.max_store_id)
        monkeypatch.setattr(engine._store, "read_metadata_json", metadata_read)
        monkeypatch.setattr(engine._store, "max_store_id", indexed_read)
        assert not engine.should_compress(10_000)
        reads = metadata_read.call_count, indexed_read.call_count
        assert not engine.should_compress(10_000)
        assert metadata_read.call_count > reads[0] and indexed_read.call_count > reads[1]
        assert len(resolved_paths) == 1  # Steady admission reuses only the bound path.
        value = getattr(engine._config, setting)
        setattr(engine._config, setting, value + (0.01 if setting.endswith("ratio") else 1))
        assert engine.should_compress(10_000)  # Target-only change releases the hold.
        assert len(resolved_paths) == 1
        setattr(engine._config, setting, value)
        fingerprint = engine._no_progress_fingerprint()
        original_path = engine._store.db_path
        engine._store.db_path = tmp_path / "another-synthetic-binding.db"
        assert engine._no_progress_fingerprint() != fingerprint
        assert len(resolved_paths) == 2
        engine._store.db_path = original_path
        assert engine._no_progress_fingerprint() == fingerprint
        assert len(resolved_paths) == 3
        engine._no_progress_bound_store = (object(), original_path, "synthetic-old-store")
        assert engine._no_progress_fingerprint() == fingerprint
        assert len(resolved_paths) == 4  # Same path on a different bound store is re-resolved.
    finally:
        engine.shutdown()


@pytest.mark.parametrize("setting", ["l2_budget_ratio", "summary_prompt_version", "model", "provider", "api_mode",
                                     "stub_first_exit_target_ratio", "proactive_recall_enabled", "proactive_recall_budget_tokens",
                                     "embeddings_enabled"])
def test_effective_settings_and_route_only_invalidate(tmp_path, setting):
    view, before = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        engine._ingest_messages(view)
        assert not engine.should_compress(10_000)
        old_window = engine.context_length
        if setting not in {"model", "provider", "api_mode"}:
            old = getattr(engine._config, setting)
            setattr(engine._config, setting, not old if isinstance(old, bool) else old + (0.01 if setting.endswith("ratio") else 1))
        else:
            route = dict(model=engine.model, provider=engine.provider, api_mode=engine.api_mode)
            route[setting] = "synthetic-route"
            engine.update_model(context_length=engine.raw_context_length, **route)
        assert engine.context_length == old_window
        assert engine.should_compress(10_000)
        assert not engine._automatic_compression_blocked()
        assert engine._store.get_session_messages("S") == before
    finally:
        engine.shutdown()


def test_recovered_direct_ingest_reuses_hold(tmp_path, monkeypatch):
    view, _ = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        engine._record_ingest_failure("synthetic", RuntimeError("synthetic failure"))
        scan = Mock(side_effect=AssertionError("recovered ingest repeated sweep"))
        monkeypatch.setattr(engine, "_prepare_compression_sweep", scan)
        assert engine.compress(view, current_tokens=10_000) == view
        assert engine._consecutive_ingest_failures == 0
        scan.assert_not_called()
    finally:
        engine.shutdown()


def test_exact_pressure_change_inside_bucket_releases_hold(tmp_path):
    view, _ = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        engine._ingest_messages(view)
        assert not engine.should_compress(10_000)
        assert engine.should_compress(10_001)
    finally:
        engine.shutdown()


def test_h2_zero_observation_preserves_legacy_gate(tmp_path):
    engine = make_engine(tmp_path, window=200_000, conversation_id="conv", survival_fit=False)
    view = [{"role": "user", "content": "synthetic current objective"},
            {"role": "assistant", "content": "synthetic reply"}]
    try:
        engine._last_gate_tokens = 50_883
        engine.compress(view, current_tokens=0)
        assert not engine._survival_pass_observed
        assert engine._last_gate_tokens == 50_883
        assert engine._no_progress_fingerprint() is None  # Explicitly unobserved pressure is unknown.
        assert engine._store.read_metadata_json(KEY) is None
    finally:
        engine.shutdown()


@pytest.mark.parametrize("observations", [1, 2])
def test_h2_default_pre_yield_hold_survives_rebuild(tmp_path, observations):
    settings = dict(window=200_000, conversation_id="conv", context_threshold=0.001,
                    threshold_full_sweep_enabled=False, temporal_rollups_enabled=False,
                    empty_lifecycle_gc_enabled=False, survival_fit=False)
    engine = make_engine(tmp_path, **settings)
    view = [{"role": "user", "content": "synthetic current objective " * 40},
            {"role": "assistant", "content": "synthetic reply " * 40}]
    try:
        assert engine._config.fresh_tail_pressure_yield_enabled
        assert engine._config.fresh_tail_pressure_yield_min_observations == 3
        engine._pressure_yield_blocked_streak = observations - 1
        view = engine.compress(view, current_tokens=10_000)
        assert engine._pressure_yield_blocked_streak == observations
        assert engine._store.read_metadata_json(KEY)["reason"] == "no_progress"
    finally:
        engine.shutdown()
    engine = make_engine(tmp_path, **settings)
    try:
        engine.last_prompt_tokens = 10_000
        engine._ingest_messages(view)
        assert engine._pressure_yield_blocked_streak == 0
        assert not engine.should_compress(10_000)
        engine._pressure_yield_blocked_streak = 2
        assert not engine.should_compress(10_000)
        engine._pressure_yield_blocked_streak = 3
        assert engine.should_compress(10_000)  # A meaningful yield-guard crossing still releases it.
    finally:
        engine.shutdown()


@pytest.mark.parametrize("consumer", ["direct", "preflight", "ingest"])
def test_h2_swallowed_audit_failure_stays_unknown_until_clean_recovery(tmp_path, monkeypatch, consumer):
    view, _ = arm(tmp_path)
    engine = rebuild(tmp_path)
    try:
        engine.should_compress(10_000)
        engine._ingest_messages(view)
        engine._clear_persisted_no_progress_hold()
        engine._no_progress_hold = None
        engine._compress_commit_proof = None
        view = copy.deepcopy(view)
        view[-1]["content"] += " changed synthetic current replay"
        failures = engine._ingest_failure_count
        prematch = engine._identity_anchor_prematch

        def fail_audit(messages, identities, cursor, audit_from=None, **kwargs):
            assert audit_from is not None  # Reach the supported swallowed audited-prefix branch.
            raise RuntimeError("synthetic audited-prefix failure")

        def invoke():
            if consumer == "direct":
                return engine.compress(view, current_tokens=10_000)
            if consumer == "preflight":
                return engine.should_compress_preflight(view)
            return engine.ingest(view)

        monkeypatch.setattr(engine, "_identity_anchor_prematch", fail_audit)
        invoke()
        assert engine._ingest_failure_count == failures + 1
        assert engine._consecutive_ingest_failures == 1
        assert engine._no_progress_replay is None
        assert engine._no_progress_fingerprint() is None
        assert engine._store.read_metadata_json(KEY) is None
        monkeypatch.setattr(engine, "_identity_anchor_prematch", prematch)
        invoke()
        assert engine._ingest_failure_count == failures + 1
        assert engine._consecutive_ingest_failures == 0
        assert engine._no_progress_replay is not None
        assert engine._no_progress_fingerprint() is not None
    finally:
        engine.shutdown()


def test_h2_host_held_pruned_stub_keeps_current_public_proof(tmp_path):
    from tests.test_issue_772_replay_keeps_host_stubs import CLEARED

    engine = make_engine(tmp_path, window=200_000, conversation_id="conv",
                         large_output_externalization_enabled=False,
                         large_output_active_replay_stubbing_enabled=False)
    view = [{"role": "user", "content": "synthetic current objective"}]
    for i in range(3):
        view += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"synthetic-{i}",
                  "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": f"synthetic-{i}", "content": f"synthetic result {i} " * 40}]
    try:
        engine.ingest(view)
        before = copy.deepcopy(engine._store.get_session_messages("S"))
        host = copy.deepcopy(view)
        host[2]["content"] = CLEARED
        assert engine._message_replay_identity(host[2], strip_carrier=False) == engine._message_replay_identity(view[2], strip_carrier=False)
        replay = engine._ingest_messages(host)
        assert replay[2]["content"] == CLEARED
        assert engine._store.get_session_messages("S") == before
        assert engine._no_progress_replay is not None
        active = engine._last_active_replay_messages[2]
        assert engine._no_progress_public_rows[id(active)]["content"] == CLEARED
    finally:
        engine.shutdown()
