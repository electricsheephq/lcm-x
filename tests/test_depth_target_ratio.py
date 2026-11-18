"""Synthetic acceptance for the threshold depth and summary-prefix ratios (#2838)."""

import hashlib
import json
import logging
import math
import sqlite3
import time
from types import SimpleNamespace

import pytest

import hermes_lcm.compaction as compaction
import hermes_lcm.externalize as externalize
import hermes_lcm.leaf_prep as leaf_prep
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode
from hermes_lcm.lifecycle_state import LifecycleBindingChangedError, LifecyclePublicationConflictError
from hermes_lcm.tokens import count_messages_tokens, count_tokens
from hermes_lcm.tools import lcm_status
from tests.test_issue_457_resume_controls import _HostCancel, _spy_resumes
from tests.test_issue_605_foreground_budget import _provider
from tests.test_issue_608_sweep_budget_stop import _pin_lcm_char_counter as _pin_lcm_char_counter
from tests.test_issue_608_sweep_budget_stop import clock as clock
from tests.test_issue_1137_hung_route_circuit import _provider as _hung_provider
from tests.test_issue_628_summary_circuit import _open_circuit
from tests.test_issue_628_summary_circuit import _provider as _circuit_provider
from tests.test_issue_671_stub_first_exit import (
    exit_engine,
    payload_of,
    tool_pair,
    tool_view,
)
from tests.test_issue_671_stub_first_exit import make_engine as make_engine
from tests.test_issue_671_stub_first_exit import summaries as summaries

WINDOW = 100_000
SUMMARY = "Earlier turns.\nExpand for details about: turns"
DEPTH_KEYS = {"depth_target_tokens", "fixed_load_tokens", "whole_prompt_tokens_after", "depth_target_met"}
MAIN_SWEEP_KEYS = {
    "status", "leaf_passes", "condensation_passes", "pre_leaf_condensation_passes",
    "pre_leaf_condensation_stop_reason", "post_drain_condensation_skipped", "total_passes",
    "duration_ms", "tokens_before", "tokens_after", "summary_prefix_tokens_before",
    "summary_prefix_tokens_after", "summary_prefix_target_tokens", "stop_reason", "budget_exhausted",
}


def _view(turns=5, *, with_tools=False):
    payload = payload_of(780)
    rows = [{"role": "system", "content": "system prompt"}]
    for index in range(turns):
        tag = f"{index:02d}"
        rows.append({"role": "user", "content": f"{tag} ask " + payload})
        if with_tools:
            rows.extend(tool_pair(f"call-{tag}", f"{tag} result " + payload))
        rows.append({"role": "assistant", "content": f"{tag} answer " + payload})
    return rows


def _leaf_tokens(view):
    next_user = next(i for i in range(2, len(view)) if view[i]["role"] == "user")
    return count_messages_tokens(view[1:next_user])


def _sweep_engine(make_engine, view, **overrides):
    depth_ratio = overrides.pop("depth_target_ratio", 0.0)
    prefix_ratio = overrides.pop("summary_prefix_target_ratio", 0.0)
    settings = {
        "leaf_chunk_tokens": _leaf_tokens(view),
        "dynamic_leaf_chunk_enabled": False,
        "fresh_tail_pressure_yield_enabled": False,
        "large_output_externalization_enabled": False,
        "large_output_active_replay_stubbing_enabled": False,
        "temporal_rollups_enabled": False,
        "reserve_tokens_floor": 0,
        "summary_prefix_target_tokens": 5_000,
        "l3_truncate_tokens": 2,
        **overrides,
    }
    engine = make_engine(**settings)
    engine._config.depth_target_ratio = depth_ratio
    engine._config.summary_prefix_target_ratio = prefix_ratio
    engine.context_length = WINDOW
    engine.threshold_tokens = count_messages_tokens(view) - 250
    return engine


def _target_after(view, leaves):
    saving = _leaf_tokens(view) - count_tokens(SUMMARY)
    return count_messages_tokens(view) - leaves * saving + 100


def _sweep(engine):
    return engine.get_status()["threshold_full_sweep"]


def _leaves(engine):
    return sorted(
        (node for node in engine._dag.get_session_nodes(engine._session_id) if node.depth == 0),
        key=lambda node: node.source_ids,
    )


@pytest.mark.parametrize("field", ["depth_target_ratio", "summary_prefix_target_ratio"])
@pytest.mark.parametrize("raw, expected", [
    (None, 0.0), ("0", 0.0), ("0.4", 0.4), ("0.999", 0.999),
    ("-0.1", 0.0), ("1", 0.0), ("1.5", 0.0), ("nan", 0.0), ("abc", 0.0),
])
def test_t1_ratio_config(monkeypatch, field, raw, expected):
    key = f"LCM_{field.upper()}"
    if raw is not None:
        monkeypatch.setenv(key, raw)
    config = LCMConfig.from_env()
    valid = raw in {"0", "0.4", "0.999"}
    assert getattr(LCMConfig(), field) == 0.0
    assert getattr(config, field) == expected
    assert config.config_sources[field] == (f"env:{key}" if valid else "default")
    assert any(key in warning for warning in config.config_source_warnings) == (raw is not None and not valid)


def test_t2_default_status_keys_match_main(make_engine, summaries):
    view = _view()
    engine = _sweep_engine(make_engine, view)

    engine.compress(view, current_tokens=count_messages_tokens(view))

    assert len(summaries) == len(_leaves(engine)) == 4
    status = _sweep(engine)
    assert set(status) == MAIN_SWEEP_KEYS | {"hidden_rows"}
    report = json.loads(lcm_status({}, engine=engine))
    assert report["threshold_full_sweep"] == status


def test_t3_depth_stop_leaves_later_rows_raw(make_engine, summaries):
    view = _view()
    target = _target_after(view, 2)
    engine = _sweep_engine(make_engine, view, depth_target_ratio=target / WINDOW)

    result = engine.compress(view, current_tokens=count_messages_tokens(view))

    assert len(summaries) == len(_leaves(engine)) == 2
    assert result[-6:] == view[-6:]
    assert _sweep(engine)["depth_target_met"] is True
    assert _sweep(engine)["depth_target_tokens"] == target


@pytest.mark.parametrize("host_figure, loaded, expected_leaves", [
    (True, False, 2), (True, True, 3), (False, True, 3), (False, False, 2),
])
def test_t4_whole_prompt_counts_fixed_load_once(make_engine, summaries, host_figure, loaded, expected_leaves):
    view = _view()
    counted = count_messages_tokens(view)
    fixed = _leaf_tokens(view) - count_tokens(SUMMARY) if loaded else 0
    engine = _sweep_engine(make_engine, view, depth_target_ratio=_target_after(view, 2) / WINDOW)
    if not host_figure and loaded:
        engine._survival_host_overhead(view, counted + fixed)  # earlier observation for this conversation

    result = engine.compress(view, **({"current_tokens": counted + fixed} if host_figure else {}))

    status = _sweep(engine)
    assert len(summaries) == len(_leaves(engine)) == expected_leaves
    assert status["fixed_load_tokens"] == fixed
    assert status["whole_prompt_tokens_after"] == engine._survival_measure(result) + fixed
    assert status["depth_target_met"] is True


def test_t5_soft_target_stops_before_second_call(make_engine, monkeypatch, clock):
    view = _view()
    engine = _sweep_engine(make_engine, view, depth_target_ratio=_target_after(view, 3) / WINDOW,
                           foreground_soft_seconds=60, summary_timeout_ms=60_000)
    provider = _provider(monkeypatch, clock, default=31.0)

    engine.compress(view, current_tokens=count_messages_tokens(view))

    assert len(provider.calls) == len(_leaves(engine)) == 1
    assert _sweep(engine)["stop_reason"] == "soft_target_reached"
    assert _sweep(engine)["depth_target_met"] is False


def test_t6_unreachable_target_drains_without_arming_hold(make_engine, summaries, caplog):
    view = _view()
    engine = _sweep_engine(make_engine, view, depth_target_ratio=0.008)
    fixed = 1_800

    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=count_messages_tokens(view) + fixed)

    assert len(summaries) == len(_leaves(engine)) == 4
    assert result[-2:] == view[-2:]
    assert _sweep(engine)["fixed_load_tokens"] == fixed
    assert _sweep(engine)["depth_target_met"] is False
    assert engine._no_progress_hold is None
    assert engine.should_compress(engine.threshold_tokens - 1) is False
    assert sum("depth target missed" in record.message for record in caplog.records) == 1


@pytest.mark.parametrize("shape", [
    "equal", "inverted", "unknown_window", "sweep_off", "force", "recovery", "cleanup", "overflow",
])
def test_t7_inert_depth_matches_disabled(make_engine, summaries, monkeypatch, caplog, shape):
    view = _view()
    monkeypatch.setattr(compaction.time, "perf_counter", lambda: 1.0)

    def run(ratio):
        engine = _sweep_engine(make_engine, view, depth_target_ratio=ratio, **(
            {"threshold_full_sweep_enabled": False} if shape == "sweep_off" else {}
        ))
        if shape == "unknown_window":
            engine.context_length = 0
        if shape in {"equal", "inverted"}:
            engine.threshold_tokens = 6_000
        if shape == "overflow":
            engine._config.max_assembly_tokens = 7_500
        if shape == "cleanup":
            # A real below-threshold preflight request: compress() takes the cleanup-only return.
            engine.threshold_tokens = count_messages_tokens(view) + 1
            assert engine._mark_below_threshold_maintenance() is True
        kwargs = {"force": {"force": True}, "recovery": {"bypass_cooldown": True}}.get(shape, {})
        result = engine.compress(view, current_tokens=count_messages_tokens(view), **kwargs)
        if shape == "cleanup":
            assert engine._last_compression_status == "sanitized"
            assert not summaries
            assert DEPTH_KEYS.isdisjoint(_sweep(engine) or {})
        return result, _sweep(engine)

    ratio = {"equal": 0.06, "inverted": 0.065}.get(shape, 0.04)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        assert run(ratio) == run(0.0)
    lines = [record for record in caplog.records if "depth target is at or above" in record.message]
    assert len(lines) == (1 if shape in {"equal", "inverted"} else 0)


@pytest.mark.parametrize("target, exits", [(120, False), (2_000, True)])
def test_t8_stub_first_uses_active_depth(make_engine, summaries, target, exits):
    view = tool_view()
    baseline = exit_engine(make_engine, temporal_rollups_enabled=False)
    baseline.compress(view, current_tokens=count_messages_tokens(view))
    assert baseline.get_status()["last_stub_first_exit"] is not None
    assert DEPTH_KEYS.isdisjoint(_sweep(baseline))
    engine = exit_engine(make_engine, temporal_rollups_enabled=False)
    engine._config.depth_target_ratio = target / 200_000

    engine.compress(view, current_tokens=count_messages_tokens(view))

    status = engine.get_status()
    if exits:
        assert status["last_stub_first_exit"]["target_tokens"] == target
        assert status["threshold_full_sweep"]["depth_target_met"] is True
        assert not summaries
    else:
        assert status["last_stub_first_exit"] is None
        assert summaries
    tool_report = json.loads(lcm_status({}, engine=engine))
    assert DEPTH_KEYS <= tool_report["threshold_full_sweep"].keys()
    assert tool_report["config"]["depth_target_ratio"] == target / 200_000


@pytest.mark.parametrize("circuit_open", [False, True], ids=["rejected-level-3", "open-circuit"])
def test_c4_zero_leaf_uses_stub_first_fallback(make_engine, monkeypatch, tmp_path, circuit_open):
    # Independent payload stores with the same reference names in both runs.
    monkeypatch.setattr(externalize, "time", SimpleNamespace(
        gmtime=lambda: time.gmtime(1_700_000_000), strftime=time.strftime, time_ns=lambda: 1, time=time.time,
    ))
    view = tool_view()
    observed = count_messages_tokens(view)
    baseline, engine = (exit_engine(
        make_engine, temporal_rollups_enabled=False, l3_truncate_tokens=2,
        large_output_externalization_path=str(tmp_path / f"payloads-{name}"),
    ) for name in ("ratio-zero", "depth-on"))
    engine._config.depth_target_ratio = math.nextafter(120 / engine.context_length, 1.0)
    provider = _circuit_provider(monkeypatch, None)
    if circuit_open:
        for instance in (baseline, engine):
            _open_circuit(instance)
            assert not instance._summary_route_available()

    expected = baseline.compress(view, current_tokens=observed)
    assert not provider.calls
    actual = engine.compress(view, current_tokens=observed)

    assert len(provider.calls) == (0 if circuit_open else 2)
    assert actual == expected and count_messages_tokens(actual) < observed
    assert not _leaves(engine)
    assert engine._no_progress_hold is None and not engine._sweep_budget_hold_active()
    assert engine.should_compress(observed) == baseline.should_compress(observed) is True
    assert engine._compress_commit_proof["stub_first"] is baseline._compress_commit_proof["stub_first"] is True
    assert engine._compress_commit_proof["output"] == baseline._compress_commit_proof["output"]
    assert engine.last_compression_status == baseline.last_compression_status == "sanitized"
    assert engine.get_status()["last_stub_first_exit"] == baseline.get_status()["last_stub_first_exit"]
    status = _sweep(engine)
    assert status["depth_target_tokens"] == 120 and status["depth_target_met"] is False
    assert status["whole_prompt_tokens_after"] == engine._survival_measure(actual) + status["fixed_load_tokens"]
    assert status["pre_leaf_condensation_passes"] == 0 and status["pre_leaf_condensation_stop_reason"] == ""
    status_fields = DEPTH_KEYS | {"duration_ms", "pre_leaf_condensation_passes", "pre_leaf_condensation_stop_reason"}
    assert {key: value for key, value in status.items() if key not in status_fields} == {
        key: value for key, value in _sweep(baseline).items() if key not in status_fields
    }


def test_c4_published_leaf_takes_precedence_over_stub_first_fallback(make_engine, monkeypatch, summaries):
    view = tool_view()
    engine = exit_engine(make_engine, temporal_rollups_enabled=False)
    engine._config.depth_target_ratio = 120 / engine.context_length
    prepare = engine._prepare_compression_sweep
    staged = []

    def record_prepared_fallback(state, *args, **kwargs):
        result = prepare(state, *args, **kwargs)
        staged.append(callable(getattr(state, "stub_first_fallback", None)))
        return result

    monkeypatch.setattr(engine, "_prepare_compression_sweep", record_prepared_fallback)

    result = engine.compress(view, current_tokens=count_messages_tokens(view))

    assert len(_leaves(engine)) == len(summaries) > 0
    assert engine.get_status()["last_stub_first_exit"] is None
    assert engine._compress_commit_proof["published"] is True
    assert engine._compress_commit_proof["stub_first"] is False
    assert _sweep(engine)["stop_reason"] != "stub_first_exit"
    assert result[-2:] == view[-2:]
    assert staged == [True]  # The free cuts were ready before the leaf, then discarded.


@pytest.fixture
def stable_payload_names(monkeypatch):
    monkeypatch.setattr(externalize, "time", SimpleNamespace(
        gmtime=lambda: time.gmtime(1_700_000_000), strftime=time.strftime, time_ns=lambda: 1, time=time.time,
    ))


@pytest.mark.parametrize("failure", ["publication-lock", "summarizer-exception"])
def test_c5_first_leaf_failure_keeps_free_cuts(
    make_engine, monkeypatch, summaries, tmp_path, stable_payload_names, caplog, failure,
):
    view = tool_view()
    observed = count_messages_tokens(view)
    baseline, engine = (exit_engine(
        make_engine, temporal_rollups_enabled=False,
        large_output_externalization_path=str(tmp_path / f"payloads-{name}"),
    ) for name in ("ratio-zero", "depth-on"))
    engine._config.depth_target_ratio = math.nextafter(120 / engine.context_length, 1.0)
    expected = baseline.compress(view, current_tokens=observed)
    assert not summaries
    failures = []
    if failure == "publication-lock":
        original_publish = engine._publish_leaf_node

        def lock_once(*args, **kwargs):
            if not failures:
                failures.append(failure)
                raise sqlite3.OperationalError("database is locked")
            return original_publish(*args, **kwargs)

        monkeypatch.setattr(engine, "_publish_leaf_node", lock_once)
    else:
        def fail_summary(**kwargs):
            failures.append(failure)
            raise RuntimeError("synthetic leaf summarizer failure")

        monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", fail_summary)

    actual = engine.compress(view, current_tokens=observed)

    assert failures == [failure] and not _leaves(engine)
    assert actual == expected and count_messages_tokens(actual) < observed
    assert engine._compress_commit_proof["stub_first"] is True
    assert engine._compress_commit_proof["output"] == baseline._compress_commit_proof["output"]
    assert engine.last_compression_status == baseline.last_compression_status == "sanitized"
    assert engine._no_progress_hold is None
    status = _sweep(engine)
    assert status["stop_reason"] == "stub_first_exit" and status["depth_target_met"] is False
    assert status["tokens_after"] == _sweep(baseline)["tokens_after"]
    if failure == "summarizer-exception":
        _assert_leaf_exception_warning(caplog)


def _assert_leaf_exception_warning(caplog):
    warnings = [record.getMessage() for record in caplog.records
                if record.name == compaction.__name__ and record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "leaf_summary_error" in warnings[0] and "RuntimeError" in warnings[0]


@pytest.mark.parametrize("stage", ["_build_leaf_publication", "_run_pre_compaction_extraction", "_publish_leaf_node"])
def test_c6_first_leaf_exception_keeps_free_cuts(
    make_engine, monkeypatch, summaries, tmp_path, stable_payload_names, caplog, stage,
):
    view = tool_view()
    observed = count_messages_tokens(view)
    baseline, engine = (exit_engine(
        make_engine, temporal_rollups_enabled=False, extraction_enabled=stage == "_run_pre_compaction_extraction",
        large_output_externalization_path=str(tmp_path / f"payloads-{name}"),
    ) for name in ("ratio-zero", "depth-on"))
    engine._config.depth_target_ratio = math.nextafter(120 / engine.context_length, 1.0)
    expected = baseline.compress(view, current_tokens=observed)
    assert not summaries
    failures = []

    def fail(*args, **kwargs):
        failures.append(stage)
        raise RuntimeError("synthetic first-leaf failure")

    monkeypatch.setattr(engine, stage, fail)
    actual = engine.compress(view, current_tokens=observed)

    assert failures == [stage]
    assert actual == expected and count_messages_tokens(actual) < observed
    assert engine._compress_commit_proof["stub_first"] is True
    assert not _leaves(engine) and engine._no_progress_hold is None
    _assert_leaf_exception_warning(caplog)


@pytest.mark.parametrize("error, stop", [
    (LifecycleBindingChangedError, "lifecycle_binding_changed"),
    (LifecyclePublicationConflictError, "publication_invariant_conflict"),
])
def test_c6_publication_lifecycle_failure_keeps_fail_open(make_engine, monkeypatch, summaries, error, stop):
    view = tool_view()
    engine = exit_engine(make_engine, temporal_rollups_enabled=False)
    engine._config.depth_target_ratio = math.nextafter(120 / engine.context_length, 1.0)

    def fail(*args, **kwargs):
        raise error("synthetic lifecycle failure")

    monkeypatch.setattr(engine, "_publish_leaf_node", fail)
    assert engine.compress(view, current_tokens=count_messages_tokens(view)) == view
    assert len(summaries) == 1 and not _leaves(engine)
    assert _sweep(engine)["stop_reason"] == stop
    assert engine.last_compression_status == "error"


def test_c6_hung_fallback_matches_real_ratio_zero(make_engine, monkeypatch, clock, tmp_path, stable_payload_names):
    view = tool_view()
    observed = count_messages_tokens(view)
    baseline, engine = (exit_engine(
        make_engine, temporal_rollups_enabled=False, summary_model="m1", summary_timeout_ms=120_000,
        l3_truncate_tokens=2, large_output_externalization_path=str(tmp_path / f"payloads-{name}"),
    ) for name in ("ratio-zero", "depth-on"))
    engine._config.depth_target_ratio = math.nextafter(120 / engine.context_length, 1.0)
    calls = _hung_provider(monkeypatch, clock)

    for pass_number in (1, 2):
        expected = baseline.compress(view, current_tokens=observed)
        assert len(calls) == pass_number - 1
        actual = engine.compress(view, current_tokens=observed)
        assert actual == expected and count_messages_tokens(actual) < observed
        assert [route for route, _ in calls] == ["m1"]
        assert _sweep(engine)["stop_reason"] == "stub_first_exit"
        assert ("depth_target_tokens" in _sweep(engine)) == (pass_number == 1)
        assert engine._hung_summary_strike == (engine._hold_conversation_key(), "m1")
        for instance in (baseline, engine):
            assert instance._sweep_budget_hold_until == 0 and not instance._sweep_budget_hold_turn_paced
            assert instance._no_progress_hold is None and instance.should_compress(observed)
            instance.note_turn_complete()
            assert instance.should_compress(observed)

    clock.offset += engine._summary_circuit_breaker.cooldown_seconds + 1
    assert engine.compress(view, current_tokens=observed) == expected
    assert [route for route, _ in calls] == ["m1", "m1"]
    assert "depth_target_tokens" in _sweep(engine) and not _leaves(engine)


@pytest.mark.parametrize("depth_ratio", [0.0, 0.001])
def test_c6_two_hangs_still_arm_hold_when_free_cuts_miss(make_engine, monkeypatch, clock, depth_ratio):
    view = _view(turns=14)
    observed = count_messages_tokens(view)
    engine = exit_engine(
        make_engine, temporal_rollups_enabled=False, summary_model="m1", summary_timeout_ms=120_000,
        l3_truncate_tokens=2,
    )
    engine._config.depth_target_ratio = depth_ratio
    calls = _hung_provider(monkeypatch, clock)
    for turn_paced in (True, False):
        assert engine.compress(view, current_tokens=observed) == view
        assert engine._hung_summary_strike == (engine._hold_conversation_key(), "m1")
        assert engine._sweep_budget_hold_active() and engine._sweep_budget_hold_turn_paced is turn_paced
        assert not engine.should_compress(observed)
        engine.note_turn_complete()
    assert [route for route, _ in calls] == ["m1", "m1"]
    assert engine._sweep_budget_hold_until - time.monotonic() == pytest.approx(600, abs=1)


def _seed_oversized_summary_prefix(engine):
    for index in range(2):
        engine._dag.add_node(SummaryNode(
            session_id=engine._session_id, depth=0, summary=f"Earlier group {index}. " + payload_of(500),
            token_count=500, source_token_count=1000, source_ids=[], source_type="messages",
            created_at=float(index + 1),
        ))


def test_c6_pre_leaf_condensation_lock_keeps_free_cuts(
    make_engine, monkeypatch, summaries, tmp_path, stable_payload_names,
):
    view = tool_view()
    observed = count_messages_tokens(view)
    baseline, engine = (exit_engine(
        make_engine, temporal_rollups_enabled=False, summary_prefix_target_tokens=100, condensation_fanin=2,
        large_output_externalization_path=str(tmp_path / f"payloads-{name}"),
    ) for name in ("ratio-zero", "depth-on"))
    for instance in (baseline, engine):
        _seed_oversized_summary_prefix(instance)
    engine._config.depth_target_ratio = math.nextafter(120 / engine.context_length, 1.0)
    expected = baseline.compress(view, current_tokens=observed)
    locks = []

    def lock(*args, **kwargs):
        locks.append(True)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(engine, "_condense_summary_nodes", lock)
    actual = engine.compress(view, current_tokens=observed)

    assert locks == [True] and not summaries
    assert actual == expected and count_messages_tokens(actual) < observed
    assert engine._compress_commit_proof["stub_first"] is True
    assert engine.last_compression_status == baseline.last_compression_status == "sanitized"
    status = _sweep(engine)
    assert status["stop_reason"] == "stub_first_exit" and status["depth_target_met"] is False
    assert status["condensation_passes"] == status["pre_leaf_condensation_passes"] == status["total_passes"] == 0


def test_c5_stub_first_fallback_reports_pre_leaf_condensation(make_engine, monkeypatch):
    engine = exit_engine(make_engine, temporal_rollups_enabled=False, l3_truncate_tokens=2,
                         summary_prefix_target_tokens=100, condensation_fanin=2)
    engine._config.depth_target_ratio = math.nextafter(120 / engine.context_length, 1.0)
    _seed_oversized_summary_prefix(engine)
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs["depth"])
        return (SUMMARY, 1) if kwargs["depth"] else ("Incomplete", 3)

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
    view = tool_view()
    engine.compress(view, current_tokens=count_messages_tokens(view))

    status = _sweep(engine)
    assert calls == [1, 0] and status["stop_reason"] == "stub_first_exit"
    assert status["leaf_passes"] == 0
    assert status["condensation_passes"] == status["pre_leaf_condensation_passes"] == status["total_passes"] == 1
    assert status["pre_leaf_condensation_stop_reason"] == "summary_prefix_target_reached"
    assert status["summary_prefix_tokens_after"] == engine._summary_frontier_tokens() < status["summary_prefix_tokens_before"]


def test_c4_depth_retry_adopts_cancelled_prefix(make_engine, monkeypatch):
    view = _view(7)
    observed = count_messages_tokens(view)
    resumed = _spy_resumes(monkeypatch)

    def retry(ratio):
        engine = _sweep_engine(make_engine, view, depth_target_ratio=ratio)
        calls = 0

        def summarize(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise _HostCancel()
            return SUMMARY, 1

        monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
        with pytest.raises(_HostCancel):
            engine.compress(view, current_tokens=observed)
        committed = _leaves(engine)
        assert len(committed) == 2
        result = engine.compress(view, current_tokens=observed)
        leaves = _leaves(engine)
        assert [leaf.node_id for leaf in leaves[:2]] == [leaf.node_id for leaf in committed]
        assert calls == len(leaves) + 1
        assert result[-2:] == view[-2:]
        return [leaf.source_ids for leaf in leaves], _sweep(engine)

    baseline, _ = retry(0)
    leaves, status = retry(_target_after(view, 3) / WINDOW)
    assert [count for count in resumed if count] == [4, 4]
    assert len(leaves) == 3 < len(baseline) and leaves == baseline[:3]
    assert status["leaf_passes"] == 1 and status["depth_target_met"] is True


def test_c4_depth_stop_covers_hidden_backlog_and_keeps_host_rows(make_engine, summaries):
    history = _view(8)
    host = history[5:]
    observed = count_messages_tokens(host)

    def run(ratio):
        engine = _sweep_engine(make_engine, history, depth_target_ratio=ratio)
        engine.threshold_tokens = observed - 250
        engine.ingest(history)
        before = _stored_row_hashes(engine)
        host_ids = [row["store_id"] for row in engine._store.get_session_messages(engine._session_id)][5:]
        assert engine._store_complete_backlog(host, 0).rows > 0

        result = engine.compress(host, current_tokens=observed)

        leaves = [leaf.source_ids for leaf in _leaves(engine)]
        covered = {store_id for leaf in leaves for store_id in leaf}
        returned = [(row["role"], engine._generated_context_carrier_remainder(row) or row["content"]) for row in result]
        assert all(store_id in covered or (row["role"], row["content"]) in returned
                   for store_id, row in zip(host_ids, host, strict=True))
        assert engine._compress_hidden_rows_consumed > 0
        assert engine._store_complete_backlog(result, 0).rows == 0
        engine.ingest(result)
        assert _stored_row_hashes(engine) == before
        return leaves, _sweep(engine)

    baseline, _ = run(0)
    leaves, status = run(_target_after(host, 2) / WINDOW)
    assert 0 < len(leaves) < len(baseline) and leaves == baseline[:len(leaves)]
    assert status["depth_target_met"] is True


@pytest.mark.parametrize("window, ratio, tokens, expected", [
    (272_000, 0.07, 0, 19_040), (1_000_000, 0.07, 0, 70_000), (272_000, 0.07, 800, 800),
    (32_000, 0, 0, 875), (272_000, 0, 0, 875), (1_000_000, 0, 0, 875),
    (0, 0.07, 0, 875), (272_000, 0.000001, 0, 1),
])
def test_t9_prefix_budget_precedence(make_engine, summaries, window, ratio, tokens, expected):
    view = _view()
    engine = _sweep_engine(make_engine, view, leaf_chunk_tokens=2_000,
                           summary_prefix_target_ratio=ratio, summary_prefix_target_tokens=tokens)
    engine.context_length = window
    engine.threshold_tokens = 3_500

    engine.compress(view, current_tokens=count_messages_tokens(view))

    assert _sweep(engine)["summary_prefix_target_tokens"] == expected
    assert json.loads(lcm_status({}, engine=engine))["config"]["summary_prefix_target_ratio"] == ratio


def _stored_row_hashes(engine):
    return {
        row["store_id"]: hashlib.sha256(row["content"].encode()).hexdigest()
        for row in engine._store.get_session_messages(engine._session_id)
    }


def test_t10_deep_pass_preserves_rows_provenance_and_tool_groups(make_engine, summaries):
    view = _view(40, with_tools=True)
    engine = _sweep_engine(make_engine, view, depth_target_ratio=_target_after(view, 5) / WINDOW,
                           max_assembly_tokens=150_000, survival_fit=False)
    engine.ingest(view)
    before = _stored_row_hashes(engine)

    result = engine.compress(view, current_tokens=count_messages_tokens(view))

    leaves = _leaves(engine)
    assert len(leaves) == len(summaries) == 5
    covered = [sid for leaf in leaves for sid in leaf.source_ids]
    assert covered == list(before)[1:1 + 4 * len(leaves)]  # start after the system anchor
    assert engine._store_complete_frontier() == covered[-1]
    assert _stored_row_hashes(engine) == before
    assert result[-(4 * (40 - len(leaves))):] == view[1 + len(covered):]
    rows = engine._store.get_session_messages(engine._session_id)
    for leaf in leaves:
        selected = [row for row in rows if row["store_id"] in leaf.source_ids]
        assert [call["id"] for row in selected for call in row.get("tool_calls") or []] == [
            row["tool_call_id"] for row in selected if row["role"] == "tool"
        ]
    engine.ingest(result)
    assert _stored_row_hashes(engine) == before


def test_t11_condensation_runs_after_depth_stop_with_raw_rows_left(make_engine, summaries):
    view = _view()
    engine = _sweep_engine(make_engine, view, depth_target_ratio=_target_after(view, 2) / WINDOW,
                           summary_prefix_target_tokens=count_tokens(SUMMARY) + 2, condensation_fanin=2)

    result = engine.compress(view, current_tokens=count_messages_tokens(view))

    status = _sweep(engine)
    assert status["leaf_passes"] == 2 and result[-6:] == view[-6:]
    assert status["pre_leaf_condensation_passes"] == 0 < status["condensation_passes"]
    assert status["stop_reason"] == "summary_prefix_target_reached"
    assert status["status"] == "completed" and status["depth_target_met"] is True


def _seed_prepared_chain(engine, view, monkeypatch, count):
    cache = leaf_prep._LeafPrep()
    monkeypatch.setattr(leaf_prep, "_LEAF_PREP", cache)
    engine.ingest(view)
    rows = engine._store.get_session_messages(engine._session_id)
    start = engine._store_complete_frontier()
    chain = []
    for index in range(count):
        pair = rows[1 + index * 2:3 + index * 2]
        ids = tuple(row["store_id"] for row in pair)
        chain.append(leaf_prep.PreparedLeaf(
            session_id=engine._session_id, conversation_id=engine._conversation_id,
            start_frontier=start, covered_store_ids=ids,
            coverage_digest=leaf_prep.committed_coverage_digest(engine._store._conn, ids),
            route_fingerprint=leaf_prep.route_fingerprint(engine._config), summary=SUMMARY,
            level=1, model="synthetic", source_tokens=count_messages_tokens(pair),
            summary_tokens=count_tokens(SUMMARY), excluded_store_ids=(rows[0]["store_id"],) if not index else (),
        ))
        start = ids[-1]
    cache._cache[(str(engine._store.db_path), engine._conversation_id)] = chain


@pytest.mark.parametrize("trigger, start, prepared", [(0.50, 62_500, 16), (0.75, 80_000, 30), (0.75, 80_000, 24)])
def test_t12_prepared_leaves_continue_to_depth(make_engine, summaries, monkeypatch, trigger, start, prepared):
    view = _view(31)
    engine = _sweep_engine(make_engine, view, depth_target_ratio=0.40, background_leaf_prep_enabled=True)
    engine.threshold_tokens = int(trigger * WINDOW)
    _seed_prepared_chain(engine, view, monkeypatch, prepared)
    expected = math.ceil((start - 40_000) / (_leaf_tokens(view) - count_tokens(SUMMARY)))
    assert 12 < expected < 30 and count_messages_tokens(view) <= start

    engine.compress(view, current_tokens=start)

    status = _sweep(engine)
    assert status["leaf_passes"] == expected
    assert status["promoted_leaves"] == min(expected, prepared)
    assert len(summaries) == expected - status["promoted_leaves"]
    assert status["whole_prompt_tokens_after"] < engine.threshold_tokens
    assert status["depth_target_met"] is True


def test_t13_hold_fingerprint_includes_each_ratio_only_when_set(make_engine, monkeypatch):
    engine = make_engine()
    view = _view()
    engine.ingest(view)
    engine.should_compress(count_messages_tokens(view))
    original_dumps = json.dumps
    payloads = []

    def capture(value, **kwargs):
        payloads.append(value)
        return original_dumps(value, **kwargs)

    monkeypatch.setattr(json, "dumps", capture)
    baseline = engine._no_progress_fingerprint()
    assert baseline
    default_config = payloads[-1][-1]
    assert len(default_config) == 44  # main's ordered config values
    engine._config.depth_target_ratio = 0.4
    depth = engine._no_progress_fingerprint()
    engine._config.depth_target_ratio = 0.0
    engine._config.summary_prefix_target_ratio = 0.4
    prefix = engine._no_progress_fingerprint()
    assert len({baseline, depth, prefix}) == 3
    engine._config.summary_prefix_target_ratio = 0.0
    assert engine._no_progress_fingerprint() == baseline


def test_deep_pass_keeps_committed_leaves_after_a_third_leaf_lock(make_engine, monkeypatch, caplog):
    view = _view(8)
    engine = _sweep_engine(make_engine, view, depth_target_ratio=_target_after(view, 5) / WINDOW)
    engine.ingest(view)
    before = _stored_row_hashes(engine)
    publish = engine._publish_leaf_node
    calls = []

    def summarize(**kwargs):
        assert not engine._store._conn.in_transaction
        calls.append(kwargs["text"])
        return SUMMARY, 1

    def publish_or_lock(*args, **kwargs):
        if len(calls) == 3:
            raise sqlite3.OperationalError("database is locked")
        return publish(*args, **kwargs)

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
    monkeypatch.setattr(engine, "_publish_leaf_node", publish_or_lock)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=count_messages_tokens(view))

    assert len(calls) == 3 and len(_leaves(engine)) == 2
    assert result[-12:] == view[-12:]
    assert _stored_row_hashes(engine) == before
    assert _sweep(engine)["stop_reason"] == "sqlite_publication_locked"
    assert _sweep(engine)["depth_target_met"] is False
