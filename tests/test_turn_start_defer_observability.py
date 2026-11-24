"""Operator diagnostics for the opt-in turn-start defer gate (#3075)."""

import json
import logging
from pathlib import Path

import pytest

from hermes_lcm import command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from tests.test_command_status import status_engine


WINDOW = 1_000_000
TRIGGER = 500_000


@pytest.fixture
def make_engine(tmp_path):
    engines = []

    def create(ceiling=0.70, **options):
        root = tmp_path / str(len(engines))
        root.mkdir()
        config = LCMConfig(
            database_path=str(root / "lcm.db"), context_threshold=0.5,
            turn_start_defer_ceiling=ceiling, temporal_rollups_enabled=False, embeddings_enabled=False, **options,
        )
        eng = LCMEngine(config=config, hermes_home=str(root))
        engines.append(eng)
        return eng

    yield create
    for eng in engines:
        eng.shutdown()


def bind(eng, session="session", **kwargs):
    eng.on_session_start(session, platform="cli", context_length=WINDOW, **kwargs)


def warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "defer ceiling" in r.getMessage()]


def deferral_levels(caplog):
    return [r.levelno for r in caplog.records if r.getMessage().startswith("LCM automatic compaction deferred:")]


def defer(eng):
    eng.update_from_response({"stop_reason": "tool_use", "prompt_tokens": TRIGGER})
    assert eng.should_compress() is False


def test_a_inert_absolute_trigger_warns_once_across_binds(make_engine, caplog):
    eng = make_engine(absolute_threshold_tokens=750_000)
    bind(eng)
    assert eng.threshold_tokens == 750_000
    assert len(warnings(caplog)) == 1
    bind(eng, "next-session")
    message, = warnings(caplog)
    assert "the defer ceiling is at or below the compaction trigger; the gate never defers" in message
    assert "ceiling_tokens=700000" in message and "threshold_tokens=750000" in message


def test_b_absolute_trigger_warns_for_each_distinct_window(make_engine, caplog):
    eng = make_engine(absolute_threshold_tokens=130_000)
    bind(eng)
    eng.update_model("wide", WINDOW)
    message, = warnings(caplog)
    assert "unanchored" in message
    assert "ceiling_tokens=700000" in message and "threshold_tokens=130000" in message

    eng.update_model("narrow", 200_000)
    eng.update_model("wide", WINDOW)
    eng.update_model("narrow", 200_000)
    first, second = warnings(caplog)
    assert first == message
    assert "unanchored" in second
    assert "ceiling_tokens=140000" in second and "threshold_tokens=130000" in second


@pytest.mark.parametrize(("model", "trigger"), [("ordinary", TRIGGER), ("special", 600_000)])
def test_c_window_anchored_ratio_is_quiet(make_engine, caplog, model, trigger):
    eng = make_engine(model_thresholds={"special": 0.6})
    eng.update_model(model, WINDOW)
    bind(eng, model=model)
    assert eng.threshold_tokens == trigger
    assert eng.should_compress(trigger) is True
    eng.update_from_response({"stop_reason": "tool_use"})
    assert eng.should_compress(trigger) is False
    assert warnings(caplog) == []


@pytest.mark.parametrize("model", ["ordinary", "special"])
def test_assembly_cap_at_the_ratio_trigger_makes_the_ceiling_inert(make_engine, caplog, model):
    eng = make_engine(model_thresholds={"special": 0.6}, max_assembly_tokens=400_000)
    eng.update_model(model, WINDOW)
    bind(eng, model=model)
    assert eng.threshold_tokens == 400_000
    eng.update_from_response({"stop_reason": "tool_use"})
    assert eng.should_compress(399_999) is False
    assert eng.should_compress(400_000) is True  # The assembly cap forces overflow recovery.
    message, = warnings(caplog)
    assert "is inert:" in message
    assert "(ceiling_tokens=400000 threshold_tokens=400000" in message
    assert "assembly_cap=400000 limits deferral" in message


def test_assembly_cap_below_the_absolute_trigger_makes_the_ceiling_inert(make_engine, caplog):
    eng = make_engine(absolute_threshold_tokens=500_000, max_assembly_tokens=400_000)
    bind(eng)
    eng.update_from_response({"stop_reason": "tool_use"})
    assert eng.should_compress(399_999) is False
    assert eng.should_compress(400_000) is True
    message, = warnings(caplog)
    assert "is inert:" in message
    assert "(ceiling_tokens=400000 threshold_tokens=500000" in message
    assert "assembly_cap=400000 limits deferral" in message


def test_assembly_cap_between_a_lowered_trigger_and_the_ceiling_limits_deferral(make_engine, caplog):
    eng = make_engine(max_assembly_tokens=650_000, reserve_tokens_floor=400_000)
    bind(eng)
    assert warnings(caplog) == []
    eng.threshold_tokens = 128_000  # Hermes lowers the trigger after bind.
    eng.update_from_response({"stop_reason": "tool_use"})
    assert eng.should_compress(599_999) is False
    assert eng.should_compress(600_000) is True
    message, = warnings(caplog)
    assert "is unanchored:" in message
    assert "(ceiling_tokens=600000 threshold_tokens=128000" in message
    assert "assembly_cap=600000 limits deferral" in message


def test_host_auxiliary_clamp_after_bind_warns_once_at_the_lowered_trigger(make_engine, caplog):
    eng = make_engine()
    eng.update_model("wide", WINDOW)
    bind(eng, model="wide")
    assert eng.threshold_tokens == TRIGGER
    assert warnings(caplog) == []
    # Hermes turn preflight applies the auxiliary cap after model update and session bind.
    eng.threshold_tokens = 128_000
    assert eng.should_compress(127_999) is False
    message, = warnings(caplog)
    assert "unanchored" in message
    assert "ceiling_tokens=700000" in message and "threshold_tokens=128000" in message
    eng.update_from_response({"stop_reason": "tool_use"})
    for _ in range(3):
        assert eng.should_compress(128_000) is False
    assert warnings(caplog) == [message]


def test_inert_warning_is_reevaluated_after_host_auxiliary_clamp(make_engine, caplog):
    eng = make_engine(model_thresholds={"wide": 0.8})
    eng.update_model("wide", WINDOW)
    bind(eng, model="wide")
    assert eng.threshold_tokens == 800_000
    inert, = warnings(caplog)
    assert "inert" in inert
    assert "ceiling_tokens=700000" in inert and "threshold_tokens=800000" in inert
    eng.threshold_tokens = 128_000
    eng.update_from_response({"stop_reason": "tool_use"})
    for _ in range(3):
        assert eng.should_compress(128_000) is False
    first, unanchored = warnings(caplog)
    assert first == inert
    assert "unanchored" in unanchored
    assert "ceiling_tokens=700000" in unanchored and "threshold_tokens=128000" in unanchored


def test_quiet_key_can_warn_when_a_different_model_is_clamped_to_it(make_engine, caplog):
    eng = make_engine(model_thresholds={"special": 0.8})
    bind(eng, model="ordinary")
    assert eng.should_compress(TRIGGER) is True
    assert warnings(caplog) == []

    eng.update_model("special", WINDOW)
    assert eng.threshold_tokens == 800_000
    eng.threshold_tokens = TRIGGER
    assert eng.should_compress(TRIGGER) is True
    inert, unanchored = warnings(caplog)
    assert "inert" in inert
    assert "unanchored" in unanchored and "threshold_tokens=500000" in unanchored


def test_seen_warning_key_skips_the_lock(make_engine, caplog, monkeypatch):
    eng = make_engine(absolute_threshold_tokens=750_000)
    bind(eng)
    message, = warnings(caplog)
    with monkeypatch.context() as patch:
        patch.setattr(eng, "_compaction_gate", None)
        eng._warn_turn_start_defer_ceiling()
    assert warnings(caplog) == [message]


def test_seen_quiet_key_skips_classification_and_the_lock(make_engine, caplog, monkeypatch):
    eng = make_engine()
    bind(eng)
    assert warnings(caplog) == []

    def unexpected_classification(*args):
        pytest.fail("a seen quiet key recalculated the ceiling")

    with monkeypatch.context() as patch:
        patch.setattr(eng, "_turn_start_defer_ceiling_tokens", unexpected_classification)
        patch.setattr(eng, "_compaction_gate", None)
        assert eng.should_compress(TRIGGER - 1) is False
    assert warnings(caplog) == []


def test_bind_rechecks_trigger_after_host_auxiliary_clamp(make_engine, caplog):
    eng = make_engine(absolute_threshold_tokens=750_000)
    eng.update_model("wide", WINDOW)
    caplog.clear()
    # Hermes #1223 applies this cap directly to the engine after update_model().
    eng.threshold_tokens = 130_000
    bind(eng, model="wide")
    assert eng.threshold_tokens == 130_000
    message, = warnings(caplog)
    assert "unanchored" in message
    assert "ceiling_tokens=700000" in message and "threshold_tokens=130000" in message


def test_d_deferral_episode_logs_once_and_status_counts_every_gate_check(make_engine, caplog):
    eng = make_engine()
    bind(eng)
    with caplog.at_level(logging.DEBUG):
        for _ in range(100):
            defer(eng)
            assert eng.should_compress() is False
    assert deferral_levels(caplog) == [logging.INFO] + [logging.DEBUG] * 199
    lines = command.handle_lcm_command("status", eng).splitlines()
    start = lines.index("threshold_tokens: 500000")
    assert lines[start + 1:start + 4] == [
        "turn_start_defer_ceiling: 0.7",
        "turn_start_defer_ceiling_tokens: 700000",
        "turn_start_deferred_checks: 200",
    ]
    assert eng.get_status()["turn_start_deferred_checks"] == 200


def test_bypassed_check_preserves_the_owner_episode(make_engine, caplog):
    eng = make_engine()
    bind(eng)
    with caplog.at_level(logging.DEBUG):
        defer(eng)
        eng._mark_thread_context_stateless("child")
        try:
            assert eng.should_compress(TRIGGER) is True
        finally:
            eng._clear_thread_context_stateless("child")
        assert eng.should_compress(TRIGGER) is False
    assert deferral_levels(caplog) == [logging.INFO, logging.DEBUG]


def test_status_ceiling_tokens_are_uninitialized_before_bind(make_engine):
    eng = make_engine()
    eng.update_model("wide", WINDOW)
    lines = command.handle_lcm_command("status", eng).splitlines()
    assert "threshold_tokens: (uninitialized)" in lines
    assert "turn_start_defer_ceiling_tokens: (uninitialized)" in lines


def test_e_idless_host_starts_a_new_episode_at_turn_completion(make_engine, caplog, monkeypatch):
    eng = make_engine()
    bind(eng)
    monkeypatch.setattr(eng, "_relay_current_turn", None)
    assert eng._host_turn_id_for_deferral() == ""
    with caplog.at_level(logging.DEBUG):
        defer(eng)
        defer(eng)
        eng.note_turn_complete()
        defer(eng)
        defer(eng)
    assert deferral_levels(caplog) == [logging.INFO, logging.DEBUG, logging.INFO, logging.DEBUG]


@pytest.mark.parametrize("boundary", ["final_response", "on_turn_complete", "compress", "gate_miss"])
def test_an_episode_ends_at_phase_clear_or_gate_miss(make_engine, caplog, boundary):
    eng = make_engine()
    bind(eng)
    with caplog.at_level(logging.DEBUG):
        defer(eng)
        defer(eng)
        if boundary == "final_response":
            eng.update_from_response({"stop_reason": "end_turn"})
        elif boundary == "on_turn_complete":
            eng.on_turn_complete([])
        elif boundary == "compress":
            eng.compress([], current_tokens=0)
        else:
            assert eng.should_compress(700_000) is True
        defer(eng)
    assert deferral_levels(caplog) == [logging.INFO, logging.DEBUG, logging.INFO]


@pytest.mark.parametrize("boundary", ["reset", "rotation"])
def test_f_counter_resets_at_session_boundaries(make_engine, boundary):
    eng = make_engine()
    bind(eng)
    defer(eng)
    assert "turn_start_deferred_checks: 1" in command.handle_lcm_command("status", eng).splitlines()
    if boundary == "reset":
        eng.on_session_reset()
        assert eng.should_compress(TRIGGER) is True
    else:
        eng.on_session_start("rotated", boundary_reason="compression", old_session_id="session", platform="cli")
        assert eng.current_session_id == "rotated"
    assert "turn_start_deferred_checks: 0" in command.handle_lcm_command("status", eng).splitlines()
    if boundary == "rotation":
        assert eng.should_compress(TRIGGER) is False  # The continued tool turn keeps its phase.


def test_g_off_is_quiet_and_preserves_status_bytes(make_engine, caplog, tmp_path, monkeypatch):
    eng = make_engine(ceiling=0, absolute_threshold_tokens=TRIGGER)
    with caplog.at_level(logging.DEBUG):
        bind(eng)
        eng.update_model("wide", WINDOW)
        eng.update_from_response({"stop_reason": "tool_use"})
        assert eng.should_compress(TRIGGER) is True
    assert warnings(caplog) == []
    assert deferral_levels(caplog) == []

    default_engine = status_engine(tmp_path, "bound", monkeypatch)
    default_engine._config.turn_start_defer_ceiling = eng._config.turn_start_defer_ceiling
    expected = json.loads((Path(__file__).parent / "fixtures" / "command_status.json").read_text())
    assert command.handle_lcm_command("status", default_engine).replace(str(tmp_path), "<tmp>") == expected["bound"]
    status_text = command.handle_lcm_command("status", eng).replace(str(tmp_path), "<tmp>")
    assert not any("turn_start_defer" in line for line in status_text.splitlines())
    assert eng._turn_start_deferred_checks == 0
