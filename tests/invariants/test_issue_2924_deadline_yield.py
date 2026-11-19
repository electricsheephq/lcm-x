"""#2924 Step A: deadline yields preserve committed ingest (I1, I2, I11)."""

import hashlib
from copy import deepcopy
from unittest.mock import Mock

import pytest

from hermes_lcm import compaction as compaction_module
from hermes_lcm import engine as engine_module
from hermes_lcm.config import LCMConfig
from tests.harness import engine_context, pin_char_counter, stored_rows
from tests.test_issue_2863_host_contract import Host, ModelClock
from tests.test_issue_581_582_store_complete_survival import _turn


@pytest.fixture
def engine(tmp_path, monkeypatch):
    pin_char_counter(monkeypatch)
    monkeypatch.setattr(engine_module, "summarize_with_escalation", lambda **_: (
        "Earlier turns.\nExpand for details about: synthetic turns", 1))
    with engine_context(
        tmp_path, window=200_000, conversation_id="C", context_threshold=0.001,
        fresh_tail_count=2, leaf_chunk_tokens=400, summary_model="probe",
        embeddings_enabled=False, threshold_full_sweep_enabled=True,
    ) as eng:
        yield eng


@pytest.fixture
def view():
    return [row for i in range(8) for row in _turn(f"T{i}", 100.0 + i, tool=i == 0)]


def test_t1_slow_ingest_yields_after_commit(engine, view, monkeypatch, caplog):
    engine._config.foreground_hard_seconds = 2
    view[0]["content"] += "\ud800"  # The canonical view is a different list.
    ingest = engine._ingest_messages
    now = 100.0

    def slow(*args, **kwargs):
        nonlocal now
        result = ingest(*args, **kwargs)
        now += 3
        return result

    monkeypatch.setattr(engine, "_ingest_messages", slow)
    sweep = Mock(wraps=engine._prepare_compression_sweep)
    monkeypatch.setattr(engine, "_prepare_compression_sweep", sweep)
    before = deepcopy(view)
    with monkeypatch.context() as patch, caplog.at_level("INFO", logger="hermes_lcm.compaction"):
        patch.setattr(compaction_module.time, "monotonic", lambda: now)
        result = engine.compress(view, current_tokens=8000)
    assert engine.last_compression_status == "deadline_yield"
    assert result is view and view == before
    sweep.assert_not_called()
    assert engine._ingest_cursor == len(view)
    assert engine._foreground_budget is None
    assert sum("check=after_ingest" in line for line in caplog.messages) == 1


def _late_ingest(engine, monkeypatch):
    clock = ModelClock(monkeypatch)
    ingest = engine._ingest_messages

    def slow(*args, **kwargs):
        result = ingest(*args, **kwargs)
        clock.offset += 121
        return result

    monkeypatch.setattr(engine, "_ingest_messages", slow)
    return clock


@pytest.mark.parametrize("mode", ["forced", "recovery", "overflow"])
def test_slow_ingest_still_fits_forced_and_recovery_passes(engine, view, monkeypatch, mode):
    engine.context_length = 4000
    engine.threshold_tokens = 500
    engine._config.survival_reserve = 0.875
    if mode == "overflow":
        engine._config.max_assembly_tokens = 500
    observed = engine._survival_measure(view)
    assert 500 < observed < engine.context_length
    _late_ingest(engine, monkeypatch)

    result = engine.compress(view, current_tokens=observed, force=mode == "forced", bypass_cooldown=mode == "recovery")

    assert engine.last_compression_status != "deadline_yield"
    fit_cap = 499 if mode == "recovery" else 500
    assert result and engine._survival_measure(result) <= fit_cap
    assert _digests(stored_rows(engine)) == _digests(view)


@pytest.mark.parametrize("held", [True, False], ids=["hold-fit-only", "default-caps-at-ceiling"])
def test_slow_ingest_still_fits_automatic_passes(engine, monkeypatch, held):
    engine.context_length = 6_000
    ceiling = engine._survival_ceiling()
    assert ceiling == 5_100
    view = [row for i in range(40 if held else 16) for row in _turn(f"T{i}", 100.0 + i)]
    measured = engine._survival_measure(view)
    assert measured > ceiling
    if held:
        engine._config.max_assembly_tokens = 100_000
        engine._start_sweep_budget_hold()
    else:
        assert engine._config.max_assembly_tokens == engine._config.reserve_tokens_floor == 0
    observed = measured if held else ceiling
    assert engine._hold_fit_only_applies(observed) is held
    assert not engine._should_force_overflow_recovery(observed, view)
    _late_ingest(engine, monkeypatch)

    result = engine.compress(view, current_tokens=observed)

    assert result and engine._survival_measure(result) <= ceiling
    assert engine.last_compression_status != "deadline_yield"
    assert engine._last_survival_fit is not None
    if held:
        assert engine.last_compression_noop_reason == "held"
    assert _digests(stored_rows(engine)) == _digests(view)


def _digests(rows):
    return [(row["role"], hashlib.sha256((row["content"] or "").encode()).hexdigest()) for row in rows]


def test_t3_next_pass_keeps_each_row_once_and_publishes(engine, view, monkeypatch):
    with monkeypatch.context() as patch:
        _late_ingest(engine, patch)
        assert engine.compress(view, current_tokens=8000) is view
        assert engine.last_compression_status == "deadline_yield"
    assert engine._ingest_cursor == len(view)
    assert _digests(stored_rows(engine)) == _digests(view)
    view = [*view, *_turn("next", 110.0)]
    result = engine.compress(view, current_tokens=8000)
    assert _digests(stored_rows(engine)) == _digests(view)
    assert engine._ingest_cursor == len(result)
    assert result != view and engine._last_compress_leaves[1] > 0


def test_t4_three_yields_leave_failure_and_hold_state_clear(engine, view, monkeypatch):
    failure = Mock(wraps=engine._record_ingest_failure)
    monkeypatch.setattr(engine, "_record_ingest_failure", failure)
    before = engine._summary_circuit_breaker.route_status(["probe"])
    streaks = {engine._hold_conversation_key(): {"passes": 1, "last_reason": "prior stop"}}
    engine._zero_leaf_streaks = deepcopy(streaks)
    engine._pressure_yield_invocation_verdict = "blocked"
    engine._pressure_yield_blocked_streak = 1
    with monkeypatch.context() as patch:
        _late_ingest(engine, patch)
        for tokens in (engine.threshold_tokens, 8000, 8000):
            assert engine.compress(view, current_tokens=tokens) is view
            assert engine.last_compression_status == "deadline_yield"
            assert not engine._automatic_compression_blocked()
            assert (engine._zero_leaf_streaks, engine._pressure_yield_invocation_verdict,
                    engine._pressure_yield_blocked_streak) == (streaks, "blocked", 1)
    failure.assert_not_called()
    assert engine._ingest_failure_count == engine._consecutive_ingest_failures == 0
    assert engine._no_progress_hold is None and not engine._retained_stop_hold_active()
    assert engine._summary_circuit_breaker.route_status(["probe"]) == before
    assert engine.should_compress(8000)
    assert engine.compress(view, current_tokens=8000) != view
    assert engine._last_compress_leaves[1] > 0


def test_t5_host_uses_three_attempts_without_rearming(engine, monkeypatch):
    host = Host(engine, ["S"] * 5, _late_ingest(engine, monkeypatch))
    compress = engine.compress

    def observe(*args, **kwargs):
        result = compress(*args, **kwargs)
        assert engine.last_compression_status == "deadline_yield"
        return result

    monkeypatch.setattr(engine, "compress", observe)
    assert len(host.run()) == 6
    assert host.attempts == host.cap == 3
    assert [(verdict, progress) for _, verdict, progress in host.verdicts] == [("aborted_no_progress", False)] * 3


@pytest.mark.parametrize(("value", "auxiliary", "configured_hard", "idle", "hard", "source"), [
    ("20", None, None, 300, 120, "config_yaml:compression.context_timeout_seconds"),
    ("400", None, 500, 400, 360, "config_yaml:compression.context_timeout_seconds"),
    ("100", 600, 700, 600, 540, "config_yaml:compression.context_timeout_seconds"),
    pytest.param("60", '"NaN"', None, 60, 30, "config_yaml:compression.context_timeout_seconds", id="nan-aux"),
    ("0", 600, 700, 0, 700, "config_yaml:compression.context_timeout_seconds"),
    ("-1", None, 500, -1, 500, "config_yaml:compression.context_timeout_seconds"),
    ("bad", None, 500, 0, 500, "default"),
    (None, None, 500, 0, 500, "default"),
])
def test_t6_host_idle_margin(engine, tmp_path, monkeypatch, value, auxiliary, configured_hard, idle, hard, source):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LCM_SUMMARY_TIMEOUT_MS", "900000")
    if value is not None:
        config = f"compression:\n  context_timeout_seconds: {value}\n"
        if auxiliary is not None:
            config += f"auxiliary:\n  compression:\n    timeout: {auxiliary}\n"
        (tmp_path / "config.yaml").write_text(config)
    engine._config = LCMConfig.from_env()
    if configured_hard is not None:
        engine._config.foreground_hard_seconds = configured_hard
        engine._config.foreground_soft_seconds = configured_hard
    budget = engine._new_foreground_budget()
    assert budget.hard == hard
    assert budget.soft == min(60 if configured_hard is None else configured_hard, hard)
    assert budget.configured_timeout == 900
    assert engine._config.hermes_context_timeout_seconds == idle
    assert engine._config.config_sources["hermes_context_timeout_seconds"] == source
