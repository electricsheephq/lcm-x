"""#1122: best-effort extraction reserves the next foreground summary call."""

import logging
import time

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from tests.test_issue_605_foreground_budget import (
    _advance_on,
    _engine,
    _leaves,
    _pin_lcm_char_counter as _pin_lcm_char_counter,
    _provider,
    _view,
)


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        offset = 0.0

        def __call__(self):
            return 1000.0 + self.offset

    fake = Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    return fake


@pytest.mark.parametrize("sweep_enabled", [True, False])
@pytest.mark.parametrize("route_estimate", [30.0, 45.0])
def test_extraction_timeout_reserves_next_summary_and_writes_leaf(
    tmp_path, monkeypatch, clock, sweep_enabled, route_estimate
):
    engine = _engine(
        tmp_path,
        extraction_enabled=True,
        summary_model="primary",
        summary_timeout_ms=LCMConfig().summary_timeout_ms,
        foreground_soft_seconds=60,
        foreground_hard_seconds=120,
        threshold_full_sweep_enabled=sweep_enabled,
    )
    if route_estimate != 30.0:
        engine._foreground_estimates.record_call("primary", route_estimate)
    provider = _provider(monkeypatch, clock, default=0.25)
    extraction_calls = []

    def hanging_extraction(**kwargs):
        budget = engine._foreground_budget
        now = time.monotonic()
        estimate = budget.estimate(engine._primary_summary_route())
        extraction_calls.append(
            (kwargs["timeout"], budget.usable_deadline - now - estimate, estimate)
        )
        clock.offset += kwargs["timeout"]
        raise TimeoutError("extraction timed out")

    monkeypatch.setattr(lcm_engine, "extract_before_compaction", hanging_extraction)
    try:
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert LCMConfig().summary_timeout_ms == 120_000
        assert len(extraction_calls) == 1
        timeout, remaining_for_extraction, estimate = extraction_calls[0]
        assert estimate == route_estimate
        assert timeout <= remaining_for_extraction
        assert len(provider.calls) == 1
        assert provider.calls[0][1] >= route_estimate
        assert len(_leaves(engine)) == 1
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep_enabled", [True, False])
def test_extraction_deadline_keeps_first_leaf_floor_headroom(
    tmp_path, monkeypatch, clock, sweep_enabled
):
    engine = _engine(
        tmp_path,
        extraction_enabled=True,
        summary_model="primary",
        summary_timeout_ms=LCMConfig().summary_timeout_ms,
        foreground_soft_seconds=60,
        foreground_hard_seconds=120,
        threshold_full_sweep_enabled=sweep_enabled,
    )
    engine._foreground_estimates.record_call("primary", 15.0)
    provider = _provider(monkeypatch, clock, default=0.25)
    extraction_calls = []
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, 75.0)
    _advance_on(monkeypatch, engine, "_serialize_messages", clock, 3.0)
    # Timeout handling/logging takes time before the first leaf's admit.
    _advance_on(monkeypatch, engine, "_run_pre_compaction_extraction", clock, 1.0)

    def hanging_extraction(**kwargs):
        budget = engine._foreground_budget
        extraction_calls.append((kwargs["timeout"], budget.usable_deadline - clock()))
        clock.offset += kwargs["timeout"]
        raise TimeoutError("extraction timed out")

    monkeypatch.setattr(lcm_engine, "extract_before_compaction", hanging_extraction)
    try:
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert len(extraction_calls) == 1
        assert extraction_calls[0][1] == pytest.approx(37.0)
        assert len(provider.calls) == 1
        assert provider.calls[0][1] >= 15.0
        assert len(_leaves(engine)) == 1
    finally:
        engine.shutdown()


def test_extraction_deadline_reserves_next_leaf_before_soft_target(
    tmp_path, monkeypatch, clock
):
    engine = _engine(
        tmp_path,
        extraction_enabled=True,
        summary_model="primary",
        summary_timeout_ms=LCMConfig().summary_timeout_ms,
        foreground_soft_seconds=60,
        foreground_hard_seconds=120,
    )
    engine._foreground_estimates.record_call("primary", 20.0)
    provider = _provider(monkeypatch, clock, default=20.0)
    extraction_calls = []

    def hanging_after_progress(**kwargs):
        budget = engine._foreground_budget
        if not budget.progress:
            return
        estimate = budget.estimate(engine._primary_summary_route())
        extraction_calls.append(
            (kwargs["timeout"], budget.t0 + budget.soft - estimate - 2.0 - clock(), estimate)
        )
        clock.offset += kwargs["timeout"]
        raise TimeoutError("extraction timed out")

    monkeypatch.setattr(lcm_engine, "extract_before_compaction", hanging_after_progress)
    try:
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert len(extraction_calls) == 1
        timeout, soft_room, estimate = extraction_calls[0]
        assert estimate == 20.0
        assert timeout <= soft_room
        assert len(provider.calls) == 2
        assert 1000.0 + provider.calls[1][2] + estimate <= 1060.0
        assert len(_leaves(engine)) == 2
    finally:
        engine.shutdown()


def test_extraction_deadline_subtracts_serialization_time(tmp_path, monkeypatch, clock):
    engine = _engine(tmp_path, extraction_enabled=True)
    extraction_calls = []
    _advance_on(monkeypatch, engine, "_serialize_messages", clock, 3.0)
    monkeypatch.setattr(
        lcm_engine, "extract_before_compaction", lambda **kwargs: extraction_calls.append(kwargs)
    )
    deadline = clock() + 10.0
    try:
        engine._run_pre_compaction_extraction(_view(), timeout_seconds=9.0, deadline=deadline)
        assert len(extraction_calls) == 1
        assert extraction_calls[0]["timeout"] == pytest.approx(deadline - clock(), abs=0.01)
        assert extraction_calls[0]["timeout"] == pytest.approx(7.0, abs=0.01)
    finally:
        engine.shutdown()


def test_extraction_deadline_skips_when_serialization_leaves_less_than_one_second(
    tmp_path, monkeypatch, clock, caplog
):
    engine = _engine(tmp_path, extraction_enabled=True)
    extraction_calls = []
    _advance_on(monkeypatch, engine, "_serialize_messages", clock, 3.0)
    monkeypatch.setattr(
        lcm_engine, "extract_before_compaction", lambda **kwargs: extraction_calls.append(kwargs)
    )
    deadline = clock() + 3.5
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
            engine._run_pre_compaction_extraction(_view(), deadline=deadline)
        assert clock() == pytest.approx(deadline - 0.5)
        assert extraction_calls == []
        assert "Pre-compaction extraction skipped" in caplog.text
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep_enabled", [True, False])
def test_disabled_extraction_is_not_called(tmp_path, monkeypatch, clock, sweep_enabled):
    engine = _engine(tmp_path, threshold_full_sweep_enabled=sweep_enabled)
    provider = _provider(monkeypatch, clock, default=0.25)
    extraction_calls = []
    monkeypatch.setattr(
        lcm_engine,
        "extract_before_compaction",
        lambda **kw: extraction_calls.append(kw),
    )
    try:
        assert engine._config.extraction_enabled is False
        view = _view()
        engine.ingest(view)
        engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert extraction_calls == []
        assert provider.calls and _leaves(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep_enabled", [True, False])
@pytest.mark.parametrize("pre_work, expected_leaves", [(84.5, 1), (101.0, 0)])
def test_extraction_skipped_when_reservation_leaves_less_than_one_second(
    tmp_path, monkeypatch, clock, caplog, sweep_enabled, pre_work, expected_leaves
):
    engine = _engine(
        tmp_path,
        extraction_enabled=True,
        summary_timeout_ms=LCMConfig().summary_timeout_ms,
        foreground_soft_seconds=60,
        foreground_hard_seconds=120,
        threshold_full_sweep_enabled=sweep_enabled,
    )
    provider = _provider(monkeypatch, clock, default=0.25)
    extraction_calls = []
    monkeypatch.setattr(
        lcm_engine,
        "extract_before_compaction",
        lambda **kw: extraction_calls.append(kw),
    )
    _advance_on(monkeypatch, engine, "_prepare_retained_user_anchor", clock, pre_work)
    try:
        view = _view()
        engine.ingest(view)
        with caplog.at_level(logging.DEBUG, logger="hermes_lcm"):
            engine.compress(view, current_tokens=engine.threshold_tokens + 1)
        assert extraction_calls == []
        assert len(provider.calls) == expected_leaves
        assert len(_leaves(engine)) == expected_leaves
        skip_lines = [
            record
            for record in caplog.records
            if "Pre-compaction extraction skipped" in record.getMessage()
        ]
        assert len(skip_lines) == 1
        assert skip_lines[0].levelno == logging.DEBUG
        assert "next summary call" in skip_lines[0].getMessage()
        if not expected_leaves:
            assert "reason=time_budget_exhausted" in caplog.text
    finally:
        engine.shutdown()
