"""#906: plain survival fits report real uncovered drops without changing the cut."""
from __future__ import annotations

import json
import logging

import pytest

from hermes_lcm.command import handle_lcm_command
from hermes_lcm.dag import SummaryNode
from hermes_lcm.diagnostics import doctor_guidance_for_check
from hermes_lcm.survival_fit import SURVIVAL_FIT_COUNTER_KEY

from tests import test_issue_738_exit_fit_coverage as coverage_fixtures
from tests.test_issue_738_exit_fit_coverage import _engine, _turns

# Register the reused pytest fixtures without shadowing unused imports.
estimators = coverage_fixtures.estimators
summaries = coverage_fixtures.summaries


def _counter(engine):
    return engine._store.read_metadata_json(SURVIVAL_FIT_COUNTER_KEY)


def _doctor_observation(engine):
    return next(line for line in handle_lcm_command("doctor", engine).splitlines()
                if "survival_fit: " in line)


def _capture_record(engine, monkeypatch):
    calls = []
    original = engine._survival_record

    def record(reason, count, ids, *args, **kwargs):
        calls.append((reason, count, list(ids)))
        return original(reason, count, ids, *args, **kwargs)

    monkeypatch.setattr(engine, "_survival_record", record)
    return calls


@pytest.mark.parametrize("covered", [False, True], ids=["uncovered_drop", "covered_drop"])
def test_held_ceiling_plain_fit_counts_actual_dropped_rows(tmp_path, monkeypatch, summaries, caplog, covered):
    """Reuse #738's held ceiling path: compress returns a plain fit without a model call."""
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        view = _turns(40)
        engine.ingest(view)
        stored = engine._store.get_session_messages("S", limit=100_000)
        all_ids = [int(row["store_id"]) for row in stored]
        if covered:
            # Duplicate coverage must count each dropped store id once.
            for label in ("Earlier turns", "Earlier turns again"):
                engine._dag.add_node(SummaryNode(session_id="S", summary=label, source_ids=all_ids))
        calls = _capture_record(engine, monkeypatch)
        engine._start_sweep_budget_hold()
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            result = engine.compress(view, current_tokens=engine._survival_measure(view) + 800)
        assert summaries == []
        assert (engine._last_compression_status, engine._last_compression_noop_reason) == ("noop", "held")
        assert len(calls) == 1
        reason, count, ids = calls[0]
        assert reason == "noop" and not reason.startswith("exit_fit:")
        assert count > 0 and len(ids) == count
        assert len(result) < len(view)
        retained_contents = {row.get("content") for row in result}
        omitted_ids = {int(row["store_id"]) for row in stored if row["content"] not in retained_contents}
        assert set(ids) == omitted_ids
        node_sources = {int(source) for node in engine._dag.get_session_nodes("S") for source in node.source_ids}
        expected = len(omitted_ids - node_sources)
        assert expected == (0 if covered else count)
        assert engine._last_survival_fit["uncovered_rows"] == expected
        assert f"uncovered_rows={expected}" in caplog.text
        record = _counter(engine)
        assert record["last_uncovered_rows"] == expected
        assert record["uncovered_fit_count"] == int(not covered)
        assert record["unknown_coverage_fit_count"] == 0
        doctor = _doctor_observation(engine)
        assert f"last_uncovered_rows {expected}" in doctor
        assert f"uncovered_fit_count {int(not covered)}" in doctor
        guidance = doctor_guidance_for_check({"check": "survival_fit", "status": "warn", "detail": record})
        assert "counted for every survival fit" in guidance["rationale"]
        # Diagnosis changes no stored rows, and the supported fit still emits its original warning.
        assert engine._store.get_session_messages("S", limit=100_000) == stored
        assert engine.get_automatic_compaction_status_message(phase="after", default_message="")
    finally:
        engine.shutdown()


@pytest.mark.parametrize("failure", ["coverage_read", "partial_ids"])
def test_unknown_non_exit_coverage_stays_unknown(tmp_path, monkeypatch, summaries, caplog, failure):
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        def coverage(ids):
            if failure == "coverage_read":
                raise RuntimeError("synthetic coverage read failure")
            pytest.fail("partial ids must not be treated as complete coverage")

        monkeypatch.setattr(engine, "_store_complete_node_covered", coverage)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine._survival_record("noop", 2, [1, 2] if failure == "coverage_read" else [1],
                                    900, 500, 600, False, "", warn_user=False)
        assert engine._last_survival_fit["uncovered_rows"] is None
        assert "uncovered_rows=unknown" in caplog.text
        assert "uncovered_rows=0" not in caplog.text
        record = _counter(engine)
        assert record["last_uncovered_rows"] is None and record["uncovered_fit_count"] == 0
        assert record["unknown_coverage_fit_count"] == 1
        assert "last_uncovered_rows unknown" in _doctor_observation(engine)
        assert "unknown_coverage_fit_count 1" in _doctor_observation(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("reason", ["noop", "exit_fit:compressed"])
def test_unshortened_fit_has_zero_without_coverage_or_applied_log(tmp_path, monkeypatch, summaries, caplog, reason):
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        def coverage(ids):
            pytest.fail("an unshortened attempt must not read coverage")

        monkeypatch.setattr(engine, "_store_complete_node_covered", coverage)
        with caplog.at_level(logging.INFO, logger="hermes_lcm"):
            engine._survival_record(reason, 0, [], 900, 900, 600, False, "", shortened=False)
        assert engine._last_survival_fit["uncovered_rows"] == 0
        assert "LCM survival fit applied" not in caplog.text
        record = _counter(engine)
        assert record["last_uncovered_rows"] == 0 and record["uncovered_fit_count"] == 0
        assert record["unknown_coverage_fit_count"] == 0
        assert engine.get_automatic_compaction_status_message(phase="after", default_message="") is None
    finally:
        engine.shutdown()


@pytest.mark.parametrize("legacy", [False, True])
def test_counter_update_keeps_legacy_aggregate_unknown_and_counts_fits(tmp_path, monkeypatch, summaries, legacy):
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        if legacy:
            engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps({"count": 4}))
            doctor = _doctor_observation(engine)
            assert "last_uncovered_rows unknown" in doctor and "uncovered_fit_count unknown" in doctor
        engine._survival_record("noop", 2, [1, 2], 900, 500, 600, False, "", warn_user=False)
        engine._survival_record("noop", 1, [3], 900, 500, 600, False, "", warn_user=False)
        record = _counter(engine)
        assert record["count"] == (6 if legacy else 2)
        assert record["last_uncovered_rows"] == 1
        if legacy:
            assert "uncovered_fit_count" not in record and "projected_count" not in record
            assert "unknown_coverage_fit_count" not in record
            assert "uncovered_fit_count unknown" in _doctor_observation(engine)
            assert "unknown_coverage_fit_count unknown" in _doctor_observation(engine)
        else:
            assert record["uncovered_fit_count"] == 2  # fits, not three uncovered rows
            assert "uncovered_fit_count 2" in _doctor_observation(engine)
        assert "last_uncovered_rows 1" in _doctor_observation(engine)
    finally:
        engine.shutdown()


def test_unknown_coverage_survives_a_later_clean_fit(tmp_path, monkeypatch, summaries):
    """An unknown-coverage fit stays visible after a later fit whose coverage is known and clean."""
    engine = _engine(tmp_path, monkeypatch, sweep=True)
    try:
        def failing(ids):
            raise RuntimeError("synthetic coverage read failure")

        monkeypatch.setattr(engine, "_store_complete_node_covered", failing)
        engine._survival_record("noop", 2, [1, 2], 900, 500, 600, False, "", warn_user=False)
        monkeypatch.setattr(engine, "_store_complete_node_covered", lambda ids: set(ids))  # every row covered
        engine._survival_record("noop", 1, [3], 900, 500, 600, False, "", warn_user=False)
        record = _counter(engine)
        assert record["last_uncovered_rows"] == 0
        assert record["uncovered_fit_count"] == 0
        assert record["unknown_coverage_fit_count"] == 1
        doctor = _doctor_observation(engine)
        assert "uncovered_fit_count 0" in doctor and "unknown_coverage_fit_count 1" in doctor
        guidance = doctor_guidance_for_check({"check": "survival_fit", "status": "warn", "detail": record})
        assert "lower bound" in guidance["rationale"]
    finally:
        engine.shutdown()
