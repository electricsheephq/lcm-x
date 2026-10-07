"""#715: exit caps exclude recovery; diagnostics reflect actual live-context cuts."""
import json
import logging

import pytest

from hermes_lcm.command import handle_lcm_command
from hermes_lcm.diagnostics import doctor_guidance_for_check
from hermes_lcm.survival_fit import SURVIVAL_FIT_COUNTER_KEY

from tests.test_issue_668_exit_fit import engine as _engine_fixture, estimators, _counter  # noqa: F401
from tests.test_issue_650_survival_fit_keeps_summary import _hidden_backlog, _spy

engine = _engine_fixture


def test_automatic_forced_overflow_keeps_plain_fit(engine, monkeypatch, caplog):
    """#738: an exit fit drops only covered turns."""
    view = _hidden_backlog(engine, list_users=True)
    observed = engine._survival_measure(view) + 2000
    engine._config.max_assembly_tokens = observed - 1
    seen = _spy(engine, monkeypatch)
    assert engine._should_force_overflow_recovery(observed_tokens=observed, messages=view)
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        result = engine.compress(view, current_tokens=observed)
    assert any("forced overflow recovery" in row.getMessage() for row in caplog.records)
    assert int(engine.threshold_tokens * 0.95) < engine._survival_measure(seen["input"]) + 2000
    assert engine._survival_measure(seen["input"]) <= engine._survival_fit_budget(view, observed)
    assert result == seen["input"] and engine._last_survival_fit is None
    assert not any("exit_fit:" in row.getMessage() for row in caplog.records)
    assert engine._compress_forced_overflow is False
    # An unrelated automatic invocation must still get its exit cap (its uncovered rows stay: the exit fit skips).
    engine._config.max_assembly_tokens = 0
    monkeypatch.setattr(engine, "_compress_impl", lambda messages, **kwargs: messages)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        engine.compress(result, current_tokens=observed)
    assert any("LCM exit fit skipped" in row.getMessage() for row in caplog.records)


@pytest.mark.parametrize("prior", ["none", "shortened", "legacy"])
def test_attempt_only_doctor_guidance(engine, prior):
    if prior == "shortened":
        engine._survival_record("test:shortened", 1, [], 900, 500, 600, False, "", warn_user=False)
    elif prior == "legacy":
        engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps({"count": 1}))
    for _ in range(2):
        engine._survival_record("test:attempt", 0, [], 900, 900, 600, False, "", shortened=False)
    record = _counter(engine)
    doctor = handle_lcm_command("doctor", engine)
    observation = next(line for line in doctor.splitlines() if "survival_fit: attempted" in line)
    guidance = doctor_guidance_for_check({"check": "survival_fit", "status": "warn", "detail": record})
    assert record["ever_shortened"] is (prior != "none")
    assert "could not shorten" in observation and "could not shorten" in guidance["rationale"]
    for text in (observation, guidance["operator_action"]):
        if prior == "none":
            assert "rollback" not in text.lower() and "restore" not in text.lower()
        else:
            assert "restore the database backup" in text
            # #919 review: a legacy record has no projected_count, so the plugin-alone target is v0.27.0
            target = "v0.27.0" if prior == "legacy" else "v0.24.5"
            assert f"the plugin alone to {target} or later is fine" in text


def test_legacy_record_without_ever_shortened_keeps_doctor_advice(engine):
    record = {"count": 1, "last_reason": "test:old", "last_shortened": False}
    engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps(record))
    doctor = handle_lcm_command("doctor", engine)
    observation = next(line for line in doctor.splitlines() if "survival_fit: attempted" in line)
    guidance = doctor_guidance_for_check({"check": "survival_fit", "status": "warn", "detail": record})
    for text in (observation, guidance["operator_action"]):
        assert "restore the database backup" in text


def test_unmapped_exit_coverage_is_unknown(engine, monkeypatch, caplog):
    monkeypatch.setattr(engine, "_store_complete_node_covered", lambda ids: set(ids))
    with caplog.at_level(logging.INFO, logger="hermes_lcm"):
        engine._survival_record("exit_fit:compressed", 2, [1], 900, 500, 600, False, "", warn_user=False)
    assert engine._last_survival_fit["uncovered_rows"] is None
    assert "uncovered_rows=unknown" in caplog.text
