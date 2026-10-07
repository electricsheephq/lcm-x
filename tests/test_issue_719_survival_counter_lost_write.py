"""#719 / #601: lost counter writes must not erase survival-fit rollback diagnostics."""

import json
import sqlite3

import pytest

import hermes_lcm.command as command
from hermes_lcm.config import LCMConfig
from hermes_lcm.diagnostics import doctor_guidance_for_check
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import SURVIVAL_FIT_COUNTER_KEY


@pytest.fixture
def engine(tmp_path):
    obj = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    obj.on_session_start("S", platform="telegram", conversation_id="conv", context_length=6000)
    yield obj
    obj.shutdown()


def _record(engine, *, shortened=True, projected=False):
    engine._survival_record("test:shortened" if shortened else "test:attempt", int(shortened), [],
                            900, 500 if shortened else 900, 600, projected, "", shortened=shortened,
                            warn_user=False)


def _counter(engine):
    return engine._store.read_metadata_json(SURVIVAL_FIT_COUNTER_KEY)


def _fail_once(engine, monkeypatch):
    original = engine._store.update_metadata_json
    failed = False

    def locked(key, update):
        nonlocal failed
        if key == SURVIVAL_FIT_COUNTER_KEY and not failed:
            failed = True
            raise sqlite3.OperationalError("database is locked")
        return original(key, update)

    monkeypatch.setattr(engine._store, "update_metadata_json", locked)


def _doctor(engine, monkeypatch):
    checks = []
    original = command.doctor_guidance_for_checks

    def capture(items):
        checks.extend(items)
        return original(items)

    with monkeypatch.context() as patch:
        patch.setattr(command, "doctor_guidance_for_checks", capture)
        text = command._doctor_text(engine)
    check = next(item for item in checks if item["check"] == "survival_fit")
    observation = next(line for line in text.splitlines() if "survival_fit:" in line and " — " not in line)
    assert check["status"] == "warn"
    assert check["detail"].get("ever_shortened") is not False
    return observation, doctor_guidance_for_check(check)["operator_action"]


def _assert_rollback(observation, guidance):
    assert "a rollback to v0.23.3 keeps the database file and needs native recovery ON" in observation
    for text in (observation, guidance):
        assert "restore the database backup" in text
        assert "the plugin alone to v0.24.5 or later is fine" in text
    assert "LCM_NATIVE_RECOVERY=true" in guidance
    assert "restore the pre-migration config.yaml" in guidance


def test_t1_lost_shortened_write_then_attempt_keeps_rollback(engine, monkeypatch):
    _fail_once(engine, monkeypatch)
    _record(engine)
    assert _counter(engine) is None
    _record(engine, shortened=False)
    record = _counter(engine)
    assert record["count"] == 1 and record["ever_shortened"] is True
    assert record["last_shortened"] is False and record["projected_count"] == 0
    observation, guidance = _doctor(engine, monkeypatch)
    assert "could not shorten the list on the last attempt" in observation
    _assert_rollback(observation, guidance)
    # Successful reconciliation clears the instance marker: an explicitly empty counter is silent again.
    engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps({"count": 0}))
    assert "survival_fit:" not in command._doctor_text(engine)


@pytest.mark.parametrize("prior", [None, {"count": 2, "projected_count": 1},
                                  {"count": "damaged", "projected_count": 1}],
                         ids=["empty", "existing", "count-lost"])
def test_t2_lost_projection_stays_unknown(engine, monkeypatch, prior):
    if prior is not None:
        engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps(prior))
    _fail_once(engine, monkeypatch)
    _record(engine, projected=True)
    _record(engine, shortened=False)
    record = _counter(engine)
    assert record["ever_shortened"] is True
    assert "projected_count" not in record
    if prior and prior["count"] == "damaged":
        assert record["count_lost"] is True
    _assert_rollback(*_doctor(engine, monkeypatch))
    assert "projected_count unknown" in command._doctor_text(engine)
    _record(engine, projected=True)
    assert "projected_count" not in _counter(engine)
    assert "projected_count unknown" in command._doctor_text(engine)


@pytest.mark.parametrize("prior", [None, {"count": 0, "ever_shortened": False, "projected_count": 0}],
                         ids=["no-record", "count-zero"])
@pytest.mark.parametrize("projected", [False, True], ids=["drop-only", "projected"])
def test_t3_only_lost_write_still_shows_rollback(engine, monkeypatch, prior, projected):
    if prior is not None:
        engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps(prior))
    _fail_once(engine, monkeypatch)
    _record(engine, projected=projected)
    assert _counter(engine) == prior
    observation, guidance = _doctor(engine, monkeypatch)
    assert "applied an unknown number of times (a counter write failed)" in observation
    if projected:
        assert "projected_count unknown" in observation
    _assert_rollback(observation, guidance)
    assert _counter(engine) == prior  # the diagnostic overlay never rewrites metadata


@pytest.mark.parametrize("sequence", [((True, False),), ((False, False),), ((True, True),),
                                      ((True, False), (False, False)),
                                      ((True, True), (False, False), (True, True))],
                         ids=["shortened", "attempt-only", "projected", "shortened-attempt", "mixed"])
def test_t4_no_failures_preserve_exact_base_record(engine, sequence):
    ever_shortened = False
    for index, (shortened, projected) in enumerate(sequence, 1):
        _record(engine, shortened=shortened, projected=projected)
        record = _counter(engine)
        assert isinstance(record.pop("last_at"), (int, float))
        ever_shortened = ever_shortened or shortened
        assert record == {
            "count": index,
            "last_reason": "test:shortened" if shortened else "test:attempt",
            "last_conversation": "conv",
            "last_reached_budget": shortened,
            "last_shortened": shortened,
            "ever_shortened": ever_shortened,
            "unreached_budget_count": sum(not item[0] for item in sequence[:index]),
            "projected_count": sum(item[1] for item in sequence[:index]),
        }


@pytest.mark.parametrize("projected", [False, True])
def test_t5_lost_attempt_only_does_not_claim_shortening(engine, monkeypatch, projected):
    _fail_once(engine, monkeypatch)
    _record(engine, shortened=False, projected=projected)
    assert _counter(engine) is None
    assert "survival_fit:" not in command._doctor_text(engine)
    _record(engine, shortened=False)
    record = _counter(engine)
    assert record["ever_shortened"] is False and record["projected_count"] == 0
    text = command._doctor_text(engine)
    assert "these attempts removed no rows" in text
    assert "a rollback to v0.23.3" not in text


def test_multiple_lost_writes_keep_the_lost_projection(engine, monkeypatch):
    _fail_once(engine, monkeypatch)
    _record(engine, projected=True)
    with monkeypatch.context() as patch:
        _fail_once(engine, patch)
        _record(engine)
    _record(engine, shortened=False)
    assert _counter(engine)["ever_shortened"] is True
    assert "projected_count" not in _counter(engine)
    _assert_rollback(*_doctor(engine, monkeypatch))
