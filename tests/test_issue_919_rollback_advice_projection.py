"""#919: projected survival fits set a conservative plugin-only rollback floor."""

from __future__ import annotations

import json

import pytest

from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.diagnostics import doctor_guidance_for_check
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import SURVIVAL_FIT_COUNTER_KEY

CLAUSE = (
    "; a survival fit projected rows (or the projection count is unknown): from v0.27.0 on, "
    "a plugin-only rollback must target v0.27.0 or later; v0.26.x or earlier cannot recognise "
    "a short projected row (head + mark) and stores it again on a cold resume (#601 duplicates); "
    "to roll back further, stop Hermes and move the configured database file (by default lcm.db, "
    "with its -wal and -shm companions) aside and keep it"
)
BASE_OBSERVATION = (
    "- survival_fit: applied 2 time(s); last reason publication_invariant_conflict; projected_count 0; "
    "unreached_budget_count 0; last_reached_budget unknown; last_uncovered_rows unknown; uncovered_fit_count "
    "unknown; unknown_coverage_fit_count unknown; rollback to a 0.24.x version older than "
    "v0.24.5: that plugin cannot compact stored rows that a survival fit removed from the live context; "
    "stop Hermes, move the configured database file (by default lcm.db, with its -wal and -shm companions) "
    "aside and keep it, then restore the database backup taken before the first v0.24.5 install with "
    "the plugin; a rollback of the plugin alone to v0.24.5 or later is fine; a rollback to v0.23.3 "
    "keeps the database file and needs native recovery ON (see triage_guidance)"
)
# #919 review: with the floor, the observation never also calls v0.24.5 through v0.26.x a safe plugin-only target.
FLOOR_OBSERVATION = BASE_OBSERVATION.replace("the plugin alone to v0.24.5 or later is fine",
                                             "the plugin alone to v0.27.0 or later is fine")
GUIDANCE_OLD = "a rollback of the plugin alone to v0.24.5 or later is fine. To v0.23.3:"
GUIDANCE_FLOOR = (
    "a rollback of the plugin alone to v0.27.0 or later is fine; a survival fit projected rows (or the projection "
    "count is unknown) and v0.26.x or earlier cannot recognise a short projected row (head + mark) and stores it "
    "again on a cold resume (#601 duplicates), so to v0.24.5 through v0.26.x stop Hermes, move the configured "
    "database file (by default lcm.db, with its -wal and -shm companions) aside and keep it. To v0.23.3:"
)


def _observations(tmp_path, record):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    engine.on_session_start("S", platform="telegram", context_length=6000, conversation_id="conv")
    try:
        engine._store.write_metadata_json([SURVIVAL_FIT_COUNTER_KEY], json.dumps(record, sort_keys=True))
        return [line.strip() for line in handle_lcm_command("doctor", engine).splitlines()
                if line.strip().startswith("- survival_fit:") and " — " not in line]
    finally:
        engine.shutdown()


def test_projected_rows_add_the_floor_and_keep_older_advice(tmp_path):
    observation, = _observations(tmp_path, {
        "count": 2, "last_reason": "publication_invariant_conflict", "projected_count": 3,
    })
    assert observation == FLOOR_OBSERVATION.replace("projected_count 0;", "projected_count 3;") + CLAUSE


def test_unknown_projection_count_adds_the_floor(tmp_path):
    observation, = _observations(tmp_path, {"count": 2, "last_reason": "publication_invariant_conflict"})
    assert observation == FLOOR_OBSERVATION.replace("projected_count 0;", "projected_count unknown;") + CLAUSE


def test_zero_projected_rows_are_byte_identical_to_base(tmp_path):
    assert _observations(tmp_path, {
        "count": 2, "last_reason": "publication_invariant_conflict", "projected_count": 0,
    }) == [BASE_OBSERVATION]


def test_no_fit_has_no_observation(tmp_path):
    assert _observations(tmp_path, {"count": 0, "projected_count": 3}) == []


@pytest.mark.parametrize("projected", ["3", [1], {"a": 1}], ids=["text", "list", "dict"])
def test_damaged_projection_count_is_unknown_and_shows_the_clause(tmp_path, projected):
    observation, = _observations(tmp_path, {
        "count": 2, "last_reason": "publication_invariant_conflict", "projected_count": projected,
    })
    assert observation == FLOOR_OBSERVATION.replace("projected_count 0;", "projected_count unknown;") + CLAUSE


def test_projected_rows_without_shortening_have_self_contained_advice(tmp_path):
    observation, = _observations(tmp_path, {
        "count": 2, "projected_count": 2, "ever_shortened": False,
    })
    assert CLAUSE in observation
    assert "stop Hermes and move the configured database file (by default lcm.db, " in observation
    assert "with its -wal and -shm companions) aside and keep it" in observation
    assert "as described above" not in observation


def _guidance(record):
    return doctor_guidance_for_check({"check": "survival_fit", "status": "warn", "detail": record})["operator_action"]


@pytest.mark.parametrize("record", [
    {"count": 2, "projected_count": 3},
    {"count": 2},
    {"count": 2, "projected_count": "3"},
    {"count": 1, "projected_count": 0, "count_lost": True},
    {"count": 1, "projected_count": 1, "count_lost": True},
], ids=["projected", "absent", "damaged", "count-lost-zero", "count-lost-one"])
def test_triage_guidance_applies_the_same_floor(record):
    """#919 review: the structured guidance names the same v0.27.0 floor and never calls v0.24.5 safe."""
    text = _guidance(record)
    assert GUIDANCE_FLOOR in text and "v0.24.5 or later is fine" not in text


def test_zero_projections_keep_the_original_guidance():
    text = _guidance({"count": 2, "projected_count": 0})
    assert GUIDANCE_OLD in text and "v0.27.0" not in text


@pytest.mark.parametrize("projected", [0, 1])
def test_lost_projection_history_is_unknown_and_shows_the_floor(tmp_path, projected):
    """#919 review: a record rebuilt after a lost count (count_lost) restarts projected_count, so earlier
    projections are unknown and the doctor applies the floor whatever the current count."""
    observation, = _observations(tmp_path, {
        "count": 2, "last_reason": "publication_invariant_conflict", "projected_count": projected,
        "count_lost": True,
    })
    assert observation == FLOOR_OBSERVATION.replace("projected_count 0;", "projected_count unknown;") + CLAUSE
