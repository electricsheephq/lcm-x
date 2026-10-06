"""#919: projected survival fits set a conservative plugin-only rollback floor."""

from __future__ import annotations

import json

from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.survival_fit import SURVIVAL_FIT_COUNTER_KEY

CLAUSE = (
    "; a survival fit projected rows (or the projection count is unknown): from v0.27.0 on, "
    "a plugin-only rollback must target v0.27.0 or later; v0.26.x or earlier cannot recognise "
    "a short projected row (head + mark) and stores it again on a cold resume (#601 duplicates); "
    "to roll back further, stop Hermes and move the database aside as described above"
)
BASE_OBSERVATION = (
    "- survival_fit: applied 2 time(s); last reason publication_invariant_conflict; projected_count 0; "
    "unreached_budget_count 0; last_reached_budget unknown; rollback to a 0.24.x version older than "
    "v0.24.5: that plugin cannot compact stored rows that a survival fit removed from the live context; "
    "stop Hermes, move the configured database file (by default lcm.db, with its -wal and -shm companions) "
    "aside and keep it, then restore the database backup taken before the first v0.24.5 install with "
    "the plugin; a rollback of the plugin alone to v0.24.5 or later is fine; a rollback to v0.23.3 "
    "keeps the database file and needs native recovery ON (see triage_guidance)"
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
    assert observation == BASE_OBSERVATION.replace("projected_count 0;", "projected_count 3;") + CLAUSE


def test_unknown_projection_count_adds_the_floor(tmp_path):
    observation, = _observations(tmp_path, {"count": 2, "last_reason": "publication_invariant_conflict"})
    assert observation == BASE_OBSERVATION.replace("projected_count 0;", "projected_count unknown;") + CLAUSE


def test_zero_projected_rows_are_byte_identical_to_base(tmp_path):
    assert _observations(tmp_path, {
        "count": 2, "last_reason": "publication_invariant_conflict", "projected_count": 0,
    }) == [BASE_OBSERVATION]


def test_no_fit_has_no_observation(tmp_path):
    assert _observations(tmp_path, {"count": 0, "projected_count": 3}) == []
