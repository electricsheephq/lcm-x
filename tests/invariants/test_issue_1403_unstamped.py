"""I1/I2: proof loss must not let an old NULL-stamp copy consume a fresh turn.

Permanent seed derived from the #1395 real-engine verification probe. The
occurrence ledger is independent of the anchor's matching and backfill plan.
"""
from copy import deepcopy

import pytest

from tests.invariants.test_exactly_once_machine import RestartMachine, identity


@pytest.mark.parametrize("old_stamp", ["legacy-null", "current-unstamped", "stamped-control"])
@pytest.mark.parametrize("tool_count", [0, 2])
def test_fresh_occurrence_survives_null_copy(monkeypatch, old_stamp, tool_count):
    machine = RestartMachine()
    machine.include_known_open = True
    try:
        if old_stamp == "current-unstamped":
            original_stamp = machine.stamp

            def first_user_unstamped(row):
                stamped = original_stamp(row)
                if machine.sequence == 1:
                    stamped.pop("timestamp")
                return stamped

            monkeypatch.setattr(machine, "stamp", first_user_unstamped)
        machine.new_turn("alpha", False, 0, False, "lookup")
        old_id = machine.engine._store.get_session_tail(machine.session_id)[0]["store_id"]
        if old_stamp == "legacy-null":
            with machine.engine._store.connection as connection:
                connection.execute("UPDATE messages SET observed_at = NULL, observed_at_source = NULL "
                                   "WHERE store_id = ?", (old_id,))
        machine.new_turn("next prompt", False, tool_count, False, "lookup")
        machine.compress()
        assert machine.engine._last_compression_status == "compacted"
        assert machine.engine._compress_commit_proof
        view = deepcopy(machine.host)
        assert not any(identity(row) == identity(machine.occurrences[0]) for row in view)
        machine.replay()
        assert machine.host == view
        assert machine.engine._compress_commit_proof is None
        machine.exactly_once()
        machine.new_turn("alpha", False, 0, False, "lookup")
        machine.exactly_once()
        # Replaying the fresh occurrence does not create another copy.
        machine.replay()
        machine.exactly_once()
        if old_stamp != "stamped-control":
            assert machine.engine._store.get(old_id)["observed_at"] is None
    finally:
        machine.teardown()
