"""#1179 siblings (#1446): a retry after compaction, then a cold resume beside the host-merged scaffold.

The retry stores its rows below the cursor, so the durable proof is re-keyed to the host's list (#1286):
the cold resume proves the merged row instead of storing it again.
"""

import pytest

pytest.importorskip("hypothesis")

from tests.invariants.test_exactly_once_machine import RestartMachine


def test_retry_host_merged_scaffold_after_compaction():
    """R5: the retry repeats the last exchange of the cold-resumed compacted view."""
    state = RestartMachine()
    try:
        state.new_turn("alpha", False, 0, False, "lookup")
        state.exactly_once()
        state.cold_resume()
        state.exactly_once()
        state.compress()
        state.exactly_once()
        state.cold_resume()
        state.exactly_once()
        state.new_turn("alpha", False, 0, True, "lookup")
        assert state.retry_occurrence_index is not None  # a retry, not a new turn with the same text
        state.exactly_once()
        state.cold_resume()
        state.exactly_once()
    finally:
        state.teardown()


def test_retry_after_a_new_turn_beside_merged_scaffold():
    """S1: a new turn after the compaction, its retry, then the cold resume."""
    state = RestartMachine()
    try:
        state.new_turn("alpha", False, 0, False, "lookup")
        state.cold_resume()
        state.compress()
        state.new_turn("alpha 2", False, 0, False, "lookup")
        state.exactly_once()
        state.new_turn("alpha 2", False, 0, True, "lookup")
        assert state.retry_occurrence_index is not None
        state.exactly_once()
        state.cold_resume()
        state.exactly_once()
    finally:
        state.teardown()
