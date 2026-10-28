"""#1179 sibling: retry after compaction and host-merged scaffold cold resume."""

import pytest

pytest.importorskip("hypothesis")
from copy import deepcopy

from tests.invariants.test_exactly_once_machine import RestartMachine, identity


class RetryScaffoldDuplicate(AssertionError):
    """Only the replay's extra merged scaffold row, with every delivery retained."""


@pytest.mark.xfail(strict=True, raises=RetryScaffoldDuplicate,
                   reason="#1179: retry beside a host-merged scaffold after compaction")


def test_retry_host_merged_scaffold_after_compaction():
    state = RestartMachine()
    state.include_known_open = True
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
        state.exactly_once()
        state.cold_resume()
        rows = state.rows()
        wanted, actual = list(map(identity, state.occurrences)), list(map(identity, rows))
        if actual == wanted[:8] + [identity(state.host[0])] + wanted[8:]:
            retained = rows[:8] + rows[9:]
            assert [row["session_id"] for row in retained] == [
                row["session_id"] for row in state.occurrences
            ], "session ownership"
            raise RetryScaffoldDuplicate("cold resume re-stores the earlier merged summary + next prompt")
        state.exactly_once()  # Other losses, duplicates or ownership errors must fail normally.
    finally:
        state.teardown()


def test_wrong_session_delivery_is_not_the_known_scaffold_duplicate(monkeypatch):
    rows = RestartMachine.rows

    def wrong_owner_on_duplicate(state):
        retained = rows(state)
        if len(retained) == len(state.occurrences) + 1:
            retained[-1]["session_id"] = "wrong-session"
        return retained

    monkeypatch.setattr(RestartMachine, "rows", wrong_owner_on_duplicate)
    with pytest.raises(AssertionError, match="session ownership"):
        test_retry_host_merged_scaffold_after_compaction()


def test_guard_only_retry_beside_verified_merged_scaffold():
    from tests.invariants.test_exactly_once_machine import retry_beside_merged_scaffold

    state = RestartMachine()
    try:
        state.new_turn("alpha", False, 0, False, "lookup")
        state.cold_resume()
        state.compress()
        state.cold_resume()
        user = {"role": "user", "content": "next prompt"}
        assert retry_beside_merged_scaffold(state.engine, state.host, user)
        lookalike = deepcopy(state.host)
        lookalike[0]["content"] = lookalike[0]["content"].replace("[USER]: alpha", "[USER]: invented")
        assert not retry_beside_merged_scaffold(state.engine, lookalike, user)
        assert not retry_beside_merged_scaffold(state.engine, state.host, dict(user, content="different"))
        before = deepcopy(state.host), deepcopy(state.occurrences), state.rows()
        state.new_turn("alpha", False, 0, True, "lookup")
        assert (state.host, state.occurrences, state.rows()) == before
        state.cold_resume()
        state.exactly_once()
        # Ordinary turns still advance even with the same verified merged prefix.
        delivered = len(state.occurrences)
        state.new_turn("different", False, 0, False, "lookup")
        assert len(state.occurrences) == delivered + 2
        state.exactly_once()
    finally:
        state.teardown()
