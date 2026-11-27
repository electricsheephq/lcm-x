"""An unstamped occurrence stays unstamped across a leaf and two ingests."""

import pytest

from tests.harness.engine import stored_rows
from tests.invariants.test_issue_2855_kept_row_keys import (
    _history, _row, _stamp, open_engine as open_engine,
    open_seeded_engine as open_seeded_engine, uid_mode as uid_mode,
)

pytestmark = pytest.mark.usefixtures("uid_mode")


def test_two_ingests_after_leaf(open_seeded_engine):
    history = _history()
    current = _row("user", "Current question: " + "epsilon zeta eta theta " * 40)
    _stamp(current, 101)
    stamp = current.pop("timestamp")
    with open_seeded_engine() as engine:
        engine.ingest([*history, current])
        current["timestamp"] = stamp
        result = engine.compress([*history, current], force=True)
        assert engine._last_compress_leaves[1] == 1
        tail, = [row for row in result if row["content"] == current["content"]]
        assert "timestamp" not in tail
        for turn in range(3):
            matching = [row for row in stored_rows(engine) if row["content"] == current["content"]]
            print(f"turn {turn}: {len(matching)} row, observed_at={[row['observed_at'] for row in matching]}")
            assert len(matching) == 1
            assert matching[0]["observed_at"] is None
            if turn < 2:
                result.append(_row("assistant" if turn == 0 else "user", f"Next turn {turn}"))
                engine.ingest(result)
