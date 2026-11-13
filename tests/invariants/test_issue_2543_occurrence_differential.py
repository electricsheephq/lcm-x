"""#2543 seeded engine differential: content coverage and bounded occurrence copies (I1/I2/I6)."""

import copy
import random

import pytest

from tests.test_issue_2543_occurrence_resume import (
    FINAL, SHAPES, gc_copy, history, shape_row,
)
from tests.test_issue_2543_occurrence_resume import lanes as lanes


@pytest.mark.parametrize("seed", range(200))
def test_seeded_occurrence_differential(lanes, monkeypatch, seed):
    rng = random.Random(seed)
    lane = lanes("mutated")
    shape = rng.choice(SHAPES[:-1])
    first = history(shape)
    # A transformed result exercises recorded-form survival through the real payload-GC rewrite.
    first[4] = {"role": "tool", "tool_call_id": "m1", "content": shape_row("T6ii")["content"]}
    lane.seed(first)
    transformed = sum(not lane.engine._session_end_rows_equal(row, message, session_id="S1")
                      for row, message in zip(lane.rows(), first))
    assert transformed == 2
    view = copy.deepcopy(first)
    view[1]["content"] = f"edited {seed}"
    del view[rng.randrange(6, 10)]
    view.insert(rng.randrange(2, 6), {"role": "user", "content": f"inserted {seed}"})
    view += [FINAL, {"role": "user", "content": f"appended {seed}"}]
    edits, novel = 1, 3
    ends = rng.randint(2, 4)
    for end in range(ends):
        if end == ends - 1:
            # Identical final end across both restart and GC must add zero rows.
            gc_copy(lane)
            lane.restart()
        elif end:
            if rng.choice((False, True)):
                view[1]["content"] = f"edited {seed} version {end}"
                edits += 1
            else:
                view.append({"role": "assistant", "content": f"later {seed} {end}"})
                novel += 1
        before = lane.rows()
        lane.end(view, monkeypatch, busy=bool((seed + end) % 2))
        lane.assert_present(view)
        after = lane.rows()
        assert after[:len(before)] == before
        copies = len(after) - len(first) - novel
        assert copies <= transformed + edits
        if end == ends - 1:
            assert after == before

    # Paired plain controls use the exact prefix route, including redaction-enabled runs.
    plain = lanes("plain")
    plain.engine._config.sensitive_patterns_enabled = bool(seed % 2)
    plain.engine._config.sensitive_patterns = ["api_key"]
    base = history()
    base[2]["content"] = f"api_key=synthetic2543value{seed}"
    plain.seed(base)
    control = base + [FINAL]
    for end in range(ends):
        if end == ends - 1:
            plain.restart()
            # The fixture opens with defaults; restore the same redaction posture after restart.
            plain.engine._config.sensitive_patterns_enabled = bool(seed % 2)
            plain.engine._config.sensitive_patterns = ["api_key"]
        before = plain.rows()
        plain.end(control, monkeypatch, busy=bool((seed + end) % 2))
        plain.assert_present(control)
        assert len(plain.rows()) == len(base) + 1
        assert not plain.forms()
        if end:
            assert plain.rows() == before
