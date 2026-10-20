"""Direct tests for pure assembly-cap and threshold helpers."""

import pytest

from hermes_lcm.compaction_policy import (
    assembly_token_cap,
    threshold_tokens,
)


@pytest.mark.parametrize("window,maximum,reserve,expected", [
    (0, 0, 10, (None, False)),
    (100, 0, 0, (None, False)),
    (100, 80, 10, (80, False)),
    (100, 80, 99, (1, False)),
    (100, 80, 100, (80, True)),
    (100, 0, 101, (None, True)),
])
def test_assembly_cap(window, maximum, reserve, expected):
    assert assembly_token_cap(window, maximum, reserve) == expected


@pytest.mark.parametrize("trigger,cap,expected", [
    (0, None, 0), (0, 10, 10), (-1, 10, 10),
    (5, 10, 5), (20, 10, 10),
])
def test_threshold_token_precedence(trigger, cap, expected):
    assert threshold_tokens(trigger, cap) == expected
