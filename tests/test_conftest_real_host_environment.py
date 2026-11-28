"""#2636: report every kind of leaked real-host control."""

import pytest

from tests.conftest import _real_host_environment_changes


@pytest.mark.parametrize("before, after, expected", [
    ({}, {"LCM_REAL_ADDED": "value"}, ["LCM_REAL_ADDED added"]),
    ({"LCM_REAL_REMOVED": "value"}, {}, ["LCM_REAL_REMOVED removed"]),
    ({"LCM_REAL_CHANGED": "old"}, {"LCM_REAL_CHANGED": "new"}, ["LCM_REAL_CHANGED changed"]),
    ({"LCM_REAL_SAME": "value"}, {"LCM_REAL_SAME": "value"}, []),
    ({"OTHER": "old"}, {"OTHER": "new"}, []),
])
def test_real_host_environment_changes(before, after, expected):
    assert _real_host_environment_changes(before, after) == expected
