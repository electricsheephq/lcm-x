"""Synthetic parity checks for END read's recursion depth and exception order."""
from __future__ import annotations

import json
import sys

import pytest

from bench.tools import d5_end_read as er
from bench.tools.tests.test_d5_end_read import START, _fake_read, host, identity, stored
from bench.tools.tests.test_d5_end_read_r2 import EMPTY, _reference, _uncached_hint


def test_end_read_reports_stop_iteration_from_a_later_store_identity(monkeypatch, tmp_path):
    later = stored(12, "s1", "later")
    end = {"host": [*START["host"], host(2, "s1", "first")],
           "store": [*START["store"], stored(11, "s1", "first"), later]}

    def row_identity(row):
        if row is later:
            raise StopIteration
        return identity(row)

    result = _fake_read(monkeypatch, tmp_path, end, [], row_identity=row_identity)
    assert result["reason"] == "check 2 could not be read (StopIteration)"


def _first_recursion_error(probe):
    low, high = 0, 32_768
    probe(low)
    with pytest.raises(RecursionError):
        probe(high)
    while high - low > 1:
        depth = (low + high) // 2
        try:
            probe(depth)
        except RecursionError:
            high = depth
        else:
            low = depth
    return high


# CPython 3.12 gave C recursion its own limit. On 3.11 json decoding shares the Python frame budget,
# so the first failing depth moves with the frame difference listed in the PR's known limits.
_C_RECURSION_LIMIT = pytest.mark.skipif(sys.version_info < (3, 12),
                                        reason="exact parity needs CPython 3.12+ (separate C recursion limit)")


def _nested_array(depth):
    return "[" * depth + "0" + "]" * depth


@_C_RECURSION_LIMIT
def test_later_store_identity_has_the_reference_recursion_limit(tmp_path):
    rows = [stored(1, "s", "first"), stored(2, "s", "later")]
    hosts = [host(1, "s", "first")]

    def row_identity(row):
        if row is rows[1]:
            json.loads(row["tool_calls"])
        return identity(row)

    def reference(depth):
        rows[1]["tool_calls"] = _nested_array(depth)
        er.bars._ordered(rows, hosts, lambda r, h: row_identity(r) == row_identity(h))

    def candidate(depth):
        rows[1]["tool_calls"] = _nested_array(depth)
        er.coverage(EMPTY, {"store": rows, "host": hosts}, row_identity, tmp_path)

    assert _first_recursion_error(candidate) == _first_recursion_error(reference)


@_C_RECURSION_LIMIT
def test_payload_hint_has_the_reference_recursion_limit(tmp_path):
    payloads = tmp_path / "lcm-large-outputs"
    payloads.mkdir()
    path = payloads / "nested.json"
    row = host(1, "s", "decoded", role="tool")
    rows = [stored(1, "s", _reference("nested"), role="tool")]

    def reference(depth):
        path.write_text('{"content":' + _nested_array(depth) + "}", encoding="utf-8")
        _uncached_hint(row, rows, tmp_path)

    def candidate(depth):
        path.write_text('{"content":' + _nested_array(depth) + "}", encoding="utf-8")
        er._hint(row, rows, er._hint_cache(tmp_path))

    assert _first_recursion_error(candidate) == _first_recursion_error(reference)


class _BadEquality:
    def __eq__(self, other):
        raise ZeroDivisionError("first comparison")


@pytest.mark.parametrize("later_side", ["host", "store"])
def test_fallback_compares_before_calling_later_identities(tmp_path, later_side):
    rows = [stored(1, "s", "first store")]
    hosts = [host(1, "s", "first host")]
    if later_side == "host":
        hosts.append(host(2, "s", "later"))
    else:
        rows.append(stored(2, "s", "later"))

    def row_identity(row):
        if row["content"] == "later":
            raise KeyError("later identity")
        return _BadEquality()

    with pytest.raises(ZeroDivisionError, match="first comparison"):
        er.bars._ordered(rows, hosts, lambda r, h: row_identity(r) == row_identity(h))
    with pytest.raises(ZeroDivisionError, match="first comparison"):
        er.coverage(EMPTY, {"store": rows, "host": hosts}, row_identity, tmp_path)
