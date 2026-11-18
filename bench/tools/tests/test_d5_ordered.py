"""Synthetic proofs for the END read's ordered matcher."""
from __future__ import annotations

import os
import random
import sys
import time
from collections import Counter
from operator import eq

import pytest

from bench.instruments.d5_clone import bars
from bench.tools.d5_end_read import coverage
from bench.tools.d5_ordered import identity_key, ordered_pairs


def test_ordered_pairs_match_reference_on_3200_seeded_cases():
    rng = random.Random(61018)
    for case in range(3200):
        alphabet = case % 8 + 1
        rows = [rng.randrange(alphabet) for _ in range(rng.randrange(71))]
        hosts = [rng.randrange(alphabet) for _ in range(rng.randrange(71))]
        assert ordered_pairs(rows, hosts) == bars._ordered(rows, hosts, eq), (case, rows, hosts)


@pytest.mark.parametrize(("rows", "hosts"), [
    ([], []),
    ([], [1] * 70),
    ([1] * 70, []),
    ([1] * 35, [1] * 70),
    ([1] * 70, [1] * 35),
    ([1] * 70, [2] * 70),
    ([1, 2], [2, 1]),
    ([1, 2, 1], [2, 1, 2]),
])
def test_ordered_pairs_match_reference_on_edge_cases(rows, hosts):
    assert ordered_pairs(rows, hosts) == bars._ordered(rows, hosts, eq)


def test_a_shared_nan_key_does_not_match_itself():
    nan = float("nan")
    rows, hosts = ["a", nan], [nan, "a"]
    assert ordered_pairs(rows, hosts) == bars._ordered(rows, hosts, eq) == [(0, 1)]


def test_structured_identities_preserve_equality_as_hashable_keys():
    values = [
        ("user", [{"text": "a", "n": 1}]),
        ("user", [{"n": True, "text": "a"}]),
        ("user", [{"text": "a", "n": 2}]),
        ("user", ({"text": "a", "n": 1},)),
        ("user", {"text": "a", "n": 1}),
        ("user", {"text": "a", "n": 2}),
        ("user", {"a", "b"}),
        ("user", frozenset({"b", "a"})),
        ("tool", bytearray(b"ok")),
        ("tool", b"ok"),
    ]
    keys = list(map(identity_key, values))
    for value, key in zip(values, keys, strict=True):
        for other, other_key in zip(values, keys, strict=True):
            assert (key == other_key) == (value == other)
            if value == other:
                assert hash(key) == hash(other_key)
    assert ordered_pairs(keys, keys[::-1]) == bars._ordered(values, values[::-1], eq)


def test_coverage_calls_identity_once_per_matching_input_row(tmp_path):
    calls = Counter()

    def identity(row):
        calls[row["token"]] += 1
        return "user", [{"n": row["key"]}]

    def host(i, session, key):
        return {"id": i, "session_id": session, "key": key, "token": f"h{i}"}

    def stored(i, session, key):
        return {"store_id": i, "session_id": session, "key": key, "token": f"s{i}"}

    start = {"host": [host(1, "s1", "old")], "store": [stored(1, "s1", "old")]}
    end = {"host": [*start["host"], host(2, "s1", "b"), host(3, "s1", "a"), host(4, "s2", "c")],
           "store": [*start["store"], stored(2, "s1", "a"), stored(3, "s1", "a"), stored(4, "s1", "b")]}
    result = coverage(start, end, identity, tmp_path)
    assert (result["matched"], result["unmatched"], result["lost"]) == (1, 1, 1)
    assert calls == {"h2": 1, "h3": 1, "h4": 1, "s2": 1, "s3": 1, "s4": 1}


@pytest.mark.skipif(os.environ.get("LCMX_ENDREAD_FIELD_SCALE") != "1", reason="set LCMX_ENDREAD_FIELD_SCALE=1")
def test_ordered_pairs_at_field_scale():
    import resource

    rows, hosts, expected = [], [], []
    repeated = range(200)

    def add_matches(keys):
        for key in keys:
            expected.append((len(rows), len(hosts)))
            rows.append(key)
            hosts.append(key)

    for copy in range(150):
        anchor = 200 + 14 * copy
        add_matches(range(anchor, anchor + 14))
        # Host-only rows interleaved with the next copy of the shared block.
        hosts.extend(range(-150 * (copy + 1), -150 * copy))
        if copy % 30 == 0:
            rows.extend(repeated)
        else:
            add_matches(repeated)
    add_matches(range(2300, 2368))
    hosts.extend(range(-22611, -22500))
    assert (len(rows), len(hosts), len(expected)) == (32_168, 53_779, 31_168)

    started = time.perf_counter()
    pairs = ordered_pairs(rows, hosts)
    elapsed = time.perf_counter() - started
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_bytes *= 1024  # Linux reports KiB and macOS reports bytes.
    print(f"Field scale 32168 x 53779: {elapsed:.3f} s wall, {peak_bytes} bytes peak RSS")
    assert pairs == expected
    assert elapsed < 300
    assert peak_bytes < 8_000_000_000
