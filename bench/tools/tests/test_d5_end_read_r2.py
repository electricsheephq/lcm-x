"""Synthetic checks for END read identity ordering and hint caching."""
from __future__ import annotations

from collections import Counter, OrderedDict
import json
from operator import eq
import random
import time

import pytest

from bench.tools import d5_end_read as er
from bench.tools.tests.test_d5_end_read import START, _fake_read, host, identity, stored


EMPTY = {"host": [], "store": []}


def test_never_stored_session_skips_identity_and_counts_lost(tmp_path):
    row = host(1, "never-stored", "new")
    calls = []

    def invalid_identity(row):
        calls.append(row)
        raise KeyError("missing identity column")

    result = er.coverage(EMPTY, {"host": [row], "store": []}, invalid_identity, tmp_path)
    assert result["lost"] == 1
    assert result["unmatched_rows"][0]["class"] == "LOST"
    assert calls == []


def test_a_host_identity_error_precedes_a_later_store_identity_error(tmp_path):
    rows = [stored(1, "s", "ok"), stored(2, "s", "invalid store identity")]
    hosts = [host(1, "s", "invalid host identity")]
    calls = []

    def row_identity(row):
        calls.append(row)
        if row is hosts[0]:
            raise KeyError("host")
        if row is rows[1]:
            raise ValueError("store")
        return identity(row)

    with pytest.raises(KeyError, match="host"):
        er.coverage(EMPTY, {"host": hosts, "store": rows}, row_identity, tmp_path)
    assert calls == [rows[0], hosts[0]]


def test_end_read_fails_for_a_never_stored_session_with_a_bad_host_identity(monkeypatch, tmp_path):
    row = host(2, "never-stored", b"host content that the kit cannot decode as JSON")
    end = {"host": [*START["host"], row], "store": START["store"]}

    def row_identity(row):
        if isinstance(row["content"], bytes):
            raise TypeError("unsupported content")
        return identity(row)

    result = _fake_read(monkeypatch, tmp_path, end, [], row_identity=row_identity)
    assert result["check1_preservation"]["verdict"] == er.PASS
    assert result["verdict"] == er.FAIL
    assert result["check2_coverage"]["lost"] == 1


class _Text(str):
    pass


@pytest.mark.parametrize(("row_key", "host_key", "matches"), [
    (OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), False),
    (Counter(a=1), Counter(a=1, b=0), True),
    ({1: 0}.keys(), {1}, True),
    (("user", [{"n": 1}]), ("user", [{"n": True}]), True),
    ((True,), (1,), True),
    ((_Text("a"),), ("a",), True),
], ids=["ordered-dict", "counter", "dict-keys", "nested", "bool", "str-subclass"])
def test_non_plain_identities_use_reference_equality(monkeypatch, tmp_path, row_key, host_key, matches):
    reference = er.bars._ordered
    expected = reference([row_key], [host_key], eq)
    assert expected == ([(0, 0)] if matches else [])
    calls = []

    def track_reference(rows, hosts, same):
        pairs = reference(rows, hosts, same)
        assert pairs == expected
        calls.append((rows, hosts))
        return pairs

    def unexpected_fast_path(*args):
        pytest.fail("non-plain identity used the fast path")

    monkeypatch.setattr(er.bars, "_ordered", track_reference)
    monkeypatch.setattr(er, "ordered_pairs", unexpected_fast_path)
    end = {"host": [host(1, "s", "host")], "store": [stored(1, "s", "store")]}
    result = er.coverage(EMPTY, end, lambda row: row_key if "store_id" in row else host_key, tmp_path)
    assert calls == [(end["store"], end["host"])]
    assert result["matched"] == len(expected)


def test_plain_identities_use_the_fast_path(monkeypatch, tmp_path):
    ordered_pairs = er.ordered_pairs
    row_key = ("user", 1, 2.0, b"text", None)
    host_key = ("user", 1.0, 2, b"text", None)
    calls = []

    def track_fast_path(rows, hosts):
        calls.append((rows, hosts))
        return ordered_pairs(rows, hosts)

    def unexpected_reference(*args):
        pytest.fail("plain identity used the reference path")

    monkeypatch.setattr(er.bars, "_ordered", unexpected_reference)
    monkeypatch.setattr(er, "ordered_pairs", track_fast_path)
    end = {"host": [host(1, "s", "host")], "store": [stored(1, "s", "store")]}
    result = er.coverage(EMPTY, end, lambda row: row_key if "store_id" in row else host_key, tmp_path)
    assert calls == [([row_key], [host_key])]
    assert calls[0][0][0] is row_key and calls[0][1][0] is host_key
    assert result["matched"] == 1


def _uncached_hint(host, candidates, end_dir):
    """The body of _hint from 9b97645a, with its original call order."""
    text = host.get("content")
    for row in candidates:
        if row.get("role") != host.get("role"):
            continue
        stored = row.get("content")
        if isinstance(text, str) and isinstance(stored, str) and text != stored and text.strip() == stored.strip():
            return er.HINTS[0], row["store_id"]
        if (isinstance(text, str) and er.runner._swap_payload(end_dir, stored) == text) or \
                (isinstance(stored, str) and er.runner._swap_payload(end_dir, text) == stored):  # Either side the reference.
            return er.HINTS[1], row["store_id"]
    return er.HINTS[4], None


def _reference(name):
    return f"[Externalized tool output: 20 chars; ref={name}.json]"


def test_cached_hints_equal_the_uncached_original_on_2000_seeded_sessions(monkeypatch, tmp_path):
    payloads = tmp_path / "lcm-large-outputs"
    payloads.mkdir()
    for name, payload in {"valid": {"content": "decoded"}, "array": ["decoded"], "number": {"content": 3}}.items():
        (payloads / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    rng = random.Random(610182)
    contents = ["a", " a ", "\ta\n", "b", "decoded", " decoded ", None, b"a", b"decoded",
                *map(_reference, ("valid", "missing", "array", "number"))]
    roles = ["user", "assistant", "tool"]
    sessions = []
    for i in range(2000):
        session = f"s{i}"
        hosts = [host(j, session, rng.choice(contents), rng.choice(roles)) for j in range(1, rng.randrange(2, 15))]
        rows = [stored(j, session, rng.choice(contents), rng.choice(roles)) for j in range(1, rng.randrange(2, 15))]
        sessions.append({"host": hosts, "store": rows})

    actual = [er.coverage(EMPTY, end, identity, tmp_path) for end in sessions]
    for hint in (er.HINTS[0], er.HINTS[1], er.HINTS[4]):
        assert any(result["unmatched_by_hint"][hint] for result in actual)

    monkeypatch.setattr(er, "_hint", lambda host, candidates, cache: _uncached_hint(host, candidates, tmp_path))
    for end, result in zip(sessions, actual, strict=True):
        assert result == er.coverage(EMPTY, end, identity, tmp_path)


def test_payload_swaps_are_cached_once_per_plain_string_per_session(monkeypatch, tmp_path):
    reference = _reference("missing")
    calls = Counter()
    swap_payload = er.runner._swap_payload

    def track_swap(end_dir, value):
        calls[type(value), value] += 1
        return swap_payload(end_dir, value)

    monkeypatch.setattr(er.runner, "_swap_payload", track_swap)
    end = {"host": [], "store": []}
    for offset, session in [(0, "s1"), (10, "s2")]:
        end["host"].extend(host(offset + j, session, "host") for j in range(1, 4))
        end["store"].extend(stored(offset + j, session, value)
                            for j, value in enumerate([reference, reference, None, b"bytes", _Text("host")], 1))

    # Use side-specific identities so even equal text reaches the hint loop.
    result = er.coverage(EMPTY, end, lambda row: ("store_id" in row,), tmp_path)
    assert result["unmatched"] == 6
    assert calls == {(str, reference): 2, (str, "host"): 2, (type(None), None): 6,
                     (bytes, b"bytes"): 6, (_Text, "host"): 6}


def test_hint_cache_keeps_the_first_payload_error_in_loop_order(monkeypatch, tmp_path):
    calls = []

    def swap_payload(end_dir, value):
        calls.append(value)
        if value == "first eligible":
            raise KeyError("first")
        if value == "later":
            raise ValueError("later")
        return None

    monkeypatch.setattr(er.runner, "_swap_payload", swap_payload)
    end = {"host": [host(1, "s", "host")], "store": [stored(1, "s", "later", role="tool"),
           stored(2, "s", "first eligible"), stored(3, "s", "later")]}
    with pytest.raises(KeyError, match="first"):
        er.coverage(EMPTY, end, identity, tmp_path)
    assert calls == ["first eligible"]


def test_hint_cache_on_20000_unmatched_hosts_and_5000_spare_rows(tmp_path):
    payloads = tmp_path / "lcm-large-outputs"
    payloads.mkdir()
    for i in range(500):
        (payloads / f"{i}.json").write_text(json.dumps({"content": f"payload {i}"}), encoding="utf-8")
    end = {"host": [host(i + 1, "s", f"host {i}", "tool") for i in range(20_000)],
           "store": [stored(i + 1, "s", _reference(i // 10) if i % 10 == 0 else f"store {i}", "tool")
                     for i in range(5_000)]}

    started = time.perf_counter()
    result = er.coverage(EMPTY, end, identity, tmp_path)
    elapsed = time.perf_counter() - started
    print(f"Hint coverage 20000 x 5000, 500 payload references: {elapsed:.3f} s wall")
    assert (result["matched"], result["unmatched"], result["stored_rows_not_matched"], result["lost"]) == (0, 20_000, 5_000, 0)
    assert result["unmatched_by_hint"][er.HINTS[4]] == 20_000
    assert elapsed < 60
