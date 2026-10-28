"""#580 r3c: ``_match_occurrences`` is a maximum, deterministic matching, equal to an in-order greedy walk when
every row has one key, with no rows x occurrences structure (it runs inside compaction's 100k-row gap-fill)."""

from __future__ import annotations

import logging
import random
import time

import pytest

from hermes_lcm import recognition_matching
from hermes_lcm.identity_anchor import _match_occurrences, _reserve_shown

KEYS = ("a", "b", "c")
BOUND = 20.0  # s: >= 20x the measured times (<= 0.9 s) on a shared box; the no-WARNING asserts are the guard


@pytest.fixture(params=[None, pytest.param(BOUND, marks=pytest.mark.latency, id="latency")])
def matching_bound(request):
    return request.param


def _instance(rng, max_keys):
    rows = [{"store_id": 10 + i, "keys": rng.sample(KEYS, rng.randint(1, max_keys))} for i in range(rng.randint(0, 5))]
    return rows, [(100 + j, rng.choice(KEYS)) for j in range(rng.randint(0, 5))]


def _run(rows, occurrences, shuffle=None):  # shuffle: build each key set in another insertion order
    keys_of = (lambda row: set(shuffle.sample(row["keys"], len(row["keys"])))) if shuffle else (lambda row: set(row["keys"]))
    return _match_occurrences(rows, keys_of, occurrences)


def _maximum(rows, occurrences, used=frozenset()):
    if not rows:
        return 0
    return max([_maximum(rows[1:], occurrences, used)] + [1 + _maximum(rows[1:], occurrences, used | {o})
                                                      for o, key in occurrences if key in rows[0]["keys"] and o not in used])


def test_brute_force_maximum_valid_and_deterministic():
    rng = random.Random(580)
    for _ in range(20_000):
        rows, occurrences = _instance(rng, 2)
        result = _run(rows, occurrences)
        assert len(result) == _maximum(rows, occurrences)
        assert len(set(result.values())) == len(result)
        key_of, row_of = dict(occurrences), {row["store_id"]: row for row in rows}
        assert all(key_of[o] in row_of[sid]["keys"] for sid, o in result.items())
        assert _run(rows, occurrences, random.Random(1)) == result == _run(rows, occurrences, random.Random(2))


def test_single_key_rows_equal_an_in_order_greedy_walk():
    rng = random.Random(436)
    for _ in range(20_000):
        rows, occurrences = _instance(rng, 1)
        free, greedy = list(occurrences), {}
        for row in rows:
            hit = next((pair for pair in free if pair[1] == row["keys"][0]), None)
            if hit:
                free.remove(hit)
                greedy[row["store_id"]] = hit[0]
        assert list(_run(rows, occurrences).items()) == list(greedy.items())


def _untripped(caplog, rows, keys_of, occurrences, bound=None):
    started = time.perf_counter() if bound is not None else None
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.identity_anchor"):
        result = _match_occurrences(rows, keys_of, occurrences)
    assert not caplog.records
    if bound is not None:
        assert time.perf_counter() - started < bound
    return result


def test_scale_identical_rows_and_occurrences(caplog, matching_bound):
    rows = [{"store_id": i} for i in range(100_000)]
    for n in (100_000, 10):
        assert len(_untripped(caplog, rows, lambda _row: {"k"}, [(j, "k") for j in range(n)], matching_bound)) == n


def test_scale_adversarial_chain(caplog, matching_bound):
    """Row i holds k_i; the extra row on k0 shifts the whole chain; each row on the top key then searches the chain
    down to k0 and fails (dead keys make every search after the first O(1))."""
    n, key = 20_000, (lambda i: (1.0, f"k{i:05d}"))
    rows = [{"store_id": i, "keys": {key(i), key(i + 1)}} for i in range(n)] + [{"store_id": n, "keys": {key(0)}}]
    rows += [{"store_id": n + e, "keys": {key(n)}} for e in range(1, 1001)]
    assert len(_untripped(caplog, rows, lambda row: row["keys"], [(j, key(j)) for j in range(n + 1)], matching_bound)) == n + 1


def test_scale_hub_of_distinct_types(caplog, matching_bound):
    """Codex R3c probe: n rows {a, b_i} fill hub a, rows {b_i} (i < n/2) fill b_i, then n/2 rows {a} each shift one."""
    n = 50_000
    rows = [{"store_id": i, "keys": {"a", ("b", i)}} for i in range(n)]
    rows += [{"store_id": n + i, "keys": {("b", i)}} for i in range(n // 2)]
    rows += [{"store_id": 2 * n + i, "keys": {"a"}} for i in range(n // 2)]
    occurrences = [(j, "a") for j in range(n)] + [(n + i, ("b", i)) for i in range(n)]
    assert len(_untripped(caplog, rows, lambda row: row["keys"], occurrences, matching_bound)) == 2 * n


def test_scale_hub_of_one_type(caplog, matching_bound):
    n = 50_000
    rows = [{"store_id": i, "keys": {"a", "c"}} for i in range(n)] + [{"store_id": n + i, "keys": {"a"}} for i in range(n)]
    occurrences = [(j, "a") for j in range(n)] + [(n + j, "c") for j in range(n)]
    assert len(_untripped(caplog, rows, lambda row: row["keys"], occurrences, matching_bound)) == 2 * n


def test_a_spent_budget_leaves_a_valid_deterministic_direct_matching(caplog, monkeypatch):
    monkeypatch.setattr(recognition_matching, "_MATCH_WORK_PER_ITEM", 0)
    monkeypatch.setattr(recognition_matching, "_MATCH_WORK_FLOOR", 0)
    rng = random.Random(3)
    for _ in range(500):  # nothing is searched: each row takes a free key of its own in order, or nothing
        rows, occurrences = _instance(rng, 2)
        free, direct = list(occurrences), set()
        for row in rows:
            hit = next((pair for key in sorted(row["keys"]) for pair in free if pair[1] == key), None)
            if hit:
                free.remove(hit)
                direct.add(row["store_id"])
        result, key_of = _run(rows, occurrences, random.Random(1)), dict(occurrences)
        assert set(result) == direct and len(set(result.values())) == len(result)
        assert all(key_of[o] in row["keys"] for row in rows for sid, o in result.items() if sid == row["store_id"])
        assert result == _run(rows, occurrences, random.Random(2))
    caplog.clear()
    rows = [{"store_id": 1, "keys": ["a", "b"]}, {"store_id": 2, "keys": ["b"]}] + [{"store_id": i, "keys": ["a"]} for i in (3, 4, 5)]
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.identity_anchor"):
        assert _run(rows, [(10, "a"), (11, "b")]) == {1: 10, 2: 11}
    assert [record.levelname for record in caplog.records] == ["WARNING"]


# -- #583: a NULL-stamped row the host now shows stamped is reserved by form alone, after the stamped pass.

def _null(sid, *forms):
    return {"store_id": sid, "observed_at": None, "forms": set(forms)}


def _reserve(caplog, pool, occurrences, bound=None):
    started = time.perf_counter() if bound is not None else None
    with caplog.at_level(logging.WARNING, logger="hermes_lcm.identity_anchor"):
        result = _reserve_shown(pool, lambda row: row["forms"], occurrences)
    assert not caplog.records
    if bound is not None:
        assert time.perf_counter() - started < bound
    return result


def test_null_rows_answer_a_leftover_stamped_occurrence_of_their_form_in_store_order(caplog, matching_bound):
    pool = [{"store_id": 1, "observed_at": 1.0, "forms": {"foo"}}, _null(2, "foo"), _null(3, "foo"), _null(4, "bar")]
    assert _reserve(caplog, pool, [(0, (1.0, "foo"))], bound=matching_bound) == {1}  # the stamped pass keeps its occurrence
    assert _reserve(caplog, pool, [(0, (1.0, "foo")), (1, (2.0, "foo"))], bound=matching_bound) == {1, 2}  # (a) first NULL row only
    assert _reserve(caplog, pool, [(0, (None, "foo")), (1, (2.0, "foo"))], bound=matching_bound) == {2, 3}  # unstamped: pass one
    assert _reserve(caplog, pool, [(0, (5.0, "R\n\nfoo"))], bound=matching_bound) == set()  # (b) shown inside a composite only
    assert _reserve(caplog, pool[1:], [], bound=matching_bound) == set()


def test_scale_null_rows_and_occurrences_at_distinct_stamps(caplog, matching_bound):
    """100k NULL rows x 100k occurrences of one form, each at its own stamp (r3f's rows x stamps keys: 1e10)."""
    rows = [_null(i, "f") for i in range(100_000)]
    for n in (100_000, 10):
        assert len(_reserve(caplog, rows, [(j, (float(j), "f")) for j in range(n)], bound=matching_bound)) == n


def test_scale_null_hub_of_many_forms(caplog, matching_bound):
    """NULL rows answer {a, b_i} (exact + override) and {b_i}; hub a and each b_i shown at distinct stamps."""
    n = 50_000
    rows = [_null(i, "a", ("b", i)) for i in range(n)] + [_null(n + i, ("b", i)) for i in range(n // 2)]
    rows += [_null(2 * n + i, "a") for i in range(n // 2)]
    occurrences = [(j, (float(j), "a")) for j in range(n)] + [(n + i, (float(n + i), ("b", i))) for i in range(n)]
    assert len(_reserve(caplog, rows, occurrences, bound=matching_bound)) == 2 * n


def test_a_recorded_alias_stamp_reserves_only_its_own_row(caplog, matching_bound):
    """B-ID-3: a stamped row answers its own stamp and its recorded alias stamps; another row of the same form
    never answers that alias."""
    pool = [{"store_id": 1, "observed_at": 20.0, "forms": {"u"}}, {"store_id": 2, "observed_at": 35.0, "forms": {"u"}}]
    shown = [(0, (10.0, "u"))]
    assert _reserve(caplog, pool, shown, bound=matching_bound) == set()  # no alias recorded: nothing answers 10
    assert _reserve_shown(pool, lambda row: row["forms"], shown, {1: {10.0}}) == {1}
    assert _reserve_shown(pool, lambda row: row["forms"], shown, {2: {10.0}}) == {2}
    assert _reserve_shown(pool, lambda row: row["forms"], [(0, (10.0, "u")), (1, (20.0, "u"))], {1: {10.0}}) == {1}


def test_a_null_row_takes_an_occurrence_a_stamped_row_can_move_off(caplog, matching_bound):
    """PR #590 bot P1: S answers {a, b} at t, N (NULL) only a; the view shows a@t and b@t. Pass one gives S a@t;
    S moves to b@t (another of its OWN keys) so N takes a@t. A stamped row that cannot move is never unmatched."""
    s, n = {"store_id": 1, "observed_at": 1.0, "forms": {"a", "b"}}, _null(2, "a")
    assert _reserve(caplog, [s, n], [(0, (1.0, "a")), (1, (1.0, "b"))], bound=matching_bound) == {1, 2}
    assert _reserve(caplog, [s, n], [(0, (1.0, "a"))], bound=matching_bound) == {1}  # S cannot move: S keeps it
    s2 = {"store_id": 3, "observed_at": 1.0, "forms": {"b"}}  # b@t is S2's: S has nowhere to go
    assert _reserve(caplog, [s, s2, n], [(0, (1.0, "a")), (1, (1.0, "b"))], bound=matching_bound) == {1, 3}


def test_scale_null_rows_rerouting_stamped_rows(caplog, matching_bound):
    """50k stamped rows {a, b} at their own stamps hold a (pass one), 50k NULL rows {a}: each moves one stamped
    row to its b; then 50k stuck stamped rows {a} and 50k NULL rows {a} (every holder dead after one look)."""
    n = 50_000
    rows = [{"store_id": i, "observed_at": float(i), "forms": {"a", "b"}} for i in range(n)] + [_null(n + i, "a") for i in range(n)]
    occurrences = [(i, (float(i), "a")) for i in range(n)] + [(n + i, (float(i), "b")) for i in range(n)]
    assert len(_reserve(caplog, rows, occurrences, bound=matching_bound)) == 2 * n
    rows = [{"store_id": i, "observed_at": float(i), "forms": {"a"}} for i in range(n)] + [_null(n + i, "a") for i in range(n)]
    assert len(_reserve(caplog, rows, [(i, (float(i), "a")) for i in range(n)], bound=matching_bound)) == n
