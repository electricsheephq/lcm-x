"""#2840: the ordinal planner's fallback is bounded and the re-spacing apply is set-based.

The references below are copies of plan_ordinals and apply_respace as they were before #2840
(main 2d17cbd1). Cost is pinned by counting candidate windows and statements, never by wall clock.
"""

import logging
import math
import random
from bisect import bisect_left
from collections import Counter, defaultdict

import pytest

from hermes_lcm import ingest_ledger as ledger
from hermes_lcm.store import MessageStore
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_ingest_ledger_writes import count, engines, restart, sql_rows  # noqa: F401

P = ledger.Placement
LIMIT = getattr(ledger, "_RESPACE_WINDOW_LIMIT", 256)
DIGEST = b"d" * 16


def reference_plan_ordinals(existing, placements, stats=None):
    """Pre-#2840 plan_ordinals; `stats` records each fallback window width (instrumentation only)."""
    original = sorted(set(existing))
    original_set = set(original)
    if any(not isinstance(value, int) or not 0 < value <= ledger._MAX_ORDINAL for value in original):
        raise ledger.LedgerError("existing ordinals must be positive signed 64-bit integers")
    positions, groups = original.copy(), defaultdict(list)
    previous_left, previous_right, previous_gap = 0, 0, 0
    for index, placement in enumerate(placements):
        left, right, supersedes = ledger.Placement(*placement)
        if any(value is not None and value not in original_set for value in (left, right, supersedes)):
            raise ledger.LedgerError("placement bound is not an existing ordinal")
        left_bound, right_bound = left or 0, right if right is not None else ledger._MAX_ORDINAL + 1
        if left_bound >= right_bound or left_bound < previous_left or right_bound < previous_right:
            raise ledger.LedgerError("placement bounds must be non-decreasing in view order")
        previous_left, previous_right = left_bound, right_bound
        if supersedes is not None and left_bound <= supersedes < right_bound:
            gap = bisect_left(original, supersedes) + 1
        else:
            if left is not None and right is not None and left >= right:
                raise ledger.LedgerError("placement bounds are out of order")
            gap = len(original) if right is None else bisect_left(original, right)
        previous_gap = max(gap, previous_gap)
        groups[previous_gap].append(index)
    result = [None] * sum(map(len, groups.values()))
    for gap, indexes in sorted(groups.items()):
        if result[indexes[0]] is not None:
            continue
        lower = positions[gap - 1] if gap else 0
        upper = positions[gap] if gap < len(positions) else None
        step = ledger.ORDINAL_STEP if upper is None else (upper - lower) // (len(indexes) + 1)
        if step > 0 and lower + step * len(indexes) <= ledger._MAX_ORDINAL:
            for offset, index in enumerate(indexes, 1):
                result[index] = (lower + step * offset if upper is None
                                 else lower + (upper - lower) * offset // (len(indexes) + 1))
            continue
        window = None
        for width in range(len(positions) + 1):
            for start in range(max(0, gap - width), min(gap, len(positions) - width) + 1):
                end = start + width
                lower = positions[start - 1] if start else 0
                count = width + sum(len(groups[g]) for g in range(start, end + 1))
                step = ((positions[end] - lower) // (count + 1) if end < len(positions)
                        else min(ledger.ORDINAL_STEP, (ledger._MAX_ORDINAL - lower) // count))
                if step >= ledger.MIN_GAP:
                    window = start, end, lower, step
                    break
            if window is not None:
                break
        if window is None:
            raise ledger.LedgerError("signed 64-bit ordinal space exhausted")
        start, end, value, step = window
        if stats is not None:
            stats.append(end - start)
        for g in range(start, end + 1):
            for index in groups[g]:
                value += step
                result[index] = value
            if g < end:
                value += step
                positions[g] = value
    return result, {old: new for old, new in zip(original, positions) if old != new}


def reference_apply_respace(conn, lineage, mapping):
    """Pre-#2840 apply_respace: one SELECT, then two per-label UPDATE phases."""
    ledger._require_owner(conn)
    occupied = {row[0] for row in conn.execute(
        "SELECT ordinal FROM occurrence_bindings WHERE lineage=? AND ordinal IS NOT NULL", (lineage,))}
    if (any(value <= 0 or value > ledger._MAX_ORDINAL for value in mapping.values())
            or len(set(mapping.values())) != len(mapping)
            or set(mapping.values()) & (occupied - mapping.keys())):
        raise ledger.LedgerError("invalid re-spacing map")
    temporary, count, staged = min(occupied | {0}), 0, []
    for old, new in mapping.items():
        temporary -= 1
        if temporary < -(1 << 63):
            raise ledger.LedgerError("temporary ordinal space exhausted")
        cursor = conn.execute("UPDATE occurrence_bindings SET ordinal=? WHERE lineage=? AND ordinal=?",
                              (temporary, lineage, old))
        count += cursor.rowcount
        staged.append((temporary, new))
    for temporary, new in staged:
        conn.execute("UPDATE occurrence_bindings SET ordinal=? WHERE lineage=? AND ordinal=?", (new, lineage, temporary))
    return count


def outcome(function, *args):
    try:
        return "ok", function(*args)
    except ledger.LedgerError as exc:
        return type(exc), str(exc)


def clamped_gaps(labels, placements):
    gaps, previous = [], 0
    for left, right, supersedes in placements:
        bounds = (left or 0, right if right is not None else ledger._MAX_ORDINAL + 1)
        if supersedes is not None and bounds[0] <= supersedes < bounds[1]:
            gap = bisect_left(labels, supersedes) + 1
        else:
            gap = len(labels) if right is None else bisect_left(labels, right)
        previous = max(previous, gap)
        gaps.append(previous)
    return gaps


def assert_valid_plan(existing, placements, values, mapping):
    """The plan properties of #2840 item 3, checked one by one (no reference comparison)."""
    labels, top = sorted(set(existing)), ledger._MAX_ORDINAL
    assert len(values) == len(placements)
    assert set(mapping) <= set(labels) and all(mapping[k] != k for k in mapping)  # one entry per label, moved only
    moved = [mapping.get(label, label) for label in labels]
    assert all(a < b for a, b in zip(moved, moved[1:]))  # existing labels keep their relative order
    assert all(0 < value <= top for value in moved + values)
    assert all(a < b for a, b in zip(values, values[1:]))  # strictly increasing in view order
    assert not set(values) & set(moved)
    for placement, value, gap in zip(placements, values, clamped_gaps(labels, placements)):
        low = mapping.get(placement.left, placement.left) if placement.left is not None else 0
        high = mapping.get(placement.right, placement.right) if placement.right is not None else top + 1
        assert low < value < high  # strictly between its bounds after the map
        assert (moved[gap - 1] if gap else 0) < value < (moved[gap] if gap < len(moved) else top + 1)


def random_labels(rng):
    size = rng.choice([0, 1, 2, 3, 4, 6, 9, 14, 20, 30, 45, 70, 120])
    choices = [1, 1, 2, 3, rng.randint(1, ledger.MIN_GAP), ledger.MIN_GAP - 1, ledger.MIN_GAP,
               ledger.MIN_GAP + 1, rng.randint(ledger.MIN_GAP, ledger.ORDINAL_STEP), ledger.ORDINAL_STEP]
    steps, packed = [], rng.random() < 0.6
    while len(steps) < size:
        if packed and rng.random() < 0.3:
            steps += [rng.choice([1, 1, 2, ledger.MIN_GAP - 1])] * rng.randint(2, 40)
        else:
            steps.append(rng.choice(choices))
    steps = steps[:size]
    span = sum(steps)
    first = rng.choice([1, rng.randint(1, 1 << 30), rng.randint(1, 3 * ledger.ORDINAL_STEP),
                        ledger._MAX_ORDINAL - span - rng.randint(0, 3 * ledger.ORDINAL_STEP)])
    labels, value = [], max(1, first)
    for step in steps:
        labels.append(value)
        value += step
    labels = [label for label in labels if label <= ledger._MAX_ORDINAL]
    existing = labels + rng.sample(labels, rng.randint(0, min(3, len(labels))))  # duplicates (part witnesses)
    rng.shuffle(existing)
    return labels, existing


def random_placements(rng, labels):
    placements, left_index, right_index, shapes = [], 0, 0, set()
    for gap in sorted(rng.choices(range(len(labels) + 1), k=rng.randint(1, 4))):
        for _ in range(rng.randint(1, 4)):
            shape = rng.choice(["both", "both", "both", "left", "right"])
            # Keep most sequences valid: bounds never decrease (a few invalid ones still slip through).
            shape = "left" if "left" in shapes else "both" if shape == "right" and shapes - {"right"} else shape
            shapes.add(shape)
            left_index = max(left_index, rng.randint(max(0, gap - 3), gap))
            right_index = max(right_index, rng.randint(gap, min(len(labels), gap + 3)))
            left = labels[left_index - 1] if left_index and shape != "right" else None
            right = labels[right_index] if right_index < len(labels) and shape != "left" else None
            supersedes = rng.choice(labels) if labels and rng.random() < 0.3 else None
            placements.append(P(left, right, supersedes))
    return placements


def test_planner_matches_reference_within_the_limit_and_is_valid_beyond(monkeypatch):
    rng, tally = random.Random(2840), dict(equal=0, equal_fallback=0, beyond=0, errors=0)
    for case in range(4000):
        limit = rng.choice([LIMIT, LIMIT, 0, 1, 2, 3, 5, 8, 13])
        monkeypatch.setattr(ledger, "_RESPACE_WINDOW_LIMIT", limit, raising=False)
        labels, existing = random_labels(rng)
        placements = random_placements(rng, labels)
        widths = []
        expected = outcome(reference_plan_ordinals, existing, placements, widths)
        actual = outcome(ledger.plan_ordinals, existing, placements)
        if all(width <= limit for width in widths) and expected[1] != "signed 64-bit ordinal space exhausted":
            assert actual == expected, (case, limit, existing, placements)
            tally["equal"] += 1
            tally["equal_fallback"] += bool(widths)
            tally["errors"] += expected[0] != "ok"
        else:
            assert actual[0] == "ok", (case, limit, existing, placements, actual)
            assert_valid_plan(existing, placements, *actual[1])
            tally["beyond"] += 1
    print("planner sweep", tally)
    assert tally["equal"] - tally["errors"] >= 2000 and tally["equal_fallback"] >= 300 and tally["beyond"] >= 200, tally


@pytest.mark.parametrize("existing, placements", [
    ([], [P(None, None)] * 2), ([1 << 20], [P(1 << 20, None)] * 2), ([100, 200, 300], [P(100, 300)] * 2),
    ([100, 110], [P(100, 110)] * 3), ([100, 200, 300], [P(None, None, 100)]),
    ([1024, 1025, 8192], [P(1024, 1025)] * 2), ([1, 2, 3], [P(1, 2), P(2, 3)]),
    ([100, 200, 250, 300], [P(100, 300), P(100, 300, 200)]), ([100, 200, 300], [P(100, None), P(100, None, 200)]),
    ([100, 200, 300], [P(200, 300), P(100, 300)]), ([100, 200, 300], [P(100, None), P(100, 300)]),
    ([100, 200, 300, 400], [P(100, 200, 400), P(200, 300)]), ([100, 200, 300], [P(200, 300, 100)]),
    (list(range(1, 2 * LIMIT + 1)), [P(LIMIT, LIMIT + 1)]),  # its window is exactly LIMIT wide
])
def test_existing_shapes_match_reference(existing, placements):
    assert outcome(ledger.plan_ordinals, existing, placements) == outcome(reference_plan_ordinals, existing, placements)


def count_candidates(monkeypatch, budget):
    """Count candidate windows (#2840 helper) and abort any planner whose group lookups pass `budget`."""
    calls = {"windows": 0, "lookups": 0}
    step = getattr(ledger, "_window_step", None)

    def window_step(*args):
        calls["windows"] += 1
        return step(*args)

    class Groups(defaultdict):
        def __getitem__(self, key):
            calls["lookups"] += 1
            assert calls["lookups"] <= budget, "planner exceeded its work budget: unbounded window search"
            return super().__getitem__(key)

    monkeypatch.setattr(ledger, "_window_step", window_step, raising=False)
    monkeypatch.setattr(ledger, "defaultdict", Groups)
    return calls


def reporter_shape():
    """20,014 labels; a run 5,458 wide at indices 4..5461 packed below MIN_GAP, a gap of 1 at its middle."""
    step, run = ledger.ORDINAL_STEP, 5458
    labels = [step * (i + 1) for i in range(4)]
    for i in range(run):
        labels.append(labels[-1] + (1 if i == run // 2 else ledger.MIN_GAP - 1))
    labels += [labels[-1] + step * (i + 1) for i in range(20014 - len(labels))]
    middle = 4 + run // 2
    return labels, P(labels[middle - 1], labels[middle])


def test_reporter_shape_search_is_bounded_and_packing_does_not_recur(monkeypatch):
    labels, placement = reporter_shape()
    calls = count_candidates(monkeypatch, budget=LIMIT * (LIMIT + 2))
    values, mapping = ledger.plan_ordinals(labels, [placement])
    assert calls["windows"] <= LIMIT * (LIMIT + 2)
    assert_valid_plan(labels, [placement], values, mapping)
    current, left, right = sorted([mapping.get(v, v) for v in labels] + values), values[0], mapping[placement.right]
    for _ in range(2):  # a second and a third insert at the same spot take the fast path
        calls["windows"] = 0
        values, mapping = ledger.plan_ordinals(current, [P(left, right)])
        assert mapping == {} and calls["windows"] == 0 and left < values[0] < right
        current, left = sorted(current + values), values[0]


@pytest.mark.parametrize("size", [3 * LIMIT, 3 * LIMIT + 101])
def test_beyond_the_limit_at_default_size_is_valid_and_bounded(monkeypatch, size):
    labels = list(range(1, size + 1))
    middle = size // 2
    placements = [P(labels[middle - 1], labels[middle]), P(labels[middle - 1], labels[middle], labels[middle - 1]),
                  P(labels[middle], labels[middle + 1])]
    calls = count_candidates(monkeypatch, budget=10 * size)
    values, mapping = ledger.plan_ordinals(labels, placements)
    assert calls["windows"] <= LIMIT * (LIMIT + 2)
    assert_valid_plan(labels, placements, values, mapping)
    assert set(mapping) == set(labels) - {ledger.ORDINAL_STEP * k for k in range(1, size + len(placements) + 1)}


def test_space_exhaustion_is_still_raised(monkeypatch):
    for name, value in (("_MAX_ORDINAL", 10), ("ORDINAL_STEP", 4), ("MIN_GAP", 2), ("_RESPACE_WINDOW_LIMIT", 1)):
        monkeypatch.setattr(ledger, name, value, raising=False)
    labels = list(range(1, 11))
    with pytest.raises(ledger.LedgerError, match="space exhausted"):
        ledger.plan_ordinals(labels, [P(5, 6)])  # 11 labels cannot fit in 1..10
    monkeypatch.setattr(ledger, "_MAX_ORDINAL", 20)
    values, mapping = ledger.plan_ordinals(labels, [P(5, 6)])  # the whole lineage at the largest step that fits
    assert_valid_plan(labels, [P(5, 6)], values, mapping)
    monkeypatch.setattr(ledger, "_RESPACE_WINDOW_LIMIT", 256, raising=False)  # within the limit: as before
    assert outcome(ledger.plan_ordinals, labels, [P(5, 6)]) == outcome(reference_plan_ordinals, labels, [P(5, 6)])


def test_fallback_logs_counts_only(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger=ledger.__name__)
    ledger.plan_ordinals(list(range(1, 11)), [P(5, 6)])
    monkeypatch.setattr(ledger, "_RESPACE_WINDOW_LIMIT", 2, raising=False)
    ledger.plan_ordinals(list(range(1, 11)), [P(5, 6)])
    lines = [r.getMessage() for r in caplog.records if "respace fallback" in r.getMessage()]
    assert lines == ["LCM ledger ordinals respace fallback planned: window 5, 5 labels to move",
                     "LCM ledger ordinals respace fallback planned: window whole lineage, 10 labels to move"]


@pytest.fixture
def stores(tmp_path):
    made = [MessageStore(tmp_path / f"{name}.db") for name in ("reference", "batched")]
    for store in made:
        store.ensure_ingest_ledger()
    yield made
    for store in made:
        store.close()


def table(store):
    return store.connection.execute("SELECT *, typeof(ordinal) FROM occurrence_bindings ORDER BY occ_id").fetchall()


def random_apply_case(rng):
    size = rng.choice([5, 6, 9, 15, 30, 60, 120, 250]) if rng.random() > 0.02 else rng.randint(1000, 3000)
    labels = sorted(rng.sample(range(1, 40 * size), size))
    rows = []
    for label in labels:
        rows.append(("A", label, rng.choice(["stored", "stored", "migrated", "ambiguous", "composite_part", "split_part"])))
        for _ in range(rng.choice([0, 0, 0, 1, 2])):  # part witnesses share the label
            rows.append(("A", label, rng.choice(["composite_part", "split_part"])))
    rows += [("A", None, "composite_part")] * rng.randint(0, 2)
    if rng.random() < 0.05:  # a lineage whose lowest label leaves few temporaries
        rows.append(("A", -(1 << 63) + rng.randint(0, 3), "composite_part"))
    rows += [("B", label, "stored") for label in rng.sample(labels, min(len(labels), rng.randint(0, 20)))]
    rng.shuffle(rows)
    keys = rng.sample(labels, rng.randint(1, len(labels)))
    absent = [value for value in rng.sample(range(40 * size, 80 * size), rng.randint(0, 3))]
    keys += absent
    occupied = {ordinal for lineage, ordinal, _ in rows if lineage == "A" and ordinal is not None}
    fresh = [value for value in rng.sample(range(1, 100 * size), 2 * len(keys)) if value not in occupied]
    pool = list(dict.fromkeys(keys[:len(keys) - len(absent)] + fresh))  # new values may equal other keys' old values
    rng.shuffle(pool)
    mapping = dict(zip(keys, pool))
    if rng.random() < 0.05 and len(labels) > len(keys):
        mapping[keys[0]] = next(label for label in labels if label not in mapping)  # invalid: hits an unmoved label
    return rows, mapping


def test_batched_apply_matches_reference(stores):
    rng, applied, errors, largest = random.Random(28400), 0, Counter(), 0
    for case in range(500):
        rows, mapping = random_apply_case(rng)
        largest = max(largest, sum(lineage == "A" for lineage, _ordinal, _disposition in rows))
        results = []
        for store, apply in zip(stores, (reference_apply_respace, ledger.apply_respace)):
            conn = store.connection
            with store.atomic():
                conn.execute("DELETE FROM occurrence_bindings")
                for lineage, ordinal, disposition in rows:
                    ledger.insert_bindings(conn, lineage, [dict(ordinal=ordinal, identity_digest=DIGEST,
                                                                disposition=disposition, view_seq=0)])
                results.append((outcome(apply, conn, "A", mapping), table(store)))
        assert results[0] == results[1], case
        applied += results[0][0][0] == "ok"
        errors.update([results[0][0][1]] if results[0][0][0] != "ok" else [])
    print("apply sweep", applied, "applied;", dict(errors), "largest lineage", largest)
    assert applied >= 400 and len(errors) == 2 and largest >= 1000


def test_batched_apply_statement_count_on_reporter_shape(stores):
    labels, _placement = reporter_shape()
    mapping = {v: (i + 1) * ledger.ORDINAL_STEP for i, v in enumerate(labels) if v != (i + 1) * ledger.ORDINAL_STEP}
    store = stores[1]
    conn, statements = store.connection, []
    with store.atomic():
        ledger.insert_bindings(conn, "L", [dict(ordinal=o, identity_digest=DIGEST, disposition="stored", view_seq=0)
                                           for o in labels])
        conn.set_trace_callback(statements.append)
        try:
            moved = ledger.apply_respace(conn, "L", mapping)
        finally:
            conn.set_trace_callback(None)
        assert ledger.lineage_ordinals(conn, "L") == sorted(mapping.get(v, v) for v in labels)
    assert moved == len(mapping) == 20010
    writes = [sql for sql in statements if sql.lstrip().upper().startswith(("UPDATE", "SELECT"))]
    chunk = getattr(ledger, "_RESPACE_CHUNK", 300)
    assert len(writes) <= 2 * math.ceil(len(mapping) / chunk) + 2, len(writes)


@pytest.mark.parametrize("limit", [LIMIT, 2], ids=["window", "whole-lineage"])
def test_ingest_through_the_fallback_stores_each_row_once_in_order(engines, monkeypatch, limit):  # noqa: F811
    monkeypatch.setattr(ledger, "_RESPACE_WINDOW_LIMIT", limit, raising=False)
    engine = engines()
    view = [_m("user" if i % 2 == 0 else "assistant", f"turn {i}", 10. * (i + 1)) for i in range(10)]
    engine._ingest_messages(view)
    conn = engine._store._conn
    bindings = sql_rows(engine, "occurrence_bindings")
    assert len(bindings) == 10
    with engine._store.atomic():  # pack the lineage so the middle insert must re-space
        ledger.apply_respace(conn, bindings[0]["lineage"], {b["ordinal"]: i + 1 for i, b in enumerate(bindings)})
    view = view[:5] + [_m("user", "inserted", 55.)] + view[5:]
    for _ in range(2):  # the second pass is a replay: nothing new is stored
        restart(engine)
        engine._ingest_messages(view)
        bindings = sorted(sql_rows(engine, "occurrence_bindings"), key=lambda b: b["ordinal"])
        contents = [engine._store.get(b["store_id"])["content"] for b in bindings]
        assert contents == [m["content"] for m in view]
        assert len({b["store_id"] for b in bindings}) == len(view)
        stored = [m["content"] for m in engine._store.get_session_messages(engine._session_id)]
        assert sorted(stored) == sorted(m["content"] for m in view)
        assert count(engine, "respaced") == 1


@pytest.mark.parametrize("outside", [ledger.ORDINAL_STEP, 2 * ledger.ORDINAL_STEP + 1], ids=["collides", "inverts"])
def test_partial_lineage_past_the_limit_loads_the_whole_lineage(tmp_path, outside):
    """NULL-ordinal anchors hide every right bound, so the caller's subset holds more than the limit.

    Built with direct inserts and a hand-made attempt (as the #2858 review probe does): whether ordinary
    ingest produces NULL-ordinal anchors is not established. The unseen label sits low in the range.
    """
    from hermes_lcm.engine import LCMEngine

    store = MessageStore(tmp_path / "partial.db")
    store.ensure_ingest_ledger()
    conn, top = store.connection, ledger._MAX_ORDINAL
    labels = [top - (1 << 32), *range(top - LIMIT, top + 1)]  # LIMIT + 2 distinct left bounds
    size = 3 * len(labels)
    matched = {idx: idx + 1 for idx in range(size) if idx % 3 != 1}  # a NULL anchor follows every new row
    pending = {idx: idx + 1 for idx in range(size) if idx % 3 == 1}
    with store.atomic(immediate=True):
        ledger.insert_bindings(conn, "A", [dict(ordinal=outside, identity_digest=DIGEST, disposition="stored",
                                                view_seq=0, store_id=9999)] +
                               [dict(ordinal=value, identity_digest=DIGEST, disposition="stored", view_seq=0,
                                     store_id=idx + 1) for i, label in enumerate(labels)
                                for idx, value in ((3 * i, label), (3 * i + 2, None))])
    before = [r[0] for r in conn.execute("SELECT store_id FROM occurrence_bindings WHERE ordinal IS NOT NULL "
                                         "ORDER BY ordinal")]
    entries = [ledger.ViewEntry(DIGEST, None, None, None)] * size
    salt = store.ingest_ledger_salt()
    attempt = dict(adopted=True, salt=salt, lineage="A", resolved=None, stored=pending, matched=matched,
                   entries=entries, stamps=[None] * size, chain=ledger.prefix_chain(salt, entries),
                   observations=[], remainders={})
    engine = object.__new__(LCMEngine)
    engine._ensure_storage = lambda: None
    engine._store, engine._session_id = store, "S"
    callbacks = []
    try:
        with store.ingest_atomic():
            engine._write_ingest_ledger(attempt, callbacks, {"ledger_ms": 0})
        for callback in callbacks:
            callback()
        ordinal = dict(conn.execute("SELECT store_id, ordinal FROM occurrence_bindings WHERE ordinal IS NOT NULL"))
        counters = engine.__dict__.get("_ingest_ledger_counters", {})
        assert counters.get("ledger.write_failed", 0) == 0, dict(counters)
        new = [ordinal.get(sid) for _idx, sid in sorted(pending.items())]
        assert None not in new and len(new) == len(labels)  # every new binding is written
        assert new == sorted(set(new))  # stored order of the new rows is their view order
        assert all(ordinal[sid] > ordinal[idx] for idx, sid in pending.items())  # each after its left bound
        after = sorted((sid for sid in ordinal if sid not in pending.values()), key=ordinal.get)
        assert after == before  # no existing label, seen or unseen, changes its relative order
    finally:
        store.close()
