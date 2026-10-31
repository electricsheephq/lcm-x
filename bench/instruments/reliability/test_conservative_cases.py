"""#1879: conservative errors become credit only with explicit occurrence witnesses."""
from collections import Counter, defaultdict
from itertools import permutations
import random
from types import SimpleNamespace

import pytest

from bench.instruments.reliability import probe
from bench.instruments.reliability.scorers import multiset, projections
from bench.instruments.reliability.test_witness_scoring import TREE, merge_events, witnessed_host
from tests import test_reliability_scorers as fixture


@pytest.mark.parametrize("witness", [True, False])
def test_unrelated_enclosing_row_does_not_suppress_parts(witness):
    parts = ("alpha", "beta")
    held = "\n\n".join(parts)
    enclosing = "prefix " + held + " suffix"
    host = {"chat": {"keys": {("user", multiset.h(held)): {"n": 1, "ids": [10]},
                            ("user", multiset.h(enclosing)): {"n": 1, "ids": [11]}}, "tags": {}}}
    if witness:
        multiset.merge_witnesses(host, merge_events(parts), lambda _s: "chat")
    rows = [(i, "S0", "user", text) for i, text in enumerate((*parts, enclosing), 1)]
    out = multiset.score([("user", held)], rows, host["chat"]["keys"])
    assert out["deficit_rows"] == int(not witness)
    assert out["verdict"] == ("PASS" if witness else "FAIL")
    if witness:
        assert out["held_composites_as_parts"][0]["witness"]["event_index"] == 0


@pytest.mark.parametrize("witness", [True, False])
def test_overlapping_spans_have_order_independent_assignment(witness):
    parts = ("x", "y", "x")
    text = "\n\n".join(parts)
    host = witnessed_host(parts)
    if not witness:
        host[("user", multiset.h(text))].pop("span_witnesses")
    results = []
    for expected in permutations([("user", "x"), ("user", "x\n\ny")]):
        out = multiset.score(list(expected), [(93, "S0", "user", text)], host)
        assert out["deficit_rows"] == (0 if witness else 2)
        results.append(out["embedded_span_credits"])
    assert results[0] == results[1]
    if witness:
        assert sorted((c["byte_offset"], c["bytes"]) for c in results[0]) == [(0, 4), (6, 1)]


@pytest.mark.parametrize("witness", [True, False])
def test_span_choice_preserves_disjoint_merge_parts_fallback(witness):
    host = {"chat": {"keys": {}, "tags": {}}}
    events = merge_events(("a", "b", "c"))
    if witness:
        events += merge_events(("a", "b"))
    multiset.merge_witnesses(host, events, lambda _s: "chat")
    # License the enclosing row, so PASS proves preservation and surplus accounting.
    host["chat"]["keys"][("user", multiset.h("a\n\nb\n\nc"))]["n"] = 1
    rows = [(1, "S0", "user", "a\n\nb\n\nc"), (2, "S0", "user", "a"), (3, "S0", "user", "b")]
    for expected in permutations([("user", "b\n\nc"), ("user", "a\n\nb")]):
        out = multiset.score(list(expected), rows, host["chat"]["keys"])
        assert out["deficit_rows"] == int(not witness)
        if witness:
            assert out["verdict"] == "PASS"
            assert out["embedded_span_credits"][0]["sha256"] == multiset.h("b\n\nc")
            assert out["held_composites_as_parts"][0]["witness"]["event_index"] == 1


def test_unsorted_row_cannot_be_exact_match_span_host_and_merge_part():
    k = "qq\n\npp"
    c = k + "\n\nxx"
    host = {"chat": {"keys": {("user", multiset.h(k)): {"n": 2, "ids": [2, 5]}}, "tags": {}}}
    multiset.merge_witnesses(host, merge_events(("qq", "pp")) + merge_events((k, "xx")), lambda _s: "chat")
    rows = [(5, "S0", "user", k), (2, "S0", "user", k), (7, "S0", "user", "xx")]
    for expected in permutations([("user", k), ("user", "pp"), ("user", c)]):
        out = multiset.score(list(expected), rows, host["chat"]["keys"])
        assert out["deficit_rows"] == 1 and out["verdict"] == "FAIL"


def assert_row_capacity(expected, rows, out):
    """Whole-row reservations and disjoint spans must fit each stored key's actual rows."""
    keys = {sid: (role, multiset.h(text)) for sid, _session, role, text in rows}
    part_ids = [sid for comp in out["held_composites_as_parts"] for part in comp["parts"]
                for sid in part["store_ids"]]
    hosts = {c["store_id"] for c in out["embedded_span_credits"]}
    assert len(part_ids) == len(set(part_ids))
    assert not hosts.intersection(part_ids)
    spans = defaultdict(list)
    for credit in out["embedded_span_credits"]:
        spans[credit["store_id"]].append((credit["byte_offset"], credit["byte_offset"] + credit["bytes"]))
    for intervals in spans.values():
        intervals.sort()
        assert all(end <= start for (_a, end), (start, _b) in zip(intervals, intervals[1:]))
    want = Counter((role, multiset.h(text)) for role, text in expected)
    embedded = Counter((c["role"], c["sha256"]) for c in out["embedded_span_credits"])
    missing = {(m["role"], m["sha256"]): m["expected"] - m["stored"] for m in out["missing"]}
    # Tests generate user rows only. Parts receipts identify a composite by its exact joined bytes.
    by_id = {sid: text for sid, _session, _role, text in rows}
    paired = Counter()
    for comp in out["held_composites_as_parts"]:
        text = "\n\n".join(by_id[p["store_ids"][0]] for p in comp["parts"] for _ in range(p["uses"]))
        paired[("user", multiset.h(text))] += comp["as_parts"]
    demand = Counter(keys[sid] for sid in part_ids + list(hosts))
    for key, n in want.items():
        exact = n - embedded[key] - missing.get(key, 0) - paired[key]
        assert exact >= 0
        demand[key] += exact
    supply = Counter(keys.values())
    assert all(n <= supply[key] for key, n in demand.items())


def test_fixed_seed_row_order_fuzz_has_no_capacity_violations():
    rng = random.Random(1879)
    atoms = ["aa", "bb", "cc", "dd"]

    def text():
        return "\n\n".join(rng.choice(atoms) for _ in range(rng.choice([1, 1, 2, 2, 3])))

    for case in range(2000):
        composites = sorted({text() for _ in range(rng.randint(1, 3))})
        events = []
        for composite in composites:
            parts = composite.split("\n\n")
            if len(parts) >= 2:
                events += merge_events(tuple(parts))
        nested = []
        for composite in composites[:rng.randint(0, 2)]:
            atom = rng.choice(atoms)
            events += merge_events((composite, atom))
            nested.append(composite + "\n\n" + atom)
        host_keys = {("user", multiset.h(t)): {"n": rng.randint(1, 3), "ids": [0]}
                     for t in composites + nested + [text()] if rng.random() < 0.6}
        host = {"chat": {"keys": host_keys, "tags": {}}}
        multiset.merge_witnesses(host, events, lambda _s: "chat")
        vocab = composites + nested + atoms + [p for c in composites for p in c.split("\n\n")]
        expected = [("user", rng.choice(vocab)) for _ in range(rng.randint(1, 4))]
        pool = composites + nested + atoms
        rows = [(sid, "S0", "user", rng.choice(pool)) for sid in rng.sample(range(1, 50), rng.randint(1, 5))]
        if case % 4 == 0:
            # Regularly exercise the nested exact/span/parts collision, varying ids and input order.
            k = "qq\n\npp"
            ids = sorted(rng.sample(range(1, 50), 3))
            rows = [(ids[0], "S0", "user", k), (ids[1], "S0", "user", k),
                    (ids[2], "S0", "user", "xx")]
            expected = [("user", k), ("user", "pp"), ("user", k + "\n\nxx")]
            rng.shuffle(expected)
            host = {"chat": {"keys": {("user", multiset.h(k)): {"n": 2, "ids": ids[:2]}}, "tags": {}}}
            multiset.merge_witnesses(host, merge_events(("qq", "pp")) + merge_events((k, "xx")), lambda _s: "chat")
            host_keys = host["chat"]["keys"]
        canonical = multiset.score(expected, sorted(rows), host_keys)
        rng.shuffle(rows)
        shuffled = multiset.score(expected, rows, host_keys)
        assert (shuffled["verdict"], shuffled["deficit_rows"]) == (canonical["verdict"], canonical["deficit_rows"]), case
        assert_row_capacity(expected, rows, canonical)
        assert_row_capacity(expected, rows, shuffled)


@pytest.mark.parametrize("field", ["store_id", "tokens", "head", "tail"])
def test_oversized_projection_integer_is_an_ordinary_expectation(tmp_path, field):
    original = "original " + "alpha beta gamma delta " * 1600
    renderer = projections.codec(TREE)
    view = renderer.projected_fields({"store_id": 1, "role": "user", "content": original},
                                     9210, 1200, 600)["content"]
    match = renderer.projection_mark_candidates(view, None)[0]
    group = {"tokens": 2, "store_id": 3, "head": 4, "tail": 5}[field]
    start, end = match.span(group)
    view = view[:start] + "9" * 5000 + view[end:]
    events = fixture.turn_events(1, held=view)
    events[0].update(content=original, persist=original)
    out = fixture.make(tmp_path, rows=[("user", original), ("assistant", fixture.R.format(1, 1))],
                       events=events, plugin={"tree": TREE}, bars=["B2"])
    assert out["numbers"]["B2"]["deficit_rows"] == 1
    assert out["numbers"]["B2"]["verified_survival_projections"] == []
    assert "B2" in out["failed_bars"]


def test_projection_without_original_witness_still_fails(tmp_path):
    original = "original " + "alpha beta gamma delta " * 1600
    view = projections.codec(TREE).projected_fields(
        {"store_id": 1, "role": "user", "content": original}, 9210, 1200, 600)["content"]
    events = fixture.turn_events(1, held=view)
    events[0].update(content=view, persist=view)
    out = fixture.make(tmp_path, rows=[("user", original), ("assistant", fixture.R.format(1, 1))],
                       events=events, plugin={"tree": TREE}, bars=["B2"])
    assert out["numbers"]["B2"]["deficit_rows"] == 1
    assert out["numbers"]["B2"]["verified_survival_projections"] == []


@pytest.mark.parametrize("seam", ["ingest", "on_turn_complete", "compress"])
@pytest.mark.parametrize("fault", ["emit", "carrier", "post_carrier", "persistent_emit", "context"])
@pytest.mark.parametrize("raises", [False, True])
def test_observer_fault_preserves_original_call(seam, fault, raises, caplog):
    events, calls = [], []
    messages = [{"role": "user", "content": "input"}]
    result = [{"role": "user", "content": "output"}]
    failure = RuntimeError("original failure")

    def original(self, rows, *args, **kwargs):
        calls.append((rows, args, kwargs))
        if raises:
            raise failure
        return result

    def carrier(self, message):
        if fault == "carrier" or (fault == "post_carrier" and message is result[0]):
            raise LookupError("observer only")
        return None

    def emit(**event):
        if fault == "persistent_emit" or (fault == "emit" and event["event"] == "engine_ingress"):
            raise OSError("observer only")
        events.append(event)

    def context():
        if fault == "context":
            raise ValueError("observer only")
        return {"turn": 1}

    engine = type("Engine", (), {seam: original, "_generated_context_carrier_remainder": carrier})
    probe.trace_occurrences(engine, emit, context)
    call = getattr(engine(), seam)
    if raises:
        with pytest.raises(RuntimeError) as caught:
            call(messages, "arg", flag=True)
        assert caught.value is failure
    else:
        assert call(messages, "arg", flag=True) is result
    assert calls == [(messages, ("arg",), {"flag": True})]
    if fault != "post_carrier" or (seam == "compress" and not raises):
        assert any(e["event"] == "observer_error" for e in events) or "observer_error" in caplog.text
    assert not any(e["event"] == "engine_carrier" for e in events)
    # A failed observer cannot synthesize a witness for the unobserved input.
    host = {"chat": {"keys": {}, "tags": {}}}
    multiset.merge_witnesses(host, events, lambda _s: "chat")
    assert multiset.score([("user", "input")], [], host["chat"]["keys"])["deficit_rows"] == 1


@pytest.mark.parametrize("raises", [False, True])
def test_post_merge_observer_fault_preserves_original_call(raises):
    messages = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    result, events, calls = ([{"role": "user", "content": "a\n\nb"}], 1), [], []
    failure = RuntimeError("original failure")

    def original(rows):
        calls.append(rows)
        if raises:
            raise failure
        return result

    def emit(**event):
        if event["event"] == "host_merge":
            raise OSError("observer only")
        events.append(event)

    helpers = SimpleNamespace(_merge_consecutive_users=original, _SEQUENCE_REPAIR_PASSES=(original,))
    assert probe.trace_user_merges(helpers, emit, lambda: {"session": "S0"})
    if raises:
        with pytest.raises(RuntimeError) as caught:
            helpers._SEQUENCE_REPAIR_PASSES[0](messages)
        assert caught.value is failure
    else:
        assert helpers._SEQUENCE_REPAIR_PASSES[0](messages) is result
        assert events[0]["event"] == "observer_error"
    assert calls == [messages]
