"""#1879: conservative errors become credit only with explicit occurrence witnesses."""
from itertools import permutations
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
