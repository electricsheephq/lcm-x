"""RCA #1797/#1799 regressions: exact witnesses, bounded multiplicity, and complete receipts."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.instruments.reliability import probe
from bench.instruments.reliability.scorers import externalized, multiset, projections
from tests import test_reliability_scorers as fixture

TREE = str(Path(__file__).resolve().parents[3])
ORIGINAL = "[T37] user turn 37: synthetic exact bytes end."
HELD = ORIGINAL + "\n\n" + ORIGINAL
EARLIER = "[T36] user turn 36: earlier end."
COMPOSITE = EARLIER + "\n\n" + HELD


def merge_events(parts, session="S0"):
    def merge(messages):
        messages[0]["content"] = "\n\n".join(m["content"] for m in messages)
        return messages[:1], len(messages) - 1
    helpers = SimpleNamespace(_merge_consecutive_users=merge, _SEQUENCE_REPAIR_PASSES=(merge,))
    events = []
    assert probe.trace_user_merges(helpers, lambda **e: events.append(e),
                                   lambda: {"session": session, "phase": "A", "turn": 37})
    helpers._SEQUENCE_REPAIR_PASSES[0]([{"role": "user", "content": p} for p in parts])
    return events


def witnessed_host(parts=(EARLIER, HELD), n=1):
    [event] = merge_events(parts)
    text = "\n\n".join(parts)
    return {("user", multiset.h(text)): {"n": n, "ids": [2], "span_witnesses": [
        {"witness": {"event_index": 1, "phase": "A", "turn": 37}, "spans": event["spans"]}]}}


def test_1797_host_witnessed_compound_is_not_a_byte_deficit():
    # Exact RCA payloads, with the explicit merge witness required by the SPEC.
    host = witnessed_host()
    host[("user", multiset.h(HELD))] = {"n": 1, "ids": [1]}
    result = multiset.score([("user", HELD)], [(93, "S0", "user", COMPOSITE)], host)
    assert result["deficit_rows"] == 0
    [credit] = result["embedded_span_credits"]
    assert credit["store_id"] == 93 and credit["byte_offset"] == len((EARLIER + "\n\n").encode())
    assert credit["bytes"] == len(HELD.encode()) and credit["provenance"] == "host_merge_span"


@pytest.mark.parametrize("case", ["bare_hashes", "deleted_byte", "wrong_offset", "no_host"])
def test_1797_negative_controls_stay_b2_failures(case):
    host, text = witnessed_host(), COMPOSITE
    if case == "bare_hashes":
        host[("user", multiset.h(COMPOSITE))].pop("span_witnesses")
        host[("user", multiset.h(HELD))] = {"n": 1, "ids": [1]}
    elif case == "deleted_byte":
        text = COMPOSITE[:-1]
    elif case == "wrong_offset":
        host[("user", multiset.h(COMPOSITE))]["span_witnesses"][0]["spans"][1]["byte_offset"] += 1
    else:
        host = None
    result = multiset.score([("user", HELD)], [(93, "S0", "user", text)], host)
    assert result["deficit_rows"] == 1 and result["verdict"] == "FAIL"


def test_span_is_single_use_and_host_copies_do_not_multiply_witnesses():
    rows = [(93, "S0", "user", COMPOSITE), (94, "S0", "user", COMPOSITE)]
    result = multiset.score([("user", HELD)] * 2, rows, witnessed_host())
    assert result["deficit_rows"] == 1 and len(result["embedded_span_credits"]) == 1
    # Reserving the whole composite for its own expected row also reserves its embedded span.
    result = multiset.score([("user", COMPOSITE), ("user", HELD)], rows[:1], witnessed_host())
    assert result["deficit_rows"] == 1 and result["embedded_span_credits"] == []


def test_disjoint_repeated_spans_and_utf8_offsets():
    parts = ("éarlier", HELD, HELD)
    result = multiset.score([("user", HELD)] * 2,
                            [(93, "S0", "user", "\n\n".join(parts))], witnessed_host(parts))
    assert result["deficit_rows"] == 0
    assert [c["byte_offset"] for c in result["embedded_span_credits"]] == [
        len((parts[0] + "\n\n").encode()), len((parts[0] + "\n\n" + HELD + "\n\n").encode())]


@pytest.mark.parametrize("foreign", [False, True])
def test_observed_merge_is_grouped_by_lineage(tmp_path, foreign):
    events = fixture.turn_events(37, held=HELD)
    events[0].update(content=ORIGINAL, persist=ORIGINAL)
    events += merge_events((EARLIER, HELD), "cron" if foreign else "S0")
    out = fixture.make(tmp_path, rows=[("user", COMPOSITE), ("assistant", fixture.R.format(37, 37))],
                       events=events, host=[("S0", "user", COMPOSITE, 1)], plugin={"tree": TREE}, bars=["B2"])
    assert out["numbers"]["B2"]["deficit_rows"] == int(foreign)


@pytest.mark.parametrize("case", ["valid", "altered_middle", "altered_head", "foreign", "forged_view"])
def test_1799_verified_view_projection_requires_its_stored_original(tmp_path, case):
    original = "[T01] user turn 1: " + "alpha beta gamma delta " * 1600 + "end."
    projection = projections.codec(TREE).projected_fields(
        {"store_id": 1, "role": "user", "content": original}, 9210, 1200, 600)["content"]
    events = fixture.turn_events(1, held=projection + ("!" if case == "forged_view" else ""))
    events[0].update(content=original, persist=original)
    stored = original
    if case == "altered_middle":
        stored = original[:4000] + "!" + original[4001:]
    if case == "altered_head":
        stored = "!" + original[1:]
    out = fixture.make(tmp_path, rows=[("user", stored), ("assistant", fixture.R.format(1, 1))],
                       events=events, host=[("S0", "user", original, 1)],
                       sids=["cron", "S0"] if case == "foreign" else None,
                       plugin={"tree": TREE}, bars=["B2"])
    assert out["numbers"]["B2"]["deficit_rows"] == int(case != "valid")
    assert bool(out["numbers"]["B2"]["verified_survival_projections"]) == (case == "valid")


def test_complete_missing_and_resolved_mismatch_lists(tmp_path):
    events = fixture.clean_events(55)
    rows = [("user", "wrong " + str(i)) for i in range(55)]
    out = fixture.make(tmp_path, rows=rows, events=events, bars=["B2"])
    assert len(out["failed_bars"]["B2"]["missing"]) == 110
    assert len(out["failed_bars"]["B2"]["extra"]) == 55
    numbers = {"content_mismatch_rows": 0, "content_mismatch": []}
    full = [(i, "chat", "user", "wrong " + str(i)) for i in range(55)]
    externalized.mismatches(numbers, full, set(range(55)), {"chat": ([("user", "right")], [])},
                           lambda s: s, {}, {}, None)
    assert numbers["content_mismatch_rows"] == len(numbers["content_mismatch"]) == 55


def test_resolved_surplus_is_separate_from_missing_bytes():
    numbers = {"content_mismatch_rows": 0, "content_mismatch": []}
    externalized.mismatches(numbers, [(1, "chat", "user", "unexpected")], {1},
                           {"chat": ([("user", "right")], [])}, lambda s: s, {}, {}, None,
                           {"chat": {"deficit_rows": 0}})
    assert numbers["content_mismatch_rows"] == 0 and len(numbers["resolved_surplus"]) == 1


def test_ingress_hashes_preserve_every_occurrence_and_arguments():
    calls, events = [], []

    class Engine:
        _session_id = "S0"

        def ingest(self, messages, **kw):
            calls.append((messages, kw))
            return messages
    probe.trace_occurrences(Engine, lambda **e: events.append(e), lambda: {"turn": 1})
    rows = [{"role": "user", "content": "same"}] * 3
    assert Engine().ingest(rows, evidence=True) is rows
    assert calls == [(rows, {"evidence": True})]
    assert len(events[0]["occurrences"]) == 3
    assert {r["sha256"] for r in events[0]["occurrences"]} == {hashlib.sha256(b"same").hexdigest()}
