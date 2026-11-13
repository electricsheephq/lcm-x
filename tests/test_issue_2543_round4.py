"""#2543 final-round witnesses: authored rows, eligibility, bounded scans (I1/I2)."""

import copy
import sqlite3
import sys
import time

import pytest

from hermes_lcm import engine as engine_module
from hermes_lcm.dag import SummaryNode
from tests.invariants.test_p1e_cold_tails import fail_end_read, queue_cold
from tests.test_issue_2543_occurrence_resume import FINAL, history, lanes as lanes


def busy_end(engine, monkeypatch, view):
    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", fail_end_read(engine))
        engine.on_session_end("S1", copy.deepcopy(view))


@pytest.mark.parametrize("busy", [False, True], ids=["direct", "cold"])
def test_A1_foreign_summary_is_authored_content(lanes, monkeypatch, busy):
    lane = lanes()
    first = history()
    lane.seed(first)
    engine = lane.engine
    now = time.time()
    node = engine._dag.add_node(SummaryNode(session_id="foreground", depth=0,
        summary="Quoted context that belongs to foreground", token_count=10, source_token_count=10,
        source_ids=[], source_type="messages", created_at=now, earliest_at=now, latest_at=now, expand_hint="quote"))
    node_id = getattr(node, "node_id", node)
    target = (f"[Recent Summary (d0, node {node_id})]\nQuoted context that belongs to foreground\n"
              "[Expand for details: quote]\n\npad 2")
    view = copy.deepcopy(first)
    view[2]["content"] = target
    view.append(FINAL)
    lane.end(view, monkeypatch, busy)
    contents = [row["content"] for row in lane.rows()]
    assert (contents.count(target), contents.count("FINAL"), len(contents)) == (1, 1, 12)
    lane.end(view, monkeypatch, busy)
    assert [row["content"] for row in lane.rows()] == contents


def test_A2_dedupe_promotes_later_verified_eligibility(lanes, monkeypatch):
    lane = lanes()
    engine = lane.engine
    engine.on_session_start("S", platform="cli", context_length=200_000, conversation_id="conv")
    first = history()
    view = copy.deepcopy(first)
    view[2]["content"] = "edited"
    view.append(FINAL)
    tail = queue_cold(engine, monkeypatch, view=view, stored=first)
    assert tail["occurrence_resume_through"] is None
    read = engine._store.get_range

    def blocked(session, *args, **kwargs):
        if session == "S":
            raise sqlite3.OperationalError("database is locked")
        return read(session, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "get_range", blocked)
        lane.foreground()
        engine.on_session_end("S", copy.deepcopy(view))
    assert len(engine._pending_session_tails) == 1
    engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
    contents = [row["content"] for row in engine._store.get_session_messages("S")]
    assert (contents.count("edited"), contents.count("FINAL"), len(contents)) == (1, 1, 12)
    assert not engine._pending_session_tails
    engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
    assert [row["content"] for row in engine._store.get_session_messages("S")] == contents


def repeated_witness(lane, n):
    first = [{"role": "user", "content": "ok"} for _ in range(n)]
    engine = lane.engine
    engine._append_off_current_session_end_suffix("S1", first, source="cli", conversation_id="conv1")
    lane.foreground()
    return [{"role": "user", "content": "changed"}] + first[1:] + [FINAL]


def test_A2_dedupe_does_not_refresh_existing_high_water(lanes, monkeypatch):
    """Later eligibility must compose with CR-7, never reauthorizing an already eligible old tail."""
    lane = lanes()
    first = history()
    lane.seed(first)
    engine = lane.engine
    view = copy.deepcopy(first)
    view[2]["content"] = "edited"
    view.append(FINAL)
    busy_end(engine, monkeypatch, view)
    engine._append_off_current_session_end_suffix("S1", [{"role": "user", "content": "went on"}],
        source="cli", conversation_id="conv1")
    busy_end(engine, monkeypatch, view)
    assert len(engine._pending_session_tails) == 1
    assert engine._pending_session_tails[0]["occurrence_resume_through"] == 10
    engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
    assert len(engine._pending_session_tails) == 1
    assert len(lane.rows()) == 11
    assert lane.rows()[-1]["content"] == "went on"


def test_A3_consumed_candidates_are_not_rescanned(lanes):
    lane = lanes()
    n = 8000
    view = repeated_witness(lane, n)
    scans = 0

    def profile(frame, event, arg):
        nonlocal scans
        if event == "call" and frame.f_code.co_name == "available":
            scans += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        suffix = lane.engine._session_end_occurrence_resume(view, session_id="S1", conversation_id="conv1",
            store=lane.engine._store, hermes_home=lane.engine._hermes_home, stamps=False)
    finally:
        sys.setprofile(previous)
    assert suffix == [view[0], FINAL]
    assert scans <= 4 * n, f"consumed candidates rescanned: {scans} for {n} rows"


def test_A3_expired_classification_keeps_main_answer(lanes, monkeypatch):
    lane = lanes()
    view = repeated_witness(lane, 2000)
    engine = lane.engine
    forms = engine._session_end_host_forms
    expired = False
    replay_calls = 0
    replay = engine._message_replay_identity

    def expire_after_read(*args, **kwargs):
        nonlocal expired
        result = forms(*args, **kwargs)
        expired = True
        return result

    def count_replay(*args, **kwargs):
        nonlocal replay_calls
        replay_calls += 1
        return replay(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_session_end_host_forms", expire_after_read)
        patch.setattr(engine, "_message_replay_identity", count_replay)
        patch.setattr(engine_module.time, "monotonic", lambda: 2.0 if expired else 0.0)
        suffix = engine._session_end_occurrence_resume(view, session_id="S1", conversation_id="conv1",
            store=engine._store, hermes_home=engine._hermes_home, stamps=False, deadline=1.0)
    assert suffix is None
    assert replay_calls == 0


def test_A3_deadline_during_candidate_matching(lanes, monkeypatch):
    lane = lanes()
    engine = lane.engine
    first = [{"role": "tool", "tool_call_id": "same", "content": "ok"} for _ in range(1500)]
    engine._append_off_current_session_end_suffix("S1", first, source="cli", conversation_id="conv1")
    view = [{"role": "tool", "tool_call_id": "same", "content": "changed"}, FINAL]
    equal = engine._session_end_rows_equal
    comparisons = 0

    def compare(*args, **kwargs):
        nonlocal comparisons
        comparisons += 1
        return equal(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_session_end_rows_equal", compare)
        patch.setattr(engine_module.time, "monotonic", lambda: 2.0 if comparisons >= 10 else 0.0)
        suffix = engine._session_end_occurrence_resume(view, session_id="S1", conversation_id="conv1",
            store=engine._store, hermes_home=engine._hermes_home, stamps=False, deadline=1.0)
    assert suffix is None
    assert 10 <= comparisons <= 1001


def test_L7_two_tails_same_session(lanes, monkeypatch):
    """L-7: tail 1 stores its copy and FINAL; tail 2 stays pending. Main is no better (both tails pending)."""
    lane = lanes()
    first = history("T7-wrapped")
    lane.seed(first)
    engine = lane.engine
    v1 = first + [FINAL]
    v2 = v1 + [{"role": "user", "content": "more"}, {"role": "assistant", "content": "FINAL2"}]
    busy_end(engine, monkeypatch, v1)
    busy_end(engine, monkeypatch, v2)
    assert [tail["occurrence_resume_through"] for tail in engine._pending_session_tails] == [10, 10]
    for _ in range(2):
        engine._retry_pending_session_tails(deadline=time.monotonic() + 60)
    rows = lane.rows()
    assert len(rows) == 12 and rows[-2]["tool_calls"] and rows[-1]["content"] == "FINAL"
    assert not any(row["content"] in ("more", "FINAL2") for row in rows)
    assert len(engine._pending_session_tails) == 1
    assert engine._pending_session_tails[0]["view"] == v2
