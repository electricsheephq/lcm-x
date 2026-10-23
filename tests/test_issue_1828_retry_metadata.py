"""Retry metadata preserves bounded evidence and the pending group's chronology."""
import json
import random
import sys
from types import SimpleNamespace

from hermes_lcm import recall_session, tokens
from tests.test_issue_1828_refit import RefitFixture


def retry_deadline(monkeypatch, fixture, expire_at):
    state = dict(read_started=False, checks=0)

    def monotonic():
        if fixture.store._write_lock._is_owned():
            state["read_started"] = True
        elif state["read_started"]:
            index = state["checks"]
            state["checks"] += 1
            if index >= expire_at:
                caller = sys._getframe(2).f_locals
                state["admitted"] = [h["store_id"] for g in caller["admitted"] for h in g]
                state["skipped"] = [g[1][0]["store_id"] for g in caller["skipped"]]
                return 2
        return 0

    monkeypatch.setattr(recall_session, "time", SimpleNamespace(monotonic=monotonic))
    return state


def assert_bounded(result, cap, token_cap):
    raw = json.dumps(result, ensure_ascii=False)
    assert len(raw) <= cap
    assert tokens.count_tokens_with_encoder(raw, encoder=None) <= token_cap
    delivery = result["provenance"]["delivery"]
    assert delivery["response_chars"] == len(raw)
    assert delivery["response_tokens"] == tokens.count_tokens_with_encoder(raw, encoder=None)
    sessions = {s["session_id"] for s in delivery["sessions"]}
    assert all(h["session_id"] in sessions for h in result["hits"])


def test_retry_deadline_refits_saturated_evidence(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", 1648)
    fixture = RefitFixture((1600, 200, 100), tuple("ABC"))
    state = retry_deadline(monkeypatch, fixture, 0)
    try:
        result = fixture.run(deadline=1, query="")
    finally:
        fixture.close()
    assert state["skipped"] and state["admitted"] == [3]
    assert state["checks"] == 1
    assert_bounded(result, 1648, recall_session.TOKEN_BUDGET)
    assert result["timeout"] is True
    assert result["provenance"]["delivery"]["stop_reason"] == "deadline"


def test_seeded_retry_deadline_shapes_are_bounded_without_orphans(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    rng = random.Random(1828)
    failures, expired = [], 0
    for case in range(300):
        sizes = tuple(rng.randint(1, 2400) for _ in range(rng.randint(2, 6)))
        cap, token_cap = rng.randint(1600, 7000), rng.randint(1000, 4000)
        expire_at = rng.randrange(max(1, len(sizes) // 2))
        monkeypatch.setattr(recall_session, "CHAR_BUDGET", cap)
        monkeypatch.setattr(recall_session, "TOKEN_BUDGET", token_cap)
        fixture = RefitFixture(sizes, tuple("ABCDEF"[:len(sizes)]))
        state = retry_deadline(monkeypatch, fixture, expire_at)
        try:
            result = fixture.run(deadline=1, query="")
            assert_bounded(result, cap, token_cap)
            if state["checks"] > expire_at:
                assert result["timeout"] is True
        except Exception as exc:
            failures.append((case, sizes, cap, token_cap, expire_at, repr(exc)))
        finally:
            expired += state["checks"] > expire_at
            fixture.close()
    assert expired >= 100
    assert not failures, failures


def test_successful_retry_restores_pending_event_time(monkeypatch):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(recall_session, "time", SimpleNamespace(monotonic=lambda: 0))
    fixture = RefitFixture((1600, 100))
    fields = dict(time_uncertain="carried", event_time="2020-01-01", event_time_source="synthetic")

    def run(cap):
        monkeypatch.setattr(recall_session, "CHAR_BUDGET", cap)
        return json.loads(recall_session.deliver_session(
            fixture.ordered, dict(query="needle", limit=2, provenance=dict(arms_run=[])),
            engine=fixture.engine, deadline=1, excluded=set(), allowed=None, seen_refs=set(),
            delta_requested=True, event_time_fields=lambda row: dict(fields) if row["store_id"] == 2 else {},
            occurrence=False, summary_leads=[]))

    try:
        unconstrained, retried = run(9000), run(3240)
    finally:
        fixture.close()
    assert [h["store_id"] for h in retried["hits"]] == [2]
    expected = next(h for h in unconstrained["hits"] if h["store_id"] == 2)
    assert {k: retried["hits"][0].get(k) for k in fields} == {k: expected[k] for k in fields}
