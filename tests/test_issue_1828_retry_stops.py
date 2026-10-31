"""Refit retries preserve read failures and obey the live absolute deadline."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from hermes_lcm import recall_session, tokens
from tests.test_issue_1828_refit import RefitFixture


def run_with_state(fixture, **kwargs):
    state = {}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "return" and frame.f_code is recall_session.deliver_session.__code__:
            state.update(copy.deepcopy({k: frame.f_locals[k] for k in ("omissions", "skipped")}))

    sys.setprofile(profile)
    try:
        result = fixture.run(deadline=1, **kwargs)
    finally:
        sys.setprofile(previous)
    return result, state


@pytest.mark.parametrize("session_ids,cap", [(tuple("ABC"), 3330),
                                          (("session-0", "session-1", "session-2", "session-3"), 3500)])
def test_read_error_only_marks_interrupted_and_unprocessed_sessions(monkeypatch, session_ids, cap):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(recall_session, "time", SimpleNamespace(monotonic=lambda: 0))
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", cap)
    fixture = RefitFixture((1600,) + (100,) * (len(session_ids) - 1), session_ids)
    original = fixture.store.load_session_window

    def read(sid, **kwargs):
        if sid == session_ids[2]:
            raise OSError("synthetic read failure")
        assert sid in session_ids[:2]
        return original(sid, **kwargs)

    monkeypatch.setattr(fixture.store, "load_session_window", read)
    try:
        result, state = run_with_state(fixture, query="")
    finally:
        fixture.close()
    assert state["skipped"][0][0]["session_id"] == session_ids[1]
    assert [h["store_id"] for h in result["hits"]] == [2]
    sessions = {s["session_id"]: s for s in result["provenance"]["delivery"]["sessions"]}
    for sid in session_ids[:2]:
        assert "read_error" not in {r for r, _ in state["omissions"][sid]}
    assert sessions[session_ids[1]]["complete"]
    assert sessions[session_ids[1]]["omission_reason"] is None
    for sid in session_ids[2:]:
        assert not sessions[sid]["complete"]
        assert sessions[sid]["omission_reason"] == "read_error"
        assert "read_error" in {r for r, _ in state["omissions"][sid]}
    assert result["provenance"]["delivery"]["stop_reason"] == "read_error"


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_read_error_survives_successful_budget_retry(monkeypatch, error):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(recall_session, "time", SimpleNamespace(monotonic=lambda: 0))
    fixture = RefitFixture((1600, 100))
    failed = fixture.store.append("session-1", dict(role="user", content="unread seed", timestamp=1))
    fixture.ordered.append(dict(hit=dict(session_id="session-1", store_id=failed), ranks={}, _final_score=1))
    original = fixture.store.load_session_window

    def read(sid, **kwargs):
        if kwargs["anchor_store_id"] == failed:
            raise error("synthetic read failure")
        return original(sid, **kwargs)

    monkeypatch.setattr(fixture.store, "load_session_window", read)
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", 3250)
    try:
        result, state = run_with_state(fixture)
    finally:
        fixture.close()
    assert state["skipped"][0][0]["session_id"] == "session-1"
    assert state["omissions"]["session-0"] == [("budget", None)]
    assert [h["store_id"] for h in result["hits"]] == [2]
    assert failed not in {h["store_id"] for h in result["hits"]}
    meta = next(m for m in result["provenance"]["delivery"]["sessions"] if m["session_id"] == "session-1")
    assert not meta["complete"], json.dumps(result)
    assert "read_error" in {reason for reason, _ in state["omissions"]["session-1"]}
    assert meta["omission_reason"] == "read_error"
    assert result["provenance"]["delivery"]["stop_reason"] == "read_error"


@pytest.mark.parametrize("successful_retries", [0, 1])
def test_deadline_rechecked_before_each_retry(monkeypatch, successful_retries):
    monkeypatch.setattr(tokens, "_get_encoder", lambda: None)
    monkeypatch.setattr(recall_session, "CHAR_BUDGET", 3370)
    fixture = RefitFixture((1600, 200, 100))
    read_started = False
    retry_checks = 0

    def monotonic():
        nonlocal read_started, retry_checks
        if fixture.store._write_lock._is_owned():
            read_started = True
        elif read_started:
            retry_checks += 1
            return 2 if retry_checks > successful_retries else 0
        return 0

    monkeypatch.setattr(recall_session, "time", SimpleNamespace(monotonic=monotonic))
    try:
        result = fixture.run(deadline=1)
    finally:
        fixture.close()
    assert [h["store_id"] for h in result["hits"]] == ([2] if successful_retries else [])
    assert result.get("timeout") is True
    delivery = result["provenance"]["delivery"]
    assert delivery["stop_reason"] == "deadline"
    assert delivery["truncated"] is True
    assert retry_checks == successful_retries + 1
