"""#524: a retry after a cancelled compaction, in a session with a PRIOR adopted compaction.

Attempt 1 commits and the host adopts its output (a summary carrier + rows). Attempt 2
commits too, but the host cancels and retries with its own list: attempt 1's output plus
the rows after it. Attempt 2 replaced the one durable commit proof, so nothing durable
describes that list any more. The retry must bind it as a replay of this session's own
rows below the frontier: nothing re-stored, no provider call, attempt 2's output back.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.ingest_protection import protect_messages_for_ingest
from hermes_lcm.reconcile import _COMPACTION_COMMIT_PROOF_METADATA_PREFIX
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "issue-524", "issue-524-conv"
REASON = "replayed superseded own-session emission below frontier"


class _Provider:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return f"Stub summary {self.calls}.\nExpand for details about: stub", 1


def _engine(tmp_path, session_id=SID, **overrides):
    config = LCMConfig(database_path=str(tmp_path / "l.db"))
    for key, value in overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "h"))
    engine.on_session_start(session_id, platform="cli", conversation_id=CID, context_length=200_000)
    return engine


def _transcript():
    plain, i = [], 0
    while len(plain) < 40 or count_messages_tokens(plain[:-30]) < 24_000:
        plain.append({"role": "user" if i % 2 == 0 else "assistant",
                      "content": f"owned turn {i} " + " ".join(f"w{i}x{j}" for j in range(120))})
        i += 1
    return plain


def _more(start, n, tag="second gen turn"):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"{tag} {i} " + " ".join(f"g{i}x{j}" for j in range(120))}
            for i in range(start, start + n)]


def _compress(engine, messages):
    return engine.compress(messages, current_tokens=count_messages_tokens(messages), force=True)


def _adopt(engine, host, session_id=SID):
    """The host commits a compaction: session end on its input, then the boundary start."""
    engine.on_session_end(session_id, host)
    engine.on_session_start(session_id, boundary_reason="compression", old_session_id=session_id,
                            platform="cli", conversation_id=CID, context_length=200_000)


def _proof_key(engine):
    return engine._replay_snapshot_metadata_key(_COMPACTION_COMMIT_PROOF_METADATA_PREFIX, SID)


def _rows(outputs):
    return [LCMEngine._public_compression_row(message) for message in outputs]


@pytest.fixture
def provider(monkeypatch):
    stub = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", stub)
    return stub


def _gen2(tmp_path):
    """Attempt 1 adopted; attempt 2 committed and cancelled. Returns (engine, host, out2, proof1)."""
    engine = _engine(tmp_path)
    first = _transcript()
    engine.ingest(first)
    out1 = deepcopy(_compress(engine, first))
    proof1 = engine._store.read_metadata_json(_proof_key(engine))
    _adopt(engine, first)
    host = deepcopy(out1) + _more(0, 120)
    engine.ingest(host)
    out2 = deepcopy(_compress(engine, host))
    assert engine._last_compression_status == "compacted"
    return engine, host, out2, proof1


@pytest.mark.parametrize("delta", [0, 2], ids=["exact-retry", "retry-plus-turn"])
@pytest.mark.parametrize("cold", [False, True], ids=["same-engine", "cold-engine"])
def test_gen2_cancel_after_commit_retry_binds_replay(tmp_path, provider, cold, delta):
    engine, host, out2, _proof1 = _gen2(tmp_path)
    try:
        rows = engine._store.get_session_count(SID)
        assert rows == 260
        if cold:
            engine.shutdown()
            engine = _engine(tmp_path)
        calls = provider.calls
        new = _more(120, delta)
        retry = _compress(engine, deepcopy(host) + deepcopy(new))
        assert engine._store.get_session_count(SID) == rows + delta
        assert engine._last_ingest_reconciliation["reason"] == REASON
        assert engine._last_compression_status == "compacted"
        assert provider.calls == calls
        assert _rows(retry) == _rows(out2) + _rows(new)
    finally:
        engine.shutdown()


def test_attempt_one_proof_would_have_bound_the_list(tmp_path, provider):
    """H1: the list IS attempt 1's emission; only the replaced proof stops binding it."""
    engine, host, _out2, proof1 = _gen2(tmp_path)
    try:
        assert engine._cursor_from_durable_commit_proof(host) is None
        engine._store.write_metadata_json([_proof_key(engine)], json.dumps(proof1, sort_keys=True))
        assert engine._cursor_from_durable_commit_proof(host) == len(host) == 152
    finally:
        engine.shutdown()


def _retry_and_measure(engine, provider, messages):
    calls = provider.calls
    out = _compress(engine, messages)
    return {
        "rows": engine._store.get_session_count(engine._session_id),
        "status": engine._last_compression_status,
        "calls": provider.calls - calls,
        "reason": (engine._last_ingest_reconciliation or {}).get("reason"),
        "out": _rows(out),
    }


def _lost(engine, messages):
    """Retry rows no stored row of the session holds (a carrier by its glued row)."""
    stored = [str(row.get("content") or "") for row in engine._store.get_session_messages(engine._session_id, limit=100_000)]
    missing = []
    for message in messages:
        text = engine._generated_context_carrier_remainder(message) or str(message.get("content") or "")
        if not engine._is_verified_replay_scaffold_message(message) and text not in stored:
            missing.append(text[:60])
    return missing


# -- sequences ------------------------------------------------------------------------------------

def test_gen3_retry_binds_replay(tmp_path, provider):
    """Attempts 1 and 2 adopted, attempt 3 committed and cancelled: the retry replays out2."""
    engine, host, out2, _proof1 = _gen2(tmp_path)
    try:
        _adopt(engine, host)
        host3 = deepcopy(out2) + _more(200, 120, "third gen turn")
        engine.ingest(host3)
        rows = engine._store.get_session_count(SID)
        out3 = _rows(_compress(engine, host3))
        assert engine._last_compression_status == "compacted"
        got = _retry_and_measure(engine, provider, deepcopy(host3))
        assert (got["rows"], got["status"], got["calls"], got["reason"]) == (rows, "compacted", 0, REASON)
        assert got["out"] == out3
    finally:
        engine.shutdown()


def test_double_cancel_then_retry_with_the_original_list(tmp_path, provider):
    """The resume itself is cancelled too (its output discarded); the host retries again."""
    engine, host, out2, _proof1 = _gen2(tmp_path)
    try:
        first = _retry_and_measure(engine, provider, deepcopy(host))
        second = _retry_and_measure(engine, provider, deepcopy(host))
        for got in (first, second):
            assert (got["rows"], got["status"], got["calls"], got["reason"]) == (260, "compacted", 0, REASON)
            assert got["out"] == _rows(out2)
    finally:
        engine.shutdown()


def test_continuation_after_the_resume_compacts_past_the_frontier(tmp_path, provider):
    engine, host, _out2, _proof1 = _gen2(tmp_path)
    try:
        retry = _compress(engine, deepcopy(host))
        frontier = engine._lifecycle.get_by_conversation(CID).current_frontier_store_id
        assert frontier == 228 and engine._last_compression_status == "compacted"
        _adopt(engine, host)
        cont = deepcopy(retry) + _more(300, 120, "after resume turn")
        engine.ingest(cont)
        assert engine._store.get_session_count(SID) == 260 + 120
        calls = provider.calls
        _compress(engine, cont)
        assert engine._last_compression_status == "compacted" and provider.calls > calls
        assert engine._lifecycle.get_by_conversation(CID).current_frontier_store_id > 228
        assert engine._store.get_session_count(SID) == 260 + 120
    finally:
        engine.shutdown()


@pytest.mark.parametrize("head", ["pure-summary", "objective"])
def test_scaffold_heads_bind(tmp_path, provider, head):
    """A pure LCM summary row (the first tail row is an assistant) and an objective head
    (a tool-heavy tail pushed the operative request out) are verified scaffold heads."""
    engine = _engine(tmp_path)
    try:
        first = _transcript()
        if head == "pure-summary":
            first.append({"role": "assistant" if len(first) % 2 else "user", "content": "one more owned turn"})
        else:
            first.append({"role": "user", "content": "OBJECTIVE: reconcile the vendor ledger end to end"})
            for index in range(20):
                call = f"call_{index}"
                first += [{"role": "assistant", "content": f"step {index}",
                           "tool_calls": [{"id": call, "type": "function", "function": {"name": "ledger", "arguments": "{}"}}]},
                          {"role": "tool", "tool_call_id": call, "content": f"ledger page {index} " + "x " * 80}]
        engine.ingest(first)
        out1 = deepcopy(_compress(engine, first))
        heads = [m for m in out1 if engine._is_verified_replay_scaffold_message(m)]
        assert heads and not engine._generated_context_carrier_remainder(out1[len(heads)])
        if head == "objective":
            assert any("[Current user objective preserved" in str(m.get("content")) for m in heads)
        _adopt(engine, first)
        tail_role = out1[-1]["role"]
        host = deepcopy(out1) + _more(1 if tail_role == "user" else 0, 120)
        engine.ingest(host)
        rows = engine._store.get_session_count(SID)
        out2 = _rows(_compress(engine, host))
        assert engine._last_compression_status == "compacted"
        got = _retry_and_measure(engine, provider, deepcopy(host))
        assert (got["rows"], got["status"], got["calls"], got["reason"]) == (rows, "compacted", 0, REASON)
        assert got["out"] == out2
    finally:
        engine.shutdown()


# -- controls: every failed check stores exactly what today stores (duplicates, never loss) --------

def _no_term(monkeypatch):
    monkeypatch.setattr(LCMEngine, "_cursor_from_frontier_bound_replay", lambda self, m, rows: None, raising=False)


def _identities_lost(engine, messages, lineage=()):
    """Retry occurrences the stored rows of the bound session (or of the rotation parents in
    ``lineage``, whose rows a child's DAG covers) do not account for. A multiset: an identity the
    list holds k times needs k stored rows, so a repeated turn cannot hide behind an older copy."""
    rows = [row for sid in (engine._session_id, *lineage) for row in engine._store.get_session_messages(sid, limit=100_000)]
    stored = Counter(form for row in rows for form in engine._stored_row_forms(row))
    protected = protect_messages_for_ingest(messages, config=engine._config, hermes_home=engine._hermes_home,
                                            session_id=engine._session_id)
    need, lost = Counter(), []
    for m, p in zip(messages, protected):
        if not engine._is_verified_replay_scaffold_message(m):
            need[identity := engine._message_replay_identity(p)] += 1
            lost += [str(m.get("content"))[:60]] if need[identity] > stored[identity] else []
    return lost


def _edit_row(host, out1):
    host[50] = {**host[50], "content": host[50]["content"] + " (edited by host)"}
    return host


def _forge_carrier(host, out1):
    host[0] = {**host[0], "content": host[0]["content"].replace("Stub summary 1.", "Stub summary 1!")}
    return host


def _glue_a_later_row(host, out1):
    """The carrier's summary with row 151 glued behind it instead of row 109: the run skips
    the rows between the summary's coverage and 151, so it is not a replay of the emission."""
    head = host[0]["content"]
    summary = head[: head.index("\n\n---\n\n") if "\n\n---\n\n" in head else head.rindex("]") + 1]
    return [{"role": "user", "content": f"{summary}\n\n---\n\n{host[42]['content']}"}] + host[43:]


CONTROLS = {
    "carrier_glued_to_a_later_row": _glue_a_later_row,
    "edited_replay_row": _edit_row,
    "headless_replay": lambda host, out1: host[1:],
    "forged_carrier": _forge_carrier,
    "missing_last_row": lambda host, out1: host[:-1],
    "stale_partial": lambda host, out1: host[:100],
    "old_emission_alone": lambda host, out1: host[:len(out1)],
    "lossy_row": lambda host, out1: host,
}


def _control_run(tmp_path, monkeypatch, case, *, term):
    if not term:
        _no_term(monkeypatch)
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    lossy = {"sensitive_patterns_enabled": True, "sensitive_patterns": ["password_assignment"]} if case == "lossy_row" else {}
    engine = _engine(tmp_path, **lossy)
    try:
        first = _transcript()
        if lossy:  # row 121: in attempt 1's output (redacted, digest-less), then below the new frontier
            first[120] = {**first[120], "content": "password = supersecret123 " + first[120]["content"]}
        engine.ingest(first)
        out1 = deepcopy(_compress(engine, first))
        _adopt(engine, first)
        host = deepcopy(out1) + _more(0, 120)
        engine.ingest(host)
        _compress(engine, host)
        retry = CONTROLS[case](deepcopy(host), out1)
        got = _retry_and_measure(engine, provider, retry)
        got["lost"] = _identities_lost(engine, retry)
        return got
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", sorted(CONTROLS))
def test_failed_checks_store_what_today_stores(tmp_path, monkeypatch, case):
    fixed = _control_run(tmp_path / "fixed", monkeypatch, case, term=True)
    monkeypatch.undo()
    today = _control_run(tmp_path / "today", monkeypatch, case, term=False)
    assert fixed["reason"] != REASON and fixed["lost"] == [] == today["lost"]
    assert (fixed["rows"], fixed["status"], fixed["calls"]) == (today["rows"], today["status"], today["calls"])


def _append_new_turn(host):
    return host + [{"role": "user", "content": "a brand new question after the cancel"}]


BINDS = {  # (retry list, rows the retry stores): rows after the replayed run are a new delta
    "new_turn": (_append_new_turn, 1),
    "new_row_repeats_an_earlier_row": (lambda host: host + [dict(host[-2])], 1),
    "repeat_delta": (lambda host: host + [dict(host[-1]), {"role": "user", "content": "and a new row"}], 2),
}


@pytest.mark.parametrize("case", sorted(BINDS))
def test_rows_after_the_replayed_run_are_stored(tmp_path, provider, case):
    engine, host, out2, _proof1 = _gen2(tmp_path)
    try:
        build, new = BINDS[case]
        retry = build(deepcopy(host))
        got = _retry_and_measure(engine, provider, retry)
        assert (got["rows"], got["reason"]) == (260 + new, REASON)
        assert _identities_lost(engine, retry) == []
        stored = engine._store.get_session_messages(SID, limit=100_000)
        assert [row["content"] for row in stored[-new:]] == [m["content"] for m in retry[-new:]]
    finally:
        engine.shutdown()


def test_interleaved_rows_of_another_conversation(tmp_path, provider):
    """Store ids of this session are not contiguous: identity is per own-session row."""
    engine = _engine(tmp_path)
    other = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "l.db")), hermes_home=str(tmp_path / "h"))
    other.on_session_start("other", platform="cli", conversation_id="other-conv", context_length=200_000)
    try:
        first = _transcript()
        engine.ingest(first)
        out1 = deepcopy(_compress(engine, first))
        _adopt(engine, first)
        host = deepcopy(out1)
        for start in range(0, 120, 20):
            host += _more(start, 20)
            engine.ingest(host)
            other.ingest([{"role": "user", "content": f"other {j}"} for j in range(start + 3)])
        out2 = _rows(_compress(engine, host))
        rows = engine._store.get_session_count(SID)
        got = _retry_and_measure(engine, provider, deepcopy(host))
        assert (got["rows"], got["status"], got["calls"], got["reason"]) == (rows, "compacted", 0, REASON)
        assert got["out"] == out2
    finally:
        other.shutdown()
        engine.shutdown()


def test_steady_state_and_plain_457_retry_leave_the_term_unused(tmp_path, provider):
    engine, host, out2, _proof1 = _gen2(tmp_path)
    try:
        _adopt(engine, host)
        cont = deepcopy(out2) + _more(300, 10)
        got = _retry_and_measure(engine, provider, cont)
        assert got["rows"] == 270 and got["reason"] != REASON
    finally:
        engine.shutdown()
    engine = _engine(tmp_path / "plain")
    try:
        first = _transcript()
        engine.ingest(first)
        out = _rows(_compress(engine, first))
        got = _retry_and_measure(engine, provider, deepcopy(first))
        assert (got["rows"], got["status"], got["calls"], got["out"]) == (140, "compacted", 0, out)
        assert got["reason"] != REASON
    finally:
        engine.shutdown()


def test_rollover_to_a_new_session_is_not_a_replay(tmp_path, monkeypatch):
    """A new session (not a compaction boundary) receiving the old list: the term never runs."""
    counts = []
    for term in (True, False):
        if not term:
            _no_term(monkeypatch)
        provider = _Provider()
        monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
        engine, host, _out2, _proof1 = _gen2(tmp_path / str(term))
        try:
            engine.on_session_end(SID, host)
            engine.on_session_start("issue-524-new", platform="cli", conversation_id=CID, context_length=200_000)
            got = _retry_and_measure(engine, provider, deepcopy(host))
            counts.append((got["rows"], got["status"], got["reason"] == REASON, _identities_lost(engine, host)))
        finally:
            engine.shutdown()
    assert counts[0] == counts[1] and counts[0][2] is False and counts[0][3] == []


# -- #526: a rotation child (compression.in_place=false) replays rows its parent owns --------------

CHILD = "issue-524-child"


def _child_gen2(tmp_path, head="carrier", interleave=False, exclude=None):
    """Attempt 1 in the parent; the host rotates to a child session; attempt 2 commits in the
    child and is cancelled. The child owns only its 120 rows: the carrier's glued row and the
    rows up to 140 stay the parent's. Returns (engine, host, out2, other engine or None)."""
    engine = _engine(tmp_path)
    other = None
    if interleave:
        other = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "l.db")), hermes_home=str(tmp_path / "h"))
        other.on_session_start("other", platform="cli", conversation_id="other-conv", context_length=200_000)
    first = _transcript()
    if head == "pure-summary":
        first.append({"role": "assistant" if len(first) % 2 else "user", "content": "one more owned turn"})
    engine.ingest(first)
    out1 = deepcopy(_compress(engine, first))
    assert (engine._generated_context_carrier_remainder(out1[0]) is not None) == (head == "carrier")
    engine.on_session_end(SID, first)
    engine.on_session_start(CHILD, boundary_reason="compression", old_session_id=SID,
                            platform="cli", conversation_id=CID, context_length=200_000)
    host = deepcopy(out1)
    first_new = 1 if out1[-1]["role"] == "user" else 0
    for chunk in range(first_new, first_new + 120, 20 if interleave else 120):
        host += _more(chunk, 20 if interleave else 120)
        engine.ingest(host)
        if other is not None:
            other.ingest([{"role": "user", "content": f"other {j}"} for j in range(chunk + 3)])
    if exclude is not None:  # a parent row in (C, F] that attempt 2's publication filters out
        marker = host[exclude]["content"]
        engine._compiled_ignore_message_patterns = [type("P", (), {"search": lambda self, t, timeout=None: marker in str(t) or None})()]
    out2 = _rows(_compress(engine, host))
    assert engine._last_compression_status == "compacted"
    return engine, host, out2, other


@pytest.mark.parametrize("cold", [False, True], ids=["same-engine", "cold-engine"])
def test_rotation_child_gen2_retry(tmp_path, provider, cold):
    engine, host, out2, _other = _child_gen2(tmp_path)
    try:
        rows, parent = engine._store.get_session_count(CHILD), engine._store.get_session_count(SID)
        if cold:
            engine.shutdown()
            engine = _engine(tmp_path, session_id=CHILD)
        got = _retry_and_measure(engine, provider, deepcopy(host))
        assert _identities_lost(engine, host, lineage=(SID,)) == []
        assert (got["rows"], got["status"], got["calls"]) == (rows, "compacted", 0)
        assert (got["reason"], got["out"], engine._store.get_session_count(SID)) == (REASON, out2, parent)
    finally:
        engine.shutdown()


def test_rotation_child_pure_summary_head(tmp_path, provider):
    engine, host, out2, _other = _child_gen2(tmp_path, head="pure-summary")
    try:
        rows = engine._store.get_session_count(CHILD)
        got = _retry_and_measure(engine, provider, deepcopy(host))
        assert _identities_lost(engine, host, lineage=(SID,)) == []
        assert (got["rows"], got["status"], got["calls"], got["reason"], got["out"]) == (rows, "compacted", 0, REASON, out2)
    finally:
        engine.shutdown()


def test_rotation_child_interleaved_with_another_conversation(tmp_path, provider):
    """Store ids of the lineage are not contiguous: only the exact leaf ids are the run."""
    engine, host, out2, other = _child_gen2(tmp_path, interleave=True)
    try:
        rows = engine._store.get_session_count(CHILD)
        got = _retry_and_measure(engine, provider, deepcopy(host))
        assert _identities_lost(engine, host, lineage=(SID,)) == []
        assert (got["rows"], got["status"], got["calls"], got["reason"], got["out"]) == (rows, "compacted", 0, REASON, out2)
    finally:
        other.shutdown()
        engine.shutdown()


def test_rotation_child_ambiguous_occurrences_keep_the_new_turns(tmp_path, provider):
    """#530 review P1: every turn repeats one (X, Y) pair. Read as the host omitting the parent's
    retained rows (their summary glued to the child's first X), replaying the child's rows and
    appending as many NEW identical turns, the list is byte for byte the lineage replay. The
    child's own alignment fits too: the smaller cursor wins, so every new occurrence is stored."""
    engine = _engine(tmp_path)
    x = {"role": "user", "content": "repeat question " + " ".join(f"x{j}" for j in range(150))}
    y = {"role": "assistant", "content": "repeat answer " + " ".join(f"y{j}" for j in range(150))}
    try:
        first = _transcript() + [dict(m) for _ in range(20) for m in (x, y)]
        engine.ingest(first)
        out1 = deepcopy(_compress(engine, first))
        assert engine._generated_context_carrier_remainder(out1[0]) == x["content"]
        engine.on_session_end(SID, first)
        engine.on_session_start(CHILD, boundary_reason="compression", old_session_id=SID,
                                platform="cli", conversation_id=CID, context_length=200_000)
        host = deepcopy(out1) + [dict(m) for _ in range(60) for m in (x, y)]
        engine.ingest(host)
        rows = engine._store.get_session_count(CHILD)
        _compress(engine, host)
        assert engine._last_compression_status == "compacted"
        _retry_and_measure(engine, provider, deepcopy(host))
        assert engine._store.get_session_count(CHILD) >= rows + len(out1)  # the new turns (dups allowed)
        assert _identities_lost(engine, host) == []  # the child alone holds every occurrence
    finally:
        engine.shutdown()


CHILD_CONTROLS = {  # (fixture kwargs, retry list): each stays on today's persisted-batch path
    "forged_carrier": ({}, lambda host: _forge_carrier(host, None)),
    "excluded_parent_lineage_row": ({"exclude": 5}, lambda host: host),
    "edited_parent_row": ({}, lambda host: _edit_row(host, None)),
}


def _child_control_run(tmp_path, monkeypatch, case, *, lineage):
    if not lineage:
        monkeypatch.setattr(LCMEngine, "_head_lineage_rows", lambda self, head, frontier, limit: None, raising=False)
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    kwargs, build = CHILD_CONTROLS[case]
    engine, host, _out2, _other = _child_gen2(tmp_path, **kwargs)
    try:
        retry = build(deepcopy(host))
        got = _retry_and_measure(engine, provider, retry)
        got["lost"] = _identities_lost(engine, retry, lineage=(SID,))
        got["parent"] = engine._store.get_session_count(SID)
        return got
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", sorted(CHILD_CONTROLS))
def test_rotation_child_failed_checks_store_what_today_stores(tmp_path, monkeypatch, case):
    fixed = _child_control_run(tmp_path / "fixed", monkeypatch, case, lineage=True)
    monkeypatch.undo()
    today = _child_control_run(tmp_path / "today", monkeypatch, case, lineage=False)
    assert fixed["reason"] != REASON and fixed["lost"] == [] == today["lost"]
    assert (fixed["rows"], fixed["status"], fixed["calls"], fixed["parent"]) == (
        today["rows"], today["status"], today["calls"], today["parent"])


def test_an_ambiguous_fit_is_not_a_replay(tmp_path, provider):
    """Rows that repeat with the list's period fit at several starts: the delta would be read
    as replay at the longer fit. Only one fit binds; several persist (today's behaviour)."""
    engine, _host, _out2, _proof1 = _gen2(tmp_path)
    try:
        note = {"role": "system", "content": engine._append_lcm_note_to_content("host system prompt")}

        def turn(store_id, repeating):
            text = ("continue" if store_id % 2 else "continuing") if repeating else f"turn {store_id}"
            return {"store_id": store_id, "session_id": SID, "role": "user" if store_id % 2 else "assistant", "content": text}

        def replay(ids, repeating):
            return [note] + [{k: v for k, v in turn(i, repeating).items() if k in ("role", "content")} for i in ids]

        term = engine._cursor_from_frontier_bound_replay
        assert engine._is_verified_replay_scaffold_message(note)
        unique = [turn(store_id, False) for store_id in range(221, 233)]  # F = 228: 229..232 above it
        assert term(replay(range(225, 233), False), unique) == 9
        repeating = [turn(store_id, True) for store_id in range(221, 233)]
        assert term(replay(range(225, 233), True), repeating) is None  # also fits from 227
        assert term(replay(range(225, 235), True), repeating) is None  # 223 would read the 2 new rows as replay
        assert term(replay(range(229, 233), False), unique) is None  # rows above F only: not this term's case
        engine._last_compacted_store_id = 0  # the in-process marker disagrees with the lifecycle frontier
        assert term(replay(range(225, 233), False), unique) is None
    finally:
        engine.shutdown()


def test_a_long_ambiguous_list_computes_each_identity_once(tmp_path, provider, monkeypatch):
    """400 periodic rows: every start fits, so the term stops at the second fit, and each message
    identity and each stored row's forms are computed at most once."""
    engine, _host, _out2, _proof1 = _gen2(tmp_path)
    try:
        note = {"role": "system", "content": engine._append_lcm_note_to_content("host system prompt")}
        rows = [{"store_id": i, "session_id": SID, "role": "user" if i % 2 else "assistant",
                 "content": "continue" if i % 2 else "continuing"} for i in range(29, 429)]  # F = 228
        messages = [note] + [{"role": row["role"], "content": row["content"]} for row in rows[100:]]
        live, forms = [], []
        real_identity, real_forms = engine._message_replay_identity, engine._stored_row_forms
        monkeypatch.setattr(engine, "_message_replay_identity", lambda m, **kw: (
            live.append(1) if not kw.get("stored_row") else None) or real_identity(m, **kw))
        monkeypatch.setattr(engine, "_stored_row_forms", lambda row: forms.append(1) or real_forms(row))
        assert engine._cursor_from_frontier_bound_replay(messages, rows) is None
        assert len(live) <= len(messages) and len(forms) <= len(rows), (len(live), len(forms))
    finally:
        engine.shutdown()


# -- fresh rows shaped like LCM scaffold are content, not a replay head (#524 fix round 1) ----------

NEW = "NEW never-stored user objective 524"
OBJECTIVE = "[Current user objective preserved from compacted history]"
TODO = "[Your active task list was preserved across context compression]"


def _new_objective():
    return {"role": "user", "content": f"{OBJECTIVE}\n{NEW}"}


def _glued_row(host):
    return {"role": "user", "content": host[0]["content"].split("\n\n---\n\n")[-1]}


FRESH_HEADS = {
    "objective_before_rows_below_frontier": lambda host: [_new_objective()] + host[118:],  # rows 227..260
    "objective_before_raw_rows": lambda host: [_new_objective(), _glued_row(host)] + host[1:],
    "objective_merged_into_the_carrier": lambda host: [
        {"role": "user", "content": f"{OBJECTIVE}\n{NEW}\n\n---\n\n{host[0]['content']}"}] + host[1:],
    "objective_before_the_old_carrier": lambda host: [_new_objective()] + host,
    "note_phrases_mid_system_text": lambda host: [{"role": "system", "content": (
        "[Note: This conversation uses Lossless Context Management (LCM). Earlier turns have been "
        f"compacted into hierarchical summaries below.] {NEW}")}] + host,
    "todo_head": lambda host: [{"role": "user", "content": f"{TODO}\n- [ ] {NEW}"}] + host,
}


def _fresh_head_run(tmp_path, monkeypatch, case, *, term):
    if not term:
        _no_term(monkeypatch)
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    engine, host, _out2, _proof1 = _gen2(tmp_path)
    try:
        got = _retry_and_measure(engine, provider, FRESH_HEADS[case](deepcopy(host)))
        stored = [str(row.get("content")) for row in engine._store.get_session_messages(SID, limit=100_000)]
        return {**got, "new_stored": any(NEW in text for text in stored),
                "new_returned": any(NEW in str(row.get("content")) for row in got.pop("out"))}
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", sorted(FRESH_HEADS))
def test_a_fresh_row_shaped_like_scaffold_is_not_skipped(tmp_path, monkeypatch, case):
    fixed = _fresh_head_run(tmp_path / "fixed", monkeypatch, case, term=True)
    monkeypatch.undo()
    today = _fresh_head_run(tmp_path / "today", monkeypatch, case, term=False)
    assert fixed["reason"] != REASON and fixed == today
    # Today's store matcher treats todo / note-phrase rows as scaffold (unchanged here).
    assert fixed["new_stored"] or case in ("todo_head", "note_phrases_mid_system_text"), fixed


# -- fix round 2: a head whose identity goes beyond its bytes (tool calls) is never LCM's emission --

def _tool_call(call_id, arguments):
    return {"id": call_id, "type": "function", "function": {"name": "ledger", "arguments": json.dumps(arguments)}}


def _objective_gen2(tmp_path):
    """Attempt 1 over an objective + tool-pair transcript, adopted; attempt 2 committed and cancelled."""
    engine = _engine(tmp_path)
    first = _transcript() + [{"role": "user", "content": "OBJECTIVE: reconcile the vendor ledger end to end"}]
    for index in range(20):
        first += [{"role": "assistant", "content": f"step {index}", "tool_calls": [_tool_call(f"call_{index}", {})]},
                  {"role": "tool", "tool_call_id": f"call_{index}", "content": f"ledger page {index} " + "x " * 80}]
    engine.ingest(first)
    out1 = deepcopy(_compress(engine, first))
    _adopt(engine, first)
    host = deepcopy(out1) + _more(0, 120)
    engine.ingest(host)
    _compress(engine, host)
    assert engine._last_compression_status == "compacted"
    return engine, host


def _metadata_head_run(tmp_path, monkeypatch, case, *, term):
    if not term:
        _no_term(monkeypatch)
    provider = _Provider()
    monkeypatch.setattr(lcm_engine_module, "summarize_with_escalation", provider)
    if case == "objective_head_reusing_a_stored_call_id":
        engine, host = _objective_gen2(tmp_path)
        objective = next(m for m in host if str(m.get("content")).startswith(OBJECTIVE))
        call = next(i for i, m in enumerate(host) if m.get("tool_calls"))  # a durable call below F
        head = {"role": "assistant", "content": objective["content"],
                "tool_calls": [_tool_call(host[call]["tool_calls"][0]["id"], {"query": NEW})]}
        retry = [head] + deepcopy(host[call + 1:])  # its stored result pairs with the reused id
    else:
        engine, host, _out2, _proof1 = _gen2(tmp_path)
        head_text = host[0]["content"]
        summary = head_text[: head_text.index("\n\n---\n\n") if "\n\n---\n\n" in head_text else head_text.rindex("]") + 1]
        assert engine._is_lcm_emitted_head_row({"role": "assistant", "content": summary}, 10**9)
        head = {"role": "assistant", "content": summary, "tool_calls": [_tool_call("call_new_524", {"query": NEW})]}
        retry = [head, _glued_row(host)] + deepcopy(host[1:])
    try:
        got = _retry_and_measure(engine, provider, retry)
        stored = engine._store.get_session_messages(SID, limit=100_000)
        return {**got, "new_stored": any(NEW in json.dumps(row.get("tool_calls"), default=str) for row in stored),
                "new_returned": any(NEW in json.dumps(row.get("tool_calls"), default=str) for row in got.pop("out"))}
    finally:
        engine.shutdown()


@pytest.mark.parametrize("case", ["objective_head_reusing_a_stored_call_id", "summary_head_with_tool_calls"])
def test_a_head_with_identity_beyond_its_bytes_is_not_skipped(tmp_path, monkeypatch, case):
    """LCM emits summaries, notes and objective parts as bare content: a head row carrying
    tool calls (an identity beyond its bytes) is a real row, whatever its text says."""
    fixed = _metadata_head_run(tmp_path / "fixed", monkeypatch, case, term=True)
    monkeypatch.undo()
    today = _metadata_head_run(tmp_path / "today", monkeypatch, case, term=False)
    assert fixed["reason"] != REASON and fixed == today
    assert fixed["new_stored"] and fixed["new_returned"], fixed


# -- eva-shaped cell: the real Hermes host helpers around a real (unpatched) engine ------------------

_EVA_PROBE = r'''
import importlib.util, json, sys, tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

repo, work = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("hermes_lcm", str(repo / "__init__.py"), submodule_search_locations=[str(repo)])
pkg = importlib.util.module_from_spec(spec); pkg.__path__ = [str(repo)]; sys.modules["hermes_lcm"] = pkg
from agent.agent_runtime_helpers import repair_message_sequence
import agent.conversation_compression as host_cc
from tools.todo_tool import TodoStore
import hermes_lcm.engine as lcm_engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.tokens import count_messages_tokens

SID, CID = "eva-524", "eva-524-conv"
calls = [0]
def stub(*a, **k):
    calls[0] += 1
    return f"Stub summary {calls[0]} of the ledger work.\nExpand for details about: ledger", 1
lcm_engine_module.summarize_with_escalation = stub
shape = sys.argv[3]
if len(sys.argv) > 4:
    LCMEngine._cursor_from_frontier_bound_replay = lambda self, messages, rows: None
tempfile.tempdir = str(work)
store = work / "hermes-results"; store.mkdir(parents=True, exist_ok=True)

def tool_pair(i):
    call = f"call_{i}"
    kind = ("persisted", "large", "small")[i % 3]
    if kind == "persisted":
        full = f"PERSISTED {i}:\n" + f"ledger line {i} " * 600
        path = store / f"{call}.txt"; path.write_text(full, encoding="utf-8")
        result = ("<persisted-output>\n"
                  f"This tool result was too large ({len(full):,} characters, {len(full) / 1024:.1f} KB).\n"
                  f"Full output saved to: {path}\n"
                  "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
                  f"Preview (first 30 chars):\n{full[:30]}\n...\n</persisted-output>")
    elif kind == "large":
        result = f"INLINE {i} " + " ".join(f"p{i}x{j}" for j in range(900))
    else:
        result = f"ok {i}"
    return [{"role": "assistant", "content": f"checking {i}", "tool_calls": [{"id": call, "type": "function",
             "function": {"name": "terminal", "arguments": json.dumps({"cmd": f"step {i}"})}}]},
            {"role": "tool", "tool_call_id": call, "content": result}]

def turns(start, count, tag):
    out = []
    for t in range(start, start + count):
        out.append({"role": "user", "content": f"{tag} request {t} " + " ".join(f"u{t}x{j}" for j in range(60))})
        out += tool_pair(t)
        # Hermes streams a text reply after the tool turn: an adjacent assistant row the host repair merges.
        out.append({"role": "assistant", "content": f"{tag} answer {t} " + " ".join(f"a{t}x{j}" for j in range(60))})
        if t % 4 == 0:
            out.append({"role": "assistant", "content": f"{tag} follow-up note {t}"})
    return out

def objective(start, pairs):
    out = [{"role": "user", "content": f"OBJECTIVE {start}: reconcile the vendor ledger end to end"}]
    for i in range(start, start + pairs):
        out += tool_pair(i)
    return out

todo = TodoStore(); todo.write([{"id": "1", "content": "reconcile the vendor ledger", "status": "in_progress"}])
agent = SimpleNamespace(_todo_store=todo, _repair_message_sequence=lambda m: repair_message_sequence(None, m))
def hermes_after_commit(out):
    host = deepcopy(out)
    host_cc._fold_todo_snapshot(agent, host)
    return host, repair_message_sequence(None, host)

cfg = LCMConfig(database_path=str(work / "l.db"))
cfg.large_output_externalization_enabled = True
cfg.large_output_externalization_path = str(work / "externalized")
engine = LCMEngine(config=cfg, hermes_home=str(work / "h"))
engine.on_session_start(SID, platform="cli", conversation_id=CID, context_length=200_000)
c = lambda m: engine.compress(m, current_tokens=count_messages_tokens(m), force=True)
rows_of = lambda o: [LCMEngine._public_compression_row(x) for x in o]
rec = {"shape": shape, "term": len(sys.argv) <= 4}
try:
    first = turns(0, 30, "first") + (objective(100, 20) if shape == "objective" else [
        {"role": "user" if k % 2 == 0 else "assistant", "content": f"plain turn {k} " + " ".join(f"c{k}x{j}" for j in range(40))}
        for k in range(34 if shape == "carrier" else 33)])
    rec["repairs_first"] = repair_message_sequence(None, first)  # the pre-call belt before LCM sees it
    engine.ingest(first)
    out1 = deepcopy(c(first)); rec["s1"] = engine._last_compression_status
    engine.on_session_end(SID, first)
    engine.on_session_start(SID, boundary_reason="compression", old_session_id=SID, platform="cli", conversation_id=CID, context_length=200_000)
    host, rec["repairs_adopt"] = hermes_after_commit(out1)
    rec["out1_head"] = [(m["role"], str(m.get("content"))[:48]) for m in host[:2]]
    rec["carrier_head"] = engine._generated_context_carrier_remainder(host[0]) is not None
    rec["host_todo_rows"] = sum(str(m.get("content") or "").startswith("[Your active task list") for m in host)
    host += turns(200, 30, "second") + objective(300, 20)
    rec["repairs_host"] = repair_message_sequence(None, host)
    engine.ingest(host)
    rows = engine._store.get_session_count(SID)
    out2 = rows_of(c(host)); rec["s2"] = engine._last_compression_status
    rec["F"] = engine._lifecycle.get_by_conversation(CID).current_frontier_store_id
    before = calls[0]
    retry = c(deepcopy(host))
    rec.update(rows_before_retry=rows, rows_after_retry=engine._store.get_session_count(SID),
               retry=engine._last_compression_status, noop=engine._last_compression_noop_reason,
               retry_calls=calls[0] - before, reason=(engine._last_ingest_reconciliation or {}).get("reason"),
               output_eq_attempt2=rows_of(retry) == out2, host_len=len(host),
               persisted_markers_in_host=sum("<persisted-output>" in str(m.get("content")) for m in host),
               externalized_rows=engine._store._conn.execute("SELECT COUNT(*) FROM messages WHERE content LIKE '[Externalized%'").fetchone()[0])
finally:
    engine.shutdown()
print(json.dumps(rec, default=str))
'''


def _hermes_python():
    python = os.environ.get("LCM_REAL_HERMES_PYTHON")
    if python:
        return python, os.environ.get("LCM_REAL_HERMES_SRC", "")
    if importlib.util.find_spec("hermes_cli") is not None:
        return sys.executable, ""
    return None


# #535: in the pure shape the compacted output ends in a retained real user row, and the host's
# _merge_consecutive_users glues the next user text behind it (behind the todo fold, when one is
# present). The base passed [pure] only because #516 dropped that text from the replay identity.
_ISSUE_535 = (
    "#535: the host merges the new user text behind the retained last user row and the gen-2 retry "
    "does not align it: the retry compaction errors and the retained rows are stored again "
    "(duplicates, no loss)"
)
_EVA_TODO_WRITE = 'todo.write([{"id": "1", "content": "reconcile the vendor ledger", "status": "in_progress"}])'
assert _EVA_TODO_WRITE in _EVA_PROBE  # the no-todo variant below must really drop the fold


def _run_eva_probe(tmp_path, shape, probe=_EVA_PROBE):
    python, src = _hermes_python()
    return subprocess.run(
        [python, "-c", probe, str(Path(__file__).resolve().parent.parent), str(tmp_path), shape],
        cwd=src or str(tmp_path), capture_output=True, text=True, timeout=600, check=False,
        env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin", "TMPDIR": str(tmp_path), "PYTHONDONTWRITEBYTECODE": "1"},
    )


def _assert_eva_gen2_retry(done, shape):
    assert done.returncode == 0, done.stderr[-4000:]
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["s1"] == got["s2"] == "compacted" and got["repairs_first"] > 0, got
    assert got["persisted_markers_in_host"] > 0 and got["externalized_rows"] > 0, got
    assert got["carrier_head"] is (shape == "carrier"), got
    assert got["out1_head"][0][1].startswith("[Current user objective" if shape == "objective" else "[Recent Summary"), got
    assert got["rows_after_retry"] == got["rows_before_retry"], got
    assert (got["retry"], got["retry_calls"], got["reason"], got["output_eq_attempt2"]) == ("compacted", 0, REASON, True), got


@pytest.mark.skipif(_hermes_python() is None, reason="no real Hermes runtime available")
@pytest.mark.parametrize(
    "shape", ["objective", pytest.param("pure", marks=pytest.mark.xfail(strict=True, reason=_ISSUE_535)), "carrier"]
)
def test_eva_shaped_gen2_retry(tmp_path, shape):
    """Tool pairs with persisted-output markers, externalized payloads, adjacent assistant rows the
    host repair merges, the host's todo fold, and an objective / pure-summary / carrier head."""
    _assert_eva_gen2_retry(_run_eva_probe(tmp_path, shape), shape)


@pytest.mark.skipif(_hermes_python() is None, reason="no real Hermes runtime available")
@pytest.mark.xfail(strict=True, reason=_ISSUE_535)
def test_eva_shaped_gen2_retry_plain_merge_behind_retained_user_row(tmp_path):
    """#535 pinned without the todo fold: an empty host todo store folds nothing, so the host merges
    the next user text straight behind the retained last user row (same gap, same target)."""
    _assert_eva_gen2_retry(_run_eva_probe(tmp_path, "pure", _EVA_PROBE.replace(_EVA_TODO_WRITE, "")), "pure")
