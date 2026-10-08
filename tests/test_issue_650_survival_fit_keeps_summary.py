"""#650: the survival fit keeps LCM's summary prefix.

On Hermes the list has no system row and LCM's assembled summary is the first row (role user); the fit
used to drop it with the oldest turns. The prefix now stays whenever whole oldest turns can leave
instead and the final list (system slot, prefix, fitted tail, notice) fits the budget; otherwise the old
rule applies and a WARNING says so. Summaries are stubbed; the list is assembled by compress()."""

from __future__ import annotations

import json
import logging
import sys
import types

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

PAD = " alpha beta gamma delta" * 30
NOTICE = "[LCM survival fit:"
EMERGENCY = "LCM survival fit dropped the summary prefix"


def _rough(messages) -> int:
    """A stand-in for the host's rough request estimator (chars / 4 over content and tool calls)."""
    return sum(4 + (len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or ""))) // 4
               for m in messages)


@pytest.fixture
def host(monkeypatch):
    module = types.ModuleType("agent.model_metadata")
    module.estimate_messages_tokens_rough = _rough
    monkeypatch.setitem(sys.modules, "agent.model_metadata", module)

    def summarize(**kwargs):
        return "Earlier turns." + " summary" * 150 + "\nExpand for details about: turns", 1

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    return _rough


def _engine(tmp_path, window=24_000) -> LCMEngine:
    engine = LCMEngine(config=LCMConfig(fresh_tail_count=4, leaf_chunk_tokens=1600, context_threshold=0.75,
                                        database_path=str(tmp_path / "lcm.db"),
                                        threshold_full_sweep_enabled=True))
    engine.on_session_start("S", platform="telegram", context_length=window, conversation_id="conv")
    return engine


def _turn(tag: str, ts: float, list_users: bool = False, stamped_reply: bool = False) -> list[dict]:
    text = f"[{tag}] user turn{PAD}"
    reply = {"role": "assistant", "content": f"reply to {tag}{PAD}"}
    return [{"role": "user", "content": [{"type": "text", "text": text}] if list_users else text, "timestamp": ts},
            {**reply, "timestamp": ts + 1} if stamped_reply else reply]


def _rows(engine) -> list[dict]:
    return engine._store.get_session_messages(engine._session_id, limit=100_000)


def _hidden_backlog(engine, *, system=False, list_users=False, big_newest=0, stamped_replies=False) -> list[dict]:
    """Every turn stored; the host list no longer shows the oldest 48 turns (a hidden backlog)."""
    head = [{"role": "system", "content": "system prompt"}] if system else []
    turns = [r for i in range(108) for r in _turn(f"T{i:03d}", 10.0 * i, list_users, stamped_replies)]
    if big_newest:
        turns += [{"role": "user", "content": "[N] newest", "timestamp": 5000.0},
                  {"role": "assistant", "content": "answer " * big_newest}]
    engine.ingest(head + turns)
    view = head + turns[2 * 48:]
    engine._ingest_cursor = len(view)  # the host's cursor over the list it shows
    return view


def _spy(engine, monkeypatch, *, fit=True) -> dict:
    """The list compress() hands the fit and the fit's budget; ``fit=False`` returns that list unfitted."""
    seen: dict = {}
    real_fit, real_budget = engine._survival_fit, engine._survival_fit_budget

    def spy_fit(messages, result, *args, **kwargs):
        seen["input"] = list(result) if isinstance(result, list) else result
        seen["output"] = real_fit(messages, result, *args, **kwargs) if fit else result
        return seen["output"]

    def spy_budget(*args, **kwargs):
        seen["budget"] = real_budget(*args, **kwargs)
        return seen["budget"]

    monkeypatch.setattr(engine, "_survival_fit", spy_fit)
    monkeypatch.setattr(engine, "_survival_fit_budget", spy_budget)
    return seen


def _left(before, after) -> list[dict]:
    return [m for m in before if all(m is not kept for kept in after)]


def _today(pre, lead, budget, measure) -> list[dict]:
    """The v0.24.6 rule for an all-stored list: the oldest whole turns after the system slot leave, the
    first cut at a user row whose remainder fits (the summary prefix may leave with them)."""
    for index in range(lead + 1, len(pre)):
        if pre[index].get("role") == "user" and measure(pre[:lead] + pre[index:]) <= budget:
            return pre[:lead] + pre[index:]
    raise AssertionError("no whole-turn cut fits")


def _cold_ingest_adds(tmp_path, fitted) -> int:
    cold = _engine(tmp_path)
    try:
        before = len(_rows(cold))
        cold.ingest(fitted)
        return len(_rows(cold)) - before
    finally:
        cold.shutdown()


# -- T1 (+ #650 round 2: carrier turn integrity, appended rows) ---------------------------------------

def _summary_part(engine, message) -> str:
    content = message["content"]
    return content[:engine._verified_lcm_summary_prefix_end(content)]


def _assert_turns_whole(result) -> None:
    """Every kept raw user row but the newest has its reply (a summary row is not a turn)."""
    users = [i for i, m in enumerate(result) if m["role"] == "user" and "Summary (d" not in str(m["content"])[:40]]
    assert all(result[i + 1]["role"] == "assistant" for i in users[:-1]), [m["role"] for m in result[:6]]


@pytest.mark.parametrize("list_users", [False, True], ids=["carrier-summary", "plain-summary"])
def test_t1_hermes_list_keeps_the_summary_row(tmp_path, host, monkeypatch, list_users):
    """No system row; the summary compress() assembled is the first row (a carrier when the next user row
    is a string, as Hermes merges it). Over budget: the summary stays first, only raw stored rows leave,
    whole turns only (the carrier's own user row leaves with its reply and the carrier is re-formed around
    the first kept user row), the list fits, re-ingesting it stores nothing, and a turn the host appends
    stores exactly its own rows (same process and cold)."""
    engine = _engine(tmp_path)
    try:
        view = _hidden_backlog(engine, list_users=list_users)
        seen = _spy(engine, monkeypatch)
        result = engine.compress(view, current_tokens=host(view))
        pre, budget = seen["input"], seen["budget"]
        assert pre[0]["role"] == "user" and "Summary (d" in str(pre[0]["content"])
        assert engine._survival_generated(pre[0])
        carrier = engine._generated_context_carrier_remainder(pre[0])
        assert (carrier is not None) is (not list_users)
        assert engine._survival_measure(pre) > budget and engine._last_survival_fit is not None
        store_ids = engine._get_store_id_map_for_messages(pre)
        if list_users:
            assert result[0] is pre[0], str(result[0]["content"])[:60]
            dropped = _left(pre, result)
        else:  # re-formed exactly as assembly forms a carrier: summary + the first kept raw user row
            kept = _left(result, pre)
            assert kept == [result[0]] and _summary_part(engine, result[0]) == _summary_part(engine, pre[0])
            first = next(m for m in pre if m["role"] == "user" and m["content"] ==
                         engine._generated_context_carrier_remainder(result[0]))
            assert pre.index(first) > 1 and pre[pre.index(first) + 1] is result[1]
            dropped = [m for m in _left(pre[1:], result) if m is not first]  # the first kept row is merged
            dropped.insert(0, {"role": "user", "content": carrier})  # the carrier's row left with its turn
            assert pre[1]["role"] == "assistant" and all(pre[1] is not m for m in result)
            assert not (result[0]["role"] == "user" and result[1]["role"] == "user")
        assert dropped and all(not engine._survival_generated(m) for m in dropped)
        assert all(id(m) in store_ids for m in dropped if m in pre)
        _assert_turns_whole(result)
        assert engine._survival_measure(result) <= budget and host(result) <= budget
        assert engine._last_survival_fit["dropped_rows"] == len(dropped)
        assert not any(NOTICE in str(m.get("content")) for m in result)  # no system row: no in-list notice
        stored = len(_rows(engine))
        engine.ingest(result)
        assert len(_rows(engine)) == stored
        new = _turn("NEW", 9000.0)
        engine.ingest([*result, *new])
        added = _rows(engine)[stored:]
        assert [(r["role"], r["content"]) for r in added] == [(m["role"], m["content"]) for m in new]
        engine.on_session_end("S", [*result, *new])
        assert len(_rows(engine)) == stored + 2
    finally:
        engine.shutdown()
    assert _cold_ingest_adds(tmp_path, [*result, *new]) == 0


@pytest.mark.parametrize("list_users", [False, True], ids=["carrier-summary", "plain-summary"])
def test_t1_hermes_commit_branch_then_cold_resume(tmp_path, host, monkeypatch, list_users):
    """Hermes after a fitted compaction: session end with the ORIGINAL input (the compaction-commit
    branch), then a cold process ingests the fitted list plus a new turn: only the new turn is stored."""
    engine = _engine(tmp_path)
    try:
        view = _hidden_backlog(engine, list_users=list_users)
        result = engine.compress(view, current_tokens=host(view))
        assert engine._last_survival_fit is not None
        stored = [(r["role"], r["content"]) for r in _rows(engine)]
        engine.on_session_end("S", view)
        assert [(r["role"], r["content"]) for r in _rows(engine)] == stored
    finally:
        engine.shutdown()
    new = _turn("NEW", 9000.0)
    cold = _engine(tmp_path)
    try:
        cold.ingest([*result, *new])
        rows = [(r["role"], r["content"]) for r in _rows(cold)]
        assert rows == stored + [(m["role"], m["content"]) for m in new]
    finally:
        cold.shutdown()


# -- T2 ---------------------------------------------------------------------------------------------

def test_t2_system_head_keeps_the_summary_after_it_and_the_notice_in_it(tmp_path, host, monkeypatch):
    engine = _engine(tmp_path)
    try:
        view = _hidden_backlog(engine, system=True)
        seen = _spy(engine, monkeypatch)
        result = engine.compress(view, current_tokens=host(view))
        pre, budget = seen["input"], seen["budget"]
        assert pre[0]["role"] == "system" and engine._survival_generated(pre[1])
        assert engine._survival_measure(pre) > budget and engine._last_survival_fit is not None
        assert result[0]["role"] == "system" and NOTICE in result[0]["content"]
        assert result[1] is pre[1]  # the summary, unedited, right after the system row
        assert not any(NOTICE in str(m.get("content")) for m in result[1:])
        dropped = _left(pre, result)
        store_ids = engine._get_store_id_map_for_messages(pre)
        assert dropped[0] is pre[0]  # the system row is replaced by its copy carrying the notice
        assert all(id(m) in store_ids and not engine._survival_generated(m) for m in dropped[1:])
        assert engine._last_survival_fit["dropped_rows"] == len(dropped) - 1
        assert engine._survival_measure(result) <= budget
    finally:
        engine.shutdown()


# -- T3 / T4: the emergency ---------------------------------------------------------------------------

def _emergency(tmp_path, host, monkeypatch, *, list_users):
    """The assembled list, and a budget the newest turn fits but the summary prefix plus it does not."""
    engine = _engine(tmp_path)
    view = _hidden_backlog(engine, list_users=list_users, big_newest=4000)
    seen = _spy(engine, monkeypatch, fit=False)
    result = engine.compress(view, current_tokens=host(view))
    pre = seen["input"]  # exercise the fit's assembled input, before #904's outgoing no-growth guard
    assert result is view and engine._survival_generated(pre[0])
    measure = engine._survival_measure
    newest = pre[max(i for i, m in enumerate(pre) if m["role"] == "user"):]
    prefix = measure(pre[:1])
    budget = measure(newest) + prefix // 2
    assert measure(newest) <= budget < measure(pre[:1] + newest)
    del engine._survival_fit, engine._survival_fit_budget  # the spies go; the real fit runs below
    return engine, pre, budget, prefix


def test_t3_prefix_plus_newest_turn_over_budget_falls_back_to_the_old_rule(tmp_path, host, monkeypatch, caplog):
    engine, pre, budget, prefix = _emergency(tmp_path, host, monkeypatch, list_users=True)
    try:
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine._survival_fit(pre, list(pre), 0, "test", request_cap=budget)
        assert engine._survival_measure(result) <= budget
        assert result == _today(pre, 0, budget, engine._survival_measure)
        assert result[0] is not pre[0] and not engine._survival_generated(result[0])
        warnings = [r.getMessage() for r in caplog.records if EMERGENCY in r.getMessage()]
        assert warnings == [f"{EMERGENCY} (emergency: prefix={prefix} tokens, budget={budget})"]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("emergency", [False, True], ids=["kept", "emergency"])
def test_t4_a_carrier_in_the_prefix_is_never_silently_lost(tmp_path, host, monkeypatch, caplog, emergency):
    """The carrier (summary + a real user row) either stays as a carrier, re-formed around the first kept
    user row once its own row left with its turn (counted), within budget; or, in the emergency, leaves
    whole with its turn as v0.24.6 drops it (its row stored; the emergency WARNING says so)."""
    engine, pre, budget, _ = _emergency(tmp_path, host, monkeypatch, list_users=False)
    try:
        remainder = engine._generated_context_carrier_remainder(pre[0])
        assert remainder is not None
        if not emergency:
            budget = engine._survival_measure(pre) - 1  # an old turn can leave instead
        with caplog.at_level(logging.WARNING, logger="hermes_lcm"):
            result = engine._survival_fit(pre, list(pre), 0, "test", request_cap=budget)
        assert engine._survival_measure(result) <= budget
        dropped = _left(pre, result)
        assert (EMERGENCY in caplog.text) is emergency
        assert any(r["role"] == "user" and r["content"] == remainder for r in _rows(engine))
        # emergency: the carrier left whole (v0.24.6's count); kept: the carrier object is re-formed and
        # the first kept user row merged into it, while the carrier's own row left and is counted
        assert engine._last_survival_fit["dropped_rows"] == len(dropped) - 1
        assert dropped[0] is pre[0] and pre[1]["role"] == "assistant" and dropped[1] is pre[1]
        if not emergency:
            first = engine._generated_context_carrier_remainder(result[0])
            assert first is not None and first != remainder and result[1]["role"] == "assistant"
            assert _summary_part(engine, result[0]) == _summary_part(engine, pre[0])
    finally:
        engine.shutdown()


# -- T5 ---------------------------------------------------------------------------------------------

def test_t5_a_list_under_budget_is_not_fitted(tmp_path, host, monkeypatch):
    engine = _engine(tmp_path, window=200_000)
    try:
        view = _hidden_backlog(engine)
        seen = _spy(engine, monkeypatch)
        result = engine.compress(view, current_tokens=host(view))
        assert engine._survival_measure(seen["input"]) <= seen["budget"]
        assert result == seen["input"] and engine._last_survival_fit is None
        assert engine._survival_generated(result[0])
    finally:
        engine.shutdown()


# -- round 2 F1: one store-id map over the whole list ---------------------------------------------------

@pytest.mark.parametrize("lead", ["phrase-row", "carrier"])
@pytest.mark.parametrize("stored_copy", [False, True], ids=["unstored-copy", "stored-copy"])
def test_r2_f1_the_prefix_cut_maps_the_whole_list(tmp_path, host, stored_copy, lead):
    """The review's input: a stored row P (id 3) ahead of a copy of an older stored reply (id 2). Mapped on
    the whole list the copy is new (unstored: it never leaves, as v0.24.6 keeps it) or the newest stored
    copy (id 4); a map of the list without P would take id 2 for it. Since #650 round 3 a P that only
    matches a summary phrase is an ordinary row (the v0.24.6 rule); as the remainder of a verified
    carrier it is split out of the kept prefix and leaves with its turn."""
    from hermes_lcm.dag import SummaryNode

    engine = _engine(tmp_path, window=200_000)
    reply = {"role": "assistant", "content": "An ordinary detailed reply. " * 400}
    phrase = {"role": "user", "content": "Please explain CONTEXT SUMMARY.", "timestamp": 3.0}
    try:
        stored = [engine._store.append("S", dict(m), conversation_id="conv")
                  for m in ({"role": "user", "content": "An old question.", "timestamp": 1.0}, reply, phrase)]
        assert stored == [1, 2, 3]
        copy = dict(reply)
        if stored_copy:
            assert engine._store.append("S", dict(copy), conversation_id="conv") == 4
        newest = {"role": "user", "content": "The newest question?", "timestamp": 5.0}
        first, expected = phrase, [newest]
        if lead == "carrier":
            node = engine._dag.add_node(SummaryNode(session_id="S", summary="s", token_count=1, source_token_count=1,
                                                    source_ids=[1], expand_hint="turns"))
            summary = f"[Recent Summary (d0, node {node})]\ns\n[Expand for details: turns]"
            first = {"role": "user", "content": f"{summary}\n\n{phrase['content']}"}
            assert engine._generated_context_carrier_remainder(first) == phrase["content"]
            expected = [{"role": "user", "content": summary}, newest]
        active = [first, copy, newest]
        whole = engine._get_store_id_map_for_messages(active)
        result = engine._survival_fit(active, list(active), 0, "test", after_exception=True,
                                      request_cap=engine._survival_measure(expected))
        if stored_copy:
            assert whole.get(id(copy)) == 4 and result == expected
            assert engine._get_store_id_map_for_messages(active[1:]).get(id(copy)) == 2  # a sublist map errs
            assert "store ids 3..4" in engine._last_survival_fit["notice"]
        else:
            assert id(copy) not in whole
            assert any(m is copy for m in result) and engine._last_survival_fit is None
    finally:
        engine.shutdown()


# -- round 2 F2: the cut is chosen by the final list, notice included ----------------------------------

def test_r2_f2_the_notice_never_pushes_a_whole_turn_cut_over_budget(tmp_path, monkeypatch):
    """The review's shape: a system slot, a verified summary carrier too large to keep with the newest turn,
    nine more stored rows (store ids 1..10) and a large newest turn, swept token by token across the band
    where the old cut fits without its notice. Where v0.24.6's list (its notice included) fits, the result
    fits; elsewhere it is never larger than v0.24.6's."""
    from hermes_lcm.dag import SummaryNode
    from hermes_lcm.survival_fit import _NOTICE

    monkeypatch.setitem(sys.modules, "agent.model_metadata", None)  # LCM's own character-based count
    rows = [r for i in range(5) for r in _turn(f"D{i}", 10.0 * i)]
    system = {"role": "system", "content": "systemxx"}
    engine = _engine(tmp_path)
    try:
        assert [engine._store.append("S", dict(m), conversation_id="conv") for m in rows] == list(range(1, 11))
        node = engine._dag.add_node(SummaryNode(session_id="S", summary="saved history " * 1000, token_count=1,
                                                source_token_count=1, source_ids=[1], expand_hint="turns"))
        summary = f"[Recent Summary (d0, node {node})]\n{'saved history ' * 1000}\n[Expand for details: turns]"
        carrier = {"role": "user", "content": f"{summary}\n\n{rows[0]['content']}"}
        assert engine._generated_context_carrier_remainder(carrier) == rows[0]["content"]
        measure, budget = engine._survival_measure, engine._survival_fit_budget([system], 0)
        old_head = {"role": "system", "content": engine._survival_with_notice(
            "systemxx", _NOTICE.format(n=9, first=1, last=10))}  # v0.24.6's head for this cut
        exact = 0
        for pad in range(81_384 - 200, 81_384 + 1, 4):
            turn = [{"role": "user", "content": "the newest question", "timestamp": 500.0},
                    {"role": "assistant", "content": "answer" + "t" * pad}]
            active = [system, carrier, *rows[1:], *turn]
            engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
            result = engine._survival_fit(active, list(active), 0, "test")
            old = measure([old_head, *turn])
            exact += old == budget
            assert measure(result) <= (budget if old <= budget else old), (pad, measure(result), old, budget)
        assert exact  # the sweep hit the list v0.24.6 returned at exactly the budget
    finally:
        engine.shutdown()


# -- round 3 F5: the protected prefix is provenance, never a summary phrase ----------------------------

PHRASE = "please explain this CONTEXT SUMMARY"


@pytest.mark.parametrize("system", [False, True], ids=["no-system", "system"])
def test_r3_f5_a_real_row_quoting_a_summary_phrase_is_not_protected(tmp_path, host, monkeypatch, system):
    """The first kept raw user row mentions "CONTEXT SUMMARY" and follows LCM's summary. LCM's summary is
    recognised by provenance (scaffold or verified carrier), never by the phrase; the phrase row is an
    ordinary turn: it and its reply stay together, and no kept request loses its reply."""
    engine = _engine(tmp_path)
    try:
        head = [{"role": "system", "content": "system prompt"}] if system else []
        turns = [r for i in range(108) for r in _turn(f"T{i:03d}", 10.0 * i)]
        turns[2 * 48]["content"] = f"[T048] {PHRASE}{PAD}"  # the first row the host still shows
        engine.ingest(head + turns)
        view = head + turns[2 * 48:]
        engine._ingest_cursor = len(view)
        seen = _spy(engine, monkeypatch)
        result = engine.compress(view, current_tokens=host(view))
        pre, budget, lead = seen["input"], seen["budget"], len(head)
        summary = pre[lead]
        phrase = pre[lead + 1] if system else {"role": "user", "content": view[lead]["content"]}
        reply = pre[lead + 1 + system]
        assert PHRASE in phrase["content"] and reply["role"] == "assistant"
        assert not engine._is_context_summary_content(summary["content"].split("\n\n[T048]")[0])
        assert (engine._is_replayed_context_scaffold_message(summary)
                or engine._generated_context_carrier_remainder(summary) is not None)
        if system:  # assembly emits [system, summary, phrase row, reply, ...]
            assert engine._survival_generated(phrase)  # the phrase alone matches it
        else:  # assembly glues the phrase row to the summary: a verified carrier
            assert engine._generated_context_carrier_remainder(summary) == phrase["content"]
        assert engine._last_survival_fit is not None and engine._survival_measure(result) <= budget
        assert "Summary (d" in str(result[lead]["content"])[:40]  # LCM's summary kept
        kept_phrase = any(m.get("content") == phrase["content"]
                          or engine._generated_context_carrier_remainder(m) == phrase["content"] for m in result)
        assert kept_phrase is any(m is reply for m in result)  # the turn stays whole
        _assert_turns_whole(result)
        if not system:
            assert not (result[0]["role"] == "user" and result[1]["role"] == "user")
    finally:
        engine.shutdown()


# -- round 3: the split carrier's fallback shape is assembly's own -------------------------------------

@pytest.mark.parametrize("case", ["list-content-next", "next-is-newest"])
def test_r3_split_carrier_without_a_carrier_shape_matches_assembly(tmp_path, host, case):
    """When the first kept user row has list content, or is the newest user row, assembly
    (engine._assemble_context, the carrier guard) emits the user-role summary and that row separately;
    the fit's leading rows are exactly that shape: [summary, U_k, its reply, ...]."""
    from hermes_lcm.dag import SummaryNode

    engine = _engine(tmp_path, window=200_000)
    try:
        node = engine._dag.add_node(SummaryNode(session_id="S", summary="saved history " * 50, token_count=1,
                                                source_token_count=1, source_ids=[1], expand_hint="turns"))
        summary = f"[Recent Summary (d0, node {node})]\n{'saved history ' * 50}\n[Expand for details: turns]"
        rows = _turn("U1", 1.0) + _turn("U2", 2.0, list_users=case == "list-content-next")
        if case == "list-content-next":
            rows += _turn("U3", 3.0)
        for m in rows:
            engine._store.append("S", dict(m), conversation_id="conv")
        carrier = {"role": "user", "content": f"{summary}\n\n{rows[0]['content']}"}
        assert engine._generated_context_carrier_remainder(carrier) == rows[0]["content"]
        active = [carrier, *rows[1:]]
        expected = [{"role": "user", "content": summary}, *rows[2:]]
        engine._ingest_cursor, engine._ingest_cursor_needs_reconcile = len(active), False
        result = engine._survival_fit(active, list(active), 0, "test",
                                      request_cap=engine._survival_measure(expected))
        assert result == expected
        assert engine._is_replayed_context_scaffold_message(result[0])
    finally:
        engine.shutdown()


# -- #1000: a fit's re-formed carrier is a replay, never a new row --------------------------------------

_BOUNDARY = dict(boundary_reason="compression", old_session_id="S", platform="telegram",
                 conversation_id="conv", context_length=24_000)


def _held_fit_after_inplace_boundary(tmp_path, host, monkeypatch, path="held"):
    """Hermes 0.21.x: a published compaction, the in-place boundary (no ``on_session_end`` commit), more
    turns, then a compress held at the ceiling (#618) whose survival fit re-forms the carrier around the
    first kept user row U, and the in-place boundary again (the cursor resets: the fit's proof is unconsumed).
    ``path="exception"``: the compress fails instead and the fit runs on its exception path (no proof)."""
    engine = _engine(tmp_path)
    view = _hidden_backlog(engine, stamped_replies=True)  # string users -> an assembled carrier; Hermes stamps every row
    first = engine.compress(view, current_tokens=host(view))  # publishes (durable proof)
    engine.on_session_start("S", **_BOUNDARY)  # the 0.21.x in-place commit signal
    grown = [*first, *[r for i in range(12) for r in _turn(f"G{i:02d}", 20_000.0 + 10 * i, stamped_reply=True)],
             {"role": "user", "content": "[CUR] api variant", "timestamp": 30_000.0}]
    engine.ingest(grown)  # host per-turn persistence
    if path == "held":
        monkeypatch.setattr(engine, "_hold_fit_only_applies", lambda tokens: True)  # #618 held
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("summariser down")

        monkeypatch.setattr(engine, "_compress_impl", fail)
    fitted = engine.compress(grown, current_tokens=host(grown))
    assert engine._last_survival_fit and fitted is not grown
    assert (engine.last_compression_noop_reason == "held") is (path == "held")
    rest = engine._generated_context_carrier_remainder(fitted[0])
    assert rest and rest != engine._generated_context_carrier_remainder(first[0])  # re-formed around U
    engine.on_session_start("S", **_BOUNDARY)
    assert engine._ingest_cursor == 0 and engine._ingest_cursor_needs_reconcile  # documents the reset
    return engine, fitted, rest


@pytest.mark.parametrize("host_stamp", [False, True], ids=["unstamped", "host-stamped"])
@pytest.mark.parametrize("path", ["held", "exception"])
@pytest.mark.parametrize("rewrite_current", [False, True], ids=["as-returned", "turn-end-rewrite"])
def test_1000_held_fit_then_inplace_boundary_stores_no_carrier(tmp_path, host, monkeypatch, rewrite_current, path,
                                                               host_stamp):
    """The post-turn ingest of the fitted list plus the reply stores the reply (and, after the host's
    turn-end rewrite of the current user row, that row's persist variant), never the re-formed carrier.
    ``host_stamp``: the host stamps the adopted carrier, which has none, with its own clock (Hermes 0.21.5:
    the Track H forced cell stored every carrier with a fresh ``host_message_timestamp``)."""
    engine, fitted, rest = _held_fit_after_inplace_boundary(tmp_path, host, monkeypatch, path)
    try:
        if host_stamp:
            assert "timestamp" not in fitted[0]
            fitted[0]["timestamp"] = 40_000.0  # in place: the host stamps the object it adopted
        if rewrite_current:  # turn_finalizer.py:270: the API variant becomes the persist variant
            fitted[-1] = {**fitted[-1], "content": "[CUR] persist variant"}
        stored = len(_rows(engine))
        engine.ingest([*fitted, {"role": "assistant", "content": "reply CUR", "timestamp": 30_001.0}])
        added = [r["content"] for r in _rows(engine)[stored:]]
        assert not any("Summary (d" in c[:40] for c in added), [c[:60] for c in added]  # main: the carrier
        assert sum(1 for r in _rows(engine) if r["content"] == rest) == 1  # U itself stays one row
        # the persist variant of the current user row is the separate "stored twice" shape, not #1000
        assert [c for c in added if c != "[CUR] persist variant"] == ["reply CUR"], [c[:60] for c in added]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("host_stamp", [False, True], ids=["unstamped", "host-stamped"])
@pytest.mark.parametrize("prompt", ["older-row", "carrier-row"])
def test_1000_loss_guard_host_merged_summary_with_a_repeated_prompt_is_stored(tmp_path, host, monkeypatch, prompt,
                                                                             host_stamp):
    """No carrier descriptor binds a host merge of LCM's standalone summary and a NEW prompt whose text
    equals a stored user row (an older one, or U itself: byte-equal to the emitted carrier, but a second
    occurrence the descriptor does not bind): that row is new content and is stored (duplicate over loss)."""
    engine, fitted, rest = _held_fit_after_inplace_boundary(tmp_path, host, monkeypatch)
    try:
        summary = fitted[0]["content"][:engine._verified_lcm_summary_prefix_end(fitted[0]["content"])]
        repeated = rest if prompt == "carrier-row" else next(
            r["content"] for r in _rows(engine) if r["role"] == "user" and r["content"].startswith("[T050]"))
        merged = {"role": "user", "content": f"{summary}\n\n{repeated}"}  # H1's _merge_consecutive_users
        if host_stamp:  # the host stamps both rows with its own clock
            fitted[0]["timestamp"], merged["timestamp"] = 40_000.0, 40_002.0
        assert engine._generated_context_carrier_remainder(merged) == repeated
        assert (merged["content"] == fitted[0]["content"]) is (prompt == "carrier-row")
        stored = len(_rows(engine))
        engine.ingest([*fitted, {"role": "assistant", "content": "reply CUR", "timestamp": 30_001.0}, merged])
        added = [r["content"] for r in _rows(engine)[stored:]]
        assert "reply CUR" in added
        assert sum(1 for c in added if c in (repeated, merged["content"])) >= 1, [c[:60] for c in added]
    finally:
        engine.shutdown()
