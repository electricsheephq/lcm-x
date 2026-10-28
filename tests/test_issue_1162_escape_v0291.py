"""#1162 v0.29.1: retained-store escape, covered-view restart and base zero-leaf repairs."""
from collections import Counter
import inspect
import hashlib
import json
import logging
import math

import pytest

import hermes_lcm.engine as lcm_engine
from hermes_lcm import leaf_prep
from hermes_lcm.compaction import _CompressionState, _LeafState, _THRESHOLD_FULL_SWEEP_MAX_PASSES
from hermes_lcm.tokens import count_messages_tokens
from tests.test_issue_1162_retained_escape import (
    PAD, _engine, _turn, _reorder, _restore, _passes, _rows, _frontier, _r_id,
    lines as _shared_lines, summaries as _shared_summaries,
)
from tests.test_issue_581_582_store_complete_survival import _assert_contiguous

lines = _shared_lines
summaries = _shared_summaries


@pytest.mark.parametrize("ctx", [32_000, 0])
@pytest.mark.parametrize("compress", [False, True], ids=["direct", "compress"])
def test_b5_lift_is_bounded(tmp_path, summaries, lines, monkeypatch, ctx, compress):
    from hermes_lcm import store_complete
    from hermes_lcm.store_complete import StoreCompleteMixin
    monkeypatch.setattr(store_complete, "_SCAN_LIMIT", 2)
    engine = _engine(tmp_path, False, fresh_tail_count=2, leaf_chunk_tokens=400)
    engine.context_length = ctx
    c1, c2 = ({"role": "user", "content": n + PAD, "timestamp": t}
              for n, t in (("c1", 1.0), ("c2", 3.0)))
    hidden = [{"role": "user", "content": f"h{i}" + " lorem ipsum dolor sit amet" * 40}
              for i in range(6)]
    tail = [r for i in range(2, 5) for r in _turn(f"T{i}", 100.0 * i)] if compress else []
    calls = []
    original = StoreCompleteMixin._store_complete_input
    def counted(self, *args, **kwargs):
        calls.append((args[7], kwargs))
        assert len(calls) <= 4, calls  # fail promptly instead of exhausting Python's stack
        return original(self, *args, **kwargs)
    monkeypatch.setattr(StoreCompleteMixin, "_store_complete_input", counted)
    try:
        engine.ingest([c1, *hidden, c2, *tail])
        ids = [int(r["store_id"]) for r in _rows(engine)]
        chunk, view = [c2, c1], [c2, c1, *tail]
        engine._ingest_cursor = len(view)
        if compress:
            engine.compress(view, current_tokens=10_000)
            assert engine._last_compression_status in ("noop", "compacted")
        else:
            out = engine._store_complete_input(chunk, {}, {id(c1): ids[0], id(c2): ids[7]},
                                               view, chunk, 0, [], count_messages_tokens(chunk),
                                               allow_retained_escape=True)
            assert out == []
            assert len(calls) == 2  # original budget, then exactly one lifted budget
            assert calls[1][0] > calls[0][0]
        assert 1 < len(calls) <= 4
    finally:
        engine.shutdown()


def test_b5_empty_input_keeps_tool_pairing_repair(tmp_path, summaries, lines, monkeypatch):
    engine, view = _reorder(tmp_path, False)
    view.insert(1, {"role": "tool", "tool_call_id": "orphan", "content": "orphan result"})
    engine._ingest_cursor = len(view)
    reached = []
    def no_source(*_args, **_kwargs):
        reached.append(True)
        engine._store_complete_empty_reason = "leaf input has no stored source rows (input_rows_without_source=0)"
        engine._store_complete_blocked = None
        return []
    monkeypatch.setattr(engine, "_identity_anchor_summary_input", no_source)
    try:
        out = engine.compress(view, current_tokens=10_000)
        assert reached and not summaries
        assert out is not view
        assert not any(m.get("tool_call_id") == "orphan" for m in out)
        assert engine._last_compression_status == "sanitized"
    finally:
        engine.shutdown()


def test_b6_assembled_empty_input_keeps_late_result_repair(tmp_path, summaries, lines, monkeypatch):
    engine = _engine(tmp_path, False)
    call, late_result = _turn("late", 100.0, tool=True)[1:3]
    user = {"role": "user", "content": "a later user turn"}
    # A system anchor keeps assembly's provider-leading-role guard from
    # discarding the leading assistant before the pairing repair is returned.
    view = [{"role": "system", "content": "You are helpful"}, call, user, late_result]
    try:
        engine.ingest(view)
        state = _CompressionState(working_messages=view, anchor_source_messages=view,
                                  dropped_replayed_scaffold_messages=True,
                                  budget=engine._new_foreground_budget())
        def no_source(*_args, **_kwargs):
            engine._store_complete_empty_reason = "leaf input has no stored source rows (input_rows_without_source=0)"
            return []
        monkeypatch.setattr(engine, "_identity_anchor_summary_input", no_source)
        action, out = engine._prepare_leaf_input(state, _LeafState(to_compact=[call]), set(), lambda *_a: 0)
        assert action == "break" and out is None
        # Assembly drops late T(X) and inserts a stub T(X) immediately after A.
        # Its id multiset is unchanged, so the removed rider suppressed this repair.
        expected = engine._assemble_context(view[0], view[1:])
        assert expected[1:] == [call, expected[2], user]
        assert expected[2]["role"] == "tool" and expected[2]["tool_call_id"] == late_result["tool_call_id"]
        assert expected[2]["content"] != late_result["content"] and late_result not in expected
        action, out = engine._finish_without_leaf(state, view, lambda: None)
        assert action == "return" and out == expected and out is not view
        assert engine._last_compression_status == "sanitized" and not summaries
    finally:
        engine.shutdown()


@pytest.mark.parametrize("separator", ["\n\n---\n\n", "\n\n"])
def test_b5_memo_covers_carrier(tmp_path, summaries, lines, separator):
    engine, view = _reorder(tmp_path, False)
    try:
        view, _ = _passes(engine, view, 2, lines)
        content = view[0]["content"]
        end = engine._verified_lcm_summary_prefix_end(content)
        assert end > 0
        carrier = {"role": "user", "content": content[:end] + separator + view[-1]["content"]}
        assert engine._generated_context_carrier_remainder(carrier) == view[-1]["content"]
        engine._retained_escape_memos = {}  # prove persisted reload, not just this pass's cache
        assert engine._retained_escape_covered_view(carrier)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("lookup_first", [False, True], ids=["record_read_fails", "lookup_read_fails"])
def test_b5_transient_memo_read_preserves_older_entries(tmp_path, summaries, lines, monkeypatch, lookup_first):
    engine, view = _reorder(tmp_path, False)
    try:
        old_id, new_id = int(_rows(engine)[0]["store_id"]), _r_id(engine)
        engine._record_retained_escape([(old_id, "user", "shown")])
        key = engine._replay_snapshot_metadata_key("retained_escape")
        read = engine._store.read_metadata_json
        prior = read(key)
        engine._retained_escape_memos = {}
        reads = []
        def transient(k, *args, **kwargs):
            if k == key:
                reads.append(k)
                if len(reads) == 1:
                    raise OSError("synthetic transient metadata read failure")
            return read(k, *args, **kwargs)
        monkeypatch.setattr(engine._store, "read_metadata_json", transient)
        if lookup_first:
            assert not engine._retained_escape_covered_view(view[-1])
        else:
            engine._record_retained_escape([(new_id, "user", "shown")])
            assert read(key) == prior  # a failed read cannot replace the persisted memo
        engine._record_retained_escape([(new_id, "user", "shown")])
        assert len(reads) == 2
        assert {old_id, new_id} <= {int(item["store_id"]) for item in read(key)}
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep,backlog,limit", [(True, 0, 3), (True, 12, 4),
                                               (False, 0, 10), (False, 12, 20)])
def test_b5_automatic_drain_without_clearing_holds(tmp_path, summaries, lines, sweep, backlog, limit):
    # Automatic host gate; no force, bypass, hold reset or _passes helper.
    engine, view = _reorder(tmp_path, sweep)
    if backlog:
        engine.shutdown()
        from tests.test_host_uid_shadow import _state_db
        (tmp_path / "backlog").mkdir()
        _state_db(tmp_path / "backlog", [("S", None, None)])
        engine = _engine(tmp_path / "backlog", sweep, fresh_tail_count=1, leaf_chunk_tokens=400)
        hidden = [r for i in range(backlog) for r in _turn(f"H{i}", 10.0 * i)]
        retained = {"role": "user", "content": "[R] retained" + PAD, "message_uid": "retained-r"}
        live = [r for i in range(1, 5) for r in _turn(f"L{i}", 1000.0 + i)]
        engine.ingest([*hidden, retained, *live])
        view = [*live, retained]
        engine._ingest_cursor = len(view)
    try:
        engine._config.dynamic_leaf_chunk_enabled = False
        engine._config.context_threshold = 0.003
        engine.update_model(engine.model, 200_000)
        trace = []
        for turn in range(1, limit + 1):
            blocked = engine._automatic_compression_blocked()
            if not blocked:
                view = engine.compress(view, current_tokens=10_000)
            assert not engine._no_progress_hold or engine._no_progress_hold[1] != "no_progress"
            engine.note_turn_complete()  # the only between-turn action
            assert not engine._no_progress_hold or engine._no_progress_hold[1] != "no_progress"
            trace.append((turn, blocked, _frontier(engine), count_messages_tokens(view)))
            if count_messages_tokens(view) <= engine.threshold_tokens:
                break
        assert trace[-1][-1] <= engine.threshold_tokens, trace
        assert len(trace) <= limit
        print(f"B5 automatic drain sweep={sweep} backlog={backlog} turns={len(trace)} limit={limit}")
        _audit(engine)
    finally:
        engine.shutdown()


def test_b5_rescue_reports_only_published_escapes(tmp_path, summaries, lines, monkeypatch):
    import hermes_lcm.engine as engine_module
    from tests.test_host_uid_shadow import _state_db
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=400)
    r1, r2 = ({"role": "user", "content": f"[R{i}] retained occurrence" + PAD,
               "message_uid": f"retained-r{i}"} for i in (1, 2))
    stored = [r1, r2, *[r for i in range(1, 7) for r in _turn(f"T{i}", 100.0 * i)]]
    calls = []
    def summarize(**kwargs):
        calls.append(("[R1]" in kwargs["text"], "[R2]" in kwargs["text"]))
        if len(calls) == 1:
            raise RuntimeError("maximum context length exceeded")
        return "Earlier turns.\nExpand for details about: turns", 1
    monkeypatch.setattr(engine_module, "summarize_with_escalation", summarize)
    try:
        engine.ingest(stored)
        view = [m for m in stored if m is not r1 and m is not r2] + [dict(r1), dict(r2)]
        engine._ingest_cursor = len(view)
        ids = {r["content"][:4]: int(r["store_id"]) for r in _rows(engine)}
        r1_id, r2_id = ids["[R1]"], ids["[R2]"]
        engine.compress(view, current_tokens=10_000)
        assert calls[0] == (True, True) and calls[-1] == (True, False), calls
        assert _covered(engine)[r1_id] == 1 and _covered(engine)[r2_id] == 0
        assert set(engine._retained_escape_memo().values()) == {r1_id}
        assert engine._compress_escaped == [(r1_id, "user", "shown")]
        warnings = [s for lvl, s in lines if lvl >= logging.WARNING and "retained-occurrence fallback:" in s]
        assert len(warnings) == 1 and f"store_id={r1_id} role=user kind=shown" in warnings[0]
        assert "leaf includes 1 retained row(s)" in warnings[0] and "+1 more" not in warnings[0]
        stop = next(s for _lvl, s in lines if s.startswith("LCM compaction stop:"))
        assert stop.endswith(f" escaped=store_id:{r1_id},role:user,kind:shown,n:1")
        _audit(engine)
    finally:
        engine.shutdown()


def _covered(engine):
    return Counter(int(row[0]) for row in engine._store.connection.execute(
        "SELECT source.value FROM summary_nodes AS node, json_each(node.source_ids) AS source "
        "WHERE node.source_type = 'messages'"))


def _audit(engine):
    counts = _covered(engine)
    assert all(n == 1 for n in counts.values())
    carry = engine._load_compression_carry_ranges()
    rows, truncated = engine._store_complete_owned_rows(0, _frontier(engine), carry)
    assert not truncated
    twins, twin_covered = engine._store_complete_copy_twins(rows, carry)
    dependent = False
    for row in rows:
        sid, role = int(row["store_id"]), row["role"]
        ignored = engine._matches_ignore_message_patterns(row, stored_row=True)
        # The walk's complete exclusion families: ignored, system, existing
        # node coverage, proven re-ingest copy, or reply to an ignored prompt.
        copy = any(t < sid and (counts[t] == 1 or t in twin_covered) for t in twins.get(sid, ()))
        assert counts[sid] == 1 or ignored or role == "system" or copy or (
            dependent and role in ("assistant", "tool")), (sid, role, counts)
        dependent = ignored or (dependent and role not in ("user", "system"))
    if rows and all(counts[int(row["store_id"])] == 1 for row in rows):
        _assert_contiguous(engine)


def _conflicts(lines):
    return [s for _, s in lines if "publication_invariant_conflict" in s or "no durable source coverage" in s]


@pytest.mark.parametrize("sweep", [False, True])
def test_t1_shown_tail_escape(tmp_path, summaries, lines, sweep):
    engine, view = _reorder(tmp_path, sweep)
    try:
        before, retained, r_id = _rows(engine), view[-1], _r_id(engine)
        out, trace = _passes(engine, view, 3, lines)
        assert trace[1]["frontier"] >= r_id, trace
        assert _covered(engine)[r_id] == 1
        warning = [s for lvl, s in lines if lvl >= logging.WARNING and "retained-occurrence fallback:" in s]
        assert warning and f"store_id={r_id}" in warning[0] and "kind=shown" in warning[0]
        assert any(t["line"].endswith(f" escaped=store_id:{r_id},role:user,kind:shown,n:1") for t in trace)
        assert out[-1]["content"] == retained["content"]
        assert _rows(engine) == before and not _conflicts(lines)
        _audit(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("sweep", [False, True])
def test_t2_rewrite_escape(tmp_path, summaries, lines, sweep):
    engine, view = _restore(tmp_path, sweep)
    try:
        # The rewrite ingest is part of the fixture's trigger, not escape storage.
        engine.ingest(view)
        before = _rows(engine)
        out, trace = _passes(engine, view, 3, lines)
        escaped = [t for t in trace if " escaped=" in t["line"]]
        assert escaped, trace
        assert all(t["leaves"] >= 1 for t in trace if t["frontier"] < max(int(r["store_id"]) for r in before[:-4]))
        assert any("kind=shown" in s for _, s in lines if "retained-occurrence fallback" in s)
        assert out[-1]["content"] == view[-1]["content"]
        assert _rows(engine) == before and not _conflicts(lines)
        _audit(engine)
    finally:
        engine.shutdown()


def test_t3_mapped_escape(tmp_path, summaries, lines):
    # B1: explicit full_map isolates the mapped stop; extension normally absorbs R.
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=1)
    r, a, tail = ({"role": "user", "content": label + PAD} for label in ("retained", "oldest chunk", "tail"))
    carrier = {"role": "user", "content": "unstamped carrier-like barrier"}
    try:
        engine.ingest([r, a, tail])
        stored = _rows(engine)
        mapping = {id(m): int(row["store_id"]) for m, row in zip([r, a, tail], stored)}
        options = {"allow_retained_escape": True} if "allow_retained_escape" in inspect.signature(
            engine._store_complete_input).parameters else {}
        out = engine._store_complete_input([a], {}, mapping, [a, carrier, r, tail], [a], 0, [], 1, **options)
        assert out and out[0][1] == [mapping[id(r)]]
        assert out[0][0]["content"] == r["content"]
        assert engine._store_complete_escaped == [(mapping[id(r)], "user", "mapped")]
        assert engine._store_complete_blocked[:3] == (mapping[id(r)], "user", "mapped")
        assert _rows(engine) == stored
    finally:
        engine.shutdown()


@pytest.mark.parametrize("backlog", [0, 12], ids=["protected_tail", "over_budget_backlog"])
def test_t4_bounded_drain(tmp_path, summaries, lines, backlog):
    engine, view = _reorder(tmp_path, True)
    try:
        if backlog:
            engine.shutdown()
            from tests.test_host_uid_shadow import _state_db
            (tmp_path / "backlog").mkdir()
            _state_db(tmp_path / "backlog", [("S", None, None)])
            engine = _engine(tmp_path / "backlog", True, fresh_tail_count=1, leaf_chunk_tokens=400)
            hidden = [r for i in range(backlog) for r in _turn(f"H{i}", 10.0 * i)]
            retained = {"role": "user", "content": "[R] retained" + PAD, "message_uid": "retained-r"}
            live = [r for i in range(1, 5) for r in _turn(f"L{i}", 1000.0 + i)]
            engine.ingest([*hidden, retained, *live])
            view = [*live, retained]
            engine._ingest_cursor = len(view)
        # Choose a threshold that the protected tail alone can fit, and disable
        # dynamic sizing so the bound uses the actual configured leaf budget.
        engine._config.dynamic_leaf_chunk_enabled = False
        engine._config.context_threshold = 0.003
        engine.update_model(engine.model, 200_000)
        assert engine.threshold_tokens == 600
        before = _rows(engine)
        budget = engine._config.leaf_chunk_tokens
        limit = math.ceil(math.ceil(count_messages_tokens([engine._store.to_openai_msg(r) for r in before]) / budget)
                          / _THRESHOLD_FULL_SWEEP_MAX_PASSES) + 2
        trace = []
        for _ in range(limit):
            view, step = _passes(engine, view, 1, lines)
            trace.extend(step)
            assert step[0]["leaves"] >= 1, trace
            if count_messages_tokens(view) <= engine.threshold_tokens:
                break
        assert count_messages_tokens(view) <= engine.threshold_tokens, trace
        assert _rows(engine) == before and not _conflicts(lines)
        _audit(engine)
    finally:
        engine.shutdown()


def test_t5_covered_copy_leaves_tail(tmp_path, summaries, lines, monkeypatch):
    engine, view = _reorder(tmp_path, False)
    try:
        view, _ = _passes(engine, view, 2, lines)
        retained = view[-1]
        assert engine._identity_anchor_covered_view(retained, {})
        with monkeypatch.context() as p:
            p.setattr(engine, "_retained_escape_enabled", False)
            assert not engine._retained_escape_covered_view(retained)
        full_map = engine._get_store_id_map_for_messages(view)
        engine._compress_run.store_ids_by_message_id = full_map
        candidates = [retained, *_turn("C20", 2000.0)]
        assert len(engine._identity_anchor_extend_chunk([retained], candidates)) > 1
        with monkeypatch.context() as p:
            p.setattr(engine, "_retained_escape_covered_view", lambda _m, _frontier=None: False)
            assert engine._identity_anchor_extend_chunk([retained], candidates) == [retained]
        view += [r for i in range(20, 23) for r in _turn(f"C{i}", 100.0 * i)]
        _, trace = _passes(engine, view, 8, lines)
        assert not any(t["reason"] == "selected leaf chunk lacks raw store lineage" for t in trace), trace
    finally:
        engine.shutdown()


def test_t6_cold_resume(tmp_path, summaries, lines):
    engine, view = _reorder(tmp_path, False)
    view, _ = _passes(engine, view, 2, lines)
    before = _rows(engine)
    retained = view[-1]
    engine.shutdown()
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=1)
    try:
        assert engine._identity_anchor_covered_view(retained, {})
        view, trace = _passes(engine, view, 1, lines)
        assert _rows(engine) == before
        assert not any(t["reason"] == "selected leaf chunk lacks raw store lineage" for t in trace), trace
        assert sum(r["content"] == retained["content"] for r in _rows(engine)) == 1
    finally:
        engine.shutdown()


def test_t6r_rotation_child(tmp_path, summaries, lines):
    from tests.test_host_uid_shadow import _state_db
    engine, view = _reorder(tmp_path, False)
    try:
        view, _ = _passes(engine, view, 2, lines)
        retained = view[-1]
        assert engine._identity_anchor_covered_view(retained, {})
        _state_db(tmp_path, [("S", None, "compression"), ("S2", "S", None)])
        engine.on_session_start("S2", platform="cli", context_length=200_000,
                                conversation_id="conv", boundary_reason="compression", old_session_id="S")
        assert engine._identity_anchor_covered_view(retained, {})
        # The wedge symptom: the child must still step over the escaped host copy.
        engine._compress_run.store_ids_by_message_id = engine._get_store_id_map_for_messages(view)
        assert len(engine._identity_anchor_extend_chunk([retained], [retained, *_turn("C20", 2000.0)])) > 1
        _, trace = _passes(engine, view, 1, lines)
        assert not any(t["reason"] == "selected leaf chunk lacks raw store lineage" for t in trace), trace
    finally:
        engine.shutdown()


def test_t6r_unrelated_rebind_does_not_see_memo(tmp_path, summaries, lines):
    from tests.test_host_uid_shadow import _state_db
    engine, view = _reorder(tmp_path, False)
    try:
        view, _ = _passes(engine, view, 2, lines)
        retained = view[-1]
        assert engine._identity_anchor_covered_view(retained, {})
        _state_db(tmp_path, [("S", None, "compression"), ("unrelated", None, None)])
        engine.on_session_start("unrelated", platform="cli", context_length=200_000, conversation_id="conv")
        assert not engine._retained_escape_memo()
        assert not engine._identity_anchor_covered_view(retained, {})
    finally:
        engine.shutdown()


@pytest.mark.parametrize("reverse", [False, True], ids=["retained_result", "retained_call"])
def test_t7_cross_host_tool_group(tmp_path, summaries, lines, reverse):
    engine = _engine(tmp_path, False)
    call, result = _turn("tool", 100.0, tool=True)[1:3]
    prefix = {"role": "user", "content": "prefix" + PAD}
    try:
        engine.ingest([prefix, call, result])
        stored = _rows(engine)
        mapping = {id(m): int(r["store_id"]) for m, r in zip([prefix, call, result], stored)}
        chunk = [result] if reverse else [call]
        out = engine._identity_anchor_summary_input(chunk, mapping, [prefix, call, result], chunk, budget=10000)
        # Normal walk emits a prefix first; explicitly exercise the fallback on
        # the cross-group case, at the frontier immediately before that group.
        out = engine._store_complete_input(chunk, {}, mapping, [call, result], chunk,
                                            mapping[id(prefix)], [], 10000, escape_retained=True)
        assert out == []
        assert not engine._store_complete_escaped
    finally:
        engine.shutdown()


@pytest.mark.parametrize("reverse", [False, True], ids=["retained_result", "retained_call"])
def test_b7_padded_ids_keep_escape_group_whole(tmp_path, summaries, lines, reverse):
    from tests.test_host_uid_shadow import _state_db
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=1)
    user, call, result, reply = _turn("X", 100.0, tool=True)
    call["tool_calls"][0]["id"] = result["tool_call_id"] = " a "
    stored = [*_turn("T0", 50.0), user, call, result, reply,
              *[row for i in range(2, 6) for row in _turn(f"T{i}", 100.0 * i)]]
    try:
        engine.ingest(stored)
        before = _rows(engine)
        mapping = {id(m): int(r["store_id"]) for m, r in zip(stored, before)}
        chunk = [result] if reverse else [call]
        # Exercise the escape itself at the group boundary, in both directions.
        out = engine._store_complete_input(chunk, {}, mapping, [call, result], chunk,
                                          mapping[id(call)] - 1, [], 10000, escape_retained=True)
        assert out == []  # neither half may be consumed while the host retains the other
        assert not engine._store_complete_escaped
        moved = call if reverse else result
        view = [m for m in stored if m is not moved] + [dict(moved)]
        engine._ingest_cursor = len(view)
        for _ in range(4):
            view, _ = _passes(engine, view, 1, lines)
            for text in summaries:
                if "[Tool calls:" in text:
                    # Base may repair a moved result with a stub; check the actual
                    # leaf input, rather than requiring that stub's stored ID.
                    assert "[TOOL RESULT a]:" in text
            calls = Counter(str(c.get("id") or "").strip()
                            for m in view for c in m.get("tool_calls") or ())
            results = Counter(str(m.get("tool_call_id") or "").strip()
                              for m in view if m.get("role") == "tool")
            assert calls == results  # no orphaned call or result in assembled context
            assert all(n == 1 for n in results.values())
        assert _rows(engine) == before
        _audit(engine)
    finally:
        engine.shutdown()


def test_b7_padded_result_cuts_at_frontier_call(tmp_path):
    from tests.test_host_uid_shadow import _state_db
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=1)
    user, call, result, reply = _turn("X", 100.0, tool=True)
    call["tool_calls"][0]["id"] = result["tool_call_id"] = " a "
    stored = [*_turn("T0", 50.0), user, call, result, reply]
    try:
        engine.ingest(stored)
        mapping = {id(m): int(r["store_id"]) for m, r in zip(stored, _rows(engine))}
        # The call sits at the frontier, so only the result side of group_cut can stop the chunk.
        out = engine._store_complete_input([result], {}, mapping, [call, result], [result],
                                          mapping[id(call)], [], 10000, escape_retained=True)
        assert out == []
    finally:
        engine.shutdown()


def test_t8_ordinary_cleanup_control(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False, fresh_tail_count=2, leaf_chunk_tokens=4000)
    view = [{"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<think>internal</think>Visible answer"}]
    try:
        out = engine.compress(view)
        assert out is not view and len(out) == len(view)
        assert out[-1]["content"] == "Visible answer"
        assert engine._last_compression_status == "sanitized"
        assert engine._last_compression_noop_reason == "" and not summaries
    finally:
        engine.shutdown()


def test_t9_kill_switch(tmp_path, summaries, lines):
    engine, view = _reorder(tmp_path, False)
    engine._retained_escape_enabled = False
    try:
        _, trace = _passes(engine, view, 3, lines)
        assert trace[-1]["leaves"] == 0 and _frontier(engine) < _r_id(engine)
        assert "kind:shown" in trace[-1]["line"] and " escaped=" not in trace[-1]["line"]
    finally:
        engine.shutdown()


def test_t12_contiguity_budget_lift(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    rows = [{"role": "user", "content": f"row {i}" + PAD} for i in range(3)]
    try:
        engine.ingest(rows)
        stored = _rows(engine)
        chunk = [rows[2], rows[0]]
        mapping = {id(r): int(s["store_id"]) for r, s in zip(rows, stored) if r in chunk}
        out = engine._store_complete_input(chunk, {}, mapping, chunk, chunk, 0, [], 1,
                                            scanned=(stored, False), allow_retained_escape=True)
        assert out and engine._store_complete_blocked is None
        def ids_of(pairs):
            return [sid for m, ids in pairs for sid in [*ids, *([mapping[id(m)]] if id(m) in mapping else [])]]
        # #2007: the list keeps the walk's order; the summarizer reads it in store order, not [row 2, row 0].
        assert ids_of(out) == _walk_ids(rows, chunk)
        assert ids_of(_in_store_order(engine, out)) == [1, 2, 3]
    finally:
        engine.shutdown()


def _leaf_ids(engine, out) -> list:
    full_map = engine._compress_run.store_ids_by_message_id or {}
    return [sid for m, ids in out for sid in [*ids, *([full_map[id(m)]] if id(m) in full_map else [])]]


def _walk_ids(stored, chunk) -> list:
    """The walk's (main's) order for these leaves: the chunk rows in host order, each hidden row emitted
    before the first chunk row stored after it."""
    host = [next(sid for sid, row in enumerate(stored, start=1) if row is m) for m in chunk]
    hidden, out = [sid for sid in range(1, len(stored) + 1) if sid not in host], []
    for sid in host:
        out += [h for h in hidden if h < sid and h not in out] + [sid]
    return out


def _in_store_order(engine, out) -> list:
    """#2007: the order the summarizer reads ``out`` in (the rank of a host-reordered leaf)."""
    rank = engine._store_complete_order
    return sorted(out, key=lambda pair: rank[id(pair[0])]) if rank else list(out)


def test_2007_identity_anchor_input_in_store_order(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    rows = [{"role": "user", "content": f"row {i}" + PAD, "timestamp": 10.0 * (i + 1)} for i in range(3)]
    try:
        engine.ingest(rows)
        view = [rows[2], rows[0]]  # the host moved row 0 after row 2; row 1 is hidden
        full_map = engine._get_store_id_map_for_messages(view)
        engine._compress_run.store_ids_by_message_id = full_map
        out = engine._identity_anchor_summary_input(view, full_map, view, view, budget=4000)
        assert out and _leaf_ids(engine, out) == _walk_ids(rows, view)
        assert _leaf_ids(engine, _in_store_order(engine, out)) == [1, 2, 3]
        assert [m["content"] for m, _ids in _in_store_order(engine, out)] == [r["content"] for r in rows]
    finally:
        engine.shutdown()


def test_2007_reordered_tool_group_stays_whole_in_store_order(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    user, call, result, reply = _turn("X", 100.0, tool=True)
    later = {"role": "user", "content": "later" + PAD, "timestamp": 200.0}
    try:
        engine.ingest([user, call, result, reply, later])
        view = [user, later, call, result, reply]  # the host moved the tool group after a later row
        full_map = engine._get_store_id_map_for_messages(view)
        engine._compress_run.store_ids_by_message_id = full_map
        out = engine._identity_anchor_summary_input(view, full_map, view, view, budget=4000)
        assert out and _leaf_ids(engine, out) == _walk_ids([user, call, result, reply, later], view)
        out = _in_store_order(engine, out)
        assert _leaf_ids(engine, out) == [1, 2, 3, 4, 5]
        roles = [m.get("role") for m, _ids in out]
        assert roles[1:3] == ["assistant", "tool"] and out[1][0].get("tool_calls")
        assert out[2][0]["tool_call_id"] == out[1][0]["tool_calls"][0]["id"]
    finally:
        engine.shutdown()


def _anchored_ids(engine, stored, view) -> list:
    engine.ingest(stored)
    full_map = engine._get_store_id_map_for_messages(view)
    engine._compress_run.store_ids_by_message_id = full_map
    out = engine._identity_anchor_summary_input(view, full_map, view, view, budget=4000)
    assert out and _leaf_ids(engine, out) == _walk_ids(stored, view)  # #2007 r5: the list keeps the walk's order
    return _leaf_ids(engine, _in_store_order(engine, out))  # the order the summarizer reads


def test_2007_swapped_sibling_results_in_store_order(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    user = {"role": "user", "content": "user" + PAD, "timestamp": 100.0}
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{k}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}} for k in "AB"]}
    result_a, result_b = ({"role": "tool", "tool_call_id": f"call_{k}", "content": f"result {k}" + PAD} for k in "AB")
    reply = {"role": "assistant", "content": "reply" + PAD}
    try:  # the host swapped the call's two results
        assert _anchored_ids(engine, [user, call, result_a, result_b, reply],
                             [user, call, result_b, result_a, reply]) == [1, 2, 3, 4, 5]
    finally:
        engine.shutdown()


def test_2007_hidden_result_follows_its_reordered_call(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    user, call, result, _reply = _turn("X", 100.0, tool=True)
    answer = {"role": "assistant", "content": "answer" + PAD}
    later = {"role": "user", "content": "later" + PAD, "timestamp": 200.0}
    try:  # stored u1 a2 call3 result4 u5; the host hides the result and moves the call last
        assert _anchored_ids(engine, [user, answer, call, result, later], [user, answer, later, call]) == [1, 2, 3, 4, 5]
    finally:
        engine.shutdown()


def test_2007_result_of_a_hidden_call_keeps_store_order(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    user, call, result, _reply = _turn("X", 100.0, tool=True)
    later = {"role": "user", "content": "later" + PAD, "timestamp": 200.0}
    try:  # stored u1 call2 result3 u4; the host hides the call and moves the result last
        assert _anchored_ids(engine, [user, call, result, later], [user, later, result]) == [1, 2, 3, 4]
    finally:
        engine.shutdown()


def _call(*names, content=""):
    return {"role": "assistant", "content": content, "tool_calls": [
        {"id": name, "type": "function", "function": {"name": "read_file", "arguments": "{}"}} for name in names]}


def _result(name, text):
    return {"role": "tool", "tool_call_id": name, "content": text + PAD}


def test_2007_unmatched_tool_row_between_siblings(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    user = {"role": "user", "content": "user" + PAD, "timestamp": 100.0}
    call, result_a, result_b = _call("call_A", "call_B"), _result("call_A", "result A"), _result("call_B", "result B")
    stray = {"role": "tool", "content": "tool row without a call id" + PAD}
    try:  # stored u1 call2 A3 stray4 B5; the host puts B first and the stray row between B and A
        assert _anchored_ids(engine, [user, call, result_a, stray, result_b],
                             [user, call, result_b, stray, result_a]) == [1, 2, 3, 5, 4]
    finally:
        engine.shutdown()


def _walk_order(engine, stored, chunk, mapping, claims):
    engine.ingest(stored)
    full = {id(m): s for m, s in mapping}
    out = engine._store_complete_input(chunk, {id(m): c for m, c in claims}, full, chunk, chunk, 0, [],
                                       100_000, scanned=(_rows(engine), False))
    assert out and [m for m, _ids in out] == chunk  # #2007 r5: no hidden row, so the walk keeps the chunk order
    return [m for m, _ids in _in_store_order(engine, out)]


@pytest.mark.parametrize("shape", ["G", "H"])
def test_2007_result_never_precedes_an_id_less_call(tmp_path, summaries, lines, shape):
    # A regression guard: a call row without a store id (position 0) stays ahead of its results.
    engine = _engine(tmp_path, False)
    try:
        if shape == "G":  # out [U2, C(no id, c), X3, R1->c]
            result, user, later = _result("c", "result c"), {"role": "user", "content": "U" + PAD}, \
                {"role": "user", "content": "X" + PAD}
            call = _call("c", content="unstored call")
            order = _walk_order(engine, [result, user, later], [user, call, later, result],
                                [(user, 2), (later, 3)], [(result, [1])])
            pairs = [(call, result)]
        else:  # out [C1(a), Ra3, C(no id, b), Rb2->b]
            call_a, result_b, result_a = _call("a"), _result("b", "result b"), _result("a", "result a")
            call_b = _call("b", content="unstored call")
            order = _walk_order(engine, [call_a, result_b, result_a], [call_a, result_a, call_b, result_b],
                                [(call_a, 1), (result_a, 3)], [(result_b, [2])])
            pairs = [(call_a, result_a), (call_b, result_b)]
        position = {id(m): i for i, m in enumerate(order)}
        assert all(position[id(c)] < position[id(r)] for c, r in pairs), [m["content"][:12] for m in order]
    finally:
        engine.shutdown()


def test_2007_repeated_call_id_answers_the_latest_call(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    first = {"role": "user", "content": "first" + PAD, "timestamp": 100.0}
    second = {"role": "user", "content": "second" + PAD, "timestamp": 200.0}
    call_1, result_1 = _call("call_0", content="first call"), _result("call_0", "first result")
    call_2, result_2 = _call("call_0", content="second call"), _result("call_0", "second result")
    try:  # the host moves the first turn's tool group after the second turn's
        assert _anchored_ids(engine, [first, call_1, result_1, second, call_2, result_2],
                             [first, second, call_2, result_2, call_1, result_1]) == [1, 2, 3, 4, 5, 6]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["hidden_results", "moved_result"])
def test_2007_repeated_call_id_follows_the_store(tmp_path, summaries, lines, shape):
    # #2011: a result of a reused call id joins the call stored nearest before it, not the walk's nearest.
    engine = _engine(tmp_path, False)
    call_1, result_1 = _call("call_0", content="first call"), _result("call_0", "first result")
    call_2, result_2 = _call("call_0", content="second call"), _result("call_0", "second result")
    later = {"role": "user", "content": "later" + PAD, "timestamp": 500.0}
    try:
        if shape == "hidden_results":  # stored C1 R1 C2 R2 U5; the host shows C2 C1 U5 (both results hidden)
            ids = _anchored_ids(engine, [call_1, result_1, call_2, result_2, later], [call_2, call_1, later])
        else:  # stored C1 R2 C3 R4; the host shows C1 C3 R4 R2
            ids = _anchored_ids(engine, [call_1, result_1, call_2, result_2], [call_1, call_2, result_2, result_1])
        assert ids == sorted(ids)
    finally:
        engine.shutdown()


def test_2007_in_order_tool_shapes_returned_as_built(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    user = {"role": "user", "content": "user" + PAD, "timestamp": 100.0}
    hidden = {"role": "user", "content": "hidden" + PAD, "timestamp": 150.0}
    call, result_a, result_b = _call("call_A", "call_B"), _result("call_A", "result A"), _result("call_B", "result B")
    stray = {"role": "tool", "content": "tool row without a call id" + PAD}
    again, result_c = _call("call_A", content="again"), _result("call_A", "result again")
    stored = [user, hidden, call, result_a, result_b, stray, again, result_c]
    view = [m for m in stored if m is not hidden]
    try:
        engine.ingest(stored)
        full_map = engine._get_store_id_map_for_messages(view)
        engine._compress_run.store_ids_by_message_id = full_map
        out = engine._identity_anchor_summary_input(view, full_map, view, view, budget=4000)
        assert _leaf_ids(engine, out) == list(range(1, len(stored) + 1))
        assert engine._store_complete_order == {}  # #2007 r5: an in-order leaf gets no rank
        assert [m for m, _ids in out if any(m is v for v in view)] == view
        assert [m["content"] for m, _ids in out] == [m["content"] for m in stored]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("shape", ["row_inside_group", "two_calls_then_results"])
def test_2007_store_ordered_split_group_returned_as_built(tmp_path, summaries, lines, shape):
    # F4 guard: a leaf already in store order keeps today's input, even when the store holds a row
    # between a call and its result; regrouping it would move that row out of store order.
    engine = _engine(tmp_path, False)
    if shape == "row_inside_group":  # stored C1(a) U2 Ra3
        rows = [_call("a"), {"role": "user", "content": "between" + PAD}, _result("a", "result a")]
    else:  # stored C1(a) C2(b) Ra3 Rb4
        rows = [_call("a"), _call("b", content="second call"), _result("a", "result a"), _result("b", "result b")]
    try:
        engine.ingest(rows)
        full = {id(m): s for s, m in enumerate(rows, start=1)}
        assert engine._store_complete_input(rows, {}, full, rows, rows, 0, [], 100_000,
                                            scanned=(_rows(engine), False)) is None
        assert engine._store_complete_order == {}
    finally:
        engine.shutdown()


def _leaf_calls(monkeypatch, engine, fail_first: bool):
    """The leaf summariser texts (a stub whose first call can fail retry-worthy) and each leaf's consumed ids."""
    texts, consumed = [], []

    def summarize(**kwargs):
        if kwargs.get("depth", 0) == 0:
            texts.append(kwargs["text"])
            if fail_first and len(texts) == 1:
                raise TimeoutError("request timed out")  # retry-worthy: the rescue retries a smaller prefix
        return "Earlier turns.\nExpand for details about: turns", 1

    original = engine._leaf_consumption

    def consumption(*args):
        result = original(*args)
        consumed.append(result[4])
        return result

    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summarize)
    monkeypatch.setattr(engine, "_leaf_consumption", consumption)
    return texts, consumed


def _read_ids(text, stored) -> list:
    """The store ids of the rows a summariser text holds, in the order it holds them."""
    marks = [(sid, row["content"].split(PAD)[0]) for sid, row in enumerate(stored, start=1)]
    return [sid for _at, sid in sorted((text.index(mark), sid) for sid, mark in marks if mark in text)]


def test_2007_rescue_prefix_consumes_only_rows_it_read(tmp_path, lines, monkeypatch):
    # B1: the host holds [B(2), A(1)]; the first call fails retry-worthy and the rescue retries one row.
    engine = _engine(tmp_path, False)
    a = {"role": "user", "content": "row A" + PAD, "timestamp": 10.0}
    b = {"role": "user", "content": "row B" + PAD, "timestamp": 20.0}
    stored = [a, b, *[row for i in (2, 3) for row in _turn(f"T{i}", 100.0 * i)]]
    view = [b, a, *stored[2:]]
    texts, consumed = _leaf_calls(monkeypatch, engine, fail_first=True)
    try:
        engine.ingest(stored)
        engine._ingest_cursor = len(view)
        engine.compress(view, current_tokens=10_000)
        assert len(texts) >= 2 and sorted(_read_ids(texts[0], stored)) == [1, 2] and consumed, (texts, consumed)
        read = _read_ids(texts[1], stored)
        assert len(read) == 1 and consumed[0] == sorted(read)  # no row the summary never read is consumed
        assert read == sorted(read)  # store order
    finally:
        engine.shutdown()


def _hidden_row_and_tool_group():
    user = {"role": "user", "content": "user 1" + PAD, "timestamp": 100.0}
    hidden = {"role": "user", "content": "hidden 2" + PAD, "timestamp": 150.0}
    call, result = _call("call_X", content="call 3"), _result("call_X", "result 4")
    reply = {"role": "assistant", "content": "reply 5" + PAD}
    later = {"role": "user", "content": "later 6" + PAD, "timestamp": 200.0}
    return [user, hidden, call, result, reply, later, *[row for i in (3, 4) for row in _turn(f"T{i}", 100.0 * i)]]


@pytest.mark.parametrize("rescue", [False, True], ids=["full", "rescue"])
def test_2007_summariser_reads_a_reordered_leaf_in_store_order(tmp_path, lines, monkeypatch, rescue):
    engine = _engine(tmp_path, False, leaf_chunk_tokens=4000, dynamic_leaf_chunk_max=4000)
    stored = _hidden_row_and_tool_group()
    view = [stored[0], stored[5], *stored[2:5], *stored[6:]]  # row 2 hidden; row 6 moved before the tool group
    texts, _consumed = _leaf_calls(monkeypatch, engine, fail_first=rescue)
    try:
        engine.ingest(stored)
        engine._ingest_cursor = len(view)
        engine.compress(view, current_tokens=10_000)
        read = [_read_ids(text, stored) for text in texts]
        assert read and {1, 2, 3, 4, 5, 6} <= set(read[0]) and read[0] == sorted(read[0]), read
        if rescue:  # the rescue prefix of the walk, still read in store order (the call before its result)
            assert len(read) >= 2 and {3, 4, 6} <= set(read[1]) < set(read[0]), read
            assert read[1] == sorted(read[1]), read
    finally:
        engine.shutdown()


def test_2007_in_order_leaf_serialized_as_given(tmp_path, lines, monkeypatch):
    engine = _engine(tmp_path, False, leaf_chunk_tokens=4000, dynamic_leaf_chunk_max=4000)
    stored = _hidden_row_and_tool_group()
    view = [row for i, row in enumerate(stored) if i != 1]  # row 2 hidden, the rest in store order
    texts, _consumed = _leaf_calls(monkeypatch, engine, fail_first=False)
    given, original = [], engine._summarize_leaf_chunk_with_rescue

    def summarise(chunk, **kwargs):
        given.append(engine._serialize_messages_with_clip(list(chunk))[0])
        return original(chunk, **kwargs)

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarise)
    try:
        engine.ingest(stored)
        engine._ingest_cursor = len(view)
        engine.compress(view, current_tokens=10_000)
        assert texts and given and texts[0] == given[0]  # main's serialization: the input as given, byte for byte
        assert {1, 2, 3, 4, 5, 6} <= set(_read_ids(texts[0], stored))
    finally:
        engine.shutdown()


def test_t11_prepared_escape_promotes(tmp_path, summaries, lines, monkeypatch):
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=1000,
                     context_threshold=0.75, background_leaf_prep_enabled=True)
    retained, a, b, tail = [{"role": "user", "content": name + PAD}
                             for name in ("retained", "a", "b", "tail")]
    carrier = {"role": "user", "content": "unmapped carrier barrier"}
    try:
        engine.ingest([retained, a, b, tail])
        stored = _rows(engine)
        mapping = {id(m): int(r["store_id"]) for m, r in zip([a, b, tail], stored[1:])}
        view = [a, b, carrier, retained, tail]
        engine._compress_run.store_ids_by_message_id = mapping
        covered = tuple(int(r["store_id"]) for r in stored[:3])
        prepared = leaf_prep.PreparedLeaf(
            "S", "conv", 0, covered,
            leaf_prep.coverage_digest([(r["store_id"], r["content"]) for r in stored[:3]]),
            leaf_prep.route_fingerprint(engine._config), "Earlier turns.\nExpand for details about: turns",
            1, "stub", count_messages_tokens([retained, a, b]), 20)
        monkeypatch.setattr(leaf_prep, "ready_chain", lambda *_args: [prepared])
        engine._store_complete_escaped = [(999, "user", "shown")]
        matched = engine._match_prepared_leaf([a, b, carrier], view, [], set(), 200_000,
                                               count_messages_tokens([a, b, carrier]))
        assert matched and matched[0] is prepared
        assert engine._store_complete_escaped == [(999, "user", "shown")]
        state = _CompressionState(working_messages=view, budget=engine._new_foreground_budget())
        from hermes_lcm.compaction import _LeafState
        leaf = _LeafState(to_compact=matched[1], prepared=prepared)
        engine._compress_escaped = []
        transfer, _ = engine._prepare_leaf_input(state, leaf, set(), lambda *_args: 0)
        assert transfer == "next" and leaf.prepared is prepared
        assert engine._store_complete_escaped == [(1, "user", "shown")]
        assert engine._prepared_leaf_summary(prepared)(leaf.summary_input_chunk)[2] == prepared.summary
        assert not any("host_unit_split" in s for _, s in lines)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("moved", ["result", "pair"])
def test_t7_returned_tool_groups(tmp_path, summaries, lines, moved):
    from tests.test_host_uid_shadow import _state_db
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=1)
    user, call, result, reply = _turn("X", 100.0, tool=True)
    stored = [*_turn("T0", 50.0), user, call, result, reply,
              *[row for i in range(2, 6) for row in _turn(f"T{i}", 100.0 * i)]]
    try:
        engine.ingest(stored)
        moved_rows = [result] if moved == "result" else [call, result]
        view = [m for m in stored if all(m is not r for r in moved_rows)] + [dict(r) for r in moved_rows]
        engine._ingest_cursor = len(view)
        for _ in range(4):
            view, _ = _passes(engine, view, 1, lines)
            calls = Counter(c["id"] for m in view for c in m.get("tool_calls") or ())
            results = Counter(m["tool_call_id"] for m in view if m.get("role") == "tool")
            assert calls == results  # each call/result pair is both present or both absent
            assert all(n == 1 for n in results.values())  # no orphaned or doubled result
        _audit(engine)
    finally:
        engine.shutdown()


def test_t12b_contiguity_lift_reaches_retained_row(tmp_path, summaries, lines):
    engine = _engine(tmp_path, False)
    big = " lorem ipsum dolor sit amet" * 400
    c1, c2 = ({"role": "user", "content": n + PAD} for n in ("c1", "c2"))
    r2, r3, r5 = ({"role": "user", "content": n + big} for n in ("r2", "r3", "r5"))
    retained = {"role": "user", "content": "R" + PAD}
    try:
        engine.ingest([c1, r2, r3, retained, r5, c2])
        ids = [int(r["store_id"]) for r in _rows(engine)]
        mapping = {id(c1): ids[0], id(c2): ids[5]}
        chunk, view = [c2, c1], [c2, c1, dict(retained)]
        for _ in range(2):
            out = engine._store_complete_input(chunk, {}, mapping, view, chunk, 0, [],
                                                count_messages_tokens(chunk), allow_retained_escape=True)
            if out:
                break
        assert out
        assert (ids[3], "user", "shown") in engine._store_complete_escaped
        assert ids[3] in [sid for _m, claims in out for sid in claims]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("payload", ['{not json', '[{"store_id": 1}]', '["abc"]',
                                     '[{"store_id":"x","identity_sha256":"d"}]', None, "unreadable"])
def test_b4_corrupt_memo_fails_closed(tmp_path, summaries, lines, monkeypatch, payload):
    engine, view = _reorder(tmp_path, False)
    try:
        key = engine._replay_snapshot_metadata_key("retained_escape")
        if payload not in (None, "unreadable"):
            with engine._store._write_lock:
                engine._store._conn.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, payload))
                engine._store._conn.commit()
        engine._retained_escape_memos = {}
        with monkeypatch.context() as patch:
            if payload == "unreadable":
                read = engine._store.read_metadata_json
                def unreadable(k, *args, **kwargs):
                    if k == key:
                        raise OSError("synthetic metadata read failure")
                    return read(k, *args, **kwargs)
                patch.setattr(engine._store, "read_metadata_json", unreadable)
            assert not engine._retained_escape_covered_view({"role": "user", "content": "unknown"})
            assert engine._retained_escape_memo() == {}
        _, trace = _passes(engine, view, 3, lines)
        assert all(t["status"] != "error" for t in trace)
        memo = engine._store.read_metadata_json(key)
        assert memo and all(isinstance(i["identity_sha256"], str) and int(i["store_id"]) > 0 for i in memo)
        _audit(engine)
    finally:
        engine.shutdown()


def test_b4_memo_write_failure_keeps_committed_pass(tmp_path, summaries, lines, monkeypatch):
    engine, view = _reorder(tmp_path, False)
    try:
        key = engine._replay_snapshot_metadata_key("retained_escape")
        write = engine._store.write_metadata_json
        def unavailable(keys, *args, **kwargs):
            if key in keys:
                raise OSError("synthetic metadata write failure")
            return write(keys, *args, **kwargs)
        monkeypatch.setattr(engine._store, "write_metadata_json", unavailable)
        _, trace = _passes(engine, view, 2, lines)
        assert trace[-1]["leaves"] >= 1 and trace[-1]["status"] == "compacted"
        assert _covered(engine)[_r_id(engine)] == 1
        _audit(engine)
    finally:
        engine.shutdown()


@pytest.mark.parametrize("rewrite", [False, True])
def test_b4_memo_matches_shown_host_rewrite(tmp_path, summaries, lines, rewrite):
    from tests.test_host_uid_shadow import _state_db
    _state_db(tmp_path, [("S", None, None)])
    engine = _engine(tmp_path, False, fresh_tail_count=1, leaf_chunk_tokens=1)
    retained = {"role": "user", "content": "[R] retained occurrence" + PAD + ("\n\n" if rewrite else ""),
                "message_uid": "retained-r"}
    stored = [*_turn("T1", 100.0), retained,
              *[row for i in range(2, 6) for row in _turn(f"T{i}", 100.0 * i)]]
    try:
        engine.ingest(stored)
        sid = _r_id(engine)
        row = next(r for r in _rows(engine) if int(r["store_id"]) == sid)
        live = dict(retained)
        if rewrite:
            live["content"] = retained["content"].rstrip()
            payload = {"version": 1, "content": live["content"], "stripped": ["", "\n\n"],
                       "stored_sha256": hashlib.sha256(row["content"].encode()).hexdigest()}
            engine._store.write_metadata_json([f"host_rewrite_identity:{sid}"], json.dumps(payload, sort_keys=True))
            engine._host_rewrite_state()[1].pop(sid, None)
        view = [m for m in stored if m is not retained] + [live]
        engine._ingest_cursor = len(view)
        assert engine._message_replay_identity(live, strip_carrier=False) in engine._stored_row_forms(row)
        view, trace = _passes(engine, view, 2, lines)
        assert any(" escaped=" in t["line"] for t in trace)
        assert engine._identity_anchor_covered_view(view[-1], {})
        engine._retained_escape_memos = {}  # prove the persisted forms, too
        assert engine._retained_escape_covered_view(view[-1])
        assert not _conflicts(lines)
    finally:
        engine.shutdown()


def test_b4_memo_cap_counts_rows_not_forms(tmp_path, summaries, lines, monkeypatch):
    engine = _engine(tmp_path, False)
    try:
        engine.ingest([{"role": "user", "content": f"row {i}"} for i in range(257)])
        rows = _rows(engine)
        original = engine._stored_row_forms
        def forms(row):
            form = next(iter(original(row)))
            return {form, (form[0], form[1] + " alternate", *form[2:])}
        monkeypatch.setattr(engine, "_stored_row_forms", forms)
        engine._record_retained_escape([(int(r["store_id"]), "user", "shown") for r in rows])
        memo = engine._retained_escape_memo()
        assert len(set(memo.values())) == 256 and len(memo) == 512
        assert int(rows[0]["store_id"]) not in memo.values()
    finally:
        engine.shutdown()


def test_b4_summary_failure_does_not_report_escape(tmp_path, summaries, lines, monkeypatch):
    import hermes_lcm.engine as engine_module
    engine, view = _reorder(tmp_path, False)
    try:
        view, _ = _passes(engine, view, 1, lines)
        def boom(**_kwargs):
            raise RuntimeError("synthetic route down")
        monkeypatch.setattr(engine_module, "summarize_with_escalation", boom)
        for _ in range(2):
            start = len(lines)
            with pytest.raises(RuntimeError, match="synthetic route down"):
                engine.compress(view, current_tokens=10_000)
            emitted = [message for _level, message in lines[start:]]
            stop = next(m for m in emitted if m.startswith("LCM compaction stop:"))
            assert "reason=error leaves=0" in stop and " escaped=" not in stop
            assert not any("retained-occurrence fallback:" in m for m in emitted)
            assert not engine._compress_escaped
    finally:
        engine.shutdown()


@pytest.mark.parametrize("bound", [True, False])
def test_t6b_cold_resume_after_no_leaf_condensation(tmp_path, summaries, lines, bound):
    engine, view = _reorder(tmp_path, True, bound=bound)
    try:
        view, trace = _passes(engine, view, 2, lines)
        if bound:
            assert any(" escaped=" in t["line"] for t in trace)
        else:
            assert _covered(engine)[_r_id(engine)] == 1  # unbound ingest keeps a copy, so R is hidden
        assert trace[-1]["leaves"] == 0 and "progress=condensation" in trace[-1]["line"]
        retained = next(r["content"] for r in _rows(engine) if r["content"].startswith("[R] retained"))
        engine.shutdown()
        engine = _engine(tmp_path, True, fresh_tail_count=1, leaf_chunk_tokens=1)
        engine.ingest(view)
        # A duplicate here is open #1286 (I1 exception #259); base produces the
        # same duplicate for unbound sessions. This cell asserts no loss only.
        assert any(r["content"] == retained for r in _rows(engine))
        _audit(engine)
    finally:
        engine.shutdown()
