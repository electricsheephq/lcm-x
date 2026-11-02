"""P1d: exact ledger prefixes, consume-once consumers, and restart losses."""
from collections import Counter
from copy import deepcopy
import random

import pytest

from hermes_lcm import ingest_ledger as ledger
from hermes_lcm.dag import SummaryNode
from tests.test_host_uid_shadow import _engine, _m
from tests.test_issue_2055_emission_writes import candidate as emission_capture
from tests.test_issue_2055_ingest_ledger_writes import (
    count, engines as _engines, restart, sql_rows,
)

engines = _engines


@pytest.fixture(autouse=True)
def pending(monkeypatch):
    monkeypatch.setattr(ledger, "_PENDING_EMISSIONS", {})


def prepare(engine, view):
    return engine._prepare_ingest_ledger(view, {"ledger_digest_ms": 0.})


def candidate(engine, view, *, ignored=None, attempt=None):
    attempt = attempt or prepare(engine, view)
    with engine._store.atomic(immediate=True):
        before = engine._store._conn.total_changes
        result = ledger.cold_cursor_candidate(
            engine._store._conn, session_id="S", attempt=attempt,
            ignored=ignored or [False] * len(view))
        assert engine._store._conn.total_changes == before
    return result


def force_today(engine, monkeypatch, value=0):
    """Isolate the operating seam; leave every post-cursor consumer real."""
    def reconcile(messages, *, session_counts, **kwargs):
        session_counts.append(engine._store.get_session_count("S"))
        engine._last_ingest_reconciliation = {"cursor": value, "reason": "test legacy choice"}
        return value
    monkeypatch.setattr(engine, "_reconcile_ingest_cursor_from_store", reconcile)
    engine._ingest_cursor_needs_reconcile = True


def emit(engine, view, kinds):
    """Write through the real final exit writer, without minting a durable proof."""
    capture = [dict(kind=kind, output_occurrence={"index": i, "of": len(view)}) for i, kind in kinds.items()]
    engine._compress_exit_write(view, (view, capture, None))
    assert engine._emission_handle is not None


def pure_summary(engine, role="assistant"):
    node = SummaryNode(session_id="S", summary="Synthetic earlier facts.", source_ids=[1])
    engine._dag.add_node(node)
    return dict(role=role, content=f"[Recent Summary (d0, node {node.node_id})]\n{node.summary}\n"
                f"[Expand for details: {node.expand_hint}]")


def snapshot(engine):
    return Counter((r["role"], r["content"], r["observed_at"], r.get("tool_call_id"))
                   for r in engine._store.get_session_messages("S"))


def tool_pair(stamp):
    return [_m("assistant", "", stamp, tool_calls=[
        {"id": "reused", "type": "function", "function": {"name": "read", "arguments": "{}"}}]),
        _m("tool", "ok", stamp + 1, tool_call_id="reused")]


def test_t2_zero_today_win_does_not_activate_tool_suppression(engines, monkeypatch):
    engine = engines()
    p = _m("user", "P", 1.)
    engine._ingest_messages([p, *tool_pair(2.)])
    force_today(engine, monkeypatch)
    view = [p, _m("user", "new", 4.), *tool_pair(5.)]
    engine._ingest_messages(deepcopy(view))
    assert count(engine, "cold_candidate.used_ingest") == 1
    assert engine._last_ingest_reconciliation["k"] == 1
    assert all(snapshot(engine)[(m["role"], m["content"], m["timestamp"], m.get("tool_call_id"))] == 1
               for m in view)


def test_t2_tool_multiset_excludes_consumed_ids_not_equal_content(engines):
    engine = engines()
    p, old = _m("user", "P", 1.), tool_pair(2.)
    engine._ingest_messages([p, *old])
    stored = engine._store.get_session_messages("S")
    view = [p, _m("user", "new", 9.), *old]
    helper = engine._replayed_tool_segment_indexes_after_cursor
    assert helper(view, 1) == {2, 3}
    assert helper(view, 1, consumed_store_ids={r["store_id"] for r in stored[1:]}) == set()
    # An out-of-window consumed twin must not cancel an in-window occurrence.
    assert helper(view, 1, consumed_store_ids={999999}) == {2, 3}


def test_t3_consumed_twin_cannot_be_stolen(engines, monkeypatch):
    engine = engines()
    x = _m("user", "x", 10.)
    engine._ingest_messages([x])
    force_today(engine, monkeypatch)
    engine._ingest_messages([x, _m("assistant", "new", 11.), dict(x)])
    assert count(engine, "cold_candidate.used_ingest") == 1
    assert snapshot(engine)[("user", "x", 10., None)] == 2


def test_t3_carried_copy_keeps_fetch_and_local_predecessor(engines, monkeypatch):
    outputs = []
    for enabled in (False, True):
        engine = engines(str(enabled))
        p, r = _m("user", "P", 10.), _m("user", "R", 10.)
        engine._ingest_messages([p, r])
        rows = engine._store.get_session_messages("S")
        engine._store.add_message_relations([[(rows[1]["store_id"], "carried_stamp", rows[0]["store_id"], None, 10.)]])
        # Compress ancestry is required by the real carried-stamp predicate.
        monkeypatch.setattr(engine, "_identity_anchor_chain", lambda: ["ancestor"])
        attempt = prepare(engine, [p])
        with engine._store.atomic(immediate=True):
            ledger.write_view(engine._store._conn, attempt["salt"], attempt["lineage"], "ingest", None,
                              [attempt["entries"][0]])  # ordinary unbound positional prefix
        with monkeypatch.context() as patch:
            force_today(engine, patch)
            if not enabled:
                patch.setattr(engine, "_ledger_cold_cursor", lambda *a: dict(k=0, pairs={}))
            patch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
            engine._ingest_messages([dict(p, message_uid="new-prefix-uid"), _m("user", "R", 20.)])
        if enabled:
            assert count(engine, "cold_candidate.used_ingest") == 1
            assert engine._last_ingest_reconciliation["k"] == 1
        assert len(engine._store.get_session_messages("S")) == 2
        latest = ledger.newest_view(engine._store._conn, attempt["lineage"], "ingest")
        refs = [entry.ref for entry in ledger.decode_entries(latest["entries"])]
        # P has local predecessor proof but remains an ordinary positional prefix.
        if enabled:
            assert refs[0] is None
        has_uids = engine._store._conn.execute("SELECT 1 FROM sqlite_master WHERE name='host_uid_bindings'").fetchone()
        uid_rows = engine._store._conn.execute("SELECT store_id,uid FROM host_uid_bindings").fetchall() if has_uids else []
        if enabled:
            assert (rows[0]["store_id"], "new-prefix-uid") not in uid_rows
        else:
            assert (rows[0]["store_id"], "new-prefix-uid") in uid_rows
        outputs.append((snapshot(engine), [b["ordinal"] for b in sql_rows(engine, "occurrence_bindings")]))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("shape,k,stop", [("bound_system", 2, "descriptor"),
                                          ("unbound_system", 0, "no_anchor"),
                                          ("no_system", 1, "descriptor")])
@pytest.mark.parametrize("today", [0, 2, 4])
def test_t4_compacted_emission_stops_at_descriptor(engines, shape, k, stop, today):
    engine = engines()
    p, omitted, tail = _m("user", "P", 1.), _m("assistant", "omitted", 2.), _m("user", "tail", 3.)
    system = _m("system", engine._append_lcm_note_to_content("System instructions"), .5)
    engine._ingest_messages(([system] if shape == "bound_system" else []) + [p, omitted, tail])
    summary = pure_summary(engine)
    emitted = ([] if shape == "no_system" else [system]) + [p, summary, tail]
    emit(engine, emitted, {k if shape != "unbound_system" else 2: "summary"})
    # Hermes stamps and uid-stamps generated rows when persisting/reloading them.
    view = [dict(row, timestamp=10., message_uid="summary-uid") if row is summary else dict(row)
            for row in emitted]
    run, counters = candidate(engine, view)
    assert run["k"] == k and counters["stop." + stop] == 1
    assert set(run["pairs"]) == set(range(k))
    engine._ingest_ledger_attempt = prepare(engine, view)
    prior = engine._last_ingest_reconciliation = {"cursor": today, "reason": "legacy choice"}
    with engine._store.atomic(immediate=True):
        callbacks = []
        winner = engine._ledger_cold_cursor(view, [False] * len(view), today, 3, callbacks)
        assert max(today, winner["k"]) == max(today, k)
        if today >= k:
            assert engine._last_ingest_reconciliation is prior
        else:
            assert engine._last_ingest_reconciliation["k"] == k
    engine.__dict__.pop("_ingest_ledger_attempt", None)


@pytest.mark.parametrize("kind", ["carrier", "recovery", "projection", "scaffold", "folded", "summary"])
@pytest.mark.parametrize("ignored_descriptor", [False, True])
def test_t5_every_descriptor_stops_before_ignored_rule(engines, kind, ignored_descriptor):
    engine = engines()
    p = _m("user", "P", 1.)
    engine._ingest_messages([p])
    row = pure_summary(engine)
    if kind == "folded":
        row["content"] += "\n\n---\n\nhost tail"
    if kind == "scaffold":
        row["content"] = "[Current user objective preserved from compacted history]\nHost authored objective"
    emit(engine, [p, row], {1: "summary" if kind == "folded" else kind})
    statements = []
    engine._store._conn.set_trace_callback(statements.append)
    try:
        run, counters = candidate(engine, [p, row], ignored=[False, ignored_descriptor])
    finally:
        engine._store._conn.set_trace_callback(None)
    assert run["k"] == 1 and set(run["pairs"]) == {0}
    assert counters["stop.descriptor"] == 1
    assert not any("ledger_emissions" in sql for sql in statements)


@pytest.mark.parametrize("fault", ["refused", "dropped_refusal", "lost_newer", "evicted_newer", "not_received"])
def test_t6_stale_or_undelivered_views_only_prove_prefix(engines, monkeypatch, fault):
    engine = engines()
    p = _m("user", "P", 1.)
    engine._ingest_messages([p])
    old = pure_summary(engine)
    emit(engine, [p, old], {1: "summary"})
    handle = engine._emission_handle
    if fault == "refused":
        with engine._store.atomic(immediate=True):
            ledger.set_emission_state(engine._store._conn, handle.lineage, handle.view_seq, "refused")
    if fault == "evicted_newer":
        for n in range(3):
            ledger.mark_pending_emission((engine._store.db_identity, prepare(engine, [p])["salt"]), handle.lineage,
                                         origin=engine, rows=[], synthetic=n)
        ledger._PENDING_EMISSIONS.clear()  # process loss after newest-two eviction
    # Host refused, received a newer undurable summary, or never received output.
    view = [p, _m("user", "actual new occurrence", 99.)]
    restart(engine)
    force_today(engine, monkeypatch)
    engine._ingest_messages(view)
    assert engine._last_ingest_reconciliation["k"] <= 1
    assert snapshot(engine)[("user", "actual new occurrence", 99., None)] == 1


class RollbackProxy:
    def __init__(self, conn, *, exception=RuntimeError):
        self.conn, self.exception = conn, exception

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def rollback(self):
        self.conn.rollback()
        raise self.exception("rollback sentinel")


class CommitInterruptProxy:
    def __init__(self, conn):
        self.conn, self.pending = conn, True

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def commit(self):
        self.conn.commit()
        if self.pending:
            self.pending = False
            raise KeyboardInterrupt("after COMMIT")


def test_t7_interrupt_survives_rollback_failure_and_next_ingest(engines, monkeypatch):
    engine = engines()
    engine._ingest_messages([_m("user", "P", 1.)])
    conn, write = engine._store._conn, engine._write_ingest_ledger
    engine._store._conn = RollbackProxy(conn)
    with monkeypatch.context() as patch:
        def interrupt(*a):
            raise KeyboardInterrupt("original interrupt")
        patch.setattr(engine, "_write_ingest_ledger", interrupt)
        with pytest.raises(KeyboardInterrupt, match="original interrupt"):
            engine._ingest_messages([_m("user", "P", 1.), _m("assistant", "reply", 2.)])
    assert count(engine, "rollback_failed") == 0  # Deferred telemetry survives to the next successful commit.
    assert engine._store._transaction_owner.rollback_failures == 1
    assert not conn.in_transaction
    engine._store._conn = conn
    assert engine._write_ingest_ledger == write
    engine._ingest_messages([_m("user", "P", 1.), _m("assistant", "reply", 2.)])
    assert count(engine, "rollback_failed") == 1
    assert len(engine._store.get_session_messages("S")) == 2


def test_t7_rollback_baseexception_still_propagates(engines):
    engine = engines()
    proxy = RollbackProxy(engine._store._conn, exception=SystemExit)
    with pytest.raises(SystemExit, match="rollback sentinel"):
        with engine._store._transaction_owner.atomic(proxy, immediate=True):
            raise KeyboardInterrupt("original")
    assert engine._store._transaction_owner.rollback_failures == 0


def test_t8_no_floor_withholds_only_emission_and_cold_start_records_floor(engines, monkeypatch):
    engine = engines()
    view = [_m("user", "P", 1.)]
    engine._ingest_messages(view)
    emit(engine, [*view, pure_summary(engine)], {1: "summary"})
    with engine._store.atomic(immediate=True):
        engine._store._conn.execute("DELETE FROM ledger_meta WHERE key='emission_provenance_floor'")
    run, counts = candidate(engine, view)
    assert run["k"] == 1 and run["kind"] == "ingest"
    assert counts["emission_no_floor_key"] == 1
    restart(engine)
    assert engine._store._conn.execute("SELECT value FROM ledger_meta WHERE key='emission_provenance_floor'").fetchone()


def test_t9_query_plans_use_indexes_without_sort(engines):
    conn = engines()._store._conn
    for sql, params, index in (
        (ledger._COLD_WINDOW, ("L", 0, 200), "idx_occ_lineage_ordinal"),
        (ledger._COLD_ANCHOR, ("L", b"x" * 16, 200), "idx_occ_lineage_digest"),
    ):
        plan = "\n".join(r[-1] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params))
        assert index in plan and "USE TEMP B-TREE" not in plan
        assert "SCAN occurrence_bindings" not in plan


def test_t11_ignored_and_system_rows_keep_full_index_space(engines):
    engine = engines()
    p = _m("user", "P", 2.)
    engine._ingest_messages([p])
    view = [_m("tool", "ignored", 1.),
            _m("system", engine._append_lcm_note_to_content("System instructions")), p]
    attempt = prepare(engine, view)
    with engine._store.atomic(immediate=True):
        ledger.write_view(engine._store._conn, attempt["salt"], attempt["lineage"], "ingest", None, attempt["entries"])
    run, counters = candidate(engine, view, ignored=[True, False, False])
    assert run["k"] == 1 and run["pairs"] == {}
    assert counters["stop.no_anchor"] == 1


def test_t12_persisted_lineage_wins_over_new_resolved_key(engines, monkeypatch):
    engine = engines()
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_host_uid_lineage_key", lambda *a, **kw: (None, "unresolved"))
        engine._ingest_messages([_m("user", "head", 1.)])
    view = [_m("user", "head", 1.)]
    assert prepare(engine, view)["lineage"] != "provisional:S"
    assert candidate(engine, view)[0]["k"] == 0  # No persisted key: do not guess the provisional head.
    with engine._store.atomic(immediate=True):
        engine._store._conn.execute("INSERT INTO ledger_meta VALUES ('session_lineage:S','provisional:S')")
    assert candidate(engine, view)[0]["k"] == 1
    assert candidate(engine, [_m("user", "no anchor", 2.)])[0]["k"] == 0


def test_t13_ties_keep_legacy_record_and_ingest_wins_record_tie(engines):
    engine = engines()
    view = [_m("user", "P", 1.), _m("assistant", "A", 2.)]
    engine._ingest_messages(view)
    attempt = prepare(engine, view)
    with engine._store.atomic(immediate=True):
        ledger.write_view(engine._store._conn, attempt["salt"], attempt["lineage"], "emission", "proposed", attempt["entries"])
        winner, _ = candidate(engine, view)
        assert winner["kind"] == "ingest"
        engine._ingest_ledger_attempt = attempt
        prior = engine._last_ingest_reconciliation = {"reason": "keep me"}
        callbacks = []
        assert engine._ledger_cold_cursor(view, [False] * 2, 2, 2, callbacks)["k"] == 0
        assert engine._last_ingest_reconciliation is prior
    engine.__dict__.pop("_ingest_ledger_attempt", None)


@pytest.mark.parametrize("reason", ["no_attempt", "not_ready", "read_only", "salt_changed", "not_owner", "empty_session", "complete", "length_mismatch"])
def test_t14_admission_does_not_read_ledger_tables(engines, monkeypatch, reason):
    engine = engines()
    view = [_m("user", "P", 1.)]
    engine._ingest_messages(view)
    engine._ingest_ledger_attempt = prepare(engine, view)
    if reason == "no_attempt":
        engine._ingest_ledger_attempt = None
    elif reason == "not_ready":
        monkeypatch.setattr(engine._store, "ingest_ledger_state", lambda: "absent")
    elif reason == "read_only":
        engine._ingest_read_only = RuntimeError("frozen")
    elif reason == "salt_changed":
        engine._ingest_ledger_attempt["salt"] = b"changed"
    elif reason == "length_mismatch":
        engine._ingest_ledger_attempt["entries"] = []
    statements, callbacks = [], []
    def read():
        engine._store._conn.set_trace_callback(statements.append)
        try:
            result = engine._ledger_cold_cursor(view, [False], int(reason == "complete"),
                                                0 if reason == "empty_session" else 1, callbacks)
            assert result["k"] == 0
        finally:
            engine._store._conn.set_trace_callback(None)
    if reason == "not_owner":
        read()
    else:
        with engine._store.atomic(immediate=True):
            read()
    assert not any(table in sql for table in ("ingest_views", "occurrence_bindings", "ledger_emissions", "stamp_observations")
                   for sql in statements)
    for callback in callbacks:
        callback()
    key = "length_mismatch" if reason == "length_mismatch" else "skipped." + reason
    assert count(engine, "cold_candidate." + key) == 1
    engine.__dict__.pop("_ingest_ledger_attempt", None)
    engine.__dict__.pop("_ingest_read_only", None)


@pytest.mark.parametrize("history", [["ok"], ["D", "E"], ["hi"]])
def test_t15_delta_only_unstamped_restarts_store_every_occurrence(engines, history):
    engine = engines()
    engine._ingest_messages([_m("user" if i % 2 == 0 else "assistant", text) for i, text in enumerate(history)])
    for n in range(2):
        restart(engine)
        engine._ingest_messages([_m("user", history[0])])
        assert sum(r["content"] == history[0] for r in engine._store.get_session_messages("S")) == n + 2
    assert count(engine, "cold_candidate.stop.no_occurrence_evidence") >= 1


def test_t15_preledger_history_is_not_an_anchor(engines, monkeypatch):
    engine = engines(mode="off")
    engine._ingest_messages([_m("user", "D"), _m("assistant", "E")])
    monkeypatch.setenv("LCM_INGEST_LEDGER", "shadow")
    restart(engine)
    engine._ingest_messages([_m("user", "D")])
    assert sum(r["content"] == "D" for r in engine._store.get_session_messages("S")) == 2


def test_t4_compress_return_then_fresh_engine_keeps_legacy_choice(engines, monkeypatch):
    outputs = []
    for enabled in (False, True):
        engine = engines(f"compress-{enabled}")
        before = [_m("user", "P", 1.), _m("assistant", "old", 2.), _m("user", "tail", 3.)]
        engine._ingest_messages(before)
        summary = pure_summary(engine)
        output = [before[0], summary, before[-1]]
        def compress_impl(messages, **kw):
            engine._last_compression_status = "compacted"
            engine._ingest_cursor = len(output)
            engine._ingest_cursor_needs_reconcile = False
            engine._pending_emission_candidates = [emission_capture(summary)]
            return output
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_compress_impl", compress_impl)
            patch.setattr(engine, "_survival_fit", lambda messages, result, *a, **kw: result)
            returned = engine.compress(before, current_tokens=100, force=True)
        assert engine._emission_handle is not None
        cold = _engine(engine._store.db_path.parent)
        try:
            # Actual legacy choice remains in use after a fresh engine is created.
            with monkeypatch.context() as patch:
                if not enabled:
                    patch.setattr(cold, "_ledger_cold_cursor", lambda *a: dict(k=0, pairs={}))
                else:
                    run, counters = candidate(cold, returned)
                    assert run["k"] == 1 and counters["stop.descriptor"] == 1
                cold._ingest_messages(deepcopy(returned))
                outputs.append((snapshot(cold), deepcopy(cold._last_ingest_reconciliation)))
        finally:
            cold.shutdown()
    assert outputs[0] == outputs[1]


def test_t1_preseeded_lost_row_has_distinct_stamp_from_stored_twin(engines, monkeypatch):
    engine = engines()
    p, twin, lost = _m("user", "P", 1.), _m("assistant", "same twin", 3.), _m("assistant", "same twin", 2.)
    engine._ingest_messages([p, twin])
    attempt = prepare(engine, [p, lost])
    # Stale unbound vector claims the lost row; exact row proof must veto it.
    with engine._store.atomic(immediate=True):
        ledger.write_view(engine._store._conn, attempt["salt"], attempt["lineage"], "ingest", None, attempt["entries"])
    force_today(engine, monkeypatch)
    engine._ingest_messages([p, lost])
    assert count(engine, "cold_candidate.used_ingest") == 1
    assert engine._last_ingest_reconciliation["k"] == 1
    assert snapshot(engine)[("assistant", "same twin", 2., None)] == 1
    assert snapshot(engine)[("assistant", "same twin", 3., None)] == 1


@pytest.mark.parametrize("broken,expected", [("ambiguous", "ambiguous_disposition"), ("missing", "missing_row"),
                                            ("other_session", "other_session"), ("stamp", "stamp"),
                                            ("observed_at", "observed_at"), ("ref", "not_contiguous")])
def test_row_proof_never_steps_over_a_bad_contiguous_binding(engines, broken, expected):
    engine = engines()
    view = [_m("user", "P", 1.), _m("assistant", "A", 2.)]
    engine._ingest_messages(view)
    conn = engine._store._conn
    bindings = sql_rows(engine, "occurrence_bindings")
    with engine._store.atomic(immediate=True):
        if broken == "ambiguous":
            conn.execute("UPDATE occurrence_bindings SET disposition='ambiguous' WHERE occ_id=?", (bindings[1]["occ_id"],))
        elif broken == "missing":
            conn.execute("UPDATE occurrence_bindings SET store_id=999999 WHERE occ_id=?", (bindings[1]["occ_id"],))
        elif broken == "other_session":
            conn.execute("UPDATE messages SET session_id='other' WHERE store_id=?", (bindings[1]["store_id"],))
        elif broken == "stamp":
            conn.execute("UPDATE occurrence_bindings SET stamp='99.0' WHERE occ_id=?", (bindings[1]["occ_id"],))
        elif broken == "observed_at":
            conn.execute("UPDATE messages SET observed_at=99.0 WHERE store_id=?", (bindings[1]["store_id"],))
        else:
            record = ledger.newest_view(conn, bindings[0]["lineage"], "ingest")
            entries = ledger.decode_entries(record["entries"])
            entries[1] = entries[1]._replace(ref=bindings[0]["occ_id"])
            ledger.write_view(conn, prepare(engine, view)["salt"], bindings[0]["lineage"], "ingest", None, entries)
    run, counters = candidate(engine, view)
    assert run["k"] == 1 and counters["stop." + expected] == 1


def test_stamp_observation_is_occurrence_evidence_even_with_null_binding_stamp(engines):
    engine = engines()
    view = [_m("user", "P", 1.)]
    engine._ingest_messages(view)
    b = sql_rows(engine, "occurrence_bindings")[0]
    with engine._store.atomic(immediate=True):
        engine._store._conn.execute("UPDATE occurrence_bindings SET stamp=NULL")
        engine._store._conn.execute("UPDATE messages SET observed_at=9.0")
        ledger.add_stamp_observation(engine._store._conn, b["lineage"], b["occ_id"], "1.0", "alt_stamp", b["view_seq"])
    assert candidate(engine, view)[0]["k"] == 1


@pytest.mark.parametrize("metadata", ["stamped", "unstamped", "uid"])
@pytest.mark.parametrize("prepended", [False, True])
def test_t16_fresh_user_equal_to_live_summary_is_stored(engines, monkeypatch, metadata, prepended):
    results = []
    for enabled in (False, True):
        engine = engines(f"summary-equal-{enabled}")
        p = _m("user", "real host head", 1.)
        engine._ingest_messages([p])
        summary = pure_summary(engine, "user")
        emit(engine, [summary, p], {0: "summary"})
        assert ledger.emission_candidate(engine._store._conn, prepare(engine, [p])["lineage"])[1] == "usable"
        fresh = dict(summary)
        if metadata == "stamped":
            fresh["timestamp"] = 20.
        elif metadata == "uid":
            fresh["message_uid"] = "fresh-summary-equal-user"
        view = [fresh, summary, p] if prepended else [fresh]
        run, counters = candidate(engine, view)
        assert run["k"] == 0 and counters["stop.descriptor"] == 1
        restart(engine)
        with monkeypatch.context() as patch:
            force_today(engine, patch)
            if not enabled:
                patch.setattr(engine, "_ledger_cold_cursor", lambda *a: dict(k=0, pairs={}))
            engine._ingest_messages(deepcopy(view))
        stored = snapshot(engine)
        assert stored[("user", summary["content"], fresh.get("timestamp"), None)] >= 1
        assert count(engine, "cold_candidate.used_emission") == 0
        results.append(stored)
    assert results[0] == results[1]


@pytest.mark.parametrize("shape", ["host", "summary", "scaffold"])
@pytest.mark.parametrize("observation", [False, True])
def test_t17_null_stamped_continuation_needs_own_occurrence_evidence(engines, monkeypatch, shape, observation):
    results = []
    for enabled in (False, True):
        engine = engines(f"null-continuation-{enabled}")
        p, x = _m("user", "P", 1.), _m("user", "X")
        if shape == "summary":
            x = pure_summary(engine, "user")
        elif shape == "scaffold":
            x["content"] = "[Current user objective preserved from compacted history]\nHost objective"
        engine._ingest_messages([p, x])
        bindings = sql_rows(engine, "occurrence_bindings")
        assert len(bindings) == 2 and bindings[1]["stamp"] is None
        if shape != "host":
            relations = engine._store.get_message_relations([bindings[1]["store_id"]], "generated_unproven")
            assert any(r["kind"] == "generated_unproven" for r in relations)
        if observation:
            with engine._store.atomic(immediate=True):
                ledger.add_stamp_observation(engine._store._conn, bindings[1]["lineage"], bindings[1]["occ_id"],
                                             "2.0", "alt_stamp", bindings[1]["view_seq"])
        view = [p, dict(x, timestamp=2.)]
        restart(engine)
        run, counters = candidate(engine, view)
        assert run["k"] == (2 if observation else 1)
        assert counters["stop." + ("length_end" if observation else "no_occurrence_evidence")] == 1
        with monkeypatch.context() as patch:
            force_today(engine, patch)
            if not enabled:
                patch.setattr(engine, "_ledger_cold_cursor", lambda *a: dict(k=0, pairs={}))
            engine._ingest_messages(deepcopy(view))
        if enabled:
            assert engine._last_ingest_reconciliation["k"] == run["k"]
        results.append(snapshot(engine))
    # Observation controls isolate the reader: today's anchor does not read the
    # ledger observation yet. With no witness, on/off must store the same rows.
    if not observation:
        assert results[0] == results[1]
        assert results[1][("user", x["content"], 2., None)] == 1


def test_t17_equal_known_uids_prove_a_null_stamped_continuation_on_own_ref(engines):
    engine = engines()
    view = [_m("user", "P", 1.), _m("assistant", "X", uid="own-uid")]
    engine._ingest_messages(view)
    run, counters = candidate(engine, view)
    assert run["k"] == 2 and counters["stop.length_end"] == 1


@pytest.mark.parametrize("seed", range(8))
def test_t1_seeded_restarts_refusals_interrupts_and_twins(engines, monkeypatch, seed):
    """Paired histories: multiset inclusion, no new silent surplus, seam equation.

    Random deliveries mix full views and delta-only views. Every occurrence has
    a unique stamp except the explicit unstamped gateway cases pinned by T15.
    """
    rng = random.Random(seed)
    events = [(rng.choice(("restart", "delta", "undo", "refuse", "drop", "evict", "interrupt", "commit_interrupt")),
               rng.choice(("ok", "same twin", "new"))) for _ in range(12)]
    events += [("unstamped", "ok"), ("unstamped", "ok")]
    results, baseline_steps = [], []
    for enabled in (False, True):
        engine = engines(f"history-{seed}-{enabled}")
        with monkeypatch.context() as patch:
            # Exercise provisional head -> resolved reader key.
            with patch.context() as unresolved:
                unresolved.setattr(engine, "_host_uid_lineage_key", lambda *a, **kw: (None, "unresolved"))
                engine._ingest_messages([_m("user", "head", 1.)])
            if not enabled:
                patch.setattr(engine, "_ledger_cold_cursor", lambda *a: dict(k=0, pairs={}))
            else:
                reader = engine._ledger_cold_cursor
                seam_state = {}
                def seam(messages, ignored, today, count_, callbacks):
                    run = reader(messages, ignored, today, count_, callbacks)
                    seam_state.update(bound=max(today, run["k"]), pending=True, remaps=[])
                    return run
                patch.setattr(engine, "_ledger_cold_cursor", seam)
                read_proof, prematch = engine._durable_commit_proof_payload, engine._identity_anchor_prematch
                def observe_seam(*a, **kw):
                    if seam_state.get("pending"):
                        seam_state["observed"] = engine._ingest_cursor
                        seam_state["pending"] = False
                    return read_proof(*a, **kw)
                def observe_remap(messages, identities, cursor, *a, **kw):
                    if seam_state:
                        seam_state["remaps"].append(cursor)
                    return prematch(messages, identities, cursor, *a, **kw)
                patch.setattr(engine, "_durable_commit_proof_payload", observe_seam)
                patch.setattr(engine, "_identity_anchor_prematch", observe_remap)
            host = [_m("user", "head", 1.)]
            expected = Counter({("user", "head", 1., None): 1})
            for step, (event, text) in enumerate(events, 1):
                if enabled:
                    seam_state.clear()
                stamp = float(10 * step)
                new = [_m("user", text)] if event == "unstamped" else [_m("user", text, stamp), *tool_pair(stamp + 1)]
                expected.update((m["role"], m["content"], m.get("timestamp"), m.get("tool_call_id")) for m in new)
                host = new if event in {"delta", "unstamped"} else [*host, *new]
                if event == "undo":
                    host = [host[0], *new]  # replay the same call id with fresh occurrence stamps
                if event in {"refuse", "drop", "evict", "commit_interrupt"}:
                    summary = pure_summary(engine)
                    emit(engine, [host[0], summary], {1: "summary"})
                    handle = engine._emission_handle
                    if event == "refuse":
                        with engine._store.atomic(immediate=True):
                            ledger.set_emission_state(engine._store._conn, handle.lineage, handle.view_seq, "refused")
                    if event == "evict":
                        for n in range(3):
                            ledger.mark_pending_emission((engine._store.db_identity, prepare(engine, host)["salt"]),
                                                         handle.lineage, origin=engine, rows=[], synthetic=n)
                        ledger._PENDING_EMISSIONS.clear()
                if event == "interrupt":
                    with patch.context() as fault:
                        def interrupt(*a):
                            raise KeyboardInterrupt("before commit")
                        fault.setattr(engine, "_write_ingest_ledger", interrupt)
                        with pytest.raises(KeyboardInterrupt):
                            engine._ingest_messages(deepcopy(host))
                    # Retry in this live process; failure sets needs_reconcile.
                elif event != "delta":
                    restart(engine)
                if event == "commit_interrupt":
                    conn = engine._store._conn
                    engine._store._conn = CommitInterruptProxy(conn)
                    try:
                        with pytest.raises(KeyboardInterrupt, match="after COMMIT"):
                            engine._ingest_messages(deepcopy(host))
                    finally:
                        engine._store._conn = conn
                    restart(engine)
                engine._ingest_messages(deepcopy(host))
                stored = snapshot(engine)
                if enabled and seam_state:
                    assert seam_state["observed"] == seam_state["bound"]
                    assert all(cursor <= seam_state["bound"] for cursor in seam_state["remaps"])
                if not enabled:
                    baseline_steps.append(stored)
                else:
                    baseline = baseline_steps[step - 1]
                    assert not ((expected & baseline) - stored), (seed, step, event, "lost occurrence")
                    assert not ((stored - expected) - (baseline - expected)), (seed, step, event, "silent surplus")
            results.append(snapshot(engine))
    assert not ((results[1] - expected) - (results[0] - expected)), "no new silent surplus"
