"""#2055 P1f-1: provisional adoption preserves I1/I2/I5 identity and provenance."""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from hermes_lcm import ingest_ledger as ledger
from hermes_lcm.sqlite_util import _SqliteSavepointError
from tests.invariants.test_p1e_decided_tails import contents, history
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_cold_cursor import force_today
from tests.test_issue_2055_emission_restart import _main_ledger_write
from tests.test_issue_2055_ingest_ledger_writes import (
    SALT, count, engines as engines, ledger_state, sql_rows,
)
from tests.test_issue_2055_pending_emissions import emit, fail_exit

P, L = "provisional:S", "home:root"


@pytest.fixture(autouse=True)
def isolated_pending(monkeypatch):
    monkeypatch.setattr(ledger, "_PENDING_EMISSIONS", {})


def resolution(engine, monkeypatch, value):
    original = engine._host_uid_lineage_key
    monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda sid=None, **kw:
                        (value, None) if kw.get("require_complete") else original(sid))


def seed(engine, lineage, rows, *, session="S", dispositions=None):
    store, conn = engine._store, engine._store._conn
    entries, stamps = engine._ledger_view_entries(SALT, rows)
    with store.ingest_atomic():
        ids = [store.append(session, row, source="cli", conversation_id="conv") for row in rows]
        maximum = conn.execute("SELECT max(ordinal) FROM occurrence_bindings WHERE lineage=?", (lineage,)).fetchone()[0] or 0
        seq = ledger.next_view_seq(conn)
        occs = ledger.insert_bindings(conn, lineage, [dict(store_id=sid, ordinal=maximum + (i + 1) * ledger.ORDINAL_STEP,
            identity_digest=e.identity_digest, stamp=stamp, disposition=(dispositions or ["stored"] * len(rows))[i], view_seq=seq)
            for i, (sid, e, stamp) in enumerate(zip(ids, entries, stamps))])
        ledger.write_view(conn, SALT, lineage, "ingest", None,
                          [e._replace(ref=occ) for e, occ in zip(entries, occs)], view_seq=seq)
    return ids, occs


def adopt(engine, *, resolved=L, salt=SALT, store=None, sid="S"):
    store = store or engine._store
    callbacks = []
    with store.ingest_atomic():
        result = engine._ledger_adopt(store, sid, resolved, salt, callbacks)
    for callback in callbacks:
        callback()
    return result


def meta(engine, key):
    row = engine._store._conn.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def test_merge_rekeys_every_table(engines):
    engine = engines()
    conn = engine._store._conn
    seed(engine, L, [_m("user", "target", 1.)])
    ids, occs = seed(engine, P, history()[:2], dispositions=["stored", "composite_part"])
    with engine._store.ingest_atomic():
        # Part witnesses share an ordinal; an unplaced witness stays NULL.
        ledger.insert_bindings(conn, P, [dict(store_id=ids[1], ordinal=ledger.ORDINAL_STEP,
            identity_digest=b"x" * 16, disposition="composite_part", view_seq=1),
            dict(store_id=ids[1], identity_digest=b"y" * 16, disposition="composite_part", view_seq=1)])
        ledger.add_stamp_observation(conn, P, occs[0], "9.0", "alt_stamp", 1)
        for lineage in (L, P):
            ledger.update_store_mark(conn, lineage, 10 if lineage == L else 99)
            ledger.update_leaf_mark(conn, lineage, 7 if lineage == L else 9)
            ledger.update_legacy_proof_digest(conn, lineage, b"proof" if lineage == P else None)
            if lineage == P:
                ledger._update_mark(conn, lineage, "stamp_policy", "stamp_unreliable")
            for kind in ("ingest", "emission"):
                for _ in range(2):
                    seq = ledger.write_view(conn, SALT, lineage, kind, "proposed" if kind == "emission" else None, [])
                    if kind == "emission":
                        ledger.insert_emissions(conn, lineage, seq, [dict(output_index=0, descriptor_digest=b"e" * 16, kind="summary")])
    before = ledger_state(engine)
    source = [b for b in before["occurrence_bindings"] if b["lineage"] == P]
    oldmax = max(b["ordinal"] for b in before["occurrence_bindings"] if b["lineage"] == L)
    assert adopt(engine) == L
    after = ledger_state(engine)
    for table in ("occurrence_bindings", "ingest_views", "ledger_emissions", "stamp_observations", "lineage_marks"):
        assert all(row["lineage"] == L for row in after[table])
    assert [b["occ_id"] for b in after["occurrence_bindings"]] == [b["occ_id"] for b in before["occurrence_bindings"]]
    moved = {b["occ_id"]: b for b in after["occurrence_bindings"]}
    positions = [(b["ordinal"], moved[b["occ_id"]]["ordinal"]) for b in source if b["ordinal"] is not None]
    assert all(new > oldmax for _, new in positions)
    assert len({new for _, new in positions}) == len({old for old, _ in positions})
    assert sorted(positions) == sorted(positions, key=lambda pair: pair[1])
    assert moved[source[-1]["occ_id"]]["ordinal"] is None
    assert Counter(v["kind"] for v in after["ingest_views"]) == {"ingest": 2, "emission": 2}
    assert {v["view_seq"] for v in after["ingest_views"]} == {v["view_seq"] for v in before["ingest_views"] if v["lineage"] == P}
    assert [(v["entries"], v["prefix_chain"]) for v in after["ingest_views"]] == [
        (v["entries"], v["prefix_chain"]) for v in before["ingest_views"] if v["lineage"] == P]
    mark = ledger.marks(conn, L)
    assert (mark["max_store_id"], mark["max_leaf_id"], mark["legacy_proof_digest"], mark["stamp_policy"]) == (10, 7, b"proof", "stamp_unreliable")
    assert meta(engine, "adopted_from:" + P) == meta(engine, "session_lineage:S") == L
    assert count(engine, "lineage_merged") == 1
    assert [count(engine, "lineage_merge." + key) for key in ("bindings", "views", "emissions")] == [4, 4, 2]
    snapshot = ledger_state(engine)
    adopt(engine)
    assert ledger_state(engine) == snapshot and count(engine, "lineage_merged") == 1


def test_first_resolved_turn_cold_cursor_sees_history(engines, monkeypatch):
    engine = engines(sessions=[])
    rows = history()[:2]
    engine.ingest(deepcopy(rows))
    resolution(engine, monkeypatch, L)
    force_today(engine, monkeypatch)
    engine.ingest(deepcopy([*rows, _m("user", "new", 8.)]))
    assert count(engine, "cold_candidate.used_ingest") == 1, engine._ingest_ledger_counters
    assert count(engine, "cold_candidate.no_anchor") == 0
    assert engine._last_ingest_reconciliation["k"] == 2
    assert contents(engine) == ["A", "B", "new"]


def test_first_resolved_turn_never_skips_a_row_absent_from_view(engines, monkeypatch):
    engine = engines(sessions=[])
    engine.ingest(history()[:3])
    resolution(engine, monkeypatch, L)
    force_today(engine, monkeypatch)
    engine.ingest([*history()[:2], _m("user", "new", 8.)])
    assert count(engine, "cold_candidate.used_ingest") == 1
    assert engine._last_ingest_reconciliation["k"] == 2
    assert contents(engine) == ["A", "B", "C", "new"]


def test_stranded_by_older_writer_merged(engines, monkeypatch):
    engine = engines(sessions=[])
    engine.ingest(history()[:1])
    with engine._store.ingest_atomic():
        engine._store._conn.execute("INSERT INTO ledger_meta VALUES ('session_lineage:S',?)", (L,))
    resolution(engine, monkeypatch, None)  # The persisted key alone must trigger repair.
    engine.ingest(history()[:2])
    assert {r["lineage"] for r in sql_rows(engine, "occurrence_bindings")} == {L}
    assert meta(engine, "adopted_from:" + P) == L


@pytest.mark.parametrize("failure", [False, True])
def test_tail_retry_resolution_merges_before_rematch(engines, monkeypatch, failure):
    engine = engines(sessions=[])
    engine.ingest(history()[:2])
    resolution(engine, monkeypatch, L)
    locker = sqlite3.connect(engine._store.db_path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        engine.on_session_end("S", history())
    finally:
        locker.rollback()
        locker.close()
    [tail] = engine._pending_session_tails
    assert tail["ledger"]["resolved"] == L
    matcher, witnessed = engine._match_pending_tail, []
    if failure:
        def fail(groups):
            raise ValueError("merge relation failure")
        monkeypatch.setattr(engine._store, "add_message_relations", fail)

    def match(tail):
        if tail["store"]._conn.in_transaction:
            assert meta(engine, "adopted_from:" + P) == (None if failure else L)
            assert all(b["lineage"] == (P if failure else L) for b in sql_rows(engine, "occurrence_bindings"))
            witnessed.append(True)
        return matcher(tail)

    monkeypatch.setattr(engine, "_match_pending_tail", match)
    engine._session_id = "other"  # Adoption must use the frozen session, not the current one.
    engine._retry_pending_session_tails()
    assert witnessed == [True] and not engine._pending_session_tails
    assert contents(engine) == ["A", "B", "C", "D"]
    assert {b["lineage"] for b in sql_rows(engine, "occurrence_bindings")} == {P if failure else L}
    assert count(engine, "lineage_merge_failed") == int(failure)


def test_merge_pairs_require_stamp(engines):
    engine = engines()
    row = _m("user", "ok", 1.)
    targets, _ = seed(engine, L, [row, row, _m("user", "unstamped", None)])
    sources, _ = seed(engine, P, [row, row, row, _m("user", "ok", None), _m("user", "unstamped", None)])
    adopt(engine)
    relations = [r for r in sql_rows(engine, "message_relations") if r["kind"] == "possible_replay_of"]
    assert [(r["store_id"], r["related_store_id"]) for r in relations] == list(zip(sources[:2], targets[:2]))
    assert count(engine, "lineage_merge.replay_pairs") == count(engine, "lineage_merge.unstamped_skipped") == 2
    adopt(engine)
    assert len(sql_rows(engine, "message_relations")) == 2


@pytest.mark.parametrize("target_session,expected", [("S", 1), ("other", 0)])
def test_merge_late_counted(engines, target_session, expected):
    engine = engines()
    seed(engine, P, history()[:1])
    seed(engine, L, history()[1:2], session=target_session)
    adopt(engine)
    assert count(engine, "lineage_merged") == 1
    assert count(engine, "lineage_merge.late") == expected


def test_refusal_after_merge_reaches_rekeyed_view(engines, monkeypatch):
    engine = engines(sessions=[])
    emit(engine, [_m("user", "generated", 10.)])
    handle = engine._emission_handle
    resolution(engine, monkeypatch, L)
    engine.ingest(history()[:1])
    engine._signal_emission_state("refused", "S")
    [emission] = sql_rows(engine, "ledger_emissions")
    assert handle.lineage == P and emission["lineage"] == L and emission["state"] == "refused"
    assert next(v for v in sql_rows(engine, "ingest_views") if v["kind"] == "emission")["state"] == "refused"
    assert count(engine, "emission_refused") == 1


def test_pending_digest_after_merge(engines, monkeypatch):
    engine = engines(sessions=[])
    rows = fail_exit(engine, monkeypatch)
    resolution(engine, monkeypatch, L)
    monkeypatch.setattr(engine, "_lcm_generated_marker", lambda row: None)

    def failed(*args, **kwargs):
        raise ValueError("retry remains pending")

    monkeypatch.setattr(ledger, "insert_emissions", failed)
    engine.ingest(rows)
    assert meta(engine, "session_lineage:S") == L
    assert count(engine, "generated_unproven_rows") == 1
    [relation] = sql_rows(engine, "message_relations")
    assert relation["kind"] == "generated_unproven" and engine._store.get(relation["store_id"])["content"] == "generated"
    assert ledger.pending_emission_digests((engine._store.db_identity, SALT), P)


@pytest.mark.parametrize("adopted", [False, True])
def test_signal_falls_back_to_raw_handle(engines, adopted):
    engine = engines(sessions=[])
    emit(engine, [_m("user", "generated", 10.)])
    if adopted:
        engine._signal_emission_state("adopted", "S")
    # An older writer persisted L while leaving P stranded, or its merge failed.
    with engine._store.ingest_atomic():
        engine._store._conn.execute("INSERT INTO ledger_meta VALUES ('session_lineage:S',?)", (L,))
    engine._signal_emission_state("refused", "S")
    assert sql_rows(engine, "ledger_emissions")[0]["state"] == ("adopted" if adopted else "refused")
    assert count(engine, "refused_after_adopted" if adopted else "emission_refused") == 1


@pytest.mark.parametrize("source,target", [("12", "9"), ("9", "12"), ("12", None)])
def test_recovered_mark_survives_merge(engines, source, target):
    engine = engines()
    seed(engine, P, history()[:1])
    with engine._store.ingest_atomic():
        engine._store._conn.execute("INSERT INTO ledger_meta VALUES (?,?)", ("emission_recovered:" + P, source))
        if target:
            engine._store._conn.execute("INSERT INTO ledger_meta VALUES (?,?)", ("emission_recovered:" + L, target))
    adopt(engine)
    assert int(meta(engine, "emission_recovered:" + L)) == max(int(source), int(target or 0))
    assert meta(engine, "emission_recovered:" + P) is None


def test_merge_failure_stays_provisional_and_restores_schema_flag(engines, monkeypatch):
    engine = engines(sessions=[])
    engine.ingest(history()[:1])
    resolution(engine, monkeypatch, L)
    engine._store._pending_relation_schema = False

    def failed(groups):
        engine._store._pending_relation_schema = True
        raise ValueError("relation failure after re-key")

    with monkeypatch.context() as patch:
        patch.setattr(engine._store, "add_message_relations", failed)
        engine.ingest(history()[:2])
    assert not engine._store._pending_relation_schema
    assert meta(engine, "session_lineage:S") is None
    assert {b["lineage"] for b in sql_rows(engine, "occurrence_bindings")} == {P}
    assert count(engine, "lineage_merge_failed") == 1 and count(engine, "lineage_merged") == 0
    assert not getattr(engine, "_ledger_adopted", ())
    engine.ingest(history()[:2])
    assert {b["lineage"] for b in sql_rows(engine, "occurrence_bindings")} == {L}


def test_busy_at_commit_retry_remerges(engines, monkeypatch):
    engine = engines(sessions=[])
    engine.ingest(history()[:1])
    resolution(engine, monkeypatch, L)
    atomic, attempts = engine._store.ingest_atomic, []

    @contextmanager
    def busy(**kwargs):
        with atomic(**kwargs):
            yield
            assert not getattr(engine, "_ledger_adopted", ())
            assert meta(engine, "adopted_from:" + P) == L
            attempts.append(True)
            if len(attempts) == 1:
                exc = sqlite3.OperationalError("busy at COMMIT")
                exc.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
                raise exc

    monkeypatch.setattr(engine._store, "ingest_atomic", busy)
    engine.ingest(history()[:2])
    assert attempts == [True, True]
    assert count(engine, "lineage_merged") == 1 and count(engine, "lineage_merge.bindings") == 1
    assert len(engine._ledger_adopted) == 1 and contents(engine) == ["A", "B"]


class Crash(BaseException):
    pass


@pytest.mark.parametrize("error", [Crash, KeyboardInterrupt, _SqliteSavepointError])
def test_merge_interrupt_rolls_back(engines, monkeypatch, error):
    engine = engines()
    seed(engine, P, history()[:1])
    before = ledger_state(engine)

    def fail(groups):
        raise error("inside merge")

    monkeypatch.setattr(engine._store, "add_message_relations", fail)
    with pytest.raises(error):
        adopt(engine)
    assert ledger_state(engine) == before and meta(engine, "session_lineage:S") is None
    assert not getattr(engine, "_ledger_adopted", ())


def test_crash_after_commit_fresh_engine_does_not_merge_twice(engines, monkeypatch):
    engine = engines(sessions=[])
    engine.ingest(history()[:1])
    # A separate process commits adoption, then dies before publishing its memo/counters.
    script = r'''
import os, pathlib, sys
import tests.conftest
from tests.test_host_uid_shadow import _engine
engine = _engine(pathlib.Path(sys.argv[1]))
store = engine._store
with store.ingest_atomic():
    engine._ledger_adopt(store, "S", "home:root", store.ingest_ledger_salt(), [])
os._exit(91)
'''
    root = str(engine._hermes_home)
    result = subprocess.run([sys.executable, "-c", script, root], capture_output=True, text=True, timeout=30)
    assert result.returncode == 91, result.stderr
    engine.shutdown()
    from tests.test_host_uid_shadow import _engine
    fresh = _engine(Path(engine._hermes_home))
    try:
        resolution(fresh, monkeypatch, L)
        fresh.ingest(history()[:2])
        assert contents(fresh) == ["A", "B"] and count(fresh, "lineage_merged") == 0
        assert {b["lineage"] for b in sql_rows(fresh, "occurrence_bindings")} == {L}
        assert meta(fresh, "adopted_from:" + P) == L
    finally:
        fresh.shutdown()


@pytest.mark.parametrize("guard", ["off", "read_only", "not_ready", "live_salt"])
def test_adoption_guards(engines, monkeypatch, guard):
    engine = engines()
    seed(engine, P, history()[:1])
    before, statements = ledger_state(engine), []
    if guard == "off":
        monkeypatch.setenv("LCM_INGEST_LEDGER", "off")
    elif guard == "read_only":
        engine._ingest_read_only = "frozen run"
    elif guard == "not_ready":
        monkeypatch.setattr(engine._store, "ingest_ledger_state", lambda: "absent")
    else:
        engine._store._ingest_ledger_cached_salt = SALT  # Guard must consult the live value.
        monkeypatch.setattr(engine._store, "ingest_ledger_salt", lambda: b"n" * 32)
    engine._store._conn.set_trace_callback(statements.append)
    try:
        assert adopt(engine) is None
    finally:
        engine._store._conn.set_trace_callback(None)
    assert ledger_state(engine) == before
    if guard == "off":
        assert not any("ledger" in sql or "occurrence" in sql for sql in statements)


def test_host_uid_bindings_untouched(engines, monkeypatch):
    monkeypatch.setenv("LCM_HOST_MESSAGE_UID", "shadow")
    engine = engines()
    resolution(engine, monkeypatch, None)
    engine.ingest(history()[:2])
    before = sql_rows(engine, "host_uid_bindings")
    assert before
    adopt(engine)
    assert sql_rows(engine, "host_uid_bindings") == before


def test_frozen_writer_ingests_merged_store_with_same_cursor(engines, monkeypatch):
    results = []
    for name, frozen in (("current", False), ("frozen", True)):
        engine = engines(name, sessions=[])
        engine.ingest(history()[:2])
        resolution(engine, monkeypatch, L)
        if frozen:
            monkeypatch.setattr(engine, "_write_ingest_ledger", _main_ledger_write.__get__(engine))
        force_today(engine, monkeypatch)
        engine.ingest(history()[:3])
        assert count(engine, "write_failed") == 0 and count(engine, "lineage_merged") == 1
        results.append((contents(engine), engine._last_ingest_reconciliation["k"],
                        [(b["lineage"], b["ordinal"], b["store_id"]) for b in sql_rows(engine, "occurrence_bindings")]))
    assert results[0] == results[1] and results[0][1] == 2


def test_merge_requires_owner(engines):
    engine = engines()
    with pytest.raises(ledger.LedgerTransactionError):
        ledger.merge_provisional(engine._store._conn, P, L)


@pytest.mark.parametrize("failure", ["busy", "transaction_lost"])
def test_merge_owner_failures_propagate(engines, monkeypatch, failure):
    engine = engines()
    seed(engine, P, history()[:1])
    before = ledger_state(engine)

    def fail(groups):
        if failure == "transaction_lost":
            engine._store._conn.rollback()
            raise ValueError("owner transaction lost")
        error = sqlite3.OperationalError("busy during merge")
        error.sqlite_errorcode = sqlite3.SQLITE_BUSY_SNAPSHOT
        raise error

    monkeypatch.setattr(engine._store, "add_message_relations", fail)
    with pytest.raises(ValueError if failure == "transaction_lost" else sqlite3.OperationalError):
        adopt(engine)
    assert ledger_state(engine) == before and not getattr(engine, "_ledger_adopted", ())
    assert count(engine, "lineage_merge_failed") == 0


@pytest.mark.parametrize("frozen", [False, True])
def test_direct_writer_adopts_before_writing(engines, monkeypatch, frozen):
    engine = engines(sessions=[])
    seed(engine, P, history()[:1])
    resolution(engine, monkeypatch, L)
    row = history()[1]
    attempt = engine._prepare_ingest_ledger([row], Counter())
    callbacks = []
    with engine._store.ingest_atomic():
        sid = engine._store.append("S", row, conversation_id="conv")
        attempt.update(stored={0: sid}, matched={}, remainders={}, observations=[])
        (_main_ledger_write if frozen else type(engine)._write_ingest_ledger)(engine, attempt, callbacks, Counter())
        assert not getattr(engine, "_ledger_adopted", ())
    for callback in callbacks:
        callback()
    assert count(engine, "lineage_merged") == 1 and count(engine, "write_failed") == 0
    assert {b["lineage"] for b in sql_rows(engine, "occurrence_bindings")} == {L}
    assert contents(engine) == ["A", "B"]


def test_merge_ordinal_overflow_keeps_both_histories(engines):
    engine = engines()
    seed(engine, L, history()[:1])
    with engine._store.ingest_atomic():
        ledger.apply_respace(engine._store._conn, L, {ledger.ORDINAL_STEP: (1 << 63) - 1})
    seed(engine, P, history()[1:2])
    before = ledger_state(engine)
    assert adopt(engine) == P
    assert ledger_state(engine) == before and count(engine, "lineage_merge_failed") == 1
    assert not getattr(engine, "_ledger_adopted", ())


def test_p1_gate_failed_adoption_then_later_ingests(engines, monkeypatch):
    from tests.p1_gate import expect_from_view, record, store_outcome

    engine = engines(sessions=[])
    view = history()
    engine.ingest(deepcopy(view[:2]))
    resolution(engine, monkeypatch, L)
    merge = ledger.merge_provisional
    fired = []

    def fail(*args, **kwargs):
        merge(*args, **kwargs)
        fired.append(True)
        raise sqlite3.OperationalError("injected adoption savepoint failure")

    with monkeypatch.context() as patch:
        patch.setattr(ledger, "merge_provisional", fail)
        engine.ingest(deepcopy(view[:3]))
    assert fired and count(engine, "lineage_merge_failed") >= 1
    assert meta(engine, "adopted_from:" + P) is None
    assert {row["lineage"] for row in sql_rows(engine, "occurrence_bindings")} == {P}
    engine.ingest(deepcopy(view))
    view += [_m("user", "new turn", 8.), _m("assistant", "new reply", 9.)]
    engine.ingest(deepcopy(view))
    assert contents(engine) == [row["content"] for row in view]
    outcome = store_outcome(engine._store._conn, expect_from_view(view, "conv"))
    record(dict(family="savepoint", boundary="adoption"), outcome, engine)
    assert outcome["loss"] == outcome["silent"] == 0, outcome
