"""P1c-1: engine-direct exit captures, shadow vectors and signal ownership."""
import copy
import logging
import sqlite3
import time

import pytest

from hermes_lcm import ingest_ledger as ledger
from hermes_lcm.emission_projection import _emission_identity
from hermes_lcm.dag import SummaryNode
from tests.test_issue_2055_ingest_ledger_writes import (
    SALT, count, engines as _engines, sql_rows,
)

engines = _engines

PATHS = ("normal", "stub", "fit", "recovery", "exception", "late_exception", "raise", "bypass",
         "identical", "hidden_same", "hidden_carried", "cursor", "stale", "provisional", "carried_fresh")
WRITES = {"normal", "stub", "fit", "recovery", "exception", "late_exception", "provisional", "carried_fresh"}


def candidate(row, kind="summary"):
    return dict(kind=kind, span=row["content"], row=row, full_identity=_emission_identity(row))


def prepare_path(engine, monkeypatch, path="normal"):
    """Control the pipeline result, exercising real compress bookkeeping and proof construction."""
    before = [dict(role="user" if i % 2 == 0 else "assistant", content=f"original {i}", timestamp=10. + i)
              for i in range(4)]
    generated = dict(role="user", content="generated summary", message_uid="engine-fixed")
    result = [generated, before[-1]]
    engine._ingest_messages(before)
    if path in {"hidden_carried", "carried_fresh", "stale"}:
        prior = dict(role="user", content="prior generated summary")
        before = [prior, *before[1:]]
        engine._ingest_cursor = len(before)
        engine._pending_emission_candidates = [candidate(prior)]
        engine._record_compress_commit_proof([], before)
        result = [prior, generated, before[-1]]
    if path == "provisional":
        monkeypatch.setattr(engine, "_host_uid_lineage_key", lambda *a, **kw: (None, "unresolved"))
        with engine._store.atomic():
            engine._store._conn.execute("DELETE FROM ledger_meta WHERE key='session_lineage:S'")
    if path == "bypass":
        engine._session_stateless = True
    if path in {"identical", "hidden_same"}:
        result = before
    if path == "hidden_carried":
        result = [dict(row) for row in before]
        result[1]["content"] += " sanitized"
        monkeypatch.setattr(engine, "_sanitize_active_context_messages", lambda rows: result)

    def impl(messages, **kwargs):
        if path in {"raise", "exception", "stale"}:
            raise ValueError("pipeline sentinel")
        engine._last_compression_status = "overflow_recovery" if path == "recovery" else "compacted"
        engine._ingest_cursor_needs_reconcile = False
        engine._ingest_cursor = 0 if path in {"cursor", "fit"} else len(result)
        if path not in {"identical", "hidden_same", "hidden_carried", "fit"}:
            engine._pending_emission_candidates = [candidate(generated, "recovery" if path == "recovery" else "summary")]
        if path == "stub":
            engine._last_compression_status = "sanitized"
            engine._stub_first_exit_now = dict(tokens_before=100, tokens_after=50, target_tokens=80, backlog_rows=2)
        if path.startswith("hidden_"):
            engine._foreground_budget.leaves = 1
            engine._compress_host_rows_consumed = 0
            engine._compress_hidden_rows_consumed = 1
            # The hidden-only guard must run before proof construction.
            return [dict(row, content=row["content"] + " padding") for row in before]
        return result

    def fit(messages, output, *args, **kwargs):
        if path in {"fit", "exception", "late_exception"}:
            carrier = dict(role="user", content="fitted carrier", message_uid="carrier-fixed")
            engine._survival_fit_emission = candidate(carrier, "carrier")
            fitted = [carrier, before[-1]]
            if path == "late_exception" and not kwargs.get("after_exception"):
                return output
            return fitted
        if path == "stale":
            return [dict(before[-1])]
        return output

    monkeypatch.setattr(engine, "_compress_impl", impl)
    monkeypatch.setattr(engine, "_survival_fit", fit)
    if path == "late_exception":
        record = engine._host_uid_record_engine
        calls = []

        def late(rows):
            calls.append(rows)
            if len(calls) == 1:
                raise ValueError("after proof sentinel")
            return record(rows)
        monkeypatch.setattr(engine, "_host_uid_record_engine", late)
    return before


def compress_path(engine, monkeypatch, path="normal"):
    before = prepare_path(engine, monkeypatch, path)
    return before, engine.compress(before, current_tokens=100, force=True)


def emissions(engine):
    return sql_rows(engine, "ledger_emissions")


def emission_views(engine):
    return [v for v in sql_rows(engine, "ingest_views") if v["kind"] == "emission"]


@pytest.mark.parametrize("path", PATHS)
def test_return_paths_write_only_fresh_descriptors(engines, monkeypatch, path):
    engine = engines()
    before = prepare_path(engine, monkeypatch, path)
    if path == "raise":
        with pytest.raises(ValueError, match="pipeline sentinel"):
            engine.compress(before, current_tokens=100, force=True)
        returned = None
    else:
        returned = engine.compress(before, current_tokens=100, force=True)
    rows, views = emissions(engine), emission_views(engine)
    if path not in WRITES:
        assert rows == views == []
        if path == "cursor":
            assert count(engine, "emission_no_descriptors.cursor_mismatch") == 1
        if path == "stale":
            assert count(engine, "emission_no_descriptors.no_capture") == 1
        # Make these new behavior assertions red on the base, including no-write paths.
        assert hasattr(engine, "_emission_handle") and engine._emission_handle is None
        return
    expected = "carrier" if path in {"fit", "exception", "late_exception"} else "recovery" if path == "recovery" else "summary"
    assert len(rows) == len(views) == 1
    assert rows[0]["kind"] == expected and rows[0]["state"] == views[0]["state"] == "proposed"
    assert rows[0]["source_store_ids"] == "[]" and rows[0]["next_occ_id"] is None
    assert rows[0]["view_seq"] == views[0]["view_seq"] == engine._emission_handle.view_seq
    entries = ledger.decode_entries(views[0]["entries"])
    prepared = engine._prepare_ingest_ledger(returned, {"ledger_digest_ms": 0.})
    assert len(entries) == len(returned)
    assert [e[:3] for e in entries] == [e[:3] for e in prepared["entries"]]
    index = rows[0]["output_index"]
    assert entries[index].is_descriptor and entries[index].ref == rows[0]["emission_id"]
    assert rows[0]["descriptor_digest"] == ledger.identity_digest(SALT, returned[index])
    assert all(e.ref is None and not e.is_descriptor for i, e in enumerate(entries) if i != index)
    assert entries[index].uidset_digest == ledger.uidset_digest(SALT, returned[index]["message_uid"], [])
    assert engine._last_emission_write_ms["digest"] >= 0 and engine._last_emission_write_ms["ledger"] >= 0
    if path == "fit":
        assert engine._compress_commit_reason == "cursor_mismatch"
    if path == "provisional":
        assert rows[0]["lineage"] == "provisional:S"


@pytest.mark.parametrize("damage,reason", [("list", "list_mismatch"), ("bounds", "stale"), ("duplicate", "stale"), ("of", "stale")])
def test_capture_currency_guards(engines, monkeypatch, damage, reason):
    engine = engines()
    before = prepare_path(engine, monkeypatch)
    original = engine._record_compress_commit_proof

    def capture(*args):
        rows, fresh, why = original(*args)
        fresh = copy.deepcopy(fresh)
        if damage == "list":
            rows = list(rows)
        elif damage == "duplicate":
            fresh += fresh
        else:
            fresh[0]["output_occurrence"]["index" if damage == "bounds" else "of"] = 999
        return rows, fresh, why
    monkeypatch.setattr(engine, "_record_compress_commit_proof", capture)
    engine.compress(before, force=True)
    assert emissions(engine) == []
    assert count(engine, "emission_no_descriptors." + reason) == 1


def test_verbatim_reingest_digest_and_merged_recovery_gap(engines, monkeypatch):
    engine = engines()
    _, returned = compress_path(engine, monkeypatch, "recovery")
    row = emissions(engine)[0]
    engine._ingest_messages(copy.deepcopy(returned))
    assert ledger.identity_digest(SALT, returned[0]) == row["descriptor_digest"]
    assert ledger.identity_digest(SALT, {**returned[0], "content": returned[0]["content"] + "\n\nhost suffix"}) != row["descriptor_digest"]


def test_objective_maps_to_scaffold(engines, monkeypatch):
    engine = engines()
    before = prepare_path(engine, monkeypatch)
    original = engine._compress_impl

    def objective(*a, **kw):
        result = original(*a, **kw)
        engine._pending_emission_candidates[0]["kind"] = "objective"
        return result
    monkeypatch.setattr(engine, "_compress_impl", objective)
    engine.compress(before, force=True)
    assert emissions(engine)[0]["kind"] == "scaffold"


def adopt(engine, source="S"):
    return engine._continue_in_place_compression_boundary(source, dict(platform="cli", conversation_id="conv"))


@pytest.mark.parametrize("first", ["adopted", "refused"])
def test_signals_and_copies_move_once_and_never_reverse(engines, monkeypatch, first):
    engine = engines()
    compress_path(engine, monkeypatch)
    clone = copy.copy(engine)
    for target in (engine, clone):
        if first == "adopted":
            assert adopt(target)
            target.record_rejected_compaction()
        else:
            target.record_rejected_compaction()
            assert adopt(target)
    assert {r["state"] for r in emissions(engine)} == {first}
    assert {v["state"] for v in emission_views(engine)} == {first}
    assert count(engine, "emission_" + first) == 1
    if first == "adopted":
        assert count(engine, "refused_after_adopted") == 2


@pytest.mark.parametrize("drift", [False, True])
def test_rotation_adopts_compressing_source(engines, monkeypatch, drift):
    engine = engines()
    before, _ = compress_path(engine, monkeypatch)
    engine.on_session_end("S", before)
    if drift:
        engine._dag.add_node(SummaryNode(session_id="S", summary="source summary", source_ids=[1]))
        engine._session_id = "side-channel"
    engine._continue_compression_boundary("child", "S", dict(platform="cli", conversation_id="conv"))
    assert engine._session_id == "child"
    assert emissions(engine)[0]["state"] == "adopted"
    assert count(engine, "emission_adopted") == 1


def test_false_and_skipped_boundaries_stay_proposed(engines, monkeypatch):
    engine = engines()
    compress_path(engine, monkeypatch)
    assert not engine._continue_in_place_compression_boundary("wrong", {})
    engine._session_id = "unrelated"
    engine._continue_compression_boundary("child", "missing", dict(conversation_id="other"))
    assert emissions(engine)[0]["state"] == "proposed"
    assert count(engine, "emission_boundary_unadopted") == 2
    assert engine._emission_handle is None


@pytest.mark.parametrize("reset", ["session_reset", "session_start"])
@pytest.mark.parametrize("move", ["adopted", "refused"])
def test_reset_after_a_move_counts_no_boundary(engines, monkeypatch, move, reset):
    engine = engines()
    compress_path(engine, monkeypatch)
    if move == "adopted":
        assert adopt(engine)
    else:
        engine.record_rejected_compaction()
    if reset == "session_reset":
        engine.on_session_reset()
    else:
        engine.on_session_start("next", platform="cli", conversation_id="next-conv")
    assert emissions(engine)[0]["state"] == move
    assert count(engine, "emission_boundary_unadopted") == 0
    assert engine._emission_handle is None


@pytest.mark.parametrize("conversation", ["conv", "other"])
def test_in_place_adoption_needs_the_compressing_conversation(engines, monkeypatch, conversation):
    engine = engines()
    compress_path(engine, monkeypatch)
    if conversation != "conv":  # a same-session rebind keeps the handle
        engine.on_session_start("S", platform="cli", conversation_id=conversation)
    engine.on_session_start("S", boundary_reason="compression", old_session_id="S", platform="cli",
                            conversation_id=conversation)
    assert emissions(engine)[0]["state"] == ("adopted" if conversation == "conv" else "proposed")
    assert count(engine, "emission_signal_unmatched") == (0 if conversation == "conv" else 1)


def test_bypass_source_boundary_counts_once(engines, monkeypatch, caplog):
    engine = engines()
    compress_path(engine, monkeypatch)
    engine._lcm_session_last_bypassed["S"] = True
    with caplog.at_level(logging.INFO):
        engine.on_session_start("child", boundary_reason="compression", old_session_id="S",
                                platform="cli", conversation_id="conv")
    assert "source session bypasses LCM storage" in caplog.text
    assert emissions(engine)[0]["state"] == "proposed"
    assert count(engine, "emission_boundary_unadopted") == 1
    assert engine._emission_handle is None


@pytest.mark.parametrize("path", ["bypass", "identical", "write_failure"])
def test_new_call_invalidates_old_signal(engines, monkeypatch, path):
    engine = engines()
    compress_path(engine, monkeypatch)
    if path == "write_failure":
        def fail(*a, **kw):
            raise ValueError("write sentinel")
        monkeypatch.setattr(ledger, "insert_emissions", fail)
    before = prepare_path(engine, monkeypatch, "normal" if path == "write_failure" else path)
    engine.compress(before, force=True)
    engine._session_stateless = False
    assert adopt(engine)
    assert emissions(engine)[0]["state"] == "proposed"
    assert count(engine, "emission_signal_unmatched") == 1


def test_parent_handle_survives_child_bypass(engines, monkeypatch):
    engine = engines()
    compress_path(engine, monkeypatch)
    handle = engine._emission_handle

    engine._mark_thread_context_stateless("child")
    try:
        assert engine.compress([]) == []
    finally:
        engine._clear_thread_context_stateless("child")
    assert engine._emission_handle is handle
    assert adopt(engine)
    assert emissions(engine)[0]["state"] == "adopted"


def test_signal_matching_reset_and_fresh_clones(engines, monkeypatch):
    engine = engines()
    compress_path(engine, monkeypatch)
    handle = engine._emission_handle
    engine._emission_handle = handle._replace(compress_call=handle.compress_call - 1)
    assert adopt(engine)
    engine._emission_handle = handle._replace(conversation_key="other")
    engine.record_rejected_compaction()
    assert emissions(engine)[0]["state"] == "proposed"
    assert count(engine, "emission_signal_unmatched") == 2
    engine._emission_handle = handle
    for clone in (engine.clone_for_agent(), copy.deepcopy(engine)):
        try:
            assert clone._emission_handle is None
        finally:
            clone.shutdown()
    engine._reset_session_scoped_runtime_state()
    assert engine._emission_handle is None


def test_open_transaction_is_never_joined(engines, monkeypatch):
    engine = engines()
    before = prepare_path(engine, monkeypatch, "stub")
    with engine._store.atomic(immediate=True):
        returned = engine.compress(before, force=True)
        assert returned[0]["content"] == "generated summary"
        assert engine._store._conn.in_transaction
        assert emissions(engine) == []
        assert count(engine, "emission_skipped_in_transaction") == 1
    assert engine._emission_handle is None


def test_busy_exit_wait_is_bounded(engines, monkeypatch):
    engine = engines()
    before = prepare_path(engine, monkeypatch)
    write = engine._compress_exit_write
    waits = []

    def reserved(*args):
        locker = sqlite3.connect(engine._store.db_path)
        locker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        try:
            write(*args)
        finally:
            waits.append(time.monotonic() - started)
            locker.rollback()
            locker.close()
    monkeypatch.setattr(engine, "_compress_exit_write", reserved)
    returned = engine.compress(before, force=True)
    assert returned[0]["content"] == "generated summary"
    assert count(engine, "emission_busy") == 1 and emissions(engine) == []
    assert .07 <= waits[0] < .9
    assert engine._last_emission_write_ms["ledger"] >= 70


@pytest.mark.parametrize("site,counter", [("prepare", "emission_prepare_failed"), ("insert_emissions", "emission_write_failed"),
                                         ("write_view", "emission_write_failed"), ("set_emission_state", "emission_state_failed")])
def test_ordinary_failures_count_warn_once_and_preserve_output(engines, monkeypatch, caplog, site, counter):
    engine = engines()
    if site == "set_emission_state":
        compress_path(engine, monkeypatch)

    def fail(*a, **kw):
        raise ValueError("ledger sentinel")
    monkeypatch.setattr(engine if site == "prepare" else ledger, "_ledger_view_entries" if site == "prepare" else site, fail)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            if site == "set_emission_state":
                engine.record_rejected_compaction()
            else:
                _, returned = compress_path(engine, monkeypatch)
                assert returned[0]["content"] == "generated summary"
    assert count(engine, counter) == 2
    warnings = [r.getMessage() for r in caplog.records if "ledger." + counter in r.getMessage()]
    assert len(warnings) == 1 and "compress output unaffected" in warnings[0]
    assert all(r["state"] == "proposed" for r in emissions(engine))
    if site != "set_emission_state":
        assert emissions(engine) == emission_views(engine) == []


def test_interrupt_rolls_back_and_finishes_budget(engines, monkeypatch):
    engine = engines()
    before = prepare_path(engine, monkeypatch)
    finish, finished = engine._finish_foreground_budget, []

    def interrupt(*a, **kw):
        raise KeyboardInterrupt("exit sentinel")

    def finishing(*a, **kw):
        finished.append(True)
        return finish(*a, **kw)
    monkeypatch.setattr(ledger, "write_view", interrupt)
    monkeypatch.setattr(engine, "_finish_foreground_budget", finishing)
    with pytest.raises(KeyboardInterrupt, match="exit sentinel"):
        engine.compress(before, force=True)
    assert emissions(engine) == emission_views(engine) == []
    assert finished == [True] and not engine._store._conn.in_transaction
    assert hasattr(engine, "_emission_handle") and engine._emission_handle is None


def test_rotation_skip_without_handle_counts_unadopted(engines):
    engine = engines()
    engine._session_id = "unrelated"
    engine._continue_compression_boundary("child", "missing", dict(conversation_id="other"))
    assert count(engine, "emission_boundary_unadopted") == 1
