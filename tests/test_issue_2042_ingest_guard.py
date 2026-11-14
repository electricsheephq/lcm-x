"""#2042 step 2: serialize cursor-zero reconciliation with eager writers (I1/I2/I8/I12)."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by name
from __future__ import annotations

import copy
import multiprocessing
import queue
import sqlite3
import sys
import threading
import traceback
from collections import Counter
from contextlib import ExitStack, contextmanager

import pytest

if "hermes_lcm" not in sys.modules:
    from tests import conftest as _bootstrap  # noqa: F401

from hermes_lcm import unended_catchup as catchup
from hermes_lcm.engine import LCMEngine
from tests.test_issue_1931_delegate_children import child_view
from tests.test_issue_2042_unended_catchup import (
    SID, Case, case, failed_rows, fresh_process_caches,  # noqa: F401 (fixtures)
)


def _catchup_writer(config, home, rows, begin, release, observations):
    """Use the real catch-up store path, with a zero-budget first lock attempt."""
    try:
        class ControlledStore(catchup.MessageStore):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.connection.execute("PRAGMA busy_timeout=0")
                atomic = self.atomic
                first = True

                @contextmanager
                def controlled_atomic(*args, **kw):
                    nonlocal first
                    if not first:
                        with atomic(*args, **kw):
                            yield
                        return
                    first = False
                    observations.put("ready")  # Protection finished, no write transaction yet.
                    assert begin.wait(30), "catch-up was not released"
                    with ExitStack() as stack:
                        try:
                            stack.enter_context(atomic(*args, **kw))
                        except sqlite3.OperationalError as exc:
                            assert "locked" in str(exc).lower(), str(exc)
                            assert not self.connection.in_transaction
                            observations.put("busy")
                            assert release.wait(30), "ingest did not release its write lock"
                            stack.enter_context(atomic(*args, **kw))
                        yield

                self.atomic = controlled_atomic

        before = Counter(catchup.unended_catchup_counts())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(catchup, "MessageStore", ControlledStore)
            catchup._store_candidate(LCMEngine, config, home, config.database_path,
                                    {"id": SID, "source": "desktop"}, rows, 128000)
        observations.put(dict(Counter(catchup.unended_catchup_counts()) - before))
    except BaseException:
        observations.put({"error": traceback.format_exc()})


@pytest.mark.parametrize("writer_kind", ["connection", "process"])
@pytest.mark.parametrize("order", ["before-begin", "during-hold", "after-commit"])
def test_catchup_and_eager_ingest_commit_orders(case, monkeypatch, writer_kind, order):
    live = case.bind(SID)
    assert (live._ingest_cursor, live._ingest_cursor_needs_reconcile) == (0, False)
    recovered = failed_rows(False)
    history = recovered + [{"role": "user", "content": "next seam request"},
                           {"role": "assistant", "content": "next seam answer"}]
    if writer_kind == "process":
        ctx = multiprocessing.get_context("spawn")
        begin, release, observations = ctx.Event(), ctx.Event(), ctx.Queue()
        worker_type = ctx.Process
    else:
        begin, release, observations = threading.Event(), threading.Event(), queue.Queue()
        worker_type = threading.Thread
    worker = worker_type(target=_catchup_writer, args=(
        case.config, str(case.home.resolve()), recovered, begin, release, observations))
    probes, transactions = [], []
    probe = live._store.has_session_messages

    def observed_probe(sid):
        assert live._store.connection.in_transaction
        found = probe(sid)
        probes.append(found)
        return found

    transaction = live._ingest_messages_transaction

    def held(*args, **kwargs):
        assert live._store.connection.in_transaction
        transactions.append(True)
        if order == "during-hold" and len(transactions) == 1:
            begin.set()
            assert observations.get(timeout=30) == "busy"
        return transaction(*args, **kwargs)

    monkeypatch.setattr(live._store, "has_session_messages", observed_probe)
    monkeypatch.setattr(live, "_ingest_messages_transaction", held)
    try:
        worker.start()
        assert observations.get(timeout=30) == "ready"
        if order == "before-begin":
            assert not live._store.connection.in_transaction
            begin.set()
            assert observations.get(timeout=30) == {"recovered_candidates": 1, "recovered_rows": 2}
            case.assert_rows(recovered)
        live.ingest(copy.deepcopy(history))
        assert transactions == [True]
        assert probes[0] is (order == "before-begin")
        assert not live._store.connection.in_transaction
        if order != "before-begin":
            # The waiting writer rechecks under its own lock and stores no copy.
            begin.set()
            release.set()
            assert observations.get(timeout=30) == {"raced": 1}
        worker.join(30)
        assert not worker.is_alive()
        if writer_kind == "process":
            assert worker.exitcode == 0
        case.assert_rows(history)
        live.ingest(copy.deepcopy(history))
        case.assert_rows(history)
    finally:
        begin.set()
        release.set()
        worker.join(30)
        if writer_kind == "process":
            if worker.is_alive():
                worker.terminate()
                worker.join(10)
            observations.close()
            observations.join_thread()


def test_n4_stamped_store_and_unstamped_live_history(case):
    live = case.bind(SID)
    assert not live._ingest_cursor_needs_reconcile
    stamped = failed_rows(True)
    catchup._store_candidate(LCMEngine, case.config, str(case.home.resolve()), case.db_path,
                            {"id": SID, "source": "desktop"}, stamped, 128000)
    original = case.assert_rows(stamped)
    history = failed_rows(False) + [{"role": "user", "content": "mixed next request"},
                                  {"role": "assistant", "content": "mixed next answer"}]
    live.ingest(copy.deepcopy(history))
    stored = case.assert_rows(history)
    assert [r["store_id"] for r in stored[:2]] == [r["store_id"] for r in original]
    # Review F1: reconcile keeps the stored copy's metadata on the prefix it adopts.
    assert [(r["observed_at"], r["observed_at_source"]) for r in stored[:2]] == [
        (r["observed_at"], r["observed_at_source"]) for r in original]
    assert all(r["observed_at"] is not None for r in stored[:2])  # the stamped copy's times survive
    assert live._last_ingest_reconciliation["cursor"] == 2
    live.ingest(copy.deepcopy(history))
    case.assert_rows(history)


def test_probe_error_reconciles_stored_rows_in_same_attempt(case, monkeypatch):
    live = case.bind(SID)
    assert (live._ingest_cursor, live._ingest_cursor_needs_reconcile) == (0, False)
    recovered = failed_rows(False)
    catchup._store_candidate(LCMEngine, case.config, str(case.home.resolve()), case.db_path,
                            {"id": SID, "source": "desktop"}, recovered, 128000)
    original = case.assert_rows(recovered)
    history = recovered + [{"role": "user", "content": "probe error next request"},
                           {"role": "assistant", "content": "probe error next answer"}]
    attempts, probes, scheduled = [], [], []
    transaction = live._ingest_messages_transaction
    restamp = live._rotation_restamp_metadata
    probe = live._store.has_session_messages

    def attempted(*args, **kwargs):
        assert live._store.connection.in_transaction
        attempts.append(True)
        return transaction(*args, **kwargs)

    def failed_probe(sid):
        assert sid == SID
        assert live._store.connection.in_transaction
        if probes:
            return probe(sid)  # Leave the existing placeholder-budget lookup intact.
        probes.append(True)
        raise sqlite3.OperationalError("injected session probe read failure")

    def reconciled(*args, **kwargs):
        scheduled.append(live._ingest_cursor_needs_reconcile)
        return restamp(*args, **kwargs)

    monkeypatch.setattr(live, "_ingest_messages_transaction", attempted)
    monkeypatch.setattr(live._store, "has_session_messages", failed_probe)
    monkeypatch.setattr(live, "_rotation_restamp_metadata", reconciled)
    live.ingest(copy.deepcopy(history))
    assert attempts == probes == scheduled == [True]
    stored = case.assert_rows(history)
    assert [r["store_id"] for r in stored[:2]] == [r["store_id"] for r in original]
    # Review F1: reconcile keeps the stored copy's metadata on the prefix it adopts.
    assert [(r["observed_at"], r["observed_at_source"]) for r in stored[:2]] == [
        (r["observed_at"], r["observed_at_source"]) for r in original]
    assert live._last_ingest_reconciliation["cursor"] == 2
    assert live._ingest_cursor == len(history)
    assert not live._store.connection.in_transaction
    live.ingest(copy.deepcopy(history))
    case.assert_rows(history)
    assert probes == [True]  # A positive cursor no longer needs the guard's lookup.


def _without_guard(monkeypatch, engine):
    """Hide rows from just the first probe to exercise the previous append route."""
    lookup = engine._store.has_session_messages
    first = True

    def old_entry(sid):
        nonlocal first
        if first:
            first = False
            return False
        return lookup(sid)

    monkeypatch.setattr(engine._store, "has_session_messages", old_entry)


def _identities(rows):
    return Counter((r["role"], r["content"], r.get("tool_calls_json"), r.get("tool_call_id")) for r in rows)


@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "unstamped"])
def test_delegate_child_reconciles_without_dropping_old_append_rows(tmp_path, monkeypatch, stamped):
    view = child_view(1)
    if stamped:
        for i, row in enumerate(view):
            row["timestamp"] = 1_900_000_000 + i
    results = []
    for legacy in (True, False):
        child_case = Case(tmp_path / str(legacy))
        try:
            parent = child_case.bind("parent")
            child = child_case.bind("child")
            assert (child._ingest_cursor, child._ingest_cursor_needs_reconcile) == (0, False)
            link = dict(parent_session_id="parent", source="parent_frame", child_subagent_id="sa-2042",
                        parent_subagent_id="", child_role="")
            # The same off-current append used by aux_session.py's delegate-child path.
            assert parent._append_delegate_child_rows("child", copy.deepcopy(view[:3]), link) == 3
            original = child_case.assert_rows(view[:3], "child")
            with monkeypatch.context() as patch:
                if legacy:
                    _without_guard(patch, child)
                child.ingest(copy.deepcopy(view))
            stored = child_case.rows("child")
            assert [r["store_id"] for r in stored[:3]] == [r["store_id"] for r in original]
            results.append(stored)
            if not legacy:
                child_case.assert_rows(view, "child")
                assert child._last_ingest_reconciliation["cursor"] == 3
                child.ingest(copy.deepcopy(view))
                child_case.assert_rows(view, "child")
        finally:
            child_case.close()
    # Every occurrence the old append added remains represented, including its call/result fields.
    assert _identities(results[0][3:]) <= _identities(results[1])
    assert _identities(results[0]) - _identities(results[1]) <= _identities(results[1][:3])


def test_brand_new_first_ingest_keeps_rows_and_counters(tmp_path, monkeypatch):
    outcomes = []
    for legacy in (True, False):
        fresh = Case(tmp_path / str(legacy))
        try:
            live = fresh.bind(SID)
            with monkeypatch.context() as patch:
                if legacy:
                    _without_guard(patch, live)
                live.ingest(failed_rows(False))
            rows = fresh.assert_rows(failed_rows(False))
            outcomes.append((_identities(rows), live._ingest_cursor, live._ingest_cursor_needs_reconcile,
                             dict(getattr(live, "_host_uid_counters", {})),
                             dict(getattr(live, "_ingest_ledger_counters", {})), live._ignored_message_count))
        finally:
            fresh.close()
    assert outcomes[0] == outcomes[1]
    assert outcomes[1][1:3] == (2, False)
