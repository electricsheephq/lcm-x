"""#2241: detached cold tails keep their session's ingest-protection window (I1/I2)."""

import copy
import sqlite3
from pathlib import Path

import pytest

from hermes_lcm.externalize import extract_externalized_ref, load_externalized_payload
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_ingest_ledger_writes import engines as engines


WINDOW = 1_048_576


def _committed_rows(store, session="S1"):
    with sqlite3.connect(store.db_path) as reader:
        reader.row_factory = sqlite3.Row
        return [dict(row) for row in reader.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY store_id", (session,))]


def _queue_cold_tail(engines, monkeypatch, length, *, shared=False, off_current=False):
    engine = engines(sessions=[("S1", None, None), ("S2", None, None)])
    engine._config.large_output_externalization_enabled = True
    if not shared:
        engine._config.database_path = ""
    engine.on_session_start("S1", platform="cli", context_length=WINDOW, conversation_id="conv1")
    prose = "This document describes ordinary plans and useful observations. "
    text = (prose * (length // len(prose) + 1))[:length]
    view = [_m("user", "seed", 1.), _m("assistant", "reply", 2.), _m("user", text, 3.)]
    engine.ingest(copy.deepcopy(view[:1]))
    store = engine._store
    assert store.get_context_window_tokens("S1") == WINDOW
    if off_current:
        engine.on_session_start("S2", platform="api", context_length=65_536, conversation_id="conv2")

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(store, "get_range" if off_current else "get_session_count", locked)
        engine._ingest_cursor_needs_reconcile = True
        engine.on_session_end("S1", copy.deepcopy(view))
    assert len(engine._pending_session_tails) == 1
    tail = engine._pending_session_tails[0]
    assert tail["kind"] == "cold" and tail["view"] == view
    assert [row["content"] for row in _committed_rows(store)] == ["seed"]
    return engine, tail, text


def _retry_detached(engine, tail, shared):
    engine._rebind_storage_for_home(str(Path(engine._hermes_home).parent / "other-home"))
    assert (engine._store is tail["store"]) is shared
    assert tail["store"].get_context_window_tokens("S1") == 0
    assert not engine._cold_tail_is_bound(tail)
    engine._retry_pending_session_tails()
    assert not engine._pending_session_tails
    rows = _committed_rows(tail["store"])
    assert [row["content"] for row in rows[:2]] == ["seed", "reply"]
    assert len(rows) == 3
    assert all(row["source"] == "cli" and row["conversation_id"] == "conv1" for row in rows)
    assert [row["observed_at"] for row in rows] == [1., 2., 3.]
    if not shared:
        assert not _committed_rows(engine._store)
    return rows[-1]


@pytest.mark.parametrize("shared", [False, True], ids=["separate_store", "shared_store"])
def test_detached_tail_keeps_its_window_after_rebind(engines, monkeypatch, shared):
    engine, tail, text = _queue_cold_tail(engines, monkeypatch, 150_000, shared=shared)
    row = _retry_detached(engine, tail, shared)
    assert not extract_externalized_ref(row["content"])
    assert row["content"] == text


@pytest.mark.parametrize("shared", [False, True], ids=["separate_store", "shared_store"])
def test_detached_tail_over_its_window_floor_still_externalizes(engines, monkeypatch, shared):
    engine, tail, text = _queue_cold_tail(engines, monkeypatch, 450_000, shared=shared)
    row = _retry_detached(engine, tail, shared)
    ref = extract_externalized_ref(row["content"])
    assert ref
    assert load_externalized_payload(ref, config=engine._config, hermes_home=tail["hermes_home"])["content"] == text


def test_bound_tail_window_read_is_unchanged(engines, monkeypatch):
    engine, tail, text = _queue_cold_tail(engines, monkeypatch, 150_000, off_current=True)
    store = tail["store"]
    assert engine._cold_tail_is_bound(tail)
    store.set_context_window_tokens("S1", 65_536)
    read, reads = store.get_context_window_tokens, []

    def observed(session_id):
        reads.append(session_id)
        return read(session_id)

    with monkeypatch.context() as patch:
        patch.setattr(store, "get_context_window_tokens", observed)
        engine._retry_pending_session_tails()
    assert reads == ["S1"]
    assert not engine._pending_session_tails
    rows = _committed_rows(store)
    assert len(rows) == 3
    ref = extract_externalized_ref(rows[-1]["content"])
    assert ref
    assert load_externalized_payload(ref, config=engine._config, hermes_home=tail["hermes_home"])["content"] == text
    assert not _committed_rows(store, "S2")
