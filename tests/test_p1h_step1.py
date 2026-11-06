"""#2293 step 1: attempt-local folded-tail reads preserve durable lineage proof."""

import copy
import builtins
import json
import logging
import sqlite3
from contextlib import contextmanager

import pytest

from hermes_lcm import engine as engine_module, tokens
from tests.test_host_uid_shadow import _m
from tests.test_issue_2055_ingest_ledger_writes import engines as engines


def folded(engine):
    row = _m("user", "folded source", 10.)
    engine._ingest_messages([row])
    assert engine._write_folded_tail_lineage(row, 1)
    return row


@pytest.mark.parametrize("writer", ["write", "clear", "metadata", "factory", "restore"])
def test_folded_tail_write_invalidates_attempt(engines, monkeypatch, writer):
    engine = engines()
    row = folded(engine)
    store = engine._store
    key = engine._folded_tail_lineage_metadata_key()
    original = store.read_metadata_json(key)
    with store.ingest_atomic():
        assert engine._load_folded_tail_lineage([row]) is not None
        assert store._ingest_folded_tail_reads[key] == original
        if writer == "write":
            # Same source, a changed folded digest: the writer is still real.
            assert engine._write_folded_tail_lineage({**row, "content": "new fold"}, 1)
        elif writer == "clear":
            assert engine._clear_folded_tail_lineage()
        elif writer == "metadata":
            store.write_metadata_json([key], "null")
        elif writer == "factory":
            store._append_protected_batch("S", [_m("assistant", "factory")],
                                          metadata_factory=lambda _row, _id: [(key, "null")])
        else:
            # Exercise compaction.py's restore loop after the final assembly
            # misses its target, with a real metadata write at that boundary.
            monkeypatch.setattr(engine, "_leading_anchor_count", lambda _rows: 0)
            monkeypatch.setattr(engine, "_fresh_tail_start", lambda _rows: 0)
            monkeypatch.setattr(engine, "_generated_context_carrier_remainder", lambda _row: None)
            monkeypatch.setattr(engine, "_effective_assembly_token_cap", lambda: None)
            monkeypatch.setattr(engine, "_survival_host_overhead", lambda *_args: 0)
            monkeypatch.setattr(engine, "_committed_replay_drops", lambda *_args: (False,))
            measures = iter([0, 10**9])
            monkeypatch.setattr(engine, "_survival_measure", lambda _rows: next(measures))

            def assemble(*_args, **kwargs):
                if kwargs.get("persist", True):
                    store.write_metadata_json([key], "null")
                    # Read the temporary value before compaction restores it.
                    assert engine._load_folded_tail_lineage([row]) is None
                return [row]

            monkeypatch.setattr(engine, "_assemble_committed_compaction_context", assemble)
            engine._config.large_output_active_replay_stubbing_enabled = True
            engine._config.large_output_externalization_enabled = True
            engine._config.proactive_recall_enabled = False
            assert engine._stub_first_exit([row], [row], [row], [row], 1) is None
        assert store._ingest_folded_tail_reads is None
        expected = writer == "restore"
        assert (engine._load_folded_tail_lineage([row]) is not None) == expected


def test_folded_cache_savepoint_rollback(engines):
    engine = engines()
    row = folded(engine)
    store = engine._store
    key = engine._folded_tail_lineage_metadata_key()
    with store.ingest_atomic():
        assert engine._load_folded_tail_lineage([row]) is not None
        store._conn.execute("SAVEPOINT cache_probe")
        store.write_metadata_json([key], "null")
        assert engine._load_folded_tail_lineage([row]) is None
        store._conn.execute("ROLLBACK TO cache_probe")
        store._conn.execute("RELEASE cache_probe")
        assert engine._load_folded_tail_lineage([row]) is not None
        assert store._ingest_folded_tail_reads is None


@pytest.mark.parametrize("rollback", [False, True])
def test_folded_cache_lifetime_and_outside_reads(engines, monkeypatch, rollback):
    engine = engines()
    row = folded(engine)
    store = engine._store
    reads = []
    real = store.read_metadata_json

    def read(key):
        reads.append(key)
        return real(key)

    monkeypatch.setattr(store, "read_metadata_json", read)
    assert getattr(store, "_ingest_folded_tail_reads", None) is None
    for _ in range(2):
        assert engine._load_folded_tail_lineage([row]) is not None
    assert len(reads) == 2
    try:
        with store.ingest_atomic():
            assert store._ingest_folded_tail_reads == {}
            for _ in range(2):
                assert engine._load_folded_tail_lineage([row]) is not None
            assert len(reads) == 3
            if rollback:
                raise RuntimeError("rollback probe")
    except RuntimeError:
        assert rollback
    assert store._ingest_folded_tail_reads is None
    assert engine._load_folded_tail_lineage([row]) is not None
    assert len(reads) == 4
    with store.ingest_atomic():
        assert store._ingest_folded_tail_reads == {}
        assert engine._load_folded_tail_lineage([row]) is not None
    assert len(reads) == 5
    assert store._ingest_folded_tail_reads is None


def test_folded_cache_retry_is_fresh(engines, monkeypatch):
    engine = engines()
    row = folded(engine)
    real = engine._ingest_messages_transaction
    caches = []

    def transaction(*args, **kwargs):
        cache = engine._store._ingest_folded_tail_reads
        assert cache == {}
        caches.append(cache)
        assert engine._load_folded_tail_lineage([row]) is not None
        if len(caches) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_ingest_messages_transaction", transaction)
    engine._ingest_messages([row, _m("assistant", "after retry", 11.)])
    assert len(caches) == 2 and caches[0] is not caches[1]
    assert engine._store._ingest_folded_tail_reads is None


def test_update_metadata_cannot_join_ingest(engines):
    engine = engines()
    row = folded(engine)
    store = engine._store
    key = engine._folded_tail_lineage_metadata_key()
    original = copy.deepcopy(store.read_metadata_json(key))
    with store.ingest_atomic():
        assert engine._load_folded_tail_lineage([row]) is not None
        with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
            store.update_metadata_json(key, lambda _value: None)
        assert engine._load_folded_tail_lineage([row]) is not None
    assert store.read_metadata_json(key) == original
    store.update_metadata_json(key, lambda _value: None)
    assert engine._load_folded_tail_lineage([row]) is None


def test_failed_metadata_write_disables_cache(engines):
    engine = engines()
    row = folded(engine)
    store = engine._store
    key = engine._folded_tail_lineage_metadata_key()
    with store.ingest_atomic():
        assert engine._load_folded_tail_lineage([row]) is not None
        with pytest.raises(sqlite3.ProgrammingError):
            store.write_metadata_json([key], object())
        assert store._ingest_folded_tail_reads is None
        assert engine._load_folded_tail_lineage([row]) is not None
        assert json.loads(store._conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0])


def test_view_only_phases_precede_begin_without_side_effects(engines, monkeypatch):
    engine = engines()
    engine._ingest_messages([_m("user", "seed", 10.)])
    rows = [_m("user", "seed", 10.), _m("assistant", "new reply", 11.)]
    active = [False]
    calls = []
    real_precompute = engine._precompute_ingest_view_only

    def forbidden(*_args, **_kwargs):
        if active[0]:
            raise AssertionError("precompute side effect")

    def precompute(messages):
        assert not engine._store._conn.in_transaction
        active[0] = True
        try:
            # AssertionError inside precompute is deliberately swallowed by
            # its fallback; the event list below must also remain empty.
            with monkeypatch.context() as patch:
                def record(*args, **kwargs):
                    calls.append((args, kwargs))
                    forbidden()
                patch.setattr(builtins, "open", record)
                patch.setattr(logging.Logger, "_log", record)
                for name in dir(type(engine._store)):
                    if not name.startswith('_') and callable(getattr(type(engine._store), name)):
                        patch.setattr(engine._store, name, record)
                hints = real_precompute(messages)
        finally:
            active[0] = False
        assert hints["inputs"] == messages
        assert 1 in hints["estimates"] and 1 in hints["markers"] and hints["names"] == {}
        return hints

    sql = []
    engine._store._conn.set_trace_callback(lambda statement: sql.append(statement) if active[0] else None)
    monkeypatch.setattr(engine, "_precompute_ingest_view_only", precompute)
    try:
        engine._ingest_messages(rows)
    finally:
        engine._store._conn.set_trace_callback(None)
    assert not calls and not sql
    assert engine._store.get_session_messages("S")[-1]["content"] == "new reply"


@pytest.mark.parametrize("change", ["view", "protected", "counter", "precompute-error"])
def test_view_only_hints_fall_back_under_lock(engines, monkeypatch, change):
    engine = engines()
    seed = _m("user", "seed", 10.)
    engine._ingest_messages([seed])
    new = _m("assistant", "new reply", 11.)
    view = [seed, new]
    counts = []
    real_count = engine_module.count_message_tokens

    def count(row):
        locked = engine._store._conn.in_transaction
        counts.append((locked, copy.deepcopy(row)))
        if change == "precompute-error" and not locked:
            raise ValueError("speculative error")
        return real_count(row)

    monkeypatch.setattr(engine_module, "count_message_tokens", count)
    real_atomic = engine._store.ingest_atomic

    @contextmanager
    def atomic(**kwargs):
        if change == "view":
            new["content"] = "changed while waiting"
        elif change == "counter":
            monkeypatch.setattr(tokens, "_encoder_generation", tokens._encoder_generation + 1)
        with real_atomic(**kwargs):
            yield

    monkeypatch.setattr(engine._store, "ingest_atomic", atomic)
    if change == "protected":
        real_protect = engine._protect_ingest_rows

        def protect(**kwargs):
            protected, replay = real_protect(**kwargs)
            protected[0] = {**protected[0], "content": "protected replacement"}
            return protected, replay

        monkeypatch.setattr(engine, "_protect_ingest_rows", protect)
    engine._ingest_messages(view)
    assert any(not locked for locked, _ in counts)
    assert any(locked for locked, _ in counts)
    stored = engine._store.get_session_messages("S")[-1]
    assert stored["token_estimate"] == real_count(stored)
    expected = {"view": "changed while waiting", "protected": "protected replacement"}.get(change, "new reply")
    assert stored["content"] == expected


def test_unchanged_view_reuses_token_and_marker_hints(engines, monkeypatch):
    engine = engines()
    seed = _m("user", "seed", 10.)
    engine._ingest_messages([seed])
    counts, markers, names = [], [], []
    for name, target in (("count_message_tokens", counts), ("_tool_result_names", names)):
        real = getattr(engine_module, name)

        def tracked(*args, _real=real, _target=target, **kwargs):
            _target.append(engine._store._conn.in_transaction)
            return _real(*args, **kwargs)

        monkeypatch.setattr(engine_module, name, tracked)
    real_marker = engine._lcm_generated_marker

    def marker(row):
        if row.get("content") == "fresh reply":
            markers.append(engine._store._conn.in_transaction)
        return real_marker(row)

    monkeypatch.setattr(engine, "_lcm_generated_marker", marker)
    engine._ingest_messages([seed, _m("assistant", "fresh reply", 11.)])
    assert counts == [False]
    # Other replay helpers may build their own map; the protection call reuses it.
    assert names and names[0] is False
    assert markers and markers[0] is False
