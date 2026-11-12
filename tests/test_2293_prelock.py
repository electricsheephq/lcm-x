"""#2293: classification runs before the write lock; the lock is taken at the first write and rechecked.

The ingest attempt takes BEGIN IMMEDIATE lazily at its first write. At that point it re-reads the store's write
generation (total_changes, PRAGMA data_version, PRAGMA schema_version); if anything changed since the attempt's
first read, it rolls back to its savepoint and classifies again under the held lock (design M5's fallback).

Pinned tests:
  (a) differential: every ingest-trace scenario observes the same protocol eager and lazy, apart from where the
      transaction-control statements fall;
  (b) forced change: a second connection commits a classification-relevant change between the pre-lock run and
      BEGIN; the lazy ingest reclassifies once and stores exactly what today's eager path stores;
  (c) no change: no reclassify, no poisoned attempt, and no lock held while classifying;
  plus a pre-lock write is denied and forces an in-lock reclassify, and a busy BEGIN keeps today's retry behaviour.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore

SESSION = "s-2293-prelock"
OTHER = "s-2293-other"
PAD = " lorem ipsum dolor" * 12
CLOCK_COLUMNS = {"ingested_at", "timestamp", "created_at", "updated_at", "completed_at", "current_bound_at", "last_finalized_at",
                 "debt_updated_at", "last_maintenance_attempt_at", "last_rollover_at", "last_reset_at",
                 "built_at", "failed_at", "lease_expires_at", "last_built_at"}
SKIP_TABLES = {"lcm_migration_state", "lcm_lifecycle_state"}


@pytest.fixture(autouse=True)
def _pinned_salt(monkeypatch):
    """Each store draws a random digest salt; pin it so two stores built the same way compare byte for byte."""
    import hermes_lcm.store as store_module
    monkeypatch.setattr(store_module.secrets, "token_bytes", lambda n=32: bytes(range(n)))


def _clock_key(table, row, cols):
    return table == "metadata" and re.search(r"_at(?::|$)", str(row[cols.index("key")]))


def _u(text, ts=None):
    row = {"role": "user", "content": text + PAD}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _a(text, ts=None):
    row = {"role": "assistant", "content": text + PAD}
    if ts is not None:
        row["timestamp"] = float(ts)
    return row


def _tool_pair(call_id, ts):
    return [{"role": "assistant", "content": "", "timestamp": float(ts), "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{\"p\": 1}"}}]},
            {"role": "tool", "tool_call_id": call_id, "content": f"result {call_id}" + PAD, "timestamp": float(ts) + 0.5}]


def _engine(root: Path, session=SESSION, *, prelock: bool | None = None):
    engine = LCMEngine(config=LCMConfig(database_path=str(root / "lcm.db"),
                                        large_output_externalization_enabled=False),
                       hermes_home=str(root / ("home-" + session)))
    if prelock is not None:
        engine._store._ingest_prelock = prelock
    engine.on_session_start(session, platform="cli", context_length=1_000_000)
    return engine


def _dump(db: Path) -> dict:
    """Every durable table, row ids included; wall-clock columns and FTS shadows excluded."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        out = {}
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            if table in SKIP_TABLES or table.startswith("sqlite_stat") or "_fts" in table:
                continue
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')") if r[1] not in CLOCK_COLUMNS]
            if not cols:
                continue
            select = ", ".join(f'"{c}"' for c in cols)
            try:
                rows = conn.execute(f'SELECT {select} FROM "{table}" ORDER BY rowid').fetchall()
            except sqlite3.OperationalError:  # WITHOUT ROWID
                rows = sorted(conn.execute(f'SELECT {select} FROM "{table}"').fetchall(), key=repr)
            out[table] = [row for row in rows if not _clock_key(table, row, cols)]
        return out
    finally:
        conn.close()


def _rows(engine, session=SESSION):
    return Counter((r.get("role"), str(r.get("content") or ""), r.get("tool_call_id") or "")
                   for r in engine._store.get_session_messages(session))


def _counts(engine) -> Counter:
    return Counter(getattr(engine._store, "ingest_prelock_counts", Counter()))


# -- shapes: (setup views ingested first, the judged view) ------------------------------------------------------------

def _shape(name):
    base = [_u("U0 first", 1000), _a("R0 reply", 1000.5), _u("U1 second", 1001), _a("R1 reply", 1001.5)]
    if name == "warm-append":
        return [base], base + [_u("U2 third", 1002), _a("R2 reply", 1002.5)]
    if name == "tools":
        return [base], base + _tool_pair("call-1", 1002) + [_a("R2 after tool", 1003)]
    if name == "changed-prefix":
        changed = list(base)
        changed[2] = _u("U1 rewritten second", 1004)
        return [base], changed + [_a("R4 reply", 1004.5)]
    if name == "stamp-replay":
        # A cold engine sees the stored rows again (same stamps) plus a new turn: the anchor proves the replay.
        return [base], base + [_u("U5 new after replay", 1005)]
    if name == "identical":
        return [base], list(base)
    if name == "many-new":
        rows = []
        for i in range(120):
            rows += [_u(f"bulk user {i}", 2000 + i), _a(f"bulk reply {i}", 2000.5 + i)]
        return [], rows
    raise AssertionError(name)


SHAPES = ("warm-append", "tools", "changed-prefix", "stamp-replay", "identical", "many-new")


def _foreign_commit(kind: str, root: Path, view: list) -> None:
    """One classification-relevant commit by another connection (another engine or a raw connection)."""
    db = root / "lcm.db"
    if kind == "same-session-rows":
        other = _engine(root, SESSION, prelock=False)
        try:
            other.ingest(view)  # the same session's rows, stored by another engine first
        finally:
            other.shutdown()
    elif kind == "other-session-rows":
        other = _engine(root, OTHER, prelock=False)
        try:
            other.ingest([_u("other session row", 5000), _a("other reply", 5000.5)])
        finally:
            other.shutdown()
    elif kind == "observed-at-backfill":
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            conn.execute("UPDATE messages SET observed_at = 1001.0, observed_at_source = 'test' "
                         "WHERE store_id = (SELECT MIN(store_id) FROM messages WHERE session_id = ?) "
                         "AND observed_at IS NULL", (SESSION,))
            conn.execute("INSERT INTO metadata(key, value) VALUES('test-2293-generation', '1') "
                         "ON CONFLICT(key) DO UPDATE SET value = excluded.value || 'x'")
        conn.close()
    elif kind == "alt-stamp-relation":
        conn = sqlite3.connect(db, timeout=5)
        has_table = conn.execute("SELECT 1 FROM sqlite_master WHERE name='message_relations'").fetchone()
        with conn:
            if has_table:
                conn.execute("INSERT INTO message_relations(store_id, kind, related_store_id, ordinal, observed_at, "
                             "created_at) SELECT MIN(store_id), 'alt_stamp', NULL, NULL, 1003.0, 0 FROM messages "
                             "WHERE session_id = ?", (SESSION,))
            else:
                conn.execute("INSERT INTO metadata(key, value) VALUES('test-2293-generation', '1') "
                             "ON CONFLICT(key) DO UPDATE SET value = excluded.value || 'x'")
        conn.close()
    elif kind == "folded-tail-lineage":
        # A foreign write to the folded-tail lineage key that reconcile reads (and caches per attempt).
        import json as _json
        conn = sqlite3.connect(db, timeout=5)
        with conn:
            conn.execute("INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = "
                         "excluded.value", ("folded_tail_lineage:" + SESSION, _json.dumps(
                             {"version": 1, "source_store_id": 1, "folded_identity_sha256": "0" * 64,
                              "source_identity_sha256": "0" * 64})))
        conn.close()
    elif kind == "two-commits":
        _foreign_commit("observed-at-backfill", root, view)
        _foreign_commit("other-session-rows", root, view)
    else:
        raise AssertionError(kind)


def _run(root: Path, shape: str, *, prelock: bool, foreign: str | None = None, monkeypatch=None):
    """Setup views on one engine, then the judged view on a fresh (cold) or the same engine.

    With ``foreign``, the commit runs at the test seam ``MessageStore._ingest_prelock_before_begin`` (called right
    before BEGIN IMMEDIATE in both modes): eager -> before the in-lock classification; lazy -> after the pre-lock
    classification. Both paths then classify the judged view with the commit visible.
    """
    root.mkdir(parents=True, exist_ok=True)
    setups, view = _shape(shape)
    engine = _engine(root, prelock=prelock)
    try:
        for setup in setups:
            engine.ingest(setup)
        if shape == "stamp-replay":
            engine.shutdown()
            engine = _engine(root, prelock=prelock)
        before = _counts(engine)
        fired = []
        if foreign is not None:
            original = MessageStore._ingest_prelock_before_begin

            def seam(store):
                if store is engine._store and not fired:
                    fired.append(True)
                    _foreign_commit(foreign, root, view)
                return original(store)

            monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", seam)
        result = engine.ingest(view)
        if foreign is not None:
            # Restore only the seam: undo() would also drop the autouse salt pin.
            monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", original)
            assert fired, "the foreign commit never ran: no BEGIN reached the seam"
        state = dict(cursor=engine._frontier.cursor, needs_reconcile=engine._frontier.needs_reconcile,
                     failures=getattr(engine, "_consecutive_ingest_failures", 0),
                     returned=[(m.get("role"), m.get("content"), m.get("tool_call_id")) for m in result or []],
                     rows=_rows(engine))
        counts = _counts(engine) - before
    finally:
        engine.shutdown()
    return _dump(root / "lcm.db"), state, counts


# -- (a) differential over the ingest-trace corpus --------------------------------------------------------------------

_TXN = re.compile(r'^sql \S+ "(?:BEGIN(?: IMMEDIATE| DEFERRED)?|COMMIT|ROLLBACK|END|SAVEPOINT \w+|RELEASE(?: SAVEPOINT)? \w+'
                  r'|ROLLBACK TO(?: SAVEPOINT)? \w+|PRAGMA (?:main\.)?(?:data_version|schema_version))"$')


def _normalized(trace):
    return [event for event in trace if not _TXN.match(event)]


def _scenarios():
    from tests.behaviour import ingest_trace_harness as harness
    return harness.SCENARIOS


@pytest.mark.parametrize("case", _scenarios(), ids=lambda case: case.name)
def test_a_trace_differential_eager_vs_prelock(case, tmp_path, monkeypatch):
    from tests.behaviour import ingest_trace_harness as harness
    monkeypatch.setattr(MessageStore, "_ingest_prelock", False)
    eager = harness.observe(case, tmp_path / "eager")
    monkeypatch.setattr(MessageStore, "_ingest_prelock", True)
    lazy = harness.observe(case, tmp_path / "lazy")
    eager_n, lazy_n = _normalized(eager), _normalized(lazy)
    if eager_n != lazy_n:
        first = next(i for i, (x, y) in enumerate(zip(eager_n + [None], lazy_n + [None])) if x != y)
        pytest.fail(f"{case.name}: first difference at event {first}:\n eager: {eager_n[first:first + 3]}\n"
                    f" lazy:  {lazy_n[first:first + 3]}")


@pytest.mark.parametrize("shape", SHAPES)
def test_a_store_differential_eager_vs_prelock(shape, tmp_path):
    eager_db, eager_state, _ = _run(tmp_path / "eager", shape, prelock=False)
    lazy_db, lazy_state, counts = _run(tmp_path / "lazy", shape, prelock=True)
    assert lazy_state == eager_state
    assert lazy_db == eager_db
    assert counts["reclassified"] == 0 and counts["poisoned"] == 0


# -- (b) forced change between the pre-lock run and BEGIN -------------------------------------------------------------

FOREIGN = ("same-session-rows", "other-session-rows", "observed-at-backfill", "alt-stamp-relation",
           "folded-tail-lineage", "two-commits")


@pytest.mark.parametrize("foreign", FOREIGN)
@pytest.mark.parametrize("shape", ("warm-append", "tools", "changed-prefix", "stamp-replay", "many-new"))
def test_b_forced_change_reclassifies_and_matches_eager(shape, foreign, tmp_path, monkeypatch):
    eager_db, eager_state, eager_counts = _run(tmp_path / "eager", shape, prelock=False, foreign=foreign,
                                               monkeypatch=monkeypatch)
    lazy_db, lazy_state, counts = _run(tmp_path / "lazy", shape, prelock=True, foreign=foreign,
                                       monkeypatch=monkeypatch)
    assert counts["reclassified"] == 1, counts
    assert eager_counts["reclassified"] == 0
    assert lazy_state == eager_state
    assert lazy_db == eager_db
    # Nothing lost, nothing silently doubled: every row of the judged view is stored, and each at most as often as
    # the eager path stores it.
    _, view = _shape(shape)
    expected = Counter((m["role"], m["content"], m.get("tool_call_id") or "") for m in view)
    assert all(lazy_state["rows"][key] >= 1 for key in expected)


def test_b_commit_during_classification_of_a_no_write_ingest(tmp_path, monkeypatch):
    """An identical re-ingest writes nothing; a foreign commit during its pre-lock run still forces a locked recheck,
    and the outcome equals today's path with the same commit before BEGIN."""
    eager_db, eager_state, _ = _run(tmp_path / "eager", "identical", prelock=False, foreign="same-session-rows",
                                    monkeypatch=monkeypatch)
    root = tmp_path / "lazy"
    root.mkdir()
    setups, view = _shape("identical")
    engine = _engine(root, prelock=True)
    try:
        for setup in setups:
            engine.ingest(setup)
        before = _counts(engine)
        original = MessageStore.get_session_tail
        fired = []

        def tail(store, *args, **kwargs):
            if store is engine._store and not fired and not store._conn.in_transaction:
                fired.append(True)
                _foreign_commit("same-session-rows", root, view)
            return original(store, *args, **kwargs)

        monkeypatch.setattr(MessageStore, "get_session_tail", tail)
        result = engine.ingest(view)
        monkeypatch.setattr(MessageStore, "get_session_tail", original)
        counts = _counts(engine) - before
        state = dict(cursor=engine._frontier.cursor, needs_reconcile=engine._frontier.needs_reconcile,
                     failures=getattr(engine, "_consecutive_ingest_failures", 0),
                     returned=[(m.get("role"), m.get("content"), m.get("tool_call_id")) for m in result or []],
                     rows=_rows(engine))
    finally:
        engine.shutdown()
    if fired:  # the shape may not read the tail; then the no-write path is only checked for equality
        assert counts["reclassified"] == 1, counts
    assert state == eager_state
    assert _dump(root / "lcm.db") == eager_db


# -- (c) no change: no recheck work, and no lock while classifying -----------------------------------------------------

@pytest.mark.parametrize("shape", ("warm-append", "tools", "changed-prefix"))
def test_c_no_change_no_reclassify_and_no_lock_during_classification(shape, tmp_path, monkeypatch):
    root = tmp_path / "lazy"
    root.mkdir()
    setups, view = _shape(shape)
    engine = _engine(root, prelock=True)
    try:
        for setup in setups:
            engine.ingest(setup)
        locked_while_classifying = []
        original = LCMEngine._identity_anchor_prematch

        def prematch(self, *args, **kwargs):
            if self is engine:
                locked_while_classifying.append(self._store._conn.in_transaction)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(LCMEngine, "_identity_anchor_prematch", prematch)
        triggers = []
        seam = MessageStore._ingest_prelock_before_begin

        def record(store):
            if store is engine._store:
                import traceback
                triggers.append([f.name for f in traceback.extract_stack()])
            return seam(store)

        monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", record)
        before = _counts(engine)
        engine.ingest(view)
        counts = _counts(engine) - before
    finally:
        engine.shutdown()
    assert counts["reclassified"] == 0 and counts["poisoned"] == 0
    assert locked_while_classifying and not any(locked_while_classifying)
    assert len(triggers) == 1  # one BEGIN, no recheck rerun
    if shape != "changed-prefix":  # a changed prefix may write anchor evidence before its rows
        # The lock is taken by the row INSERT, the first write of a steady-state append.
        assert "_append_protected_batch" in triggers[0]


# -- the safety net: a pre-lock write is denied, poisons the attempt, and the locked rerun commits it once -------------

def test_prelock_write_is_denied_and_reclassified_under_the_lock(tmp_path, monkeypatch):
    results = {}
    for mode in (False, True):
        root = tmp_path / ("lazy" if mode else "eager")
        root.mkdir()
        setups, view = _shape("warm-append")
        engine = _engine(root, prelock=mode)
        try:
            for setup in setups:
                engine.ingest(setup)
            original = LCMEngine._ingest_filter_rows
            refused = []

            def filter_rows(self, *args, **kwargs):
                if self is engine:
                    try:  # a fail-open writer that swallows its own error
                        self._store._conn.execute(
                            "INSERT INTO metadata(key, value) VALUES('test-2293-stray', '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = value || '1'")
                    except sqlite3.DatabaseError as exc:
                        refused.append(type(exc).__name__)
                return original(self, *args, **kwargs)

            monkeypatch.setattr(LCMEngine, "_ingest_filter_rows", filter_rows)
            before = _counts(engine)
            engine.ingest(view)
            monkeypatch.setattr(LCMEngine, "_ingest_filter_rows", original)
            results[mode] = (_counts(engine) - before, refused, _rows(engine))
        finally:
            engine.shutdown()
        results[mode] += (_dump(root / "lcm.db"),)
    (lazy_counts, lazy_refused, lazy_rows, lazy_db), (_, eager_refused, eager_rows, eager_db) = results[True], results[False]
    assert eager_refused == [] and len(lazy_refused) == 1
    assert lazy_counts["poisoned"] == 1 and lazy_counts["reclassified"] == 1
    assert lazy_rows == eager_rows
    assert lazy_db == eager_db  # the stray row is written once, inside the transaction


# -- busy at the lazy BEGIN keeps today's admission-retry and failure accounting --------------------------------------

def test_busy_at_lazy_begin_matches_eager(tmp_path):
    outcomes = {}
    for mode in (False, True):
        root = tmp_path / ("lazy" if mode else "eager")
        root.mkdir()
        setups, view = _shape("warm-append")
        engine = _engine(root, prelock=mode)
        holder = None
        try:
            for setup in setups:
                engine.ingest(setup)
            engine._store._conn.execute("PRAGMA busy_timeout = 50")
            holder = sqlite3.connect(root / "lcm.db", timeout=5, isolation_level=None)
            holder.execute("BEGIN IMMEDIATE")
            engine.ingest(view)  # ingest() records the failure; the rows are retried by the next ingest
            failed = getattr(engine, "_consecutive_ingest_failures", 0)
            holder.execute("ROLLBACK")
            holder.close()
            holder = None
            engine.ingest(view)
            outcomes[mode] = (failed, getattr(engine, "_consecutive_ingest_failures", 0), _rows(engine))
        finally:
            if holder is not None:
                holder.execute("ROLLBACK")
                holder.close()
            engine.shutdown()
        outcomes[mode] += (_dump(root / "lcm.db"),)
    assert outcomes[True][0] >= 1 and outcomes[True][1] == 0
    assert outcomes[True] == outcomes[False]


def _judged(root, shape, prelock, monkeypatch, foreign=None, spies=()):
    """Run a shape with optional foreign commit at the BEGIN seam and class-level spies; return the engine's
    counts, the spies' records and the final dump."""
    root.mkdir(parents=True, exist_ok=True)
    setups, view = _shape(shape)
    engine = _engine(root, prelock=prelock)
    try:
        for setup in setups:
            engine.ingest(setup)
        records = {name: [] for name, *_ in spies}
        with monkeypatch.context() as patch:
            for name, owner, attr, make in spies:
                patch.setattr(owner, attr, make(getattr(owner, attr), records[name], engine))
            if foreign is not None:
                original, fired = MessageStore._ingest_prelock_before_begin, []

                def seam(store):
                    if store is engine._store and not fired:
                        fired.append(True)
                        _foreign_commit(foreign, root, view)
                    return original(store)

                patch.setattr(MessageStore, "_ingest_prelock_before_begin", seam)
            before = _counts(engine)
            engine.ingest(view)
            counts = _counts(engine) - before
        rows = _rows(engine)
    finally:
        engine.shutdown()
    return counts, records, rows, _dump(root / "lcm.db")


def _folded_reads(original, record, engine):
    def read(store, key, *args, **kwargs):
        if store is engine._store and str(key).startswith("folded_tail_lineage:"):
            record.append(store._conn.in_transaction)
        return original(store, key, *args, **kwargs)
    return read


def test_b_reclassify_rereads_the_folded_tail_cache(tmp_path, monkeypatch):
    """The per-attempt folded-tail read cache (store._ingest_folded_tail_reads) is cleared for the reclassify: the
    locked pass reads the key from SQL again and sees the foreign write."""
    spies = [("folded", MessageStore, "read_metadata_json", _folded_reads)]
    _, eager, eager_rows, eager_db = _judged(tmp_path / "eager", "warm-append", False, monkeypatch,
                                             foreign="folded-tail-lineage", spies=spies)
    counts, lazy, lazy_rows, lazy_db = _judged(tmp_path / "lazy", "warm-append", True, monkeypatch,
                                               foreign="folded-tail-lineage", spies=spies)
    assert eager["folded"], "the shape must read the folded-tail key for this test to mean anything"
    assert counts["reclassified"] == 1
    assert any(not locked for locked in lazy["folded"]) and any(lazy["folded"])  # read pre-lock, read again locked
    assert lazy_rows == eager_rows and lazy_db == eager_db


def _entry_cursors(original, record, engine):
    def transaction(self, *args, **kwargs):
        if self is engine:
            record.append(self._ingest_cursor)
        return original(self, *args, **kwargs)
    return transaction


def test_b_reclassify_restarts_from_the_attempts_original_cursor(tmp_path, monkeypatch):
    """A pre-lock run that moves the cursor (a changed prefix rewinds it) must not leak that move into the locked pass."""
    spies = [("cursor", LCMEngine, "_ingest_messages_transaction", _entry_cursors)]
    _, eager, eager_rows, eager_db = _judged(tmp_path / "eager", "changed-prefix", False, monkeypatch,
                                             foreign="other-session-rows", spies=spies)
    counts, lazy, lazy_rows, lazy_db = _judged(tmp_path / "lazy", "changed-prefix", True, monkeypatch,
                                               foreign="other-session-rows", spies=spies)
    assert counts["reclassified"] == 1
    assert len(eager["cursor"]) == 1 and lazy["cursor"] == eager["cursor"] * 2
    assert lazy_rows == eager_rows and lazy_db == eager_db


def _session_end(root, prelock, monkeypatch, *, poison=False, hold_lock=False):
    root.mkdir(parents=True, exist_ok=True)
    setups, view = _shape("warm-append")
    engine = _engine(root, prelock=prelock)
    passes, holder = [], None
    try:
        for setup in setups:
            engine.ingest(setup)
        with monkeypatch.context() as patch:
            original = LCMEngine._identity_anchor_prematch

            def prematch(self, *args, **kwargs):
                if self is engine and poison:
                    # A write reached during classification whose error is NOT swallowed here: it reaches the anchor
                    # block's `except Exception: self._raise_ingest_busy(exc)` guard.
                    self._store._conn.execute("INSERT INTO metadata(key, value) VALUES('test-2293-poison', '1') "
                                              "ON CONFLICT(key) DO UPDATE SET value = value || '1'")
                return original(self, *args, **kwargs)

            patch.setattr(LCMEngine, "_identity_anchor_prematch", prematch)
            transaction = LCMEngine._ingest_messages_transaction

            def counted(self, *args, **kwargs):
                if self is engine:
                    passes.append(True)
                return transaction(self, *args, **kwargs)

            patch.setattr(LCMEngine, "_ingest_messages_transaction", counted)
            if hold_lock:
                holder = sqlite3.connect(root / "lcm.db", timeout=5, isolation_level=None)
                holder.execute("BEGIN IMMEDIATE")
            before = _counts(engine)
            engine.on_session_end(SESSION, view)
            counts = _counts(engine) - before
            if holder is not None:
                holder.execute("ROLLBACK")
                holder.close()
                holder = None
        engine.shutdown()  # retries any frozen session-end tail
        engine = None
    finally:
        if holder is not None:
            holder.execute("ROLLBACK")
            holder.close()
        if engine is not None:
            engine.shutdown()
    return counts, len(passes), _dump(root / "lcm.db")


def test_poisoned_write_under_the_busy_guard_at_session_end_matches_eager(tmp_path, monkeypatch):
    eager_counts, _, eager_db = _session_end(tmp_path / "eager", False, monkeypatch, poison=True)
    counts, _, lazy_db = _session_end(tmp_path / "lazy", True, monkeypatch, poison=True)
    assert counts["poisoned"] >= 1 and counts["reclassified"] == 1
    assert lazy_db == eager_db  # rows stored once; the classification-time write committed once, inside the lock


def test_busy_at_lazy_begin_at_session_end_matches_eager(tmp_path, monkeypatch):
    """A busy session end keeps today's admission retry and frozen tail; the admission retry runs eager, so the lazy
    path classifies exactly once more than today (its discarded pre-lock pass)."""
    _, eager_passes, eager_db = _session_end(tmp_path / "eager", False, monkeypatch, hold_lock=True)
    counts, lazy_passes, lazy_db = _session_end(tmp_path / "lazy", True, monkeypatch, hold_lock=True)
    assert counts["reclassified"] == 0
    assert lazy_passes == eager_passes + 1
    assert lazy_db == eager_db


def test_cold_tail_retry_stays_eager(tmp_path, monkeypatch):
    """A bound cold-tail retry (_ingest_messages with retried_tail) keeps BEGIN IMMEDIATE before its first read."""
    seen = []
    original = MessageStore.ingest_atomic

    def record(store, *args, **kwargs):
        seen.append(bool(kwargs.get("prelock", False)))
        return original(store, *args, **kwargs)

    monkeypatch.setattr(MessageStore, "ingest_atomic", record)
    root = tmp_path / "tail"
    root.mkdir()
    engine = _engine(root, prelock=True)
    try:
        engine.ingest(_shape("warm-append")[1])
        tail = {"kind": "cold", "view": _shape("warm-append")[1], "session_id": SESSION}
        seen.clear()
        try:
            engine._ingest_messages(tail["view"], allow_session_end_replay_proof=False, no_view=True, retried_tail=tail)
        except Exception:
            pass  # only the BEGIN mode matters here
    finally:
        engine.shutdown()
    assert seen and not any(seen)


def test_switch_defaults_on_and_off_restores_eager_begin(tmp_path):
    assert MessageStore._ingest_prelock is True
    order = {}
    for mode in (False, True):
        root = tmp_path / ("lazy" if mode else "eager")
        root.mkdir()
        setups, view = _shape("warm-append")
        engine = _engine(root, prelock=mode)
        try:
            for setup in setups:
                engine.ingest(setup)
            seen = []
            engine._store._conn.set_trace_callback(seen.append)
            engine.ingest(view)
            engine._store._conn.set_trace_callback(None)
        finally:
            engine.shutdown()
        upper = [s.strip().upper() for s in seen]
        begin = next(i for i, s in enumerate(upper) if s.startswith("BEGIN"))
        read = next(i for i, s in enumerate(upper) if s.startswith("SELECT") and "FROM MESSAGES" in s)
        order[mode] = begin < read
    assert order == {False: True, True: False}  # eager: BEGIN before the first row read; lazy: after it
