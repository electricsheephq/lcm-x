"""#2293 merge-blocking probes (release-manager ruling T1-11) for the pre-lock ingest.

The ingest classifies before the write lock. Python's implicit BEGIN IMMEDIATE (the store connection uses
isolation_level="IMMEDIATE") or an explicit seam takes the lock at the first write; the store's write generation is
then rechecked inside the transaction. Each probe runs every ingest-trace scenario (the 202-case corpus) in both
modes and compares OUTCOMES: returned rows, raised errors, engine state and the follow-up and cold ingests. The full
statement stream is not compared, because a discarded pre-lock pass legitimately adds reads.

P1  A second connection holds the write lock, so an implicit BEGIN inside a fail-open writer (placeholder-ledger
    metadata, host-uid counters, ledger adopt, relations, raw-connection ledger functions) times out and the writer
    swallows it. The lock is released just before the next explicit BEGIN. The attempt must notice and rerun eager,
    so the skipped write is redone: outcomes equal eager under the same contention.
P2  A foreign commit lands after the attempt's baseline and before its first write, so every state the pre-lock pass
    built (placeholder maps, host-uid counters, folded-tail memo, relation caches) must be reset by the rerun:
    outcomes equal eager. The rerun holds the lock, so no second mismatch is possible (a second connection cannot
    commit during it).
P3  total_changes: no foreign commit gives zero reclassifies on every no-fault scenario (no false mismatch from the
    attempt's own writes). A same-connection write by another thread between the baseline and the first write is
    caught (it reclassifies).
P4  Raw-connection ledger functions write through the store connection, so their DML joins the attempt's
    transaction: at the row INSERT, nothing the attempt wrote is visible to a second connection, and a second
    connection cannot take the write lock while the attempt holds it.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager

import pytest

from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore
from tests.behaviour import ingest_trace_harness as harness

OUTCOME_PREFIXES = ("return-", "raise-", "state-after", "attr ", "search-after", "payloads-after", "input-after",
                    "aliases-", "next-return", "next-raise", "next-aliases", "next-state", "cold-return", "cold-raise",
                    "cold-state", "payloads-cold", "quarantine-payloads-after")
# Faults that already hold the write lock or open a transaction on the ingest connection cannot take a second holder.
_OWN_LOCK_FAULTS = {"locked", "duplicate-tail", "snapshot"}
NAMES = [case.name for case in harness.SCENARIOS if case.fault not in _OWN_LOCK_FAULTS]
BY_NAME = {case.name: case for case in harness.SCENARIOS}
LEDGER_TABLES = ("occurrence_bindings", "ingest_views", "ledger_meta", "lineage_marks", "ledger_emissions",
                 "stamp_observations", "metadata", "message_relations", "messages")


@pytest.fixture(autouse=True)
def _pinned_salt(monkeypatch):
    import hermes_lcm.store as store_module
    monkeypatch.setattr(store_module.secrets, "token_bytes", lambda n=32: bytes(range(n)))


def _outcomes(trace):
    return [event for event in trace if event.startswith(OUTCOME_PREFIXES)]


def _observe(case, root, monkeypatch, *, prelock, around=None):
    """Run one corpus scenario; ``around(engine)`` is a context manager wrapped around the judged ingest only."""
    monkeypatch.setattr(MessageStore, "_ingest_prelock", prelock)
    if around is not None:
        original = harness.sqlite_fault

        @contextmanager
        def wrapped(engine, fault, events):
            with original(engine, fault, events), around(engine):
                yield

        monkeypatch.setattr(harness, "sqlite_fault", wrapped)
    try:
        return getattr(harness.observe, "__wrapped__", harness.observe)(case, root)  # bypass any lru_cache: every call is a fresh run
    finally:
        if around is not None:
            monkeypatch.setattr(harness, "sqlite_fault", original)


def _assert_same(case, eager, lazy):
    e, z = _outcomes(eager), _outcomes(lazy)
    if e != z:
        first = next(i for i, (x, y) in enumerate(zip(e + [None], z + [None])) if x != y)
        pytest.fail(f"{case.name}: first outcome difference at {first}:\n eager: {e[first:first + 2]}\n"
                    f" lazy:  {z[first:first + 2]}")


# -- P1: a lock timeout at the implicit BEGIN inside a fail-open writer -----------------------------------------------

def _held_until_explicit_begin(monkeypatch, log):
    """Hold the write lock from a second connection; release it right before the next explicit BEGIN."""
    @contextmanager
    def around(engine):
        conn = engine._store._conn
        previous = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.execute("PRAGMA busy_timeout = 20")
        holder = sqlite3.connect(engine._config.database_path, timeout=0, isolation_level=None)
        holder.execute("BEGIN IMMEDIATE")
        released = []
        seam = MessageStore._ingest_prelock_before_begin

        def release(store):
            if store is engine._store and not released:
                released.append(True)
                holder.execute("ROLLBACK")
            return seam(store)

        monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", release)
        before = dict(engine._store.ingest_prelock_counts)
        try:
            yield
        finally:
            monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", seam)
            if not released:
                holder.execute("ROLLBACK")
            holder.close()
            try:
                conn.execute(f"PRAGMA busy_timeout = {int(previous)}")
            except sqlite3.Error:
                pass
            after = engine._store.ingest_prelock_counts
            log.append({k: after.get(k, 0) - before.get(k, 0) for k in after})
    return around


@pytest.mark.parametrize("name", NAMES)
def test_p1_lock_timeout_at_implicit_begin_matches_eager(name, tmp_path, monkeypatch):
    case = BY_NAME[name]
    eager = _observe(case, tmp_path / "eager", monkeypatch, prelock=False,
                     around=_held_until_explicit_begin(monkeypatch, []))
    log = []
    lazy = _observe(case, tmp_path / "lazy", monkeypatch, prelock=True,
                    around=_held_until_explicit_begin(monkeypatch, log))
    _assert_same(case, eager, lazy)


# -- P2: a foreign commit after the baseline, before the first write --------------------------------------------------

def _foreign_after_baseline(monkeypatch, log, *, prelock):
    @contextmanager
    def around(engine):
        store = engine._store
        fired, rerun_probe = [], []

        def commit():
            with sqlite3.connect(engine._config.database_path, timeout=5) as other:
                other.execute("INSERT INTO metadata(key, value) VALUES('test-2293-foreign', 'x') "
                              "ON CONFLICT(key) DO UPDATE SET value = value || 'x'")
            other.close()

        seam = MessageStore._ingest_prelock_before_begin
        transaction = LCMEngine._ingest_messages_transaction
        passes = []

        authorizer = MessageStore._ingest_prelock_authorizer

        def after_baseline(self):
            # Lazy entry builds the authorizer right after it records G0 and before
            # the first pre-lock read or write: a foreign commit here always lands
            # after the baseline, so every pre-lock write that follows must be
            # discarded by the recheck.
            state = getattr(self, "_ingest_prelock_state", None)
            if (self is store and prelock and not fired and state is not None and state.phase == "prelock"
                    and not self._conn.in_transaction):
                fired.append(True)
                commit()
            return authorizer(self)

        def eager_seam(self):
            if self is store and not prelock and not fired:
                fired.append(True)
                commit()
            return seam(self)

        def counted(self, *args, **kwargs):
            if self is engine:
                passes.append(True)
                if len(passes) == 2:
                    # The rerun holds the lock: a second connection cannot commit, so it cannot mismatch again.
                    other = sqlite3.connect(engine._config.database_path, timeout=0, isolation_level=None)
                    try:
                        other.execute("BEGIN IMMEDIATE")
                        rerun_probe.append("committable")
                        other.execute("ROLLBACK")
                    except sqlite3.OperationalError:
                        rerun_probe.append("locked")
                    finally:
                        other.close()
            return transaction(self, *args, **kwargs)

        monkeypatch.setattr(MessageStore, "_ingest_prelock_authorizer", after_baseline)
        monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", eager_seam)
        monkeypatch.setattr(LCMEngine, "_ingest_messages_transaction", counted)
        before = dict(store.ingest_prelock_counts)
        try:
            yield
        finally:
            monkeypatch.setattr(MessageStore, "_ingest_prelock_authorizer", authorizer)
            monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", seam)
            monkeypatch.setattr(LCMEngine, "_ingest_messages_transaction", transaction)
            after = store.ingest_prelock_counts
            log.append(dict(fired=bool(fired), rerun=list(rerun_probe),
                            counts={k: after.get(k, 0) - before.get(k, 0) for k in after}))
    return around


@pytest.mark.parametrize("name", NAMES)
def test_p2_mismatch_after_prelock_writes_resets_state_and_matches_eager(name, tmp_path, monkeypatch):
    case = BY_NAME[name]
    eager = _observe(case, tmp_path / "eager", monkeypatch, prelock=False,
                     around=_foreign_after_baseline(monkeypatch, [], prelock=False))
    log = []
    lazy = _observe(case, tmp_path / "lazy", monkeypatch, prelock=True,
                    around=_foreign_after_baseline(monkeypatch, log, prelock=True))
    _assert_same(case, eager, lazy)
    entry = log[0]
    if entry["fired"] and (entry["counts"].get("acquired", 0) or entry["counts"].get("reclassified", 0)):
        assert entry["counts"].get("reclassified", 0) == 1, entry
        assert "committable" not in entry["rerun"], entry  # the second pass cannot mismatch


# -- P3: total_changes and the attempt's own writes -------------------------------------------------------------------

def _counting(log):
    @contextmanager
    def around(engine):
        before = dict(engine._store.ingest_prelock_counts)
        try:
            yield
        finally:
            after = engine._store.ingest_prelock_counts
            log.append({k: after.get(k, 0) - before.get(k, 0) for k in after})
    return around


@pytest.mark.parametrize("name", [case.name for case in harness.SCENARIOS if not case.fault])
def test_p3_no_foreign_write_means_no_reclassify(name, tmp_path, monkeypatch):
    log = []
    _observe(BY_NAME[name], tmp_path / "lazy", monkeypatch, prelock=True, around=_counting(log))
    assert log and log[0].get("reclassified", 0) == 0, log
    assert log[0].get("poisoned", 0) == 0, log


def test_p3_same_connection_write_by_another_thread_is_caught(tmp_path, monkeypatch):
    from tests.test_2293_prelock import _engine, _shape, _dump, _rows
    results = {}
    for mode in (False, True):
        root = tmp_path / ("lazy" if mode else "eager")
        root.mkdir()
        setups, view = _shape("warm-append")
        engine = _engine(root, prelock=mode)
        try:
            for setup in setups:
                engine.ingest(setup)
            conn, fired = engine._store._conn, []

            def foreign_same_connection():
                # Another thread on the SAME connection (a writer that skipped _write_lock): data_version does not
                # move for it; only total_changes does.
                t = threading.Thread(target=lambda: (conn.execute(
                    "INSERT INTO metadata(key, value) VALUES('test-2293-same-conn', 'x') "
                    "ON CONFLICT(key) DO UPDATE SET value = value || 'x'"), conn.commit()))
                t.start()
                t.join()

            read = MessageStore.read_metadata_json
            seam = MessageStore._ingest_prelock_before_begin

            def first_read(self, *args, **kwargs):
                state = getattr(self, "_ingest_prelock_state", None)
                if self is engine._store and mode and not fired and state is not None and state.phase == "prelock" \
                        and not self._conn.in_transaction:
                    fired.append(True)
                    foreign_same_connection()
                return read(self, *args, **kwargs)

            def eager_seam(self):
                if self is engine._store and not mode and not fired:
                    fired.append(True)
                    foreign_same_connection()
                return seam(self)

            monkeypatch.setattr(MessageStore, "read_metadata_json", first_read)
            monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", eager_seam)
            before = dict(engine._store.ingest_prelock_counts)
            engine.ingest(view)
            monkeypatch.setattr(MessageStore, "read_metadata_json", read)
            monkeypatch.setattr(MessageStore, "_ingest_prelock_before_begin", seam)
            after = engine._store.ingest_prelock_counts
            results[mode] = (fired, {k: after.get(k, 0) - before.get(k, 0) for k in after}, _rows(engine))
        finally:
            engine.shutdown()
        results[mode] += (_dump(root / "lcm.db"),)
    assert results[True][0] and results[False][0]
    assert results[True][1].get("reclassified", 0) == 1, results[True][1]
    assert results[True][2] == results[False][2] and results[True][3] == results[False][3]


# -- P4: raw-connection ledger writes join the attempt's transaction --------------------------------------------------

def _visible_counts(path):
    other = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        out = {}
        for table in LEDGER_TABLES:
            try:
                out[table] = other.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            except sqlite3.OperationalError:
                out[table] = None
        return out
    finally:
        other.close()


def _isolation_probe(monkeypatch, log):
    @contextmanager
    def around(engine):
        path = engine._config.database_path
        start = _visible_counts(path)
        append = MessageStore._append_protected_batch

        def probed(self, *args, **kwargs):
            if self is engine._store:
                held = self._conn.in_transaction
                visible = _visible_counts(path)
                other = sqlite3.connect(path, timeout=0, isolation_level=None)
                try:
                    other.execute("BEGIN IMMEDIATE")
                    lockable = True
                    other.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    lockable = False
                finally:
                    other.close()
                log.append(dict(held=held, leaked=visible != start, lockable=lockable))
            return append(self, *args, **kwargs)

        monkeypatch.setattr(MessageStore, "_append_protected_batch", probed)
        try:
            yield
        finally:
            monkeypatch.setattr(MessageStore, "_append_protected_batch", append)
    return around


@pytest.mark.parametrize("name", [case.name for case in harness.SCENARIOS if not case.fault])
def test_p4_prelock_writes_stay_inside_the_attempt_transaction(name, tmp_path, monkeypatch):
    log = []
    _observe(BY_NAME[name], tmp_path / "lazy", monkeypatch, prelock=True, around=_isolation_probe(monkeypatch, log))
    for entry in log:
        assert not entry["leaked"], (name, entry)  # nothing the attempt wrote is visible before COMMIT
        if entry["held"]:
            assert not entry["lockable"], (name, entry)  # our write lock is held; a second writer waits


# -- review findings (round 2) ---------------------------------------------------------------------------------------

def test_foreign_commit_after_the_engine_validated_a_read_only_attempt_does_not_escape(tmp_path):
    """Deep review F1: the engine validates a read-only attempt in its loop; _ingest_atomic's own finish must not
    re-validate after the loop has exited (a commit landing between them would raise _IngestReclassify past
    every handler)."""
    from tests.test_2293_prelock import _engine, _shape
    setups, view = _shape("warm-append")
    engine = _engine(tmp_path, prelock=True)
    try:
        for setup in setups:
            engine.ingest(setup)
        engine.ingest(view)  # an identical re-ingest is now a read-only lazy attempt
        calls, fired = [], []
        original = MessageStore._finish_ingest_prelock

        def finish(self):
            if self is engine._store:
                calls.append(self._ingest_is_prelock())
                if len(calls) == 2 and calls[1] and not fired:
                    fired.append(True)
                    with sqlite3.connect(engine._config.database_path, timeout=5) as other:
                        other.execute("INSERT INTO metadata(key, value) VALUES('probe-foreign', 'x') "
                                      "ON CONFLICT(key) DO UPDATE SET value = value || 'x'")
                    other.close()
            return original(self)

        before = engine._store.ingest_prelock_counts["no_write"]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(MessageStore, "_finish_ingest_prelock", finish)
            engine.ingest(view)  # must return; a BaseException would escape here
        assert fired
        assert engine._store.ingest_prelock_counts["no_write"] == before + 1
    finally:
        engine.shutdown()


def test_discarded_pass_rule_is_exercised_both_ways():
    """Astra F3: the held-lock rule accepts repeated reads and a P1 busy write, and rejects any write statement,
    any event eager never emits, and any deletion."""
    from tests.test_2293_prelock import _discarded_passes_only
    eager = ['sql _store "SELECT value FROM metadata WHERE key=\'a\'"', "store result read_metadata_json null",
             'sql failed _store ["BEGIN IMMEDIATE",[],"OperationalError","database is locked"]', "raise-0 x"]
    reads = eager[:2]
    assert _discarded_passes_only(eager, reads + eager) is None
    busy_write = ["sql failed _store sha12/len=abc/301",
                  'store raises write_metadata_json ["OperationalError","database is locked"]']
    assert _discarded_passes_only(eager, reads + busy_write + eager) is None
    write = ['sql _store "UPDATE ledger_meta SET value=CAST(value AS INTEGER)+1 WHERE key=\'view_seq\'"']
    assert "write" in _discarded_passes_only(eager + write, write + eager + write)
    assert "novel" in _discarded_passes_only(eager, ["log x WARNING something"] + eager)
    assert _discarded_passes_only(eager, eager[1:]) is not None
