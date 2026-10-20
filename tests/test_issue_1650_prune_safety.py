"""Fix2 regressions, using the independent review's failing inputs (#1650)."""
import dataclasses
import itertools
import json
import sqlite3

import pytest

from hermes_lcm import command
from hermes_lcm.embedding_provider import (
    EmbeddedDocumentBatch, EmbeddingProviderError, ProviderUnavailable, VoyageError,
)
from tests.test_issue_1014_incremental_embedding import FakeProvider
from tests.test_issue_1586_local_pacing import local_store as local_store

COUNTER = itertools.count(1)


@pytest.mark.parametrize("n", [20, 200])
def test_manual_scan_cache_is_bounded(local_store, n):
    engine, _ = local_store
    db = engine._store.db_path
    with sqlite3.connect(db) as conn:
        conn.executemany("INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                         [("seed", "user", f"scan message {i} " + "detail words here. " * 60, float(i))
                          for i in range(n)])
        cache = {}
        total, docs, _ = command._chunk_pending_rows(conn, _identity(db), "conversational", 7,
                                                    chunk_cache=cache)
        assert total == n and len(docs) == 7
        assert len(cache) <= 7


@pytest.mark.parametrize("successor_held", [False, True])
def test_lost_lease_prune_keeps_successor_record(local_store, monkeypatch, successor_held):
    engine, _ = local_store
    db = _seed_base(engine, engine._config, "conversational")
    identity = _identity(db)
    store = command.VectorStore(db, config=engine._config)
    conn = store.connection
    stale = command._acquire_embedding_backfill_lease(conn, ttl_s=60, heartbeat_s=60)
    real_ids = command._chunk_partial_ids

    def successor_after_read(c, ih):
        ids = real_ids(c, ih)
        with sqlite3.connect(db) as other:
            other.execute("DELETE FROM metadata WHERE key=?", (command._EMBEDDING_BACKFILL_CLAIM_KEY,))
        successor = command._acquire_embedding_backfill_lease(conn, ttl_s=60, heartbeat_s=60)
        command._record_chunk_partial_messages(conn, identity, successor, {123})
        if not successor_held:
            successor.release()
        return ids

    monkeypatch.setattr(command, "_chunk_partial_ids", successor_after_read)
    try:
        command._prune_chunk_partial_messages(conn, identity, stale, {})
        assert real_ids(conn, identity) == {123}
    finally:
        store.close()


@pytest.mark.parametrize("original_error", [False, True])
def test_final_prune_error_releases_and_closes(local_store, monkeypatch, caplog, original_error):
    engine, _ = local_store
    db = engine._store.db_path
    _seed_base(engine, engine._config, "conversational")
    real_prune, real_close = command._prune_chunk_partial_messages, command.VectorStore.close
    calls, closed = [], []

    def flaky(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise sqlite3.OperationalError("prune locked")
        if original_error:
            raise RuntimeError("original failure")
        return real_prune(*args, **kwargs)

    def close(store):
        closed.append(store.connection)
        return real_close(store)

    monkeypatch.setattr(command, "_prune_chunk_partial_messages", flaky)
    monkeypatch.setattr(command.VectorStore, "close", close)
    result = command._chunk_backfill_run(engine._config, db, apply=True, limit=32,
                                         retry_uncertain=False, policy="", incremental=True)
    assert calls[0] in closed
    assert result["status"] != "complete"
    if original_error:
        assert "original failure" in result["error"]
    assert "prune locked" in caplog.text
    with sqlite3.connect(db) as conn:
        assert not conn.execute("SELECT 1 FROM metadata WHERE key=?",
                                (command._EMBEDDING_BACKFILL_CLAIM_KEY,)).fetchone()


def test_record_updates_only_when_changed(local_store):
    engine, _ = local_store
    db = _seed_base(engine, engine._config, "conversational")
    identity = _identity(db)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE record_writes(n INTEGER)")
        conn.execute("CREATE TRIGGER count_record_updates AFTER UPDATE ON metadata "
                     "WHEN new.key LIKE 'chunk_partial_messages:%' "
                     "BEGIN INSERT INTO record_writes VALUES(1); END")
    for _ in range(3):
        command._chunk_backfill_run(engine._config, db, apply=True, limit=32,
                                    retry_uncertain=False, policy="", incremental=True)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM record_writes").fetchone()[0] == 0
        lease = command._acquire_embedding_backfill_lease(conn, ttl_s=60, heartbeat_s=60)
        command._record_chunk_partial_messages(conn, identity, lease, {123})
        command._record_chunk_partial_messages(conn, identity, lease, {123})
        assert conn.execute("SELECT COUNT(*) FROM record_writes").fetchone()[0] == 1
        lease.release()


class Distinct(FakeProvider):
    """Every vector distinct, so a rewrite or duplicate is visible."""

    def embed_document_batches(self, texts, *, before_dispatch):
        indexes = tuple(range(len(texts)))
        before_dispatch(indexes)
        yield EmbeddedDocumentBatch(indexes, tuple((float(next(COUNTER)), 0.5) for _ in texts))


def _identity(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT identity_hash FROM lcm_embedding_profile WHERE task='chunk' AND active=1").fetchone()[0]


def _record(db):
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT value FROM metadata WHERE key=?", (f"chunk_partial_messages:{_identity(db)}",)).fetchone()
    return None if row is None else json.loads(row[0])


def _published(db):
    with sqlite3.connect(db) as conn:
        return {r[0] for r in conn.execute("SELECT chunk_id FROM lcm_chunk_meta")}


def _expected(db, policy):
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall()
    return {c.chunk_id for row in rows for c in command.chunk_message(*row, policy=policy) if c.char_end > c.char_start}


def _chunks_of(db, sid, policy):
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT store_id, role, content FROM messages WHERE store_id=?", (sid,)).fetchone()
    return [c for c in command.chunk_message(*row, policy=policy) if c.char_end > c.char_start]


def _seed_base(engine, config, policy, n=3):
    db = engine._store.db_path
    for i in range(n):
        engine._store.append("seed", {"role": "assistant", "content": f"base message {i} words. " * 20})
    r = command._chunk_backfill_run(config, db, apply=True, limit=10000, retry_uncertain=False, policy=policy)
    assert isinstance(r, dict) and r["status"] == "complete" and not r["remaining"], r
    return db


def _converge(config, db, policy, slice_s, expected, *, passes=40, retry=True):
    hist = []
    for i in range(passes):
        for rt in ((False, True) if retry else (False,)):
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE lcm_embedding_backfill_inflight SET updated_at=0")
            r = command._chunk_backfill_run(config, db, apply=True, limit=32, retry_uncertain=rt, policy=policy,
                                            incremental=True, maintenance_slice=slice_s)
            assert isinstance(r, dict), r
            missing = expected - _published(db)
            hist.append((i, rt, r["status"], r["remaining"], r["embedded"], len(missing), r["stop_reason"]))
        if not missing and hist[-1][3] == 0 and hist[-1][2] == "complete":
            return hist, i
    return hist, None


def _lies(hist):
    """Passes that said complete or remaining 0 while a chunk was still missing (non-retry passes)."""
    return [h for h in hist if h[5] and not h[1] and h[3] == 0 and h[2] == "complete"]


@pytest.mark.parametrize("slice_s", [0.0, 1.0])
@pytest.mark.parametrize("policy", ["conversational", "full"])
@pytest.mark.parametrize("err", ["before_dispatch", "rate_limit", "uncertain", "uncertain_then_split"])
def test_interior_zero_published_drop(local_store, monkeypatch, slice_s, policy, err):
    engine, _ = local_store
    db = _seed_base(engine, engine._config, policy)
    m_sid = engine._store.append("seed", {"role": "assistant", "content": "interior message sentence. " * 190})
    n_sid = engine._store.append("seed", {"role": "assistant", "content": "newer short message"})
    m_chunks = _chunks_of(db, m_sid, policy)
    assert len(m_chunks) >= 3
    # One outer batch holds exactly M's chunks; N follows in the next batch.
    config = dataclasses.replace(engine._config, embedding_max_batch_items=len(m_chunks))
    expected = _expected(db, policy)
    calls = []

    class FailFirst(Distinct):
        def embed_document_batches(self, texts, *, before_dispatch):
            calls.append(len(texts))
            if len(calls) == 1:
                if err == "before_dispatch":
                    raise ProviderUnavailable("model load blip")
                if err == "rate_limit":
                    before_dispatch(tuple(range(len(texts))))
                    raise VoyageError("rate_limit", "429")
                if err == "uncertain":
                    before_dispatch(tuple(range(len(texts))))
                    raise EmbeddingProviderError("read timeout")
                # First sub-request of the batch dispatched and lost; the rest never dispatched.
                before_dispatch((0, 1))
                raise EmbeddingProviderError("read timeout")
            yield from super().embed_document_batches(texts, before_dispatch=before_dispatch)

    monkeypatch.setattr(command, "resolve_provider", lambda c, **_k: FailFirst(c.embedding_provider, c.embedding_model))
    first = command._chunk_backfill_run(config, db, apply=True, limit=32, retry_uncertain=False, policy=policy,
                                        incremental=True, maintenance_slice=slice_s)
    pub = _published(db)
    with sqlite3.connect(db) as conn:
        unc = conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight").fetchone()[0]
    print(f"PROBE interior {err} {policy} slice={slice_s} first status={first['status']} remaining={first['remaining']} "
          f"embedded={first['embedded']} rec={_record(db)} m_published={sum(1 for c in m_chunks if c.chunk_id in pub)} "
          f"n_published={any(c.startswith(f'{n_sid}:') for c in pub)} inflight={unc} calls={calls}")
    assert any(c.startswith(f"{n_sid}:") for c in pub), "N must publish so M is interior"
    monkeypatch.setattr(command, "resolve_provider", lambda c, **_k: Distinct(c.embedding_provider, c.embedding_model))
    hist, done = _converge(config, db, policy, slice_s, expected, passes=12)
    lies = _lies(hist)
    print(f"PROBE interior {err} {policy} slice={slice_s} done={done} last={hist[-1]} lies={len(lies)} first_lie={lies[:1]}")
    assert done is not None and not lies, (hist[-3:], lies[:2])


@pytest.mark.parametrize("slice_s", [0.0, 1.0])
def test_record_behind_uncertain_slots(local_store, monkeypatch, slice_s):
    """Cloud-shaped: no automatic uncertain retry. >= partial_limit newer recorded
    messages whose only remainder is uncertain occupy every record slot; does an
    older recorded message with ordinary pending chunks still progress?"""
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=2)
    db = _seed_base(engine, config, "conversational")
    s_sid = engine._store.append("seed", {"role": "assistant", "content": "older starving message sentence. " * 190})
    s_chunks = _chunks_of(db, s_sid, "conversational")
    newer = []
    for i in range(18):
        sid = engine._store.append("seed", {"role": "assistant", "content": f"two chunk message {i} sentence. " * 120})
        newer.append(sid)
    assert all(len(_chunks_of(db, sid, "conversational")) == 2 for sid in newer)
    calls = []

    class Flaky(Distinct):
        def embed_document_batches(self, texts, *, before_dispatch):
            calls.append(1)
            if any("starving" in t for t in texts):
                if len(calls) == 1:
                    yield from super().embed_document_batches(texts, before_dispatch=before_dispatch)
                    return
                raise ProviderUnavailable("blip before dispatch")  # S's later batch: claims cleared
            # two-chunk message: first chunk accepted, second dispatched and lost
            before_dispatch((0,))
            yield EmbeddedDocumentBatch((0,), ((float(next(COUNTER)), 0.5),))
            before_dispatch((1,))
            raise EmbeddingProviderError("read timeout")

    monkeypatch.setattr(command, "resolve_provider", lambda c, **_k: Flaky(c.embedding_provider, c.embedding_model))
    first = command._chunk_backfill_run(config, db, apply=True, limit=10000, retry_uncertain=False, policy="",
                                        incremental=True, maintenance_slice=slice_s)
    rec = _record(db) or []
    pub = _published(db)
    s_pub = sum(1 for c in s_chunks if c.chunk_id in pub)
    print(f"PROBE slots slice={slice_s} first={first['status']} rec_len={len(rec)} s_in_rec={s_sid in rec} "
          f"S={len(s_chunks)} s_pub={s_pub} uncertain={first['uncertain']}")
    assert s_sid in rec and 0 < s_pub < len(s_chunks)
    monkeypatch.setattr(command, "resolve_provider", lambda c, **_k: Distinct(c.embedding_provider, c.embedding_model))
    progress = []
    for i in range(10):  # retry_uncertain never runs (cloud maintenance: retries=(False,))
        r = command._chunk_backfill_run(config, db, apply=True, limit=32, retry_uncertain=False, policy="",
                                        incremental=True, maintenance_slice=slice_s)
        pub = _published(db)
        progress.append((r["status"], r["remaining"], r["embedded"], sum(1 for c in s_chunks if c.chunk_id in pub)))
    print(f"PROBE slots slice={slice_s} S progress: first={progress[0]} last={progress[-1]}")
    assert progress[-1][3] == len(s_chunks), progress[-3:]
