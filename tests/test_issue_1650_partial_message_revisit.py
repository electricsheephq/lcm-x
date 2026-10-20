"""Interrupted messages retain an exact-chunk incremental continuation (#1650)."""
import dataclasses
import sqlite3
import threading

import pytest

from hermes_lcm import command, embedding_maintenance as maintenance, tools
from hermes_lcm.embedding_provider import EmbeddedDocumentBatch
from tests.test_issue_1014_incremental_embedding import FakeProvider
from tests.test_issue_1586_local_pacing import local_store as local_store


@pytest.mark.parametrize("slice_s", [0.0, 1.0])
@pytest.mark.parametrize("path", ["gap-stop", "gap-lease", "gap-mid-exit", "cursor-stop"])
def test_interrupted_message_is_revisited(local_store, monkeypatch, slice_s, path):
    engine, _ = local_store
    config = dataclasses.replace(engine._config, embedding_max_batch_items=2)
    db = engine._store.db_path
    for body in ("low boundary " * 60, "gap message sentence. " * 250, "high boundary " * 60):
        engine._store.append("seed", {"role": "assistant", "content": body})
    result = command._chunk_backfill_run(config, db, apply=True, limit=10000,
                                         retry_uncertain=False, policy="")
    assert isinstance(result, dict) and result["remaining"] == 0
    if path.startswith("cursor"):
        engine._store.append("seed", {"role": "assistant", "content": "new message sentence. " * 250})
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall()
        target = rows[-1][0] if path.startswith("cursor") else rows[1][0]
        expected = {c.chunk_id for row in rows for c in command.chunk_message(*row, policy="conversational")}
        assert len(command.chunk_message(*next(row for row in rows if row[0] == target),
                                         policy="conversational")) > 2
        identity = conn.execute("SELECT identity_hash FROM lcm_embedding_profile WHERE task='chunk'").fetchone()[0]
        if path.startswith("gap"):
            conn.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id IN "
                         "(SELECT chunk_id FROM lcm_chunk_meta WHERE store_id=?)", (target,))
            conn.execute("DELETE FROM lcm_chunk_meta WHERE store_id=?", (target,))
            conn.execute("DELETE FROM metadata WHERE key LIKE 'chunk_policy_revision%' OR key LIKE 'chunk_gapfill_scan%'")
    stop = threading.Event()
    calls = []

    class Interrupted(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            calls.append(len(texts))
            if path == "gap-mid-exit" and len(calls) == 1:
                before_dispatch((0,))
                yield EmbeddedDocumentBatch((0,), ((1.0, 0.0),))
                stop.set()  # The other claimed chunk was never dispatched.
                return
            yield from super().embed_document_batches(texts, before_dispatch=before_dispatch)
            if len(calls) == 1 and path != "gap-lease":
                stop.set()

    renew = command._BackfillLease.renew
    renewals = []

    def lose_lease(lease, **kwargs):
        renewals.append(1)
        if len(renewals) == 2:
            lease.conn.execute("DELETE FROM metadata WHERE key=?", (command._EMBEDDING_BACKFILL_CLAIM_KEY,))
            return False
        return renew(lease, **kwargs)

    with monkeypatch.context() as interrupt:
        interrupt.setattr(command, "resolve_provider", lambda c, **_k: Interrupted(c.embedding_provider, c.embedding_model))
        if path == "gap-lease":
            interrupt.setattr(command._BackfillLease, "renew", lose_lease)
        first = command._chunk_backfill_run(config, db, apply=True, limit=32,
                                            retry_uncertain=False, policy="", incremental=True,
                                            stop_event=stop, maintenance_slice=slice_s)
    assert first["stop_reason"] == ("lease_lost" if path == "gap-lease" else "process_exit"), first
    assert first["embedded"] == (1 if path == "gap-mid-exit" else 2), first
    assert first["remaining"] > 0, first
    assert first["status"] != "complete", first
    with sqlite3.connect(db) as conn:
        before = dict(conn.execute("SELECT chunk_id, vec FROM lcm_chunk_vectors"))
        assert conn.execute("SELECT COUNT(*) FROM lcm_embedding_backfill_inflight").fetchone()[0] == 0
        assert command._chunk_backlog_estimate(conn, identity, "conversational") != "0"

    # Exercise the real cycle/readiness reporting while withholding further work.
    # Its completion decision must use the actual remaining accounting.
    with monkeypatch.context() as observe:
        remaining = command._chunk_backfill_remaining(db, identity, "conversational", 0, 0, True)[0]
        observe.setattr(command, "_chunk_backfill_run", lambda *a, **kw: dict(
            first, status="complete", embedded=0, selected=0, stop_reason=None, remaining=remaining))
        state = {}
        for _ in range(4):
            last = maintenance.run_incremental_embedding_pass(db, config, slice_state=state)
        assert last["outcome"].endswith("no_progress"), last
        assert "no_progress" in command.handle_lcm_command("status", engine)
        assert tools._embedding_readiness(engine, {"available": True}).endswith("no_progress")

    stop.clear()
    for _ in range(3):
        result = command._chunk_backfill_run(config, db, apply=True, limit=32,
                                             retry_uncertain=False, policy="", incremental=True,
                                             maintenance_slice=slice_s)
        assert not result["failed"], result
    assert result["remaining"] == 0 and result["status"] == "complete", result
    with sqlite3.connect(db) as conn:
        after = dict(conn.execute("SELECT chunk_id, vec FROM lcm_chunk_vectors"))
        assert set(after) == expected
        assert all(after[key] == value for key, value in before.items())
        assert conn.execute("SELECT COUNT(*) FROM lcm_chunk_meta").fetchone()[0] == len(expected)
        saved = conn.execute("SELECT value FROM metadata WHERE key=?",
                             (f"chunk_partial_messages:{identity}",)).fetchone()
        assert not saved or not command._chunk_partial_ids(conn, identity)
