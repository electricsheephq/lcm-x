"""#1518: a live dispatch is distinct from recovered durable uncertainty."""

import re
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from hermes_lcm import command
from hermes_lcm.embedding_provider import EmbeddedDocumentBatch
from tests.test_embedding_backfill import FakeProvider, _engine, _seed


def test_held_dispatch_is_in_flight(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    _seed(engine, 2)
    started, release = threading.Event(), threading.Event()
    results = []

    class HeldProvider(FakeProvider):
        def embed_document_batches(self, texts, *, before_dispatch):
            indexes = tuple(range(len(texts)))
            before_dispatch(indexes)
            started.set()
            assert release.wait(5)
            yield EmbeddedDocumentBatch(indexes, tuple((1.0, 0.0) for _ in indexes))

    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_k: HeldProvider())
    worker = threading.Thread(target=lambda: results.append(command.handle_lcm_command("embed backfill --apply", engine)))
    worker.start()
    try:
        assert started.wait(5)
        assert command._embedding_maintenance_status_lines(engine)[0] == "embedding_backlog_summaries: 0 (in flight 2)"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert "status: complete" in results[0]
    assert command._embedding_maintenance_status_lines(engine)[0] == "embedding_backlog_summaries: 0"


def test_abandoned_dispatch_recovers_as_uncertain(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    ids = _seed(engine, 1)
    with sqlite3.connect(engine._store.db_path) as conn:
        command._ensure_inflight_table(conn)
        identity = command._embedding_current_profile(conn)["identity_hash"]
        conn.execute("INSERT INTO lcm_embedding_backfill_inflight "
                     "(embedded_id, identity_hash, lease_id, generation, state, request_id, updated_at) "
                     "VALUES (?, ?, 'abandoned', 1, 'dispatched', 'old-request', 1)", (str(ids[0]), identity))
    provider = FakeProvider()
    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_k: provider)
    command.handle_lcm_command("embed backfill --apply", engine)
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT state FROM lcm_embedding_backfill_inflight").fetchone()[0] == "uncertain"
    assert provider.calls == []
    assert command._embedding_maintenance_status_lines(engine)[0] == "embedding_backlog_summaries: 0 (uncertain 1)"


@pytest.mark.parametrize("dispatched,uncertain,suffix", [
    (0, 0, ""), (2, 0, " (in flight 2)"), (0, 3, " (uncertain 3)"),
    (2, 3, " (in flight 2, uncertain 3)"),
])
def test_status_line_parser_shapes(monkeypatch, dispatched, uncertain, suffix):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE lcm_embedding_backfill_inflight(identity_hash TEXT, state TEXT)")
    conn.executemany("INSERT INTO lcm_embedding_backfill_inflight VALUES ('summary', ?)",
                     [("dispatched",)] * dispatched + [("uncertain",)] * uncertain)
    monkeypatch.setattr(command, "_embedding_read_connection", lambda _p: conn)
    monkeypatch.setattr(command, "_embedding_current_profile", lambda _c: {"identity_hash": "summary"})
    monkeypatch.setattr(command, "_chunk_current_profile", lambda _c: None)
    monkeypatch.setattr(command, "_embedding_pending_rows", lambda *_a: (7, []))
    monkeypatch.setattr(command, "incremental_embedding_state", lambda _p: (None, 0))
    engine = SimpleNamespace(_store=SimpleNamespace(db_path="fixture.db"), _config=SimpleNamespace())
    line = command._embedding_maintenance_status_lines(engine)[0]
    assert line == "embedding_backlog_summaries: 7" + suffix
    match = re.fullmatch(r"embedding_backlog_summaries: (\d+)(?: \((?:in flight (\d+))?(?:, )?(?:uncertain (\d+))?\))?", line)
    assert match is not None
    pending, live, durable = match.groups()
    assert (int(pending), int(live or 0), int(durable or 0)) == (7, dispatched, uncertain)
