"""#1518: a live dispatch is distinct from recovered durable uncertainty."""

import json
import re
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import command
from hermes_lcm.embedding_provider import EmbeddedDocumentBatch
from tests.test_embedding_backfill import FakeProvider, _engine, _seed
from tests.test_chunk_backfill import _seed_messages, _user_msgs
from hermes_lcm import tools


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
        conn.row_factory = sqlite3.Row
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
    conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO metadata VALUES (?, ?)", (command._EMBEDDING_BACKFILL_CLAIM_KEY,
                 json.dumps(dict(owner="live", generation=1, heartbeat_at=time.time()))))
    conn.execute("CREATE TABLE lcm_embedding_backfill_inflight(identity_hash TEXT, state TEXT, lease_id TEXT DEFAULT 'live', generation INTEGER DEFAULT 1)")
    conn.executemany("INSERT INTO lcm_embedding_backfill_inflight(identity_hash, state) VALUES ('summary', ?)",
                     [("dispatched",)] * dispatched + [("uncertain",)] * uncertain)
    monkeypatch.setattr(command, "_embedding_read_connection", lambda _p: conn)
    monkeypatch.setattr(command, "_embedding_current_profile", lambda _c: {"identity_hash": "summary"})
    monkeypatch.setattr(command, "_chunk_current_profile", lambda _c: {"identity_hash": "summary"})
    monkeypatch.setattr(command, "_chunk_backlog_estimate", lambda *_a: "7+ (gap-fill)")
    monkeypatch.setattr(command, "_embedding_pending_rows", lambda *_a: (7, []))
    monkeypatch.setattr(command, "incremental_embedding_state", lambda _p: (None, 0))
    engine = SimpleNamespace(_store=SimpleNamespace(db_path="fixture.db"), _config=SimpleNamespace())
    lines = command._embedding_maintenance_status_lines(engine)
    line = lines[0]
    assert line == "embedding_backlog_summaries: 7" + suffix
    match = re.fullmatch(r"embedding_backlog_summaries: (\d+)(?: \((?:in flight (\d+))?(?:, )?(?:uncertain (\d+))?\))?", line)
    assert match is not None
    pending, live, durable = match.groups()
    assert (int(pending), int(live or 0), int(durable or 0)) == (7, dispatched, uncertain)
    assert lines[1] == "embedding_backlog_chunks: 7+ (gap-fill)" + suffix
    match = re.fullmatch(r"embedding_backlog_chunks: (\d+)\+?(?: \(gap-fill\))?(?: \((?:in flight (\d+))?(?:, )?(?:uncertain (\d+))?\))?", lines[1])
    assert match is not None
    assert tuple(int(v or 0) for v in match.groups()) == (7, dispatched, uncertain)


def _dispatch_fixture(tmp_path, corpus):
    engine = _engine(tmp_path)
    if corpus == "summary":
        embedded_id = str(_seed(engine, 1)[0])
    else:
        _seed_messages(engine, _user_msgs(1))
        embedded_id = command.chunk_message(1, "user", "u" * 60, policy="conversational")[0].chunk_id
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        command._ensure_inflight_table(conn)
        profile = (command._embedding_current_profile if corpus == "summary" else command._chunk_current_profile)(conn)
        identity = profile["identity_hash"]
        lease = command._acquire_embedding_backfill_lease(conn, ttl_s=600, heartbeat_s=60)
        conn.execute("INSERT INTO lcm_embedding_backfill_inflight "
                     "(embedded_id, identity_hash, lease_id, generation, state, request_id, updated_at) "
                     "VALUES (?, ?, ?, ?, 'dispatched', 'request', ?)",
                     (embedded_id, identity, lease.lease_id, lease.generation, time.time()))
    return engine, identity


@pytest.mark.parametrize("corpus", ["summary", "chunk"])
def test_all_readers_reclassify_expired_dispatch_without_backfill(tmp_path, monkeypatch, corpus):
    engine, identity = _dispatch_fixture(tmp_path, corpus)
    index = 0 if corpus == "summary" else 1
    prefix = "embedding_backlog_summaries" if index == 0 else "embedding_backlog_chunks"
    provider = FakeProvider()
    # Cached advice may come from a pass before the lease changed.
    command._record_embedding_catchup(engine._store.db_path, corpus, identity, 1, operator_retry=1)

    def remaining():
        if corpus == "summary":
            return command._embedding_backfill_remaining(engine._store.db_path, identity, 1, 0)
        return command._chunk_backfill_remaining(engine._store.db_path, identity, "conversational", 1, 0)

    for expired in (False, True):
        if expired:
            with sqlite3.connect(engine._store.db_path) as conn:
                conn.execute("UPDATE metadata SET value = json_set(value, '$.heartbeat_at', ?) WHERE key=?",
                             (time.time() - 601, command._EMBEDDING_BACKFILL_CLAIM_KEY))
        label = "uncertain 1" if expired else "in flight 1"
        assert command._embedding_maintenance_status_lines(engine)[index].endswith(f" ({label})")
        pending, live, uncertain = remaining()
        assert (live, uncertain) == ((0, 1) if expired else (1, 0))
        report = command._embedding_backfill_report(
            mode="apply", status="partial", provider="ollama", model="model-a", pending=1,
            selected=0, estimated_tokens=0, estimated_cost_tokens=0, estimated_batches=0,
            embedded=0, skipped=[], failed=[], remaining=pending, duration=0, consumed_tokens=0,
            in_flight=live, uncertain=uncertain,
        )
        assert f"in_flight: {live}\nuncertain_remote_acceptance: {uncertain}" in report
        assert ("--retry-uncertain" in report) is expired
        reason = tools._lcm_recall_catchup_reason(engine, provider, corpus)
        assert ("--retry-uncertain" in (reason or "")) is expired
    # Expiry classification is read-only: no recovery run changed the row.
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute("SELECT state FROM lcm_embedding_backfill_inflight").fetchone()[0] == "dispatched"
        conn.execute("DELETE FROM metadata WHERE key=?", (command._EMBEDDING_BACKFILL_CLAIM_KEY,))
    assert command._embedding_maintenance_status_lines(engine)[index].endswith(" (uncertain 1)")
    assert prefix in command._embedding_maintenance_status_lines(engine)[index]


@pytest.mark.parametrize("change", ["expired", "missing", "owner", "generation", "recovered"])
def test_doctor_uses_same_lease_split(tmp_path, monkeypatch, change):
    engine, identity = _dispatch_fixture(tmp_path, "summary")
    # Readiness only counts coverage for the configured FastEmbed model.
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE lcm_embedding_profile SET provider='fastembed'")
    engine._config.embedding_provider = "fastembed"
    monkeypatch.setattr(tools, "probe_provider_availability", lambda _c: dict(
        available=True, provider="fastembed", model="model-a"))
    from hermes_lcm import embedding_maintenance
    monkeypatch.setattr(embedding_maintenance, "local_model_state", lambda _c: None)

    def readiness():
        return tools._embedding_provider_health_check(engine)["detail"]["readiness"]

    assert "; in flight 1" in readiness()
    assert "uncertain" not in readiness()
    with sqlite3.connect(engine._store.db_path) as conn:
        if change == "recovered":
            conn.execute("UPDATE lcm_embedding_backfill_inflight SET state='uncertain'")
        elif change == "missing":
            conn.execute("DELETE FROM metadata WHERE key=?", (command._EMBEDDING_BACKFILL_CLAIM_KEY,))
        else:
            field, value = {"expired": ("heartbeat_at", time.time() - 601),
                            "owner": ("owner", "successor"), "generation": ("generation", 2)}[change]
            conn.execute("UPDATE metadata SET value=json_set(value, ?, ?) WHERE key=?",
                         (f"$.{field}", value, command._EMBEDDING_BACKFILL_CLAIM_KEY))
    assert "; uncertain 1" in readiness()
    assert "in flight" not in readiness()
