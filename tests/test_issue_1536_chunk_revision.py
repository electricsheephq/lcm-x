"""#1536: rollback cursor growth must invalidate only the unproven coverage range."""

from collections.abc import Iterator
import importlib
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

command = importlib.import_module("hermes_lcm.command")
chunking = importlib.import_module("hermes_lcm.chunking")
config_module = importlib.import_module("hermes_lcm.config")
vector_module = importlib.import_module("hermes_lcm.vector_store")

LONG = "The archive contains the complete synthetic migration history. " * 8
SHORT = "My bicycle is blue."
Store = tuple[Any, Any, Any, list[str]]


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    config = config_module.LCMConfig(
        database_path=str(tmp_path / "rollback.db"), embeddings_enabled=True,
        embedding_provider="ollama", embedding_model="model-a",
    )
    vectors = vector_module.VectorStore(config.database_path, config=config)
    vectors.ensure_chunk_schema()
    vectors.register_profile("model-a", "ollama", 2, task="chunk")
    identity = vector_module.EmbeddingIdentity.canonical(
        "ollama", "model-a", "", 2, "float32", "little", "chunk",
    )
    vectors.connection.execute(
        "CREATE TABLE messages(store_id INTEGER PRIMARY KEY, role TEXT, content TEXT, "
        "session_id TEXT NOT NULL DEFAULT 'synthetic', source TEXT DEFAULT '', "
        "timestamp REAL NOT NULL DEFAULT 0, conversation_id TEXT DEFAULT '')"
    )
    texts: list[str] = []

    def embed(documents: list[str]) -> list[list[float]]:
        texts.extend(documents)
        return [[1.0, 0.0] for _ in documents]

    provider = SimpleNamespace(provider_id="ollama", model_id="model-a",
                               last_skipped_documents=[], embed_documents=embed)
    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_k: provider)
    yield config, vectors, identity, texts
    vectors.close()


def add_rows(store: Store, rows: list[tuple[int, str, str]], embedded: set[int]) -> None:
    _, vectors, identity, _ = store
    vectors.connection.executemany("INSERT INTO messages(store_id, role, content) VALUES(?, ?, ?)", rows)
    # c67d7633 chunking.py:226-229 skips <40-token turns. Its command.py:5020-5026
    # then advances MIN/MAX from published long rows and never revisits enclosed shorts.
    for row in rows:
        if row[0] in embedded:
            for chunk in chunking.chunk_message(*row, policy="full"):
                vectors.record_chunk_embedding(
                    chunk.chunk_id, "model-a", [1.0, 0.0], store_id=chunk.store_id,
                    chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                    char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity,
                )


def metadata(store: Store, key: str, value: int | None = None) -> str | None:
    _, vectors, identity, _ = store
    key = f"{key}:{identity.identity_hash}"
    if value is not None:
        vectors.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)", (key, str(value)))
    row = vectors.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def run(store: Store, *, limit: int = 32, incremental: bool = True) -> None:
    config, vectors, _, _ = store
    before = list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
    result = command._chunk_backfill_run(
        config, config.database_path, apply=True, limit=limit,
        retry_uncertain=False, policy="conversational", incremental=incremental,
    )
    assert isinstance(result, dict) and result["error"] is None and not result["failed"], result
    assert before == list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))


def counts(store: Store) -> dict[int, int]:
    _, vectors, identity, _ = store
    return dict(vectors.connection.execute(
        "SELECT store_id, COUNT(*) FROM lcm_chunk_meta WHERE identity_hash=? GROUP BY store_id",
        (identity.identity_hash,),
    ))


@pytest.mark.parametrize("bound", [None, 3])
def test_completed_marker_repairs_rollback_and_already_affected_store(store: Store, bound: int | None) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG)], {1, 2, 3})
    metadata(store, "chunk_policy_revision", 2)  # rc1 completed its original fill.
    if bound is not None:
        metadata(store, "chunk_policy_revision_through", bound)
    # v0.27.0 publishes long 5 but leaves short 4 enclosed; all marker keys survive.
    add_rows(store, [(4, "user", "I work Fridays."), (5, "assistant", LONG)], {5})
    for _ in range(3):
        run(store)
    assert counts(store) == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    assert store[3] == ["I work Fridays."]
    assert metadata(store, "chunk_policy_revision") == "2"
    assert metadata(store, "chunk_policy_revision_through") == "5"
    run(store)
    assert store[3] == ["I work Fridays."]


@pytest.mark.parametrize("bound", [None, 3])
def test_uncovered_status_is_read_only_and_reports_gapfill(store: Store, bound: int | None) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG),
                     (4, "user", SHORT), (5, "assistant", LONG)], {1, 2, 3, 5})
    metadata(store, "chunk_policy_revision", 2)
    if bound is not None:
        metadata(store, "chunk_policy_revision_through", bound)
    config, vectors, _, _ = store
    before = list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
    assert "embedding_backlog_chunks: 1 (gap-fill)" in command._embedding_maintenance_status_lines(engine)
    assert before == list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))


@pytest.mark.parametrize("scoped_skip", [False, True])
def test_mid_fill_skip_does_not_hide_new_rollback_rows(store: Store, scoped_skip: bool) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "user", "I work Fridays."),
                     (4, "user", "\t\n"), (5, "assistant", LONG)], {1, 3, 5})
    metadata(store, "chunk_gapfill_skip_below", 4)  # rc1 stopped after publishing short 3.
    if scoped_skip:
        metadata(store, "chunk_gapfill_skip_through", 5)
    add_rows(store, [(6, "user", "My cat is grey."), (7, "assistant", LONG)], {7})
    for _ in range(5):
        run(store, limit=1)
    assert counts(store) == {1: 1, 2: 1, 3: 1, 5: 1, 6: 1, 7: 1}
    assert store[3] == ["My cat is grey.", SHORT]
    assert metadata(store, "chunk_policy_revision_through") == "7"


def test_legacy_marker_with_full_coverage_is_revalidated_once(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG)], {1, 2, 3})
    metadata(store, "chunk_policy_revision", 2)
    run(store)
    assert metadata(store, "chunk_policy_revision_through") == "3"

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("steady coverage must not repeat gap-fill")

    monkeypatch.setattr(command, "_chunk_gapfill_rows", unexpected)
    run(store)
    assert not store[3]


def test_unpublished_gaps_do_not_advance_coverage_bound(store: Store) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "user", "I work Fridays."),
                     (4, "assistant", LONG)], {1, 4})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 1)
    run(store, limit=1)
    assert counts(store) == {1: 1, 3: 1, 4: 1}
    assert metadata(store, "chunk_policy_revision_through") == "1"
    for _ in range(3):
        run(store, limit=1)
    assert counts(store) == {1: 1, 2: 1, 3: 1, 4: 1}
    assert metadata(store, "chunk_policy_revision_through") == "4"


def test_full_backfill_stamps_coverage_bound(store: Store) -> None:
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG)], {1, 3})
    run(store, incremental=False)
    assert counts(store) == {1: 1, 2: 1, 3: 1}
    assert metadata(store, "chunk_policy_revision_through") == "3"


@pytest.mark.parametrize("kind", ["whitespace", "heads", "provider"])
def test_exceptional_gap_scan_stays_flat_with_appends(
    store: Store, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    config, vectors, identity, _ = store
    policy = "heads" if kind == "heads" else "conversational"
    hole = "REFUSE this document" if kind == "provider" else "\n\t"
    role = "tool" if kind == "heads" else "user"
    add_rows(store, [(i, "user", LONG) for i in range(1, 101)], set(range(1, 101)))
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 100)
    add_rows(store, [(101, role, hole), (102, "assistant", LONG)], {102})
    config.embedding_max_batch_items = 1

    def embed(documents: list[str]) -> list[list[float]]:
        if hole in documents:
            raise command.ProviderPreDispatchError("synthetic local rejection")
        return [[1.0, 0.0] for _ in documents]

    provider = SimpleNamespace(provider_id="ollama", model_id="model-a",
                               last_skipped_documents=[], embed_documents=embed)
    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_kw: provider)
    sql: list[str] = []
    connect = sqlite3.connect

    def traced(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn: sqlite3.Connection = connect(*args, **kwargs)
        conn.set_trace_callback(sql.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", traced)
    visits: list[int] = []
    statuses: list[str] = []
    for turn in range(24):
        high = 102 + 8 * turn
        add_rows(store, [(high + i, "user", LONG) for i in range(1, 9)], set())
        before = list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
        sql.clear()
        result = command._chunk_backfill_run(
            config, config.database_path, apply=True, limit=32, retry_uncertain=False,
            policy=policy, incremental=True,
        )
        assert isinstance(result, dict), result
        if kind != "provider":
            assert result["error"] is None and not result["failed"], result
        scanned = 0
        for statement in sql:
            if "FROM messages AS m" not in statement:
                continue
            bounds = re.search(r"store_id >= (\d+) AND store_id <= (\d+) AND store_id < (\d+)", statement)
            assert bounds is not None, statement
            low, upper, ceiling = map(int, bounds.groups())
            scanned += max(0, min(upper + 1, ceiling) - low)
        visits.append(scanned)
        statuses.append(command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, policy))
        assert before == list(vectors.connection.execute("SELECT * FROM messages ORDER BY store_id"))
    assert max(visits[2:]) <= visits[1] + 16, visits
    assert statuses == (["1 (gap-fill)"] * 24 if kind == "provider" else ["0"] * 24)
    assert set(counts(store)) == set(range(1, 295)) - {101}


def test_rejected_gap_remains_backlog_then_recovers_within_two_passes(
    store: Store, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, vectors, identity, _ = store
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG)], {1, 3})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 1)
    rejecting = True

    def embed(documents: list[str]) -> list[list[float]]:
        if rejecting and SHORT in documents:
            raise command.ProviderPreDispatchError("synthetic local rejection")
        return [[1.0, 0.0] for _ in documents]

    provider = SimpleNamespace(provider_id="ollama", model_id="model-a",
                               last_skipped_documents=[], embed_documents=embed)
    monkeypatch.setattr(command, "resolve_provider", lambda *_a, **_kw: provider)
    for turn in range(5):
        add_rows(store, [(turn + 4, "user", LONG)], set())
        result = command._chunk_backfill_run(
            config, config.database_path, apply=True, limit=32, retry_uncertain=False,
            policy="conversational", incremental=True,
        )
        assert isinstance(result, dict) and result["failed"], result
        assert 2 not in counts(store)
        assert "(gap-fill)" in command._chunk_backlog_estimate(
            vectors.connection, identity.identity_hash, "conversational",
        )
    rejecting = False
    run(store)
    run(store)
    assert counts(store) == dict.fromkeys(range(1, 9), 1)
    assert command._chunk_backlog_estimate(vectors.connection, identity.identity_hash, "conversational") == "0"


def test_healthy_appends_have_no_gapfill_status_suffix(store: Store) -> None:
    add_rows(store, [(1, "user", LONG)], {1})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 1)
    for turn in range(20):
        add_rows(store, [(turn + 2, "user", LONG)], set())
        run(store)
        config, vectors, _, _ = store
        before = list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
        engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
        assert "embedding_backlog_chunks: 0" in command._embedding_maintenance_status_lines(engine)
        assert before == list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))


@pytest.mark.parametrize("pending_count", [500, 501, 1000, 1999])
def test_pending_recheck_respects_sqlite_variable_limit(
    store: Store, monkeypatch: pytest.MonkeyPatch, pending_count: int,
) -> None:
    _, vectors, identity, _ = store
    high = pending_count + 2
    add_rows(store, [(1, "user", LONG),
                     *[(i, "user", SHORT) for i in range(2, high)],
                     (high, "assistant", LONG)], {1, high})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 1)
    selected = list(command._chunk_frontier_rows(
        vectors.connection, identity.identity_hash, policy="conversational", record_progress=True,
    ))
    assert {row[0] for row in selected} == set(range(2, high))
    saved = metadata(store, "chunk_gapfill_scan")
    assert saved is not None and set(json.loads(saved)["pending"]) == set(range(2, high))
    connect = sqlite3.connect

    def limited(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn: sqlite3.Connection = connect(*args, **kwargs)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        return conn

    monkeypatch.setattr(sqlite3, "connect", limited)
    vectors.connection.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    run(store, limit=pending_count)
    assert counts(store) == dict.fromkeys(range(1, high + 1), 1)


@pytest.mark.parametrize("operation", ["status", "apply"])
@pytest.mark.parametrize("record", [
    "not-json", '{"revision":2', "[1,2,3]",
    '{"revision":2,"policy":"conversational","pending":[3]}',
    '{"revision":2,"policy":"conversational","high":"5","pending":[3]}',
    *[json.dumps({"revision": 2, "policy": "conversational", "high": 5, "pending": [3], **invalid})
      for invalid in ({"revision": "2"}, {"revision": True}, {"high": False},
                      {"policy": 1}, {"pending": "3"}, {"pending": ["3"]},
                      {"pending": [True]}, {"ceiling": "5"})],
])
def test_invalid_scan_record_revalidates_full_gap(store: Store, record: str, operation: str) -> None:
    config, vectors, identity, _ = store
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "user", SHORT),
                     (4, "user", SHORT), (5, "assistant", LONG)], {1, 5})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 1)
    vectors.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)",
                               (f"chunk_gapfill_scan:{identity.identity_hash}", record))
    if operation == "status":
        before = list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
        engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
        assert "embedding_backlog_chunks: 3 (gap-fill)" in command._embedding_maintenance_status_lines(engine)
        assert before == list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
    else:
        run(store)
        assert counts(store) == dict.fromkeys(range(1, 6), 1)


@pytest.mark.parametrize("operation", ["status", "apply"])
@pytest.mark.parametrize("bound", ["not-a-number", "", "1.5", "[1]"])
def test_unreadable_revision_bound_revalidates_full_gap(store: Store, bound: str, operation: str) -> None:
    config, vectors, identity, _ = store
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "user", SHORT),
                     (4, "user", SHORT), (5, "assistant", LONG)], {1, 5})
    metadata(store, "chunk_policy_revision", 2)
    vectors.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)",
                               (f"chunk_policy_revision_through:{identity.identity_hash}", bound))
    if operation == "status":
        before = list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
        engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
        assert "embedding_backlog_chunks: 3 (gap-fill)" in command._embedding_maintenance_status_lines(engine)
        assert before == list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
    else:
        run(store)
        assert counts(store) == dict.fromkeys(range(1, 6), 1)


@pytest.mark.parametrize("extra", [{}, {"format": 2, "cursor": {"low": 1, "high": 5}},
                                   {"pending": [3, *range(1000, 1600)]}])
def test_scan_record_from_a_newer_revision_is_not_trusted(store: Store, extra: dict[str, Any]) -> None:
    _, vectors, identity, _ = store
    add_rows(store, [(1, "user", LONG), (2, "user", SHORT), (3, "user", SHORT),
                     (4, "user", SHORT), (5, "assistant", LONG)], {1, 5})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 1)
    newer = {"revision": command.CHUNK_POLICY_REVISION + 1, "policy": "conversational",
             "high": 5, "pending": [3], **extra}
    vectors.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, ?)",
                               (f"chunk_gapfill_scan:{identity.identity_hash}", json.dumps(newer)))
    run(store)
    # Trusting the newer record would recheck only row 3 and skip rows 2 and 4.
    assert counts(store) == dict.fromkeys(range(1, 6), 1)
