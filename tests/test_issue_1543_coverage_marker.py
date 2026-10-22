"""Coverage certificates bind both cursor ends and the content policy (#1543/#1550/#1624)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_issue_1536_chunk_revision import (
    LONG, SHORT, Store, add_rows, chunking, command, counts, metadata,
    store,  # noqa: F401 -- shared offline fake Ollama fixture
)

ROWS = [(1, "user", LONG), (2, "user", SHORT), (3, "assistant", LONG),
        (4, "user", "I work Fridays."), (5, "assistant", LONG),
        (6, "user", LONG), (7, "assistant", LONG)]


def raw(store):
    return list(store[1].connection.execute("SELECT * FROM messages ORDER BY store_id"))


def run(store, *, policy="conversational", limit=32, incremental=True):
    config = store[0]
    before = raw(store)
    result = command._chunk_backfill_run(
        config, config.database_path, apply=True, limit=limit,
        retry_uncertain=False, policy=policy, incremental=incremental,
    )
    assert isinstance(result, dict) and result["error"] is None and not result["failed"], result
    assert raw(store) == before


def marker(store):
    return json.loads(metadata(store, "chunk_policy_revision_through"))


def certify(store, bounds, policy="conversational"):
    _, vectors, identity, _ = store
    assert list(command._chunk_gapfill_rows(
        vectors.connection, identity.identity_hash, bounds, policy, 32, record_progress=True,
    )) == []


def older_walk(store, ids):
    """Only insert below the low: rewriting covered rows would wipe the proof."""
    _, vectors, identity, _ = store
    before = raw(store)
    proof = metadata(store, "chunk_policy_revision_through")
    low = command._chunk_cursor(vectors.connection, identity.identity_hash)[0]
    for row in vectors.connection.execute("SELECT store_id, role, content FROM messages ORDER BY store_id").fetchall():
        if row[0] in ids:
            assert row[0] < low and row[0] not in counts(store)
            for chunk in chunking.chunk_message(*row, policy="full"):
                vectors.record_chunk_embedding(
                    chunk.chunk_id, "model-a", [1.0, 0.0], store_id=chunk.store_id,
                    chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                    char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity,
                )
    assert metadata(store, "chunk_policy_revision_through") == proof
    assert raw(store) == before


def assert_status(store, policy, expected):
    config, vectors, _, _ = store
    config.embedding_content_policy = policy
    before = list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key"))
    messages = raw(store)
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
    assert f"embedding_backlog_chunks: {expected}" in command._embedding_maintenance_status_lines(engine)
    assert list(vectors.connection.execute("SELECT * FROM metadata ORDER BY key")) == before
    assert raw(store) == messages


@pytest.mark.parametrize("staged", [False, True], ids=["natural", "staged"])
def test_1543_low_side_round_trip(store: Store, staged):
    add_rows(store, ROWS, {6, 7} if staged else set())
    if staged:
        metadata(store, "chunk_policy_revision", 2)
        certify(store, (6, 7))
    else:
        run(store, limit=2)
        run(store, limit=1)
    older_walk(store, {1, 3} if not staged else {1, 3, 5})
    for _ in range(3):
        run(store)
    assert counts(store) == dict.fromkeys(range(1, 8), 1)
    assert marker(store) == {"from": 1, "through": 7, "policy": "conversational"}


def test_1624_backward_cursor_older_writer(store: Store):
    add_rows(store, [(i, "user", LONG) for i in range(1, 7)], {4, 5, 6})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (4, 6))
    older_walk(store, {2})
    for _ in range(3):
        run(store)
    assert counts(store) == dict.fromkeys(range(1, 7), 1)


@pytest.mark.parametrize("manual", [False, True], ids=["incremental", "manual"])
def test_1550_policy_switch_repairs_older_tools(store: Store, manual):
    add_rows(store, [(i, "tool" if i in (2, 4, 6) else "user", LONG) for i in range(1, 9)], set())
    for _ in range(3):
        run(store, incremental=not manual)
    assert set(counts(store)) == {1, 3, 5, 7, 8}
    for _ in range(3):
        run(store, policy="heads")
    assert counts(store) == dict.fromkeys(range(1, 9), 1)
    assert marker(store) == {"from": 1, "through": 8, "policy": "heads"}


@pytest.mark.parametrize("kind", ["band", "policy"])
def test_status_reports_unproven_candidates_without_writes(store: Store, kind):
    if kind == "band":
        add_rows(store, ROWS, {6, 7})
        metadata(store, "chunk_policy_revision", 2)
        certify(store, (6, 7))
        older_walk(store, {1, 3, 5})
        policy, pending = "conversational", 2
    else:
        add_rows(store, [(i, "tool" if i in (2, 4, 6) else "user", LONG) for i in range(1, 9)], set())
        run(store)
        run(store)
        policy, pending = "heads", 3
    assert_status(store, policy, f"{pending} (gap-fill)")


def test_repaired_steady_state_does_not_scan(store: Store, monkeypatch):
    add_rows(store, ROWS, {6, 7})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (6, 7))
    older_walk(store, {1, 3, 5})
    for _ in range(3):
        run(store)
    assert counts(store) == dict.fromkeys(range(1, 8), 1)

    def unexpected(*args, **kwargs):
        pytest.fail("steady coverage must not repeat gap-fill")

    monkeypatch.setattr(command, "_chunk_gapfill_rows", unexpected)
    run(store)
    assert_status(store, "conversational", "0")


def test_own_history_walk_scans_at_most_one_walked_band(store: Store, monkeypatch):
    add_rows(store, [(i, "user", LONG) for i in range(1, 18)], {16, 17})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (16, 17))
    gapfill = command._chunk_gapfill_rows
    bands = []

    def counted(*args, **kwargs):
        bands.append(args[2])
        yield from gapfill(*args, **kwargs)

    monkeypatch.setattr(command, "_chunk_gapfill_rows", counted)
    monkeypatch.setattr(command, "_CHUNK_HISTORY_SCAN_MESSAGES", 4)
    previous_low = 16
    previous_walk = 0
    for _ in range(6):
        bands.clear()
        run(store, limit=4)
        assert len(bands) <= 1
        if bands:
            assert bands == [(previous_low, previous_low + previous_walk - 1)]
            assert bands[0][1] - bands[0][0] + 1 <= 4
        low = command._chunk_cursor(store[1].connection, store[2].identity_hash)[0]
        previous_walk, previous_low = previous_low - low, low
    assert counts(store) == dict.fromkeys(range(1, 18), 1)
    assert marker(store) == {"from": 1, "through": 17, "policy": "conversational"}


@pytest.mark.parametrize("policy", ["conversational", "heads"])
def test_legacy_integer_trusted_for_default_and_revalidated_once_for_heads(store: Store, monkeypatch, policy):
    add_rows(store, [(1, "user", LONG), (2, "tool", LONG), (3, "user", LONG)], {1, 3})
    metadata(store, "chunk_policy_revision", 2)
    metadata(store, "chunk_policy_revision_through", 3)
    gapfill = command._chunk_gapfill_rows
    scans = []

    def counted(*args, **kwargs):
        scans.append(args[2])
        yield from gapfill(*args, **kwargs)

    monkeypatch.setattr(command, "_chunk_gapfill_rows", counted)
    for _ in range(3):
        run(store, policy=policy)
    assert scans == ([(1, 3), (1, 3)] if policy == "heads" else [])
    assert set(counts(store)) == ({1, 2, 3} if policy == "heads" else {1, 3})
    assert marker(store) == {"from": 1, "through": 3, "policy": policy}
    scans.clear()
    run(store, policy=policy)
    assert scans == []


def test_v0281_reader_falls_back_to_full_range_on_json(store: Store):
    add_rows(store, ROWS, {6, 7})
    metadata(store, "chunk_policy_revision", 2)
    certify(store, (6, 7))
    assert isinstance(marker(store), dict)
    namespace = {"CHUNK_POLICY_REVISION": command.CHUNK_POLICY_REVISION}
    source = Path(__file__).with_name("fixtures") / "issue_1543_v0281_gapfill_bounds.py.txt"
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    assert namespace["_chunk_gapfill_bounds"](store[1].connection, store[2].identity_hash, (6, 7), 2) == (6, 7)
