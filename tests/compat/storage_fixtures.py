"""Offline storage fixtures shared with the pinned previous-GA subprocess."""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from tests.compat.helpers import FLEET_POLICY

SESSION = "compat-storage"
CONVERSATION = "compat-conversation"
VECTOR_TABLES = (
    "lcm_embedding_profile", "lcm_embedding_meta", "lcm_embedding_vectors", "lcm_embedding_binary",
    "lcm_chunk_meta", "lcm_chunk_vectors", "lcm_chunk_binary",
)


def plugin(name: str) -> Any:
    # pytest and the old-process bootstrap register different pinned package roots.
    return importlib.import_module(f"hermes_lcm.{name}")


class OfflineProvider:
    provider_id = "ollama"
    model_id = "compat-model"
    dim = 2
    last_usage_tokens = 1

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.last_skipped_documents: list[int] = []

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [1.0 + digest[0] / 255, 1.0 + digest[1] / 255]

    def embed_document_batches(
        self, texts: Sequence[str], *, before_dispatch: Callable[[tuple[int, ...]], None],
    ) -> Iterator[Any]:
        indexes = tuple(range(len(texts)))
        before_dispatch(indexes)
        self.documents.extend(texts)
        yield plugin("embedding_provider").EmbeddedDocumentBatch(
            indexes, tuple(tuple(self.embed_query(text)) for text in texts))


def summary(text: str = "", **kwargs: Any) -> tuple[str, int]:
    return "Synthetic summary " + hashlib.sha256(text.encode()).hexdigest(), 1


def install_fakes(monkeypatch: pytest.MonkeyPatch | None = None) -> OfflineProvider:
    provider = OfflineProvider()

    def replace(target: Any, name: str, value: Any) -> None:
        if monkeypatch is None:
            setattr(target, name, value)
        else:
            monkeypatch.setattr(target, name, value)

    replace(plugin("command"), "resolve_provider", lambda *args, **kwargs: provider)
    replace(plugin("embedding_provider"), "probe_provider_availability",
            lambda *args, **kwargs: {"available": True, "detail": "offline fixture"})
    replace(plugin("engine"), "summarize_with_escalation", summary)
    replace(plugin("rollup_builder"), "summarize_with_escalation", summary)
    return provider


def storage_config(directory: Path, *, fleet: bool = False, embeddings: bool = False,
                   dtype: str = "float32", small: bool = False) -> Any:
    options: dict[str, Any] = {}
    if fleet:
        options.update(FLEET_POLICY)
    if small:
        options.update(fresh_tail_count=4, leaf_chunk_tokens=1000)
    if embeddings:
        options.update(embeddings_enabled=True, embedding_provider="ollama",
                       embedding_model=OfflineProvider.model_id, embedding_storage_dtype=dtype)
    return plugin("config").LCMConfig(
        database_path=str(directory / "lcm.db"),
        large_output_externalization_path=str(directory / "payloads"), **options)


def bind(engine: Any) -> None:
    engine.on_session_start(SESSION, platform="cli", context_length=200_000,
                            conversation_id=CONVERSATION)
    assert engine.drain_rollup_maintenance(timeout=10)


def seed_history(engine: Any, label: str, *, turns: int = 40) -> None:
    view: list[dict[str, Any]] = [
        {"role": "user", "content": f"{label} run the report", "timestamp": 100.0},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": label, "type": "function", "function": {"name": "report", "arguments": "{}"}}]},
        {"role": "tool", "name": "report", "tool_call_id": label,
         "content": f"{label} payload " + "report data " * 5000},
        {"role": "assistant", "content": f"{label} report complete", "timestamp": 101.0},
    ]
    for turn in range(turns):
        for offset, role in enumerate(("user", "assistant")):
            view.append({"role": role, "content": f"{label} {role} {turn}: " + "detail " * 400,
                         "timestamp": 110.0 + 2 * turn + offset})
    engine.ingest(view)
    engine.compress(view, current_tokens=180_000, force=True)
    assert engine._dag.get_session_nodes(SESSION), "public compaction must publish summaries"


def rebuild_rollup(engine: Any) -> None:
    assert engine.drain_rollup_maintenance(timeout=10)
    node = engine._dag.get_session_nodes(SESSION)[0]
    day = datetime.fromtimestamp(node.earliest_at or node.created_at, timezone.utc).date()
    result = plugin("command").handle_lcm_command(f"rollups rebuild day {day}", engine)
    assert "status: complete" in result, result
    conn = engine._store.connection
    assert conn.execute("SELECT count(*) FROM lcm_rollups WHERE status='ready'").fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM lcm_rollup_sources").fetchone()[0] > 0


def embed_all(engine: Any, *, limit: int | None = None) -> str:
    suffix = "" if limit is None else f" --limit {limit}"
    result: str = plugin("command").handle_lcm_command(
        "embed backfill --corpus both --apply" + suffix, engine)
    assert result.count("status: complete") == 2, result
    return result


def warmup(engine: Any) -> None:
    result = plugin("command").handle_lcm_command("embed warmup", engine)
    assert "status: ready" in result, result


def table_hashes(directory: Path, tables: Sequence[str]) -> dict[str, dict[int, str]]:
    """Hash complete SQLite rows, including lossless BLOB bytes, as snapshot() does."""
    with sqlite3.connect(directory / "lcm.db") as conn:
        return {table: {row[0]: hashlib.sha256(json.dumps(
            row[1:], default=lambda value: {"blob_hex": value.hex()}).encode()).hexdigest()
            for row in conn.execute(f'SELECT rowid, * FROM "{table}" ORDER BY rowid')}
            for table in tables}


def payload_hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (directory / "payloads").rglob("*") if path.is_file()}


def assert_preserved(directory: Path, before: dict[str, dict[int, str]]) -> None:
    after = table_hashes(directory, tuple(before))
    for table, rows in before.items():
        assert rows.items() <= after[table].items(), f"pre-existing rows changed in {table}"


def vector_ids(engine: Any) -> tuple[set[str], set[str]]:
    conn = engine._store.connection
    return (set(row[0] for row in conn.execute("SELECT embedded_id FROM lcm_embedding_vectors")),
            set(row[0] for row in conn.execute("SELECT chunk_id FROM lcm_chunk_vectors")))


def assert_complete_vectors(engine: Any) -> None:
    conn = engine._store.connection
    summaries = {str(row[0]) for row in conn.execute("SELECT node_id FROM summary_nodes WHERE depth=0")}
    expected_chunks = {chunk.chunk_id for row in conn.execute("SELECT store_id, role, content FROM messages")
                       for chunk in plugin("command").chunk_message(*row, policy="conversational")}
    actual_summaries, actual_chunks = vector_ids(engine)
    assert actual_summaries == summaries and summaries
    assert actual_chunks == expected_chunks and expected_chunks
    for meta, vectors, key in (("lcm_embedding_meta", "lcm_embedding_vectors", "embedded_id"),
                               ("lcm_chunk_meta", "lcm_chunk_vectors", "chunk_id")):
        ids = [row[0] for row in conn.execute(f"SELECT {key} FROM {meta}")]
        assert len(ids) == len(set(ids)), f"duplicate source ids in {meta}"
        assert set(ids) == {row[0] for row in conn.execute(f"SELECT {key} FROM {vectors}")}
    # The store connection can report false FTS5 corruption after summary writes
    # through the other connection merge segments (#1578), so check a fresh one.
    with closing(sqlite3.connect(engine._store.db_path)) as fresh:
        assert fresh.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def assert_calls_once(provider: OfflineProvider, previous: tuple[set[str], set[str]], engine: Any) -> None:
    current = vector_ids(engine)
    new_summaries, new_chunks = (now - old for now, old in zip(current, previous))
    conn = engine._store.connection
    expected = [str(row[1]) for row in conn.execute("SELECT node_id, summary FROM summary_nodes")
                if str(row[0]) in new_summaries]
    expected.extend(chunk.text for row in conn.execute("SELECT store_id, role, content FROM messages")
                    for chunk in plugin("command").chunk_message(*row, policy="conversational")
                    if chunk.chunk_id in new_chunks)
    assert Counter(provider.documents) == Counter(expected)


def finish_embeddings(engine: Any) -> None:
    """Drain queued work, then run bounded incremental passes without sleeps."""
    assert engine.drain_embedding_maintenance(timeout=10)
    for _ in range(8):
        before = vector_ids(engine)
        outcome = plugin("embedding_maintenance").run_incremental_embedding_pass(
            engine._config.database_path, engine._config)
        assert "error" not in outcome and "refused" not in outcome, outcome
        if vector_ids(engine) == before:
            break
    else:
        raise AssertionError("bounded incremental catch-up did not converge")
    assert_complete_vectors(engine)
