"""Frozen real-runner backfill traces; privacy APIs are opaque fixture boundaries.

Regenerate deliberately with --regenerate. SQL retains whitespace and ordering;
large events use the ingest oracle's sha12/length encoding and optional payloads.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, tzinfo
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
from types import FunctionType, MethodType, ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator
import uuid
from urllib.parse import quote
from enum import Enum

import pytest

if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--repo", type=Path)
    selected, _ = pre.parse_known_args()
    package = ModuleType("hermes_lcm")
    package.__path__ = [str((selected.repo or Path(__file__).resolve().parents[2]).resolve())]
    sys.modules["hermes_lcm"] = package

import tests.conftest  # noqa: E402, F401
from tests.compat.storage_fixtures import OfflineProvider, storage_config  # noqa: E402
from tests.behaviour.reconcile_trace_harness import first_difference as first_difference  # noqa: E402
from tests.phase0_hygiene import product_env_keys  # noqa: E402
from tests.test_embedding_backfill import _seed  # noqa: E402
from tests.test_chunk_backfill import _seed_messages  # noqa: E402

command: Any = importlib.import_module("hermes_lcm.command")
vectors: Any = importlib.import_module("hermes_lcm.vector_store")
provider_module: Any = importlib.import_module("hermes_lcm.embedding_provider")
ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/fixtures/backfill_trace.json"
TRACE_GUIDANCE = "Regenerate deliberately: python -m tests.behaviour.backfill_trace_harness --repo . --regenerate"
COMMON = (
    "disabled", "db-open-error", "profile-query-error", "dtype-mismatch",
    "provider-missing", "provider-mismatch", "dry-empty", "dry-limited", "dry-token-cap",
    "apply-empty", "apply-one", "page-B-minus-1", "page-B", "page-B-plus-1",
    "limit-then-reopen-resume", "lease-held", "identity-switch-before-dispatch",
    "lease-renew-lost-after-page", "budget-after-page", "stop-before-dispatch",
    "stop-after-accepted-subbatch", "provider-auth-rejection", "provider-ambiguous-after-subbatch",
    "publish-call-error", "publish-row-error", "publication-owner-loss",
    "publication-identity-superseded", "release-failure", "invalid-accepted-index",
    "unresolved-request", "profile-absent-dry", "invalid-dispatch-index",
    "dtype-match", "read-open-non-sqlite-error", "profile-read-non-sqlite-error",
    "discovery-read-non-sqlite-error",
)
CHUNK_ONLY = (
    "cloud-dry-consent-off", "cloud-dry-consent-on", "cloud-apply-consent-off",
    "cloud-apply-consent-on", "profile-absent-apply", "incremental-boundary-resume",
    "incremental-revision-gap-checkpoint", "contextual-model-cap-and-estimates",
)
# Independent, frozen observer inventories. New calls/writes must be registered.
STORE_PRIMITIVES = frozenset({
    "_as_node_id", "_encode_stored_vector", "_ensure_embedding_schema", "_identity_from_profile", "_migration_step_present",
    "_init_db", "_table_columns", "_now", "_write_transaction", "capture_identity",
    "_resolve_profile", "_current_profile", "ensure_chunk_schema", "_chunk_tables_exist",
    "publish_embedding_batch_under_lease", "publish_chunk_embedding_batch_under_lease",
    "_run_publish_batch", "publish_embedding_under_lease", "publish_chunk_embedding_under_lease",
    "_publish_under_lease", "_profile_by_identity", "_write_embedding_row", "_write_chunk_row",
    "_validate_embedding_write", "_normalized", "_bump_data_version", "close",
})
PROVIDER_PRIMITIVES = frozenset({
    "embed_query", "embed_documents", "embed_document_batches", "embed_chunk_group_batches",
})
WRITE_TABLES = frozenset({
    "sqlite_master", "sqlite_temp_master", "metadata", "lcm_migration_state", "lcm_embedding_profile",
    "lcm_embedding_meta", "lcm_embedding_vectors", "lcm_embedding_binary",
    "lcm_chunk_meta", "lcm_chunk_vectors", "lcm_chunk_binary", "lcm_embedding_backfill_inflight",
    "lcm_lifecycle_state", "lcm_rollups", "lcm_rollup_sources", "lcm_rollup_invalidations",
    "lcm_rollup_state", "lcm_assertion_sources", "lcm_assertions", "lcm_assertion_relations",
    "messages", "summary_nodes", "lcm_embedding_backfill_predecessor_batch",
})
SNAPSHOT_TABLES = tuple(sorted(WRITE_TABLES - {"sqlite_master", "sqlite_temp_master"}))
WRITE_ACTIONS = frozenset({sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                           sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE, sqlite3.SQLITE_ALTER_TABLE})


@dataclasses.dataclass(frozen=True)
class Scenario:
    runner: str
    shape: str

    @property
    def name(self) -> str:
        return self.runner + "/" + self.shape

    @property
    def group(self) -> str:
        if self.shape in CHUNK_ONLY:
            return "chunk-only"
        if self.shape in COMMON[:6] or self.shape == "profile-absent-dry":
            return "admission"
        if self.shape.startswith("dry-"):
            return "preview"
        if self.shape in COMMON[9:15]:
            return "pages"
        if self.shape in COMMON[17:21]:
            return "interruption"
        return "dispatch/recovery"


SCENARIOS = tuple(Scenario(runner, shape) for runner in ("summary", "chunks") for shape in COMMON) + tuple(
    Scenario("chunks", shape) for shape in CHUNK_ONLY)


class Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.ordinal = 0

    def monotonic(self) -> float:
        return 100.0 + self.elapsed

    def time(self) -> float:
        return 1_700_000_000.0 + self.elapsed

    def nonce(self) -> uuid.UUID:
        self.ordinal += 1
        return uuid.UUID(int=self.ordinal)


class Observer:
    def __init__(self, root: Path, payloads: list[str] | None = None) -> None:
        self.root, self.payloads = str(root), payloads
        self.roots = sorted({self.root, os.path.realpath(root),
                             quote(self.root), quote(os.path.realpath(root))}, key=lambda item: (-len(item), item))
        self.events: list[str] = []
        self.errors: list[str] = []
        self.called: set[str] = set()
        self.written: set[str] = set()
        self.quiet = False
        self.active = True
        self.connections: list[Connection] = []
        self.read_connect = sqlite3.connect
        self.aliases: dict[int, str] = {}
        self.retained: list[Any] = []
        self.fault_used = False
        self.commit_fault = False
        self.case = Scenario("summary", "apply-one")

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            for root in self.roots:
                value = value.replace(root, "<sandbox>")
            return value
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, Enum):
            return {"enum": type(value).__name__, "value": self.value(value.value)}
        if isinstance(value, BaseException):
            return [type(value).__name__, str(value), self.value(value.__cause__)]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {field.name: self.value(getattr(value, field.name)) for field in dataclasses.fields(value)}
        if isinstance(value, sqlite3.Row):
            return {key: self.value(value[key]) for key in value.keys()}
        if isinstance(value, dict):
            return {str(key): self.value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(self.value(item) for item in value)
        if callable(value):
            return "<callback>"
        return {"type": type(value).__name__}

    def emit(self, label: str, value: Any) -> None:
        if self.quiet or not self.active:
            return
        text = json.dumps(self.value(value), ensure_ascii=False, separators=(",", ":"))
        if self.payloads is not None:
            self.payloads.append(label + " " + text)
        compact = text if len(text) <= 100 else "sha12/len=" + hashlib.sha256(text.encode()).hexdigest()[:12] + "/" + str(len(text))
        self.events.append(label + " " + compact)

    def alias(self, value: Any) -> str:
        if id(value) not in self.aliases:
            self.aliases[id(value)] = "object-" + str(len(self.aliases) + 1)
            self.retained.append(value)
        return self.aliases[id(value)]

    def primitive(self, name: str) -> None:
        self.called.add(name)
        if name not in STORE_PRIMITIVES:
            self.errors.append("unregistered store primitive: " + name)

    def authorizer(self, action: int, table: str | None, column: str | None, db: str | None, trigger: str | None) -> int:
        if self.quiet:
            return sqlite3.SQLITE_OK
        if action in WRITE_ACTIONS and table:
            self.written.add(table)
            if table not in WRITE_TABLES:
                self.errors.append("unregistered table write: " + table)
                return sqlite3.SQLITE_DENY
        if (not self.fault_used and self.case.shape == "profile-query-error"
                and action == sqlite3.SQLITE_READ and table == "lcm_embedding_profile"):
            self.fault_used = True
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def files(self, root: Path) -> None:
        # Only this case's private sandbox; never inspect unrelated disk state.
        self.emit("files", sorted(path.relative_to(root).as_posix()
                                  for path in root.rglob("*") if path.is_file()))

    def snapshot(self, path: Path, label: str) -> None:
        previous, self.quiet = self.quiet, True
        try:
            with self.read_connect(path) as conn:
                existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                tables = {name: [tuple(row) for row in conn.execute(f'SELECT * FROM "{name}" ORDER BY 1, 2')]
                          for name in SNAPSHOT_TABLES if name in existing}
                identities = [row[0] for row in conn.execute("SELECT identity_hash FROM lcm_embedding_profile WHERE active=1")]
                cursors = {identity: command._chunk_cursor(conn, identity) for identity in identities}
            conn.close()
        finally:
            self.quiet = previous
        self.emit(label, {"tables": tables, "cursors": cursors})

    def complete(self) -> None:
        assert not self.errors, "; ".join(dict.fromkeys(self.errors))


class Connection(sqlite3.Connection):
    observer: Observer
    label: str

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        observer = self.observer
        release = (observer.case.shape == "release-failure" and not observer.fault_used
                   and sql.lstrip().startswith("DELETE FROM metadata"))
        if release:
            observer.fault_used = True
            self.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DELETE else sqlite3.SQLITE_OK)
        try:
            fault_table = {"profile-read-non-sqlite-error": "lcm_embedding_profile",
                           "discovery-read-non-sqlite-error": "summary_nodes" if observer.case.runner == "summary" else "messages"}
            if (not observer.fault_used and self.label == "connection-1"
                    and fault_table.get(observer.case.shape, "__absent__") in sql):
                observer.fault_used = True
                raise RuntimeError("scripted read failure")
            return super().execute(sql, parameters)
        except Exception as exc:
            observer.emit("sql raises " + self.label, [sql, parameters, exc])
            raise
        finally:
            if release:
                self.set_authorizer(observer.authorizer)

    def commit(self) -> None:
        if self.observer.commit_fault:
            self.observer.emit("commit raises " + self.label, "scripted commit failure")
            raise RuntimeError("scripted commit failure")
        super().commit()

    def close(self) -> None:
        self.observer.emit("connection close", self.label)
        super().close()


class Logs(logging.Handler):
    def __init__(self, observer: Observer) -> None:
        super().__init__()
        self.observer = observer

    def emit(self, record: logging.LogRecord) -> None:
        self.observer.emit("log", [record.name, record.levelname, record.msg, record.args])


class ScriptedProvider(OfflineProvider):
    def __init__(self, case: Scenario, observer: Observer, clock: Clock, stop: threading.Event,
                 config: Any) -> None:
        super().__init__()
        self.case, self.observer, self.clock, self.stop = case, observer, clock, stop
        self.provider_id, self.model_id = config.embedding_provider, config.embedding_model
        self.supports_contextualized_grouping = case.shape == "contextual-model-cap-and-estimates"

    def __getattribute__(self, name: str) -> Any:
        original = super().__getattribute__(name)
        if name.startswith("_") or not isinstance(original, (FunctionType, MethodType)):
            return original
        def observed(*args: Any, **kwargs: Any) -> Any:
            observer = self.observer
            if name not in PROVIDER_PRIMITIVES:
                observer.errors.append("unregistered provider method: " + name)
                raise AssertionError(observer.errors[-1])
            observer.emit("provider call " + name, [args, kwargs])
            return original(*args, **kwargs)
        return observed

    def embed_documents(self, texts: Any) -> list[list[float]]:
        self.documents.extend(texts)
        return [self.embed_query(text) for text in texts]

    def embed_document_batches(self, texts: Any, *, before_dispatch: Callable[[tuple[int, ...]], None]) -> Iterator[Any]:
        texts = list(texts)
        o, shape = self.observer, self.case.shape
        self.last_skipped_documents = []
        o.emit("provider input", texts)
        if shape == "invalid-dispatch-index":
            o.emit("callback indexes", (len(texts),))
            before_dispatch((len(texts),))
            return
        indexes = tuple(range(len(texts)))
        if shape in {"stop-after-accepted-subbatch", "provider-ambiguous-after-subbatch"}:
            indexes = (0,)
        o.emit("callback indexes", indexes)
        before_dispatch(indexes)
        o.emit("provider dispatch", indexes)
        self.documents.extend(texts[index] for index in indexes)
        if shape == "provider-auth-rejection":
            raise provider_module.VoyageError("auth", "scripted rejection")
        if shape == "unresolved-request":
            return
        accepted = (len(texts),) if shape == "invalid-accepted-index" else indexes
        if shape == "dry-token-cap":
            self.last_skipped_documents = [i for i, text in enumerate(texts) if len(text) > 27000]
            accepted = tuple(i for i in indexes if i not in self.last_skipped_documents)
        batch = provider_module.EmbeddedDocumentBatch(accepted, tuple(tuple(self.embed_query(texts[i])) for i in indexes[:len(accepted)]))
        o.emit("provider accepted", batch)
        if shape == "stop-after-accepted-subbatch":
            self.stop.set()
        yield batch
        if shape == "provider-ambiguous-after-subbatch" and len(texts) > 1:
            indexes = tuple(range(1, len(texts)))
            o.emit("callback indexes", indexes)
            before_dispatch(indexes)
            o.emit("provider dispatch", indexes)
            raise provider_module.EmbeddingProviderError("scripted ambiguous acceptance")

    def embed_chunk_group_batches(self, groups: Any, *, before_dispatch: Callable[[tuple[int, ...]], None]) -> Iterator[Any]:
        # Real contextual planner, fake accepted responses; no provider transport.
        self.observer.emit("provider groups", groups)
        self.last_skipped_documents = [index for group in groups for index, text in group
                                       if len(text) > provider_module._VOYAGE_CONTEXT_MAX_CHUNK_TOKENS]
        kept = [[(index, text) for index, text in group if index not in self.last_skipped_documents]
                for group in groups]
        plans = provider_module._plan_contextualized_requests(
            [[len(text) for _, text in group] for group in kept],
            doc_token_budget=provider_module._VOYAGE_CONTEXT_DOCUMENT_TOKEN_BUDGET,
            request_token_budget=provider_module._VOYAGE_CONTEXT_REQUEST_TOKEN_BUDGET,
            request_chunk_budget=provider_module._VOYAGE_CONTEXT_MAX_REQUEST_CHUNKS, max_inputs=2)
        for plan in plans:
            flat = [kept[d][c] for group in plan for d, c in group]
            indexes = tuple(item[0] for item in flat)
            self.observer.emit("callback indexes", indexes)
            before_dispatch(indexes)
            self.observer.emit("provider dispatch", indexes)
            batch = provider_module.EmbeddedDocumentBatch(indexes, tuple(tuple(self.embed_query(item[1])) for item in flat))
            self.observer.emit("provider accepted", batch)
            yield batch


def seed(root: Path, case: Scenario, patch: pytest.MonkeyPatch) -> Any:
    root.mkdir(parents=True, exist_ok=True)
    config: Any = storage_config(root, embeddings=True)
    config.embedding_max_batch_items = 3 if case.shape.startswith("page-B") else 2
    config.embeddings_enabled = case.shape != "disabled"
    cloud = case.shape.startswith("cloud-") or case.shape in {"dry-token-cap", "contextual-model-cap-and-estimates"}
    if cloud:
        config.embedding_provider = "voyage"
        config.embedding_model = "voyage-context-3" if case.shape == "contextual-model-cap-and-estimates" else "voyage-4-lite"
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=Path(config.database_path)))
    count = 0 if case.shape in {"dry-empty", "apply-empty"} else 3
    if case.shape == "apply-one":
        count = 1
    if case.shape.startswith("page-B"):
        count = {"page-B-minus-1": 2, "page-B": 3, "page-B-plus-1": 4}[case.shape]
    if case.shape == "incremental-revision-gap-checkpoint":
        count = 4
    seed_summaries: Any = _seed
    seed_chunks: Any = _seed_messages
    seed_summaries(engine, count, register=False)
    text = 'synthetic row "quoted" café ' * 3
    if case.shape == "cloud-dry-consent-on":
        text = text.replace("row", "map")  # Equal-sized content keeps the consent previews identical.
    if case.shape == "dry-token-cap":
        text = "x" * 28001
        with sqlite3.connect(config.database_path) as conn:
            conn.execute("UPDATE summary_nodes SET summary=? WHERE depth=0 AND node_id=1", (text,))
            conn.execute("UPDATE summary_nodes SET summary=? WHERE depth=0 AND node_id=2", ("y" * 27000,))
        conn.close()
    rows = [(i, "trace-session", "history", "user", text + str(i), float(i)) for i in range(1, count + 1)]
    if case.shape == "dry-token-cap":
        rows[1] = (2, "trace-session", "history", "user", "y" * 27000, 2.)
    if case.shape == "contextual-model-cap-and-estimates":
        rows = [(1, "trace-session", "history", "user", "a" * 30000, 1.),
                (2, "trace-session", "history", "user", "b" * 33001, 2.),
                (3, "trace-session", "history", "user", "short sentence. " * 90, 3.)]
    if case.shape == "incremental-boundary-resume":
        rows = [(1, "trace-session", "history", "user", "synthetic sentence. " * 100, 1.)]
    seed_chunks(engine, rows, register=False)
    store = vectors.VectorStore(config.database_path, config=config)
    try:
        store.ensure_chunk_schema()
        if not case.shape.startswith("profile-absent"):
            store.register_profile(config.embedding_model, config.embedding_provider, 2, revision="fixture-revision", task="chunk" if case.runner == "chunks" else "summary")
        if case.shape == "incremental-revision-gap-checkpoint":
            identity = vectors.EmbeddingIdentity.canonical(config.embedding_provider, config.embedding_model,
                "fixture-revision", 2, "float32", "little", "chunk")
            for row in (rows[0], rows[-1]):
                for chunk in command.chunk_message(row[0], row[3], row[4], policy="conversational"):
                    store.record_chunk_embedding(chunk.chunk_id, config.embedding_model, [1., 1.],
                        store_id=chunk.store_id, chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                        char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=identity)
            store.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?, '1')",
                                     ("chunk_policy_revision:" + identity.identity_hash,))
        if case.shape == "lease-held":
            command._acquire_embedding_backfill_lease(store.connection, ttl_s=300., heartbeat_s=1.)
    finally:
        store.close()
    # SQLite's SQL clock is separate from Python's frozen clock. These are
    # generated fixture-setup timestamps, never source content or runner writes.
    with sqlite3.connect(config.database_path) as conn:
        conn.execute("UPDATE lcm_migration_state SET completed_at=1700000000.0")
    conn.close()
    return config


def observe(case: Scenario, root: Path, *, payloads: list[str] | None = None) -> list[str]:
    o, clock, stop = Observer(root, payloads), Clock(), threading.Event()
    o.case = case
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> FrozenDateTime:
            return cls.fromtimestamp(clock.time(), tz)
    with pytest.MonkeyPatch.context() as patch:
        for key in product_env_keys(ROOT):
            patch.delenv(key, raising=False)
        patch.setattr(command, "time", clock)
        patch.setattr(vectors, "time", clock)
        patch.setattr(vectors, "datetime", FrozenDateTime)
        patch.setattr(importlib.import_module("hermes_lcm.db_bootstrap"), "time", clock)
        patch.setattr(command.uuid, "uuid4", clock.nonce)
        patch.setattr(vectors.VectorStore, "_now", staticmethod(lambda: "2023-11-14T22:13:20+00:00"))
        patch.setattr(command, "count_tokens", lambda text: len(str(text)))
        patch.setattr(importlib.import_module("hermes_lcm.chunking"), "count_tokens", lambda text: len(str(text)))
        # Never execute or inspect privacy implementations. Only call order is observed.
        def revision(*args: Any) -> str:
            o.emit("boundary revision", None)
            return "fixture-revision"
        patch.setattr(command, "embedding_privacy_revision", revision)
        def prepare(documents: Any, **kwargs: Any) -> Any:
            o.emit("boundary preparation", None)
            o.emit("document aliases", [o.alias(documents), [o.alias(row) for row in documents]])
            o.emit("discovery checkpoint", None)
            o.snapshot(Path(config.database_path), "discovery state")
            privacy_revision = "fixture-revision" if command.embedding_provider_requires_privacy(kwargs["provider_name"]) else None
            return documents, privacy_revision, 0, 0, None
        patch.setattr(command, "_prepare_embedding_provider_documents", prepare)
        patch.setattr(command, "validate_embedding_privacy_dispatch", lambda *args, **kwargs: o.emit("boundary dispatch", None))
        config = seed(root, case, patch)
        provider = ScriptedProvider(case, o, clock, stop, config)
        if case.shape.startswith("incremental-"):
            provider.case = Scenario(case.runner, "stop-after-accepted-subbatch")
        if case.shape == "provider-mismatch":
            provider.model_id = "other-model"
        def resolve(*args: Any, **kwargs: Any) -> Any:
            o.emit("provider resolve", {"for_backfill": kwargs.get("for_backfill")})
            return None if case.shape == "provider-missing" else provider
        patch.setattr(command, "resolve_provider", resolve)
        # Socket operations fail immediately, including accidental fallback transports.
        def offline(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("network forbidden in backfill trace")
        import socket
        patch.setattr(socket, "socket", offline)
        patch.setattr(socket, "getaddrinfo", offline)
        real_connect = sqlite3.connect
        def connect(*args: Any, **kwargs: Any) -> Connection:
            if case.shape == "read-open-non-sqlite-error" and not o.fault_used:
                o.fault_used = True
                o.emit("connection open raises", "scripted open failure")
                raise RuntimeError("scripted open failure")
            kwargs.update(factory=Connection, cached_statements=0)
            conn: Connection = real_connect(*args, **kwargs)
            conn.observer = o
            conn.label = "connection-" + str(len(o.connections) + 1)
            o.connections.append(conn)
            o.emit("connection open", [conn.label, args, kwargs.get("isolation_level")])
            conn.set_authorizer(o.authorizer)
            conn.set_trace_callback(lambda sql: o.emit("sql " + conn.label, sql))
            return conn
        patch.setattr(sqlite3, "connect", connect)
        real_store = vectors.VectorStore
        publish_name = "publish_chunk_embedding_batch_under_lease" if case.runner == "chunks" else "publish_embedding_batch_under_lease"
        row_name = "publish_chunk_embedding_under_lease" if case.runner == "chunks" else "publish_embedding_under_lease"
        published = 0
        stores: list[Any] = []
        def race(kind: str) -> None:
            o.emit("race", kind)
            if kind == "identity-newer":
                other_store = real_store(config.database_path, config=config)
                try:
                    other_store.register_profile(config.embedding_model, config.embedding_provider, 2,
                                                 revision="newer-fixture-revision", task="summary")
                finally:
                    other_store.close()
                return
            with sqlite3.connect(config.database_path) as other:
                if kind == "owner":
                    other.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps({"owner": "other-owner", "generation": 9, "heartbeat_at": clock.time()}), command._EMBEDDING_BACKFILL_CLAIM_KEY))
                else:
                    other.execute("UPDATE lcm_embedding_profile SET active=0 WHERE active=1")
            other.close()
        class Store(real_store):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                stores.append(self)
                super().__init__(*args, **kwargs)

            def __getattribute__(self, name: str) -> Any:
                original = super().__getattribute__(name)
                if name.startswith("__") or not isinstance(original, (FunctionType, MethodType)) or o.quiet or not o.active:
                    return original
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    nonlocal published
                    o.primitive(name)
                    o.emit("store enter " + name, [args, kwargs])
                    if name == publish_name:
                        if not o.fault_used and case.shape in {"publication-owner-loss", "publication-identity-superseded"}:
                            o.fault_used = True
                            race("owner" if case.shape == "publication-owner-loss" else "identity")
                        o.commit_fault = case.shape == "publish-call-error" and not o.fault_used
                        o.fault_used |= o.commit_fault
                    try:
                        if name == row_name and case.shape == "publish-row-error" and published == 1 and not o.fault_used:
                            o.fault_used = True
                            conn = self.connection
                            conn.set_authorizer(lambda action, table, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table in {"lcm_embedding_vectors", "lcm_chunk_vectors"} else o.authorizer(action, table, None, None, None))
                        result = original(*args, **kwargs)
                    except Exception as exc:
                        o.emit("store raises " + name, exc)
                        raise
                    finally:
                        if name == row_name and case.shape == "publish-row-error":
                            self.connection.set_authorizer(o.authorizer)
                        if name == publish_name:
                            o.commit_fault = False
                    o.emit("store exit " + name, result)
                    if name in {"_init_db", publish_name}:
                        o.files(root)
                    if name == "_init_db" and case.shape == "identity-switch-before-dispatch" and not o.fault_used:
                        o.fault_used = True
                        race("identity-newer" if case.runner == "summary" else "identity")
                    if name == row_name:
                        published += 1
                    if name == publish_name:
                        o.snapshot(Path(config.database_path), "accepted checkpoint")
                        if case.shape == "lease-renew-lost-after-page" and not o.fault_used:
                            o.fault_used = True
                            race("owner")
                            clock.elapsed += 10.
                        if case.shape == "budget-after-page":
                            clock.elapsed += 1000.
                    return result
                return wrapped
        patch.setattr(command, "VectorStore", Store)
        patch.setenv("LCM_EMBEDDING_BACKFILL_HEARTBEAT_S", "1")
        patch.setenv("LCM_EMBEDDING_BACKFILL_BUDGET_S", "1" if case.shape == "budget-after-page" else "0")
        if case.shape == "stop-before-dispatch":
            stop.set()
        handler, logger = Logs(o), logging.getLogger("hermes_lcm")
        old_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            o.snapshot(Path(config.database_path), "before")
            o.files(root)
            passes = 3 if case.shape == "incremental-revision-gap-checkpoint" else 2 if case.shape in {
                "limit-then-reopen-resume", "incremental-boundary-resume", "provider-ambiguous-after-subbatch",
                "contextual-model-cap-and-estimates"} else 1
            for attempt in range(passes):
                apply = not (case.shape.startswith("dry-") or case.shape in {"dtype-match", "discovery-read-non-sqlite-error", "profile-absent-dry", "cloud-dry-consent-off", "cloud-dry-consent-on"})
                if case.shape == "contextual-model-cap-and-estimates":
                    apply = attempt > 0
                if attempt:
                    o.emit("reopen", attempt)
                    stop.clear()
                    provider.case = Scenario(case.runner, "apply-one")
                    o.snapshot(Path(config.database_path), "reopened state")
                options: dict[str, Any] = dict(apply=apply, limit=1 if case.shape in {"dry-limited", "limit-then-reopen-resume", "incremental-boundary-resume", "incremental-revision-gap-checkpoint"} and attempt == 0 else 100,
                    retry_uncertain=case.shape == "provider-ambiguous-after-subbatch" and attempt > 0, stop_event=stop,
                    expected_dtype="int8" if case.shape == "dtype-mismatch" else "float32" if case.shape == "dtype-match" else None)
                db_path = Path(config.database_path) if case.shape != "db-open-error" else root / "missing" / "lcm.db"
                if case.runner == "chunks":
                    options.update(policy="conversational", confirm_raw_text=case.shape.endswith("-on") or case.shape in {"dry-token-cap", "contextual-model-cap-and-estimates"}, incremental=case.shape.startswith("incremental-"))
                runner = command._chunk_backfill_run if case.runner == "chunks" else command._embedding_backfill_summary_run
                try:
                    result = runner(config, db_path, **options)
                    o.emit("return type", [type(result).__name__, getattr(result, "reason", None)])
                    # An ordered pair list pins field order independently of the renderer.
                    o.emit("report fields", list(result.items()) if isinstance(result, dict) else result)
                    o.emit("rendered", command._embedding_backfill_report(**result) if isinstance(result, dict) else str(result))
                except Exception as exc:
                    o.emit("runner raises", exc)
                o.snapshot(Path(config.database_path), "after")
                o.files(root)
            o.emit("observer inventory", [sorted(o.called), sorted(o.written)])
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
            for store in stores:
                try:
                    if store._conn is not None:
                        store.close()
                except Exception as exc:
                    o.emit("teardown raises", exc)
            for conn in o.connections:
                try:
                    conn.close()
                except Exception as exc:
                    o.emit("teardown raises", exc)
            o.active = False
        o.complete()
    return o.events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--payloads", type=Path)
    parser.add_argument("--scenario")
    args = parser.parse_args()
    if not args.regenerate and args.out is None:
        parser.error("pass --out or explicitly --regenerate")
    if args.regenerate and args.scenario:
        parser.error("cannot regenerate a partial inventory")
    assert Path(command.__file__).resolve() == args.repo.resolve() / "command.py", "wrong product tree"
    traces, payloads = {}, {}
    with tempfile.TemporaryDirectory(prefix="backfill-trace-") as directory:
        for case in SCENARIOS:
            if args.scenario and case.name != args.scenario:
                continue
            full: list[str] = []
            traces[case.name] = observe(case, Path(directory) / case.name, payloads=full)
            payloads[case.name] = full
    destination = args.out or GOLDEN
    destination.write_text(json.dumps(traces, indent=0, ensure_ascii=False) + "\n")
    if args.payloads:
        args.payloads.write_text(json.dumps(payloads, ensure_ascii=False) + "\n")
    print(f"Wrote {len(traces)} scenarios to {destination.name}")


if __name__ == "__main__":
    main()
