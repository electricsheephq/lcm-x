"""#1192: ``lcm-x mcp`` -- a read-only stdio MCP server over one live Hermes store.

REGRESSION tests encode new behaviour (red on the v0.27.0 base); CONTROL tests pin
today's Hermes behaviour at the seams this change adds (green on base too).
Every fixture is a v5 store built in ``tmp_path`` through the real write API.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import hermes_lcm.embedding_provider as embedding_provider
import hermes_lcm.retrieval_core as retrieval_core
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.db_bootstrap import join_background_stem_backfills
from hermes_lcm.externalize import DEFAULT_LARGE_OUTPUT_DIRNAME
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

PLUGIN_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PLUGIN_DIR / "scripts" / "lcm_x_mcp.py"
PROVIDER, MODEL, DIM = "ollama", "mock-model", 16
NON_LCM_SECRET = "dummy-provider-key-1192"
FIVE = ["lcm_recall", "lcm_grep", "lcm_expand", "lcm_load_session", "lcm_describe"]


def _reader_modules():
    import hermes_lcm.mcp_reader as mcp_reader
    import hermes_lcm.readonly_store as readonly_store

    return mcp_reader, readonly_store


# -- deterministic local embedding provider (no network) -------------------------

def _hash_vector(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.sha1(word.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec] if any(vec) else [1.0] + [0.0] * (dim - 1)


class HashProvider:
    """Bag-of-words hashing embedder, shaped like the suite's MockProvider."""

    provider_id = PROVIDER
    model_id = MODEL

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return _hash_vector(text, self.dim)

    def embed_documents(self, texts):
        return [_hash_vector(t, self.dim) for t in texts]


@pytest.fixture(autouse=True)
def _hash_provider(monkeypatch):
    provider = HashProvider()
    monkeypatch.setattr(embedding_provider, "resolve_provider", lambda _config, **_kw: provider)
    monkeypatch.setattr(lcm_tools, "resolve_provider", lambda _config, **_kw: provider)
    retrieval_core._reset_vector_store_pool()
    yield provider
    retrieval_core._reset_vector_store_pool()
    while _RUNNING_AGENTS:
        _RUNNING_AGENTS.pop().close()


# Idle connections standing in for a running Hermes: they keep the store's -wal/-shm in
# place (this SQLite deletes both on the last close) and never write.
_RUNNING_AGENTS: list[sqlite3.Connection] = []


def start_agent(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    _RUNNING_AGENTS.append(conn)
    return conn


def stop_agents() -> None:
    while _RUNNING_AGENTS:
        _RUNNING_AGENTS.pop().close()


# -- fixture store ---------------------------------------------------------------

@dataclass
class Fixture:
    home: Path
    db: Path
    config: LCMConfig
    s1_ids: list[int]
    s2_tool_id: int
    node_ids: list[int]
    ref: str


PROFILE_ENV = (
    "# profile env: only LCM_* keys may reach the reader\n"
    "LCM_EMBEDDINGS_ENABLED=true\n"
    f"LCM_EMBEDDING_PROVIDER={PROVIDER}\n"
    f"export LCM_EMBEDDING_MODEL='{MODEL}'\n"
    "LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS=200  # small, for the fixture\n"
    "LCM_RECALL_QUERY_TIMEOUT_S=30\n"
    "LCM_EMBEDDING_QUERY_TIMEOUT_S=30\n"
    f"VOYAGE_API_KEY={NON_LCM_SECRET}\n"
)


def _profile_config(db: Path) -> LCMConfig:
    return LCMConfig(
        database_path=str(db),
        embeddings_enabled=True,
        embedding_provider=PROVIDER,
        embedding_model=MODEL,
        large_output_externalization_threshold_chars=200,
        recall_query_timeout_s=30.0,
        embedding_query_timeout_s=30.0,
    )


def _node(session_id, summary, source_ids, *, depth=0, source_type="messages", at=1_700_000_000.0):
    return SummaryNode(
        session_id=session_id, depth=depth, summary=summary, token_count=20,
        source_token_count=60, source_ids=list(source_ids), source_type=source_type,
        created_at=at, earliest_at=at, latest_at=at, expand_hint=f"Expand: {summary[:24]}",
    )


def _seed_vectors(db: Path, config: LCMConfig, nodes: dict[int, str], messages: dict[int, str], *,
                  dim: int = DIM, revision: str = "") -> None:
    vs = VectorStore(db, config=config)
    try:
        vs.register_profile(MODEL, PROVIDER, dim, revision=revision)
        vs.register_profile(MODEL, PROVIDER, dim, revision=revision, task="chunk")
        summary = EmbeddingIdentity.canonical(PROVIDER, MODEL, revision, dim, "float32", "little", "summary")
        chunk = EmbeddingIdentity.canonical(PROVIDER, MODEL, revision, dim, "float32", "little", "chunk")
        for node_id, text in nodes.items():
            vs.record_embedding(str(node_id), "summary", MODEL, _hash_vector(text, dim), identity=summary)
        for store_id, text in messages.items():
            vs.record_chunk_embedding(
                f"{store_id}:0", MODEL, _hash_vector(text, dim), store_id=store_id, chunk_index=0,
                char_start=0, char_end=len(text), token_estimate=max(1, len(text) // 4), identity=chunk,
            )
    finally:
        vs.close()


def build_store(tmp_path: Path, *, vectors: bool = True, dim: int = DIM, revision: str = "",
                env: str = PROFILE_ENV) -> Fixture:
    home = tmp_path / "profile"
    home.mkdir(parents=True)
    (home / ".env").write_text(env)
    db = home / "lcm.db"
    config = _profile_config(db)
    store = MessageStore(db, ingest_protection_config=config, hermes_home=str(home))
    dag = SummaryDAG(db)
    texts = [
        ("user", "We chose the blue heron logo for the Aurora launch."),
        ("assistant", "Noted: the Aurora launch uses the blue heron logo."),
        ("user", "The Aurora budget is 42k and the venue is the old boathouse."),
        ("assistant", "Budget 42k and the boathouse venue are recorded."),
    ]
    try:
        s1_ids = [
            store.append("s1", {"role": role, "content": text, "timestamp": 1_700_000_000.0 + i})
            for i, (role, text) in enumerate(texts)
        ]
        store.append("s2", {"role": "user", "content": "Run the inventory report for the warehouse."})
        big = "inventory row " + " ".join(f"item-{n} quantity {n * 3}" for n in range(60))
        s2_tool_id = store.append(
            "s2", {"role": "tool", "content": big, "tool_call_id": "call-inv-1", "tool_name": "inventory"}
        )
        store.append("s2", {"role": "assistant", "content": "The warehouse inventory report is ready."})
        n1 = dag.add_node(_node("s1", "Aurora launch: blue heron logo chosen.", s1_ids[:2]),
                          escalation_level=1, model="mock-summarizer")
        n2 = dag.add_node(_node("s1", "Aurora budget 42k; venue is the old boathouse.", s1_ids[2:]),
                          escalation_level=2, model="mock-summarizer")
        n3 = dag.add_node(_node("s1", "Aurora launch plan: heron logo, 42k budget, boathouse.", [n1, n2],
                                depth=1, source_type="nodes"))
    finally:
        store.close()
        dag.close()
    join_background_stem_backfills(timeout=10)
    if vectors:
        _seed_vectors(
            db, config,
            {n1: "Aurora launch: blue heron logo chosen.",
             n2: "Aurora budget 42k; venue is the old boathouse.",
             n3: "Aurora launch plan: heron logo, 42k budget, boathouse."},
            {sid: text for sid, (_role, text) in zip(s1_ids, texts)},
            dim=dim, revision=revision,
        )
    payload_dir = home / DEFAULT_LARGE_OUTPUT_DIRNAME
    refs = sorted(p.name for p in payload_dir.iterdir()) if payload_dir.is_dir() else []
    assert len(refs) == 1, refs
    start_agent(db)
    return Fixture(home, db, config, s1_ids, s2_tool_id, [n1, n2, n3], refs[0])


@pytest.fixture
def fx(tmp_path):
    return build_store(tmp_path)


def make_reader(fixture: Fixture, monkeypatch, *, prepare: bool = True):
    """The reader as the launcher builds it: the profile's LCM_* view, then LCMConfig.from_env()."""
    mcp_reader, _ = _reader_modules()
    for key in [k for k in os.environ if k.startswith("LCM_")]:
        monkeypatch.delenv(key)
    for key, value in mcp_reader.read_profile_env(fixture.home).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HERMES_HOME", str(fixture.home))
    reader = mcp_reader.StoreReader(fixture.db, fixture.home, config=LCMConfig.from_env())
    if prepare:
        reader.prepare(tokenizer_wait_s=0.5)
    return reader


def call(reader, name, args):
    text, is_error = reader.call(name, args)
    return json.loads(text), is_error, text


def strip_envelope(text: str) -> str:
    """Remove the reader's ``store_uuid``/``reader`` prefix; the production bytes remain."""
    decoded = json.loads(text)
    head = json.dumps({"store_uuid": decoded["store_uuid"], "reader": decoded["reader"]})[:-1]
    assert text.startswith(head), "envelope must be a pure prefix"
    rest = text[len(head):]
    return "{}" if rest == "}" else "{" + rest[len(", "):]


def sweep(fixture: Fixture) -> list[tuple[str, dict]]:
    """A full five-tool sweep (FTS, semantic, hybrid, vectors, nodes, payloads, paging)."""
    n1, _n2, n3 = fixture.node_ids
    return [
        ("lcm_grep", {"query": "heron logo"}),
        ("lcm_grep", {"query": "Aurora budget", "mode": "hybrid"}),
        ("lcm_grep", {"query": "heron", "mode": "semantic", "session_scope": "session", "session_id": "s1"}),
        ("lcm_grep", {"query": "inventory", "session_scope": "session", "session_id": "s2"}),
        ("lcm_recall", {"query": "which logo did we choose for Aurora"}),
        ("lcm_recall", {"query": "Aurora venue budget", "detail": "answer_ready", "include_occurrence_time": True}),
        ("lcm_recall", {"query": "warehouse inventory", "include": "verbatim"}),
        ("lcm_expand", {"store_id": fixture.s1_ids[0], "include_exact_ref": True}),
        ("lcm_expand", {"store_id": fixture.s2_tool_id}),
        ("lcm_expand", {"node_id": n1, "session_id": "s1"}),
        ("lcm_expand", {"node_id": n3}),
        ("lcm_expand", {"externalized_ref": fixture.ref}),
        ("lcm_describe", {"node_id": n3, "session_id": "s1"}),
        ("lcm_describe", {"session_id": "s1"}),
        ("lcm_describe", {"externalized_ref": fixture.ref}),
        ("lcm_load_session", {"session_id": "s1", "limit": 2}),
        ("lcm_load_session", {"session_id": "s1", "after_store_id": fixture.s1_ids[1], "include_exact_ref": True}),
    ]


# -- T1: zero main-database writes on every connection the reader opens ----------

_TEMP_OK = "temp"
_WRITE_ACTIONS = {
    getattr(sqlite3, name) for name in dir(sqlite3)
    if name.startswith(("SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_",
                        "SQLITE_DROP_", "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE"))
}
_ALWAYS_DENIED = {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}
_WRITE_PRAGMAS = {"wal_checkpoint", "incremental_vacuum", "optimize", "journal_mode", "user_version",
                  "schema_version", "application_id", "auto_vacuum", "page_size", "writable_schema",
                  "wal_autocheckpoint", "synchronous", "journal_size_limit", "mmap_size"}
_READ_PRAGMAS_WITH_ARG = {"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo",
                          "foreign_key_list", "busy_timeout", "query_only"}


class _Denials(list):
    opened: list


def install_write_auditor(monkeypatch) -> _Denials:
    """Patch sqlite3.connect so EVERY new connection (product private ones too) is audited.

    Rule: writes whose database is 'temp' are allowed (TEMP staging); any other write,
    DDL, ATTACH/DETACH or write PRAGMA (wal_checkpoint included) is a denial.
    """
    denials = _Denials()
    real_connect = sqlite3.connect

    def authorizer(action, arg1, arg2, db_name, _source):
        if action in _ALWAYS_DENIED:
            denials.append((action, arg1, arg2, db_name))
            return sqlite3.SQLITE_DENY
        if action in _WRITE_ACTIONS and db_name != _TEMP_OK:
            denials.append((action, arg1, arg2, db_name))
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_PRAGMA:
            name = str(arg1 or "").lower()
            if name in _WRITE_PRAGMAS and (arg2 is not None or name in {"wal_checkpoint", "optimize",
                                                                       "incremental_vacuum"}):
                denials.append((action, arg1, arg2, db_name))
                return sqlite3.SQLITE_DENY
            if arg2 is not None and name not in _READ_PRAGMAS_WITH_ARG and name not in _WRITE_PRAGMAS:
                denials.append((action, arg1, arg2, db_name))
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    opened = []

    def audited_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_authorizer(authorizer)
        opened.append(args[0] if args else kwargs.get("database"))
        return conn

    monkeypatch.setattr(sqlite3, "connect", audited_connect)
    denials.opened = opened
    return denials


def test_t1_full_sweep_makes_zero_main_database_writes(fx, monkeypatch):
    """REGRESSION: an authorizer on every connection the reader process opens sees no write."""
    denials = install_write_auditor(monkeypatch)
    reader = make_reader(fx, monkeypatch)
    results = [call(reader, name, args) for name, args in sweep(fx)]
    assert [name for (name, _a), (_p, err, _t) in zip(sweep(fx), results) if err] == []
    assert len(denials.opened) >= 3 * len(results)  # per-call ro connections were audited
    assert any(isinstance(entry["store"], _reader_modules()[1].ReadOnlyVectorStore)
               for entry in retrieval_core._vector_store_pool.values())
    # Teardown is part of the sweep: pooled-store close and every finalizer run under audit.
    retrieval_core._reset_vector_store_pool()
    del reader
    gc.collect()
    assert denials == []


def test_t1_auditor_detects_a_stock_constructor_write(fx, monkeypatch):
    """CONTROL: the auditor is live -- a stock writable open IS counted (positive control)."""
    denials = install_write_auditor(monkeypatch)
    with pytest.raises(sqlite3.DatabaseError):
        MessageStore(fx.db, ingest_protection_config=fx.config, hermes_home=str(fx.home))
    assert denials


# -- T2: parity with a normal engine on the same quiesced store ------------------

# Fields excluded from the byte comparison, and why: none of the five tools emit timing
# or elapsed fields today; the reader adds only the ``store_uuid``/``reader`` envelope.
PARITY_EXCLUDED_FIELDS: tuple[str, ...] = ()


def _normal_engine(fixture: Fixture):
    from hermes_lcm.engine import LCMEngine

    return LCMEngine(config=_profile_config(fixture.db), hermes_home=str(fixture.home))


def _drop_fields(text: str) -> str:
    if not PARITY_EXCLUDED_FIELDS:
        return text
    payload = json.loads(text)
    for name in PARITY_EXCLUDED_FIELDS:
        payload.pop(name, None)
    return json.dumps(payload, sort_keys=True)


def test_t2_outputs_equal_a_normal_engine_byte_for_byte(fx, monkeypatch):
    """REGRESSION: reader output minus the envelope == the same call through a normal engine."""
    reader = make_reader(fx, monkeypatch)
    reader_texts = []
    for name, args in sweep(fx):
        text, is_error = reader.call(name, dict(args))
        assert not is_error, (name, text)
        reader_texts.append(strip_envelope(text))
    # Not vacuous: vectors served, hits returned, payload content and node sources expanded.
    recall = json.loads(reader_texts[4])
    assert recall["hits"] and recall["provenance"]["coverage"]["summary"] in {"full", "bounded"}
    assert "item-59 quantity 177" in json.loads(reader_texts[11])["content"]
    assert json.loads(reader_texts[9])["expanded"]
    retrieval_core._reset_vector_store_pool()
    engine = _normal_engine(fx)
    compared = 0
    try:
        for (name, args), reader_text in zip(sweep(fx), reader_texts):
            normal_args = dict(args)
            if name == "lcm_grep":
                normal_args.setdefault("session_scope", "all")  # the reader's server-side default
            if name in {"lcm_expand", "lcm_describe"} and "node_id" in args and "session_id" not in args:
                normal_args["session_id"] = "s1"  # the reader resolves the node's own session
            # The reader reaches payloads from any session; a normal engine needs the payload's
            # session as its current one (only the payload-session-dependent calls differ).
            current = "s2" if "externalized_ref" in args else ""
            engine._foreground_session_id = current
            # The store-id expansion of the s2 externalized row: the normal engine under the
            # reader's any-session policy, ordinary storage, no current session.
            any_session = name == "lcm_expand" and args.get("store_id") == fx.s2_tool_id
            if any_session:
                engine._reader_any_session = True
            try:
                normal = getattr(lcm_tools, name)(normal_args, engine=engine)
            finally:
                if any_session:
                    del engine._reader_any_session
            assert _drop_fields(reader_text) == _drop_fields(normal), name
            compared += 1
    finally:
        engine.shutdown()
    assert compared == len(sweep(fx))


# -- T3: store bytes unchanged; no embedding tables materialized ------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_t3_sweep_leaves_store_bytes_unchanged(fx, monkeypatch):
    """REGRESSION: with no writes (an idle agent holds the store), lcm.db hashes the same after a full sweep."""
    before = _sha256(fx.db)
    reader = make_reader(fx, monkeypatch)
    for name, args in sweep(fx):
        assert not reader.call(name, args)[1], name
    retrieval_core._reset_vector_store_pool()
    assert _sha256(fx.db) == before


def test_t3_store_without_embedding_tables_gets_none(tmp_path, monkeypatch):
    """REGRESSION: embeddings enabled in the profile, absent in the store: nothing is created."""
    fixture = build_store(tmp_path, vectors=False)
    before = _sha256(fixture.db)
    reader = make_reader(fixture, monkeypatch)
    for name, args in sweep(fixture):
        reader.call(name, args)
    with sqlite3.connect(f"{fixture.db.resolve().as_uri()}?mode=ro", uri=True) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'lcm_embedding%' "
                                             "OR name LIKE 'lcm_chunk%'")}
    assert tables == set()
    assert _sha256(fixture.db) == before
    payload, _err, _ = call(reader, "lcm_recall", {"query": "heron logo"})
    assert payload["reader"]["capabilities"]["embeddings"] is False


# -- T4: a writer's commit is visible on the next call (FTS + vectors) ------------

def test_t4_live_writer_commit_visible_on_next_call(fx, monkeypatch):
    """REGRESSION: a new message and a new vector committed between two calls are both seen."""
    reader = make_reader(fx, monkeypatch)
    first, _, _ = call(reader, "lcm_grep", {"query": "zebra crossing"})
    assert first["total_results"] == 0
    first_recall, _, _ = call(reader, "lcm_recall", {"query": "zebra crossing mural", "include": "summaries"})
    assert all("zebra" not in json.dumps(hit).lower() for hit in first_recall["hits"])

    store = MessageStore(fx.db, ingest_protection_config=fx.config, hermes_home=str(fx.home))
    dag = SummaryDAG(fx.db)
    vs = VectorStore(fx.db, config=fx.config)
    try:
        sid = store.append("s3", {"role": "user", "content": "Paint the zebra crossing mural by Friday."})
        node = dag.add_node(_node("s3", "Zebra crossing mural due Friday.", [sid]))
        identity = vs.capture_identity(MODEL, provider=PROVIDER)
        vs.record_embedding(str(node), "summary", MODEL, _hash_vector("zebra crossing mural", DIM),
                            identity=identity)
    finally:
        store.close()
        dag.close()
        vs.close()

    second, _, _ = call(reader, "lcm_grep", {"query": "zebra crossing"})
    assert [r.get("store_id") for r in second["results"]] == [sid]
    second_recall, _, _ = call(reader, "lcm_recall", {"query": "zebra crossing mural", "include": "summaries"})
    assert second_recall["provenance"]["coverage"]["summary"] in {"full", "bounded"}
    assert node in [hit.get("node_id") for hit in second_recall["hits"]]


# -- T5: schema refusal and optional capabilities ---------------------------------

def _set_stamp(db: Path, value: str) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE metadata SET value = ? WHERE key = 'schema_version'", (value,))


def test_t5_newer_stamp_is_refused_clearly(fx, monkeypatch):
    """REGRESSION: a store stamped above this build is refused, not opened."""
    _set_stamp(fx.db, "6")
    before = _sha256(fx.db)
    reader = make_reader(fx, monkeypatch)
    payload, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
    assert is_error and payload["error"].startswith("schema_refused:")
    assert "6" in payload["error"]
    assert _sha256(fx.db) == before


def test_t5_stamp_below_minimum_is_refused_clearly(fx, monkeypatch):
    """REGRESSION: a store stamped below v5 is refused with the minimum named."""
    _set_stamp(fx.db, "4")
    reader = make_reader(fx, monkeypatch)
    payload, is_error, _ = call(reader, "lcm_describe", {"session_id": "s1"})
    assert is_error
    assert "schema_refused" in payload["error"] and "minimum v5" in payload["error"]


def test_t5_missing_optional_capabilities_read_with_flags(tmp_path, monkeypatch):
    """REGRESSION: no embedding tables, no provenance sidecar, no stem index: reads work, flags say so."""
    fixture = build_store(tmp_path, vectors=False)
    with sqlite3.connect(fixture.db) as conn:
        conn.execute("DROP TABLE summary_node_provenance")
        for (trigger,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND sql LIKE '%messages_fts_stem%'"
        ).fetchall():
            conn.execute(f"DROP TRIGGER {trigger}")
        conn.execute("DROP TABLE IF EXISTS messages_fts_stem")
    reader = make_reader(fixture, monkeypatch)
    payload, is_error, _ = call(reader, "lcm_describe", {"node_id": fixture.node_ids[0], "session_id": "s1"})
    assert not is_error, payload
    assert "escalation_level" not in payload and payload["node_id"] == fixture.node_ids[0]
    caps = payload["reader"]["capabilities"]
    assert caps["summary_node_provenance"] is False
    assert caps["embeddings"] is False and caps["chunks"] is False
    assert caps["stem"] is False
    assert caps["fts_messages"] is True and caps["schema_version"] == 5
    grep, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
    assert not is_error and grep["total_results"] >= 1


def test_t5_full_store_reports_all_capabilities(fx, monkeypatch):
    """REGRESSION: a complete v5 store reports every capability, provenance included."""
    reader = make_reader(fx, monkeypatch)
    payload, _, _ = call(reader, "lcm_describe", {"node_id": fx.node_ids[0], "session_id": "s1"})
    assert payload["escalation_level"] == 1 and payload["model"] == "mock-summarizer"
    caps = payload["reader"]["capabilities"]
    assert caps == {
        "schema_version": 5, "fts_messages": True, "fts_nodes": True, "stem": True,
        "embeddings": True, "chunks": True, "summary_node_provenance": True,
        "lifecycle_state": True, "host_uid_bindings": caps["host_uid_bindings"],
    }


# -- T6: the writer's checkpoints keep progressing --------------------------------

def test_t6_writer_checkpoints_never_busy_across_50_calls(fx, monkeypatch):
    """REGRESSION: after each of 50 reader calls a writer commit + PASSIVE checkpoint reports busy == 0
    and checkpoints every WAL frame (a reader snapshot left open would hold frames back)."""
    reader = make_reader(fx, monkeypatch)
    store = MessageStore(fx.db, ingest_protection_config=fx.config, hermes_home=str(fx.home))
    busy_counts, lagging = [], []
    try:
        plan = sweep(fx)
        for index in range(50):
            name, args = plan[index % len(plan)]
            reader.call(name, args)
            store.append("s9", {"role": "user", "content": f"checkpoint probe {index}"})
            busy, log_frames, checkpointed = store._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            busy_counts.append(busy)
            if checkpointed != log_frames:
                lagging.append((index, name, log_frames, checkpointed))
    finally:
        store.close()
    assert any(isinstance(e["store"], _reader_modules()[1].ReadOnlyVectorStore)
               for e in retrieval_core._vector_store_pool.values())
    assert busy_counts == [0] * 50
    assert lagging == []


# -- T7: full-identity gate --------------------------------------------------------

def test_t7_matching_identity_uses_vector_arms(fx, monkeypatch, _hash_provider):
    """REGRESSION: provider identity == store's active identity -> summary/chunk KNN served."""
    reader = make_reader(fx, monkeypatch)
    payload, _, _ = call(reader, "lcm_recall", {"query": "which logo did we choose for Aurora"})
    assert payload["reader"]["degraded"] == []
    assert payload["provenance"]["coverage"]["summary"] in {"full", "bounded"}
    assert payload["provenance"]["coverage"]["chunk"] in {"full", "bounded"}
    assert "which logo did we choose for Aurora" in _hash_provider.queries


@pytest.mark.parametrize("variant", ["dim", "revision"])
def test_t7_identity_mismatch_means_no_vectors(tmp_path, monkeypatch, variant):
    """REGRESSION: a dim or revision mismatch -> no vectors, degraded flag, no VectorStore built."""
    if variant == "dim":
        fixture = build_store(tmp_path, dim=DIM * 2)
    else:
        fixture = build_store(tmp_path, revision="r2-other-weights")
    reader = make_reader(fixture, monkeypatch)
    payload, is_error, _ = call(reader, "lcm_recall", {"query": "which logo did we choose for Aurora"})
    assert not is_error
    assert payload["reader"]["degraded"] == ["embedding_identity_mismatch"]
    assert payload["provenance"]["coverage"].get("summary") == "disabled"
    assert retrieval_core._vector_store_pool == {}
    assert payload["hits"], "full text still answers"


def test_t7_unavailable_provider_degrades_to_full_text(fx, monkeypatch):
    """REGRESSION: a provider that cannot warm (no local model) -> provider_unavailable, FTS only."""
    def broken(_config, **_kw):
        raise embedding_provider.ProviderNotWarmedUp("model not cached")

    monkeypatch.setattr(embedding_provider, "resolve_provider", broken)
    reader = make_reader(fx, monkeypatch)
    payload, is_error, _ = call(reader, "lcm_recall", {"query": "heron logo"})
    assert not is_error and payload["reader"]["degraded"] == ["provider_unavailable"]
    assert payload["hits"]


def test_t7_cloud_provider_is_never_called(fx, monkeypatch):
    """REGRESSION: a cloud provider (keys are never loaded) is not dispatched; FTS only."""
    cloud = HashProvider()
    cloud.provider_id = "voyage"
    monkeypatch.setattr(embedding_provider, "resolve_provider", lambda _c, **_k: cloud)
    reader = make_reader(fx, monkeypatch)
    payload, _, _ = call(reader, "lcm_recall", {"query": "heron logo"})
    assert payload["reader"]["degraded"] == ["provider_unavailable"]
    assert cloud.queries == []


# -- T8: externalized payloads from a non-current session, from --hermes-home ------

def _spawn(fixture: Fixture, tmp_path: Path, *, extra_env: dict | None = None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LCM_")}
    env.pop("PYTHONPATH", None)
    env.update({"HF_HUB_OFFLINE": "1", "HERMES_HOME": str(tmp_path / "not-the-profile")})
    env.update(extra_env or {})
    log_dir = tmp_path / "logs"
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--store", str(fixture.db), "--hermes-home", str(fixture.home),
         "--log-dir", str(log_dir)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        cwd=str(tmp_path), text=True,
    )
    return proc, log_dir


def _exchange(proc, lines: list[str], timeout: float = 120.0) -> list[dict]:
    out, err = proc.communicate("\n".join(lines) + "\n", timeout=timeout)
    assert proc.returncode == 0, err
    assert err == ""
    replies = [json.loads(line) for line in out.splitlines()]
    assert all(isinstance(r, dict) and r.get("jsonrpc") == "2.0" for r in replies), out
    return replies


def _req(ident, method, params=None):
    message = {"jsonrpc": "2.0", "id": ident, "method": method}
    if params is not None:
        message["params"] = params
    return json.dumps(message)


SUBPROCESS_ENV = (
    "LCM_EMBEDDINGS_ENABLED=true\n"
    "LCM_EMBEDDING_PROVIDER=fastembed\n"
    "LCM_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5\n"
    "LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS=200\n"
    "LCM_DISABLED_TOOLS=lcm_describe\n"
    f"VOYAGE_API_KEY={NON_LCM_SECRET}\n"
)


@pytest.fixture
def spawned_fx(tmp_path):
    fixture = build_store(tmp_path)
    cache = tmp_path / "empty-fastembed-cache"
    cache.mkdir()
    (fixture.home / ".env").write_text(SUBPROCESS_ENV + f"FASTEMBED_CACHE_PATH={cache}\n")
    return fixture


def test_t8_externalized_payload_from_other_session_resolves_from_hermes_home(spawned_fx, tmp_path):
    """REGRESSION: payloads of a non-current session resolve by ref and via store-id expansion,
    against --hermes-home, with the reader's own LCM_HERMES_BASE_DIR pointing elsewhere."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    proc, _ = _spawn(spawned_fx, tmp_path, extra_env={"LCM_HERMES_BASE_DIR": str(elsewhere)})
    replies = _exchange(proc, [
        _req(1, "initialize", {"protocolVersion": "2025-06-18"}),
        _req(2, "tools/call", {"name": "lcm_expand", "arguments": {"externalized_ref": spawned_fx.ref}}),
        _req(3, "tools/call", {"name": "lcm_expand", "arguments": {"store_id": spawned_fx.s2_tool_id}}),
    ])
    by_ref = json.loads(replies[1]["result"]["content"][0]["text"])
    assert replies[1]["result"]["isError"] is False, by_ref
    assert by_ref["session_id"] == "s2" and "item-59 quantity 177" in by_ref["content"]
    by_store_id = json.loads(replies[2]["result"]["content"][0]["text"])
    assert by_store_id["externalized_ref"] == spawned_fx.ref
    assert by_store_id["externalized_payloads"][0]["session_id"] == "s2"
    assert "externalized_note" not in by_store_id


def test_t8_any_session_reach_in_process(fx, monkeypatch):
    """REGRESSION: lcm_describe by ref reaches the s2 payload with no current session."""
    reader = make_reader(fx, monkeypatch)
    payload, is_error, _ = call(reader, "lcm_describe", {"externalized_ref": fx.ref})
    assert not is_error and payload["session_id"] == "s2"
    assert payload["tool_call_id"] == "call-inv-1"


# -- T9: protocol ------------------------------------------------------------------

def test_t9_protocol_session_over_stdio(spawned_fx, tmp_path):
    """REGRESSION: initialize / tools/list / tools/call / ping; bad lines answered, serving continues."""
    proc, log_dir = _spawn(spawned_fx, tmp_path)
    replies = _exchange(proc, [
        _req(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "test", "version": "0"}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        _req(2, "initialize", {"protocolVersion": "1999-01-01"}),
        _req(3, "tools/list"),
        "[1, 2, 3]",
        "42",
        "this is not json",
        _req(4, "ping"),
        _req(5, "tools/call", {"name": "lcm_grep", "arguments": {"query": "heron logo"}}),
        _req(6, "tools/call", {"name": "lcm_describe", "arguments": {"session_id": "s1"}}),
        _req(7, "no/such/method"),
        "",
        _req(8, "tools/call", {"name": "lcm_load_session", "arguments": {"session_id": "s1", "limit": 1}}),
    ])
    by_id = {r.get("id"): r for r in replies if r.get("id") is not None}
    assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"
    assert by_id[1]["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert by_id[2]["result"]["protocolVersion"] == "2025-06-18"
    tools = by_id[3]["result"]["tools"]
    assert [t["name"] for t in tools] == ["lcm_recall", "lcm_grep", "lcm_expand", "lcm_load_session"]
    assert all(t["inputSchema"]["type"] == "object" for t in tools)
    bad = [r for r in replies if r.get("id") is None]
    assert [r["error"]["code"] for r in bad] == [-32600, -32600, -32700]
    assert by_id[4]["result"] == {}
    grep = json.loads(by_id[5]["result"]["content"][0]["text"])
    assert by_id[5]["result"]["isError"] is False and grep["total_results"] >= 1
    assert re.fullmatch(r"[0-9a-f]{32}", grep["store_uuid"])
    assert grep["reader"]["degraded"] == ["provider_unavailable"]  # no local model cached: FTS only
    assert by_id[6]["result"]["isError"] is True
    assert by_id[7]["error"]["code"] == -32601
    page = json.loads(by_id[8]["result"]["content"][0]["text"])
    assert page["session_id"] == "s1" and len(page["messages"]) == 1
    assert len(replies) == 11  # 8 id-bearing requests + 3 bad lines; none for the notification
    # T13 (subprocess half): no non-LCM key from the profile .env reaches the log.
    log_text = "".join(p.read_text() for p in log_dir.iterdir())
    assert NON_LCM_SECRET not in log_text and "VOYAGE_API_KEY" not in log_text
    assert '"tool": "lcm_grep"' in log_text


def test_t9_replies_arrive_line_by_line_before_stdin_closes(spawned_fx, tmp_path):
    """REGRESSION: an MCP client waits for each reply; the server answers without more input."""
    proc, _ = _spawn(spawned_fx, tmp_path)
    try:
        for ident, line in enumerate([_req(1, "initialize", {"protocolVersion": "2025-06-18"}),
                                      _req(2, "tools/call", {"name": "lcm_grep", "arguments": {"query": "heron"}})],
                                     start=1):
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
            reply = json.loads(proc.stdout.readline())
            assert reply["id"] == ident and "result" in reply
    finally:
        proc.stdin.close()
        assert proc.wait(timeout=60) == 0


def test_t9_launcher_never_imports_the_engine(tmp_path):
    """REGRESSION: the launcher bootstrap reaches mcp_reader without hermes_lcm.engine or __init__."""
    probe = (
        "import runpy, sys; sys.argv=['x', '--help']\n"
        f"try:\n    runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
        "except SystemExit:\n    pass\n"
        "mods = sorted(m for m in sys.modules if m.startswith('hermes_lcm'))\n"
        "assert 'hermes_lcm.mcp_reader' in mods, mods\n"
        "assert 'hermes_lcm.engine' not in mods, mods\n"
        "assert not hasattr(sys.modules['hermes_lcm'], 'register')\n"
        "print('ok')\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            env=env, cwd=str(tmp_path), timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_t9_tools_list_honours_disabled_tools_and_production_schemas(monkeypatch):
    """REGRESSION: five tools with production schemas plus the reader note; disabled ones omitted."""
    mcp_reader, _ = _reader_modules()
    from hermes_lcm import schemas

    definitions = mcp_reader.tool_definitions(set())
    assert [d["name"] for d in definitions] == FIVE
    for definition in definitions:
        production = getattr(schemas, definition["name"].upper())
        assert definition["inputSchema"] == production["parameters"]
        assert definition["description"] == production["description"] + mcp_reader.READER_NOTE
    assert [d["name"] for d in mcp_reader.tool_definitions({"lcm_grep", "lcm_recall"})] == [
        "lcm_expand", "lcm_load_session", "lcm_describe"]


# -- T10: store offline --------------------------------------------------------------


@pytest.mark.parametrize("format_bytes", [(1, 2), (2, 1)], ids=["read-byte-wal", "write-byte-wal"])
def test_t10_mixed_wal_header_without_sidecars_is_offline(fx, monkeypatch, format_bytes):
    """REGRESSION (read-byte-wal; write-byte-wal is a CONTROL): SQLite takes the WAL path when either
    header format byte (18, 19) is 2, so a
    mixed header with no sidecars must also be store_offline and create no file."""
    reader = make_reader(fx, monkeypatch)
    ok, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
    assert not is_error
    stop_agents()
    for suffix in ("wal", "shm"):
        Path(f"{fx.db}-{suffix}").unlink(missing_ok=True)
    with open(fx.db, "r+b") as handle:
        handle.seek(18)
        original = handle.read(2)
        handle.seek(18)
        handle.write(bytes(format_bytes))
    try:
        listing = sorted(os.listdir(fx.home))
        offline, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
        assert is_error and offline["error"].startswith("store_offline:")
        assert sorted(os.listdir(fx.home)) == listing
    finally:
        with open(fx.db, "r+b") as handle:
            handle.seek(18)
            handle.write(original)

@pytest.mark.parametrize("missing", ["both", "wal", "shm"])
def test_t10_stopped_store_is_offline_and_reader_creates_no_file(fx, monkeypatch, missing):
    """REGRESSION: writable directory, WAL store, agent stopped and sidecar(s) gone -> store_offline,
    and the directory listing is unchanged (a mode=ro open would recreate them); the agent's
    restart makes the next call succeed."""
    reader = make_reader(fx, monkeypatch)
    ok, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
    assert not is_error
    sidecars = {suffix: Path(f"{fx.db}-{suffix}") for suffix in ("wal", "shm")}
    saved = {suffix: path.read_bytes() for suffix, path in sidecars.items()}
    stop_agents()  # the last close; this SQLite removes both sidecars, persistent-WAL builds keep them
    for suffix, path in sidecars.items():
        path.unlink(missing_ok=True)
        if missing != "both" and suffix != missing:
            path.write_bytes(saved[suffix])  # leave only the named sidecar missing
    listing = sorted(os.listdir(fx.home))
    offline, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
    assert is_error and offline["error"].startswith("store_offline:")
    assert offline["store_uuid"] == ok["store_uuid"]
    again, is_error, _ = call(reader, "lcm_recall", {"query": "heron"})
    assert is_error and again["error"].startswith("store_offline:")
    assert sorted(os.listdir(fx.home)) == listing
    for path in sidecars.values():
        path.unlink(missing_ok=True)
    writer = MessageStore(fx.db, ingest_protection_config=fx.config, hermes_home=str(fx.home))
    try:
        back, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
        assert not is_error and back["total_results"] == ok["total_results"]
    finally:
        writer.close()


def test_t10_store_offline_then_recovers(fx, monkeypatch):
    """CONTROL: a stopped WAL store without -wal/-shm in an unwritable directory -> store_offline
    (no crash); a restored writer makes the next call succeed."""
    reader = make_reader(fx, monkeypatch)
    ok, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
    assert not is_error
    stop_agents()
    for suffix in ("-wal", "-shm"):
        Path(f"{fx.db}{suffix}").unlink(missing_ok=True)
    os.chmod(fx.home, 0o500)  # owner read+search only: an unwritable directory
    try:
        offline, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
        assert is_error and offline["error"].startswith("store_offline:")
        assert offline["store_uuid"] == ok["store_uuid"]
        again, is_error, _ = call(reader, "lcm_recall", {"query": "heron"})
        assert is_error and again["error"].startswith("store_offline:")
    finally:
        os.chmod(fx.home, 0o700)
    writer = MessageStore(fx.db, ingest_protection_config=fx.config, hermes_home=str(fx.home))
    try:
        back, is_error, _ = call(reader, "lcm_grep", {"query": "heron"})
        assert not is_error and back["total_results"] == ok["total_results"]
    finally:
        writer.close()


# -- T11: store_uuid ---------------------------------------------------------------

def _metadata_uuid(db: Path) -> str | None:
    with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'store_uuid'").fetchone()
    return row[0] if row else None


def test_t11_writer_stamps_store_uuid_once_and_keeps_it(tmp_path):
    """REGRESSION: the migration chain writes store_uuid once; reopening never changes it."""
    db = tmp_path / "lcm.db"
    MessageStore(db).close()
    first = _metadata_uuid(db)
    assert first and re.fullmatch(r"[0-9a-f]{32}", first)
    SummaryDAG(db).close()
    MessageStore(db).close()
    VectorStore(db, config=LCMConfig(database_path=str(db))).close()
    assert _metadata_uuid(db) == first


def test_t11_reader_returns_store_uuid_and_accepts_qualified_ids(fx, monkeypatch):
    """REGRESSION: every result carries the store's uuid; own-prefixed ids == bare ids."""
    reader = make_reader(fx, monkeypatch)
    stamped = _metadata_uuid(fx.db)
    bare, _, bare_text = call(reader, "lcm_expand", {"store_id": fx.s1_ids[0]})
    qualified, is_error, qualified_text = call(reader, "lcm_expand", {"store_id": f"{stamped}:{fx.s1_ids[0]}"})
    assert not is_error and bare["store_uuid"] == stamped == qualified["store_uuid"]
    assert bare["reader"]["store_uuid_source"] == "metadata"
    assert strip_envelope(bare_text) == strip_envelope(qualified_text)
    node, is_error, _ = call(reader, "lcm_describe", {"node_id": f"{stamped}:{fx.node_ids[0]}",
                                                      "session_id": f"{stamped}:s1"})
    assert not is_error and node["node_id"] == fx.node_ids[0]


def test_t11_foreign_prefix_is_refused(fx, monkeypatch):
    """REGRESSION: an id qualified with another store's uuid is refused, never resolved locally."""
    reader = make_reader(fx, monkeypatch)
    other = "0" * 32
    payload, is_error, _ = call(reader, "lcm_expand", {"store_id": f"{other}:{fx.s1_ids[0]}"})
    assert is_error and payload["error"].startswith("foreign_store:")
    payload, is_error, _ = call(reader, "lcm_recall", {"query": "heron", "detail": "answer_ready",
                                                      "seen_refs": [f"{other}:lcm:1:0-5"]})
    assert is_error and payload["error"].startswith("foreign_store:")


def test_t11_absent_store_uuid_is_derived_and_flagged(fx, monkeypatch):
    """REGRESSION: a store without the stamp gets sha256(realpath, st_dev, st_ino)[:32], flagged."""
    with sqlite3.connect(fx.db) as conn:
        conn.execute("DELETE FROM metadata WHERE key = 'store_uuid'")
    reader = make_reader(fx, monkeypatch)
    payload, _, _ = call(reader, "lcm_describe", {"session_id": "s1"})
    real = os.path.realpath(fx.db)
    st = os.stat(real)
    expected = hashlib.sha256(f"{real}\x1f{st.st_dev}\x1f{st.st_ino}".encode()).hexdigest()[:32]
    assert payload["store_uuid"] == expected
    assert payload["reader"]["store_uuid_source"] == "derived"


# -- T12: product seams default to today's behaviour --------------------------------

def _semantic_engine(tmp_path):
    db = tmp_path / "seam.db"
    config = LCMConfig(database_path=str(db), embeddings_enabled=True, embedding_provider=PROVIDER,
                       embedding_model=MODEL, embedding_query_timeout_s=10.0)
    store = MessageStore(db, ingest_protection_config=config)
    dag = SummaryDAG(db)
    node = dag.add_node(_node("seam", "semantic seam summary", []))
    _seed_vectors(db, config, {node: "semantic seam summary"}, {})
    return SimpleNamespace(_config=config, _store=store, _dag=dag, current_session_id="seam")


def test_t12_knn_backend_defaults_to_module_vectorstore(tmp_path, monkeypatch):
    """CONTROL: an engine without ``_vector_store_cls`` keeps ``tools.VectorStore`` as the backend."""
    built = []

    class Recording(VectorStore):
        _supports_pooling = False

        def __init__(self, *args, **kwargs):
            built.append(type(self).__name__)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(lcm_tools, "VectorStore", Recording)
    engine = _semantic_engine(tmp_path)
    try:
        payload = json.loads(lcm_tools.lcm_grep({"query": "semantic seam", "mode": "semantic"}, engine=engine))
        assert payload["degraded_to_fts"] is False
        assert built == ["Recording"]
        mock_engine = MagicMock(wraps=engine)
        mock_engine._config, mock_engine._store, mock_engine._dag = engine._config, engine._store, engine._dag
        mock_engine.current_session_id = "seam"
        mock_engine._lcm_embedding_provider_cache = None
        lcm_tools.lcm_grep({"query": "semantic seam", "mode": "semantic"}, engine=mock_engine)
        assert built == ["Recording", "Recording"]  # a mock's auto-attribute is not a class binding
    finally:
        engine._store.close()
        engine._dag.close()


def test_t12_externalized_payload_stays_session_scoped_by_default(fx):
    """CONTROL: without ``_reader_any_session`` a cross-session payload is still refused."""
    engine = SimpleNamespace(_config=fx.config, _hermes_home=str(fx.home), current_session_id="s1",
                             _store=None, _dag=None)
    payload = json.loads(lcm_tools.lcm_expand({"externalized_ref": fx.ref}, engine=engine))
    assert payload == {"error": f"Externalized payload {fx.ref} not found in current session"}
    mock_engine = MagicMock()
    mock_engine._config, mock_engine._hermes_home, mock_engine.current_session_id = fx.config, str(fx.home), "s1"
    mock_engine._rotation_predecessor_session_ids = None
    payload = json.loads(lcm_tools.lcm_expand({"externalized_ref": fx.ref}, engine=mock_engine))
    assert "error" in payload  # a truthy mock attribute does not widen the reach
    engine.current_session_id = "s2"
    assert json.loads(lcm_tools.lcm_expand({"externalized_ref": fx.ref}, engine=engine))["session_id"] == "s2"


def test_t12_only_construction_site_is_the_engine_bound_class(fx, monkeypatch):
    """REGRESSION: no stock ``VectorStore(`` is reachable from the five tools through the reader."""
    stock_inits = []
    real_init = VectorStore.__init__

    def guarded_init(self, *args, **kwargs):
        stock_inits.append(type(self).__name__)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(VectorStore, "__init__", guarded_init)
    reader = make_reader(fx, monkeypatch)
    for name, args in sweep(fx):
        reader.call(name, args)
    assert stock_inits == []
    source = (PLUGIN_DIR / "tools.py").read_text() + (PLUGIN_DIR / "retrieval_core.py").read_text()
    assert "vector_store_cls=VectorStore" not in source
    assert not re.search(r"(?<![\w.])VectorStore\(", source)


# -- T13: profile configuration ------------------------------------------------------

def test_t13_profile_env_reaches_config_and_nothing_else(fx, monkeypatch):
    """REGRESSION: profile LCM_* keys build LCMConfig; non-LCM keys and the reader's own LCM_* never land."""
    mcp_reader, _ = _reader_modules()
    environ = {"LCM_HERMES_BASE_DIR": "/somewhere/else", "LCM_EMBEDDINGS_ENABLED": "false", "PATH": "/bin"}
    applied = mcp_reader.apply_profile_environment(fx.home, environ)
    assert "VOYAGE_API_KEY" not in environ and NON_LCM_SECRET not in environ.values()
    assert "LCM_HERMES_BASE_DIR" not in environ
    assert environ["LCM_EMBEDDINGS_ENABLED"] == "true"
    assert environ["LCM_EMBEDDING_MODEL"] == MODEL  # export + quotes parsed
    assert environ["LCM_LARGE_OUTPUT_EXTERNALIZATION_THRESHOLD_CHARS"] == "200"  # inline comment dropped
    assert environ["HERMES_HOME"] == str(fx.home) and environ["PATH"] == "/bin"
    assert all(key.startswith("LCM_") for key in applied)
    for key in [k for k in os.environ if k.startswith("LCM_")]:
        monkeypatch.delenv(key)
    for key, value in environ.items():
        monkeypatch.setenv(key, value)
    config = LCMConfig.from_env()
    assert config.embeddings_enabled is True and config.embedding_provider == PROVIDER
    assert config.large_output_externalization_threshold_chars == 200
    assert os.environ.get("VOYAGE_API_KEY") != NON_LCM_SECRET


# -- seams of the read-only construction itself ---------------------------------------

def test_readonly_classes_carry_every_stock_instance_field(fx):
    """REGRESSION: the constructor-free views keep the stock instance fields (drift guard)."""
    _, readonly_store = _reader_modules()
    stock_store = MessageStore(fx.db, ingest_protection_config=fx.config, hermes_home=str(fx.home))
    stock_dag = SummaryDAG(fx.db)
    stock_vs = VectorStore(fx.db, config=fx.config)
    conn = readonly_store.open_readonly_connection(fx.db)
    try:
        ro_store = readonly_store.ReadOnlyMessageStore(fx.db, conn, config=fx.config,
                                                       hermes_home=str(fx.home), stem_usable=True)
        ro_dag = readonly_store.ReadOnlySummaryDAG(fx.db, conn)
        ro_vs = readonly_store.ReadOnlyVectorStore(fx.db, config=fx.config)
        stock_fields = set(vars(stock_store)) - {"_identity_anchor_schema_ready", "_host_uid_schema_ready"}
        assert stock_fields <= set(vars(ro_store))
        assert set(vars(stock_dag)) <= set(vars(ro_dag))
        assert set(vars(stock_vs)) == set(vars(ro_vs))
        ro_vs._matrix_cache["k"] = "v"
        ro_vs.close()
        assert ro_vs._matrix_cache == {"k": "v"}  # close never clears caches shared with copies
    finally:
        stock_store.close()
        stock_dag.close()
        stock_vs.close()
        conn.close()


def test_readonly_open_refuses_writes_but_allows_temp_staging(fx):
    """REGRESSION: mode=ro without query_only -- TEMP works, main writes fail, no checkpoint on close."""
    _, readonly_store = _reader_modules()
    conn = readonly_store.open_readonly_connection(fx.db)
    try:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 0
        conn.execute("CREATE TEMP TABLE scratch_1192(x)")
        conn.execute("INSERT INTO scratch_1192 VALUES (1)")
        assert conn.in_transaction is False
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO metadata(key, value) VALUES('x', 'y')")
    finally:
        conn.close()
    store = readonly_store.ReadOnlyMessageStore(
        fx.db, readonly_store.open_readonly_connection(fx.db), config=fx.config,
        hermes_home=str(fx.home), stem_usable=True)
    statements = []
    store._conn.set_trace_callback(statements.append)
    store.close()
    assert not any("wal_checkpoint" in s for s in statements)
