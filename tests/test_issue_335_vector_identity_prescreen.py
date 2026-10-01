"""Cloud privacy revisions must distinguish binary-prescreen corpora (#335)."""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

import hermes_lcm.command as command_mod
import hermes_lcm.tools as tools_mod
import hermes_lcm.vector_store as vector_mod
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.ingest_protection import embedding_privacy_revision
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore


class CaptureProvider:
    provider_id = "voyage"
    model_id = "voyage-4-large"

    def __init__(self):
        self.queries = []
        self.documents = []

    def embed_query(self, text):
        self.queries.append(str(text))
        return [1.0, 0.0]

    def embed_documents(self, texts):
        self.documents.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


def _config(tmp_path, *, prescreen=False, privacy=True):
    return LCMConfig(
        database_path=str(tmp_path / "vectors.db"), embeddings_enabled=True,
        embedding_provider="voyage", embedding_model="voyage-4-large",
        embedding_binary_prescreen=prescreen, embedding_privacy_enabled=privacy,
        sensitive_patterns=["api_key", "password_assignment"],
    )


def _register(config, *, task="summary"):
    model = config.embedding_model
    if task == "chunk":
        model = command_mod.default_chunk_model(config.embedding_provider, model)
    store = VectorStore(config.database_path, config=config)
    try:
        identity = store.register_profile(
            model, config.embedding_provider, 2,
            revision=embedding_privacy_revision(config), task=task,
        )
        revision = store.connection.execute(
            "SELECT revision FROM lcm_embedding_profile WHERE identity_hash=?", (identity,),
        ).fetchone()[0]
        return identity, revision
    finally:
        store.close()


def _engine_with_summary(tmp_path, *, prescreen=False, count=1):
    config = _config(tmp_path, prescreen=prescreen)
    dag = SummaryDAG(config.database_path)
    try:
        for index in range(count):
            dag.add_node(SummaryNode(
                session_id="test-session", depth=0,
                summary=f"summary {index}: api_key=abcdefghijklmnop",
                created_at=float(index + 1), latest_at=float(index + 1),
            ))
    finally:
        dag.close()
    _register(config)
    return SimpleNamespace(
        _config=config, _store=SimpleNamespace(db_path=config.database_path),
    )


def _summary_backfill(engine, *, apply):
    return command_mod._embedding_backfill_summary_text(
        engine, apply=apply, limit=10, retry_uncertain=False,
    )


def _counts(config, identity):
    with sqlite3.connect(config.database_path) as conn:
        return tuple(conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE identity_hash=?", (identity,),
        ).fetchone()[0] for table in ("lcm_embedding_vectors", "lcm_embedding_binary"))


@pytest.mark.parametrize("privacy", [True, False])
def test_prescreen_on_cloud_privacy_revision_mints_distinct_identity(tmp_path, privacy):
    off = _config(tmp_path, privacy=privacy)
    h_off, _ = _register(off)
    on = _config(tmp_path, prescreen=True, privacy=privacy)
    h_on, revision = _register(on)
    assert h_on != h_off
    assert revision == embedding_privacy_revision(on) + "+binprescreen"


@pytest.mark.parametrize("privacy", [True, False])
def test_prescreen_on_cloud_chunk_identity_is_distinct(tmp_path, privacy):
    off = _config(tmp_path, privacy=privacy)
    h_off, _ = _register(off, task="chunk")
    on = _config(tmp_path, prescreen=True, privacy=privacy)
    h_on, revision = _register(on, task="chunk")
    assert h_on != h_off
    assert revision == embedding_privacy_revision(on) + "+binprescreen"


def test_cloud_prescreen_flip_backfills_sign_bits_for_every_vector(tmp_path, monkeypatch):
    engine = _engine_with_summary(tmp_path, count=2)
    provider = CaptureProvider()
    monkeypatch.setattr(command_mod, "resolve_provider", lambda _c, **_k: provider)
    h_off, _ = _register(engine._config)
    assert "status: complete" in _summary_backfill(engine, apply=True)
    assert _counts(engine._config, h_off) == (2, 0)
    engine._config = _config(tmp_path, prescreen=True)
    h_on, _ = _register(engine._config)
    assert "pending: 2" in _summary_backfill(engine, apply=False)
    assert "status: complete" in _summary_backfill(engine, apply=True)
    assert _counts(engine._config, h_on) == (2, 2)
    assert len(provider.documents) == 2  # The new corpus was actually re-embedded.
    store = VectorStore(engine._config.database_path, config=engine._config)
    try:
        assert store._binary_fully_synced(h_on, chunk=False)
    finally:
        store.close()
    engine._config = _config(tmp_path)
    assert _register(engine._config)[0] == h_off
    assert _counts(engine._config, h_off) == (2, 0)


def test_semantic_query_accepts_composed_prescreen_revision(tmp_path):
    # Passes at base; after composition this requires the tools.py consumer strip.
    engine = _engine_with_summary(tmp_path, prescreen=True)
    provider = CaptureProvider()
    assert tools_mod._lcm_grep_embed_query(
        provider, "api_key=abcdefghijklmnop", engine=engine,
        task="summary", remaining_s=1.0,
    ) == [1.0, 0.0]
    assert len(provider.queries) == 1
    assert "abcdefghijklmnop" not in provider.queries[0]


def test_summary_backfill_dry_run_accepts_composed_revision(tmp_path):
    # Passes at base; after composition this requires the summary consumer strip.
    engine = _engine_with_summary(tmp_path, prescreen=True)
    report = _summary_backfill(engine, apply=False)
    assert "privacy policy differs" not in report
    assert "status: dry-run" in report
    assert "pending: 1" in report


def test_chunk_backfill_dry_run_accepts_composed_revision(tmp_path, monkeypatch):
    # Passes at base; after composition this requires the chunk consumer strip.
    config = _config(tmp_path, prescreen=True)
    _register(config, task="chunk")
    with sqlite3.connect(config.database_path) as conn:
        conn.execute("CREATE TABLE messages (store_id INTEGER PRIMARY KEY, "
                     "session_id TEXT, source TEXT, role TEXT, content TEXT, timestamp REAL)")
        conn.execute("INSERT INTO messages VALUES (1, 'test-session', 'history', "
                     "'user', ?, 1.0)", ("api_key=abcdefghijklmnop " * 10,))
    monkeypatch.setattr(command_mod, "count_tokens", lambda text: len(str(text)))
    import hermes_lcm.chunking as chunking
    monkeypatch.setattr(chunking, "count_tokens", lambda text: len(str(text)))
    engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=config.database_path))
    report = command_mod._chunk_backfill_text(
        engine, apply=False, limit=10, retry_uncertain=False, policy="conversational",
    )
    assert "privacy policy differs" not in report
    assert "status: dry-run" in report
    assert "pending: 1" in report


def test_prescreen_off_identities_unchanged(tmp_path):
    config = _config(tmp_path)
    cases = [("ollama", "model-a", ""), ("voyage", "voyage-4-large", embedding_privacy_revision(config)),
             ("voyage", "voyage-4-large", "privacy:off"), ("ollama", "model-a", "custom")]
    store = VectorStore(config.database_path, config=config)
    try:
        for provider, model, revision in cases:
            for task in ("summary", "chunk"):
                expected = EmbeddingIdentity.canonical(provider, model, revision, 2, "float32", "little", task)
                assert store.register_profile(model, provider, 2, revision=revision, task=task) == expected.identity_hash
        assert store.register_profile("model-a", "ollama", 2) == (
            "46275ea2874ce922b9e4551d7f1af936c4a643cb1ce1903e59f84941433d9f7c"
        )
    finally:
        store.close()


def test_prescreen_on_local_explicit_and_int8_identities_unchanged(tmp_path):
    config = _config(tmp_path, prescreen=True)
    store = VectorStore(config.database_path, config=config)
    try:
        assert store.register_profile("model-a", "ollama", 2) == (
            "2d7ee8936e9479a0b37ad910fd0c91f6f35468a69c8b56c262af49ae061f4cfe"
        )
        for task in ("summary", "chunk"):
            for revision, dtype in (("prescreen", "float32"), (embedding_privacy_revision(config), "int8")):
                expected = EmbeddingIdentity.canonical("voyage", "voyage-4-large", revision, 2, dtype, "little", task)
                assert store.register_profile("voyage-4-large", "voyage", 2, revision=revision, dtype=dtype, task=task) == expected.identity_hash
    finally:
        store.close()


def test_privacy_revisions_carry_the_composable_prefix(tmp_path):
    for privacy in (True, False):
        revision = embedding_privacy_revision(_config(tmp_path, privacy=privacy))
        assert revision.startswith(vector_mod._PRIVACY_REVISION_PREFIX)
        assert "+" not in revision


def test_strip_prescreen_revision_round_trip(tmp_path):
    config = _config(tmp_path, prescreen=True)
    store = VectorStore(config.database_path, config=config)
    try:
        for privacy in (True, False):
            revision = embedding_privacy_revision(_config(tmp_path, privacy=privacy))
            composed = store._prescreen_revision(revision, dtype="float32")
            assert vector_mod.strip_prescreen_revision(composed) == revision
            assert store._prescreen_revision(composed, dtype="float32") == composed
        for revision in ("", "binprescreen", "prescreen", "custom+binprescreen"):
            assert vector_mod.strip_prescreen_revision(revision) == revision
    finally:
        store.close()
