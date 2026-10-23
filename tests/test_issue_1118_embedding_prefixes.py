"""Prefix contracts; compat tests intentionally pass on the pinned BASE."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from hermes_lcm import command, config as config_mod, retrieval_core, tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.embedding_provider import HttpResponse, resolve_provider
from hermes_lcm.ingest_protection import (
    embedding_privacy_revision,
    protect_embedding_text,
)
from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore
from tests.harness import engine_context


PROVIDERS = ("ollama", "openai-compatible")
METHODS = (
    "embed_query", "embed_query_interactive", "embed_documents", "embed_document_batches",
)
QUERY_PREFIX = " operator-query: "
DOCUMENT_PREFIX = " operator-document: "


def configured(provider, query="", document=""):
    config = LCMConfig(
        embeddings_enabled=True,
        embedding_provider=provider,
        embedding_model="prefix-test",
        embedding_base_url="https://example.invalid/v1",
    )
    # Assignment lets BASE reach the behavioral assertion rather than failing
    # with an unexpected-constructor-keyword error before exercising the path.
    config.embedding_query_prefix = query
    config.embedding_document_prefix = document
    return config


def digest(query=QUERY_PREFIX, document=DOCUMENT_PREFIX):
    return hashlib.sha256(json.dumps([query, document], ensure_ascii=False).encode()).hexdigest()[:12]


def captured(store, provider_name, task):
    if task == "summary":
        return store.capture_identity("prefix-test", provider=provider_name)
    return store._identity_from_profile(
        store._resolve_chunk_profile("prefix-test", provider=provider_name)
    )


def provider_with_transport(config, monkeypatch):
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "fixture-key")
    provider = resolve_provider(config)
    payloads = []

    def transport(**kwargs):
        payload = kwargs["payload"]
        payloads.append(json.dumps(payload).encode())
        vectors = [[1.0, 0.0] for _ in payload["input"]]
        body = (
            {"embeddings": vectors}
            if provider.provider_id == "ollama"
            else {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}
        )
        return HttpResponse(200, {}, json.dumps(body).encode())

    provider._transport = transport
    return provider, payloads


def invoke(provider, method, text):
    if method == "embed_document_batches":
        return list(provider.embed_document_batches([text]))
    if method == "embed_documents":
        return provider.embed_documents([text])
    if method == "embed_query_interactive":
        return provider.embed_query_interactive(text, timeout=2.0)
    return provider.embed_query(text)


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("method", METHODS)
def test_compat_unset_request_bytes(provider_name, method, monkeypatch):
    provider, payloads = provider_with_transport(configured(provider_name), monkeypatch)
    invoke(provider, method, "hello é")
    expected = b'{"model": "prefix-test", "input": ["hello \\u00e9"]'
    expected += b', "truncate": false}' if provider_name == "ollama" else b'}'
    assert payloads == [expected]


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("task", ("summary", "chunk"))
def test_compat_unset_identity(provider_name, task, tmp_path):
    store = VectorStore(tmp_path / "vectors.db", config=configured(provider_name))
    try:
        expected = hashlib.sha256(
            "\x1f".join((provider_name, "prefix-test", "", "2", "float32", "little", task)).encode()
        ).hexdigest()
        assert store.register_profile("prefix-test", provider_name, 2, task=task) == expected
        assert captured(store, provider_name, task).revision == ""
    finally:
        store.close()


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("method", METHODS)
def test_set_exact_request(provider_name, method, monkeypatch):
    config = configured(provider_name, QUERY_PREFIX, DOCUMENT_PREFIX)
    provider, payloads = provider_with_transport(config, monkeypatch)
    invoke(provider, method, "hello")
    prefix = QUERY_PREFIX if method.startswith("embed_query") else DOCUMENT_PREFIX
    assert json.loads(payloads[0])["input"] == [prefix + "hello"]


@pytest.mark.parametrize("source", ("env", "yaml", "yaml-fallback"))
def test_loader_keeps_whitespace(source, monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    if source == "env":
        monkeypatch.setenv("LCM_EMBEDDING_QUERY_PREFIX", QUERY_PREFIX)
        monkeypatch.setenv("LCM_EMBEDDING_DOCUMENT_PREFIX", DOCUMENT_PREFIX)
    else:
        (home / "config.yaml").write_text(
            f'lcm:\n  embedding_query_prefix: "{QUERY_PREFIX}"\n'
            f'  embedding_document_prefix: "{DOCUMENT_PREFIX}"\n'
        )
        if source == "yaml-fallback":
            monkeypatch.setattr(config_mod, "yaml", None)
    config = LCMConfig.from_env()
    assert getattr(config, "embedding_query_prefix", None) == QUERY_PREFIX
    assert getattr(config, "embedding_document_prefix", None) == DOCUMENT_PREFIX


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("task", ("summary", "chunk"))
@pytest.mark.parametrize("pair", ((QUERY_PREFIX, ""), ("", DOCUMENT_PREFIX), (QUERY_PREFIX, DOCUMENT_PREFIX)))
def test_set_identity_digest(provider_name, task, pair, tmp_path, monkeypatch):
    config = configured(provider_name, *pair)
    store = VectorStore(tmp_path / "vectors.db", config=config)
    try:
        provider, _ = provider_with_transport(config, monkeypatch)
        actual = store.register_profile("prefix-test", provider_name, 2, task=task,
                                        revision_suffix=provider.identity_revision_suffix)
        expected = EmbeddingIdentity.canonical(
            provider_name, "prefix-test", "+prefix:" + digest(*pair), 2, "float32", "little", task,
        )
        assert actual == expected.identity_hash
        assert captured(store, provider_name, task) == expected
    finally:
        store.close()


@pytest.mark.parametrize("task", ("summary", "chunk"))
def test_cloud_query_write_identity_and_privacy_order(task, tmp_path, monkeypatch):
    config = configured("openai-compatible", "password=fixture-prefix ", DOCUMENT_PREFIX)
    store = VectorStore(tmp_path / "vectors.db", config=config)
    try:
        revision = embedding_privacy_revision(config)
        provider, payloads = provider_with_transport(config, monkeypatch)
        store.register_profile("prefix-test", "openai-compatible", 2, revision=revision, task=task,
                               revision_suffix=provider.identity_revision_suffix)
        identity = captured(store, "openai-compatible", task)
        engine = SimpleNamespace(_config=config, _store=SimpleNamespace(db_path=store.db_path))
        tools._lcm_grep_embed_query(
            provider, "password=fixture-content", engine=engine, task=task, remaining_s=2.0,
        )
        protected, _, _ = protect_embedding_text("password=fixture-content", config)
        assert "fixture-content" not in protected
        assert json.loads(payloads[0])["input"] == [config.embedding_query_prefix + protected]
        assert identity.revision.endswith(
            "+prefix:" + digest(config.embedding_query_prefix, DOCUMENT_PREFIX)
        )
    finally:
        store.close()


@pytest.mark.parametrize("surface", ("doctor", "status"))
def test_diagnostics_show_digest_without_prefix_text(surface, tmp_path):
    with engine_context(tmp_path, window=8192, embeddings_enabled=True) as engine:
        config = engine._config
        config.embedding_provider = "ollama"
        config.embedding_model = "prefix-test"
        config.embedding_query_prefix = QUERY_PREFIX
        config.embedding_document_prefix = DOCUMENT_PREFIX
        output = (
            command._doctor_text(engine) if surface == "doctor"
            else tools.lcm_status({}, engine=engine)
        )
        assert digest() in output
        assert "query" in output and "document" in output
        assert QUERY_PREFIX not in output
        assert DOCUMENT_PREFIX not in output


@pytest.mark.parametrize("revision", ("", "binprescreen", embedding_privacy_revision(configured("openai-compatible"))))
@pytest.mark.parametrize("suffix", ("", "+prefix:0123456789ab"))
def test_split_keeps_base(revision, suffix):
    from hermes_lcm.ingest_policy import split_prefix_revision
    assert split_prefix_revision(revision + suffix) == (revision, suffix)


def test_pair_encoding_and_aliases():
    from hermes_lcm.ingest_policy import embedding_prefix_suffix
    a = configured("fastembed", "a\x1fb", "c")
    b = configured("fastembed", "a", "b\x1fc")
    assert embedding_prefix_suffix(a, "ollama") != embedding_prefix_suffix(b, "ollama")
    assert embedding_prefix_suffix(a, "openai") == embedding_prefix_suffix(a, "siliconflow")
    assert embedding_prefix_suffix(a, "openai") == "+prefix:" + digest("a\x1fb", "c")
    assert embedding_prefix_suffix(a, "fastembed") == ""
    assert embedding_prefix_suffix(a, "voyage") == ""


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("method", METHODS)
def test_long_input_sent_complete(provider_name, method, monkeypatch):
    provider, payloads = provider_with_transport(
        configured(provider_name, QUERY_PREFIX, DOCUMENT_PREFIX), monkeypatch,
    )
    text = "é long text " * 10000
    invoke(provider, method, text)
    prefix = QUERY_PREFIX if method.startswith("embed_query") else DOCUMENT_PREFIX
    assert json.loads(payloads[0])["input"] == [prefix + text]
    if provider_name == "ollama":
        assert json.loads(payloads[0])["truncate"] is False


def seeded_engine(tmp_path, monkeypatch, provider_name="ollama", prefix=""):
    from tests.test_embedding_backfill import _engine, _seed
    from tests.test_chunk_backfill import _seed_messages
    engine = _engine(tmp_path)
    engine._config.embedding_provider = provider_name
    engine._config.embedding_query_prefix = prefix
    engine._config.embedding_document_prefix = DOCUMENT_PREFIX if prefix else ""
    engine._config.embedding_model = "prefix-test"
    engine._config.embedding_base_url = "https://example.invalid/v1"
    ids = _seed(engine, 1, register=False)
    _seed_messages(engine, [(1, "session-a", "history", "user", "hello world", 1.0)], register=False)
    provider, payloads = provider_with_transport(engine._config, monkeypatch)
    monkeypatch.setattr(command, "resolve_provider", lambda *a, **kw: provider)
    result = command._embedding_register_profiles(engine._config, engine._store.db_path, allow_download=False)
    assert isinstance(result, dict), result
    return engine, ids[0], provider, payloads


def backfill(engine, task):
    if task == "summary":
        return command._embedding_backfill_summary_run(
            engine._config, engine._store.db_path, apply=True, limit=10, retry_uncertain=False,
        )
    return command._chunk_backfill_run(
        engine._config, engine._store.db_path, apply=True, limit=10,
        retry_uncertain=False, policy="conversational", confirm_raw_text=True,
    )


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize("status", (400, 413))
@pytest.mark.parametrize("task", ("summary", "chunk"))
def test_server_rejection_no_vector_write(provider_name, status, task, tmp_path, monkeypatch):
    engine, _, provider, _ = seeded_engine(tmp_path, monkeypatch, provider_name, QUERY_PREFIX)
    requests = []
    def reject(**kwargs):
        requests.append(kwargs["payload"])
        return HttpResponse(status, {}, b'{}')
    provider._transport = reject
    result = backfill(engine, task)
    assert isinstance(result, dict), result
    assert result["embedded"] == 0 and result["failed"]
    assert str(status) in str(result["failed"])
    assert requests[0]["input"][0].startswith(DOCUMENT_PREFIX)
    table = "lcm_embedding_meta" if task == "summary" else "lcm_chunk_meta"
    with sqlite3.connect(engine._store.db_path) as conn:
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.parametrize("task", ("summary", "chunk"))
@pytest.mark.parametrize("initial,new", (("", QUERY_PREFIX), (QUERY_PREFIX, "")))
def test_local_change_stales_then_warmup_restores(task, initial, new, tmp_path, monkeypatch):
    engine, _, _, _ = seeded_engine(tmp_path, monkeypatch, prefix=initial)
    assert backfill(engine, task)["embedded"] == 1
    config = engine._config
    config.embedding_query_prefix = new
    config.embedding_document_prefix = DOCUMENT_PREFIX if new else ""
    provider, _ = provider_with_transport(config, monkeypatch)
    monkeypatch.setattr(command, "resolve_provider", lambda *a, **kw: provider)
    def coverage():
        store = VectorStore(engine._store.db_path, config=config)
        try:
            method = store.knn if task == "summary" else store.knn_chunks
            result = method([1.0, 0.0], model="prefix-test", provider="ollama",
                            revision_suffix=provider.identity_revision_suffix)
            resolver = store._resolve_profile if task == "summary" else store._resolve_chunk_profile
            assert resolver("prefix-test", provider="ollama") is not None  # external BASE behavior
            return result.coverage
        finally:
            store.close()
    assert coverage() == "none"
    def usable():
        return tools._lcm_recall_has_usable_vector_corpus(
            engine, run_summary=task == "summary", run_chunk=task == "chunk",
            provider_name="ollama", model_name="prefix-test", excluded_session_ids=set(),
            deadline=time.monotonic() + 5,
        )
    assert not usable()
    assert "embedding prefix settings changed" in tools._lcm_recall_catchup_reason(engine, provider, task)
    result = backfill(engine, task)
    assert "does not match" in str(result) or "prefix settings changed" in str(result)
    command._embedding_register_profiles(config, engine._store.db_path, allow_download=False)
    assert backfill(engine, task)["embedded"] == 1
    assert coverage() != "none"
    assert usable()


@pytest.mark.parametrize("task", ("summary", "chunk"))
def test_cloud_changed_prefix_reason(task, tmp_path, monkeypatch):
    engine, _, _, _ = seeded_engine(tmp_path, monkeypatch, "openai-compatible")
    engine._config.embedding_query_prefix = QUERY_PREFIX
    provider, payloads = provider_with_transport(engine._config, monkeypatch)
    from hermes_lcm.ingest_policy import EmbeddingIdentityStaleError
    with pytest.raises(EmbeddingIdentityStaleError, match="embedding prefix settings changed"):
        tools._lcm_grep_embed_query(provider, "hello", engine=engine, task=task, remaining_s=2)
    assert payloads == []


def test_prescreen_before_suffix(tmp_path, monkeypatch):
    config = configured("ollama", QUERY_PREFIX, DOCUMENT_PREFIX)
    provider, _ = provider_with_transport(config, monkeypatch)
    hashes = []
    for enabled in (False, True):
        config.embedding_binary_prescreen = enabled
        store = VectorStore(tmp_path / "vectors.db", config=config)
        try:
            hashes.append(store.register_profile("prefix-test", "ollama", 2,
                          revision_suffix=provider.identity_revision_suffix))
            assert captured(store, "ollama", "summary").revision == (
                ("binprescreen" if enabled else "") + "+prefix:" + digest()
            )
        finally:
            store.close()
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize("provider_name", ("fastembed", "voyage"))
def test_other_provider_ignores_prefixes(provider_name, tmp_path, monkeypatch):
    config = configured(provider_name, QUERY_PREFIX, DOCUMENT_PREFIX)
    provider = resolve_provider(config)
    assert getattr(provider, "identity_revision_suffix", "") == ""
    store = VectorStore(tmp_path / "vectors.db", config=config)
    try:
        actual = store.register_profile("prefix-test", provider_name, 2)
        assert actual == EmbeddingIdentity.canonical(provider_name, "prefix-test", "", 2, "float32", "little", "summary").identity_hash
    finally:
        store.close()

    inputs = []
    if provider_name == "fastembed":
        def encode(texts):
            inputs.extend(texts)
            return [[1.0, 0.0] for _ in texts]
        monkeypatch.setattr(provider, "_ensure_local", lambda: SimpleNamespace(embed=encode, query_embed=encode))
        for method in ("embed_query", "embed_documents", "embed_document_batches"):
            invoke(provider, method, "hello")
    else:
        monkeypatch.setenv("VOYAGE_API_KEY", "fixture-key")
        provider, payloads = provider_with_transport(config, monkeypatch)
        for method in METHODS:
            invoke(provider, method, "hello")
        inputs = [text for payload in payloads for text in json.loads(payload)["input"]]
    assert inputs == ["hello"] * (3 if provider_name == "fastembed" else 4)


def test_provider_override_uses_own_id(tmp_path, monkeypatch):
    config = configured("fastembed", QUERY_PREFIX, DOCUMENT_PREFIX)
    engine = SimpleNamespace(_config=config, _embedding_provider_cache={})
    provider = tools._resolve_recall_provider(engine, provider_override="ollama")
    assert provider.identity_revision_suffix == "+prefix:" + digest()
    config = configured("ollama", QUERY_PREFIX, DOCUMENT_PREFIX)
    engine = SimpleNamespace(_config=config, _embedding_provider_cache={})
    provider = tools._resolve_recall_provider(engine, provider_override="fastembed")
    assert getattr(provider, "identity_revision_suffix", "") == ""

    engine, _, _, _ = seeded_engine(tmp_path, monkeypatch, prefix=QUERY_PREFIX)
    assert backfill(engine, "summary")["embedded"] == 1
    engine._config.embedding_provider = "fastembed"
    override = tools._resolve_recall_provider(engine, provider_override="ollama")
    assert override.identity_revision_suffix == "+prefix:" + digest()
    assert len(engine._lcm_proactive_provider_cache[0]) == 2
    assert tools._lcm_recall_has_usable_vector_corpus(
        engine, run_summary=True, run_chunk=False, provider_name=override.provider_id,
        model_name=override.model_id, excluded_session_ids=set(), deadline=time.monotonic() + 5,
    )


@pytest.mark.parametrize("task", ("summary", "chunk"))
def test_cloud_document_privacy_before_prefix(task, tmp_path, monkeypatch):
    engine, _, _, payloads = seeded_engine(tmp_path, monkeypatch, "openai-compatible", QUERY_PREFIX)
    engine._config.embedding_document_prefix = "password=fixture-prefix "
    provider, payloads = provider_with_transport(engine._config, monkeypatch)
    monkeypatch.setattr(command, "resolve_provider", lambda *a, **kw: provider)
    command._embedding_register_profiles(engine._config, engine._store.db_path, allow_download=False)
    with sqlite3.connect(engine._store.db_path) as conn:
        conn.execute("UPDATE summary_nodes SET summary='password=fixture-content' WHERE depth=0")
        conn.execute("UPDATE messages SET content='password=fixture-content'")
    result = backfill(engine, task)
    assert result["embedded"] == 1, result
    text = json.loads(payloads[-1])["input"][0]
    assert text.startswith("password=fixture-prefix ")
    assert "fixture-content" not in text


@pytest.mark.parametrize("spelling", ("ollama", "Ollama", " ollama", "OpenAI"))
@pytest.mark.parametrize("task", ("summary", "chunk"))
def test_backfill_normalized_provider_suffix(spelling, task, tmp_path, monkeypatch):
    engine, _, provider, _ = seeded_engine(tmp_path, monkeypatch, spelling, QUERY_PREFIX)
    # Exercise the constructor and the apply head using the same non-canonical config.
    assert provider.identity_revision_suffix == "+prefix:" + digest()
    result = backfill(engine, task)
    assert isinstance(result, dict) and result["embedded"] == 1, result


@pytest.mark.parametrize("spelling", ("Ollama", " ollama", "OpenAI"))
def test_status_normalized_provider_suffix(spelling):
    from hermes_lcm.ingest_policy import embedding_prefix_status
    assert embedding_prefix_status(configured(spelling, QUERY_PREFIX, DOCUMENT_PREFIX)) == {
        "embedding_prefix": {"query": True, "document": True, "digest": digest()},
    }


@pytest.mark.parametrize("surface", ("doctor", "status"))
def test_prefix_text_is_key_value_lines(surface, tmp_path):
    from hermes_lcm import command_status
    with engine_context(tmp_path, window=8192, embeddings_enabled=True) as engine:
        engine._config.embedding_provider = "Ollama"
        engine._config.embedding_query_prefix = QUERY_PREFIX
        engine._config.embedding_document_prefix = DOCUMENT_PREFIX
        output = (command._doctor_text(engine) if surface == "doctor"
                  else command_status._status_text(engine))
        assert output.splitlines()[-3:] == [
            "embedding_prefix_query: True", "embedding_prefix_document: True",
            "embedding_prefix_digest: " + digest(),
        ]
        assert QUERY_PREFIX not in output and DOCUMENT_PREFIX not in output


@pytest.mark.parametrize("task", ("summary", "chunk"))
@pytest.mark.parametrize("initial,new", (("", QUERY_PREFIX), (QUERY_PREFIX, "")))
def test_product_knn_goes_stale(task, initial, new, tmp_path, monkeypatch):
    engine, _, provider, _ = seeded_engine(tmp_path, monkeypatch, prefix=initial)
    assert backfill(engine, task)["embedded"] == 1
    run = retrieval_core.run_knn if task == "summary" else retrieval_core.run_chunk_knn

    def coverage():
        return run(
            engine, query_vector=[1.0, 0.0], provider=provider, knn_limit=5,
            deadline=time.monotonic() + 5, since=None, until=None,
            conversation_ids=None, source=None, vector_store_cls=VectorStore,
        ).coverage

    assert coverage() != "none"
    engine._config.embedding_query_prefix = new
    engine._config.embedding_document_prefix = DOCUMENT_PREFIX if new else ""
    provider, _ = provider_with_transport(engine._config, monkeypatch)
    monkeypatch.setattr(command, "resolve_provider", lambda *a, **kw: provider)
    assert coverage() == "none"
    command._embedding_register_profiles(engine._config, engine._store.db_path, allow_download=False)
    assert backfill(engine, task)["embedded"] == 1
    assert coverage() != "none"
