"""Prefix contracts; compat tests intentionally pass on the pinned BASE."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from hermes_lcm import command, config as config_mod, tools
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
    return hashlib.sha256((query + "\x1f" + document).encode()).hexdigest()[:12]


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
def test_set_identity_digest(provider_name, task, pair, tmp_path):
    config = configured(provider_name, *pair)
    store = VectorStore(tmp_path / "vectors.db", config=config)
    try:
        actual = store.register_profile("prefix-test", provider_name, 2, task=task)
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
        store.register_profile("prefix-test", "openai-compatible", 2, revision=revision, task=task)
        identity = captured(store, "openai-compatible", task)
        provider, payloads = provider_with_transport(config, monkeypatch)
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
