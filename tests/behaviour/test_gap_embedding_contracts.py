"""Provider transport and optional dependency contracts without network or models."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

providers = importlib.import_module("hermes_lcm.embedding_provider")


def test_spend_guard_clear_reopens_without_advancing_time() -> None:
    guard = providers.EmbeddingSpendGuard(max_calls=2, backoff_seconds=100)
    guard.record_call(now=1)
    guard.record_call(now=1)
    assert not guard.allows(now=1)
    guard.clear()
    assert guard.allows(now=1)
    guard.record_call(now=1)
    assert guard.allows(now=1)


@pytest.mark.parametrize("name", ["voyage", "ollama", "openai-compatible"])
def test_http_documents_report_discovered_dimension(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def transport(**kwargs: Any) -> Any:
        calls.append(kwargs)
        body = {"embeddings": [[3.0, 4.0]]} if name == "ollama" else {"data": [{"index": 0, "embedding": [3.0, 4.0]}]}
        return providers.HttpResponse(status=200, headers={}, body=json.dumps(body).encode())

    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-fixture")
    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "synthetic-fixture")
    monkeypatch.setattr(providers, "count_tokens", lambda _text: 1)
    if name == "voyage":
        provider = providers.VoyageProvider("fixture", transport=transport)
    elif name == "ollama":
        provider = providers.OllamaProvider("fixture", transport=transport)
    else:
        provider = providers.OpenAICompatibleProvider("fixture", base_url="https://example.test/v1", transport=transport)
    assert provider.dim == 0
    assert provider.embed_documents(["document"]) == [[3.0, 4.0]]
    assert provider.dim == 2
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["openai-compatible", "voyage"])
@pytest.mark.parametrize("dispatch", [False, True])
def test_provider_batches_keep_input_indexes_and_dispatch_order(
    monkeypatch: pytest.MonkeyPatch, dispatch: bool, name: str,
) -> None:
    events: list[Any] = []
    documents = ["first", "second", "third", "fourth"]
    expected_vectors = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    vectors_by_document = dict(zip(documents, expected_vectors))

    def transport(**kwargs: Any) -> Any:
        payload = kwargs["payload"]["input"]
        events.append(payload)
        # Response indexes are request-local, deliberately returned out of order.
        data = [{"index": index, "embedding": vectors_by_document[text]}
                for index, text in reversed(list(enumerate(payload)))]
        return providers.HttpResponse(status=200, headers={}, body=json.dumps({"data": data}).encode())

    monkeypatch.setenv("LCM_EMBEDDING_API_KEY", "synthetic-fixture")
    monkeypatch.setenv("VOYAGE_API_KEY", "synthetic-fixture")
    monkeypatch.setattr(providers, "count_tokens", lambda _text: 1)
    if name == "voyage":
        provider = providers.VoyageProvider("fixture", transport=transport, max_batch_items=2)
        expected_indexes = [(0, 1), (2, 3)]
        expected_payloads = [["first", "second"], ["third", "fourth"]]
    else:
        # observed behaviour; see REPORT: OpenAI-compatible has no batching knob.
        provider = providers.OpenAICompatibleProvider("fixture", base_url="https://example.test/v1", transport=transport)
        expected_indexes = [(0, 1, 2, 3)]
        expected_payloads = [["first", "second", "third", "fourth"]]
    batches = list(provider.embed_document_batches(documents, before_dispatch=events.append if dispatch else None))
    assert [batch.indexes for batch in batches] == expected_indexes
    assert [vector for batch in batches for vector in batch.vectors] == expected_vectors
    assert [event for event in events if isinstance(event, list)] == expected_payloads
    expected_events: list[Any] = []
    for indexes, payload in zip(expected_indexes, expected_payloads):
        if dispatch:
            expected_events.append(indexes)
        expected_events.append(payload)
    assert events == expected_events
    assert list(provider.embed_document_batches([])) == []
    assert events == expected_events


def test_fastembed_import_constructs_local_only_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[dict[str, Any]] = []

    class TextEmbedding:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)

        def embed(self, texts: Any) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    module = ModuleType("fastembed")
    module.TextEmbedding = TextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    provider = providers.FastembedProvider("fixture", cache_dir=tmp_path)
    assert provider.embed_documents(["first"]) == [[1.0, 0.0]]
    assert provider.dim == 2
    assert constructed == [{"model_name": "fixture", "cache_dir": str(tmp_path), "threads": 2, "local_files_only": True}]
