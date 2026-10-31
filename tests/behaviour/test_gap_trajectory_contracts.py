"""State-index chunking through the public backfill path, with an offline codec."""

from __future__ import annotations

import hashlib
import importlib
import struct
from pathlib import Path
from typing import Any

import pytest

trajectory = importlib.import_module("hermes_lcm.trajectory_store")
tokens = importlib.import_module("hermes_lcm.tokens")
harness = importlib.import_module("tests.harness")


class ByteEncoder:
    """Deliberately splits UTF-8 codepoints; chunker must repair the boundary."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, token: int) -> bytes:
        return bytes([token])


class RecordingProvider:
    provider_id = "fixture"
    model_id = "byte-fixture"
    last_usage_tokens = 0

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.document_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.documents.extend(texts)
        self.document_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.documents.append(text)
        return [1.0, 0.0]


def make_store(
    tmp_path: Path, provider: RecordingProvider,
    texts: tuple[str, ...] = ("visible 🐑 café 🐑 boundary",),
) -> Any:
    asset = tmp_path / "state.png"
    asset.write_bytes(b"fixture")
    identity = trajectory.CorpusIdentity(
        dataset_name="example/gaps", dataset_revision="fixture-v1", harness_commit="fixture-v1",
        tier="small", domain="web", ingest_config_digest="fixture-v1",
    )
    store = trajectory.TrajectoryStore(tmp_path / "lcm.db", identity, asset_root=tmp_path, embedding_provider=provider)
    states = tuple(trajectory.TrajectoryState(
        state_index=index, step=index, url="https://example.test/", incoming_action=None,
        thoughts="Inspect", text=text, screenshot_path=asset,
    ) for index, text in enumerate(texts))
    store.insert(trajectory.TrajectorySource(
        trajectory_id="fixture", ordinal=0, goal="Inspect", start_url=states[0].url,
        outcome="done", states=states, source_payload={"id": "fixture"},
    ))
    store.finalize(["fixture"])
    return store


@pytest.mark.parametrize("budget", [1, 7])
def test_public_state_backfill_preserves_unicode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: int,
) -> None:
    harness.pin_char_counter(monkeypatch)
    monkeypatch.setattr(tokens, "_get_encoder", lambda: ByteEncoder())
    provider = RecordingProvider()
    store = make_store(tmp_path, provider)
    try:
        if budget == 1:
            with pytest.raises(ValueError, match="Unicode character boundary"):
                store.build_state_semantic_index(document_token_budget=budget)
            assert store.active_state_semantic_profile() is None
            # State chunks are pooled in memory, not stored in a separate chunk table.
            for table in ("lcm_trajectory_state_embeddings", "lcm_trajectory_state_embedding_profiles"):
                assert store.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        stats = store.build_state_semantic_index(document_token_budget=7)
        assert stats["states_embedded"] == stats["chunked_states"] == 1
        assert all(len(text.encode("utf-8")) <= 7 for text in provider.documents)
        assert "".join(provider.document_texts) == "visible 🐑 café 🐑 boundary"
        profiles = store.connection.execute(
            "SELECT profile_digest, active FROM lcm_trajectory_state_embedding_profiles"
        ).fetchall()
        assert [tuple(row) for row in profiles] == [(stats["profile_digest"], 1)]
        rows = store.connection.execute("SELECT text FROM lcm_trajectory_states").fetchall()
        assert [row[0] for row in rows] == ["visible 🐑 café 🐑 boundary"]
    finally:
        store.close()


@pytest.mark.xfail(strict=True, reason="known gap: completed state-index resume resends text in a probe; issue #1457")
def test_completed_state_backfill_resumes_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.pin_char_counter(monkeypatch)
    provider = RecordingProvider()
    store = make_store(tmp_path, provider, ("synthetic state text",))
    try:
        store.build_state_semantic_index()
        provider.documents.clear()
        resumed = store.build_state_semantic_index(resume=True)
        assert resumed["already_embedded"] == 1
        assert resumed["states_embedded"] == 0
        assert (provider.documents, resumed["provider_calls"]) == ([], 0)
    finally:
        store.close()


def test_interrupted_state_backfill_resumes_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.pin_char_counter(monkeypatch)
    texts = ("first state", "second state", "third state")
    provider = RecordingProvider()
    store = make_store(tmp_path, provider, texts)
    identity = store.identity
    requests: list[list[str]] = []
    embed_documents = provider.embed_documents

    def fail_second_request(documents: list[str]) -> list[list[float]]:
        requests.append(list(documents))
        if len(requests) == 2:
            raise RuntimeError("fixture interrupted backfill")
        return embed_documents(documents)

    monkeypatch.setattr(provider, "embed_documents", fail_second_request)
    try:
        state_ids = [row[0] for row in store.connection.execute(
            "SELECT state_id FROM lcm_trajectory_states ORDER BY state_id"
        )]
        with pytest.raises(RuntimeError, match="fixture interrupted backfill"):
            store.build_state_semantic_index(batch_max_items=1)
        partial = [tuple(row) for row in store.connection.execute(
            "SELECT state_id, profile_digest, document_sha256, vector "
            "FROM lcm_trajectory_state_embeddings ORDER BY state_id"
        )]
        assert [row[0] for row in partial] == state_ids[:1]
        assert requests == [[texts[0]], [texts[1]]]
        assert store.active_state_semantic_profile() is None
    finally:
        store.close()

    provider = RecordingProvider()
    store = trajectory.TrajectoryStore(
        tmp_path / "lcm.db", identity, asset_root=tmp_path, embedding_provider=provider,
    )
    try:
        assert store.active_state_semantic_profile() is None
        activations: list[Any] = []
        stats = store.build_state_semantic_index(
            batch_max_items=1,
            progress_callback=lambda _stats: activations.append(store.active_state_semantic_profile()),
        )
        assert activations and all(profile is None for profile in activations)
        assert stats["already_embedded"] == 1
        assert stats["states_embedded"] == 2
        # Document requests must only embed the remainder; no contract requires a query probe.
        assert provider.document_texts == list(texts[1:])
        rows = [tuple(row) for row in store.connection.execute(
            "SELECT state_id, profile_digest, document_sha256, vector "
            "FROM lcm_trajectory_state_embeddings ORDER BY state_id"
        )]
        assert rows[:1] == partial
        assert rows == [
            (state_id, stats["profile_digest"], hashlib.sha256(text.encode()).hexdigest(), struct.pack("<ff", 1.0, 0.0))
            for state_id, text in zip(state_ids, texts)
        ]
        profiles = store.connection.execute(
            "SELECT profile_digest, state_count, active FROM lcm_trajectory_state_embedding_profiles"
        ).fetchall()
        assert [tuple(row) for row in profiles] == [(stats["profile_digest"], 3, 1)]
    finally:
        store.close()


def test_query_keeps_lexical_hits_when_attempt_introspection_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness.pin_char_counter(monkeypatch)
    provider = RecordingProvider()
    store = make_store(tmp_path, provider)

    class BrokenIdentity:
        def __str__(self) -> str:
            raise ValueError("fixture identity introspection failure")

    try:
        store.build_semantic_index()
        monkeypatch.setattr(provider, "provider_id", BrokenIdentity())
        hits = store.query("visible")
        assert hits
        assert all(hit.match_kind == "fts" for hit in hits)
        assert store.last_semantic_attempt()["reason"] == "attempt_record_introspection_failed"
        assert store.semantic_attempt_counters()["fallbacks"] == 1
    finally:
        store.close()
