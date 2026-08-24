"""Content-hash embedding cache and LongMemEval cache CLI regressions."""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import benchmarking.longmemeval as lme
from tests.conftest import load_cli as _load_cli


class _CountingProvider:
    provider_id = "voyage"
    dim = 2

    def __init__(self, model_id: str = "voyage-test", *, offset: float = 0.0):
        self.model_id = model_id
        self.offset = offset
        self.calls = 0
        self.documents: list[list[str]] = []

    def embed_documents(self, texts):
        batch = [str(text) for text in texts]
        self.calls += 1
        self.documents.append(batch)
        return [
            [self.offset + float(len(text)), self.offset + float(sum(map(ord, text)))]
            for text in batch
        ]

    def embed_query(self, text):
        return [self.offset, float(len(str(text)))]


class TestCacheHit:
    def test_hit_skips_provider_call(self, tmp_path):
        raw = _CountingProvider()
        cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "embeddings.db")

        expected = cached.embed_documents(["same document"])
        actual = cached.embed_documents(["same document"])

        assert actual == expected
        assert raw.calls == 1
        assert cached.hits == 1
        assert cached.misses == 1


class TestCacheMiss:
    def test_miss_populates_sqlite_row(self, tmp_path):
        raw = _CountingProvider()
        path = tmp_path / "embeddings.db"
        cached = lme.ContentHashEmbeddingCache(raw, path)

        vector = cached.embed_documents(["new document"])[0]

        assert raw.calls == 1
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT provider, model, content_sha256, vector_dim, length(vector_f64_le) "
                "FROM embedding_cache"
            ).fetchone()
        assert row == (
            "voyage",
            "voyage-test",
            lme.ContentHashEmbeddingCache.content_sha256("new document"),
            len(vector),
            len(vector) * 8,
        )


class TestCacheIdentity:
    def test_provider_and_model_are_part_of_key(self, tmp_path):
        path = tmp_path / "embeddings.db"
        first = _CountingProvider("model-a", offset=1.0)
        second = _CountingProvider("model-b", offset=2.0)
        other_provider = _CountingProvider("model-a", offset=3.0)

        vector_a = lme.ContentHashEmbeddingCache(first, path).embed_documents(["shared"])
        vector_b = lme.ContentHashEmbeddingCache(second, path).embed_documents(["shared"])
        vector_other = lme.ContentHashEmbeddingCache(
            other_provider, path, provider_id="other"
        ).embed_documents(["shared"])

        assert first.calls == second.calls == other_provider.calls == 1
        assert vector_a != vector_b != vector_other
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 3


class _ContextualProvider(_CountingProvider):
    supports_contextualized_grouping = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_dispatches = 0

    def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
        self.provider_dispatches += 1
        if before_dispatch is not None:
            raise AssertionError("observation must not select the callback retry contract")
        for group in groups:
            indexes = tuple(index for index, _text in group)
            context = "\u241f".join(text for _index, text in group)
            yield indexes, tuple(self.embed_documents([context + text])[0] for _index, text in group)


def test_contextual_cache_binds_pair_order_and_privacy_and_reuses_without_transport(tmp_path):
    path = tmp_path / "contextual.db"
    groups = [[(0, "alpha"), (1, "beta")], [(2, "gamma")]]
    first = _ContextualProvider("context-model")
    cached = lme.ContentHashEmbeddingCache(first, path, privacy_namespace="public-a")
    expected = list(cached.embed_chunk_group_batches(groups))
    first_dispatches = first.provider_dispatches
    actual = list(cached.embed_chunk_group_batches(groups))

    assert actual == expected
    assert first.provider_dispatches == first_dispatches
    changed = _ContextualProvider("context-model")
    other = lme.ContentHashEmbeddingCache(changed, path, privacy_namespace="public-b")
    list(other.embed_chunk_group_batches([[ (0, "beta"), (1, "alpha") ]]))
    assert changed.provider_dispatches == 1


def test_contextual_cache_observation_preserves_provider_retry_contract(tmp_path):
    class RetryingProvider(_ContextualProvider):
        def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
            if before_dispatch is not None:
                raise AssertionError("callback disables safe retries")
            self.provider_dispatches += 2
            yield from super().embed_chunk_group_batches(groups)
            self.provider_dispatches -= 1

    raw = RetryingProvider("context-model")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "retry.db")
    accounting = lme.ProviderAccounting()
    wrapped = lme._AccountingProvider(cached, accounting, "chunk_documents")
    assert list(wrapped.embed_chunk_group_batches([[(0, "chunk")]]))
    assert accounting.snapshot()["chunk_documents"]["provider_dispatches"] == 2


@pytest.mark.parametrize("path", ["documents", "contextual"])
def test_cache_preserves_completed_dispatch_before_later_refusal(tmp_path, path):
    class PartiallyRefusingProvider(_CountingProvider):
        supports_contextualized_grouping = True

        def _batches(self, indexes, texts):
            yield type(
                "Batch",
                (),
                {
                    "indexes": indexes,
                    "vectors": tuple(self.embed_documents([text])[0] for text in texts),
                },
            )()
            error = RuntimeError("later request refused")
            error.transport_started = False
            raise error

        def embed_document_batches(self, texts):
            yield from self._batches((0,), texts[:1])

        def embed_chunk_group_batches(self, groups):
            index, text = groups[0][0]
            yield from self._batches((index,), (text,))

    cached = lme.ContentHashEmbeddingCache(
        PartiallyRefusingProvider("context-model"), tmp_path / f"{path}.db"
    )
    with pytest.raises(RuntimeError, match="later request refused"):
        if path == "documents":
            cached.embed_documents(["first", "second"])
        else:
            list(cached.embed_chunk_group_batches([[(0, "first")], [(1, "second")]]))
    assert cached.provider_dispatches == 1


def test_contextual_cache_hit_adds_no_stale_usage(tmp_path):
    class UsageProvider(_ContextualProvider):
        last_usage_tokens = 0

        def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
            self.last_usage_tokens = 5
            yield from super().embed_chunk_group_batches(
                groups, before_dispatch=before_dispatch
            )

    cached = lme.ContentHashEmbeddingCache(
        UsageProvider("context-model"), tmp_path / "usage.db"
    )
    accounting = lme.ProviderAccounting()
    wrapped = lme._AccountingProvider(cached, accounting, "chunk_documents")
    groups = [[(0, "chunk")]]
    list(wrapped.embed_chunk_group_batches(groups))
    list(wrapped.embed_chunk_group_batches(groups))
    row = accounting.snapshot()["chunk_documents"]
    assert row["provider_dispatches"] == 1
    assert row["usage_tokens"] == 5


def test_concurrent_contextual_writers_return_first_durable_vectors(tmp_path):
    path = tmp_path / "contextual-race.db"
    barrier = threading.Barrier(2)

    class RacingContextualProvider(_ContextualProvider):
        def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
            barrier.wait(timeout=5)
            yield from super().embed_chunk_group_batches(
                groups, before_dispatch=before_dispatch
            )

    left = lme.ContentHashEmbeddingCache(
        RacingContextualProvider(offset=10.0), path
    )
    right = lme.ContentHashEmbeddingCache(
        RacingContextualProvider(offset=20.0), path
    )
    groups = [[(0, "shared context")]]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(lambda: list(left.embed_chunk_group_batches(groups))),
            executor.submit(lambda: list(right.embed_chunk_group_batches(groups))),
        ]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] == results[1]


class TestCacheEnvGate:
    def test_unset_env_returns_provider_object_unchanged(self, monkeypatch):
        raw = _CountingProvider()
        monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)

        resolved = lme._maybe_cache_harness_provider(raw, provider_name="voyage")

        assert resolved is raw


class TestCacheConcurrentWriters:
    def test_two_threads_use_wal_without_corruption(self, tmp_path):
        path = tmp_path / "embeddings.db"
        barrier = threading.Barrier(2)

        class _RacingProvider(_CountingProvider):
            def embed_documents(self, texts):
                barrier.wait(timeout=5)
                return super().embed_documents(texts)

        left_raw = _RacingProvider(offset=10.0)
        right_raw = _RacingProvider(offset=20.0)
        left = lme.ContentHashEmbeddingCache(left_raw, path)
        right = lme.ContentHashEmbeddingCache(right_raw, path)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(left.embed_documents, ["raced document"]),
                executor.submit(right.embed_documents, ["raced document"]),
            ]
            results = [future.result(timeout=10) for future in futures]

        assert results[0] == results[1]
        with sqlite3.connect(path) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 1


def _question() -> lme.Question:
    return lme.Question(
        question_id="q0",
        question_type="single-session-user",
        question="what was said?",
        haystack_session_ids=["s0"],
        haystack_sessions=[[{"role": "user", "content": "cache this exact content"}]],
        answer_session_ids=["s0"],
    )


def test_prewarm_is_resumable_and_skips_all_warm_units(tmp_path):
    raw = _CountingProvider()
    cached = lme.ContentHashEmbeddingCache(
        raw, tmp_path / "embeddings.db", provider_id="stub", model_id="stub-hash-64"
    )

    first = lme.prewarm_embedding_cache([_question()], cached)
    calls_after_first = raw.calls
    second = lme.prewarm_embedding_cache([_question()], cached)

    assert first["unique_request_units"] >= 1
    assert first["populated"] == first["unique_request_units"]
    assert second["already_cached"] == second["unique_request_units"]
    assert second["populated"] == 0
    assert raw.calls == calls_after_first


def test_prewarm_rejects_split_summary_chunk_identity_before_embedding(tmp_path):
    raw = _CountingProvider("voyage-4-large")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "embeddings.db")

    with pytest.raises(ValueError, match="summary and chunk embedding identities differ"):
        lme.prewarm_embedding_cache([_question()], cached)

    assert raw.calls == 0


def test_cache_cli_subcommands_parse_without_execution():
    cli = _load_cli()

    prewarm = cli._parse_args(
        [
            "prewarm-cache",
            "--prepared-dir",
            "prepared",
            "--shards-manifest",
            "shards",
            "--model",
            "voyage-3-large",
        ]
    )
    probe = cli._parse_args(
        [
            "determinism-probe",
            "--prepared-dir",
            "prepared",
            "--shards-manifest",
            "shards",
            "--model",
            "voyage-3-large",
        ]
    )

    assert prewarm.command == "prewarm-cache"
    assert probe.command == "determinism-probe"
    assert probe.sample_size == 20


def test_empty_cache_env_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, "")

    with pytest.raises(ValueError, match="non-empty SQLite path"):
        lme._maybe_cache_harness_provider(_CountingProvider(), provider_name="voyage")


def test_cache_database_creation_syncs_parent_directory(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(lme, "_fsync_parent_directory", synced.append)
    path = tmp_path / "embeddings.db"

    lme.ContentHashEmbeddingCache(_CountingProvider(), path)

    assert synced == [path]


def test_report_discloses_cache_stats_only_when_env_is_set(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lme, "evaluate_question", lambda question, *_args, **_kwargs: {
            **{
                arm: {
                    "recall@1": 1.0,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "ndcg@10": 1.0,
                    "latency_ms": 1.0,
                    "turn": {
                        "recall@1": 1.0,
                        "recall@5": 1.0,
                        "recall@10": 1.0,
                        "ndcg@10": 1.0,
                        "session_granularity": False,
                    },
                }
                for arm in lme.ARMS
            },
            "ingest_ms": 1.0,
        },
    )
    monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)
    without_cache = lme.run_harness(
        [_question()], provider_name="stub", model="", tmp_dir=tmp_path, reuse_db_template=False
    )
    assert "embed_cache" not in without_cache["ingest"]

    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.db"))
    with_cache = lme.run_harness(
        [_question()], provider_name="stub", model="", tmp_dir=tmp_path, reuse_db_template=False
    )
    assert with_cache["ingest"]["embed_cache"] == {"hits": 0, "misses": 0}


def test_fastembed_prewarm_resolves_with_run_path_warmup(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.db"))
    monkeypatch.setattr(cli, "_prepared_shard_questions", lambda _args: [])
    calls = []

    class _Provider:
        pass

    def _resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return _Provider()

    monkeypatch.setattr(cli, "resolve_harness_provider", _resolve)
    monkeypatch.setattr(cli, "prewarm_embedding_cache", lambda *_args, **_kwargs: {})
    args = cli._parse_args(
        [
            "prewarm-cache",
            "--prepared-dir",
            "prepared",
            "--shards-manifest",
            "shards",
            "--provider",
            "fastembed",
            "--model",
            "local-model",
        ]
    )

    assert cli._cmd_prewarm_cache(args) == 0
    assert calls == [(("fastembed", "local-model"), {"timeout": 300.0, "warmup": True})]
