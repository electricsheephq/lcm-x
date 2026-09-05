"""Content-hash embedding cache and LongMemEval cache CLI regressions."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

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


def _question(question_id: str = "q0") -> lme.Question:
    return lme.Question(
        question_id=question_id,
        question_type="single-session-user",
        question="what was said?",
        haystack_session_ids=["s0"],
        haystack_sessions=[[{"role": "user", "content": "cache this exact content"}]],
        answer_session_ids=["s0"],
    )


def test_prewarm_is_resumable_and_skips_all_warm_units(tmp_path):
    raw = _CountingProvider()
    cache_path = tmp_path / "embeddings.db"
    cached = lme.ContentHashEmbeddingCache(
        raw, cache_path, provider_id="stub", model_id="stub-hash-64"
    )

    dry_run = lme.prewarm_embedding_cache([_question()], cached, dry_run=True)
    assert dry_run["would_populate"] == dry_run["unique_request_units"]
    assert dry_run["populated"] == 0
    assert dry_run["dry_run"] is True
    assert raw.calls == 0
    with sqlite3.connect(cache_path) as connection:
        assert connection.execute("SELECT count(*) FROM embedding_cache").fetchone()[0] == 0

    first = lme.prewarm_embedding_cache([_question()], cached)
    calls_after_first = raw.calls
    second = lme.prewarm_embedding_cache([_question()], cached, dry_run=True)

    assert first["unique_request_units"] >= 1
    assert first["populated"] == first["unique_request_units"]
    assert second["already_cached"] == second["unique_request_units"]
    assert second["populated"] == 0
    assert second["would_populate"] == 0
    assert second["dry_run"] is True
    assert raw.calls == calls_after_first


def test_prewarm_changed_manifest_truncates_to_current_scan(tmp_path, monkeypatch):
    documents = ["ordinary document", "password: hunter2000abc"]
    monkeypatch.setattr(
        lme,
        "iter_ingest_embedding_request_units",
        lambda _question: iter(documents),
    )
    raw = _CountingProvider("voyage-context-4")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "manifest-cache.sqlite3")
    manifest = tmp_path / "nested" / "changed.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"unrelated":true}\n', encoding="utf-8")

    dry_run = lme.prewarm_embedding_cache(
        [_question("q-manifest-rebank")],
        cached,
        dry_run=True,
        changed_manifest=manifest,
    )
    real_run = lme.prewarm_embedding_cache(
        [_question("q-manifest-rebank")],
        cached,
        changed_manifest=manifest,
    )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["question_id"] == "q-manifest-rebank"
    assert dry_run["changed_units"] == real_run["changed_units"] == 1
    assert dry_run["privacy"]["changed"] == real_run["privacy"]["changed"] == 1


def test_prewarm_refuses_changed_manifest_that_aliases_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lme,
        "iter_ingest_embedding_request_units",
        lambda _question: iter(["ordinary document"]),
    )
    raw = _CountingProvider("voyage-context-4")
    cache_path = tmp_path / "alias-cache.sqlite3"
    cached = lme.ContentHashEmbeddingCache(raw, cache_path)
    lme.prewarm_embedding_cache([_question("q-alias-seed")], cached)
    before = cache_path.read_bytes()
    calls_after_seed = raw.calls
    assert before

    with pytest.raises(ValueError, match="changed-manifest must not be the embedding cache"):
        lme.prewarm_embedding_cache(
            [_question("q-alias-seed")], cached, dry_run=True, changed_manifest=cache_path
        )
    assert cache_path.read_bytes() == before

    alias = tmp_path / "manifest-alias.jsonl"
    alias.symlink_to(cache_path)
    with pytest.raises(ValueError, match="changed-manifest must not be the embedding cache"):
        lme.prewarm_embedding_cache(
            [_question("q-alias-seed")], cached, dry_run=True, changed_manifest=alias
        )
    assert cache_path.read_bytes() == before

    hard_link = tmp_path / "manifest-hardlink.jsonl"
    os.link(cache_path, hard_link)
    with pytest.raises(ValueError, match="changed-manifest must not be the embedding cache"):
        lme.prewarm_embedding_cache(
            [_question("q-alias-seed")], cached, dry_run=True, changed_manifest=hard_link
        )
    assert cache_path.read_bytes() == before
    assert raw.calls == calls_after_seed


def test_evaluate_question_reports_exact_embed_cache_delta(tmp_path):
    raw = _CountingProvider()
    cached = lme.ContentHashEmbeddingCache(
        raw, tmp_path / "embeddings.db", provider_id="stub", model_id="stub-hash-64"
    )

    first = lme.evaluate_question(
        _question("q-first"),
        cached,
        provider_name="stub",
        tmp_dir=tmp_path / "first",
        embeddings_enabled=True,
        chunk_provider=cached,
    )
    assert first["embed_cache"] == {"hits": 0, "misses": cached.misses}
    before_hits, before_misses = cached.hits, cached.misses

    second = lme.evaluate_question(
        _question("q-second"),
        cached,
        provider_name="stub",
        tmp_dir=tmp_path / "second",
        embeddings_enabled=True,
        chunk_provider=cached,
    )
    assert second["embed_cache"] == {
        "hits": cached.hits - before_hits,
        "misses": cached.misses - before_misses,
    }
    assert second["embed_cache"]["hits"] > 0
    assert second["embed_cache"]["misses"] == 0


def test_resume_aggregate_adds_restored_and_live_embed_cache_counts(tmp_path, monkeypatch):
    questions = [_question("q-first"), _question("q-second")]
    checkpoint = tmp_path / "run" / "per_question_checkpoint.jsonl"
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.sqlite3"))

    def _scored(embed_cache):
        metric = {
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
        return {
            **{arm: dict(metric) for arm in lme.ARMS},
            "hybrid_rerank": {**metric, "rerank_mode": lme.RERANK_MODE_PLACEHOLDER},
            "ingest_ms": 1.0,
            "privacy": {key: 0 for key in lme._PRIVACY_KEYS},
            "corpus_counts": {"messages": 1, "summary_nodes": 1, "chunks": 0},
            "embed_cache": dict(embed_cache),
        }

    resolved = []

    def _resolve(*_args, **_kwargs):
        provider = SimpleNamespace(
            provider_id="stub",
            model_id="stub-hash-64",
            hits=0,
            misses=0,
        )
        resolved.append(provider)
        return lme.HarnessProviderSet(
            summary=provider,
            chunk=provider,
            summary_binding=("stub", "stub-hash-64"),
            chunk_binding=("stub", "stub-hash-64"),
        )

    def _evaluate(_question, provider, **_kwargs):
        provider.hits += 2
        provider.misses += 1
        return _scored({"hits": 2, "misses": 1})

    monkeypatch.setattr(lme, "resolve_harness_providers", _resolve)
    monkeypatch.setattr(lme, "evaluate_question", _evaluate)
    lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "initial",
        reuse_db_template=False,
        checkpoint_path=checkpoint,
    )

    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    header_line = lines[0]
    first_row = json.loads(lines[1])
    first_row["embed_cache"] = {"hits": 10, "misses": 20}
    checkpoint.write_text(
        header_line + "\n" + json.dumps(first_row, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    resumed = lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        reuse_db_template=False,
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id for question in questions],
    )

    assert resumed["ingest"]["embed_cache"] == {"hits": 12, "misses": 21}


def _cached_two_question_run(tmp_path, monkeypatch, *, traffic=(2, 1), cached=True):
    """First run: two checkpoint rows, each carrying ``embed_cache == traffic``, written
    under an embed cache when ``cached`` (rows then carry ``embed_cache_enabled=True``)."""
    questions = [_question("q-cache-a"), _question("q-cache-b")]
    checkpoint = tmp_path / "run" / "per_question_checkpoint.jsonl"
    if cached:
        monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.sqlite3"))
    else:
        monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)
    metric = {
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

    def _resolve(*_args, **_kwargs):
        provider = SimpleNamespace(
            provider_id="stub", model_id="stub-hash-64", hits=0, misses=0
        )
        return lme.HarnessProviderSet(
            summary=provider,
            chunk=provider,
            summary_binding=("stub", "stub-hash-64"),
            chunk_binding=("stub", "stub-hash-64"),
        )

    def _evaluate(_question, provider, **_kwargs):
        provider.hits += traffic[0]
        provider.misses += traffic[1]
        return {
            **{arm: dict(metric) for arm in lme.ARMS},
            "hybrid_rerank": {**metric, "rerank_mode": lme.RERANK_MODE_PLACEHOLDER},
            "ingest_ms": 1.0,
            "privacy": {key: 0 for key in lme._PRIVACY_KEYS},
            "corpus_counts": {"messages": 1, "summary_nodes": 1, "chunks": 0},
            "embed_cache": {"hits": traffic[0], "misses": traffic[1]},
        }

    monkeypatch.setattr(lme, "resolve_harness_providers", _resolve)
    monkeypatch.setattr(lme, "evaluate_question", _evaluate)
    lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "initial",
        reuse_db_template=False,
        checkpoint_path=checkpoint,
    )
    rows = [
        json.loads(line)
        for line in checkpoint.read_text(encoding="utf-8").splitlines()[1:]
    ]
    expected_traffic = {"hits": traffic[0], "misses": traffic[1]}
    assert [row["embed_cache"] for row in rows] == [expected_traffic] * 2
    assert [row["embed_cache_enabled"] for row in rows] == [cached, cached]
    return questions, checkpoint


def test_completed_resume_reports_restored_embed_cache_without_cache_env(
    tmp_path, monkeypatch
):
    """Re-reporting a completed cached shard without the cache env must keep the
    aggregate cache-pair evidence the rows carry (and construct no provider)."""
    questions, checkpoint = _cached_two_question_run(tmp_path, monkeypatch)
    monkeypatch.delenv(lme.EMBED_CACHE_ENV)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a fully completed resume must not resolve a provider")

    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden)
    resumed = lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        reuse_db_template=False,
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id for question in questions],
    )
    assert resumed["ingest"]["embed_cache"] == {"hits": 4, "misses": 2}


def _partial_resume(tmp_path, monkeypatch, questions, checkpoint):
    """Truncate the checkpoint to its first row, forbid provider resolution, resume."""
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    checkpoint.write_text(lines[0] + "\n" + lines[1] + "\n", encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the mix must be refused before any provider resolves")

    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden)
    return lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        reuse_db_template=False,
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id for question in questions],
    )


@pytest.mark.parametrize("traffic", [(2, 1), (0, 0)], ids=["traffic", "zero-traffic"])
def test_partial_resume_without_cache_env_refuses_rows_written_under_a_cache(
    tmp_path, monkeypatch, traffic
):
    """A cached half must not be summed with an uncached half; refused before any
    provider resolves. The zero-traffic case is the bypass a traffic heuristic left
    open (a cached checkpoint holding only abstention-like rows)."""
    questions, checkpoint = _cached_two_question_run(tmp_path, monkeypatch, traffic=traffic)
    monkeypatch.delenv(lme.EMBED_CACHE_ENV)
    with pytest.raises(
        ValueError,
        match=f"embed_cache_enabled=True but the resumed run has {lme.EMBED_CACHE_ENV} unset",
    ):
        _partial_resume(tmp_path, monkeypatch, questions, checkpoint)


def test_partial_resume_with_cache_env_refuses_uncached_rows(tmp_path, monkeypatch):
    """The inverse mix: rows written without a cache must not be completed under one."""
    questions, checkpoint = _cached_two_question_run(tmp_path, monkeypatch, cached=False)
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "late-cache.sqlite3"))
    with pytest.raises(
        ValueError,
        match=f"embed_cache_enabled=False but the resumed run has {lme.EMBED_CACHE_ENV} set",
    ):
        _partial_resume(tmp_path, monkeypatch, questions, checkpoint)


def test_completed_resume_of_zero_traffic_cached_rows_reports_embed_cache_without_env(
    tmp_path, monkeypatch
):
    """The marker, not the traffic, decides whether the aggregate belongs in the report."""
    questions, checkpoint = _cached_two_question_run(tmp_path, monkeypatch, traffic=(0, 0))
    monkeypatch.delenv(lme.EMBED_CACHE_ENV)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a fully completed resume must not resolve a provider")

    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden)
    resumed = lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        reuse_db_template=False,
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id for question in questions],
    )
    assert resumed["ingest"]["embed_cache"] == {"hits": 0, "misses": 0}


@pytest.mark.parametrize("cached", [True, False], ids=["cached-rows", "uncached-rows"])
@pytest.mark.parametrize("live_env", [True, False], ids=["env-set", "env-unset"])
def test_completed_resume_report_follows_the_rows_posture_not_the_live_env(
    tmp_path, monkeypatch, cached, live_env
):
    """A completed resume constructs no provider; its aggregate embed_cache must
    follow the rows' recorded posture in both directions of live-env disagreement."""
    questions, checkpoint = _cached_two_question_run(tmp_path, monkeypatch, cached=cached)
    if live_env:
        monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "live-cache.sqlite3"))
    else:
        monkeypatch.delenv(lme.EMBED_CACHE_ENV, raising=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a fully completed resume must not resolve a provider")

    monkeypatch.setattr(lme, "resolve_harness_providers", forbidden)
    resumed = lme.run_harness(
        questions,
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "resume",
        reuse_db_template=False,
        checkpoint_path=checkpoint,
        resume=True,
        selected_question_ids=[question.question_id for question in questions],
    )
    if cached:
        assert resumed["ingest"]["embed_cache"] == {"hits": 4, "misses": 2}
    else:
        assert "embed_cache" not in resumed["ingest"]


def test_abstention_row_carries_the_embed_cache_marker(tmp_path, monkeypatch):
    """Abstention rows skip evaluation and carry zero traffic; the marker must still
    record the posture they were written under."""
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.sqlite3"))
    question = _question("q-cache_abs")
    assert question.is_abstention
    checkpoint = tmp_path / "run" / "per_question_checkpoint.jsonl"
    lme.run_harness(
        [question],
        provider_name="stub",
        model="",
        tmp_dir=tmp_path / "initial",
        embeddings_enabled=False,
        reuse_db_template=False,
        checkpoint_path=checkpoint,
    )
    row = json.loads(checkpoint.read_text(encoding="utf-8").splitlines()[1])
    assert row["abstention"] is True
    assert row["embed_cache"] == {"hits": 0, "misses": 0}
    assert row["embed_cache_enabled"] is True


def test_prewarm_rejects_split_summary_chunk_identity_before_embedding(tmp_path):
    raw = _CountingProvider("voyage-4-large")
    cached = lme.ContentHashEmbeddingCache(raw, tmp_path / "embeddings.db")

    with pytest.raises(ValueError, match="summary and chunk embedding identities differ"):
        lme.prewarm_embedding_cache([_question()], cached)

    assert raw.calls == 0


def test_determinism_probe_reports_chunk_mode_only_for_the_run_chunk_identity():
    """The probe embeds the summary identity only; its chunk_embedding_mode may only
    describe the provider the run uses for chunks. voyage-4-large maps to
    voyage-context-4, so the field is null and the split is recorded; a model that
    maps to itself reports its mode."""
    split = _CountingProvider("voyage-4-large")
    report = lme.embedding_determinism_report([_question()], split, sample_size=1)
    assert report["chunk_embedding_mode"] is None
    assert report["chunk_identity"] == {
        "summary": ["voyage", "voyage-4-large"],
        "chunk": ["voyage", "voyage-context-4"],
        "matches_summary": False,
    }
    assert split.calls > 0  # the summary-determinism measurement still ran

    same = _CountingProvider("voyage-context-3")
    report = lme.embedding_determinism_report([_question()], same, sample_size=1)
    assert report["chunk_embedding_mode"] in {"flat", "contextual"}
    assert report["chunk_identity"]["matches_summary"] is True


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
            "--changed-manifest",
            "changed.jsonl",
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
    assert prewarm.changed_manifest == "changed.jsonl"
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


def test_prewarm_cli_refuses_changed_manifest_that_aliases_an_input(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.db"))
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    prepared_manifest = prepared_dir / "manifest.json"
    prepared_question = prepared_dir / "q1.json"
    prepared_question.write_text('{"question_id": "q1"}', encoding="utf-8")
    prepared_manifest.write_text(
        '{"questions": [{"file": "q1.json", "question_id": "q1"}]}', encoding="utf-8"
    )
    shards_dir = tmp_path / "shards"
    (shards_dir / "shard-0").mkdir(parents=True)
    shard_manifest = shards_dir / "shard-0" / "manifest.json"
    shard_question = shards_dir / "shard-0" / "q2.json"
    shard_question.write_text('{"question_id": "q2"}', encoding="utf-8")
    shard_manifest.write_text(
        '{"questions": [{"file": "q2.json", "question_id": "q2"}]}', encoding="utf-8"
    )
    before = {
        path: path.read_bytes()
        for path in (prepared_manifest, shard_manifest, prepared_question, shard_question)
    }
    monkeypatch.setattr(
        cli, "_prepared_shard_questions", lambda _args: pytest.fail("inputs must not be read")
    )
    monkeypatch.setattr(
        cli, "prewarm_embedding_cache", lambda *_a, **_k: pytest.fail("prewarm must not run")
    )
    hard_link = tmp_path / "hardlink.jsonl"
    os.link(shard_manifest, hard_link)
    # Hard links to the per-question files themselves resolve outside every
    # guarded directory yet name the same inode as a corpus input.
    question_hard_link = tmp_path / "question-hardlink.jsonl"
    os.link(prepared_question, question_hard_link)
    shard_question_hard_link = tmp_path / "shard-question-hardlink.jsonl"
    os.link(shard_question, shard_question_hard_link)

    def _args(changed_manifest):
        return SimpleNamespace(
            prepared_dir=str(prepared_dir),
            shards_manifest=str(shards_dir),
            dataset_label="m",
            provider="stub",
            model="stub",
            timeout=30.0,
            dry_run=True,
            changed_manifest=str(changed_manifest),
        )

    for alias in (
        shard_manifest,
        prepared_manifest,
        shards_dir / "shard-0" / "changed.jsonl",
        prepared_dir / "changed.jsonl",
        hard_link,
        question_hard_link,
        shard_question_hard_link,
    ):
        with pytest.raises(SystemExit, match="Refusing --changed-manifest"):
            cli._cmd_prewarm_cache(_args(alias))
    assert {path: path.read_bytes() for path in before} == before


def test_fastembed_prewarm_resolves_with_run_path_warmup(tmp_path, monkeypatch):
    cli = _load_cli()
    monkeypatch.setenv(lme.EMBED_CACHE_ENV, str(tmp_path / "cache.db"))
    monkeypatch.setattr(cli, "_prepared_shard_questions", lambda _args: [])
    calls = []
    prewarm_calls = []

    class _Provider:
        pass

    def _resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return _Provider()

    monkeypatch.setattr(cli, "resolve_harness_provider", _resolve)

    def _prewarm(*args, **kwargs):
        prewarm_calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(cli, "prewarm_embedding_cache", _prewarm)
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
            "--dry-run",
            "--changed-manifest",
            str(tmp_path / "changed.jsonl"),
        ]
    )

    assert cli._cmd_prewarm_cache(args) == 0
    assert calls == [(("fastembed", "local-model"), {"timeout": 300.0, "warmup": True})]
    assert prewarm_calls and prewarm_calls[0][1]["dry_run"] is True
    assert prewarm_calls[0][1]["changed_manifest"] == str(
        tmp_path / "changed.jsonl"
    )
