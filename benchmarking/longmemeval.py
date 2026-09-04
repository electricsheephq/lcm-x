"""Offline LongMemEval retrieval-quality harness for hermes-lcm.

This module ingests each LongMemEval question's conversation history into a
fresh temporary LCM store (reusing ``store``/``dag``/``vector_store`` APIs
directly, with no live Hermes host), builds one deterministic summary per
session, optionally backfills summary embeddings, then scores each retrieval
arm against the dataset's labeled evidence sessions.

It is retrieval-only: LongMemEval labels the evidence session(s) per question,
so recall@k / NDCG@k are computable offline without an LLM judge.

Dataset: LongMemEval (Wu et al., ICLR 2025), canonical Hugging Face dataset
``xiaowu0162/longmemeval``. The small default (``longmemeval_s``) and medium
tier (``longmemeval_m``) share one pinned revision (see :data:`DATASET_COORDS`).
The dataset is downloaded once by an explicit operator command and never during
a run.

Export hygiene mirrors ``scripts/lcm_benchmark.py``: the default output is
aggregate-only and contains no transcript content, session ids, or local paths.
The opt-in ``--dump-candidates`` sidecar contains ranked session/turn ids only.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Sequence

from .standalone import ensure_agent_context_engine_importable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOGGER = logging.getLogger(__name__)

BENCHMARK_VERSION = 1
SCHEMA_VERSION = 1
RRF_K = 60

# F7: embed a question's session summaries in one batched ``embed_documents`` call
# instead of one call per session. Sub-batching only guards against a pathologically
# large haystack tripping the provider's per-call deadline; for a typical LongMemEval
# question (tens of sessions) this collapses to the single call the F7 item asks for.
EMBED_BATCH_SIZE = 64

# Canonical LongMemEval dataset coordinates. The revision is PINNED so a run is
# reproducible: `longmemeval_s` has been byte-stable since it was introduced;
# this is the current `main` commit at implementation time (2026-07-17).
DATASET_REPO_ID = "xiaowu0162/longmemeval"
DATASET_REVISION = "2ec2a557f339b6c0369619b1ed5793734cc87533"
DATASET_FILENAME = "longmemeval_s"
DATASET_COORDS = {
    "s": {
        "name": "LongMemEval_S",
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "file": DATASET_FILENAME,
    },
    "m": {
        "name": "LongMemEval_M",
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "file": "longmemeval_m",
    },
}
PREPARED_MANIFEST_SCHEMA_VERSION = 1
PER_QUESTION_CHECKPOINT_FILENAME = "per_question_checkpoint.jsonl"
_CHECKPOINT_HEADER_KEY = "__checkpoint_header__"
_DUMP_HEADER_KEY = "__dump_header__"
_CANDIDATE_DUMP_TOP_K = 10
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IJSON_MIN_VERSION = (3, 2)

PROVIDERS = ("stub", "fastembed", "voyage", "ollama")

# Provider accounting is deliberately transient.  The #347 telemetry surface owns
# serialization and presentation; this module only exposes a small in-memory
# object so tests and an embedding caller can inspect exact request accounting
# without changing the normal checkpoint/report bytes.
ACCOUNTING_ROLES = (
    "summary_documents",
    "chunk_documents",
    "harness_queries",
    "production_lcm_recall_queries",
)

_KNOWN_PROVIDER_DEGRADED_REASONS = (
    ("semantic retrieval is disabled", "provider_disabled"),
    ("embedding provider is not configured", "summary_provider_disabled"),
    ("chunk embedding provider is not configured", "chunk_provider_disabled"),
    ("embedding provider unavailable:", "summary_provider_unavailable"),
    ("chunk embedding provider unavailable:", "chunk_provider_unavailable"),
    ("query embedding failed:", "summary_query_failed"),
    ("chunk query embedding failed:", "chunk_query_failed"),
    ("summary vectors are unavailable", "summary_vectors_unavailable"),
    ("chunk vectors are unavailable", "chunk_vectors_unavailable"),
)
# ``chunk_vectors`` scores the raw-chunk KNN corpus; ``hybrid_rrf3`` fuses it as a
# third arm alongside FTS + summary vectors. ``lcm_recall`` scores the ACTUAL
# production tool (weighted RRF + scope/recency prior + chunk-vs-FTS dedup), not a
# harness reimplementation. All are appended so the earlier arms keep byte-identical
# outputs and report ordering.
ARMS = (
    "fts",
    "summary_vectors",
    "hybrid_rrf",
    "hybrid_rerank",
    "chunk_vectors",
    "hybrid_rrf3",
    "lcm_recall",
)

# LongMemEval `question_type` -> reported category label. Abstention questions
# (``question_id`` ends with ``_abs``) are excluded from recall scoring and
# reported separately as an ``abstention`` count.
CATEGORY_LABELS = {
    "single-session-user": "single-session-user",
    "single-session-assistant": "single-session-assistant",
    "single-session-preference": "single-session-preference",
    "multi-session": "multi-session",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge-update",
}

_WHITESPACE_RE = re.compile(r"\s+")
_STUB_MODEL = "stub-hash-64"
_STUB_DIM = 64
EMBED_CACHE_ENV = "LCM_LONGMEMEVAL_EMBED_CACHE"
CHUNK_EMBEDDING_MODE_ENV = "LCM_LONGMEMEVAL_CHUNK_EMBEDDING_MODE"


def _chunk_embedding_mode() -> str:
    """Return the configured chunk-embedding mode, failing on unknown values."""
    mode = os.environ.get(CHUNK_EMBEDDING_MODE_ENV, "auto").strip().lower()
    if mode not in {"auto", "flat", "contextual"}:
        raise ValueError(
            f"{CHUNK_EMBEDDING_MODE_ENV} must be one of: auto, flat, contextual"
        )
    return mode


@dataclass
class ProviderAccounting:
    """Transient, aggregate-only accounting for harness provider work.

    ``requests`` counts logical provider calls observed by the harness and
    ``provider_dispatches`` counts calls that reached the wrapped provider.  A
    cache hit therefore contributes documents but no dispatch.  Token usage is
    explicitly incomplete when a provider does not expose a per-call usage
    counter; callers must not infer zero usage from that state.
    """

    requests: dict[str, int] = field(
        default_factory=lambda: {role: 0 for role in ACCOUNTING_ROLES}
    )
    documents: dict[str, int] = field(
        default_factory=lambda: {role: 0 for role in ACCOUNTING_ROLES}
    )
    provider_dispatches: dict[str, int] = field(
        default_factory=lambda: {role: 0 for role in ACCOUNTING_ROLES}
    )
    usage_tokens: dict[str, int] = field(
        default_factory=lambda: {role: 0 for role in ACCOUNTING_ROLES}
    )
    usage_tokens_complete: dict[str, bool] = field(
        default_factory=lambda: {role: True for role in ACCOUNTING_ROLES}
    )
    degraded_outcomes: list[str] = field(default_factory=list)

    def _validate_role(self, role: str) -> str:
        normalized = str(role).strip()
        if normalized not in ACCOUNTING_ROLES:
            raise ValueError(f"unknown provider accounting role: {role!r}")
        return normalized

    def record_call(
        self,
        role: str,
        *,
        document_count: int = 0,
        dispatches: int = 1,
        usage_tokens: int = 0,
        usage_complete: bool = True,
    ) -> None:
        role = self._validate_role(role)
        self.requests[role] += 1
        self.documents[role] += max(0, int(document_count))
        self.provider_dispatches[role] += max(0, int(dispatches))
        self.usage_tokens[role] += max(0, int(usage_tokens))
        if not usage_complete:
            self.usage_tokens_complete[role] = False

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic redacted snapshot suitable for local evidence."""
        return {
            role: {
                "requests": int(self.requests[role]),
                "documents": int(self.documents[role]),
                "provider_dispatches": int(self.provider_dispatches[role]),
                "known_usage_tokens": int(self.usage_tokens[role]),
                "usage_tokens": (
                    int(self.usage_tokens[role])
                    if self.usage_tokens_complete[role]
                    else None
                ),
                "usage_tokens_complete": bool(self.usage_tokens_complete[role]),
            }
            for role in ACCOUNTING_ROLES
        }

    def record_degraded_outcomes(self, outcomes: Sequence[str]) -> None:
        self.degraded_outcomes.extend(str(outcome) for outcome in outcomes)


def _is_provider_predispatch_error(exc: BaseException) -> bool:
    return getattr(exc, "transport_started", None) is False


def _accounted_provider_attempt(
    provider: Any,
    accounting: ProviderAccounting | None,
    role: str,
    operation: Callable[[], Any],
) -> Any:
    """Run one provider attempt and record every observable dispatch and usage."""
    if accounting is None:
        return operation()
    before_calls = getattr(provider, "provider_dispatches", None)
    failure: Exception | None = None
    succeeded = False
    try:
        result = operation()
        succeeded = True
        return result
    except Exception as exc:
        failure = exc
        raise
    finally:
        after_calls = getattr(provider, "provider_dispatches", None)
        if failure is not None and _is_provider_predispatch_error(failure):
            dispatches = 0
        elif isinstance(before_calls, int) and isinstance(after_calls, int):
            dispatches = max(0, after_calls - before_calls)
        else:
            dispatches = 1
        raw_usage = getattr(provider, "last_usage_tokens", None) if succeeded else None
        try:
            usage_tokens = max(0, int(raw_usage))
            usage_complete = True
        except (TypeError, ValueError, OverflowError):
            usage_tokens = 0
            usage_complete = dispatches == 0
        accounting.record_call(
            role,
            document_count=1,
            dispatches=dispatches,
            usage_tokens=usage_tokens,
            usage_complete=usage_complete,
        )


class _AccountingProvider:
    """Role-bound provider proxy; it never changes provider behavior or output."""

    def __init__(self, provider: Any, accounting: ProviderAccounting, role: str):
        self._provider = provider
        self._accounting = accounting
        self._role = accounting._validate_role(role)

    def _usage_tokens(self) -> int | None:
        raw = getattr(self._provider, "last_usage_tokens", None)
        if raw is None:
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError, OverflowError):
            return None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        before_calls = getattr(self._provider, "provider_dispatches", None)
        batch_method = getattr(type(self._provider), "embed_document_batches", None)
        dispatches = 0
        known_usage = 0
        usage_complete = True
        try:
            if callable(batch_method):
                vectors_by_index: dict[int, Sequence[float]] = {}

                def before_dispatch(_indexes):
                    nonlocal dispatches
                    dispatches += 1

                for batch in batch_method(
                    self._provider, values, before_dispatch=before_dispatch
                ):
                    indexes = tuple(batch.indexes)
                    vectors = tuple(batch.vectors)
                    if len(indexes) != len(vectors):
                        raise ValueError(
                            "embedding provider returned mismatched indexes/vectors"
                        )
                    vectors_by_index.update(zip(indexes, vectors))
                    usage = self._usage_tokens()
                    if usage is None:
                        usage_complete = False
                    else:
                        known_usage += usage
                if len(vectors_by_index) != len(values):
                    raise ValueError(
                        "embedding provider returned incomplete document vectors"
                    )
                result = [vectors_by_index[index] for index in range(len(values))]
            else:
                result = self._provider.embed_documents(values)
                after_calls = getattr(self._provider, "provider_dispatches", None)
                if isinstance(before_calls, int) and isinstance(after_calls, int):
                    dispatches = max(0, after_calls - before_calls)
                else:
                    dispatches = 1
                usage = self._usage_tokens()
                if dispatches and usage is None:
                    usage_complete = False
                elif dispatches:
                    known_usage = usage
            return [list(vector) for vector in result]
        except Exception as exc:
            after_calls = getattr(self._provider, "provider_dispatches", None)
            if _is_provider_predispatch_error(exc):
                if isinstance(before_calls, int) and isinstance(after_calls, int):
                    dispatches = max(0, after_calls - before_calls)
                else:
                    dispatches = max(0, dispatches - 1)
            elif isinstance(before_calls, int) and isinstance(after_calls, int):
                dispatches = max(dispatches, max(0, after_calls - before_calls))
            elif dispatches == 0:
                dispatches = 1
            usage_complete = dispatches == 0
            raise
        finally:
            self._accounting.record_call(
                self._role,
                document_count=len(values),
                dispatches=dispatches,
                usage_tokens=known_usage,
                usage_complete=usage_complete,
            )

    def embed_query(self, text: str) -> list[float]:
        before_calls = getattr(self._provider, "provider_dispatches", None)
        dispatches = 0
        usage_tokens = 0
        usage_complete = True
        try:
            result = self._provider.embed_query(text)
            after_calls = getattr(self._provider, "provider_dispatches", None)
            if isinstance(before_calls, int) and isinstance(after_calls, int):
                dispatches = max(0, after_calls - before_calls)
            else:
                dispatches = 1
            usage = self._usage_tokens()
            if dispatches and usage is None:
                usage_complete = False
            elif dispatches:
                usage_tokens = usage
            return list(result)
        except Exception as exc:
            after_calls = getattr(self._provider, "provider_dispatches", None)
            if isinstance(before_calls, int) and isinstance(after_calls, int):
                dispatches = max(0, after_calls - before_calls)
            elif not _is_provider_predispatch_error(exc):
                dispatches = 1
            usage_complete = dispatches == 0
            raise
        finally:
            self._accounting.record_call(
                self._role,
                document_count=1,
                dispatches=dispatches,
                usage_tokens=usage_tokens,
                usage_complete=usage_complete,
            )

    def embed_chunk_group_batches(self, groups, *, before_dispatch=None):
        """Forward contextual grouping while accounting every accepted request."""

        document_count = sum(len(group) for group in groups)
        local_dispatches = 0
        known_usage = 0
        usage_complete = True

        def dispatch(indexes):
            nonlocal local_dispatches
            local_dispatches += 1
            if before_dispatch is not None:
                before_dispatch(indexes)

        method = getattr(self._provider, "embed_chunk_group_batches", None)
        if not callable(method):
            raise AttributeError("provider does not support contextualized grouping")
        try:
            result = method(groups, before_dispatch=dispatch)
        except Exception as exc:
            if _is_provider_predispatch_error(exc):
                local_dispatches = max(0, local_dispatches - 1)
            self._accounting.record_call(
                self._role,
                document_count=document_count,
                dispatches=local_dispatches,
                usage_tokens=0,
                usage_complete=local_dispatches == 0,
            )
            raise
        # Generators perform work during iteration, so account after consumption
        # rather than before returning the iterator.
        def consume():
            nonlocal known_usage, local_dispatches, usage_complete
            try:
                for batch in result:
                    usage = self._usage_tokens()
                    if usage is None:
                        usage_complete = False
                    else:
                        known_usage += usage
                    yield batch
            except Exception as exc:
                if _is_provider_predispatch_error(exc):
                    local_dispatches = max(0, local_dispatches - 1)
                usage_complete = local_dispatches == 0
                raise
            finally:
                self._accounting.record_call(
                    self._role,
                    document_count=document_count,
                    dispatches=local_dispatches,
                    usage_tokens=known_usage,
                    usage_complete=usage_complete,
                )

        return consume()

    def __getattr__(self, name: str):
        return getattr(self._provider, name)


@dataclass(frozen=True)
class HarnessProviderSet:
    """Resolved summary/chunk providers with immutable provider/model bindings."""

    summary: Any
    chunk: Any
    summary_binding: tuple[str, str]
    chunk_binding: tuple[str, str]

def _provider_binding(provider: Any, fallback_provider: str, fallback_model: str) -> tuple[str, str]:
    return (
        str(getattr(provider, "provider_id", fallback_provider) or fallback_provider)
        .strip()
        .lower(),
        str(getattr(provider, "model_id", fallback_model) or fallback_model).strip(),
    )


def _configured_chunk_binding(provider: str, model: str) -> tuple[str, str]:
    """Resolve the production chunk binding without resolving any provider."""
    _ensure_hermes_lcm_package()
    from hermes_lcm.embedding_provider import default_chunk_model

    normalized_provider = str(provider).strip().lower()
    return normalized_provider, default_chunk_model(normalized_provider, model)


_PRIVACY_KEYS = (
    "documents",
    "changed",
    "blocked",
    "queries",
    "queries_changed",
    "queries_blocked",
)
_PRIVACY_COUNTS = {key: 0 for key in _PRIVACY_KEYS}
_PRIVACY_QUESTION = {key: 0 for key in _PRIVACY_KEYS}


def _reset_privacy_counts() -> None:
    _PRIVACY_COUNTS.update({key: 0 for key in _PRIVACY_KEYS})
    _PRIVACY_QUESTION.update({key: 0 for key in _PRIVACY_KEYS})


def _attach_privacy_counts(exc: BaseException) -> BaseException:
    """Attach the current aggregate privacy counts to a policy exception."""
    if not hasattr(exc, "privacy_counts"):
        exc.privacy_counts = dict(_PRIVACY_COUNTS)
    return exc


def _record_validator_block(
    exc: BaseException, *, kind: str, count: int
) -> BaseException:
    """Count a dispatch-validator refusal, then attach aggregate counters."""
    if kind == "document":
        blocked_key = "blocked"
    elif kind == "query":
        blocked_key = "queries_blocked"
    else:
        raise ValueError(f"unsupported privacy protection kind: {kind!r}")
    for counters in (_PRIVACY_COUNTS, _PRIVACY_QUESTION):
        counters[blocked_key] += count
    return _attach_privacy_counts(exc)


def _protected(
    text: str,
    config,
    *,
    expected_revision: str | None = None,
    kind: str = "document",
    count: bool = True,
) -> str:
    """Protect one provider-bound value and record transform outcomes.

    ``count=False`` suppresses total/changed counters only; a policy block is
    always counted so refusals remain visible even during de-duplication scans.
    ``blocked`` means documents refused dispatch by either the transform or the
    dispatch validator (and ``queries_blocked`` is the query equivalent).
    """
    if kind not in {"document", "query"}:
        raise ValueError(f"unsupported privacy protection kind: {kind!r}")
    _ensure_hermes_lcm_package()
    from hermes_lcm.ingest_protection import (
        EmbeddingPrivacyPolicyError,
        protect_embedding_text,
    )

    if kind == "document":
        total_key, changed_key, blocked_key = (
            "documents",
            "changed",
            "blocked",
        )
    else:
        total_key, changed_key, blocked_key = (
            "queries",
            "queries_changed",
            "queries_blocked",
        )
    if count:
        _PRIVACY_COUNTS[total_key] += 1
        _PRIVACY_QUESTION[total_key] += 1
    try:
        protected, _revision, changed = protect_embedding_text(
            text, config, expected_revision=expected_revision
        )
    except EmbeddingPrivacyPolicyError as exc:
        _PRIVACY_COUNTS[blocked_key] += 1
        _PRIVACY_QUESTION[blocked_key] += 1
        _attach_privacy_counts(exc)
        raise
    if count and changed:
        _PRIVACY_COUNTS[changed_key] += 1
        _PRIVACY_QUESTION[changed_key] += 1
    return protected


def _embedding_privacy_context(
    provider: str, model: str, *, embeddings_enabled: bool = True
) -> tuple[Any | None, str | None]:
    """Return the production-parity cloud dispatch config and active revision."""
    _ensure_hermes_lcm_package()
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.ingest_protection import (
        embedding_privacy_revision,
        embedding_provider_requires_privacy,
    )

    normalized_provider = str(provider or "").strip().lower()
    requires_privacy = bool(embeddings_enabled) and embedding_provider_requires_privacy(
        normalized_provider
    )
    if not requires_privacy:
        return None, None
    config = LCMConfig(
        embeddings_enabled=True,
        embedding_provider=normalized_provider,
        embedding_model=str(model or "").strip(),
        sensitive_patterns_enabled=False,
        embedding_privacy_enabled=None,
    )
    return config, embedding_privacy_revision(config)


def _ensure_hermes_lcm_package() -> None:
    """Make this source checkout importable as ``hermes_lcm`` (no plugin registration)."""
    ensure_agent_context_engine_importable()
    if "hermes_lcm" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "hermes_lcm",
        _REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(_REPO_ROOT)],
    )
    if spec is None:
        raise RuntimeError("could not create hermes_lcm package spec")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(_REPO_ROOT)]
    module.__package__ = "hermes_lcm"
    sys.modules["hermes_lcm"] = module


# --------------------------------------------------------------------------- #
# Deterministic stub embedder (pure plumbing; scores are meaningless with it).
# --------------------------------------------------------------------------- #


class StubEmbedder:
    """Hash-based unit vectors for offline plumbing tests. No provider calls."""

    provider_id = "stub"

    def __init__(self, dim: int = _STUB_DIM):
        self.model_id = _STUB_MODEL
        self.dim = int(dim)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _WHITESPACE_RE.sub(" ", str(text).lower()).split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            vector[-1] = 1.0
            return vector
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ContentHashEmbeddingCache:
    """Per-document embedding cache keyed by exact provider input text.

    ``embed_documents`` is the cache boundary because Voyage treats each string
    in its outer request list as one independently returned document vector.
    Query embeddings and provider-specific methods are delegated unchanged.
    """

    provider_id: str
    supports_contextualized_grouping = False

    def __init__(
        self,
        provider,
        cache_path: str | Path,
        *,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._provider = provider
        self.cache_path = Path(cache_path)
        self.provider_id = str(
            provider_id or getattr(provider, "provider_id", "")
        ).strip()
        self._model_id = str(
            model_id or getattr(provider, "model_id", "")
        ).strip()
        if not self.provider_id:
            raise ValueError("embedding cache provider id must not be empty")
        if not self._model_id:
            raise ValueError("embedding cache model id must not be empty")
        self.hits = 0
        self.misses = 0
        # Process-local only; never persisted in the cache database or reports.
        self.provider_dispatches = 0
        self.last_usage_tokens: int | None = None
        self._initialize()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return int(getattr(self._provider, "dim", 0))

    @staticmethod
    def content_sha256(text: str) -> str:
        """Hash the exact UTF-8 string passed as one provider document."""
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.cache_path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        cache_existed = self.cache_path.exists()
        with self._connect() as connection:
            mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0])
            if mode.casefold() != "wal":
                raise RuntimeError(
                    f"embedding cache requires SQLite WAL mode, got {mode!r}"
                )
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
                    vector_dim INTEGER NOT NULL CHECK(vector_dim > 0),
                    vector_f64_le BLOB NOT NULL,
                    PRIMARY KEY (provider, model, content_sha256)
                ) WITHOUT ROWID
                """
            )
        if not cache_existed:
            _fsync_parent_directory(self.cache_path)

    @staticmethod
    def _encode_vector(vector: Sequence[float]) -> tuple[int, bytes]:
        values = [float(value) for value in vector]
        if not values:
            raise ValueError("embedding provider returned an empty vector")
        return len(values), struct.pack(f"<{len(values)}d", *values)

    @staticmethod
    def _decode_vector(dimension: int, payload: bytes) -> list[float]:
        expected = int(dimension) * 8
        if int(dimension) <= 0 or len(payload) != expected:
            raise ValueError("embedding cache vector payload is corrupt")
        return list(struct.unpack(f"<{int(dimension)}d", payload))

    def _lookup(self, digests: Sequence[str]) -> dict[str, list[float]]:
        if not digests:
            return {}
        found: dict[str, list[float]] = {}
        with self._connect() as connection:
            for digest in dict.fromkeys(digests):
                row = connection.execute(
                    """
                    SELECT vector_dim, vector_f64_le
                    FROM embedding_cache
                    WHERE provider = ? AND model = ? AND content_sha256 = ?
                    """,
                    (self.provider_id, self.model_id, digest),
                ).fetchone()
                if row is not None:
                    found[digest] = self._decode_vector(int(row[0]), bytes(row[1]))
        return found

    def cached_count(self, texts: Sequence[str]) -> int:
        digests = [self.content_sha256(str(text)) for text in texts]
        return len(self._lookup(digests))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        documents = [str(text) for text in texts]
        if not documents:
            return []
        digests = [self.content_sha256(text) for text in documents]
        cached = self._lookup(digests)
        self.hits += sum(digest in cached for digest in digests)
        self.misses += sum(digest not in cached for digest in digests)
        missing_by_digest: dict[str, str] = {}
        for digest, text in zip(digests, documents):
            if digest not in cached:
                missing_by_digest.setdefault(digest, text)

        if missing_by_digest:
            self.last_usage_tokens = None
            missing_digests = list(missing_by_digest)
            missing_texts = [missing_by_digest[digest] for digest in missing_digests]
            batch_method = getattr(type(self._provider), "embed_document_batches", None)
            dispatches = 0
            known_usage = 0
            usage_complete = True
            if callable(batch_method):
                vectors_by_index: dict[int, Sequence[float]] = {}

                def before_dispatch(_indexes):
                    nonlocal dispatches
                    dispatches += 1

                try:
                    for provider_batch in batch_method(
                        self._provider,
                        missing_texts,
                        before_dispatch=before_dispatch,
                    ):
                        indexes = tuple(provider_batch.indexes)
                        batch_vectors = tuple(provider_batch.vectors)
                        if len(indexes) != len(batch_vectors):
                            raise ValueError(
                                "embedding provider returned mismatched indexes/vectors"
                            )
                        vectors_by_index.update(zip(indexes, batch_vectors))
                        usage = getattr(self._provider, "last_usage_tokens", None)
                        try:
                            known_usage += max(0, int(usage))
                        except (TypeError, ValueError, OverflowError):
                            usage_complete = False
                except Exception as exc:
                    if _is_provider_predispatch_error(exc):
                        dispatches = max(0, dispatches - 1)
                    raise
                finally:
                    self.provider_dispatches += dispatches
                vectors = [vectors_by_index[index] for index in range(len(missing_texts))]
            else:
                self.provider_dispatches += 1
                dispatches = 1
                try:
                    vectors = list(self._provider.embed_documents(missing_texts))
                except Exception as exc:
                    if _is_provider_predispatch_error(exc):
                        self.provider_dispatches -= 1
                    raise
                usage = getattr(self._provider, "last_usage_tokens", None)
                try:
                    known_usage = max(0, int(usage))
                except (TypeError, ValueError, OverflowError):
                    usage_complete = False
            self.last_usage_tokens = known_usage if usage_complete else None
            if len(vectors) != len(missing_texts):
                raise ValueError(
                    "embedding provider returned "
                    f"{len(vectors)} vectors for {len(missing_texts)} texts"
                )
            encoded = [self._encode_vector(vector) for vector in vectors]
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO embedding_cache (
                        provider, model, content_sha256, vector_dim, vector_f64_le
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            self.provider_id,
                            self.model_id,
                            digest,
                            dimension,
                            sqlite3.Binary(payload),
                        )
                        for digest, (dimension, payload) in zip(missing_digests, encoded)
                    ],
                )
            # Another shard may have won INSERT OR IGNORE. Re-read so every
            # caller observes the first vector set durably stored for the key.
            cached.update(self._lookup(missing_digests))

        try:
            return [list(cached[digest]) for digest in digests]
        except KeyError as exc:  # pragma: no cover - defensive storage failure
            raise RuntimeError("embedding cache did not retain a populated key") from exc

    def embed_query(self, text: str) -> list[float]:
        self.provider_dispatches += 1
        succeeded = False
        try:
            result = self._provider.embed_query(text)
            succeeded = True
            return result
        except Exception as exc:
            if _is_provider_predispatch_error(exc):
                self.provider_dispatches -= 1
            raise
        finally:
            usage = (
                getattr(self._provider, "last_usage_tokens", None)
                if succeeded
                else None
            )
            try:
                self.last_usage_tokens = max(0, int(usage))
            except (TypeError, ValueError, OverflowError):
                self.last_usage_tokens = None

    def __getattr__(self, name: str):
        return getattr(self._provider, name)


def _resolved_chunk_embedding_mode(
    chunk_provider: Any, *, enforce_cache_guard: bool = True
) -> str:
    """Resolve flat/contextual semantics against the unwrapped chunk provider."""
    requested = _chunk_embedding_mode()
    underlying = (
        getattr(chunk_provider, "_provider", chunk_provider)
        if isinstance(chunk_provider, ContentHashEmbeddingCache)
        else chunk_provider
    )
    supports_contextual = bool(
        getattr(underlying, "supports_contextualized_grouping", False)
    ) and callable(getattr(underlying, "embed_chunk_group_batches", None))
    if requested == "flat":
        resolved = "flat"
    elif requested == "contextual":
        if not supports_contextual:
            raise ValueError(
                "contextual chunk embedding requested but the chunk provider does not "
                "support contextualized grouping"
            )
        resolved = "contextual"
    else:
        resolved = "contextual" if supports_contextual else "flat"
    if (
        resolved == "contextual"
        and enforce_cache_guard
        and isinstance(chunk_provider, ContentHashEmbeddingCache)
    ):
        raise ValueError(
            "contextual chunk grouping is not cache-backed: set "
            "LCM_LONGMEMEVAL_CHUNK_EMBEDDING_MODE=flat or unset "
            "LCM_LONGMEMEVAL_EMBED_CACHE"
        )
    return resolved


def _embed_cache_totals(providers: Iterable[Any]) -> tuple[int, int]:
    """Return process-local cache hit/miss totals, de-duplicated by provider id."""
    unique = {id(provider): provider for provider in providers}
    return (
        sum(int(getattr(provider, "hits", 0)) for provider in unique.values()),
        sum(int(getattr(provider, "misses", 0)) for provider in unique.values()),
    )


def _maybe_cache_harness_provider(provider, *, provider_name: str):
    cache_path = os.environ.get(EMBED_CACHE_ENV)
    if cache_path is None:
        return provider
    if not cache_path.strip():
        raise ValueError(f"{EMBED_CACHE_ENV} must be a non-empty SQLite path")
    return ContentHashEmbeddingCache(
        provider,
        cache_path,
        provider_id=provider_name,
        model_id=str(getattr(provider, "model_id", "")),
    )


def _fastembed_cache_dir() -> str | None:
    """Cache dir for FastEmbed models, honoring an env override.

    The provider default is ``~/.cache/fastembed``; ``LCM_LONGMEMEVAL_FASTEMBED_CACHE``
    (or ``FASTEMBED_CACHE_PATH``) redirects it, e.g. to a roomy volume.
    """
    import os

    override = os.environ.get("LCM_LONGMEMEVAL_FASTEMBED_CACHE") or os.environ.get(
        "FASTEMBED_CACHE_PATH"
    )
    return override or None


def resolve_harness_provider(
    provider: str,
    model: str,
    *,
    timeout: float = 300.0,
    use_embed_cache: bool = True,
    warmup: bool = True,
    accounting: ProviderAccounting | None = None,
    accounting_role: str = "summary_documents",
):
    """Return a WARMED embedder for ``provider``. ``stub`` stays fully offline.

    Non-stub providers are warmed up once here so ``.dim`` is populated (FastEmbed
    reports dim only after the first embed) and any model download happens before
    the scoring loop rather than inside a per-question deadline.
    """
    resolved = None
    if provider == "stub":
        resolved = StubEmbedder()
    else:
        _ensure_hermes_lcm_package()
    if provider != "stub" and not model:
        raise ValueError(f"--model is required for --provider {provider}")
    if provider in {"fastembed", "fast-embed"}:
        from hermes_lcm.embedding_provider import EmbeddingSpendGuard, FastembedProvider

        # max_calls=0 disables the per-minute call-rate guard, matching the
        # bulk-backfill contract (resolve_provider(for_backfill=True)); the
        # harness embeds thousands of summaries in one pass.
        resolved = FastembedProvider(
            model,
            cache_dir=_fastembed_cache_dir(),
            timeout=timeout,
            spend_guard=EmbeddingSpendGuard(max_calls=0),
        )
        if warmup:
            _accounted_provider_attempt(
                resolved, accounting, accounting_role, resolved.warmup
            )
    elif provider != "stub":
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.embedding_provider import resolve_provider

        config = LCMConfig(
            embedding_provider=provider,
            embedding_model=model,
            embedding_backfill_timeout_s=timeout,
        )
        resolved = resolve_provider(config, for_backfill=True)
    if resolved is None:
        raise ValueError(f"could not resolve embedding provider {provider!r}")
    if warmup and provider != "stub" and int(getattr(resolved, "dim", 0)) == 0:
        _accounted_provider_attempt(
            resolved,
            accounting,
            accounting_role,
            lambda: resolved.embed_query("warmup"),
        )
    if not use_embed_cache:
        return resolved
    return _maybe_cache_harness_provider(resolved, provider_name=provider)


def resolve_harness_providers(
    provider: str,
    model: str,
    *,
    timeout: float = 300.0,
    use_embed_cache: bool = True,
    warmup: bool = True,
    accounting: ProviderAccounting | None = None,
) -> HarnessProviderSet:
    """Resolve one cached provider per production summary/chunk identity.

    The configured model is the summary identity.  Chunk identity follows the
    production mapping, which is especially important for Voyage where plain
    summary models map to ``voyage-context-4``.  A small per-run cache avoids
    resolving the same provider/model twice while retaining immutable identity
    tuples for query-vector reuse decisions.
    """
    cache: dict[tuple[str, str], Any] = {}

    def resolve(requested_model: str, role: str) -> Any:
        key = (str(provider).strip().lower(), str(requested_model).strip())
        if key not in cache:
            cache[key] = resolve_harness_provider(
                provider,
                requested_model,
                timeout=timeout,
                use_embed_cache=use_embed_cache,
                warmup=warmup,
                accounting=accounting,
                accounting_role=role,
            )
        return cache[key]

    summary = resolve(model, "summary_documents")
    summary_binding = _provider_binding(summary, provider, model)
    _ensure_hermes_lcm_package()
    from hermes_lcm.embedding_provider import default_chunk_model

    chunk_model = default_chunk_model(summary_binding[0], summary_binding[1])
    chunk = (
        summary
        if chunk_model == summary_binding[1]
        else resolve(chunk_model, "chunk_documents")
    )
    chunk_binding = _provider_binding(chunk, summary_binding[0], chunk_model)
    return HarnessProviderSet(
        summary=summary,
        chunk=chunk,
        summary_binding=summary_binding,
        chunk_binding=chunk_binding,
    )


# --------------------------------------------------------------------------- #
# Dataset loading + question shape.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Question:
    """A single LongMemEval question with its haystack and labeled evidence."""

    question_id: str
    question_type: str
    question: str
    haystack_session_ids: list[str]
    haystack_sessions: list[list[dict[str, Any]]]
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def category(self) -> str:
        return CATEGORY_LABELS.get(self.question_type, self.question_type)


def parse_question(raw: dict[str, Any]) -> Question:
    return Question(
        question_id=str(raw["question_id"]),
        question_type=str(raw["question_type"]),
        question=str(raw["question"]),
        haystack_session_ids=[str(s) for s in raw.get("haystack_session_ids", [])],
        haystack_sessions=list(raw.get("haystack_sessions", [])),
        answer_session_ids=[str(s) for s in raw.get("answer_session_ids", [])],
    )


def _questions_from_bytes(payload: bytes, *, limit: int | None = None) -> list[Question]:
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("LongMemEval dataset must be a JSON array of questions")
    questions = [parse_question(row) for row in data]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        questions = questions[:limit]
    return questions


def load_questions(path: str | Path, *, limit: int | None = None) -> list[Question]:
    """Load LongMemEval questions from the downloaded JSON file."""
    return _questions_from_bytes(Path(path).read_bytes(), limit=limit)


def load_questions_with_sha256(
    path: str | Path, *, limit: int | None = None
) -> tuple[list[Question], str]:
    """Parse and hash one immutable byte read of a direct dataset file."""
    payload = Path(path).read_bytes()
    return _questions_from_bytes(payload, limit=limit), hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of ``path`` without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_coordinates(dataset_label: str) -> dict[str, str]:
    """Return a copy of the pinned coordinates for a supported dataset label."""
    try:
        return dict(DATASET_COORDS[dataset_label])
    except KeyError as exc:
        raise ValueError(f"unsupported dataset label: {dataset_label!r}") from exc


def validate_dataset_path_label(path: str | Path, dataset_label: str) -> None:
    """Fail closed when a direct dataset filename does not match its label."""
    expected = dataset_coordinates(dataset_label)["file"]
    actual = Path(path).name
    if actual != expected:
        raise ValueError(
            f"dataset label {dataset_label!r} requires filename {expected!r}; got {actual!r}"
        )


class _HashingReader:
    """Minimal binary reader that hashes the same bytes consumed by ``ijson``."""

    def __init__(self, source):
        self._source = source
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self._source.read(size)
        self._digest.update(chunk)
        return chunk

    def drain(self) -> None:
        """Hash bytes after the top-level array; reject non-whitespace trailers.

        Trailing garbage (e.g. a concatenated second array) would otherwise be
        hashed into source_sha256 while its content silently never prepares.
        """
        for chunk in iter(lambda: self.read(1024 * 1024), b""):
            if chunk.strip():
                raise ValueError(
                    "invalid LongMemEval dataset JSON: trailing content after the top-level array"
                )

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _require_ijson():
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            "ijson is required for `prepare`; install it for that command only"
        ) from exc

    installed = getattr(ijson, "__version__", None)
    if installed is None:
        try:
            from importlib.metadata import version

            installed = version("ijson")
        except Exception:
            installed = None
    if installed is not None:
        match = re.match(r"^(\d+)\.(\d+)", str(installed))
        if match and tuple(map(int, match.groups())) < _IJSON_MIN_VERSION:
            raise RuntimeError(
                "ijson >= 3.2 is required for `prepare`; "
                f"found ijson {installed!r}"
            )
    return ijson


def _dataset_root_type(first_byte: int | None) -> str:
    if first_byte is None:
        return "empty input"
    if first_byte == ord("{"):
        return "object"
    if first_byte == ord('"'):
        return "scalar (string)"
    if first_byte in b"-0123456789":
        return "scalar (number)"
    if first_byte in b"tf":
        return "scalar (boolean)"
    if first_byte == ord("n"):
        return "scalar (null)"
    return "invalid"


def _validate_dataset_root(source) -> None:
    """Check the JSON root on the same stream that prepare later consumes."""
    prefix = source.read(3)
    if prefix == b"\xef\xbb\xbf":
        prefix = b""

    first_byte: int | None = None
    while first_byte is None:
        for byte in prefix:
            if byte not in b" \t\r\n":
                first_byte = byte
                break
        if first_byte is not None:
            break
        prefix = source.read(4096)
        if not prefix:
            break

    if first_byte != ord("["):
        raise ValueError(
            "LongMemEval dataset root must be a JSON array; "
            f"got {_dataset_root_type(first_byte)}"
        )


def _iter_dataset_rows(source) -> Iterator[dict[str, Any]]:
    """Incrementally yield top-level array items from a LongMemEval corpus."""
    ijson = _require_ijson()
    try:
        for row in ijson.items(source, "item", use_float=True):
            if not isinstance(row, dict):
                raise ValueError("LongMemEval dataset entries must be JSON objects")
            yield row
    except (ijson.JSONError, ValueError, KeyError, TypeError) as exc:
        # Our own row-shape error is already specific — pass it through unwrapped.
        if isinstance(exc, ValueError) and str(exc).startswith(
            "LongMemEval dataset entries must be JSON objects"
        ):
            raise
        # A TypeError from the call signature means an ijson too old for
        # ``use_float``; any other TypeError is a backend data error and maps
        # to the standard invalid-dataset ValueError below.
        if isinstance(exc, TypeError) and "use_float" in str(exc):
            raise RuntimeError(
                "ijson >= 3.2 is required for `prepare`; "
                "upgrade it for that command only"
            ) from exc
        detail = str(exc).strip()
        offset = next(
            (
                getattr(exc, attribute, None)
                for attribute in ("pos", "position", "offset")
                if getattr(exc, attribute, None) is not None
            ),
            None,
        )
        if offset is not None and str(offset) not in detail:
            detail = f"{detail} (offset {offset})" if detail else f"offset {offset}"
        message = "invalid LongMemEval dataset JSON"
        if detail:
            # Backend builders may report their byte/character offset only in
            # the exception text. Preserve that context while normalizing the
            # backend-specific exception type for callers.
            message = f"{message}: {detail}"
        raise ValueError(message) from exc


def _question_filename(question_id: str) -> str:
    stem = Path(question_id).stem.casefold()
    windows_reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    if (
        not question_id
        or question_id.casefold() in {".", "..", "manifest", "_template"}
        or question_id.startswith(".")
        or stem in windows_reserved
        or "/" in question_id
        or "\\" in question_id
        or any(character in question_id for character in ':*?"<>|')
    ):
        raise ValueError(f"unsafe question_id for prepared output: {question_id!r}")
    return f"{question_id}.json"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def prepare_dataset(
    source_path: str | Path,
    prepared_dir: str | Path,
    *,
    dataset_label: str,
) -> dict[str, Any]:
    """Stream a corpus into checksum-addressed, per-question prepared files.

    The hidden ``.{name}.prepare-*`` staging directory is adjacent to the
    destination and renamed into place atomically.
    """
    source_path = Path(source_path)
    validate_dataset_path_label(source_path, dataset_label)
    if not source_path.is_file():
        raise ValueError(f"dataset file not found: {source_path}")

    prepared_dir = Path(prepared_dir)
    try:
        prepared_dir.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            "prepared staging parent is not writable: "
            f"{prepared_dir.parent} (check --prepared-dir {prepared_dir})"
        ) from exc
    if prepared_dir.exists():
        if not prepared_dir.is_dir() or any(prepared_dir.iterdir()):
            raise ValueError(f"prepared directory must be empty: {prepared_dir}")

    try:
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{prepared_dir.name}.prepare-", dir=prepared_dir.parent)
        )
    except OSError as exc:
        raise OSError(
            "could not create prepared staging directory next to "
            f"{prepared_dir.parent} (check --prepared-dir {prepared_dir})"
        ) from exc
    try:
        questions: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        with source_path.open("rb") as raw_source:
            _validate_dataset_root(raw_source)
            raw_source.seek(0)
            source = _HashingReader(raw_source)
            for row in _iter_dataset_rows(source):
                raw_question_id = row.get("question_id")
                question_id = "" if raw_question_id is None else str(raw_question_id)
                for field in ("question_id", "question_type", "question"):
                    if field not in row or row[field] is None:
                        raise ValueError(
                            f"question {question_id!r} missing required field {field!r}"
                        )
                # Mirror the consumption path's collection requirements so a
                # malformed entry cannot publish and fail later at load time.
                for field in ("haystack_session_ids", "haystack_sessions", "answer_session_ids"):
                    if not isinstance(row.get(field), list):
                        raise ValueError(
                            f"question {question_id!r} field {field!r} must be a list"
                        )
                for index, session in enumerate(row["haystack_sessions"]):
                    if not isinstance(session, list):
                        raise ValueError(
                            f"question {question_id!r} haystack_sessions[{index}] "
                            "must be a list of messages"
                        )
                if question_id.casefold() in seen_ids:
                    raise ValueError(
                        f"duplicate question_id in dataset: {question_id!r} "
                        "(ids are compared case-insensitively; prepared filenames "
                        "collide on case-insensitive filesystems)"
                    )
                filename = _question_filename(question_id)
                payload = _canonical_json_bytes(row)
                (staging_dir / filename).write_bytes(payload)
                questions.append(
                    {
                        "question_id": question_id,
                        "file": filename,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
                seen_ids.add(question_id.casefold())
            source.drain()
            if not questions:
                raise ValueError("LongMemEval dataset must contain at least one question")

            manifest = {
                "schema_version": PREPARED_MANIFEST_SCHEMA_VERSION,
                "dataset_label": dataset_label,
                "source_file": source_path.name,
                "source_sha256": source.hexdigest,
                "question_count": len(questions),
                "questions": questions,
            }
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if prepared_dir.exists():
            # Verified empty above; os.replace cannot replace an existing
            # directory on Windows, so remove the empty target first.
            prepared_dir.rmdir()
        os.replace(staging_dir, prepared_dir)
        return manifest
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


@dataclass(frozen=True)
class PreparedDataset:
    directory: Path
    dataset_label: str
    source_sha256: str
    manifest_sha256: str
    question_count: int
    questions: tuple[dict[str, str], ...]

    def _selected_entries(self, limit: int | None) -> tuple[dict[str, str], ...]:
        if limit is not None and limit <= 0:
            raise ValueError("--limit must be a positive integer")
        return self.questions if limit is None else self.questions[:limit]

    def _read_entry(self, entry: dict[str, str]) -> dict[str, Any]:
        path = self.directory / entry["file"]
        payload = path.read_bytes()
        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != entry["sha256"]:
            raise ValueError(f"prepared question checksum mismatch: {entry['file']}")
        try:
            raw = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid prepared question JSON: {entry['file']}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"prepared question must be a JSON object: {entry['file']}")
        if str(raw.get("question_id", "")) != entry["question_id"]:
            raise ValueError(f"prepared question id mismatch: {entry['file']}")
        return raw

    def validate_question_ids(self, *, limit: int | None = None) -> None:
        """Preflight manifest ids and files without reading question bytes."""
        self._selected_entries(limit)
        expected_files = {entry["file"] for entry in self.questions}
        for entry in self.questions:
            path = self.directory / entry["file"]
            if not path.is_file():
                raise ValueError(f"prepared question file not found: {path}")
        actual_files = {
            path.name for path in self.directory.glob("*.json") if path.name != "manifest.json"
        }
        if actual_files != expected_files:
            raise ValueError("prepared directory JSON file set does not match manifest")

    def selected_question_ids(self, *, limit: int | None = None) -> tuple[str, ...]:
        """Return the manifest-selected ids without reading question payloads."""
        return tuple(entry["question_id"] for entry in self._selected_entries(limit))

    def iter_questions(self, *, limit: int | None = None) -> Iterator[Question]:
        for entry in self._selected_entries(limit):
            yield parse_question(self._read_entry(entry))

    def iter_question_ids(self, question_ids: Sequence[str]) -> Iterator[Question]:
        """Yield an explicit manifest-selected subset in caller order."""
        by_id = {entry["question_id"]: entry for entry in self.questions}
        for question_id in question_ids:
            entry = by_id.get(str(question_id))
            if entry is None:
                raise ValueError(
                    f"shard question id is absent from prepared dataset: {question_id!r}"
                )
            yield parse_question(self._read_entry(entry))


def load_prepared_dataset(
    prepared_dir: str | Path, *, dataset_label: str
) -> PreparedDataset:
    """Validate an entire prepared manifest/file set before returning a lazy reader."""
    prepared_dir = Path(prepared_dir)
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"prepared manifest not found: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid prepared manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("prepared manifest must be a JSON object")
    if manifest.get("schema_version") != PREPARED_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported prepared manifest schema_version")
    if manifest.get("dataset_label") != dataset_label:
        raise ValueError(
            f"dataset label mismatch: requested {dataset_label!r}, "
            f"manifest has {manifest.get('dataset_label')!r}"
        )
    source_sha256 = str(manifest.get("source_sha256", ""))
    if not _SHA256_RE.fullmatch(source_sha256):
        raise ValueError("prepared manifest source_sha256 is invalid")
    expected_source_file = dataset_coordinates(dataset_label)["file"]
    if manifest.get("source_file") != expected_source_file:
        raise ValueError(
            f"prepared manifest source_file does not match label {dataset_label!r}"
        )

    raw_questions = manifest.get("questions")
    if not isinstance(raw_questions, list):
        raise ValueError("prepared manifest questions must be a list")
    if manifest.get("question_count") != len(raw_questions):
        raise ValueError("prepared manifest question_count mismatch")

    questions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    seen_files_folded: set[str] = set()
    for raw_entry in raw_questions:
        if not isinstance(raw_entry, dict):
            raise ValueError("prepared manifest question entries must be objects")
        question_id = str(raw_entry.get("question_id", ""))
        filename = str(raw_entry.get("file", ""))
        expected_filename = _question_filename(question_id)
        checksum = str(raw_entry.get("sha256", ""))
        if filename != expected_filename:
            raise ValueError(f"prepared filename mismatch for question {question_id!r}")
        if question_id.casefold() in seen_ids or filename.casefold() in seen_files_folded:
            raise ValueError(
                f"duplicate prepared question entry: {question_id!r} "
                "(ids and filenames are compared case-insensitively, mirroring prepare)"
            )
        if not _SHA256_RE.fullmatch(checksum):
            raise ValueError(f"invalid prepared question checksum: {filename}")
        path = prepared_dir / filename
        if not path.is_file():
            raise ValueError(f"prepared question file not found: {path}")
        questions.append(
            {"question_id": question_id, "file": filename, "sha256": checksum}
        )
        seen_ids.add(question_id.casefold())
        seen_files.add(filename)
        seen_files_folded.add(filename.casefold())

    actual_files = {path.name for path in prepared_dir.glob("*.json")}
    expected_files = seen_files | {"manifest.json"}
    if actual_files != expected_files:
        raise ValueError("prepared directory JSON file set does not match manifest")

    return PreparedDataset(
        directory=prepared_dir,
        dataset_label=dataset_label,
        source_sha256=source_sha256,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        question_count=len(questions),
        questions=tuple(questions),
    )


def evidence_sessions(question: Question) -> set[str]:
    """Session-level evidence set (empty for abstention questions)."""
    if question.is_abstention:
        return set()
    return set(question.answer_session_ids)


def evidence_turns(question: Question) -> set[tuple[str, int]]:
    """Turn-level evidence: ``(session_id, turn_index)`` where ``has_answer``."""
    turns: set[tuple[str, int]] = set()
    if question.is_abstention:
        return turns
    for session_id, session in zip(question.haystack_session_ids, question.haystack_sessions):
        for index, turn in enumerate(session):
            if isinstance(turn, dict) and turn.get("has_answer"):
                turns.add((str(session_id), index))
    return turns


# --------------------------------------------------------------------------- #
# Deterministic summarization stub (no LLM, content-derived, offline).
# --------------------------------------------------------------------------- #


def deterministic_session_summary(turns: Sequence[dict[str, Any]], *, max_chars: int = 1200) -> str:
    """Condense a session's turns into a deterministic, content-bearing summary.

    Lexical content is preserved (collapsed whitespace, truncated) so the FTS
    and embedding arms both see meaningful text. No provider is consulted.
    """
    parts: list[str] = []
    for turn in turns:
        role = str(turn.get("role", "unknown")) if isinstance(turn, dict) else "unknown"
        content = turn.get("content", "") if isinstance(turn, dict) else str(turn)
        parts.append(f"{role}: {content}")
    condensed = _WHITESPACE_RE.sub(" ", " ".join(parts)).strip()
    if not condensed:
        # LongMemEval_S contains empty haystack sessions; cloud embedding
        # endpoints (voyage) reject empty inputs with HTTP 400, so an empty
        # session gets a deterministic non-empty placeholder instead.
        return "(empty session)"
    return condensed[:max_chars]


def iter_ingest_embedding_request_units(question: Question) -> Iterator[str]:
    """Yield every exact document string embedded during one question's ingest.

    Raw conversational chunks are sent one document at a time; deterministic
    session summaries are sent as a batch, but each batch item remains one
    independently keyed Voyage document request unit.
    """
    _ensure_hermes_lcm_package()
    from hermes_lcm.chunking import iter_message_chunks

    next_store_id = 1
    for session in question.haystack_sessions:
        messages = [
            {
                "role": str(turn.get("role", "user")) if isinstance(turn, dict) else "user",
                "content": turn.get("content", "") if isinstance(turn, dict) else str(turn),
            }
            for turn in session
        ]
        if messages:
            rows = [
                {
                    "store_id": store_id,
                    "role": message["role"],
                    "content": message["content"],
                }
                for store_id, message in zip(
                    range(next_store_id, next_store_id + len(messages)), messages
                )
            ]
            next_store_id += len(messages)
            for chunk in iter_message_chunks(rows, policy="conversational"):
                yield str(chunk.text)
        yield deterministic_session_summary(session)


def load_shard_question_ids(shards_manifest: str | Path) -> tuple[str, ...]:
    """Load ordered qids from one shard manifest or a directory of shard manifests."""
    path = Path(shards_manifest)
    if path.is_dir():
        manifest_paths = sorted(path.glob("shard-*/manifest.json"))
        if not manifest_paths and (path / "manifest.json").is_file():
            manifest_paths = [path / "manifest.json"]
    elif path.is_file():
        manifest_paths = [path]
    else:
        raise ValueError(f"shards manifest not found: {path}")
    if not manifest_paths:
        raise ValueError(f"no shard manifests found under: {path}")

    question_ids: list[str] = []
    seen: set[str] = set()
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid shard manifest: {manifest_path}") from exc
        raw_questions = manifest.get("questions") if isinstance(manifest, dict) else None
        if not isinstance(raw_questions, list):
            raise ValueError(f"shard manifest questions must be a list: {manifest_path}")
        for entry in raw_questions:
            question_id = str(entry.get("question_id", "")) if isinstance(entry, dict) else ""
            if not question_id:
                raise ValueError(f"invalid shard question entry: {manifest_path}")
            if question_id in seen:
                raise ValueError(f"duplicate question id across shard manifests: {question_id!r}")
            seen.add(question_id)
            question_ids.append(question_id)
    return tuple(question_ids)


def prewarm_embedding_cache(
    questions: Iterable[Question],
    provider: ContentHashEmbeddingCache,
    *,
    progress_every: int = 100,
    progress: Callable[[int], None] | None = None,
    dry_run: bool = False,
    changed_manifest: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Populate all unique ingest document units, skipping already-cached keys."""
    _reset_privacy_counts()
    if not isinstance(provider, ContentHashEmbeddingCache):
        raise ValueError(f"{EMBED_CACHE_ENV} must be set for prewarm-cache")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    chunk_embedding_mode = _resolved_chunk_embedding_mode(
        provider, enforce_cache_guard=False
    )
    if chunk_embedding_mode == "contextual":
        raise ValueError(
            "prewarm-cache populates flat chunk units; contextual mode is not cache-backed"
        )
    configured_chunk = _configured_chunk_binding(
        provider.provider_id, provider.model_id
    )
    if configured_chunk != (provider.provider_id, provider.model_id):
        raise ValueError(
            "prewarm-cache is unsupported when production summary and chunk "
            "embedding identities differ; use the run path so each unit is cached "
            "under its resolved provider/model pair"
        )
    privacy_config, privacy_revision = _embedding_privacy_context(
        provider.provider_id, provider.model_id
    )
    if privacy_revision is not None:
        from hermes_lcm.ingest_protection import (
            EmbeddingPrivacyPolicyError,
            validate_embedding_privacy_dispatch,
        )

    seen: set[str] = set()
    batch: list[str] = []
    already_cached = 0
    processed = 0
    would_populate = 0
    changed_units = 0
    changed_manifest_path = (
        Path(changed_manifest) if changed_manifest is not None else None
    )
    changed_manifest_file = None
    if changed_manifest_path is not None:
        changed_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        changed_manifest_file = changed_manifest_path.open("w", encoding="utf-8")

    def flush_batch() -> None:
        nonlocal already_cached, processed, would_populate
        if not batch:
            return
        cached = provider.cached_count(batch)
        already_cached += cached
        if privacy_revision is not None:
            try:
                validate_embedding_privacy_dispatch(
                    batch,
                    privacy_config,
                    expected_revision=privacy_revision,
                )
            except EmbeddingPrivacyPolicyError as exc:
                raise _record_validator_block(
                    exc, kind="document", count=len(batch)
                )
        if dry_run:
            would_populate += len(batch) - cached
        else:
            provider.embed_documents(batch)
        processed += len(batch)
        if progress is not None:
            progress(processed)
        batch.clear()

    try:
        for question in questions:
            for unit_index, raw_text in enumerate(
                iter_ingest_embedding_request_units(question)
            ):
                text = raw_text
                if privacy_revision is not None:
                    text = _protected(
                        raw_text,
                        privacy_config,
                        expected_revision=privacy_revision,
                    )
                if text != raw_text:
                    changed_units += 1
                    if changed_manifest_file is not None:
                        changed_manifest_file.write(
                            json.dumps(
                                {
                                    "question_id": question.question_id,
                                    "unit_index": unit_index,
                                    "raw_sha256": provider.content_sha256(raw_text),
                                    "protected_sha256": provider.content_sha256(text),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                digest = provider.content_sha256(text)
                if digest in seen:
                    continue
                seen.add(digest)
                batch.append(text)
                if len(batch) == progress_every:
                    flush_batch()
        flush_batch()
    finally:
        if changed_manifest_file is not None:
            changed_manifest_file.close()
    return {
        "provider": provider.provider_id,
        "model": provider.model_id,
        "cache_path": str(provider.cache_path),
        "unique_request_units": len(seen),
        "already_cached": already_cached,
        "populated": 0 if dry_run else len(seen) - already_cached,
        "dry_run": dry_run,
        "would_populate": would_populate,
        "chunk_embedding_mode": chunk_embedding_mode,
        "changed_manifest": (
            str(changed_manifest_path) if changed_manifest_path is not None else None
        ),
        "changed_units": changed_units,
        "privacy": dict(_PRIVACY_COUNTS),
        "privacy_scope": "corpus",
    }


def embedding_determinism_report(
    questions: Iterable[Question],
    provider,
    *,
    sample_size: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Embed random unique session summaries twice and compare float bits."""
    _reset_privacy_counts()
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    chunk_embedding_mode = _resolved_chunk_embedding_mode(provider)
    privacy_config, privacy_revision = _embedding_privacy_context(
        str(getattr(provider, "provider_id", "")),
        str(getattr(provider, "model_id", "")),
    )
    before_dispatch = None
    if privacy_revision is not None:
        from hermes_lcm.ingest_protection import (
            EmbeddingPrivacyPolicyError,
            validate_embedding_privacy_dispatch,
        )

        def before_dispatch(texts: Sequence[str]) -> None:
            try:
                validate_embedding_privacy_dispatch(
                    texts,
                    privacy_config,
                    expected_revision=privacy_revision,
                )
            except EmbeddingPrivacyPolicyError as exc:
                raise _record_validator_block(
                    exc, kind="document", count=len(texts)
                )

    unique: dict[str, str] = {}
    unique_raw: dict[str, str] = {}
    for question in questions:
        for session in question.haystack_sessions:
            raw = deterministic_session_summary(session)
            text = raw
            if privacy_revision is not None:
                text = _protected(
                    raw,
                    privacy_config,
                    expected_revision=privacy_revision,
                    count=False,
                )
            digest = ContentHashEmbeddingCache.content_sha256(text)
            unique.setdefault(digest, text)
            unique_raw.setdefault(digest, raw)
    if len(unique) < sample_size:
        raise ValueError(
            f"determinism probe needs {sample_size} unique sessions; found {len(unique)}"
        )

    sampled_digests = random.Random(seed).sample(list(unique), sample_size)
    sampled_texts: list[str] = []
    for digest in sampled_digests:
        if privacy_revision is None:
            sampled_texts.append(unique[digest])
            continue
        protected = _protected(
            unique_raw[digest],
            privacy_config,
            expected_revision=privacy_revision,
            count=True,
        )
        if protected != unique[digest]:
            raise ValueError("determinism probe privacy protection is not deterministic")
        sampled_texts.append(protected)
    first = _embed_in_batches(
        provider, sampled_texts, before_dispatch=before_dispatch
    )
    second = _embed_in_batches(
        provider, sampled_texts, before_dispatch=before_dispatch
    )
    if len(first) != sample_size or len(second) != sample_size:
        raise ValueError("determinism probe provider returned the wrong vector count")

    identical = 0
    max_abs_diff = 0.0
    for left, right in zip(first, second):
        if len(left) != len(right):
            raise ValueError("determinism probe vector dimensions changed")
        left_bits = struct.pack(f"<{len(left)}d", *(float(value) for value in left))
        right_bits = struct.pack(f"<{len(right)}d", *(float(value) for value in right))
        if left_bits == right_bits:
            identical += 1
        for left_value, right_value in zip(left, right):
            difference = abs(float(left_value) - float(right_value))
            if not math.isfinite(difference):
                raise ValueError("determinism probe received a non-finite vector value")
            max_abs_diff = max(max_abs_diff, difference)
    return {
        "provider": str(getattr(provider, "provider_id", "")),
        "model": str(getattr(provider, "model_id", "")),
        "sample_size": sample_size,
        "seed": seed,
        "sample_content_sha256": sampled_digests,
        "bitwise_identical_count": identical,
        "non_identical_count": sample_size - identical,
        "max_abs_diff": max_abs_diff,
        "chunk_embedding_mode": chunk_embedding_mode,
        "privacy": dict(_PRIVACY_COUNTS),
        "privacy_scope": "sample",
    }


# --------------------------------------------------------------------------- #
# Metric math (pure, testable).
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Set recall@k: fraction of relevant items present in the top-k retrieved."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(dict.fromkeys(retrieved))[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance NDCG@k over a deduplicated ranked list."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = list(dict.fromkeys(retrieved))[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# Turn-level relevance is coverage-based: each retrieved item covers a
# ``(session, turn_index)`` range, and a hit is an item whose range intersects the
# labeled evidence turns. Precise items (raw-message FTS rows, raw chunks) cover a
# single turn ``(session, index)``; a summary item cannot localize a turn, so it is
# a session-granularity marker ``(session, None)`` that covers *every* evidence turn
# of its session at once. Callers surface that coarseness with an asterisk in output.
TurnKey = tuple[str, "int | None"]


def _evidence_turns_by_session(evidence_turns: Iterable[TurnKey]) -> dict[str, set[TurnKey]]:
    by_session: dict[str, set[TurnKey]] = {}
    for key in evidence_turns:
        by_session.setdefault(key[0], set()).add(key)
    return by_session


def turn_recall_at_k(turn_keys: Sequence[TurnKey], relevant: Iterable[TurnKey], k: int) -> float:
    """Coverage recall@k over turn keys.

    Top-k is a budget of ranked *items*. A precise ``(session, index)`` item covers
    itself; a ``(session, None)`` summary marker covers all of that session's
    evidence turns (session granularity). Returns covered-evidence / total-evidence.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    by_session = _evidence_turns_by_session(relevant_set)
    covered: set[TurnKey] = set()
    seen = 0
    for key in dict.fromkeys(turn_keys):
        if seen >= k:
            break
        seen += 1
        session, index = key
        if index is None:
            covered |= by_session.get(session, set())
        elif key in relevant_set:
            covered.add(key)
    return len(covered) / len(relevant_set)


def turn_ndcg_at_k(turn_keys: Sequence[TurnKey], relevant: Iterable[TurnKey], k: int) -> float:
    """Binary-relevance NDCG@k over a deduplicated ranked list of turn keys.

    An item is relevant if it is a labeled evidence turn, or a summary marker for a
    session that contains any evidence turn. IDCG assumes the relevant items in the
    list are ranked first.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    sessions_with_evidence = {key[0] for key in relevant_set}

    def is_relevant(key: TurnKey) -> bool:
        session, index = key
        if index is None:
            return session in sessions_with_evidence
        return key in relevant_set

    deduped = list(dict.fromkeys(turn_keys))
    top_k = deduped[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, key in enumerate(top_k, start=1) if is_relevant(key))
    ideal_hits = min(sum(1 for key in deduped if is_relevant(key)), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def percentiles(values: Sequence[float], points: Sequence[int] = (50, 90, 99)) -> dict[str, float]:
    """Nearest-rank percentiles for latency reporting."""
    ordered = sorted(values)
    result: dict[str, float] = {}
    for point in points:
        if not ordered:
            result[f"p{point}"] = 0.0
            continue
        rank = max(1, math.ceil(point / 100 * len(ordered)))
        result[f"p{point}"] = round(ordered[min(rank, len(ordered)) - 1], 3)
    return result


# --------------------------------------------------------------------------- #
# Retrieval arms.
# --------------------------------------------------------------------------- #


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _unit(values: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude == 0.0:
        return list(values)
    return [value / magnitude for value in values]


def _dedup_sessions(session_ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(session_ids))


# A LongMemEval question is a natural-language sentence; SQLite FTS5 MATCH ANDs
# its tokens, so a single non-matching word (e.g. "what") zeroes the arm. A
# standard lexical retrieval arm ORs the salient query terms (BM25-style), so we
# build the MATCH query from the repo's own term extractor minus light stopwords.
_FTS_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
        "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me",
        "my", "of", "on", "or", "that", "the", "to", "was", "were", "what",
        "when", "where", "which", "who", "why", "with", "you", "your",
    }
)


def build_fts_query(question: str) -> str:
    """Build an OR-of-terms FTS5 MATCH query from a natural-language question.

    Each term is reduced to a bareword (alphanumerics/underscore only) so no FTS5
    operator character survives to trip a syntax error and force the LIKE
    fallback; empty and stopword tokens are dropped.
    """
    _ensure_hermes_lcm_package()
    from hermes_lcm.search_query import extract_search_terms

    barewords: list[str] = []
    for term in extract_search_terms(question):
        cleaned = re.sub(r"\W+", "", term, flags=re.UNICODE)
        if cleaned and cleaned.lower() not in _FTS_STOPWORDS:
            barewords.append(cleaned)
    return " OR ".join(dict.fromkeys(barewords))


def fts_hits(store, query: str, fetch: int) -> list[tuple[str, int]]:
    """Raw FTS arm: ranked ``(session_id, store_id)`` message hits (no dedup).

    The ``store_id`` localizes each hit to a single turn for turn-level scoring;
    :func:`fts_sessions` collapses it to the session ranking.
    """
    match_query = build_fts_query(query)
    if not match_query:
        return []
    # build_fts_query composes FTS5 syntax on purpose (barewords joined by OR),
    # so it opts in to operator handling. Raw prose never does.
    rows = store.search(
        match_query, session_id=None, limit=fetch, allow_operators=True
    )
    hits: list[tuple[str, int]] = []
    for row in rows:
        store_id = row.get("store_id")
        if store_id is None:
            continue
        hits.append((str(row.get("session_id", "")), int(store_id)))
    return hits


def fts_sessions(store, query: str, fetch: int) -> list[str]:
    """Rank evidence sessions by the raw-message FTS arm (``store.search``)."""
    return _dedup_sessions(session for session, _ in fts_hits(store, query, fetch))


def vector_sessions(vector_store, dag, query_vec, model, provider, fetch: int) -> list[str]:
    """Rank evidence sessions by the summary-vector arm (``vector_store.knn``)."""
    result = vector_store.knn(query_vec, k=fetch, model=model, provider=provider)
    sessions: list[str] = []
    for embedded_id, _score, _kind in result:
        node = dag.get_node(int(embedded_id))
        if node is not None:
            sessions.append(str(node.session_id))
    return _dedup_sessions(sessions)


def _chunk_store_id(chunk_id) -> int | None:
    """Extract the source ``store_id`` from a ``store_id:chunk_index`` chunk id."""
    try:
        return int(str(chunk_id).split(":", 1)[0])
    except (TypeError, ValueError):
        return None


def chunk_hits(vector_store, query_vec, model, provider, fetch: int) -> list:
    """Raw chunk arm: the ranked ``knn_chunks`` result (``store_id:chunk_index`` ids)."""
    return list(vector_store.knn_chunks(query_vec, k=fetch, model=model, provider=provider))


def _map_chunk_sessions(result, store_id_to_session: dict[int, str]) -> list[str]:
    sessions: list[str] = []
    for chunk_id, _score, _kind in result:
        store_id = _chunk_store_id(chunk_id)
        session_id = None if store_id is None else store_id_to_session.get(store_id)
        if session_id is not None:
            sessions.append(str(session_id))
    return _dedup_sessions(sessions)


def chunk_sessions(
    vector_store, query_vec, model, provider, fetch: int,
    store_id_to_session: dict[int, str],
) -> list[str]:
    """Rank evidence sessions by the raw-chunk KNN arm (``knn_chunks``).

    Each chunk id is ``store_id:chunk_index``; its store_id maps back to the
    session that owns the source message, so a chunk hit votes for its session.
    """
    result = chunk_hits(vector_store, query_vec, model, provider, fetch)
    return _map_chunk_sessions(result, store_id_to_session)


# --------------------------------------------------------------------------- #
# Turn-key projections (parallel to the session rankings above).
# --------------------------------------------------------------------------- #


def fts_turn_keys(hits: Sequence[tuple[str, int]], store_id_to_turn: dict[int, TurnKey]) -> list[TurnKey]:
    """Project raw FTS message hits to precise ``(session, turn_index)`` keys."""
    keys: list[TurnKey] = []
    for _session, store_id in hits:
        key = store_id_to_turn.get(int(store_id))
        if key is not None:
            keys.append(key)
    return keys


def chunk_turn_keys(result, store_id_to_turn: dict[int, TurnKey]) -> list[TurnKey]:
    """Project raw chunk hits to precise ``(session, turn_index)`` keys via store_id."""
    keys: list[TurnKey] = []
    for chunk_id, _score, _kind in result:
        store_id = _chunk_store_id(chunk_id)
        key = None if store_id is None else store_id_to_turn.get(store_id)
        if key is not None:
            keys.append(key)
    return keys


def summary_turn_keys(session_ranked: Sequence[str]) -> list[TurnKey]:
    """A summary covers a whole session, so it localizes only to ``(session, None)``."""
    return [(session, None) for session in session_ranked]


# --------------------------------------------------------------------------- #
# Production ``lcm_recall`` arm — the tool users actually call.
# --------------------------------------------------------------------------- #

# A synthetic current-session id for the probe engine. lcm_recall applies a scope
# prior that BOOSTS hits belonging to the current conversation; benchmarking the
# production path honestly means the probe conversation must sit OUTSIDE the
# dataset's sessions so no evidence session is silently lifted by conversation
# membership. (The recency prior still applies to every hit — that is the honest
# production behavior, noted in benchmarks/README.md.)
_LCM_RECALL_FRESH_SESSION = "__lcm_recall_fresh_probe__"


def fresh_recall_session_id(question: "Question") -> str:
    """A current-session id guaranteed absent from ``question``'s haystack.

    Deterministic (sentinel + question id) so runs are reproducible, and disjoint
    from the haystack so the scope prior never boosts a dataset session.
    """
    existing = set(question.haystack_session_ids)
    candidate = f"{_LCM_RECALL_FRESH_SESSION}{question.question_id}"
    suffix = 0
    while candidate in existing:
        suffix += 1
        candidate = f"{_LCM_RECALL_FRESH_SESSION}{question.question_id}-{suffix}"
    return candidate


def recall_hit_sessions(hits: Sequence[dict[str, Any]]) -> list[str]:
    """Dedup production recall hits to a session ranking (hit order preserved)."""
    return _dedup_sessions(
        str(hit.get("session_id")) for hit in hits if hit.get("session_id")
    )


def recall_hit_turn_keys(
    hits: Sequence[dict[str, Any]], store_id_to_turn: dict[int, TurnKey]
) -> list[TurnKey]:
    """Project production recall hits to turn keys.

    A verbatim hit (``kind`` != ``summary``) carries a ``store_id`` that localizes
    it to one ``(session, turn_index)`` via the ingest map; a summary hit covers a
    whole session, so it becomes a ``(session, None)`` marker — the
    session-granularity credit the ``*`` asterisk warns about.
    """
    keys: list[TurnKey] = []
    for hit in hits:
        session_id = hit.get("session_id")
        if hit.get("kind") == "summary":
            if session_id:
                keys.append((str(session_id), None))
            continue
        store_id = hit.get("store_id")
        if store_id is None:
            continue
        key = store_id_to_turn.get(int(store_id))
        if key is not None:
            keys.append(key)
    return keys


def _typed_provider_degraded_outcomes(payload: dict[str, Any]) -> tuple[str, ...]:
    """Extract provider outcomes without treating general recall degradation as one."""
    if not payload.get("degraded"):
        return ()
    known_codes = {code for _prefix, code in _KNOWN_PROVIDER_DEGRADED_REASONS}
    explicit = payload.get("provider_degraded_outcomes")
    if explicit is not None:
        if not isinstance(explicit, list) or any(
            not isinstance(outcome, str) or outcome not in known_codes
            for outcome in explicit
        ):
            raise ValueError("lcm_recall returned unknown provider degraded outcomes")
        return tuple(dict.fromkeys(explicit))
    raw_reason = payload.get("degraded_reason")
    if not isinstance(raw_reason, str) or not raw_reason.strip():
        raise ValueError("lcm_recall degraded without a non-empty degraded_reason")
    outcomes: list[str] = []
    for reason in (part.strip() for part in raw_reason.split(";")):
        outcome = next(
            (
                code
                for prefix, code in _KNOWN_PROVIDER_DEGRADED_REASONS
                if reason == prefix or reason.startswith(prefix)
            ),
            None,
        )
        if outcome is None:
            # Aggregate recall degradation also contains general retrieval state
            # (bounded coverage, arm failures, and scope timeouts).  Accounting
            # ignores that prose, but still fails closed when an untyped reason
            # claims to describe provider/embedding behavior.
            if re.search(r"\b(provider|embedding|unsupported)\b", reason, re.I):
                raise ValueError(f"unknown provider degraded reason: {reason}")
            continue
        outcomes.append(outcome)
    return tuple(dict.fromkeys(outcomes))


def production_recall_hits(
    question: "Question",
    config,
    store,
    dag,
    provider_embedder,
    *,
    chunk_provider_embedder=None,
    provider_name: str,
    chunk_provider_name: str | None = None,
    tmp_dir: Path,
    embeddings_enabled: bool,
    limit: int,
    return_status: bool = False,
    accounting: ProviderAccounting | None = None,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str, list[list[Any]]]:
    """Invoke the REAL ``tools.lcm_recall`` against this question's temp store.

    This is the tool users call: weighted RRF over the FTS + summary + chunk arms
    (``retrieval_core.rrf_fuse`` with ``LCM_RECALL_ARM_WEIGHTS``), the scope/recency
    prior, and chunk-vs-FTS dedup by ``store_id`` — none of which the per-arm harness
    measurements exercise. The engine is the proven smoke-test stand-in (a
    ``SimpleNamespace`` exposing the already-open store/dag/config with a fresh
    current-session id). The warmed harness embedder is injected through
    ``lcm_recall``'s provider cache so no second model load or network call occurs;
    ``lcm_recall`` clamps ``limit`` to its own production ceiling.
    """
    _ensure_hermes_lcm_package()
    import hermes_lcm.tools as lcm_tools

    engine = SimpleNamespace(
        _config=config,
        _store=store,
        _dag=dag,
        _hermes_home=str(tmp_dir),
        current_session_id=fresh_recall_session_id(question),
    )
    if embeddings_enabled:
        cache_key = (
            str(provider_name).strip().lower(),
            str(provider_embedder.model_id).strip(),
        )
        engine._lcm_embedding_provider_cache = (cache_key, provider_embedder)
        chunk_provider_embedder = chunk_provider_embedder or provider_embedder
        chunk_cache_key = (
            str(chunk_provider_name or provider_name).strip().lower(),
            str(chunk_provider_embedder.model_id).strip(),
        )
        engine._lcm_chunk_provider_cache = (
            chunk_cache_key,
            chunk_provider_embedder,
        )
    payload = json.loads(
        lcm_tools.lcm_recall({"query": question.question, "limit": limit}, engine=engine)
    )
    if accounting is not None:
        degraded_outcomes = _typed_provider_degraded_outcomes(payload)
        accounting.record_degraded_outcomes(degraded_outcomes)
    hits = list(payload.get("hits", []))
    if not return_status:
        return hits
    status = payload.get("provenance", {}).get("rerank")
    if not isinstance(status, str) or not status:
        raise ValueError("lcm_recall response is missing provenance.rerank status")
    scores = payload.get("provenance", {}).get("rerank_scores", [])
    if not isinstance(scores, list):
        raise ValueError("lcm_recall response has invalid provenance.rerank_scores")
    return hits, status, scores


def _fuse_tiebreak(item):
    """Total-order tie-break for fused ids: plain session strings sort as-is;
    turn keys ``(session, turn|None)`` map a None turn (summary = whole
    session) to -1 so ties never compare ``None < int`` (crash observed when
    a summary turn key tied a localized turn key on score AND best rank)."""
    if isinstance(item, tuple):
        return tuple(-1 if part is None else part for part in item)
    return item


def rrf_fuse(*ranked_lists: Sequence[str]) -> list[str]:
    """Reciprocal-rank fusion over per-arm session rankings (``RRF_K`` = 60)."""
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in ranked_lists:
        for rank, session_id in enumerate(ranked, start=1):
            scores[session_id] = scores.get(session_id, 0.0) + 1.0 / (RRF_K + rank)
            best_rank[session_id] = min(best_rank.get(session_id, rank), rank)
    return sorted(scores, key=lambda sid: (-scores[sid], best_rank[sid], _fuse_tiebreak(sid)))


def rerank_by_cosine(
    sessions: Sequence[str], query_vec, session_vectors: dict[str, list[float]]
) -> list[str]:
    """Rerank fused candidates by cosine to the query embedding.

    This is a deterministic embedding-cosine reranker over the fused candidate
    pool, a placeholder for a real cross-encoder. Per the MemDelta caveat we
    only ever compare it against *this* configuration, never as a universal
    verdict.
    """
    normalized_query = _unit(list(query_vec))

    def score(session_id: str) -> float:
        vector = session_vectors.get(session_id)
        if vector is None:
            return -math.inf
        return _dot(normalized_query, vector)

    return sorted(sessions, key=lambda sid: (-score(sid), sid))


# The real rerank arm reranks a bounded candidate window (top fused sessions) in a
# single cross-encoder call under an absolute per-question budget; the rest of the
# fused ranking is appended unchanged. These bound live-provider cost/latency.
RERANK_CANDIDATE_WINDOW = 20
RERANK_TIMEOUT_S = 10.0
RERANK_MODE_PLACEHOLDER = "placeholder-cosine"
RERANK_MODE_VOYAGE = "voyage:rerank-2.5-lite"
# Run-level label when some (but not all) questions used the real reranker while
# others silently fell back -- the run is neither a clean real nor placeholder run.
RERANK_MODE_MIXED = "mixed"


def rerank_sessions_voyage(
    reranker,
    query: str,
    sessions: Sequence[str],
    session_summaries: dict[str, str],
    *,
    window: int = RERANK_CANDIDATE_WINDOW,
    timeout: float = RERANK_TIMEOUT_S,
    privacy_config=None,
    privacy_revision: str | None = None,
) -> list[str] | None:
    """Rerank the top-``window`` fused sessions with a real cross-encoder.

    Uses ``VoyageProvider.rerank`` (rerank-2.5-lite) over each candidate session's
    deterministic summary in one API call. Returns the reordered candidate window
    followed by the untouched fused tail, or ``None`` to signal the caller should
    fall back to the deterministic placeholder (empty window, any provider error,
    or an empty/degenerate response that does not cover every candidate).
    """
    candidates = list(sessions[:window])
    if not candidates:
        return None
    documents = [session_summaries.get(session, "") for session in candidates]
    if privacy_config is not None and privacy_revision is not None:
        # Production protects rerank payloads under the embedding-privacy
        # resolution (#371); the harness mirrors it — only the provider-bound
        # copies are transformed, ordering still applies to the originals.
        query = _protected(
            query,
            privacy_config,
            expected_revision=privacy_revision,
            kind="query",
        )
        documents = [
            _protected(
                doc,
                privacy_config,
                expected_revision=privacy_revision,
                kind="document",
            )
            for doc in documents
        ]
    try:
        ranked = reranker.rerank(query, documents, top_k=len(documents), timeout=timeout)
    except Exception:
        return None
    # A non-exception but empty/degenerate response (e.g. ``data: []`` or scores
    # covering only some candidates) is NOT a trustworthy real rerank -- treat it
    # exactly like a provider error so it is labeled a placeholder fallback rather
    # than silently counted as real voyage rerank.
    covered = {index for index, _score in ranked if 0 <= index < len(candidates)}
    if len(covered) != len(candidates):
        return None
    reordered = [candidates[index] for index, _score in ranked if 0 <= index < len(candidates)]
    seen = set(reordered)
    for session in sessions:
        if session not in seen:
            reordered.append(session)
            seen.add(session)
    return reordered


# --------------------------------------------------------------------------- #
# Per-question ingest + evaluation.
# --------------------------------------------------------------------------- #


@dataclass
class ArmSamples:
    recalls: dict[int, list[float]] = field(default_factory=lambda: {1: [], 5: [], 10: []})
    ndcg10: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    turn_recalls: dict[int, list[float]] = field(default_factory=lambda: {1: [], 5: [], 10: []})
    turn_ndcg10: list[float] = field(default_factory=list)
    # True when the arm's turn ranking includes summary (session-granularity) items,
    # so its turn-level numbers carry the coarse-localization asterisk.
    session_granularity: bool = False


def _new_arm_samples() -> dict[str, ArmSamples]:
    return {arm: ArmSamples() for arm in ARMS}


def _bootstrap_db_template(template_path: Path, config) -> None:
    """Create one fully-migrated empty LCM DB to clone per question.

    Opening ``MessageStore``/``SummaryDAG``/``VectorStore`` runs the schema
    bootstrap + FTS/migration DDL once; each subsequent question copies this file
    (idempotent re-open, no migrations) instead of paying that cost 500x.
    """
    _ensure_hermes_lcm_package()
    from hermes_lcm.dag import SummaryDAG
    from hermes_lcm.store import MessageStore
    from hermes_lcm.vector_store import VectorStore

    store = MessageStore(str(template_path), ingest_protection_config=config)
    dag = SummaryDAG(str(template_path))
    vector_store = VectorStore(str(template_path), config=config)
    vector_store.close()
    dag.close()
    store.close()


def evaluate_question(
    question: Question,
    provider_embedder,
    *,
    provider_name: str,
    tmp_dir: Path,
    embeddings_enabled: bool,
    top_k: int = 10,
    use_rerank: bool = False,
    recall_rerank: bool = False,
    recall_rerank_window: int = 0,
    recall_rerank_margin: float = 0.0,
    db_template: Path | None = None,
    embedding_batch_size: int | None = None,
    include_rankings: bool = False,
    chunk_provider=None,
    accounting: ProviderAccounting | None = None,
) -> dict[str, Any]:
    """Ingest one question into a fresh store and score every retrieval arm.

    Each arm reports session-level ``recall@1/5/10`` + ``ndcg@10`` + ``latency_ms``
    and a nested ``turn`` block with the same recall/NDCG at turn granularity plus a
    ``session_granularity`` flag. ``ingest_ms`` (per-question ingest wall time) and,
    for ``hybrid_rerank``, ``rerank_mode`` ride alongside for aggregation.
    """
    _PRIVACY_QUESTION.update({key: 0 for key in _PRIVACY_KEYS})
    _ensure_hermes_lcm_package()
    from hermes_lcm.chunking import group_by_store_id, iter_message_chunks
    from hermes_lcm.config import LCMConfig
    from hermes_lcm.dag import SummaryDAG, SummaryNode
    from hermes_lcm.embedding_provider import default_chunk_model
    from hermes_lcm.store import MessageStore
    from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

    db_path = tmp_dir / f"{_safe(question.question_id)}.db"
    if db_path.name == "_template.db":
        # The prepared path rejects this id at prepare time; guard the direct
        # --dataset path too, or the question DB would clobber the seeded template.
        raise ValueError(
            f"question_id {question.question_id!r} collides with the template database"
        )
    summary_provider = provider_embedder
    chunk_provider = chunk_provider or summary_provider
    cache_start_hits, cache_start_misses = _embed_cache_totals(
        (summary_provider, chunk_provider)
    )
    summary_name, model = _provider_binding(summary_provider, provider_name, "")
    chunk_name, chunk_model = _provider_binding(
        chunk_provider, summary_name, default_chunk_model(summary_name, model)
    )
    dim = int(summary_provider.dim)
    chunk_dim = int(chunk_provider.dim)
    summary_documents_provider = (
        _AccountingProvider(summary_provider, accounting, "summary_documents")
        if accounting is not None
        else summary_provider
    )
    chunk_documents_provider = (
        _AccountingProvider(chunk_provider, accounting, "chunk_documents")
        if accounting is not None
        else chunk_provider
    )
    harness_query_provider = (
        _AccountingProvider(summary_provider, accounting, "harness_queries")
        if accounting is not None
        else summary_provider
    )
    harness_chunk_query_provider = (
        _AccountingProvider(chunk_provider, accounting, "harness_queries")
        if accounting is not None
        else chunk_provider
    )
    production_summary_provider = (
        _AccountingProvider(
            summary_provider, accounting, "production_lcm_recall_queries"
        )
        if accounting is not None
        else summary_provider
    )
    production_chunk_provider = (
        _AccountingProvider(
            chunk_provider, accounting, "production_lcm_recall_queries"
        )
        if accounting is not None
        else chunk_provider
    )
    from hermes_lcm.ingest_protection import (
        EmbeddingPrivacyPolicyError,
        embedding_privacy_revision,
        embedding_provider_requires_privacy,
        validate_embedding_privacy_dispatch,
    )

    summary_requires_privacy = embedding_provider_requires_privacy(summary_name)
    chunk_requires_privacy = embedding_provider_requires_privacy(chunk_name)

    config = LCMConfig(
        database_path=str(db_path),
        embeddings_enabled=embeddings_enabled,
        embedding_provider=summary_name,
        embedding_model=model,
        rerank_enabled=recall_rerank,
        rerank_window_limit=recall_rerank_window,
        rerank_margin=recall_rerank_margin,
        # Keep the benchmark corpus lossless. Cloud privacy applies only to
        # provider-bound copies and resolves on automatically for cloud names.
        sensitive_patterns_enabled=False,
        embedding_privacy_enabled=None,
    )
    ingest_start = time.perf_counter()
    # F7: clone a pre-migrated template instead of re-running schema bootstrap.
    if db_template is not None and db_template.is_file():
        shutil.copyfile(db_template, db_path)
    store = MessageStore(str(db_path), ingest_protection_config=config)
    dag = SummaryDAG(str(db_path))
    vector_store = VectorStore(str(db_path), config=config)
    session_vectors: dict[str, list[float]] = {}
    session_summaries: dict[str, str] = {}
    # store_id -> owning session (chunk vote) and -> (session, turn) (turn scoring).
    store_id_to_session: dict[int, str] = {}
    store_id_to_turn: dict[int, TurnKey] = {}
    chunk_identity = None
    summary_revision: str | None = None
    chunk_revision: str | None = None
    # F7: collect every session summary, then embed the whole corpus in one batched
    # ``embed_documents`` call instead of one call per session. (Raw chunks stay
    # per-item: for local ONNX providers, batching pads every text to the batch's
    # longest, so per-chunk embedding is actually faster; the summary-call collapse
    # is the win that matters for network/live providers.)
    summary_specs: list[tuple[str, int, str]] = []  # (session_id, node_id, summary_text)
    chunk_items: list[Any] = []
    flat_chunk_batch: list[Any] = []
    chunk_embedding_mode = _resolved_chunk_embedding_mode(chunk_provider)
    supports_grouping = embeddings_enabled and chunk_embedding_mode == "contextual"
    flat_chunk_batch_size = max(1, int(embedding_batch_size or 1))
    try:
        if embeddings_enabled:
            # Production posture (#367/#370): cloud providers register their
            # vector identity under the active privacy revision, exactly as
            # the engine does — otherwise the recall path's identity check
            # fails loud on a revision mismatch.
            summary_revision = (
                embedding_privacy_revision(config)
                if summary_requires_privacy
                else None
            )
            chunk_revision = (
                embedding_privacy_revision(config)
                if chunk_requires_privacy
                else None
            )
            vector_store.register_profile(
                model, summary_name, dim, revision=summary_revision or ""
            )
            identity = vector_store.capture_identity(model, provider=summary_name)
            # The raw-chunk corpus is a distinct task='chunk' profile/identity.
            vector_store.register_profile(
                chunk_model, chunk_name, chunk_dim, task="chunk",
                revision=chunk_revision or "",
            )
            chunk_identity = EmbeddingIdentity.canonical(
                chunk_name,
                chunk_model,
                chunk_revision or "",
                chunk_dim,
                "float32",
                "little",
                "chunk",
            )

        def flush_flat_chunks() -> None:
            if not flat_chunk_batch:
                return
            chunk_texts = [str(chunk.text) for chunk in flat_chunk_batch]
            before_dispatch = None
            if chunk_revision is not None:
                chunk_texts = [
                    _protected(
                        text,
                        config,
                        expected_revision=chunk_revision,
                    )
                    for text in chunk_texts
                ]

                def before_dispatch(texts: Sequence[str]) -> None:
                    try:
                        validate_embedding_privacy_dispatch(
                            texts,
                            config,
                            expected_revision=chunk_revision,
                        )
                    except EmbeddingPrivacyPolicyError as exc:
                        raise _record_validator_block(
                            exc, kind="document", count=len(texts)
                        )

            vectors = _embed_in_batches(
                chunk_documents_provider,
                chunk_texts,
                batch_size=flat_chunk_batch_size,
                before_dispatch=before_dispatch,
            )
            for chunk, vector in zip(flat_chunk_batch, vectors):
                vector_store.record_chunk_embedding(
                    chunk.chunk_id,
                    chunk_model,
                    vector,
                    store_id=chunk.store_id,
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_estimate=chunk.token_estimate,
                    identity=chunk_identity,
                )
            flat_chunk_batch.clear()

        for order, (session_id, session) in enumerate(
            zip(question.haystack_session_ids, question.haystack_sessions), start=1
        ):
            messages = [
                {
                    "role": str(turn.get("role", "user")) if isinstance(turn, dict) else "user",
                    "content": turn.get("content", "") if isinstance(turn, dict) else str(turn),
                }
                for turn in session
            ]
            if messages:
                store_ids = store.append_batch(
                    session_id, messages, source="benchmark", conversation_id=session_id
                )
                # Message i is turn index i (1:1 with the haystack turns), so its
                # store_id resolves to (session, turn) for turn-level scoring.
                for turn_index, store_id in enumerate(store_ids):
                    store_id_to_turn[int(store_id)] = (session_id, turn_index)
                if embeddings_enabled and chunk_identity is not None:
                    for store_id in store_ids:
                        store_id_to_session[int(store_id)] = session_id
                    rows = [
                        {"store_id": sid, "role": m["role"], "content": m["content"]}
                        for sid, m in zip(store_ids, messages)
                    ]
                    # Keep production's contiguous per-message grouping for
                    # contextualized providers. Stream the local/non-context
                    # fallback in bounded batches so it never retains the full
                    # question corpus or forces one longest-text padded batch.
                    for chunk in iter_message_chunks(rows, policy="conversational"):
                        if supports_grouping:
                            chunk_items.append(chunk)
                        else:
                            flat_chunk_batch.append(chunk)
                            if len(flat_chunk_batch) >= flat_chunk_batch_size:
                                flush_flat_chunks()
            summary_text = deterministic_session_summary(session)
            session_summaries[session_id] = summary_text
            node_id = dag.add_node(
                SummaryNode(
                    session_id=session_id,
                    depth=0,
                    summary=summary_text,
                    token_count=len(summary_text.split()),
                    source_token_count=sum(len(m["content"].split()) for m in messages),
                    source_type="messages",
                    created_at=float(order),
                )
            )
            summary_specs.append((session_id, node_id, summary_text))

        if embeddings_enabled and not supports_grouping:
            flush_flat_chunks()

        if embeddings_enabled and supports_grouping and chunk_items:
            chunk_texts = [str(chunk.text) for chunk in chunk_items]
            contextual_before_dispatch = None
            if chunk_revision is not None:
                chunk_texts = [
                    _protected(
                        text,
                        config,
                        expected_revision=chunk_revision,
                    )
                    for text in chunk_texts
                ]

                def contextual_before_dispatch(indexes: Sequence[int]) -> None:
                    try:
                        validate_embedding_privacy_dispatch(
                            [chunk_texts[int(index)] for index in indexes],
                            config,
                            expected_revision=chunk_revision,
                        )
                    except EmbeddingPrivacyPolicyError as exc:
                        raise _record_validator_block(
                            exc, kind="document", count=len(indexes)
                        )

            groups = [
                [(index, chunk_texts[index]) for index in group]
                for group in group_by_store_id(
                    [int(chunk.store_id) for chunk in chunk_items]
                )
            ]
            chunk_batches = chunk_documents_provider.embed_chunk_group_batches(
                groups,
                before_dispatch=contextual_before_dispatch,
            )
            chunk_vectors_by_index: dict[int, Sequence[float]] = {}
            for batch in chunk_batches:
                indexes = getattr(batch, "indexes", None)
                vectors = getattr(batch, "vectors", None)
                if indexes is None or vectors is None:
                    try:
                        indexes, vectors = batch
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "contextual chunk provider returned malformed batch"
                        ) from exc
                if len(indexes) != len(vectors):
                    raise ValueError(
                        "contextual chunk provider returned mismatched indexes/vectors"
                    )
                chunk_vectors_by_index.update(
                    {int(index): vector for index, vector in zip(indexes, vectors)}
                )
            if len(chunk_vectors_by_index) != len(chunk_items):
                raise ValueError("contextual chunk provider returned incomplete vectors")
            for index in range(len(chunk_items)):
                chunk = chunk_items[index]
                vector_store.record_chunk_embedding(
                    chunk.chunk_id,
                    chunk_model,
                    chunk_vectors_by_index[index],
                    store_id=chunk.store_id,
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_estimate=chunk.token_estimate,
                    identity=chunk_identity,
                )

        if embeddings_enabled and summary_specs:
            summary_texts = [text for _session, _node, text in summary_specs]
            summary_before_dispatch = None
            if summary_revision is not None:
                summary_texts = [
                    _protected(
                        text,
                        config,
                        expected_revision=summary_revision,
                    )
                    for text in summary_texts
                ]

                def summary_before_dispatch(texts: Sequence[str]) -> None:
                    try:
                        validate_embedding_privacy_dispatch(
                            texts,
                            config,
                            expected_revision=summary_revision,
                        )
                    except EmbeddingPrivacyPolicyError as exc:
                        raise _record_validator_block(
                            exc, kind="document", count=len(texts)
                        )

            summary_vectors = _embed_in_batches(
                summary_documents_provider,
                summary_texts,
                batch_size=embedding_batch_size,
                before_dispatch=summary_before_dispatch,
            )
            for (session_id, node_id, _text), vector in zip(summary_specs, summary_vectors):
                vector_store.record_embedding(
                    str(node_id), "summary", model, vector, identity=identity
                )
                session_vectors[session_id] = _unit(list(vector))
        ingest_ms = (time.perf_counter() - ingest_start) * 1000.0

        relevant = evidence_sessions(question)
        relevant_turns = evidence_turns(question)
        summary_query = str(question.question)
        if summary_revision is not None:
            summary_query = _protected(
                summary_query,
                config,
                expected_revision=summary_revision,
                kind="query",
            )
            try:
                validate_embedding_privacy_dispatch(
                    [summary_query],
                    config,
                    expected_revision=summary_revision,
                )
            except EmbeddingPrivacyPolicyError as exc:
                raise _record_validator_block(exc, kind="query", count=1)
        query_vec = (
            harness_query_provider.embed_query(summary_query)
            if embeddings_enabled
            else None
        )
        fetch = max(top_k * 5, 50)

        # Session rankings + parallel turn-key projections for every arm.
        fts_raw, fts_ms = _timed(lambda: fts_hits(store, question.question, fetch))
        fts_ranked = _dedup_sessions(session for session, _ in fts_raw)
        fts_turns = fts_turn_keys(fts_raw, store_id_to_turn)

        if embeddings_enabled:
            vector_ranked, vector_ms = _timed(
                lambda: vector_sessions(
                    vector_store, dag, query_vec, model, summary_name, fetch
                )
            )
        else:
            vector_ranked, vector_ms = [], 0.0
        summary_turns = summary_turn_keys(vector_ranked)

        hybrid_ranked, hybrid_ms = _timed(lambda: rrf_fuse(fts_ranked, vector_ranked))
        # C6: a hybrid arm's turn keys are session-granularity markers projected from
        # its fused SESSION ranking, NOT an RRF over the raw per-arm turn-key lists.
        # Fusing precise (fts/chunk) and coarse (summary) keys in one ranked list let
        # non-evidence precise keys consume the fixed top-k coverage budget ahead of
        # the summary markers of high-ranked evidence sessions, cratering turn recall
        # below every input arm (measured 25q: rrf3 tR@5 0.40 vs chunk 0.62 / summary
        # 0.76; session-marker projection restores it to 0.88). The fused session
        # ranking is the arm's trustworthy signal, so its turn localization is honestly
        # session-granularity (carried by the ``*`` asterisk), same as summary_vectors.
        hybrid_turns = summary_turn_keys(hybrid_ranked)

        rerank_mode = RERANK_MODE_PLACEHOLDER
        rerank_start = time.perf_counter()
        rerank_ranked: list[str]
        if (
            use_rerank
            and embeddings_enabled
            and provider_name == "voyage"
            and hasattr(provider_embedder, "rerank")
        ):
            real = rerank_sessions_voyage(
                provider_embedder, question.question, hybrid_ranked, session_summaries,
                privacy_config=config if summary_revision is not None else None,
                privacy_revision=summary_revision,
            )
            if real is not None:
                rerank_ranked = real
                rerank_mode = RERANK_MODE_VOYAGE
            else:
                rerank_ranked = rerank_by_cosine(hybrid_ranked, query_vec, session_vectors)
        elif embeddings_enabled:
            rerank_ranked = rerank_by_cosine(hybrid_ranked, query_vec, session_vectors)
        else:
            rerank_ranked = list(hybrid_ranked)
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0
        # C6: session-granularity markers follow the reranked session order.
        rerank_turns = summary_turn_keys(rerank_ranked)

        if embeddings_enabled:
            if (summary_name, model) == (chunk_name, chunk_model):
                chunk_query_vec = query_vec
            else:
                chunk_query = str(question.question)
                if chunk_revision is not None:
                    chunk_query = _protected(
                        chunk_query,
                        config,
                        expected_revision=chunk_revision,
                        kind="query",
                    )
                    try:
                        validate_embedding_privacy_dispatch(
                            [chunk_query],
                            config,
                            expected_revision=chunk_revision,
                        )
                    except EmbeddingPrivacyPolicyError as exc:
                        raise _record_validator_block(exc, kind="query", count=1)
                chunk_query_vec = harness_chunk_query_provider.embed_query(chunk_query)
            chunk_raw, chunk_ms = _timed(
                lambda: chunk_hits(
                    vector_store,
                    chunk_query_vec,
                    chunk_model,
                    chunk_name,
                    fetch,
                )
            )
        else:
            chunk_query_vec = None
            chunk_raw, chunk_ms = [], 0.0
        chunk_ranked = _map_chunk_sessions(chunk_raw, store_id_to_session)
        chunk_turns = chunk_turn_keys(chunk_raw, store_id_to_turn)

        hybrid_rrf3_ranked, rrf3_ms = _timed(
            lambda: rrf_fuse(fts_ranked, vector_ranked, chunk_ranked)
        )
        # C6: session-granularity markers projected from the fused 3-arm ranking.
        rrf3_turns = summary_turn_keys(hybrid_rrf3_ranked)

        # The production tool: fetch drives lcm_recall's own limit (clamped to its
        # 25-hit ceiling). Its hits carry session_id directly (session ranking) and
        # store_id/node_id for turn projection. Its turn keys mix precise verbatim
        # keys with (session, None) summary markers, so it carries the asterisk.
        recall_result, recall_ms = _timed(
            lambda: production_recall_hits(
                question,
                config,
                store,
                dag,
                production_summary_provider,
                chunk_provider_embedder=production_chunk_provider,
                provider_name=summary_name,
                chunk_provider_name=chunk_name,
                tmp_dir=tmp_dir,
                embeddings_enabled=embeddings_enabled, limit=fetch,
                return_status=recall_rerank,
                accounting=accounting,
            )
        )
        if recall_rerank:
            recall_raw, recall_rerank_status, recall_rerank_scores = recall_result
        else:
            recall_raw = recall_result
        recall_ranked = recall_hit_sessions(recall_raw)
        recall_turns = recall_hit_turn_keys(recall_raw, store_id_to_turn)

        # arm -> (session ranking, latency, turn keys, session_granularity asterisk).
        ranked_by_arm: dict[str, tuple[list[str], float, list[TurnKey], bool]] = {
            "fts": (fts_ranked, fts_ms, fts_turns, False),
            "summary_vectors": (vector_ranked, vector_ms, summary_turns, True),
            "hybrid_rrf": (hybrid_ranked, hybrid_ms, hybrid_turns, True),
            "hybrid_rerank": (rerank_ranked, rerank_ms, rerank_turns, True),
            "chunk_vectors": (chunk_ranked, chunk_ms, chunk_turns, False),
            "hybrid_rrf3": (hybrid_rrf3_ranked, rrf3_ms, rrf3_turns, True),
            "lcm_recall": (recall_ranked, recall_ms, recall_turns, True),
        }
        scored: dict[str, Any] = {"ingest_ms": ingest_ms}
        for arm, (ranked, elapsed_ms, turn_keys, session_granularity) in ranked_by_arm.items():
            scored[arm] = {
                "recall@1": recall_at_k(ranked, relevant, 1),
                "recall@5": recall_at_k(ranked, relevant, 5),
                "recall@10": recall_at_k(ranked, relevant, 10),
                "ndcg@10": ndcg_at_k(ranked, relevant, 10),
                "latency_ms": elapsed_ms,
                "turn": {
                    "recall@1": turn_recall_at_k(turn_keys, relevant_turns, 1),
                    "recall@5": turn_recall_at_k(turn_keys, relevant_turns, 5),
                    "recall@10": turn_recall_at_k(turn_keys, relevant_turns, 10),
                    "ndcg@10": turn_ndcg_at_k(turn_keys, relevant_turns, 10),
                    "session_granularity": session_granularity,
                },
            }
        scored["hybrid_rerank"]["rerank_mode"] = rerank_mode
        if recall_rerank:
            scored["lcm_recall"]["recall_rerank_status"] = recall_rerank_status
        if include_rankings:
            candidate_rankings = {
                arm: {
                    "sessions": list(dict.fromkeys(ranked))[:_CANDIDATE_DUMP_TOP_K],
                    "turns": list(dict.fromkeys(turn_keys))[:_CANDIDATE_DUMP_TOP_K],
                }
                for arm, (ranked, _elapsed_ms, turn_keys, _session_granularity)
                in ranked_by_arm.items()
            }
            if recall_rerank:
                candidate_rankings["lcm_recall"]["rerank_scores"] = list(
                    recall_rerank_scores
                )
            scored["_candidate_rankings"] = candidate_rankings
        store_connection = store.connection
        dag_connection = dag.connection
        messages_count = int(
            store_connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        ) if store_connection is not None else 0
        summary_nodes_count = int(
            dag_connection.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0]
        ) if dag_connection is not None else 0
        chunks_count = 0
        vector_connection = vector_store.connection
        if vector_connection is not None:
            try:
                chunks_count = int(
                    vector_connection.execute(
                        "SELECT COUNT(*) FROM lcm_chunk_meta"
                    ).fetchone()[0]
                )
            except sqlite3.OperationalError as exc:
                # Chunk tables are lazy and absent when the chunk arm is disabled
                # or no chunk rows were emitted for this question.
                if "no such table" not in str(exc).lower():
                    raise
                chunks_count = 0
        scored["chunk_embedding_mode"] = chunk_embedding_mode
        scored["privacy"] = dict(_PRIVACY_QUESTION)
        scored["corpus_counts"] = {
            "messages": messages_count,
            "summary_nodes": summary_nodes_count,
            "chunks": chunks_count,
        }
        cache_end_hits, cache_end_misses = _embed_cache_totals(
            (summary_provider, chunk_provider)
        )
        scored["embed_cache"] = {
            "hits": cache_end_hits - cache_start_hits,
            "misses": cache_end_misses - cache_start_misses,
        }
        return scored
    finally:
        vector_store.close()
        dag.close()
        store.close()


def _embedding_batch_size() -> int:
    raw = os.environ.get("LCM_EMBEDDING_MAX_BATCH_ITEMS")
    if raw is None:
        return EMBED_BATCH_SIZE
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError(
            "LCM_EMBEDDING_MAX_BATCH_ITEMS must be an integer; must match ^[0-9]+$"
        )
    value = int(raw)
    if value <= 0:
        raise ValueError("LCM_EMBEDDING_MAX_BATCH_ITEMS must be positive")
    return value


def _embed_in_batches(
    embedder,
    texts: Sequence[str],
    batch_size: int | None = None,
    *,
    before_dispatch=None,
) -> list:
    """Embed ``texts`` in ``batch_size`` sub-batches, concatenating the results.

    One ``embed_documents`` call per sub-batch (F7 amortization) while each call
    stays inside the provider's per-call deadline. Per-text vectors are identical to
    embedding one text at a time for the deterministic/independent providers used here.
    """
    if batch_size is None:
        batch_size = _embedding_batch_size()
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    vectors: list = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        if before_dispatch is not None:
            before_dispatch(batch)
        embedded = list(embedder.embed_documents(batch))
        if len(embedded) != len(batch):
            raise ValueError(
                f"embedding provider returned {len(embedded)} vectors for {len(batch)} texts"
            )
        vectors.extend(embedded)
    return vectors


def _timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._") or "question"


def _cleanup_question_db(tmp_dir: Path, question_id: str) -> None:
    """Remove one question's SQLite database and sidecar files after scoring."""
    db_path = Path(tmp_dir) / f"{_safe(question_id)}.db"
    if db_path.name == "_template.db":
        return
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            _LOGGER.warning("could not delete LongMemEval question database %s: %s", path, exc)


# --------------------------------------------------------------------------- #
# Aggregation + report.
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _aggregate_rerank_mode(mode_counts: dict[str, int]) -> dict[str, Any]:
    """Collapse per-question rerank modes into one auditable run-level label.

    A run is labeled ``real`` (voyage) only if EVERY scored rerank-arm question
    used the real reranker; if any question silently fell back to the placeholder
    the run is ``mixed`` (never mislabeled as real); if none used voyage it is
    ``placeholder``. Per-mode counts ride alongside so the label is verifiable
    against the run rather than reflecting only the final question.
    """
    voyage = mode_counts.get(RERANK_MODE_VOYAGE, 0)
    placeholder = mode_counts.get(RERANK_MODE_PLACEHOLDER, 0)
    total = sum(mode_counts.values())
    if total == 0 or voyage == 0:
        mode = RERANK_MODE_PLACEHOLDER
    elif voyage == total:
        mode = RERANK_MODE_VOYAGE
    else:
        mode = RERANK_MODE_MIXED
    return {
        "mode": mode,
        "real_count": voyage,
        "placeholder_count": placeholder,
        "counts": dict(mode_counts),
    }


def _checkpoint_header(
    *,
    provider: str,
    model: str,
    rerank: bool,
    recall_rerank: bool = False,
    recall_rerank_window: int = 0,
    recall_rerank_margin: float = 0.0,
    embeddings_enabled: bool,
    dataset_label: str,
    direct_source_sha256: str | None,
    manifest_sha256: str | None,
    reuse_db_template: bool,
    embedding_batch_size: int | None,
) -> dict[str, Any]:
    chunk_provider, chunk_model = _configured_chunk_binding(provider, model)
    _privacy_config, privacy_revision = _embedding_privacy_context(
        provider, model, embeddings_enabled=embeddings_enabled
    )
    bindings: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "chunk_provider": chunk_provider,
        "chunk_model": chunk_model,
        "rerank": rerank,
        "recall_rerank": recall_rerank,
        "recall_rerank_window": recall_rerank_window,
        "embeddings_enabled": embeddings_enabled,
        "embedding_privacy_revision": privacy_revision,
        "dataset_label": dataset_label,
        "reuse_db_template": reuse_db_template,
        "embedding_batch_size": embedding_batch_size,
    }
    if direct_source_sha256 is not None:
        bindings["source_sha256"] = direct_source_sha256
    if manifest_sha256 is not None:
        bindings["manifest_sha256"] = manifest_sha256
    if recall_rerank or recall_rerank_margin != 0.0:
        bindings["recall_rerank_margin"] = recall_rerank_margin
    return {_CHECKPOINT_HEADER_KEY: bindings}


def _candidate_dump_header(
    *,
    provider: str,
    model: str,
    rerank: bool,
    recall_rerank: bool = False,
    recall_rerank_window: int = 0,
    recall_rerank_margin: float = 0.0,
    embeddings_enabled: bool,
    dataset_label: str,
    direct_source_sha256: str | None,
    manifest_sha256: str | None,
    top_k: int = _CANDIDATE_DUMP_TOP_K,
) -> dict[str, Any]:
    # Bind BOTH corpus digests, mirroring the checkpoint header: a direct
    # --dataset run records source_sha256 (manifest is None) and a prepared-dir
    # run records manifest_sha256, so an existing dump from a different corpus
    # under the same label fails validation instead of being appended to.
    chunk_provider, chunk_model = _configured_chunk_binding(provider, model)
    _privacy_config, privacy_revision = _embedding_privacy_context(
        provider, model, embeddings_enabled=embeddings_enabled
    )
    bindings: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "chunk_provider": chunk_provider,
        "chunk_model": chunk_model,
        "dataset_label": dataset_label,
        "source_sha256": direct_source_sha256,
        "manifest_sha256": manifest_sha256,
        "embeddings_enabled": embeddings_enabled,
        "embedding_privacy_revision": privacy_revision,
        "rerank": rerank,
        "recall_rerank": recall_rerank,
        "recall_rerank_window": recall_rerank_window,
        "top_k": top_k,
    }
    if recall_rerank or recall_rerank_margin != 0.0:
        bindings["recall_rerank_margin"] = recall_rerank_margin
    return {_DUMP_HEADER_KEY: bindings}


def _validate_candidate_dump_header(
    record: Any, *, expected_header: dict[str, Any], path: Path
) -> None:
    if record != expected_header:
        actual_revision = (
            record.get(_DUMP_HEADER_KEY, {}).get("embedding_privacy_revision")
            if isinstance(record, dict)
            and isinstance(record.get(_DUMP_HEADER_KEY), dict)
            else None
        )
        expected_revision = expected_header[_DUMP_HEADER_KEY].get(
            "embedding_privacy_revision"
        )
        if actual_revision != expected_revision:
            raise ValueError(
                "candidate dump privacy revision mismatch for "
                f"{path}: artifact={actual_revision!r}, current={expected_revision!r}"
            )
        raise ValueError(f"candidate dump configuration mismatch for {path}")


def _candidate_dump_record(
    question: Question,
    rankings: dict[str, Any] | None,
    *,
    recall_rerank: bool = False,
) -> dict[str, Any]:
    def _turn_sort_key(turn_key: TurnKey) -> tuple[str, int]:
        return (str(turn_key[0]), -1 if turn_key[1] is None else int(turn_key[1]))

    arms: dict[str, Any] = {}
    if not question.is_abstention:
        # Fail loud on incomplete rankings: a missing arm or a malformed arm
        # entry must never degrade into a valid-looking row with empty lists.
        # (Genuinely empty rankings are legal — an arm may return no candidates.)
        provided = rankings or {}
        if set(provided) != set(ARMS):
            raise RuntimeError(
                "candidate dump rankings are incomplete for question "
                f"{question.question_id!r}: expected arms {sorted(ARMS)}, "
                f"got {sorted(provided)}"
            )
        for arm in ARMS:
            ranking = provided[arm]
            if "sessions" not in ranking or "turns" not in ranking:
                raise RuntimeError(
                    "candidate dump rankings are malformed for question "
                    f"{question.question_id!r} arm {arm!r}: missing "
                    "sessions/turns"
                )
            session_ids = list(dict.fromkeys(ranking["sessions"]))[:_CANDIDATE_DUMP_TOP_K]
            turn_keys = list(dict.fromkeys(ranking["turns"]))[:_CANDIDATE_DUMP_TOP_K]
            arms[arm] = {
                "sessions_top10": session_ids,
                "turns_top10": [list(turn_key) for turn_key in turn_keys],
            }
            if recall_rerank and arm == "lcm_recall" and "rerank_scores" in ranking:
                arms[arm]["rerank_scores"] = list(ranking["rerank_scores"])

    return {
        "question_id": question.question_id,
        "category": question.category,
        "abstention": question.is_abstention,
        "gold_sessions": sorted(evidence_sessions(question)),
        "gold_turns": [
            list(turn_key)
            for turn_key in sorted(evidence_turns(question), key=_turn_sort_key)
        ],
        "arms": arms,
    }


def _write_candidate_dump_record(dump_file, record: dict[str, Any]) -> None:
    dump_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    dump_file.flush()
    os.fsync(dump_file.fileno())


def _validate_checkpoint_header(
    record: Any, *, expected_header: dict[str, Any], path: Path
) -> None:
    if not isinstance(record, dict) or _CHECKPOINT_HEADER_KEY not in record:
        raise ValueError(
            f"checkpoint is missing required {_CHECKPOINT_HEADER_KEY}; "
            f"old-format checkpoints cannot be resumed: {path}"
        )
    if set(record) != {_CHECKPOINT_HEADER_KEY} or not isinstance(
        record[_CHECKPOINT_HEADER_KEY], dict
    ):
        raise ValueError(f"invalid checkpoint header record: {path}")

    actual = record[_CHECKPOINT_HEADER_KEY]
    expected = expected_header[_CHECKPOINT_HEADER_KEY]
    mismatches = [
        f"{field}: checkpoint={actual.get(field)!r}, current={expected.get(field)!r}"
        for field in sorted(set(actual) | set(expected))
        if field not in actual or field not in expected or actual[field] != expected[field]
    ]
    if mismatches:
        raise ValueError(
            f"checkpoint configuration mismatch for {path}: " + "; ".join(mismatches)
        )


def _load_question_checkpoint(
    path: Path,
    *,
    selected_question_ids: set[str],
    expected_header: dict[str, Any],
) -> list[dict[str, Any]]:
    """Load a checkpoint, truncating only a malformed final crash-torn line."""
    payload = path.read_bytes()
    recall_rerank = bool(expected_header[_CHECKPOINT_HEADER_KEY].get("recall_rerank", False))
    lines = payload.splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    offset = 0
    dropped_torn_line = False
    header_seen = False
    for index, raw_line in enumerate(lines):
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index != len(lines) - 1 or raw_line.endswith(b"\n"):
                raise ValueError(
                    f"invalid checkpoint JSON at line {index + 1}: {path}"
                ) from exc
            _LOGGER.warning(
                "dropping torn final checkpoint line %d from %s; its question will be rerun",
                index + 1,
                path,
            )
            with path.open("r+b") as checkpoint_file:
                checkpoint_file.truncate(offset)
                checkpoint_file.flush()
                os.fsync(checkpoint_file.fileno())
            dropped_torn_line = True
            break
        if index == 0:
            _validate_checkpoint_header(record, expected_header=expected_header, path=path)
            header_seen = True
            offset += len(raw_line)
            continue
        if not isinstance(record, dict):
            raise ValueError(f"checkpoint line {index + 1} must be a JSON object: {path}")
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"checkpoint line {index + 1} has no question_id: {path}")
        if question_id not in selected_question_ids:
            raise ValueError(
                f"checkpoint question_id {question_id!r} is not in the selected question set; "
                f"refusing wrong output directory: {path}"
            )
        if question_id in seen:
            raise ValueError(f"duplicate checkpoint question_id {question_id!r}: {path}")
        _validate_restored_checkpoint_metrics(
            record, line_number=index + 1, path=path, recall_rerank=recall_rerank
        )
        seen.add(question_id)
        records.append(record)
        offset += len(raw_line)
    if not header_seen:
        raise ValueError(
            f"checkpoint is missing required {_CHECKPOINT_HEADER_KEY}; "
            f"cannot resume: {path}"
        )
    if payload and not dropped_torn_line and not payload.endswith(b"\n"):
        # A crash may land after the JSON object but before its newline. Preserve
        # the valid record and restore the separator before later appends.
        with path.open("ab") as checkpoint_file:
            checkpoint_file.write(b"\n")
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
    return records


def _write_checkpoint_record(checkpoint_file, record: dict[str, Any]) -> None:
    checkpoint_file.write(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    )
    checkpoint_file.flush()
    os.fsync(checkpoint_file.fileno())


def _fsync_parent_directory(path: Path) -> None:
    """Durably publish a new file entry where directory fsync is supported."""
    dir_fd: int | None = None
    try:
        dir_fd = os.open(str(Path(path).parent), os.O_RDONLY)
        os.fsync(dir_fd)
    except (OSError, NotImplementedError):
        # Windows and some filesystems cannot open/fsync directory handles.
        return
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


def _validate_restored_checkpoint_metrics(
    record: dict[str, Any], *, line_number: int, path: Path, recall_rerank: bool = False
) -> None:
    """Validate every aggregate input restored from a scored checkpoint row."""
    if record.get("abstention") is True:
        return

    def require_real(field: str, value: Any) -> None:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"checkpoint line {line_number} field {field} must be a finite real number: {path}"
            )

    require_real("ingest_ms", record.get("ingest_ms"))
    arms = record.get("arms")
    if not isinstance(arms, dict):
        raise ValueError(f"checkpoint line {line_number} field arms must be an object: {path}")
    for arm in ARMS:
        metrics = arms.get(arm)
        arm_field = f"arms.{arm}"
        if not isinstance(metrics, dict):
            raise ValueError(
                f"checkpoint line {line_number} field {arm_field} must be an object: {path}"
            )
        for metric in ("recall@1", "recall@5", "recall@10", "ndcg@10", "latency_ms"):
            require_real(f"{arm_field}.{metric}", metrics.get(metric))
        turn = metrics.get("turn")
        turn_field = f"{arm_field}.turn"
        if not isinstance(turn, dict):
            raise ValueError(
                f"checkpoint line {line_number} field {turn_field} must be an object: {path}"
            )
        for metric in ("recall@1", "recall@5", "recall@10", "ndcg@10"):
            require_real(f"{turn_field}.{metric}", turn.get(metric))
        if not isinstance(turn.get("session_granularity"), bool):
            raise ValueError(
                f"checkpoint line {line_number} field {turn_field}.session_granularity "
                f"must be a boolean: {path}"
            )
    if recall_rerank:
        recall_metrics = arms.get("lcm_recall")
        status = (
            recall_metrics.get("recall_rerank_status")
            if isinstance(recall_metrics, dict)
            else None
        )
        if not isinstance(status, str) or not status:
            raise ValueError(
                f"checkpoint line {line_number} field arms.lcm_recall.recall_rerank_status "
                f"must be a non-empty string: {path}"
            )


def _question_checkpoint_record(
    question: Question,
    scored: dict[str, Any] | None,
    *,
    chunk_embedding_mode: str | None = None,
) -> dict[str, Any]:
    zero_privacy = {key: 0 for key in _PRIVACY_KEYS}
    if scored is None:
        record = {
            "question_id": question.question_id,
            "category": question.category,
            "abstention": True,
            "rerank_mode": None,
            "ingest_ms": 0.0,
            "privacy": zero_privacy,
            "corpus_counts": {"messages": 0, "summary_nodes": 0, "chunks": 0},
            "embed_cache": {"hits": 0, "misses": 0},
            "arms": {},
        }
        if chunk_embedding_mode is not None:
            record["chunk_embedding_mode"] = chunk_embedding_mode
        return record
    checkpoint_scored = copy.deepcopy(scored)
    checkpoint_scored.pop("_candidate_rankings", None)
    scored_chunk_embedding_mode = checkpoint_scored.pop(
        "chunk_embedding_mode", chunk_embedding_mode
    )
    ingest_ms = checkpoint_scored.pop("ingest_ms", 0.0)
    privacy = checkpoint_scored.pop(
        "privacy", zero_privacy
    )
    corpus_counts = checkpoint_scored.pop(
        "corpus_counts", {"messages": 0, "summary_nodes": 0, "chunks": 0}
    )
    embed_cache = checkpoint_scored.pop(
        "embed_cache", {"hits": 0, "misses": 0}
    )
    rerank_mode = checkpoint_scored["hybrid_rerank"].pop(
        "rerank_mode", RERANK_MODE_PLACEHOLDER
    )
    record = {
        "question_id": question.question_id,
        "category": question.category,
        "abstention": False,
        "rerank_mode": rerank_mode,
        "ingest_ms": ingest_ms,
        "privacy": privacy,
        "corpus_counts": corpus_counts,
        "embed_cache": embed_cache,
        "arms": checkpoint_scored,
    }
    if scored_chunk_embedding_mode is not None:
        record["chunk_embedding_mode"] = scored_chunk_embedding_mode
    return record


def _restore_privacy_counts(records: Iterable[dict[str, Any]]) -> None:
    """Restore additive privacy totals from old or new checkpoint rows."""
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = record.get("question_id", "?")
        if "privacy" not in record or record["privacy"] is None:
            continue
        privacy = record["privacy"]
        if not isinstance(privacy, dict):
            raise ValueError(
                f"checkpoint row {question_id}: malformed privacy={privacy!r}"
            )
        for key in _PRIVACY_KEYS:
            value = privacy.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"checkpoint row {question_id}: malformed privacy.{key}={value!r}"
                )
            _PRIVACY_COUNTS[key] += value


def _restore_embed_cache_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Restore additive per-question cache totals from old or new checkpoint rows."""
    totals = {"hits": 0, "misses": 0}
    for record in records:
        if not isinstance(record, dict):
            continue
        question_id = record.get("question_id", "?")
        if "embed_cache" not in record or record["embed_cache"] is None:
            continue
        embed_cache = record["embed_cache"]
        if not isinstance(embed_cache, dict):
            raise ValueError(
                f"checkpoint row {question_id}: malformed embed_cache={embed_cache!r}"
            )
        for key in totals:
            value = embed_cache.get(key, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"checkpoint row {question_id}: malformed embed_cache.{key}={value!r}"
                )
            totals[key] += value
    return totals


def _accumulate_question_checkpoint(
    record: dict[str, Any],
    *,
    by_category: dict[str, dict[str, ArmSamples]],
    overall: dict[str, ArmSamples],
    ingest_samples: list[float],
    rerank_mode_counts: dict[str, int],
    recall_rerank_status_counts: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Seed aggregate state from one live or resumed per-question record."""
    if record.get("abstention") is True:
        return 0, 1
    if record.get("abstention") is not False:
        raise ValueError(f"checkpoint question {record.get('question_id')!r} has invalid abstention")
    category = record.get("category")
    rerank_mode = record.get("rerank_mode")
    arms = record.get("arms")
    if not isinstance(category, str) or not category:
        raise ValueError(f"checkpoint question {record.get('question_id')!r} has invalid category")
    if not isinstance(rerank_mode, str) or not rerank_mode:
        raise ValueError(
            f"checkpoint question {record.get('question_id')!r} has invalid rerank_mode"
        )
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError(f"checkpoint question {record.get('question_id')!r} has invalid arms")
    ingest_ms = record.get("ingest_ms")
    if not isinstance(ingest_ms, (int, float)) or isinstance(ingest_ms, bool):
        raise ValueError(f"checkpoint question {record.get('question_id')!r} has invalid ingest_ms")

    ingest_samples.append(float(ingest_ms))
    rerank_mode_counts[rerank_mode] = rerank_mode_counts.get(rerank_mode, 0) + 1
    if recall_rerank_status_counts is not None:
        recall_rerank_status = arms.get("lcm_recall", {}).get("recall_rerank_status")
        if not isinstance(recall_rerank_status, str) or not recall_rerank_status:
            raise ValueError(
                f"checkpoint question {record.get('question_id')!r} has invalid "
                "recall_rerank_status"
            )
        recall_rerank_status_counts[recall_rerank_status] = (
            recall_rerank_status_counts.get(recall_rerank_status, 0) + 1
        )
    bucket = by_category.setdefault(category, _new_arm_samples())
    try:
        for arm in ARMS:
            metrics = arms[arm]
            turn = metrics["turn"]
            for k in (1, 5, 10):
                bucket[arm].recalls[k].append(metrics[f"recall@{k}"])
                overall[arm].recalls[k].append(metrics[f"recall@{k}"])
                bucket[arm].turn_recalls[k].append(turn[f"recall@{k}"])
                overall[arm].turn_recalls[k].append(turn[f"recall@{k}"])
            bucket[arm].ndcg10.append(metrics["ndcg@10"])
            overall[arm].ndcg10.append(metrics["ndcg@10"])
            bucket[arm].turn_ndcg10.append(turn["ndcg@10"])
            overall[arm].turn_ndcg10.append(turn["ndcg@10"])
            bucket[arm].latency_ms.append(metrics["latency_ms"])
            overall[arm].latency_ms.append(metrics["latency_ms"])
            if turn["session_granularity"]:
                bucket[arm].session_granularity = True
                overall[arm].session_granularity = True
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"checkpoint question {record.get('question_id')!r} has invalid arm metrics"
        ) from exc
    return 1, 0


def run_harness(
    questions: Iterable[Question],
    *,
    provider_name: str,
    model: str,
    tmp_dir: Path,
    embeddings_enabled: bool | None = None,
    use_rerank: bool = False,
    recall_rerank: bool = False,
    recall_rerank_window: int = 0,
    recall_rerank_margin: float = 0.0,
    reuse_db_template: bool = True,
    question_count: int | None = None,
    dataset_label: str = "s",
    source_sha256: str | None = None,
    direct_source_sha256: str | None = None,
    manifest_sha256: str | None = None,
    checkpoint_path: Path | None = None,
    dump_candidates_path: Path | None = None,
    resume: bool = False,
    selected_question_ids: Iterable[str] | None = None,
    accounting: ProviderAccounting | None = None,
) -> dict[str, Any]:
    """Run every arm over every question and return an aggregate-only report."""
    _reset_privacy_counts()
    dataset_report: dict[str, Any] = dataset_coordinates(dataset_label)
    if source_sha256 is not None:
        if not _SHA256_RE.fullmatch(source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
        dataset_report["source_sha256"] = source_sha256
    if direct_source_sha256 is not None and not _SHA256_RE.fullmatch(
        direct_source_sha256
    ):
        raise ValueError("direct_source_sha256 must be a lowercase SHA-256 hex digest")
    if manifest_sha256 is not None:
        if not _SHA256_RE.fullmatch(manifest_sha256):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 hex digest")
        dataset_report["manifest_sha256"] = manifest_sha256
    if resume and selected_question_ids is None:
        if isinstance(questions, Sequence):
            selected_question_ids = tuple(question.question_id for question in questions)
        else:
            raise ValueError(
                "selected_question_ids is required to resume a streaming question iterator"
            )
    selected_id_sequence = tuple(selected_question_ids or ())
    selected_ids = set(selected_id_sequence)
    if resume and len(selected_ids) != len(selected_id_sequence):
        raise ValueError("selected question ids must be unique when resuming")

    if embeddings_enabled is None:
        embeddings_enabled = provider_name != "none"
    effective_embedding_batch_size = (
        _embedding_batch_size() if embeddings_enabled else None
    )
    expected_checkpoint_header = _checkpoint_header(
        provider=provider_name,
        model=model,
        rerank=use_rerank,
        recall_rerank=recall_rerank,
        recall_rerank_window=recall_rerank_window,
        recall_rerank_margin=recall_rerank_margin,
        embeddings_enabled=embeddings_enabled,
        dataset_label=dataset_label,
        direct_source_sha256=direct_source_sha256,
        manifest_sha256=manifest_sha256,
        reuse_db_template=reuse_db_template,
        embedding_batch_size=effective_embedding_batch_size,
    )
    expected_dump_header = _candidate_dump_header(
        provider=provider_name,
        model=model,
        rerank=use_rerank,
        recall_rerank=recall_rerank,
        recall_rerank_window=recall_rerank_window,
        recall_rerank_margin=recall_rerank_margin,
        embeddings_enabled=embeddings_enabled,
        dataset_label=dataset_label,
        direct_source_sha256=direct_source_sha256,
        manifest_sha256=manifest_sha256,
    )

    if resume and checkpoint_path is None:
        raise ValueError("resume=True requires a checkpoint_path")
    checkpoint_records: list[dict[str, Any]] = []
    restored_embed_cache = {"hits": 0, "misses": 0}
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_exists = checkpoint_path.exists()
        checkpoint_nonempty = checkpoint_exists and checkpoint_path.stat().st_size > 0
        if not resume and checkpoint_nonempty:
            raise ValueError(
                "checkpoint exists from a previous run; pass --resume to continue it "
                "or use a fresh --output"
            )
        if resume:
            if not checkpoint_nonempty:
                raise ValueError(
                    f"cannot resume without a non-empty checkpoint containing "
                    f"{_CHECKPOINT_HEADER_KEY}: {checkpoint_path}"
                )
            checkpoint_records = _load_question_checkpoint(
                checkpoint_path,
                selected_question_ids=selected_ids,
                expected_header=expected_checkpoint_header,
            )
            _restore_privacy_counts(checkpoint_records)
            restored_embed_cache = _restore_embed_cache_counts(checkpoint_records)

    by_category: dict[str, dict[str, ArmSamples]] = {}
    overall = _new_arm_samples()
    scored_count = 0
    abstention_count = 0
    ingest_samples: list[float] = []
    # Track per-question rerank modes so the run-level label is an aggregate, not
    # whatever the final question happened to use (FIX-2).
    rerank_mode_counts: dict[str, int] = {}
    recall_rerank_status_counts: dict[str, int] | None = (
        {} if recall_rerank else None
    )
    consumed_count = 0

    completed_question_ids = {
        str(record["question_id"]) for record in checkpoint_records
    }
    for record in checkpoint_records:
        scored_delta, abstention_delta = _accumulate_question_checkpoint(
            record,
            by_category=by_category,
            overall=overall,
            ingest_samples=ingest_samples,
            rerank_mode_counts=rerank_mode_counts,
            recall_rerank_status_counts=recall_rerank_status_counts,
        )
        scored_count += scored_delta
        abstention_count += abstention_delta

    fully_completed_resume = resume and completed_question_ids == selected_ids
    embed_cache_path = os.environ.get(EMBED_CACHE_ENV)
    if embed_cache_path is not None and not embed_cache_path.strip():
        raise ValueError(f"{EMBED_CACHE_ENV} must be a non-empty SQLite path")

    provider_set: HarnessProviderSet | None = None
    db_template: Path | None = None
    resolved_chunk_embedding_mode: str | None = None
    if not fully_completed_resume:
        if embeddings_enabled:
            provider_set = resolve_harness_providers(
                provider_name, model, accounting=accounting
            )
        else:
            # Lexical-only runs retain the historical summary-provider setup but
            # never resolve or warm the distinct production chunk identity.
            summary_provider = resolve_harness_provider(
                provider_name,
                model,
                accounting=accounting,
                accounting_role="summary_documents",
            )
            summary_binding = _provider_binding(summary_provider, provider_name, model)
            provider_set = HarnessProviderSet(
                summary=summary_provider,
                chunk=summary_provider,
                summary_binding=summary_binding,
                chunk_binding=_configured_chunk_binding(*summary_binding),
            )
        resolved_chunk_embedding_mode = _resolved_chunk_embedding_mode(
            provider_set.chunk
        )
        if resume:
            for record in checkpoint_records:
                question_id = record["question_id"]
                if "chunk_embedding_mode" not in record:
                    raise ValueError(
                        f"checkpoint row {question_id}: missing chunk_embedding_mode; "
                        "rows written before this instrument cannot be partially "
                        "resumed — start a fresh output root"
                    )
                recorded_mode = record["chunk_embedding_mode"]
                if recorded_mode != resolved_chunk_embedding_mode:
                    raise ValueError(
                        f"checkpoint row {question_id}: "
                        f"chunk_embedding_mode={recorded_mode!r} but the resumed run "
                        f"resolves {resolved_chunk_embedding_mode!r}"
                    )
        if reuse_db_template:
            _ensure_hermes_lcm_package()
            from hermes_lcm.config import LCMConfig

            db_template = Path(tmp_dir) / "_template.db"
            _bootstrap_db_template(
                db_template,
                LCMConfig(
                    database_path=str(db_template),
                    embeddings_enabled=embeddings_enabled,
                    embedding_provider=provider_set.summary_binding[0],
                    embedding_model=provider_set.summary_binding[1],
                    # Match the per-question lossless/provider-copy split.
                    sensitive_patterns_enabled=False,
                    embedding_privacy_enabled=None,
                ),
            )

    checkpoint_file = None
    if checkpoint_path is not None and not fully_completed_resume:
        checkpoint_file = checkpoint_path.open("a" if resume else "w", encoding="utf-8")
        if not resume:
            _write_checkpoint_record(checkpoint_file, expected_checkpoint_header)
            # Make the directory ENTRY durable too — record fsyncs alone don't
            # survive a crash that predates the dirent reaching disk.
            _fsync_parent_directory(checkpoint_path)

    candidate_dump_file = None
    if dump_candidates_path is not None and not fully_completed_resume:
        dump_candidates_path = Path(dump_candidates_path)
        dump_candidates_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dump = (
            dump_candidates_path.read_bytes() if dump_candidates_path.exists() else b""
        )
        # Validate BEFORE any mutation, and FAIL CLOSED on uncertain ownership:
        # only a file whose first newline-terminated line parses as OUR header
        # (matching this run's config binding) may be repaired or appended to.
        # A nonempty file with no complete first line cannot prove ownership --
        # even a prefix of our own torn header write is indistinguishable from
        # a torn foreign JSON -- so it is rejected byte-intact; the operator
        # deletes it deliberately (dump rows always precede their checkpoint
        # records, so nothing checkpointed is lost by starting fresh).
        if raw_dump and b"\n" not in raw_dump:
            raise ValueError(
                "candidate dump target exists without a complete header line "
                f"(cannot prove ownership, refusing to touch it): {dump_candidates_path}"
            )
        if raw_dump:
            raw_header = raw_dump.split(b"\n", 1)[0]
            try:
                dump_header = json.loads(raw_header)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid candidate dump header: {dump_candidates_path}"
                ) from exc
            _validate_candidate_dump_header(
                dump_header, expected_header=expected_dump_header, path=dump_candidates_path
            )
        if raw_dump and not raw_dump.endswith(b"\n"):
            # Now provably our own dump. A torn final row always precedes its
            # checkpoint record -- that question re-evaluates on resume -- so
            # truncating back to the last complete row loses nothing and keeps
            # every remaining line parseable.
            keep = raw_dump.rfind(b"\n") + 1
            with dump_candidates_path.open("r+b") as existing_dump:
                existing_dump.truncate(keep)
                existing_dump.flush()
                os.fsync(existing_dump.fileno())
            raw_dump = raw_dump[:keep]
        candidate_dump_file = dump_candidates_path.open("a", encoding="utf-8")
        if not raw_dump:
            _write_candidate_dump_record(candidate_dump_file, expected_dump_header)
            _fsync_parent_directory(dump_candidates_path)

    observed_question_ids: set[str] = set()
    if fully_completed_resume:
        observed_question_ids.update(selected_ids)
        consumed_count = len(selected_id_sequence)
    try:
        for question in (() if fully_completed_resume else questions):
            consumed_count += 1
            observed_question_ids.add(question.question_id)
            if resume and question.question_id not in selected_ids:
                raise ValueError(
                    f"question_id {question.question_id!r} is not in the selected question set"
                )
            if question.question_id in completed_question_ids:
                continue
            try:
                scored = None
                if not question.is_abstention:
                    scored = evaluate_question(
                        question,
                        provider_set.summary,
                        provider_name=provider_name,
                        tmp_dir=tmp_dir,
                        embeddings_enabled=embeddings_enabled,
                        use_rerank=use_rerank,
                        recall_rerank=recall_rerank,
                        recall_rerank_window=recall_rerank_window,
                        recall_rerank_margin=recall_rerank_margin,
                        db_template=db_template,
                        embedding_batch_size=effective_embedding_batch_size,
                        include_rankings=dump_candidates_path is not None,
                        chunk_provider=provider_set.chunk,
                        accounting=accounting,
                    )
                candidate_rankings = None
                if dump_candidates_path is not None and scored is not None:
                    candidate_rankings = scored.pop("_candidate_rankings", None)
                    if candidate_rankings is None:
                        raise RuntimeError(
                            "candidate dump rankings were not returned for a scored question"
                        )
                record = _question_checkpoint_record(
                    question,
                    scored,
                    chunk_embedding_mode=resolved_chunk_embedding_mode,
                )
                scored_delta, abstention_delta = _accumulate_question_checkpoint(
                    record,
                    by_category=by_category,
                    overall=overall,
                    ingest_samples=ingest_samples,
                    rerank_mode_counts=rerank_mode_counts,
                    recall_rerank_status_counts=recall_rerank_status_counts,
                )
                scored_count += scored_delta
                abstention_count += abstention_delta
                # Sidecar row FIRST, checkpoint second: a crash between the two
                # re-evaluates the question on resume (its checkpoint record
                # never landed) and appends a fresh sidecar row. The worst case
                # is a duplicate sidecar row for one question -- consumers
                # dedupe by question_id keeping the last -- never a checkpointed
                # question with a silently missing sidecar row.
                if candidate_dump_file is not None:
                    _write_candidate_dump_record(
                        candidate_dump_file,
                        _candidate_dump_record(
                            question,
                            candidate_rankings,
                            recall_rerank=recall_rerank,
                        ),
                    )
                if checkpoint_file is not None:
                    _write_checkpoint_record(checkpoint_file, record)
            finally:
                _cleanup_question_db(tmp_dir, question.question_id)
    finally:
        if checkpoint_file is not None:
            checkpoint_file.close()
        if candidate_dump_file is not None:
            candidate_dump_file.close()

    if resume and observed_question_ids != selected_ids:
        missing = sorted(selected_ids - observed_question_ids)
        extra = sorted(observed_question_ids - selected_ids)
        raise ValueError(
            f"selected question ids do not match consumed questions: missing={missing}, extra={extra}"
        )

    if question_count is not None and consumed_count != question_count:
        raise ValueError(
            f"question count mismatch: expected {question_count}, consumed {consumed_count}"
        )

    if resolved_chunk_embedding_mode is None:
        recorded_modes = {
            record.get("chunk_embedding_mode")
            for record in checkpoint_records
            if record.get("chunk_embedding_mode") in {"flat", "contextual"}
        }
        if len(recorded_modes) == 1:
            resolved_chunk_embedding_mode = recorded_modes.pop()
        elif len(recorded_modes) > 1:
            raise ValueError("checkpoint rows contain mixed chunk embedding modes")
        elif fully_completed_resume and not any(
            "chunk_embedding_mode" in record for record in checkpoint_records
        ):
            resolved_chunk_embedding_mode = "unknown"
        else:
            legacy_provider_set = resolve_harness_providers(
                provider_name, model, accounting=accounting
            )
            resolved_chunk_embedding_mode = _resolved_chunk_embedding_mode(
                legacy_provider_set.chunk
            )

    ingest_report: dict[str, Any] = {
        "batched_embeddings": embeddings_enabled,
        "chunk_embedding_mode": resolved_chunk_embedding_mode,
        "reuse_db_template": reuse_db_template,
        "per_question_ms": percentiles(ingest_samples),
        "privacy": dict(_PRIVACY_COUNTS),
    }
    if dataset_label == "m" or manifest_sha256 is not None:
        ingest_report["embedding_batch_size"] = effective_embedding_batch_size
    if embed_cache_path is not None:
        live_cache_hits, live_cache_misses = _embed_cache_totals(
            ()
            if provider_set is None
            else (provider_set.summary, provider_set.chunk)
        )
        ingest_report["embed_cache"] = {
            "hits": restored_embed_cache["hits"] + live_cache_hits,
            "misses": restored_embed_cache["misses"] + live_cache_misses,
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transcript_contents_included": False,
        "dataset": dataset_report,
        "provider": provider_name,
        "model": model,
        "embeddings_enabled": embeddings_enabled,
        "question_count": consumed_count,
        "scored_count": scored_count,
        "abstention_excluded": abstention_count,
        "rerank": {
            **_aggregate_rerank_mode(rerank_mode_counts),
            "candidate_window": RERANK_CANDIDATE_WINDOW,
            "timeout_s": RERANK_TIMEOUT_S,
        },
        "ingest": ingest_report,
        "arms": {
            arm: _arm_report(overall[arm]) for arm in ARMS
        },
        "per_category": {
            category: {arm: _arm_report(samples[arm]) for arm in ARMS}
            for category, samples in sorted(by_category.items())
        },
    }
    if recall_rerank:
        report["recall_rerank_modes"] = dict(recall_rerank_status_counts or {})
    return report


def _arm_report(samples: ArmSamples) -> dict[str, Any]:
    return {
        "recall@1": _mean(samples.recalls[1]),
        "recall@5": _mean(samples.recalls[5]),
        "recall@10": _mean(samples.recalls[10]),
        "ndcg@10": _mean(samples.ndcg10),
        "n": len(samples.ndcg10),
        "latency_ms": percentiles(samples.latency_ms),
        "turn": {
            "recall@1": _mean(samples.turn_recalls[1]),
            "recall@5": _mean(samples.turn_recalls[5]),
            "recall@10": _mean(samples.turn_recalls[10]),
            "ndcg@10": _mean(samples.turn_ndcg10),
            "session_granularity": samples.session_granularity,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Aggregate-only markdown table of overall per-arm session + turn recall/NDCG.

    ``*`` on an arm name marks turn-level numbers that are session-granularity: the
    arm retrieves summaries, which localize only to a whole session, so a hit credits
    every evidence turn of that session at once.
    """
    rerank = report.get("rerank", {})
    ingest = report.get("ingest", {})
    per_q = ingest.get("per_question_ms", {})
    lines = [
        f"# {report['dataset']['name']} retrieval — provider={report['provider']} "
        f"model={report['model'] or 'n/a'}",
        "",
        f"scored={report['scored_count']} abstention_excluded={report['abstention_excluded']} "
        f"dataset={report['dataset']['repo_id']}@{report['dataset']['revision'][:7]}",
        f"rerank={rerank.get('mode', 'n/a')} (window={rerank.get('candidate_window', 'n/a')}) "
        f"ingest_p50={per_q.get('p50', 0.0):.1f}ms "
        f"batched_embeddings={ingest.get('batched_embeddings', False)} "
        f"reuse_db_template={ingest.get('reuse_db_template', False)}",
        "",
        "| Arm | R@1 | R@5 | R@10 | NDCG@10 | tR@1 | tR@5 | tR@10 | tNDCG@10 | p50 ms | p90 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = report["arms"][arm]
        turn = row["turn"]
        label = f"{arm}*" if turn.get("session_granularity") else arm
        lines.append(
            f"| {label} | {row['recall@1']:.3f} | {row['recall@5']:.3f} | "
            f"{row['recall@10']:.3f} | {row['ndcg@10']:.3f} | "
            f"{turn['recall@1']:.3f} | {turn['recall@5']:.3f} | "
            f"{turn['recall@10']:.3f} | {turn['ndcg@10']:.3f} | "
            f"{row['latency_ms']['p50']:.1f} | {row['latency_ms']['p90']:.1f} |"
        )
    lines.append("")
    lines.append(
        "`t*` columns are turn-level. `*` = session-granularity turn scoring "
        "(summary arms cannot localize to a turn)."
    )
    return "\n".join(lines)
