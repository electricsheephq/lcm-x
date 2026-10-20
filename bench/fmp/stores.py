"""Headless production stores, built once; mismatches retain old generations."""
import importlib.metadata
import json
import os
import sqlite3
import struct
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .common import commit, sha, write_json
from .delivery import question_clock

MODELS = {"off": ("", ""), "local": ("fastembed", "BAAI/bge-small-en-v1.5"),
          "voyage": ("voyage", "voyage-context-3")}
# Optional directory of earlier embedding caches, read only; misses use the run-owned cache.
CACHE_ROOT = Path(os.environ["FMP_CACHE_ROOT"]) if os.environ.get("FMP_CACHE_ROOT") else None


def safe(value):
    # Hash arbitrary dataset ids, avoiding path traversal and sanitization collisions.
    return sha(str(value).encode())[:24]


class ReadCache:
    """Read existing cache entries; misses use a run-owned production cache."""
    def __init__(self, provider, paths):
        self.provider, self.paths = provider, paths

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def embed_documents(self, texts):
        found = {}
        for path in self.paths:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
                for text in texts:
                    row = con.execute("SELECT vector_dim, vector_f64_le FROM embedding_cache "
                                      "WHERE provider=? AND model=? AND content_sha256=?",
                                      (self.provider_id, self.model_id, sha(text.encode()))).fetchone()
                    if row:
                        found[text] = list(struct.unpack(f"<{row[0]}d", row[1]))
        missing = list(dict.fromkeys(t for t in texts if t not in found))
        if missing:
            found.update(zip(missing, self.provider.embed_documents(missing)))
        return [found[t] for t in texts]


class Stores:
    def __init__(self, root, config, repo, dataset_sha, lme, summaries="none", cache=None, store_text="speaker-prefix"):
        self.root, self.config, self.repo = Path(root), config, Path(repo)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_sha, self.lme, self.summaries = dataset_sha, lme, summaries
        self.store_text = store_text
        self.base, sep, self.rerank = config.partition("+rerank:")
        if self.base not in MODELS or (sep and not self.rerank):
            raise ValueError("unsupported store configuration")
        self.provider_name, self.model = MODELS[self.base]
        self.provider = None
        if self.base != "off":
            if self.base == "local" and importlib.metadata.version("fastembed") != "0.8.0":
                raise ValueError("local requires fastembed==0.8.0")
            from benchmarking.longmemeval import ContentHashEmbeddingCache
            provider = lme.resolve_harness_provider(self.provider_name, self.model, use_embed_cache=False)
            local = ContentHashEmbeddingCache(provider, self.root / f"{self.base}-embed-cache.sqlite")
            paths = []
            for path in ([Path(cache)] if cache else sorted(CACHE_ROOT.glob("*cache*.sqlite")) if CACHE_ROOT else []):
                try:
                    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
                        if con.execute("SELECT 1 FROM embedding_cache WHERE provider=? AND model=? LIMIT 1",
                                       (self.provider_name, self.model)).fetchone():
                            paths.append(path)
                except sqlite3.Error:
                    if cache:
                        raise
            self.provider = ReadCache(local, paths)
        self.identity = {"lcmx_commit": commit(repo), "config": self.base,
                         "dataset_sha256": dataset_sha, "summaries": summaries,
                         "provider": self.provider_name, "model": self.model,
                         "embedding_privacy": "raw-public-benchmark", "format": 3, "store_text": store_text}
        # D8: the path names every build input, so no build is reused across inputs.
        self.generation = "-".join([self.base, summaries, self.identity["lcmx_commit"][:12],
                                    sha(json.dumps(self.identity, sort_keys=True).encode())[:12]])

    def content(self, turn, date):
        name = turn.get("speaker")
        if not name or self.store_text == "plain":
            return turn["content"]
        if self.store_text == "name-date":
            name += " (" + datetime.fromisoformat(date).astimezone(timezone.utc).date().isoformat() + ")"
        return name + ": " + turn["content"]

    def directory(self, q):
        # Reranking is recall-time only; variants share the same ingestion.
        return self.root / self.generation / safe(q.store_id or q.qid)

    def expected(self, q):
        return {**self.identity, "id": q.store_id or q.qid, "session_map": q.session_map}

    @contextmanager
    def engine(self, q, path=None, frozen=True):
        from hermes_lcm.config import LCMConfig
        from hermes_lcm.store import MessageStore
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.vector_store import VectorStore
        directory = path or self.directory(q)
        if path is None and json.loads((directory / "build-manifest.json").read_text()) != self.expected(q):
            raise RuntimeError("store manifest mismatch")
        config = LCMConfig(database_path=str(directory / "lcm.db"), embeddings_enabled=self.base != "off",
                           embedding_provider=self.provider_name or "fastembed", embedding_model=self.model,
                           embedding_privacy_enabled=False, rerank_enabled=bool(self.rerank),
                           rerank_model=self.rerank or "rerank-2.5-lite")
        store = MessageStore(config.database_path, ingest_protection_config=config)
        dag = SummaryDAG(config.database_path)
        vector = VectorStore(config.database_path, config=config)
        engine = SimpleNamespace(_config=config, _store=store, _dag=dag, _hermes_home=str(directory),
                                 current_session_id="__fmp_probe__" + safe(q.qid))
        if self.provider:
            engine._lcm_embedding_provider_cache = ((self.provider_name, self.model), self.provider)
        if frozen:
            for obj in (store, dag, vector):
                obj._conn.execute("PRAGMA query_only=ON")
        try:
            yield engine, vector
        finally:
            vector.close()
            dag.close()
            store.close()

    def build(self, q):
        path = self.directory(q)
        expected = self.expected(q)
        manifest = path / "build-manifest.json"
        if manifest.exists() and (path / "lcm.db").exists() and json.loads(manifest.read_text()) == expected:
            return "reused"
        stage = path.with_name(path.name + ".building-" + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        speakers = {}
        with self.engine(q, stage, frozen=False) as (engine, vectors):
            from hermes_lcm.chunking import iter_message_chunks
            from hermes_lcm.dag import SummaryNode
            from hermes_lcm.vector_store import EmbeddingIdentity
            if self.provider:
                vectors.register_profile(self.model, self.provider_name, self.provider.dim, task="chunk")
                identity = EmbeddingIdentity.canonical(self.provider_name, self.model, "", self.provider.dim,
                                                       "float32", "little", "chunk")
            for order, session in enumerate(q.sessions, 1):
                rows = [{"role": t["role"], "content": self.content(t, session["date"]),
                         "timestamp": session["date"]}
                        for t in session["turns"]]
                # Production ingest stamps timestamp/ingested_at from the store clock: run it at the
                # session date, so recency and event times behave as in a live store (LME and LoCoMo alike).
                with question_clock(session["date"]):
                    ids = engine._store.append_batch(session["session_id"], rows, source="benchmark",
                                                     conversation_id=session["session_id"])
                speakers.update({str(i): t["speaker"] for i, t in zip(ids, session["turns"]) if t.get("speaker")})
                if self.provider:
                    chunks = list(iter_message_chunks([{**r, "store_id": i} for i, r in zip(ids, rows)],
                                                       policy="conversational"))
                    for start in range(0, len(chunks), 64):
                        batch = chunks[start:start + 64]
                        embedded = self.provider.embed_documents([c.text for c in batch])
                        for c, v in zip(batch, embedded, strict=True):
                            vectors.record_chunk_embedding(c.chunk_id, self.model, v, store_id=c.store_id,
                                chunk_index=c.chunk_index, char_start=c.char_start, char_end=c.char_end,
                                token_estimate=c.token_estimate, identity=identity)
                if self.summaries == "deterministic":
                    text = self.lme.deterministic_session_summary(rows)
                    stamp = datetime.fromisoformat(session["date"]).timestamp() if session["date"] else None
                    node_id = engine._dag.add_node(SummaryNode(session_id=session["session_id"], depth=0, summary=text,
                        token_count=len(text.split()), source_token_count=sum(len(r["content"].split()) for r in rows),
                        source_type="messages", created_at=float(order), earliest_at=stamp, latest_at=stamp))
                    if self.provider:
                        vectors.register_profile(self.model, self.provider_name, self.provider.dim)
                        summary_identity = vectors.capture_identity(self.model, provider=self.provider_name)
                        vectors.record_embedding(str(node_id), "summary", self.model,
                            self.provider.embed_documents([text])[0], identity=summary_identity)
        write_json(stage / "speakers.json", speakers)
        write_json(stage / "build-manifest.json", expected)
        if path.exists():
            path.rename(path.with_name(path.name + ".superseded-" + uuid.uuid4().hex))
        stage.rename(path)
        return "built"
