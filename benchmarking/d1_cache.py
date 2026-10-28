"""D1 harness-only sealed cache; product chunking/retrieval remain untouched."""
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .longmemeval import ContentHashEmbeddingCache


def seal(path):
    """Publish the identity and ordered entry hashes after the sole writer exits."""
    path = Path(path)
    with closing(sqlite3.connect(path)) as db:
        identity = db.execute("SELECT identity FROM binding").fetchone()[0]
        rows = db.execute("SELECT key, checksum FROM entries ORDER BY key").fetchall()
    manifest = json.dumps({"identity": identity, "entries": rows}, separators=(",", ":"))
    path.with_suffix(".manifest.json").write_text(manifest)
    path.with_suffix(".manifest.json").chmod(0o444)
    path.chmod(0o444)
    return hashlib.sha256(manifest.encode()).hexdigest()


def merge(parts, target):
    parts = list(parts)
    identities = []
    for part in parts:
        with closing(sqlite3.connect(part)) as db:
            identities.append(db.execute("SELECT identity FROM binding").fetchone()[0])
    if len(set(identities)) != 1:
        raise ValueError("D1 cache part model/options mismatch")
    entries, duplicates = 0, 0
    with closing(sqlite3.connect(target)) as db, db:
        db.execute("CREATE TABLE binding (identity TEXT NOT NULL)")
        db.execute("CREATE TABLE entries (key TEXT PRIMARY KEY, dim INTEGER, payload BLOB, checksum TEXT)")
        db.execute("INSERT INTO binding VALUES (?)", (identities[0],))
        for part in parts:
            with closing(sqlite3.connect(part)) as source:
                for key, dim, payload, checksum in source.execute("SELECT key, dim, payload, checksum FROM entries"):
                    existing = db.execute("SELECT dim, payload, checksum FROM entries WHERE key=?", (key,)).fetchone()
                    if existing is not None:
                        if existing != (dim, payload, checksum):
                            raise ValueError("D1 cache conflicting duplicate key")
                        duplicates += 1
                    else:
                        db.execute("INSERT INTO entries VALUES (?,?,?,?)", (key, dim, payload, checksum))
                        entries += 1
    return entries, duplicates


class D1Cache(ContentHashEmbeddingCache):
    """Exact document boundary, one options binding per DB, read-only fallbacks."""

    def __init__(self, provider, path, *, options, readonly=True):
        self.readonly = readonly
        self.partition = None
        partition = os.environ.get("LCM_D1_BUILD_PARTITION") if not readonly else None
        if partition is not None:
            values = partition.split("/")
            if len(values) != 2 or not all(value.isdecimal() for value in values):
                raise ValueError("D1 build partition must be i/P with 0 <= i < P")
            i, count = map(int, values)
            if not 0 <= i < count:
                raise ValueError("D1 build partition must be i/P with 0 <= i < P")
            self.partition = i, count
        self.options = options
        self.identity = json.dumps(
            [provider.provider_id, provider.model_id, options], sort_keys=True,
            separators=(",", ":"),
        )
        self.wall_ms = 0.0
        self.corruptions = 0
        self.manifest_hash = None
        self.expected = {}
        super().__init__(provider, path)
        if readonly:
            raw = Path(path).with_suffix(".manifest.json").read_bytes()
            manifest = json.loads(raw)
            if manifest["identity"] != self.identity:
                raise ValueError("D1 cache manifest model/options mismatch")
            self.expected = dict(manifest["entries"])
            self.manifest_hash = hashlib.sha256(raw).hexdigest()

    def _connect(self):
        return sqlite3.connect(self.cache_path.resolve().as_uri() + "?mode=ro", uri=True) if self.readonly else sqlite3.connect(self.cache_path)

    def _initialize(self):
        with closing(self._connect()) as db, db:
            if not self.readonly:
                db.execute("CREATE TABLE IF NOT EXISTS binding (identity TEXT NOT NULL)")
                db.execute("CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, dim INTEGER, payload BLOB, checksum TEXT)")
                if not db.execute("SELECT 1 FROM binding").fetchone():
                    db.execute("INSERT INTO binding VALUES (?)", (self.identity,))
            if db.execute("SELECT identity FROM binding").fetchall() != [(self.identity,)]:
                raise ValueError("D1 cache model/options mismatch")

    def content_sha256(self, text):
        return hashlib.sha256((self.identity + "\0" + text).encode()).hexdigest()

    def _lookup(self, digests):
        found = {}
        with closing(self._connect()) as db:
            for key in dict.fromkeys(digests):
                row = db.execute("SELECT dim, payload, checksum FROM entries WHERE key=?", (key,)).fetchone()
                if row is None:
                    self.corruptions += int(self.readonly and key in self.expected)
                    continue
                dim, payload, checksum = row
                if not isinstance(payload, bytes):
                    self.corruptions += 1
                    continue
                actual = hashlib.sha256(key.encode() + payload).hexdigest()
                if actual != checksum or (self.readonly and self.expected.get(key) != checksum) or dim != self.dim:
                    self.corruptions += 1
                    continue
                try:
                    found[key] = self._decode_vector(dim, payload)
                except ValueError:
                    self.corruptions += 1
        return found

    def embed_documents(self, texts):
        start = time.perf_counter()
        try:
            keys = [self.content_sha256(text) for text in texts]
            found = self._lookup(keys)
            self.hits += sum(key in found for key in keys)
            self.misses += sum(key not in found for key in keys)
            if self.partition is not None:
                i, count = self.partition
                # prewarm_embedding_cache.flush_batch discards these placeholder vectors.
                found.update((key, [0.0] * self.dim) for key in keys if int(key[:16], 16) % count != i)
            missing = dict((key, text) for key, text in zip(keys, texts) if key not in found)
            if missing:
                self.provider_dispatches += 1
                vectors = list(self._provider.embed_documents(list(missing.values())))
                if len(vectors) != len(missing):
                    raise ValueError("D1 provider vector count mismatch")
                with closing(self._connect()) as db, db:
                    for key, vector in zip(missing, vectors):
                        dim, payload = self._encode_vector(vector)
                        if dim != self.dim:
                            raise ValueError("D1 provider dimension mismatch")
                        found[key] = list(vector)
                        if not self.readonly:
                            db.execute("INSERT OR REPLACE INTO entries VALUES (?,?,?,?)", (key, dim, payload, hashlib.sha256(key.encode() + payload).hexdigest()))
            return [found[key] for key in keys]
        finally:
            self.wall_ms += (time.perf_counter() - start) * 1000

    def embed_query(self, text):
        start = time.perf_counter()
        try:
            return super().embed_query(text)
        finally:
            self.wall_ms += (time.perf_counter() - start) * 1000
