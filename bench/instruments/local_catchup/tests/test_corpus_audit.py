from pathlib import Path
import sqlite3

import pytest

from bench.instruments.local_catchup.audit import audit, connect, preimage
from bench.instruments.local_catchup.common import MODEL, product
from bench.instruments.local_catchup.corpus import build

REPO = Path(__file__).resolve().parents[4]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path/"home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    host = Path(__import__("os").environ.get("INSTRUMENT_HERMES_SOURCE", REPO))
    db = tmp_path/"corpus.db"
    manifest = build(REPO, host, db, 8)
    return db, manifest, product(REPO, host)


def test_manifest_deterministic(corpus, tmp_path):
    db, manifest, mod = corpus
    other = build(REPO, REPO, tmp_path/"other.db", 8)
    assert manifest == other
    assert manifest["rows"] == 8 and manifest["pairs"] == 4
    assert len(manifest["expected"]["chunks"]) == 8
    assert manifest["system_rows"] == manifest["tool_rows"] == 0
    assert mod("tokens").count_tokens("abc") > 0


def test_shards_unique(corpus, tmp_path):
    manifest = build(REPO, REPO, tmp_path/"shards.db", 1002, "mixed")
    assert len(set(manifest["sessions"])) == 3
    assert len({c["id"] for c in manifest["canaries"]}) == 3
    assert manifest["input_tokens"] == sum(int(k)*v for k, v in manifest["token_distribution"].items())


def test_upgrade_has_short_gaps_and_partial_long(corpus, tmp_path):
    manifest = build(REPO, REPO, tmp_path/"upgrade.db", 8, baseline="upgrade")
    assert manifest["upgrade_long_chunks"] > 32
    assert f"chunks/{manifest['identities']['chunk']}/1:0" in manifest["preimage"]["vectors"]
    assert f"chunks/{manifest['identities']['chunk']}/1:1" not in manifest["preimage"]["vectors"]
    assert any(ident.startswith("2:") for ident in manifest["expected"]["chunks"])


def fill(db, manifest, mod):
    store = mod("vector_store").VectorStore(db)
    chunk_identity = mod("vector_store").EmbeddingIdentity.canonical("fastembed", MODEL, "", 384, "float32", "little", "chunk")
    leaf_identity = mod("vector_store").EmbeddingIdentity.canonical("fastembed", MODEL, "", 384, "float32", "little", "summary")
    with connect(db) as c:
        for sid, role, text in c.execute("SELECT store_id,role,content FROM messages"):
            for chunk in mod("chunking").chunk_message(sid, role, text):
                store.record_chunk_embedding(chunk.chunk_id, MODEL, [1.]+[0.]*383,
                    store_id=sid, chunk_index=chunk.chunk_index, char_start=chunk.char_start,
                    char_end=chunk.char_end, token_estimate=chunk.token_estimate, identity=chunk_identity)
    for leaf in manifest["expected"]["leaves"]:
        store.record_embedding(leaf, "summary", MODEL, [1.]+[0.]*383, identity=leaf_identity)
    store.close()


@pytest.mark.parametrize("failure", [None, "chunk", "leaf", "uncertain", "changed", "raw"])
def test_complete_audit_and_failure_kinds(corpus, failure):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    with connect(db) as c:
        manifest["preimage"] = preimage(c)
    with sqlite3.connect(db) as c:
        if failure == "chunk":
            c.execute("DELETE FROM lcm_chunk_vectors WHERE chunk_id='1:0'")
        elif failure == "leaf":
            c.execute("DELETE FROM lcm_embedding_vectors")
        elif failure == "uncertain":
            mod("command")._ensure_inflight_table(c)
            c.execute("INSERT INTO lcm_embedding_backfill_inflight(embedded_id,identity_hash,state) VALUES('1:0',?,'uncertain')", (manifest["identities"]["chunk"],))
        elif failure == "changed":
            c.execute("UPDATE lcm_chunk_vectors SET vec=zeroblob(1536) WHERE chunk_id='1:0'")
        elif failure == "raw":
            c.execute("UPDATE messages SET content='changed' WHERE store_id=1")
    result = audit(db, manifest)
    assert result["complete"] is (failure is None)
    if failure in ("chunk", "leaf"):
        cls = "chunks" if failure == "chunk" else "leaves"
        assert result["classes"][cls]["missing_count"] == 1
    if failure == "uncertain":
        assert result["residue"] == {"uncertain": 1}
    if failure == "changed":
        assert len(result["changed_vectors"]) == 1
    if failure == "raw":
        assert not result["raw_unchanged"]
