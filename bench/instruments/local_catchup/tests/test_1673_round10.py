"""A vector is observed only together with its metadata (one snapshot), so metadata
that first appears after its vector cannot be baselined after a later change."""
import sqlite3

import pytest

from bench.instruments.local_catchup import audit
from bench.instruments.local_catchup.tests.test_corpus_audit import fill

pytest_plugins = ["bench.instruments.local_catchup.tests.test_corpus_audit"]


# embedded_at is covered only by the metadata digest (the semantic expected-metadata check
# reads other columns), so a missed snapshot would let its change pass unnoticed.
@pytest.mark.parametrize("table,field", [("lcm_chunk_meta", "embedded_at"),
                                         ("lcm_embedding_meta", "embedded_at")])
def test_late_metadata_is_snapshotted_with_its_vector(corpus, table, field):
    db, manifest, mod = corpus
    fill(db, manifest, mod)
    kind = "chunks" if table == "lcm_chunk_meta" else "leaves"
    with sqlite3.connect(db) as c:
        cur = c.execute(f"SELECT * FROM {table}")
        columns = [d[0] for d in cur.description]
        saved = cur.fetchall()
        c.execute(f"DELETE FROM {table}")
    assert saved
    audit.progress(db, manifest)
    assert not [k for k in manifest.get("observed_vectors", {}) if k.startswith(kind+"/")]
    with sqlite3.connect(db) as c:
        c.executemany(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?'*len(columns))})", saved)
    audit.progress(db, manifest)
    assert [k for k in manifest["observed_vectors"] if k.startswith(kind+"/")]
    with sqlite3.connect(db) as c:
        c.execute(f"UPDATE {table} SET {field}={field}+1")
    assert audit.audit(db, manifest)["changed_metadata"]
